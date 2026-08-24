//! Real local inference engine backed by llama.cpp (via `llama-cpp-2`).
//!
//! Loads a GGUF file directly — tokenizer and chat template come from the
//! file's metadata, so a single `.gguf` is all the user needs.
//!
//! Each loaded model owns a dedicated worker thread holding a **persistent**
//! `LlamaContext`. Across turns the KV cache is reused by longest-common-prefix
//! matching: only the newly added tokens are decoded, so a long conversation no
//! longer re-processes its whole history every turn. This is also the substrate
//! for KV snapshot / branching.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tauri::ipc::Channel;

use super::{ChatMessage, GenParams, GenRequest, GenStats, InferenceBackend, ModelInfo, Role, StreamEvent};

// ── GPU crash guard (issue #5) ──────────────────────────────────────────────
// A broken Vulkan driver aborts the whole process DURING model load — no
// Result to catch, the window just vanishes. The guard turns that one-shot
// death into a persistent CPU fallback: an `inflight` marker is written just
// before the load and removed when load() returns (Ok or Err — a clean error
// is not a crash); if the marker is still there at the NEXT startup, the
// previous load killed the process, so GPU offload gets blocked for good
// (GGML_VK_VISIBLE_DEVICES="" → the Vulkan backend registers zero devices)
// and the UI is told to say so.

const GPU_INFLIGHT: &str = "llama-load.inflight";
const GPU_BLOCKED: &str = "llama-gpu-blocked";

use crate::errlog::chaty_data_dir;

/// Pure state machine over a base dir (testable): promote a stale inflight
/// marker to the persistent block, and report whether GPU must stay off.
/// PURE means no logging — the first version appended to the REAL error log
/// from in here, so every `cargo test` run stamped a false "gpu crashed"
/// entry into the dev machine's user log (the errlog-pollution sin, third
/// occurrence). The production caller logs; the state machine doesn't.
fn gpu_guard_check(base: &Path) -> bool {
    let inflight = base.join(GPU_INFLIGHT);
    let blocked = base.join(GPU_BLOCKED);
    if inflight.exists() {
        let _ = std::fs::write(&blocked, "previous model load crashed the process\n");
        let _ = std::fs::remove_file(&inflight);
    }
    blocked.exists()
}

/// Known bad-conversion tells, per model family. Two cases so far:
/// * MiniCPM5 is plain llama-arch BY DESIGN — its tell is the tokenizer:
///   official conversions declare `tokenizer.ggml.pre = "minicpm5"`, while
///   files made with pre-MiniCPM5 convert scripts fall back to "llama-bpe"
///   and degenerate (the owner's two downloads: deterministic whitespace on
///   zh prompts, token salad on en — both with a textbook-perfect prompt).
/// * MiniCPM 1–3 DO need their own architecture (µP scalers live in the
///   arch handling), so those exported as plain "llama" are broken.
fn conversion_suspect(name: &str, arch: &str, tokenizer_pre: &str) -> bool {
    let _ = tokenizer_pre;
    let n = name.to_lowercase();
    // MiniCPM5: even the OFFICIAL GGUF (llama arch, llama-bpe pre) degenerates
    // on the llama.cpp this build bundles — upstream b10330 runs the same file
    // fine, so the support gap is in the engine version, not any one file.
    // Flag the whole family until the bundled engine catches up; the MLX
    // build runs perfectly through the sidecar meanwhile.
    if n.contains("minicpm5") {
        return true;
    }
    // MiniCPM 1-3 need their own arch (µP scalers); plain-llama exports are broken.
    if n.contains("minicpm") {
        return arch == "llama";
    }
    false
}

/// Call once at process start, BEFORE any llama backend init. Returns true
/// when GPU offload was blocked because a previous load crashed the process.
/// Windows-only in effect: the crash class is the Vulkan driver, macOS runs
/// Metal (which ignores the env), and an active guard off-Windows can only
/// produce FALSE "gpu crashed" warnings — a cargo-test on the dev Mac once
/// raced the real app into exactly that.
pub fn apply_gpu_crash_guard() -> bool {
    #[cfg(windows)]
    {
        let base = chaty_data_dir();
        let promoted = base.join(GPU_INFLIGHT).exists();
        let blocked = gpu_guard_check(&base);
        if promoted && blocked {
            crate::errlog::append_error(
                "gpu-crash-guard",
                "previous model load crashed the process (likely GPU driver abort, issue #5 class); GPU offload disabled — running CPU-only from now on",
            );
        }
        if blocked {
            // Vulkan backend: zero visible devices = never touches the
            // driver's allocation/pipeline paths again.
            std::env::set_var("GGML_VK_VISIBLE_DEVICES", "");
        }
        return blocked;
    }
    #[cfg(not(windows))]
    {
        // Hygiene only: a stale inflight marker (crashed dev build, killed
        // test run) is removed without promoting it to a block.
        let _ = std::fs::remove_file(chaty_data_dir().join(GPU_INFLIGHT));
        false
    }
}

/// Whether the guard is currently blocking GPU offload (for the load reply).
pub fn gpu_crash_blocked() -> bool {
    #[cfg(windows)]
    {
        return chaty_data_dir().join(GPU_BLOCKED).exists();
    }
    #[cfg(not(windows))]
    false
}

/// Removes the marker when load() returns — normally OR with an error. Only
/// a process death leaves it behind, which is exactly the signal we want.
/// Armed on Windows only (see apply_gpu_crash_guard).
struct LoadGuard(Option<std::path::PathBuf>);
impl LoadGuard {
    fn arm_at(base: &Path) -> Self {
        let p = base.join(GPU_INFLIGHT);
        let _ = std::fs::write(&p, "loading\n");
        Self(Some(p))
    }
    fn arm() -> Self {
        if cfg!(windows) {
            Self::arm_at(&chaty_data_dir())
        } else {
            Self(None)
        }
    }
}
impl Drop for LoadGuard {
    fn drop(&mut self) {
        if let Some(p) = &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Process-wide llama.cpp backend. It may only be initialized once.
static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

fn llama_backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = LLAMA_BACKEND.get() {
        return Ok(b);
    }
    let _guard = INIT_LOCK.lock().unwrap();
    if let Some(b) = LLAMA_BACKEND.get() {
        return Ok(b);
    }
    let mut backend = LlamaBackend::init().context("failed to initialize llama.cpp backend")?;
    backend.void_logs();
    let _ = LLAMA_BACKEND.set(backend);
    Ok(LLAMA_BACKEND.get().unwrap())
}

/// Crate-public accessor so other engines (the RAG embedder) share the same
/// process-wide llama.cpp backend.
pub fn llama_backend_pub() -> Result<&'static LlamaBackend> {
    llama_backend()
}

/// Quickly read a model's transformer-layer count via a vocab-only load (no
/// weights), used to size the GPU offload before the real load.
///
/// NOTE: `n_layer()` returns 0 in vocab-only mode (the architecture isn't
/// built), so we read `<arch>.block_count` from the GGUF metadata, which *is*
/// available.
fn probe_n_layer(backend: &LlamaBackend, path: &str) -> Option<u32> {
    let params = LlamaModelParams::default().with_vocab_only(true);
    let model = LlamaModel::load_from_file(backend, path, &params).ok()?;
    let arch = model.meta_val_str("general.architecture").ok()?;
    model
        .meta_val_str(&format!("{arch}.block_count"))
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .or_else(|| match model.n_layer() {
            0 => None,
            n => Some(n),
        })
}

/// Find the vision encoder (mmproj) GGUF paired with a model file.
///
/// Convention: a vision model lives in its own folder together with its
/// `mmproj*.gguf` (that's the layout the in-app downloader creates). To avoid
/// mispairing in a flat `models/` folder holding many models, a same-dir
/// mmproj is only paired when the model is the *only* main GGUF there, or the
/// mmproj filename mentions the model's stem.
pub fn find_mmproj(model_path: &str) -> Option<std::path::PathBuf> {
    let path = Path::new(model_path);
    let dir = path.parent()?;
    let is_mmproj = |n: &str| n.to_lowercase().contains("mmproj");
    let name = path.file_name()?.to_str()?;
    if is_mmproj(name) {
        return None; // the mmproj itself is not a chat model
    }

    let mut mains = 0usize;
    let mut projs: Vec<std::path::PathBuf> = Vec::new();
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        let Some(n) = p.file_name().and_then(|s| s.to_str()) else { continue };
        if !n.to_lowercase().ends_with(".gguf") {
            continue;
        }
        if is_mmproj(n) {
            projs.push(p);
        } else {
            mains += 1;
        }
    }
    if projs.is_empty() {
        return None;
    }
    // Prefer smaller-precision projections (F16 over F32) — visually
    // indistinguishable, half the memory.
    projs.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX));
    if mains <= 1 {
        return projs.into_iter().next();
    }
    // Crowded folder: require a filename affinity (first stem token).
    let stem = name.trim_end_matches(".gguf").to_lowercase();
    let token = stem.split(['-', '_', '.']).next().unwrap_or(&stem).to_string();
    projs
        .into_iter()
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| token.len() >= 3 && n.to_lowercase().contains(&token))
                .unwrap_or(false)
        })
}

/// KV-cache state of the *media* (mtmd) prefill regime — the multimodal
/// analogue of the token-prefix cache. Stores the rendered prompt string and
/// image identities as of the last prefill, so the next turn of the same
/// conversation only evaluates the appended tail (old images are NOT
/// re-encoded).
struct MediaCache {
    /// Rendered prompt (with media markers) that is resident in the KV.
    prompt: String,
    /// Identity keys of the images already encoded into the KV, in order.
    image_keys: Vec<String>,
    /// Number of positions resident right after that prefill.
    n_past: i32,
    /// The prompt WITHOUT the generation tail (assistant header + thinking
    /// prefill) — the stable anchor the next turn's render string-extends
    /// even when the tail diverges (Qwen3.5+ `<think>` prefill). Empty when
    /// the anchor position couldn't be observed (disables anchor reuse).
    body: String,
    /// Positions resident at the end of `body`.
    n_past_body: i32,
    /// Whether `prompt`/`n_past` describe the KV *including* the reply this
    /// turn generated. When they do, a prompt that string-extends them is a
    /// pure append with nothing to truncate — which is the only way a
    /// hybrid/recurrent model can reuse a media prefill at all, since its
    /// state cannot be rewound to drop a generation tail.
    complete: bool,
}

/// Total images pushed through the vision encoder (observability: the media
/// cache exists so old images are NOT re-encoded — tests assert on this).
static IMG_ENCODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Times the media cache had to fall back to a full re-prefill because the
/// model's memory doesn't support partial removal (hybrid/recurrent archs
/// like Qwen3.6 — their state can't be rewound to a mid-sequence point).
static CACHE_FALLBACKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) fn img_encode_count() -> usize {
    IMG_ENCODES.load(Ordering::Relaxed)
}
#[cfg(test)]
pub(crate) fn cache_fallback_count() -> usize {
    CACHE_FALLBACKS.load(Ordering::Relaxed)
}

/// Cheap identity for an image file (path + size + mtime).
fn image_cache_key(p: &str) -> String {
    match std::fs::metadata(p) {
        Ok(m) => format!("{p}|{}|{:?}", m.len(), m.modified().ok()),
        Err(_) => format!("{p}|missing"),
    }
}

/// Clone messages, prepending one media marker per attached image to the
/// message text — mtmd's tokenizer replaces each marker with that image's
/// embedding chunks.
fn inject_media_markers(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| {
            if m.images.is_empty() {
                m.clone()
            } else {
                let mut content =
                    String::with_capacity(m.content.len() + m.images.len() * 16);
                for _ in &m.images {
                    content.push_str(mtmd_default_marker());
                    content.push('\n');
                }
                content.push_str(&m.content);
                ChatMessage {
                    role: m.role.clone(),
                    content,
                    images: m.images.clone(),
                    reasoning_content: m.reasoning_content.clone(),
                }
            }
        })
        .collect()
}

/// A unit of work sent to a model's worker thread.
enum Job {
    Generate {
        req: GenRequest,
        sink: Channel<StreamEvent>,
        cancel: Arc<AtomicBool>,
        done: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// One-shot generation collected in-process (no streaming) — the substrate
    /// for vision analysis by Code mode / KB / Canvas. Runs on the same worker
    /// so it never races the persistent context.
    Collect {
        req: GenRequest,
        cancel: Arc<AtomicBool>,
        done: tokio::sync::oneshot::Sender<Result<String>>,
    },
}

/// In-process sink that concatenates streamed token text (the non-IPC analogue
/// of the Tauri `Channel`). Lives entirely on the worker thread.
struct StringSink {
    buf: std::cell::RefCell<String>,
}
impl EventSink for StringSink {
    fn emit(&self, ev: StreamEvent) -> Result<()> {
        if let StreamEvent::Token { text } = ev {
            self.buf.borrow_mut().push_str(&text);
        }
        Ok(())
    }
}

/// Handle to a loaded model. Generation is funneled to the worker thread, which
/// owns the persistent context (serializing requests, which also avoids any
/// concurrent-context issues).
pub struct LlamaEngine {
    /// `None` after `unload()` — the worker is gone and generation must fail.
    tx: Mutex<Option<Sender<Job>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl LlamaEngine {
    /// Load a GGUF file and spin up its worker thread. Blocking; run off-thread.
    ///
    /// `gpu_pref`: `None`/negative = auto‑tune by VRAM, `Some(0)` = force CPU,
    /// `Some(n>0)` = offload exactly `n` layers.
    pub fn load(path: &str, gpu_pref: Option<i32>, n_ctx_pref: Option<u32>) -> Result<(Self, ModelInfo)> {
        // Armed for the whole load: if the process dies in here (Vulkan
        // driver abort — issue #5), the marker survives and the next start
        // falls back to CPU instead of dying again.
        let _crash_guard = LoadGuard::arm();
        let backend = llama_backend()?;
        if !Path::new(path).exists() {
            bail!("model file not found: {path}");
        }

        // ---- pre-flight: refuse loads that can only end in a swap-freeze ----
        // Only the hard impossibility (weights exceed physical RAM) is checked.
        // Deliberately NOT checking "available" memory: macOS counts file cache
        // as used, so available_memory() wildly underestimates what a load can
        // actually obtain — it falsely rejected MoE models. Memory pressure
        // during a feasible load is handled by the synchronous eject, the KV
        // context clamp and the OOM back-off instead.
        {
            let weights = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
            let margin = 1_500u64 * 1024 * 1024; // ctx + compute buffers + app slack
            if weights.saturating_add(margin) > sys.total_memory() {
                bail!(
                    "模型体积约 {:.1} GiB，超过本机内存 {:.1} GiB，无法加载，请换用更小的量化版本。(Model ≈{:.1} GiB exceeds the {:.1} GiB of RAM — use a smaller quantization.)",
                    gib(weights),
                    gib(sys.total_memory()),
                    gib(weights),
                    gib(sys.total_memory())
                );
            }
        }

        // ---- GPU auto-tuning: offload as many layers as fit in VRAM ----
        let gpu = crate::gpu::detect_gpu();
        let requested = match gpu_pref {
            Some(0) => 0,            // force CPU
            Some(n) if n > 0 => n,   // manual layer count
            _ => match &gpu {        // auto (None / negative sentinel)
                Some(g) => match probe_n_layer(backend, path) {
                    Some(nl) => crate::gpu::auto_gpu_layers(Path::new(path), nl, g.vram_mb),
                    // Layer count unknown: try to offload everything; the
                    // retry-halving below backs off if it doesn't fit.
                    None => 999,
                },
                None => 0,
            },
        };

        // CPU-side worker threads. On Apple Silicon this is the performance-core
        // count (efficiency cores hurt throughput); elsewhere the logical CPUs.
        let n_threads = crate::gpu::cpu_worker_threads() as i32;

        // Vision: pair a sibling mmproj GGUF (folder layout) when one exists.
        let mmproj = find_mmproj(path).map(|p| p.to_string_lossy().to_string());

        // Load the weights AND allocate the inference context, backing off
        // `n_gpu_layers` on any out-of-memory failure. This covers BOTH the
        // weights and the KV-cache/compute buffers — the latter often OOMs a
        // small GPU even when the weights fit. If even a pure-CPU load runs out
        // of memory, return a clear error instead of a cryptic crash.
        let mut layers = requested.max(0);
        let mut oom_fallback = false;
        let (model, tx, handle, mtmd_err) = loop {
            let params = LlamaModelParams::default().with_n_gpu_layers(layers.max(0) as u32);
            // macOS: load via malloc instead of mmap. Freeing malloc'd weights
            // is synchronous, whereas the Metal-wired pages of an mmap'd MoE
            // model have been observed to never return to the kernel after
            // unload — the next big load then swap-freezes the machine.
            #[cfg(target_os = "macos")]
            let params = params.with_use_mmap(false);
            let model = match LlamaModel::load_from_file(backend, path, &params) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    let msg = format!("{e:#}");
                    if layers > 0 && is_oom(&msg) {
                        oom_fallback = true;
                        eprintln!("weight-load OOM at {layers} gpu layers; backing off");
                        layers = backoff_layers(layers);
                        continue;
                    }
                    if is_oom(&msg) {
                        bail!("加载模型权重时内存不足 (out of memory while loading the model weights)");
                    }
                    return Err(e).with_context(|| format!("failed to load GGUF model: {path}"));
                }
            };

            // Create the context in the worker and wait for the result, so a
            // KV/compute-buffer OOM is caught here and folded into the back-off.
            let n_ctx = mem_safe_n_ctx(&model, clamp_n_ctx(model.n_ctx_train(), n_ctx_pref));
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            // Init result: Err = fatal context failure; Ok(Some(msg)) = context
            // fine but the mmproj failed to load (vision off, non-fatal).
            let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<Option<String>, String>>();
            let worker_model = model.clone();
            let worker_mmproj = mmproj.clone();
            let worker_gpu = layers > 0;
            let handle = std::thread::Builder::new()
                .name("chaty-llama".into())
                .spawn(move || {
                    worker(worker_model, n_ctx, n_threads, worker_mmproj, worker_gpu, rx, init_tx)
                })
                .context("failed to start inference thread")?;

            match init_rx.recv() {
                Ok(Ok(mtmd_err)) => break (model, tx, handle, mtmd_err),
                Ok(Err(msg)) => {
                    drop(tx); // the worker already exited; release this attempt
                    // A failed context allocation (incl. a null return from
                    // llama.cpp) is almost always memory pressure — back off the
                    // GPU offload and retry rather than giving up.
                    if layers > 0 {
                        oom_fallback = true;
                        eprintln!("context init failed at {layers} gpu layers ({msg}); backing off");
                        layers = backoff_layers(layers);
                        continue;
                    }
                    if is_oom(&msg) {
                        bail!("out of memory while allocating the model context");
                    }
                    bail!("failed to initialize inference context: {msg}");
                }
                Err(_) => bail!("inference thread exited during initialization"),
            }
        };

        // `layers` may be n_layer+1 (output layer) or 999 (offload-all); clamp
        // the reported count to the block count so the panel reads e.g. "28/28".
        let gpu_layers = layers.min(model.n_layer() as i32).max(0);
        let gpu_name = if gpu_layers > 0 {
            gpu.as_ref().map(|g| g.name.clone())
        } else {
            None
        };
        let n_ctx_train = model.n_ctx_train();
        let n_ctx_wanted = clamp_n_ctx(n_ctx_train, n_ctx_pref);
        let n_ctx = mem_safe_n_ctx(&model, n_ctx_wanted);
        // If we had to drop below the requested offload to fit memory, flag it;
        // a context window clamped to fit unified memory is worth a note too.
        // A paired mmproj that failed to load degrades to text-only — surfaced
        // so the UI can say "vision unavailable" instead of silently ignoring
        // images.
        let vision_ready = mmproj.is_some() && mtmd_err.is_none();
        if let Some(err) = &mtmd_err {
            eprintln!("mmproj load failed (vision disabled): {err}");
        }
        let warning = if gpu_crash_blocked() {
            // A previous load crashed the process (issue #5: broken Vulkan
            // driver aborts mid-load) — this run is CPU-only by the guard.
            Some("gpu-crash-cpu".to_string())
        } else if oom_fallback {
            Some("gpu-oom".to_string())
        } else if mtmd_err.is_some() {
            Some("mmproj-failed".to_string())
        } else if n_ctx < n_ctx_wanted {
            Some("ctx-clamped".to_string())
        } else if conversion_suspect(
            &format!(
                "{} {}",
                Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or(""),
                model.meta_val_str("general.name").unwrap_or_default()
            ),
            &model.meta_val_str("general.architecture").unwrap_or_default(),
            &model.meta_val_str("tokenizer.ggml.pre").unwrap_or_default(),
        ) {
            Some("conversion-suspect".to_string())
        } else {
            None
        };
        let name = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.gguf")
            .to_string();

        // ---- intelligent GGUF probe (best-effort metadata sniffing) ----
        let arch = model.meta_val_str("general.architecture").unwrap_or_default();
        let arch_lc = arch.to_lowercase();
        let model_name = model
            .meta_val_str("general.name")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let name_lc = format!(
            "{} {}",
            model_name.as_deref().unwrap_or(""),
            name
        )
        .to_lowercase();
        let template = model.meta_val_str("tokenizer.chat_template").ok();
        let template_lc = template.as_deref().unwrap_or("").to_lowercase();

        let supports_tools = template_lc.contains("tool");
        // Qwen3.5/3.6 (arch "qwen35*"/"qwen36*") use the <think> paradigm with
        // no soft switch — the architecture field is authoritative over
        // template text, which community finetunes frequently customize.
        let is_qwen35plus = is_qwen3_5_plus_arch(&arch_lc);
        // Can the engine turn thinking off by itself, by prefilling an empty
        // reasoning block after the assistant header? Same question
        // `template_uses_think` answers at render time — asked here so the soft
        // switch knows whether it is needed. Architecture is authoritative: the
        // Qwen3 line reasons in `<think>` blocks whatever template a finetune
        // shipped, and some ship none at all.
        let engine_can_prefill_think =
            is_think_paradigm_arch(&arch_lc) || template_lc.contains("<think>");
        let supports_thinking = is_qwen35plus
            || template_lc.contains("think")
            || template_lc.contains("reasoning")
            || ["qwen3", "qwq", "deepseek-r1", "-r1", "reasoning", "thinking", "magistral", "cogito"]
                .iter()
                .any(|k| name_lc.contains(k) || arch_lc.contains(k));
        // The `/no_think` soft switch is a Qwen3-era convention. Qwen3.5+ dropped
        // it for an `enable_thinking` template flag, so the toggle must NOT inject
        // `/no_think` there (it would just leak into the prompt as noise). Detect
        // the switch by the template actually mentioning it — AND require the
        // embedded template to actually be usable: when we render through a
        // fallback (e.g. Gemma 4's unparseable Jinja), soft-switch text would
        // reach the model as literal prompt noise and can derail generation.
        let template_usable = model
            .chat_template(None)
            .ok()
            .and_then(|t| {
                let probe = vec![LlamaChatMessage::new("user".to_string(), "hi".to_string()).ok()?];
                model.apply_chat_template(&t, &probe, true).ok()
            })
            .is_some();
        // `/no_think` appended to the user's message: Qwen3's soft switch, and
        // the last resort. It is offered only when nothing better exists,
        // because it is worse in both directions. It mutates the turn, so the
        // next prompt — rendered without it — diverges at that message and the
        // cache dies there: 51-71% reuse against 100% for the engine flag,
        // measured over four turns on two Qwen3 builds. And on those same two it
        // does not even work — both kept reasoning with `/no_think` appended,
        // the trained behaviour apparently lost in the finetune, while the flag
        // silenced them.
        //
        // So: whenever the engine can prefill an empty reasoning block itself,
        // that is the mechanism, and this stays off.
        //
        // The probe also has to drop `</think>` before asking, because that
        // closing tag CONTAINS `/think` — every template that closes a block
        // matched, which is all of them. LFM2 was handed a switch it has never
        // heard of, the app stopped sending the flag on its behalf, and thinking
        // could not be turned off on it at all.
        let switch_probe = template_lc.replace("</think>", "");
        let think_switch = !is_qwen35plus
            && !engine_can_prefill_think
            && template_usable
            && (switch_probe.contains("no_think") || switch_probe.contains("/think"));
        let multimodal = mmproj.is_some()
            || model
                .meta_val_str(&format!("{arch}.vision.block_count"))
                .is_ok()
            || model.meta_val_str("clip.has_vision_encoder").is_ok()
            || [
                "-vl", " vl", "vision", "llava", "mllama", "qwen2vl", "qwen2.5-vl",
                "minicpm-v", "internvl", "pixtral", "idefics", "smolvlm", "gemma-3",
            ]
            .iter()
            .any(|k| name_lc.contains(k) || arch_lc.contains(k));
        let quant = model
            .meta_val_str("general.file_type")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(|ft| quant_name(ft).to_string());

        let effort_levels = effort_levels_of(template.as_deref().unwrap_or(""));
        let tool_role = probe_tool_role(&model);
        // llama.cpp's chat message carries a role and a body and nothing else,
        // so a structured reasoning field can never reach a template here.
        let reasoning_field = false;
        let info = ModelInfo {
            name,
            path: path.to_string(),
            backend: "llama.cpp".to_string(),
            loaded: true,
            arch: (!arch.is_empty()).then(|| arch.clone()),
            size_mb: Some(model.size() / (1024 * 1024)),
            params_b: Some(model.n_params() as f64 / 1e9),
            n_ctx_train: Some(n_ctx_train),
            n_ctx: Some(n_ctx),
            n_layer: Some(model.n_layer()),
            gpu_layers,
            gpu_name,
            model_name,
            quant,
            n_embd: Some(model.n_embd() as u32),
            has_chat_template: template.is_some(),
            supports_thinking,
            think_switch,
            effort_levels,
            tool_role,
            reasoning_field,
            supports_tools,
            multimodal,
            vision_ready,
            mmproj,
            warning,
        };

        // `tx` + the worker came from the load/back-off loop above.
        Ok((
            Self {
                tx: Mutex::new(Some(tx)),
                worker: Mutex::new(Some(handle)),
            },
            info,
        ))
    }
}

#[async_trait]
impl InferenceBackend for LlamaEngine {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn unload(&self) {
        // Drop the job sender (worker's recv() then errors and the thread
        // winds down, freeing the context + weights), then block until it has
        // actually exited so the caller can safely load a replacement.
        drop(self.tx.lock().unwrap().take());
        if let Some(h) = self.worker.lock().unwrap().take() {
            let _ = h.join();
        }
    }

    async fn generate(&self, req: GenRequest, sink: Channel<StreamEvent>, cancel: Arc<AtomicBool>) -> Result<()> {
        let tx = self
            .tx
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("模型已卸载 (model unloaded)"))?;
        let (done, done_rx) = tokio::sync::oneshot::channel();
        tx.send(Job::Generate { req, sink, cancel, done })
            .map_err(|_| anyhow::anyhow!("推理线程已退出 (inference thread exited)"))?;
        match done_rx.await {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!("推理线程在生成过程中断开 (inference thread disconnected mid-generation)")),
        }
    }

    async fn generate_collect(&self, req: GenRequest, cancel: Arc<AtomicBool>) -> Result<String> {
        let tx = self
            .tx
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("模型已卸载 (model unloaded)"))?;
        let (done, done_rx) = tokio::sync::oneshot::channel();
        tx.send(Job::Collect { req, cancel, done })
            .map_err(|_| anyhow::anyhow!("推理线程已退出 (inference thread exited)"))?;
        match done_rx.await {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!("推理线程在生成过程中断开 (inference thread disconnected)")),
        }
    }
}

/// Owns the persistent context for one model and serves jobs until the engine
/// (and its `Sender`) is dropped.
fn worker(
    model: Arc<LlamaModel>,
    n_ctx: u32,
    n_threads: i32,
    mmproj: Option<String>,
    use_gpu: bool,
    rx: Receiver<Job>,
    init: Sender<Result<Option<String>, String>>,
) {
    let backend = match llama_backend() {
        Ok(b) => b,
        Err(e) => {
            let _ = init.send(Err(format!("{e:#}")));
            return;
        }
    };
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);
    // Flash attention (less KV memory + faster long-context decode, a real win on
    // Metal) is left at llama.cpp's default policy (AUTO), which enables it
    // automatically on Apple Silicon when the model supports it. To force it,
    // `with_flash_attention_policy(..)` takes a raw `llama_flash_attn_type`.
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            // Almost always a VRAM/RAM OOM allocating the KV cache + compute
            // buffers; report it so `load()` can back off the GPU offload.
            let _ = init.send(Err(format!("{e:#}")));
            return;
        }
    };
    // Vision encoder (mmproj), when the model ships one. A failure here is
    // non-fatal: the model still chats, images are just unavailable.
    let mut mtmd_err: Option<String> = None;
    let mtmd = mmproj.as_deref().and_then(|p| {
        let params = MtmdContextParams {
            use_gpu,
            n_threads,
            ..MtmdContextParams::default()
        };
        match MtmdContext::init_from_file(p, &model, &params) {
            Ok(c) => {
                if c.support_vision() {
                    Some(c)
                } else {
                    mtmd_err = Some("mmproj has no vision encoder".to_string());
                    None
                }
            }
            Err(e) => {
                mtmd_err = Some(format!("{e}"));
                None
            }
        }
    });
    let _ = init.send(Ok(mtmd_err));

    // Tokens currently resident in the KV cache for sequence 0 (positions 0..len).
    let mut cached: Vec<LlamaToken> = Vec::new();
    // Multimodal analogue of `cached` (see `MediaCache`).
    let mut media_cache: Option<MediaCache> = None;

    while let Ok(job) = rx.recv() {
        match job {
            Job::Generate { req, sink, cancel, done } => {
                let result = run_turn(
                    &model,
                    &mut ctx,
                    &mut cached,
                    mtmd.as_ref(),
                    &mut media_cache,
                    n_ctx,
                    &req,
                    &sink,
                    &cancel,
                );
                let _ = done.send(result);
            }
            Job::Collect { req, cancel, done } => {
                let sink = StringSink { buf: std::cell::RefCell::new(String::new()) };
                let result = run_turn(
                    &model,
                    &mut ctx,
                    &mut cached,
                    mtmd.as_ref(),
                    &mut media_cache,
                    n_ctx,
                    &req,
                    &sink,
                    &cancel,
                )
                .map(|()| sink.buf.into_inner());
                let _ = done.send(result);
            }
        }
    }
}

/// Decode `tokens[from..]` into the context in `n_batch`-sized chunks, setting
/// logits on the final token. `n_batch` must be ≥ 1.
fn decode_prompt(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
    from: usize,
    n_batch: usize,
    mut on_batch: impl FnMut(usize, usize),
) -> Result<()> {
    let n_prompt = tokens.len();
    let mut pos = from;
    on_batch(pos, n_prompt); // initial position (KV-reused prefix counts as done)
    while pos < n_prompt {
        let end = (pos + n_batch).min(n_prompt);
        batch.clear();
        for (j, tok) in tokens[pos..end].iter().enumerate() {
            batch.add(*tok, (pos + j) as i32, &[0], pos + j == n_prompt - 1)?;
        }
        ctx.decode(batch).context("decode failed")?;
        pos = end;
        on_batch(pos, n_prompt);
    }
    Ok(())
}

/// Where generated events go. Production streams them over a Tauri IPC channel;
/// tests collect them in-process, which lets `run_turn` (the real decode loop)
/// be driven headless against a live model.
pub trait EventSink {
    fn emit(&self, ev: StreamEvent) -> Result<()>;
}
impl EventSink for Channel<StreamEvent> {
    fn emit(&self, ev: StreamEvent) -> Result<()> {
        self.send(ev)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_turn(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    cached: &mut Vec<LlamaToken>,
    mtmd: Option<&MtmdContext>,
    media_cache: &mut Option<MediaCache>,
    n_ctx: u32,
    req: &GenRequest,
    sink: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<()> {
    sink.emit(StreamEvent::Started)?;

    let has_images = req.messages.iter().any(|m| !m.images.is_empty());
    let media_turn = has_images && mtmd.is_some();
    if has_images && mtmd.is_none() {
        // Defensive: the frontend only sends images to vision-ready models.
        eprintln!("images attached but no mmproj is loaded — ignoring them");
    }

    let (prompt, prompt_body) = if media_turn {
        build_prompt_pair(model, &inject_media_markers(&req.messages), req.params.think)?
    } else {
        (build_prompt(model, &req.messages, req.params.think)?, String::new())
    };
    // Native reasoning-effort rung (Qwen3.8). The template rendered with
    // llama.cpp's default kwargs already carries the `xhigh` sentence, so a
    // different rung is a verbatim swap. BOTH renders get it: the body is the
    // media-cache anchor and must stay a prefix of the full prompt.
    let (prompt, prompt_body) = match req.params.effort.as_deref() {
        Some(level) if req.params.think != Some(false) => (
            apply_effort(&prompt, level),
            if prompt_body.is_empty() { prompt_body } else { apply_effort(&prompt_body, level) },
        ),
        _ => (prompt, prompt_body),
    };
    // Qwen3.5/3.6-style templates PRE-OPEN the reasoning block: the prompt
    // ends with "<think>\n" and the model starts mid-reasoning, so the UI
    // would never see an opening tag. Emit a synthetic one so the stream is
    // well-formed for the frontend's think-panel parser.
    if req.params.think != Some(false) && prompt.trim_end().ends_with("<think>") {
        sink.emit(StreamEvent::Token { text: "<think>\n".to_string() })?;
    }

    let n_batch = (ctx.n_batch() as usize).max(1);
    let mut batch = LlamaBatch::new(n_batch, 1);

    // ---- prefill: two regimes sharing one generation loop below ----
    // `n_prompt_pos` = positions resident after prefill; `idx` = where to
    // sample the first token (-1 = "last logits" after an mtmd prefill).
    let (n_prompt_pos, mut idx): (i32, i32);
    // How many prompt tokens the KV already held — the same observability the
    // MLX engine reports, and the number that shows an agent turn is a pure
    // append rather than a full re-read.
    let mut kv_reused = 0u32;

    if media_turn {
        let mtmd = mtmd.expect("media_turn implies mtmd");
        // The token-prefix cache can't describe media chunks; it is empty
        // while a conversation is in the media regime (and vice versa).
        cached.clear();

        n_prompt_pos = prefill_media(
            ctx,
            mtmd,
            media_cache,
            &prompt,
            &prompt_body,
            &req.messages,
            n_ctx,
            n_batch as i32,
            sink,
            cancel,
        )?;
        if cancel.load(Ordering::Relaxed) {
            return done_event(sink, n_prompt_pos as u32, 0, 0.0, "cancelled");
        }
        idx = -1;
    } else {
        // Leaving the media regime: the KV holds media embeddings the token
        // cache can't account for — start clean.
        if media_cache.take().is_some() {
            ctx.clear_kv_cache();
            cached.clear();
        }

        // Diagnostic parity with the MLX engine: what the model actually sees.
        // Off unless asked for — prompts carry user content.
        if std::env::var("CHATY_DUMP_PROMPT").as_deref() == Ok("1") {
            eprintln!("PROMPT[{} chars]>>>{prompt}<<<END", prompt.len());
        }
        let tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .context("tokenization failed")?;
        let n_prompt = tokens.len();

        if n_prompt + 4 >= n_ctx as usize {
            ctx.clear_kv_cache();
            cached.clear();
            bail!("提示词 {n_prompt} tokens 超出上下文窗口 {n_ctx}，请新建对话或缩短输入。(Prompt exceeds the {n_ctx}-token context window — start a new chat or shorten the input.)");
        }

        // Longest common prefix with the cached sequence → reuse that part of the KV.
        let mut prefix = 0usize;
        let max_match = cached.len().min(n_prompt);
        while prefix < max_match && cached[prefix] == tokens[prefix] {
            prefix += 1;
        }
        // Always leave at least one token to decode so we have fresh logits to sample.
        if prefix == n_prompt {
            prefix = n_prompt - 1;
        }

        // Drop everything in the KV at/after `prefix`, then decode only the new
        // tail. Partial removal is unsupported on hybrid/recurrent memories
        // (Qwen3.5/3.6 — seq_rm reports false): silently proceeding would
        // leave the previous conversation's state in place and the model
        // would see BOTH conversations at once. Fall back to a full clear.
        if prefix < cached.len()
            && ctx.clear_kv_cache_seq(Some(0), Some(prefix as u32), None) != Ok(true)
        {
            ctx.clear_kv_cache();
            cached.clear();
            prefix = 0;
        }
        cached.truncate(prefix);

        if cancel.load(Ordering::Relaxed) {
            ctx.clear_kv_cache();
            cached.clear();
            return done_event(sink, n_prompt as u32, 0, 0.0, "cancelled");
        }

        // Report prompt-processing progress, but only when the new tail spans
        // more than one batch — short prefills would just flash a ring.
        let progress = |from: usize| {
            move |done: usize, total: usize| {
                if total.saturating_sub(from) > n_batch {
                    let _ = sink.emit(StreamEvent::Prefill {
                        processed: done as u32,
                        total: total as u32,
                    });
                }
            }
        };

        // Decode the new tail, reusing the cached KV prefix. Some models don't
        // tolerate partial KV reuse (llama.cpp's decode returns an error); if so,
        // clear the KV and decode the whole prompt fresh. If that still fails, reset
        // state so the next turn / new chat starts clean instead of staying broken.
        if let Err(e) = decode_prompt(ctx, &mut batch, &tokens, prefix, n_batch, progress(prefix)) {
            eprintln!("prompt decode (reuse from {prefix}) failed: {e:#}; retrying from a clean KV");
            ctx.clear_kv_cache();
            if let Err(e2) = decode_prompt(ctx, &mut batch, &tokens, 0, n_batch, progress(0)) {
                ctx.clear_kv_cache();
                cached.clear();
                return Err(e2).context("prompt decode failed");
            }
        }
        kv_reused = prefix as u32;
        *cached = tokens; // KV now holds the full prompt
        n_prompt_pos = n_prompt as i32;
        idx = batch.n_tokens() - 1;
    }

    // Diagnostic parity with the MLX engine's CHATY_MLX_DUMP_TOKENS: the exact
    // ids this turn generated. The text a turn streams is what the NEXT turn's
    // prompt is rebuilt from, so any id whose piece renders empty — a control
    // token — is an id the next prompt cannot reproduce, and the KV prefix
    // diverges at exactly that position.
    let dump_gen = std::env::var("CHATY_DUMP_GEN_TOKENS").as_deref() == Ok("1");
    let mut gen_ids: Vec<i32> = Vec::new();

    // How much of `out` is actually resident in the KV. See the snapshot below.
    let mut decoded_len = 0usize;

    let mut sampler = build_sampler(&req.params);
    // Robust incremental UTF-8 assembly: accumulate raw token bytes and only
    // emit the valid-UTF-8 prefix, carrying any incomplete trailing bytes to
    // the next token. This never drops a byte (the old streaming decoder could
    // silently swallow a char at a token boundary).
    let mut pending: Vec<u8> = Vec::new();
    let start = Instant::now();
    let mut n_decoded: u32 = 0;
    let mut n_past = n_prompt_pos;

    // Stop sequences: hold back up to `max_stop-1` chars before emitting so a stop
    // string straddling token boundaries is still caught, then trim at the match.
    let mut stops: Vec<String> = req
        .params
        .stop
        .iter()
        .map(|s| s.replace("\\n", "\n"))
        .filter(|s| !s.is_empty())
        .collect();
    // Implicit turn-boundary stops for Gemma 4: insurance in case the GGUF
    // doesn't mark `<turn|>` as an end-of-generation token. Hitting one of
    // these is a natural end, not a user stop sequence.
    let implicit_from = stops.len();
    if is_gemma4(model) {
        stops.push("<turn|>".to_string());
        stops.push("<|turn>".to_string());
    }
    let max_stop = stops.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = String::new(); // all decoded text so far
    let mut emitted = 0usize; // bytes of `out` already streamed
    let mut stopped = false;
    let mut stop_reason = "eos";

    loop {
        if cancel.load(Ordering::Relaxed) {
            stop_reason = "cancelled";
            break;
        }
        let token = sampler.sample(ctx, idx);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        if dump_gen {
            gen_ids.push(token.0);
        }
        pending.extend_from_slice(&piece_bytes(model, token));
        let valid = match std::str::from_utf8(&pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            out.push_str(&text);
        }

        // Emit the portion of `out` that is safe to send.
        if max_stop == 0 {
            if emitted < out.len() {
                let chunk = out[emitted..].to_string();
                emitted = out.len();
                if !chunk.is_empty() {
                    sink.emit(StreamEvent::Token { text: chunk })?;
                }
            }
        } else if let Some((si, rel)) = stops
            .iter()
            .enumerate()
            .filter_map(|(i, s)| out[emitted..].find(s).map(|r| (i, r)))
            .min_by_key(|&(_, r)| r)
        {
            let abs = emitted + rel;
            if abs > emitted {
                sink.emit(StreamEvent::Token { text: out[emitted..abs].to_string() })?;
            }
            emitted = abs;
            stopped = true;
            // An implicit turn-boundary stop is a natural end of the reply.
            stop_reason = if si >= implicit_from { "eos" } else { "stop" };
            break;
        } else {
            let mut safe = out.len().saturating_sub(max_stop - 1);
            while safe > emitted && !out.is_char_boundary(safe) {
                safe -= 1;
            }
            if safe > emitted {
                sink.emit(StreamEvent::Token { text: out[emitted..safe].to_string() })?;
                emitted = safe;
            }
        }

        n_decoded += 1;
        // ── Degenerate-output watchdog ── a model that has produced 32
        // tokens of pure whitespace is not thinking, it is broken (the
        // MiniCPM5-as-llama case: a GGUF converted under the wrong
        // architecture loses its embed/logit scalers and deterministically
        // emits spaces until the token cap). Say so instead of streaming a
        // silent screenful of nothing — the user reads "no answer", when
        // the truth is "this model file is a bad conversion".
        if n_decoded == 32 && out.trim().is_empty() {
            sink.emit(StreamEvent::Token {
                text: "⚠️ 模型输出退化(连续空白),已中止。该模型文件很可能转换损坏(常见:预分词器或架构元数据声明错误,如 MiniCPM5 被标成 llama-bpe 预分词)。请改用该模型的官方 GGUF 或 MLX 版本。\n(Model output degenerated into pure whitespace — aborted. The file is likely a broken conversion — commonly a wrong tokenizer-pre or architecture declaration. Use the model's official GGUF or MLX build.)".to_string(),
            })?;
            stop_reason = "degenerate";
            break;
        }
        // max_tokens == 0 means "no per-reply cap" (the context window still bounds us).
        if req.params.max_tokens > 0 && n_decoded >= req.params.max_tokens {
            stop_reason = "length";
            break;
        }
        if n_past + 1 >= n_ctx as i32 {
            stop_reason = "context";
            break;
        }
        batch.clear();
        batch.add(token, n_past, &[0], true)?;
        n_past += 1;
        if let Err(e) = ctx.decode(&mut batch) {
            // The ledger must never run ahead of the cache — reset both
            // rather than leaving a phantom token the next turn would reuse.
            ctx.clear_kv_cache();
            cached.clear();
            *media_cache = None;
            return Err(e).context("decode failed");
        }
        if !media_turn {
            cached.push(token);
        }
        // The text of everything the KV now holds. A token reaches the cache
        // only here, at the end of its iteration — a stop sequence breaks out
        // above, with that token's piece already in `out` but never decoded —
        // so the snapshot has to be taken after the decode, not before.
        decoded_len = out.len();
        idx = batch.n_tokens() - 1;
    }
    // Flush the unsent tail (unless we halted on a stop sequence).
    if !stopped {
        if emitted < out.len() {
            let _ = sink.emit(StreamEvent::Token { text: out[emitted..].to_string() });
        }
        if !pending.is_empty() {
            let text = String::from_utf8_lossy(&pending).into_owned();
            if !text.is_empty() {
                let _ = sink.emit(StreamEvent::Token { text });
            }
        }
    }

    if dump_gen {
        let pieces: Vec<String> = gen_ids
            .iter()
            .map(|t| {
                let text = String::from_utf8_lossy(&piece_bytes(model, LlamaToken(*t))).to_string();
                if text.is_empty() {
                    // Nothing should land here now that control tokens render;
                    // if one does, it is a token the next prompt cannot
                    // reproduce and the prefix will break on it.
                    format!("{t}:<EMPTY>")
                } else {
                    format!("{t}:{text:?}")
                }
            })
            .collect();
        eprintln!("GEN_TOKENS[{}]>>>{}<<<END", gen_ids.len(), pieces.join(" "));
    }

    // Record the reply into the media ledger, so the next turn's prompt — which
    // contains this very reply — extends it instead of colliding with a tail
    // nothing describes. Without this the next prefill had to rewind the KV
    // past the generated tokens, and a hybrid model cannot rewind: it re-read
    // the whole conversation and re-encoded every image, on every single round.
    //
    // Only when `pending` is empty, because a multi-byte character split across
    // tokens leaves bytes that have not reached `out` yet — the ledger would
    // then describe less than the KV holds, and the next turn would evaluate a
    // tail the model has already seen. In that case the ledger stays
    // prompt-only and the old rewind path handles it.
    if media_turn && pending.is_empty() && decoded_len > 0 {
        if let Some(c) = media_cache.as_mut() {
            c.prompt.push_str(&out[..decoded_len]);
            c.n_past = n_past;
            c.complete = true;
        }
    }

    let secs = start.elapsed().as_secs_f32().max(1e-3);
    done_event_reused(sink, n_prompt_pos as u32, n_decoded, n_decoded as f32 / secs, stop_reason, kv_reused)
}

/// Prefill a multimodal prompt through mtmd, reusing the media KV cache
/// incrementally when the conversation merely grew (the common case): only the
/// appended tail — and only the *new* images — are evaluated. Returns the
/// number of positions resident after the prefill.
#[allow(clippy::too_many_arguments)]
fn prefill_media(
    ctx: &mut LlamaContext,
    mtmd: &MtmdContext,
    media_cache: &mut Option<MediaCache>,
    prompt: &str,
    prompt_body: &str,
    messages: &[ChatMessage],
    n_ctx: u32,
    n_batch: i32,
    sink: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<i32> {
    let images: Vec<&String> = messages.iter().flat_map(|m| m.images.iter()).collect();
    let image_keys: Vec<String> = images.iter().map(|p| image_cache_key(p)).collect();

    // Incremental reuse, best plan first. Generated tokens beyond the cached
    // prefill are dropped (mirrors the text path, which re-renders the
    // assistant turn from the template rather than trusting raw output).
    //
    // 1. EXTEND: the new prompt string-extends the FULL cached prompt
    //    (generation tail included) — nothing to truncate, evaluate the tail.
    // 2. ANCHOR: the new prompt extends the cached BODY but not the full
    //    prompt — the generation tail diverged (Qwen3.5+ `<think>` prefill vs
    //    the re-rendered assistant turn). Truncate the KV back to the body
    //    and evaluate from there: a handful of tail tokens re-decoded, every
    //    already-encoded image kept.
    // 3. Otherwise start clean.
    let img_prefix_ok = |c: &MediaCache| {
        image_keys.len() >= c.image_keys.len() && image_keys[..c.image_keys.len()] == c.image_keys[..]
    };
    // Strict extension: an identical prompt (e.g. regenerate) must re-eval —
    // an empty tail would leave the sampler without fresh logits.
    let reuse = media_cache.as_ref().and_then(|c| {
        if prompt.len() > c.prompt.len() && prompt.starts_with(c.prompt.as_str()) && img_prefix_ok(c) {
            Some((c.n_past, c.prompt.len(), c.image_keys.len(), c.complete))
        } else if !c.body.is_empty()
            && prompt.len() > c.body.len()
            && prompt.starts_with(c.body.as_str())
            && img_prefix_ok(c)
        {
            // The anchor deliberately points behind the generation tail, so
            // this one always has something to truncate.
            Some((c.n_past_body, c.body.len(), c.image_keys.len(), false))
        } else {
            None
        }
    });
    let (start_past, tail, new_images) = match reuse {
        // Reuse needs the generation tail truncated out of the KV first.
        // Hybrid/recurrent models (Qwen3.6) can't partially rewind their
        // state — seq_rm reports false — so fall back to a clean full
        // prefill (correct, just slower; the frontend caps how many images
        // ride along, so the re-encode cost stays bounded).
        // A complete ledger means the KV ends exactly where the cached prompt
        // does: the new prompt appends to it, so there is nothing to remove and
        // no rewind to ask for. This is what lets a hybrid model reuse at all.
        Some((n_past, prompt_len, n_imgs, true)) => {
            (n_past, &prompt[prompt_len..], &images[n_imgs..])
        }
        Some((n_past, prompt_len, n_imgs, false))
            if ctx.clear_kv_cache_seq(Some(0), Some(n_past as u32), None) == Ok(true) =>
        {
            (n_past, &prompt[prompt_len..], &images[n_imgs..])
        }
        Some(_) => {
            CACHE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            eprintln!("media cache: partial KV removal unsupported by this model — full re-prefill");
            ctx.clear_kv_cache();
            *media_cache = None;
            (0, prompt, &images[..])
        }
        None => {
            ctx.clear_kv_cache();
            *media_cache = None;
            (0, prompt, &images[..])
        }
    };

    // From here until the prefill completes, the KV holds a state no record
    // describes (truncated to the anchor, partially evaluated, …). The ledger
    // is re-established at the end — any early exit (cancel, error) leaves it
    // empty so the NEXT turn starts from a clean cache instead of trusting a
    // stale description of what's in the KV.
    *media_cache = None;

    let clear_all = |ctx: &mut LlamaContext, cache: &mut Option<MediaCache>| {
        ctx.clear_kv_cache();
        *cache = None;
    };

    let mut bitmaps: Vec<MtmdBitmap> = Vec::with_capacity(new_images.len());
    for p in new_images {
        if cancel.load(Ordering::Relaxed) {
            return Ok(start_past.max(0));
        }
        // Oversized screenshots (full-page captures at 2x scale reach tens of
        // megapixels) are downscaled before they hit the vision encoder — the
        // model would compress them internally anyway, and pre-shrinking cuts
        // decode + preprocessing time and caps the visual-token count. The
        // original file is untouched (UI previews stay full-res).
        let feed = downscale_for_vision(p);
        IMG_ENCODES.fetch_add(1, Ordering::Relaxed);
        // `placeholder: false` → decode the actual pixels.
        match MtmdBitmap::from_file(mtmd, &feed, false) {
            Ok(b) => bitmaps.push(b),
            Err(e) => {
                clear_all(ctx, media_cache);
                bail!("无法读取图片 (failed to read image) {p}: {e}");
            }
        }
    }

    // ── Segmented prefill with progress ─────────────────────────────────
    // The helper's eval_chunks() is a single opaque call, so a screenshot
    // turn used to sit silent for seconds (image encode + decode) with no
    // prefill events. Split the tail at the media markers instead — one
    // segment per image — tokenize each piece, then eval them sequentially,
    // emitting progress between segments. Positions stay identical to a
    // whole-tail eval because every split point is a marker (special-token)
    // boundary. The segment containing the BODY boundary (end of the render
    // without the generation tail) is split once more, so the anchor's KV
    // position is observable for the next turn's reuse. If the segment count
    // doesn't line up (defensive), fall back to a single whole-tail eval.
    let marker = mtmd_default_marker();
    let parts: Vec<&str> = tail.split(marker).collect();
    let segmented = parts.len() == new_images.len() + 1;

    // (text, bitmap indices into `bitmaps`)
    let mut seg_texts: Vec<(String, Vec<usize>)> = Vec::new();
    if segmented {
        for (i, part) in parts.iter().enumerate() {
            if i == 0 {
                if !part.is_empty() {
                    seg_texts.push(((*part).to_string(), Vec::new()));
                }
            } else {
                seg_texts.push((format!("{marker}{part}"), vec![i - 1]));
            }
        }
        if seg_texts.is_empty() {
            seg_texts.push((String::new(), Vec::new()));
        }
    } else {
        seg_texts.push((tail.to_string(), (0..bitmaps.len()).collect()));
    }

    // Split the segment containing the body boundary so `n_past` right at the
    // anchor is observable. The boundary is where the generation header
    // starts — a special-token edge, so re-tokenizing the halves separately
    // is BPE-safe. Skipped (anchor disabled) if it would cut a marker.
    let tail_start = prompt.len() - tail.len();
    let mut body_seg: Option<usize> = None;
    if segmented && prompt_body.len() > tail_start && prompt_body.len() <= prompt.len() {
        let rel = prompt_body.len() - tail_start;
        let mut acc = 0usize;
        for i in 0..seg_texts.len() {
            let len = seg_texts[i].0.len();
            if rel == acc {
                // boundary right at a segment edge — previous segment ends the body
                if i > 0 {
                    body_seg = Some(i - 1);
                }
                break;
            }
            if rel < acc + len {
                let off = rel - acc;
                let (txt, bms) = seg_texts[i].clone();
                if !txt.is_char_boundary(off) || (!bms.is_empty() && off < marker.len()) {
                    break; // would cut a marker / char — disable the anchor
                }
                let (a, b) = txt.split_at(off);
                seg_texts[i] = (a.to_string(), bms);
                seg_texts.insert(i + 1, (b.to_string(), Vec::new()));
                body_seg = Some(i);
                break;
            }
            acc += len;
        }
        if body_seg.is_none() && rel >= seg_texts.iter().map(|(t, _)| t.len()).sum::<usize>() {
            // body runs to the very end of the tail (no generation tail?)
            body_seg = Some(seg_texts.len() - 1);
        }
    }

    struct Seg {
        chunks: llama_cpp_2::mtmd::MtmdInputChunks,
        positions: i32,
    }

    let mut segs: Vec<Seg> = Vec::new();
    let mut total_pos: i32 = 0;
    for (i, (text, bm_idx)) in seg_texts.iter().enumerate() {
        let seg_bitmaps: Vec<&MtmdBitmap> = bm_idx.iter().map(|&j| &bitmaps[j]).collect();
        let input = MtmdInputText {
            text: text.clone(),
            // BOS only at the very start of the whole sequence.
            add_special: start_past == 0 && i == 0,
            parse_special: true,
        };
        match mtmd.tokenize(input, &seg_bitmaps) {
            Ok(c) => {
                let p = c.total_positions();
                total_pos += p;
                segs.push(Seg { chunks: c, positions: p });
            }
            Err(e) => {
                clear_all(ctx, media_cache);
                bail!("多模态分词失败 (multimodal tokenization failed): {e}");
            }
        }
    }

    let grand_total = start_past + total_pos;
    if grand_total + 4 >= n_ctx as i32 {
        clear_all(ctx, media_cache);
        bail!(
            "图文提示共 {grand_total} 个位置，超出上下文窗口 {n_ctx}，请新建对话、缩短输入或减少图片。(The multimodal prompt needs {grand_total} positions — over the {n_ctx} context window; start a new chat, shorten the input or drop images.)"
        );
    }
    if cancel.load(Ordering::Relaxed) {
        return Ok(start_past.max(0));
    }

    // Image turns are always worth a ring (encoding takes seconds); text-only
    // media-regime tails follow the text path's rule (skip short ones).
    let report = !new_images.is_empty() || total_pos > n_batch;
    if report {
        let _ = sink.emit(StreamEvent::Prefill {
            processed: start_past.max(0) as u32,
            total: grand_total.max(0) as u32,
        });
    }

    // Encode image chunks + decode text chunks; llama.cpp's helper handles
    // non-causal attention and M-RoPE position bookkeeping per model.
    let mut n_past = start_past;
    let mut n_past_body: i32 = 0;
    let last = segs.len() - 1;
    for (i, seg) in segs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            // Mid-prefill cancel: the KV holds a partial media prefill the
            // cache can't describe — drop it so the next turn starts clean.
            clear_all(ctx, media_cache);
            return Ok(start_past.max(0));
        }
        n_past = match seg.chunks.eval_chunks(mtmd, ctx, n_past, 0, n_batch, i == last) {
            Ok(p) => p,
            Err(e) => {
                clear_all(ctx, media_cache);
                bail!("图文预填充失败 (multimodal prefill failed): {e}");
            }
        };
        let _ = seg.positions; // tracked via n_past
        if body_seg == Some(i) {
            n_past_body = n_past;
        }
        if report {
            let _ = sink.emit(StreamEvent::Prefill {
                processed: n_past.max(0) as u32,
                total: grand_total.max(0) as u32,
            });
        }
    }

    let anchored = body_seg.is_some() && n_past_body > 0;
    *media_cache = Some(MediaCache {
        prompt: prompt.to_string(),
        image_keys,
        n_past,
        body: if anchored { prompt_body.to_string() } else { String::new() },
        n_past_body,
        // The reply has not been generated yet; the generation loop marks this
        // true once it has recorded what it decoded.
        complete: false,
    });
    Ok(n_past)
}

/// Feed-side image budget for the vision encoder. Full-page browser
/// screenshots (2x device scale, many viewports tall) reach 10-25 MP; the
/// encoder's preprocessing would shrink them internally anyway, so feeding a
/// pre-shrunk copy is pure speed (and it caps visual tokens / context use).
const MAX_VISION_PIXELS: u64 = 2_000_000;

/// Return a path whose image is at most `max_pixels`: the original path if
/// it's already small enough (or unreadable — let mtmd report that), else a
/// cached downscaled JPEG in the temp dir keyed by path+size+mtime.
///
/// Ordering is the point (owner-reported memory spikes): the size check
/// reads only the file HEADER, and the cache key needs only fs metadata —
/// so the full pixel decode (a 41 MP screenshot costs ~120 MB of transient
/// RSS, paid inside the same process the GGUF model lives in) happens ONLY
/// on a true cache miss. The agent re-sends its attached screenshots on
/// every step; decoding them anew each step stacked transient spikes onto an
/// already-full machine and got the MLX sidecar jetsammed.
pub(crate) fn downscale_for_vision(path: &str) -> String {
    downscale_for_vision_capped(path, MAX_VISION_PIXELS)
}

pub(crate) fn downscale_for_vision_capped(path: &str, max_pixels: u64) -> String {
    // 1. Header-only dimensions — no pixel decode.
    let Ok((w32, h32)) = image::image_dimensions(path) else { return path.to_string() };
    let (w, h) = (w32 as u64, h32 as u64);
    if w * h <= max_pixels {
        return path.to_string();
    }
    // 2. Cache lookup from fs metadata alone — still no decode.
    use std::hash::{Hash, Hasher};
    let meta = std::fs::metadata(path).ok();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    max_pixels.hash(&mut hasher);
    meta.as_ref().map(|m| m.len()).unwrap_or(0).hash(&mut hasher);
    meta.and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .hash(&mut hasher);
    let dir = std::env::temp_dir().join("chaty-vision-imgs");
    let _ = std::fs::create_dir_all(&dir);
    let out = dir.join(format!("{:016x}.jpg", hasher.finish()));
    if out.is_file() {
        return out.to_string_lossy().to_string();
    }
    // 3. True miss — decode once, downscale, cache.
    let Ok(img) = image::open(path) else { return path.to_string() };
    let scale = ((max_pixels as f64) / ((w * h) as f64)).sqrt();
    let nw = ((img.width() as f64 * scale) as u32).max(1);
    let nh = ((img.height() as f64 * scale) as u32).max(1);
    let resized = img.resize(nw, nh, image::imageops::FilterType::Triangle);
    // JPEG has no alpha channel — flatten to RGB.
    let rgb = image::DynamicImage::ImageRgb8(resized.to_rgb8());
    match rgb.save_with_format(&out, image::ImageFormat::Jpeg) {
        Ok(()) => {
            eprintln!(
                "vision: downscaled {path} {}x{} -> {nw}x{nh} for the encoder",
                img.width(),
                img.height()
            );
            out.to_string_lossy().to_string()
        }
        Err(_) => path.to_string(),
    }
}

fn done_event(
    sink: &dyn EventSink,
    prompt_tokens: u32,
    completion_tokens: u32,
    tps: f32,
    stop_reason: &str,
) -> Result<()> {
    done_event_reused(sink, prompt_tokens, completion_tokens, tps, stop_reason, 0)
}

fn done_event_reused(
    sink: &dyn EventSink,
    prompt_tokens: u32,
    completion_tokens: u32,
    tps: f32,
    stop_reason: &str,
    reused: u32,
) -> Result<()> {
    sink.emit(StreamEvent::Done {
        stats: GenStats {
            prompt_tokens,
            completion_tokens,
            tokens_per_second: tps,
            stop_reason: stop_reason.to_string(),
            reused,
        },
    })?;
    Ok(())
}

/// Render messages into a prompt using the model's embedded chat template,
/// falling back to ChatML if the GGUF doesn't carry one.
fn build_prompt(model: &LlamaModel, messages: &[ChatMessage], think: Option<bool>) -> Result<String> {
    build_prompt_pair(model, messages, think).map(|(full, _)| full)
}

/// Render the prompt twice: the FULL prompt (generation header + any thinking
/// prefill — what actually gets prefilled), and the BODY (the same render
/// WITHOUT the generation tail). The body is the media-cache anchor: the next
/// turn's prompt always re-renders history identically, so it string-extends
/// the body even when the generation tail (e.g. Qwen3.5+'s `<think>` prefill)
/// diverges from how the assistant turn is later re-rendered. Truncating the
/// KV back to the body costs a handful of tokens — re-encoding every image
/// (the old behavior on any tail divergence) cost seconds per turn.
/// The empty reasoning block Chaty prefills when thinking is off, as it must
/// appear at the head of a stored assistant turn. Kept next to the injection
/// site below — the two spellings have to stay identical.
const THINK_OFF_PREFIX: &str = "<think>\n\n</think>\n\n";

/// Put that block back in front of prior assistant turns, so re-rendering the
/// conversation reproduces the prompt the model actually continued from. A turn
/// that already opens with a reasoning block is left alone — the model wrote one
/// despite the request, and doubling it would break the prefix just as badly.
fn with_think_off_prefix(
    model: &LlamaModel,
    messages: &[ChatMessage],
    think: Option<bool>,
) -> Vec<ChatMessage> {
    prefixed_assistant_turns(messages, think == Some(false) && template_uses_think(model))
}

/// The message rewrite itself, with the model question already answered so the
/// rule is testable on its own.
fn prefixed_assistant_turns(messages: &[ChatMessage], apply: bool) -> Vec<ChatMessage> {
    if !apply {
        return messages.to_vec();
    }
    messages
        .iter()
        .map(|m| {
            if !matches!(m.role, Role::Assistant) || m.content.trim_start().starts_with("<think>") {
                return m.clone();
            }
            ChatMessage {
                content: format!("{THINK_OFF_PREFIX}{}", m.content),
                ..m.clone()
            }
        })
        .collect()
}

/// Does this model's template use the `<think>` convention? Architecture is
/// authoritative: Qwen3.5/3.6 use it even when a finetune ships a custom
/// template without the markers.
fn template_uses_think(model: &LlamaModel) -> bool {
    let arch = model
        .meta_val_str("general.architecture")
        .map(|a| is_think_paradigm_arch(&a.to_lowercase()))
        .unwrap_or(false);
    arch || model
        .meta_val_str("tokenizer.chat_template")
        .map(|t| t.contains("<think>"))
        .unwrap_or(false)
}

fn build_prompt_pair(
    model: &LlamaModel,
    messages: &[ChatMessage],
    think: Option<bool>,
) -> Result<(String, String)> {
    // Gemma 4 ships a Jinja template the vendored llama.cpp can't parse, and
    // the old built-in "gemma" template uses the wrong (<start_of_turn>) turn
    // delimiters — render its documented format natively instead.
    if is_gemma4(model) {
        return Ok((
            render_gemma4(messages, think, true),
            render_gemma4(messages, think, false),
        ));
    }
    // What Chaty appends after the assistant header when thinking is off (see
    // below) has to appear in front of every STORED assistant turn too, or the
    // next prompt is not an append of the last one. Round one ends with
    // `…assistant\n<think>\n\n</think>\n\n` and the model continues from there;
    // if round two renders that same turn as `…assistant\n<answer>`, the common
    // prefix ends at the header and every turn after the first pays a full
    // prefill. Measured at 0% KV reuse on qwen35 and lfm2 with thinking off —
    // which is the default in code mode.
    let messages = with_think_off_prefix(model, messages, think);
    let messages = messages.as_slice();
    let body = render_chat(model, messages, false).unwrap_or_default();
    let mut prompt = render_chat(model, messages, true)?;

    // Qwen3.5+ dropped the `/no_think` soft switch and default to reasoning. To
    // honour a "thinking off" request we pre-fill an empty reasoning block right
    // after the assistant header, the same way the Qwen template does when
    // `enable_thinking=false` — the model then skips straight to the answer.
    // Only do this for models whose template actually uses the `<think>`
    // convention: injecting it into e.g. channel-style reasoners (Gemma 4)
    // feeds them tokens they never saw in training and triggers degenerate
    // repetition loops.
    // Architecture is authoritative: Qwen3.5/3.6 use the <think> paradigm
    // even when a finetune ships a legacy/custom template without the markers.
    let qwen35plus = model
        .meta_val_str("general.architecture")
        .map(|a| is_qwen3_5_plus_arch(&a.to_lowercase()))
        .unwrap_or(false);
    let template_uses_think = template_uses_think(model);
    // Thinking ON for a model whose official template PRE-OPENS the block
    // after the assistant header (Qwen3.5/3.6: literal `'<think>\n'` in the
    // generation section — Qwen3 emits the tag itself and must NOT get this):
    // llama.cpp's generic ChatML fallback renderer drops the pre-open, and
    // pre-open-trained models don't re-emit the tag, so their reasoning
    // streams untagged. Restore the official shape ourselves.
    let template_preopens_think = qwen35plus
        || model
            .meta_val_str("tokenizer.chat_template")
            // Converters store the jinja `'<think>\n'` either with a literal
            // backslash-n or a real newline — accept both spellings.
            .map(|t| t.contains("'<think>\\n'") || t.contains("'<think>\n'"))
            .unwrap_or(false);
    if think != Some(false) && template_preopens_think {
        let tail = prompt.trim_end();
        if !tail.ends_with("<think>") && !tail.ends_with("</think>") {
            if !prompt.ends_with('\n') {
                prompt.push('\n');
            }
            prompt.push_str("<think>\n");
        }
    }
    if think == Some(false) && template_uses_think {
        let tail = prompt.trim_end();
        let already_closed = tail.ends_with("</think>");
        let already_open = tail.ends_with("<think>");
        if !already_closed {
            let needs_newline = !prompt.ends_with('\n');
            if needs_newline {
                prompt.push('\n');
            }
            if already_open {
                // The template already opened a reasoning block — just close it.
                prompt.push_str("\n</think>\n\n");
            } else {
                prompt.push_str(THINK_OFF_PREFIX);
            }
        }
    }

    // The anchor only works if it really is a prefix of the full render
    // (true for every sane template; guard against odd ones).
    let body = if prompt.starts_with(&body) { body } else { String::new() };
    Ok((prompt, body))
}

/// Qwen 3.5 and everything after it (3.6, 3.8, …) share one paradigm: no
/// `/no_think` soft switch, a pre-opened `<think>` block. Parse the minor
/// version out of the GGUF arch string instead of listing each release —
/// `qwen35`, `qwen36`, `qwen38` all qualify, `qwen3`/`qwen3moe` do not.
/// Does this architecture reason in `<think>` blocks at all? The whole Qwen3
/// line does — 3, 3.5, 3.6, 3.8 — and that is a property of the model, not of
/// whatever chat template a finetune happened to ship. Some ship none: the
/// abliterated Qwen3 4B carries an EMPTY template, so llama.cpp renders it with
/// its ChatML fallback and every think-off mechanism that keyed off template
/// text silently did nothing. Thinking could not be turned off on that model at
/// all, in either mode, and the only sign was a slow reply.
pub(crate) fn is_think_paradigm_arch(arch_lc: &str) -> bool {
    arch_lc.starts_with("qwen3")
}

pub(crate) fn is_qwen3_5_plus_arch(arch_lc: &str) -> bool {
    let Some(rest) = arch_lc.strip_prefix("qwen3") else { return false };
    match rest.chars().next() {
        Some(d) if d.is_ascii_digit() => d >= '5',
        _ => false,
    }
}

/// The three sentences Qwen3.8's chat template injects for its
/// `reasoning_effort` ladder, verbatim from the official template. `medium`
/// deliberately injects nothing — it is the neutral baseline.
pub(crate) const EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
pub(crate) const EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

/// The ladder a template offers, weakest first — detected from the template
/// text, never from the model name (finetunes rename freely, and a template
/// that takes the kwarg is exactly the set of models that honour it).
/// Does delivering a tool result under its own role keep the next prompt an
/// APPEND onto the last one?
///
/// That is the property the KV cache actually needs: round two must begin with
/// round one's prompt followed by exactly the text the model generated on top
/// of it. Templates decide "does this turn still belong to the request being
/// answered" from the index of the last *user* message, so a result wearing
/// the user role pushes that index past every assistant turn — some templates
/// then drop their reasoning, others re-wrap the stored turn inside an empty
/// thinking block. Either way the prompt no longer reproduces what the model
/// just wrote, the prefix dies at the first assistant turn, and a model whose
/// memory cannot rewind re-reads the entire conversation every step.
///
/// Testing the append property directly, rather than looking for reasoning in
/// the output, is what distinguishes a template that genuinely preserves the
/// turn from one that merely passes the markup through as content.
fn probe_tool_role(model: &LlamaModel) -> bool {
    const REASONED: &str = "PROBE_REASONING\n</think>\n\nPROBE_ANSWER";
    let msg = |role: Role, content: &str| ChatMessage {
    reasoning_content: None,
        role,
        content: content.into(),
        images: vec![],
    };
    let opening = [msg(Role::System, "s"), msg(Role::User, "q")];
    let Ok(first) = render_chat(model, &opening, true) else { return false };
    // A template that pre-opens the thinking block supplies that tag itself, so
    // the model's own output starts after it — and the loop stores the turn the
    // same way, tag included.
    // The stored turn always carries the opening tag; what the model *generated*
    // does not when the template pre-opened it. Both conventions exist (Qwen3.5
    // pre-opens, Qwen3 emits the tag itself), and getting this backwards makes
    // the probe test a shape that never occurs.
    let stored = format!("<think>\n{REASONED}");
    let generated = if first.trim_end().ends_with("<think>") {
        REASONED.to_string()
    } else {
        stored.clone()
    };
    let appends = |tool_role: Role| {
        let second = [
            opening[0].clone(),
            opening[1].clone(),
            msg(Role::Assistant, &stored),
            msg(tool_role, "result"),
        ];
        render_chat(model, &second, true)
            .map(|p| p.starts_with(&format!("{first}{generated}")))
            .unwrap_or(false)
    };
    appends(Role::Tool) && !appends(Role::User)
}

pub(crate) fn effort_levels_of(template: &str) -> Vec<String> {
    if !template.contains("reasoning_effort") {
        return Vec::new();
    }
    ["low", "medium", "xhigh"]
        .iter()
        .filter(|lvl| template.contains(&format!("'{lvl}'")) || template.contains(&format!("\"{lvl}\"")))
        .map(|s| s.to_string())
        .collect()
}

/// llama.cpp renders the chat template with DEFAULT kwargs, so a thinking
/// Qwen3.8 prompt already carries the `xhigh` sentence. Requesting another
/// rung is therefore a verbatim substitution on the rendered prompt — the
/// result is byte-identical to what the official template produces for that
/// rung (`medium` = the sentence removed, its empty-instruction branch).
/// Anything unexpected (finetuned template, fallback renderer that never
/// emitted the sentence) leaves the prompt untouched.
pub(crate) fn apply_effort(prompt: &str, effort: &str) -> String {
    if !prompt.contains(EFFORT_XHIGH) {
        return prompt.to_string();
    }
    match effort {
        "xhigh" => prompt.to_string(),
        "low" => prompt.replace(EFFORT_XHIGH, EFFORT_LOW),
        // The template emits `instructions + '\n\n'` before the system body,
        // or a system block holding only the sentence — drop both shapes.
        "medium" => prompt
            .replace(&format!("<|im_start|>system\n{EFFORT_XHIGH}<|im_end|>\n"), "")
            .replace(&format!("{EFFORT_XHIGH}\n\n"), ""),
        _ => prompt.to_string(),
    }
}

/// Gemma 4 uses `<|turn>role\n…<turn|>` turn delimiters (the template string
/// is the reliable marker — the arch string varies across conversions).
fn is_gemma4(model: &LlamaModel) -> bool {
    model
        .meta_val_str("tokenizer.chat_template")
        .map(|t| t.contains("<|turn>"))
        .unwrap_or(false)
}

/// Native renderer for Gemma 4's documented chat format:
///
/// ```text
/// {bos}<|turn>system\n[<|think|>\n]SYSTEM<turn|>\n
/// <|turn>user\nUSER<turn|>\n
/// <|turn>model\nREPLY<turn|>\n
/// <|turn>model\n            ← generation prompt
/// ```
///
/// Thinking defaults ON and is controlled by the `<|think|>` token at the
/// start of the system turn; the model then emits
/// `<|channel>thought\n…<channel|>` before its answer. (BOS is added by the
/// tokenizer via `AddBos::Always`.)
fn render_gemma4(messages: &[ChatMessage], think: Option<bool>, add_gen: bool) -> String {
    let think_on = think != Some(false);
    let sys_text = messages
        .iter()
        .filter(|m| matches!(m.role, Role::System))
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut p = String::new();
    if think_on || !sys_text.is_empty() {
        p.push_str("<|turn>system\n");
        if think_on {
            p.push_str("<|think|>\n");
        }
        p.push_str(&sys_text);
        p.push_str("<turn|>\n");
    }
    for m in messages {
        let role = match m.role {
            Role::System => continue,
            // Gemma's format has no tool turn — a result is spoken by the user,
            // which is exactly the shape this renderer already produced.
            Role::User | Role::Tool => "user",
            Role::Assistant => "model",
        };
        // A turn keeps the channel it was written in.
        //
        // This is a DELIBERATE departure from Google's documented contract, so
        // do not quietly restore the strip. Their guidance is in two parts:
        // thoughts must NOT be removed between the tool calls of one model turn,
        // and they must be removed from previous turns before the next one.
        // Gemma's own template obeys neither half cleanly — `strip_thinking`
        // runs over every model message unconditionally, so it breaks the
        // tool-call half, which is what made an agent step re-read its whole
        // transcript.
        //
        // Keeping the whole channel fixes that half and costs the other: 13%
        // cache reuse across four chat turns became 100%. The owner ran it and
        // found no drop in answer quality with reasoning in history, and took
        // the trade knowingly. Compaction reclaims stale reasoning when the
        // window tightens, which is also what Google suggests for long sessions
        // — carry the substance, not the raw trace.
        //
        // https://ai.google.dev/gemma/docs/capabilities/thinking
        let content = m.content.clone();
        p.push_str("<|turn>");
        p.push_str(role);
        p.push('\n');
        p.push_str(content.trim());
        p.push_str("<turn|>\n");
    }
    if add_gen {
        p.push_str("<|turn>model\n");
    }
    p
}

/// Apply the chat template with a robust fallback chain:
/// 1. the GGUF's embedded template as-is;
/// 2. the embedded template with system messages folded into the first user
///    turn — Gemma-family templates raise "system role not supported";
/// 3. llama.cpp's *built-in* template for the architecture — newer models
///    (e.g. Gemma 3/4) often embed Jinja the vendored llama.cpp can't parse
///    even though the wire format is unchanged;
/// 4. ChatML as a last resort.
fn render_chat(model: &LlamaModel, messages: &[ChatMessage], add_ass: bool) -> Result<String> {
    fn to_chat(msgs: &[ChatMessage]) -> Result<Vec<LlamaChatMessage>> {
        msgs.iter()
            .map(|m| LlamaChatMessage::new(role_str(&m.role).to_string(), m.content.clone()))
            .collect::<std::result::Result<_, _>>()
            .context("invalid message content")
    }

    let chat = to_chat(messages)?;
    let folded = fold_system(messages);
    let folded_chat = to_chat(&folded)?;

    if let Ok(t) = model.chat_template(None) {
        if let Ok(p) = model.apply_chat_template(&t, &chat, add_ass) {
            return Ok(p);
        }
        if let Ok(p) = model.apply_chat_template(&t, &folded_chat, add_ass) {
            eprintln!("chat template rejected the system role; folded it into the user turn");
            return Ok(p);
        }
    }

    let arch = model
        .meta_val_str("general.architecture")
        .unwrap_or_default()
        .to_lowercase();
    for name in builtin_template_candidates(&arch) {
        if let Ok(t) = LlamaChatTemplate::new(name) {
            if let Ok(p) = model.apply_chat_template(&t, &folded_chat, add_ass) {
                eprintln!("embedded chat template unusable; using built-in '{name}' (arch: {arch})");
                return Ok(p);
            }
        }
    }
    bail!("no usable chat template (arch: {arch})")
}

/// Merge any system messages into the first user turn (for templates that
/// reject the system role). Returns the messages unchanged if there are none.
fn fold_system(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let sys: Vec<&str> = messages
        .iter()
        .filter(|m| matches!(m.role, Role::System))
        .map(|m| m.content.as_str())
        .collect();
    if sys.is_empty() {
        return messages.to_vec();
    }
    let sys_text = sys.join("\n\n");
    let mut out = Vec::with_capacity(messages.len());
    let mut injected = false;
    for m in messages {
        match m.role {
            Role::System => {}
            Role::User if !injected => {
                injected = true;
                out.push(ChatMessage { images: Vec::new(),
                    role: Role::User,
                    content: format!("{sys_text}\n\n{}", m.content),
                    reasoning_content: None,
                });
            }
            _ => out.push(m.clone()),
        }
    }
    if !injected {
        out.insert(
            0,
            ChatMessage {
                images: Vec::new(),
                role: Role::User,
                content: sys_text,
                reasoning_content: None,
            },
        );
    }
    out
}

/// llama.cpp built-in template names worth trying for an architecture, in
/// order. These are the stable wire formats; "chatml" is the universal layout
/// most instruction-tuned models tolerate.
fn builtin_template_candidates(arch: &str) -> &'static [&'static str] {
    if arch.starts_with("gemma") {
        &["gemma", "chatml"]
    } else if arch.starts_with("llama") {
        &["llama3", "chatml"]
    } else if arch.starts_with("mistral") {
        &["mistral-v7", "chatml"]
    } else if arch.starts_with("phi") {
        &["phi4", "chatml"]
    } else {
        &["chatml"]
    }
}

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Raw bytes of a token's piece, handling pieces longer than the initial
/// buffer.
///
/// `special = true`: a control token renders as its literal text rather than as
/// nothing. This used to be `false`, which quietly deleted information the rest
/// of Chaty needs. A turn's streamed text is what the NEXT turn's prompt is
/// rebuilt from, so a token whose piece renders empty is a token the next
/// prompt cannot reproduce, and the KV prefix diverges at exactly that
/// position — every later turn paid a full prefill. Worse, LFM2 emits its tool
/// calls as `<|tool_call_start|>[read_file(path='x')]<|tool_call_end|>` no
/// matter what format the system prompt asks for: with the markers deleted the
/// text read as ordinary prose and the tool call never fired at all.
///
/// The MLX engine has always kept them (swift-transformers decodes with
/// `skipSpecialTokens: false`), and the front end already normalises such
/// markers away for display in `normalizeChannels` — this restores the same
/// contract on the GGUF side: the engine streams what the model actually
/// produced, and the front end decides what a person sees.
///
/// End-of-generation tokens never reach here — the sampling loop breaks on
/// `is_eog_token` first.
fn piece_bytes(model: &LlamaModel, token: LlamaToken) -> Vec<u8> {
    match model.token_to_piece_bytes(token, 32, true, None) {
        Ok(b) => b,
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(i)) => model
            .token_to_piece_bytes(token, (-i) as usize, true, None)
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Heuristic: does this llama.cpp error look like an out-of-memory failure?
fn is_oom(msg: &str) -> bool {
    let m = msg.to_lowercase();
    [
        "out of memory",
        "outofmemory",
        "out of device memory",
        "failed to allocate",
        "cannot allocate",
        "unable to allocate",
        "insufficient memory",
        "bad_alloc",
        "vk_error_out_of",
        "erroroutofdevicememory",
        "ggml_vk_create_buffer",
        "alloc_buffer",
    ]
    .iter()
    .any(|k| m.contains(k))
}

/// Next-lower GPU layer count when an attempt OOMs (halve, then drop to CPU).
fn backoff_layers(layers: i32) -> i32 {
    if layers > 8 {
        layers / 2
    } else {
        0
    }
}

/// Resolve the context window to load with. Honours a user preference (clamped to
/// the model's trained length and never below 512); otherwise defaults to a
/// memory-friendly 8192 cap even when the model was trained far longer.
fn clamp_n_ctx(trained: u32, pref: Option<u32>) -> u32 {
    let ceiling = trained.max(512);
    match pref {
        Some(n) if n > 0 => n.clamp(512, ceiling),
        // Auto: on macOS ask for the model's full trained length — the KV
        // memory clamp (mem_safe_n_ctx) then fits it to unified memory, which
        // is the meaningful "auto". Elsewhere keep the conservative default.
        #[cfg(target_os = "macos")]
        _ => ceiling,
        #[cfg(not(target_os = "macos"))]
        _ => trained.clamp(512, 8192),
    }
}

/// Estimated KV-cache bytes per context token (K + V, f16), from the model's
/// GQA geometry. 0 if the metadata is missing.
fn kv_bytes_per_token(model: &LlamaModel) -> u64 {
    let arch = model.meta_val_str("general.architecture").unwrap_or_default();
    let meta_u64 = |key: &str| -> Option<u64> {
        model
            .meta_val_str(key)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
    };
    let n_layer = model.n_layer() as u64;
    let n_embd = model.n_embd() as u64;
    let n_head = meta_u64(&format!("{arch}.attention.head_count")).unwrap_or(1).max(1);
    let n_head_kv = meta_u64(&format!("{arch}.attention.head_count_kv")).unwrap_or(n_head);
    let n_embd_gqa = n_embd / n_head * n_head_kv;
    n_layer * n_embd_gqa * 2 /* K+V */ * 2 /* f16 bytes */
}

/// Cap the context length so weights + KV cache fit in the memory budget.
///
/// On unified memory (Apple Silicon) an oversized KV cache does NOT fail fast —
/// the allocation succeeds, the machine swaps, and the system freezes. The OOM
/// back-off never fires, so we must clamp *proactively* here. On discrete GPUs
/// allocations fail fast and the existing back-off handles it, so this is a
/// no-op off macOS.
fn mem_safe_n_ctx(model: &LlamaModel, n_ctx: u32) -> u32 {
    #[cfg(target_os = "macos")]
    {
        let per_tok = kv_bytes_per_token(model);
        let Some(gpu) = crate::gpu::detect_gpu() else { return n_ctx };
        if per_tok == 0 {
            return n_ctx;
        }
        let budget = gpu.vram_mb * 1024 * 1024; // Metal working-set size
        // Headroom for compute buffers, the app, WebView and the OS.
        let headroom = 2u64 * 1024 * 1024 * 1024;
        let avail = budget.saturating_sub(model.size()).saturating_sub(headroom);
        let fit = (avail / per_tok) as u32;
        // Round down to a 256 boundary; never go below a usable minimum.
        let fit = (fit / 256 * 256).max(2048);
        if fit < n_ctx {
            eprintln!(
                "clamping n_ctx {n_ctx} -> {fit} to fit unified memory \
                 (weights {} MiB + KV {} KiB/token, budget {} MiB)",
                model.size() / (1024 * 1024),
                per_tok / 1024,
                budget / (1024 * 1024),
            );
            return fit;
        }
        n_ctx
    }
    #[cfg(not(target_os = "macos"))]
    {
        n_ctx
    }
}

/// Map a GGUF `general.file_type` enum to a readable quant name.
fn quant_name(ft: u32) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        _ => "mixed",
    }
}

fn build_sampler(params: &GenParams) -> LlamaSampler {
    let seed = params.seed.map_or(0xFFFF_FFFF, |s| s as u32);
    // Repetition penalty applies to greedy and sampled decoding alike.
    let repeat = if params.repeat_penalty > 0.0 {
        params.repeat_penalty
    } else {
        1.0
    };
    let penalties = LlamaSampler::penalties(64, repeat, 0.0, 0.0);
    if params.temperature <= 0.0 {
        LlamaSampler::chain_simple([penalties, LlamaSampler::greedy()])
    } else {
        let top_k = if params.top_k == 0 {
            -1
        } else {
            params.top_k as i32
        };
        LlamaSampler::chain_simple([
            penalties,
            LlamaSampler::top_k(top_k),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::min_p(params.min_p.max(0.0), 1),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::dist(seed),
        ])
    }
}

#[cfg(test)]
mod tests {

    /// The Qwen3.5+ paradigm test parses the minor version instead of listing
    /// releases — 3.8 must qualify the day it ships, `qwen3`/`qwen3moe` must
    /// not (they still use the `/no_think` soft switch).
    #[test]
    fn qwen3_family_predicate_reads_the_minor_version() {
        for a in ["qwen35", "qwen35moe", "qwen36", "qwen38", "qwen39moe"] {
            assert!(super::is_qwen3_5_plus_arch(a), "{a} should be 3.5+");
        }
        for a in ["qwen3", "qwen3moe", "qwen2", "qwen34", "llama", "qwen"] {
            assert!(!super::is_qwen3_5_plus_arch(a), "{a} should NOT be 3.5+");
        }
    }

    /// The effort ladder is detected from the template text (never the model
    /// name), and a rung request rewrites the rendered prompt to exactly what
    /// the official template emits for that rung.
    /// The wire name a template sees. The sidecar used to fold every role it
    /// did not recognise into `user`, which silently undid the whole point.
    #[test]
    fn tool_role_goes_over_the_wire_as_tool() {
        assert_eq!(super::role_str(&Role::Tool), "tool");
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
        let back: Role = serde_json::from_str("\"tool\"").unwrap();
        assert!(matches!(back, Role::Tool));
    }

    /// Gemma's format has no tool turn, so a result is spoken by the user —
    /// exactly the bytes this renderer produced before the role existed.
    #[test]
    fn gemma4_renders_a_tool_result_exactly_as_it_did_a_user_turn() {
        let msg = |role: Role, text: &str| ChatMessage {
            role,
            content: text.into(),
            images: vec![],
            reasoning_content: None,
};
        let with = |role: Role| {
            super::render_gemma4(
                &[
                    msg(Role::System, "s"),
                    msg(Role::User, "q"),
                    msg(Role::Assistant, "a"),
                    msg(role, "<tool_result name=\"ls\">x</tool_result>"),
                ],
                Some(true),
                true,
            )
        };
        assert_eq!(with(Role::Tool), with(Role::User));
    }

    #[test]
    fn reasoning_effort_ladder_detect_and_apply() {
        let tmpl = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\
                    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}";
        assert_eq!(super::effort_levels_of(tmpl), vec!["low", "medium", "xhigh"]);
        // Templates without the kwarg have no ladder — the UI keeps on/off.
        assert!(super::effort_levels_of("{% if enable_thinking %}<think>{% endif %}").is_empty());

        // A rendered prompt as llama.cpp produces it (default kwargs ⇒ xhigh).
        let rendered = format!(
            "<|im_start|>system\n{}\n\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
            super::EFFORT_XHIGH
        );
        // xhigh is what the template already rendered — byte-identical no-op.
        assert_eq!(super::apply_effort(&rendered, "xhigh"), rendered);
        // low swaps the sentence, keeping everything else identical.
        let low = super::apply_effort(&rendered, "low");
        assert!(low.contains(super::EFFORT_LOW), "{low}");
        assert!(!low.contains(super::EFFORT_XHIGH));
        assert!(low.contains("You are helpful."));
        assert_eq!(low.matches("<|im_start|>system").count(), 1);
        // medium is the template's empty-instruction branch: sentence gone,
        // system body intact.
        let med = super::apply_effort(&rendered, "medium");
        assert!(!med.contains("Reasoning effort is set to"), "{med}");
        assert!(med.contains("<|im_start|>system\nYou are helpful.<|im_end|>"), "{med}");
        // A system block holding ONLY the sentence disappears entirely.
        let only = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n",
            super::EFFORT_XHIGH
        );
        assert_eq!(
            super::apply_effort(&only, "medium"),
            "<|im_start|>user\nhi<|im_end|>\n"
        );
        // Unknown rungs and prompts the sentence never reached are untouched.
        assert_eq!(super::apply_effort(&rendered, "bogus"), rendered);
        assert_eq!(super::apply_effort("plain prompt", "low"), "plain prompt");
    }

    use super::*;

    /// The crash-guard state machine (issue #5): a leftover inflight marker
    /// means the previous load killed the process → promote it to the
    /// persistent block; a clean dir stays unblocked; the block persists.
    /// Broken conversions must be flagged; official files and ordinary
    /// llama models must not. MiniCPM5's tell is the pre-tokenizer (its
    /// llama arch is legitimate); MiniCPM 1–3's tell is the arch itself.
    #[test]
    fn conversion_suspect_flags_wrong_converter() {
        // MiniCPM5 GGUFs (official included) degenerate on the bundled
        // engine — the whole family is flagged until the engine catches up.
        assert!(conversion_suspect("MiniCPM5-1B-F16.gguf MiniCPM5 1B", "llama", "llama-bpe"));
        assert!(conversion_suspect("MiniCPM5-1B-F16.gguf MiniCPM5 1B", "llama", "minicpm5"));
        // Older MiniCPM families need their own arch.
        assert!(conversion_suspect("minicpm-2b.Q4.gguf ", "llama", "llama-bpe"));
        assert!(!conversion_suspect("MiniCPM3-4B.gguf MiniCPM3", "minicpm3", "minicpm3"));
        // Ordinary models never flag.
        assert!(!conversion_suspect("Llama-3-8B.gguf Meta Llama 3", "llama", "llama-bpe"));
        assert!(!conversion_suspect("Qwen3.5-0.8B.gguf Qwen", "qwen3", "qwen2"));
    }

    #[test]
    fn gpu_crash_guard_state_machine() {
        let base = std::env::temp_dir().join(format!("chaty-gpu-guard-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        assert!(!gpu_guard_check(&base), "clean dir must not block");
        std::fs::write(base.join(GPU_INFLIGHT), "loading").unwrap();
        assert!(gpu_guard_check(&base), "stale inflight must block");
        assert!(!base.join(GPU_INFLIGHT).exists(), "inflight must be consumed");
        assert!(base.join(GPU_BLOCKED).exists(), "block must persist");
        assert!(gpu_guard_check(&base), "block persists across restarts");
        std::fs::remove_dir_all(&base).ok();
    }

    /// LoadGuard removes its marker on drop — Ok AND Err paths both clean
    /// up; only a process death leaves it behind. Tested against a TEMP dir:
    /// the first version armed the REAL app-data dir and raced the running
    /// app into a false gpu-blocked promotion on the dev machine.
    #[test]
    fn load_guard_cleans_up_on_drop() {
        let base = std::env::temp_dir().join(format!("chaty-loadguard-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let g = LoadGuard::arm_at(&base);
        let p = g.0.clone().unwrap();
        assert!(p.exists());
        drop(g);
        assert!(!p.exists());
        std::fs::remove_dir_all(&base).ok();
    }

    // ---- find_mmproj: the vision folder-layout pairing rules ----

    fn mk(dir: &std::path::Path, name: &str) {
        std::fs::write(dir.join(name), b"x").unwrap();
    }
    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("chaty-mmproj-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn pairs_mmproj_in_dedicated_folder() {
        let d = tmp("folder");
        mk(&d, "Qwen3.5-4B-Q4_K_M.gguf");
        mk(&d, "mmproj-F16.gguf");
        let got = find_mmproj(&d.join("Qwen3.5-4B-Q4_K_M.gguf").to_string_lossy());
        assert_eq!(got, Some(d.join("mmproj-F16.gguf")));
    }

    #[test]
    fn prefers_smaller_mmproj() {
        let d = tmp("smaller");
        mk(&d, "model.gguf");
        std::fs::write(d.join("mmproj-F32.gguf"), vec![0u8; 64]).unwrap();
        std::fs::write(d.join("mmproj-F16.gguf"), vec![0u8; 8]).unwrap();
        let got = find_mmproj(&d.join("model.gguf").to_string_lossy());
        assert_eq!(got, Some(d.join("mmproj-F16.gguf")));
    }

    #[test]
    fn crowded_folder_requires_name_affinity() {
        let d = tmp("crowded");
        mk(&d, "gemma-4-E4B-it-Q4.gguf");
        mk(&d, "OtherModel-Q4.gguf");
        mk(&d, "mmproj-gemma-4-E4B-F16.gguf");
        // gemma main ↔ gemma mmproj pair by shared stem token
        let got = find_mmproj(&d.join("gemma-4-E4B-it-Q4.gguf").to_string_lossy());
        assert_eq!(got, Some(d.join("mmproj-gemma-4-E4B-F16.gguf")));
        // the unrelated model must NOT pair with it
        let other = find_mmproj(&d.join("OtherModel-Q4.gguf").to_string_lossy());
        assert_eq!(other, None);
    }

    #[test]
    fn no_mmproj_means_none_and_mmproj_is_not_a_model() {
        let d = tmp("none");
        mk(&d, "model.gguf");
        assert_eq!(find_mmproj(&d.join("model.gguf").to_string_lossy()), None);
        mk(&d, "mmproj-F16.gguf");
        // asking for the mmproj itself never pairs
        assert_eq!(find_mmproj(&d.join("mmproj-F16.gguf").to_string_lossy()), None);
    }
}

/// End-to-end agent-loop test against a REAL model. Ignored by default (needs a
/// GGUF). Run with:
///   CHATY_TEST_MODEL=/path/to/model.gguf cargo test --release -p chaty --lib \
///     agent_e2e -- --ignored --nocapture
#[cfg(test)]
mod agent_e2e {
    use super::*;
    use std::cell::RefCell;

    /// Collects streamed tokens in-process (the headless analogue of the IPC channel).
    struct Collector {
        buf: RefCell<String>,
    }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev {
                self.buf.borrow_mut().push_str(&text);
            }
            Ok(())
        }
    }

    fn strip_think(s: &str) -> String {
        let mut out = String::new();
        let mut rest = s;
        while let Some(i) = rest.find("<think>") {
            out.push_str(&rest[..i]);
            if let Some(j) = rest[i..].find("</think>") {
                rest = &rest[i + j + "</think>".len()..];
            } else {
                rest = "";
            }
        }
        out.push_str(rest);
        out.replace("<think>", "").replace("</think>", "").trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") {
            body = &body[..c];
        }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?;
        let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        // Accept "arguments" or "parameters"; else treat the remaining fields as args.
        let args = json
            .get("arguments")
            .or_else(|| json.get("parameters"))
            .cloned()
            .unwrap_or_else(|| {
                let mut m = json.clone();
                if let Some(o) = m.as_object_mut() {
                    o.remove("name");
                }
                m
            });
        Some((name, args))
    }

    fn exec_tool(name: &str, args: &serde_json::Value) -> String {
        let get = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        match name {
            "read_file" => {
                let sym = get("symbol");
                crate::agent::agent_read_file(
                    get("path"),
                    None,
                    None,
                    None,
                    (!sym.is_empty()).then_some(sym),
                )
                .unwrap_or_else(|e| format!("ERROR: {e}"))
            }
            "search_code" => crate::agent::agent_search_code(get("query"), None)
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            "validate_change" => {
                let files: Option<Vec<String>> = args
                    .get("files")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(crate::agent::agent_validate_change(files))
                    .unwrap_or_else(|e| format!("ERROR: {e}"))
            }
            "list_dir" => {
                let p = get("path");
                match crate::agent::agent_list_dir(if p.is_empty() { None } else { Some(p) }) {
                    Ok(es) => es
                        .iter()
                        .map(|e| format!("{}{}", e.name, if e.is_dir { "/" } else { "" }))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Err(e) => format!("ERROR: {e}"),
                }
            }
            "write_file" => crate::agent::agent_write_file(get("path"), get("content"))
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            // One merged edit tool: an `edits` array applies several atomically,
            // otherwise it's a single old_string/new_string replacement.
            // `multi_edit` stays a tolerated alias.
            "edit_file" | "multi_edit" => {
                let edits: Vec<crate::agent::EditOp> = args
                    .get("edits")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                if edits.is_empty() {
                    crate::agent::agent_edit_file(
                        get("path"),
                        get("old_string"),
                        get("new_string"),
                        args.get("replace_all").and_then(|v| v.as_bool()),
                    )
                    .unwrap_or_else(|e| format!("ERROR: {e}"))
                } else {
                    crate::agent::agent_multi_edit(get("path"), edits)
                        .unwrap_or_else(|e| format!("ERROR: {e}"))
                }
            }
            "outline" => crate::agent::agent_outline(get("path"))
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            "glob" => crate::agent::agent_glob(get("pattern"))
                .map(|h| h.join("\n"))
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            "grep" => crate::agent::agent_grep(get("pattern"), None, None)
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            "search_files" => crate::agent::agent_search_files(
                get("query"),
                None,
                args.get("names_only").and_then(|v| v.as_bool()),
            )
            .unwrap_or_else(|e| format!("ERROR: {e}")),
            "bash" => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                match rt.block_on(crate::agent::agent_bash(get("command"), Some(60), None)) {
                    Ok(r) => format!("{}\n{}\n[exit {}]", r.stdout, r.stderr, r.code),
                    Err(e) => format!("ERROR: {e}"),
                }
            }
            "web_search" => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let site = get("site");
                if site.is_empty() {
                    match rt.block_on(crate::search::web_search(get("query"))) {
                        Ok(hits) => hits
                            .iter()
                            .take(8)
                            .map(|h| format!("{} — {}\n{}", h.title, h.url, h.snippet))
                            .collect::<Vec<_>>()
                            .join("\n"),
                        Err(e) => format!("ERROR: {e}"),
                    }
                } else {
                    match rt.block_on(crate::webx::site_search(site, get("query"))) {
                        Ok(hits) => hits
                            .iter()
                            .take(10)
                            .map(|h| format!("[{}] {} — {}\n{}", h.kind, h.title, h.url, h.snippet))
                            .collect::<Vec<_>>()
                            .join("\n"),
                        Err(e) => format!("ERROR: {e}"),
                    }
                }
            }
            "web_fetch" => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                match rt.block_on(crate::webx::fetch_page_ex(
                    get("url"),
                    args.get("raw").and_then(|v| v.as_bool()),
                )) {
                    Ok(p) => {
                        let mut s = format!("{} [{}]\n{}\n", p.url, p.kind, p.text.chars().take(8000).collect::<String>());
                        if !p.links.is_empty() {
                            s.push_str("— links —\n");
                            for l in p.links.iter().take(12) {
                                s.push_str(&format!("- {} {}\n", l.text, l.url));
                            }
                        }
                        if !p.images.is_empty() {
                            s.push_str("— images —\n");
                            for i in p.images.iter().take(8) {
                                s.push_str(&format!("- {i}\n"));
                            }
                        }
                        s
                    }
                    Err(e) => format!("ERROR: {e}"),
                }
            }
            "web_download" => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(crate::agent::agent_web_download(get("url"), get("path")))
                    .unwrap_or_else(|e| format!("ERROR: {e}"))
            }
            _ => format!("unknown tool: {name}"),
        }
    }

    const SYS: &str = r#"你是 Chaty 的编程智能体,在工作区目录中完成编码任务。工作区根目录:{WS}

可用工具(路径相对工作区,越界会被拒绝):
- read_file: {"path": string}
- write_file: {"path": string, "content": string}
- edit_file: {"path": string, "old_string": string, "new_string": string}
- list_dir: {"path"?: string}
- glob: {"pattern": string}
- grep: {"pattern": string}
- bash: {"command": string}

规则(严格遵守):
- 每次只调用一个工具。调用时只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call> 然后立即停止,不要有其它内容。
- 系统会用 <tool_result>...</tool_result> 返回结果,你再继续。
- 任务完成后不要再调用工具,直接用一两句话总结你做了什么。"#;

    /// Richer prompt that documents the two meta-tools (update_plan / ask_user),
    /// mirroring src/lib/agentLoop.ts. Used to verify the real model emits them
    /// as valid JSON that the parser + loop handle correctly.
    const SYS_META: &str = r#"你是 Chaty 的编程智能体,在工作区目录中完成编码任务。工作区根目录:{WS}

可用工具(路径相对工作区,越界会被拒绝):
- read_file: {"path": string}
- write_file: {"path": string, "content": string}
- edit_file: {"path": string, "old_string": string, "new_string": string}
- list_dir: {"path"?: string}
- glob: {"pattern": string}
- grep: {"pattern": string}
- bash: {"command": string}
- update_plan: 制定或更新任务计划(待办清单)。args: {"todos": [{"content": string, "status": "pending"|"in_progress"|"done"}]}
- ask_user: 需要用户拍板时提一个选择题。args: {"question": string, "options": string[]}

规则(严格遵守):
- 每次只调用一个工具。调用时只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call> 然后立即停止,不要有其它内容。
- 系统会用 <tool_result>...</tool_result> 返回结果,你再继续。
- 开始复杂任务时先用 update_plan 列出步骤,完成一步就更新状态。
- 遇到需要用户决定的事(如命名、格式)用 ask_user 提问,不要自己乱猜。
- 任务完成后不要再调用工具,直接用一两句话总结你做了什么。"#;

    /// Prompt documenting the code-editing tools (merged edit_file / outline),
    /// mirroring src/lib/agentLoop.ts.
    const SYS_CODE: &str = r#"你是 Chaty 的编程智能体,在工作区目录中完成编码任务。工作区根目录:{WS}

可用工具(路径相对工作区,越界会被拒绝):
- read_file: {"path": string, "offset"?: number, "limit"?: number}
- write_file: 新建或整体重写文件。修改已有文件请用 edit_file。args: {"path": string, "content": string}
- edit_file: 精确替换文件内容(old_string 需逐字匹配且唯一)。改一处给 old_string/new_string;同一文件改多处给 edits 数组一次原子提交(任何一条失败则整体不改动)。args: 单处 {"path": string, "old_string": string, "new_string": string, "replace_all"?: boolean} 或 多处 {"path": string, "edits": [{"old_string": string, "new_string": string, "replace_all"?: boolean}]}
- outline: 列出文件的定义大纲(函数/类 + 行号),不读全文即可掌握结构。args: {"path": string}
- list_dir: {"path"?: string}
- grep: {"pattern": string}
- search_files: 按关键词(字面)一次搜文件名和内容。args: {"query": string, "names_only"?: boolean}
- search_code: 按含义提问代码库("哪里处理邮箱校验"),返回按相关度排序的文件清单+关键定义+片段,直接据此挑文件。args: {"query": string}
- validate_change: 改完代码后一键验证:自动找出与改动文件相关的测试并只跑最小集,返回通过/失败与失败摘要。args: {"files"?: string[]}
- bash: {"command": string}

规则(严格遵守):
- 每次只调用一个工具。调用时只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call> 然后立即停止,不要有其它内容。
- 系统会用 <tool_result>...</tool_result> 返回结果,你再继续。
- 修改前先用 outline / read_file 了解结构;同一文件多处修改用一次 edit_file(给 edits 数组)。
- 定位"哪里处理 X"优先用 search_code;改完代码先用 validate_change 验证。
- 任务完成后不要再调用工具,直接用一两句话总结你做了什么。"#;

    /// Refactor e2e: rename a function + update all call sites across the
    /// file — the natural shape for outline + one atomic multi_edit — then
    /// prove the behaviour is unchanged by running the script.
    /// Run: CHATY_TEST_MODEL=… cargo test -p chaty agent_refactors_with_multi_edit -- --ignored --nocapture
    #[test]
    #[ignore]
    fn agent_locates_with_search_files() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let ws = std::env::temp_dir().join(format!("chaty-agent-sf-e2e-{}", std::process::id()));
        std::fs::create_dir_all(ws.join("src")).unwrap();
        std::fs::write(ws.join("src/token_store.py"), "SECRET = 'refresh the token here'\n").unwrap();
        std::fs::write(ws.join("src/api.py"), "def call():\n    # attach the auth token\n    pass\n").unwrap();
        std::fs::write(ws.join("src/unrelated.py"), "x = 42\n").unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS_CODE.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "用 search_files 找出这个项目里所有和 \"token\" 有关的文件和代码,把命中的文件路径列出来。".into(),
                reasoning_content: None,
},
        ];
        let think = Some(false);
        let cancel = AtomicBool::new(false);
        let mut used_search_files = false;
        let mut finished = false;
        let mut final_text = String::new();
        let mut cached: Vec<LlamaToken> = Vec::new();

        for step in 0..10 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 1024,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think,
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} ────────\n{}", raw.trim().chars().take(400).collect::<String>());
            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    used_search_files |= name == "search_files";
                    eprintln!("  ▶ TOOL  {name}  {}", args.to_string().chars().take(200).collect::<String>());
                    let result = exec_tool(&name, &args);
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(500).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });
                    messages.push(ChatMessage { images: Vec::new(),
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                        reasoning_content: None,
});
                }
                None => {
                    final_text = strip_think(&raw);
                    eprintln!("  ✔ FINAL\n{final_text}");
                    finished = true;
                    break;
                }
            }
        }
        std::fs::remove_dir_all(&ws).ok();
        eprintln!("\n════ VERDICT: finished={finished} · used_search_files={used_search_files} ════");
        assert!(finished, "agent never finished");
        assert!(used_search_files, "agent should have used search_files");
        assert!(final_text.contains("token_store.py"), "should have located token_store.py:\n{final_text}");
    }

    #[test]
    #[ignore]
    fn agent_refactors_with_multi_edit() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        eprintln!("loading model: {model_path}");
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 16384u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let ws = std::env::temp_dir().join(format!("chaty-agent-refactor-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(
            ws.join("shop.py"),
            r#"def calc_total(items, tax_rate):
    return sum(items) * (1 + tax_rate)

def receipt(items, tax_rate):
    total = calc_total(items, tax_rate)
    return f"TOTAL: {total:.2f}"

def audit(items):
    # audits use zero tax
    return calc_total(items, 0)

if __name__ == "__main__":
    print(receipt([10, 20], 0.1))
    print(f"AUDIT: {audit([10, 20]):.2f}")
"#,
        )
        .unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS_CODE.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "把 shop.py 里的函数 calc_total 重命名为 compute_total,并更新文件里所有调用它的地方(先用 outline 看结构,同一文件的多处修改用一次 edit_file 的 edits 数组一次完成)。改完运行 python3 shop.py 确认输出仍然是 TOTAL: 33.00 和 AUDIT: 30.00。".into(),
                reasoning_content: None,
},
        ];
        let think = Some(false);
        let cancel = AtomicBool::new(false);
        let mut finished = false;
        let mut used_multi_edit = false;
        let mut used_outline = false;
        let mut cached: Vec<LlamaToken> = Vec::new();

        for step in 0..16 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 2048,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think,
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW ────────\n{}", raw.trim().chars().take(500).collect::<String>());

            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    // Multi-spot edit via the merged edit_file (edits array) or
                    // the multi_edit alias — either counts.
                    used_multi_edit |= (name == "multi_edit" || name == "edit_file")
                        && args.get("edits").and_then(|v| v.as_array()).is_some_and(|a| a.len() >= 2);
                    used_outline |= name == "outline";
                    eprintln!("  ▶ TOOL  {name}  {}", args.to_string().chars().take(300).collect::<String>());
                    let result = exec_tool(&name, &args);
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(600).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });
                    messages.push(ChatMessage { images: Vec::new(),
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                        reasoning_content: None,
});
                }
                None => {
                    eprintln!("  ✔ FINAL\n{}", strip_think(&raw));
                    finished = true;
                    break;
                }
            }
        }

        // Independent behaviour check.
        let out = std::process::Command::new("python3")
            .arg("shop.py")
            .current_dir(&ws)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let src = std::fs::read_to_string(ws.join("shop.py")).unwrap_or_default();
        eprintln!("\n════════ VERDICT: finished={finished} · multi_edit={used_multi_edit} · outline={used_outline} ════════");
        eprintln!("---- run output ----\n{out}");
        std::fs::remove_dir_all(&ws).ok();

        assert!(finished, "agent never produced a final answer");
        assert!(used_multi_edit, "agent should have used one edit_file with an edits array for the multi-site rename");
        assert!(src.contains("compute_total") && !src.contains("calc_total"), "rename incomplete:\n{src}");
        assert!(out.contains("TOTAL: 33.00") && out.contains("AUDIT: 30.00"), "behaviour changed: {out}");
    }

    /// Prompt documenting the web tools, mirroring src/lib/agentLoop.ts.
    const SYS_WEB: &str = r#"你是 Chaty 的编程智能体,在工作区目录中完成任务。工作区根目录:{WS}

可用工具(路径相对工作区,越界会被拒绝):
- read_file: {"path": string}
- write_file: {"path": string, "content": string}
- list_dir: {"path"?: string}
- web_search: 联网搜索。加 site 参数做站内搜索:site="github.com" 返回仓库/issue/代码匹配;site="reddit.com" 搜帖子;site="youtube.com"/"bilibili.com" 返回视频;其他域名限定站内。搜索源偶尔会抽风,返回不相关的结果——连续 2 次搜出来都和问题无关,就说明此刻再换措辞重搜也没用,立即改道:用 web_fetch 直接抓最可能的页面(官方文档、GitHub 仓库、crates.io、项目官网都能猜出 URL)。args: {"query": string, "site"?: string}
- web_fetch: 抓取 URL:文章→Markdown;代码/JSON→原文;GitHub 文件页自动取 raw 源码;YouTube 视频→元信息+完整字幕转写;B站视频→公开元信息+简介;结果附页面链接和图片 URL,可继续 fetch 深入。args: {"url": string, "raw"?: boolean}
- web_download: 把 URL 文件下载到工作区路径。args: {"url": string, "path": string}

规则(严格遵守):
- 每次只调用一个工具。调用时只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call> 然后立即停止,不要有其它内容。
- 系统会用 <tool_result>...</tool_result> 返回结果,你再继续。
- 任务完成后不要再调用工具,直接用一两句话总结你做了什么。"#;

    /// Video-understanding e2e: the real model must find a video via in-site
    /// search, pull its caption transcript through web_fetch, and act on the
    /// CONTENT of the video (not its title) — proving the pipeline delivers
    /// understanding, not just links.
    /// Run: CHATY_TEST_MODEL=… cargo test -p chaty agent_understands_video -- --ignored --nocapture
    #[test]
    #[ignore]
    fn agent_understands_video() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        eprintln!("loading model: {model_path}");
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 16384u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let ws = std::env::temp_dir().join(format!("chaty-agent-video-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS_WEB.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "在 YouTube 上搜索 \"me at the zoo\",找到 YouTube 历史上的第一条视频,用 web_fetch 获取它的字幕转写,然后把视频中拍摄者实际谈论的动物和他说的重点写进 NOTES.md(必须依据字幕内容,不要凭标题猜)。".into(),
                reasoning_content: None,
},
        ];
        let think = Some(false);
        let cancel = AtomicBool::new(false);
        let mut finished = false;
        let mut cached: Vec<LlamaToken> = Vec::new();

        for step in 0..14 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 2048,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think,
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW ────────\n{}", raw.trim().chars().take(400).collect::<String>());
            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    eprintln!("  ▶ TOOL  {name}  {}", args.to_string().chars().take(200).collect::<String>());
                    let result = exec_tool(&name, &args);
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(500).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });
                    messages.push(ChatMessage { images: Vec::new(),
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                        reasoning_content: None,
});
                }
                None => {
                    eprintln!("  ✔ FINAL\n{}", strip_think(&raw));
                    finished = true;
                    break;
                }
            }
        }

        let notes = std::fs::read_to_string(ws.join("NOTES.md")).unwrap_or_default();
        eprintln!("\n════════ VERDICT: finished={finished} · notes={} chars ════════\n{notes}", notes.len());
        std::fs::remove_dir_all(&ws).ok();

        assert!(finished, "agent never produced a final answer");
        let lower = notes.to_lowercase();
        // "front of the elephants … really long trunks" — only knowable from
        // the transcript, never from the title.
        assert!(
            lower.contains("elephant") || notes.contains("大象"),
            "NOTES.md must mention the elephants from the transcript:\n{notes}"
        );
        assert!(
            lower.contains("trunk") || notes.contains("象鼻") || notes.contains("鼻子"),
            "NOTES.md should capture the point about trunks:\n{notes}"
        );
    }

    /// Full-stack online research e2e: the real model must use site-scoped
    /// GitHub search, fetch a page, write findings, and download a binary —
    /// exercising every new web tool against the live internet.
    /// Run: CHATY_TEST_MODEL=… cargo test -p chaty agent_researches_online -- --ignored --nocapture
    #[test]
    #[ignore]
    fn agent_researches_online() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        eprintln!("loading model: {model_path}");
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 16384u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let ws = std::env::temp_dir().join(format!("chaty-agent-web-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS_WEB.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "帮我调研一个小众 Rust 库:在 GitHub 上搜索 \"dom_smoothie readability\",找到那个把 Mozilla Readability 移植到 Rust 的仓库;用 web_fetch 打开它的仓库页面,把仓库全名和一句话简介写入 RESEARCH.md;最后用 web_download 把页面上列出的任意一张图片保存为 logo.png。全部完成后总结。".into(),
                reasoning_content: None,
},
        ];
        let think = Some(false);
        let cancel = AtomicBool::new(false);
        let mut finished = false;
        let mut web_calls = 0u32;
        let mut cached: Vec<LlamaToken> = Vec::new();

        for step in 0..20 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 2048,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think,
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW ────────\n{}", raw.trim().chars().take(600).collect::<String>());

            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    if name.starts_with("web_") {
                        web_calls += 1;
                    }
                    eprintln!("  ▶ TOOL  {name}  {args}");
                    let result = exec_tool(&name, &args);
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(700).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });
                    messages.push(ChatMessage { images: Vec::new(),
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                        reasoning_content: None,
});
                }
                None => {
                    eprintln!("  ✔ FINAL\n{}", strip_think(&raw));
                    finished = true;
                    break;
                }
            }
        }

        let research = std::fs::read_to_string(ws.join("RESEARCH.md")).unwrap_or_default();
        let logo_bytes = std::fs::metadata(ws.join("logo.png")).map(|m| m.len()).unwrap_or(0);
        eprintln!("\n════════ VERDICT: finished={finished} · web_calls={web_calls} · research={} chars · logo={} bytes ════════", research.len(), logo_bytes);
        eprintln!("---- RESEARCH.md ----\n{research}");
        std::fs::remove_dir_all(&ws).ok();

        assert!(finished, "agent never produced a final answer");
        assert!(web_calls >= 3, "agent should have searched, fetched, and downloaded");
        assert!(research.to_lowercase().contains("dom_smoothie"), "RESEARCH.md should name the repo");
        assert!(logo_bytes > 500, "logo.png should have been downloaded");
    }

    /// Search-flail failover e2e: web_search is rigged to return IRRELEVANT
    /// results every time (a degraded backend), mirroring the production
    /// loop's nudge (3rd/4th consecutive search) and intercept (5th+). The
    /// model must stop rephrasing queries and fail over to web_fetch on a
    /// guessable URL, still completing the task.
    /// Run: CHATY_TEST_MODEL=… cargo test -p chaty agent_fails_over_from_flaky_search -- --ignored --nocapture
    #[test]
    #[ignore]
    fn agent_fails_over_from_flaky_search() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        eprintln!("loading model: {model_path}");
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 16384u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let ws = std::env::temp_dir().join(format!("chaty-agent-flaky-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();

        // Degraded search backend: plausible-looking but utterly irrelevant.
        const GARBAGE: &str = "1. 十大人气奶茶配方大公开 — https://example.com/boba\n   在家自制珍珠奶茶的完整教程…\n2. 2026 春季旅行地推荐 — https://example.com/travel\n   这些小众目的地值得一去…\n3. 如何挑选适合自己的跑鞋 — https://example.com/shoes\n   跑步爱好者的选鞋指南…";

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS_WEB.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "调研一下 Rust crate dom_smoothie 是做什么用的,把一句话结论写入 FINDING.md,然后总结。".into(),
                reasoning_content: None,
},
        ];
        let think = Some(false);
        let cancel = AtomicBool::new(false);
        let mut finished = false;
        let mut searches_executed = 0u32;
        let mut searches_intercepted = 0u32;
        let mut fetch_calls = 0u32;
        let mut search_streak = 0u32;
        let mut cached: Vec<LlamaToken> = Vec::new();

        for step in 0..16 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 2048,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think,
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW ────────\n{}", raw.trim().chars().take(400).collect::<String>());

            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    eprintln!("  ▶ TOOL  {name}  {args}");
                    let result = if name == "web_search" {
                        search_streak += 1;
                        if search_streak >= 5 {
                            // Production intercept, mirrored from agentLoop.ts.
                            searches_intercepted += 1;
                            format!("搜索被拦截:这已是连续第 {search_streak} 次 web_search,前几次都没解决问题,说明搜索源此刻不可靠——继续换措辞重搜不会有新结果。请换策略:用 web_fetch 直接抓取最可能的页面(官方文档 / GitHub 仓库 / crates.io 的 URL 通常能直接猜出来)。用过其它工具后可以再搜索。")
                        } else {
                            searches_executed += 1;
                            let mut r = GARBAGE.to_string();
                            if search_streak >= 3 {
                                // Production nudge, mirrored from agentLoop.ts.
                                r.push_str(&format!("\n\n[系统提示] 这已是连续第 {search_streak} 次搜索。若以上结果仍与问题无关,说明搜索源此刻不可靠——不要再换措辞重搜,改用 web_fetch 直接抓取最可能的页面(官方文档/GitHub/crates.io)。"));
                            }
                            r
                        }
                    } else {
                        search_streak = 0;
                        if name == "web_fetch" {
                            fetch_calls += 1;
                        }
                        exec_tool(&name, &args)
                    };
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(500).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });
                    messages.push(ChatMessage { images: Vec::new(),
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                        reasoning_content: None,
});
                }
                None => {
                    eprintln!("  ✔ FINAL\n{}", strip_think(&raw));
                    finished = true;
                    break;
                }
            }
        }

        let finding = std::fs::read_to_string(ws.join("FINDING.md")).unwrap_or_default();
        eprintln!("\n════════ VERDICT: finished={finished} · searches={searches_executed} (+{searches_intercepted} intercepted) · fetches={fetch_calls} ════════");
        eprintln!("---- FINDING.md ----\n{finding}");
        std::fs::remove_dir_all(&ws).ok();

        assert!(finished, "agent never produced a final answer");
        assert!(
            searches_executed <= 4,
            "the model kept flailing on search: {searches_executed} executed searches"
        );
        assert!(fetch_calls >= 1, "the model never failed over to web_fetch");
        let lower = finding.to_lowercase();
        assert!(
            lower.contains("readability") || lower.contains("正文") || lower.contains("可读"),
            "FINDING.md should describe dom_smoothie (readability extraction): {finding}"
        );
    }

    /// Round-closing e2e for the "smart tools" upgrade: a real model must fix
    /// a bug it has to FIND first (multi-file project), with search_code
    /// doing the locating and validate_change doing the verifying — the
    /// decide/filter/verify work that used to burn many fragile steps.
    /// Run: CHATY_TEST_MODEL=… cargo test -p chaty agent_uses_smart_tools_e2e -- --ignored --nocapture
    #[test]
    #[ignore]
    fn agent_uses_smart_tools_e2e() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        eprintln!("loading model: {model_path}");
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 16384u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let ws = std::env::temp_dir().join(format!("chaty-agent-smart-{}", std::process::id()));
        std::fs::create_dir_all(ws.join("src")).unwrap();
        std::fs::write(
            ws.join("src/validators.py"),
            "def validate_email(addr):\n    # BUG: accepts anything longer than 3 chars\n    return len(addr) > 3\n\n\ndef validate_age(age):\n    return 0 < age < 150\n",
        )
        .unwrap();
        std::fs::write(
            ws.join("src/signup.py"),
            "from src.validators import validate_email\n\n\ndef signup(email):\n    if not validate_email(email):\n        raise ValueError('bad email')\n    return {'email': email}\n",
        )
        .unwrap();
        std::fs::write(
            ws.join("src/render.py"),
            "def draw():\n    return 'pixels'\n",
        )
        .unwrap();
        std::fs::write(ws.join("src/__init__.py"), "").unwrap();
        std::fs::write(
            ws.join("test_signup.py"),
            "from src.validators import validate_email\n\n\ndef test_rejects_missing_at():\n    assert validate_email('nope') is False\n\n\ndef test_accepts_normal():\n    assert validate_email('a@b.co') is True\n",
        )
        .unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();
        let _cp = crate::agent::agent_checkpoint_begin();

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS_CODE.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "这个项目的注册邮箱校验有 bug:没有 @ 的字符串也能通过校验。找到相关代码修复(返回值必须仍是布尔值),然后验证修复是否正确。".into(),
                reasoning_content: None,
},
        ];
        let think = Some(false);
        let cancel = AtomicBool::new(false);
        let mut finished = false;
        let mut used_search_code = 0u32;
        let mut used_validate = 0u32;
        let mut last_validate = String::new();
        let mut cached: Vec<LlamaToken> = Vec::new();

        for step in 0..18 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 2048,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think,
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW ────────\n{}", raw.trim().chars().take(400).collect::<String>());

            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    eprintln!("  ▶ TOOL  {name}  {args}");
                    if name == "search_code" {
                        used_search_code += 1;
                    }
                    let result = exec_tool(&name, &args);
                    if name == "validate_change" {
                        used_validate += 1;
                        last_validate = result.clone();
                    }
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(600).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });
                    messages.push(ChatMessage { images: Vec::new(),
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                        reasoning_content: None,
});
                }
                None => {
                    eprintln!("  ✔ FINAL\n{}", strip_think(&raw));
                    finished = true;
                    break;
                }
            }
        }

        let fixed = std::fs::read_to_string(ws.join("src/validators.py")).unwrap_or_default();
        eprintln!("\n════════ VERDICT: finished={finished} · search_code={used_search_code} · validate={used_validate} ════════");
        eprintln!("---- validators.py ----\n{fixed}");
        std::fs::remove_dir_all(&ws).ok();

        assert!(finished, "agent never produced a final answer");
        assert!(fixed.contains('@'), "validate_email must now check for @:\n{fixed}");
        assert!(used_search_code >= 1, "the agent should locate code via search_code");
        assert!(used_validate >= 1, "the agent should verify via validate_change");
        assert!(
            last_validate.contains("✓ 通过"),
            "the last validation must pass:\n{last_validate}"
        );
    }

    #[test]
    #[ignore]
    fn agent_runs_real_task() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        eprintln!("loading model: {model_path}");
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        // A realistic mini-project with a FAILING test the agent must fix:
        // calc.py is missing `subtract`, which test_calc.py exercises.
        let ws = std::env::temp_dir().join(format!("chaty-agent-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("calc.py"), "def add(a, b):\n    return a + b\n").unwrap();
        std::fs::write(
            ws.join("test_calc.py"),
            "from calc import add, subtract\n\nassert add(2, 3) == 5\nassert subtract(5, 2) == 3\nprint('ALL TESTS PASSED')\n",
        )
        .unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "这个项目有一个失败的测试。请运行 `python3 test_calc.py`,找出失败原因并修复代码,直到测试全部通过(输出 ALL TESTS PASSED)。".into(),
                reasoning_content: None,
},
        ];
        // Simulate the app's thinking config: default forces no-think (Some(false)),
        // but CHATY_TEST_THINK=none reproduces the "model may think" path.
        let think = match std::env::var("CHATY_TEST_THINK").as_deref() {
            Ok("none") => None,
            Ok("true") => Some(true),
            _ => Some(false),
        };
        eprintln!("think = {think:?}");
        let cancel = AtomicBool::new(false);
        let mut finished = false;
        let mut transcript = String::new(); // what the user would see, in order

        // Persist the KV cache across steps (mirrors the app's worker), so the
        // prompt-reuse path is exercised the same way it is in production.
        let mut cached: Vec<LlamaToken> = Vec::new();

        for step in 0..20 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 2048,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think,
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            // The app resets its KV between our direct calls differently; keep it
            // simple and correct by re-decoding each step from the cache we hold.
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW MODEL OUTPUT ────────\n{}", raw.trim());

            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    let prose = strip_think(&raw);
                    let prose = prose.split("<tool_call>").next().unwrap_or("").trim();
                    if !prose.is_empty() {
                        transcript.push_str(&format!("💬 {prose}\n"));
                    }
                    transcript.push_str(&format!("🔧 {name}({args})\n"));
                    eprintln!("  ▶ TOOL  {name}  {args}");
                    let result = exec_tool(&name, &args);
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(800).collect::<String>());
                    transcript.push_str(&format!(
                        "   → {}\n",
                        result.lines().take(3).collect::<Vec<_>>().join(" | ")
                    ));
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });
                    messages.push(ChatMessage { images: Vec::new(),
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                        reasoning_content: None,
});
                }
                None => {
                    let final_text = strip_think(&raw);
                    eprintln!("  ✔ FINAL\n{final_text}");
                    transcript.push_str(&format!("✅ {final_text}\n"));
                    finished = true;
                    break;
                }
            }
        }

        // Independently verify the fix actually works.
        let final_run = std::process::Command::new("python3")
            .arg("test_calc.py")
            .current_dir(&ws)
            .output();
        let passed = final_run
            .as_ref()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("ALL TESTS PASSED"))
            .unwrap_or(false);
        let calc_src = std::fs::read_to_string(ws.join("calc.py")).unwrap_or_default();

        eprintln!("\n════════ USER-VISIBLE TRANSCRIPT ════════\n{transcript}");
        eprintln!("════════ VERDICT: finished={finished} · tests_pass={passed} ════════");
        eprintln!("---- final calc.py ----\n{calc_src}");
        std::fs::remove_dir_all(&ws).ok();

        assert!(finished, "agent never produced a final answer");
        assert!(calc_src.contains("subtract"), "agent should have added subtract()");
        assert!(passed, "the test suite should pass after the agent's fix");
    }

    /// Verifies the real model actually EMITS update_plan and ask_user as valid
    /// tool calls our parser + loop handle, and that answering ask_user steers it.
    /// Run: CHATY_TEST_MODEL=/path/to/model.gguf cargo test -p chaty agent_plans_and_asks -- --ignored --nocapture
    #[test]
    #[ignore]
    fn agent_plans_and_asks() {
        let model_path = match std::env::var("CHATY_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL=/path/to/model.gguf");
                return;
            }
        };
        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let ws = std::env::temp_dir().join(format!("chaty-agent-meta-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        crate::agent::agent_set_workspace(ws.to_string_lossy().to_string()).unwrap();

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: SYS_META.replace("{WS}", &ws.to_string_lossy()), reasoning_content: None },
            ChatMessage { images: Vec::new(),
                role: Role::User,
                content: "请在工作区创建一个 Python 模块 greet.py,实现 greet(name) 函数,再写 test_greet.py 并用 bash 运行确认通过。开始前先用 update_plan 列出步骤。问候语的语言(中文还是英文)由我决定,请用 ask_user 问我。".into(),
                reasoning_content: None,
},
        ];
        let cancel = AtomicBool::new(false);
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut plan_used = false;
        let mut ask_used = false;
        let mut finished = false;
        let mut last_plan: Vec<serde_json::Value> = vec![];

        for step in 0..24 {
            let req = GenRequest {
                messages: messages.clone(),
                params: GenParams {
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 2048,
                    repeat_penalty: 1.05,
                    stop: vec!["</tool_call>".to_string()],
                    think: Some(false),
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, None, &mut None, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} ────────\n{}", raw.trim());

            let Some((name, args)) = parse_tool_call(&raw) else {
                eprintln!("  ✔ FINAL\n{}", strip_think(&raw));
                finished = true;
                break;
            };
            let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&with_close), reasoning_content: None });

            let result = match name.as_str() {
                "update_plan" => {
                    plan_used = true;
                    if let Some(todos) = args.get("todos").and_then(|v| v.as_array()) {
                        last_plan = todos.clone();
                        eprintln!("  📋 PLAN ({} items)", todos.len());
                        for t in todos {
                            let c = t.get("content").and_then(|v| v.as_str()).unwrap_or("");
                            let s = t.get("status").and_then(|v| v.as_str()).unwrap_or("");
                            eprintln!("     [{s}] {c}");
                        }
                    }
                    "计划已更新。".to_string()
                }
                "ask_user" => {
                    ask_used = true;
                    let q = args.get("question").and_then(|v| v.as_str()).unwrap_or("");
                    let opts: Vec<String> = args
                        .get("options")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|o| o.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    // Pick the option that looks like English, else the first.
                    let choice = opts
                        .iter()
                        .find(|o| o.to_lowercase().contains("英") || o.to_lowercase().contains("english") || o.to_lowercase().contains("en"))
                        .cloned()
                        .or_else(|| opts.first().cloned())
                        .unwrap_or_else(|| "English".to_string());
                    eprintln!("  ❓ ASK: {q}\n     options={opts:?} → chose {choice}");
                    format!("用户的选择是:{choice}")
                }
                _ => {
                    eprintln!("  ▶ {name}({args})");
                    let r = exec_tool(&name, &args);
                    eprintln!("  ◀ {}", r.chars().take(400).collect::<String>());
                    r
                }
            };
            messages.push(ChatMessage { images: Vec::new(),
                role: Role::User,
                content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
                reasoning_content: None,
});
        }

        let greet_src = std::fs::read_to_string(ws.join("greet.py")).unwrap_or_default();
        eprintln!(
            "\n════════ VERDICT: finished={finished} · plan_used={plan_used} · ask_used={ask_used} · plan_items={} ════════",
            last_plan.len()
        );
        eprintln!("---- greet.py ----\n{greet_src}");
        std::fs::remove_dir_all(&ws).ok();

        assert!(plan_used, "model should have called update_plan");
        assert!(ask_used, "model should have called ask_user");
        assert!(finished, "agent never produced a final answer");
        assert!(greet_src.contains("greet"), "greet.py should define greet()");
    }
}

/// Vision e2e against a REAL vision model (main GGUF + mmproj side by side in
/// one folder — the layout the downloader creates). Ignored by default. Run:
///   CHATY_TEST_VLM=/path/to/Folder/model.gguf cargo test -p chaty vision_e2e -- --ignored --nocapture
#[cfg(test)]
mod vision_e2e {
    use super::*;
    use std::cell::RefCell;

    struct Collector {
        buf: RefCell<String>,
    }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev {
                self.buf.borrow_mut().push_str(&text);
            }
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn vision_sees_image_and_reuses_media_cache() {
        let model_path = match std::env::var("CHATY_TEST_VLM") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_VLM=/path/to/vlm.gguf (mmproj beside it)");
                return;
            }
        };
        // The folder-layout pairing is part of what's under test.
        let mmproj = find_mmproj(&model_path).expect("no mmproj found next to the test VLM");
        eprintln!("paired mmproj: {}", mmproj.display());

        let backend = llama_backend().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load model");
        let n_ctx = 4096u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");

        let mtmd_params = MtmdContextParams {
            use_gpu: true,
            n_threads: nt,
            ..MtmdContextParams::default()
        };
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params)
            .expect("mtmd init");
        assert!(mtmd.support_vision(), "mmproj must provide a vision encoder");

        // A solid red image — unambiguous even for a 500M model.
        let img_path = std::env::temp_dir().join(format!("chaty-vision-e2e-{}.png", std::process::id()));
        image::RgbImage::from_pixel(224, 224, image::Rgb([214, 30, 30]))
            .save(&img_path)
            .expect("write test image");

        let ask = |messages: Vec<ChatMessage>,
                   ctx: &mut LlamaContext,
                   cached: &mut Vec<LlamaToken>,
                   media_cache: &mut Option<MediaCache>|
         -> String {
            let req = GenRequest {
                messages,
                params: GenParams {
                    temperature: 0.1,
                    max_tokens: 64,
                    think: Some(false),
                    ..Default::default()
                },
            };
            let sink = Collector { buf: RefCell::new(String::new()) };
            let cancel = AtomicBool::new(false);
            run_turn(&model, ctx, cached, Some(&mtmd), media_cache, n_ctx, &req, &sink, &cancel)
                .expect("run_turn");
            let out = sink.buf.into_inner();
            eprintln!("--- reply: {out}");
            out
        };

        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;

        // ---- turn 1: the model must actually SEE the image ----
        let mut messages = vec![ChatMessage {
            images: vec![img_path.to_string_lossy().to_string()],
            role: Role::User,
            content: "What is the dominant color of this image? Answer with one English word.".into(),
            reasoning_content: None,
}];
        let ans1 = ask(messages.clone(), &mut ctx, &mut cached, &mut media_cache);
        assert!(
            ans1.to_lowercase().contains("red"),
            "expected 'red' in the answer, got: {ans1}"
        );
        let cache1 = media_cache.as_ref().expect("media cache after a vision turn");
        let (prompt1, n_past1) = (cache1.prompt.clone(), cache1.n_past);
        assert!(n_past1 > 0 && !prompt1.is_empty());
        assert!(cached.is_empty(), "token cache must stay empty in the media regime");

        // ---- turn 2: follow-up reuses the media prefill incrementally ----
        messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: ans1, reasoning_content: None });
        messages.push(ChatMessage {
            images: Vec::new(),
            role: Role::User,
            content: "Is this image mostly red? Answer strictly yes or no.".into(),
            reasoning_content: None,
});
        // The behavioral property the media cache exists for: a follow-up
        // turn must NOT re-encode the already-seen image. (Checked via the
        // encoder counter rather than string prefixes — on Qwen3.5+ the
        // generation tail legitimately diverges and the BODY anchor carries
        // the reuse.)
        let _ = prompt1;
        let encodes_before = img_encode_count();
        let fallbacks_before = cache_fallback_count();
        let ans2 = ask(messages, &mut ctx, &mut cached, &mut media_cache);
        // Either the media cache reused the KV (no re-encode) — standard
        // attention models — or the model's memory can't partially rewind
        // (hybrid/recurrent, e.g. Qwen3.6) and the CLEAN fallback re-prefilled
        // everything. Silent re-encoding through any other path is a bug.
        let reused = img_encode_count() == encodes_before;
        let clean_fallback = cache_fallback_count() > fallbacks_before;
        assert!(
            reused || clean_fallback,
            "turn 2 re-encoded the image without going through the documented fallback"
        );
        eprintln!("turn-2 reuse: reused={reused} clean_fallback={clean_fallback}");
        assert!(
            ans2.to_lowercase().contains("yes"),
            "expected 'yes' (image still visible through the reused KV), got: {ans2}"
        );
        let cache2 = media_cache.as_ref().unwrap();
        assert!(cache2.n_past > n_past1, "prefill must have grown, not restarted");

        // ---- turn 3: a text-only request leaves the media regime cleanly ----
        let ans3 = ask(
            vec![ChatMessage {
                images: Vec::new(),
                role: Role::User,
                content: "Say the single word 'hello'.".into(),
                reasoning_content: None,
}],
            &mut ctx,
            &mut cached,
            &mut media_cache,
        );
        assert!(!ans3.trim().is_empty(), "text-only turn after vision must still generate");
        assert!(media_cache.is_none(), "media cache must clear when leaving the media regime");
        assert!(!cached.is_empty(), "token cache must resume in the text regime");

        let _ = std::fs::remove_file(&img_path);
    }
}

/// Engine-level wiring test: `LlamaEngine::load` must auto-pair the mmproj and
/// report `vision_ready` (the worker inits the mtmd context). Ignored; run:
///   CHATY_TEST_VLM=… cargo test -p chaty engine_load_pairs_mmproj -- --ignored --nocapture
#[cfg(test)]
mod vision_engine {
    use super::*;

    #[test]
    #[ignore]
    fn engine_load_pairs_mmproj() {
        let model_path = match std::env::var("CHATY_TEST_VLM") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_VLM=/path/to/vlm.gguf (mmproj beside it)");
                return;
            }
        };
        let (engine, info) = LlamaEngine::load(&model_path, None, Some(4096)).expect("load");
        eprintln!("vision_ready={} mmproj={:?} warning={:?}", info.vision_ready, info.mmproj, info.warning);
        assert!(info.multimodal, "VLM must be flagged multimodal");
        assert!(info.vision_ready, "mmproj must be paired and loaded");
        assert!(info.mmproj.as_deref().map_or(false, |p| p.to_lowercase().contains("mmproj")));
        assert!(info.warning.is_none(), "clean load expected, got {:?}", info.warning);

        // generate_collect (the one-shot vision path used by KB/Canvas/browser)
        let img = std::env::temp_dir().join(format!("chaty-collect-e2e-{}.png", std::process::id()));
        image::RgbImage::from_pixel(160, 160, image::Rgb([25, 120, 220]))
            .save(&img)
            .unwrap();
        let req = GenRequest {
            messages: vec![ChatMessage {
                images: vec![img.to_string_lossy().to_string()],
                role: Role::User,
                content: "What is the dominant color? One English word.".into(),
                reasoning_content: None,
}],
            params: GenParams { temperature: 0.1, max_tokens: 32, think: Some(false), ..Default::default() },
        };
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let out =
            tokio::runtime::Runtime::new().unwrap().block_on(engine.generate_collect(req, cancel));
        let text = out.expect("generate_collect");
        eprintln!("collect reply: {text}");
        assert!(text.to_lowercase().contains("blue"), "expected 'blue', got: {text}");
        let _ = std::fs::remove_file(&img);

        // A richer scene → a descriptive, retrieval-friendly caption (the KB
        // image-indexing path). Draw a red disc and a green rectangle.
        let mut scene = image::RgbImage::from_pixel(320, 240, image::Rgb([245, 245, 245]));
        for y in 0..240i32 {
            for x in 0..320i32 {
                let (dx, dy) = (x - 90, y - 120);
                if dx * dx + dy * dy < 55 * 55 {
                    scene.put_pixel(x as u32, y as u32, image::Rgb([220, 30, 30]));
                }
                if (190..=280).contains(&x) && (80..=170).contains(&y) {
                    scene.put_pixel(x as u32, y as u32, image::Rgb([30, 170, 70]));
                }
            }
        }
        let scene_path = std::env::temp_dir().join(format!("chaty-scene-e2e-{}.png", std::process::id()));
        scene.save(&scene_path).unwrap();
        let cap_req = GenRequest {
            messages: vec![ChatMessage {
                images: vec![scene_path.to_string_lossy().to_string()],
                role: Role::User,
                content: "Describe this image thoroughly for search: shapes, colors, layout.".into(),
                reasoning_content: None,
}],
            params: GenParams { temperature: 0.3, max_tokens: 200, think: Some(false), ..Default::default() },
        };
        let cancel2 = std::sync::Arc::new(AtomicBool::new(false));
        let caption = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(engine.generate_collect(cap_req, cancel2))
            .expect("caption");
        eprintln!("=== KB caption:\n{caption}\n===");
        let lc = caption.to_lowercase();
        assert!(caption.trim().len() > 40, "caption should be descriptive");
        assert!(
            lc.contains("red") || lc.contains("green") || lc.contains("circle") || lc.contains("rectang") || lc.contains("square"),
            "caption should mention a shape/color, got: {caption}"
        );
        let _ = std::fs::remove_file(&scene_path);

        engine.unload();
    }

    // The full "browser + vision verification" loop: render a web page in the
    // headless CDP browser, screenshot it, and have the vision model describe
    // what it sees. Needs Chrome + CHATY_TEST_VLM.
    #[test]
    #[ignore]
    fn browser_vision_verify() {
        let model_path = match std::env::var("CHATY_TEST_VLM") {
            Ok(p) => p,
            Err(_) => { eprintln!("SKIP: set CHATY_TEST_VLM"); return; }
        };
        if crate::browser::chrome_path_pub().is_none() {
            eprintln!("SKIP: no Chrome"); return;
        }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");
        let (engine, info) = LlamaEngine::load(&model_path, None, Some(4096)).expect("load");
        assert!(info.vision_ready);

        let html = "<!doctype html><html><head><title>Invoice</title></head>\
            <body style='font-family:sans-serif;padding:40px'>\
            <h1 style='color:#0a7'>Monthly Invoice</h1>\
            <p>Total due: <b>$4,200</b></p>\
            <button>Pay now</button>\
            <script>console.error('checkout failed: card declined')</script></body></html>";
        let path = std::env::temp_dir().join(format!("chaty-bv-{}.html", std::process::id()));
        std::fs::write(&path, html).unwrap();
        let url = format!("file://{}", path.display());
        crate::browser::navigate(&url).expect("navigate");
        let png = crate::browser::screenshot().expect("screenshot");
        let shot = std::env::temp_dir().join(format!("chaty-bv-{}.png", std::process::id()));
        std::fs::write(&shot, &png).unwrap();

        let console = crate::browser::console().expect("console");
        eprintln!("console: {console}");
        assert!(console.contains("card declined"), "console error should be captured");

        let req = GenRequest {
            messages: vec![ChatMessage {
                images: vec![shot.to_string_lossy().to_string()],
                role: Role::User,
                content: "What is the heading text and the total amount shown on this page?".into(),
                reasoning_content: None,
}],
            params: GenParams { temperature: 0.2, max_tokens: 96, think: Some(false), ..Default::default() },
        };
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let ans = tokio::runtime::Runtime::new().unwrap()
            .block_on(engine.generate_collect(req, cancel)).expect("vision");
        eprintln!("=== vision read of the page: {ans}");
        let lc = ans.to_lowercase();
        assert!(lc.contains("invoice") || lc.contains("4,200") || lc.contains("4200") || lc.contains("pay"),
            "model should read the page content, got: {ans}");

        crate::browser::shutdown();
        engine.unload();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&shot);
    }
}

/// PROBE: drive a realistic multi-page form task with the agent tool-loop so we
/// can see where a real model gets stuck. A local site: home → Contact link →
/// form (name/email/message) → submit → thank-you. Needs Chrome + CHATY_TEST_VLM.
#[cfg(test)]
mod browser_task_probe {
    use super::*;
    use std::cell::RefCell;

    struct Collector { buf: RefCell<String> }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev { self.buf.borrow_mut().push_str(&text); }
            Ok(())
        }
    }
    fn strip_think(s: &str) -> String {
        let mut out = String::new(); let mut rest = s;
        while let Some(i) = rest.find("<think>") { out.push_str(&rest[..i]); if let Some(j)=rest[i..].find("</think>"){rest=&rest[i+j+8..];} else {rest="";} }
        out.push_str(rest); out.trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") { body = &body[..c]; }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?; let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        let args = json.get("arguments").or_else(|| json.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
        Some((name, args))
    }

    #[test]
    #[ignore]
    fn agent_fills_and_submits_a_form() {
        let model_path = match std::env::var("CHATY_TEST_VLM") { Ok(p) => p, Err(_) => { eprintln!("SKIP: CHATY_TEST_VLM"); return; } };
        if crate::browser::chrome_path_pub().is_none() { eprintln!("SKIP: no Chrome"); return; }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");

        // Two-page local site.
        let dir = std::env::temp_dir().join(format!("chaty-formsite-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"),
            "<!doctype html><title>Acme</title><body style='font-family:sans-serif;padding:40px'>\
             <h1>Acme Inc.</h1><nav><a href='contact.html'>Contact</a> · <a href='about.html'>About</a></nav>\
             <p>Welcome to Acme.</p></body>").unwrap();
        std::fs::write(dir.join("contact.html"),
            "<!doctype html><title>Contact Acme</title><body style='font-family:sans-serif;padding:40px'>\
             <h1>Contact us</h1>\
             <form onsubmit=\"event.preventDefault();document.body.innerHTML='<h1 id=ok>Thanks, we got your message!</h1>'\">\
             <input name='name' placeholder='Your name'><br>\
             <input name='email' type='email' placeholder='Your email'><br>\
             <textarea name='message' placeholder='Your message'></textarea><br>\
             <button type='submit'>Send message</button></form></body>").unwrap();
        let home = format!("file://{}/index.html", dir.display());

        let backend = crate::inference::llama::llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_threads(nt).with_n_threads_batch(nt);
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;

        // Minimal browser-tool system prompt + the task.
        let sys = format!(
            "你是浏览器自动化助手。每步只输出一行 <tool_call>{{\"name\":..,\"arguments\":{{..}}}}</tool_call> 然后停止。可用工具:\n\
             - browser_navigate {{url}}:打开页面,返回可交互元素清单\n\
             - browser_read {{}}:刷新当前页面的元素清单\n\
             - browser_screenshot {{}}:看当前页面(会把截图给你)\n\
             - browser_click {{text}} 或 {{selector}}:优先用 text 按可见文字点击\n\
             - browser_type {{label,text}}:按占位符/字段名填输入框\n\
             系统会用 <tool_result> 回你。完成后不要再调用工具,直接说\"完成\"。\n\
             CSS 选择器只支持标准语法(没有 :contains/:has-text);按文字点用 browser_click 的 text。首个页面:{home}"
        );
        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: sys, reasoning_content: None },
            ChatMessage { images: Vec::new(), role: Role::User,
                content: format!("打开 {home},进入 Contact 页面,填写姓名 Alice、邮箱 alice@example.com、留言 Hello,然后提交表单。\n/no_think"),
                reasoning_content: None,
},
        ];

        let cancel = AtomicBool::new(false);
        let mut submitted = false;
        let mut step_log: Vec<String> = Vec::new();
        for step in 0..16 {
            let req = GenRequest { messages: messages.clone(), params: GenParams { temperature: 0.3, max_tokens: 1024, think: Some(false), ..Default::default() } };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, Some(&mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            let call = parse_tool_call(&raw);
            eprintln!("--- step {step}: {}", call.as_ref().map(|(n,a)| format!("{n} {a}")).unwrap_or_else(|| format!("(no tool) {}", raw.trim().chars().take(80).collect::<String>())));
            let Some((name, args)) = call else { break; };
            step_log.push(name.clone());
            let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
            let (result, image): (String, Option<String>) = match name.as_str() {
                "browser_navigate" => (crate::browser::navigate(&g("url").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_read" => (crate::browser::read_page().unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_click" => (crate::browser::click(g("selector"), g("text")).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_type" => (crate::browser::type_text(g("selector"), g("label"), g("text").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_screenshot" => {
                    let png = crate::browser::screenshot().unwrap();
                    let p = std::env::temp_dir().join(format!("chaty-probe-{}-{step}.png", std::process::id()));
                    std::fs::write(&p, png).unwrap();
                    ("(截图已附上)".into(), Some(p.to_string_lossy().to_string()))
                }
                _ => (format!("未知工具 {name}"), None),
            };
            // Did the form submit?
            if let Ok(ok) = crate::browser::eval("document.getElementById('ok')?document.getElementById('ok').textContent:''") {
                if ok.contains("Thanks") { submitted = true; }
            }
            // Echo the model's ACTUAL output (with args) — the real loop does this;
            // a name-only echo makes the model mimic argument-less calls.
            let asst = strip_think(&raw);
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: asst, reasoning_content: None });
            let mut m = ChatMessage { images: image.clone().into_iter().collect(), role: Role::User,
                content: format!("<tool_result>{}</tool_result>\n/no_think", result),
                reasoning_content: None,
};
            if image.is_some() { m.content = "<tool_result>这是当前页面截图,请查看后继续。</tool_result>\n/no_think".into(); }
            messages.push(m);
            if submitted { eprintln!("=== FORM SUBMITTED at step {step}"); break; }
        }
        crate::browser::shutdown();
        eprintln!("=== steps: {}", step_log.join(" → "));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(submitted, "the agent should have filled + submitted the form; steps: {step_log:?}");
    }
}

/// PRODUCTION E2E: drive the 35B through several realistic, multi-step web tasks
/// on a local "real" site (SPA shop→checkout, login→secret, lazy-load feed) plus
/// a real web_fetch research task — and VERIFY the actual end state via the DOM,
/// not the model's self-report. Also mirrors the production loop's identical-call
/// breaker WITH the scroll/observe exemption, so a legit multi-scroll isn't
/// wrongly intercepted (regression for the "scroll 300px ×2" false positive), and
/// records step counts + flags sub-optimal tool choices (browser for pure
/// research, redundant reads, screenshot-instead-of-snapshot).
///
///   CHATY_TEST_VLM=/path/Folder/model.gguf cargo test -p chaty \
///     agent_completes_production_web_tasks -- --ignored --nocapture
#[cfg(test)]
mod browser_tasks_e2e {
    use super::*;
    use std::cell::RefCell;

    struct Collector { buf: RefCell<String> }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev { self.buf.borrow_mut().push_str(&text); }
            Ok(())
        }
    }
    fn strip_think(s: &str) -> String {
        let mut out = String::new(); let mut rest = s;
        while let Some(i) = rest.find("<think>") { out.push_str(&rest[..i]); if let Some(j)=rest[i..].find("</think>"){rest=&rest[i+j+8..];} else {rest="";} }
        out.push_str(rest); out.trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") { body = &body[..c]; }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?; let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        let args = json.get("arguments").or_else(|| json.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
        Some((name, args))
    }

    // Tools whose repeated identical call is legitimate progress/observation —
    // MUST match REPEAT_EXEMPT in src/lib/agentLoop.ts (kept in sync by hand).
    // NOT browser_navigate / view_image: an identical call there re-fetches the
    // SAME target (a real degenerate loop the breaker must stop).
    fn repeat_exempt(name: &str) -> bool {
        matches!(name,
            "browser_scroll" | "browser_screenshot" | "browser_snapshot" |
            "browser_read" | "browser_console" | "bg_output")
    }

    struct TaskReport {
        name: &'static str,
        done: bool,
        steps: Vec<String>,
        final_text: String,
        notes: Vec<String>,
        wrongly_blocked: bool,
    }

    /// Faithful mirror of the production agent loop for ONE task. Executes real
    /// browser/web tools, echoes the model's full output, attaches screenshots
    /// as images, and applies the identical-call breaker (with the exemption).
    #[allow(clippy::too_many_arguments)]
    fn drive(
        model: &LlamaModel,
        backend: &LlamaBackend,
        mtmd: &MtmdContext,
        n_ctx: u32,
        nt: i32,
        rt: &tokio::runtime::Runtime,
        name: &'static str,
        sys: &str,
        task: &str,
        max_steps: usize,
        break_on_done: bool,
        done: &dyn Fn() -> bool,
    ) -> TaskReport {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;

        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: sys.to_string(), reasoning_content: None },
            ChatMessage { images: Vec::new(), role: Role::User, content: format!("{task}\n/no_think"), reasoning_content: None },
        ];
        let cancel = AtomicBool::new(false);

        let mut steps: Vec<String> = Vec::new();
        let mut final_text = String::new();
        let mut last_key = String::new();
        let mut repeat = 0usize;
        let mut screenshot_ct = 0usize;

        for step in 0..max_steps {
            let req = GenRequest { messages: messages.clone(), params: GenParams { temperature: 0.3, max_tokens: 1024, think: Some(false), ..Default::default() } };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(model, &mut ctx, &mut cached, Some(mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            let call = parse_tool_call(&raw);
            let Some((tname, args)) = call else {
                final_text = strip_think(&raw);
                eprintln!("--- [{name}] step {step}: FINAL: {}", final_text.chars().take(120).collect::<String>());
                break;
            };
            eprintln!("--- [{name}] step {step}: {tname} {args}");

            // ── Identical-call breaker (mirror of production) ──
            let call_key = format!("{tname}:{args}");
            if repeat_exempt(&tname) {
                last_key.clear(); repeat = 0;
            } else if call_key == last_key {
                repeat += 1;
            } else {
                last_key = call_key.clone(); repeat = 0;
            }
            if repeat >= 1 && !repeat_exempt(&tname) {
                // Production intercepts the 2nd identical NON-exempt call. For an
                // exempt tool (scroll/observe) this branch is unreachable — which
                // is exactly what proves a legit repeated scroll is NOT blocked.
                eprintln!("    (breaker: intercepted identical non-exempt call {tname})");
                steps.push(format!("{tname}*BLOCKED"));
                messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
                messages.push(ChatMessage { images: Vec::new(), role: Role::User,
                    content: "<tool_result>这一步和上一步完全相同,已拦截。换一种做法或读取当前状态。</tool_result>\n/no_think".into(),
                    reasoning_content: None,
});
                continue;
            }

            steps.push(tname.clone());
            let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
            let (result, image): (String, Option<String>) = match tname.as_str() {
                "browser_navigate" => (crate::browser::navigate(&g("url").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_read" => (crate::browser::read_page().unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_console" => (crate::browser::console().unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_click" => (crate::browser::click(g("selector"), g("text")).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_type" => (crate::browser::type_text(g("selector"), g("label"), g("text").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_scroll" => {
                    let to = g("to");
                    let by = args.get("by").and_then(|v| v.as_f64());
                    (crate::browser::scroll_page(to, by).unwrap_or_else(|e| format!("ERROR: {e}")), None)
                }
                "browser_snapshot" | "browser_screenshot" => {
                    screenshot_ct += 1;
                    let png = if tname == "browser_snapshot" { crate::browser::snapshot() } else { crate::browser::screenshot() };
                    match png {
                        Ok(bytes) => {
                            let p = std::env::temp_dir().join(format!("chaty-e2e-{}-{step}.png", std::process::id()));
                            std::fs::write(&p, bytes).unwrap();
                            ("(截图已附上)".into(), Some(p.to_string_lossy().to_string()))
                        }
                        Err(e) => (format!("ERROR: {e}"), None),
                    }
                }
                "web_fetch" => {
                    let url = g("url").unwrap_or_default();
                    match rt.block_on(crate::search::fetch_url(url)) {
                        Ok(pc) => (format!("标题: {}\n正文:\n{}", pc.title, pc.text.chars().take(1500).collect::<String>()), None),
                        Err(e) => (format!("ERROR: {e}"), None),
                    }
                }
                "web_search" => {
                    let q = g("query").unwrap_or_default();
                    match rt.block_on(crate::search::web_search(q)) {
                        Ok(rs) => {
                            let s = rs.iter().take(5)
                                .map(|r| format!("- {} — {}\n  {}", r.title, r.url, r.snippet))
                                .collect::<Vec<_>>().join("\n");
                            (if s.is_empty() { "(no results)".into() } else { s }, None)
                        }
                        Err(e) => (format!("ERROR: {e}"), None),
                    }
                }
                other => (format!("未知工具 {other}"), None),
            };

            // Echo the model's ACTUAL output (with args), then the tool result.
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
            if let Some(img) = image {
                messages.push(ChatMessage { images: vec![img], role: Role::User,
                    content: "<tool_result>这是当前页面截图,请查看后继续。</tool_result>\n/no_think".into(),
                    reasoning_content: None,
});
            } else {
                messages.push(ChatMessage { images: Vec::new(), role: Role::User,
                    content: format!("<tool_result>{result}</tool_result>\n/no_think"),
                    reasoning_content: None,
});
            }

            if break_on_done && done() { break; }
        }

        // ── Efficiency / optimal-choice notes ──
        let mut notes = Vec::new();
        // 1. redundant browser_read directly after browser_navigate (navigate
        //    already returns the digest).
        for w in steps.windows(2) {
            if w[0] == "browser_navigate" && w[1] == "browser_read" {
                notes.push("次优: browser_navigate 后紧跟 browser_read(navigate 已返回元素清单,多余一步)".into());
                break;
            }
        }
        // 2. full-page screenshots when a viewport snapshot would do.
        let full_shots = steps.iter().filter(|s| *s == "browser_screenshot").count();
        if full_shots >= 2 {
            notes.push(format!("次优: 用了 {full_shots} 次整页 browser_screenshot(较慢);多数情况 browser_snapshot 更快"));
        }
        // 3. any scroll/observe tool that got intercepted means the breaker
        //    misfired. Checked against a LITERAL list (not repeat_exempt) so the
        //    test still fails loudly if someone deletes the exemption.
        const SHOULD_NEVER_BLOCK: &[&str] = &[
            "browser_scroll", "browser_screenshot", "browser_snapshot",
            "browser_read", "browser_console", "browser_navigate", "view_image", "bg_output",
        ];
        let wrongly_blocked = steps.iter().any(|s| {
            s.ends_with("*BLOCKED") && SHOULD_NEVER_BLOCK.contains(&s.trim_end_matches("*BLOCKED"))
        });
        let _ = screenshot_ct;

        TaskReport { name, done: done(), steps, final_text, notes, wrongly_blocked }
    }

    fn shop_html() -> String {
        r#"<!doctype html><title>Nimbus Shop</title>
<body style="font-family:sans-serif;padding:32px;max-width:640px;margin:auto">
<h1>Nimbus Shop</h1><div id="view"></div>
<script>
var products={"Nimbus Pro":{price:299,desc:"Flagship. Everything included."},
 "Nimbus Lite":{price:99,desc:"Essentials for starters."},
 "Nimbus Max":{price:499,desc:"For power users."}};
var view=document.getElementById('view');
function grid(){var h='<h2>Products</h2>';for(var k in products){h+='<div style="margin:12px 0"><button onclick="detail(\''+k+'\')">'+k+'</button> — $'+products[k].price+'</div>';}view.innerHTML=h;}
function detail(name){var p=products[name];view.innerHTML='<h2>'+name+'</h2><p>'+p.desc+'</p><p>Price: $'+p.price+'</p><button onclick="addcart(\''+name+'\')">Add to cart</button> <button onclick="grid()">Back</button>';}
function addcart(name){view.innerHTML='<h2>Cart</h2><p>1 x '+name+' added.</p><button onclick="checkout()">Proceed to checkout</button>';}
function checkout(){view.innerHTML='<h2>Checkout</h2><p><input placeholder="Full name" id="f_name"></p><p><input placeholder="Email" id="f_email"></p><p><input placeholder="Shipping address" id="f_addr"></p><button onclick="place()">Place order</button>';}
function place(){var n=document.getElementById('f_name').value,e=document.getElementById('f_email').value,a=document.getElementById('f_addr').value;if(!n||!e||!a){alert('fill all');return;}document.title='Order Confirmed';view.innerHTML='<h2>Thank you, '+n+'!</h2><div id="order-number">ORD-4821</div><p>Confirmation sent to '+e+'</p>';}
grid();
</script></body>"#.to_string()
    }
    fn login_html() -> String {
        r#"<!doctype html><title>Nimbus Login</title>
<body style="font-family:sans-serif;padding:32px"><h1>Sign in</h1>
<div id="app"><p><input placeholder="Username" id="u"></p><p><input placeholder="Password" type="password" id="p"></p>
<button onclick="signin()">Sign in</button><div id="err" style="color:crimson"></div></div>
<script>
function signin(){var u=document.getElementById('u').value,p=document.getElementById('p').value;
if(u==='admin'&&p==='nimbus2026'){document.getElementById('app').innerHTML='<h2>Dashboard</h2><p>Your API key:</p><div id="secret">TOKEN-7Q2X</div>';}
else{console.error('login failed for user '+u);document.getElementById('err').textContent='Invalid credentials';}}
</script></body>"#.to_string()
    }
    // A longer content page whose key fact (the Enterprise price) sits well
    // below the fold. The optimal move is ONE full-page screenshot to see the
    // whole thing at once; browser_read returns no body text, so reading the
    // price REQUIRES vision. Tests "screenshot-first" for page research.
    fn docs_html() -> String {
        r#"<!doctype html><title>Nimbus Pricing</title>
<body style="font-family:sans-serif;max-width:760px;margin:0 auto;padding:24px;line-height:1.6">
<h1>Nimbus Cloud — Pricing</h1>
<p>Nimbus Cloud is our managed platform. Below are the plans. Every plan includes the core runtime, automatic backups, and email support. Annual billing saves 20%.</p>
<h2>Starter</h2><p>For individuals getting started. Includes 1 project, 5 GB storage, community support. Price: 9 dollars per month.</p>
<h2>Team</h2><p>For small teams shipping together. Includes 10 projects, 100 GB storage, priority email support, and role-based access. Price: 49 dollars per month.</p>
<h2>Business</h2><p>For growing companies. Includes unlimited projects, 1 TB storage, SSO, audit logs, and a 99.9% uptime SLA. Price: 199 dollars per month.</p>
<h2>Enterprise</h2>
<p style="font-size:22px"><b>The Enterprise plan costs 999 dollars per month.</b> It adds dedicated infrastructure, a named support engineer, custom contracts, and a 99.99% uptime SLA.</p>
<p>Contact sales for volume discounts. All prices are in USD and exclude local taxes.</p>
</body>"#.to_string()
    }
    fn feed_html() -> String {
        r#"<!doctype html><title>Nimbus Feed</title>
<body style="font-family:sans-serif;margin:0;padding:16px"><h1>Feed</h1><div id="feed"></div>
<script>
var n=0,max=30;
function add(k){var d=document.createElement('div');d.id='item-'+k;d.style.height='150px';d.style.borderBottom='1px solid gray';d.textContent='Item '+k+(k===max?' — LAST (item-30)':'');document.getElementById('feed').appendChild(d);}
function batch(){for(var i=0;i<8&&n<max;i++){add(++n);}}
batch();
addEventListener('scroll',function(){if(window.innerHeight+window.scrollY>=document.body.scrollHeight-250){batch();}});
</script></body>"#.to_string()
    }

    const SYS: &str = "你是浏览器/网页自动化助手。每步只输出一行 <tool_call>{\"name\":..,\"arguments\":{..}}</tool_call> 然后停止,系统会用 <tool_result> 回你。\n\
可用工具:\n\
- web_search {query}:联网搜索,返回结果标题+链接+摘要。\n\
- web_fetch {url}:抓取一个网址的正文(查资料/读文档首选,比开浏览器快)。\n\
- browser_navigate {url}:打开页面,返回页面上可交互元素清单。\n\
- browser_read {}:刷新当前页面的可交互元素清单(只含可点/可填元素,不含正文;点击/跳转后核实状态用)。\n\
- browser_screenshot {}:截取整页给你看。打开一个要研究/读内容的页面后,第一步就用它一次性拿到整页全貌,快速定位元素或读正文,省掉反复 snapshot+scroll。\n\
- browser_snapshot {}:只截当前视口给你看(整页某处要放大细看、或某次交互后只想确认这一屏时用)。\n\
- browser_scroll {to?,by?}:向下滚动加载更多内容(连续多次滚动是正常进度)。\n\
- browser_click {text|selector}:优先用 text 按可见文字点击。\n\
- browser_type {label,text}:按占位符/字段名 label 向输入框填 text。\n\
- browser_console {}:读控制台输出与报错。\n\
规则:①任何点击/输入/跳转/滚动后,页面可能变了——先看工具返回的元素清单,或用 browser_read/browser_snapshot 核实当前状态,再进行下一步,别凭猜测连续操作。②CSS 选择器只支持标准语法(不存在 :contains/:has-text);按文字点用 browser_click 的 text。③纯查资料/读文档优先用 web_fetch/web_search(更快,不必开浏览器);web_fetch/web_search 一旦拿到能回答问题的内容,就直接回答,不要再多开浏览器重复核实。完成任务后不要再调用工具,直接用一句话回答结果。";

    #[test]
    #[ignore]
    fn agent_completes_production_web_tasks() {
        let model_path = match std::env::var("CHATY_TEST_VLM") { Ok(p) => p, Err(_) => { eprintln!("SKIP: set CHATY_TEST_VLM"); return; } };
        if crate::browser::chrome_path_pub().is_none() { eprintln!("SKIP: no Chrome"); return; }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");

        let dir = std::env::temp_dir().join(format!("chaty-e2e-site-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shop.html"), shop_html()).unwrap();
        std::fs::write(dir.join("login.html"), login_html()).unwrap();
        std::fs::write(dir.join("feed.html"), feed_html()).unwrap();
        std::fs::write(dir.join("pricing.html"), docs_html()).unwrap();
        let shop = format!("file://{}/shop.html", dir.display());
        let login = format!("file://{}/login.html", dir.display());
        let feed = format!("file://{}/feed.html", dir.display());
        let pricing = format!("file://{}/pricing.html", dir.display());

        let backend = crate::inference::llama::llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");
        let rt = tokio::runtime::Runtime::new().unwrap();

        let mut reports: Vec<TaskReport> = Vec::new();

        // ── Task 1: shop → product → cart → checkout form → order confirmed ──
        {
            let shop2 = shop.clone();
            let done = move || {
                crate::browser::eval("(document.title==='Order Confirmed'&&/ORD-/.test(document.body.innerText))?'Y':'N'")
                    .map(|s| s.contains('Y')).unwrap_or(false)
            };
            let task = format!("打开 {shop2},进入商品 Nimbus Pro,把它加入购物车,前往结账,\
                姓名填 Alice、邮箱填 alice@example.com、收货地址填 1 Rue Nimbus,然后下单(Place order)。");
            reports.push(drive(&model, backend, &mtmd, n_ctx, nt, &rt, "shop-checkout", SYS, &task, 16, true, &done));
        }

        // ── Task 2: login with correct creds → read the revealed secret token ──
        {
            let done = || {
                crate::browser::eval("(function(){var s=document.getElementById('secret');return s?s.textContent:'';})()")
                    .map(|s| s.contains("TOKEN-7Q2X")).unwrap_or(false)
            };
            let task = format!("打开 {login},用用户名 admin、密码 nimbus2026 登录,\
                然后告诉我登录后页面显示的 API 密钥是什么。");
            reports.push(drive(&model, backend, &mtmd, n_ctx, nt, &rt, "login-secret", SYS, &task, 12, false, &done));
        }

        // ── Task 3: lazy-load feed — scroll repeatedly until item-30 appears ──
        {
            let done = || {
                crate::browser::eval("document.getElementById('item-30')?'Y':'N'")
                    .map(|s| s.contains('Y')).unwrap_or(false)
            };
            let task = format!("打开 {feed},这是一个懒加载的信息流,一直向下滚动直到出现第 30 条 (item-30),\
                然后告诉我第 30 条的文字内容。");
            reports.push(drive(&model, backend, &mtmd, n_ctx, nt, &rt, "lazy-feed", SYS, &task, 14, true, &done));
        }

        // ── Task 4: research a page — should full-page screenshot FIRST to
        //    locate the fact, not snapshot+scroll around. Answer is below the
        //    fold and browser_read has no body text, so vision is required. ──
        {
            let done = || false; // content question — judged by final text + tool path
            let task = format!("打开 {pricing},这是 Nimbus 的定价页面,告诉我 Enterprise(企业版)套餐每月多少钱。");
            reports.push(drive(&model, backend, &mtmd, n_ctx, nt, &rt, "locate-price", SYS, &task, 8, false, &done));
        }

        // ── Task 5: pure research — should pick web_fetch, NOT the browser ──
        let research_online = {
            let done = || false; // no DOM; judged by final text + tool choice
            let task = "查一下 https://example.com 这个网页的主标题(H1)是什么,一句话告诉我。".to_string();
            let r = drive(&model, backend, &mtmd, n_ctx, nt, &rt, "research", SYS, &task, 6, false, &done);
            // If the fetch itself failed (offline/sandbox), don't hard-fail the suite.
            let online = !r.final_text.is_empty()
                && (r.final_text.to_lowercase().contains("example domain")
                    || r.steps.iter().any(|s| s == "web_fetch"));
            reports.push(r);
            online
        };

        crate::browser::shutdown();
        let _ = std::fs::remove_dir_all(&dir);

        // ── Report ──
        eprintln!("\n================ PRODUCTION WEB-TASK E2E REPORT ================");
        for r in &reports {
            eprintln!("[{}] done={} steps={} :: {}", r.name, r.done, r.steps.len(), r.steps.join(" → "));
            if !r.final_text.is_empty() { eprintln!("    final: {}", r.final_text.chars().take(140).collect::<String>()); }
            for n in &r.notes { eprintln!("    ⚠ {n}"); }
            if r.wrongly_blocked { eprintln!("    ✗ a legit exempt tool was wrongly blocked by the breaker!"); }
        }
        eprintln!("===============================================================\n");

        // ── Hard assertions on ACTUAL end state (not the model's claim) ──
        let get = |name: &str| reports.iter().find(|r| r.name == name).unwrap();
        assert!(get("shop-checkout").done, "shop task: order was NOT actually confirmed in the DOM; steps: {:?}", get("shop-checkout").steps);
        assert!(get("login-secret").done, "login task: the secret token was NOT actually revealed; steps: {:?}", get("login-secret").steps);
        assert!(get("login-secret").final_text.to_uppercase().contains("TOKEN-7Q2X"), "login task: model didn't report the token; got: {}", get("login-secret").final_text);
        assert!(get("lazy-feed").done, "feed task: item-30 was NOT actually reached; steps: {:?}", get("lazy-feed").steps);
        // No exempt tool may ever be wrongly blocked (regression for scroll ×2).
        for r in &reports { assert!(!r.wrongly_blocked, "[{}] a scroll/observe tool was wrongly intercepted as a repeat", r.name); }
        // Locate-on-page: HARD gate = correct answer (999) read off the page.
        // SOFT check = did it screenshot the whole page FIRST (optimal) instead
        // of a snapshot+scroll hunt? Surfaced as an efficiency note.
        let locate = get("locate-price");
        assert!(locate.final_text.contains("999"), "locate task: expected the Enterprise price 999 in the answer; got: {}", locate.final_text);
        let first_visual = locate.steps.iter().find(|s| *s == "browser_screenshot" || *s == "browser_snapshot");
        let scroll_hunt = locate.steps.iter().filter(|s| *s == "browser_snapshot" || *s == "browser_scroll").count();
        match first_visual.map(|s| s.as_str()) {
            Some("browser_screenshot") if scroll_hunt == 0 =>
                eprintln!("✓ locate task took the full-page screenshot first (optimal): {:?}", locate.steps),
            _ =>
                eprintln!("⚠ EFFICIENCY: locate task did NOT lead with a full-page screenshot — steps: {:?} (prompt now steers screenshot-first for page research)", locate.steps),
        }

        // Research: only assert when the fetch actually reached the network.
        // HARD gate = the answer is correct (real extraction). Tool CHOICE is a
        // probabilistic model behaviour, so a sub-optimal choice (opening the
        // browser after web_fetch already answered) is surfaced as an efficiency
        // WARNING, not a flaky hard failure — the regression suite stays
        // deterministic while still making the inefficiency visible.
        let research = get("research");
        if research_online {
            assert!(research.final_text.to_lowercase().contains("example domain"), "research task: expected 'Example Domain' in the answer; got: {}", research.final_text);
            if research.steps.iter().any(|s| s == "browser_navigate") {
                eprintln!("⚠ EFFICIENCY: research task opened the browser after web_fetch already had the answer — steps: {:?} (prompt now discourages this; watch the trend)", research.steps);
            } else {
                eprintln!("✓ research task used web_fetch only (optimal): {:?}", research.steps);
            }
        } else {
            eprintln!("NOTE: research task network unreachable — skipped its assertions.");
        }
    }
}

/// E2E: the decode loop must emit `Prefill` progress events for a LONG prompt
/// (several batches — drives the Code-mode progress ring), monotonically up to
/// 100% — and must NOT emit them when the KV prefix cache leaves only a short
/// tail to decode (no ring flash on fast steps).
///
///   CHATY_TEST_MODEL=/path/model.gguf cargo test -p chaty prefill_progress -- --ignored --nocapture
#[cfg(test)]
mod prefill_progress_e2e {
    use super::*;
    use std::cell::RefCell;

    struct AllEvents {
        evs: RefCell<Vec<StreamEvent>>,
    }
    impl EventSink for AllEvents {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            self.evs.borrow_mut().push(ev);
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn prefill_progress_long_yes_short_no() {
        let model_path = match std::env::var("CHATY_TEST_MODEL")
            .or_else(|_| std::env::var("CHATY_TEST_VLM"))
        {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_MODEL (or CHATY_TEST_VLM)");
                return;
            }
        };
        let backend = llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;
        let cancel = AtomicBool::new(false);

        // ── Turn 1: ~several-thousand-token prompt → multiple decode batches ──
        let filler = "这是一段用来撑长提示词的资料文本,内容本身不重要。".repeat(120);
        let mut messages = vec![ChatMessage {
            images: Vec::new(),
            role: Role::User,
            content: format!("{filler}\n读完以上资料,回答:一加一等于几?只答数字。\n/no_think"),
            reasoning_content: None,
}];
        let req = GenRequest {
            messages: messages.clone(),
            params: GenParams { temperature: 0.0, max_tokens: 8, think: Some(false), ..Default::default() },
        };
        let sink = AllEvents { evs: RefCell::new(Vec::new()) };
        run_turn(&model, &mut ctx, &mut cached, None, &mut media_cache, n_ctx, &req, &sink, &cancel)
            .expect("run_turn long");
        let evs = sink.evs.into_inner();
        let reply: String = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let prefills: Vec<(u32, u32)> = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Prefill { processed, total } => Some((*processed, *total)),
                _ => None,
            })
            .collect();
        eprintln!(
            "long turn: {} prefill events, first {:?}, last {:?}, reply {:?}",
            prefills.len(),
            prefills.first(),
            prefills.last(),
            reply.trim().chars().take(40).collect::<String>()
        );
        assert!(prefills.len() >= 3, "long prompt must emit several progress events, got {prefills:?}");
        let total = prefills[0].1;
        assert!(total > 1000, "the test prompt should be well over a batch, total={total}");
        assert!(prefills.iter().all(|(_, t)| *t == total), "total must be stable");
        assert!(prefills.windows(2).all(|w| w[0].0 <= w[1].0), "progress must be monotonic: {prefills:?}");
        assert_eq!(prefills.last().unwrap().0, total, "progress must end at 100%");

        // ── Turn 2: append a short exchange → KV prefix reuse leaves a tiny
        //    tail (< one batch) → NO prefill events (the ring must not flash) ──
        messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: reply, reasoning_content: None });
        messages.push(ChatMessage { images: Vec::new(), role: Role::User, content: "再答一次,只答数字。\n/no_think".into(), reasoning_content: None });
        let req2 = GenRequest {
            messages,
            params: GenParams { temperature: 0.0, max_tokens: 8, think: Some(false), ..Default::default() },
        };
        let sink2 = AllEvents { evs: RefCell::new(Vec::new()) };
        run_turn(&model, &mut ctx, &mut cached, None, &mut media_cache, n_ctx, &req2, &sink2, &cancel)
            .expect("run_turn short");
        let prefills2: Vec<(u32, u32)> = sink2
            .evs
            .into_inner()
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Prefill { processed, total } => Some((*processed, *total)),
                _ => None,
            })
            .collect();
        eprintln!("short incremental turn: {} prefill events (want 0)", prefills2.len());
        assert!(prefills2.is_empty(), "a short cached-prefix tail must not emit progress: {prefills2:?}");
    }
}

#[cfg(test)]
mod vision_speed_tests {
    use super::*;

    #[test]
    fn downscale_caps_pixels_and_caches() {
        // 3000x1000 = 3 MP > 2 MP budget → downscaled JPEG in temp cache.
        let dir = std::env::temp_dir().join(format!("chaty-dsc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let big = dir.join("big.png");
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(3000, 1000, image::Rgb([200, 30, 30])))
            .save(&big)
            .unwrap();
        let fed = downscale_for_vision(&big.to_string_lossy());
        assert_ne!(fed, big.to_string_lossy(), "oversized image must be re-routed");
        let out = image::open(&fed).unwrap();
        let px = (out.width() as u64) * (out.height() as u64);
        assert!(px <= MAX_VISION_PIXELS, "downscaled to {px} px");
        // Aspect ratio preserved (3:1).
        let ratio = out.width() as f64 / out.height() as f64;
        assert!((ratio - 3.0).abs() < 0.05, "ratio {ratio}");
        // Second call hits the cache (same path returned, file already there).
        let fed2 = downscale_for_vision(&big.to_string_lossy());
        assert_eq!(fed, fed2);

        // A small image passes through untouched.
        let small = dir.join("small.png");
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(640, 480, image::Rgb([30, 200, 30])))
            .save(&small)
            .unwrap();
        assert_eq!(downscale_for_vision(&small.to_string_lossy()), small.to_string_lossy());

        // The MLX cap is TIGHTER: the same 3 MP image under a 1 MP cap must
        // produce a DIFFERENT (smaller) cached file than the 2 MP one — the
        // cap is part of the cache key, or the two engines would poison each
        // other's caches.
        let fed_mlx = downscale_for_vision_capped(&big.to_string_lossy(), 1_000_000);
        assert_ne!(fed_mlx, fed, "per-cap cache keys must differ");
        let out_mlx = image::open(&fed_mlx).unwrap();
        let px_mlx = (out_mlx.width() as u64) * (out_mlx.height() as u64);
        assert!(px_mlx <= 1_000_000, "mlx-capped to {px_mlx} px");
        // And a 1.5 MP image: passes the GGUF cap untouched, shrinks for MLX.
        let mid = dir.join("mid.png");
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1500, 1000, image::Rgb([30, 30, 200])))
            .save(&mid)
            .unwrap();
        assert_eq!(downscale_for_vision(&mid.to_string_lossy()), mid.to_string_lossy());
        assert_ne!(
            downscale_for_vision_capped(&mid.to_string_lossy(), 1_000_000),
            mid.to_string_lossy()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The size gate must be header-only: a file whose header parses but
    /// whose pixel data is corrupt still passes through when under the cap
    /// (no decode happened), and still resolves from the cache when a cached
    /// copy exists — the every-step re-decode was the memory-spike bug.
    #[test]
    fn downscale_size_gate_reads_header_not_pixels() {
        let dir = std::env::temp_dir().join(format!("chaty-dsh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A valid small PNG, then truncate the pixel data: header (IHDR) is
        // intact, full decode would fail.
        let p = dir.join("truncated.png");
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(640, 480, image::Rgb([9, 9, 9])))
            .save(&p)
            .unwrap();
        let bytes = std::fs::read(&p).unwrap();
        std::fs::write(&p, &bytes[..64]).unwrap(); // IHDR survives, IDAT gone
        assert_eq!(
            image::image_dimensions(&p).unwrap(),
            (640, 480),
            "header must still parse"
        );
        assert!(image::open(&p).is_err(), "full decode must fail");
        // Under-cap → returned as-is WITHOUT decoding (decode would error).
        assert_eq!(
            downscale_for_vision(&p.to_string_lossy()),
            p.to_string_lossy()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// E2E: an image turn must emit `Prefill` progress events (the segmented media
/// prefill), and the model must still understand the DOWNSCALED image — the
/// oversized original is shrunk before the vision encoder.
///
///   CHATY_TEST_VLM=… cargo test -p chaty media_prefill_progress -- --ignored --nocapture
#[cfg(test)]
mod media_prefill_e2e {
    use super::*;
    use std::cell::RefCell;

    struct AllEvents {
        evs: RefCell<Vec<StreamEvent>>,
    }
    impl EventSink for AllEvents {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            self.evs.borrow_mut().push(ev);
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn media_prefill_progress_and_downscaled_understanding() {
        let model_path = match std::env::var("CHATY_TEST_VLM") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: set CHATY_TEST_VLM");
                return;
            }
        };
        // A 3000x1500 (4.5 MP) pure-red image — over the 2 MP feed budget, so
        // the encoder sees the downscaled copy.
        let dir = std::env::temp_dir().join(format!("chaty-media-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let img_path = dir.join("red.png");
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(3000, 1500, image::Rgb([220, 20, 20])))
            .save(&img_path)
            .unwrap();

        let backend = llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(nt)
            .with_n_threads_batch(nt);
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;
        let cancel = AtomicBool::new(false);

        let req = GenRequest {
            messages: vec![ChatMessage {
                images: vec![img_path.to_string_lossy().to_string()],
                role: Role::User,
                content: "What is the dominant color of this image? Answer with one word.".into(),
                reasoning_content: None,
}],
            params: GenParams { temperature: 0.0, max_tokens: 12, think: Some(false), ..Default::default() },
        };
        let sink = AllEvents { evs: RefCell::new(Vec::new()) };
        run_turn(&model, &mut ctx, &mut cached, Some(&mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel)
            .expect("run_turn media");
        let evs = sink.evs.into_inner();
        let prefills: Vec<(u32, u32)> = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Prefill { processed, total } => Some((*processed, *total)),
                _ => None,
            })
            .collect();
        let reply: String = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        eprintln!("media turn: {} prefill events {:?} … {:?}; reply {:?}", prefills.len(), prefills.first(), prefills.last(), reply.trim());

        // 1. The image turn emits progress: an initial event + one per segment,
        //    monotonic, ending at 100%.
        assert!(prefills.len() >= 2, "image turn must emit prefill progress: {prefills:?}");
        let total = prefills[0].1;
        assert!(prefills.windows(2).all(|w| w[0].0 <= w[1].0), "monotonic: {prefills:?}");
        assert_eq!(prefills.last().unwrap().0, total, "ends at 100%");
        // 2. Vision still works on the downscaled feed.
        assert!(reply.to_lowercase().contains("red"), "model should see red, got: {reply}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// PRODUCTION E2E: a progressive-rules "password game" (a faithful local clone
/// of neal.fun/password-game — rules appear one at a time as you satisfy them).
/// Proves the model solves it FAST using ONLY `browser_read` (text) +
/// `browser_type` — NO screenshots — because the rich text digest surfaces every
/// dynamically-shown rule. Generalizes to any dynamic page.
///
///   CHATY_TEST_VLM=… cargo test -p chaty password_game_no_vision -- --ignored --nocapture
#[cfg(test)]
mod password_game_e2e {
    use super::*;
    use std::cell::RefCell;

    struct Collector { buf: RefCell<String> }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev { self.buf.borrow_mut().push_str(&text); }
            Ok(())
        }
    }
    fn strip_think(s: &str) -> String {
        let mut out = String::new(); let mut rest = s;
        while let Some(i) = rest.find("<think>") { out.push_str(&rest[..i]); if let Some(j)=rest[i..].find("</think>"){rest=&rest[i+j+8..];} else {rest="";} }
        out.push_str(rest); out.trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") { body = &body[..c]; }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?; let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        let args = json.get("arguments").or_else(|| json.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
        Some((name, args))
    }

    const GAME_HTML: &str = r##"<!doctype html><meta charset="utf-8"><title>Password Game</title>
<body style="font-family:sans-serif;max-width:640px;margin:2rem auto">
<h1>The Password Game</h1><p>Please choose a password.</p>
<textarea id="pw" rows="3" style="width:100%;font-size:18px" placeholder="password"></textarea>
<div id="rules"></div><div id="score"></div>
<script>
var RULES=[
 {n:1,text:"Your password must be at least 8 characters.",t:function(p){return p.length>=8;}},
 {n:2,text:"Your password must include a number.",t:function(p){return /[0-9]/.test(p);}},
 {n:3,text:"Your password must include an uppercase letter.",t:function(p){return /[A-Z]/.test(p);}},
 {n:4,text:"Your password must include a special character (one of ! @ % &).",t:function(p){return /[!@%&]/.test(p);}},
 {n:5,text:"Your password must include a month of the year (e.g. December).",t:function(p){return /(january|february|march|april|may|june|july|august|september|october|november|december)/i.test(p);}},
 {n:6,text:"Your password must include the Roman numeral XV.",t:function(p){return /XV/.test(p);}},
 {n:7,text:"Your password must include one of our sponsors: shell.",t:function(p){return /shell/i.test(p);}},
 {n:8,text:"Your password must include the word chaty.",t:function(p){return /chaty/i.test(p);}}
];
var pw=document.getElementById("pw");
function render(){
 var p=pw.value, revealed=0;
 for(var i=0;i<RULES.length;i++){revealed=i+1;if(!RULES[i].t(p))break;}
 var solved=0,html="";
 for(var i=0;i<revealed;i++){var ok=RULES[i].t(p);if(ok)solved++;html+="<div>Rule "+RULES[i].n+" — "+(ok?"PASS: ":"TODO: ")+RULES[i].text+"</div>";}
 document.getElementById("rules").innerHTML=html;
 var win=solved===RULES.length;
 document.getElementById("score").innerHTML="<b>Rules solved: "+solved+" of "+RULES.length+(win?" — YOU WIN":"")+"</b>";
 document.body.dataset.solved=solved;
}
pw.addEventListener("input",render);render();
</script></body>"##;

    #[test]
    #[ignore]
    fn password_game_no_vision() {
        let model_path = match std::env::var("CHATY_TEST_VLM") { Ok(p) => p, Err(_) => { eprintln!("SKIP: CHATY_TEST_VLM"); return; } };
        if crate::browser::chrome_path_pub().is_none() { eprintln!("SKIP: no Chrome"); return; }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");

        let dir = std::env::temp_dir().join(format!("chaty-pwgame-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("game.html"), GAME_HTML).unwrap();
        let url = format!("file://{}/game.html", dir.display());

        let backend = crate::inference::llama::llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_threads(nt).with_n_threads_batch(nt);
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;

        let sys = format!(
            "你是浏览器自动化助手。每步只输出一行 <tool_call>{{\"name\":..,\"arguments\":{{..}}}}</tool_call> 然后停止,系统会用 <tool_result> 回你。\n\
             可用工具:\n\
             - browser_navigate {{url}}:打开页面,返回页面全部可见文字+元素。\n\
             - browser_read {{}}:读取当前页面的全部可见文字(规则/提示都在里面)+ 输入框当前值。**看页面只用它,不要截图。**\n\
             - browser_type {{text}}:把密码输入框的内容设为 text(整体替换,不是追加)。\n\
             这是一个「密码游戏」:页面会**逐条**给出对密码的要求,你满足当前所有已出现的要求后,会**出现新的一条要求**。\n\
             做法:browser_read 看当前所有规则 → 想一个**同时满足所有已出现规则**的密码 → browser_type 输入完整密码 → 看返回的新页面文字 → 如果出现新规则或某条没过,再调整密码重新 browser_type。每次都要输入**完整**密码(工具是整体替换)。\n\
             **只用 browser_read/browser_type 文字操作,绝不要用 browser_screenshot/browser_snapshot。** 首个页面:{url}"
        );
        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: sys, reasoning_content: None },
            ChatMessage { images: Vec::new(), role: Role::User, content: format!("打开 {url},玩这个密码游戏,尽量多满足几条规则。\n/no_think"), reasoning_content: None },
        ];

        let cancel = AtomicBool::new(false);
        let mut steps: Vec<String> = Vec::new();
        let solved_now = || -> i64 {
            crate::browser::eval("document.body.dataset.solved||'0'")
                .ok().and_then(|s| s.trim_matches('"').parse::<i64>().ok()).unwrap_or(0)
        };
        let started = std::time::Instant::now();
        let mut best = 0i64;

        for step in 0..20 {
            let req = GenRequest { messages: messages.clone(), params: GenParams { temperature: 0.3, max_tokens: 700, think: Some(false), ..Default::default() } };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, Some(&mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            let Some((name, args)) = parse_tool_call(&raw) else {
                eprintln!("--- step {step}: (no tool) {}", strip_think(&raw).chars().take(80).collect::<String>());
                break;
            };
            let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
            eprintln!("--- step {step}: {name} {}", g("text").unwrap_or_default().chars().take(60).collect::<String>());
            steps.push(name.clone());
            let result = match name.as_str() {
                "browser_navigate" => crate::browser::navigate(&g("url").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")),
                "browser_read" => crate::browser::read_page().unwrap_or_else(|e| format!("ERROR: {e}")),
                "browser_type" => crate::browser::type_text(g("selector"), g("label"), g("text").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")),
                other => format!("未知工具 {other}"),
            };
            best = best.max(solved_now());
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
            messages.push(ChatMessage { images: Vec::new(), role: Role::User, content: format!("<tool_result>{}</tool_result>\n/no_think", result), reasoning_content: None });
            if best >= 8 { break; }
        }
        let elapsed = started.elapsed();
        crate::browser::shutdown();
        let _ = std::fs::remove_dir_all(&dir);

        let screenshots = steps.iter().filter(|s| *s == "browser_screenshot" || *s == "browser_snapshot").count();
        eprintln!("=== password game: solved {best}/8 in {} steps, {:.1}s, screenshots={screenshots} :: {}", steps.len(), elapsed.as_secs_f32(), steps.join(" → "));

        // The point of the feature: solve MANY rules FAST, purely from text.
        assert_eq!(screenshots, 0, "must solve WITHOUT any screenshot (rich text digest is the vision substitute)");
        assert!(best >= 6, "should progressively solve most rules from text alone; got {best}/8");
    }
}

/// PRODUCTION E2E: a LONG, scrollable form with MANY fields spread top-to-bottom
/// plus a checkbox and a submit button at the very end. Proves the 35B fills it
/// accurately, optimally (batched multi-field `browser_type` + multi-target
/// `browser_click`), and fast — and every step is audited (printed) and the
/// FINAL DOM state is asserted field-by-field (not the model's self-report).
///
///   CHATY_TEST_VLM=… cargo test -p chaty long_form_fill_submit -- --ignored --nocapture
#[cfg(test)]
mod long_form_e2e {
    use super::*;
    use std::cell::RefCell;

    struct Collector { buf: RefCell<String> }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev { self.buf.borrow_mut().push_str(&text); }
            Ok(())
        }
    }
    fn strip_think(s: &str) -> String {
        let mut out = String::new(); let mut rest = s;
        while let Some(i) = rest.find("<think>") { out.push_str(&rest[..i]); if let Some(j)=rest[i..].find("</think>"){rest=&rest[i+j+8..];} else {rest="";} }
        out.push_str(rest); out.trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") { body = &body[..c]; }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?; let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        let args = json.get("arguments").or_else(|| json.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
        Some((name, args))
    }

    // A tall form: intro text, 3 fields near the top, 3 fields far below the
    // fold, a consent checkbox, and a Submit at the very bottom. Submit checks
    // every field is filled AND the box is ticked before confirming.
    const FORM_HTML: &str = r##"<!doctype html><meta charset="utf-8"><title>Apply</title>
<body style="font-family:sans-serif;max-width:640px;margin:0 auto;padding:24px;line-height:1.7">
<h1>Job Application</h1>
<p>Thanks for applying to Nimbus. Please complete every field below and accept the terms, then submit. This form is long — the submit button is at the bottom.</p>
<div style="height:220px;background:aliceblue;border-radius:8px;padding:12px">About the role: you'll build local-first AI apps. We value craft, honesty and speed. Fill in your details carefully.</div>
<h3>Your details</h3>
<p><label>Full name<br><input id="f_name" placeholder="Full name" style="width:100%"></label></p>
<p><label>Email<br><input id="f_email" type="email" placeholder="Email" style="width:100%"></label></p>
<p><label>Current company<br><input id="f_company" placeholder="Company" style="width:100%"></label></p>
<div style="height:900px;background:seashell;border-radius:8px;padding:12px">Portfolio guidelines … (long section, scroll down to continue the form) …</div>
<h3>More about you</h3>
<p><label>Phone<br><input id="f_phone" placeholder="Phone" style="width:100%"></label></p>
<p><label>City<br><input id="f_city" placeholder="City" style="width:100%"></label></p>
<p><label>Why do you want this job?<br><textarea id="f_why" placeholder="Motivation" style="width:100%" rows="3"></textarea></label></p>
<div style="height:500px;background:honeydew;border-radius:8px;padding:12px">Legal … (long) …</div>
<p><label><input type="checkbox" id="f_agree"> I agree to the terms and privacy policy.</label></p>
<p><button id="submit">Submit application</button></p>
<div id="result" style="font-weight:bold"></div>
<script>
function val(id){return (document.getElementById(id).value||"").trim();}
document.getElementById("submit").addEventListener("click",function(){
  var fields={name:val("f_name"),email:val("f_email"),company:val("f_company"),phone:val("f_phone"),city:val("f_city"),why:val("f_why")};
  var missing=Object.keys(fields).filter(function(k){return !fields[k];});
  var agreed=document.getElementById("f_agree").checked;
  if(missing.length){document.getElementById("result").textContent="ERROR: please fill: "+missing.join(", ");return;}
  if(!agreed){document.getElementById("result").textContent="ERROR: please accept the terms.";return;}
  document.title="Submitted";
  document.body.dataset.submitted="1";
  document.getElementById("result").textContent="APPLICATION SUBMITTED — ref APP-7788. Thanks, "+fields.name+"!";
});
</script></body>"##;

    #[test]
    #[ignore]
    fn long_form_fill_submit() {
        let model_path = match std::env::var("CHATY_TEST_VLM") { Ok(p) => p, Err(_) => { eprintln!("SKIP: CHATY_TEST_VLM"); return; } };
        if crate::browser::chrome_path_pub().is_none() { eprintln!("SKIP: no Chrome"); return; }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");

        let dir = std::env::temp_dir().join(format!("chaty-longform-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("apply.html"), FORM_HTML).unwrap();
        let url = format!("file://{}/apply.html", dir.display());

        let backend = crate::inference::llama::llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_threads(nt).with_n_threads_batch(nt);
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;

        let sys = format!(
            "你是浏览器自动化助手。每步只输出一行 <tool_call>{{\"name\":..,\"arguments\":{{..}}}}</tool_call> 然后停止,系统会用 <tool_result> 回你。\n\
             可用工具:\n\
             - browser_navigate {{url}}:打开页面,返回全部可见文字+元素。\n\
             - browser_read {{}}:读取当前页面全部可见文字+输入框当前值(看页面用它,别截图)。\n\
             - browser_type:填输入框。**一次填多个**用 steps:{{\"steps\":[{{\"label\":\"Full name\",\"text\":\"...\"}},{{\"label\":\"Email\",\"text\":\"...\"}}]}}。\n\
             - browser_click:点击。**一次点多处**用 steps:{{\"steps\":[{{\"text\":\"I agree\"}},{{\"text\":\"Submit\"}}]}};也可 {{\"text\":\"...\"}}。\n\
             - browser_scroll {{to?,by?}}:滚动。\n\
             这是一个很长、需要滚动的求职表单。**尽量用 steps 一次填多个字段、一次点多个按钮**以求最快。要求:填完所有字段(姓名/邮箱/公司/电话/城市/求职动机)、勾选同意条款、点提交。每次操作后读返回文字确认。首个页面:{url}"
        );
        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: sys, reasoning_content: None },
            ChatMessage { images: Vec::new(), role: Role::User, content: format!(
                "打开 {url},填写这张求职表单并提交:姓名 Alice Chen、邮箱 alice@nimbus.io、公司 Acme、电话 5551234567、城市 Montreal、求职动机随便写一句,勾选同意条款,然后提交。\n/no_think"),
                reasoning_content: None,
},
        ];

        let cancel = AtomicBool::new(false);
        let mut steps_log: Vec<String> = Vec::new();
        let mut screenshots = 0usize;
        let mut type_calls = 0usize;
        let mut fields_typed = 0usize;
        let submitted = || crate::browser::eval("document.body.dataset.submitted||'0'").map(|s| s.contains('1')).unwrap_or(false);
        let started = std::time::Instant::now();

        for step in 0..18 {
            let req = GenRequest { messages: messages.clone(), params: GenParams { temperature: 0.3, max_tokens: 900, think: Some(false), ..Default::default() } };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, Some(&mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            let Some((name, args)) = parse_tool_call(&raw) else {
                eprintln!("--- step {step}: FINAL: {}", strip_think(&raw).chars().take(100).collect::<String>());
                break;
            };
            // AUDIT: print the full call each step.
            eprintln!("--- step {step} AUDIT: {name} {}", args.to_string().chars().take(200).collect::<String>());
            let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
            let result = match name.as_str() {
                "browser_navigate" => crate::browser::navigate(&g("url").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")),
                "browser_read" => crate::browser::read_page().unwrap_or_else(|e| format!("ERROR: {e}")),
                "browser_scroll" => crate::browser::scroll_page(g("to"), args.get("by").and_then(|v| v.as_f64())).unwrap_or_else(|e| format!("ERROR: {e}")),
                "browser_type" => {
                    type_calls += 1;
                    if let Some(arr) = args.get("steps").and_then(|v| v.as_array()) {
                        fields_typed += arr.len();
                        let steps: Vec<(Option<String>, Option<String>, String)> = arr.iter().map(|s| (
                            s.get("selector").and_then(|v| v.as_str()).map(String::from),
                            s.get("label").and_then(|v| v.as_str()).map(String::from),
                            s.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                        )).collect();
                        crate::browser::type_seq(steps).unwrap_or_else(|e| format!("ERROR: {e}"))
                    } else {
                        fields_typed += 1;
                        crate::browser::type_text(g("selector"), g("label"), g("text").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}"))
                    }
                }
                "browser_click" => {
                    if let Some(arr) = args.get("steps").and_then(|v| v.as_array()) {
                        let steps: Vec<(Option<String>, Option<String>)> = arr.iter().map(|s| (
                            s.get("selector").and_then(|v| v.as_str()).map(String::from),
                            s.get("text").and_then(|v| v.as_str()).map(String::from),
                        )).collect();
                        crate::browser::click_seq(steps).unwrap_or_else(|e| format!("ERROR: {e}"))
                    } else {
                        crate::browser::click(g("selector"), g("text")).unwrap_or_else(|e| format!("ERROR: {e}"))
                    }
                }
                "browser_snapshot" | "browser_screenshot" => { screenshots += 1; "(截图)".into() }
                other => format!("未知工具 {other}"),
            };
            steps_log.push(name.clone());
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
            messages.push(ChatMessage { images: Vec::new(), role: Role::User, content: format!("<tool_result>{}</tool_result>\n/no_think", result), reasoning_content: None });
            if submitted() { break; }
        }
        let elapsed = started.elapsed();

        // ── AUDIT: field-by-field final DOM state (not the model's word) ──
        let field = |id: &str| crate::browser::eval(&format!("document.getElementById('{id}')?document.getElementById('{id}').value:''")).unwrap_or_default().trim_matches('"').to_string();
        let name = field("f_name");
        let email = field("f_email");
        let company = field("f_company");
        let phone = field("f_phone");
        let city = field("f_city");
        let why = field("f_why");
        let agreed = crate::browser::eval("document.getElementById('f_agree').checked").unwrap_or_default().contains("true");
        let done = submitted();
        crate::browser::shutdown();
        let _ = std::fs::remove_dir_all(&dir);

        eprintln!("\n===== LONG FORM AUDIT =====");
        eprintln!("steps={} ({:.1}s) type_calls={type_calls} fields_typed={fields_typed} screenshots={screenshots}", steps_log.len(), elapsed.as_secs_f32());
        eprintln!("path: {}", steps_log.join(" → "));
        eprintln!("name='{name}' email='{email}' company='{company}' phone='{phone}' city='{city}' why='{}' agreed={agreed} submitted={done}", why.chars().take(30).collect::<String>());
        eprintln!("===========================\n");

        // ── Hard assertions on the ACTUAL DOM (accuracy) ──
        assert!(done, "the form was NOT actually submitted; steps: {steps_log:?}");
        assert_eq!(name, "Alice Chen", "name mis-filled");
        assert_eq!(email, "alice@nimbus.io", "email mis-filled");
        assert_eq!(company, "Acme", "company mis-filled");
        assert_eq!(phone, "5551234567", "phone mis-filled");
        assert_eq!(city, "Montreal", "city mis-filled");
        assert!(!why.is_empty(), "motivation left empty");
        assert!(agreed, "terms checkbox not ticked");
        // Optimality: 6 fields via batching should take far fewer type calls
        // than 6, and the whole thing well under the step budget.
        assert!(steps_log.len() <= 12, "took too many steps ({}), not optimal", steps_log.len());
        eprintln!("✓ accurate + optimal: {} fields filled in {} type call(s), {} total steps", fields_typed, type_calls, steps_log.len());
    }
}

/// PRODUCTION E2E (balance check): a task whose correctness is VISUAL — the page
/// shows a raster IMAGE (red field + yellow disc) with NO text describing it, so
/// the ONLY way to answer "what/what color is in the picture" is to LOOK
/// (screenshot → vision). Proves the rebalanced prompt still makes the model
/// reach for vision when text can't answer — the complement to
/// `password_game_no_vision` (pure text → 0 screenshots).
///
///   CHATY_TEST_VLM=… cargo test -p chaty visual_task_uses_vision -- --ignored --nocapture
#[cfg(test)]
mod visual_verify_e2e {
    use super::*;
    use std::cell::RefCell;

    struct Collector { buf: RefCell<String> }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev { self.buf.borrow_mut().push_str(&text); }
            Ok(())
        }
    }
    fn strip_think(s: &str) -> String {
        let mut out = String::new(); let mut rest = s;
        while let Some(i) = rest.find("<think>") { out.push_str(&rest[..i]); if let Some(j)=rest[i..].find("</think>"){rest=&rest[i+j+8..];} else {rest="";} }
        out.push_str(rest); out.trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") { body = &body[..c]; }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?; let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        let args = json.get("arguments").or_else(|| json.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
        Some((name, args))
    }

    #[test]
    #[ignore]
    fn visual_task_uses_vision() {
        let model_path = match std::env::var("CHATY_TEST_VLM") { Ok(p) => p, Err(_) => { eprintln!("SKIP: CHATY_TEST_VLM"); return; } };
        if crate::browser::chrome_path_pub().is_none() { eprintln!("SKIP: no Chrome"); return; }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");

        let dir = std::env::temp_dir().join(format!("chaty-visual-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A picture whose content lives ONLY in pixels: a solid RED field with a
        // big YELLOW disc in the middle. Nothing in the DOM/text says so.
        let mut img = image::RgbImage::from_pixel(480, 360, image::Rgb([210, 30, 30]));
        let (cx, cy, r) = (240i32, 180i32, 110i32);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let (dx, dy) = (x as i32 - cx, y as i32 - cy);
            if dx * dx + dy * dy <= r * r { *px = image::Rgb([240, 210, 20]); }
        }
        let img_path = dir.join("pic.png");
        image::DynamicImage::ImageRgb8(img).save(&img_path).unwrap();
        // Deliberately mislead a text-only reader: the ALT text is generic.
        let html = format!(
            "<!doctype html><meta charset=utf-8><title>Gallery</title><body style='font-family:sans-serif;text-align:center'>\
             <h1>My Gallery</h1><p>Here is today's featured picture.</p>\
             <img src='file://{}/pic.png' alt='featured image' style='width:480px'>\
             <p>Enjoy the artwork.</p></body>",
            dir.display()
        );
        std::fs::write(dir.join("g.html"), html).unwrap();
        let url = format!("file://{}/g.html", dir.display());

        let backend = crate::inference::llama::llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_threads(nt).with_n_threads_batch(nt);
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;

        let sys = format!(
            "你是浏览器自动化助手。每步只输出一行 <tool_call>{{\"name\":..,\"arguments\":{{..}}}}</tool_call> 然后停止,系统会用 <tool_result> 回你。\n\
             工具:browser_navigate {{url}} / browser_read {{}}(读文字) / browser_screenshot {{}}(整页截图给你看) / browser_snapshot {{}}(当前屏截图给你看)。\n\
             判断:要文字/状态用 browser_read;要判断图片画的是什么、颜色、外观,读文字看不出来,必须用 browser_screenshot/snapshot 亲眼看。首个页面:{url}"
        );
        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: sys, reasoning_content: None },
            ChatMessage { images: Vec::new(), role: Role::User, content: format!("打开 {url},页面上有一张图片。告诉我这张图片主要是什么颜色、中间画的是什么形状。\n/no_think"), reasoning_content: None },
        ];

        let cancel = AtomicBool::new(false);
        let mut steps: Vec<String> = Vec::new();
        let mut final_text = String::new();

        for step in 0..10 {
            let req = GenRequest { messages: messages.clone(), params: GenParams { temperature: 0.3, max_tokens: 500, think: Some(false), ..Default::default() } };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, Some(&mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            let Some((name, args)) = parse_tool_call(&raw) else {
                final_text = strip_think(&raw);
                eprintln!("--- step {step}: FINAL: {}", final_text.chars().take(120).collect::<String>());
                break;
            };
            eprintln!("--- step {step}: {name}");
            steps.push(name.clone());
            let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
            let (result, image): (String, Option<String>) = match name.as_str() {
                "browser_navigate" => (crate::browser::navigate(&g("url").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_read" => (crate::browser::read_page().unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_snapshot" | "browser_screenshot" => {
                    let png = if name == "browser_snapshot" { crate::browser::snapshot() } else { crate::browser::screenshot() };
                    match png {
                        Ok(b) => { let p = dir.join(format!("shot-{step}.png")); std::fs::write(&p, b).unwrap(); ("(截图已附上)".into(), Some(p.to_string_lossy().to_string())) }
                        Err(e) => (format!("ERROR: {e}"), None),
                    }
                }
                other => (format!("未知工具 {other}"), None),
            };
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
            if let Some(img) = image {
                messages.push(ChatMessage { images: vec![img], role: Role::User, content: "<tool_result>这是当前页面截图,请查看后回答。</tool_result>\n/no_think".into(), reasoning_content: None });
            } else {
                messages.push(ChatMessage { images: Vec::new(), role: Role::User, content: format!("<tool_result>{result}</tool_result>\n/no_think"), reasoning_content: None });
            }
        }
        crate::browser::shutdown();
        let _ = std::fs::remove_dir_all(&dir);

        let shots = steps.iter().filter(|s| *s == "browser_screenshot" || *s == "browser_snapshot").count();
        let low = final_text.to_lowercase();
        let saw_red = low.contains("red") || final_text.contains("红");
        let saw_yellow = low.contains("yellow") || final_text.contains("黄") || low.contains("gold");
        let saw_circle = low.contains("circle") || low.contains("disc") || final_text.contains("圆") || final_text.contains("圆形");
        eprintln!("=== visual task: steps={} shots={shots} :: {} | answer red={saw_red} yellow={saw_yellow} circle={saw_circle}\n    {final_text}", steps.len(), steps.join(" → "));

        // BALANCE: a visual-correctness task MUST use vision (not just read text).
        assert!(shots >= 1, "a visual task must use a screenshot/snapshot — the model skipped vision; steps: {steps:?}");
        // And it must get the picture right from the pixels.
        assert!(saw_red && (saw_yellow || saw_circle), "vision answer wrong (expected red field + yellow circle): {final_text}");
    }
}

/// PRODUCTION E2E: a Duolingo-style "build the sentence by tapping words in
/// order" task. Proves the model (a) picks the words with ONE batched
/// browser_click(steps) instead of one-at-a-time clicks, and (b) takes a
/// snapshot/screenshot to VISUALLY confirm the assembled sentence BEFORE it
/// clicks the irreversible Check/Submit button.
///
///   CHATY_TEST_VLM=… cargo test -p chaty duolingo_order_click_verify -- --ignored --nocapture
#[cfg(test)]
mod duolingo_e2e {
    use super::*;
    use std::cell::RefCell;

    struct Collector { buf: RefCell<String> }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev { self.buf.borrow_mut().push_str(&text); }
            Ok(())
        }
    }
    fn strip_think(s: &str) -> String {
        let mut out = String::new(); let mut rest = s;
        while let Some(i) = rest.find("<think>") { out.push_str(&rest[..i]); if let Some(j)=rest[i..].find("</think>"){rest=&rest[i+j+8..];} else {rest="";} }
        out.push_str(rest); out.trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") { body = &body[..c]; }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?; let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        let args = json.get("arguments").or_else(|| json.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
        Some((name, args))
    }

    // A word-bank sentence builder: tapping a word appends it to the answer
    // line; "Check" grades the sentence against the target. Target sentence is
    // "I drink coffee every morning" — the bank has those words plus distractors
    // in a shuffled order, so the model must tap them in the RIGHT order.
    const HTML: &str = r##"<!doctype html><meta charset="utf-8"><title>Translate</title>
<body style="font-family:sans-serif;max-width:640px;margin:2rem auto;text-align:center">
<h2>Build the sentence</h2>
<p>Translate: “我每天早上喝咖啡”</p>
<div id="answer" style="min-height:40px;border-bottom:2px solid teal;font-size:20px;padding:8px">&nbsp;</div>
<div id="bank" style="margin-top:20px"></div>
<p><button id="check" style="font-size:16px;padding:8px 20px">Check</button></p>
<div id="result" style="font-weight:bold;margin-top:16px"></div>
<script>
var TARGET="I drink coffee every morning";
var WORDS=["coffee","every","I","banana","drink","morning","quickly"];
var picked=[];
var bank=document.getElementById("bank");
WORDS.forEach(function(w){
  var b=document.createElement("button");
  b.textContent=w; b.style.margin="4px"; b.style.fontSize="16px"; b.style.padding="6px 12px";
  b.addEventListener("click",function(){ picked.push(w); b.disabled=true; b.style.opacity=0.4; render(); });
  bank.appendChild(b);
});
function render(){ document.getElementById("answer").textContent = picked.join(" ")||" "; }
document.getElementById("check").addEventListener("click",function(){
  var ok = picked.join(" ")===TARGET;
  document.getElementById("result").textContent = ok? "CORRECT — well done!" : ("WRONG: you built \""+picked.join(" ")+"\"");
  document.body.dataset.correct = ok? "1":"0";
  document.body.dataset.checked = "1";
});
</script></body>"##;

    #[test]
    #[ignore]
    fn duolingo_order_click_verify() {
        let model_path = match std::env::var("CHATY_TEST_VLM") { Ok(p) => p, Err(_) => { eprintln!("SKIP: CHATY_TEST_VLM"); return; } };
        if crate::browser::chrome_path_pub().is_none() { eprintln!("SKIP: no Chrome"); return; }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");

        let dir = std::env::temp_dir().join(format!("chaty-duo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("q.html"), HTML).unwrap();
        let url = format!("file://{}/q.html", dir.display());

        let backend = crate::inference::llama::llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_threads(nt).with_n_threads_batch(nt);
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;

        let sys = format!(
            "你是浏览器自动化助手。每步只输出一行 <tool_call>{{\"name\":..,\"arguments\":{{..}}}}</tool_call> 然后停止,系统会用 <tool_result> 回你。\n\
             工具:browser_navigate {{url}} / browser_read {{}} / browser_snapshot {{}}(截当前屏给你看) / browser_screenshot {{}} / browser_click(点按钮;**一次按顺序点多个用 steps**,如 {{\"steps\":[{{\"text\":\"I\"}},{{\"text\":\"like\"}}]}})。\n\
             这是一道选词造句题:点击词库里的单词按正确顺序拼成目标句子,再点 Check。\n\
             要求:①想好完整顺序后**用 browser_click 的 steps 一次把所有词按顺序点完**,不要一个一个点;②点 Check(会判分、不可逆)**之前先用 browser_snapshot 截屏,用视觉确认拼出的句子完全正确**,再点 Check。首个页面:{url}"
        );
        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: sys, reasoning_content: None },
            ChatMessage { images: Vec::new(), role: Role::User, content: format!("打开 {url},把「我每天早上喝咖啡」这道选词造句题做对(目标英文句子是 I drink coffee every morning),按要求先批量选词、提交前截图确认,再点 Check。\n/no_think"), reasoning_content: None },
        ];

        let cancel = AtomicBool::new(false);
        let mut steps: Vec<(String, serde_json::Value)> = Vec::new();
        let checked = || crate::browser::eval("document.body.dataset.checked||'0'").map(|s| s.contains('1')).unwrap_or(false);

        for step in 0..14 {
            let req = GenRequest { messages: messages.clone(), params: GenParams { temperature: 0.2, max_tokens: 600, think: Some(false), ..Default::default() } };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(&model, &mut ctx, &mut cached, Some(&mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            let Some((name, args)) = parse_tool_call(&raw) else {
                eprintln!("--- step {step}: FINAL: {}", strip_think(&raw).chars().take(90).collect::<String>());
                break;
            };
            eprintln!("--- step {step}: {name} {}", args.to_string().chars().take(150).collect::<String>());
            steps.push((name.clone(), args.clone()));
            let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
            let (result, image): (String, Option<String>) = match name.as_str() {
                "browser_navigate" => (crate::browser::navigate(&g("url").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_read" => (crate::browser::read_page().unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_click" => {
                    if let Some(arr) = args.get("steps").and_then(|v| v.as_array()) {
                        let s: Vec<(Option<String>,Option<String>)> = arr.iter().map(|x| (x.get("selector").and_then(|v|v.as_str()).map(String::from), x.get("text").and_then(|v|v.as_str()).map(String::from))).collect();
                        (crate::browser::click_seq(s).unwrap_or_else(|e| format!("ERROR: {e}")), None)
                    } else {
                        (crate::browser::click(g("selector"), g("text")).unwrap_or_else(|e| format!("ERROR: {e}")), None)
                    }
                }
                "browser_snapshot" | "browser_screenshot" => {
                    let png = if name == "browser_snapshot" { crate::browser::snapshot() } else { crate::browser::screenshot() };
                    match png { Ok(b) => { let p = dir.join(format!("s-{step}.png")); std::fs::write(&p, b).unwrap(); ("(截图已附上)".into(), Some(p.to_string_lossy().to_string())) } Err(e) => (format!("ERROR: {e}"), None) }
                }
                other => (format!("未知工具 {other}"), None),
            };
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
            if let Some(img) = image {
                messages.push(ChatMessage { images: vec![img], role: Role::User, content: "<tool_result>这是当前页面截图,请查看后继续。</tool_result>\n/no_think".into(), reasoning_content: None });
            } else {
                messages.push(ChatMessage { images: Vec::new(), role: Role::User, content: format!("<tool_result>{result}</tool_result>\n/no_think"), reasoning_content: None });
            }
            if checked() { break; }
        }

        // ── Analyse the tool trace ──
        let names: Vec<&str> = steps.iter().map(|(n,_)| n.as_str()).collect();
        // words picked via batched click(steps): count clicks that carried a steps array of length >=2
        let batched_word_clicks = steps.iter().any(|(n,a)| n=="browser_click" && a.get("steps").and_then(|v|v.as_array()).map(|x|x.len()>=2).unwrap_or(false));
        let single_word_clicks = steps.iter().filter(|(n,a)| n=="browser_click" && a.get("steps").is_none() && a.get("text").and_then(|v|v.as_str()).map(|t| t!="Check").unwrap_or(false)).count();
        // a snapshot/screenshot BEFORE the Check click?
        let check_idx = steps.iter().position(|(n,a)| n=="browser_click" && (a.get("text").and_then(|v|v.as_str())==Some("Check") || a.get("steps").and_then(|v|v.as_array()).map(|x| x.iter().any(|s| s.get("text").and_then(|v|v.as_str())==Some("Check"))).unwrap_or(false)));
        let shot_before_check = match check_idx {
            Some(ci) => steps[..ci].iter().any(|(n,_)| n=="browser_snapshot" || n=="browser_screenshot"),
            None => false,
        };
        let correct = crate::browser::eval("document.body.dataset.correct||'0'").map(|s| s.contains('1')).unwrap_or(false);
        let done = checked();
        crate::browser::shutdown();
        let _ = std::fs::remove_dir_all(&dir);

        eprintln!("\n===== DUOLINGO AUDIT =====");
        eprintln!("steps={} path: {}", names.len(), names.join(" → "));
        eprintln!("batched_word_clicks={batched_word_clicks} single_word_clicks={single_word_clicks} snapshot_before_check={shot_before_check} checked={done} correct={correct}");
        eprintln!("==========================\n");

        // ── Hard assertions ──
        assert!(done, "the model never clicked Check; steps: {names:?}");
        assert!(correct, "the built sentence was wrong");
        // (1) ordered words picked as ONE batch, not one-at-a-time.
        assert!(batched_word_clicks, "words should be picked with ONE batched browser_click(steps), not single clicks; steps: {names:?}");
        assert!(single_word_clicks <= 1, "too many one-at-a-time word clicks ({single_word_clicks}) — should batch");
        // (2) a visual confirmation BEFORE hitting the irreversible Check.
        assert!(shot_before_check, "must snapshot/screenshot to visually confirm BEFORE clicking Check; steps: {names:?}");
        eprintln!("✓ batched the word taps AND visually confirmed before submitting");
    }
}

/// PROBE (not a hard-pass gate): run the 35B against MANY REAL websites — real
/// browser, real model — covering the tool-selection decision points that keep
/// going wrong (web_fetch vs browser; batched vs single clicks; when to
/// screenshot). Prints a scorecard per scenario so we can see where the model
/// picks a sub-optimal tool or fails, then optimize the prompt/tools. Uses
/// quotes.toscrape.com (built for automation practice, stable) + Wikipedia / HN
/// / example.com. Network + Chrome required; skips offline.
///
///   CHATY_TEST_VLM=… cargo test -p chaty real_web_scenarios -- --ignored --nocapture
#[cfg(test)]
mod real_scenarios_e2e {
    use super::*;
    use std::cell::RefCell;

    struct Collector { buf: RefCell<String> }
    impl EventSink for Collector {
        fn emit(&self, ev: StreamEvent) -> Result<()> {
            if let StreamEvent::Token { text } = ev { self.buf.borrow_mut().push_str(&text); }
            Ok(())
        }
    }
    fn strip_think(s: &str) -> String {
        let mut out = String::new(); let mut rest = s;
        while let Some(i) = rest.find("<think>") { out.push_str(&rest[..i]); if let Some(j)=rest[i..].find("</think>"){rest=&rest[i+j+8..];} else {rest="";} }
        out.push_str(rest); out.trim().to_string()
    }
    fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        let open = text.find("<tool_call>")?;
        let mut body = &text[open + "<tool_call>".len()..];
        if let Some(c) = body.find("</tool_call>") { body = &body[..c]; }
        let body = body.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
        let s = body.find('{')?; let e = body.rfind('}')?;
        let json: serde_json::Value = serde_json::from_str(&body[s..=e]).ok()?;
        let name = json.get("name")?.as_str()?.to_string();
        let args = json.get("arguments").or_else(|| json.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
        Some((name, args))
    }

    struct Report { name: &'static str, done: bool, steps: Vec<String>, final_text: String, note: String }

    #[allow(clippy::too_many_arguments)]
    fn drive(
        model: &LlamaModel, backend: &LlamaBackend, mtmd: &MtmdContext, n_ctx: u32, nt: i32,
        rt: &tokio::runtime::Runtime, name: &'static str, sys: &str, task: &str, max_steps: usize,
        done: &dyn Fn() -> bool,
    ) -> Report {
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_threads(nt).with_n_threads_batch(nt);
        let mut ctx = model.new_context(backend, ctx_params).expect("ctx");
        let mut cached: Vec<LlamaToken> = Vec::new();
        let mut media_cache: Option<MediaCache> = None;
        let mut messages = vec![
            ChatMessage { images: Vec::new(), role: Role::System, content: sys.to_string(), reasoning_content: None },
            ChatMessage { images: Vec::new(), role: Role::User, content: format!("{task}\n/no_think"), reasoning_content: None },
        ];
        let cancel = AtomicBool::new(false);
        let mut steps: Vec<String> = Vec::new();
        let mut final_text = String::new();
        let mut last_key = String::new();
        let mut repeat = 0usize;
        for step in 0..max_steps {
            let req = GenRequest { messages: messages.clone(), params: GenParams { temperature: 0.25, max_tokens: 900, think: Some(false), ..Default::default() } };
            let sink = Collector { buf: RefCell::new(String::new()) };
            run_turn(model, &mut ctx, &mut cached, Some(mtmd), &mut media_cache, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            let Some((tname, args)) = parse_tool_call(&raw) else { final_text = strip_think(&raw); eprintln!("  [{name}] s{step} FINAL: {}", final_text.chars().take(90).collect::<String>()); break; };
            eprintln!("  [{name}] s{step}: {tname} {}", args.to_string().chars().take(120).collect::<String>());
            // repeat breaker (non-exempt identical)
            let key = format!("{tname}:{args}");
            let exempt = matches!(tname.as_str(), "browser_scroll"|"browser_screenshot"|"browser_snapshot"|"browser_read"|"browser_console"|"bg_output");
            if exempt { last_key.clear(); repeat = 0; } else if key == last_key { repeat += 1; } else { last_key = key; repeat = 0; }
            if repeat >= 1 && !exempt {
                steps.push(format!("{tname}*BLK"));
                messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
                messages.push(ChatMessage { images: Vec::new(), role: Role::User, content: "<tool_result>这一步和上一步完全相同,已拦截,换做法。</tool_result>\n/no_think".into(), reasoning_content: None });
                continue;
            }
            steps.push(tname.clone());
            let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
            let click_steps = || args.get("steps").and_then(|v| v.as_array()).map(|arr| arr.iter().map(|x| (x.get("selector").and_then(|v|v.as_str()).map(String::from), x.get("text").and_then(|v|v.as_str()).map(String::from))).collect::<Vec<_>>());
            let type_steps = || args.get("steps").and_then(|v| v.as_array()).map(|arr| arr.iter().map(|x| (x.get("selector").and_then(|v|v.as_str()).map(String::from), x.get("label").and_then(|v|v.as_str()).map(String::from), x.get("text").and_then(|v|v.as_str()).unwrap_or_default().to_string())).collect::<Vec<_>>());
            let (result, image): (String, Option<String>) = match tname.as_str() {
                "browser_navigate" => (crate::browser::navigate(&g("url").unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_read" => (crate::browser::read_page().unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_console" => (crate::browser::console().unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_scroll" => (crate::browser::scroll_page(g("to"), args.get("by").and_then(|v|v.as_f64())).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_click" => (if let Some(s)=click_steps(){crate::browser::click_seq(s)}else{crate::browser::click(g("selector"),g("text"))}.unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_type" => (if let Some(s)=type_steps(){crate::browser::type_seq(s)}else{crate::browser::type_text(g("selector"),g("label"),g("text").unwrap_or_default())}.unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_eval" => (crate::browser::eval(&g("expression").or_else(||g("expr")).unwrap_or_default()).unwrap_or_else(|e| format!("ERROR: {e}")), None),
                "browser_snapshot" | "browser_screenshot" => {
                    let png = if tname=="browser_snapshot"{crate::browser::snapshot()}else{crate::browser::screenshot()};
                    match png { Ok(b)=>{let p=std::env::temp_dir().join(format!("chaty-rs-{}-{step}.png",std::process::id()));std::fs::write(&p,b).unwrap();("(截图已附上)".into(),Some(p.to_string_lossy().to_string()))} Err(e)=>(format!("ERROR: {e}"),None) }
                }
                "web_fetch" => match rt.block_on(crate::search::fetch_url(g("url").unwrap_or_default())) { Ok(pc)=>(format!("标题:{}\n正文:\n{}",pc.title,pc.text.chars().take(2000).collect::<String>()),None), Err(e)=>(format!("ERROR: {e}"),None) },
                "web_search" => match rt.block_on(crate::search::web_search(g("query").unwrap_or_default())) { Ok(rs)=>{let s=rs.iter().take(6).map(|r|format!("- {} — {}\n  {}",r.title,r.url,r.snippet)).collect::<Vec<_>>().join("\n");(if s.is_empty(){"(无结果)".into()}else{s},None)} Err(e)=>(format!("ERROR: {e}"),None) },
                other => (format!("未知工具 {other}"), None),
            };
            messages.push(ChatMessage { images: Vec::new(), role: Role::Assistant, content: strip_think(&raw), reasoning_content: None });
            if let Some(img) = image {
                messages.push(ChatMessage { images: vec![img], role: Role::User, content: "<tool_result>这是当前页面截图,请查看后继续。</tool_result>\n/no_think".into(), reasoning_content: None });
            } else {
                messages.push(ChatMessage { images: Vec::new(), role: Role::User, content: format!("<tool_result>{}</tool_result>\n/no_think", result.chars().take(6000).collect::<String>()), reasoning_content: None });
            }
            if done() && !final_text.is_empty() { break; }
        }
        let finished = done() || !final_text.is_empty();
        Report { name, done: finished, steps, final_text, note: String::new() }
    }

    #[test]
    #[ignore]
    fn real_web_scenarios() {
        let model_path = match std::env::var("CHATY_TEST_VLM") { Ok(p)=>p, Err(_)=>{eprintln!("SKIP: CHATY_TEST_VLM");return;} };
        if crate::browser::chrome_path_pub().is_none() { eprintln!("SKIP: no Chrome"); return; }
        std::env::set_var("CHATY_BROWSER_HEADLESS", "1");
        // Quick net check.
        let rt = tokio::runtime::Runtime::new().unwrap();
        if rt.block_on(crate::search::fetch_url("https://example.com".into())).is_err() { eprintln!("SKIP: no network"); return; }

        let backend = crate::inference::llama::llama_backend_pub().unwrap();
        let mparams = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(backend, &model_path, &mparams).expect("load");
        let n_ctx = 8192u32;
        let nt = crate::gpu::cpu_worker_threads() as i32;
        let mtmd_params = MtmdContextParams { use_gpu: true, n_threads: nt, ..MtmdContextParams::default() };
        let mmproj = find_mmproj(&model_path).expect("mmproj");
        let mtmd = MtmdContext::init_from_file(&mmproj.to_string_lossy(), &model, &mtmd_params).expect("mtmd");

        // Reuse the PRODUCTION system prompt so the probe measures what ships.
        let sys = systemPrompt_for_probe();

        let mut reports: Vec<Report> = Vec::new();
        macro_rules! run { ($n:expr,$t:expr,$m:expr,$d:expr) => { reports.push(drive(&model,backend,&mtmd,n_ctx,nt,&rt,$n,&sys,$t,$m,&$d)); }; }

        // 1. PURE READ — should use web_fetch, NOT the browser.
        run!("read-wikipedia", "查一下英文维基百科 Coffee 词条,咖啡因(caffeine)属于哪一类生物碱/化合物?一句话回答。", 6, || false);
        // 2. PURE READ — HN front page top story (web_fetch).
        run!("read-hn", "打开 Hacker News 首页 https://news.ycombinator.com,告诉我当前排在最上面那条帖子的标题。", 6, || false);
        // 3. BROWSER navigation + pagination.
        run!("quotes-page2", "打开 https://quotes.toscrape.com,翻到第 2 页(点 Next),告诉我第 2 页第一条名言的作者是谁。", 10, || false);
        // 4. BROWSER click a tag filter.
        run!("quotes-tag", "打开 https://quotes.toscrape.com,点击标签 'love' 进入该标签页,告诉我该标签页上第一条名言的作者是谁。", 8, || false);
        // 5. BROWSER login form (accepts any credentials).
        run!("quotes-login", "打开 https://quotes.toscrape.com/login,用户名 admin 密码 admin 登录,登录成功后页面右上角会出现 Logout 链接,确认登录成功。", 10,
            || crate::browser::eval("/logout/i.test(document.body.innerText)?'Y':'N'").map(|s|s.contains('Y')).unwrap_or(false));
        // 6. BROWSER search form with two dropdowns + submit (ordered selects).
        run!("quotes-search", "打开 https://quotes.toscrape.com/search.aspx,在 author 下拉选 'Albert Einstein',再在 tag 下拉里选一个可选的标签,点 Search,告诉我搜到的第一条名言。", 12, || false);

        let get = |n: &str| reports.iter().find(|r| r.name==n).unwrap();
        let uses = |r: &Report, t: &str| r.steps.iter().any(|s| s==t);

        eprintln!("\n================= REAL WEB SCENARIOS SCORECARD =================");
        let mut completed = 0;
        for r in &reports {
            let browser = uses(r, "browser_navigate");
            let webfetch = uses(r, "web_fetch") || uses(r, "web_search");
            let shots = r.steps.iter().filter(|s| *s=="browser_screenshot"||*s=="browser_snapshot").count();
            if r.done { completed += 1; }
            eprintln!("[{}] done={} steps={} browser={browser} web={webfetch} shots={shots}\n    path: {}\n    final: {}",
                r.name, r.done, r.steps.len(), r.steps.join(" → "), r.final_text.chars().take(110).collect::<String>());
        }
        // Optimality notes (read tasks should NOT open the browser).
        for n in ["read-wikipedia","read-hn"] { let r=get(n); if uses(r,"browser_navigate"){ eprintln!("⚠ {n}: opened browser for a pure-read task (web_fetch was optimal)"); } }
        eprintln!("completed {completed}/{} scenarios", reports.len());
        eprintln!("===============================================================\n");

        // Tolerant gate (real sites can flake): most scenarios must complete.
        assert!(completed >= reports.len() - 1, "too many real-web scenarios failed: {completed}/{}", reports.len());
    }

    // The production Code-mode system prompt (vision-ready), rendered for the
    // probe so it measures the SAME guidance users get. Kept in sync by hand
    // with agentLoop.ts systemPrompt(); if they drift, the probe still works —
    // it just measures this copy.
    fn systemPrompt_for_probe() -> String {
        // A faithful condensation of the shipping browser+web guidance.
        "你是 Chaty 的浏览器/网页自动化助手,帮用户在真实网页上完成任务。每步只输出一行 <tool_call>{\"name\":..,\"arguments\":{..}}</tool_call> 然后停止,系统会用 <tool_result> 回你。\n\
         工具:\n\
         - web_search {query} / web_fetch {url}:联网搜索 / 抓取网址正文。**纯查资料、读文章、找一个事实,优先用它们**(比开浏览器快得多);拿到答案就直接回答,别再开浏览器重复核实。\n\
         - browser_navigate {url}:打开页面,返回页面全部可见文字+可交互元素。\n\
         - browser_read {}:读当前页面全部可见文字+输入框当前值(看页面/确认状态用它,不用截图)。\n\
         - browser_snapshot {} / browser_screenshot {}:截图用视觉看(要判断排版/图片/渲染是否正确,或提交不可逆操作前确认时用)。\n\
         - browser_click {text|selector|steps}:点击;**已想好顺序的多次点击必须用 steps 一次点完**,如 {\"steps\":[{\"text\":\"A\"},{\"text\":\"B\"}]}。\n\
         - browser_type {label|selector|text|steps}:填输入框;**下拉框(select)也用它——text 传选项可见文字即可,别去 click 下拉选项**;**多个字段用 steps 一次填完**。\n\
         - browser_scroll {to?,by?}:滚动。\n\
         规则:①纯读资料用 web_fetch/web_search,不要开浏览器;②需要真实操作网页(点、填、登录、翻页、必须亲眼看到渲染)才用浏览器;③浏览器里看页面/确认状态优先 browser_read,交互返回已带最新页面文字;④点提交/登录/不可逆按钮前若涉及正确性,先截图视觉确认;⑤顺序点击/多字段填写用 steps 批量;⑥点登录/翻页/Search/提交等按钮后先读返回文字确认结果,成功就别重复点,翻页要点一次读一次别猛点。任务完成后直接用一句话回答。".to_string()
    }
}

/// Cross-conversation isolation for the GGUF engine. Hybrid/recurrent
/// memories (Qwen3.5 / 3.6) cannot partially rewind — `clear_kv_cache_seq`
/// reports `false` — and the engine must fall back to a full clear instead
/// of silently appending the new conversation on top of the old one's state.
///   CHATY_TEST_GGUF=/path/to/model.gguf \
///     cargo test -p chaty --lib gguf_e2e_no_cross_conversation_bleed -- --ignored --nocapture
#[cfg(test)]
mod gguf_kv_e2e {
    use super::*;
    use crate::inference::{ChatMessage, GenParams, GenRequest, InferenceBackend, Role};
    use std::sync::atomic::AtomicBool;

    #[test]
    #[ignore]
    fn gguf_e2e_no_cross_conversation_bleed() {
        let Ok(path) = std::env::var("CHATY_TEST_GGUF") else {
            eprintln!("SKIP: set CHATY_TEST_GGUF=/path/to/model.gguf");
            return;
        };
        let (engine, info) = LlamaEngine::load(&path, None, Some(4096)).expect("load");
        eprintln!("arch: {:?}", info.arch);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let run = |content: &str| -> String {
            let req = GenRequest {
                messages: vec![ChatMessage {
                    role: Role::User,
                    content: content.into(),
                    images: vec![],
                    reasoning_content: None,
}],
                params: GenParams { max_tokens: 96, think: Some(false), ..Default::default() },
            };
            rt.block_on(engine.generate_collect(req, Arc::new(AtomicBool::new(false))))
                .expect("generate")
        };

        // Conversation A plants a canary; conversation B must never see it.
        let a = run("The secret codeword is BANANA. Acknowledge with OK and nothing else.");
        eprintln!("conv A: {a:?}");
        let b = run("Name the capital of France. Answer with one word only.");
        eprintln!("conv B: {b:?}");
        let lb = b.to_lowercase();
        assert!(lb.contains("paris"), "expected 'Paris' in: {b}");
        assert!(
            !lb.contains("banana") && !lb.contains("codeword") && !lb.contains("secret"),
            "conversation A leaked into B: {b}"
        );
        engine.unload();
    }
}
/// Layers the model's next-token-prediction head occupies, read from the GGUF
/// metadata (`<arch>.nextn_predict_layers`). llama.cpp has a C accessor for
/// this; the Rust binding does not re-export it.
pub(crate) fn nextn_layers(model: &LlamaModel) -> u32 {
    let arch = model.meta_val_str("general.architecture").unwrap_or_default();
    model
        .meta_val_str(&format!("{arch}.nextn_predict_layers"))
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0)
}

/// What llama.cpp's multi-token-prediction drafter needs before it will draft,
/// pinned as an executable note. Everything an MTP integration does is
/// downstream of these four facts, and each one cost a round of guessing:
///
/// - the draft context is a SECOND context over the same model, asked for with
///   `LlamaContextType::Mtp`, and it stands up with GPU offload just fine;
/// - the target's prompt must be decoded with an output requested on EVERY
///   position — the nextn head reads hidden states, and llama.cpp only
///   produces the rows a batch asked for;
/// - `process` must see each prefill batch, and `begin` only afterwards: it
///   inspects the DRAFT context's position and warns if the prompt never
///   reached it;
/// - `draft`'s `n_past` is the number of tokens resident in the cache, and
///   `id_last` is the last of them — a token whose hidden state already exists.
///   Drafting from a token that has not been decoded yet cannot work.
///
///   CHATY_TEST_MTP=<gguf with nextn layers> [CHATY_TEST_MTP_GPU=999] \
///     cargo test --lib mtp_probe -- --ignored --nocapture
#[cfg(test)]
mod mtp_probe {
    use super::*;
    use llama_cpp_2::context::params::LlamaContextType;
    use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};

    #[test]
    #[ignore]
    fn mtp_draft_context_stands_up_and_drafts() {
        let path = std::env::var("CHATY_TEST_MTP").expect("set CHATY_TEST_MTP=<gguf>");
        let backend = llama_backend().expect("backend");
        let gpu: u32 = std::env::var("CHATY_TEST_MTP_GPU")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let model = LlamaModel::load_from_file(
            backend,
            &path,
            &LlamaModelParams::default().with_n_gpu_layers(gpu),
        )
        .expect("load model");
        assert!(
            nextn_layers(&model) > 0,
            "this model carries no nextn layers — nothing to draft with"
        );

        let base = || {
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(2048))
                .with_n_threads(4)
                .with_n_threads_batch(4)
        };
        let target = model.new_context(backend, base()).expect("target context");
        let draft = model
            .new_context(backend, base().with_context_type(LlamaContextType::Mtp))
            .expect("MTP draft context");

        let mut spec = MtpSpeculative::new(target, draft, MtpSpeculativeParams::default())
            .expect("MtpSpeculative::new");

        let prompt = model
            .str_to_token("The capital of France is", AddBos::Always)
            .expect("tokenize");

        // An output on every position, or the nextn head has no hidden states
        // to read; `process` before `begin`, or the drafter has no prompt.
        let mut batch = LlamaBatch::new(512, 1);
        for (i, t) in prompt.iter().enumerate() {
            batch.add(*t, i as i32, &[0], true).expect("add");
        }
        spec.target_context_mut().decode(&mut batch).expect("decode prompt");
        spec.process(&batch).expect("process");
        spec.begin(&prompt).expect("begin");

        let last = *prompt.last().unwrap();
        let drafted = spec
            .draft(prompt.len() as i32, last, &prompt)
            .expect("draft");
        eprintln!(
            "MTP drafted {} token(s): {:?}",
            drafted.len(),
            drafted
                .iter()
                .map(|t| String::from_utf8_lossy(&piece_bytes(&model, *t)).to_string())
                .collect::<Vec<_>>()
        );
        // `accept` is only valid while a draft is pending — an empty one is not.
        if !drafted.is_empty() {
            spec.accept(drafted.len() as u16).expect("accept");
        }
        assert!(!drafted.is_empty(), "an MTP model should draft at least one token");
    }
}


#[cfg(test)]
mod think_off_prefix {
    use super::*;

    fn msg(role: Role, content: &str) -> ChatMessage {
        ChatMessage { role, content: content.into(), images: vec![], reasoning_content: None }
    }

    #[test]
    fn a_stored_turn_carries_what_the_prompt_prefilled() {
        // Round one ends with `…assistant\n<think>\n\n</think>\n\n` and the model
        // continues from there. Unless round two puts the same block in front of
        // that stored turn, the common prefix ends at the assistant header and
        // the whole conversation is prefilled again — measured at 0% KV reuse on
        // qwen35 and lfm2 with thinking off, which is code mode's default.
        let out = prefixed_assistant_turns(
            &[msg(Role::User, "q"), msg(Role::Assistant, "the answer")],
            true,
        );
        assert_eq!(out[1].content, format!("{THINK_OFF_PREFIX}the answer"));
    }

    #[test]
    fn only_assistant_turns_are_touched() {
        let out = prefixed_assistant_turns(
            &[
                msg(Role::System, "s"),
                msg(Role::User, "q"),
                msg(Role::Tool, "result"),
            ],
            true,
        );
        assert_eq!(out[0].content, "s");
        assert_eq!(out[1].content, "q");
        assert_eq!(out[2].content, "result");
    }

    #[test]
    fn a_turn_that_already_reasons_is_left_alone() {
        // The model wrote a block despite the request; doubling it would break
        // the prefix exactly as badly as omitting it.
        let already = "<think>\nhmm\n</think>\n\nthe answer";
        let out = prefixed_assistant_turns(&[msg(Role::Assistant, already)], true);
        assert_eq!(out[0].content, already);
    }

    #[test]
    fn thinking_on_changes_nothing() {
        let before = [msg(Role::User, "q"), msg(Role::Assistant, "a")];
        let out = prefixed_assistant_turns(&before, false);
        assert_eq!(out[1].content, "a");
    }

    #[test]
    fn the_prefix_matches_what_the_prompt_appends() {
        // The two spellings live apart; if they drift the prefix silently dies.
        assert_eq!(THINK_OFF_PREFIX, "<think>\n\n</think>\n\n");
    }
}
