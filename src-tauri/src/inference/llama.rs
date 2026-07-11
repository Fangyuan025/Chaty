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
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tauri::ipc::Channel;

use super::{ChatMessage, GenParams, GenRequest, GenStats, InferenceBackend, ModelInfo, Role, StreamEvent};

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

/// A unit of work sent to a model's worker thread.
enum Job {
    Generate {
        req: GenRequest,
        sink: Channel<StreamEvent>,
        cancel: Arc<AtomicBool>,
        done: tokio::sync::oneshot::Sender<Result<()>>,
    },
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

        // Load the weights AND allocate the inference context, backing off
        // `n_gpu_layers` on any out-of-memory failure. This covers BOTH the
        // weights and the KV-cache/compute buffers — the latter often OOMs a
        // small GPU even when the weights fit. If even a pure-CPU load runs out
        // of memory, return a clear error instead of a cryptic crash.
        let mut layers = requested.max(0);
        let mut oom_fallback = false;
        let (model, tx, handle) = loop {
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
            let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
            let worker_model = model.clone();
            let handle = std::thread::Builder::new()
                .name("chaty-llama".into())
                .spawn(move || worker(worker_model, n_ctx, n_threads, rx, init_tx))
                .context("failed to start inference thread")?;

            match init_rx.recv() {
                Ok(Ok(())) => break (model, tx, handle),
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
        let warning = if oom_fallback {
            Some("gpu-oom".to_string())
        } else if n_ctx < n_ctx_wanted {
            Some("ctx-clamped".to_string())
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
        let is_qwen35plus = arch_lc.starts_with("qwen35") || arch_lc.starts_with("qwen36");
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
        // The soft switch is Qwen3-only; never offer it on 3.5+/finetunes
        // whose legacy templates still mention it.
        let think_switch = !is_qwen35plus
            && template_usable
            && (template_lc.contains("no_think") || template_lc.contains("/think"));
        let multimodal = model
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
            supports_tools,
            multimodal,
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
}

/// Owns the persistent context for one model and serves jobs until the engine
/// (and its `Sender`) is dropped.
fn worker(
    model: Arc<LlamaModel>,
    n_ctx: u32,
    n_threads: i32,
    rx: Receiver<Job>,
    init: Sender<Result<(), String>>,
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
    let _ = init.send(Ok(()));

    // Tokens currently resident in the KV cache for sequence 0 (positions 0..len).
    let mut cached: Vec<LlamaToken> = Vec::new();

    while let Ok(job) = rx.recv() {
        match job {
            Job::Generate { req, sink, cancel, done } => {
                let result = run_turn(&model, &mut ctx, &mut cached, n_ctx, &req, &sink, &cancel);
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
) -> Result<()> {
    let n_prompt = tokens.len();
    let mut pos = from;
    while pos < n_prompt {
        let end = (pos + n_batch).min(n_prompt);
        batch.clear();
        for (j, tok) in tokens[pos..end].iter().enumerate() {
            batch.add(*tok, (pos + j) as i32, &[0], pos + j == n_prompt - 1)?;
        }
        ctx.decode(batch).context("decode failed")?;
        pos = end;
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
    n_ctx: u32,
    req: &GenRequest,
    sink: &dyn EventSink,
    cancel: &AtomicBool,
) -> Result<()> {
    sink.emit(StreamEvent::Started)?;

    let prompt = build_prompt(model, &req.messages, req.params.think)?;
    // Qwen3.5/3.6-style templates PRE-OPEN the reasoning block: the prompt
    // ends with "<think>\n" and the model starts mid-reasoning, so the UI
    // would never see an opening tag. Emit a synthetic one so the stream is
    // well-formed for the frontend's think-panel parser.
    if req.params.think != Some(false) && prompt.trim_end().ends_with("<think>") {
        sink.emit(StreamEvent::Token { text: "<think>\n".to_string() })?;
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

    // Drop everything in the KV at/after `prefix`, then decode only the new tail.
    if prefix < cached.len() {
        let _ = ctx.clear_kv_cache_seq(Some(0), Some(prefix as u32), None);
    }
    cached.truncate(prefix);

    if cancel.load(Ordering::Relaxed) {
        ctx.clear_kv_cache();
        cached.clear();
        return done_event(sink, n_prompt as u32, 0, 0.0, "cancelled");
    }

    let n_batch = (ctx.n_batch() as usize).max(1);
    let mut batch = LlamaBatch::new(n_batch, 1);

    // Decode the new tail, reusing the cached KV prefix. Some models don't
    // tolerate partial KV reuse (llama.cpp's decode returns an error); if so,
    // clear the KV and decode the whole prompt fresh. If that still fails, reset
    // state so the next turn / new chat starts clean instead of staying broken.
    if let Err(e) = decode_prompt(ctx, &mut batch, &tokens, prefix, n_batch) {
        eprintln!("prompt decode (reuse from {prefix}) failed: {e:#}; retrying from a clean KV");
        ctx.clear_kv_cache();
        if let Err(e2) = decode_prompt(ctx, &mut batch, &tokens, 0, n_batch) {
            ctx.clear_kv_cache();
            cached.clear();
            return Err(e2).context("prompt decode failed");
        }
    }
    *cached = tokens; // KV now holds the full prompt
    let mut idx = batch.n_tokens() - 1;

    let mut sampler = build_sampler(&req.params);
    // Robust incremental UTF-8 assembly: accumulate raw token bytes and only
    // emit the valid-UTF-8 prefix, carrying any incomplete trailing bytes to
    // the next token. This never drops a byte (the old streaming decoder could
    // silently swallow a char at a token boundary).
    let mut pending: Vec<u8> = Vec::new();
    let start = Instant::now();
    let mut n_decoded: u32 = 0;
    let mut n_past = n_prompt as i32;

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
        cached.push(token);
        n_past += 1;
        ctx.decode(&mut batch).context("decode failed")?;
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

    let secs = start.elapsed().as_secs_f32().max(1e-3);
    done_event(sink, n_prompt as u32, n_decoded, n_decoded as f32 / secs, stop_reason)
}

fn done_event(
    sink: &dyn EventSink,
    prompt_tokens: u32,
    completion_tokens: u32,
    tps: f32,
    stop_reason: &str,
) -> Result<()> {
    sink.emit(StreamEvent::Done {
        stats: GenStats {
            prompt_tokens,
            completion_tokens,
            tokens_per_second: tps,
            stop_reason: stop_reason.to_string(),
        },
    })?;
    Ok(())
}

/// Render messages into a prompt using the model's embedded chat template,
/// falling back to ChatML if the GGUF doesn't carry one.
fn build_prompt(model: &LlamaModel, messages: &[ChatMessage], think: Option<bool>) -> Result<String> {
    // Gemma 4 ships a Jinja template the vendored llama.cpp can't parse, and
    // the old built-in "gemma" template uses the wrong (<start_of_turn>) turn
    // delimiters — render its documented format natively instead.
    if is_gemma4(model) {
        return Ok(render_gemma4(messages, think));
    }
    let mut prompt = render_chat(model, messages)?;

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
        .map(|a| {
            let a = a.to_lowercase();
            a.starts_with("qwen35") || a.starts_with("qwen36")
        })
        .unwrap_or(false);
    let template_uses_think = qwen35plus
        || model
            .meta_val_str("tokenizer.chat_template")
            .map(|t| t.contains("<think>"))
            .unwrap_or(false);
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
                prompt.push_str("<think>\n\n</think>\n\n");
            }
        }
    }

    Ok(prompt)
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
fn render_gemma4(messages: &[ChatMessage], think: Option<bool>) -> String {
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
            Role::User => "user",
            Role::Assistant => "model",
        };
        // Strip reasoning channels from prior assistant turns — official
        // templates never feed thought traces back into the context.
        let content = if matches!(m.role, Role::Assistant) {
            strip_thought_channels(&m.content)
        } else {
            m.content.clone()
        };
        p.push_str("<|turn>");
        p.push_str(role);
        p.push('\n');
        p.push_str(content.trim());
        p.push_str("<turn|>\n");
    }
    p.push_str("<|turn>model\n");
    p
}

/// Remove `<|channel>…<channel|>` reasoning spans (and stray markers).
fn strip_thought_channels(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find("<|channel>") {
        out.push_str(&rest[..open]);
        match rest[open..].find("<channel|>") {
            Some(close) => rest = &rest[open + close + "<channel|>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Apply the chat template with a robust fallback chain:
/// 1. the GGUF's embedded template as-is;
/// 2. the embedded template with system messages folded into the first user
///    turn — Gemma-family templates raise "system role not supported";
/// 3. llama.cpp's *built-in* template for the architecture — newer models
///    (e.g. Gemma 3/4) often embed Jinja the vendored llama.cpp can't parse
///    even though the wire format is unchanged;
/// 4. ChatML as a last resort.
fn render_chat(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String> {
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
        if let Ok(p) = model.apply_chat_template(&t, &chat, true) {
            return Ok(p);
        }
        if let Ok(p) = model.apply_chat_template(&t, &folded_chat, true) {
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
            if let Ok(p) = model.apply_chat_template(&t, &folded_chat, true) {
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
                out.push(ChatMessage {
                    role: Role::User,
                    content: format!("{sys_text}\n\n{}", m.content),
                });
            }
            _ => out.push(m.clone()),
        }
    }
    if !injected {
        out.insert(0, ChatMessage { role: Role::User, content: sys_text });
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
    }
}

/// Raw bytes of a token's piece, handling pieces longer than the initial
/// buffer. `special = false` so control tokens render empty.
fn piece_bytes(model: &LlamaModel, token: LlamaToken) -> Vec<u8> {
    match model.token_to_piece_bytes(token, 32, false, None) {
        Ok(b) => b,
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(i)) => model
            .token_to_piece_bytes(token, (-i) as usize, false, None)
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
    use super::*;

    #[test]
    fn strips_a_single_channel_span() {
        assert_eq!(strip_thought_channels("a<|channel>secret<channel|>b"), "ab");
    }

    #[test]
    fn strips_multiple_channel_spans() {
        assert_eq!(
            strip_thought_channels("x<|channel>1<channel|>y<|channel>2<channel|>z"),
            "xyz"
        );
    }

    #[test]
    fn drops_unterminated_channel_tail() {
        // No closing marker: everything from the open tag on is discarded.
        assert_eq!(strip_thought_channels("answer<|channel>still thinking"), "answer");
    }

    #[test]
    fn leaves_plain_text_untouched() {
        assert_eq!(strip_thought_channels("  just a normal reply  "), "just a normal reply");
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
            "read_file" => crate::agent::agent_read_file(get("path"), None, None, None)
                .unwrap_or_else(|e| format!("ERROR: {e}")),
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
            "edit_file" => crate::agent::agent_edit_file(
                get("path"),
                get("old_string"),
                get("new_string"),
                args.get("replace_all").and_then(|v| v.as_bool()),
            )
            .unwrap_or_else(|e| format!("ERROR: {e}")),
            "multi_edit" => {
                let edits: Vec<crate::agent::EditOp> = args
                    .get("edits")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                crate::agent::agent_multi_edit(get("path"), edits)
                    .unwrap_or_else(|e| format!("ERROR: {e}"))
            }
            "outline" => crate::agent::agent_outline(get("path"))
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            "glob" => crate::agent::agent_glob(get("pattern"))
                .map(|h| h.join("\n"))
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            "grep" => crate::agent::agent_grep(get("pattern"), None, None)
                .unwrap_or_else(|e| format!("ERROR: {e}")),
            "bash" => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                match rt.block_on(crate::agent::agent_bash(get("command"), Some(60))) {
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

    /// Prompt documenting the code-editing power tools (multi_edit / outline),
    /// mirroring src/lib/agentLoop.ts.
    const SYS_CODE: &str = r#"你是 Chaty 的编程智能体,在工作区目录中完成编码任务。工作区根目录:{WS}

可用工具(路径相对工作区,越界会被拒绝):
- read_file: {"path": string, "offset"?: number, "limit"?: number}
- write_file: {"path": string, "content": string}
- edit_file: 精确替换一段文本(old_string 需逐字匹配且唯一)。匹配失败会提示文件中最相似的位置。args: {"path": string, "old_string": string, "new_string": string, "replace_all"?: boolean}
- multi_edit: 对同一文件一次提交多处精确替换,原子生效(任何一条失败则整体不改动)。同一文件改多处必须用它。args: {"path": string, "edits": [{"old_string": string, "new_string": string, "replace_all"?: boolean}]}
- outline: 列出文件的定义大纲(函数/类 + 行号),不读全文即可掌握结构。args: {"path": string}
- list_dir: {"path"?: string}
- grep: {"pattern": string}
- bash: {"command": string}

规则(严格遵守):
- 每次只调用一个工具。调用时只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call> 然后立即停止,不要有其它内容。
- 系统会用 <tool_result>...</tool_result> 返回结果,你再继续。
- 修改前先用 outline / read_file 了解结构;同一文件多处修改用一次 multi_edit。
- 任务完成后不要再调用工具,直接用一两句话总结你做了什么。"#;

    /// Refactor e2e: rename a function + update all call sites across the
    /// file — the natural shape for outline + one atomic multi_edit — then
    /// prove the behaviour is unchanged by running the script.
    /// Run: CHATY_TEST_MODEL=… cargo test -p chaty agent_refactors_with_multi_edit -- --ignored --nocapture
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
            ChatMessage { role: Role::System, content: SYS_CODE.replace("{WS}", &ws.to_string_lossy()) },
            ChatMessage {
                role: Role::User,
                content: "把 shop.py 里的函数 calc_total 重命名为 compute_total,并更新文件里所有调用它的地方(先用 outline 看结构,同一文件的多处修改用一次 multi_edit 完成)。改完运行 python3 shop.py 确认输出仍然是 TOTAL: 33.00 和 AUDIT: 30.00。".into(),
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
            run_turn(&model, &mut ctx, &mut cached, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW ────────\n{}", raw.trim().chars().take(500).collect::<String>());

            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    used_multi_edit |= name == "multi_edit";
                    used_outline |= name == "outline";
                    eprintln!("  ▶ TOOL  {name}  {}", args.to_string().chars().take(300).collect::<String>());
                    let result = exec_tool(&name, &args);
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(600).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { role: Role::Assistant, content: strip_think(&with_close) });
                    messages.push(ChatMessage {
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
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
        assert!(used_multi_edit, "agent should have used multi_edit for the multi-site rename");
        assert!(src.contains("compute_total") && !src.contains("calc_total"), "rename incomplete:\n{src}");
        assert!(out.contains("TOTAL: 33.00") && out.contains("AUDIT: 30.00"), "behaviour changed: {out}");
    }

    /// Prompt documenting the web tools, mirroring src/lib/agentLoop.ts.
    const SYS_WEB: &str = r#"你是 Chaty 的编程智能体,在工作区目录中完成任务。工作区根目录:{WS}

可用工具(路径相对工作区,越界会被拒绝):
- read_file: {"path": string}
- write_file: {"path": string, "content": string}
- list_dir: {"path"?: string}
- web_search: 联网搜索。加 site 参数做站内搜索:site="github.com" 返回仓库/issue/代码匹配;site="reddit.com" 搜帖子;site="youtube.com"/"bilibili.com" 返回视频;其他域名限定站内。args: {"query": string, "site"?: string}
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
            ChatMessage { role: Role::System, content: SYS_WEB.replace("{WS}", &ws.to_string_lossy()) },
            ChatMessage {
                role: Role::User,
                content: "在 YouTube 上搜索 \"me at the zoo\",找到 YouTube 历史上的第一条视频,用 web_fetch 获取它的字幕转写,然后把视频中拍摄者实际谈论的动物和他说的重点写进 NOTES.md(必须依据字幕内容,不要凭标题猜)。".into(),
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
            run_turn(&model, &mut ctx, &mut cached, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} · RAW ────────\n{}", raw.trim().chars().take(400).collect::<String>());
            match parse_tool_call(&raw) {
                Some((name, args)) => {
                    eprintln!("  ▶ TOOL  {name}  {}", args.to_string().chars().take(200).collect::<String>());
                    let result = exec_tool(&name, &args);
                    eprintln!("  ◀ RESULT\n{}", result.chars().take(500).collect::<String>());
                    let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
                    messages.push(ChatMessage { role: Role::Assistant, content: strip_think(&with_close) });
                    messages.push(ChatMessage {
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
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
            ChatMessage { role: Role::System, content: SYS_WEB.replace("{WS}", &ws.to_string_lossy()) },
            ChatMessage {
                role: Role::User,
                content: "帮我调研一个小众 Rust 库:在 GitHub 上搜索 \"dom_smoothie readability\",找到那个把 Mozilla Readability 移植到 Rust 的仓库;用 web_fetch 打开它的仓库页面,把仓库全名和一句话简介写入 RESEARCH.md;最后用 web_download 把页面上列出的任意一张图片保存为 logo.png。全部完成后总结。".into(),
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
            run_turn(&model, &mut ctx, &mut cached, n_ctx, &req, &sink, &cancel).expect("run_turn");
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
                    messages.push(ChatMessage { role: Role::Assistant, content: strip_think(&with_close) });
                    messages.push(ChatMessage {
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
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
            ChatMessage { role: Role::System, content: SYS.replace("{WS}", &ws.to_string_lossy()) },
            ChatMessage {
                role: Role::User,
                content: "这个项目有一个失败的测试。请运行 `python3 test_calc.py`,找出失败原因并修复代码,直到测试全部通过(输出 ALL TESTS PASSED)。".into(),
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
            run_turn(&model, &mut ctx, &mut cached, n_ctx, &req, &sink, &cancel).expect("run_turn");
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
                    messages.push(ChatMessage { role: Role::Assistant, content: strip_think(&with_close) });
                    messages.push(ChatMessage {
                        role: Role::User,
                        content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
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
            ChatMessage { role: Role::System, content: SYS_META.replace("{WS}", &ws.to_string_lossy()) },
            ChatMessage {
                role: Role::User,
                content: "请在工作区创建一个 Python 模块 greet.py,实现 greet(name) 函数,再写 test_greet.py 并用 bash 运行确认通过。开始前先用 update_plan 列出步骤。问候语的语言(中文还是英文)由我决定,请用 ask_user 问我。".into(),
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
            run_turn(&model, &mut ctx, &mut cached, n_ctx, &req, &sink, &cancel).expect("run_turn");
            let raw = sink.buf.into_inner();
            eprintln!("\n──────── STEP {step} ────────\n{}", raw.trim());

            let Some((name, args)) = parse_tool_call(&raw) else {
                eprintln!("  ✔ FINAL\n{}", strip_think(&raw));
                finished = true;
                break;
            };
            let with_close = if raw.contains("</tool_call>") { raw.clone() } else { format!("{raw}</tool_call>") };
            messages.push(ChatMessage { role: Role::Assistant, content: strip_think(&with_close) });

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
            messages.push(ChatMessage {
                role: Role::User,
                content: format!("<tool_result name=\"{name}\">\n{result}\n</tool_result>"),
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
