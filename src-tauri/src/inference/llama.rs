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

#[allow(clippy::too_many_arguments)]
fn run_turn(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    cached: &mut Vec<LlamaToken>,
    n_ctx: u32,
    req: &GenRequest,
    sink: &Channel<StreamEvent>,
    cancel: &AtomicBool,
) -> Result<()> {
    sink.send(StreamEvent::Started)?;

    let prompt = build_prompt(model, &req.messages, req.params.think)?;
    // Qwen3.5/3.6-style templates PRE-OPEN the reasoning block: the prompt
    // ends with "<think>\n" and the model starts mid-reasoning, so the UI
    // would never see an opening tag. Emit a synthetic one so the stream is
    // well-formed for the frontend's think-panel parser.
    if req.params.think != Some(false) && prompt.trim_end().ends_with("<think>") {
        sink.send(StreamEvent::Token { text: "<think>\n".to_string() })?;
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
                    sink.send(StreamEvent::Token { text: chunk })?;
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
                sink.send(StreamEvent::Token { text: out[emitted..abs].to_string() })?;
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
                sink.send(StreamEvent::Token { text: out[emitted..safe].to_string() })?;
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
            let _ = sink.send(StreamEvent::Token { text: out[emitted..].to_string() });
        }
        if !pending.is_empty() {
            let text = String::from_utf8_lossy(&pending).into_owned();
            if !text.is_empty() {
                let _ = sink.send(StreamEvent::Token { text });
            }
        }
    }

    let secs = start.elapsed().as_secs_f32().max(1e-3);
    done_event(sink, n_prompt as u32, n_decoded, n_decoded as f32 / secs, stop_reason)
}

fn done_event(
    sink: &Channel<StreamEvent>,
    prompt_tokens: u32,
    completion_tokens: u32,
    tps: f32,
    stop_reason: &str,
) -> Result<()> {
    sink.send(StreamEvent::Done {
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
