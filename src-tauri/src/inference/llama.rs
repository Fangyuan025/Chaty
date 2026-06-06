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
    tx: Sender<Job>,
}

impl LlamaEngine {
    /// Load a GGUF file and spin up its worker thread. Blocking; run off-thread.
    ///
    /// `gpu_pref`: `None`/negative = auto‑tune by VRAM, `Some(0)` = force CPU,
    /// `Some(n>0)` = offload exactly `n` layers.
    pub fn load(path: &str, gpu_pref: Option<i32>) -> Result<(Self, ModelInfo)> {
        let backend = llama_backend()?;
        if !Path::new(path).exists() {
            bail!("model file not found: {path}");
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

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        // Load the weights AND allocate the inference context, backing off
        // `n_gpu_layers` on any out-of-memory failure. This covers BOTH the
        // weights and the KV-cache/compute buffers — the latter often OOMs a
        // small GPU even when the weights fit. If even a pure-CPU load runs out
        // of memory, return a clear error instead of a cryptic crash.
        let mut layers = requested.max(0);
        let mut oom_fallback = false;
        let (model, tx) = loop {
            let params = LlamaModelParams::default().with_n_gpu_layers(layers.max(0) as u32);
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
                        bail!("out of memory while loading the model weights");
                    }
                    return Err(e).with_context(|| format!("failed to load GGUF model: {path}"));
                }
            };

            // Create the context in the worker and wait for the result, so a
            // KV/compute-buffer OOM is caught here and folded into the back-off.
            let n_ctx = model.n_ctx_train().clamp(512, 8192);
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
            let worker_model = model.clone();
            std::thread::Builder::new()
                .name("chaty-llama".into())
                .spawn(move || worker(worker_model, n_ctx, n_threads, rx, init_tx))
                .context("failed to start inference thread")?;

            match init_rx.recv() {
                Ok(Ok(())) => break (model, tx),
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
        // If we had to drop below the requested offload to fit memory, flag it.
        let warning = oom_fallback.then(|| "gpu-oom".to_string());

        let n_ctx_train = model.n_ctx_train();
        let n_ctx = n_ctx_train.clamp(512, 8192);
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
        let supports_thinking = template_lc.contains("think")
            || template_lc.contains("reasoning")
            || ["qwen3", "qwq", "deepseek-r1", "-r1", "reasoning", "thinking", "magistral", "cogito"]
                .iter()
                .any(|k| name_lc.contains(k) || arch_lc.contains(k));
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
            supports_tools,
            multimodal,
            warning,
        };

        // `tx` + the worker came from the load/back-off loop above.
        Ok((Self { tx }, info))
    }
}

#[async_trait]
impl InferenceBackend for LlamaEngine {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    async fn generate(&self, req: GenRequest, sink: Channel<StreamEvent>, cancel: Arc<AtomicBool>) -> Result<()> {
        let (done, done_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Job::Generate { req, sink, cancel, done })
            .map_err(|_| anyhow::anyhow!("推理线程已退出"))?;
        match done_rx.await {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!("推理线程在生成过程中断开")),
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

    let prompt = build_prompt(model, &req.messages)?;
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .context("tokenization failed")?;
    let n_prompt = tokens.len();

    if n_prompt + 4 >= n_ctx as usize {
        ctx.clear_kv_cache();
        cached.clear();
        bail!("提示词 {n_prompt} tokens 超出上下文窗口 {n_ctx}，请新建对话或缩短输入。");
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
        return done_event(sink, n_prompt as u32, 0, 0.0);
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

    loop {
        if cancel.load(Ordering::Relaxed) {
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
            if !text.is_empty() {
                sink.send(StreamEvent::Token { text })?;
            }
        }
        n_decoded += 1;
        if n_decoded >= req.params.max_tokens || n_past + 1 >= n_ctx as i32 {
            break;
        }
        batch.clear();
        batch.add(token, n_past, &[0], true)?;
        cached.push(token);
        n_past += 1;
        ctx.decode(&mut batch).context("decode failed")?;
        idx = batch.n_tokens() - 1;
    }
    // Flush any trailing bytes left in the buffer at end of generation.
    if !pending.is_empty() {
        let text = String::from_utf8_lossy(&pending).into_owned();
        if !text.is_empty() {
            let _ = sink.send(StreamEvent::Token { text });
        }
    }

    let secs = start.elapsed().as_secs_f32().max(1e-3);
    done_event(sink, n_prompt as u32, n_decoded, n_decoded as f32 / secs)
}

fn done_event(sink: &Channel<StreamEvent>, prompt_tokens: u32, completion_tokens: u32, tps: f32) -> Result<()> {
    sink.send(StreamEvent::Done {
        stats: GenStats {
            prompt_tokens,
            completion_tokens,
            tokens_per_second: tps,
        },
    })?;
    Ok(())
}

/// Render messages into a prompt using the model's embedded chat template,
/// falling back to ChatML if the GGUF doesn't carry one.
fn build_prompt(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String> {
    let template = model
        .chat_template(None)
        .or_else(|_| LlamaChatTemplate::new("chatml"))
        .context("no usable chat template")?;

    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| LlamaChatMessage::new(role_str(&m.role).to_string(), m.content.clone()))
        .collect::<std::result::Result<_, _>>()
        .context("invalid message content")?;

    model
        .apply_chat_template(&template, &chat, true)
        .context("failed to apply chat template")
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
    if params.temperature <= 0.0 {
        LlamaSampler::chain_simple([LlamaSampler::greedy()])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::dist(seed),
        ])
    }
}
