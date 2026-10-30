//! The knowledge base's embedder: bge-m3 through llama.cpp, on a worker
//! thread with a persistent embeddings context. It runs inside the chaty-llama
//! sidecar (`embed_load` / `embed`), like the chat model; rag.rs talks to it
//! through `llama_sidecar::EmbedClient`.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::mpsc::Sender;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

enum EmbedJob {
    Embed {
        texts: Vec<String>,
        reply: Sender<Result<Vec<Vec<f32>>, String>>,
    },
}

/// A loaded embedding model. Dropping it closes the worker's channel and joins
/// it, so the memory is back by the time the drop returns.
pub struct LocalEmbedder {
    tx: Option<Sender<EmbedJob>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Drop for LocalEmbedder {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// How many of the embedder's layers may go on the GPU.
///
/// This load happens with a chat model ALREADY resident, which is the one case
/// where "offload everything" is certain to be wrong: a card with room for a 9B
/// is not a card with room for a 9B and 730 MB more. And an over-ask is not an
/// allocation error to back off from — the Vulkan driver takes the process down
/// with it (issue #9), which is how an import could end with the window
/// vanishing and nothing at all in the log (issue #13). Sizing is therefore the
/// chat model's: against what is FREE, under any cap a previous crash left.
///
/// The same rule on every platform, Apple Silicon included: `fit_layers`
/// against what is free right now. On a Mac "free" has to satisfy both Metal's
/// working set and the machine's memory, because the GPU and CPU share one pool
/// and an MLX chat model lives in a different process from this one.
pub fn embedder_gpu_layers(backend: &llama_cpp_2::llama_backend::LlamaBackend, path: &str) -> i32 {
    let want = crate::gpu::detect_gpu().and_then(|g| {
        let file = std::fs::metadata(path).map(|m| m.len()).ok()?;
        let n_layer = super::llama::probe_n_layer(backend, path)?;
        Some(crate::gpu::fit_layers(file, n_layer, crate::gpu::embed_free_bytes(g.vram_mb)))
    });
    embedder_layers(want, super::llama::gpu_layer_cap())
}

/// The most tokens one chunk may hand the embedder.
///
/// bge-m3 is an ENCODER, and llama.cpp cannot split one sequence across
/// micro-batches for an encoder — so a chunk longer than the context's
/// micro-batch is not an error it returns. It is `GGML_ASSERT(n_ubatch >=
/// n_tokens)` and `abort()`: the whole app gone, nothing raised, nothing in the
/// log. That was issue #13. The micro-batch defaulted to 512 while chunks were
/// cut at 1020, and 800 characters of Chinese run far past 512 tokens — on the
/// reported book 283 of its 377 chunks did (median 652, largest 882), so the
/// first batch killed the app every time, while English, at roughly 200 tokens
/// a chunk, never came near the line. Hence whichever is smaller: the window
/// (less room for special tokens) or the micro-batch the context really got.
fn embed_token_limit(n_ctx: u32, n_ubatch: u32) -> usize {
    (n_ctx as usize).saturating_sub(4).min(n_ubatch as usize).max(1)
}

/// The decision itself, separated from the machinery needed to ask the
/// questions: what VRAM allows, and what a previous crash allows.
///
/// `None` means the question could not be answered — no GPU, or a file whose
/// layer count would not read. Neither is a reason to guess upward with a chat
/// model already on the card; the CPU is slower and survives.
fn embedder_layers(want: Option<i32>, cap: Option<i32>) -> i32 {
    let want = want.unwrap_or(0);
    cap.map_or(want, |c| want.min(c))
}

impl LocalEmbedder {
    /// Start the embedding worker. `layers` forces an offload (tests compare CPU
    /// against GPU); `None` lets `embedder_gpu_layers` decide, as the app does.
    pub fn start(model_path: &Path, layers: Option<i32>) -> Result<Self, String> {
        let path = model_path.to_string_lossy().to_string();
        let (tx, rx) = std::sync::mpsc::channel::<EmbedJob>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        let worker = std::thread::Builder::new()
            .name("chaty-embed".into())
            .spawn(move || {
                let backend = match super::llama::llama_backend_pub() {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = init_tx.send(Err(format!("{e:#}")));
                        return;
                    }
                };
                // On the GPU everywhere, sized by the rule the chat model uses rather
                // than by asking for all of it — it loads with a chat model already
                // resident, the one case where "everything" is certainly wrong.
                let layers = layers.unwrap_or_else(|| embedder_gpu_layers(backend, &path));
                let params = LlamaModelParams::default().with_n_gpu_layers(layers.max(0) as u32);
                // macOS loads through malloc, exactly as the chat model does. What
                // kept this embedder on the CPU was an mmap'd load on Metal leaving
                // its pages wired after unload, never returned to the kernel; malloc
                // is freed synchronously, and residency sets are off app-wide
                // (GGML_METAL_NO_RESIDENCY). Measured before this moved: see
                // `rag_embed_gpu_probe`.
                #[cfg(target_os = "macos")]
                let params = params.with_use_mmap(false);
                // A death in here is remembered: the same guard the chat model arms
                // (Windows, where drivers take the process down), so the next
                // attempt asks for less instead of repeating this one.
                let _crash_guard = super::llama::LoadGuard::arm(layers);
                let model = match LlamaModel::load_from_file(backend, &path, &params) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = init_tx.send(Err(trf!("加载嵌入模型失败:{e:#}", "failed to load embedding model: {e:#}")));
                        return;
                    }
                };
                let n_ctx = 1024u32;
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(n_ctx))
                    // A whole chunk in one micro-batch — see `embed_token_limit`.
                    .with_n_batch(n_ctx)
                    .with_n_ubatch(n_ctx)
                    .with_embeddings(true)
                    .with_pooling_type(LlamaPoolingType::Mean)
                    .with_n_threads(crate::gpu::cpu_worker_threads() as i32);
                let mut ctx = match model.new_context(backend, ctx_params) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = init_tx.send(Err(trf!("创建嵌入上下文失败:{e:#}", "failed to create embedding context: {e:#}")));
                        return;
                    }
                };
                let _ = init_tx.send(Ok(()));
                // Read back rather than assumed: whatever the backend actually
                // granted is the number the assertion checks against.
                let limit = embed_token_limit(n_ctx, ctx.n_ubatch());

                while let Ok(job) = rx.recv() {
                    match job {
                        EmbedJob::Embed { texts, reply } => {
                            let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
                            let mut failed: Option<String> = None;
                            for text in &texts {
                                let tokens = super::llama::tokenize(&model, text, true);
                                let take = tokens.len().min(limit);
                                let mut batch = LlamaBatch::new(take.max(1), 1);
                                let mut add_err = None;
                                for (i, tok) in tokens[..take].iter().enumerate() {
                                    if let Err(e) = batch.add(*tok, i as i32, &[0], true) {
                                        add_err = Some(e.to_string());
                                        break;
                                    }
                                }
                                if let Some(e) = add_err {
                                    failed = Some(e);
                                    break;
                                }
                                ctx.clear_kv_cache();
                                if let Err(e) = ctx.decode(&mut batch) {
                                    failed = Some(trf!("嵌入解码失败:{e}", "embed decode failed: {e}"));
                                    break;
                                }
                                match ctx.embeddings_seq_ith(0) {
                                    Ok(v) => {
                                        let mut v = v.to_vec();
                                        l2_normalize(&mut v);
                                        out.push(v);
                                    }
                                    Err(e) => {
                                        failed = Some(format!("embeddings unavailable: {e}"));
                                        break;
                                    }
                                }
                            }
                            let _ = reply.send(match failed {
                                Some(e) => Err(e),
                                None => Ok(out),
                            });
                        }
                    }
                }
            })
            .map_err(|e| e.to_string())?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self { tx: Some(tx), worker: Some(worker) }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(trf!("嵌入线程启动失败", "embedder thread failed to start")),
        }
    }

    /// Embed a batch of texts: one L2-normalized vector each.
    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.tx
            .as_ref()
            .ok_or_else(|| trf!("嵌入线程已退出", "embedder exited"))?
            .send(EmbedJob::Embed { texts, reply: reply_tx })
            .map_err(|_| trf!("嵌入线程已退出", "embedder exited"))?;
        reply_rx.recv().map_err(|_| trf!("嵌入线程已退出", "embedder exited"))?
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}


#[cfg(test)]
mod tests {
    /// A chunk may never exceed the micro-batch the encoder context was given:
    /// past it llama.cpp aborts the process instead of returning an error.
    #[test]
    fn a_chunk_never_exceeds_the_encoders_micro_batch() {
        use super::embed_token_limit;
        assert_eq!(embed_token_limit(1024, 1024), 1020, "window minus special tokens");
        assert_eq!(
            embed_token_limit(1024, 512),
            512,
            "a smaller micro-batch binds — 512 was the default, and the crash"
        );
        assert!(embed_token_limit(1024, 512) < 606, "the reported book's first batch would now fit");
        assert_eq!(embed_token_limit(0, 512), 1, "never zero tokens");
    }

    /// The embedder shares a card with a chat model that is already on it, so
    /// "offload everything" is the one answer that is certainly wrong — and an
    /// over-ask is not an error to retry, it is a Vulkan driver taking the
    /// process down (issue #13 arrived as an import that closed the window and
    /// logged nothing). What VRAM allows and what a previous crash allows both
    /// bind; neither is optional.
    #[test]
    fn the_embedder_never_asks_for_more_gpu_than_it_is_allowed() {
        use super::embedder_layers;
        assert_eq!(embedder_layers(Some(20), None), 20, "no crash, no cap: what fits");
        assert_eq!(embedder_layers(Some(20), Some(4)), 4, "a cap from a crash binds");
        assert_eq!(
            embedder_layers(Some(3), Some(9)),
            3,
            "a cap wider than the ask leaves it alone"
        );
        assert_eq!(embedder_layers(Some(20), Some(0)), 0, "CPU-only after a crash");
        assert_eq!(embedder_layers(None, None), 0, "no GPU, or an unreadable file: CPU");
        assert_eq!(
            embedder_layers(None, Some(40)),
            0,
            "an unanswered question is never widened by a generous cap"
        );
    }
}
