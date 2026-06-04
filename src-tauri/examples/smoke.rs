//! Headless smoke test for the llama.cpp engine path, including the
//! cross-turn **KV-cache reuse** logic used by the real engine.
//!
//!   cargo run --example smoke -- <model.gguf>
//!
//! Runs a 2-turn conversation against one persistent context and prints, per
//! turn, how many prompt tokens were reused vs. freshly decoded.

use std::io::Write;
use std::num::NonZeroU32;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

fn turn(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    cached: &mut Vec<LlamaToken>,
    n_ctx: u32,
    messages: &[(String, String)],
    max_tokens: u32,
) -> Result<String> {
    let template = model
        .chat_template(None)
        .or_else(|_| LlamaChatTemplate::new("chatml"))?;
    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|(r, c)| LlamaChatMessage::new(r.clone(), c.clone()))
        .collect::<std::result::Result<_, _>>()?;
    let prompt = model.apply_chat_template(&template, &chat, true)?;
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let n_prompt = tokens.len();

    let mut prefix = 0usize;
    let max_match = cached.len().min(n_prompt);
    while prefix < max_match && cached[prefix] == tokens[prefix] {
        prefix += 1;
    }
    if prefix == n_prompt {
        prefix = n_prompt - 1;
    }
    if prefix < cached.len() {
        let _ = ctx.clear_kv_cache_seq(Some(0), Some(prefix as u32), None);
    }
    cached.truncate(prefix);
    eprintln!(
        "[turn] prompt={n_prompt} reused={prefix} decode={}",
        n_prompt - prefix
    );

    let n_batch = ctx.n_batch() as usize;
    let mut batch = LlamaBatch::new(n_batch.max(1), 1);
    let mut pos = prefix;
    while pos < n_prompt {
        let end = (pos + n_batch).min(n_prompt);
        batch.clear();
        for (j, tok) in tokens[pos..end].iter().enumerate() {
            let is_last = pos + j == n_prompt - 1;
            batch.add(*tok, (pos + j) as i32, &[0], is_last)?;
        }
        ctx.decode(&mut batch)?;
        pos = end;
    }
    *cached = tokens;
    let mut idx = batch.n_tokens() - 1;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(42),
    ]);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut out = String::new();
    let mut n_decoded = 0u32;
    let mut n_past = n_prompt as i32;
    loop {
        let token = sampler.sample(ctx, idx);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, false, None)
            .unwrap_or_default();
        print!("{piece}");
        std::io::stdout().flush().ok();
        out.push_str(&piece);
        n_decoded += 1;
        if n_decoded >= max_tokens || n_past + 1 >= n_ctx as i32 {
            break;
        }
        batch.clear();
        batch.add(token, n_past, &[0], true)?;
        cached.push(token);
        n_past += 1;
        ctx.decode(&mut batch)?;
        idx = batch.n_tokens() - 1;
    }
    Ok(out)
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: smoke <model.gguf>")?;

    let mut backend = LlamaBackend::init()?;
    backend.void_logs();
    let model =
        LlamaModel::load_from_file(&backend, &path, &LlamaModelParams::default().with_n_gpu_layers(0))
            .context("load model")?;
    let n_ctx = model.n_ctx_train().clamp(512, 4096);
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);
    let mut ctx = model.new_context(&backend, ctx_params)?;
    let mut cached: Vec<LlamaToken> = Vec::new();

    let mut convo: Vec<(String, String)> = Vec::new();

    eprintln!("\n===== TURN 1 =====");
    convo.push(("user".into(), "用一句话介绍勾股定理。/no_think".into()));
    let r1 = turn(&model, &mut ctx, &mut cached, n_ctx, &convo, 120)?;
    convo.push(("assistant".into(), r1));

    eprintln!("\n\n===== TURN 2 (expect high reuse of turn-1 prefix) =====");
    convo.push((
        "user".into(),
        "用它算直角边 3 和 4 的斜边长度。/no_think".into(),
    ));
    let _r2 = turn(&model, &mut ctx, &mut cached, n_ctx, &convo, 160)?;

    eprintln!("\n\nDONE");
    Ok(())
}
