//! A zero-dependency fake engine.
//!
//! It echoes the last user message and streams a canned paragraph token by
//! token so the whole IPC + UI pipeline can be exercised without a model.
//! Retained as a test double / offline smoke-test backend.
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tauri::ipc::Channel;

use super::{GenRequest, GenStats, InferenceBackend, Role, StreamEvent};

pub struct MockBackend {
    model_name: String,
}

impl MockBackend {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }
}

#[async_trait]
impl InferenceBackend for MockBackend {
    fn name(&self) -> &str {
        "mock"
    }

    async fn generate(
        &self,
        req: GenRequest,
        sink: Channel<StreamEvent>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        sink.send(StreamEvent::Started)?;

        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::User))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let reply = format!(
            "**[mock · {}]** 收到你的消息：「{}」。\n\n\
             这是占位引擎在按 token 流式输出，用来验证 Rust ↔ WebView 的 `Channel` 管线。\
             下一步会把真正的 **llama.cpp** 引擎接到同一个 `InferenceBackend` trait 后面，\
             届时这段文字会变成本地 GGUF 模型的真实推理结果。",
            self.model_name, last_user
        );

        let start = Instant::now();
        let mut completion_tokens = 0u32;
        for chunk in pseudo_tokens(&reply) {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(18)).await;
            sink.send(StreamEvent::Token { text: chunk })?;
            completion_tokens += 1;
        }

        let secs = start.elapsed().as_secs_f32().max(1e-3);
        sink.send(StreamEvent::Done {
            stats: GenStats {
                prompt_tokens: 0,
                completion_tokens,
                tokens_per_second: completion_tokens as f32 / secs,
            },
        })?;
        Ok(())
    }
}

/// Split text into small chunks that *look* like a token stream: latin words
/// (with trailing whitespace) stay together, CJK characters stream one by one.
fn pseudo_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if ch.is_whitespace() || !ch.is_ascii() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
