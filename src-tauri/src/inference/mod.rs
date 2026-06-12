//! Inference abstraction layer.
//!
//! Every engine (mock, llama.cpp, candle/mistral.rs, remote …) implements
//! [`InferenceBackend`]. The rest of the app only ever talks to this trait, so
//! swapping or adding engines never touches the command/UI layer.

pub mod llama;
pub use llama::llama_backend_pub;
pub mod mock;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

/// A single chat turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Sampling / decoding parameters. Sensible defaults so the UI can omit them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub seed: Option<u64>,
    /// Top-k cutoff (0 = disabled).
    pub top_k: u32,
    /// Min-p cutoff (0 = disabled).
    pub min_p: f32,
    /// Repetition penalty (1.0 = off).
    pub repeat_penalty: f32,
    /// Stop sequences — generation halts (and the trailing match is trimmed) when
    /// any of these appears in the output.
    pub stop: Vec<String>,
    /// Reasoning control for models without the `/no_think` soft switch
    /// (Qwen3.5+): `Some(false)` force-disables thinking by pre-filling an empty
    /// `<think></think>` block; `Some(true)`/`None` leave the model default.
    pub think: Option<bool>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 512,
            seed: None,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.1,
            stop: Vec::new(),
            think: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub params: GenParams,
}

/// Streaming protocol pushed to the frontend over a Tauri `Channel`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum StreamEvent {
    /// Generation accepted; prompt is being processed.
    Started,
    /// One decoded piece of text (not necessarily a whole token).
    Token { text: String },
    /// Generation finished cleanly.
    Done { stats: GenStats },
    /// Generation aborted with an error.
    Error { message: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub tokens_per_second: f32,
    /// Why generation ended: "eos" (model finished), "length" (max_tokens hit),
    /// "context" (context window full), "stop" (stop sequence), "cancelled".
    pub stop_reason: String,
}

/// Metadata about the currently loaded model, surfaced to the UI.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub backend: String,
    pub loaded: bool,
    /// Model architecture from GGUF metadata (e.g. "llama", "qwen2").
    pub arch: Option<String>,
    /// On-disk size of the tensors, in MB.
    pub size_mb: Option<u64>,
    /// Parameter count in billions.
    pub params_b: Option<f64>,
    /// Context length the model was trained with.
    pub n_ctx_train: Option<u32>,
    /// Context length we actually loaded it with.
    pub n_ctx: Option<u32>,
    /// Total transformer layers in the model.
    pub n_layer: Option<u32>,
    /// Layers actually offloaded to the GPU (0 = pure CPU).
    pub gpu_layers: i32,
    /// Name of the GPU used for offload, if any.
    pub gpu_name: Option<String>,
    /// Pretty model name from `general.name`, if present.
    pub model_name: Option<String>,
    /// Quantization (e.g. "Q5_K_M"), derived from `general.file_type`.
    pub quant: Option<String>,
    /// Embedding dimension.
    pub n_embd: Option<u32>,
    /// Whether the GGUF ships a chat template.
    pub has_chat_template: bool,
    /// Best-effort: the model appears to support `<think>` reasoning.
    pub supports_thinking: bool,
    /// The chat template honours the `/no_think` soft switch (Qwen3, not 3.5+).
    pub think_switch: bool,
    /// Best-effort: the chat template supports tool / function calling.
    pub supports_tools: bool,
    /// Best-effort: the model appears to be multimodal (vision).
    pub multimodal: bool,
    /// Non-fatal load warning code for the UI (e.g. "gpu-oom" when the GPU
    /// offload had to be reduced to fit memory). `None` on a clean load.
    pub warning: Option<String>,
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Synchronously release the model's memory (block until freed). Called
    /// before loading a replacement so two models never coexist in RAM.
    fn unload(&self) {}
    /// Short identifier for telemetry / UI ("mock", "llama.cpp", …).
    fn name(&self) -> &str;

    /// Stream a completion for `req`, emitting [`StreamEvent`]s on `sink`.
    /// Implementations should send `Started` first and exactly one terminal
    /// `Done` or `Error` last, and stop early when `cancel` becomes `true`.
    async fn generate(
        &self,
        req: GenRequest,
        sink: Channel<StreamEvent>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()>;
}
