//! Inference abstraction layer.
//!
//! Every engine (mock, llama.cpp, candle/mistral.rs, remote …) implements
//! [`InferenceBackend`]. The rest of the app only ever talks to this trait, so
//! swapping or adding engines never touches the command/UI layer.

pub mod llama;
pub use llama::llama_backend_pub;
pub mod mlx;
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
    /// A tool's result, delivered under its own role rather than folded into a
    /// user turn. Chat templates key "is this turn still part of the current
    /// request" off the last *user* message, so a tool result posing as one
    /// makes the template discard the assistant reasoning that preceded it —
    /// costing the model the thread of its own work and voiding the KV prefix
    /// every step. Used only where the model's template actually renders it
    /// (probed at load, never assumed).
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// The turn's thinking, kept out of `content`. Some templates read
    /// reasoning only from a structured field and never split it back out of
    /// the content — a turn stored inline reaches them as an empty thought
    /// followed by its own markup. Sent only where the template is probed to
    /// use it; `None` everywhere else keeps the wire shape unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Image attachments (absolute file paths) for vision models. Ignored —
    /// and expected empty — when the loaded model has no mmproj.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
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
    /// Native reasoning-effort level for models whose chat template takes a
    /// `reasoning_effort` kwarg (Qwen3.8: `low` | `medium` | `xhigh`). `None`
    /// leaves the model's own default; ignored by models without the ladder.
    pub effort: Option<String>,
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
            effort: None,
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

/// The conversation and message a streaming reply belongs to.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveTarget {
    pub conversation_id: String,
    pub message_id: String,
}

/// Streaming protocol pushed to the frontend over a Tauri `Channel`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum StreamEvent {
    /// Generation accepted; prompt is being processed.
    Started,
    /// Prompt-processing progress: `processed` of `total` prompt tokens are in
    /// the KV cache. Only emitted for prefills long enough to be worth a
    /// progress ring (more than one decode batch of new tokens).
    Prefill { processed: u32, total: u32 },
    /// One decoded piece of text (not necessarily a whole token).
    Token { text: String },
    /// Generation finished cleanly.
    Done { stats: GenStats },
    /// Generation aborted with an error.
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub tokens_per_second: f32,
    /// Why generation ended: "eos" (model finished), "length" (max_tokens hit),
    /// "context" (context window full), "stop" (stop sequence), "cancelled".
    pub stop_reason: String,
    /// Prompt tokens resumed from the previous turn's KV cache (0 = evaluated
    /// from scratch). Cross-conversation contamination shows up here first.
    pub reused: u32,
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
    /// Native reasoning-effort ladder the chat template accepts, weakest
    /// first (Qwen3.8: `["low", "medium", "xhigh"]`). Empty ⇒ the model has
    /// no effort control and the UI keeps its plain thinking toggle.
    #[serde(default)]
    pub effort_levels: Vec<String>,
    /// The chat template renders a tool result under its own role AND keeps
    /// the assistant reasoning that preceded it — which a result posing as a
    /// user turn discards. Probed at load; false ⇒ results stay user turns and
    /// the rendered prompt is byte-identical to what earlier builds produced.
    #[serde(default)]
    pub tool_role: bool,
    /// The template reads a turn's thinking from a structured
    /// `reasoning_content` field rather than splitting it out of the content.
    #[serde(default)]
    pub reasoning_field: bool,
    /// Best-effort: the chat template supports tool / function calling.
    pub supports_tools: bool,
    /// Best-effort: the model appears to be multimodal (vision).
    pub multimodal: bool,
    /// The vision encoder (mmproj) is loaded — the model can actually see
    /// images this session. `multimodal && !vision_ready` means "the model
    /// could do vision, but its mmproj GGUF is missing next to the weights".
    pub vision_ready: bool,
    /// Whether ONE prompt may carry several pictures. Gemma-4 through MLX
    /// cannot: it encodes the first and then rejects the token-count mismatch,
    /// failing the whole round — which is what a tall page's tiled screenshot
    /// hands it. Everything else here says true.
    pub multi_image: bool,
    /// Path of the paired mmproj GGUF, when one was found.
    pub mmproj: Option<String>,
    /// The model file carries a multi-token-prediction head — the extra block
    /// speculative decoding guesses with. A capability of the FILE, so it stays
    /// true when the user has the feature switched off; the settings toggle is
    /// enabled on this and greyed out on everything else.
    #[serde(default)]
    pub speculative: bool,
    /// Speculative decoding is actually running for this load: the model has a
    /// head, the setting allows it, and the head loaded. Never true when
    /// `speculative` is false.
    #[serde(default)]
    pub speculative_on: bool,
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

    /// One-shot, non-streaming generation returning the full text. Powers
    /// vision analysis (Code mode / KB / Canvas). Default: unsupported.
    async fn generate_collect(
        &self,
        _req: GenRequest,
        _cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<String> {
        anyhow::bail!("this backend does not support one-shot generation")
    }
}
