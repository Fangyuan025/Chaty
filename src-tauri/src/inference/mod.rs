//! Inference abstraction layer.
//!
//! Every engine (mock, llama.cpp, candle/mistral.rs, remote …) implements
//! [`InferenceBackend`]. The rest of the app only ever talks to this trait, so
//! swapping or adding engines never touches the command/UI layer.

pub mod jinja;
pub mod llama;
pub mod llama_log;
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
    /// A side generation — a chat title, a search-query rewrite — that must
    /// leave the conversation's cache alone. Run on the conversation's own
    /// cache it replaced everything the conversation had built, so the turn
    /// after it re-read the whole of it: 0% reused on every turn of a chat with
    /// web search on (measured on Qwen3.6 35B).
    pub scratch: bool,
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
            scratch: false,
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
    /// The tool-call format the chat template teaches its model, read from the
    /// template at load (see `native_tool_format`): "xml", "json", "gemma" or
    /// "lfm". None ⇒ the template names none; the agent uses the fallback the
    /// user picked.
    #[serde(default)]
    pub tool_format: Option<String>,
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

/// Which tool-call format a chat template teaches its model. The template is
/// how the model was trained to write a call; a guess from its name is not.
/// Most specific first — a Qwen3.5 template mentions `<tool_call>` too.
///  - "gemma": Gemma 4, `<|tool_call>call:name{key:<|"|>text<|"|>}<tool_call|>`
///  - "lfm":   LFM2, `<|tool_call_start|>[name(key='v')]<|tool_call_end|>`
///  - "xml":   Qwen3.5/3.6/3.8, `<tool_call>\n<function=name>\n<parameter=key>\n…`
///  - "json":  Hermes, `<tool_call>{"name": …, "arguments": {…}}</tool_call>`
///    (Qwen2.5/Qwen3, QwQ)
pub fn native_tool_format(template: &str) -> Option<&'static str> {
    // K2 Horizon: `<ifm|tool_call>name` + `<ifm|arg_key>`/`<ifm|arg_value>`
    // pairs, all special tokens of its own.
    if template.contains("<ifm|tool_call>") {
        Some("ifm")
    } else if template.contains("<arg_key>") {
        // GLM-4.5/4.6/4.7: the same pairs without a namespace, after a bare
        // tool name in `<tool_call>` — which JSON would otherwise claim.
        Some("glm")
    } else if template.contains("<param name=") {
        // MiniCPM5: `<function name="…"><param name="…">…</param></function>`.
        Some("minicpm")
    } else if template.contains("<|tool_call>") {
        Some("gemma")
    } else if template.contains("<|tool_call_start|>") {
        Some("lfm")
    } else if template.contains("<function=") && template.contains("<parameter=") {
        Some("xml")
    } else if template.contains("<tool_call>") {
        Some("json")
    } else {
        None
    }
}

#[cfg(test)]
mod tool_format_tests {
    use super::native_tool_format;

    /// Fragments of the real templates each family ships.
    #[test]
    fn each_family_template_names_its_own_format() {
        let qwen35 = "{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{{- '<parameter=' + args_name + '>\\n' }}";
        let qwen3 = "{{- '<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}";
        let gemma4 = "{{- '<|tool_call>call:' + function['name'] + '{' -}}";
        let lfm = "<|tool_call_start|>[{{ tool_call.name }}(...)]<|tool_call_end|>";
        assert_eq!(native_tool_format(qwen35), Some("xml"));
        assert_eq!(native_tool_format(qwen3), Some("json"));
        assert_eq!(native_tool_format(gemma4), Some("gemma"));
        assert_eq!(native_tool_format(lfm), Some("lfm"));
        let k2 = "{{- \"\\n<ifm|tool_call>\" + tool_call.name + \"\\n\" }}{{- \"<ifm|arg_key>\" + key + \"</ifm|arg_key>\\n\" }}";
        assert_eq!(native_tool_format(k2), Some("ifm"));
        // GLM-4.7-Flash, verbatim.
        let glm = "{{- '<tool_call>' + tc.name -}}\n{% set _args = tc.arguments %}{% for k, v in _args.items() %}<arg_key>{{ k }}</arg_key><arg_value>{{ v | tojson(ensure_ascii=False) if v is not string else v }}</arg_value>{% endfor %}</tool_call>";
        assert_eq!(native_tool_format(glm), Some("glm"));
        let minicpm5 = "{{- '<function name=\"' ~ tool_call.name ~ '\">' }}{{- '<param name=\"' ~ param_name ~ '\">' }}";
        assert_eq!(native_tool_format(minicpm5), Some("minicpm"));
        assert_eq!(native_tool_format("{% for m in messages %}{{ m.content }}{% endfor %}"), None);
    }

    /// An MLX folder's template: chat_template.jinja, or the tokenizer config's
    /// chat_template — a string, or a list of named ones (read together).
    #[test]
    fn an_mlx_folder_template_is_found_in_either_place() {
        use super::mlx::mlx_chat_template;
        let dir = std::env::temp_dir().join(format!("chaty-mlx-template-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(mlx_chat_template(&dir), None);
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":[{"name":"default","template":"plain"},{"name":"tool_use","template":"<tool_call>{{ x | tojson }}</tool_call>"}]}"#,
        )
        .unwrap();
        assert_eq!(mlx_chat_template(&dir).as_deref().and_then(native_tool_format), Some("json"));
        std::fs::write(dir.join("chat_template.jinja"), "<|tool_call>call:{{ name }}").unwrap();
        assert_eq!(mlx_chat_template(&dir).as_deref().and_then(native_tool_format), Some("gemma"));
        std::fs::remove_dir_all(&dir).ok();
    }
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
