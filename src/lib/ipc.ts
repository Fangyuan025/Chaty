// Typed bridge to the Rust backend. Keep this the single source of truth for
// the IPC contract so the UI never touches `invoke` string names directly.
import { invoke, Channel } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface GenParams {
  temperature: number;
  topP: number;
  maxTokens: number;
  seed?: number | null;
  /** Reasoning control for switch-less models (Qwen3.5+): false force-disables
   *  thinking, true/undefined leaves the model default. */
  think?: boolean | null;
}

export interface GenRequest {
  messages: ChatMessage[];
  params?: Partial<GenParams>;
}

export interface ModelInfo {
  name: string;
  path: string;
  backend: string;
  loaded: boolean;
  arch?: string | null;
  sizeMb?: number | null;
  paramsB?: number | null;
  nCtxTrain?: number | null;
  nCtx?: number | null;
  nLayer?: number | null;
  gpuLayers: number;
  gpuName?: string | null;
  modelName?: string | null;
  quant?: string | null;
  nEmbd?: number | null;
  hasChatTemplate: boolean;
  supportsThinking: boolean;
  /** The chat template honours the `/no_think` soft switch (Qwen3, not 3.5+). */
  thinkSwitch: boolean;
  supportsTools: boolean;
  multimodal: boolean;
  /** Non-fatal load warning code (e.g. "gpu-oom"), or null. */
  warning?: string | null;
}

export interface GpuInfo {
  name: string;
  vramMb: number;
}

export interface HardwareInfo {
  cpu: string;
  cpuThreads: number;
  ramMb: number;
  gpu?: GpuInfo | null;
  /** Compiled GPU backend for the LLM ("Vulkan" | "CPU"). */
  gpuBackend: string;
}

export async function getHardwareInfo(): Promise<HardwareInfo> {
  return await invoke<HardwareInfo>("get_hardware_info");
}

export interface GpuUsage {
  usedMb: number;
  totalMb: number;
}

export async function getGpuUsage(): Promise<GpuUsage | null> {
  return await invoke<GpuUsage | null>("get_gpu_usage");
}

export interface UpdateInfo {
  available: boolean;
  current: string;
  latest: string;
  url?: string | null;
  notes?: string | null;
}

/** Check GitHub Releases for a newer version (never throws). */
export async function checkUpdate(): Promise<UpdateInfo> {
  return await invoke<UpdateInfo>("check_update");
}

/** Download the installer and launch it (the app will exit). */
export async function runUpdate(url: string): Promise<void> {
  await invoke("run_update", { url });
}

export interface GenStats {
  promptTokens: number;
  completionTokens: number;
  tokensPerSecond: number;
}

export type StreamEvent =
  | { type: "started" }
  | { type: "token"; text: string }
  | { type: "done"; stats: GenStats }
  | { type: "error"; message: string };

/** Open a native file picker filtered to .gguf and return the absolute path. */
export async function pickModelFile(): Promise<string | null> {
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "GGUF model", extensions: ["gguf"] }],
  });
  return typeof selected === "string" ? selected : null;
}

/**
 * Load a GGUF. `gpuLayers`: omit/-1 = auto‑tune by VRAM, 0 = CPU, n = n layers.
 * `nCtx`: omit = memory-friendly default (≤8192), n = desired context window
 * (clamped to the model's trained length).
 */
export async function loadModel(
  path: string,
  gpuLayers?: number,
  nCtx?: number,
): Promise<ModelInfo> {
  return await invoke<ModelInfo>("load_model", { path, gpuLayers, nCtx });
}

export async function getModel(): Promise<ModelInfo | null> {
  return await invoke<ModelInfo | null>("get_model");
}

export interface ModelEntry {
  name: string;
  path: string;
}

/** List `.gguf` models found in the install/app-data `models/` folders. */
export async function listModels(): Promise<ModelEntry[]> {
  return await invoke<ModelEntry[]>("list_models");
}

/** Re-label the system-tray menu to the given UI language. */
export async function setTrayLanguage(lang: string): Promise<void> {
  await invoke("set_tray_language", { lang });
}

/**
 * Stream a completion. `onEvent` fires for every backend event; the returned
 * promise resolves when generation has fully finished.
 */
export async function generate(
  request: GenRequest,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const channel = new Channel<StreamEvent>();
  channel.onmessage = onEvent;
  await invoke("generate", { request, onEvent: channel });
}

/** Request the in-flight generation to stop early. */
export async function cancelGeneration(): Promise<void> {
  await invoke("cancel_generation");
}

// ---------- Conversation persistence ----------

export interface Conversation {
  id: string;
  title: string;
  modelPath?: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface StoredMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
}

export async function saveConversation(
  id: string,
  title: string,
  modelPath: string | null,
): Promise<void> {
  await invoke("save_conversation", { id, title, modelPath });
}

export async function saveMessage(
  id: string,
  conversationId: string,
  role: Role,
  content: string,
): Promise<void> {
  await invoke("save_message", { id, conversationId, role, content });
}

export async function listConversations(): Promise<Conversation[]> {
  return await invoke<Conversation[]>("list_conversations");
}

export async function getMessages(conversationId: string): Promise<StoredMessage[]> {
  return await invoke<StoredMessage[]>("get_messages", { conversationId });
}

/** Replace all messages of a conversation (for edit / regenerate truncation). */
export async function replaceMessages(
  conversationId: string,
  messages: { id: string; role: Role; content: string }[],
): Promise<void> {
  await invoke("replace_messages", { conversationId, messages });
}

export async function deleteConversation(id: string): Promise<void> {
  await invoke("delete_conversation", { id });
}

export async function renameConversation(id: string, title: string): Promise<void> {
  await invoke("rename_conversation", { id, title });
}

// ---------- Web search ----------

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

export async function webSearch(query: string): Promise<SearchResult[]> {
  return await invoke<SearchResult[]>("web_search", { query });
}

export interface PageContent {
  title: string;
  url: string;
  text: string;
}

export interface WebResearch {
  results: SearchResult[];
  pages: PageContent[];
}

export async function webResearch(query: string): Promise<WebResearch> {
  return await invoke<WebResearch>("web_research", { query });
}

export async function fetchUrl(url: string): Promise<PageContent> {
  return await invoke<PageContent>("fetch_url", { url });
}

// ---------- Voice (STT / TTS) ----------

export interface SynthAudio {
  audio: string; // base64 of little-endian f32 PCM
  sampleRate: number;
}

/** Transcribe base64 f32 PCM audio → text (Whisper). */
export async function transcribe(audio: string, sampleRate: number): Promise<string> {
  return await invoke<string>("transcribe", { audio, sampleRate });
}

/** Synthesize speech for text (Kokoro) → base64 f32 PCM + sample rate. */
export async function synthesize(
  text: string,
  speed?: number,
  sid?: number,
): Promise<SynthAudio> {
  return await invoke<SynthAudio>("synthesize", { text, speed, sid });
}

// ---------- Attachments ----------

export interface Attachment {
  name: string;
  kind: string;
  text: string;
  chars: number;
  truncated: boolean;
}

export async function pickAttachmentFile(): Promise<string | null> {
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [
      {
        name: "文档 / 图片",
        extensions: [
          "txt", "md", "markdown", "pdf", "csv", "json", "log", "rs", "py",
          "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "hpp", "go",
          "rb", "php", "html", "css", "xml", "yaml", "yml", "toml", "ini", "sh",
          "png", "jpg", "jpeg", "webp", "bmp", "gif",
        ],
      },
    ],
  });
  return typeof selected === "string" ? selected : null;
}

export async function readAttachment(path: string): Promise<Attachment> {
  return await invoke<Attachment>("read_attachment", { path });
}
