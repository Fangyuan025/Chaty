// Typed bridge to the Rust backend. Keep this the single source of truth for
// the IPC contract so the UI never touches `invoke` string names directly.
import { invoke, Channel } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
  /** Image attachment paths — only honoured by vision-ready models (mmproj loaded). */
  images?: string[];
}

export interface GenParams {
  temperature: number;
  topP: number;
  maxTokens: number;
  seed?: number | null;
  topK?: number;
  minP?: number;
  repeatPenalty?: number;
  stop?: string[];
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
  /** The vision encoder (mmproj) is loaded — images actually work this session. */
  visionReady: boolean;
  /** Path of the paired mmproj GGUF, when one was found. */
  mmproj?: string | null;
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
  /** Why generation ended: "eos" | "length" | "context" | "stop" | "cancelled". */
  stopReason: string;
}

// ---------- Local RAG (knowledge base) ----------

export interface RagStatus {
  modelReady: boolean;
  docs: number;
  chunks: number;
}

export interface RagDoc {
  id: number;
  name: string;
  chunks: number;
  /** Whether this document participates in retrieval (custom query scope). */
  enabled: boolean;
}

export interface RagHit {
  docName: string;
  seq: number;
  text: string;
  score: number;
}

export interface RagProgress {
  /** "extract" | "embed" | "done" */
  phase: string;
  frac: number;
}

export async function ragStatus(): Promise<RagStatus> {
  return invoke<RagStatus>("rag_status");
}

export async function ragListDocuments(): Promise<RagDoc[]> {
  return invoke<RagDoc[]>("rag_list_documents");
}

export async function ragRemoveDocument(id: number): Promise<void> {
  await invoke("rag_remove_document", { id });
}

/** Empty the whole knowledge base (all documents + chunks). */
export async function ragClearAll(): Promise<void> {
  await invoke("rag_clear_all");
}

export async function ragSetDocEnabled(id: number, enabled: boolean): Promise<void> {
  await invoke("rag_set_doc_enabled", { id, enabled });
}

/** Concatenated text from the enabled KB documents (for podcast transcript). */
export async function ragCorpus(maxChars?: number): Promise<string> {
  return invoke<string>("rag_corpus", { maxChars });
}

export interface RagDocText {
  name: string;
  text: string;
}

/** Per-document text from the enabled KB (fair per-file budget) — for grounding
 *  an overview report with one citation per file. */
export async function ragCorpusDocs(maxChars?: number): Promise<RagDocText[]> {
  return invoke<RagDocText[]>("rag_corpus_docs", { maxChars });
}

export async function ragSearch(query: string, k?: number): Promise<RagHit[]> {
  return invoke<RagHit[]>("rag_search", { query, k });
}

export async function ragAddDocument(
  path: string,
  onProgress: (p: RagProgress) => void,
  /** When ingesting from a folder, the selected folder path — the document is
   *  then named by its path relative to that folder (preserving structure). */
  root?: string,
): Promise<void> {
  const channel = new Channel<RagProgress>();
  channel.onmessage = onProgress;
  await invoke("rag_add_document", { path, root: root ?? null, onProgress: channel });
}

/** Recursively list ingestable files under a folder (for folder import). */
export async function ragListSupportedFiles(dir: string): Promise<string[]> {
  return invoke<string[]>("rag_list_supported_files", { dir });
}

// ---------- Agentic coding (Code mode) ----------
// All file/bash ops are confined to the workspace on the Rust side; paths that
// escape it are rejected. Bash is sandboxed (seatbelt) on macOS.

export interface AgentDirEntry {
  name: string;
  isDir: boolean;
  size: number;
}

export interface AgentBashResult {
  stdout: string;
  stderr: string;
  code: number;
  timedOut: boolean;
}

export async function agentSetWorkspace(path: string): Promise<string> {
  return invoke<string>("agent_set_workspace", { path });
}
export async function agentGetWorkspace(): Promise<string | null> {
  return invoke<string | null>("agent_get_workspace");
}
/** Session directory grants: access beyond the workspace, user-approved. */
export async function agentGrantDir(path: string): Promise<string> {
  return invoke<string>("agent_grant_dir", { path });
}
export async function agentRevokeDir(path: string): Promise<void> {
  await invoke("agent_revoke_dir", { path });
}
export async function agentListGrants(): Promise<string[]> {
  return invoke<string[]>("agent_list_grants");
}
export async function agentClearGrants(): Promise<void> {
  await invoke("agent_clear_grants");
}
export async function agentReadFile(
  path: string,
  offset?: number,
  limit?: number,
  maxChars?: number,
  symbol?: string,
): Promise<string> {
  return invoke<string>("agent_read_file", { path, offset, limit, maxChars, symbol });
}

/** Extract text (+ cached embedded images) from a workspace document —
 *  pdf / docx / xlsx / pptx. Scanned PDFs get automatic OCR. */
export async function agentReadDoc(path: string): Promise<string> {
  return await invoke<string>("agent_read_doc", { path });
}

/** Run the minimal set of tests related to the changed files, summarized. */
export async function agentValidateChange(files?: string[]): Promise<string> {
  return await invoke<string>("agent_validate_change", { files: files ?? null });
}
export async function agentWriteFile(path: string, content: string): Promise<string> {
  return invoke<string>("agent_write_file", { path, content });
}
export async function agentEditFile(
  path: string,
  oldString: string,
  newString: string,
  replaceAll?: boolean,
): Promise<string> {
  return invoke<string>("agent_edit_file", { path, oldString, newString, replaceAll });
}
export async function agentListDir(path?: string): Promise<AgentDirEntry[]> {
  return invoke<AgentDirEntry[]>("agent_list_dir", { path });
}
export async function agentGlob(pattern: string): Promise<string[]> {
  return invoke<string[]>("agent_glob", { pattern });
}
/** Filename search for the @-mention picker (substring, capped, skips VCS/build dirs). */
export async function agentListFiles(query?: string, limit?: number): Promise<string[]> {
  return invoke<string[]>("agent_list_files", { query, limit });
}

/** Fused, file-ranked code search digest (BM25 + filename + exact-phrase,
 *  with matching definition lines) — ready to show the model as-is. */
export async function agentSearchCode(query: string, k?: number): Promise<string> {
  return invoke<string>("agent_search_code", { query, k });
}

// ---------- Background commands (Code mode) ----------

export interface AgentBgInfo {
  id: number;
  command: string;
  running: boolean;
  code?: number | null;
  elapsedSecs: number;
  tail: string;
}

/** Start a background command; returns its id immediately. */
export async function agentBashBg(command: string): Promise<number> {
  return invoke<number>("agent_bash_bg", { command });
}
/** Current status + output tail of one background command. */
export async function agentBgOutput(id: number): Promise<AgentBgInfo> {
  return invoke<AgentBgInfo>("agent_bg_output", { id });
}
/** Kill a background command (whole process tree). */
export async function agentBgKill(id: number): Promise<string> {
  return invoke<string>("agent_bg_kill", { id });
}
/** Finished-but-unreported background commands (each returned exactly once). */
export async function agentBgReap(): Promise<AgentBgInfo[]> {
  return invoke<AgentBgInfo[]>("agent_bg_reap");
}
/** All currently running background commands (UI indicator). */
export async function agentBgList(): Promise<AgentBgInfo[]> {
  return invoke<AgentBgInfo[]>("agent_bg_list");
}

// ---------- Checkpoints (Code mode rewind) ----------

/** Open a checkpoint for the coming turn; agent writes/edits journal into it. */
export async function agentCheckpointBegin(): Promise<number> {
  return invoke<number>("agent_checkpoint_begin");
}
/** Restore the workspace to the state before checkpoint `id` (reverts newer turns too). */
export async function agentCheckpointRevertTo(id: number): Promise<string> {
  return invoke<string>("agent_checkpoint_revert_to", { id });
}
/** Literal keyword search across file names AND contents (no regex). */
export async function agentSearchFiles(query: string, path?: string, namesOnly?: boolean): Promise<string> {
  return invoke<string>("agent_search_files", { query, path, namesOnly });
}

export async function agentGrep(pattern: string, path?: string, glob?: string): Promise<string> {
  return invoke<string>("agent_grep", { pattern, path, glob });
}
export async function agentBash(
  command: string,
  timeoutSecs?: number,
  /** User's sudo password, when a sudo command was approved — piped to
   *  `sudo -S` on stdin, never placed in argv or logs. */
  sudoPassword?: string,
): Promise<AgentBashResult> {
  return invoke<AgentBashResult>("agent_bash", {
    command,
    timeoutSecs,
    sudoPassword: sudoPassword ?? null,
  });
}

// ---------- Code-mode session persistence ----------

export interface CodeSessionMeta {
  id: string;
  title: string;
  workspace: string | null;
  updatedAt: number;
}

/** Persist a whole coding session as a JSON blob (frontend owns the shape). */
export async function codeSessionSave(
  id: string,
  title: string,
  workspace: string | null,
  data: string,
): Promise<void> {
  await invoke("code_session_save", { id, title, workspace, data });
}
export async function codeSessionList(): Promise<CodeSessionMeta[]> {
  return invoke<CodeSessionMeta[]>("code_session_list");
}
export async function codeSessionLoad(id: string): Promise<string | null> {
  return invoke<string | null>("code_session_load", { id });
}
export async function codeSessionDelete(id: string): Promise<void> {
  await invoke("code_session_delete", { id });
}

export async function ragDownloadModel(
  onProgress: (p: DownloadProgress) => void,
): Promise<void> {
  const channel = new Channel<DownloadProgress>();
  channel.onmessage = onProgress;
  await invoke("rag_download_model", { onProgress: channel });
}

/** Reveal the writable models folder in Finder/Explorer (creates it if needed). */
export async function openModelsDir(): Promise<string> {
  return invoke<string>("open_models_dir");
}

/** Reveal the app data folder (DB, models, indexes) for manual backup. */
export async function openDataDir(): Promise<string> {
  return invoke<string>("open_data_dir");
}

/** Write an HTML doc and open it in the default browser (used to export a Deep
 *  Research report → print to PDF, since WKWebView can't print itself). */
export async function openHtmlReport(html: string, name?: string): Promise<string> {
  return invoke<string>("open_html_report", { html, name });
}

/** Open a URL/file in the default browser. Routed through Rust (fork-free
 *  posix_spawn) instead of the opener plugin, which forks and crashes libmalloc
 *  in this multithreaded WebKit process on macOS. */
export async function openExternal(target: string): Promise<void> {
  return invoke<void>("open_external", { target });
}

/** Native webview page zoom (Settings → UI scale). CSS `zoom` breaks fixed/vw
 *  layout, so the platform zoom does the scaling instead. */
export async function setUiZoom(factor: number): Promise<void> {
  return invoke<void>("set_ui_zoom", { factor });
}

export type StreamEvent =
  | { type: "started" }
  /** Prompt-processing progress (long prefills only): `processed`/`total` prompt tokens. */
  | { type: "prefill"; processed: number; total: number }
  | { type: "token"; text: string }
  | { type: "done"; stats: GenStats }
  | { type: "error"; message: string };

/**
 * Open a native folder picker and return the absolute path. Folders are the
 * one loading entry point — GGUF (main + optional mmproj) and MLX
 * (config.json + safetensors) alike; the backend resolves the format.
 */
export async function pickModelFolder(): Promise<string | null> {
  const selected = await open({ multiple: false, directory: true });
  return typeof selected === "string" ? selected : null;
}

/**
 * Load a GGUF. `gpuLayers`: omit/-1 = auto‑tune by VRAM, 0 = CPU, n = n layers.
 * `nCtx`: omit = memory-friendly default (≤8192), n = desired context window
 * (clamped to the model's trained length).
 */
export interface LoadProgress {
  /** "eject" (freeing the old model) | "weights" (loading) | "ready". */
  phase: string;
  frac: number;
}

export async function loadModel(
  path: string,
  gpuLayers?: number,
  nCtx?: number,
  onProgress?: (p: LoadProgress) => void,
): Promise<ModelInfo> {
  const channel = new Channel<LoadProgress>();
  if (onProgress) channel.onmessage = onProgress;
  return await invoke<ModelInfo>("load_model", { path, gpuLayers, nCtx, onProgress: channel });
}

export async function getModel(): Promise<ModelInfo | null> {
  return await invoke<ModelInfo | null>("get_model");
}

/** Unload the active model and return to the empty state (frees its memory). */
export async function ejectModel(): Promise<void> {
  await invoke("eject_model");
}

export interface ModelEntry {
  name: string;
  path: string;
  sizeMb?: number;
  /** Paired vision encoder (mmproj) path, for the picker's vision badge. */
  mmproj?: string | null;
  /** Weight format: "gguf" (llama.cpp) or "mlx" (Apple-Silicon sidecar folder). */
  format?: "gguf" | "mlx";
  /** Vision-capable once loaded (GGUF: paired mmproj; MLX: built-in tower). */
  vision?: boolean;
}

export interface HfFile {
  name: string;
  size: number;
  url: string;
}

/** List `.gguf` files in a HuggingFace repo (`owner/name` or a full URL). */
export async function listHfGgufs(repo: string): Promise<HfFile[]> {
  return await invoke<HfFile[]>("list_hf_ggufs", { repo });
}

export type DownloadProgress =
  | { type: "progress"; downloaded: number; total: number }
  | { type: "done"; path: string }
  | { type: "error"; message: string };

/** Download a GGUF into the models folder, streaming progress to `onProgress`. */
export async function downloadModel(
  url: string,
  filename: string,
  onProgress: (p: DownloadProgress) => void,
  /** Optional folder under models/ — the vision-model layout (main + mmproj side by side). */
  subdir?: string,
): Promise<void> {
  const channel = new Channel<DownloadProgress>();
  channel.onmessage = onProgress;
  await invoke("download_model", { url, filename, subdir: subdir ?? null, onProgress: channel });
}

/** Sentinel rejection message of a user-cancelled download. */
export const DOWNLOAD_CANCELLED = "DOWNLOAD_CANCELLED";

// ---- HF model store (search / browse / quant-level detail) ----

export interface HfModelHit {
  /** Repo id, `owner/name`. */
  id: string;
  name: string;
  author: string;
  downloads: number;
  likes: number;
  updatedAt: string;
  vision: boolean;
  paramsB?: number | null;
}

/** One downloadable variant — a GGUF quant (maybe multi-part) or the MLX build. */
export interface QuantOption {
  label: string;
  size: number;
  /** Repo-relative paths in download order; empty for MLX (whole-repo unit). */
  files: string[];
}

export interface HfModelDetail {
  id: string;
  format: "gguf" | "mlx";
  vision: boolean;
  paramsB?: number | null;
  arch?: string | null;
  quants: QuantOption[];
  mmproj?: string | null;
  mmprojSize: number;
  readme: string;
  totalRamMb: number;
}

/** Search/browse HF models. Empty query = trending storefront. */
export async function hfSearch(
  query: string,
  format: "gguf" | "mlx",
  sort: "trending" | "downloads" | "likes" | "updated",
): Promise<HfModelHit[]> {
  return await invoke<HfModelHit[]>("hf_search", { query, format, sort, limit: 30 });
}

/** Quant-level detail for one repo (auto-detects MLX when it has no GGUFs). */
export async function hfModelDetail(repo: string, format: "gguf" | "mlx"): Promise<HfModelDetail> {
  return await invoke<HfModelDetail>("hf_model_detail", { repo, format });
}

/** The author's official HF avatar URL (org or user), or null. */
export async function hfAuthorAvatar(author: string): Promise<string | null> {
  return await invoke<string | null>("hf_author_avatar", { author });
}

/** Resolve-URL for a repo file (store downloads reuse the file downloader). */
export function hfResolveUrl(repo: string, path: string): string {
  return `https://huggingface.co/${repo}/resolve/main/${path}?download=true`;
}

export interface MlxRepoInfo {
  /** Suggested model folder name (the repo's name segment). */
  name: string;
  files: number;
  totalSize: number;
}

/** Probe a HuggingFace repo as an MLX folder model (config.json + safetensors). */
export async function listHfMlx(repo: string): Promise<MlxRepoInfo> {
  return await invoke<MlxRepoInfo>("list_hf_mlx", { repo });
}

/**
 * Download a whole MLX repo into `models/<Name>/` with aggregate progress.
 * Cancel with `cancelDownload(info.name)` (the folder name is the key).
 * Resolves to the model folder path.
 */
export async function downloadMlxRepo(
  repo: string,
  onProgress: (p: DownloadProgress) => void,
): Promise<string> {
  const channel = new Channel<DownloadProgress>();
  channel.onmessage = onProgress;
  return await invoke<string>("download_mlx_repo", { repo, onProgress: channel });
}

/** True when an HF GGUF filename is a vision encoder (mmproj), not a chat model. */
export function isMmprojFile(name: string): boolean {
  return /mmproj/i.test(name);
}

/** Best mmproj companion in a repo's GGUF list: F16 > BF16 > Q8 > rest, then smallest. */
export function pickMmproj(files: HfFile[]): HfFile | null {
  const projs = files.filter((f) => isMmprojFile(f.name));
  if (projs.length === 0) return null;
  const score = (n: string) => (/f16/i.test(n) && !/bf16/i.test(n) ? 0 : /bf16/i.test(n) ? 1 : /q8/i.test(n) ? 2 : 3);
  return [...projs].sort((a, b) => score(a.name) - score(b.name) || a.size - b.size)[0];
}

/** Folder name for the vision-model layout ("owner/Name-GGUF" → "Name"). */
export function modelFolderFor(repo: string): string {
  const base = repo.split("/").filter(Boolean).pop() ?? repo;
  const clean = base.replace(/-gguf$/i, "").replace(/[/\\:*?"<>|]/g, "_").trim();
  return clean || "model";
}

/** Ask an in-flight `downloadModel(…, filename)` to stop (partial file removed). */
export async function cancelDownload(filename: string): Promise<void> {
  await invoke("cancel_download", { key: filename });
}

/** Ask the in-flight knowledge-base embedding-model download to stop. */
export async function ragCancelDownload(): Promise<void> {
  await invoke("cancel_download", { key: "rag-embed" });
}

/** List `.gguf` models found in the install/app-data `models/` folders. */
export async function listModels(): Promise<ModelEntry[]> {
  return await invoke<ModelEntry[]>("list_models");
}

/** Permanently delete a GGUF file from the models folder (not the active one). */
export async function deleteModelFile(path: string): Promise<void> {
  await invoke("delete_model_file", { path });
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
  pinned: boolean;
}

export interface StoredMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  images?: string[];
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
  images?: string[],
): Promise<void> {
  await invoke("save_message", { id, conversationId, role, content, images: images ?? null });
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
  messages: { id: string; role: Role; content: string; images?: string[] }[],
): Promise<void> {
  await invoke("replace_messages", { conversationId, messages });
}

export async function deleteConversation(id: string): Promise<void> {
  await invoke("delete_conversation", { id });
}

/** Delete every conversation and message (Settings → clear all chats). */
export interface DataStats {
  conversations: number;
  messages: number;
  codeSessions: number;
  dbBytes: number;
}
/** Aggregate counters for the Settings → Data statistics panel. */
export async function dataStats(): Promise<DataStats> {
  return await invoke<DataStats>("data_stats");
}

export async function clearAllConversations(): Promise<void> {
  await invoke("clear_all_conversations");
}

/** Conversation ids whose message bodies match `query` (case-insensitive). */
export async function searchConversations(query: string): Promise<string[]> {
  return await invoke<string[]>("search_conversations", { query });
}

/** Show a save dialog and write `content`; returns true if the user saved. */
export async function exportTextFile(
  defaultName: string,
  content: string,
  ext: "md" | "json",
): Promise<boolean> {
  const path = await save({
    defaultPath: defaultName,
    filters: [{ name: ext === "md" ? "Markdown" : "JSON", extensions: [ext] }],
  });
  if (!path) return false;
  await invoke("write_text_file", { path, content });
  return true;
}

/** Show a save dialog and write an HTML file (used by the Canvas studio). */
export async function exportHtmlFile(defaultName: string, content: string): Promise<boolean> {
  const path = await save({
    defaultPath: defaultName,
    filters: [{ name: "HTML", extensions: ["html"] }],
  });
  if (!path) return false;
  await invoke("write_text_file", { path, content });
  return true;
}

export async function renameConversation(id: string, title: string): Promise<void> {
  await invoke("rename_conversation", { id, title });
}

/** Pin/unpin a conversation (pinned ones sort to the top of the sidebar). */
export async function setConversationPinned(id: string, pinned: boolean): Promise<void> {
  await invoke("set_conversation_pinned", { id, pinned });
}

/** Save base64 f32 mono PCM as a .wav (16-bit) via a native save dialog. */
export async function exportWavFile(
  defaultName: string,
  audioBase64: string,
  sampleRate: number,
): Promise<boolean> {
  const path = await save({
    defaultPath: defaultName,
    filters: [{ name: "WAV audio", extensions: ["wav"] }],
  });
  if (!path) return false;
  await invoke("write_wav_file", { path, audio: audioBase64, sampleRate });
  return true;
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

// ---- extended web tools (coding agent) ----

export interface SiteResult {
  /** repo | issue | code | post | web */
  kind: string;
  title: string;
  url: string;
  snippet: string;
}

export interface LinkRef {
  url: string;
  text: string;
}

export interface PageEx {
  url: string;
  /** markdown | source | text | pdf | binary */
  kind: string;
  contentType: string;
  title: string;
  text: string;
  truncated: boolean;
  links: LinkRef[];
  images: string[];
  bytes?: number | null;
}

/** Structured in-site search: GitHub API + code search, Reddit RSS, or an
 *  engine `site:` query for any other domain. */
export async function siteSearch(site: string, query: string): Promise<SiteResult[]> {
  return await invoke<SiteResult[]>("site_search", { site, query });
}

/** Content-type-aware fetch: markdown for articles, raw source on demand,
 *  text passthrough for code/JSON, PDF extraction, metadata for binaries. */
export async function fetchPageEx(url: string, raw?: boolean): Promise<PageEx> {
  return await invoke<PageEx>("fetch_page_ex", { url, raw });
}

/** Download a URL into the agent workspace (sandboxed + rewind-journaled). */
export async function agentWebDownload(url: string, path: string): Promise<string> {
  return await invoke<string>("agent_web_download", { url, path });
}
/** A background download's live status (progress badge / reap). */
export interface AgentDlInfo {
  id: number;
  url: string;
  path: string;
  downloaded: number;
  total: number | null;
  done: boolean;
  error: string | null;
}
export async function agentDlList(): Promise<AgentDlInfo[]> {
  return invoke<AgentDlInfo[]>("agent_dl_list");
}
export async function agentDlReap(): Promise<AgentDlInfo[]> {
  return invoke<AgentDlInfo[]>("agent_dl_reap");
}

export interface EditOp {
  old_string: string;
  new_string: string;
  replace_all?: boolean;
}

/** Several exact-match edits to one file, applied atomically. */
export async function agentMultiEdit(path: string, edits: EditOp[]): Promise<string> {
  return await invoke<string>("agent_multi_edit", { path, edits });
}

/** Definition lines (functions/classes/…) of a file, with line numbers. */
export async function agentOutline(path: string): Promise<string> {
  return await invoke<string>("agent_outline", { path });
}

/** Resolve a workspace image path to a confined absolute path for `view_image`. */
export async function agentResolveImage(path: string): Promise<string> {
  return await invoke<string>("agent_resolve_image", { path });
}

// ---------- Browser automation (Code mode, CDP-driven Chrome) ----------
export async function browserNavigate(url: string): Promise<string> {
  return await invoke<string>("browser_navigate", { url });
}
/** Full-page screenshot (auto-scrolls to trigger lazy content) → temp PNG path. */
export async function browserScreenshot(): Promise<string> {
  return await invoke<string>("browser_screenshot");
}
/** Snapshot of just the current viewport (immediate) → temp PNG path. */
export async function browserSnapshot(): Promise<string> {
  return await invoke<string>("browser_snapshot");
}
/** Scroll the page (to "bottom"/"top" or by pixels) to trigger lazy loading. */
export async function browserScroll(to?: "bottom" | "top", by?: number): Promise<string> {
  return await invoke<string>("browser_scroll", { to: to ?? null, by: by ?? null });
}
export async function browserEval(expression: string): Promise<string> {
  return await invoke<string>("browser_eval", { expression });
}
export interface ClickStep {
  selector?: string;
  text?: string;
}
export interface TypeStep {
  selector?: string;
  label?: string;
  text: string;
}
export async function browserClick(
  selector?: string,
  text?: string,
  steps?: ClickStep[],
): Promise<string> {
  return await invoke<string>("browser_click", {
    selector: selector ?? null,
    text: text ?? null,
    steps: steps ?? null,
  });
}
export async function browserType(
  selector?: string,
  label?: string,
  text = "",
  steps?: TypeStep[],
): Promise<string> {
  return await invoke<string>("browser_type", {
    selector: selector ?? null,
    label: label ?? null,
    text,
    steps: steps ?? null,
  });
}
export async function browserConsole(): Promise<string> {
  return await invoke<string>("browser_console");
}
export async function browserRead(): Promise<string> {
  return await invoke<string>("browser_read");
}
export async function browserClose(): Promise<string> {
  return await invoke<string>("browser_close");
}
/** Settings → Code: run the agent's browser hidden (applies on next launch). */
export async function browserSetHeadless(on: boolean): Promise<void> {
  await invoke("browser_set_headless", { on });
}
/** Render self-contained HTML in a dedicated headless browser (no window),
 *  returning a screenshot path + the page's console output (Canvas). */
export interface CanvasCapture {
  image: string;
  console: string;
}
export async function browserRenderHtml(html: string): Promise<CanvasCapture> {
  return await invoke<CanvasCapture>("browser_render_html", { html });
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
  /** Original file path — set for vision-image attachments (sent as pixels, not OCR text). */
  path?: string;
  /** Images embedded INSIDE the document (extracted copies) — vision models see them too. */
  images?: string[];
}

export async function pickAttachmentFile(): Promise<string | null> {
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [
      {
        name: "文档 / 图片",
        extensions: [
          "txt", "md", "markdown", "pdf", "docx", "xlsx", "pptx", "csv", "json", "log", "rs", "py",
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

/** Downscaled data-URL thumbnail of a local image (chat-bubble preview). */
export async function imageThumb(path: string, maxDim?: number): Promise<string> {
  return await invoke<string>("image_thumb", { path, maxDim: maxDim ?? null });
}

/** Full-resolution image as a data URL (no downscaling) — crisp preview. */
export async function imageDataUrl(path: string): Promise<string> {
  return await invoke<string>("image_data_url", { path });
}

/** Copy `src` to `dest` (a user-chosen path from the save dialog). */
export async function saveFile(src: string, dest: string): Promise<void> {
  await invoke("save_file", { src, dest });
}

/** Save an image to a user-picked location via the native dialog. Returns the
 *  destination path, or null if the user cancelled. */
export async function saveImageAs(src: string, suggestedName = "screenshot.png"): Promise<string | null> {
  const dest = await save({
    defaultPath: suggestedName,
    filters: [{ name: "PNG", extensions: ["png"] }],
  });
  if (!dest) return null;
  await saveFile(src, dest);
  return dest;
}

/** One-shot vision analysis: run the loaded vision model on image paths + a
 *  prompt, return its full text answer. Throws if no vision model is loaded. */
export async function visionQuery(
  images: string[],
  prompt: string,
  maxTokens?: number,
): Promise<string> {
  return await invoke<string>("vision_query", { images, prompt, maxTokens: maxTokens ?? null });
}

/** File extensions the vision pipeline accepts as direct image input. */
export const VISION_IMAGE_EXTS = ["png", "jpg", "jpeg", "webp", "bmp", "gif"];

/** True when `path` looks like an image a vision model can consume directly. */
export function isVisionImagePath(path: string): boolean {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return VISION_IMAGE_EXTS.includes(ext);
}
