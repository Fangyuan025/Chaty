import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { applyCodeTheme } from "./lib/codeTheme";
import { platform } from "@tauri-apps/plugin-os";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { getCurrent as getDeepLinks, onOpenUrl } from "@tauri-apps/plugin-deep-link";
import { AssistantMessage } from "./components/AssistantMessage";
import { ContextMenu } from "./components/ContextMenu";
import { DownloadModal } from "./components/DownloadModal";
import { HardwarePanel } from "./components/HardwarePanel";
import { LiveMode } from "./components/LiveMode";
import { ModelInfoPanel } from "./components/ModelInfoPanel";
import { WindowControls } from "./components/WindowControls";
import { useI18n, type Lang, type TKey } from "./lib/i18n";
import {
  decodeAudio,
  encodeAudio,
  playAudio,
  SpeechQueue,
  startRecording,
  type Recorder,
} from "./lib/audio";
import { SettingsPanel, type GenSettings, defaultSettings, parseStops } from "./components/SettingsPanel";
import { SetupModal } from "./components/SetupModal";
import { KnowledgePanel } from "./components/KnowledgePanel";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { Icon } from "./components/Icon";
import { CanvasPanel, type CanvasVersion } from "./components/CanvasPanel";
import { fixInstruction } from "./lib/canvasSource";
import { CanvasOpenContext, CodeCollapseContext } from "./components/Markdown";
import { useConfirm } from "./components/ConfirmModal";
import { IconPin, IconPinFilled, IconEdit } from "./components/icons";
import { PodcastPanel } from "./components/PodcastPanel";
import { DeepResearchPanel } from "./components/DeepResearchPanel";
import { CodeMode } from "./components/CodeMode";
import { answerOnly, cleanTitle, cutSentences, forSpeech, stripThink } from "./lib/voiceText";
import { copyToClipboard } from "./lib/clipboard";
import {
  cancelGeneration,
  checkUpdate,
  deleteConversation,
  fetchUrl,
  generate,
  getMessages,
  getModel,
  listConversations,
  listModels,
  loadModel,
  ejectModel,
  deleteModelFile,
  openExternal,
  setUiZoom,
  browserSetHeadless,
  openModelsDir,
  setHfEndpoint,
  openDataDir,
  ragSearch,
  ragStatus,
  pickAttachmentFile,
  pickModelFolder,
  readAttachment,
  imageThumb,
  browserRenderHtml,

  isVisionImagePath,
  renameConversation,
  setConversationPinned,
  replaceMessages,
  runUpdate,
  searchConversations,
  exportTextFile,
  exportHtmlFile,
  openHtmlReport,
  saveConversation,
  saveMessage,
  setTrayLanguage,
  synthesize,
  transcribe,
  webResearch,
  type Attachment,
  type ChatMessage,
  type Conversation,
  type GenStats,
  type LoadProgress,
  type ModelEntry,
  type ModelInfo,
  type Role,
  type SearchResult,
  type StreamEvent,
  type UpdateInfo,
} from "./lib/ipc";
import "./App.css";
import { fmtGbFromMb } from "./lib/fmt";

interface UiMessage extends ChatMessage {
  id: string;
  sources?: SearchResult[];
}

/** Module-level thumbnail cache: path → data URL (survives re-renders). */
const thumbCache = new Map<string, string>();

/** Self-loading thumbnail for a local image path; hides itself if unreadable. */
/** User-message text with a clamp for pasted walls of text: over ~15 lines or
 *  1200 chars it renders a 220px preview with a fade + expand pill. */
function UserText({ content, expandLabel, collapseLabel }: { content: string; expandLabel: string; collapseLabel: string }) {
  const long = content.length > 1200 || content.split("\n").length > 15;
  const [open, setOpen] = useState(false);
  if (!long) return <span className="user-text">{content}</span>;
  return (
    <>
      <span className={`user-text ${open ? "" : "clamped"}`}>{content}</span>
      <button className="user-expand" type="button" onClick={() => setOpen(!open)}>
        {open ? collapseLabel : expandLabel}
      </button>
    </>
  );
}

/** Hover copy button on user messages (mirrors the edit pencil). */
function UserCopy({ content, title }: { content: string; title: string }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      className="user-edit user-copy"
      title={title}
      onClick={() =>
        void copyToClipboard(content).then(() => {
          setOk(true);
          setTimeout(() => setOk(false), 1400);
        })
      }
    >
      {ok ? (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
          <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15V5a2 2 0 0 1 2-2h8" strokeLinecap="round" />
        </svg>
      )}
    </button>
  );
}

function ImageThumb({ path, size = 168 }: { path: string; size?: number }) {
  const [src, setSrc] = useState<string | null>(thumbCache.get(path) ?? null);
  useEffect(() => {
    let live = true;
    if (!thumbCache.has(path)) {
      imageThumb(path, 512)
        .then((d) => {
          thumbCache.set(path, d);
          if (live) setSrc(d);
        })
        .catch(() => {
          if (live) setSrc("");
        });
    }
    return () => {
      live = false;
    };
  }, [path]);
  if (src === "") return null; // moved/deleted — degrade quietly
  return (
    <span className="img-thumb" style={{ maxWidth: size, maxHeight: size }}>
      {src ? <img src={src} alt="" /> : <span className="img-thumb-ph" />}
    </span>
  );
}

// Host OS, resolved once at startup. Drives native window chrome on macOS
// (traffic lights instead of our custom min/max/close buttons).
const OS_PLATFORM = (() => {
  try {
    return platform();
  } catch {
    return "windows";
  }
})();
const IS_MACOS = OS_PLATFORM === "macos";

const uid = () => Math.random().toString(36).slice(2);
const fmtK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n));
const SETTINGS_KEY = "chaty.settings";
const LAST_MODEL_KEY = "chaty.lastModel";
/** Boot auto-load must run once per page load, not once per (Strict)mount. */
let bootLoadStarted = false;
const SIDEBAR_DEFAULT = 248;
const SIDEBAR_MIN = 200;
const SIDEBAR_MAX = 440;
const convTitle = (t: string) => t.replace(/\s+/g, " ").trim().slice(0, 40) || "新对话";

/** Parse `chaty://open_from_hf?model=<repo>&file=<file>` from a deep link. */
function parseHfDeepLink(raw: string): { repo: string; file?: string } | null {
  try {
    const u = new URL(raw);
    if (u.protocol !== "chaty:") return null;
    const repo = u.searchParams.get("model");
    if (!repo) return null;
    return { repo, file: u.searchParams.get("file") || undefined };
  } catch {
    return null;
  }
}

const copyText = (t: string) => {
  void copyToClipboard(t);
};

/** Clean a model-generated search query (strip reasoning/quotes, keep it short). */
function cleanQuery(raw: string): string {
  const t = answerOnly(raw);
  const firstLine = t.split("\n").map((s) => s.trim()).find(Boolean) ?? "";
  return firstLine.replace(/^["'「『《]+|["'」』》]+$/g, "").trim().slice(0, 80);
}

const URL_RE = /https?:\/\/[^\s)）】"'<>，。、]+/g;

const SUGGESTIONS_ZH = [
  "用简单的话解释什么是量子纠缠",
  "帮我写一封礼貌专业的请假邮件",
  "用 Python 实现快速排序并讲解思路",
  "给我三个适合周末的短途旅行点子",
];
const SUGGESTIONS_EN = [
  "Explain quantum entanglement in simple terms",
  "Write a polite, professional time-off request email",
  "Implement quicksort in Python and explain the idea",
  "Give me three ideas for a weekend getaway",
];

/** System prompt for `/webdesign` mode — pushes the model to produce a single,
 *  polished, self-contained HTML UI (pairs with the in-app HTML preview). */
const WEBDESIGN_PROMPT = `You are an elite front-end designer and developer. The user will describe a UI or web page; you deliver ONE complete, self-contained HTML file that renders it beautifully.

Hard requirements:
- Output ONLY a single \`\`\`html code block containing a full document (<!doctype html> … </html>), with ALL CSS in a <style> tag and ALL JavaScript in a <script> tag. No external files and no build step (a Google Fonts <link> is allowed). It must run as-is when saved as one .html file.
- No placeholders, no "TODO", no lorem ipsum — write real, specific, relevant content and copy.

Design bar (make it look like a senior product designer built it, not a template):
- Clear visual hierarchy, generous whitespace, a deliberate type scale, and a cohesive, restrained color palette. Avoid the generic purple-gradient "AI" look.
- Polished details: consistent spacing, tasteful rounded corners, subtle shadows or hairline borders, and hover/focus states with smooth transitions.
- Responsive (mobile-first) and accessible: semantic HTML, labels, strong contrast, keyboard-focusable controls.
- Use a tasteful Google Font or clean system fonts, and inline SVG for icons.

Put at most one short sentence before the code block; let the design speak for itself.`;

function greetingKey(): TKey {
  const h = new Date().getHours();
  if (h < 6) return "greetNight";
  if (h < 12) return "greetMorning";
  if (h < 14) return "greetNoon";
  if (h < 18) return "greetAfternoon";
  return "greetEvening";
}

function formatDate(lang: Lang): string {
  return new Date().toLocaleDateString(lang === "zh" ? "zh-CN" : "en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    weekday: "long",
  });
}

/** Rough token estimate: CJK ≈ 1 token/char, other text ≈ 1 token per ~3.6 chars. */
function estimateTokens(text: string): number {
  let cjk = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0) ?? 0;
    if (
      (c >= 0x3000 && c <= 0x9fff) ||
      (c >= 0xac00 && c <= 0xd7a3) ||
      (c >= 0xf900 && c <= 0xfaff) ||
      (c >= 0xff00 && c <= 0xffef)
    ) {
      cjk++;
    }
  }
  return Math.ceil(cjk + (text.length - cjk) / 3.6);
}

function loadSettings(): GenSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...defaultSettings, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return defaultSettings;
}

export default function App() {
  const { t, lang } = useI18n();
  const confirm = useConfirm();
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelEntry[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [busy, setBusy] = useState(false);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [stats, setStats] = useState<GenStats | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);
  // Weight ticks come from a poller thread and can race the final "ready" —
  // drop anything that would move the bar backwards or resurrect it.
  const onLoadProgress = useCallback((p: LoadProgress) => {
    setLoadProgress((prev) => {
      if (prev && p.phase === "weights") {
        if (prev.phase === "ready") return prev;
        if (prev.phase === "weights" && p.frac <= prev.frac) return prev;
      }
      return p;
    });
  }, []);
  const [settings, setSettings] = useState<GenSettings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [showCmdk, setShowCmdk] = useState(false);
  const [canvasOpen, setCanvasOpen] = useState(false);
  const [canvasVersions, setCanvasVersions] = useState<CanvasVersion[]>([]);
  const [canvasIndex, setCanvasIndex] = useState(0);
  // Canvas iterations survive closing: sessions are keyed by the reply's
  // ORIGINAL html block, so reopening the same reply resumes its versions
  // while a different reply starts fresh. In-memory, app-session scoped.
  const canvasSessions = useRef(new Map<string, { versions: CanvasVersion[]; index: number }>());
  const [canvasKey, setCanvasKey] = useState("");
  const [canvasBusy, setCanvasBusy] = useState(false);
  // Set by the canvas Stop button; the generation flow then discards the
  // partial output instead of reporting a "no HTML" error.
  const canvasCancelRef = useRef(false);
  const [canvasStream, setCanvasStream] = useState<string | null>(null);
  const canvasStreamRef = useRef<{ acc: string; timer: number | null }>({ acc: "", timer: null });
  const [showHardware, setShowHardware] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [showDownload, setShowDownload] = useState(false);
  const [deepLink, setDeepLink] = useState<{ repo: string; file?: string } | null>(null);

  // Preview-only deep entry for the screenshot pipeline: ?open=store opens
  // the model store, &repo=Owner/Name jumps straight to a detail view,
  // &theme=light|dark overrides the theme (light-mode marketing shots).
  // Never active in the real app (VITE_UI_PREVIEW gates the mock build).
  useEffect(() => {
    if (!import.meta.env.VITE_UI_PREVIEW) return;
    const q = new URLSearchParams(window.location.search);
    const theme = q.get("theme");
    if (theme === "light" || theme === "dark") setSettings((s) => ({ ...s, theme }));
    if (q.get("open") === "store") {
      const repo = q.get("repo");
      if (repo) setDeepLink({ repo });
      setShowDownload(true);
    }
  }, []);
  const [showSetup, setShowSetup] = useState(false);
  const [notice, setNotice] = useState<{ kind: "warn" | "error"; text: string } | null>(null);
  const noticeTimer = useRef<number | null>(null);
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [updating, setUpdating] = useState(false);
  const [webEnabled, setWebEnabled] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);
  const [showKb, setShowKb] = useState(false);
  const [showPodcast, setShowPodcast] = useState(false);
  const [showDeepResearch, setShowDeepResearch] = useState(false);
  const [showKbReport, setShowKbReport] = useState(false);
  const [appMode, setAppMode] = useState<"chat" | "code">("chat");
  const [thinkEnabled, setThinkEnabled] = useState(() => {
    try {
      return localStorage.getItem("chaty.think") !== "0";
    } catch {
      return true;
    }
  });
  const [webDesign, setWebDesign] = useState(() => {
    try {
      return localStorage.getItem("chaty.webdesign") === "1";
    } catch {
      return false;
    }
  });
  const [searching, setSearching] = useState<"" | "web" | "kb" | "mix">("");
  const [composing, setComposing] = useState(false);
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [convQuery, setConvQuery] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [contentMatches, setContentMatches] = useState<Set<string>>(new Set());
  const [recorder, setRecorder] = useState<Recorder | null>(null);
  const recorderRef = useRef<Recorder | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [speakReplies, setSpeakReplies] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [showLive, setShowLive] = useState(false);
  const [showToolsMenu, setShowToolsMenu] = useState(false);
  const liveConvRef = useRef<string | null>(null);
  const playbackRef = useRef<{ stop: () => void } | null>(null);
  const speechRef = useRef<SpeechQueue | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevMsgCount = useRef(0);
  const prevConvId = useRef<string | null>(null);
  const followRef = useRef(true);
  const showJumpRef = useRef(false);
  const [showJump, setShowJump] = useState(false);
  const asideRef = useRef<HTMLElement>(null);
  const [sidebarW, setSidebarW] = useState(() => {
    try {
      const v = Number(localStorage.getItem("chaty.sidebarW"));
      if (Number.isFinite(v) && v >= SIDEBAR_MIN && v <= SIDEBAR_MAX) return v;
    } catch {
      /* ignore */
    }
    return SIDEBAR_DEFAULT;
  });

  // Drag the sidebar's right edge to resize. The width is driven through state
  // (rAF-throttled to one update per frame) so a concurrent re-render — e.g. a
  // reply streaming in — can't fight the drag and snap the width back. The
  // memoized message list doesn't re-render with it. Persists on release;
  // double-click the handle resets to the default width.
  function startSidebarResize(e: React.PointerEvent) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = asideRef.current?.offsetWidth ?? sidebarW;
    let frame: number | null = null;
    let latest = startW;
    document.body.classList.add("resizing-x");
    const onMove = (ev: PointerEvent) => {
      latest = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, startW + (ev.clientX - startX)));
      if (frame == null)
        frame = requestAnimationFrame(() => {
          frame = null;
          setSidebarW(latest);
        });
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      if (frame != null) cancelAnimationFrame(frame);
      document.body.classList.remove("resizing-x");
      setSidebarW(latest);
      try {
        localStorage.setItem("chaty.sidebarW", String(latest));
      } catch {
        /* ignore */
      }
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }

  function resetSidebarW() {
    setSidebarW(SIDEBAR_DEFAULT);
    try {
      localStorage.setItem("chaty.sidebarW", String(SIDEBAR_DEFAULT));
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    // StrictMode double-mounts this effect in dev; both invocations used to
    // race past the getModel() check and spawn TWO model loads — with a 35B
    // MLX model that's two sidecars resident at once. One boot, one load.
    if (bootLoadStarted) return;
    bootLoadStarted = true;
    (async () => {
      try {
        const models = await listModels().catch(() => [] as ModelEntry[]);
        setAvailableModels(models);
        // If the backend still holds a model (e.g. after a frontend-only reload), reuse it.
        const current = await getModel();
        if (current) {
          setModel(current);
          return;
        }
        // Otherwise auto-load: last session's model, else the first GGUF found
        // in the models/ folder.
        if (!settings.autoLoadLast) return;
        // Only ever auto-load the model the user explicitly loaded last —
        // never auto-pick models[0]. A models folder can contain something
        // far bigger than this machine (a 32B alphabetically-first model
        // froze a 48 GB Mac twice); first-run model choice belongs to the
        // user / "Set up for me", not a directory sort order.
        const last = localStorage.getItem(LAST_MODEL_KEY);
        const target = last;
        if (!target) return;
        setLoadingModel(true);
        try {
          const info = await loadModel(target, settings.gpuLayers, settings.contextLength || undefined, onLoadProgress);
          setModel(info);
          localStorage.setItem(LAST_MODEL_KEY, info.path);
          noticeForLoad(info);
        } catch (e) {
          if (last) localStorage.removeItem(LAST_MODEL_KEY); // file moved/deleted
          showLoadError(e);
        } finally {
          setLoadingModel(false);
      setLoadProgress(null);
        }
      } catch (e) {
        console.error(e);
      }
    })();
    refreshConversations();
  }, []);

  // Keep the tray menu labels in sync with the UI language.
  useEffect(() => {
    setTrayLanguage(lang).catch(() => {});
  }, [lang]);

  // Check GitHub for a newer release shortly after launch.
  useEffect(() => {
    const id = window.setTimeout(() => {
      checkUpdate()
        .then((u) => {
          if (u.available) setUpdate(u);
        })
        .catch(() => {});
    }, 3000);
    return () => window.clearTimeout(id);
  }, []);

  // Global ⌘K / Ctrl+K toggles the command palette. (DOM KeyboardEvent — the
  // bare name is React's here, imported above for composer key handling.)
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setShowCmdk((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Handle chaty:// deep links (HuggingFace "Use this model" → open the
  // downloader pre-filled with the repo). Covers cold start + while running.
  useEffect(() => {
    const open = (raw: string) => {
      const parsed = parseHfDeepLink(raw);
      if (!parsed) return;
      setDeepLink(parsed);
      setShowDownload(true);
      void getCurrentWindow().setFocus().catch(() => {});
    };
    let unlisten: (() => void) | undefined;
    (async () => {
      try {
        const cur = await getDeepLinks();
        if (cur && cur.length) open(cur[0]);
        unlisten = await onOpenUrl((urls) => {
          if (urls.length) open(urls[0]);
        });
      } catch (e) {
        console.error(e);
      }
    })();
    return () => unlisten?.();
  }, []);

  // Open external links (from model output, reports, anywhere) in the system
  // browser. Without this, clicking a link navigates the webview itself and the
  // app is gone. Capture phase so we beat the default navigation.
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented) return;
      const a = (e.target as HTMLElement | null)?.closest?.("a[href]") as HTMLAnchorElement | null;
      if (!a) return;
      const href = a.getAttribute("href") || "";
      if (/^(https?:|mailto:)/i.test(href)) {
        e.preventDefault();
        void openExternal(href).catch(console.error);
      }
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, []);

  // Close the model picker when clicking outside it.
  useEffect(() => {
    if (!showModelMenu) return;
    const close = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest(".model-wrap")) setShowModelMenu(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [showModelMenu]);

  // Close the tools menu when clicking outside it.
  useEffect(() => {
    if (!showToolsMenu) return;
    const close = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest(".tools-wrap")) setShowToolsMenu(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [showToolsMenu]);

  function openLive() {
    if (!model) return;
    liveConvRef.current = conversationId; // continue current chat, or null → new
    setShowToolsMenu(false);
    setShowLive(true);
  }

  /** Persist a completed live-mode exchange into the conversation history. */
  async function recordLiveTurn(userText: string, assistantText: string) {
    const fresh = !liveConvRef.current;
    const convId = liveConvRef.current ?? uid();
    liveConvRef.current = convId;
    const uMsg: UiMessage = { id: uid(), role: "user", content: userText };
    const aMsg: UiMessage = { id: uid(), role: "assistant", content: assistantText };
    setMessages((cur) => [...cur, uMsg, aMsg]);
    try {
      if (fresh) {
        setConversationId(convId);
        await saveConversation(convId, convTitle(userText), model?.path ?? null);
      }
      await saveMessage(uMsg.id, convId, "user", userText);
      await saveMessage(aMsg.id, convId, "assistant", assistantText);
      await refreshConversations();
      if (fresh) void makeTitle(convId, userText);
    } catch (e) {
      console.error(e);
    }
  }

  // Follow-the-stream is an *intent*, not a position: any upward wheel motion
  // releases it immediately (a distance check alone loses to the next stream
  // frame re-pinning the bottom before the user escapes the threshold), and
  // parking back at the bottom re-arms it.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const dist = () => el.scrollHeight - el.scrollTop - el.clientHeight;
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) followRef.current = false;
      else if (dist() < 40) followRef.current = true;
    };
    const onScroll = () => {
      // Covers scrollbar drags and keyboard scrolling; programmatic pins land
      // at the bottom, so they only ever re-arm.
      const d = dist();
      if (d < 4) followRef.current = true;
      else if (d > 240) followRef.current = false;
      const jump = d > 320;
      if (jump !== showJumpRef.current) {
        showJumpRef.current = jump;
        setShowJump(jump);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("scroll", onScroll);
    };
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Structural change (a message sent, a fresh assistant bubble, or switching
    // conversations) → jump to the bottom and re-arm following. While a reply
    // streams (same message count, content just growing) only stick to the
    // bottom while the user hasn't scrolled away.
    const structural =
      messages.length !== prevMsgCount.current || conversationId !== prevConvId.current;
    prevMsgCount.current = messages.length;
    prevConvId.current = conversationId;
    if (structural) {
      followRef.current = true;
      // A conversation switch fires no scroll event — without this the pill
      // from the previous chat lingers over an empty/short one.
      showJumpRef.current = false;
      setShowJump(false);
    }
    if (followRef.current) el.scrollTo({ top: el.scrollHeight });
  }, [messages, conversationId]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  // Mirror the HF endpoint setting into the ipc module, so every store
  // search / download / KB-model fetch follows it (incl. resolve URLs).
  useEffect(() => {
    setHfEndpoint(settings.hfEndpoint);
  }, [settings.hfEndpoint]);

  // Apply the colour theme to the document root (CSS keys off [data-theme],
  // plus per-appearance palettes on [data-dark] / [data-light]).
  useEffect(() => {
    const el = document.documentElement;
    el.dataset.theme = settings.theme;
    el.dataset.dark = settings.darkScheme;
    el.dataset.light = settings.lightScheme;
  }, [settings.theme, settings.darkScheme, settings.lightScheme]);

  // Keep each reply's canvas session current (cheap: refs into state).
  useEffect(() => {
    if (canvasKey && canvasVersions.length) {
      canvasSessions.current.set(canvasKey, { versions: canvasVersions, index: canvasIndex });
    }
  }, [canvasKey, canvasVersions, canvasIndex]);

  // Chat code-block highlight palette. Palettes with a light sibling follow
  // the app appearance (incl. live OS switches under the system theme).
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const apply = () =>
      applyCodeTheme(
        settings.codeTheme,
        settings.theme === "light" || (settings.theme === "system" && mq.matches),
      );
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [settings.codeTheme, settings.theme]);

  // Display preferences: UI zoom, motion kill-switch, answer reading size.
  // Zoom is the native webview page zoom — CSS `zoom` reflows the document
  // without the viewport, so fixed/vw elements overflow and the backdrop
  // shows through when shrinking.
  useEffect(() => {
    void setUiZoom(settings.uiScale).catch(console.error);
    document.documentElement.dataset.motion = settings.reduceMotion ? "reduce" : "";
    document.documentElement.dataset.answer = settings.answerSize;
  }, [settings.uiScale, settings.reduceMotion, settings.answerSize]);

  // Settings → Code: agent-browser visibility. Changing it closes any open
  // agent browser (Rust side) so the next tool call relaunches hidden/visible
  // as picked — the setting used to look ignored for the rest of the session.
  useEffect(() => {
    void browserSetHeadless(settings.codeBrowserHeadless).catch(console.error);
  }, [settings.codeBrowserHeadless]);

  // Tag the document root with the host OS once, so CSS can adapt the title bar
  // (e.g. macOS leaves room for the native traffic lights). CSS keys off [data-os].
  useEffect(() => {
    document.documentElement.dataset.os = OS_PLATFORM;
  }, []);

  // Debounced full-text search over message bodies (titles are matched locally).
  useEffect(() => {
    const q = convQuery.trim();
    if (q.length < 2) {
      setContentMatches(new Set());
      return;
    }
    let cancelled = false;
    const id = window.setTimeout(() => {
      searchConversations(q)
        .then((ids) => {
          if (!cancelled) setContentMatches(new Set(ids));
        })
        .catch(() => {});
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(id);
    };
  }, [convQuery]);

  useEffect(() => {
    try {
      localStorage.setItem("chaty.think", thinkEnabled ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [thinkEnabled]);

  useEffect(() => {
    try {
      localStorage.setItem("chaty.webdesign", webDesign ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [webDesign]);

  // Native file drag-and-drop onto the window → load as an attachment. Tauri
  // intercepts OS file drops and emits these events (HTML5 DnD is disabled).
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const p = event.payload;
        if (p.type === "enter" || p.type === "over") {
          setDragging(true);
        } else if (p.type === "leave") {
          setDragging(false);
        } else if (p.type === "drop") {
          setDragging(false);
          const path = p.paths?.[0];
          if (path) void loadAttachmentPath(path);
        }
      })
      .then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      unlisten?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshConversations() {
    try {
      setConversations(await listConversations());
    } catch (e) {
      console.error(e);
    }
  }

  /** Backend reasoning flag forcing no-think on switch-less models (Qwen3.5+),
   *  used by the title/query/summary helpers that should never spend time
   *  reasoning. `undefined` for everything else (no effect). */
  const noThinkFlag = (): boolean | undefined =>
    model?.supportsThinking && !model.thinkSwitch ? false : undefined;

  /** Ask the model for a concise title for a freshly created conversation. */
  async function makeTitle(convId: string, firstMsg: string) {
    if (!settings.autoTitle) return;
    try {
      let acc = "";
      await generate(
        {
          messages: [
            {
              role: "system",
              content:
                lang === "zh"
                  ? "请用一个不超过12个汉字的简短短语，概括下面这条消息的主题，作为对话标题。只输出标题本身，不要引号、标点、解释或思考过程。"
                  : "Summarize the topic of the following message as a short chat title (max ~5 words). Output only the title — no quotes, punctuation, explanation, or reasoning.",
            },
            { role: "user", content: `${firstMsg}${model?.thinkSwitch ? "\n/no_think" : ""}` },
          ],
          // Generous budget: thinking-specialised models (Qwen3.6) reason even
          // with an empty <think/> pre-fill — give them room to finish and
          // still emit the title after the block closes.
          params: { temperature: 0.2, topP: 0.9, maxTokens: 512, think: noThinkFlag() },
        },
        (ev) => {
          if (ev.type === "token") {
            acc += ev.text;
            // Live-scan feed, throttled: the panel re-diffs on every update.
            const st = canvasStreamRef.current;
            st.acc = acc;
            if (st.timer === null) {
              st.timer = window.setTimeout(() => {
                st.timer = null;
                // A stale timer from a finished generation must never
                // resurrect the stream (the ref object is replaced per run).
                if (canvasStreamRef.current === st) setCanvasStream(st.acc);
              }, 150);
            }
          }
        },
      );
      const title = cleanTitle(acc);
      if (title) {
        await renameConversation(convId, title);
        await refreshConversations();
      }
    } catch (e) {
      console.error(e);
    }
  }

  // ----- Canvas (design studio) -----

  /** Pull a single-file HTML document out of model output (fenced or raw). */
  function extractHtml(text: string): string {
    const fences = [...text.matchAll(/```(?:html|htm)?\s*\n?([\s\S]*?)```/gi)].map((m) => m[1]);
    const fromFence = fences.find((c) => /<!doctype|<html/i.test(c)) ?? fences[0];
    let html = (fromFence ?? text).trim();
    const start = html.search(/<!doctype html|<html/i);
    if (start >= 0) {
      const end = html.toLowerCase().lastIndexOf("</html>");
      html = end >= 0 ? html.slice(start, end + 7) : html.slice(start);
    }
    return html.trim();
  }

  /** Open an HTML snippet (e.g. from a chat message) in the Canvas studio. */
  function openInCanvas(raw: string) {
    const html = extractHtml(raw) || raw.trim();
    if (!html) return;
    const prior = canvasSessions.current.get(html);
    if (prior) {
      setCanvasVersions(prior.versions);
      setCanvasIndex(prior.index);
    } else {
      setCanvasVersions([{ html, note: t("canvasInitial") }]);
      setCanvasIndex(0);
    }
    setCanvasKey(html);
    setCanvasOpen(true);
  }
  // DEV hook: drive the full canvas stack from the console/harness without a
  // model round-trip (the console-pipeline hunt needed exactly this).
  if (import.meta.env.DEV) {
    (window as unknown as { __openCanvas?: (raw: string) => void }).__openCanvas = openInCanvas;
  }

  /** Parse `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` edit blocks. */
  function parseEdits(text: string): { search: string; replace: string }[] {
    const re =
      /<{5,}\s*SEARCH[^\n]*\r?\n([\s\S]*?)\r?\n={3,}[^\n]*\r?\n([\s\S]*?)\r?\n>{5,}\s*REPLACE/g;
    const out: { search: string; replace: string }[] = [];
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) out.push({ search: m[1], replace: m[2] });
    return out;
  }

  /** Apply search/replace edits to `base`. Returns null if any search misses. */
  function applyEdits(base: string, edits: { search: string; replace: string }[]): string | null {
    let html = base;
    const norm = (s: string) => s.replace(/\r/g, "").replace(/[ \t]+$/gm, "");
    for (const e of edits) {
      if (!e.search) continue;
      if (html.includes(e.search)) {
        html = html.replace(e.search, e.replace);
        continue;
      }
      // Lenient: ignore CRs and trailing whitespace.
      const nHtml = norm(html);
      const nSearch = norm(e.search);
      if (nSearch && nHtml.includes(nSearch)) {
        html = nHtml.replace(nSearch, norm(e.replace));
        continue;
      }
      return null;
    }
    return html;
  }

  /** Generate a new Canvas version from the current one (an edit or a fix). */
  async function generateCanvasVersion(kind: "edit" | "fix", payload: string) {
    if (busy || canvasBusy) return;
    if (!model) {
      showNotice("error", t("canvasNeedsModel"));
      return;
    }
    const base = canvasVersions[canvasIndex]?.html ?? "";
    const newIndex = canvasVersions.length;
    // Prefer a minimal search/replace patch (small output = fast), but allow the
    // model to return the full file instead — small local models rarely echo an
    // exact SEARCH block, so the full-HTML path is the reliable fallback.
    // Rewrite mode: the model streams the whole corrected document and the
    // system computes the live diff (buildScanView's full mode). No verbatim
    // SEARCH echoes required — the reliable path for smaller models.
    const sys =
      settings.canvasEditMode === "rewrite"
        ? lang === "zh"
          ? "你是一个网页设计助手，正在修改一个单文件 HTML 文档。\n请在一个 ```html 代码块里返回**完整修正后**的单文件 HTML(从 <!doctype html> 到 </html>,包含所有未改动的部分)。\n所有资源必须内联（禁止外部 CDN、字体或图片链接）；不要输出任何解释或思考。"
          : "You are a web-design assistant editing a single-file HTML document.\nReturn the COMPLETE corrected single-file HTML in one ```html code block (from <!doctype html> to </html>, including every unchanged part).\nKeep everything inline (no external CDNs, fonts or image URLs); output no explanation or reasoning."
        : lang === "zh"
          ? "你是一个网页设计助手，正在修改一个单文件 HTML 文档。\n优先用「查找/替换」补丁做最小改动（更快）：\n<<<<<<< SEARCH\n（从当前 HTML 原样复制、要被替换的片段）\n=======\n（替换后的新片段）\n>>>>>>> REPLACE\n可以输出多个这样的块，每个 SEARCH 必须与当前 HTML 完全一致。\n如果不方便用补丁，就直接在一个 ```html 代码块里返回**完整修正后**的单文件 HTML。\n所有资源必须内联（禁止外部 CDN、字体或图片链接）；不要输出任何解释或思考。"
          : "You are a web-design assistant editing a single-file HTML document.\nPrefer minimal search/replace patches for speed:\n<<<<<<< SEARCH\n(exact snippet copied verbatim from the current HTML)\n=======\n(the new snippet)\n>>>>>>> REPLACE\nYou may output several blocks; each SEARCH must match the current HTML exactly.\nIf a patch is awkward, instead return the COMPLETE corrected single-file HTML in one ```html code block.\nKeep everything inline (no external CDNs, fonts or image URLs); output no explanation or reasoning.";
    const how =
      settings.canvasEditMode === "rewrite"
        ? lang === "zh" ? "（返回完整修正后的 HTML）" : "(return the complete corrected HTML)"
        : lang === "zh" ? "（用查找/替换补丁，或返回完整 HTML）" : "(as a search/replace patch, or return the full HTML)";
    const user =
      kind === "edit"
        ? lang === "zh"
          ? `当前页面的完整 HTML：\n\`\`\`html\n${base}\n\`\`\`\n请实现以下修改${how}：${payload}`
          : `Current full HTML:\n\`\`\`html\n${base}\n\`\`\`\nApply this change ${how}: ${payload}`
        : lang === "zh"
          ? `当前页面的完整 HTML：\n\`\`\`html\n${base}\n\`\`\`\n${fixInstruction(payload, "zh", how)}`
          : `Current full HTML:\n\`\`\`html\n${base}\n\`\`\`\n${fixInstruction(payload, "en", how)}`;
    const note =
      (kind === "edit" ? `${t("canvasEdit")}：${payload}` : `${t("canvasFix")}：${payload}`).slice(
        0,
        48,
      );
    setCanvasBusy(true);
    setBusy(true);
    canvasCancelRef.current = false;
    canvasStreamRef.current = { acc: "", timer: null };
    setCanvasStream("");
    // Visual context: a vision model also SEES the current page (a real
    // headless-browser screenshot) and its live console errors, so an edit
    // request isn't judged from the HTML source alone. Best-effort — a browser
    // failure degrades to the text-only path.
    let shotImages: string[] | undefined;
    let visualNote = "";
    if (model.visionReady && base.trim()) {
      try {
        const cap = await browserRenderHtml(base);
        shotImages = [cap.image];
        const con = cap.console;
        if (con && !/console is empty|控制台无输出/.test(con)) {
          visualNote =
            lang === "zh"
              ? `\n\n当前页面的浏览器控制台输出/报错:\n${con.slice(0, 1500)}`
              : `\n\nCurrent browser console output/errors:\n${con.slice(0, 1500)}`;
        }
      } catch (e) {
        console.error("canvas visual capture failed", e);
      }
    }
    const userText =
      `${user}${visualNote}` +
      (shotImages
        ? lang === "zh"
          ? "\n\n(上方附有当前页面的实际渲染截图,请据此判断视觉效果后再修改。)"
          : "\n\n(A screenshot of how the page currently renders is attached — judge the visual result from it before editing.)"
        : "") +
      (model?.thinkSwitch ? "\n/no_think" : "");
    try {
      let acc = "";
      await generate(
        {
          messages: [
            { role: "system", content: sys },
            { role: "user", content: userText, images: shotImages },
          ],
          params: { temperature: 0.4, topP: 0.9, maxTokens: 8192, think: noThinkFlag() },
        },
        (ev) => {
          if (ev.type === "token") {
            acc += ev.text;
            // Live-scan feed, throttled: the panel re-diffs on every update.
            const st = canvasStreamRef.current;
            st.acc = acc;
            if (st.timer === null) {
              st.timer = window.setTimeout(() => {
                st.timer = null;
                // A stale timer from a finished generation must never
                // resurrect the stream (the ref object is replaced per run).
                if (canvasStreamRef.current === st) setCanvasStream(st.acc);
              }, 150);
            }
          }
        },
      );
      // A deliberate stop discards the partial output quietly — no error
      // notice, no version.
      if (canvasCancelRef.current) return;
      // Prefer applying the patch; fall back to a full document if the model
      // returned one instead (or the patch didn't apply cleanly).
      const edits = parseEdits(acc);
      let html = edits.length ? applyEdits(base, edits) : null;
      if (!html) {
        // The model returned a full document (or the patch didn't apply) —
        // accept any complete single-file HTML.
        const full = extractHtml(acc);
        if (full && /<!doctype|<html/i.test(full)) html = full;
      }
      if (!html) {
        showNotice("error", t("canvasNoHtml"));
        return;
      }
      setCanvasVersions((vs) => [...vs, { html, note }]);
      setCanvasIndex(newIndex);
    } catch (e) {
      if (canvasCancelRef.current) return;
      console.error(e);
      showLoadError(e);
    } finally {
      const st = canvasStreamRef.current;
      if (st.timer !== null) window.clearTimeout(st.timer);
      st.timer = null;
      setCanvasStream(null);
      setCanvasBusy(false);
      setBusy(false);
    }
  }

  /** Rewrite the latest question into a standalone search query using context. */
  async function rewriteQuery(prior: UiMessage[], latest: string): Promise<string> {
    try {
      const recent = prior
        .slice(-6)
        .map((m) => `${m.role === "user" ? "用户" : "助手"}: ${stripThink(m.content).slice(0, 240)}`)
        .join("\n");
      let acc = "";
      await generate(
        {
          messages: [
            {
              role: "system",
              content:
                lang === "zh"
                  ? "根据下面的对话历史，把用户最新的问题改写成一个用于网页搜索的查询词：补全省略的指代和主体，使其能独立用于搜索。只输出查询词本身，不要解释、引号或思考过程。"
                  : "Using the conversation history, rewrite the user's latest question into a standalone web search query (resolve pronouns / omitted subjects). Output only the query — no explanation, quotes, or reasoning.",
            },
            {
              role: "user",
              content: `${recent}\n\n最新问题：${latest}${model?.thinkSwitch ? "\n/no_think" : ""}`,
            },
          ],
          params: { temperature: 0.2, topP: 0.9, maxTokens: 512, think: noThinkFlag() },
        },
        (ev) => {
          if (ev.type === "token") {
            acc += ev.text;
            // Live-scan feed, throttled: the panel re-diffs on every update.
            const st = canvasStreamRef.current;
            st.acc = acc;
            if (st.timer === null) {
              st.timer = window.setTimeout(() => {
                st.timer = null;
                // A stale timer from a finished generation must never
                // resurrect the stream (the ref object is replaced per run).
                if (canvasStreamRef.current === st) setCanvasStream(st.acc);
              }, 150);
            }
          }
        },
      );
      return cleanQuery(acc) || latest;
    } catch {
      return latest;
    }
  }

  function showNotice(kind: "warn" | "error", text: string) {
    if (noticeTimer.current) window.clearTimeout(noticeTimer.current);
    setNotice({ kind, text });
    noticeTimer.current = window.setTimeout(
      () => setNotice(null),
      kind === "error" ? 9000 : 6000,
    );
  }

  /** Open the models folder + a hint that drop-ins appear on picker reopen —
   *  first-time users read "open folder" as a file picker (issue #1). */
  function openModelsFolder() {
    openModelsDir()
      .then(() => showNotice("warn", t("modelsDirHintToast")))
      .catch(console.error);
  }

  /** Surface a non-fatal load warning (e.g. GPU offload reduced to fit memory). */
  function noticeForLoad(info: ModelInfo) {
    if (info.warning === "gpu-oom") {
      showNotice(
        "warn",
        info.gpuLayers > 0
          ? t("oomPartial", { a: info.gpuLayers, b: info.nLayer ?? "?" })
          : t("oomCpu"),
      );
    } else if (info.warning === "mmproj-failed") {
      showNotice("warn", t("mmprojFailed"));
    } else if (info.warning === "ctx-clamped" && info.nCtx) {
      showNotice("warn", t("ctxClamped", { n: info.nCtx }));
    }
  }

  /** Turn a model-load failure into a friendly toast (OOM gets a clear hint). */
  function showLoadError(e: unknown) {
    const msg = typeof e === "string" ? e : ((e as Error)?.message ?? String(e));
    if (/out of memory|内存不足|allocate|insufficient memory/i.test(msg)) {
      showNotice("error", t("oomFail"));
    } else {
      showNotice("error", msg.slice(0, 220));
    }
  }

  async function applyUpdate() {
    if (!update?.url || updating) return;
    setUpdating(true);
    try {
      await runUpdate(update.url); // downloads + launches installer, then app exits
    } catch (e) {
      setUpdating(false);
      showLoadError(e);
    }
  }

  async function refreshModels() {
    try {
      setAvailableModels(await listModels());
    } catch (e) {
      console.error(e);
    }
  }

  /** Hot-swap to another already-discovered model. */
  async function switchModel(path: string) {
    setShowModelMenu(false);
    if (busy || model?.path === path) return;
    setLoadingModel(true);
    try {
      const info = await loadModel(path, settings.gpuLayers, settings.contextLength || undefined, onLoadProgress);
      setModel(info);
      localStorage.setItem(LAST_MODEL_KEY, info.path);
      noticeForLoad(info);
    } catch (e) {
      console.error(e);
      showLoadError(e);
    } finally {
      setLoadingModel(false);
      setLoadProgress(null);
    }
  }

  /** Reload the current model so settings (context length, GPU layers) take effect. */
  async function reloadModel() {
    if (!model || busy || loadingModel) return;
    setLoadingModel(true);
    try {
      const info = await loadModel(model.path, settings.gpuLayers, settings.contextLength || undefined, onLoadProgress);
      setModel(info);
      noticeForLoad(info);
    } catch (e) {
      console.error(e);
      showLoadError(e);
    } finally {
      setLoadingModel(false);
      setLoadProgress(null);
    }
  }

  /** Unload the active model and return Chaty to the empty state. */
  async function handleEject() {
    setShowModelMenu(false);
    if (!model || busy || loadingModel) return;
    setLoadingModel(true);
    try {
      await ejectModel();
      setModel(null);
      localStorage.removeItem(LAST_MODEL_KEY);
    } catch (e) {
      console.error(e);
      showLoadError(e);
    } finally {
      setLoadingModel(false);
      setLoadProgress(null);
    }
  }

  /** Permanently delete a model file from disk (after a confirm). */
  async function handleDeleteModel(m: ModelEntry) {
    if (busy || loadingModel) return;
    if (
      !(await confirm({
        message: t("confirmDeleteModel", { name: m.name }),
        title: t("deleteModelFile"),
        confirmLabel: t("confirmDelete"),
        danger: true,
      }))
    ) {
      return;
    }
    try {
      await deleteModelFile(m.path);
      await refreshModels();
    } catch (e) {
      console.error(e);
      showNotice("error", typeof e === "string" ? e : ((e as Error)?.message ?? String(e)));
    }
  }

  // One folder per model is the canonical layout for BOTH formats (GGUF
  // main+mmproj, MLX config+safetensors) — the backend resolves whichever
  // the folder contains.
  async function handleLoadFolder() {
    setShowModelMenu(false);
    try {
      const path = await pickModelFolder();
      if (!path) return;
      setLoadingModel(true);
      const info = await loadModel(path, settings.gpuLayers, settings.contextLength || undefined, onLoadProgress);
      setModel(info);
      localStorage.setItem(LAST_MODEL_KEY, info.path);
      noticeForLoad(info);
      void refreshModels();
    } catch (e) {
      console.error(e);
      showLoadError(e);
    } finally {
      setLoadingModel(false);
      setLoadProgress(null);
    }
  }

  function handleNewChat() {
    if (busy) return;
    setConversationId(null);
    setMessages([]);
    setStats(null);
    setAttachment(null);
    setAttachError("");
  }

  function exportConversation(fmt: "md" | "json") {
    setShowExport(false);
    const list = messages.filter((m) => m.content.trim());
    if (list.length === 0) return;
    const conv = conversations.find((c) => c.id === conversationId);
    const title = conv?.title ?? "chat";
    const safe = title.replace(/[\\/:*?"<>|\n]+/g, "_").slice(0, 60) || "chat";
    const userL = lang === "zh" ? "用户" : "You";
    const asstL = lang === "zh" ? "助手" : "Assistant";
    let content: string;
    if (fmt === "json") {
      content = JSON.stringify(
        {
          title,
          model: model?.name ?? null,
          exportedAt: new Date().toISOString(),
          messages: list.map((m) => ({ role: m.role, content: m.content })),
        },
        null,
        2,
      );
    } else {
      const lines = [`# ${title}`, ""];
      for (const m of list) {
        lines.push(`## ${m.role === "user" ? userL : asstL}`, "", stripThink(m.content).trim(), "");
      }
      content = lines.join("\n");
    }
    void exportTextFile(`${safe}.${fmt}`, content, fmt).catch((e) => {
      console.error(e);
      showNotice("error", t("exportFailed"));
    });
  }

  async function loadAttachmentPath(path: string) {
    setAttachError("");
    // Vision-ready model + image file → attach the pixels themselves (the
    // model sees the image); otherwise images fall back to the OCR text path.
    if (model?.visionReady && isVisionImagePath(path)) {
      const name = path.split(/[/\\]/).pop() ?? "image";
      setAttachment({ name, kind: "vision", text: "", chars: 0, truncated: false, path });
      return;
    }
    try {
      setAttaching(true);
      setAttachment(await readAttachment(path));
    } catch (e) {
      setAttachment(null);
      setAttachError(typeof e === "string" ? e : t("readAttachFailed"));
    } finally {
      setAttaching(false);
    }
  }

  async function handleAttach() {
    const path = await pickAttachmentFile();
    if (!path) return;
    await loadAttachmentPath(path);
  }

  async function openConversation(id: string) {
    if (busy || id === conversationId) return;
    try {
      const stored = await getMessages(id);
      setMessages(
        stored.map((m) => ({ id: m.id, role: m.role, content: m.content, images: m.images })),
      );
      setConversationId(id);
      setStats(null);
      setAttachment(null);
      setAttachError("");
    } catch (e) {
      console.error(e);
    }
  }

  async function handleDelete(id: string) {
    if (
      !(await confirm({
        message: t("confirmDeleteConv"),
        title: t("deleteConv"),
        confirmLabel: t("confirmDelete"),
        danger: true,
      }))
    ) {
      return;
    }
    try {
      // Deleting the conversation you're viewing must also clear the chat area —
      // even mid-generation. handleNewChat's busy-guard exists for the "+ New
      // chat" button (don't silently abandon a running reply via the button); a
      // destructive delete should always reset, so cancel any in-flight reply
      // and reset the view inline rather than going through that guard.
      const isCurrent = id === conversationId;
      // Only a *chat* stream is ours to cancel here; busy from Deep Research /
      // Podcast (streamingId is null) must not be force-unlocked by a delete.
      const wasStreaming = isCurrent && streamingId != null;
      if (wasStreaming) {
        try {
          await cancelGeneration();
        } catch (err) {
          console.error(err);
        }
      }
      await deleteConversation(id);
      if (isCurrent) {
        setConversationId(null);
        setMessages([]);
        setStats(null);
        setAttachment(null);
        setAttachError("");
        if (wasStreaming) {
          setStreamingId(null);
          setBusy(false);
        }
      }
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }
  }

  /** Toggle a conversation's pinned state (pinned ones float to the top). */
  async function handleTogglePin(c: Conversation) {
    try {
      await setConversationPinned(c.id, !c.pinned);
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }
  }

  function startRename(c: Conversation) {
    setRenamingId(c.id);
    setRenameDraft(c.title);
  }

  async function commitRename() {
    const id = renamingId;
    const title = renameDraft.trim();
    setRenamingId(null);
    if (!id || !title) return;
    try {
      await renameConversation(id, title);
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }
  }

  /** Fork a new conversation from the messages up to (and including) `index`. */
  async function handleFork(index: number) {
    if (busy) return;
    const slice = messages.slice(0, index + 1).filter((m) => m.content.trim());
    if (slice.length === 0) return;
    const newId = uid();
    const parent = conversations.find((c) => c.id === conversationId);
    const firstUser = slice.find((m) => m.role === "user");
    const title = parent?.title ?? convTitle(firstUser?.content ?? "新对话");
    try {
      await saveConversation(newId, title, model?.path ?? null);
      const copied: UiMessage[] = [];
      for (const m of slice) {
        const id = uid();
        await saveMessage(id, newId, m.role, m.content, m.images);
        copied.push({ id, role: m.role, content: m.content, images: m.images });
      }
      setConversationId(newId);
      setMessages(copied);
      setStats(null);
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }
  }

  /** When a conversation nears the model's context window, summarise the older
   *  turns into one note and keep only the recent tail verbatim. Returns `null`
   *  when there's still plenty of room (the common case). Non-destructive: the
   *  stored/displayed messages are never touched — only the prompt we send. */
  async function composeContext<T extends { role: Role; content: string }>(
    msgs: T[],
  ): Promise<{ summary: string; tail: T[] } | null> {
    const nCtx = model?.nCtx ?? 0;
    if (!nCtx || msgs.length < 6) return null;

    // Room for the answer + chat markup. With no reply cap, reserve a sane slice.
    const reserve = (settings.limitTokens ? settings.maxTokens : 2048) + 700;
    const budget = Math.max(1024, nCtx - reserve);
    const cost = (m: { content: string }) => estimateTokens(m.content) + 8;
    const total = msgs.reduce((s, m) => s + cost(m), 0);
    if (total <= budget * 0.85) return null; // still comfortable

    // Keep the most recent turns within ~half the budget; summarise the rest.
    const tailCap = budget * 0.5;
    let acc = 0;
    let keep = 0;
    for (let i = msgs.length - 1; i >= 0; i--) {
      acc += cost(msgs[i]);
      if (acc > tailCap && keep >= 2) break;
      keep++;
    }
    const splitAt = Math.max(1, msgs.length - keep);
    const head = msgs.slice(0, splitAt);
    const tail = msgs.slice(splitAt);
    if (head.length === 0) return null;

    let transcript = head
      .map((m) => `${m.role === "user" ? "用户" : "助手"}: ${m.content}`)
      .join("\n");
    if (transcript.length > 7000) transcript = transcript.slice(-7000);

    setComposing(true);
    let out = "";
    try {
      await generate(
        {
          messages: [
            {
              role: "system",
              content:
                lang === "zh"
                  ? "请把下面这段较早的对话压缩成简洁的要点摘要，保留关键事实、结论、用户偏好和未决事项，省略寒暄。只输出摘要正文，不要解释或思考过程。"
                  : "Condense the earlier conversation below into a concise summary of key facts, conclusions, user preferences, and open threads. Output only the summary — no preamble or reasoning.",
            },
            {
              role: "user",
              content: `${transcript}${model?.thinkSwitch ? "\n/no_think" : ""}`,
            },
          ],
          params: { temperature: 0.3, topP: 0.9, maxTokens: 400, think: noThinkFlag() },
        },
        (ev) => {
          if (ev.type === "token") out += ev.text;
        },
      );
    } finally {
      setComposing(false);
    }

    const summary = stripThink(out).trim();
    if (!summary) return null;
    return { summary: t("contextSummary") + summary, tail };
  }

  /** Core generation turn: web search → build prompt → stream into `asstId`,
   *  then persist. Reused by send, regenerate and edit. */
  async function streamAssistant(
    history: UiMessage[],
    asstId: string,
    convId: string,
    opts: { freshConv: boolean },
  ) {
    const text = [...history].reverse().find((m) => m.role === "user")?.content ?? "";
    const prior = history.slice(0, -1);
    setBusy(true);
    setStreamingId(asstId);
    setStats(null);

    let webContext = "";
    const urls = (text.match(URL_RE) ?? []).slice(0, 3);
    if (webEnabled || ragEnabled || urls.length > 0) {
      const webish = webEnabled || urls.length > 0;
      setSearching(ragEnabled && webish ? "mix" : ragEnabled ? "kb" : "web");
      try {
        const blocks: string[] = [];
        const usedSources: SearchResult[] = [];

        // D — fetch any URLs the user pasted (highest priority).
        for (const url of urls) {
          try {
            const page = await fetchUrl(url);
            blocks.push(`【${blocks.length + 1}】 ${page.title}\n${page.text.slice(0, 5000)}`);
            usedSources.push({ title: page.title, url: page.url, snippet: page.text.slice(0, 360) });
          } catch (e) {
            console.error(e);
          }
        }

        // B2 — local knowledge base (hybrid retrieval over indexed documents).
        if (ragEnabled) {
          try {
            const query = prior.length > 0 ? await rewriteQuery(prior, text) : text;
            const hits = await ragSearch(query, 6);
            // Group retrieved chunks by their source file so the user sees one
            // citation per document, not one per chunk. First-seen order keeps
            // the best-scoring file first; chunks within a file go in document
            // order. Block ↔ source stay paired so 【N】 numbering lines up.
            const byDoc = new Map<string, typeof hits>();
            for (const h of hits) {
              const g = byDoc.get(h.docName);
              if (g) g.push(h);
              else byDoc.set(h.docName, [h]);
            }
            for (const [docName, group] of byDoc) {
              const ordered = [...group].sort((a, b) => a.seq - b.seq);
              const body = ordered.map((h) => h.text.slice(0, 2200)).join("\n\n…\n\n");
              blocks.push(`【${blocks.length + 1}】 ${docName}\n${body}`);
              usedSources.push({
                title: docName,
                url: "",
                snippet: ordered.map((h) => h.text).join("\n\n").slice(0, 600),
              });
            }
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            if (msg.includes("RAG_MODEL_MISSING")) showNotice("warn", t("kbNeedSetup"));
            else console.error(e);
          }
        }

        // C — web search using a context-aware, rewritten query.
        if (webEnabled) {
          const query = prior.length > 0 ? await rewriteQuery(prior, text) : text;
          const research = await webResearch(query);
          let budget = 5400;
          let added = 0;
          for (const p of research.pages) {
            if (budget <= 0) break;
            const txt = p.text.slice(0, budget);
            if (txt.length < 40) continue;
            blocks.push(`【${blocks.length + 1}】 ${p.title}\n${txt}`);
            usedSources.push({ title: p.title, url: p.url, snippet: txt.slice(0, 360) });
            budget -= txt.length;
            added++;
          }
          if (added === 0 && research.results.length) {
            research.results.slice(0, 6).forEach((r) => {
              blocks.push(`【${blocks.length + 1}】 ${r.title}\n${r.snippet}`);
              usedSources.push({ title: r.title, url: r.url, snippet: r.snippet.slice(0, 360) });
            });
          }
        }

        if (usedSources.length) {
          setMessages((cur) =>
            cur.map((m) => (m.id === asstId ? { ...m, sources: usedSources } : m)),
          );
        }
        if (blocks.length) {
          // KB mode gets the strict-grounding instruction: answer only from the
          // retrieved passages, never invent, admit when they don't cover it.
          webContext = (ragEnabled ? t("ragInstruction") : t("webInstruction")) + blocks.join("\n\n---\n\n");
        }
      } catch (e) {
        console.error(e);
      } finally {
        setSearching("");
      }
    }

    // Never feed a prior turn's reasoning back to the model: Qwen's own guidance
    // is that history should carry only the final answer, and stale <think> blocks
    // just waste context and confuse newer (3.5+) reasoning parsers.
    const historyForModel = history.map(({ role, content, images }) => ({
      role,
      content: role === "assistant" ? stripThink(content) : content,
      // Only vision-ready models get pixels; otherwise images were already
      // OCR'd into text at attach time (or never attached).
      ...(role === "user" && images?.length && model?.visionReady ? { images } : {}),
    }));

    // Near the context limit: summarise the older turns so the user can keep the
    // conversation going. This is non-destructive — the UI still shows every
    // message; only the model-facing prompt is compacted into a summary + tail.
    let summaryNote = "";
    let modelHistory = historyForModel;
    try {
      const comp = await composeContext(historyForModel);
      if (comp) {
        summaryNote = comp.summary;
        modelHistory = comp.tail;
      }
    } catch (e) {
      console.error(e);
    }

    // Thinking-mode control. Two mechanisms, picked by what the model supports:
    //  • Qwen3 (`thinkSwitch`): append the `/no_think` soft switch to the prompt.
    //  • Qwen3.5+ (reasoning, but no soft switch): tell the backend to pre-fill an
    //    empty <think></think> block via the `think` param below.
    // Web search forces no-think either way to keep answers concise.
    const wantNoThink = !thinkEnabled || webEnabled;
    if (modelHistory.length > 0 && model?.thinkSwitch && wantNoThink) {
      const last = modelHistory[modelHistory.length - 1];
      last.content = `${last.content}\n/no_think`;
    }
    // For switch-less reasoning models, drive thinking through the backend flag.
    const thinkParam =
      model?.supportsThinking && !model.thinkSwitch ? !wantNoThink : undefined;

    // Only tell the model today's date when the question is actually time-related,
    // otherwise short prompts can trigger the model to recite the date.
    const needsDate =
      webEnabled ||
      /\b(today|date|now|current|recent|yesterday|tomorrow|this (?:week|month|year)|what day|weekday)\b|今天|日期|现在|最近|几号|星期|今年|去年|明年|昨天|明天|当前|目前/i.test(
        text,
      );

    const sys = settings.systemPrompt.trim();
    const sent: ChatMessage[] = [
      ...(needsDate
        ? [{ role: "system" as const, content: t("todayNote", { date: formatDate(lang) }) }]
        : []),
      ...(webDesign ? [{ role: "system" as const, content: WEBDESIGN_PROMPT }] : []),
      ...(sys ? [{ role: "system" as const, content: sys }] : []),
      ...(attachment && attachment.kind !== "vision"
        ? [
            {
              role: "system" as const,
              content:
                t("attachInstruction", { name: attachment.name }) +
                attachment.text.slice(0, 9000),
            },
          ]
        : []),
      ...(webContext ? [{ role: "system" as const, content: webContext }] : []),
      ...(summaryNote ? [{ role: "system" as const, content: summaryNote }] : []),
      ...modelHistory,
    ];

    // Streaming text-to-speech: synthesize & play sentence-by-sentence as the
    // answer arrives, so audio starts long before generation finishes.
    const useTTS = speakReplies && lang === "en";
    let speech: SpeechQueue | null = null;
    let synthChain: Promise<void> = Promise.resolve();
    let spokenLen = 0;
    if (useTTS) {
      stopSpeaking();
      speech = new SpeechQueue();
      speechRef.current = speech;
      setSpeaking(true);
    }
    const enqueueChunk = (raw: string) => {
      const clean = forSpeech(raw);
      if (!clean || !speech) return;
      const q = speech;
      synthChain = synthChain.then(async () => {
        if (q.isStopped) return;
        try {
          const { audio, sampleRate } = await synthesize(clean, settings.voiceSpeed, settings.voiceSid);
          if (!q.isStopped) q.enqueue(decodeAudio(audio), sampleRate);
        } catch (e) {
          console.error(e);
        }
      });
    };
    const pumpSpeech = (final: boolean) => {
      if (!useTTS || !speech) return;
      const ans = answerOnly(acc.text);
      let pending = ans.slice(spokenLen);
      if (final) {
        spokenLen = ans.length;
      } else {
        const [done] = cutSentences(pending);
        if (!done) return;
        pending = done;
        spokenLen += done.length;
      }
      enqueueChunk(pending);
    };

    const acc = { text: "" };
    // Coalesce token → state into one re-render per animation frame (was: one
    // setMessages per token = a full re-render of the list on every token).
    let rafId: number | null = null;
    const renderMsg = () =>
      setMessages((cur) => cur.map((m) => (m.id === asstId ? { ...m, content: acc.text } : m)));
    const scheduleRender = () => {
      if (rafId == null)
        rafId = requestAnimationFrame(() => {
          rafId = null;
          renderMsg();
        });
    };
    try {
      await generate(
        {
          messages: sent,
          params: {
            temperature: settings.temperature,
            topP: settings.topP,
            // 0 = no per-reply cap (backend treats it as unlimited within the context)
            maxTokens: settings.limitTokens ? settings.maxTokens : 0,
            topK: settings.topK,
            minP: settings.minP,
            repeatPenalty: settings.repeatPenalty,
            stop: parseStops(settings.stop),
            think: thinkParam,
          },
        },
        (ev: StreamEvent) => {
          if (ev.type === "token") {
            acc.text += ev.text;
            scheduleRender();
            pumpSpeech(false);
          } else if (ev.type === "done") {
            if (rafId != null) {
              cancelAnimationFrame(rafId);
              rafId = null;
            }
            renderMsg();
            setStats(ev.stats);
          } else if (ev.type === "error") {
            acc.text += `\n\n**${ev.message}**`;
            if (rafId != null) {
              cancelAnimationFrame(rafId);
              rafId = null;
            }
            renderMsg();
          }
        },
      );
    } catch (e) {
      console.error(e);
    } finally {
      // Ensure the final text is rendered even if the stream ended without a
      // clean "done" (cancel / error), and stop any pending frame.
      if (rafId != null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      renderMsg();
      setBusy(false);
      setStreamingId(null);
      try {
        if (acc.text.trim()) await saveMessage(asstId, convId, "assistant", acc.text);
        await refreshConversations();
      } catch (e) {
        console.error(e);
      }
      // Let the model name a fresh conversation from its first question.
      if (opts.freshConv && acc.text.trim()) void makeTitle(convId, text);
      // Flush the trailing sentence and clear the speaking state once audio ends.
      if (useTTS && speech) {
        pumpSpeech(true);
        const q = speech;
        void synthChain
          .then(() => q.whenIdle())
          .finally(() => {
            if (speechRef.current === q) {
              speechRef.current = null;
              setSpeaking(false);
            }
          });
      }
    }
  }

  async function handleSend(override?: string) {
    const text = (override ?? input).trim();
    if (!text || busy) return;
    // Slash command: `/webdesign` toggles web-design mode (no message sent).
    if (!override && /^\/webdesign\s*$/i.test(text)) {
      setWebDesign((v) => !v);
      setInput("");
      return;
    }
    if (!model) {
      await handleLoadFolder();
      return;
    }

    const freshConv = conversationId === null;
    const convId = conversationId ?? uid();
    // A vision attachment rides on this very message (pixels, not OCR text)
    // and is consumed by the send; document attachments stay pinned.
    const visionImgs =
      attachment?.kind === "vision" && attachment.path
        ? [attachment.path]
        : attachment?.images?.length
          ? attachment.images
          : undefined;
    const userMsg: UiMessage = { id: uid(), role: "user", content: text, images: visionImgs };
    const asstMsg: UiMessage = { id: uid(), role: "assistant", content: "" };
    const history = [...messages, userMsg];
    setMessages([...history, asstMsg]);
    setInput("");
    if (visionImgs) setAttachment(null);

    try {
      if (freshConv) {
        setConversationId(convId);
        await saveConversation(convId, convTitle(text), model.path);
      }
      await saveMessage(userMsg.id, convId, "user", text, visionImgs);
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }

    await streamAssistant(history, asstMsg.id, convId, { freshConv });
  }

  /** Re-run the assistant turn at `index` (drops anything after it). */
  async function regenerate(index: number) {
    if (busy || !model || !conversationId) return;
    const target = messages[index];
    if (!target || target.role !== "assistant") return;
    const history = messages.slice(0, index);
    if (!history.some((m) => m.role === "user")) return;
    const newAsst: UiMessage = { id: target.id, role: "assistant", content: "" };
    setMessages([...history, newAsst]);
    try {
      await replaceMessages(
        conversationId,
        history.map((m) => ({ id: m.id, role: m.role, content: m.content, images: m.images })),
      );
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }
    await streamAssistant(history, newAsst.id, conversationId, { freshConv: false });
  }

  /** Edit a user message in place (drops anything after it) and regenerate. */
  async function editUser(index: number, newText: string) {
    const txt = newText.trim();
    setEditingId(null);
    if (busy || !model || !conversationId || !txt) return;
    const target = messages[index];
    if (!target || target.role !== "user") return;
    const editedUser: UiMessage = { id: target.id, role: "user", content: txt, images: target.images };
    const asstMsg: UiMessage = { id: uid(), role: "assistant", content: "" };
    const history = [...messages.slice(0, index), editedUser];
    setMessages([...history, asstMsg]);
    try {
      await replaceMessages(
        conversationId,
        history.map((m) => ({ id: m.id, role: m.role, content: m.content, images: m.images })),
      );
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }
    await streamAssistant(history, asstMsg.id, conversationId, { freshConv: false });
  }

  async function handleStop() {
    try {
      await cancelGeneration();
    } catch (e) {
      console.error(e);
    }
  }

  /** Stop the recorder, transcribe, then either fill the input or auto-send. */
  async function finishRecording(rec: Recorder | null, autoSend: boolean) {
    if (!rec || recorderRef.current !== rec) return; // already finished
    recorderRef.current = null;
    setRecorder(null);
    setTranscribing(true);
    try {
      const { samples, sampleRate } = await rec.stop();
      if (samples.length > sampleRate * 0.25) {
        const text = await transcribe(encodeAudio(samples), sampleRate);
        if (text) {
          if (autoSend) {
            setInput("");
            await handleSend(text);
          } else {
            setInput((prev) => (prev.trim() ? prev.trim() + " " : "") + text);
          }
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setTranscribing(false);
    }
  }

  /** Tap to start; the recording auto-ends on silence (VAD) and sends, or tap
   *  again to stop and drop the text into the input for review. */
  async function handleMic() {
    if (transcribing) return;
    if (recorder) {
      await finishRecording(recorder, false);
      return;
    }
    try {
      const rec = await startRecording({
        onAutoStop: () => void finishRecording(recorderRef.current, true),
      });
      recorderRef.current = rec;
      setRecorder(rec);
    } catch (e) {
      console.error(e);
      setAttachError("无法访问麦克风 / Microphone unavailable");
    }
  }

  /** Replay a full reply from scratch (the "reread" button). */
  async function speakText(text: string) {
    const clean = forSpeech(text);
    if (!clean) return;
    stopSpeaking();
    try {
      setSpeaking(true);
      const { audio, sampleRate } = await synthesize(clean, settings.voiceSpeed, settings.voiceSid);
      const pb = playAudio(decodeAudio(audio), sampleRate);
      playbackRef.current = pb;
      await pb.done;
    } catch (e) {
      console.error(e);
    } finally {
      if (playbackRef.current) playbackRef.current = null;
      setSpeaking(false);
    }
  }

  function stopSpeaking() {
    playbackRef.current?.stop();
    playbackRef.current = null;
    speechRef.current?.stop();
    speechRef.current = null;
    setSpeaking(false);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter") return;
    if (settings.sendKey === "modEnter") {
      // Combo sends; plain Enter falls through and inserts a newline.
      if (e.metaKey || e.ctrlKey) {
        e.preventDefault();
        handleSend();
      }
    } else if (!e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // Sidebar filter: instant title match, unioned with backend content matches.
  const q = convQuery.trim().toLowerCase();
  const visibleConvs = q
    ? conversations.filter(
        (c) => c.title.toLowerCase().includes(q) || contentMatches.has(c.id),
      )
    : conversations;

  // Command-palette actions: static commands + load-model + jump-to-conversation.
  const commands: Command[] = [
    { id: "new", label: t("newChat"), keywords: "new chat 新对话", run: handleNewChat },
    {
      id: "mode",
      label: appMode === "code" ? t("cmdkGoChat") : t("cmdkGoCode"),
      keywords: "code chat mode agent 切换 模式 编码",
      run: () => setAppMode((m) => (m === "code" ? "chat" : "code")),
    },
    {
      id: "settings",
      label: t("settingsTitle"),
      keywords: "settings 设置 偏好",
      run: () => setShowSettings(true),
    },
    {
      id: "download",
      label: t("dlTitle"),
      keywords: "download model 下载 模型",
      run: () => setShowDownload(true),
    },
    { id: "live", label: t("cmdkLive"), keywords: "voice live 语音", run: () => setShowLive(true) },
    {
      id: "kb",
      label: ragEnabled ? t("cmdkKbOff") : t("cmdkKbOn"),
      keywords: "knowledge base rag 知识库",
      run: () => setRagEnabled((v) => !v),
    },
    {
      id: "web",
      label: webEnabled ? t("cmdkWebOff") : t("cmdkWebOn"),
      keywords: "web search 联网 搜索",
      run: () => setWebEnabled((v) => !v),
    },
    {
      id: "models-dir",
      label: t("openModelsDir"),
      keywords: "models folder 模型 文件夹",
      run: () => openModelsFolder(),
    },
    {
      id: "data-dir",
      label: t("openDataDir"),
      keywords: "data folder backup 数据 备份",
      run: () => void openDataDir().catch(console.error),
    },
    ...(model
      ? [
          {
            id: "eject",
            label: t("ejectModel"),
            keywords: "eject unload 卸载",
            run: () => void handleEject(),
          },
        ]
      : []),
    ...availableModels
      .filter((m) => m.path !== model?.path)
      .map((m) => ({
        id: `model:${m.path}`,
        label: t("cmdkLoadModel", { name: m.name }),
        hint: m.sizeMb ? fmtGbFromMb(m.sizeMb) : undefined,
        keywords: `model 模型 ${m.name}`,
        run: () => void switchModel(m.path),
      })),
    ...conversations.map((c) => ({
      id: `conv:${c.id}`,
      label: c.title,
      hint: t("cmdkChatHint"),
      keywords: `chat conversation 对话 ${c.title}`,
      run: () => void openConversation(c.id),
    })),
  ];

  return (
    <CanvasOpenContext.Provider value={openInCanvas}>
    <CodeCollapseContext.Provider value={settings.chatCollapseCode}>
    <div className="app">
      <CommandPalette open={showCmdk} onClose={() => setShowCmdk(false)} commands={commands} />
      <CanvasPanel
        open={canvasOpen}
        versions={canvasVersions}
        index={canvasIndex}
        busy={canvasBusy}
        streamText={canvasStream}
        onSelectVersion={setCanvasIndex}
        onIterate={(instr) => void generateCanvasVersion("edit", instr)}
        onReset={() => {
          setCanvasVersions((vs) => (vs.length ? [vs[0]] : vs));
          setCanvasIndex(0);
        }}
        onManualEdit={(html) => {
          const at = canvasVersions.length;
          setCanvasVersions((vs) => [...vs, { html, note: t("canvasManualNote") }]);
          setCanvasIndex(at);
        }}
        onFix={(err) => void generateCanvasVersion("fix", err)}
        onStop={() => {
          canvasCancelRef.current = true;
          void cancelGeneration().catch(console.error);
        }}
        onExport={(html) => void exportHtmlFile("design.html", html).catch(console.error)}
        onOpenExternal={(html) => void openHtmlReport(html, "canvas").catch(console.error)}
        onClose={() => setCanvasOpen(false)}
      />
      {dragging && (
        <div className="drop-overlay">
          <div className="drop-card">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
              <path d="M12 16V4M12 4l-4 4M12 4l4 4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" />
            </svg>
            <span>{t("dropToAttach")}</span>
          </div>
        </div>
      )}
      <header className="titlebar" data-tauri-drag-region>
        <div className="brand">
          <span className="brand-dot" />
          Chaty
        </div>

        <div className="mode-switch" role="tablist" aria-label="Mode">
          <button
            className={`mode-tab ${appMode === "chat" ? "active" : ""}`}
            onClick={() => setAppMode("chat")}
          >
            {t("modeChat")}
          </button>
          <button
            className={`mode-tab ${appMode === "code" ? "active" : ""}`}
            onClick={() => setAppMode("code")}
          >
            {t("modeCode")}
          </button>
        </div>

        <div className="model-wrap">
          <button
            className={`model-chip ${model ? "loaded" : ""}`}
            onClick={() => {
              if (!showModelMenu) void refreshModels();
              setShowModelMenu((v) => !v);
            }}
            title={t("changeModel")}
          >
            {model ? (
              <>
                <span className="chip-name">{model.name.replace(/\.gguf$/i, "")}</span>
                {model.paramsB ? (
                  <span className="chip-meta">{model.paramsB.toFixed(1)}B</span>
                ) : null}
                {model.sizeMb ? (
                  <span className="chip-meta">{fmtGbFromMb(model.sizeMb)}</span>
                ) : null}
              </>
            ) : loadingModel ? (
              loadProgress?.phase === "weights"
                ? `${t("loadingModel")} ${Math.round(loadProgress.frac * 100)}%`
                : loadProgress?.phase === "eject"
                  ? t("ejectingModel")
                  : t("loadingModel")
            ) : (
              t("noModel")
            )}
            <span className={`chip-caret ${showModelMenu ? "open" : ""}`}><Icon name="chevron-down" size={11} strokeWidth={2} /></span>
          </button>
          {showModelMenu && (
            <div className="model-menu">
              <div className="model-menu-head">
                <span>{t("modelsHeader")}</span>
                <button
                  className="model-menu-refresh"
                  onClick={() => void refreshModels()}
                  title={t("refreshModels")}
                >
                  ⟳
                </button>
              </div>
              <div className="model-menu-list">
                {availableModels.length === 0 ? (
                  <div className="model-menu-empty">{t("noModelsFound")}</div>
                ) : (
                  availableModels.map((m) => {
                    const active = model?.path === m.path;
                    return (
                      <div
                        key={m.path}
                        className={`model-menu-item ${active ? "active" : ""}`}
                      >
                        <button
                          className="mm-pick"
                          onClick={() => switchModel(m.path)}
                          disabled={busy}
                          title={m.path}
                        >
                          <span className="mm-name">{m.name}</span>
                          {m.format === "mlx" ? (
                            <span className="mm-fmt" title={t("mlxBadgeTip")}>
                              MLX
                            </span>
                          ) : null}
                          {m.mmproj || m.vision ? (
                            <span className="mm-vision" title={t("visionBadgeTip")}>
                              {t("visionBadge")}
                            </span>
                          ) : null}
                          {m.sizeMb ? (
                            <span className="mm-size">{fmtGbFromMb(m.sizeMb)}</span>
                          ) : null}
                        </button>
                        <span className="mm-trail">
                          {active ? (
                            <span className="mm-dot" />
                          ) : (
                            <button
                              className="mm-del"
                              onClick={() => void handleDeleteModel(m)}
                              disabled={busy}
                              title={t("deleteModelFile")}
                              aria-label={t("deleteModelFile")}
                            >
                              <Icon name="x" size={11} strokeWidth={2.2} />
                            </button>
                          )}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
              <button className="model-menu-file" onClick={handleLoadFolder}>
                {t("loadFromFolder")}
              </button>
              <button
                className="model-menu-file"
                onClick={() => {
                  setShowModelMenu(false);
                  setShowDownload(true);
                }}
              >
                {t("dlTitle")}
              </button>
              <button
                className="model-menu-file"
                onClick={() => {
                  setShowModelMenu(false);
                  openModelsFolder();
                }}
              >
                {t("openModelsDir")}
              </button>
              {model && (
                <button
                  className="model-menu-file model-menu-eject"
                  onClick={handleEject}
                  disabled={busy || loadingModel}
                  title={t("ejectModel")}
                >
                  {t("ejectModel")}
                </button>
              )}
            </div>
          )}
        </div>

        <div className="settings-wrap">
          <button
            className={`icon-btn ${showModelInfo ? "active" : ""}`}
            onClick={() => setShowModelInfo((v) => !v)}
            title={t("miTitleBtn")}
            disabled={!model}
          >
            <svg
              width="17"
              height="17"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M12 11v5" strokeLinecap="round" />
              <circle cx="12" cy="7.6" r="0.7" fill="currentColor" stroke="none" />
            </svg>
          </button>
          {showModelInfo && (
            <ModelInfoPanel model={model} onClose={() => setShowModelInfo(false)} />
          )}
        </div>

        {messages.length > 0 && (
          <div className="settings-wrap">
            <button
              className={`icon-btn ${showExport ? "active" : ""}`}
              onClick={() => setShowExport((v) => !v)}
              title={t("exportTitle")}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                <path d="M12 3v12M12 15l-4-4M12 15l4-4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" />
              </svg>
            </button>
            {showExport && (
              <>
                <div className="popover-backdrop" onClick={() => setShowExport(false)} />
                <div className="export-menu">
                  <button onClick={() => exportConversation("md")}>{t("exportMd")}</button>
                  <button onClick={() => exportConversation("json")}>{t("exportJson")}</button>
                </div>
              </>
            )}
          </div>
        )}

        <div className="settings-wrap">
          <button
            className={`icon-btn ${showHardware ? "active" : ""}`}
            onClick={() => setShowHardware((v) => !v)}
            title={t("hwTitleBtn")}
          >
            <svg
              width="17"
              height="17"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <rect x="7" y="7" width="10" height="10" rx="1.5" />
              <path
                d="M10 4v2M14 4v2M10 18v2M14 18v2M4 10h2M4 14h2M18 10h2M18 14h2"
                strokeLinecap="round"
              />
            </svg>
          </button>
          {showHardware && (
            <HardwarePanel model={model} onClose={() => setShowHardware(false)} />
          )}
        </div>

        <div className="settings-wrap">
          <button
            className={`icon-btn ${showSettings ? "active" : ""}`}
            onClick={() => setShowSettings((v) => !v)}
            title={t("settingsTitle")}
          >
            <svg
              width="17"
              height="17"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </button>
          <SettingsPanel
            open={showSettings}
            value={settings}
            onChange={setSettings}
            onClose={() => setShowSettings(false)}
            maxTokensLimit={Math.max(1024, model?.nCtx ?? 4096)}
            ctxTrainLimit={model?.nCtxTrain}
            onReloadModel={model ? () => void reloadModel() : undefined}
            reloading={loadingModel}
            onDataCleared={() => {
              handleNewChat();
              void refreshConversations();
            }}
          />
        </div>

        {/* macOS uses native traffic lights (titleBarStyle: Overlay); our
            custom controls are only for Windows/Linux. */}
        {!IS_MACOS && <WindowControls />}
      </header>
      {loadingModel && (
        <div className="load-bar">
          <div
            className={`load-bar-fill ${loadProgress?.phase === "weights" ? "" : "indeterminate"}`}
            style={
              loadProgress?.phase === "weights"
                ? { width: `${Math.max(2, Math.round(loadProgress.frac * 100))}%` }
                : undefined
            }
          />
        </div>
      )}

      <CodeMode
        model={model}
        active={appMode === "code"}
        maxSteps={settings.codeMaxSteps}
        bashTimeout={settings.codeBashTimeout}
        temperature={settings.codeTemperature}
        thinkBudget={settings.codeThinkBudget}
        maxGenTokens={settings.codeMaxTokens}
        autoApproveEdits={settings.codeAutoApproveEdits}
        autoRunReadOnly={settings.codeAutoRunReadOnly}
        skills={settings.codeSkills}
        disabledSkills={settings.codeDisabledSkills}
        memoryEnabled={settings.codeMemory}
        allowedCommands={settings.codeAllowedCommands}
        sendKey={settings.sendKey}
        autoTitle={settings.autoTitle}
      />

      <div className="body" style={appMode === "code" ? { display: "none" } : undefined}>
        <aside className="sidebar" ref={asideRef} style={{ width: sidebarW }}>
          <button className="new-chat" onClick={handleNewChat} disabled={busy}>
            <Icon name="plus" size={13} strokeWidth={2} /> {t("newChat")}
          </button>
          {conversations.length > 0 && (
            <div className="conv-search">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <circle cx="11" cy="11" r="7" />
                <path d="M21 21l-4.3-4.3" strokeLinecap="round" />
              </svg>
              <input
                type="text"
                placeholder={t("searchConv")}
                value={convQuery}
                onChange={(e) => setConvQuery(e.target.value)}
              />
              {convQuery && (
                <button className="conv-search-clear" onClick={() => setConvQuery("")} title={t("cancel")}>
                  <Icon name="x" size={11} strokeWidth={2.2} />
                </button>
              )}
            </div>
          )}
          <div className="conv-list">
            {visibleConvs.length === 0 ? (
              <div className="conv-empty">
                {conversations.length === 0 ? t("noConversations") : t("noMatches")}
              </div>
            ) : (
              visibleConvs.map((c) => (
                <div
                  key={c.id}
                  className={`conv-item ${c.id === conversationId ? "active" : ""} ${
                    c.pinned ? "pinned" : ""
                  }`}
                  onClick={() => renamingId !== c.id && openConversation(c.id)}
                >
                  {renamingId === c.id ? (
                    <input
                      className="conv-rename"
                      autoFocus
                      value={renameDraft}
                      onChange={(e) => setRenameDraft(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          void commitRename();
                        } else if (e.key === "Escape") {
                          e.preventDefault();
                          setRenamingId(null);
                        }
                      }}
                      onBlur={() => void commitRename()}
                    />
                  ) : (
                    <>
                      <span className="conv-title">{c.title}</span>
                      <div className="conv-actions">
                        <button
                          className={`conv-act ${c.pinned ? "on" : ""}`}
                          title={c.pinned ? t("unpinConv") : t("pinConv")}
                          onClick={(e) => {
                            e.stopPropagation();
                            void handleTogglePin(c);
                          }}
                        >
                          {c.pinned ? <IconPinFilled size={13} /> : <IconPin size={13} />}
                        </button>
                        <button
                          className="conv-act"
                          title={t("renameConv")}
                          onClick={(e) => {
                            e.stopPropagation();
                            startRename(c);
                          }}
                        >
                          <IconEdit size={13} />
                        </button>
                        <button
                          className="conv-del"
                          title={t("deleteConv")}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(c.id);
                          }}
                        >
                          <Icon name="x" size={11} strokeWidth={2.2} />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
          </div>
          <div className="side-status" title={model ? model.name : ""}>
            <span className="ss-dot" />
            <span className="ss-meta">v{__APP_VERSION__}</span>
          </div>
          <div
            className="sidebar-resizer"
            role="separator"
            aria-orientation="vertical"
            title={t("resizeSidebar")}
            onPointerDown={startSidebarResize}
            onDoubleClick={resetSidebarW}
          />
        </aside>

        <div className="main">
          {showDeepResearch && (
            <DeepResearchPanel
              model={model}
              onClose={() => setShowDeepResearch(false)}
              onLockChange={setBusy}
            />
          )}
          {showKbReport && (
            <DeepResearchPanel
              mode="kb"
              model={model}
              onClose={() => setShowKbReport(false)}
              onLockChange={setBusy}
            />
          )}
          <div className="chat-wrap">
          <main className="chat" ref={scrollRef}>
            {messages.length === 0 ? (
              <div className="empty">
                <div className="empty-hero">
                  <div className="empty-greeting">{t(greetingKey())}</div>
                  <div className="empty-sub">
                    {model ? t("readyMsg") : t("loadToStart")}
                  </div>
                </div>
                {!model && (
                  <button className="setup-cta" onClick={() => setShowSetup(true)}>
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                      aria-hidden="true"
                      style={{ verticalAlign: "-2px", marginRight: "7px" }}
                    >
                      <path d="M12 2.5l2.2 6.3L20.5 11l-6.3 2.2L12 19.5l-2.2-6.3L3.5 11l6.3-2.2z" />
                    </svg>
                    {t("setupBtn")}
                  </button>
                )}
                {model && (
                  <div className="suggestions">
                    {(lang === "zh" ? SUGGESTIONS_ZH : SUGGESTIONS_EN).map((s) => (
                      <button key={s} className="suggestion" onClick={() => setInput(s)}>
                        {s}
                      </button>
                    ))}
                  </div>
                )}
                <button
                  className="cmdk-hint-chip"
                  onClick={() => setShowCmdk(true)}
                  title={t("cmdkHint")}
                >
                  <kbd>{IS_MACOS ? "⌘" : "Ctrl"}</kbd>
                  <kbd>K</kbd>
                  <span>{t("cmdkHint")}</span>
                </button>
              </div>
            ) : (
              messages.map((m, i) =>
                m.role === "assistant" ? (
                  <div key={m.id} className="msg assistant">
                    <AssistantMessage
                      content={m.content}
                      streaming={streamingId === m.id}
                      searching={streamingId === m.id ? searching : ""}
                      composing={streamingId === m.id && composing}
                      hideThinking={!thinkEnabled}
                      sources={m.sources}
                    />
                    {m.sources && m.sources.length > 0 && (
                      <div className="sources">
                        <span className="sources-label">{t("sources")}</span>
                        <div className="sources-list">
                          {m.sources.map((s, k) => (
                            <button
                              key={k}
                              className="source-chip"
                              title={s.url || undefined}
                              onClick={() => {
                                if (s.url) void openExternal(s.url).catch(() => {});
                              }}
                            >
                              <span className="source-idx">{k + 1}</span>
                              <span className="source-title">{s.title}</span>
                              {s.snippet && (
                                <span className="cite-pop">
                                  <span className="cite-pop-title">{s.title}</span>
                                  <span className="cite-pop-text">{s.snippet}</span>
                                </span>
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {streamingId !== m.id && m.content.trim() && (
                      <div className="msg-actions">
                        <button
                          className="msg-action"
                          title={t("copyTitle")}
                          onClick={() => copyText(stripThink(m.content))}
                        >
                          {t("copy")}
                        </button>
                        <button
                          className="msg-action"
                          title={t("regenTitle")}
                          onClick={() => regenerate(i)}
                          disabled={busy}
                        >
                          {t("regenerate")}
                        </button>
                        <button
                          className="msg-action"
                          title={t("forkTitle")}
                          onClick={() => handleFork(i)}
                        >
                          {t("fork")}
                        </button>
                        {lang === "en" && (
                          <button
                            className="msg-action"
                            title={t("rereadTitle")}
                            onClick={() => (speaking ? stopSpeaking() : speakText(m.content))}
                          >
                            {speaking ? t("rereadStop") : t("reread")}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div key={m.id} className="msg user">
                    {editingId === m.id ? (
                      <div className="edit-box">
                        <textarea
                          value={editingText}
                          autoFocus
                          rows={Math.min(8, editingText.split("\n").length + 1)}
                          onChange={(e) => setEditingText(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              editUser(i, editingText);
                            } else if (e.key === "Escape") {
                              setEditingId(null);
                            }
                          }}
                        />
                        <div className="edit-actions">
                          <button className="msg-action" onClick={() => setEditingId(null)}>
                            {t("cancel")}
                          </button>
                          <button
                            className="msg-action primary"
                            onClick={() => editUser(i, editingText)}
                            disabled={busy || !editingText.trim()}
                          >
                            {t("saveEdit")}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="bubble">
                        {m.images && m.images.length > 0 && (
                          <span className="msg-images">
                            {m.images.map((p) => (
                              <ImageThumb key={p} path={p} />
                            ))}
                          </span>
                        )}
                        <UserText content={m.content} expandLabel={t("expandAll")} collapseLabel={t("collapseText")} />
                        <UserCopy content={m.content} title={t("copyMsg")} />
                        {!busy && (
                          <button
                            className="user-edit"
                            title={t("editMsg")}
                            onClick={() => {
                              setEditingText(m.content);
                              setEditingId(m.id);
                            }}
                          >
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                              <path d="M14.5 5.5l4 4M4 20l1-4L16 5a2 2 0 0 1 3 3L8 19l-4 1z" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                ),
              )
            )}
          </main>
          {showJump && messages.length > 0 && (
            <button
              className="jump-bottom"
              title={t("jumpLatest")}
              onClick={() => {
                followRef.current = true;
                scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
              }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
          </div>

          <footer className="composer">
            {stats && (
              <div className="stats">
                {stats.completionTokens} tokens · {stats.tokensPerSecond.toFixed(1)} tok/s
                {stats.stopReason ? (
                  <span
                    className={`stop-reason ${
                      stats.stopReason === "context" || stats.stopReason === "length"
                        ? "warn"
                        : ""
                    }`}
                  >
                    {" · "}
                    {stats.stopReason === "eos"
                      ? t("stopEos")
                      : stats.stopReason === "length"
                        ? t("stopLength")
                        : stats.stopReason === "context"
                          ? t("stopContext")
                          : stats.stopReason === "stop"
                            ? t("stopStop")
                            : stats.stopReason === "cancelled"
                              ? t("stopCancelled")
                              : stats.stopReason}
                  </span>
                ) : null}
                {model?.nCtx
                  ? (() => {
                      const used = stats.promptTokens + stats.completionTokens;
                      const pct = Math.min(1, used / model.nCtx);
                      const C = 2 * Math.PI * 7; // ring circumference (r=7)
                      return (
                        <span className="ctx-meter" title={t("ctxUsage")}>
                          {" · ctx "}
                          {fmtK(used)}/{fmtK(model.nCtx)}
                          <svg
                            className={`ctx-ring ${pct > 0.95 ? "danger" : pct > 0.8 ? "warn" : ""}`}
                            viewBox="0 0 18 18"
                            aria-hidden="true"
                          >
                            <circle className="ctx-ring-track" cx="9" cy="9" r="7" />
                            <circle
                              className="ctx-ring-fill"
                              cx="9"
                              cy="9"
                              r="7"
                              strokeDasharray={C}
                              strokeDashoffset={C * (1 - pct)}
                            />
                          </svg>
                        </span>
                      );
                    })()
                  : null}
              </div>
            )}
            {(attachment || attachError) && (
              <div className="attach-bar">
                {attachment && (
                  <div className="attach-chip">
                    {attachment.kind === "vision" && attachment.path && (
                      <ImageThumb path={attachment.path} size={26} />
                    )}
                    <span className="attach-name">
                      <svg
                        width="13"
                        height="13"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                        style={{ verticalAlign: "-2px", marginRight: "5px" }}
                      >
                        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
                        <path d="M14 3v5h5" />
                      </svg>
                      {attachment.name}
                    </span>
                    <span className="attach-meta">
                      {attachment.kind === "vision"
                        ? t("visionAttach")
                        : t("charsLabel", { n: attachment.chars }) +
                          (attachment.truncated ? t("truncatedSuffix") : "")}
                    </span>
                    <button
                      className="attach-remove"
                      title={t("removeAttach")}
                      onClick={() => setAttachment(null)}
                    >
                      <Icon name="x" size={11} strokeWidth={2.2} />
                    </button>
                  </div>
                )}
                {attachError && <span className="attach-error">{attachError}</span>}
              </div>
            )}
            {webDesign && (
              <div className="mode-bar">
                <span className="mode-chip">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <rect x="3" y="4.5" width="18" height="15" rx="2" />
                    <path d="M3 9h18" strokeLinecap="round" />
                  </svg>
                  {t("webDesignChip")}
                  <button
                    className="mode-chip-x"
                    onClick={() => setWebDesign(false)}
                    title={t("webDesignOff")}
                  >
                    <Icon name="x" size={11} strokeWidth={2.2} />
                  </button>
                </span>
              </div>
            )}
            <div className="input-row">
              <div className="tools-wrap">
                <button
                  className={`tool-toggle ${showToolsMenu ? "active" : ""}`}
                  title={t("toolsMenu")}
                  onClick={() => setShowToolsMenu((v) => !v)}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    aria-hidden="true"
                  >
                    <path d="M12 5v14M5 12h14" strokeLinecap="round" />
                  </svg>
                </button>
                {showToolsMenu && (
                  <div className="tools-menu">
                    <button
                      className="tool-item"
                      onClick={() => {
                        setShowToolsMenu(false);
                        handleAttach();
                      }}
                      disabled={attaching}
                    >
                      <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                        <path d="M21.5 11.5l-9 9a5.5 5.5 0 0 1-7.8-7.8l9-9a3.6 3.6 0 1 1 5.1 5.1l-9 9a1.8 1.8 0 0 1-2.5-2.5l8.3-8.3" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      <span className="ti-label">{t("toolAttach")}</span>
                    </button>
                    {/* Knowledge base group — hover reveals retrieve / manage */}
                    <div className="tool-group">
                      <div className={`tool-item tool-parent ${ragEnabled ? "on" : ""}`}>
                        <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" strokeLinecap="round" />
                          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" strokeLinejoin="round" />
                        </svg>
                        <span className="ti-label">{t("toolKbGroup")}</span>
                        <span className="ti-check">{ragEnabled ? <Icon name="check" size={12} strokeWidth={2.4} /> : ""}</span>
                        <span className="ti-caret">›</span>
                      </div>
                      <div className="tool-submenu">
                        <button
                          className={`tool-item ${ragEnabled ? "on" : ""}`}
                          onClick={() => {
                            const next = !ragEnabled;
                            if (next) {
                              ragStatus()
                                .then((st) => {
                                  if (!st.modelReady || st.docs === 0) {
                                    setShowToolsMenu(false);
                                    setShowKb(true);
                                  } else {
                                    setRagEnabled(true);
                                  }
                                })
                                .catch(() => setShowKb(true));
                            } else {
                              setRagEnabled(false);
                            }
                          }}
                        >
                          <span className="ti-label">{t("toolKb")}</span>
                          <span className="ti-check">{ragEnabled ? <Icon name="check" size={12} strokeWidth={2.4} /> : ""}</span>
                        </button>
                        <button
                          className="tool-item"
                          onClick={() => {
                            setShowToolsMenu(false);
                            setShowKb(true);
                          }}
                        >
                          <span className="ti-label">{t("toolKbManage")}</span>
                        </button>
                      </div>
                    </div>
                    {/* Web group — hover reveals Deep Research / web search */}
                    <div className="tool-group">
                      <div className={`tool-item tool-parent ${webEnabled ? "on" : ""}`}>
                        <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                          <circle cx="12" cy="12" r="9" />
                          <path d="M3 12h18" />
                          <path d="M12 3c2.6 2.7 2.6 15.3 0 18M12 3c-2.6 2.7-2.6 15.3 0 18" />
                        </svg>
                        <span className="ti-label">{t("toolWebGroup")}</span>
                        <span className="ti-check">{webEnabled ? <Icon name="check" size={12} strokeWidth={2.4} /> : ""}</span>
                        <span className="ti-caret">›</span>
                      </div>
                      <div className="tool-submenu">
                        <button
                          className="tool-item"
                          onClick={() => {
                            setShowToolsMenu(false);
                            setShowDeepResearch(true);
                          }}
                        >
                          <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                            <circle cx="11" cy="11" r="7" />
                            <path d="M21 21l-4.3-4.3" strokeLinecap="round" />
                          </svg>
                          <span className="ti-label">{t("toolDeepResearch")}</span>
                        </button>
                        <button
                          className={`tool-item ${webEnabled ? "on" : ""}`}
                          onClick={() => {
                            const next = !webEnabled;
                            setWebEnabled(next);
                            if (next) setThinkEnabled(false); // web search ⇄ thinking are exclusive
                          }}
                        >
                          <span className="ti-label">{t("toolWeb")}</span>
                          <span className="ti-check">{webEnabled ? <Icon name="check" size={12} strokeWidth={2.4} /> : ""}</span>
                        </button>
                      </div>
                    </div>
                    <button
                      className={`tool-item ${thinkEnabled && model?.supportsThinking ? "on" : ""}`}
                      onClick={() => {
                        const next = !thinkEnabled;
                        setThinkEnabled(next);
                        if (next) setWebEnabled(false); // thinking ⇄ web search are exclusive
                      }}
                      disabled={!model?.supportsThinking}
                      title={model && !model.supportsThinking ? t("thinkUnsupported") : undefined}
                    >
                      <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                        <path d="M9.5 18h5M10.5 21h3" strokeLinecap="round" />
                        <path d="M12 3a6 6 0 0 0-3.5 10.9c.6.4 1 1.1 1 1.8v.3h5v-.3c0-.7.4-1.4 1-1.8A6 6 0 0 0 12 3z" strokeLinejoin="round" />
                      </svg>
                      <span className="ti-label">{t("toolThink")}</span>
                      <span className="ti-check">
                        {thinkEnabled && model?.supportsThinking ? <Icon name="check" size={12} strokeWidth={2.4} /> : ""}
                      </span>
                    </button>
                    <button
                      className={`tool-item ${webDesign ? "on" : ""}`}
                      onClick={() => setWebDesign((v) => !v)}
                    >
                      <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                        <rect x="3" y="4.5" width="18" height="15" rx="2" />
                        <path d="M3 9h18" strokeLinecap="round" />
                        <path d="M6 6.7h.01M8.4 6.7h.01" strokeLinecap="round" />
                      </svg>
                      <span className="ti-label">{t("toolDesign")}</span>
                      <span className="ti-check">{webDesign ? <Icon name="check" size={12} strokeWidth={2.4} /> : ""}</span>
                    </button>
                    {lang === "en" && (
                      <>
                        <div className="tools-sep" />
                        <button
                          className={`tool-item ${speakReplies ? "on" : ""}`}
                          onClick={() => {
                            if (speaking) stopSpeaking();
                            else if (speakReplies) stopSpeaking();
                            setSpeakReplies((v) => !v);
                          }}
                        >
                          <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                            <path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5H4z" strokeLinejoin="round" />
                            <path d="M15.5 8.5a4.2 4.2 0 0 1 0 7" strokeLinecap="round" />
                          </svg>
                          <span className="ti-label">{t("speakAloud")}</span>
                          <span className="ti-check">{speakReplies ? <Icon name="check" size={12} strokeWidth={2.4} /> : ""}</span>
                        </button>
                        <button
                          className="tool-item"
                          onClick={openLive}
                          disabled={!model}
                        >
                          <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                            <circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" />
                            <path d="M7.5 7.5a6 6 0 0 0 0 9M16.5 7.5a6 6 0 0 1 0 9" strokeLinecap="round" />
                          </svg>
                          <span className="ti-label">{t("liveStart")}</span>
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={
                  !model
                    ? t("inputPhNoModel")
                    : webDesign
                      ? t("inputPhDesign")
                      : webEnabled
                        ? t("inputPhWeb")
                        : settings.sendKey === "modEnter"
                          ? t("inputPhMod")
                          : t("inputPh")
                }
                rows={1}
              />
              {lang === "en" && (
                <button
                  className={`mic-btn ${recorder ? "recording" : ""}`}
                  title={recorder ? t("micStop") : t("micStart")}
                  onClick={handleMic}
                  disabled={transcribing}
                >
                  {transcribing ? (
                    <span className="mini-spinner" />
                  ) : recorder ? (
                    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
                      <rect x="5" y="5" width="14" height="14" rx="3" fill="currentColor" />
                    </svg>
                  ) : (
                    <svg
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      aria-hidden="true"
                    >
                      <rect x="9" y="3" width="6" height="11" rx="3" />
                      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" strokeLinecap="round" />
                    </svg>
                  )}
                </button>
              )}
              {busy ? (
                <button className="send-btn stop" onClick={handleStop} title={t("stopTitle")}>
                  <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
                    <rect x="6" y="6" width="12" height="12" rx="3" fill="currentColor" />
                  </svg>
                </button>
              ) : (
                <button
                  className="send-btn"
                  onClick={() => handleSend()}
                  disabled={!input.trim()}
                  title={t("sendTitle")}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
                    <path
                      d="M11 20a1 1 0 0 0 2 0V8.4l4.3 4.3a1 1 0 0 0 1.4-1.4l-6-6a1 1 0 0 0-1.4 0l-6 6a1 1 0 0 0 1.4 1.4L11 8.4z"
                      fill="currentColor"
                    />
                  </svg>
                </button>
              )}
            </div>
          </footer>
        </div>
      </div>
      <ContextMenu />
      {update?.available && (
        <div className="update-banner">
          <span className="update-text">{t("updateAvailable", { v: update.latest })}</span>
          <button className="update-btn primary" onClick={applyUpdate} disabled={updating}>
            {updating ? t("updateDownloading") : t("updateNow")}
          </button>
          <button
            className="update-btn"
            onClick={() => setUpdate(null)}
            disabled={updating}
          >
            {t("updateLater")}
          </button>
        </div>
      )}
      {notice && (
        <div
          className={`toast toast-${notice.kind}`}
          onClick={() => setNotice(null)}
          title={t("toastDismiss")}
        >
          {notice.text}
        </div>
      )}
      {showLive && (
        <LiveMode
          onClose={() => setShowLive(false)}
          preamble={[
            t("todayNote", { date: formatDate(lang) }),
            settings.systemPrompt.trim(),
            "You are a voice assistant in a live, spoken conversation. Talk the way people actually speak out loud: relaxed, warm, and flowing, using contractions. Keep replies short — usually one to three sentences. Absolutely NO bullet points, numbered lists, headings, markdown, code, or emoji. If you would normally list things, weave them into a natural sentence instead (say \"you could try A, B, or C\" rather than listing them). Just talk naturally, like a friend on the phone. Never end with offers to help more such as \"let me know if you want more\", \"feel free to ask\", or \"hope this helps\" — just answer and stop.",
          ]
            .filter(Boolean)
            .join("\n\n")}
          initialHistory={messages
            .filter((m) => m.content.trim())
            .map((m) => ({
              role: m.role,
              content: m.role === "assistant" ? stripThink(m.content) : m.content,
            }))}
          onTurn={recordLiveTurn}
          appendNoThink={model?.thinkSwitch ?? false}
          forceNoThink={(model?.supportsThinking && !model.thinkSwitch) ?? false}
          voiceSid={settings.voiceSid}
          voiceSpeed={settings.voiceSpeed}
        />
      )}
      {showDownload && (
        <DownloadModal
          onClose={() => {
            setShowDownload(false);
            setDeepLink(null);
          }}
          onDownloaded={refreshModels}
          initialRepo={deepLink?.repo}
          initialFile={deepLink?.file}
        />
      )}
      {showKb && (
        <KnowledgePanel
          onClose={() => setShowKb(false)}
          onPodcast={() => {
            setShowKb(false);
            setShowPodcast(true);
          }}
          onReport={() => {
            setShowKb(false);
            setShowKbReport(true);
          }}
        />
      )}
      {showPodcast && (
        <PodcastPanel
          model={model}
          voiceSpeed={settings.voiceSpeed}
          onClose={() => setShowPodcast(false)}
          onLockChange={setBusy}
        />
      )}
      {showSetup && (
        <SetupModal
          onClose={() => setShowSetup(false)}
          onOpenStore={() => {
            setShowSetup(false);
            setShowDownload(true);
          }}
          onLoad={(path) => {
            setShowSetup(false);
            void refreshModels();
            void switchModel(path);
          }}
        />
      )}
    </div>
    </CodeCollapseContext.Provider>
    </CanvasOpenContext.Provider>
  );
}
