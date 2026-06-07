import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { AssistantMessage } from "./components/AssistantMessage";
import { ContextMenu } from "./components/ContextMenu";
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
import { SettingsPanel, type GenSettings, defaultSettings } from "./components/SettingsPanel";
import { answerOnly, cutSentences, forSpeech, stripThink } from "./lib/voiceText";
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
  pickAttachmentFile,
  pickModelFile,
  readAttachment,
  renameConversation,
  replaceMessages,
  runUpdate,
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
  type ModelEntry,
  type ModelInfo,
  type SearchResult,
  type StreamEvent,
  type UpdateInfo,
} from "./lib/ipc";
import "./App.css";

interface UiMessage extends ChatMessage {
  id: string;
  sources?: SearchResult[];
}

const uid = () => Math.random().toString(36).slice(2);
const fmtK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n));
const SETTINGS_KEY = "chaty.settings";
const LAST_MODEL_KEY = "chaty.lastModel";
const convTitle = (t: string) => t.replace(/\s+/g, " ").trim().slice(0, 40) || "新对话";

const copyText = (t: string) => {
  navigator.clipboard?.writeText(t).catch(() => {});
};

/** Strip a model's reasoning/quotes from a generated title and clamp length. */
function cleanTitle(raw: string): string {
  let t = raw.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, "");
  const firstLine = t.split("\n").map((s) => s.trim()).find(Boolean) ?? "";
  return firstLine
    .replace(/^["'「『《<[(]+|["'」』》>\])。.!！?？:：]+$/g, "")
    .trim()
    .slice(0, 24);
}

/** Clean a model-generated search query (strip reasoning/quotes, keep it short). */
function cleanQuery(raw: string): string {
  const t = raw.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, "");
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
  const [settings, setSettings] = useState<GenSettings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [showHardware, setShowHardware] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [notice, setNotice] = useState<{ kind: "warn" | "error"; text: string } | null>(null);
  const noticeTimer = useRef<number | null>(null);
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [updating, setUpdating] = useState(false);
  const [webEnabled, setWebEnabled] = useState(false);
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
  const [searching, setSearching] = useState(false);
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState("");
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

  useEffect(() => {
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
        const last = localStorage.getItem(LAST_MODEL_KEY);
        const target = last ?? models[0]?.path ?? null;
        if (!target) return;
        setLoadingModel(true);
        try {
          const info = await loadModel(target, settings.gpuLayers);
          setModel(info);
          localStorage.setItem(LAST_MODEL_KEY, info.path);
          noticeForLoad(info);
        } catch (e) {
          if (last) localStorage.removeItem(LAST_MODEL_KEY); // file moved/deleted
          showLoadError(e);
        } finally {
          setLoadingModel(false);
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

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

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

  async function refreshConversations() {
    try {
      setConversations(await listConversations());
    } catch (e) {
      console.error(e);
    }
  }

  /** Ask the model for a concise title for a freshly created conversation. */
  async function makeTitle(convId: string, firstMsg: string) {
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
            { role: "user", content: `${firstMsg}\n/no_think` },
          ],
          params: { temperature: 0.2, topP: 0.9, maxTokens: 48 },
        },
        (ev) => {
          if (ev.type === "token") acc += ev.text;
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
            { role: "user", content: `${recent}\n\n最新问题：${latest}\n/no_think` },
          ],
          params: { temperature: 0.2, topP: 0.9, maxTokens: 64 },
        },
        (ev) => {
          if (ev.type === "token") acc += ev.text;
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

  /** Surface a non-fatal load warning (e.g. GPU offload reduced to fit memory). */
  function noticeForLoad(info: ModelInfo) {
    if (info.warning === "gpu-oom") {
      showNotice(
        "warn",
        info.gpuLayers > 0
          ? t("oomPartial", { a: info.gpuLayers, b: info.nLayer ?? "?" })
          : t("oomCpu"),
      );
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
      const info = await loadModel(path, settings.gpuLayers);
      setModel(info);
      localStorage.setItem(LAST_MODEL_KEY, info.path);
      noticeForLoad(info);
    } catch (e) {
      console.error(e);
      showLoadError(e);
    } finally {
      setLoadingModel(false);
    }
  }

  async function handleLoad() {
    setShowModelMenu(false);
    try {
      const path = await pickModelFile();
      if (!path) return;
      setLoadingModel(true);
      const info = await loadModel(path, settings.gpuLayers);
      setModel(info);
      localStorage.setItem(LAST_MODEL_KEY, info.path);
      noticeForLoad(info);
      void refreshModels();
    } catch (e) {
      console.error(e);
      showLoadError(e);
    } finally {
      setLoadingModel(false);
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

  async function handleAttach() {
    setAttachError("");
    try {
      const path = await pickAttachmentFile();
      if (!path) return;
      setAttaching(true);
      setAttachment(await readAttachment(path));
    } catch (e) {
      setAttachment(null);
      setAttachError(typeof e === "string" ? e : t("readAttachFailed"));
    } finally {
      setAttaching(false);
    }
  }

  async function openConversation(id: string) {
    if (busy || id === conversationId) return;
    try {
      const stored = await getMessages(id);
      setMessages(stored.map((m) => ({ id: m.id, role: m.role, content: m.content })));
      setConversationId(id);
      setStats(null);
      setAttachment(null);
      setAttachError("");
    } catch (e) {
      console.error(e);
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteConversation(id);
      if (id === conversationId) handleNewChat();
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
        await saveMessage(id, newId, m.role, m.content);
        copied.push({ id, role: m.role, content: m.content });
      }
      setConversationId(newId);
      setMessages(copied);
      setStats(null);
      await refreshConversations();
    } catch (e) {
      console.error(e);
    }
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
    if (webEnabled || urls.length > 0) {
      setSearching(true);
      try {
        const blocks: string[] = [];
        const usedSources: SearchResult[] = [];
        const label = lang === "zh" ? "资料" : "Source";

        // D — fetch any URLs the user pasted (highest priority).
        for (const url of urls) {
          try {
            const page = await fetchUrl(url);
            blocks.push(`${label}${blocks.length + 1}：${page.title}\n${page.text.slice(0, 5000)}`);
            usedSources.push({ title: page.title, url: page.url, snippet: "" });
          } catch (e) {
            console.error(e);
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
            blocks.push(`${label}${blocks.length + 1}：${p.title}\n${txt}`);
            usedSources.push({ title: p.title, url: p.url, snippet: "" });
            budget -= txt.length;
            added++;
          }
          if (added === 0 && research.results.length) {
            research.results.slice(0, 6).forEach((r) => {
              blocks.push(`${label}${blocks.length + 1}：${r.title}\n${r.snippet}`);
              usedSources.push({ title: r.title, url: r.url, snippet: "" });
            });
          }
        }

        if (usedSources.length) {
          setMessages((cur) =>
            cur.map((m) => (m.id === asstId ? { ...m, sources: usedSources } : m)),
          );
        }
        if (blocks.length) {
          webContext = t("webInstruction") + blocks.join("\n\n---\n\n");
        }
      } catch (e) {
        console.error(e);
      } finally {
        setSearching(false);
      }
    }

    // Thinking-mode control (Qwen3-style `/think` · `/no_think`): only meaningful
    // when the model actually supports reasoning — otherwise the tag is noise the
    // model may echo. Web search also forces no-think to keep answers concise.
    const historyForModel = history.map(({ role, content }) => ({ role, content }));
    if (
      historyForModel.length > 0 &&
      model?.supportsThinking &&
      (!thinkEnabled || webEnabled)
    ) {
      const last = historyForModel[historyForModel.length - 1];
      last.content = `${last.content}\n/no_think`;
    }

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
      ...(attachment
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
      ...historyForModel,
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
          const { audio, sampleRate } = await synthesize(clean);
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
    try {
      await generate(
        {
          messages: sent,
          params: {
            temperature: settings.temperature,
            topP: settings.topP,
            maxTokens: settings.maxTokens,
          },
        },
        (ev: StreamEvent) => {
          if (ev.type === "token") {
            acc.text += ev.text;
            setMessages((cur) =>
              cur.map((m) => (m.id === asstId ? { ...m, content: m.content + ev.text } : m)),
            );
            pumpSpeech(false);
          } else if (ev.type === "done") {
            setStats(ev.stats);
          } else if (ev.type === "error") {
            setMessages((cur) =>
              cur.map((m) =>
                m.id === asstId
                  ? { ...m, content: `${m.content}\n\n⚠️ ${ev.message}` }
                  : m,
              ),
            );
          }
        },
      );
    } catch (e) {
      console.error(e);
    } finally {
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
      await handleLoad();
      return;
    }

    const freshConv = conversationId === null;
    const convId = conversationId ?? uid();
    const userMsg: UiMessage = { id: uid(), role: "user", content: text };
    const asstMsg: UiMessage = { id: uid(), role: "assistant", content: "" };
    const history = [...messages, userMsg];
    setMessages([...history, asstMsg]);
    setInput("");

    try {
      if (freshConv) {
        setConversationId(convId);
        await saveConversation(convId, convTitle(text), model.path);
      }
      await saveMessage(userMsg.id, convId, "user", text);
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
        history.map((m) => ({ id: m.id, role: m.role, content: m.content })),
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
    const editedUser: UiMessage = { id: target.id, role: "user", content: txt };
    const asstMsg: UiMessage = { id: uid(), role: "assistant", content: "" };
    const history = [...messages.slice(0, index), editedUser];
    setMessages([...history, asstMsg]);
    try {
      await replaceMessages(
        conversationId,
        history.map((m) => ({ id: m.id, role: m.role, content: m.content })),
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
      const { audio, sampleRate } = await synthesize(clean);
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
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="app">
      <header className="titlebar" data-tauri-drag-region>
        <div className="brand">
          <span className="brand-dot" />
          Chaty
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
                <span className="chip-name">{model.name}</span>
                {model.paramsB ? (
                  <span className="chip-meta">{model.paramsB.toFixed(1)}B</span>
                ) : null}
                {model.sizeMb ? (
                  <span className="chip-meta">{(model.sizeMb / 1024).toFixed(1)} GB</span>
                ) : null}
                <span className="chip-backend">{model.backend}</span>
              </>
            ) : loadingModel ? (
              t("loadingModel")
            ) : (
              t("noModel")
            )}
            <span className={`chip-caret ${showModelMenu ? "open" : ""}`}>▾</span>
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
                  availableModels.map((m) => (
                    <button
                      key={m.path}
                      className={`model-menu-item ${model?.path === m.path ? "active" : ""}`}
                      onClick={() => switchModel(m.path)}
                      disabled={busy}
                      title={m.path}
                    >
                      <span className="mm-name">{m.name}</span>
                      {model?.path === m.path && <span className="mm-dot" />}
                    </button>
                  ))
                )}
              </div>
              <button className="model-menu-file" onClick={handleLoad}>
                {t("loadFromFile")}
              </button>
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
            ⚙
          </button>
          {showSettings && (
            <SettingsPanel
              value={settings}
              onChange={setSettings}
              onClose={() => setShowSettings(false)}
              maxTokensLimit={Math.max(1024, model?.nCtx ?? 4096)}
            />
          )}
        </div>

        <WindowControls />
      </header>

      <div className="body">
        <aside className="sidebar">
          <button className="new-chat" onClick={handleNewChat} disabled={busy}>
            ＋ {t("newChat")}
          </button>
          <div className="conv-list">
            {conversations.length === 0 ? (
              <div className="conv-empty">{t("noConversations")}</div>
            ) : (
              conversations.map((c) => (
                <div
                  key={c.id}
                  className={`conv-item ${c.id === conversationId ? "active" : ""}`}
                  onClick={() => openConversation(c.id)}
                >
                  <span className="conv-title">{c.title}</span>
                  <button
                    className="conv-del"
                    title={t("deleteConv")}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(c.id);
                    }}
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>

        <div className="main">
          <main className="chat" ref={scrollRef}>
            {messages.length === 0 ? (
              <div className="empty">
                <div className="empty-hero">
                  <div className="empty-greeting">{t(greetingKey())}</div>
                  <div className="empty-sub">
                    {model ? t("readyMsg") : t("loadToStart")}
                  </div>
                </div>
                {model && (
                  <div className="suggestions">
                    {(lang === "zh" ? SUGGESTIONS_ZH : SUGGESTIONS_EN).map((s) => (
                      <button key={s} className="suggestion" onClick={() => setInput(s)}>
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              messages.map((m, i) =>
                m.role === "assistant" ? (
                  <div key={m.id} className="msg assistant">
                    <AssistantMessage
                      content={m.content}
                      streaming={streamingId === m.id}
                      searching={streamingId === m.id && searching}
                      hideThinking={!thinkEnabled}
                    />
                    {m.sources && m.sources.length > 0 && (
                      <div className="sources">
                        <span className="sources-label">{t("sources")}</span>
                        <div className="sources-list">
                          {m.sources.map((s, k) => (
                            <button
                              key={k}
                              className="source-chip"
                              title={s.url}
                              onClick={() => openUrl(s.url).catch(() => {})}
                            >
                              <span className="source-idx">{k + 1}</span>
                              <span className="source-title">{s.title}</span>
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
                        <span className="user-text">{m.content}</span>
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

          <footer className="composer">
            {stats && (
              <div className="stats">
                {stats.completionTokens} tokens · {stats.tokensPerSecond.toFixed(1)} tok/s
                {model?.nCtx ? (
                  <span className="ctx-meter" title={t("ctxUsage")}>
                    {" · ctx "}
                    {fmtK(stats.promptTokens + stats.completionTokens)}/{fmtK(model.nCtx)}
                    <span className="ctx-bar">
                      <span
                        className="ctx-bar-fill"
                        style={{
                          width: `${Math.min(
                            100,
                            ((stats.promptTokens + stats.completionTokens) / model.nCtx) * 100,
                          )}%`,
                        }}
                      />
                    </span>
                  </span>
                ) : null}
              </div>
            )}
            {(attachment || attachError) && (
              <div className="attach-bar">
                {attachment && (
                  <div className="attach-chip">
                    <span className="attach-name">📄 {attachment.name}</span>
                    <span className="attach-meta">
                      {t("charsLabel", { n: attachment.chars })}
                      {attachment.truncated ? t("truncatedSuffix") : ""}
                    </span>
                    <button
                      className="attach-remove"
                      title={t("removeAttach")}
                      onClick={() => setAttachment(null)}
                    >
                      ×
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
                    ×
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
                    <button
                      className={`tool-item ${webEnabled ? "on" : ""}`}
                      onClick={() => {
                        const next = !webEnabled;
                        setWebEnabled(next);
                        if (next) setThinkEnabled(false); // web search ⇄ thinking are exclusive
                      }}
                    >
                      <svg className="ti-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                        <circle cx="12" cy="12" r="9" />
                        <path d="M3 12h18" />
                        <path d="M12 3c2.6 2.7 2.6 15.3 0 18M12 3c-2.6 2.7-2.6 15.3 0 18" />
                      </svg>
                      <span className="ti-label">{t("toolWeb")}</span>
                      <span className="ti-check">{webEnabled ? "✓" : ""}</span>
                    </button>
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
                        {thinkEnabled && model?.supportsThinking ? "✓" : ""}
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
                      <span className="ti-check">{webDesign ? "✓" : ""}</span>
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
                          <span className="ti-check">{speakReplies ? "✓" : ""}</span>
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
                  <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                    <path
                      d="M12 19V6M6 12l6-6 6 6"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
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
          appendNoThink={model?.supportsThinking ?? false}
        />
      )}
    </div>
  );
}
