import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { AssistantMessage } from "./components/AssistantMessage";
import { ContextMenu } from "./components/ContextMenu";
import { HardwarePanel } from "./components/HardwarePanel";
import { LiveMode } from "./components/LiveMode";
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
} from "./lib/ipc";
import "./App.css";

interface UiMessage extends ChatMessage {
  id: string;
  sources?: SearchResult[];
}

const uid = () => Math.random().toString(36).slice(2);
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
  const [busy, setBusy] = useState(false);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [stats, setStats] = useState<GenStats | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [settings, setSettings] = useState<GenSettings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [showHardware, setShowHardware] = useState(false);
  const [webEnabled, setWebEnabled] = useState(false);
  const [thinkEnabled, setThinkEnabled] = useState(() => {
    try {
      return localStorage.getItem("chaty.think") !== "0";
    } catch {
      return true;
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
  const [showVoiceMenu, setShowVoiceMenu] = useState(false);
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
        } catch {
          if (last) localStorage.removeItem(LAST_MODEL_KEY); // file moved/deleted
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

  // Close the model picker when clicking outside it.
  useEffect(() => {
    if (!showModelMenu) return;
    const close = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest(".model-wrap")) setShowModelMenu(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [showModelMenu]);

  // Close the voice menu when clicking outside it.
  useEffect(() => {
    if (!showVoiceMenu) return;
    const close = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest(".voice-wrap")) setShowVoiceMenu(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [showVoiceMenu]);

  function openLive() {
    if (!model) return;
    liveConvRef.current = conversationId; // continue current chat, or null → new
    setShowVoiceMenu(false);
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
    } catch (e) {
      console.error(e);
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
      void refreshModels();
    } catch (e) {
      console.error(e);
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

  async function handleSend(override?: string) {
    const text = (override ?? input).trim();
    if (!text || busy) return;
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
    setBusy(true);
    setStreamingId(asstMsg.id);
    setStats(null);

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
          const query = messages.length > 0 ? await rewriteQuery(messages, text) : text;
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
            cur.map((m) => (m.id === asstMsg.id ? { ...m, sources: usedSources } : m)),
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

    // Thinking-mode control (Qwen3-style `/think` · `/no_think`): the user toggle
    // is the default; web search forces no-think to keep web answers concise.
    const historyForModel = history.map(({ role, content }) => ({ role, content }));
    if (historyForModel.length > 0 && (!thinkEnabled || webEnabled)) {
      const last = historyForModel[historyForModel.length - 1];
      last.content = `${last.content}\n/no_think`;
    }

    const sys = settings.systemPrompt.trim();
    const sent: ChatMessage[] = [
      { role: "system", content: t("todayNote", { date: formatDate(lang) }) },
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
              cur.map((m) =>
                m.id === asstMsg.id ? { ...m, content: m.content + ev.text } : m,
              ),
            );
            pumpSpeech(false);
          } else if (ev.type === "done") {
            setStats(ev.stats);
          } else if (ev.type === "error") {
            setMessages((cur) =>
              cur.map((m) =>
                m.id === asstMsg.id
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
        if (acc.text.trim()) await saveMessage(asstMsg.id, convId, "assistant", acc.text);
        await refreshConversations();
      } catch (e) {
        console.error(e);
      }
      // Let the model name a fresh conversation from its first question.
      if (freshConv && acc.text.trim()) void makeTitle(convId, text);
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
                    <div className="bubble">
                      <span className="user-text">{m.content}</span>
                    </div>
                  </div>
                ),
              )
            )}
          </main>

          <footer className="composer">
            {stats && (
              <div className="stats">
                {stats.completionTokens} tokens · {stats.tokensPerSecond.toFixed(1)} tok/s
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
            <div className="input-row">
              <button
                className="tool-toggle"
                title={t("attachTitle")}
                onClick={handleAttach}
                disabled={attaching}
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.7"
                  aria-hidden="true"
                >
                  <path
                    d="M21.5 11.5l-9 9a5.5 5.5 0 0 1-7.8-7.8l9-9a3.6 3.6 0 1 1 5.1 5.1l-9 9a1.8 1.8 0 0 1-2.5-2.5l8.3-8.3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <button
                className={`tool-toggle ${webEnabled ? "active" : ""}`}
                title={webEnabled ? t("webOn") : t("webOff")}
                onClick={() => setWebEnabled((v) => !v)}
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.7"
                  aria-hidden="true"
                >
                  <circle cx="12" cy="12" r="9" />
                  <path d="M3 12h18" />
                  <path d="M12 3c2.6 2.7 2.6 15.3 0 18M12 3c-2.6 2.7-2.6 15.3 0 18" />
                </svg>
              </button>
              <button
                className={`tool-toggle ${thinkEnabled ? "active" : ""}`}
                title={thinkEnabled ? t("thinkOn") : t("thinkOff")}
                onClick={() => setThinkEnabled((v) => !v)}
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.7"
                  aria-hidden="true"
                >
                  <path d="M9.5 18h5M10.5 21h3" strokeLinecap="round" />
                  <path
                    d="M12 3a6 6 0 0 0-3.5 10.9c.6.4 1 1.1 1 1.8v.3h5v-.3c0-.7.4-1.4 1-1.8A6 6 0 0 0 12 3z"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              {lang === "en" && (
                <div className="voice-wrap">
                  <button
                    className={`tool-toggle ${speakReplies || speaking ? "active" : ""} ${
                      speaking ? "speaking" : ""
                    }`}
                    title={t("voiceMenu")}
                    onClick={() => setShowVoiceMenu((v) => !v)}
                  >
                    <svg
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      aria-hidden="true"
                    >
                      <path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5H4z" strokeLinejoin="round" />
                      <path
                        d="M15.5 8.5a4.2 4.2 0 0 1 0 7M17.8 6a7.5 7.5 0 0 1 0 12"
                        strokeLinecap="round"
                      />
                    </svg>
                  </button>
                  {showVoiceMenu && (
                    <div className="voice-menu">
                      <button
                        className={`voice-item ${speakReplies ? "on" : ""}`}
                        onClick={() => {
                          if (speaking) stopSpeaking();
                          else if (speakReplies) stopSpeaking();
                          setSpeakReplies((v) => !v);
                          setShowVoiceMenu(false);
                        }}
                      >
                        <span className="vi-check">{speakReplies ? "✓" : ""}</span>
                        {t("speakAloud")}
                      </button>
                      <button className="voice-item" onClick={openLive} disabled={!model}>
                        <span className="vi-check">◉</span>
                        {t("liveStart")}
                      </button>
                    </div>
                  )}
                </div>
              )}
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={
                  model ? (webEnabled ? t("inputPhWeb") : t("inputPh")) : t("inputPhNoModel")
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
      {showLive && (
        <LiveMode
          onClose={() => setShowLive(false)}
          preamble={[
            t("todayNote", { date: formatDate(lang) }),
            settings.systemPrompt.trim(),
            "You are a voice assistant in a live, spoken conversation. Talk the way people actually speak out loud: relaxed, warm, and flowing, using contractions. Keep replies short — usually one to three sentences. Absolutely NO bullet points, numbered lists, headings, markdown, code, or emoji. If you would normally list things, weave them into a natural sentence instead (say \"you could try A, B, or C\" rather than listing them). Just talk naturally, like a friend on the phone.",
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
        />
      )}
    </div>
  );
}
