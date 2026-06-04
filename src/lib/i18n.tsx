import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Lang = "zh" | "en";

const LANG_KEY = "chaty.lang";

type Entry = { zh: string; en: string };

export const T = {
  // titlebar / sidebar
  newChat: { zh: "新对话", en: "New chat" },
  noModel: { zh: "未加载模型", en: "No model" },
  loadingModel: { zh: "加载中…", en: "Loading…" },
  changeModel: { zh: "更换模型", en: "Change model" },
  loadGguf: { zh: "加载 GGUF", en: "Load GGUF" },
  modelsHeader: { zh: "models 文件夹", en: "Models folder" },
  loadFromFile: { zh: "从文件选择…", en: "Load from file…" },
  noModelsFound: { zh: "models 文件夹中暂无模型", en: "No models in the models folder" },
  refreshModels: { zh: "刷新列表", en: "Refresh" },
  // html preview
  htmlPreview: { zh: "预览", en: "Preview" },
  htmlPreviewTitle: { zh: "预览 HTML 渲染效果", en: "Preview rendered HTML" },
  openInBrowser: { zh: "在浏览器打开", en: "Open in browser" },
  closePreview: { zh: "关闭预览", en: "Close preview" },
  noConversations: { zh: "暂无历史会话", en: "No conversations yet" },
  deleteConv: { zh: "删除会话", en: "Delete conversation" },
  // empty state
  readyMsg: { zh: "我已就绪——完全本地运行，对话不出本机。", en: "Ready — everything runs locally, nothing leaves your machine." },
  loadToStart: { zh: "加载一个本地 GGUF 模型，即可开始对话。", en: "Load a local GGUF model to start chatting." },
  // composer
  inputPh: { zh: "输入消息，Enter 发送，Shift+Enter 换行", en: "Message… (Enter to send, Shift+Enter for newline)" },
  inputPhWeb: { zh: "联网搜索已开启，输入问题…", en: "Web search on — ask anything…" },
  inputPhNoModel: { zh: "先加载一个 GGUF 模型…", en: "Load a GGUF model first…" },
  webOn: { zh: "联网搜索：已开启", en: "Web search: on" },
  webOff: { zh: "联网搜索：已关闭", en: "Web search: off" },
  thinkOn: { zh: "思考模式：已开启", en: "Thinking mode: on" },
  thinkOff: { zh: "思考模式：已关闭", en: "Thinking mode: off" },
  attachTitle: { zh: "添加附件（文档 / PDF / 图片）", en: "Attach a file (document / PDF / image)" },
  stopTitle: { zh: "停止生成", en: "Stop" },
  sendTitle: { zh: "发送", en: "Send" },
  micStart: { zh: "语音输入", en: "Voice input" },
  micStop: { zh: "停止录音", en: "Stop recording" },
  liveStart: { zh: "实时语音对话", en: "Live voice chat" },
  liveExit: { zh: "退出实时模式", en: "Exit live mode" },
  liveListening: { zh: "聆听中…", en: "Listening…" },
  liveThinking: { zh: "思考中…", en: "Thinking…" },
  liveSpeaking: { zh: "回答中…", en: "Speaking…" },
  speakOn: { zh: "朗读回复：开", en: "Read replies aloud: on" },
  speakOff: { zh: "朗读回复：关", en: "Read replies aloud: off" },
  voiceMenu: { zh: "语音", en: "Voice" },
  speakAloud: { zh: "朗读回复", en: "Read replies aloud" },
  // context menu
  ctxCut: { zh: "剪切", en: "Cut" },
  ctxCopy: { zh: "复制", en: "Copy" },
  ctxPaste: { zh: "粘贴", en: "Paste" },
  ctxSelectAll: { zh: "全选", en: "Select all" },
  // window controls
  minimize: { zh: "最小化", en: "Minimize" },
  maximize: { zh: "最大化", en: "Maximize" },
  restore: { zh: "向下还原", en: "Restore" },
  close: { zh: "关闭", en: "Close" },
  // message actions
  sources: { zh: "来源", en: "Sources" },
  copy: { zh: "复制", en: "Copy" },
  fork: { zh: "⑂ 分叉", en: "⑂ Branch" },
  copyTitle: { zh: "复制回答", en: "Copy reply" },
  forkTitle: { zh: "从这里分叉为新对话", en: "Branch into a new chat from here" },
  reread: { zh: "朗读", en: "Replay" },
  rereadStop: { zh: "停止", en: "Stop" },
  rereadTitle: { zh: "朗读这条回复", en: "Read this reply aloud" },
  // assistant message
  searching: { zh: "正在联网搜索…", en: "Searching the web…" },
  thinking: { zh: "正在思考", en: "Thinking" },
  thoughtExpand: { zh: "已深度思考 · 点击展开", en: "Reasoned · click to expand" },
  thoughtCollapse: { zh: "已深度思考 · 点击收起", en: "Reasoned · click to collapse" },
  // attachment
  removeAttach: { zh: "移除附件", en: "Remove attachment" },
  truncatedSuffix: { zh: " · 已截断", en: " · truncated" },
  charsLabel: { zh: "{n} 字", en: "{n} chars" },
  readAttachFailed: { zh: "读取附件失败", en: "Failed to read file" },
  // settings
  settingsTitle: { zh: "生成设置", en: "Generation settings" },
  language: { zh: "语言", en: "Language" },
  systemPrompt: { zh: "系统提示词", en: "System prompt" },
  systemPromptPh: { zh: "可选，例如：你是一个简洁、专业的助手。", en: "Optional, e.g. You are a concise, professional assistant." },
  temperature: { zh: "温度", en: "Temperature" },
  maxTokens: { zh: "最大生成长度", en: "Max length" },
  resetDefaults: { zh: "恢复默认", en: "Reset" },
  // greetings (time of day)
  greetMorning: { zh: "早上好", en: "Good morning" },
  greetNoon: { zh: "中午好", en: "Good afternoon" },
  greetAfternoon: { zh: "下午好", en: "Good afternoon" },
  greetEvening: { zh: "晚上好", en: "Good evening" },
  greetNight: { zh: "夜深了", en: "Working late" },
  // model-facing prompts
  todayNote: {
    zh: "当前日期是 {date}。涉及“今天/最近/现在”等时间时以此为准。",
    en: 'Today is {date}. When the question refers to "today/recent/now", use this date.',
  },
  webInstruction: {
    zh: "下面是联网检索到的资料。请综合它们，用自然连贯的语言直接回答用户的问题；不要在正文里插入“【来源N】”“(来源N)”之类的标注（来源会单独展示给用户）。若资料不足以回答，请直说。\n\n",
    en: "Below is information retrieved from the web. Use it to answer the user's question in natural prose; do NOT insert inline citation markers like [N] or (source N) (sources are shown separately). If it is insufficient, say so.\n\n",
  },
  attachInstruction: {
    zh: "用户上传了文件《{name}》，其内容如下，回答时请优先依据它：\n\n",
    en: 'The user uploaded a file "{name}". Its content is below; base your answer on it:\n\n',
  },
} satisfies Record<string, Entry>;

export type TKey = keyof typeof T;

function detectLang(): Lang {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === "zh" || saved === "en") return saved;
  } catch {
    /* ignore */
  }
  // Default first language is English (voice features are English-only and are
  // hidden when the user explicitly switches to Chinese).
  return "en";
}

interface I18n {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (key: TKey, vars?: Record<string, string | number>) => string;
}

const LangContext = createContext<I18n | null>(null);

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(detectLang);

  useEffect(() => {
    try {
      localStorage.setItem(LANG_KEY, lang);
    } catch {
      /* ignore */
    }
  }, [lang]);

  const t = useCallback(
    (key: TKey, vars?: Record<string, string | number>) => {
      let s: string = T[key]?.[lang] ?? key;
      if (vars) {
        for (const k of Object.keys(vars)) s = s.replace(`{${k}}`, String(vars[k]));
      }
      return s;
    },
    [lang],
  );

  const value = useMemo<I18n>(() => ({ lang, setLang, t }), [lang, t]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useI18n(): I18n {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useI18n must be used within LangProvider");
  return ctx;
}
