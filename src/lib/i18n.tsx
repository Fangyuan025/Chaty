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
  zoomIn: { zh: "放大（Ctrl +）", en: "Zoom in (Ctrl +)" },
  zoomOut: { zh: "缩小（Ctrl −）", en: "Zoom out (Ctrl −)" },
  zoomReset: { zh: "重置缩放（Ctrl 0）", en: "Reset zoom (Ctrl 0)" },
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
  webDesignOn: { zh: "网页设计模式：已开启（/webdesign 切换）", en: "Web design mode: on (/webdesign to toggle)" },
  webDesignOff: { zh: "网页设计模式：已关闭（/webdesign 切换）", en: "Web design mode: off (/webdesign to toggle)" },
  webDesignChip: { zh: "网页设计模式", en: "Web design mode" },
  inputPhDesign: { zh: "描述你想要的界面，模型会生成单文件 HTML…", en: "Describe the UI you want — get a single-file HTML…" },
  attachTitle: { zh: "添加附件（文档 / PDF / 图片）", en: "Attach a file (document / PDF / image)" },
  toolsMenu: { zh: "工具", en: "Tools" },
  toolAttach: { zh: "添加附件", en: "Attach a file" },
  toolWeb: { zh: "联网搜索", en: "Web search" },
  toolThink: { zh: "思考模式", en: "Thinking mode" },
  toolDesign: { zh: "网页设计模式", en: "Web design mode" },
  thinkUnsupported: { zh: "当前模型不支持思考模式", en: "This model doesn't support thinking" },
  thinkAuto: {
    zh: "该模型（Qwen3.5+）由架构自动管理思考，无需手动开关",
    en: "Thinking is auto-managed by this model (Qwen3.5+); no manual switch",
  },
  // update banner
  updateAvailable: { zh: "发现新版本 v{v}", en: "Update available — v{v}" },
  updateNow: { zh: "立即更新", en: "Update now" },
  updateLater: { zh: "稍后", en: "Later" },
  updateDownloading: { zh: "下载中…", en: "Downloading…" },
  // context usage
  ctxUsage: { zh: "上下文占用", en: "Context usage" },
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
  regenerate: { zh: "重新生成", en: "Regenerate" },
  regenTitle: { zh: "重新生成这条回答（会丢弃其后的对话）", en: "Regenerate this reply (drops what follows)" },
  editMsg: { zh: "编辑", en: "Edit" },
  saveEdit: { zh: "保存并重发", en: "Save & resend" },
  cancel: { zh: "取消", en: "Cancel" },
  // assistant message
  searching: { zh: "正在联网搜索…", en: "Searching the web…" },
  composing: { zh: "正在整理上文…", en: "Composing earlier context…" },
  contextSummary: {
    zh: "【早前对话摘要，供延续参考】\n",
    en: "[Summary of earlier conversation, for continuity]\n",
  },
  thinking: { zh: "正在思考", en: "Thinking" },
  thoughtExpand: { zh: "已深度思考 · 点击展开", en: "Reasoned · click to expand" },
  thoughtCollapse: { zh: "已深度思考 · 点击收起", en: "Reasoned · click to collapse" },
  // attachment
  removeAttach: { zh: "移除附件", en: "Remove attachment" },
  truncatedSuffix: { zh: " · 已截断", en: " · truncated" },
  charsLabel: { zh: "{n} 字", en: "{n} chars" },
  readAttachFailed: { zh: "读取附件失败", en: "Failed to read file" },
  dropToAttach: { zh: "松开以添加为附件", en: "Drop to attach" },
  searchConv: { zh: "搜索对话…", en: "Search chats…" },
  noMatches: { zh: "无匹配对话", en: "No matching chats" },
  exportTitle: { zh: "导出对话", en: "Export chat" },
  exportMd: { zh: "导出为 Markdown", en: "Export as Markdown" },
  exportJson: { zh: "导出为 JSON", en: "Export as JSON" },
  exportFailed: { zh: "导出失败", en: "Export failed" },
  dlTitle: { zh: "下载模型", en: "Download model" },
  dlRepoPh: { zh: "HuggingFace 仓库，如 Qwen/Qwen3-4B-GGUF", en: "HuggingFace repo, e.g. Qwen/Qwen3-4B-GGUF" },
  dlSearch: { zh: "查找", en: "Find" },
  dlSearching: { zh: "查找中…", en: "Finding…" },
  dlHint: {
    zh: "输入 HuggingFace 仓库（owner/name）或其网址，列出其中的 GGUF 文件后下载到模型文件夹。",
    en: "Enter a HuggingFace repo (owner/name) or URL to list its GGUF files and download into your models folder.",
  },
  dlGet: { zh: "下载", en: "Download" },
  dlSearchFailed: { zh: "查找失败", en: "Lookup failed" },
  dlFailed: { zh: "下载失败", en: "Download failed" },
  // settings
  settingsTitle: { zh: "生成设置", en: "Generation settings" },
  language: { zh: "语言", en: "Language" },
  systemPrompt: { zh: "系统提示词", en: "System prompt" },
  systemPromptPh: { zh: "可选，例如：你是一个简洁、专业的助手。", en: "Optional, e.g. You are a concise, professional assistant." },
  temperature: { zh: "温度", en: "Temperature" },
  maxTokens: { zh: "最大生成长度", en: "Max length" },
  off: { zh: "关", en: "off" },
  repeatPenalty: { zh: "重复惩罚", en: "Repeat penalty" },
  stopSeqs: { zh: "停止词（每行一个）", en: "Stop sequences (one per line)" },
  stopSeqsPh: { zh: "例如：\nUser:\n###", en: "e.g.\nUser:\n###" },
  presets: { zh: "提示词预设", en: "Prompt presets" },
  presetNamePh: { zh: "预设名称…", en: "Preset name…" },
  presetSave: { zh: "保存当前", en: "Save" },
  voice: { zh: "语音发音人", en: "Voice" },
  voiceSpeed: { zh: "语速", en: "Speech rate" },
  theme: { zh: "主题", en: "Theme" },
  themeSystem: { zh: "跟随系统", en: "System" },
  themeLight: { zh: "浅色", en: "Light" },
  themeDark: { zh: "深色", en: "Dark" },
  resetDefaults: { zh: "恢复默认", en: "Reset" },
  // GPU acceleration settings
  gpuAccel: { zh: "GPU 加速", en: "GPU acceleration" },
  gpuAuto: { zh: "自动", en: "Auto" },
  gpuOff: { zh: "关闭", en: "Off" },
  gpuCustom: { zh: "自定义", en: "Custom" },
  gpuLayersLabel: { zh: "GPU 层数", en: "GPU layers" },
  gpuHint: {
    zh: "自动模式会按显存把尽量多的层放到 GPU。更改将在下次加载模型时生效。",
    en: "Auto fills the GPU with as many layers as VRAM allows. Changes apply on the next model load.",
  },
  // context length settings
  ctxLength: { zh: "上下文长度", en: "Context length" },
  ctxAuto: { zh: "自动", en: "Auto" },
  ctxTokens: { zh: "上下文 Token", en: "Context tokens" },
  ctxHint: {
    zh: "自动 = 在内存允许范围内尽量用满模型的训练上下文。自定义值也会按内存上限自动回落。",
    en: "Auto = as much of the model's trained context as memory allows. Custom values are also capped to fit memory.",
  },
  reloadApply: { zh: "重新加载模型以生效", en: "Reload model to apply" },
  ejectingModel: { zh: "正在卸载旧模型…", en: "Ejecting old model…" },
  noLimit: { zh: "不限制", en: "No limit" },
  openModelsDir: { zh: "打开模型文件夹", en: "Open models folder" },
  // stop reasons (shown in the stats line after generation)
  stopEos: { zh: "自然结束", en: "finished" },
  stopLength: { zh: "达到长度上限", en: "length limit" },
  stopContext: { zh: "上下文已满", en: "context full" },
  stopStop: { zh: "命中停止词", en: "stop sequence" },
  stopCancelled: { zh: "已手动停止", en: "cancelled" },
  // hardware panel
  hwTitleBtn: { zh: "硬件信息", en: "Hardware" },
  hwTitle: { zh: "本机硬件", en: "Hardware" },
  hwCpu: { zh: "处理器", en: "CPU" },
  hwRam: { zh: "内存", en: "Memory" },
  hwGpu: { zh: "显卡", en: "GPU" },
  hwVram: { zh: "显存占用", en: "VRAM usage" },
  hwBackend: { zh: "GPU 后端", en: "GPU backend" },
  hwAccel: { zh: "当前模型加速", en: "Current model" },
  hwNoGpu: { zh: "未检测到独立显卡", en: "No discrete GPU detected" },
  hwThreads: { zh: "{n} 线程", en: "{n} threads" },
  hwLayersOn: { zh: "{a}/{b} 层在 GPU", en: "{a}/{b} layers on GPU" },
  hwCpuOnly: { zh: "纯 CPU 运行", en: "Running on CPU" },
  hwNoModel: { zh: "未加载模型", en: "No model loaded" },
  // model info panel
  miTitleBtn: { zh: "模型信息", en: "Model info" },
  miTitle: { zh: "模型信息", en: "Model info" },
  miName: { zh: "名称", en: "Name" },
  miArch: { zh: "架构", en: "Architecture" },
  miParams: { zh: "参数量", en: "Parameters" },
  miQuant: { zh: "量化", en: "Quantization" },
  miSize: { zh: "大小", en: "Size" },
  miContext: { zh: "上下文", en: "Context" },
  miTrained: { zh: "训练", en: "trained" },
  miLayers: { zh: "层数", en: "Layers" },
  miEmbed: { zh: "嵌入维度", en: "Embedding" },
  miTemplate: { zh: "对话模板", en: "Chat template" },
  miThinking: { zh: "思考推理", en: "Thinking" },
  miTools: { zh: "工具调用", en: "Tool calls" },
  miMultimodal: { zh: "多模态", en: "Multimodal" },
  miNoModel: { zh: "未加载模型", en: "No model loaded" },
  // load notices / OOM
  oomPartial: {
    zh: "显存不足，已自动减少 GPU 层数（{a}/{b} 层在 GPU）",
    en: "Low VRAM — GPU offload reduced to {a}/{b} layers",
  },
  oomCpu: { zh: "显存不足，已回退到 CPU 运行", en: "Low VRAM — fell back to CPU" },
  ctxClamped: {
    zh: "上下文已按内存自动调整为 {n}（模型权重 + KV 缓存需放入统一内存）",
    en: "Context auto-fitted to {n} (weights + KV cache must fit in unified memory)",
  },
  oomFail: {
    zh: "内存不足，无法加载该模型。试试更小 / 更高量化的模型，或关闭其他占用内存的程序。",
    en: "Out of memory — couldn't load this model. Try a smaller / more-quantized model, or free up RAM.",
  },
  toastDismiss: { zh: "点击关闭", en: "Click to dismiss" },
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
