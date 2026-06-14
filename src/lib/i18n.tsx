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
  searchingKb: { zh: "正在检索知识库…", en: "Searching the knowledge base…" },
  searchingMix: { zh: "正在检索知识库与网络…", en: "Searching knowledge base & web…" },
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
  // knowledge base (local RAG)
  kbTitle: { zh: "本地知识库", en: "Local knowledge base" },
  kbDocs: { zh: "个文档", en: "docs" },
  kbChunks: { zh: "个片段", en: "chunks" },
  kbModelNote: {
    zh: "首次使用需下载多语嵌入模型 bge-m3（约 730 MB，一次性，之后完全离线）。",
    en: "First use downloads the multilingual bge-m3 embedding model (~730 MB, one-time; fully offline after).",
  },
  kbDownloadModel: { zh: "下载嵌入模型", en: "Download embedding model" },
  kbEmpty: { zh: "还没有文档 — 添加 PDF / 文本 / 图片开始", en: "No documents yet — add a PDF / text / image file" },
  kbAdd: { zh: "添加文档", en: "Add documents" },
  kbScopeTip: { zh: "勾选 = 参与检索", en: "Checked = included in retrieval" },
  kbScopeHint: {
    zh: "勾选要检索的文档，未勾选的不参与回答。",
    en: "Check the documents to search; unchecked ones are excluded from answers.",
  },
  kbRemove: { zh: "移除", en: "Remove" },
  kbIndexing: { zh: "索引中", en: "Indexing" },
  // deep-dive podcast (NotebookLM-style)
  kbPodcast: { zh: "生成深度播客", en: "Generate deep-dive podcast" },
  podcastTitle: { zh: "深度播客", en: "Deep-dive podcast" },
  podcastSub: {
    zh: "双主持人 · 英文 · 基于本地知识库",
    en: "Two hosts · English · from your knowledge base",
  },
  podcastIntro: {
    zh: "根据你启用的知识库文档，由模型撰写一段英文双人对话脚本，再用 Kokoro 一男一女两种音色交替朗读。生成期间其他 LLM 功能会暂时锁定，可随时取消，完成后可播放并导出音频。（仅生成英文播客）",
    en: "From your enabled documents, the model writes a two-host English script, then Kokoro reads it aloud with alternating male/female voices. Other LLM features are locked while it runs; you can cancel anytime, then play and export the audio. (English podcast only.)",
  },
  podcastStart: { zh: "开始生成", en: "Generate" },
  podcastWriting: { zh: "正在撰写脚本…", en: "Writing the script…" },
  podcastVoicing: { zh: "正在合成语音…", en: "Synthesizing voices…" },
  podcastEta: { zh: "预计剩余", en: "Time left" },
  podcastCancel: { zh: "取消生成", en: "Cancel" },
  podcastTurns: { zh: "段对话", en: "turns" },
  podcastPlay: { zh: "播放", en: "Play" },
  podcastStop: { zh: "停止", en: "Stop" },
  podcastExport: { zh: "导出音频", en: "Export audio" },
  podcastRegen: { zh: "重新生成", en: "Regenerate" },
  podcastRetry: { zh: "重试", en: "Try again" },
  podcastNeedModel: { zh: "请先加载一个聊天模型", en: "Load a chat model first" },
  podcastNoScript: {
    zh: "脚本生成失败，请重试或更换模型",
    en: "Could not produce a usable script — try again or switch models",
  },
  podcastFootZh: {
    zh: "提示：播客内容为英文，适合用于英语听力与学习。",
    en: "",
  },
  toolKb: { zh: "知识库检索", en: "Knowledge base" },
  toolKbManage: { zh: "管理知识库…", en: "Manage knowledge base…" },
  kbNeedSetup: { zh: "请先在知识库面板下载嵌入模型并添加文档", en: "Download the embedding model and add documents first" },
  // tools-menu groups (hover submenus)
  toolKbGroup: { zh: "知识库", en: "Knowledge base" },
  toolWebGroup: { zh: "联网功能", en: "Web" },
  // deep research
  toolDeepResearch: { zh: "深度研究", en: "Deep Research" },
  drTitle: { zh: "深度研究", en: "Deep Research" },
  drSub: {
    zh: "多轮联网检索 + 推理，生成含来源的深度报告",
    en: "Multi-round web search + reasoning → an in-depth, cited report",
  },
  drTopicPh: {
    zh: "输入研究主题，例如：固态电池的最新进展与商业化前景",
    en: "Enter a topic, e.g. the state of solid-state batteries and commercialization outlook",
  },
  drDepth: { zh: "深度", en: "Depth" },
  drDepthQuick: { zh: "快速（2 轮）", en: "Quick (2 rounds)" },
  drDepthStd: { zh: "标准（3 轮）", en: "Standard (3 rounds)" },
  drDepthDeep: { zh: "深入（4 轮）", en: "Deep (4 rounds)" },
  drRun: { zh: "开始研究", en: "Research" },
  drStop: { zh: "停止", en: "Stop" },
  drEmpty: {
    zh: "给出一个主题，模型会自动多轮检索网络、边查边推理，最后写成一篇带参考来源的深度报告，可导出 PDF。",
    en: "Give a topic; the model searches the web over several rounds, reasons as it goes, and writes a cited in-depth report you can export to PDF.",
  },
  drRound: { zh: "第", en: "round" },
  drQueries: { zh: "次检索", en: "searches" },
  drSources: { zh: "个来源", en: "sources" },
  drExportPdf: { zh: "导出 PDF", en: "Export PDF" },
  drExportMd: { zh: "导出 Markdown", en: "Export Markdown" },
  drPhasePlanning: { zh: "正在规划检索方向…", en: "Planning the research…" },
  drPhaseSearching: { zh: "正在联网检索…", en: "Searching the web…" },
  drPhaseReasoning: { zh: "正在分析并寻找信息缺口…", en: "Analyzing and finding gaps…" },
  drPhaseWriting: { zh: "正在撰写报告…", en: "Writing the report…" },
  drPhaseDone: { zh: "完成", en: "Done" },
  // first-launch setup
  setupBtn: { zh: "一键配置", en: "Set up for me" },
  setupTitle: { zh: "为这台电脑挑选模型", en: "Models picked for this machine" },
  setupBudget: { zh: "可用模型内存约", en: "model memory budget ≈" },
  setupDownload: { zh: "下载", en: "Download" },
  setupResolving: { zh: "正在查找…", en: "Resolving…" },
  setupLoad: { zh: "加载此模型", en: "Load this model" },
  setupNotFound: { zh: "未找到合适的 GGUF 文件", en: "No suitable GGUF file found" },
  setupFoot: {
    zh: "推荐按本机内存自动匹配规模与量化，下载自 HuggingFace，存入应用的 models 文件夹。",
    en: "Sized & quantized for your memory, downloaded from HuggingFace into the app's models folder.",
  },
  // parameter tooltips
  tipTemperature: {
    zh: "随机性：越低回答越确定，越高越发散有创意（常用 0.7）",
    en: "Randomness: lower = more deterministic, higher = more creative (0.7 is typical)",
  },
  tipTopP: {
    zh: "核采样：只从累计概率前 P 的候选词中选择，与温度配合控制多样性",
    en: "Nucleus sampling: choose only from tokens within the top-P cumulative probability",
  },
  tipTopK: {
    zh: "只考虑概率最高的 K 个候选词；0 = 关闭此限制",
    en: "Consider only the K most likely tokens; 0 disables the limit",
  },
  tipMinP: {
    zh: "过滤相对概率低于最高候选 P 倍的词；0 = 关闭",
    en: "Drop tokens whose probability is below P× the top token's; 0 disables",
  },
  tipRepeatPenalty: {
    zh: "大于 1 时抑制重复用词；调得过高会伤害流畅度",
    en: ">1 discourages repetition; too high hurts fluency",
  },
  tipMaxTokens: {
    zh: "单次回复的最大 token 数；“不限制”时由上下文窗口决定",
    en: "Max tokens per reply; with no limit, the context window is the bound",
  },
  tipStopSeqs: {
    zh: "模型一旦输出这些字符串就立即停止生成",
    en: "Generation stops immediately when the model emits any of these strings",
  },
  tipGpuAccel: {
    zh: "放到 GPU 的网络层数；自动 = 按显存放尽量多的层",
    en: "How many layers run on the GPU; Auto fills as many as memory allows",
  },
  tipCtxLength: {
    zh: "模型能“记住”的对话长度（token）；越大越耗内存",
    en: "How much conversation the model can hold (tokens); larger uses more memory",
  },
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
    zh: "下面是联网检索到的资料，已按【1】【2】…编号。请综合它们，用自然连贯的语言直接回答用户的问题；在用到某条资料的句子末尾标注对应角标，如【1】或【1】【3】（不要写“来源”二字，只写数字角标）。若资料不足以回答，请直说。\n\n",
    en: "Below is information retrieved from the web, numbered 【1】【2】…. Use it to answer the user's question in natural prose, and append the matching citation marker(s) — e.g. 【1】 or 【1】【3】 — at the end of each sentence that draws on a source. If the material is insufficient, say so.\n\n",
  },
  ragInstruction: {
    zh: "下面是从用户本地知识库检索到的文档片段，已按【1】【2】…编号。严格依据这些片段回答：只陈述片段中明确支持的内容，绝不编造、不引入片段之外的事实或数字；若片段不足以回答，必须直接说明“当前文档未提及”。在用到某条片段的句子末尾标注对应角标，如【1】或【1】【3】（只写数字角标）。\n\n",
    en: "Below are passages retrieved from the user's local knowledge base, numbered 【1】【2】…. Answer STRICTLY from these passages: state only what they explicitly support, never invent facts or numbers beyond them; if they do not contain the answer, you must say the current documents do not mention it. Append the matching citation marker(s) — e.g. 【1】 or 【1】【3】 — at the end of each sentence that draws on a passage.\n\n",
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
