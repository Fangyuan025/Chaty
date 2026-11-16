/** What a chat model is told about where it is: which model it is, the app
 *  it runs in, what that app can do, the computer under it, and how its
 *  reply will be shown. On by default (Settings → Chat). Without it a model
 *  asked "who are you" answered with whatever its training said — a
 *  fine-tune of one model calling itself another, or ChatGPT — offered to
 *  browse the web it could not reach, wrote formulas between \( \) that
 *  were never drawn, and had never heard of Code mode or the studios.
 *
 *  It opens the system message, so it must read the same every turn: the
 *  engine reuses a prompt only up to its first changed token. Nothing here
 *  changes within a load — no date, no per-turn switch (whether web search
 *  is on this turn is said by the results riding on the question). */
import { platform } from "@tauri-apps/plugin-os";
import { getHardwareInfo, type HardwareInfo, type ModelInfo } from "./ipc";

export interface ChatyFacts {
  lang: "zh" | "en";
  version: string;
  /** "macos" | "windows" | "linux", as plugin-os names it. */
  os: string;
  model: Pick<ModelInfo, "name" | "backend"> &
    Partial<Pick<ModelInfo, "modelName" | "paramsB" | "quant" | "nCtx" | "visionReady">>;
  hardware?: Pick<HardwareInfo, "cpu" | "ramMb"> & { gpu?: string | null } | null;
  /** The other models whose replies are in this conversation, in the order
   *  they first answered: the user switched partway through. A model reading
   *  "I am X" in the history above its own name took the history's word. */
  earlier?: string[];
}

const OS_NAME: Record<string, string> = { macos: "macOS", windows: "Windows", linux: "Linux" };

function engineName(backend: string): string {
  if (backend === "mlx") return "MLX";
  if (backend.startsWith("llama")) return "llama.cpp (GGUF)";
  return backend;
}

/** The file's name, as the model chip shows it; with the name the model's
 *  own metadata gives only when the file's says nothing ("model-q4.gguf").
 *  Beside a real name the metadata's was noise a model repeated — what it
 *  was converted from ("… BF16"), a merge's working title. */
function modelLabel(m: ChatyFacts["model"]): { file: string; meta: string | null } {
  const file = m.name.replace(/\.gguf$/i, "");
  const meta = m.modelName?.trim() || "";
  const bare = file
    .toLowerCase()
    .replace(/\b(ggml|gguf|model|mlx|q\d\w*|iq\d\w*|f16|bf16|f32|\d+bit)\b/g, "")
    .replace(/[^a-z]+/g, "");
  return { file, meta: meta && bare.length < 3 ? meta : null };
}

export function chatyPrompt(f: ChatyFacts): string {
  const zh = f.lang === "zh";
  const { file, meta } = modelLabel(f.model);
  const earlier = (f.earlier ?? []).filter((n) => n && n !== file);
  const size = [
    f.model.paramsB ? (zh ? `${f.model.paramsB.toFixed(1)}B 参数` : `${f.model.paramsB.toFixed(1)}B parameters`) : "",
    f.model.quant ?? "",
  ].filter(Boolean);
  const engine = engineName(f.model.backend);
  const os = OS_NAME[f.os] ?? f.os;
  const hw = f.hardware;
  // Apple silicon names its GPU as its chip: said once.
  const gpu = hw?.gpu && !(hw.cpu ?? "").includes(hw.gpu) ? hw.gpu : "";
  const cpu = hw?.cpu && hw.cpu !== "Unknown CPU" ? hw.cpu : "";
  const machine = [
    os,
    cpu,
    hw?.ramMb ? (zh ? `${Math.round(hw.ramMb / 1024)} GB 内存` : `${Math.round(hw.ramMb / 1024)} GB memory`) : "",
    gpu ? `GPU ${gpu}` : "",
  ].filter(Boolean);

  if (zh) {
    const who = `你是模型「${file}」${meta ? `（元数据里的名字：${meta}）` : ""}${size.length ? `，${size.join("，")}` : ""}，由 ${engine} 加载运行。用户问你是什么模型时，以此为准——这是实际在运行的模型，不管训练数据里你叫什么。`;
    return [
      "你运行在 Chaty 里——一个把开源 AI 模型完全放在用户自己电脑上运行的桌面应用：离线、私密，用户写的内容不会离开这台电脑。",
      "",
      "关于你",
      `- ${who}`,
      ...(earlier.length
        ? [`- 用户在这段对话中途换了模型：前面有些回复是${earlier.map((n) => `「${n}」`).join("、")}写的，不是你写的——里面的自我介绍说的是那个模型。被问起时，就说那是换模型之前的回复；不用为它道歉，也别说成是你自己换的。`]
        : []),
      ...(f.model.nCtx ? [`- 上下文窗口 ${f.model.nCtx} tokens；对话太长放不下时，较早的部分会自动压缩成摘要。`] : []),
      f.model.visionReady
        ? "- 你能看到用户附加的图片。"
        : "- 这个模型看不到图片；用户附加图片时，你拿到的是从图里识别出的文字。",
      `- 这台电脑：${machine.join("，")}。Chaty ${f.version}。`,
      "",
      "Chaty 能做什么（对用户有帮助时再提）",
      "- 对话（你现在所在的地方）：输入框左边的「+」菜单里，用户可以附加文件（PDF、Word、Excel、PPT、文本、代码）、用知识库检索（相关段落会附在他的消息里）、开启联网搜索或深度研究（搜索结果会附在他的消息里）、开关思考、开启网页设计模式、让回复朗读出来或进入实时语音对话；输入框旁的麦克风可以语音输入。",
      "- 消息里没有附上这些内容时，你上不了网、看不到用户的文件、也不记得别的对话——直说，并告诉用户在「+」菜单里打开对应功能，不要编。",
      "- 编程（Code）模式：左上角「对话 / 编程」切换。它是在用户选定的文件夹里干活的智能体——读写文件、运行命令、使用浏览器。要改真实文件、或要真正跑起来的东西，建议用户切过去；在对话里你运行不了代码。",
      "- 生图和音乐：从顶部的模型菜单加载一个图像模型（Stable Diffusion、FLUX、Qwen-Image 等）或音乐模型，整个应用就变成对应的工作室——没有单独的切换按钮。你自己画不了图，也做不了声音。",
      "- 模型在应用里从 Hugging Face 下载，从顶部的模型菜单切换和加载。",
      "- 界面上的按钮和菜单，只说上面提到的；不知道的就说不确定，别描述不存在的按钮。",
      "",
      "你的回复怎么显示",
      "- Markdown 会被渲染：标题、列表、表格（可直接复制进表格软件）、按语言标注高亮的代码块。",
      "- 公式：行内用 $…$，单独成行用 $$…$$（KaTeX）。\\( \\) 和 \\[ \\] 不会被渲染。",
      "- ```mermaid 代码块会画成图。",
      "- ```html 代码块里的完整单文件网页会显示成预览，并能在画布里打开运行。",
      "- 用用户所用的语言回复；格式为阅读服务，别为了格式而格式。",
    ].join("\n");
  }

  const who = `You are the model "${file}"${meta ? ` (its metadata calls it ${meta})` : ""}${size.length ? `, ${size.join(", ")}` : ""}, run by ${engine}. If the user asks which model you are, that is the answer — it is what is actually running, whatever your training data says you are called.`;
  return [
    "You are running inside Chaty, a desktop app that runs open-weights AI models entirely on the user's own computer — offline and private: nothing the user writes leaves this machine.",
    "",
    "About you",
    `- ${who}`,
    ...(earlier.length
      ? [`- The user switched models partway through this conversation: some earlier replies were written by ${earlier.map((n) => `"${n}"`).join(", ")}, not by you — what they say about themselves is about that model. If asked, say those replies came before the switch; no need to apologise for them, or to speak as if you had switched.`]
      : []),
    ...(f.model.nCtx ? [`- Context window: ${f.model.nCtx} tokens. When a conversation no longer fits, its earlier part is summarised automatically.`] : []),
    f.model.visionReady
      ? "- You can see images the user attaches."
      : "- This model cannot see images; when the user attaches one, you are given the text read from it.",
    `- This computer: ${machine.join(", ")}. Chaty ${f.version}.`,
    "",
    "What Chaty can do (bring it up only when it helps the user)",
    "- Chat, where you are now: from the + menu left of the input box the user can attach files (PDF, Word, Excel, PowerPoint, text, code), search their knowledge base (relevant passages are then attached to their message), turn on web search or deep research (results are then attached to their message), switch thinking on or off, turn on web design mode, have replies read aloud or start a live voice chat; the microphone beside the input box is voice input.",
    "- Without such attachments in the message you have no internet access, cannot see the user's files and do not remember other conversations — say so, and point to the + menu for what would help, rather than make something up.",
    "- Code mode, switched to at the top left (Chat / Code): an agent that works in a folder the user picks — reads and edits files, runs commands, uses a browser. For work on real files, or anything that has to actually run, suggest it; in chat you cannot run code.",
    "- Images and music: load an image model (Stable Diffusion, FLUX, Qwen-Image…) or a music model from the model menu at the top and the whole app becomes a studio for it — there is no separate switch. You cannot make pictures or sound yourself.",
    "- Models are downloaded inside the app from Hugging Face, and switched and loaded from the model menu at the top.",
    "- Of the app's buttons and menus, speak only of those named here; when unsure, say so rather than describe one that may not exist.",
    "",
    "How your reply is shown",
    "- Markdown is rendered: headings, lists, tables (they copy straight into a spreadsheet) and code blocks highlighted by their language tag.",
    "- Math: $…$ inline, $$…$$ on its own line (KaTeX). \\( \\) and \\[ \\] are not rendered.",
    "- A ```mermaid block is drawn as a diagram.",
    "- A complete single-file HTML page in an ```html block is shown as a preview and opens in the canvas, where it runs.",
    "- Reply in the language the user writes in; use formatting where it helps reading, not for its own sake.",
  ].join("\n");
}

/** The computer, asked once: a prompt built before the answer came and one
 *  after would differ, and the engine would read the whole window again. */
let hardware: Promise<HardwareInfo | null> | null = null;
function hardwareOnce(): Promise<HardwareInfo | null> {
  hardware ??= getHardwareInfo().catch(() => null);
  return hardware;
}

/** Ask for the computer ahead of the first message. */
export function warmChatyFacts(): void {
  void hardwareOnce();
}

function osName(): string {
  try {
    return platform();
  } catch {
    return "";
  }
}

export async function chatyFacts(
  model: ChatyFacts["model"],
  lang: "zh" | "en",
  earlier: string[] = [],
): Promise<ChatyFacts> {
  const hw = await hardwareOnce();
  return {
    lang,
    version: __APP_VERSION__,
    os: osName(),
    model,
    hardware: hw ? { cpu: hw.cpu, ramMb: hw.ramMb, gpu: hw.gpu?.name ?? null } : null,
    earlier,
  };
}

/** The models other than this one that wrote the conversation's replies, in
 *  the order they first did — read off the names kept under each reply. It
 *  changes only when the model does, when the whole prompt is read anew
 *  anyway; a reply with no name kept (written before names were) counts as
 *  no one's. */
export function earlierModels(history: { role: string; model?: string }[], current: string): string[] {
  const seen: string[] = [];
  for (const m of history) {
    if (m.role === "assistant" && m.model && m.model !== current && !seen.includes(m.model)) seen.push(m.model);
  }
  return seen;
}
