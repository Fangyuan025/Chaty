// The agentic-coding brain for Code mode. Reuses the local model via generate()
// and drives a tool loop: the model emits ONE <tool_call>{json}</tool_call>, we
// stop generation there, run the tool (confined + sandboxed on the Rust side),
// feed the result back as <tool_result>, and repeat until the model answers with
// no tool call. Works on any instruct model (no native function-calling needed);
// it degrades gracefully when the model doesn't follow the format.

import {
  agentBash,
  agentBashBg,
  agentBgKill,
  agentBgOutput,
  agentBgReap,
  agentEditFile,
  agentMultiEdit,
  agentOutline,
  type EditOp,
  agentGlob,
  agentGrep,
  agentSearchFiles,
  agentListDir,
  agentReadFile,
  agentSearchCode,
  agentWriteFile,
  cancelGeneration,
  fetchPageEx,
  siteSearch,
  agentWebDownload,
  generate,
  ragSearch,
  webSearch,
  type ChatMessage,
} from "./ipc";
import { normalizeChannels } from "./voiceText";
import { diffLines } from "./diff";

export type AgentToolName =
  | "read_file"
  | "write_file"
  | "edit_file"
  | "multi_edit"
  | "outline"
  | "list_dir"
  | "glob"
  | "grep"
  | "search_files"
  | "search_code"
  | "search_docs"
  | "bash"
  | "bash_bg"
  | "bg_output"
  | "bg_kill"
  | "web_search"
  | "web_fetch"
  | "web_download"
  | "ask_user"
  | "update_plan";

/** Tools that change the world (or run code) → need approval unless bypassed. */
export const MUTATING_TOOLS = new Set<AgentToolName>([
  "write_file",
  "edit_file",
  "multi_edit",
  "bash",
  "bash_bg",
  "web_download",
]);

export interface ToolCall {
  name: AgentToolName;
  args: Record<string, unknown>;
}

export type StepStatus = "running" | "done" | "error" | "denied";

/** How much the model reasons before each action. */
export type ThinkMode = "off" | "normal" | "deep";

/** A single item in the agent's task plan (todo list). */
export type PlanStatus = "pending" | "in_progress" | "done";
export interface PlanItem {
  content: string;
  status: PlanStatus;
}

export interface ToolStep {
  id: string;
  call: ToolCall;
  status: StepStatus;
  /** The model's reasoning that led to this tool call (shown collapsed). */
  thinking?: string;
  /** Human-readable result/output (for the UI). */
  result?: string;
  /** For edit/write, the before/after so the UI can render a diff. */
  diff?: { path: string; before: string; after: string };
}

export class AgentSignal {
  cancelled = false;
  cancel() {
    this.cancelled = true;
    void cancelGeneration().catch(() => {});
  }
}

export interface AgentCallbacks {
  /** Streaming reasoning for the current step (shown in a think panel). */
  onThinking: (full: string) => void;
  /** Streaming assistant prose for the current turn (before/around a tool call). */
  onAssistantText: (full: string) => void;
  /** A tool step was created or updated. */
  onStep: (step: ToolStep) => void;
  /** Live generation stats: total tokens this turn + current tokens/sec. */
  onStats?: (tokens: number, tps: number) => void;
  /** Context window position after a step (prompt + output tokens used). */
  onContext?: (used: number) => void;
  /** The model asks the user to pick between options. Resolves with the choice. */
  onAskUser?: (question: string, options: string[]) => Promise<string>;
  /** The model set/updated its task plan (todo list). */
  onPlan?: (todos: PlanItem[]) => void;
  /** Context was auto-compacted (old history/tool results elided). Fires once per turn. */
  onCompacted?: () => void;
  /** The model finished the task (no more tool calls). `reason` is "steps"
   *  when the turn paused at the step limit rather than truly finishing. */
  onFinal: (text: string, thinking?: string, reason?: "done" | "steps") => void;
  onError: (message: string) => void;
}

export interface AgentOptions {
  /** Reasoning depth: off = no thinking, normal = default, deep = thorough. */
  thinkMode: ThinkMode;
  /** From ModelInfo — picks the right no-think mechanism per model family
   *  (Qwen3 soft switch vs. Qwen3.5+/Gemma think-flag), mirroring chat mode. */
  supportsThinking?: boolean;
  /** Model uses the `/no_think` soft switch (Qwen3) instead of the think flag. */
  thinkSwitch?: boolean;
  nCtx?: number;
  maxSteps?: number;
  /** Default timeout for bash commands (seconds) when the model doesn't set one. */
  bashTimeout?: number;
  /** Project guide (AGENTS.md / PROJECT.md / CLAUDE.md) injected into the
   *  system prompt — the /init loop's other half. */
  projectDoc?: { name: string; text: string };
  signal: AgentSignal;
  /** Gate a mutating tool call. Return true to run, false to deny. Bypass mode
   *  passes a function that always resolves true. */
  approve: (call: ToolCall) => Promise<boolean>;
}

const uid = () => Math.random().toString(36).slice(2);

function stripThink(raw: string): string {
  // Channel-style reasoning markers (Gemma 4 / Harmony) → <think> convention,
  // same normalization chat mode applies before parsing.
  const s = normalizeChannels(raw);
  const o = s.indexOf("<think>");
  if (o === -1) {
    // Orphan close: reasoning streamed without an opening tag (pre-open-trained
    // models) — everything before the close is reasoning.
    const c = s.indexOf("</think>");
    if (c !== -1) return s.slice(c + "</think>".length);
  }
  return s.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, "");
}

/** The reasoning inside `<think>…</think>` (or after `<think>` if not yet closed). */
function thinkPart(raw: string): string {
  const s = normalizeChannels(raw);
  const o = s.indexOf("<think>");
  if (o === -1) {
    const c = s.indexOf("</think>");
    return c === -1 ? "" : s.slice(0, c).trim(); // orphan close
  }
  const after = s.slice(o + "<think>".length);
  const c = after.indexOf("</think>");
  return (c === -1 ? after : after.slice(0, c)).trim();
}

/** Prose after any think block and before any tool call. */
function proseAfter(raw: string): string {
  let t = normalizeChannels(raw);
  const c = t.indexOf("</think>");
  if (c !== -1) t = t.slice(c + "</think>".length);
  else if (t.includes("<think>")) return ""; // still thinking → no prose yet
  const tc = t.indexOf("<tool_call>");
  return (tc === -1 ? t : t.slice(0, tc)).trim();
}

const TOOLS_DOC = `
- read_file: 读取文件,一般一次调用即可读完整个文件;只有超出上下文预算的超大文件才分页,此时结果末尾会直接给出下一页的 offset,照着传即可。args: { "path": string, "offset"?: number(起始行,从1开始), "limit"?: number(行数) }
- write_file: 新建文件,或对一个文件做整体重写(会覆盖全部内容)。**修改已有文件请优先用 edit_file / multi_edit**,不要用 write_file 重写整个文件来做局部改动。args: { "path": string, "content": string }
- edit_file: 精确替换文件内容(old_string 必须与文件逐字匹配且唯一,除非 replace_all=true);匹配失败会提示文件中最相似的位置。改一处直接给 old_string/new_string;同一文件要改多处时给 edits 数组,一次原子提交(任何一条失败则整体不改动),不要拆成多次调用。args: 单处 { "path": string, "old_string": string, "new_string": string, "replace_all"?: boolean } 或 多处 { "path": string, "edits": [{ "old_string": string, "new_string": string, "replace_all"?: boolean }] }
- outline: 列出文件的定义大纲(函数/类/结构体等 + 行号),不读全文即可掌握文件结构;之后用 read_file 带 offset 精确读需要的区段。args: { "path": string }
- list_dir: 列出目录一层内容(不传 path = 工作区根;看子目录请传相对路径,如 {"path":"src"})。args: { "path"?: string }
- glob: 按通配符找文件(如 "src/**/*.ts")。args: { "pattern": string }
- grep: 用正则搜索文件内容(需要正则或精确匹配时用)。args: { "pattern": string, "path"?: string, "glob"?: string }
- search_files: 按关键词(字面,不是正则)一次性搜文件名和文件内容 —— "跟 X 有关的东西在哪"最快用它。传 names_only=true 只搜文件名。args: { "query": string, "path"?: string, "names_only"?: boolean }
- search_code: 按含义搜索代码("哪里处理登录鉴权"),返回排序过的相关代码块(文件+行号)。探索陌生代码库优先用它。args: { "query": string, "k"?: number }
- search_docs: 检索用户的知识库文档(需求文档、设计稿、笔记)。当任务涉及用户自己的资料时用。args: { "query": string }
- bash: 在工作区里执行 shell 命令(macOS 沙箱,写限工作区)。会等命令结束;不要用它启动 dev server 等不会退出的进程。args: { "command": string, "timeout_secs"?: number }
- bash_bg: 在后台启动长时间运行的命令(dev server、慢构建、长测试),立即返回一个 id,期间你可以继续做别的;它结束时系统会自动把结果告诉你。args: { "command": string }
- bg_output: 查看某个后台命令的当前状态和最近输出(比如确认 server 已启动)。args: { "id": number }
- bg_kill: 终止某个后台命令(整棵进程树)。args: { "id": number }
- web_search: 联网搜索(标题+链接+摘要)。查资料、找文档、查报错时用。加 site 参数可做站内搜索:site="github.com" 返回结构化的仓库/issue/代码匹配;site="reddit.com"(或 "reddit.com/r/某版块")搜帖子;site="youtube.com" / "bilibili.com" 返回视频(标题/时长/UP主/播放量);其他任意域名(docs.python.org、stackoverflow.com、x.com、weibo.com 等)都会限定在该站内搜(登录墙站点只能拿到搜索引擎快照级的标题/摘要)。args: { "query": string, "site"?: string }
- web_fetch: 抓取任意 URL,按内容类型自动处理:文章页→干净的 Markdown 正文;代码/JSON/配置文件→原文;GitHub 文件页自动取 raw 源文件;Reddit 帖子→正文+评论;YouTube 视频→元信息+完整字幕转写;B站视频→公开元信息+简介(播放/点赞/弹幕);PDF→提取文本;图片等二进制→返回元信息(用 web_download 保存)。结果还会列出页面上的链接和图片 URL——想深入子页面就继续 fetch 那些链接。要 HTML 源码时传 raw=true。args: { "url": string, "raw"?: boolean }
- web_download: 把 URL 指向的文件(图片、压缩包、任意资源)下载到工作区指定路径。args: { "url": string, "path": string }
- update_plan: 制定或更新任务计划(待办清单),让用户看到你的推进步骤。开始复杂任务时先列计划,完成一步就把它标为 done、把下一步标为 in_progress。args: { "todos": [{ "content": string, "status": "pending"|"in_progress"|"done" }] }
- ask_user: 当遇到需要用户拍板的决策(方案分歧、需求不明、破坏性操作确认)时,向用户提一个选择题;不要自己乱猜。args: { "question": string, "options": string[] }`;

function systemPrompt(
  workspace: string,
  zh: boolean,
  mode: ThinkMode,
  projectDoc?: { name: string; text: string },
): string {
  const doc = projectDoc
    ? zh
      ? `\n\n项目说明(来自工作区的 ${projectDoc.name},请遵循其中的约定):\n${projectDoc.text}`
      : `\n\nProject guide (from ${projectDoc.name} in the workspace — follow its conventions):\n${projectDoc.text}`
    : "";
  const think =
    mode === "deep"
      ? zh
        ? "\n- 在每次行动前,先在 <think>…</think> 中充分思考:分析现状、权衡多种方案、考虑边界情况,再决定调用哪个工具。"
        : "\n- Before each action, reason thoroughly inside <think>…</think>: analyze the state, weigh options and edge cases, then decide which tool to call."
      : mode === "normal"
        ? zh
          ? "\n- 行动前可在 <think>…</think> 中简要思考下一步,再调用工具。"
          : "\n- You may think briefly inside <think>…</think> before each tool call."
        : "";
  if (zh) {
    return `你是 Chaty 的编程智能体,在一个工作区目录中帮用户完成编码任务。工作区根目录:${workspace}

你可以调用下列工具(所有路径都相对于工作区;越界路径会被拒绝):
${TOOLS_DOC}

调用规则(务必严格遵守):
- 每次只调用一个工具。要调用时,只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call> 然后立即停止,不要在同一条消息里写其它内容。
- 系统会把结果以 <tool_result>...</tool_result> 返回给你,你再继续。
- 没有"当前目录"的概念:每条 bash 都是从工作区根目录启动的全新 shell,单独的 cd 不会保留到下一条命令。访问子目录请直接用相对路径(ls src、read_file "src/app.ts"),或在同一条命令内组合(cd src && npm test)。
- 修改代码前,先用 outline 看文件结构、read_file / grep / list_dir 了解现状;改完可用 bash 跑测试/构建验证。
- 读大文件别从头翻到尾:先用 search_code / grep 定位到相关位置,再用 read_file 带 offset/limit 只读需要的区段。
- 工具选择:**新建文件**用 write_file 一次写完;**修改已有文件**用 edit_file(改一处给 old_string/new_string,改多处给 edits 数组一次原子提交)——不要用 write_file 整体重写已有文件来做局部改动,那会覆盖全文、极易丢失你没重写的内容。只有确实要把整个文件推倒重来时才用 write_file。不要把一个改动拆成许多细碎小步。
- dev server、npm run dev、长构建等不会很快退出的命令必须用 bash_bg 后台运行,再用 bg_output 确认启动成功;用完记得 bg_kill。
- 遇到不认识的报错、需要查库/API 文档时,用 web_search / web_fetch 联网查证,不要凭空猜测。
- 任务较复杂时,先用 update_plan 列出待办步骤,推进中及时更新状态;需要用户拍板时用 ask_user 提问。
- 任务完成后,不要再调用工具,直接用简洁的中文总结你做了什么。
- 谨慎对待 write_file / edit_file / bash(它们会真实改动文件或执行命令)。${think}${doc}`;
  }
  return `You are Chaty's coding agent, working inside a workspace directory. Workspace root: ${workspace}

You can call these tools (all paths are relative to the workspace; escaping paths are rejected):
${TOOLS_DOC}

Rules (follow strictly):
- Call ONE tool at a time. To call it, output a single line <tool_call>{"name":"tool","arguments":{...}}</tool_call> and STOP immediately — nothing else in that message.
- You'll get the result as <tool_result>...</tool_result>, then continue.
- There is NO persistent working directory: every bash command starts a fresh shell at the workspace root, so a lone cd does NOT carry over. Use relative paths directly (ls src, read_file "src/app.ts") or combine in one command (cd src && npm test).
- Before editing, understand the code with read_file / grep / list_dir; after editing, you can run tests/builds with bash.
- Tool choice: create **new** files with ONE write_file; **modify existing** files with edit_file (one spot via old_string/new_string, or several at once via an atomic edits array) — do NOT rewrite an existing file wholesale with write_file to make a local change, that overwrites everything and easily drops content you didn't retype. Use write_file on an existing file only when you truly mean to replace the entire thing. Never split one change into many tiny steps.
- Commands that don't exit quickly (dev servers, npm run dev, long builds) MUST run via bash_bg; check they started with bg_output, and bg_kill them when done.
- For unfamiliar errors or library/API docs, verify with web_search / web_fetch instead of guessing.
- For non-trivial tasks, lay out a todo list with update_plan first and keep its statuses current as you go; use ask_user when a decision is the user's to make.
- When done, DON'T call a tool — just give a concise summary of what you did.
- Be careful with write_file / edit_file / bash (they really change files / run commands).${think}${doc}`;
}

/** Pull the first tool call out of model output. Tolerant of the closing tag
 *  being cut by the stop sequence, and of ```json fences. */
function parseToolCall(text: string): ToolCall | null {
  const open = text.indexOf("<tool_call>");
  if (open === -1) return null;
  let body = text.slice(open + "<tool_call>".length);
  const close = body.indexOf("</tool_call>");
  if (close !== -1) body = body.slice(0, close);
  body = body.trim().replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  // Grab the outermost {...}
  const s = body.indexOf("{");
  const e = body.lastIndexOf("}");
  if (s === -1 || e === -1 || e < s) return null;
  try {
    const obj = JSON.parse(body.slice(s, e + 1)) as Record<string, unknown>;
    if (obj && typeof obj.name === "string") {
      // Accept "arguments" or "parameters"; else treat the rest as the args.
      let args = obj.arguments ?? obj.parameters;
      if (!args || typeof args !== "object") {
        const { name: _n, ...rest } = obj;
        args = rest;
      }
      return { name: obj.name as AgentToolName, args: args as Record<string, unknown> };
    }
  } catch {
    /* malformed → handled by the caller (retry) */
  }
  return null;
}

/** Text to show as the assistant's prose (drop the tool-call markup + think). */
function proseOnly(text: string): string {
  const open = text.indexOf("<tool_call>");
  const visible = open === -1 ? text : text.slice(0, open);
  return stripThink(visible).trim();
}

/** Runaway-reasoning check: so far the output is *only* reasoning — no tool
 *  call and no real answer yet. Small models fall into this and keep thinking
 *  forever; gated behind a token budget so normal reasoning isn't cut short. */
function isThinkOnly(raw: string): boolean {
  if (raw.includes("<tool_call>")) return false;
  return proseAfter(raw).trim() === "";
}

/** Degenerate looping: a short chunk repeated ≥3× back-to-back at the tail
 *  ("I need to check… I need to check… I need to check…"). Very unlikely in
 *  legitimate prose or code, so a positive is a reliable stuck signal. */
function isDegenerateRepeat(raw: string): boolean {
  return /([\s\S]{16,90}?)\1\1/.test(raw.slice(-320));
}

const asStr = (v: unknown): string => (typeof v === "string" ? v : v == null ? "" : String(v));

// Models sometimes name arguments differently (path/file_path/filename…) —
// normalize the common aliases instead of failing with a confusing OS error.
export const argPath = (a: Record<string, unknown>): string =>
  asStr(a.path ?? a.file_path ?? a.filename ?? a.file);
export const argContent = (a: Record<string, unknown>): string =>
  asStr(a.content ?? a.text ?? a.contents ?? a.body ?? a.file_text);
export const argOld = (a: Record<string, unknown>): string =>
  asStr(a.old_string ?? a.old_str ?? a.old ?? a.search ?? a.from);
export const argNew = (a: Record<string, unknown>): string =>
  asStr(a.new_string ?? a.new_str ?? a.new ?? a.replace ?? a.to);
/** multi_edit's edits array, with per-item aliases normalized. */
export const argEdits = (a: Record<string, unknown>): EditOp[] => {
  const raw = a.edits ?? a.changes ?? a.replacements;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((e): e is Record<string, unknown> => !!e && typeof e === "object")
    .map((e) => ({
      old_string: argOld(e),
      new_string: argNew(e),
      replace_all: e.replace_all === true,
    }));
};

const MISSING_PATH =
  'ERROR: 缺少 "path" 参数(文件路径)。请重新调用并在 arguments 中带上 "path"。(Missing "path" — re-issue the tool call with a "path" argument.)';

/** Normalize the model's update_plan args into a clean PlanItem[]. */
function parsePlan(args: Record<string, unknown>): PlanItem[] {
  const raw = Array.isArray(args.todos)
    ? args.todos
    : Array.isArray(args.plan)
      ? args.plan
      : Array.isArray(args.items)
        ? args.items
        : [];
  const out: PlanItem[] = [];
  for (const it of raw as unknown[]) {
    if (typeof it === "string") {
      out.push({ content: it, status: "pending" });
      continue;
    }
    if (it && typeof it === "object") {
      const o = it as Record<string, unknown>;
      const content = asStr(o.content ?? o.text ?? o.task ?? o.title);
      if (!content) continue;
      const st = asStr(o.status).toLowerCase();
      const status: PlanStatus =
        st === "done" || st === "completed" || st === "complete"
          ? "done"
          : st === "in_progress" || st === "in-progress" || st === "active" || st === "doing"
            ? "in_progress"
            : "pending";
      out.push({ content, status });
    }
  }
  return out;
}
const asNum = (v: unknown): number | undefined =>
  typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : undefined;

/** Read a file's FULL content for a diff snapshot — no pagination footer, no
 *  truncation (up to the Rust byte cap). The model-facing read is budgeted for
 *  context; diffs need the whole file so the +N/−M count and the rendered hunks
 *  are correct even for large files. */
function readFull(path: string): Promise<string> {
  return agentReadFile(path, undefined, undefined, 400_000);
}

/** Execute a tool call → a text result for the model, plus optional diff data. */
async function execTool(
  call: ToolCall,
  bashTimeout?: number,
  readChars?: number,
): Promise<{ result: string; diff?: ToolStep["diff"] }> {
  const a = call.args;
  switch (call.name) {
    case "read_file": {
      const path = argPath(a);
      if (!path) return { result: MISSING_PATH };
      return { result: await agentReadFile(path, asNum(a.offset), asNum(a.limit), readChars) };
    }
    case "list_dir": {
      const entries = await agentListDir(a.path ? asStr(a.path) : undefined);
      const body =
        entries.map((e) => `${e.isDir ? "📁 " : "📄 "}${e.name}${e.isDir ? "/" : ""}`).join("\n") ||
        "(空目录 / empty)";
      return { result: body };
    }
    case "glob": {
      const hits = await agentGlob(asStr(a.pattern));
      return { result: hits.length ? hits.join("\n") : "(无匹配 / no matches)" };
    }
    case "grep":
      return {
        result: await agentGrep(
          asStr(a.pattern),
          a.path ? asStr(a.path) : undefined,
          a.glob ? asStr(a.glob) : undefined,
        ),
      };
    case "search_files": {
      const q = asStr(a.query);
      if (!q) return { result: 'ERROR: 缺少 "query" 参数 (missing "query")' };
      return {
        result: await agentSearchFiles(
          q,
          a.path ? asStr(a.path) : undefined,
          a.names_only === true || a.namesOnly === true,
        ),
      };
    }
    case "search_code": {
      const q = asStr(a.query);
      if (!q) return { result: 'ERROR: 缺少 "query" 参数 (missing "query")' };
      const hits = await agentSearchCode(q, asNum(a.k));
      if (!hits.length) return { result: "(没有匹配的代码 / no matches)" };
      return {
        result: hits
          .map((h) => `── ${h.path}:${h.line} ──\n${h.snippet}`)
          .join("\n\n"),
      };
    }
    case "search_docs": {
      const q = asStr(a.query);
      if (!q) return { result: 'ERROR: 缺少 "query" 参数 (missing "query")' };
      try {
        const hits = await ragSearch(q, 6);
        if (!hits.length) return { result: "(知识库中没有相关内容 / nothing relevant in the knowledge base)" };
        return {
          result: hits.map((h) => `── ${h.docName} ──\n${h.text}`).join("\n\n"),
        };
      } catch (e) {
        return { result: `知识库不可用 (knowledge base unavailable): ${e instanceof Error ? e.message : String(e)}` };
      }
    }
    case "write_file": {
      const path = argPath(a);
      if (!path) return { result: MISSING_PATH };
      let before = "";
      try {
        before = await readFull(path);
      } catch {
        /* new file */
      }
      const after = argContent(a);
      // Guardrail: a small change to an existing sizable file should be an
      // edit, not a full rewrite. write_file overwrites everything, so when the
      // model regenerates a big file just to tweak a few lines it risks dropping
      // content it didn't retype. Intercept the clear cases and steer to edit.
      if (before) {
        const oldLines = before.split("\n").length;
        const { added, removed } = diffLines(before, after);
        const changed = added + removed;
        if (oldLines >= 40 && changed > 0 && changed < oldLines * 0.5) {
          return {
            result:
              `未写入 (not written)。这是对已有文件的局部改动(约 ${changed} 行,文件共 ${oldLines} 行)——请改用 edit_file(改一处给 old_string/new_string,改多处给 edits 数组)精确替换。` +
              `不要用 write_file 整体重写来做小改动:它会覆盖全文,容易丢失你没重写的内容。` +
              ` (This is a partial change to an existing file — use edit_file instead of a full write_file rewrite, which can drop content you didn't retype.)`,
          };
        }
      }
      const result = await agentWriteFile(path, after);
      return { result, diff: { path, before, after } };
    }
    // One edit tool: a single replacement (old_string/new_string) OR several
    // at once (edits array) — both applied atomically. `multi_edit` is kept as
    // a tolerated alias for models that still emit it.
    case "edit_file":
    case "multi_edit": {
      const path = argPath(a);
      if (!path) return { result: MISSING_PATH };
      const edits = argEdits(a);
      let before = "";
      try {
        before = await readFull(path);
      } catch {
        /* edit will re-fail with a clear message */
      }
      const result =
        edits.length > 0
          ? await agentMultiEdit(path, edits)
          : await agentEditFile(path, argOld(a), argNew(a), a.replace_all === true);
      let after = before;
      try {
        after = await readFull(path);
      } catch {
        /* ignore */
      }
      return { result, diff: { path, before, after } };
    }
    case "outline": {
      const path = argPath(a);
      if (!path) return { result: MISSING_PATH };
      return { result: await agentOutline(path) };
    }
    case "bash": {
      const cmd = asStr(a.command).trim();
      // A lone `cd` can't work — there is no persistent shell. Catch it before
      // wasting a real execution and tell the model what to do instead.
      if (/^cd\s+[^;&|()<>]+$/.test(cmd)) {
        return {
          result:
            "提示:没有持久的工作目录,单独的 cd 不会保留到下一条命令。请直接用相对路径(如 ls src、read_file \"src/app.ts\"),或在同一条命令内组合:cd 子目录 && 你的命令。(No persistent cwd — combine `cd dir && cmd` in one command, or just use relative paths.)",
        };
      }
      const r = await agentBash(cmd, asNum(a.timeout_secs) ?? bashTimeout);
      const parts: string[] = [];
      if (r.stdout) parts.push(r.stdout);
      if (r.stderr) parts.push(`[stderr]\n${r.stderr}`);
      parts.push(`[exit ${r.code}${r.timedOut ? " · 超时/timed out" : ""}]`);
      return { result: parts.join("\n") };
    }
    case "bash_bg": {
      const id = await agentBashBg(asStr(a.command));
      return {
        result: `后台命令已启动 (background job started): #${id}。结束时会自动通知你;可用 bg_output 查看进度。`,
      };
    }
    case "bg_output": {
      const info = await agentBgOutput(Number(a.id));
      const head = info.running
        ? `#${info.id} 运行中 (running, ${info.elapsedSecs}s): ${info.command}`
        : `#${info.id} 已结束 (finished, exit ${info.code}): ${info.command}`;
      return { result: `${head}\n--- 最近输出 (recent output) ---\n${info.tail || "(无输出 / no output yet)"}` };
    }
    case "bg_kill":
      return { result: await agentBgKill(Number(a.id)) };
    case "web_search": {
      const q = asStr(a.query);
      if (!q) return { result: 'ERROR: 缺少 "query" 参数 (missing "query")' };
      const site = asStr(a.site);
      if (site) {
        const hits = await siteSearch(site, q);
        if (!hits.length) return { result: "(没有搜索结果 / no results)" };
        return {
          result: hits
            .slice(0, 16)
            .map((h, i) => `${i + 1}. [${h.kind}] ${h.title}\n   ${h.url}\n   ${h.snippet}`)
            .join("\n"),
        };
      }
      const hits = await webSearch(q);
      if (!hits.length) return { result: "(没有搜索结果 / no results)" };
      return {
        result: hits
          .slice(0, 8)
          .map((h, i) => `${i + 1}. ${h.title}\n   ${h.url}\n   ${h.snippet}`)
          .join("\n"),
      };
    }
    case "web_fetch": {
      const url = asStr(a.url);
      if (!url) return { result: 'ERROR: 缺少 "url" 参数 (missing "url")' };
      const raw = a.raw === true || a.raw === "true";
      const p = await fetchPageEx(url, raw);
      const parts: string[] = [];
      parts.push(`${p.title ? p.title + "\n" : ""}${p.url} [${p.kind}${p.truncated ? ", 已截断/truncated" : ""}]`);
      parts.push("");
      parts.push(p.text);
      if (p.links.length) {
        parts.push("");
        parts.push("— 页面链接 (links on this page, fetch to go deeper) —");
        parts.push(p.links.map((l) => `- ${l.text ? l.text + " — " : ""}${l.url}`).join("\n"));
      }
      if (p.images.length) {
        parts.push("");
        parts.push("— 图片 (images, save with web_download) —");
        parts.push(p.images.map((u) => `- ${u}`).join("\n"));
      }
      return { result: parts.join("\n") };
    }
    case "web_download": {
      const url = asStr(a.url);
      const path = asStr(a.path) || asStr(a.file_path) || asStr(a.dest);
      if (!url) return { result: 'ERROR: 缺少 "url" 参数 (missing "url")' };
      if (!path) return { result: 'ERROR: 缺少 "path" 参数 (missing "path")' };
      return { result: await agentWebDownload(url, path) };
    }
    default:
      return { result: `未知工具 (unknown tool): ${call.name}` };
  }
}

function toolResultMsg(name: string, content: string): string {
  // read_file sizes itself in Rust from the model's real context window (plus
  // an actionable next-offset footer) — never chop that off with a blind cap.
  const cap = name === "read_file" ? 400000 : name === "web_fetch" ? 48000 : 12000;
  let capped: string;
  if (content.length <= cap) {
    capped = content;
  } else if (name === "bash" || name === "bash_bg" || name === "bg_output") {
    // Command output: the failure is almost always at the END (panics, test
    // summaries, exit codes) — keep head AND tail instead of chopping the tail.
    const head = content.slice(0, Math.floor(cap * 0.3));
    const tail = content.slice(-Math.floor(cap * 0.7));
    capped = `${head}\n… (中间省略 ${content.length - head.length - tail.length} 字符 / middle omitted) …\n${tail}`;
  } else {
    capped = content.slice(0, cap) + "\n… (截断/truncated)";
  }
  return `<tool_result name="${name}">\n${capped}\n</tool_result>`;
}

/** Rough transcript size in tokens (mixed code/CJK ≈ 2.5 chars per token,
 *  plus a little chat-template overhead per message). */
function estimateTokens(messages: ChatMessage[]): number {
  return messages.reduce((n, m) => n + Math.ceil(m.content.length / 2.5) + 8, 0);
}

/** Auto-compaction: when the transcript nears the context window, elide the
 *  OLDEST tool results (they are the bulkiest and least useful verbatim) while
 *  keeping the most recent ones intact — Claude-Code-style compaction without
 *  spending a model round-trip. The system prompt, the task, and all assistant
 *  turns are never touched. */
function compactMessages(messages: ChatMessage[], nCtx: number): boolean {
  const limit = Math.floor(nCtx * 0.8);
  if (estimateTokens(messages) <= limit) return false;
  const results = messages
    .map((m, i) => ({ m, i }))
    .filter(({ m }) => m.role === "user" && m.content.startsWith("<tool_result"));
  const KEEP = 3; // most recent results stay verbatim
  let changed = false;
  for (let k = 0; k < results.length - KEEP; k++) {
    const { m, i } = results[k];
    if (m.content.length < 200) continue; // already tiny
    const name = /name="([^"]+)"/.exec(m.content)?.[1] ?? "tool";
    messages[i] = {
      role: "user",
      content: `<tool_result name="${name}">\n(较早的结果已被上下文压缩省略 / elided by context compaction)\n</tool_result>`,
    };
    changed = true;
    if (estimateTokens(messages) <= limit) break;
  }
  return changed;
}

/** Cross-turn compaction: if the prior conversation alone would eat too much of
 *  the window, keep only the most recent exchanges and note the elision. */
function trimHistory(history: ChatMessage[], nCtx: number): { history: ChatMessage[]; trimmed: boolean } {
  const budget = Math.floor(nCtx * 0.4);
  if (estimateTokens(history) <= budget) return { history, trimmed: false };
  const kept = [...history];
  while (kept.length > 2 && estimateTokens(kept) > budget) {
    kept.shift();
  }
  // Never start the kept slice mid-exchange with an assistant message.
  while (kept.length && kept[0].role === "assistant") kept.shift();
  kept.unshift({
    role: "user",
    content:
      "(提示:更早的对话内容较长,已被自动压缩省略,以下是最近的部分。/ Note: earlier conversation was auto-compacted; what follows is the most recent part.)",
  });
  return { history: kept, trimmed: true };
}

/** Run one user turn to completion (possibly many tool steps). `history` is the
 *  prior conversation as plain chat messages. */
export async function runAgentTurn(
  userInput: string,
  history: ChatMessage[],
  workspace: string,
  lang: "zh" | "en",
  opts: AgentOptions,
  cb: AgentCallbacks,
): Promise<void> {
  const maxSteps = opts.maxSteps ?? 32;
  // Thinking control mirrors chat mode's per-model mechanisms:
  //  • Qwen3 (`thinkSwitch`): append the `/no_think` soft switch to user turns.
  //  • Switch-less reasoning models (Qwen3.5+ / Gemma): drive the think flag
  //    (true = reason, false = pre-fill an empty think block).
  //  • Models without model info fall back to the old flag-only behavior.
  const wantNoThink = opts.thinkMode === "off";
  const think =
    opts.supportsThinking === undefined
      ? wantNoThink
        ? false
        : undefined
      : opts.supportsThinking && !opts.thinkSwitch
        ? !wantNoThink
        : undefined;
  const noThinkSuffix = wantNoThink && opts.thinkSwitch ? "\n/no_think" : "";
  // A generous token budget so a long reasoning block can't bury the tool call,
  // but never so large that generation crowds the prompt out of the window.
  const nCtx = opts.nCtx ?? 8192;
  const budget = opts.thinkMode === "deep" ? 8192 : opts.thinkMode === "normal" ? 6144 : 4096;
  const maxTokens = Math.min(budget, Math.max(1024, Math.floor(nCtx * 0.75)));
  // Think gate: uninterrupted reasoning past this many tokens with no tool call
  // and no answer = the model is looping in its own head. Deep mode gets more
  // headroom; still far below maxTokens so a genuine long reason isn't cut.
  const thinkCap = opts.thinkMode === "deep" ? 5000 : 3000;
  // read_file budget: use most of the real context window for one read so even
  // long files come back in a single call (the #1 agent frustration). We leave
  // ~5k tokens of headroom for the system prompt + room to act, then ~3 chars/
  // token; compaction reclaims the space on later steps. Small-context models
  // get a proportionally smaller (safe) budget; big ones read up to ~384 KB.
  const readChars = Math.min(384000, Math.max(8000, Math.floor((nCtx - 5000) * 3)));
  const { history: keptHistory, trimmed } = trimHistory(history, nCtx);
  const messages: ChatMessage[] = [
    { role: "system", content: systemPrompt(workspace, lang === "zh", opts.thinkMode, opts.projectDoc) },
    ...keptHistory,
    { role: "user", content: userInput + noThinkSuffix },
  ];

  // Every user-role turn (tool results, nudges) carries the soft switch too,
  // since the model reads the LAST user message when deciding to think.
  const pushUser = (content: string) =>
    messages.push({ role: "user", content: content + noThinkSuffix });

  let baseTokens = 0; // tokens from completed steps this turn
  let lastTps = 0; // last trustworthy tokens/sec (engine-reported or warmed-up live)
  // Loop breaker: fingerprint of the last tool call, to catch a model repeating
  // the exact same call (e.g. `ls .` forever). Escalation: 2nd identical call
  // is intercepted (not executed) + the next step samples hotter to break the
  // pattern attractor; 3rd pauses the turn for the user.
  let lastCallKey = "";
  let repeatCount = 0;
  let hotNext = false;
  // Think gate state: consecutive stuck-thinking steps, and a one-shot flag to
  // physically disable reasoning on the recovery step.
  let stuckThinkCount = 0;
  let forceNoThinkNext = false;
  let compactNotified = false;
  const noteCompacted = () => {
    if (!compactNotified) {
      compactNotified = true;
      cb.onCompacted?.();
    }
  };
  if (trimmed) noteCompacted();

  try {
    for (let step = 0; step < maxSteps; step++) {
      if (opts.signal.cancelled) return;

      // Background commands that finished since the last step → tell the model
      // (and show a completion card), then it can react on this very step.
      try {
        for (const j of await agentBgReap()) {
          const head =
            lang === "zh"
              ? `后台命令 #${j.id} 已结束 (exit ${j.code}): ${j.command}`
              : `Background job #${j.id} finished (exit ${j.code}): ${j.command}`;
          cb.onStep({
            id: uid(),
            call: { name: "bash_bg", args: { command: j.command, id: j.id } },
            status: j.code === 0 ? "done" : "error",
            result: `${head}\n${j.tail}`,
          });
          pushUser(toolResultMsg("bash_bg", `${head}\n--- 输出 (output tail) ---\n${j.tail}`));
        }
      } catch {
        /* no workspace yet — nothing to reap */
      }

      // keep the running transcript inside the context window
      if (compactMessages(messages, nCtx)) noteCompacted();

      let raw = "";
      let liveTokens = 0;
      let thinkGateTripped = false;
      const t0 = performance.now();
      // After an intercepted repeat, sample hotter once to escape the pattern.
      const stepTemp = hotNext ? 0.7 : 0.3;
      hotNext = false;
      // Recovery step after the think gate: reasoning off so the model MUST act.
      const stepThink = forceNoThinkNext ? false : think;
      forceNoThinkNext = false;
      await generate(
        {
          messages,
          params: {
            temperature: stepTemp,
            topP: 0.9,
            maxTokens,
            repeatPenalty: 1.05,
            stop: ["</tool_call>"],
            think: stepThink,
          },
        },
        (ev) => {
          if (ev.type === "token") {
            raw += ev.text;
            liveTokens++;
            // The live rate is meaningless for the first fraction of a second
            // (1 token / ~0ms ⇒ absurd tok/s) — hold the last real value until
            // the measurement has warmed up.
            const secs = (performance.now() - t0) / 1000;
            if (secs >= 0.35) lastTps = liveTokens / secs;
            cb.onStats?.(baseTokens + liveTokens, lastTps);
            cb.onThinking(thinkPart(raw));
            cb.onAssistantText(proseAfter(raw));
            // ── Think gate (mid-stream) ── stop a runaway before it fills the
            // whole budget: too much uninterrupted reasoning with no output, or
            // degenerate looping. Checked periodically to stay cheap.
            if (!thinkGateTripped && liveTokens % 48 === 0) {
              const runaway = liveTokens > thinkCap && isThinkOnly(raw);
              const looping = liveTokens > 400 && isThinkOnly(raw) && isDegenerateRepeat(raw);
              if (runaway || looping) {
                thinkGateTripped = true;
                void cancelGeneration().catch(() => {});
              }
            }
          } else if (ev.type === "done") {
            baseTokens += ev.stats.completionTokens;
            lastTps = ev.stats.tokensPerSecond;
            cb.onStats?.(baseTokens, lastTps);
            // prompt + this step's output ≈ current position in the context window
            cb.onContext?.(ev.stats.promptTokens + ev.stats.completionTokens);
          }
        },
      );
      if (opts.signal.cancelled) return;
      const thinking = thinkPart(raw);

      const call = parseToolCall(raw);
      if (!call) {
        const answer = proseAfter(raw).trim() || stripThink(raw).trim();
        // ── Think gate (post-generation) ── the step produced only reasoning:
        // either we cut a runaway mid-stream, or it finished with no tool call
        // and an empty answer despite thinking. Don't accept that as "done" —
        // force reasoning off, sample hotter, and demand an action. Bounded so
        // the recovery itself can't spin: a 3rd stuck step pauses for the user.
        const stuckThinking = thinkGateTripped || (answer === "" && thinking.trim() !== "");
        if (stuckThinking && step < maxSteps - 1) {
          stuckThinkCount++;
          if (stuckThinkCount >= 3) {
            cb.onFinal(
              lang === "zh"
                ? "模型连续陷入思考循环、迟迟没有产出结果,已暂停以免空转。点「继续」重试,或把任务拆得更具体一些。"
                : 'The model kept looping in its own reasoning without producing anything — paused to avoid spinning. Hit "Continue" to retry, or break the task into more concrete steps.',
              undefined,
              "steps",
            );
            return;
          }
          // Break the attractor on the next step: reasoning physically off
          // (empty think prefill for flag models + /no_think for switch models),
          // hotter sampling, and a firm instruction to act now. Don't feed the
          // runaway reasoning back into context — just a short marker.
          forceNoThinkNext = true;
          hotNext = true;
          messages.push({ role: "assistant", content: proseOnly(raw).slice(0, 300) });
          const stopSuffix = opts.thinkSwitch ? "\n/no_think" : "";
          messages.push({
            role: "user",
            content:
              (lang === "zh"
                ? "停止思考。你已经反复推理却没有产出任何结果。现在立刻二选一:要么输出一行 <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call> 执行一个具体动作,要么直接给出简短的最终答案。不要再写任何思考过程。"
                : "Stop thinking. You have been reasoning in circles without producing anything. Right now, do ONE of two things: output a single line <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call> to take a concrete action, or give a short final answer directly. Do not write any more reasoning.") +
              stopSuffix,
          });
          cb.onThinking("");
          cb.onAssistantText("");
          continue;
        }
        // A `<tool_call>` was attempted but couldn't be parsed → don't leak the
        // raw markup into the answer; nudge the model to re-emit valid JSON.
        // (Bounded by maxSteps.) Otherwise it's a genuine final answer.
        if (raw.includes("<tool_call>") && step < maxSteps - 1) {
          messages.push({ role: "assistant", content: proseOnly(raw) });
          pushUser(
            lang === "zh"
              ? '你上一个工具调用的格式无效。请严格用一行 <tool_call>{"name":"...","arguments":{...}}</tool_call> 重新调用。'
              : 'Your last tool call was not valid. Re-issue it as exactly one line: <tool_call>{"name":"...","arguments":{...}}</tool_call>.',
          );
          continue;
        }
        cb.onFinal(answer, thinking);
        return;
      }
      // A valid tool call = real progress; clear the stuck-thinking streak.
      stuckThinkCount = 0;

      // Record the assistant turn (its reasoning + the tool call, tag restored).
      const withClose = raw.includes("</tool_call>") ? raw : `${raw}</tool_call>`;
      messages.push({ role: "assistant", content: stripThink(withClose).trim() });

      const stepObj: ToolStep = { id: uid(), call, status: "running", thinking };

      // ── Loop breaker: identical call to the previous one? ──
      const callKey = `${call.name}:${JSON.stringify(call.args)}`;
      if (callKey === lastCallKey) {
        repeatCount++;
      } else {
        lastCallKey = callKey;
        repeatCount = 0;
      }
      if (repeatCount >= 2) {
        // Third identical call — pause instead of spinning to the step limit.
        cb.onFinal(
          lang === "zh"
            ? `检测到模型连续 ${repeatCount + 1} 次发出完全相同的调用,已暂停以免空转。可点「继续」重试,或换一种说法明确指出要看的子目录/文件。`
            : `The model issued the exact same call ${repeatCount + 1} times in a row — paused to avoid spinning. Hit "Continue" to retry, or rephrase with the specific subdirectory/file to look at.`,
          undefined,
          "steps",
        );
        return;
      }
      if (repeatCount === 1) {
        // Second identical call — intercept without executing, teach, and let
        // the next generation sample hotter to break the attractor.
        hotNext = true;
        const note =
          lang === "zh"
            ? "调用被拦截:这和上一步完全相同,结果不会变化。请换一种做法——传入具体的子目录/文件路径(如 list_dir {\"path\":\"src\"}、read_file \"src/app.ts\")、换个工具,或用 update_plan 重新梳理。提醒:没有持久的工作目录,cd 不会保留。"
            : 'Intercepted: this call is identical to the previous one — the result cannot change. Do something different: pass a concrete subdirectory/file path (list_dir {"path":"src"}, read_file "src/app.ts"), use another tool, or re-plan with update_plan. Reminder: there is no persistent cwd.';
        stepObj.status = "error";
        stepObj.result = note;
        cb.onStep(stepObj);
        pushUser(toolResultMsg(call.name, note));
        continue;
      }

      // ── Meta-tools handled in the loop (no backend call, no approval) ──
      // update_plan renders as a dedicated live plan panel, not a step card.
      if (call.name === "update_plan") {
        const todos = parsePlan(call.args);
        cb.onPlan?.(todos);
        pushUser(toolResultMsg("update_plan", lang === "zh" ? "计划已更新。" : "Plan updated."));
        continue;
      }
      if (call.name === "ask_user") {
        const question = asStr(call.args.question);
        const options = Array.isArray(call.args.options)
          ? (call.args.options as unknown[]).map(asStr).filter(Boolean)
          : [];
        cb.onStep(stepObj);
        const choice = cb.onAskUser
          ? await cb.onAskUser(question, options)
          : options[0] ?? "";
        if (opts.signal.cancelled) return;
        stepObj.status = "done";
        stepObj.result = (lang === "zh" ? "用户选择:" : "User chose: ") + choice;
        cb.onStep(stepObj);
        pushUser(
          toolResultMsg("ask_user", (lang === "zh" ? "用户的选择是:" : "The user chose: ") + choice),
        );
        continue;
      }

      // Approval gate for mutating tools.
      if (MUTATING_TOOLS.has(call.name)) {
        const ok = await opts.approve(call);
        if (opts.signal.cancelled) return;
        if (!ok) {
          stepObj.status = "denied";
          stepObj.result = lang === "zh" ? "已被用户拒绝" : "denied by the user";
          cb.onStep(stepObj);
          pushUser(
            toolResultMsg(
              call.name,
              lang === "zh"
                ? "用户拒绝了此操作。请调整方案或询问用户。"
                : "The user denied this action. Adjust the plan or ask the user.",
            ),
          );
          continue;
        }
      }

      cb.onStep(stepObj);
      let resultText: string;
      try {
        const out = await execTool(call, opts.bashTimeout, readChars);
        resultText = out.result;
        stepObj.status = "done";
        stepObj.result = out.result;
        stepObj.diff = out.diff;
      } catch (e) {
        resultText = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
        stepObj.status = "error";
        stepObj.result = resultText;
      }
      if (opts.signal.cancelled) return;
      cb.onStep(stepObj);
      pushUser(toolResultMsg(call.name, resultText));
    }
    cb.onFinal(
      lang === "zh"
        ? `已达到本轮 ${maxSteps} 步上限,先暂停。点击「继续」可接着当前进度做;也可在 设置 → Code 调高上限。`
        : `Paused at the ${maxSteps}-step limit for this turn. Hit "Continue" to pick up where it left off, or raise the limit in Settings → Code.`,
      undefined,
      "steps",
    );
  } catch (e) {
    if (!opts.signal.cancelled) cb.onError(e instanceof Error ? e.message : String(e));
  }
}
