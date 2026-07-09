// The agentic-coding brain for Code mode. Reuses the local model via generate()
// and drives a tool loop: the model emits ONE <tool_call>{json}</tool_call>, we
// stop generation there, run the tool (confined + sandboxed on the Rust side),
// feed the result back as <tool_result>, and repeat until the model answers with
// no tool call. Works on any instruct model (no native function-calling needed);
// it degrades gracefully when the model doesn't follow the format.

import {
  agentBash,
  agentEditFile,
  agentGlob,
  agentGrep,
  agentListDir,
  agentReadFile,
  agentWriteFile,
  cancelGeneration,
  generate,
  type ChatMessage,
} from "./ipc";

export type AgentToolName =
  | "read_file"
  | "write_file"
  | "edit_file"
  | "list_dir"
  | "glob"
  | "grep"
  | "bash"
  | "ask_user"
  | "update_plan";

/** Tools that change the world (or run code) → need approval unless bypassed. */
export const MUTATING_TOOLS = new Set<AgentToolName>(["write_file", "edit_file", "bash"]);

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
  nCtx?: number;
  maxSteps?: number;
  /** Default timeout for bash commands (seconds) when the model doesn't set one. */
  bashTimeout?: number;
  signal: AgentSignal;
  /** Gate a mutating tool call. Return true to run, false to deny. Bypass mode
   *  passes a function that always resolves true. */
  approve: (call: ToolCall) => Promise<boolean>;
}

const uid = () => Math.random().toString(36).slice(2);

function stripThink(s: string): string {
  return s.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, "");
}

/** The reasoning inside `<think>…</think>` (or after `<think>` if not yet closed). */
function thinkPart(s: string): string {
  const o = s.indexOf("<think>");
  if (o === -1) return "";
  const after = s.slice(o + "<think>".length);
  const c = after.indexOf("</think>");
  return (c === -1 ? after : after.slice(0, c)).trim();
}

/** Prose after any think block and before any tool call. */
function proseAfter(s: string): string {
  let t = s;
  const c = t.indexOf("</think>");
  if (c !== -1) t = t.slice(c + "</think>".length);
  else if (t.includes("<think>")) return ""; // still thinking → no prose yet
  const tc = t.indexOf("<tool_call>");
  return (tc === -1 ? t : t.slice(0, tc)).trim();
}

const TOOLS_DOC = `
- read_file: 读取文件。args: { "path": string, "offset"?: number, "limit"?: number }
- write_file: 新建或覆盖文件。args: { "path": string, "content": string }
- edit_file: 精确替换文件中的一段文本(old_string 必须与文件内容逐字匹配且唯一,除非 replace_all=true)。args: { "path": string, "old_string": string, "new_string": string, "replace_all"?: boolean }
- list_dir: 列出目录一层内容。args: { "path"?: string }
- glob: 按通配符找文件(如 "src/**/*.ts")。args: { "pattern": string }
- grep: 用正则搜索文件内容。args: { "pattern": string, "path"?: string, "glob"?: string }
- bash: 在工作区里执行 shell 命令(macOS 沙箱,写限工作区)。args: { "command": string, "timeout_secs"?: number }
- update_plan: 制定或更新任务计划(待办清单),让用户看到你的推进步骤。开始复杂任务时先列计划,完成一步就把它标为 done、把下一步标为 in_progress。args: { "todos": [{ "content": string, "status": "pending"|"in_progress"|"done" }] }
- ask_user: 当遇到需要用户拍板的决策(方案分歧、需求不明、破坏性操作确认)时,向用户提一个选择题;不要自己乱猜。args: { "question": string, "options": string[] }`;

function systemPrompt(workspace: string, zh: boolean, mode: ThinkMode): string {
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
- 修改代码前,先用 read_file / grep / list_dir 了解现状;改完可用 bash 跑测试/构建验证。
- 步数宝贵:新建文件用 write_file 一次写入完整内容;同一文件的多处修改合并成一次 edit_file,或直接用 write_file 重写整个文件。不要把一个改动拆成许多细碎小步。
- 任务较复杂时,先用 update_plan 列出待办步骤,推进中及时更新状态;需要用户拍板时用 ask_user 提问。
- 任务完成后,不要再调用工具,直接用简洁的中文总结你做了什么。
- 谨慎对待 write_file / edit_file / bash(它们会真实改动文件或执行命令)。${think}`;
  }
  return `You are Chaty's coding agent, working inside a workspace directory. Workspace root: ${workspace}

You can call these tools (all paths are relative to the workspace; escaping paths are rejected):
${TOOLS_DOC}

Rules (follow strictly):
- Call ONE tool at a time. To call it, output a single line <tool_call>{"name":"tool","arguments":{...}}</tool_call> and STOP immediately — nothing else in that message.
- You'll get the result as <tool_result>...</tool_result>, then continue.
- Before editing, understand the code with read_file / grep / list_dir; after editing, you can run tests/builds with bash.
- Steps are precious: create new files with ONE write_file containing the complete content; merge multiple changes to the same file into one edit_file, or rewrite the whole file with write_file. Never split one change into many tiny steps.
- For non-trivial tasks, lay out a todo list with update_plan first and keep its statuses current as you go; use ask_user when a decision is the user's to make.
- When done, DON'T call a tool — just give a concise summary of what you did.
- Be careful with write_file / edit_file / bash (they really change files / run commands).${think}`;
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

/** Execute a tool call → a text result for the model, plus optional diff data. */
async function execTool(
  call: ToolCall,
  bashTimeout?: number,
): Promise<{ result: string; diff?: ToolStep["diff"] }> {
  const a = call.args;
  switch (call.name) {
    case "read_file": {
      const path = argPath(a);
      if (!path) return { result: MISSING_PATH };
      return { result: await agentReadFile(path, asNum(a.offset), asNum(a.limit)) };
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
    case "write_file": {
      const path = argPath(a);
      if (!path) return { result: MISSING_PATH };
      let before = "";
      try {
        before = await agentReadFile(path);
      } catch {
        /* new file */
      }
      const after = argContent(a);
      const result = await agentWriteFile(path, after);
      return { result, diff: { path, before, after } };
    }
    case "edit_file": {
      const path = argPath(a);
      if (!path) return { result: MISSING_PATH };
      let before = "";
      try {
        before = await agentReadFile(path);
      } catch {
        /* read will re-fail in edit with a clear message */
      }
      const result = await agentEditFile(
        path,
        argOld(a),
        argNew(a),
        a.replace_all === true,
      );
      let after = before;
      try {
        after = await agentReadFile(path);
      } catch {
        /* ignore */
      }
      return { result, diff: { path, before, after } };
    }
    case "bash": {
      const r = await agentBash(asStr(a.command), asNum(a.timeout_secs) ?? bashTimeout);
      const parts: string[] = [];
      if (r.stdout) parts.push(r.stdout);
      if (r.stderr) parts.push(`[stderr]\n${r.stderr}`);
      parts.push(`[exit ${r.code}${r.timedOut ? " · 超时/timed out" : ""}]`);
      return { result: parts.join("\n") };
    }
    default:
      return { result: `未知工具 (unknown tool): ${call.name}` };
  }
}

function toolResultMsg(name: string, content: string): string {
  const capped = content.length > 12000 ? content.slice(0, 12000) + "\n… (截断/truncated)" : content;
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
  // off → no-think (prefill empty <think>); normal/deep → let the model reason.
  // A generous token budget so a long reasoning block can't bury the tool call,
  // but never so large that generation crowds the prompt out of the window.
  const think = opts.thinkMode === "off" ? false : undefined;
  const nCtx = opts.nCtx ?? 8192;
  const budget = opts.thinkMode === "deep" ? 8192 : opts.thinkMode === "normal" ? 6144 : 4096;
  const maxTokens = Math.min(budget, Math.max(1024, Math.floor(nCtx * 0.75)));
  const { history: keptHistory, trimmed } = trimHistory(history, nCtx);
  const messages: ChatMessage[] = [
    { role: "system", content: systemPrompt(workspace, lang === "zh", opts.thinkMode) },
    ...keptHistory,
    { role: "user", content: userInput },
  ];

  let baseTokens = 0; // tokens from completed steps this turn
  let lastTps = 0; // last trustworthy tokens/sec (engine-reported or warmed-up live)
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
      // keep the running transcript inside the context window
      if (compactMessages(messages, nCtx)) noteCompacted();

      let raw = "";
      let liveTokens = 0;
      const t0 = performance.now();
      await generate(
        {
          messages,
          params: {
            temperature: 0.3,
            topP: 0.9,
            maxTokens,
            repeatPenalty: 1.05,
            stop: ["</tool_call>"],
            think,
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
        // A `<tool_call>` was attempted but couldn't be parsed → don't leak the
        // raw markup into the answer; nudge the model to re-emit valid JSON.
        // (Bounded by maxSteps.) Otherwise it's a genuine final answer.
        if (raw.includes("<tool_call>") && step < maxSteps - 1) {
          messages.push({ role: "assistant", content: proseOnly(raw) });
          messages.push({
            role: "user",
            content:
              lang === "zh"
                ? '你上一个工具调用的格式无效。请严格用一行 <tool_call>{"name":"...","arguments":{...}}</tool_call> 重新调用。'
                : 'Your last tool call was not valid. Re-issue it as exactly one line: <tool_call>{"name":"...","arguments":{...}}</tool_call>.',
          });
          continue;
        }
        cb.onFinal(proseAfter(raw) || stripThink(raw).trim(), thinking);
        return;
      }

      // Record the assistant turn (its reasoning + the tool call, tag restored).
      const withClose = raw.includes("</tool_call>") ? raw : `${raw}</tool_call>`;
      messages.push({ role: "assistant", content: stripThink(withClose).trim() });

      const stepObj: ToolStep = { id: uid(), call, status: "running", thinking };

      // ── Meta-tools handled in the loop (no backend call, no approval) ──
      // update_plan renders as a dedicated live plan panel, not a step card.
      if (call.name === "update_plan") {
        const todos = parsePlan(call.args);
        cb.onPlan?.(todos);
        messages.push({
          role: "user",
          content: toolResultMsg("update_plan", lang === "zh" ? "计划已更新。" : "Plan updated."),
        });
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
        messages.push({
          role: "user",
          content: toolResultMsg(
            "ask_user",
            (lang === "zh" ? "用户的选择是:" : "The user chose: ") + choice,
          ),
        });
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
          messages.push({
            role: "user",
            content: toolResultMsg(
              call.name,
              lang === "zh"
                ? "用户拒绝了此操作。请调整方案或询问用户。"
                : "The user denied this action. Adjust the plan or ask the user.",
            ),
          });
          continue;
        }
      }

      cb.onStep(stepObj);
      let resultText: string;
      try {
        const out = await execTool(call, opts.bashTimeout);
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
      messages.push({ role: "user", content: toolResultMsg(call.name, resultText) });
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
