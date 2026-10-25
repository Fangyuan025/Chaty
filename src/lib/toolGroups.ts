/** Code mode's collapsed view of a turn (Settings → Code → collapse tool
 *  calls): the steps between two things the model said fold into one line —
 *  what it is doing while they run, what it did once they are done ("Read 3
 *  files, ran 2 commands"), and the cards themselves when opened. This module
 *  is the shape of that view; the components draw it. */

import type { ToolStep } from "./agentLoop";

/** What a turn shows, in order: the model's words, and the runs of tool
 *  steps between them. */
export type Segment =
  | { kind: "say"; key: string; text: string }
  | { kind: "tools"; key: string; steps: ToolStep[] };

/** Words said with a step's call (`step.said`) or with a call that has no
 *  card (`said`, after how many steps) end one run of steps and open the
 *  next. */
export function segmentsOf(steps: ToolStep[], said: { after: number; text: string }[] = []): Segment[] {
  const out: Segment[] = [];
  let run: ToolStep[] = [];
  const close = () => {
    if (run.length) out.push({ kind: "tools", key: run[0].id, steps: run });
    run = [];
  };
  steps.forEach((s, i) => {
    const before = said.filter((w) => w.after === i);
    if (before.length || s.said) {
      close();
      before.forEach((w, j) => out.push({ kind: "say", key: `said-${i}-${j}`, text: w.text }));
      if (s.said) out.push({ kind: "say", key: `said-${s.id}`, text: s.said });
    }
    run.push(s);
  });
  close();
  said
    .filter((w) => w.after >= steps.length)
    .forEach((w, j) => out.push({ kind: "say", key: `said-end-${j}`, text: w.text }));
  return out;
}

/** The kinds of work a run of steps is summed up as. */
export type WorkKind =
  | "read"
  | "create"
  | "edit"
  | "command"
  | "search"
  | "websearch"
  | "fetch"
  | "browser"
  | "validate"
  | "image"
  | "other";

const KIND: Record<string, WorkKind> = {
  read_file: "read",
  multi_read: "read",
  write_file: "edit",
  edit_file: "edit",
  multi_edit: "edit",
  edit_lines: "edit",
  bash: "command",
  bash_bg: "command",
  bg_input: "command",
  grep: "search",
  glob: "search",
  list_dir: "search",
  outline: "search",
  search_code: "search",
  search_files: "search",
  search_docs: "search",
  search_history: "search",
  read_history: "search",
  understand_repo: "search",
  web_search: "websearch",
  web_fetch: "fetch",
  validate_change: "validate",
  view_image: "image",
};

export function kindOf(name: string): WorkKind {
  if (name.startsWith("browser_")) return "browser";
  return KIND[name] ?? "other";
}

/** The paths a file step touches — several for a multi_read. */
function pathsOf(step: ToolStep): string[] {
  const a = step.call.args ?? {};
  const list = a.paths ?? a.files ?? a.file_paths ?? a.filenames;
  if (Array.isArray(list)) return list.map(String);
  const one = a.path ?? a.file_path ?? a.file ?? a.filename;
  return typeof one === "string" && one ? [one] : [];
}

export interface WorkItem {
  kind: WorkKind;
  n: number;
  /** Lines a created or edited file gained and lost, over the run. */
  added?: number;
  removed?: number;
}

/** Lines a step's diff adds and removes. */
export type DiffCounter = (step: ToolStep) => { added: number; removed: number } | null;

const FILE_KINDS: WorkKind[] = ["read", "create", "edit"];

/** What a run of steps did, kind by kind in the order each first came up.
 *  Files are counted once however often they were touched, and only where
 *  the step went through — a file written where none was is created, any
 *  other write or edit an edit, each with the lines it gained and lost.
 *  Everything else counts its calls. A run whose steps all failed sums up
 *  as the tools it used. */
export function workSummary(steps: ToolStep[], countDiff?: DiffCounter): WorkItem[] {
  const order: WorkKind[] = [];
  const files = new Map<WorkKind, Set<string>>();
  const calls = new Map<WorkKind, number>();
  const lines = new Map<WorkKind, { added: number; removed: number }>();
  for (const s of steps) {
    // Not run — the user said no, or plan mode held it back: counted apart.
    if (s.status === "denied") continue;
    let k = kindOf(s.call.name);
    if (FILE_KINDS.includes(k)) {
      if (s.status !== "done") continue;
      if (s.call.name === "write_file" && s.diff && s.diff.before === "") k = "create";
      const set = files.get(k) ?? new Set<string>();
      for (const p of pathsOf(s)) set.add(p);
      files.set(k, set);
      const d = k === "read" ? null : countDiff?.(s);
      if (d) {
        const sum = lines.get(k) ?? { added: 0, removed: 0 };
        lines.set(k, { added: sum.added + d.added, removed: sum.removed + d.removed });
      }
    }
    if (!order.includes(k)) order.push(k);
    calls.set(k, (calls.get(k) ?? 0) + 1);
  }
  const out: WorkItem[] = order.map((k) => {
    const distinct = files.get(k)?.size ?? 0;
    return { kind: k, n: distinct > 0 ? distinct : (calls.get(k) ?? 0), ...(lines.get(k) ?? {}) };
  });
  const ran = steps.filter((s) => s.status !== "denied").length;
  return out.length ? out : ran ? [{ kind: "other", n: ran }] : [];
}

export function deniedCount(steps: ToolStep[]): number {
  return steps.filter((s) => s.status === "denied").length;
}

/** A step that failed: one that errored, or a command that ran and exited
 *  non-zero — its result went back to the model, but it did not work. */
export function stepFailed(s: ToolStep): boolean {
  if (s.status === "error") return true;
  if (s.status !== "done" || s.diff) return false;
  const m = (s.result ?? "").match(/\[exit (-?\d+)/);
  return !!m && m[1] !== "0";
}

export function failedCount(steps: ToolStep[]): number {
  return steps.filter(stepFailed).length;
}

/** What a running step is doing, as a key for the line that says so and the
 *  object of the verb (a path, a command, a query) when there is one — each
 *  tool its own words, so the line says what is going on, not just that a
 *  tool is. */
export function activityOf(step: ToolStep): { key: string; x?: string } {
  const a = (step.call.args ?? {}) as Record<string, unknown>;
  const str = (v: unknown) => (typeof v === "string" ? v.trim() : "");
  const first = (...vs: unknown[]) => vs.map(str).find(Boolean) ?? "";
  const path = pathsOf(step).join(", ");
  const name = step.call.name as string;
  switch (name) {
    case "read_file":
    case "multi_read":
      return { key: "actReading", x: path };
    case "write_file":
      return { key: "actWriting", x: path };
    case "edit_file":
    case "multi_edit":
    case "edit_lines":
      return { key: "actEditing", x: path };
    case "outline":
      return { key: "actOutline", x: path };
    case "list_dir": {
      const dir = first(a.path).replace(/^\.\/?$/, "");
      return dir ? { key: "actListing", x: dir } : { key: "actListingRoot" };
    }
    case "glob":
    case "search_files":
      return { key: "actFinding", x: first(a.pattern, a.query, a.glob) };
    case "grep":
      return { key: "actSearching", x: first(a.pattern, a.query) };
    case "search_code":
      return { key: "actSearchCode", x: first(a.query) };
    case "search_docs":
      return { key: "actSearchDocs", x: first(a.query) };
    case "search_history":
      return { key: "actHistory", x: first(a.query) };
    case "read_history":
      return { key: "actReadHistory" };
    case "understand_repo":
      return { key: "actOrient" };
    case "bash":
      return { key: "actRunning", x: first(a.command, a.cmd) };
    case "bash_bg":
      return { key: "actRunningBg", x: first(a.command, a.cmd) };
    case "bg_output":
      return { key: "actBgOutput" };
    case "bg_input":
      return { key: "actBgInput" };
    case "bg_kill":
      return { key: "actBgKill" };
    case "validate_change":
      return { key: "actValidating" };
    case "web_search":
      return { key: "actWebSearch", x: first(a.query) };
    case "web_fetch":
      return { key: "actFetching", x: first(a.url) };
    case "web_download":
      return { key: "actDownloading", x: first(a.url) };
    case "update_plan":
      return { key: "actPlan" };
    case "ask_user":
      return { key: "actAsking" };
    case "view_image":
      return { key: "actViewing", x: path };
    case "use_skill":
      return { key: "actSkill", x: first(a.name, a.skill) };
    case "remember":
      return { key: "actRemember" };
    case "browser_navigate":
      return { key: "actOpening", x: first(a.url) };
    case "browser_tabs":
      return { key: "actTabs" };
    case "browser_refresh":
      return { key: "actReload" };
    case "browser_read":
      return { key: "actPageRead" };
    case "browser_screenshot":
    case "browser_snapshot":
      return { key: "actScreenshot" };
    case "browser_scroll":
      return { key: "actScroll" };
    case "browser_console":
      return { key: "actConsole" };
    case "browser_click":
      return { key: "actClicking", x: first(a.text, a.label, a.selector) };
    case "browser_type":
      return { key: "actPageType" };
    case "browser_key":
      return { key: "actKey", x: first(a.key, a.keys) };
    case "browser_eval":
      return { key: "actPageScript" };
    case "browser_close":
      return { key: "actBrowserClose" };
  }
  if (name.startsWith("browser_")) return { key: "actBrowsing" };
  // An MCP server's tool: `server__tool`.
  const mcp = name.indexOf("__");
  if (mcp > 0) return { key: "actMcp", x: `${name.slice(mcp + 2)} · ${name.slice(0, mcp)}` };
  return { key: "actTool", x: name };
}
