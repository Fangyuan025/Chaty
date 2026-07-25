/** The single source of truth for agent tools (2.0 M0).
 *
 *  Every per-tool property that used to live in a parallel structure inside
 *  agentLoop.ts — the name union, the doc strings, MUTATING/REPEAT_EXEMPT/
 *  UNTRUSTED membership, required-args validation, result caps — is one field
 *  on one ToolSpec here, and the old structures are derived exports. The
 *  refactor is byte-exact by construction: promptGolden.test.ts holds the
 *  pre-refactor system prompt, toolRegistry.test.ts holds the pre-refactor
 *  set memberships.
 *
 *  This is also the seam MCP servers (M1) and skills (M3) plug into:
 *  registerTool() adds a dynamic spec; `tier: "deferred"` keeps a tool out of
 *  the always-loaded doc block so unlimited tools can ride behind a fixed
 *  context budget — nothing ships deferred yet, but the mechanism is live and
 *  unit-tested. */

import {
  ANCHOR_READ_NOTE,
  BROWSER_TAIL_TEXT,
  BROWSER_TAIL_VISION,
  DOC_LINES,
  type Bi,
} from "./toolDocs";

/** Where a tool comes from. Natives are compiled in; mcp/skill arrive at
 *  runtime via registerTool (M1/M3). */
export type ToolSource = "native" | "mcp" | "skill";

/** Coarse permission class, the vocabulary of the M1 permission UI.
 *  Orthogonal to `mutating` (the approval gate): web_download is network AND
 *  approval-gated; bg_kill is exec but approval-free. */
export type PermissionClass = "read" | "write" | "exec" | "network" | "ui";

/** core = doc line always in the system prompt; deferred = listed in a
 *  one-line index only, full doc loads on demand (the budget mechanism that
 *  will let MCP servers bring unlimited tools to a 16K model). */
export type LoadTier = "core" | "deferred";

export interface ToolSpec {
  name: string;
  source: ToolSource;
  /** Which doc block the line belongs to. */
  suite: "core" | "browser";
  perm: PermissionClass;
  tier: LoadTier;
  /** Model-visible one-line doc. Absent → executable but undocumented
   *  (multi_edit). */
  docLine?: Bi;
  /** Doc line exists but is swapped in only by anchor mode (edit_lines). */
  docHidden?: boolean;
  /** Vision models only — dropped from the text-browser suite. */
  visionOnly?: boolean;
  /** Changes the world (or runs code) → needs approval unless bypassed. */
  mutating?: boolean;
  /** Exempt from the identical-call loop breaker (same args still make
   *  progress or re-observe a changed world). */
  repeatExempt?: boolean;
  /** Output is untrusted external content → injection defense wraps it. */
  untrusted?: boolean;
  /** Required string args, validated before the call enters the record. */
  requiredArgs?: string[];
  /** Filled-in example for the missing-arg correction message. */
  argExample?: string;
  /** Result cap in chars (default 12000). */
  resultCap?: number;
  /** Cap keeps head AND tail (command output: failures live at the end). */
  capKeepsTail?: boolean;
  /** One-phrase summary for the deferred index line (deferred tools only). */
  hint?: Bi;
}

const d = (name: string): Bi => DOC_LINES[name];

/** The compile-time union of native tool names (order irrelevant). Coverage
 *  against NATIVE_SPECS is asserted in toolRegistry.test.ts. */
export const NATIVE_TOOL_NAMES = [
  "read_file",
  "write_file",
  "edit_file",
  "edit_lines",
  "multi_edit",
  "outline",
  "list_dir",
  "glob",
  "grep",
  "search_files",
  "search_code",
  "search_docs",
  "bash",
  "bash_bg",
  "bg_output",
  "bg_kill",
  "web_search",
  "web_fetch",
  "web_download",
  "view_image",
  "browser_navigate",
  "browser_screenshot",
  "browser_snapshot",
  "browser_scroll",
  "browser_click",
  "browser_type",
  "browser_eval",
  "browser_console",
  "browser_read",
  "browser_close",
  "ask_user",
  "update_plan",
  "validate_change",
  "understand_repo",
] as const;
export type AgentToolName = (typeof NATIVE_TOOL_NAMES)[number];

/** Native tools, in system-prompt doc order (core suite, then browser). */
const NATIVE_SPECS: (ToolSpec & { name: AgentToolName })[] = [
  { name: "read_file", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("read_file"), requiredArgs: ["path"], argExample: '{"path":"src/app.ts"}', resultCap: 400000 },
  { name: "write_file", source: "native", suite: "core", perm: "write", tier: "core", docLine: d("write_file"), mutating: true, requiredArgs: ["path"], argExample: '{"path":"notes.md","content":"…"}' },
  { name: "edit_file", source: "native", suite: "core", perm: "write", tier: "core", docLine: d("edit_file"), mutating: true, requiredArgs: ["path"], argExample: '{"path":"src/app.ts","old_string":"…","new_string":"…"}' },
  { name: "edit_lines", source: "native", suite: "core", perm: "write", tier: "core", docLine: d("edit_lines"), docHidden: true, mutating: true, requiredArgs: ["path"], argExample: '{"path":"src/app.ts","edits":[{"op":"replace","anchor":"22:abc","content":"…"}]}' },
  { name: "multi_edit", source: "native", suite: "core", perm: "write", tier: "core", mutating: true, requiredArgs: ["path"], argExample: '{"path":"src/app.ts","edits":[{"old_string":"…","new_string":"…"}]}' },
  { name: "outline", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("outline") },
  { name: "list_dir", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("list_dir") },
  { name: "glob", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("glob") },
  { name: "grep", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("grep"), requiredArgs: ["pattern"], argExample: '{"pattern":"TODO"}' },
  { name: "search_files", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("search_files"), requiredArgs: ["query"], argExample: '{"query":"logging config"}' },
  { name: "search_code", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("search_code"), requiredArgs: ["query"], argExample: '{"query":"where login auth is handled"}' },
  { name: "search_docs", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("search_docs"), requiredArgs: ["query"], argExample: '{"query":"how uploads are stored"}' },
  { name: "bash", source: "native", suite: "core", perm: "exec", tier: "core", docLine: d("bash"), mutating: true, requiredArgs: ["command"], argExample: '{"command":"ls src"}', capKeepsTail: true },
  { name: "bash_bg", source: "native", suite: "core", perm: "exec", tier: "core", docLine: d("bash_bg"), mutating: true, requiredArgs: ["command"], argExample: '{"command":"npm run dev"}', capKeepsTail: true },
  { name: "bg_output", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("bg_output"), repeatExempt: true, capKeepsTail: true },
  { name: "bg_kill", source: "native", suite: "core", perm: "exec", tier: "core", docLine: d("bg_kill") },
  { name: "understand_repo", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("understand_repo") },
  { name: "validate_change", source: "native", suite: "core", perm: "exec", tier: "core", docLine: d("validate_change"), mutating: true },
  { name: "web_search", source: "native", suite: "core", perm: "network", tier: "core", docLine: d("web_search"), untrusted: true, requiredArgs: ["query"], argExample: '{"query":"tauri updater docs"}' },
  { name: "web_fetch", source: "native", suite: "core", perm: "network", tier: "core", docLine: d("web_fetch"), untrusted: true, requiredArgs: ["url"], argExample: '{"url":"https://example.com/docs"}', resultCap: 48000 },
  { name: "web_download", source: "native", suite: "core", perm: "network", tier: "core", docLine: d("web_download"), mutating: true, requiredArgs: ["url", "path"], argExample: '{"url":"https://…/file.zip","path":"downloads/file.zip"}' },
  { name: "update_plan", source: "native", suite: "core", perm: "ui", tier: "core", docLine: d("update_plan") },
  { name: "ask_user", source: "native", suite: "core", perm: "ui", tier: "core", docLine: d("ask_user") },
  { name: "view_image", source: "native", suite: "core", perm: "read", tier: "core", docLine: d("view_image"), requiredArgs: ["path"], argExample: '{"path":"shot.png"}' },
  { name: "browser_navigate", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_navigate"), untrusted: true, requiredArgs: ["url"], argExample: '{"url":"https://example.com"}' },
  { name: "browser_read", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_read"), untrusted: true, repeatExempt: true },
  { name: "browser_screenshot", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_screenshot"), visionOnly: true, repeatExempt: true },
  { name: "browser_snapshot", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_snapshot"), visionOnly: true, repeatExempt: true },
  { name: "browser_scroll", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_scroll"), untrusted: true, repeatExempt: true },
  { name: "browser_close", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_close") },
  { name: "browser_console", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_console"), untrusted: true, repeatExempt: true },
  { name: "browser_click", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_click"), untrusted: true },
  { name: "browser_type", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_type"), untrusted: true },
  { name: "browser_eval", source: "native", suite: "browser", perm: "network", tier: "core", docLine: d("browser_eval"), untrusted: true },
];

// ── Live registry: natives + runtime registrations (mcp/skill, M1/M3) ──────

const specs: ToolSpec[] = [...NATIVE_SPECS];
const byName = new Map(specs.map((s) => [s.name, s]));

export function toolSpec(name: string): ToolSpec | undefined {
  return byName.get(name);
}

export function allToolSpecs(): readonly ToolSpec[] {
  return specs;
}

/** Add a runtime tool (MCP server tool, skill-provided tool). Native names
 *  cannot be shadowed. Returns false if the name is taken. */
export function registerTool(spec: ToolSpec): boolean {
  if (byName.has(spec.name)) return false;
  specs.push(spec);
  byName.set(spec.name, spec);
  return true;
}

/** Remove a runtime tool (e.g. its MCP server disconnected). Natives are
 *  permanent. */
export function unregisterTool(name: string): boolean {
  const spec = byName.get(name);
  if (!spec || spec.source === "native") return false;
  byName.delete(name);
  specs.splice(specs.indexOf(spec), 1);
  return true;
}

// ── Derived views (the shapes agentLoop consumed before M0) ────────────────

function derive(pred: (s: ToolSpec) => boolean): Set<AgentToolName> {
  return new Set(NATIVE_SPECS.filter(pred).map((s) => s.name as AgentToolName));
}

/** Tools that change the world (or run code) → need approval unless bypassed. */
export const MUTATING_TOOLS = derive((s) => s.mutating === true);

/** Tools exempt from the identical-call loop breaker — see ToolSpec.repeatExempt. */
export const REPEAT_EXEMPT = derive((s) => s.repeatExempt === true);

/** Tools whose output is untrusted external content — see ToolSpec.untrusted. */
export const UNTRUSTED_TOOLS = derive((s) => s.untrusted === true);

export const REQUIRED_ARGS: Record<string, string[]> = Object.fromEntries(
  NATIVE_SPECS.filter((s) => s.requiredArgs).map((s) => [s.name, s.requiredArgs as string[]]),
);

export const ARG_EXAMPLE: Record<string, string> = Object.fromEntries(
  NATIVE_SPECS.filter((s) => s.argExample).map((s) => [s.name, s.argExample as string]),
);

export function resultCap(name: string): number {
  return byName.get(name)?.resultCap ?? 12000;
}

export function capKeepsTail(name: string): boolean {
  return byName.get(name)?.capKeepsTail === true;
}

// ── System-prompt doc assembly (byte-exact vs the pre-M0 inline strings) ────

const DEFERRED_INDEX_HEAD: Bi = {
  zh: "以下工具已可用但文档未加载(按需加载,M1 接线):",
  en: "Also available, docs not loaded (loaded on demand; wired in M1):",
};

/** Build the tools section of the system prompt.
 *
 *  Byte-compatibility contract (locked by promptGolden.test.ts):
 *  - core block = "\n" + one line per documented core tool, doc order;
 *  - vision adds "\n" + browser lines + tail; browserText drops the two
 *    visionOnly tools and swaps the tail;
 *  - anchor mode swaps edit_file's line for edit_lines' and annotates
 *    read_file's — the whole-prompt edit_file→edit_lines rename stays in
 *    systemPrompt, after this returns.
 *  Deferred tools (none shipped yet) collapse into a single index line. */
export function buildToolsDoc(
  l: "zh" | "en",
  opts: { vision?: boolean; browserText?: boolean; anchors?: boolean },
): string {
  const live = (suite: ToolSpec["suite"]) =>
    specs.filter((s) => s.suite === suite && s.docLine && !s.docHidden && s.tier === "core");

  let lines = live("core").map((s) => (s.docLine as Bi)[l]);
  let doc = "\n" + lines.join("\n");

  if (opts.vision || opts.browserText) {
    let browser = live("browser");
    if (!opts.vision) browser = browser.filter((s) => !s.visionOnly);
    lines = browser.map((s) => (s.docLine as Bi)[l]);
    const tail = opts.vision ? BROWSER_TAIL_VISION[l] : BROWSER_TAIL_TEXT[l];
    doc += "\n" + [...lines, tail].join("\n");
  }

  const deferred = specs.filter((s) => s.tier === "deferred");
  if (deferred.length) {
    const index = deferred
      .map((s) => `${s.name}${s.hint ? `(${s.hint[l]})` : ""}`)
      .join(", ");
    doc += `\n${DEFERRED_INDEX_HEAD[l]} ${index}`;
  }

  if (opts.anchors) {
    doc = doc
      .split("\n")
      .map((line) => {
        if (line.startsWith("- edit_file:")) return (byName.get("edit_lines")?.docLine as Bi)[l];
        if (line.startsWith("- read_file:")) return `${line} ${ANCHOR_READ_NOTE[l]}`;
        return line;
      })
      .join("\n");
  }
  return doc;
}
