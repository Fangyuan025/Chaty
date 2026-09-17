// A file write or edit the model is still writing, read out of its unfinished
// output so Code mode can show the card — and the diff — while the arguments
// arrive, instead of only once the call has run.
//
// Every call format a model may write in is read (xml, json, gemma, lfm; see
// callFormat.ts). Nothing here decides what runs: the finished output still
// goes through parseToolCall. This only has to be right about what is on the
// screen, and tolerant of output that ends anywhere — mid-key, mid-escape,
// mid-marker.

import { callStart } from "./callFormat";
import { diffLines } from "./diff";

export type LiveTool = "write_file" | "edit_file" | "multi_edit";

const LIVE_TOOLS = new Set<string>(["write_file", "edit_file", "multi_edit"]);

/** What a live card knows so far. A `…Done` flag is set once that argument's
 *  closing delimiter has arrived; until then the text is a prefix of it. */
export interface LiveView {
  name: LiveTool;
  path?: string;
  /** The file as it was before the call, once read ("" for a new file). */
  before?: string;
  content?: string;
  contentDone?: boolean;
  old?: string;
  oldDone?: boolean;
  new?: string;
  newDone?: boolean;
  /** The replacements, in the order the model wrote them: one for
   *  old_string/new_string, several for an edits array. */
  edits?: LiveEdit[];
  /** Still being written: the call has not been accepted to run. A host must
   *  not keep such a card if the session is saved. */
  pending?: boolean;
}

/** One replacement as far as it has arrived. */
export interface LiveEdit {
  old?: string;
  oldDone?: boolean;
  new?: string;
  newDone?: boolean;
}

interface Arg {
  key: string;
  /** Text so far for a string value; undefined for anything else. */
  text?: string;
  /** An array or object value, as written so far. */
  raw?: string;
  done: boolean;
}

const PATH_KEYS = ["path", "file_path", "filename", "file"];
const CONTENT_KEYS = ["content", "text", "contents", "body", "file_text"];
const OLD_KEYS = ["old_string", "old_str", "old", "search", "from"];
const NEW_KEYS = ["new_string", "new_str", "new", "replace", "to"];
const EDITS_KEYS = ["edits", "changes", "replacements"];

/** The call at the end of the model's output, from where it opens; null when
 *  there is none yet or the model is still reasoning. A call quoted inside a
 *  thought is not the call — but a thought marker AFTER the call opens is
 *  text the call is writing (a file may well contain `<think>`). */
function callText(raw: string): string | null {
  const c = callStart(raw);
  if (c === -1) return null;
  const first = (tags: string[]) =>
    tags.reduce((m, t) => {
      const k = raw.indexOf(t);
      return k !== -1 && (m === -1 || k < m) ? k : m;
    }, -1);
  const open = first(["<think>", "<|channel>"]);
  const close = first(["</think>", "<channel|>"]);
  if ((open === -1 || open > c) && (close === -1 || close > c)) return raw.slice(c);
  if (close === -1) return null; // still reasoning
  const rest = raw.slice(close + (raw.startsWith("</think>", close) ? "</think>".length : "<channel|>".length));
  const c2 = callStart(rest);
  return c2 === -1 ? null : rest.slice(c2);
}

/** `v` without a trailing prefix of `marker` — the start of a closing marker
 *  that has not finished arriving is not part of the value. */
function withoutPartialMarker(v: string, marker: string): string {
  for (let k = Math.min(marker.length - 1, v.length); k > 0; k--) {
    if (v.endsWith(marker.slice(0, k))) return v.slice(0, -k);
  }
  return v;
}

// ── xml: <function=name> <parameter=key>\nvalue\n</parameter> ──
function xmlArgs(body: string): Arg[] {
  const args: Arg[] = [];
  const opens = [...body.matchAll(/<parameter=["']?([^>\s"']+)["']?\s*>/g)];
  for (let n = 0; n < opens.length; n++) {
    const m = opens[n];
    let start = (m.index ?? 0) + m[0].length;
    if (body[start] === "\n") start++;
    const close = body.indexOf("</parameter>", start);
    const limit = n + 1 < opens.length ? (opens[n + 1].index ?? body.length) : body.length;
    if (close !== -1 && close <= limit) {
      let v = body.slice(start, close);
      if (v.endsWith("\n")) v = v.slice(0, -1);
      args.push({ key: m[1], text: v, done: true });
    } else {
      args.push({ key: m[1], text: withoutPartialMarker(body.slice(start, limit), "\n</parameter>"), done: false });
    }
  }
  return args;
}

// ── A cursor over text that may end at any point ──
class Scan {
  i = 0;
  constructor(readonly s: string) {}
  get end(): boolean {
    return this.i >= this.s.length;
  }
  ws(): void {
    while (this.i < this.s.length && /\s/.test(this.s[this.i])) this.i++;
  }
  /** A quoted string with backslash escapes, as JSON and LFM write them. */
  quoted(q: string): { text: string; done: boolean } {
    let out = "";
    this.i++; // the opening quote
    while (this.i < this.s.length) {
      const c = this.s[this.i];
      if (c === q) {
        this.i++;
        return { text: out, done: true };
      }
      if (c === "\\") {
        if (this.i + 1 >= this.s.length) break; // the escape has not arrived
        const e = this.s[this.i + 1];
        if (e === "u") {
          const hex = this.s.slice(this.i + 2, this.i + 6);
          if (hex.length < 4) break;
          out += String.fromCharCode(parseInt(hex, 16) || 0);
          this.i += 6;
          continue;
        }
        out += e === "n" ? "\n" : e === "t" ? "\t" : e === "r" ? "\r" : e === "b" ? "\b" : e === "f" ? "\f" : e;
        this.i += 2;
        continue;
      }
      out += c;
      this.i++;
    }
    this.i = this.s.length;
    return { text: out, done: false };
  }
  /** Skip a bracketed value; false when it has not closed yet. */
  skipNested(): boolean {
    const stack: string[] = [];
    while (this.i < this.s.length) {
      const c = this.s[this.i];
      if (c === '"' || c === "'") {
        if (!this.quoted(c).done) return false;
        continue;
      }
      if (this.s.startsWith('<|"|>', this.i)) {
        const close = this.s.indexOf('<|"|>', this.i + 5);
        if (close === -1) return false;
        this.i = close + 5;
        continue;
      }
      if (c === "{" || c === "[") stack.push(c === "{" ? "}" : "]");
      else if (c === "}" || c === "]") {
        stack.pop();
        if (stack.length === 0) {
          this.i++;
          return true;
        }
      }
      this.i++;
    }
    return false;
  }
}

// ── json: {"name": "…", "arguments": {…}} — or the arguments laid out flat ──
function jsonArgs(src: string): { name?: string; args: Arg[] } {
  const sc = new Scan(src);
  const top: Arg[] = [];
  const nested: Arg[] = [];
  let name: string | undefined;
  let sawNested = false;
  const object = (into: Arg[], depth: number): boolean => {
    sc.i++; // {
    for (;;) {
      sc.ws();
      if (sc.end) return false;
      if (sc.s[sc.i] === "}") {
        sc.i++;
        return true;
      }
      if (sc.s[sc.i] === ",") {
        sc.i++;
        continue;
      }
      if (sc.s[sc.i] !== '"') return false;
      const key = sc.quoted('"');
      if (!key.done) return false;
      sc.ws();
      if (sc.s[sc.i] !== ":") return false;
      sc.i++;
      sc.ws();
      if (sc.end) return false;
      const c = sc.s[sc.i];
      if (c === '"') {
        const v = sc.quoted('"');
        if (depth === 0 && key.text === "name") {
          if (v.done) name = v.text;
        } else into.push({ key: key.text, text: v.text, done: v.done });
        if (!v.done) return false;
      } else if (c === "{" && depth === 0 && (key.text === "arguments" || key.text === "parameters")) {
        sawNested = true;
        if (!object(nested, 1)) return false;
      } else if (c === "{" || c === "[") {
        const from = sc.i;
        const closed = sc.skipNested();
        into.push({ key: key.text, raw: sc.s.slice(from, sc.i), done: closed });
        if (!closed) return false;
      } else {
        while (!sc.end && !/[,}\s]/.test(sc.s[sc.i])) sc.i++;
        into.push({ key: key.text, done: !sc.end });
      }
    }
  };
  const brace = src.indexOf("{");
  if (brace === -1) return { args: [] };
  sc.i = brace;
  object(top, 0);
  return { name, args: sawNested && nested.length > 0 ? nested : top };
}

// ── gemma: call:name{key:<|"|>text<|"|>,n:5} ──
function gemmaArgs(src: string): Arg[] {
  const Q = '<|"|>';
  const sc = new Scan(src);
  const args: Arg[] = [];
  for (;;) {
    sc.ws();
    if (sc.end || sc.s[sc.i] === "}") return args;
    if (sc.s[sc.i] === ",") {
      sc.i++;
      continue;
    }
    const key = /^([A-Za-z_][\w-]*)\s*:\s*/.exec(sc.s.slice(sc.i, sc.i + 80));
    if (!key) return args;
    sc.i += key[0].length;
    if (sc.end) return args;
    if (sc.s.startsWith(Q, sc.i)) {
      const close = sc.s.indexOf(Q, sc.i + Q.length);
      if (close === -1) {
        args.push({ key: key[1], text: withoutPartialMarker(sc.s.slice(sc.i + Q.length), Q), done: false });
        return args;
      }
      args.push({ key: key[1], text: sc.s.slice(sc.i + Q.length, close), done: true });
      sc.i = close + Q.length;
    } else if (Q.startsWith(sc.s.slice(sc.i))) {
      return args; // the opening marker itself is still arriving
    } else if (sc.s[sc.i] === '"' || sc.s[sc.i] === "'") {
      const v = sc.quoted(sc.s[sc.i]);
      args.push({ key: key[1], text: v.text, done: v.done });
      if (!v.done) return args;
    } else if (sc.s[sc.i] === "{" || sc.s[sc.i] === "[") {
      const from = sc.i;
      const closed = sc.skipNested();
      args.push({ key: key[1], raw: sc.s.slice(from, sc.i), done: closed });
      if (!closed) return args;
    } else {
      while (!sc.end && !",}".includes(sc.s[sc.i])) sc.i++;
      args.push({ key: key[1], done: !sc.end });
    }
  }
}

// ── lfm: [name(key="text", n=5)] ──
function lfmArgs(src: string): Arg[] {
  const sc = new Scan(src);
  const args: Arg[] = [];
  for (;;) {
    sc.ws();
    if (sc.end || sc.s[sc.i] === ")") return args;
    if (sc.s[sc.i] === ",") {
      sc.i++;
      continue;
    }
    const key = /^([A-Za-z_]\w*)\s*=\s*/.exec(sc.s.slice(sc.i, sc.i + 80));
    if (!key) return args;
    sc.i += key[0].length;
    if (sc.end) return args;
    const c = sc.s[sc.i];
    if (c === '"' || c === "'") {
      const v = sc.quoted(c);
      args.push({ key: key[1], text: v.text, done: v.done });
      if (!v.done) return args;
    } else if (c === "{" || c === "[") {
      const from = sc.i;
      const closed = sc.skipNested();
      args.push({ key: key[1], raw: sc.s.slice(from, sc.i), done: closed });
      if (!closed) return args;
    } else {
      while (!sc.end && !",)".includes(sc.s[sc.i])) sc.i++;
      args.push({ key: key[1], done: !sc.end });
    }
  }
}

/** An edits array as far as it has arrived — `[{old_string, new_string}, …]`
 *  in any of the spellings the call formats produce: JSON, Python-style
 *  quotes, Gemma's bare keys and <|"|> strings. It stops at the first value
 *  that has not finished. */
function partialEdits(src: string): LiveEdit[] {
  const Q = '<|"|>';
  const out: LiveEdit[] = [];
  const open = src.indexOf("[");
  if (open === -1) return out;
  const sc = new Scan(src);
  sc.i = open + 1;
  for (;;) {
    sc.ws();
    if (sc.end || sc.s[sc.i] === "]") return out;
    if (sc.s[sc.i] === ",") {
      sc.i++;
      continue;
    }
    if (sc.s[sc.i] !== "{") return out;
    sc.i++;
    const edit: LiveEdit = {};
    out.push(edit);
    for (;;) {
      sc.ws();
      if (sc.end) return out;
      const c = sc.s[sc.i];
      if (c === "}") {
        sc.i++;
        break;
      }
      if (c === ",") {
        sc.i++;
        continue;
      }
      let key: string;
      if (sc.s.startsWith(Q, sc.i)) {
        const close = sc.s.indexOf(Q, sc.i + Q.length);
        if (close === -1) return out;
        key = sc.s.slice(sc.i + Q.length, close);
        sc.i = close + Q.length;
      } else if (c === '"' || c === "'") {
        const k = sc.quoted(c);
        if (!k.done) return out;
        key = k.text;
      } else {
        const m = /^[A-Za-z_]\w*/.exec(sc.s.slice(sc.i, sc.i + 60));
        if (!m) return out;
        key = m[0];
        sc.i += key.length;
      }
      sc.ws();
      if (sc.s[sc.i] !== ":" && sc.s[sc.i] !== "=") return out;
      sc.i++;
      sc.ws();
      if (sc.end) return out;
      let text: string | undefined;
      let done = true;
      if (sc.s.startsWith(Q, sc.i)) {
        const close = sc.s.indexOf(Q, sc.i + Q.length);
        if (close === -1) {
          text = withoutPartialMarker(sc.s.slice(sc.i + Q.length), Q);
          done = false;
        } else {
          text = sc.s.slice(sc.i + Q.length, close);
          sc.i = close + Q.length;
        }
      } else if (Q.startsWith(sc.s.slice(sc.i))) {
        return out;
      } else if (sc.s[sc.i] === '"' || sc.s[sc.i] === "'") {
        const v = sc.quoted(sc.s[sc.i]);
        text = v.text;
        done = v.done;
      } else if (sc.s[sc.i] === "{" || sc.s[sc.i] === "[") {
        if (!sc.skipNested()) return out;
      } else {
        while (!sc.end && !",}".includes(sc.s[sc.i])) sc.i++;
      }
      if (text !== undefined) {
        if (OLD_KEYS.includes(key)) {
          edit.old = text;
          edit.oldDone = done;
        } else if (NEW_KEYS.includes(key)) {
          edit.new = text;
          edit.newDone = done;
        }
        if (!done) return out;
      }
    }
  }
}

/** The file write or edit being written at the end of `raw`, as far as it has
 *  got; null when the output is not (yet) such a call. */
export function liveFileCall(raw: string): LiveView | null {
  const s = callText(raw);
  if (s === null) return null;
  let name: string | undefined;
  let args: Arg[] = [];
  if (s.startsWith("<|tool_call>")) {
    const head = /^<\|tool_call>call:([\w.-]+)\s*\{/.exec(s);
    if (!head || !LIVE_TOOLS.has(head[1])) return null;
    name = head[1];
    args = gemmaArgs(s.slice(head[0].length));
  } else if (s.startsWith("<|tool_call_start|>")) {
    const head = /^<\|tool_call_start\|>\s*\[?\s*([\w.-]+)\s*\(/.exec(s);
    if (!head || !LIVE_TOOLS.has(head[1])) return null;
    name = head[1];
    args = lfmArgs(s.slice(head[0].length));
  } else {
    const body = s.startsWith("<tool_call>") ? s.slice("<tool_call>".length) : s;
    const trimmed = body.trimStart();
    if (trimmed.startsWith("{")) {
      const j = jsonArgs(trimmed);
      name = j.name;
      args = j.args;
    } else {
      const fn = /<function=["']?([^>\s"']+)["']?\s*>/.exec(body);
      if (!fn) return null;
      name = fn[1];
      const end = body.indexOf("</function>", fn.index + fn[0].length);
      args = xmlArgs(body.slice(fn.index + fn[0].length, end === -1 ? undefined : end));
    }
    if (!name || !LIVE_TOOLS.has(name)) return null;
  }
  const pick = (keys: string[]) => args.find((a) => keys.includes(a.key));
  const view: LiveView = { name: name as LiveTool };
  const path = pick(PATH_KEYS);
  if (path?.done && path.text) view.path = path.text;
  const editsArg = pick(EDITS_KEYS);
  if (editsArg) {
    // XML writes the array as the parameter's text; the others as a value.
    view.edits = partialEdits(editsArg.raw ?? editsArg.text ?? "");
    return view;
  }
  const content = pick(CONTENT_KEYS);
  if (content?.text !== undefined) {
    view.content = content.text;
    view.contentDone = content.done;
  }
  const old = pick(OLD_KEYS);
  const nu = pick(NEW_KEYS);
  if (old?.text !== undefined || nu?.text !== undefined) {
    view.edits = [
      {
        ...(old?.text !== undefined ? { old: old.text, oldDone: old.done } : {}),
        ...(nu?.text !== undefined ? { new: nu.text, newDone: nu.done } : {}),
      },
    ];
  }
  return view;
}

/** How much of a file a live card lays out whole: past this, the changes are
 *  shown on their own. */
const LIVE_BEFORE_MAX = 200_000;
/** A live view redraws several times a second — a smaller diff matrix than
 *  the finished card may use. */
const LIVE_LCS_CELLS = 1_000_000;

/** `pending`: an old line the text being written has not reached yet — not
 *  removed, not kept, only not there yet (drawn dimmed). */
export type LiveRowKind = "ctx" | "add" | "del" | "pending";
export interface LiveRow {
  kind: LiveRowKind;
  text: string;
  /** Part of the block the model is copying out to replace (being found). */
  found?: boolean;
}

/** What a live card draws: the file, with the changes in it so far, and the
 *  line the model is at — the view follows it the way Canvas's patch view
 *  does: to a block, through its rewrite, on to the next. */
export interface LiveScan {
  rows: LiveRow[];
  /** Row the model is at, or null before it is anywhere. */
  head: number | null;
  added: number;
  removed: number;
}

const splitLines = (s: string): string[] => (s.length ? s.split("\n") : []);
/** Never a line of anything: keeps a half-written text's last line from being
 *  matched against the end of the file. */
const SENTINEL = "\u0000chaty-live-end\u0000";

/** a → b as rows over the whole of both: shared lines at either end as they
 *  are, the stretch between them diffed. */
function wholeRows(a: string[], b: string[]): LiveRow[] {
  let s = 0;
  while (s < a.length && s < b.length && a[s] === b[s]) s++;
  let ea = a.length;
  let eb = b.length;
  while (ea > s && eb > s && a[ea - 1] === b[eb - 1]) {
    ea--;
    eb--;
  }
  const rows: LiveRow[] = a.slice(0, s).map((text) => ({ kind: "ctx", text }));
  if (ea > s || eb > s) {
    const mid = diffLines(a.slice(s, ea).join("\n"), b.slice(s, eb).join("\n"), Number.POSITIVE_INFINITY, LIVE_LCS_CELLS);
    // A middle that is all one side diffs to rows from a joined empty string —
    // keep the line counts exact.
    if (ea === s) rows.push(...b.slice(s, eb).map((text): LiveRow => ({ kind: "add", text })));
    else if (eb === s) rows.push(...a.slice(s, ea).map((text): LiveRow => ({ kind: "del", text })));
    else rows.push(...mid.rows.map((r): LiveRow => ({ kind: r.kind, text: r.text })));
  }
  rows.push(...a.slice(ea).map((text): LiveRow => ({ kind: "ctx", text })));
  return rows;
}

/**
 * a → b where b is still being written. The lines written so far are matched
 * against the old ones; old lines past the last one matched are `pending` and
 * go after what has been written, dimmed — the model has not got there, and
 * may yet write them again. The head is the line being written.
 */
function writingRows(a: string[], b: string[]): { rows: LiveRow[]; head: number } {
  const raw = wholeRows(a, [...b, SENTINEL]).filter((r) => r.text !== SENTINEL);
  let lastKept = -1;
  raw.forEach((r, k) => {
    if (r.kind === "ctx") lastKept = k;
  });
  const written: LiveRow[] = [];
  const pending: LiveRow[] = [];
  raw.forEach((r, k) => {
    if (r.kind === "del" && k > lastKept) pending.push({ kind: "pending", text: r.text });
    else written.push(r);
  });
  // The line being typed is the start of the next old line: an unchanged line
  // being copied out, shown as it is rather than flashing up as new until its
  // last character arrives.
  const typing = b[b.length - 1];
  const tail = written[written.length - 1];
  if (typing && tail?.kind === "add" && tail.text === typing && pending[0]?.text.startsWith(typing)) {
    written[written.length - 1] = { kind: "ctx", text: pending[0].text };
    pending.shift();
  }
  let head = -1;
  for (let k = 0; k < written.length; k++) if (written[k].kind !== "del") head = k;
  return { rows: [...written, ...pending], head: Math.max(0, head) };
}

/** A replacement placed in the file: the whole lines it covers in the text as
 *  it stands, and what surrounds the replaced text on its first and last. */
interface Placed {
  start: number;
  end: number;
  prefix: string;
  suffix: string;
}

/** Where `old` sits in `text`: verbatim first, then line by line ignoring
 *  indentation and trailing space (the edit tool accepts that match too). A
 *  `partial` old may stop mid-line. */
function place(text: string, old: string, partial: boolean): Placed | null {
  const at = text.indexOf(old);
  if (at !== -1) {
    const start = text.lastIndexOf("\n", at - 1) + 1;
    const oldEnd = at + old.length;
    const lineEnd = text.indexOf("\n", oldEnd);
    const stop = lineEnd === -1 ? text.length : lineEnd;
    const startLine = start === 0 ? 0 : text.slice(0, start).split("\n").length - 1;
    const endLine = startLine + text.slice(start, stop).split("\n").length;
    return { start: startLine, end: endLine, prefix: text.slice(start, at), suffix: text.slice(oldEnd, stop) };
  }
  const lines = splitLines(text).map((l) => l.trim());
  const want = old.split("\n").map((l) => l.trim());
  if (partial && want.length > 1 && want[want.length - 1] === "") want.pop();
  while (want.length && want[0] === "") want.shift();
  if (!want.length || (want.length === 1 && want[0].length < 3)) return null;
  for (let i = 0; i + want.length <= lines.length; i++) {
    let ok = true;
    for (let j = 0; j < want.length && ok; j++) {
      const last = j === want.length - 1;
      ok = partial && last ? lines[i + j].startsWith(want[j]) : lines[i + j] === want[j];
    }
    if (ok) return { start: i, end: i + want.length, prefix: "", suffix: "" };
  }
  return null;
}

interface Hunk {
  /** Where it is in the file as it was. */
  baseStart: number;
  baseLen: number;
  lines: string[];
  /** Still being written. */
  writing: boolean;
}

/** A line range of the text with every earlier hunk applied, in the file as it
 *  was; null when it runs into one of those hunks. */
function toBase(hunks: Hunk[], start: number, end: number): { start: number; end: number } | null {
  let delta = 0;
  for (const h of hunks) {
    const vStart = h.baseStart + delta;
    const vEnd = vStart + h.lines.length;
    if (end <= vStart) break;
    if (start >= vEnd) {
      delta += h.lines.length - h.baseLen;
      continue;
    }
    return null;
  }
  return { start: start - delta, end: end - delta };
}

/** The text of `base` with `hunks` applied. */
function applied(base: string[], hunks: Hunk[]): string {
  const out: string[] = [];
  let at = 0;
  for (const h of hunks) {
    out.push(...base.slice(at, h.baseStart), ...h.lines);
    at = h.baseStart + h.baseLen;
  }
  out.push(...base.slice(at));
  return out.join("\n");
}

function counted(rows: LiveRow[], head: number | null): LiveScan {
  let added = 0;
  let removed = 0;
  for (const r of rows) {
    if (r.kind === "add") added++;
    else if (r.kind === "del") removed++;
  }
  return { rows, head, added, removed };
}

/** Replacements with no file to place them in: each one on its own. */
function editsAlone(edits: LiveEdit[]): LiveScan | null {
  const rows: LiveRow[] = [];
  let head: number | null = null;
  edits.forEach((e, k) => {
    if (e.old === undefined && e.new === undefined) return;
    if (k > 0 && rows.length) rows.push({ kind: "ctx", text: "⋯" });
    const a = splitLines(e.old ?? "");
    if (e.new === undefined) {
      rows.push(...a.map((text): LiveRow => ({ kind: "ctx", text, found: true })));
      head = rows.length - 1;
      return;
    }
    if (e.newDone) {
      rows.push(...wholeRows(a, splitLines(e.new)));
      head = rows.length - 1;
      return;
    }
    const w = writingRows(a, splitLines(e.new));
    head = rows.length + w.head;
    rows.push(...w.rows);
  });
  return rows.length ? counted(rows, head) : null;
}

/** What a live card shows for what has arrived so far. */
export function liveScan(v: LiveView): LiveScan | null {
  const before = v.before !== undefined && v.before.length <= LIVE_BEFORE_MAX ? v.before : undefined;
  if (v.name === "write_file") {
    if (v.content === undefined) return null;
    const a = splitLines(before ?? "");
    const b = splitLines(v.content);
    if (v.contentDone) {
      const rows = wholeRows(a, b);
      let head = -1;
      rows.forEach((r, k) => {
        if (r.kind !== "ctx") head = k;
      });
      return counted(rows, head >= 0 ? head : null);
    }
    const w = writingRows(a, b);
    return counted(w.rows, w.head);
  }
  const edits = v.edits ?? [];
  if (!edits.length) return null;
  if (before === undefined) return editsAlone(edits);

  // The edits in the order they were written, each placed in the file as the
  // ones before it left it — as the tool will apply them.
  const base = splitLines(before);
  const hunks: Hunk[] = [];
  let found: { start: number; end: number } | null = null;
  let aimed: "found" | "writing" | null = null;
  for (const e of edits) {
    if (!e.old) continue;
    const text = applied(base, hunks);
    const at = place(text, e.old, !e.oldDone);
    if (!at) continue;
    const range = toBase(hunks, at.start, at.end);
    if (!range) continue;
    if (!e.oldDone || e.new === undefined) {
      // Still copying out the text to replace: that is where it is looking.
      found = range;
      aimed = "found";
      continue;
    }
    const lines = splitLines(at.prefix + e.new + (e.newDone ? at.suffix : ""));
    hunks.push({ baseStart: range.start, baseLen: range.end - range.start, lines, writing: !e.newDone });
    hunks.sort((x, y) => x.baseStart - y.baseStart);
    if (!e.newDone) aimed = "writing";
    else aimed = null;
  }

  const rows: LiveRow[] = [];
  let head: number | null = null;
  let lastChange: number | null = null;
  let at = 0;
  const context = (from: number, to: number) => {
    for (let i = from; i < to; i++) {
      const inFound = found !== null && i >= found.start && i < found.end;
      rows.push(inFound ? { kind: "ctx", text: base[i], found: true } : { kind: "ctx", text: base[i] });
      if (inFound && aimed === "found") head = rows.length - 1;
    }
  };
  for (const h of hunks) {
    context(at, h.baseStart);
    const old = base.slice(h.baseStart, h.baseStart + h.baseLen);
    if (h.writing) {
      const w = writingRows(old, h.lines);
      if (aimed === "writing") head = rows.length + w.head;
      rows.push(...w.rows);
    } else {
      rows.push(...wholeRows(old, h.lines));
      lastChange = rows.length - 1;
    }
    at = h.baseStart + h.baseLen;
  }
  context(at, base.length);
  return counted(rows, head ?? lastChange);
}
