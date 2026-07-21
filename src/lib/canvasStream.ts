// Cursor-style live "apply" view for Canvas iterations: while the model
// streams SEARCH/REPLACE patches (or a full document), turn the partial
// output into a dynamic diff against the current version — resolved changes
// show as add/del rows, the region currently being rewritten carries the
// scan head, and (full-document mode) the not-yet-reached tail of the old
// code renders dimmed as "pending" instead of falsely deleted.
import { diffLines, type DiffRow } from "./diff";

export type ScanRowKind = "ctx" | "add" | "del" | "pending";
export interface ScanRow {
  kind: ScanRowKind;
  text: string;
}
export interface ScanView {
  rows: ScanRow[];
  /** Row index the scan head sits on (animated), or null while waiting. */
  scanIndex: number | null;
  mode: "patch" | "full" | "waiting";
}

interface StreamPatch {
  done: { search: string; replace: string }[];
  active: { search: string; replace: string | null } | null;
}

const norm = (s: string) => s.replace(/\r/g, "").replace(/[ \t]+$/gm, "");

/** Parse completed SEARCH/REPLACE blocks plus the one still streaming in. */
export function parseStreamPatches(text: string): StreamPatch {
  const done: StreamPatch["done"] = [];
  const re =
    /<{5,}\s*SEARCH[^\n]*\r?\n([\s\S]*?)\r?\n={3,}[^\n]*\r?\n([\s\S]*?)\r?\n>{5,}\s*REPLACE/g;
  let m: RegExpExecArray | null;
  let lastEnd = 0;
  while ((m = re.exec(text))) {
    done.push({ search: m[1], replace: m[2] });
    lastEnd = m.index + m[0].length;
  }
  const rest = text.slice(lastEnd);
  const start = /<{5,}\s*SEARCH[^\n]*\r?\n/.exec(rest);
  if (!start || start.index === undefined) return { done, active: null };
  const body = rest.slice(start.index + start[0].length);
  const sep = /\r?\n={3,}[^\n]*(?:\r?\n|$)/.exec(body);
  if (!sep || sep.index === undefined) {
    return { done, active: { search: body, replace: null } };
  }
  const search = body.slice(0, sep.index);
  const replace = body.slice(sep.index + sep[0].length);
  return { done, active: { search, replace } };
}

/** Replace `search` in `html` leniently (CR/trailing-space tolerant). */
function replaceOnce(html: string, search: string, replace: string): string | null {
  if (!search) return null;
  if (html.includes(search)) return html.replace(search, replace);
  const nHtml = norm(html);
  const nSearch = norm(search);
  if (nSearch && nHtml.includes(nSearch)) return nHtml.replace(nSearch, norm(replace));
  return null;
}

/** Partial full-document extraction: the content of an (unclosed) ```html
 *  fence, from its doctype/html start. */
function partialFullHtml(text: string): string | null {
  const fence = /```(?:html|htm)?\s*\n/i.exec(text);
  if (!fence || fence.index === undefined) return null;
  let body = text.slice(fence.index + fence[0].length);
  const closing = body.indexOf("```");
  if (closing >= 0) body = body.slice(0, closing);
  const start = body.search(/<!doctype html|<html/i);
  if (start < 0) return null;
  return body.slice(start);
}

function toScanRows(rows: DiffRow[]): ScanRow[] {
  return rows.map((r) => ({ kind: r.kind, text: r.text }));
}

/** Full-document diff rows: the whole file as ctx with the changed hunk
 *  inline (diffLines itself trims to hunk±3 — a scan wants the full scroll). */
function fullDiffRows(base: string, next: string): ScanRow[] {
  const a = base.length ? base.split("\n") : [];
  const b = next.length ? next.split("\n") : [];
  let s = 0;
  while (s < a.length && s < b.length && a[s] === b[s]) s++;
  let ea = a.length;
  let eb = b.length;
  while (ea > s && eb > s && a[ea - 1] === b[eb - 1]) {
    ea--;
    eb--;
  }
  const rows: ScanRow[] = a.slice(0, s).map((text) => ({ kind: "ctx" as const, text }));
  if (ea > s || eb > s) {
    const mid = diffLines(a.slice(s, ea).join("\n"), b.slice(s, eb).join("\n"), Number.MAX_SAFE_INTEGER);
    rows.push(...toScanRows(mid.rows));
  }
  rows.push(...a.slice(ea).map((text) => ({ kind: "ctx" as const, text })));
  return rows;
}

// Applying the done patches re-walks the whole document on every stream tick,
// but the done set only grows when a block completes — cache the applied
// result and re-derive only the active block per tick.
let applyCache: { base: string; count: number; virtual: string } | null = null;

export function buildScanView(base: string, acc: string): ScanView {
  // Full-document mode wins when a fenced document has started streaming.
  const full = partialFullHtml(acc);
  if (full !== null) {
    const raw = fullDiffRows(base, full);
    // Old lines past the last line the stream has matched are simply not
    // reached yet: every del AFTER the final ctx row is pending, not removed.
    // They render below the streamed content, dimmed — the shrinking tail.
    let lastCtx = -1;
    raw.forEach((r, k) => {
      if (r.kind === "ctx") lastCtx = k;
    });
    const resolved: ScanRow[] = [];
    const pending: ScanRow[] = [];
    raw.forEach((r, k) => {
      if (r.kind === "del" && k > lastCtx) pending.push({ kind: "pending", text: r.text });
      else resolved.push(r);
    });
    const rows = [...resolved, ...pending];
    let scan = -1;
    for (let k = 0; k < resolved.length; k++) if (rows[k].kind !== "ctx") scan = k;
    if (scan < 0 && resolved.length) scan = resolved.length - 1;
    return { rows, scanIndex: scan >= 0 ? scan : null, mode: "full" };
  }

  const { done, active } = parseStreamPatches(acc);
  if (!done.length && !active) {
    // Nothing parseable yet (reasoning/prose still streaming) — show the
    // current code untouched, scan head parked at the top.
    return {
      rows: base.split("\n").map((text) => ({ kind: "ctx" as const, text })),
      scanIndex: null,
      mode: "waiting",
    };
  }

  let virtual: string;
  if (applyCache && applyCache.base === base && applyCache.count === done.length) {
    virtual = applyCache.virtual;
  } else {
    virtual = base;
    for (const e of done) {
      const next = replaceOnce(virtual, e.search, e.replace);
      if (next !== null) virtual = next;
    }
    applyCache = { base, count: done.length, virtual };
  }
  let working = virtual;
  let searchAnchor: string | null = null;
  if (active) {
    if (active.replace !== null) {
      const next = replaceOnce(working, active.search, active.replace);
      if (next !== null) working = next;
      else searchAnchor = active.search.split("\n")[0] || null;
    } else {
      // Still copying the SEARCH block: highlight where it points so far.
      searchAnchor = active.search.split("\n")[0] || null;
    }
  }

  const rows = fullDiffRows(base, working);
  let lastChanged = -1;
  for (let k = 0; k < rows.length; k++) {
    if (rows[k].kind === "add" || rows[k].kind === "del") lastChanged = k;
  }
  // While the model is COPYING a SEARCH block it is "reading" that source
  // region — the scan head belongs there, ahead of already-resolved patches.
  let anchorIdx = -1;
  if (searchAnchor) {
    // The SEARCH text streams in character by character — match by prefix.
    const target = searchAnchor.trim();
    if (target.length >= 3) {
      anchorIdx = rows.findIndex((r) => r.kind === "ctx" && r.text.trim().startsWith(target));
    }
  }
  const scan = anchorIdx >= 0 ? anchorIdx : lastChanged;
  return { rows, scanIndex: scan >= 0 ? scan : null, mode: "patch" };
}
