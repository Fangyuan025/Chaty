// Line-level diff for Code mode's edit previews and diffstat badges.
//
// The old approach trimmed the common prefix/suffix and treated the whole
// middle as "all removed then all added" — which mis-counts when changes are
// scattered (unchanged middle lines got flagged as changed) and the render was
// hard-capped at 60 rows, so the +N/−M badge silently under-counted big edits.
//
// This does a real LCS diff (only actually-changed lines are add/del) and
// keeps the stat separate from the render cap, so the badge is always exact.

export type DiffKind = "ctx" | "add" | "del";
export interface DiffRow {
  kind: DiffKind;
  text: string;
}
export interface DiffResult {
  rows: DiffRow[];
  /** Exact totals over the WHOLE diff, independent of any render cap. */
  added: number;
  removed: number;
  /** True when `rows` was capped for display (badge stays exact regardless). */
  truncated: boolean;
}

/** LCS diff of two line arrays → unified add/del/ctx rows. O(n·m) time+space;
 *  callers bound n·m before calling (see diffLines). */
function lcsDiff(a: string[], b: string[]): DiffRow[] {
  const n = a.length;
  const m = b.length;
  // dp[i][j] = LCS length of a[i:] and b[j:].
  const dp: Uint32Array[] = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    const row = dp[i];
    const next = dp[i + 1];
    for (let j = m - 1; j >= 0; j--) {
      row[j] = a[i] === b[j] ? next[j + 1] + 1 : Math.max(next[j], row[j + 1]);
    }
  }
  const rows: DiffRow[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      rows.push({ kind: "ctx", text: a[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      rows.push({ kind: "del", text: a[i] });
      i++;
    } else {
      rows.push({ kind: "add", text: b[j] });
      j++;
    }
  }
  while (i < n) rows.push({ kind: "del", text: a[i++] });
  while (j < m) rows.push({ kind: "add", text: b[j++] });
  return rows;
}

const MAX_RENDER_ROWS = 400;
// Above this the middle LCS matrix (differing region) would be too large; fall
// back to a coarse "all removed then all added" for that region only.
const LCS_CELL_LIMIT = 4_000_000;

/**
 * Diff `before` → `after` by lines. Trims the common prefix/suffix cheaply,
 * runs LCS only on the differing middle (so it stays fast on big files with a
 * small edit), and returns exact add/removed counts plus render-ready rows
 * (with a few lines of surrounding context).
 */
export function diffLines(before: string, after: string, renderCap = MAX_RENDER_ROWS): DiffResult {
  const a = before.length ? before.split("\n") : [];
  const b = after.length ? after.split("\n") : [];

  // Common prefix / suffix — untouched regions we never need to render or count.
  let s = 0;
  while (s < a.length && s < b.length && a[s] === b[s]) s++;
  let ea = a.length;
  let eb = b.length;
  while (ea > s && eb > s && a[ea - 1] === b[eb - 1]) {
    ea--;
    eb--;
  }

  const midA = a.slice(s, ea);
  const midB = b.slice(s, eb);
  let mid: DiffRow[];
  if (midA.length * midB.length > LCS_CELL_LIMIT) {
    mid = [
      ...midA.map((text): DiffRow => ({ kind: "del", text })),
      ...midB.map((text): DiffRow => ({ kind: "add", text })),
    ];
  } else {
    mid = lcsDiff(midA, midB);
  }

  // Exact stat over the full middle (context lines never count).
  const added = mid.reduce((k, r) => k + (r.kind === "add" ? 1 : 0), 0);
  const removed = mid.reduce((k, r) => k + (r.kind === "del" ? 1 : 0), 0);

  // Assemble with up to 3 lines of surrounding context.
  const rows: DiffRow[] = [];
  for (let i = Math.max(0, s - 3); i < s; i++) rows.push({ kind: "ctx", text: a[i] });
  rows.push(...mid);
  for (let i = ea; i < Math.min(a.length, ea + 3); i++) rows.push({ kind: "ctx", text: a[i] });

  const truncated = rows.length > renderCap;
  return {
    rows: truncated ? rows.slice(0, renderCap) : rows,
    added,
    removed,
    truncated,
  };
}
