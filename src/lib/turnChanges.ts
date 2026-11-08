import { changeCounts } from "./diff";
import type { CpChange } from "./ipc";

/** One file a turn changed, as the card under its answer keeps it. */
export interface TurnChange {
  /** Absolute path — what undo through the turn's checkpoint is keyed by. */
  path: string;
  /** Relative to the workspace: what the card shows, and what a kept-copy
   *  undo writes back to. */
  rel: string;
  added: number;
  removed: number;
  created?: boolean;
  deleted?: boolean;
  binary?: boolean;
  /** Contents kept for the diff and for undo once the checkpoint is gone —
   *  absent when too large to carry in the session. */
  before?: string;
  after?: string;
  undone?: boolean;
}

/** Past this, a file's two versions are not carried in the session file: the
 *  counts stay exact, the diff and the kept-copy undo go. Same bound as a
 *  step card's diff. */
export const KEEP_CHARS = 200_000;

export function toTurnChange(c: CpChange): TurnChange {
  const before = c.before ?? "";
  const after = c.after ?? "";
  // Counted as a step's card and a folded run count it (changeCounts).
  const counts = c.binary ? { added: 0, removed: 0 } : changeCounts(before, after);
  const keep = !c.binary && before.length + after.length <= KEEP_CHARS;
  return {
    path: c.path,
    rel: c.rel,
    added: counts.added,
    removed: counts.removed,
    ...(c.created ? { created: true } : {}),
    ...(c.deleted ? { deleted: true } : {}),
    ...(c.binary ? { binary: true } : {}),
    ...(keep ? { before, after } : {}),
  };
}

/** Past this many files the card starts folded to its totals: a turn that
 *  touched thirty files would otherwise push its own answer off screen. */
export const FOLD_OVER = 5;

export function startsFolded(files: number): boolean {
  return files > FOLD_OVER;
}

/** Lines added and removed across all of a turn's files. */
export function changeTotals(changes: TurnChange[]): { added: number; removed: number } {
  return changes.reduce(
    (t, c) => ({ added: t.added + c.added, removed: t.removed + c.removed }),
    { added: 0, removed: 0 },
  );
}

/** How a file comes back without the turn's checkpoint: null removes it
 *  (the turn created it), a string is written back; undefined — the card
 *  kept no copy — means it cannot. */
export function keptRestore(c: TurnChange): string | null | undefined {
  if (c.created) return null;
  if (c.binary) return undefined;
  return c.before;
}
