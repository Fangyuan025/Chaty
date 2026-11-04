/**
 * Work that goes on while the window may well be elsewhere — model and voice
 * downloads, pictures, music, engine and app updates. While any is under way
 * its progress shows on the Dock icon / taskbar button and the system does not
 * idle into sleep (a 20 GB download overnight stopped when the Mac slept). When
 * a long one ends with the window in the background, the Dock icon bounces
 * once / the taskbar button flashes.
 *
 * The Code agent holds the machine awake while it works too (`holdAwake`), and
 * asks for attention when it stops to ask (`askAttention`).
 */
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow, ProgressBarStatus, UserAttentionType } from "@tauri-apps/api/window";

/** A job's progress: a fraction, or null while how far is not known. */
const jobs = new Map<number, number | null>();
/** Jobs that keep the machine awake but put nothing on the Dock icon. */
const silent = new Set<number>();
/** Holds that keep the machine awake without a progress of their own. */
const holds = new Set<string>();
let nextId = 1;
let pending: ReturnType<typeof setTimeout> | 0 = 0;
let awake = false;

/** Long enough that its end is worth a nudge when the window is elsewhere. */
const NUDGE_AFTER_MS = 15_000;

function win() {
  try {
    return getCurrentWindow();
  } catch {
    return null; // a plain-browser preview has no native window
  }
}

/** Show the state, at most every 400 ms: progress events arrive far faster. */
function schedule() {
  if (pending) return;
  pending = globalThis.setTimeout(() => {
    pending = 0;
    apply();
  }, 400);
}

function apply() {
  const w = win();
  const shown = [...jobs.entries()].filter(([id]) => !silent.has(id)).map(([, f]) => f);
  const known = shown.filter((f): f is number => f != null);
  if (shown.length === 0) {
    void w?.setProgressBar({ status: ProgressBarStatus.None }).catch(() => {});
  } else if (known.length === 0) {
    void w?.setProgressBar({ status: ProgressBarStatus.Indeterminate }).catch(() => {});
  } else {
    const frac = known.reduce((a, b) => a + b, 0) / known.length;
    const progress = Math.max(1, Math.min(100, Math.round(frac * 100)));
    void w?.setProgressBar({ status: ProgressBarStatus.Normal, progress }).catch(() => {});
  }
  const want = jobs.size > 0 || holds.size > 0;
  if (want !== awake) {
    awake = want;
    void invoke("keep_awake", { on: want }).catch(() => {});
  }
}

export interface Job {
  /** How far, 0..1; null: under way, amount unknown. */
  update(frac: number | null): void;
  /** Done (or failed, or stopped): gone from the Dock icon; a long one that
   *  finished well nudges a window in the background. */
  end(ok?: boolean): void;
}

/** `show: false` — a chat reply: it keeps the machine awake and nudges when a
 *  long one ends, but a bar on the Dock icon for a few seconds is noise. */
export function startJob(opts: { show?: boolean } = {}): Job {
  const id = nextId++;
  const quiet = opts.show === false;
  if (quiet) silent.add(id);
  const started = Date.now();
  let ended = false;
  jobs.set(id, null);
  schedule();
  return {
    update(frac) {
      if (ended) return;
      jobs.set(id, frac == null || !Number.isFinite(frac) ? null : Math.max(0, Math.min(1, frac)));
      schedule();
    },
    end(ok = true) {
      if (ended) return;
      ended = true;
      jobs.delete(id);
      silent.delete(id);
      schedule();
      // A model call inside an agent's run is one step of it: the run nudges
      // once, when it ends, not at every long step.
      if (ok && Date.now() - started >= NUDGE_AFTER_MS && !(quiet && holds.size > 0)) askAttention();
    },
  };
}

/** Run `work` as a job: it ends when the promise settles. */
export async function asJob<T>(work: (job: Job) => Promise<T>, opts: { show?: boolean } = {}): Promise<T> {
  const job = startJob(opts);
  let ok = false;
  try {
    const out = await work(job);
    ok = true;
    return out;
  } finally {
    job.end(ok);
  }
}

/** Keep the machine from idling into sleep while `key` is held. */
export function holdAwake(key: string, on: boolean) {
  if (on) holds.add(key);
  else holds.delete(key);
  schedule();
}

/** The Dock icon bounces once / the taskbar button flashes — only when the
 *  window is not the one in front. */
export function askAttention() {
  if (typeof document !== "undefined" && document.hasFocus()) return;
  void win()?.requestUserAttention(UserAttentionType.Informational).catch(() => {});
}
