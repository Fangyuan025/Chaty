/** Model downloads, held by the app rather than by the window that started
 *  them. The store and "Set up for me" used to keep a download's progress in
 *  their own state: the store could not be closed while one ran, and one
 *  started from the setup cards never showed in the store. Now either can be
 *  closed — the download goes on in the background, the rest of Chaty in use
 *  — and opening the store again shows every download as it stands, wherever
 *  it was started. */
import { useSyncExternalStore } from "react";
import { etaSeconds, type EtaSample } from "./eta";
import { cancelDownload } from "./ipc";

export type DlState = "running" | "done" | "error" | "cancelled";

export interface DlTask {
  /** One per thing being fetched: the store's repo and quant, a setup card. */
  id: string;
  /** What it is, as the store or the setup card names it. */
  name: string;
  /** Which build of it — a quant, the MLX folder. */
  variant: string;
  /** The repo, for the store to open it on a click. */
  repo: string;
  done: number;
  total: number;
  /** Seconds left, once there is enough to say. */
  eta: number | null;
  state: DlState;
  error?: string;
  /** The file in flight: what a cancel names to the backend. */
  key: string | null;
  /** The model's file or folder, once it is in. */
  path?: string;
  startedAt: number;
}

let tasks: DlTask[] = [];
const listeners = new Set<() => void>();
const doneListeners = new Set<(t: DlTask) => void>();
const samples = new Map<string, EtaSample[]>();
/** Asked to stop: a download of several files stops between them too. */
const stopping = new Set<string>();

function emit() {
  for (const l of listeners) l();
}

function patch(id: string, p: Partial<DlTask>) {
  tasks = tasks.map((x) => (x.id === id ? { ...x, ...p } : x));
  emit();
}

export function dlGet(id: string): DlTask | undefined {
  return tasks.find((x) => x.id === id);
}

/** Begin one; false when the same thing is already being fetched. */
export function dlBegin(t: Pick<DlTask, "id" | "name" | "variant" | "repo" | "total">): boolean {
  const cur = dlGet(t.id);
  if (cur?.state === "running") return false;
  stopping.delete(t.id);
  samples.set(t.id, []);
  const task: DlTask = { ...t, done: 0, eta: null, state: "running", key: null, startedAt: Date.now() };
  tasks = [task, ...tasks.filter((x) => x.id !== t.id)];
  emit();
  return true;
}

/** The file now in flight (what a cancel stops). */
export function dlFile(id: string, key: string) {
  patch(id, { key });
}

export function dlProgress(id: string, done: number, total?: number) {
  const cur = dlGet(id);
  if (!cur || cur.state !== "running") return;
  const tot = total ?? cur.total;
  patch(id, { done, total: tot, eta: etaSeconds(samples.get(id) ?? [], done, tot) });
}

export function dlDone(id: string, path?: string) {
  const cur = dlGet(id);
  if (!cur) return;
  patch(id, { state: "done", path, done: cur.total || cur.done, eta: null, key: null });
  const done = dlGet(id);
  if (done) for (const l of doneListeners) l(done);
}

export function dlFailed(id: string, message: string) {
  patch(id, { state: "error", error: message, eta: null, key: null });
}

/** Stopped by the user: nothing to show for it, so it goes. */
export function dlCancelled(id: string) {
  stopping.delete(id);
  samples.delete(id);
  tasks = tasks.filter((x) => x.id !== id);
  emit();
}

/** Whether a stop was asked for: checked between the files of one download. */
export function dlStopping(id: string): boolean {
  return stopping.has(id);
}

/** Stop one: the file in flight is cancelled (its partial removed), and the
 *  files after it are not started. */
export async function dlCancel(id: string) {
  const cur = dlGet(id);
  if (!cur || cur.state !== "running") return;
  stopping.add(id);
  if (cur.key) await cancelDownload(cur.key).catch(() => {});
}

/** Put away a finished, failed or cancelled one. */
export function dlDismiss(id: string) {
  const cur = dlGet(id);
  if (!cur || cur.state === "running") return;
  tasks = tasks.filter((x) => x.id !== id);
  samples.delete(id);
  emit();
}

/** Every download finished, as it finishes. */
export function onDownloadDone(fn: (t: DlTask) => void): () => void {
  doneListeners.add(fn);
  return () => doneListeners.delete(fn);
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function useDownloads(): DlTask[] {
  return useSyncExternalStore(subscribe, () => tasks);
}

/** Of the running ones, how far along together (0–1), or null when none run. */
export function dlOverall(list: DlTask[]): number | null {
  const run = list.filter((x) => x.state === "running");
  if (!run.length) return null;
  const total = run.reduce((a, x) => a + x.total, 0);
  const done = run.reduce((a, x) => a + Math.min(x.done, x.total || x.done), 0);
  return total > 0 ? done / total : 0;
}
