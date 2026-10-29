/** The one download of a new version, shared by everything that can start it
 *  or show it — the banner and Settings → About each kept their own, so a
 *  download started in one left the other offering to start a second. */
import { useSyncExternalStore } from "react";
import { runUpdate, type UpdateProgress } from "./ipc";

export interface UpdateDownload {
  running: boolean;
  progress: UpdateProgress | null;
  /** Why the last attempt stopped, when it did. */
  error: string;
}

let state: UpdateDownload = { running: false, progress: null, error: "" };
const listeners = new Set<() => void>();

function set(next: Partial<UpdateDownload>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

/** Start the download of `url`, unless one is already going. Resolves when it
 *  fails (on success the app exits to install); the reason is in `error`. */
export async function startUpdateDownload(url: string): Promise<void> {
  if (state.running) return;
  set({ running: true, progress: null, error: "" });
  try {
    await runUpdate(url, (p) => set({ progress: p }));
  } catch (e) {
    set({ running: false, progress: null, error: typeof e === "string" ? e : ((e as Error)?.message ?? String(e)) });
  }
}

export function useUpdateDownload(): UpdateDownload {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => state,
  );
}
