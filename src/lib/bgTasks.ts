import type { AgentBgInfo } from "./ipc";

export type BgStatus = "running" | "done" | "stopped" | "failed";

/** How a background command ended — or that it has not yet. Stopping one is
 *  not failing: a killed process exits non-zero, and reading that as a failure
 *  would paint every dev server the user shut down red. */
export function bgStatus(j: Pick<AgentBgInfo, "running" | "code" | "killed">): BgStatus {
  if (j.running) return "running";
  if (j.killed) return "stopped";
  return j.code === 0 ? "done" : "failed";
}

/** Running time as the panel shows it: 45s, 3m 20s, 1h 05m. */
export function fmtElapsed(secs: number): string {
  const s = Math.max(0, Math.floor(secs));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

/** The line a job is known by: its command's first non-empty line. */
export function bgTitle(command: string): string {
  return (
    command
      .split("\n")
      .map((l) => l.trim())
      .find(Boolean) ?? command
  );
}
