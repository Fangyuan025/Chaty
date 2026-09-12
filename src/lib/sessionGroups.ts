import type { CodeSessionMeta } from "./ipc";

export interface SessionGroup {
  /** The workspace path, or null for sessions that never had one. */
  path: string | null;
  /** What the rail shows: the folder's own name. */
  name: string;
  sessions: CodeSessionMeta[];
}

/** A workspace's folder name, whichever separator its path uses. */
export function workspaceName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

/** The session rail grouped by workspace. Sessions arrive newest first, and
 *  so do the groups: a workspace sits where its most recent session would, and
 *  keeps its sessions in the order they came. */
export function groupSessionsByWorkspace(sessions: CodeSessionMeta[]): SessionGroup[] {
  const groups = new Map<string | null, SessionGroup>();
  for (const s of sessions) {
    const path = s.workspace || null;
    let g = groups.get(path);
    if (!g) {
      g = { path, name: path ? workspaceName(path) : "", sessions: [] };
      groups.set(path, g);
    }
    g.sessions.push(s);
  }
  return [...groups.values()];
}
