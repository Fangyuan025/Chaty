/** Conversations grouped the way a person looks for them — by how recently
 *  they were touched — with pinned ones first. Order inside a group is kept as
 *  given (the list arrives newest first). Day boundaries are the viewer's
 *  local midnights, not 24-hour windows: something from last night is
 *  "yesterday" at nine in the morning. */
export type RecencyKey = "pinned" | "today" | "yesterday" | "week" | "month" | "older";

export interface RecencyGroup<T> {
  key: RecencyKey;
  items: T[];
}

const ORDER: RecencyKey[] = ["pinned", "today", "yesterday", "week", "month", "older"];

export function recencyGroups<T extends { updatedAt: number; pinned?: boolean }>(
  items: T[],
  now: number = Date.now(),
): RecencyGroup<T>[] {
  const midnight = new Date(now);
  midnight.setHours(0, 0, 0, 0);
  const today = midnight.getTime();
  const day = 86_400_000;
  const keyOf = (c: T): RecencyKey => {
    if (c.pinned) return "pinned";
    const t = c.updatedAt;
    if (t >= today) return "today";
    if (t >= today - day) return "yesterday";
    if (t >= today - 7 * day) return "week";
    if (t >= today - 30 * day) return "month";
    return "older";
  };
  const by = new Map<RecencyKey, T[]>();
  for (const c of items) {
    const k = keyOf(c);
    const list = by.get(k);
    if (list) list.push(c);
    else by.set(k, [c]);
  }
  return ORDER.filter((k) => by.has(k)).map((key) => ({ key, items: by.get(key)! }));
}
