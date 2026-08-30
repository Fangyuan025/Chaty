import type { TKey } from "./i18n";

/** The label for one rung of a model's own reasoning-effort ladder.
 *
 *  The key map lives here rather than in `i18n.tsx` on purpose: the dead-key
 *  guard reads that file as definitions only, so a map kept inside it would
 *  make every rung look unreferenced.
 *
 *  A rung Chaty has no name for is shown verbatim — a ladder it has not seen
 *  before is better read as the model wrote it than mislabelled as one it has.
 */
export function effortLabel(rung: string, t: (key: TKey) => string): string {
  const keys: Record<string, TKey> = {
    low: "effortLow",
    medium: "effortMedium",
    high: "effortHigh",
    xhigh: "effortXhigh",
  };
  const key = keys[rung];
  return key ? t(key) : rung;
}
