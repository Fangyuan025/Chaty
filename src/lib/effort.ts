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

/** Which tab of the code-mode thinking switch is lit.
 *
 *  Without a ladder the tab IS the mode. With one, every tab but `off` stands
 *  for a rung of the model's own ladder, so what is lit follows the rung —
 *  never the generic intensity that rung happens to map onto.
 */
export function thinkTabActive(
  tab: string,
  { nativeEffort, thinkMode, rung }: { nativeEffort: boolean; thinkMode: string; rung: string },
): boolean {
  if (!nativeEffort) return thinkMode === tab;
  if (tab === "off") return thinkMode === "off";
  return thinkMode !== "off" && rung === tab;
}

/** Which of Chaty's own intensities a rung stands at.
 *
 *  The intensity still drives the thinking budget and the prompt-side nudge,
 *  neither of which the model's ladder says anything about. Read from the
 *  rung's POSITION rather than its name, so a ladder of any length works —
 *  and so a three-rung ladder lands exactly where the old fixed mapping put
 *  it.
 */
export function intensityOf(levels: string[], rung: string): "off" | "low" | "normal" | "deep" {
  const i = levels.indexOf(rung);
  if (i <= 0) return "low";
  return i === levels.length - 1 ? "deep" : "normal";
}

/** The rung a model will actually use, given one remembered from another.
 *
 *  The choice is kept per app, not per model, and ladders differ in length —
 *  so a rung picked on a four-rung model can be absent from a three-rung one.
 *  Left unresolved the menu shows nothing selected while the model quietly
 *  falls back to whatever its template defaults to, which is not necessarily
 *  what the menu last showed.
 *
 *  Clamped by position, so the intent survives the move: the third rung of
 *  four becomes the third of three rather than the bottom one.
 */
export function clampRung(levels: string[], rung: string): string {
  if (levels.length === 0 || levels.includes(rung)) return rung;
  const ladder = ["low", "medium", "high", "xhigh"];
  const from = ladder.indexOf(rung);
  return levels[from < 0 ? levels.length - 1 : Math.min(from, levels.length - 1)];
}
