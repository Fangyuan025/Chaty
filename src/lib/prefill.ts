/** Prompt processing as a whole percent, the one number every place shows it
 *  — the ring in chat and under Code's composer, the folded run's line: what
 *  the engine has reported, never ahead of it, and 100 only once the whole
 *  prompt is read. The ring used to run its own number — crept ahead of the
 *  report between events and held under 100 — so it read a few percent off
 *  the line beside it, and at the end flickered 99, 100, 99, 100 as the creep
 *  cap and the report took turns. */
export function prefillPct(frac: number): number {
  if (!Number.isFinite(frac)) return 0;
  return Math.floor(Math.min(1, Math.max(0, frac)) * 100 + 1e-9);
}
