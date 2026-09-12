/** How far (px, ≥ 0, leftwards) to move a hover preview that hangs off its
 *  anchor's left edge so it ends inside [boundLeft, boundRight]. A column
 *  narrower than the preview keeps its left edge in bounds instead. */
export function popShift(
  anchorLeft: number,
  popWidth: number,
  boundLeft: number,
  boundRight: number,
  margin = 8,
): number {
  const over = anchorLeft + popWidth - (boundRight - margin);
  if (over <= 0) return 0;
  return Math.max(0, Math.min(over, anchorLeft - (boundLeft + margin)));
}

/** Keep an anchor's `.cite-pop` inside the chat column. The preview hangs off
 *  its anchor at a fixed width, so under a source chip near the right edge it
 *  poked past the column and gave it a horizontal scrollbar. Where scrollbars
 *  take room (Windows) that moved the chip out from under the pointer, which
 *  hid the preview, which dropped the scrollbar and brought the chip back —
 *  a flicker for as long as the pointer stayed (issue #14). */
export function fitPop(anchor: HTMLElement): void {
  const pop = anchor.querySelector<HTMLElement>(".cite-pop");
  if (!pop) return;
  const col = anchor.closest<HTMLElement>(".chat") ?? document.documentElement;
  const left = col.getBoundingClientRect().left + col.clientLeft;
  const width = pop.offsetWidth || Math.min(320, window.innerWidth * 0.7);
  const shift = popShift(anchor.getBoundingClientRect().left, width, left, left + col.clientWidth);
  pop.style.left = `${-shift}px`;
}
