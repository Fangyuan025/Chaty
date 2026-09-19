/** When the way back to the end is offered. The button is decided by the
 *  DISTANCE to the end — and a transcript shrinks on its own (a thinking
 *  panel folds away, a live card is withdrawn) with no scroll event to
 *  recompute it, which is how it came to sit over a conversation whose end
 *  was already on screen. */
import { expect, it } from "vitest";

/** The rule itself, as CodeMode applies it. */
const offersJump = (scrollHeight: number, scrollTop: number, clientHeight: number) =>
  scrollHeight - scrollTop - clientHeight > 320;

it("offers the way back only when the end is far off", () => {
  // A short session, wherever it sits: never.
  expect(offersJump(657, 0, 590)).toBe(false);
  expect(offersJump(657, 67, 590)).toBe(false);
  // A long one, scrolled up: yes.
  expect(offersJump(2073, 0, 590)).toBe(true);
  // …and the moment the content shrinks under it — same scrollTop, no scroll
  // event — the end is back in view and the offer goes.
  expect(offersJump(873, 0, 590)).toBe(false);
  // The boundary is the boundary.
  expect(offersJump(911, 0, 590)).toBe(true);
  expect(offersJump(910, 0, 590)).toBe(false);
});
