// Things that open and close — a folded code block, a thought, the sources,
// a long message — go from one height to the other instead of jumping there.
// The change is made at once (React renders it in the same handler, through
// flushSync), and before the frame is painted the element is put back at the
// height it had and let go to its new one.
import { flushSync } from "react-dom";

/** The user asked their system for less motion. */
export function prefersStill(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const running = new WeakMap<HTMLElement, () => void>();

/** Until when something on the page is folding away on its own (a thought
 *  folding as its answer starts, a block settling into its window). While it
 *  does, a view kept at the end is moved up by the page shrinking under it —
 *  which is not the user scrolling, and must not be read as such (see the
 *  transcripts' follow). */
let shrinkingUntil = 0;
export function noteShrink(ms: number): void {
  shrinkingUntil = Math.max(shrinkingUntil, performance.now() + ms);
}
export function shrinking(): boolean {
  return performance.now() < shrinkingUntil;
}

/** Take `el` from height `from` to the height it has now.
 *
 *  Opening, the height it goes to is read again every frame: what opens is
 *  not always all there the moment it opens — a diagram drawn a moment later,
 *  a picture that loads, highlighted code — and a height fixed when it opened
 *  ran out short of it, the rest arriving in a jump as the animation let go.
 *  Closing goes to the height it was given. */
export function animateHeight(el: HTMLElement, from: number, ms = 260): void {
  running.get(el)?.();
  const to = el.getBoundingClientRect().height;
  if (Math.abs(to - from) < 2 || prefersStill()) return;
  const opening = to > from;
  if (!opening) noteShrink(ms * 3 + 120);
  const cs = getComputedStyle(el);
  const borders = (parseFloat(cs.borderTopWidth) || 0) + (parseFloat(cs.borderBottomWidth) || 0);
  const keep = { height: el.style.height, maxHeight: el.style.maxHeight, overflow: el.style.overflow, transition: el.style.transition };
  el.style.transition = "none";
  el.style.maxHeight = "none";
  el.style.overflow = "hidden";
  el.style.height = `${from}px`;
  let raf = 0;
  const done = () => {
    cancelAnimationFrame(raf);
    running.delete(el);
    el.style.height = keep.height;
    el.style.maxHeight = keep.maxHeight;
    el.style.overflow = keep.overflow;
    el.style.transition = keep.transition;
  };
  // Followed rather than tweened: each frame closes a share of the gap to
  // where it is going (by the time since the last), so a target that moves
  // part-way — more content arriving — is gone on to without a break, where a
  // fixed curve recomputed toward it leapt there in a frame.
  const tau = ms / 4;
  const t0 = performance.now();
  let last = t0;
  let h = from;
  const step = (now: number) => {
    const dt = Math.max(0, now - last);
    last = now;
    // What it holds now, all of it (the box is held shorter, so its own
    // height says nothing — its content's does).
    const target = opening ? Math.max(to, el.scrollHeight + borders) : to;
    h += (target - h) * (1 - Math.exp(-dt / tau));
    if (Math.abs(target - h) < 0.5 || now - t0 > ms * 3) {
      done();
      return;
    }
    el.style.height = `${h}px`;
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  running.set(el, done);
}

/** Make the change `change` makes to `el` (a state update) gradually. */
export function smoothly(el: HTMLElement | null | undefined, change: () => void, ms = 260): void {
  if (!el || prefersStill()) {
    change();
    return;
  }
  // Where it is now — part-way through an earlier change, it goes on from
  // there rather than from where that one started.
  const from = el.getBoundingClientRect().height;
  running.get(el)?.();
  flushSync(change);
  animateHeight(el, from, ms);
}
