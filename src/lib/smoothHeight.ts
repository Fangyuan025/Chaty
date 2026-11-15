// Things that open and close — a folded code block, a thought, the sources,
// a long message — go from one height to the other instead of jumping there.
// The change is made at once (React renders it in the same handler, through
// flushSync), and before the frame is painted the element is put back at the
// height it had and let go to its new one.
import { useCallback, useLayoutEffect, useRef } from "react";
import { flushSync } from "react-dom";

/** The user asked for less motion — of their system, or in Settings (which
 *  the stylesheet already honoured, and the motion made from here did not). */
export function prefersStill(): boolean {
  if (typeof document !== "undefined" && document.documentElement.dataset.motion === "reduce") return true;
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
    // A frame held up (what opened being drawn) moves it on by a frame's
    // worth, not by all the time it took: a slow frame slows it, where it
    // leapt.
    const dt = Math.min(20, Math.max(0, now - last));
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

/** Height animated whenever `key` changes — by a click or by itself (a card
 *  opening as its call is written, a thought folding as it ends): every open
 *  and close of the element the returned ref is put on goes the same way.
 *  Its height is kept as it changes (a ResizeObserver, read after layout) to
 *  go from; the first render is drawn where it is. */
export function useHeightOnChange<T extends HTMLElement>(key: unknown, ms = 260): (node: T | null) => void {
  const el = useRef<T | null>(null);
  const lastH = useRef(0);
  const ro = useRef<ResizeObserver | null>(null);
  const ref = useCallback((node: T | null) => {
    ro.current?.disconnect();
    el.current = node;
    lastH.current = 0;
    if (node && typeof ResizeObserver !== "undefined") {
      ro.current = new ResizeObserver(([e]) => {
        lastH.current = e.borderBoxSize?.[0]?.blockSize ?? node.getBoundingClientRect().height;
      });
      ro.current.observe(node);
    }
  }, []);
  const first = useRef(true);
  useLayoutEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const node = el.current;
    if (node && lastH.current > 0) animateHeight(node, lastH.current, ms);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return ref;
}
