// Things that open and close — a folded code block, a thought, the sources,
// a long message — go from one height to the other instead of jumping there.
// The change is made at once (React renders it in the same handler, through
// flushSync), and before the frame is painted the element is put back at the
// height it had and let go to its new one.
import { flushSync } from "react-dom";

const EASE = "cubic-bezier(0.22, 0.8, 0.24, 1)";

/** The user asked their system for less motion. */
export function prefersStill(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const running = new WeakMap<HTMLElement, () => void>();

/** Take `el` from height `from` to the height it has now. */
export function animateHeight(el: HTMLElement, from: number, ms = 260): void {
  running.get(el)?.();
  const to = el.getBoundingClientRect().height;
  if (Math.abs(to - from) < 2 || prefersStill()) return;
  const keep = { height: el.style.height, maxHeight: el.style.maxHeight, overflow: el.style.overflow, transition: el.style.transition };
  el.style.transition = "none";
  el.style.height = `${from}px`;
  el.style.maxHeight = "none";
  el.style.overflow = "hidden";
  void el.offsetHeight;
  el.style.transition = `height ${ms}ms ${EASE}`;
  el.style.height = `${to}px`;
  let timer = 0;
  const done = () => {
    el.removeEventListener("transitionend", onEnd);
    clearTimeout(timer);
    running.delete(el);
    el.style.height = keep.height;
    el.style.maxHeight = keep.maxHeight;
    el.style.overflow = keep.overflow;
    el.style.transition = keep.transition;
  };
  const onEnd = (e: TransitionEvent) => {
    if (e.target === el && e.propertyName === "height") done();
  };
  el.addEventListener("transitionend", onEnd);
  timer = window.setTimeout(done, ms + 120);
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
