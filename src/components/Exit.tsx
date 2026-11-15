// What appears over the page — a menu, a sheet, a preview — goes the way it
// came, fading out over a moment, where it used to vanish in a frame. Wrap
// the conditional: <Exit>{open && <Menu … />}</Exit>. Taken away, the last
// of it stays a moment, not to be clicked, while it fades.
//
// `collapse`: something in the flow of the page rather than over it (a live
// thought that a step's card takes over from) — it opens from nothing as it
// comes and folds to nothing as it fades, so what is under it moves smoothly
// instead of jumping: the gap it had between its neighbours (a column's
// `gap`) closing with it, which went in a frame as it was taken away.
//
// `away`: said rather than read from the children — going while still drawn
// as it is now (its own parts changing as it folds), not as it last was.
import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import { animateHeight, noteShrink, prefersStill } from "../lib/smoothHeight";

/** Whether a conditional's result draws anything: `text && <X/>` is "" when
 *  the text is, and that is as gone as false. */
function drawn(children: ReactNode): boolean {
  return children != null && typeof children !== "boolean" && children !== "";
}

/** Its height in the layout — not the box a transform draws it in: a
 *  spinner turned 45° measured half again as tall, and folded from there it
 *  first grew. */
function laidHeight(el: Element): number {
  if (el instanceof HTMLElement && getComputedStyle(el).transform !== "none") return el.offsetHeight;
  return el.getBoundingClientRect().height;
}

/** The gap a column puts before or after what is in it — given back as that
 *  folds away, taken as it comes. */
function gapOf(wrap: HTMLElement | null): number {
  let p = wrap?.parentElement ?? null;
  while (p && getComputedStyle(p).display === "contents") p = p.parentElement;
  if (!p) return 0;
  const cs = getComputedStyle(p);
  const column = cs.display.includes("grid") || (cs.display.includes("flex") && cs.flexDirection.startsWith("column"));
  return column ? parseFloat(cs.rowGap) || 0 : 0;
}

export function Exit({
  children,
  ms = 140,
  collapse = false,
  away,
}: {
  children: ReactNode;
  ms?: number;
  collapse?: boolean;
  away?: boolean;
}) {
  const present = away === undefined ? drawn(children) : !away;
  const last = useRef<ReactNode>(children);
  if (present || away !== undefined) last.current = children;
  const [gone, setGone] = useState(!present);
  const wrap = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (present) {
      setGone(false);
      return;
    }
    const timer = setTimeout(() => setGone(true), ms);
    return () => clearTimeout(timer);
  }, [present, ms]);
  // Coming (not on the first render — what is there as the page is drawn is
  // just there): from no height to the one it has, read again as it grows.
  // Back before it had gone: from where its folding had got to.
  const was = useRef(present);
  const folding = useRef<Animation | null>(null);
  useLayoutEffect(() => {
    const came = present && !was.current;
    was.current = present;
    if (!came || !collapse) return;
    const child = wrap.current?.firstElementChild as HTMLElement | null;
    const back = folding.current;
    const now = back && child ? getComputedStyle(child) : null;
    const from = back && child ? laidHeight(child) : 0;
    const op = now ? parseFloat(now.opacity) || 0 : 0;
    const gap = gapOf(wrap.current);
    const top0 = now ? now.marginTop : `${-gap}px`;
    const bottom0 = now ? now.marginBottom : "0px";
    back?.cancel();
    folding.current = null;
    if (back && child) child.style.overflow = "";
    if (!child || prefersStill()) return;
    const cs = getComputedStyle(child);
    animateHeight(child, from, ms);
    child.animate?.(
      [
        { opacity: op, marginTop: top0, marginBottom: bottom0 },
        { opacity: 1, marginTop: cs.marginTop, marginBottom: cs.marginBottom },
      ],
      { duration: ms, easing: "cubic-bezier(0.22, 0.8, 0.24, 1)" },
    );
  }, [present, collapse, ms]);
  // Folding away: from the height it has to none, before the frame it starts
  // in is drawn.
  useLayoutEffect(() => {
    if (present || !collapse || prefersStill()) return;
    const child = wrap.current?.firstElementChild as HTMLElement | null;
    if (!child?.animate) return;
    const h = laidHeight(child);
    if (h < 2) return;
    noteShrink(ms + 80);
    const gap = gapOf(wrap.current);
    child.style.overflow = "hidden";
    folding.current = child.animate(
      [
        { height: `${h}px`, opacity: 1 },
        {
          height: "0px",
          opacity: 0,
          marginTop: `${-gap}px`,
          marginBottom: "0px",
          paddingTop: "0px",
          paddingBottom: "0px",
          // A box's borders are in its height too: left, a ringed spinner
          // stopped at 3px and went in a frame.
          borderTopWidth: "0px",
          borderBottomWidth: "0px",
        },
      ],
      { duration: ms, easing: "cubic-bezier(0.22, 0.8, 0.24, 1)", fill: "forwards" },
    );
  }, [present, collapse, ms]);
  if (!present && gone) return null;
  return (
    <div ref={wrap} className={`exit-wrap${present ? "" : " leaving"}${collapse ? " collapse" : ""}`}>
      {present || away !== undefined ? children : last.current}
    </div>
  );
}
