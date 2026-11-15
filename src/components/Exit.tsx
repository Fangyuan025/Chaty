// What appears over the page — a menu, a sheet, a preview — goes the way it
// came, fading out over a moment, where it used to vanish in a frame. Wrap
// the conditional: <Exit>{open && <Menu … />}</Exit>. Taken away, the last
// of it stays a moment, not to be clicked, while it fades.
import { type ReactNode, useEffect, useRef, useState } from "react";

export function Exit({ children, ms = 140 }: { children: ReactNode; ms?: number }) {
  const present = children != null && children !== false;
  const last = useRef<ReactNode>(children);
  if (present) last.current = children;
  const [gone, setGone] = useState(!present);
  useEffect(() => {
    if (present) {
      setGone(false);
      return;
    }
    const timer = setTimeout(() => setGone(true), ms);
    return () => clearTimeout(timer);
  }, [present, ms]);
  if (!present && gone) return null;
  return <div className={present ? "exit-wrap" : "exit-wrap leaving"}>{present ? children : last.current}</div>;
}
