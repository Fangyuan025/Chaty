// How much of the context window the last reply used: a ring and its share,
// the exact counts on hover. The same in chat and Code mode.
import { useEffect, useRef, useState } from "react";
import { useI18n } from "../lib/i18n";

/** How long the ring takes to reach a new share. */
const SWEEP_MS = 520;

export function ContextRing({ used, total }: { used: number; total: number }) {
  const { t } = useI18n();
  const frac = Math.min(1, Math.max(0, used / total));
  const pct = Math.round(frac * 100);
  // A new count sweeps the ring there, the number with it, instead of the
  // ring jumping: what the conversation just took is seen being taken. The
  // ring as it first appears is drawn where it is.
  const [shown, setShown] = useState(frac);
  const at = useRef(frac);
  useEffect(() => {
    const from = at.current;
    if (Math.abs(frac - from) < 0.0005) return;
    const still = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (still) {
      at.current = frac;
      setShown(frac);
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / SWEEP_MS);
      const v = from + (frac - from) * (1 - Math.pow(1 - k, 3));
      at.current = v;
      setShown(v);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [frac]);
  const r = 7;
  const circ = 2 * Math.PI * r;
  const tone = pct >= 90 ? "hot" : pct >= 70 ? "warn" : "";
  return (
    <span className={`cm-ctx ${tone}`} title={`${used.toLocaleString()} / ${total.toLocaleString()} tokens · ${t("cmContext")}`}>
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
        <circle cx="9" cy="9" r={r} fill="none" stroke="var(--border-strong)" strokeWidth="2.2" />
        <circle
          cx="9" cy="9" r={r} fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={circ * (1 - shown)}
          transform="rotate(-90 9 9)"
        />
      </svg>
      <span className="cm-ctx-pct">{Math.round(shown * 100)}%</span>
    </span>
  );
}
