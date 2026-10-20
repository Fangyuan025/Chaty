import { useEffect, useRef, useState } from "react";

/** Circular prompt-processing progress: a small ring that fills 0→100% while a
 *  long prompt is prefilling (the silent gap before tokens stream), plus the
 *  live percentage. Replaces the generic spinner whenever progress is known. */
export function PrefillRing({ frac, size = 16 }: { frac: number; size?: number }) {
  const r = (size - 3) / 2; // stroke 2.5 + hairline padding
  const c = 2 * Math.PI * r;
  // Progress events arrive per decode batch / per media segment — seconds
  // apart on big prompts, which read as jumps. Smooth the DISPLAYED value:
  // chase a new target quickly, and between events creep gently ahead (capped
  // a few percent past the last report) so the ring visibly keeps moving
  // through long silent stretches like image encoding.
  const [shown, setShown] = useState(frac);
  const targetRef = useRef(frac);
  targetRef.current = frac;
  useEffect(() => {
    const timer = setInterval(() => {
      setShown((s) => {
        const t = targetRef.current;
        if (t < s - 0.08) return t; // new turn restarted — snap back
        if (t > s) return Math.min(t, s + Math.max((t - s) * 0.3, 0.005)); // chase
        return Math.min(s + 0.0015, t + 0.05, 0.99); // creep between events
      });
    }, 80);
    return () => clearInterval(timer);
  }, []);
  const pct = Math.round(Math.min(1, Math.max(0, shown)) * 100);
  return (
    <span className="cm-prefill" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border-strong)" strokeWidth="2.5" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct / 100)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className="cm-prefill-arc"
        />
      </svg>
      <span className="cm-prefill-pct">{pct}%</span>
    </span>
  );
}
