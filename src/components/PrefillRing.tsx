import { prefillPct } from "../lib/prefill";

/** Circular prompt-processing progress: a small ring that fills 0→100% while a
 *  long prompt is prefilling (the silent gap before tokens stream), plus the
 *  percentage. Replaces the generic spinner whenever progress is known.
 *
 *  The number is the engine's report as `prefillPct` gives it — the same one
 *  the folded run's line shows. Reports come per decode batch, seconds apart
 *  on a big prompt; the arc eases between them (`.cm-prefill-arc`), the
 *  number never runs ahead of what was read. */
export function PrefillRing({ frac, size = 16 }: { frac: number; size?: number }) {
  const r = (size - 3) / 2; // stroke 2.5 + hairline padding
  const c = 2 * Math.PI * r;
  const pct = prefillPct(frac);
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
