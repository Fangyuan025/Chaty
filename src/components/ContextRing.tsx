// How much of the context window the last reply used: a ring and its share,
// the exact counts on hover. The same in chat and Code mode.
import { useI18n } from "../lib/i18n";

export function ContextRing({ used, total }: { used: number; total: number }) {
  const { t } = useI18n();
  const pct = Math.min(100, Math.round((used / total) * 100));
  const r = 7;
  const circ = 2 * Math.PI * r;
  const tone = pct >= 90 ? "hot" : pct >= 70 ? "warn" : "";
  return (
    <span className={`cm-ctx ${tone}`} title={`${used.toLocaleString()} / ${total.toLocaleString()} tokens · ${t("cmContext")}`}>
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
        <circle cx="9" cy="9" r={r} fill="none" stroke="var(--border-strong)" strokeWidth="2.2" />
        <circle
          cx="9" cy="9" r={r} fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={circ * (1 - pct / 100)}
          transform="rotate(-90 9 9)"
        />
      </svg>
      <span className="cm-ctx-pct">{pct}%</span>
    </span>
  );
}
