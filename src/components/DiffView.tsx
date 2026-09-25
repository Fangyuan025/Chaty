import { useMemo, useState } from "react";
import { diffLines } from "../lib/diff";
import { useI18n } from "../lib/i18n";

type Contents = { before: string; after: string };

/**
 * A file's change as diff rows — a step card's, or a turn's changed file.
 *
 * Long diffs render their first few hundred rows and say how much is left;
 * that line opens the rest. When the contents on hand were themselves cut
 * short (a step card keeps a capped copy of a big file), `loadFull` fetches
 * the whole of both sides first.
 */
export function DiffView({
  before,
  after,
  total,
  className,
  loadFull,
}: {
  before: string;
  after: string;
  /** Changed lines in all, for the line that offers the rest. */
  total: number;
  className?: string;
  /** The uncut contents, when the ones given were capped. */
  loadFull?: () => Promise<Contents | null>;
}) {
  const { t } = useI18n();
  const [all, setAll] = useState(false);
  const [full, setFull] = useState<Contents | null>(null);
  const [loading, setLoading] = useState(false);
  const src = full ?? { before, after };
  const d = useMemo(
    () => diffLines(src.before, src.after, all ? Infinity : undefined),
    [src.before, src.after, all],
  );
  const more = !all && (d.truncated || (!!loadFull && !full));
  const expand = async () => {
    if (loading) return;
    if (loadFull && !full) {
      setLoading(true);
      const got = await loadFull().catch(() => null);
      setLoading(false);
      if (got) setFull(got);
    }
    setAll(true);
  };
  return (
    <pre className={`cm-diff${className ? ` ${className}` : ""}`}>
      {d.rows.map((l, i) => (
        <div key={i} className={`cm-dl ${l.kind}`}>
          <span className="cm-dl-mark">{l.kind === "add" ? "+" : l.kind === "del" ? "-" : " "}</span>
          {l.text}
        </div>
      ))}
      {more && (
        <button type="button" className="cm-dl ctx cm-dl-more" onClick={() => void expand()} disabled={loading}>
          {loading ? "…" : t("cmDiffMore").replace("{n}", String(total))}
        </button>
      )}
    </pre>
  );
}
