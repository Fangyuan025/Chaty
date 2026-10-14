// The sources under an answer — web pages and knowledge-base passages — as
// one line ("wikipedia.org and 11 more sources") that opens into the list:
// each with its number (the 【n】 in the answer), title, site and passage. A
// web source opens its page; a knowledge-base passage, which has no page,
// opens to its whole retrieved text.
import { useState } from "react";
import { useI18n } from "../lib/i18n";
import { openExternal, type SearchResult } from "../lib/ipc";
import { Icon } from "./Icon";

/** A web source's site ("en.wikipedia.org"), or a document's name. */
function sourceName(s: SearchResult): string {
  if (!s.url) return s.title;
  try {
    return new URL(s.url).hostname.replace(/^www\./, "");
  } catch {
    return s.title;
  }
}

function clip(s: string, n: number): string {
  return [...s].length > n ? [...s].slice(0, n - 1).join("") + "…" : s;
}

export function Sources({ sources }: { sources: SearchResult[] }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [full, setFull] = useState<Set<number>>(new Set());
  const n = sources.length;
  const first = clip(sourceName(sources[0]), 32);
  const label =
    n === 1
      ? t("sourcesOne", { first })
      : n === 2
      ? t("sourcesTwo", { first })
      : t("sourcesMany", { first, n, more: n - 1 });
  return (
    <div className={`sources${open ? " open" : ""}`}>
      <button className="sources-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="sources-sum">{label}</span>
        <Icon name="chevron-down" size={13} strokeWidth={2} />
      </button>
      {open && (
        <ol className="sources-detail">
          {sources.map((s, k) => {
            const web = !!s.url;
            const whole = full.has(k);
            return (
              <li key={k}>
                <button
                  className={`source-row${web ? " web" : ""}`}
                  title={web ? s.url : undefined}
                  onClick={() => {
                    if (web) {
                      void openExternal(s.url).catch(() => {});
                    } else {
                      setFull((cur) => {
                        const next = new Set(cur);
                        if (next.has(k)) next.delete(k);
                        else next.add(k);
                        return next;
                      });
                    }
                  }}
                >
                  <span className="source-idx">{k + 1}</span>
                  <span className="source-main">
                    <span className="source-head">
                      <span className="source-title">{s.title}</span>
                      <span className="source-site">{web ? sourceName(s) : t("sourceKb")}</span>
                    </span>
                    {s.snippet && <span className={`source-snippet${whole ? " whole" : ""}`}>{s.snippet}</span>}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
