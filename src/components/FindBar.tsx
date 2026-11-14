import { useEffect, useRef } from "react";
import { useI18n } from "../lib/i18n";
import { imeEnter } from "../lib/ime";
import { Icon } from "./Icon";

/**
 * Find in the open chat (⌘/Ctrl F): the matches marked, the current one
 * scrolled to and marked more strongly, Enter / ⇧Enter (or the arrows) to
 * move between them, Esc to close. The app's web view has no find of its own.
 */
export function FindBar({
  query,
  onQuery,
  index,
  count,
  onStep,
  onClose,
}: {
  query: string;
  onQuery: (q: string) => void;
  /** The current match, 0-based; -1 for none. */
  index: number;
  count: number;
  onStep: (dir: 1 | -1) => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    input.current?.focus();
    input.current?.select();
  }, []);
  return (
    <div className="find-bar" role="search">
      <Icon name="search" size={13} strokeWidth={1.9} />
      <input
        ref={input}
        value={query}
        placeholder={t("findInChat")}
        aria-label={t("findInChat")}
        onChange={(e) => onQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !imeEnter(e)) {
            e.preventDefault();
            onStep(e.shiftKey ? -1 : 1);
          } else if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            onClose();
          } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            onStep(e.key === "ArrowDown" ? 1 : -1);
          }
        }}
      />
      <span className={`find-count ${query && !count ? "none" : ""}`}>
        {query ? (count ? `${index + 1}/${count}` : t("findNone")) : ""}
      </span>
      <button className="find-btn" type="button" disabled={!count} title={t("findPrev")} aria-label={t("findPrev")} onClick={() => onStep(-1)}>
        <Icon name="chevron-up" size={13} strokeWidth={2} />
      </button>
      <button className="find-btn" type="button" disabled={!count} title={t("findNext")} aria-label={t("findNext")} onClick={() => onStep(1)}>
        <Icon name="chevron-down" size={13} strokeWidth={2} />
      </button>
      <button className="find-btn" type="button" title={t("close")} aria-label={t("close")} onClick={onClose}>
        <Icon name="x" size={12} strokeWidth={2.2} />
      </button>
    </div>
  );
}
