import { useMemo, useState } from "react";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { diffLines } from "../lib/diff";
import type { TurnChange } from "../lib/turnChanges";

/** Every file a turn changed, under its answer: the net +/− of each, its diff
 *  one click away, and an undo per file (and for all of them). What the turn
 *  did to the workspace was otherwise spread across its steps — one card per
 *  write, each with that write's diff, none with where the file ended up. */
export function TurnChangesCard({
  changes,
  disabled,
  onUndo,
}: {
  changes: TurnChange[];
  /** A turn is running: undo waits until it is done. */
  disabled: boolean;
  onUndo: (paths: string[]) => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState<string | null>(null);
  const pending = changes.filter((c) => !c.undone);
  return (
    <div className="cm-changes">
      <div className="cm-changes-head">
        <span>{t("cmChangesTitle", { n: String(changes.length) })}</span>
        {pending.length > 1 && (
          <button
            className="cm-changes-undo"
            disabled={disabled}
            onClick={() => onUndo(pending.map((c) => c.path))}
          >
            {t("cmChangesUndoAll")}
          </button>
        )}
      </div>
      {changes.map((c) => (
        <div key={c.path} className={`cm-change ${c.undone ? "undone" : ""}`}>
          <div className="cm-change-row">
            <button
              className="cm-change-file"
              title={c.path}
              aria-expanded={open === c.path}
              onClick={() => setOpen((o) => (o === c.path ? null : c.path))}
            >
              <Icon name={open === c.path ? "chevron-down" : "chevron-right"} size={12} />
              <span className="cm-change-name">{c.rel}</span>
              {c.created && <span className="cm-change-tag">{t("cmChangeNew")}</span>}
              {c.deleted && <span className="cm-change-tag">{t("cmChangeDeleted")}</span>}
              {c.binary ? (
                <span className="cm-change-tag">{t("cmChangeBinary")}</span>
              ) : (
                <span className="cm-step-diffstat">
                  <em className="plus">+{c.added}</em>
                  <em className="minus">-{c.removed}</em>
                </span>
              )}
            </button>
            {c.undone ? (
              <span className="cm-change-undone">{t("cmChangeUndone")}</span>
            ) : (
              <button className="cm-change-undo" disabled={disabled} onClick={() => onUndo([c.path])}>
                {t("cmChangeUndo")}
              </button>
            )}
          </div>
          {open === c.path && <ChangeDiff change={c} />}
        </div>
      ))}
    </div>
  );
}

function ChangeDiff({ change }: { change: TurnChange }) {
  const { t } = useI18n();
  const d = useMemo(
    () =>
      change.before !== undefined || change.after !== undefined
        ? diffLines(change.before ?? "", change.after ?? "")
        : null,
    [change],
  );
  if (change.binary) return <div className="cm-change-note">{t("cmChangeBinaryNote")}</div>;
  if (!d) return <div className="cm-change-note">{t("cmChangeTooBig")}</div>;
  return (
    <pre className="cm-diff cm-change-diff">
      {d.rows.map((l, i) => (
        <div key={i} className={`cm-dl ${l.kind}`}>
          <span className="cm-dl-mark">{l.kind === "add" ? "+" : l.kind === "del" ? "-" : " "}</span>
          {l.text}
        </div>
      ))}
      {d.truncated && (
        <div className="cm-dl ctx cm-dl-more">
          {t("cmDiffMore").replace("{n}", String(change.added + change.removed))}
        </div>
      )}
    </pre>
  );
}
