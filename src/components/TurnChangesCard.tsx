import { useState } from "react";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { DiffView } from "./DiffView";
import { changeTotals, startsFolded, type TurnChange } from "../lib/turnChanges";

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
  // Open up to five files; past that, folded to its totals until asked.
  const [expanded, setExpanded] = useState(() => !startsFolded(changes.length));
  const [open, setOpen] = useState<string | null>(null);
  const pending = changes.filter((c) => !c.undone);
  const total = changeTotals(changes);
  return (
    <div className={`cm-changes ${expanded ? "open" : ""}`}>
      <div className="cm-changes-head">
        <button
          className="cm-changes-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
        >
          <span className="cm-changes-chev">
            <Icon name="chevron-right" size={13} />
          </span>
          <span>{t("cmChangesTitle", { n: String(changes.length) })}</span>
          <span className="cm-step-diffstat">
            <em className="plus">+{total.added}</em>
            <em className="minus">-{total.removed}</em>
          </span>
        </button>
        {expanded && pending.length > 1 && (
          <button
            className="cm-changes-undo"
            disabled={disabled}
            onClick={() => onUndo(pending.map((c) => c.path))}
          >
            {t("cmChangesUndoAll")}
          </button>
        )}
      </div>
      {expanded && changes.map((c) => (
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
  if (change.binary) return <div className="cm-change-note">{t("cmChangeBinaryNote")}</div>;
  if (change.before === undefined && change.after === undefined) {
    return <div className="cm-change-note">{t("cmChangeTooBig")}</div>;
  }
  return (
    <DiffView
      className="cm-change-diff"
      before={change.before ?? ""}
      after={change.after ?? ""}
      total={change.added + change.removed}
    />
  );
}
