import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";

/**
 * Selecting several rows of a sidebar list to delete them together — the
 * chat's conversations, Code's sessions, the image studio's sessions all work
 * the same way (issue #20 asked for it in chat; one hook keeps the three
 * from drifting). A plain click toggles a row, Shift-click extends from the
 * last one ticked across the list as it is shown, Esc leaves.
 */
export function useMultiSelect() {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const last = useRef<string | null>(null);
  /** The list's ids in the order shown, handed over each render (`order`). */
  const shown = useRef<string[]>([]);

  const exit = useCallback(() => {
    setSelecting(false);
    setSelected(new Set());
    last.current = null;
  }, []);

  const pick = useCallback((id: string, range: boolean) => {
    // Read before the update runs: by then the ref names this click.
    const from = last.current;
    setSelecting(true);
    setSelected((cur) => {
      const next = new Set(cur);
      if (range && from && from !== id) {
        const order = shown.current;
        const a = order.indexOf(from);
        const b = order.indexOf(id);
        if (a >= 0 && b >= 0) {
          for (const x of order.slice(Math.min(a, b), Math.max(a, b) + 1)) next.add(x);
          return next;
        }
      }
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    last.current = id;
  }, []);

  const toggleAll = useCallback(() => {
    setSelected((cur) => {
      const all = shown.current;
      return all.length > 0 && all.every((id) => cur.has(id)) ? new Set() : new Set(all);
    });
  }, []);

  useEffect(() => {
    if (!selecting) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") exit();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selecting, exit]);

  return {
    selecting,
    selected,
    start: () => setSelecting(true),
    exit,
    pick,
    toggleAll,
    /** Tell it the order the list is shown in (search / grouping). */
    order: (ids: string[]) => {
      shown.current = ids;
    },
    allPicked: () => shown.current.length > 0 && shown.current.every((id) => selected.has(id)),
    /** Ids still in the list that are ticked, in the list's order. */
    picked: () => shown.current.filter((id) => selected.has(id)),
  };
}

export type MultiSelect = ReturnType<typeof useMultiSelect>;

/** Row click in a selectable list: toggles in selection mode, and a Ctrl /
 *  ⌘-click starts one, like a file list. Returns whether it handled the click. */
export function selectClick(ms: MultiSelect, id: string, e: { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean }): boolean {
  if (!ms.selecting && !e.metaKey && !e.ctrlKey) return false;
  ms.pick(id, e.shiftKey);
  return true;
}

/** The button that turns selection on and off: a ticked box. */
export function SelectToggle({ ms, className }: { ms: MultiSelect; className?: string }) {
  const { t } = useI18n();
  return (
    <button
      className={`select-toggle ${ms.selecting ? "on" : ""} ${className ?? ""}`}
      title={ms.selecting ? t("cancel") : t("selectConvs")}
      aria-pressed={ms.selecting}
      onClick={() => (ms.selecting ? ms.exit() : ms.start())}
    >
      {/* A ticked box: selection, not "confirm" as a bare tick read. */}
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <rect x="3.5" y="3.5" width="17" height="17" rx="4" />
        <path d="M8 12.2l2.7 2.7L16 9.6" />
      </svg>
    </button>
  );
}

/** A row's tick box while selecting. */
export function SelectCheck({ on }: { on: boolean }) {
  return (
    <span className={`select-check ${on ? "on" : ""}`} aria-hidden="true">
      {on && <Icon name="check" size={10} strokeWidth={2.6} />}
    </span>
  );
}

/** The foot of the list while selecting: count, all/none, delete, cancel. */
export function SelectBar({ ms, onDelete, busy }: { ms: MultiSelect; onDelete: () => void; busy?: boolean }) {
  const { t } = useI18n();
  if (!ms.selecting) return null;
  return (
    <div className="select-bar">
      <span className="sb-count">{t("selectedN", { n: ms.selected.size })}</span>
      <button className="sb-btn" onClick={ms.toggleAll}>
        {ms.allPicked() ? t("selectNone") : t("selectAll")}
      </button>
      <button className="sb-btn danger" disabled={ms.selected.size === 0 || busy} onClick={onDelete}>
        {t("confirmDelete")}
      </button>
      <button className="sb-btn" onClick={ms.exit}>
        {t("cancel")}
      </button>
    </div>
  );
}
