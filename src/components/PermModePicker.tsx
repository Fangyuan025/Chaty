import { useEffect, useRef, useState } from "react";
import { useI18n, type TKey } from "../lib/i18n";
import { PERM_MODES, type PermMode } from "../lib/permMode";
import { Icon } from "./Icon";

const ICON: Record<PermMode, string> = {
  auto: "shield",
  manual: "lock",
  acceptEdits: "pencil",
  plan: "map",
  bypass: "bolt",
};

export const PERM_LABEL: Record<PermMode, TKey> = {
  auto: "permAuto",
  manual: "permManual",
  acceptEdits: "permAcceptEdits",
  plan: "permPlan",
  bypass: "permBypass",
};

const PERM_DESC: Record<PermMode, TKey> = {
  auto: "permAutoDesc",
  manual: "permManualDesc",
  acceptEdits: "permAcceptEditsDesc",
  plan: "permPlanDesc",
  bypass: "permBypassDesc",
};

/** The permission mode under Code's input: the mode in force, and a menu of
 *  all five — named, described, numbered as their keys pick them. */
export function PermModePicker({ mode, onPick }: { mode: PermMode; onPick: (m: PermMode) => void }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setOpen(false);
        return;
      }
      const n = Number(e.key);
      if (n >= 1 && n <= PERM_MODES.length && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        onPick(PERM_MODES[n - 1]);
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    // Capture: the number keys belong to the menu while it is open, not to
    // the input under it.
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [open, onPick]);

  return (
    <div className="cm-perm" ref={root}>
      <button
        type="button"
        className={`cm-bypass cm-perm-trigger${mode === "bypass" ? " on" : ""}${open ? " open" : ""}`}
        title={t("permHint")}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon name={ICON[mode]} size={12} strokeWidth={1.9} />
        <span className="cm-head-label">{t(PERM_LABEL[mode])}</span>
      </button>
      {open && (
        <div className="cm-perm-menu" role="menu">
          <div className="cm-perm-title">{t("permTitle")}</div>
          {PERM_MODES.map((m, i) => (
            <button
              type="button"
              key={m}
              role="menuitemradio"
              aria-checked={m === mode}
              className={`cm-perm-opt${m === mode ? " active" : ""}`}
              onClick={() => {
                onPick(m);
                setOpen(false);
              }}
            >
              <span className="cm-perm-text">
                <span className="cm-perm-name">
                  {t(PERM_LABEL[m])}
                  {m === "auto" && <span className="cm-perm-badge">{t("permRecommended")}</span>}
                </span>
                <span className="cm-perm-desc">{t(PERM_DESC[m])}</span>
              </span>
              <span className="cm-perm-side">
                {m === mode && <Icon name="check" size={13} strokeWidth={2.2} />}
                <span className="cm-perm-num">{i + 1}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
