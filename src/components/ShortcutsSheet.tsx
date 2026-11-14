import { useEffect } from "react";
import { createPortal } from "react-dom";
import { useI18n, type TKey } from "../lib/i18n";
import { Icon } from "./Icon";

/** Every keyboard shortcut on one sheet (⌘/ or Ctrl+/), the keys as this
 *  computer writes them. */
export function ShortcutsSheet({ mac, onClose }: { mac: boolean; onClose: () => void }) {
  const { t } = useI18n();
  const mod = mac ? "⌘" : "Ctrl";
  const shift = mac ? "⇧" : "Shift";
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);
  const groups: { title: TKey; rows: [string[], TKey][] }[] = [
    {
      title: "skGeneral",
      rows: [
        [[mod, "K"], "skPalette"],
        [[mod, "N"], "skNew"],
        [[mod, ","], "skSettings"],
        [[mod, "1"], "skChat"],
        [[mod, "2"], "skCode"],
        [[mod, "\\"], "skSidebar"],
        [[mod, "/"], "skThis"],
      ],
    },
    {
      title: "skInChat",
      rows: [
        [[mod, "F"], "skFind"],
        [[mod, shift, "F"], "skSearchSide"],
        [["↑"], "skEditLast"],
        [["Esc"], "skStop"],
        [[shift, "Enter"], "skNewline"],
        [[mod, "V"], "skPaste"],
      ],
    },
    {
      title: "skInCode",
      rows: [
        [[shift, "Tab"], "skPermMode"],
        [["↑"], "skRecallTask"],
        [["Esc"], "skStop"],
      ],
    },
    {
      title: "skAnywhere",
      rows: [[["Ctrl", shift, "Space"], "skSummon"]],
    },
  ];
  return createPortal(
    <div className="kb-overlay" onMouseDown={onClose}>
      <div className="kb-sheet" role="dialog" aria-label={t("skTitle")} onMouseDown={(e) => e.stopPropagation()}>
        <div className="kb-head">
          <span>{t("skTitle")}</span>
          <button className="kb-close" type="button" title={t("close")} aria-label={t("close")} onClick={onClose}>
            <Icon name="x" size={13} strokeWidth={2} />
          </button>
        </div>
        <div className="kb-groups">
          {groups.map((g) => (
            <section key={g.title} className="kb-group">
              <h4>{t(g.title)}</h4>
              {g.rows.map(([keys, label]) => (
                <div key={label + keys.join("")} className="kb-row">
                  <span className="kb-label">{t(label)}</span>
                  <span className="kb-keys">
                    {keys.map((k) => (
                      <kbd key={k}>{k}</kbd>
                    ))}
                  </span>
                </div>
              ))}
            </section>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  );
}
