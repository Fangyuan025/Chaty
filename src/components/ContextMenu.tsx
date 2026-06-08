import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useI18n } from "../lib/i18n";
import { copyToClipboard, readFromClipboard } from "../lib/clipboard";

interface MenuItem {
  label: string;
  action: () => void;
  disabled?: boolean;
}

interface MenuState {
  x: number;
  y: number;
  items: MenuItem[];
}

function asField(el: HTMLElement | null): HTMLInputElement | HTMLTextAreaElement | null {
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return el;
  return null;
}

/** Update a (possibly React-controlled) field's value so onChange fires. */
function setNativeValue(el: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const proto =
    el instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
  setter?.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

function replaceSelection(field: HTMLInputElement | HTMLTextAreaElement, text: string) {
  const start = field.selectionStart ?? field.value.length;
  const end = field.selectionEnd ?? field.value.length;
  setNativeValue(field, field.value.slice(0, start) + text + field.value.slice(end));
  const pos = start + text.length;
  field.focus();
  field.setSelectionRange(pos, pos);
}

/** A custom right-click menu that replaces the WebView's native context menu. */
export function ContextMenu() {
  const { t } = useI18n();
  const [menu, setMenu] = useState<MenuState | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number }>({ left: 0, top: 0 });

  useEffect(() => {
    const onCtx = (e: MouseEvent) => {
      e.preventDefault();
      const target = e.target as HTMLElement;
      const field = asField(target.closest("input, textarea") as HTMLElement | null);
      const selection = window.getSelection()?.toString() ?? "";
      const items: MenuItem[] = [];

      if (field) {
        const hasSel = field.selectionStart !== field.selectionEnd;
        items.push({
          label: t("ctxCut"),
          disabled: !hasSel,
          action: () => {
            const s = field.value.slice(field.selectionStart ?? 0, field.selectionEnd ?? 0);
            void copyToClipboard(s);
            replaceSelection(field, "");
          },
        });
        items.push({
          label: t("ctxCopy"),
          disabled: !hasSel,
          action: () => {
            const s = field.value.slice(field.selectionStart ?? 0, field.selectionEnd ?? 0);
            void copyToClipboard(s);
          },
        });
        items.push({
          label: t("ctxPaste"),
          action: async () => {
            const text = await readFromClipboard();
            if (text) replaceSelection(field, text);
          },
        });
        items.push({
          label: t("ctxSelectAll"),
          disabled: field.value.length === 0,
          action: () => {
            field.focus();
            field.select();
          },
        });
      } else {
        items.push({
          label: t("ctxCopy"),
          disabled: !selection,
          action: () => void copyToClipboard(selection),
        });
        items.push({
          label: t("ctxSelectAll"),
          action: () => {
            const sel = window.getSelection();
            const block = target.closest(".answer, .bubble, .think-body, main") ?? document.body;
            const range = document.createRange();
            range.selectNodeContents(block);
            sel?.removeAllRanges();
            sel?.addRange(range);
          },
        });
      }

      setMenu({ x: e.clientX, y: e.clientY, items });
    };

    const dismiss = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(null);
    };
    window.addEventListener("contextmenu", onCtx);
    window.addEventListener("mousedown", dismiss);
    window.addEventListener("blur", dismiss);
    window.addEventListener("resize", dismiss);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("contextmenu", onCtx);
      window.removeEventListener("mousedown", dismiss);
      window.removeEventListener("blur", dismiss);
      window.removeEventListener("resize", dismiss);
      window.removeEventListener("keydown", onKey);
    };
  }, [t]);

  // Keep the menu fully on-screen.
  useLayoutEffect(() => {
    if (!menu) return;
    const el = ref.current;
    const w = el?.offsetWidth ?? 180;
    const h = el?.offsetHeight ?? 10;
    const left = Math.min(menu.x, window.innerWidth - w - 8);
    const top = Math.min(menu.y, window.innerHeight - h - 8);
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [menu]);

  if (!menu) return null;

  return (
    <div
      className="ctx-menu"
      ref={ref}
      style={{ left: pos.left, top: pos.top }}
      onMouseDown={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {menu.items.map((it, i) => (
        <button
          key={i}
          className="ctx-item"
          disabled={it.disabled}
          onClick={() => {
            it.action();
            setMenu(null);
          }}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}
