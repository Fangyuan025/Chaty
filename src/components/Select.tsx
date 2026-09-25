import { useEffect, useRef, useState } from "react";

export type SelectOption<T extends string | number> = { value: T; label: string };

/** Chaty-native dropdown — replaces the OS-native <select> so menus match the
 *  app's look (rounded panel, accent check, light/dark themed) on every platform.
 *  Closes on outside-click, Escape, or selection. */
export function Select<T extends string | number>({
  value,
  options,
  onChange,
  disabled,
  className,
  ariaLabel,
}: {
  value: T;
  options: SelectOption<T>[];
  onChange: (v: T) => void;
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  // Opens upward when the window has no room below — a menu in a popover
  // over the composer would otherwise run off the bottom of the window.
  const [up, setUp] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const current = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className={`csel${open ? " open" : ""}${up ? " up" : ""}${className ? " " + className : ""}`} ref={rootRef}>
      <button
        type="button"
        className="csel-trigger"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        onClick={() => {
          if (disabled) return;
          if (!open && rootRef.current) {
            // The menu is at most 260px tall (.csel-menu).
            const r = rootRef.current.getBoundingClientRect();
            const below = window.innerHeight - r.bottom;
            setUp(below < 272 && r.top > below);
          }
          setOpen((o) => !o);
        }}
      >
        <span className="csel-value">{current?.label ?? ""}</span>
        <svg className="csel-chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
          <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="csel-menu" role="listbox">
          {options.map((o) => (
            <button
              type="button"
              key={String(o.value)}
              role="option"
              aria-selected={o.value === value}
              className={`csel-opt${o.value === value ? " active" : ""}`}
              onClick={() => {
                onChange(o.value);
                setOpen(false);
              }}
            >
              <span>{o.label}</span>
              {o.value === value && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" aria-hidden="true">
                  <path d="M5 12l5 5L20 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
