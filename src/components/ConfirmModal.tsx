import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";

export interface ConfirmOptions {
  message: string;
  title?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Style the confirm button as destructive (red). */
  danger?: boolean;
}

type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>;

const ConfirmContext = createContext<ConfirmFn | null>(null);

/** Imperative `confirm(opts) => Promise<boolean>` backed by an in-app modal. */
export function useConfirm(): ConfirmFn {
  const fn = useContext(ConfirmContext);
  if (!fn) throw new Error("useConfirm must be used within <ConfirmProvider>");
  return fn;
}

/**
 * Provides an in-app (Chaty-styled) confirmation dialog instead of the native
 * OS dialog — `window.confirm` doesn't even render inside WKWebView, and the
 * native dialog looks out of place.
 */
export function ConfirmProvider({ children }: { children: React.ReactNode }) {
  const { t } = useI18n();
  const [opts, setOpts] = useState<ConfirmOptions | null>(null);
  const resolver = useRef<((v: boolean) => void) | null>(null);

  const confirm = useCallback<ConfirmFn>((next) => {
    return new Promise<boolean>((resolve) => {
      resolver.current = resolve;
      setOpts(next);
    });
  }, []);

  const close = useCallback((value: boolean) => {
    resolver.current?.(value);
    resolver.current = null;
    setOpts(null);
  }, []);

  useEffect(() => {
    if (!opts) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") close(false);
      else if (e.key === "Enter") close(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [opts, close]);

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      {opts &&
        createPortal(
          <div className="confirm-overlay" onMouseDown={() => close(false)}>
            <div className="confirm-modal" onMouseDown={(e) => e.stopPropagation()}>
              {opts.title && <div className="confirm-title">{opts.title}</div>}
              <div className="confirm-msg">{opts.message}</div>
              <div className="confirm-actions">
                <button className="confirm-cancel" onClick={() => close(false)}>
                  {opts.cancelLabel ?? t("cancel")}
                </button>
                <button
                  className={`confirm-ok ${opts.danger ? "danger" : ""}`}
                  onClick={() => close(true)}
                  autoFocus
                >
                  {opts.confirmLabel ?? t("confirm")}
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </ConfirmContext.Provider>
  );
}
