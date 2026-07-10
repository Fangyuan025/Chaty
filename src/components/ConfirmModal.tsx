import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { useExitTransition } from "../lib/useExit";

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
    // Resolve immediately — only the unmount is delayed for the exit animation.
    resolver.current?.(value);
    resolver.current = null;
    setOpts(null);
  }, []);

  const { mounted, closing } = useExitTransition(opts != null);
  const lastOpts = useRef<ConfirmOptions | null>(null);
  if (opts) lastOpts.current = opts;
  const shown = opts ?? lastOpts.current;

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
      {mounted &&
        shown &&
        createPortal(
          <div
            className={`confirm-overlay ${closing ? "closing" : ""}`}
            onMouseDown={() => close(false)}
          >
            <div className="confirm-modal" onMouseDown={(e) => e.stopPropagation()}>
              {shown.title && <div className="confirm-title">{shown.title}</div>}
              <div className="confirm-msg">{shown.message}</div>
              <div className="confirm-actions">
                <button className="confirm-cancel" onClick={() => close(false)}>
                  {shown.cancelLabel ?? t("cancel")}
                </button>
                <button
                  className={`confirm-ok ${shown.danger ? "danger" : ""}`}
                  onClick={() => close(true)}
                  autoFocus
                >
                  {shown.confirmLabel ?? t("confirm")}
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </ConfirmContext.Provider>
  );
}
