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
  /** A notice rather than a question: one button, nothing to decline. */
  hideCancel?: boolean;
  /** A second way to go ahead, beside the confirm button (see useChoice). */
  altLabel?: string;
}

type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>;
/** Which button: confirm, the other one, or neither (cancelled). */
export type Choice = "ok" | "alt" | null;
type ChoiceFn = (opts: ConfirmOptions & { altLabel: string }) => Promise<Choice>;

const ConfirmContext = createContext<ChoiceFn | null>(null);

/** Imperative `confirm(opts) => Promise<boolean>` backed by an in-app modal. */
export function useConfirm(): ConfirmFn {
  const fn = useContext(ConfirmContext);
  if (!fn) throw new Error("useConfirm must be used within <ConfirmProvider>");
  return useCallback((opts: ConfirmOptions) => fn(opts as ConfirmOptions & { altLabel: string }).then((v) => v === "ok"), [fn]);
}

/** The same dialog with two ways forward — `altLabel` beside the confirm
 *  button — and Escape or the backdrop for neither. */
export function useChoice(): ChoiceFn {
  const fn = useContext(ConfirmContext);
  if (!fn) throw new Error("useChoice must be used within <ConfirmProvider>");
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
  const resolver = useRef<((v: Choice) => void) | null>(null);

  const choose = useCallback<ChoiceFn>((next) => {
    return new Promise<Choice>((resolve) => {
      resolver.current = resolve;
      setOpts(next);
    });
  }, []);

  const close = useCallback((value: Choice) => {
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
      if (e.key === "Escape") close(null);
      else if (e.key === "Enter") close("ok");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [opts, close]);

  return (
    <ConfirmContext.Provider value={choose}>
      {children}
      {mounted &&
        shown &&
        createPortal(
          <div
            className={`confirm-overlay ${closing ? "closing" : ""}`}
            onMouseDown={() => close(null)}
          >
            <div className="confirm-modal" onMouseDown={(e) => e.stopPropagation()}>
              {shown.title && <div className="confirm-title">{shown.title}</div>}
              <div className="confirm-msg">{shown.message}</div>
              <div className="confirm-actions">
                {!shown.hideCancel && (
                  <button className="confirm-cancel" onClick={() => close(null)}>
                    {shown.cancelLabel ?? t("cancel")}
                  </button>
                )}
                {shown.altLabel && (
                  <button className="confirm-cancel confirm-alt" onClick={() => close("alt")}>
                    {shown.altLabel}
                  </button>
                )}
                <button
                  className={`confirm-ok ${shown.danger ? "danger" : ""}`}
                  onClick={() => close("ok")}
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
