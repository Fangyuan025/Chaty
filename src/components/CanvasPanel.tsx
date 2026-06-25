import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { withStorageShim } from "./Markdown";
import { IconDownload, IconEdit } from "./icons";

export interface CanvasVersion {
  html: string;
  /** Short label for the version list (e.g. "初始" / "修改：…" / "修复：…"). */
  note: string;
}

/** A runtime error reported from inside the sandboxed preview. */
interface CanvasError {
  kind: string;
  message: string;
  detail: string;
}

/**
 * Capture-shim injected into the preview iframe: forwards uncaught errors,
 * resource-load failures and unhandled rejections to the parent (Canvas) via
 * postMessage. Self-dedups so a render loop can't spam the same error.
 */
const ERROR_SHIM = `<script>(function(){
  var seen = {};
  function send(kind, msg, detail){
    var sig = kind + '|' + msg + '|' + (detail||'');
    if (seen[sig]) return; seen[sig] = 1;
    try { parent.postMessage({ __chatyCanvasError: { kind: kind, message: String(msg).slice(0,400), detail: String(detail||'').slice(0,400) } }, '*'); } catch(_){}
  }
  window.addEventListener('error', function(e){
    if (e && e.target && (e.target.src || e.target.href)) {
      send('resource', 'Failed to load ' + (e.target.src || e.target.href), e.target.tagName);
    } else if (e && e.message) {
      send('error', e.message, (e.filename||'') + ':' + (e.lineno||0) + ':' + (e.colno||0));
    }
  }, true);
  window.addEventListener('unhandledrejection', function(e){
    var r = e && e.reason;
    send('promise', (r && r.message) ? r.message : String(r), (r && r.stack) ? r.stack : '');
  });
})();</script>`;

/**
 * Navigation guard: an in-page anchor (`href="#x"`) navigates a sandboxed
 * srcdoc iframe to `about:srcdoc#x`, which WebKit renders as a blank page.
 * Intercept those clicks and smooth-scroll in JS instead so the page stays put.
 */
const NAV_GUARD = `<script>(function(){
  document.addEventListener('click', function(e){
    var a = e.target && e.target.closest && e.target.closest('a[href]');
    if (!a) return;
    var href = a.getAttribute('href') || '';
    if (href.charAt(0) !== '#' || href.length < 2) return;
    var id = href.slice(1), el = document.getElementById(id);
    if (!el) { try { el = document.querySelector(href); } catch(_){} }
    if (!el && document.getElementsByName) el = document.getElementsByName(id)[0];
    if (el) { e.preventDefault(); el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  }, true);
})()<\/script>`;

/** Inject the shims right after <head> so they install before page scripts. */
function withErrorShim(html: string): string {
  const shims = ERROR_SHIM + NAV_GUARD;
  const head = html.match(/<head[^>]*>/i);
  if (head && head.index !== undefined) {
    const at = head.index + head[0].length;
    return html.slice(0, at) + shims + html.slice(at);
  }
  const tag = html.match(/<html[^>]*>/i);
  if (tag && tag.index !== undefined) {
    const at = tag.index + tag[0].length;
    return html.slice(0, at) + shims + html.slice(at);
  }
  return shims + html;
}

export function CanvasPanel({
  open,
  versions,
  index,
  busy,
  onSelectVersion,
  onIterate,
  onFix,
  onExport,
  onOpenExternal,
  onClose,
}: {
  open: boolean;
  versions: CanvasVersion[];
  index: number;
  busy: boolean;
  onSelectVersion: (i: number) => void;
  onIterate: (instruction: string) => void;
  onFix: (errorText: string) => void;
  onExport: (html: string) => void;
  onOpenExternal: (html: string) => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [instruction, setInstruction] = useState("");
  const [error, setError] = useState<CanvasError | null>(null);
  const [muted, setMuted] = useState(false);

  const current = versions[index];
  const srcDoc = useMemo(
    () => (current ? withErrorShim(withStorageShim(current.html)) : ""),
    [current],
  );

  // A new version (or switch) means the old error no longer applies.
  useEffect(() => {
    setError(null);
  }, [index, current?.html]);

  // Receive runtime errors from the sandboxed (null-origin) preview. Keep only
  // the first per render; the shim already de-dups identical ones.
  useEffect(() => {
    if (!open) return;
    const onMsg = (e: MessageEvent) => {
      const d = (e.data as { __chatyCanvasError?: CanvasError })?.__chatyCanvasError;
      if (!d || muted) return;
      setError((prev) => prev ?? d);
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, [open, muted]);

  // Esc closes the studio.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open || !current) return null;

  const submit = () => {
    const text = instruction.trim();
    if (!text || busy) return;
    setInstruction("");
    onIterate(text);
  };

  return createPortal(
    <div className="canvas-overlay">
      <div className="canvas">
        <div className="canvas-head">
          <span className="canvas-title">{t("canvasTitle")}</span>
          <span className="canvas-ver-label">
            v{index + 1}/{versions.length}
          </span>
          <div className="canvas-head-actions">
            <button
              className="canvas-hbtn"
              title={t("canvasOpenExt")}
              onClick={() => onOpenExternal(current.html)}
            >
              {t("canvasOpenExt")}
            </button>
            <button
              className="canvas-hbtn"
              title={t("canvasExport")}
              onClick={() => onExport(current.html)}
            >
              <IconDownload size={13} style={{ marginRight: 5 }} />
              {t("canvasExport")}
            </button>
            <button className="canvas-close" title={t("closePreview")} onClick={onClose}>
              ×
            </button>
          </div>
        </div>

        <div className="canvas-body">
          <div className="canvas-versions">
            {versions.map((v, i) => (
              <button
                key={i}
                className={`canvas-ver ${i === index ? "active" : ""}`}
                onClick={() => onSelectVersion(i)}
                title={v.note}
              >
                <span className="cv-num">v{i + 1}</span>
                <span className="cv-note">{v.note}</span>
              </button>
            ))}
          </div>

          <div className="canvas-stage">
            <iframe
              key={index}
              className="canvas-frame"
              title="Canvas preview"
              srcDoc={srcDoc}
              sandbox="allow-scripts allow-modals allow-forms allow-popups allow-pointer-lock"
            />
            {busy && (
              <div className="canvas-busy">
                <span className="canvas-spinner" />
                {t("canvasGenerating")}
              </div>
            )}
          </div>
        </div>

        {error && !muted && (
          <div className="canvas-heal">
            <span className="canvas-heal-msg">
              {t("canvasHealMsg")}
              <code>{error.message}</code>
            </span>
            <div className="canvas-heal-actions">
              <button
                className="canvas-heal-fix"
                disabled={busy}
                onClick={() => {
                  onFix(`${error.message}${error.detail ? ` (${error.detail})` : ""}`);
                  setError(null);
                }}
              >
                {t("canvasFixBtn")}
              </button>
              <button className="canvas-heal-ghost" onClick={() => setError(null)}>
                {t("canvasIgnore")}
              </button>
              <button
                className="canvas-heal-ghost"
                onClick={() => {
                  setMuted(true);
                  setError(null);
                }}
              >
                {t("canvasMute")}
              </button>
            </div>
          </div>
        )}

        <div className="canvas-composer">
          <IconEdit size={15} style={{ color: "var(--faint)", flex: "none" }} />
          <input
            className="canvas-input"
            placeholder={t("canvasIterate")}
            value={instruction}
            disabled={busy}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submit();
              }
            }}
          />
          <button className="canvas-send" disabled={busy || !instruction.trim()} onClick={submit}>
            {t("canvasSend")}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
