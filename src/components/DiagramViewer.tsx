import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";

/**
 * A diagram on its own, over the window: fitted to it at first, then zoomed
 * (pinch, ⌘/Ctrl + wheel, the buttons, + and −) about the pointer, moved by
 * dragging or scrolling, a double-click between fitted and actual size — and
 * saved as SVG or PNG. Inline in a reply, a tall or wide diagram was drawn
 * too small to read and could go nowhere else.
 */
export function DiagramViewer({
  svg,
  onExport,
  onClose,
}: {
  svg: string;
  /** Save the diagram; resolves true when it was saved. */
  onExport: (kind: "svg" | "png") => Promise<boolean>;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const stage = useRef<HTMLDivElement>(null);
  const art = useRef<HTMLDivElement>(null);
  /** The drawing's own size, from its viewBox. */
  const size = useRef({ w: 800, h: 600 });
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const fitK = useRef(1);
  const [busy, setBusy] = useState<"" | "svg" | "png">("");
  const [saved, setSaved] = useState<"" | "svg" | "png">("");

  const fit = useCallback(() => {
    const el = stage.current;
    if (!el) return;
    const { w, h } = size.current;
    // Clear of the window's top row and of the toolbar under it.
    const top = 44;
    const bottom = 84;
    const room = el.clientHeight - top - bottom;
    const k = Math.min((el.clientWidth * 0.92) / w, room / h, 2);
    fitK.current = k;
    setView({ k, x: (el.clientWidth - w * k) / 2, y: top + (room - h * k) / 2 });
  }, []);

  // The drawing at its own size (Mermaid's is sized to its container), fitted.
  useLayoutEffect(() => {
    // Put in here, not by React: its re-renders set the markup again and took
    // back the drawing's size.
    if (!art.current) return;
    art.current.innerHTML = svg;
    const s = art.current.querySelector("svg");
    if (!s) return;
    const vb = (s.getAttribute("viewBox") ?? "").split(/[\s,]+/).map(Number);
    const w = vb[2] > 0 ? vb[2] : s.getBoundingClientRect().width || 800;
    const h = vb[3] > 0 ? vb[3] : s.getBoundingClientRect().height || 600;
    size.current = { w, h };
    // Mermaid sizes its drawing to the reply's column (max-width: 100%); here
    // it is laid out at its own size, which the zoom then scales.
    s.removeAttribute("style");
    s.setAttribute("width", String(w));
    s.setAttribute("height", String(h));
    s.style.maxWidth = "none";
    s.style.width = `${w}px`;
    s.style.height = `${h}px`;
    fit();
  }, [svg, fit]);

  /** Zoom to `k`, keeping the point (px, py) of the stage where it is. */
  const zoomTo = useCallback((k: number, px?: number, py?: number) => {
    const el = stage.current;
    if (!el) return;
    setView((v) => {
      const nk = Math.min(8, Math.max(0.05, k));
      const cx = px ?? el.clientWidth / 2;
      const cy = py ?? el.clientHeight / 2;
      return { k: nk, x: cx - ((cx - v.x) / v.k) * nk, y: cy - ((cy - v.y) / v.k) * nk };
    });
  }, []);

  useEffect(() => {
    const el = stage.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      if (e.ctrlKey || e.metaKey) {
        // A pinch arrives as a wheel with Ctrl held.
        setView((v) => {
          const nk = Math.min(8, Math.max(0.05, v.k * Math.exp(-e.deltaY / 200)));
          const cx = e.clientX - r.left;
          const cy = e.clientY - r.top;
          return { k: nk, x: cx - ((cx - v.x) / v.k) * nk, y: cy - ((cy - v.y) / v.k) * nk };
        });
      } else {
        setView((v) => ({ ...v, x: v.x - e.deltaX, y: v.y - e.deltaY }));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    const onResize = () => fit();
    window.addEventListener("resize", onResize);
    return () => {
      el.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
    };
  }, [fit]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      } else if (e.key === "+" || e.key === "=") zoomTo(view.k * 1.25);
      else if (e.key === "-") zoomTo(view.k / 1.25);
      else if (e.key === "0") fit();
      else if (e.key === "1") zoomTo(1);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose, zoomTo, fit, view.k]);

  // Dragging moves the drawing.
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    setView((v) => ({ ...v, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y }));
  };
  const onPointerUp = () => {
    drag.current = null;
  };
  const onDoubleClick = (e: React.MouseEvent) => {
    const r = stage.current!.getBoundingClientRect();
    if (Math.abs(view.k - 1) < 0.01) fit();
    else zoomTo(1, e.clientX - r.left, e.clientY - r.top);
  };

  const save = async (kind: "svg" | "png") => {
    if (busy) return;
    setBusy(kind);
    try {
      if (await onExport(kind)) {
        setSaved(kind);
        setTimeout(() => setSaved(""), 1600);
      }
    } catch (e) {
      console.error("diagram export", e);
    } finally {
      setBusy("");
    }
  };

  return createPortal(
    <div className="dv-overlay" role="dialog" aria-label={t("diagramViewer")}>
      <div
        ref={stage}
        className={`dv-stage ${drag.current ? "dragging" : ""}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={onDoubleClick}
      >
        <div
          ref={art}
          className="dv-art"
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}
        />
      </div>
      <div className="dv-bar" onPointerDown={(e) => e.stopPropagation()}>
        <button className="dv-btn" type="button" title={t("zoomOut")} aria-label={t("zoomOut")} onClick={() => zoomTo(view.k / 1.25)}>
          <Icon name="minus" size={14} strokeWidth={1.9} />
        </button>
        <button className="dv-btn dv-pct" type="button" title={t("zoomActual")} onClick={() => zoomTo(1)}>
          {Math.round(view.k * 100)}%
        </button>
        <button className="dv-btn" type="button" title={t("zoomIn")} aria-label={t("zoomIn")} onClick={() => zoomTo(view.k * 1.25)}>
          <Icon name="plus" size={14} strokeWidth={1.9} />
        </button>
        <button className="dv-btn" type="button" title={t("zoomFit")} aria-label={t("zoomFit")} onClick={fit}>
          <Icon name="fit" size={14} strokeWidth={1.8} />
        </button>
        <span className="dv-sep" />
        <button className="dv-btn dv-text" type="button" disabled={!!busy} onClick={() => void save("svg")}>
          {saved === "svg" ? <Icon name="check" size={13} strokeWidth={2.2} /> : null}
          {busy === "svg" ? "…" : t("saveSvg")}
        </button>
        <button className="dv-btn dv-text" type="button" disabled={!!busy} onClick={() => void save("png")}>
          {saved === "png" ? <Icon name="check" size={13} strokeWidth={2.2} /> : null}
          {busy === "png" ? "…" : t("savePng")}
        </button>
        <span className="dv-sep" />
        <button className="dv-btn" type="button" title={t("close")} aria-label={t("close")} onClick={onClose}>
          <Icon name="x" size={14} strokeWidth={1.9} />
        </button>
      </div>
    </div>,
    document.body,
  );
}
