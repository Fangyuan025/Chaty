import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { useConfirm } from "./ConfirmModal";
import { Icon } from "./Icon";
import { withStorageShim } from "./Markdown";
import { IconDownload, IconEdit } from "./icons";
import { annotate, buildFixPayload, highlightLines, INSPECT_SHIM } from "../lib/canvasSource";
import { diffLines } from "../lib/diff";
import { buildScanView } from "../lib/canvasStream";

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
 * srcdoc/null-origin compatibility shim: APIs that work on a normally-served
 * page but THROW inside a sandboxed srcdoc frame (SecurityError and friends).
 * A page that runs fine in the user's browser must not error here. Installed
 * before any page script; every shim is best-effort.
 */
const COMPAT_SHIM = `<script>(function(){
  try { // history API throws on about:srcdoc
    var hp = history.pushState.bind(history), hr = history.replaceState.bind(history);
    history.pushState = function(){ try { return hp.apply(null, arguments); } catch(_){} };
    history.replaceState = function(){ try { return hr.apply(null, arguments); } catch(_){} };
  } catch(_){}
  try { // cookies throw in sandboxed docs — in-memory jar
    var jar = '';
    Object.defineProperty(document, 'cookie', {
      get: function(){ return jar; },
      set: function(v){ try { var kv = String(v).split(';')[0]; if (kv && kv.indexOf('=') > 0) {
        var k = kv.split('=')[0].trim(); var parts = jar ? jar.split('; ') : [];
        parts = parts.filter(function(p){ return p.split('=')[0] !== k; });
        parts.push(kv.trim()); jar = parts.join('; ');
      } } catch(_){} return true; },
      configurable: true
    });
  } catch(_){}
  try { // clipboard rejects on null origin — succeed quietly
    if (!navigator.clipboard) {
      Object.defineProperty(navigator, 'clipboard', { value: {
        writeText: function(){ return Promise.resolve(); },
        readText: function(){ return Promise.resolve(''); }
      }, configurable: true });
    } else if (navigator.clipboard.writeText) {
      var wt = navigator.clipboard.writeText.bind(navigator.clipboard);
      try { navigator.clipboard.writeText = function(t){ return wt(t).catch(function(){}); }; } catch(_){}
    }
  } catch(_){}
})();</script>`;

/**
 * Console capture: mirrors every console call, uncaught error and rejection
 * up to the Canvas so the code pane's Console tab shows the CURRENT version's
 * output. Capped; strings truncated.
 */
const CONSOLE_SHIM = `<script>(function(){
  var count = 0;
  function send(level, args){
    if (count >= 300) return; count++;
    var text = '';
    try { text = Array.prototype.map.call(args, function(a){
      if (typeof a === 'string') return a;
      try { return JSON.stringify(a); } catch(_) { return String(a); }
    }).join(' '); } catch(_) { text = '[unserializable]'; }
    try { parent.postMessage({ __chatyCvConsole: { level: level, text: String(text).slice(0, level === 'error' ? 2000 : 600), nonce: window.__CV_NONCE || '' } }, '*'); } catch(_){}
  }
  ['log','info','warn','error','debug'].forEach(function(l){
    var orig = console[l] && console[l].bind(console);
    console[l] = function(){ send(l === 'info' || l === 'debug' ? 'log' : l, arguments); if (orig) orig.apply(null, arguments); };
  });
  window.addEventListener('error', function(e){
    if (e && e.target && (e.target.src || e.target.href)) send('error', ['Failed to load resource: ' + (e.target.src || e.target.href)]);
    else if (e && e.message) send('error', [e.message + ' (' + (e.filename||'') + ':' + (e.lineno||0) + ')' + (e.error && e.error.stack ? '\n' + e.error.stack : '')]);
  }, true);
  window.addEventListener('unhandledrejection', function(e){
    var r = e && e.reason; send('error', ['Unhandled rejection: ' + ((r && r.message) || String(r))]);
  });
})();</script>`;

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
    try { parent.postMessage({ __chatyCanvasError: { kind: kind, message: String(msg).slice(0,1000), detail: String(detail||'').slice(0,2000), nonce: window.__CV_NONCE || '' } }, '*'); } catch(_){}
  }
  window.addEventListener('error', function(e){
    if (e && e.target && (e.target.src || e.target.href)) {
      send('resource', 'Failed to load ' + (e.target.src || e.target.href), e.target.tagName);
    } else if (e && e.message) {
      send('error', e.message, (e.filename||'') + ':' + (e.lineno||0) + ':' + (e.colno||0) + (e.error && e.error.stack ? '\n' + e.error.stack : ''));
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

/**
 * Scrollbar that matches the previewed page. With "show scrollbars: always"
 * (or a mouse plugged in) the UA paints a light track+gutter in the srcdoc
 * frame, which glares beside the dark pages models like to build — WebKit only
 * draws a dark scrollbar when the document declares a `color-scheme`.
 *
 * So declare one FOR the page, inferred from what it actually renders, and let
 * the UA draw its own native scrollbar. Deliberately not `::-webkit-scrollbar`
 * rules: styling those switches the engine to a custom scrollbar whose gutter
 * is backed by the white canvas, not by the page's background — a transparent
 * track then reads as PURE WHITE (measured: rgb(255,255,255)), which is worse
 * than the bug. Setting color-scheme also darkens that canvas base, so the
 * gutter stops glowing at all.
 *
 * A page that declares its own color-scheme is left completely alone.
 */
const SCROLL_SCHEME_SHIM = `<script>(function(){
  function bright(c){
    var m=/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?/.exec(c||'');
    if(!m) return null;
    if(m[4]!==undefined && parseFloat(m[4])===0) return null; // fully transparent
    return 0.299*+m[1]+0.587*+m[2]+0.114*+m[3];              // perceived brightness
  }
  function apply(){
    try{
      var de=document.documentElement, b=document.body; if(!b) return;
      var declared=getComputedStyle(de).colorScheme;
      if(declared && declared!=='normal') return;             // the page decides
      var bg=bright(getComputedStyle(b).backgroundColor);
      if(bg===null) bg=bright(getComputedStyle(de).backgroundColor);
      var dark;
      if(bg!==null) dark = bg < 110;
      else {                                                  // gradient/image bg:
        var t=bright(getComputedStyle(b).color);               // light text ⇒ dark page
        dark = t!==null && t > 128;
      }
      de.style.colorScheme = dark ? 'dark' : 'light';
    }catch(_){}
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',apply); else apply();
  setTimeout(apply,300); // late CSS (webfonts, async styles) can flip the verdict
})()<\/script>`;

/** Inject the shims right after <head> so they install before page scripts. */
function withShims(html: string, nonce: string): string {
  // The nonce ties every message from this document generation to the
  // srcDoc that produced it: WKWebView can start reparsing (and posting
  // errors) BEFORE React's post-commit clear effect runs, so a timing-based
  // clear silently ate early errors — generation filtering is order-proof.
  const shims =
    `<script>window.__CV_NONCE=${JSON.stringify(nonce)};</script>` +
    COMPAT_SHIM + CONSOLE_SHIM + ERROR_SHIM + NAV_GUARD + INSPECT_SHIM + SCROLL_SCHEME_SHIM;
  // Inject at the very TOP of the document (only the doctype may precede us,
  // or it would flip the page into quirks mode). Injecting "after <head>" by
  // regex trusted the DOCUMENT's structure: models produce html like
  // `<html lang="zh-CN"` with the closing > missing, whose tokenizer recovery
  // eats the literal <head> — and the shims injected beside it — so NOTHING
  // was instrumented and the console stayed silent no matter how broken the
  // page was. Scripts between doctype and <html> are spec-legal; the parser
  // hoists them into the implied head and later merges the real tag.
  const dt = html.match(/^\s*<!doctype[^>]*>/i);
  const at = dt && dt.index !== undefined ? dt.index + dt[0].length : 0;
  return html.slice(0, at) + shims + html.slice(at);
}

export function CanvasPanel({
  open,
  versions,
  index,
  busy,
  streamText = null,
  onSelectVersion,
  onReset,
  onManualEdit,
  onIterate,
  onFix,
  onStop,
  onExport,
  onOpenExternal,
  onClose,
}: {
  open: boolean;
  versions: CanvasVersion[];
  index: number;
  busy: boolean;
  /** The model's partial output while an iteration streams (null when idle) —
   *  drives the Cursor-style live scan/diff in the code pane. */
  streamText?: string | null;
  onSelectVersion: (i: number) => void;
  /** Drop all iterations back to v1 (guarded by a confirm dialog here). */
  onReset: () => void;
  /** A hand-written source edit — lands as a new version. */
  onManualEdit: (html: string) => void;
  onIterate: (instruction: string) => void;
  onFix: (errorText: string) => void;
  /** Abort the in-flight iteration (wired to the engine's cancel). */
  onStop?: () => void;
  onExport: (html: string) => void;
  onOpenExternal: (html: string) => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [instruction, setInstruction] = useState("");
  const [error, setError] = useState<CanvasError | null>(null);
  const [muted, setMuted] = useState(false);
  const [view, setView] = useState<"code" | "diff" | "console">("code");
  const [consoleLog, setConsoleLog] = useState<{ level: string; text: string; nonce?: string }[]>([]);
  // Bumping remounts the iframe: scripts re-run from scratch (page refresh).
  const [reloadNonce, setReloadNonce] = useState(0);
  // Inspect selection: cv ids the user clicked (⌘/Ctrl toggles membership).
  const [selected, setSelected] = useState<number[]>([]);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  // Caret's current line in the manual editor — drives the backdrop's
  // active-line stripe (the textarea itself is transparent).
  const [editLine, setEditLine] = useState(0);
  const caretLine = (el: HTMLTextAreaElement) =>
    el.value.slice(0, el.selectionStart ?? 0).split("\n").length - 1;
  const editHlRef = useRef<HTMLPreElement>(null);
  const [inspect, setInspect] = useState(false);
  const [hotLine, setHotLine] = useState<number | null>(null);
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const codeRef = useRef<HTMLDivElement | null>(null);
  const prevCount = useRef(versions.length);
  // Resizable layout: version-rail width and code-pane share, persisted.
  const [railW, setRailW] = useState(() => {
    try { return JSON.parse(localStorage.getItem("chaty.canvasLayout") ?? "{}").railW ?? 150; } catch { return 150; }
  });
  const [codePct, setCodePct] = useState(() => {
    try { return JSON.parse(localStorage.getItem("chaty.canvasLayout") ?? "{}").codePct ?? 50; } catch { return 50; }
  });
  const [dragging, setDragging] = useState<null | "rail" | "split">(null);
  const [full, setFull] = useState(() => {
    try { return !!JSON.parse(localStorage.getItem("chaty.canvasLayout") ?? "{}").full; } catch { return false; }
  });
  const confirmDialog = useConfirm();
  // The whole drag can land inside one JS task (fast hands, automation) —
  // before React commits `dragging`. The ref is the synchronous truth.
  const dragRef = useRef<null | "rail" | "split">(null);
  useEffect(() => {
    try { localStorage.setItem("chaty.canvasLayout", JSON.stringify({ railW, codePct, full })); } catch { /* ignore */ }
  }, [railW, codePct, full]);

  const current = versions[index];

  // Live scan while an iteration streams in.
  // Guarded by `busy`: a scan view must be impossible unless a generation is
  // actually running (a stray stream value once left the badge stuck on).
  const scan = useMemo(
    () => (busy && streamText !== null && streamText !== undefined && current ? buildScanView(current.html, streamText) : null),
    [busy, streamText, current],
  );

  // Element↔code correspondence: annotate the source once per version; the
  // preview renders the annotated html, the code pane renders the ORIGINAL
  // (what the model wrote — annotations are plumbing, not content).
  const annotated = useMemo(() => (current ? annotate(current.html) : null), [current]);
  const frameNonce = useMemo(
    () => (annotated ? Math.random().toString(36).slice(2, 10) : ""),
    [annotated],
  );
  const nonceRef = useRef(frameNonce);
  nonceRef.current = frameNonce;
  const srcDoc = useMemo(
    () => (annotated ? withShims(withStorageShim(annotated.html), frameNonce) : ""),
    [annotated, frameNonce],
  );
  const codeLines = useMemo(() => (current ? highlightLines(current.html) : []), [current]);
  const diff = useMemo(
    () => (index > 0 && current ? diffLines(versions[index - 1].html, current.html) : null),
    [versions, index, current],
  );

  // A fresh iteration/fix landing = show WHAT CHANGED first, not a black box.
  useEffect(() => {
    if (versions.length > prevCount.current && versions.length >= 2) {
      setView("diff");
    }
    prevCount.current = versions.length;
  }, [versions.length]);

  // Old error / hot line don't apply across version switches; and a version
  // with no predecessor has no diff to show — fall back to the code view.
  useEffect(() => {
    setError(null);
    setHotLine(null);
    // Keep entries from the CURRENT document generation: on WKWebView the
    // new srcdoc can post its first errors before this effect runs, and a
    // blind wipe ate them (the reported "console shows nothing" file).
    setConsoleLog((prev) => prev.filter((c) => c.nonce && c.nonce === nonceRef.current));
    setSelected([]);
    setEditing(false);
    if (index === 0 && view === "diff") setView("code");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, current?.html]);

  // Messages from the sandboxed (null-origin) preview: runtime errors + the
  // inspect shim's hover/pick reports.
  useEffect(() => {
    if (!open) return;
    const onMsg = (e: MessageEvent) => {
      const data = e.data as {
        __chatyCanvasError?: CanvasError;
        __chatyCvHover?: string;
        __chatyCvPick?: string;
      };
      const con = (data as { __chatyCvConsole?: { level: string; text: string; nonce?: string } })
        ?.__chatyCvConsole;
      if (con && (!con.nonce || con.nonce === nonceRef.current)) {
        setConsoleLog((prev) => (prev.length >= 300 ? prev : [...prev, con]));
      }
      if (data?.__chatyCanvasError && !muted) {
        const d = data.__chatyCanvasError as CanvasError & { nonce?: string };
        if (!d.nonce || d.nonce === nonceRef.current) setError((prev) => prev ?? d);
      }
      const sel = (data as { __chatyCvSelect?: { cv: string; multi: boolean } })?.__chatyCvSelect;
      if (sel && annotated) {
        const id = Number(sel.cv);
        setSelected((prev) => {
          if (sel.multi) {
            return prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
          }
          return prev.length === 1 && prev[0] === id ? [] : [id];
        });
        const line = annotated.lineOf[id];
        if (line !== undefined && !scan) {
          setView("code");
          setHotLine(line);
        }
      }
      const cv = data?.__chatyCvHover;
      if (cv !== undefined && annotated) {
        const line = annotated.lineOf[Number(cv)];
        if (line !== undefined && !scan) {
          setView("code");
          setHotLine(line);
        }
      }
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, [open, muted, annotated]);

  // Follow the scan head while the model streams — unless the user scrolled
  // away to inspect (wheel/touch breaks follow; a pill resumes it). A fresh
  // generation always re-arms following.
  const scanRef = useRef<HTMLDivElement | null>(null);
  const [followScan, setFollowScan] = useState(true);
  // Rows outside a window around the head render as two spacer blocks, so a
  // long document doesn't rebuild thousands of DOM rows every stream tick.
  const [scanWin, setScanWin] = useState(0);
  const SCAN_ROW_H = 19; // .cvp-code line-height — spacer math relies on it
  const SCAN_OVERSCAN = 140;
  useEffect(() => {
    if (busy) setFollowScan(true);
  }, [busy]);
  useEffect(() => {
    if (!scan) return;
    if (followScan) {
      const head = scan.scanIndex ?? scan.rows.length - 1;
      setScanWin(Math.max(0, head - SCAN_OVERSCAN));
    }
  }, [scan, followScan]);
  useEffect(() => {
    if (!followScan || !scan || scan.scanIndex === null || !scanRef.current) return;
    // Row-arithmetic scroll (rows are a fixed 19px): element-based
    // scrollIntoView chases geometry that the window spacers shift in the
    // same commit, so its animation lands short.
    const el = scanRef.current;
    const target = Math.max(0, scan.scanIndex * SCAN_ROW_H + 8 - el.clientHeight / 2);
    el.scrollTo({ top: target, behavior: "smooth" });
  }, [scan, followScan, scanWin]);
  const onScanUserScroll = () => setFollowScan(false);
  const onScanScroll = () => {
    // Off-follow, the render window tracks the viewport instead of the head.
    if (followScan || !scanRef.current) return;
    const first = Math.floor(scanRef.current.scrollTop / SCAN_ROW_H);
    setScanWin((w) => {
      const next = Math.max(0, first - SCAN_OVERSCAN);
      return Math.abs(next - w) > SCAN_OVERSCAN / 2 ? next : w;
    });
  };

  // Keep the hot line in view while inspecting from the preview side.
  useEffect(() => {
    if (hotLine === null || !codeRef.current) return;
    const el = codeRef.current.querySelector(`[data-ln="${hotLine}"]`);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [hotLine]);

  // (Re-)arm the inspect shim on toggle and after every preview reload.
  const armInspect = (on: boolean) => {
    frameRef.current?.contentWindow?.postMessage({ __chatyCvArm: on }, "*");
  };
  useEffect(() => {
    armInspect(inspect);
    if (!inspect) setSelected([]);
  }, [inspect, srcDoc]);

  // Mirror the selection into the preview (and re-apply after reloads).
  useEffect(() => {
    frameRef.current?.contentWindow?.postMessage({ __chatyCvSetSel: selected.map(String) }, "*");
  }, [selected, srcDoc, reloadNonce]);

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
    if (selected.length && annotated && current) {
      const lines = current.html.split("\n");
      const items = selected
        .map((id) => {
          const ln = annotated.lineOf[id];
          const snippet = (lines[ln] ?? "").trim().slice(0, 100);
          return `- L${ln + 1}: ${snippet}`;
        })
        .join("\n");
      setSelected([]);
      onIterate(`${t("canvasSelPrefix")}\n${items}\n\n${text}`);
      return;
    }
    onIterate(text);
  };

  // Divider drags (pointer capture; the iframe eats moves otherwise, so the
  // stage carries a .dragging class that turns off frame pointer events).
  const startDrag = (which: "rail" | "split") => (e: React.PointerEvent) => {
    e.preventDefault();
    try {
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointers may lack a capturable id */
    }
    dragRef.current = which;
    setDragging(which);
  };
  const onDragMove = (e: React.PointerEvent) => {
    const dragging = dragRef.current;
    if (!dragging) return;
    const body = e.currentTarget as HTMLElement;
    if (dragging === "rail") {
      const r = body.getBoundingClientRect();
      setRailW(Math.min(340, Math.max(96, e.clientX - r.left)));
    } else {
      const stage = body.querySelector(".canvas-stage")?.getBoundingClientRect();
      if (!stage || stage.width < 1) return;
      const pct = ((stage.right - e.clientX) / stage.width) * 100;
      setCodePct(Math.min(85, Math.max(15, pct)));
    }
  };
  const endDrag = () => {
    dragRef.current = null;
    setDragging(null);
  };

  // Code line → preview element: flash the first element that starts on (or
  // before) the clicked line.
  const flashLine = (ln: number) => {
    if (!annotated) return;
    setHotLine(ln);
    let best = -1;
    for (let cv = 0; cv < annotated.lineOf.length; cv++) {
      if (annotated.lineOf[cv] <= ln) best = cv;
      else break;
    }
    if (best >= 0) {
      frameRef.current?.contentWindow?.postMessage({ __chatyCvFlash: String(best) }, "*");
    }
  };

  return createPortal(
    <div className={`canvas-overlay ${full ? "full" : ""}`}>
      <div className="canvas">
        <div className="canvas-head" data-tauri-drag-region>
          <span className="canvas-title">{t("canvasTitle")}</span>
          <span className="canvas-ver-label">
            v{index + 1}/{versions.length}
          </span>
          <div className="canvas-head-actions">
            <button
              className="canvas-hbtn"
              title={t("canvasResetHint")}
              disabled={versions.length <= 1 || busy}
              onClick={() => {
                void (async () => {
                  if (
                    await confirmDialog({
                      title: t("canvasReset"),
                      message: t("canvasResetConfirm"),
                      confirmLabel: t("canvasReset"),
                      danger: true,
                    })
                  ) {
                    onReset();
                  }
                })();
              }}
            >
              {t("canvasReset")}
            </button>
            <button
              className="canvas-hbtn icon"
              title={t("canvasReload")}
              onClick={() => {
                setConsoleLog([]);
                setError(null);
                setReloadNonce((n) => n + 1);
              }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <path d="M20 12a8 8 0 1 1-2.34-5.66M20 4v5h-5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <button
              className="canvas-hbtn icon"
              title={full ? t("canvasExitFull") : t("canvasFull")}
              onClick={() => setFull(!full)}
            >
              {full ? (
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                  <path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                  <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </button>
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
              <Icon name="x" size={12} strokeWidth={2.2} />
            </button>
          </div>
        </div>

        <div
          className={`canvas-body ${dragging ? "dragging" : ""}`}
          onPointerMove={onDragMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          <div className="canvas-versions" style={{ width: railW, flex: "none" }}>
            {versions.map((v, i) => (
              <button
                key={i}
                className={`canvas-ver ${i === index ? "active" : ""}`}
                // Mid-generation the scan diffs the stream against the version
                // that was current at start — switching now would render the
                // scan against the wrong base.
                disabled={busy && i !== index}
                onClick={() => onSelectVersion(i)}
                title={v.note}
              >
                <span className="cv-num">v{i + 1}</span>
                <span className="cv-note">{v.note}</span>
              </button>
            ))}
          </div>

          <div
            className="cv-divider rail"
            title={t("canvasDragHint")}
            onPointerDown={startDrag("rail")}
            onDoubleClick={() => setRailW(150)}
          />
          <div className="canvas-stage split">
            <div className="canvas-pane preview" style={{ flex: `1 1 ${100 - codePct}%` }}>
              <iframe
                key={`${index}-${reloadNonce}`}
                ref={frameRef}
                className="canvas-frame"
                title="Canvas preview"
                srcDoc={srcDoc}
                sandbox="allow-scripts allow-modals allow-forms allow-popups allow-pointer-lock"
                onLoad={() => armInspect(inspect)}
              />
              {busy && !scan && (
                <div className="canvas-busy">
                  <span className="canvas-spinner" />
                  {t("canvasGenerating")}
                </div>
              )}
            </div>

            <div
              className="cv-divider split"
              title={t("canvasDragHint")}
              onPointerDown={startDrag("split")}
              onDoubleClick={() => setCodePct(50)}
            />
            <div className="canvas-pane codepane" style={{ flex: `1 1 ${codePct}%` }}>
              <div className="cvp-head">
                <button
                  className={`cvp-inspect ${inspect ? "active" : ""}`}
                  title={t("canvasInspectHint")}
                  onClick={() => setInspect(!inspect)}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <path d="M3 3l8 19 2.5-7.5L21 12z" strokeLinejoin="round" />
                  </svg>
                  {t("canvasInspect")}
                </button>
                <span className="cvp-spacer" />
                {scan && (
                  <span className="cvp-scanning">
                    <span className="canvas-spinner" />
                    {t("canvasScanning")}
                  </span>
                )}
                {diff && (
                  <span className="cvp-diffstat">
                    <span className="plus">+{diff.added}</span> <span className="minus">−{diff.removed}</span>
                  </span>
                )}
                <button
                  className={`cvp-tab ${view === "code" && !scan ? "active" : ""}`}
                  disabled={!!scan}
                  onClick={() => setView("code")}
                >
                  {t("canvasCode")}
                </button>
                <button
                  className={`cvp-tab ${view === "diff" && !scan ? "active" : ""}`}
                  disabled={!diff || !!scan}
                  title={diff ? "" : t("canvasNoDiff")}
                  onClick={() => diff && setView("diff")}
                >
                  {t("canvasDiff")}
                </button>
                <button
                  className="cvp-tab"
                  disabled={!!scan || busy || editing}
                  title={t("canvasEditCodeHint")}
                  onClick={() => {
                    setDraft(current.html);
                    setEditing(true);
                  }}
                >
                  {t("canvasEditCode")}
                </button>
                <button
                  className={`cvp-tab ${view === "console" && !scan ? "active" : ""}`}
                  disabled={!!scan}
                  onClick={() => setView("console")}
                >
                  {t("canvasConsole")}
                  {(() => {
                    const errs = consoleLog.filter((c) => c.level === "error").length;
                    const warns = consoleLog.filter((c) => c.level === "warn").length;
                    return errs + warns > 0 ? (
                      <span className={`cvp-conbadge ${errs ? "err" : "warn"}`}>
                        {errs || warns}
                      </span>
                    ) : null;
                  })()}
                </button>
              </div>

              {editing ? (
                <div className="cvp-editwrap">
                  {/* Highlight lives in a backdrop <pre>; the textarea floats
                      transparent on top (text invisible, caret kept) so hand
                      edits stay syntax-colored. Oversized docs fall back to
                      the plain editor rather than re-highlighting per keystroke. */}
                  {draft.length <= 60000 ? (
                    <div className="cvp-editstack">
                      <pre className="cvp-edit-hl hljs" ref={editHlRef} aria-hidden="true">
                        {highlightLines(draft).map((h, i) => (
                          <div
                            key={i}
                            className={i === editLine ? "cvp-edit-line active" : "cvp-edit-line"}
                            dangerouslySetInnerHTML={{ __html: h || "&nbsp;" }}
                          />
                        ))}
                      </pre>
                      {/* wrap="off": WKWebView soft-wraps textareas regardless
                          of white-space:pre — one wrapped long line shifts
                          every later line off its highlight (the reported
                          selection drift). */}
                      <textarea
                        className="cvp-editor overlay"
                        value={draft}
                        wrap="off"
                        spellCheck={false}
                        onChange={(e) => {
                          setDraft(e.target.value);
                          setEditLine(caretLine(e.currentTarget));
                        }}
                        onSelect={(e) => setEditLine(caretLine(e.currentTarget))}
                        onKeyUp={(e) => setEditLine(caretLine(e.currentTarget))}
                        onClick={(e) => setEditLine(caretLine(e.currentTarget))}
                        onScroll={(e) => {
                          const el = editHlRef.current;
                          if (el) {
                            el.scrollTop = e.currentTarget.scrollTop;
                            el.scrollLeft = e.currentTarget.scrollLeft;
                          }
                        }}
                      />
                    </div>
                  ) : (
                    <textarea
                      className="cvp-editor"
                      value={draft}
                      wrap="off"
                      spellCheck={false}
                      onChange={(e) => setDraft(e.target.value)}
                    />
                  )}
                  <div className="cvp-editbar">
                    <button className="cvp-tab" onClick={() => setEditing(false)}>
                      {t("cancel")}
                    </button>
                    <button
                      className="cvp-tab primary"
                      disabled={!draft.trim() || draft === current.html}
                      onClick={() => {
                        setEditing(false);
                        onManualEdit(draft);
                      }}
                    >
                      {t("canvasSaveEdit")}
                    </button>
                  </div>
                </div>
              ) : scan ? (
                <div
                  className="cvp-code cvp-scanview"
                  ref={scanRef}
                  onWheel={onScanUserScroll}
                  onTouchMove={onScanUserScroll}
                  onScroll={onScanScroll}
                >
                  {scan.mode === "waiting" && (
                    <div className="cvp-scan-note">{t("canvasScanWaiting")}</div>
                  )}
                  {scanWin > 0 && <div style={{ height: scanWin * SCAN_ROW_H }} aria-hidden="true" />}
                  {scan.rows.slice(scanWin, scanWin + SCAN_OVERSCAN * 2).map((r, k) => {
                    const i = scanWin + k;
                    return (
                      <div
                        key={i}
                        data-scan={i}
                        className={`cm-dl ${r.kind === "pending" ? "ctx cvp-pending" : r.kind} ${i === scan.scanIndex ? "cvp-scanhead" : ""}`}
                      >
                        <span className="cm-dl-mark">
                          {r.kind === "add" ? "+" : r.kind === "del" ? "-" : " "}
                        </span>
                        {r.text}
                      </div>
                    );
                  })}
                  {scan.rows.length > scanWin + SCAN_OVERSCAN * 2 && (
                    <div
                      style={{ height: (scan.rows.length - scanWin - SCAN_OVERSCAN * 2) * SCAN_ROW_H }}
                      aria-hidden="true"
                    />
                  )}
                  {!followScan && (
                    <button className="cvp-follow" onClick={() => setFollowScan(true)}>
                      {t("canvasFollow")}
                    </button>
                  )}
                </div>
              ) : view === "console" ? (
                <div className="cvp-code cvp-console">
                  {consoleLog.length === 0 ? (
                    <div className="cvp-scan-note">{t("canvasConsoleEmpty")}</div>
                  ) : (
                    consoleLog.map((c, i) => (
                      <div key={i} className={`cvp-con-row ${c.level}`}>
                        <span className="cvp-con-lv">{c.level}</span>
                        {c.text}
                      </div>
                    ))
                  )}
                </div>
              ) : view === "code" ? (
                <div className="cvp-code hljs" ref={codeRef}>
                  {codeLines.map((h, ln) => (
                    <div
                      key={ln}
                      data-ln={ln}
                      className={`cvp-line ${ln === hotLine ? "hot" : ""}`}
                      onClick={() => flashLine(ln)}
                    >
                      <span className="cvp-ln">{ln + 1}</span>
                      <span className="cvp-src" dangerouslySetInnerHTML={{ __html: h || " " }} />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="cvp-code cvp-diffview">
                  {diff?.rows.map((r, i) => (
                    <div key={i} className={`cm-dl ${r.kind}`}>
                      <span className="cm-dl-mark">{r.kind === "add" ? "+" : r.kind === "del" ? "-" : " "}</span>
                      {r.text}
                    </div>
                  ))}
                  {diff?.truncated && <div className="cm-dl ctx cm-dl-more">…</div>}
                </div>
              )}
            </div>
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
                  // One click hands the model EVERY current error, not just
                  // the banner's — fixing them one round-trip at a time was
                  // the reported multi-error grind.
                  onFix(
                    buildFixPayload(
                      `${error.message}${error.detail ? ` (${error.detail})` : ""}`,
                      consoleLog.filter((c) => c.level === "error").map((c) => c.text),
                    ),
                  );
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

        {selected.length > 0 && annotated && (
          <div className="canvas-selbar">
            <span className="canvas-selhint">
              {t("canvasSelHint").replace("{n}", String(selected.length))}
            </span>
            {selected.map((id) => (
              <button
                key={id}
                className="canvas-selchip"
                title={`L${(annotated.lineOf[id] ?? 0) + 1}`}
                onClick={() => setSelected((prev) => prev.filter((x) => x !== id))}
              >
                {"<" + (annotated.tagOf[id] ?? "?") + ">"} · L{(annotated.lineOf[id] ?? 0) + 1}
                <span className="canvas-selx">×</span>
              </button>
            ))}
            <button className="canvas-selclear" onClick={() => setSelected([])}>
              {t("canvasSelClear")}
            </button>
          </div>
        )}
        <div className="canvas-composer">
          <IconEdit size={15} style={{ color: "var(--faint)", flex: "none" }} />
          <input
            className="canvas-input"
            placeholder={editing ? t("canvasComposerEditing") : t("canvasIterate")}
            value={instruction}
            disabled={busy || editing}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submit();
              }
            }}
          />
          {busy && onStop ? (
            <button className="canvas-send stop" onClick={onStop} title={t("stopTitle")}>
              {t("canvasStop")}
            </button>
          ) : (
            <button className="canvas-send" disabled={busy || editing || !instruction.trim()} onClick={submit}>
              {t("canvasSend")}
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
