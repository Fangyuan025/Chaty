import {
  useEffect,
  useRef,
  useState,
  type ComponentPropsWithoutRef,
  type ReactElement,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";
import { useI18n } from "../lib/i18n";
import { copyToClipboard } from "../lib/clipboard";

const remarkPlugins = [remarkGfm, remarkMath];
// `throwOnError: false` keeps partial LaTeX from crashing mid-stream;
// `ignoreMissing` tolerates unknown code-fence languages.
const rehypePlugins = [
  [rehypeKatex, { throwOnError: false }],
  [rehypeHighlight, { ignoreMissing: true }],
] as const;

/** Fenced language from the inner <code className="language-xxx">. */
function codeLang(children: ReactNode): string {
  const el = children as ReactElement<{ className?: string }>;
  const cls = el?.props?.className ?? "";
  const m = /language-(\w+)/.exec(cls);
  return m ? m[1].toLowerCase() : "";
}

/** Plain text of a fenced block's inner <code>. */
function codeText(children: ReactNode): string {
  const el = children as ReactElement<{ children?: ReactNode }>;
  const inner = el?.props?.children;
  if (typeof inner === "string") return inner;
  if (Array.isArray(inner)) return inner.filter((c) => typeof c === "string").join("");
  return "";
}

let mermaidReady: Promise<typeof import("mermaid").default> | null = null;
let mermaidSeq = 0;

/** Lazily-loaded Mermaid diagram. Falls back to the raw code on parse errors
 *  (e.g. while the block is still streaming in). */
function Mermaid({ code }: { code: string }) {
  const [svg, setSvg] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    const trimmed = code.trim();
    if (!trimmed) return;
    if (!mermaidReady) {
      const dark = !document.documentElement.dataset.theme?.includes("light");
      mermaidReady = import("mermaid").then((m) => {
        // "loose" — strict rejects common model-generated labels (e.g. with
        // HTML entities) and silently blanked the diagram. We render into our
        // own sandboxed context, and the SVG is inert markup.
        m.default.initialize({
          startOnLoad: false,
          theme: dark ? "dark" : "default",
          securityLevel: "loose",
        });
        return m.default;
      });
    }
    mermaidReady
      .then(async (mermaid) => {
        try {
          await mermaid.parse(trimmed);
          const { svg: out } = await mermaid.render(`mmd-${mermaidSeq++}`, trimmed);
          if (alive) {
            setSvg(out);
            setError("");
          }
        } catch (e) {
          // Expected while the block is still streaming in; the final code
          // triggers another attempt. Keep the message for the fallback.
          if (alive) setError((e as Error)?.message ?? String(e));
        }
      })
      .catch((e) => {
        if (alive) setError(`mermaid failed to load: ${(e as Error)?.message ?? e}`);
      });
    return () => {
      alive = false;
    };
  }, [code]);

  if (svg && !error) {
    return <div className="mermaid-diagram" dangerouslySetInnerHTML={{ __html: svg }} />;
  }
  return (
    <pre className="mermaid-src">
      <code>{code}</code>
      {error ? <div className="mermaid-err">{error.slice(0, 300)}</div> : null}
    </pre>
  );
}

/**
 * In-memory localStorage/sessionStorage shim, injected ahead of the snippet.
 * Sandboxed (non-same-origin) iframes throw SecurityError on storage access,
 * which crashes e.g. single-file games that save a highscore on boot.
 */
const STORAGE_SHIM = `<script>(function(){
  try { localStorage.getItem(""); } catch (_) {
    var m = new Map();
    var shim = {
      getItem: function(k){ k=String(k); return m.has(k) ? m.get(k) : null; },
      setItem: function(k,v){ m.set(String(k), String(v)); },
      removeItem: function(k){ m.delete(String(k)); },
      clear: function(){ m.clear(); },
      key: function(i){ return Array.from(m.keys())[i] ?? null; },
    };
    Object.defineProperty(shim, "length", { get: function(){ return m.size; } });
    try { Object.defineProperty(window, "localStorage", { value: shim }); } catch (_) {}
    try { Object.defineProperty(window, "sessionStorage", { value: shim }); } catch (_) {}
  }
})()</script>`;

/** Inject the storage shim so it runs before any of the snippet's scripts. */
function withStorageShim(html: string): string {
  const head = html.match(/<head[^>]*>/i);
  if (head && head.index !== undefined) {
    const at = head.index + head[0].length;
    return html.slice(0, at) + STORAGE_SHIM + html.slice(at);
  }
  const tag = html.match(/<html[^>]*>/i);
  if (tag && tag.index !== undefined) {
    const at = tag.index + tag[0].length;
    return html.slice(0, at) + STORAGE_SHIM + html.slice(at);
  }
  return STORAGE_SHIM + html;
}

/** A live, sandboxed preview of an HTML snippet rendered in an overlay. */
function HtmlPreview({ html, onClose }: { html: string; onClose: () => void }) {
  const { t } = useI18n();
  const [zoom, setZoom] = useState(1);
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const clampZoom = (z: number) => Math.min(2.5, Math.max(0.4, Math.round(z * 100) / 100));
  // Keyboard-driven games listen inside the iframe document — without focus,
  // Space/arrows land on the host window instead. Focus on load and keep it
  // through zoom-induced re-layouts.
  const focusFrame = () => {
    try {
      frameRef.current?.contentWindow?.focus();
    } catch {
      frameRef.current?.focus();
    }
  };
  useEffect(() => {
    focusFrame();
  }, [zoom]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if ((e.ctrlKey || e.metaKey) && (e.key === "=" || e.key === "+")) {
        e.preventDefault();
        setZoom((z) => clampZoom(z + 0.1));
      } else if ((e.ctrlKey || e.metaKey) && e.key === "-") {
        e.preventDefault();
        setZoom((z) => clampZoom(z - 0.1));
      } else if ((e.ctrlKey || e.metaKey) && e.key === "0") {
        e.preventDefault();
        setZoom(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div className="preview-overlay" onMouseDown={onClose}>
      <div className="preview-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="preview-bar">
          <span className="preview-title">HTML</span>
          <div className="preview-zoom">
            <button onClick={() => setZoom((z) => clampZoom(z - 0.1))} title={t("zoomOut")}>
              −
            </button>
            <button className="preview-zoom-val" onClick={() => setZoom(1)} title={t("zoomReset")}>
              {Math.round(zoom * 100)}%
            </button>
            <button onClick={() => setZoom((z) => clampZoom(z + 0.1))} title={t("zoomIn")}>
              +
            </button>
          </div>
          <button className="preview-close" onClick={onClose} title={t("closePreview")}>
            ×
          </button>
        </div>
        <div className="preview-body">
          <iframe
            ref={frameRef}
            className="preview-frame"
            title="HTML preview"
            srcDoc={withStorageShim(html)}
            sandbox="allow-scripts allow-modals allow-forms allow-popups allow-pointer-lock"
            onLoad={focusFrame}
            style={{
              width: `${100 / zoom}%`,
              height: `${100 / zoom}%`,
              transform: `scale(${zoom})`,
              transformOrigin: "0 0",
            }}
          />
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** A fenced code block with copy + (for HTML) live-preview buttons. */
function CodeBlock({ children, ...props }: ComponentPropsWithoutRef<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [isHtml, setIsHtml] = useState(false);
  const { t } = useI18n();

  const lang = codeLang(children);

  useEffect(() => {
    const text = (ref.current?.textContent ?? "").trimStart().toLowerCase();
    const sniff = text.startsWith("<!doctype") || text.startsWith("<html") || text.startsWith("<svg");
    setIsHtml(lang === "html" || lang === "htm" || (lang === "" && sniff));
  });

  // Mermaid diagrams render in place of the code block.
  if (lang === "mermaid") {
    return <Mermaid code={codeText(children)} />;
  }

  const copy = () => {
    const text = ref.current?.textContent ?? "";
    void copyToClipboard(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    });
  };

  return (
    <div className="code-block">
      <div className="code-actions">
        {isHtml && (
          <button
            className="code-btn"
            onClick={() => setPreviewing(true)}
            title={t("htmlPreviewTitle")}
            type="button"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              />
              <circle cx="12" cy="12" r="2.6" fill="currentColor" />
            </svg>
            {t("htmlPreview")}
          </button>
        )}
        <button
          className={`code-btn icon ${copied ? "done" : ""}`}
          onClick={copy}
          title="Copy code"
          type="button"
        >
          {copied ? (
            <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M5 13l4 4L19 7"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
              <rect
                x="9"
                y="9"
                width="11"
                height="11"
                rx="2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
              />
              <path
                d="M5 15V5a2 2 0 0 1 2-2h8"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
              />
            </svg>
          )}
        </button>
      </div>
      <pre ref={ref} {...props}>
        {children}
      </pre>
      {previewing && (
        <HtmlPreview html={ref.current?.textContent ?? ""} onClose={() => setPreviewing(false)} />
      )}
    </div>
  );
}

const components = { pre: CodeBlock } as const;

/** Markdown renderer with GFM tables, KaTeX math, and code highlighting. */
export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={remarkPlugins}
      rehypePlugins={rehypePlugins as never}
      components={components}
    >
      {children}
    </ReactMarkdown>
  );
}
