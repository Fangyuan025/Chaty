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

/** A live, sandboxed preview of an HTML snippet rendered in an overlay. */
function HtmlPreview({ html, onClose }: { html: string; onClose: () => void }) {
  const { t } = useI18n();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div className="preview-overlay" onMouseDown={onClose}>
      <div className="preview-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="preview-bar">
          <span className="preview-title">HTML</span>
          <button className="preview-close" onClick={onClose} title={t("closePreview")}>
            ×
          </button>
        </div>
        <iframe
          className="preview-frame"
          title="HTML preview"
          srcDoc={html}
          sandbox="allow-scripts allow-modals allow-forms allow-popups allow-pointer-lock"
        />
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

  const copy = () => {
    const text = ref.current?.textContent ?? "";
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      })
      .catch(() => {});
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
