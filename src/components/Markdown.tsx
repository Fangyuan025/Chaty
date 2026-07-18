import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentPropsWithoutRef,
  type ReactElement,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import "katex/dist/katex.min.css";
import { useI18n } from "../lib/i18n";
import { copyToClipboard } from "../lib/clipboard";

const remarkPlugins = [remarkGfm, remarkMath];
// `throwOnError: false` keeps partial LaTeX from crashing mid-stream;
// `ignoreMissing` tolerates unknown code-fence languages.
const rehypePlugins = [
  [rehypeKatex, { throwOnError: false }],
  [rehypeHighlight, { ignoreMissing: true }],
] as const;

// ---------------------------------------------------------------------------
// Inline citations: 【N】 tokens → hoverable superscript anchors
// ---------------------------------------------------------------------------

export interface CiteSource {
  title: string;
  url: string;
  snippet: string;
}

const CitesContext = createContext<CiteSource[]>([]);

/** When set by App, HTML code blocks gain an "open in Canvas" action that hands
 *  the snippet to the design studio. Null = feature unavailable (no handler). */
export const CanvasOpenContext = createContext<((html: string) => void) | null>(null);

const CITE_RE = /【(\d{1,2})】|\[(\d{1,2})\]/g;

/* eslint-disable @typescript-eslint/no-explicit-any */
/** Rehype plugin: split text nodes on 【N】 / [N] into `<sup data-cite="N">`.
 *  Skips code/pre and out-of-range numbers (left as plain text). */
function rehypeCites({ count }: { count: number }) {
  const walk = (node: any) => {
    if (!node || node.type === "comment") return;
    if (node.type === "element" && ["code", "pre", "sup", "a"].includes(node.tagName)) return;
    const kids: any[] = node.children;
    if (!kids) return;
    for (let i = 0; i < kids.length; i++) {
      const child = kids[i];
      if (child.type !== "text") {
        walk(child);
        continue;
      }
      const value: string = child.value;
      CITE_RE.lastIndex = 0;
      let m: RegExpExecArray | null;
      let last = 0;
      let replaced = false;
      const out: any[] = [];
      while ((m = CITE_RE.exec(value))) {
        const n = parseInt(m[1] ?? m[2], 10);
        if (!(n >= 1 && n <= count)) continue;
        if (m.index > last) out.push({ type: "text", value: value.slice(last, m.index) });
        out.push({
          type: "element",
          tagName: "sup",
          properties: { dataCite: String(n) },
          children: [{ type: "text", value: String(n) }],
        });
        last = m.index + m[0].length;
        replaced = true;
      }
      if (!replaced) continue;
      if (last < value.length) out.push({ type: "text", value: value.slice(last) });
      kids.splice(i, 1, ...out);
      i += out.length - 1;
    }
  };
  return (tree: any) => walk(tree);
}
/* eslint-enable @typescript-eslint/no-explicit-any */

/** Superscript citation anchor with a hover preview of the cited source. */
function CiteMark({ n }: { n: number }) {
  const cites = useContext(CitesContext);
  const s = cites[n - 1];
  if (!s) return <sup>{n}</sup>;
  return (
    <sup className="cite">
      <span className="cite-n">{n}</span>
      <span className="cite-pop">
        <span className="cite-pop-title">{s.title}</span>
        {s.snippet && <span className="cite-pop-text">{s.snippet}</span>}
      </span>
    </sup>
  );
}

function SupRenderer(
  props: ComponentPropsWithoutRef<"sup"> & { node?: unknown; "data-cite"?: string; dataCite?: string },
) {
  const { node: _node, ...rest } = props;
  const dc = rest["data-cite"] ?? rest.dataCite;
  if (dc) return <CiteMark n={Number(dc)} />;
  return <sup {...rest} />;
}

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
export function withStorageShim(html: string): string {
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


/** A fenced code block with copy + (for HTML) live-preview buttons, a
 *  language label, and a line-number gutter on longer blocks. */
function CodeBlock({ children, ...props }: ComponentPropsWithoutRef<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const [isHtml, setIsHtml] = useState(false);
  const [lines, setLines] = useState(0);
  const { t } = useI18n();
  const openCanvas = useContext(CanvasOpenContext);

  const lang = codeLang(children);

  useEffect(() => {
    const raw = ref.current?.textContent ?? "";
    const text = raw.trimStart().toLowerCase();
    const sniff = text.startsWith("<!doctype") || text.startsWith("<html") || text.startsWith("<svg");
    setIsHtml(lang === "html" || lang === "htm" || (lang === "" && sniff));
    // Line count for the gutter (grows live while the block streams in).
    setLines(raw ? raw.split("\n").length - (raw.endsWith("\n") ? 1 : 0) : 0);
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

  const showGutter = lines >= 4;

  return (
    <div className="code-block">
      {lang && <span className="code-lang">{lang}</span>}
      <div className="code-actions">
        {isHtml && openCanvas && (
          <button
            className="code-btn"
            onClick={() => openCanvas(ref.current?.textContent ?? "")}
            title={t("openInCanvas")}
            type="button"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true">
              <rect
                x="3"
                y="4"
                width="18"
                height="16"
                rx="2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
              />
              <path d="M3 9h18M8 4v5" fill="none" stroke="currentColor" strokeWidth="1.7" />
            </svg>
            {t("canvas")}
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
      <div className={`code-body ${showGutter ? "numbered" : ""}`}>
        {showGutter && (
          <div className="code-gutter" aria-hidden="true">
            <code>{Array.from({ length: lines }, (_, i) => i + 1).join("\n")}</code>
          </div>
        )}
        <pre ref={ref} {...props}>
          {children}
        </pre>
      </div>
    </div>
  );
}

const components = { pre: CodeBlock, sup: SupRenderer } as const;

/** Markdown renderer with GFM tables, KaTeX math, and code highlighting.
 *  When `cites` is given, inline 【N】 markers become hoverable anchors. */
export function Markdown({ children, cites }: { children: string; cites?: CiteSource[] }) {
  const count = cites?.length ?? 0;
  const plugins = useMemo(
    () => (count > 0 ? [...rehypePlugins, [rehypeCites, { count }]] : rehypePlugins),
    [count],
  );
  return (
    <CitesContext.Provider value={cites ?? []}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={plugins as never}
        components={components}
      >
        {children}
      </ReactMarkdown>
    </CitesContext.Provider>
  );
}
