import {
  createContext,
  memo,
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
import hljs from "highlight.js/lib/common";
import "katex/dist/katex.min.css";
import { mdBlocks, type BlockCache } from "../lib/mdBlocks";
import { useI18n } from "../lib/i18n";
import { copyToClipboard } from "../lib/clipboard";
import { fitPop } from "../lib/popFit";

const remarkPlugins = [remarkGfm, remarkMath];
// `throwOnError: false` keeps partial LaTeX from crashing mid-stream;
// `ignoreMissing` tolerates unknown code-fence languages.
const rehypePlugins = [
  [rehypeKatex, { throwOnError: false }],
  [rehypeHighlight, { ignoreMissing: true }],
] as const;
/** For a fenced block still being written: its lines are highlighted one by
 *  one as they complete (StreamLinesCode), not the whole block every frame. */
const rehypeStreamingPlugins = [[rehypeKatex, { throwOnError: false }]] as const;

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

/** Settings → Chat: long code blocks collapse to a header (think-panel style). */
export const CodeCollapseContext = createContext(false);
/** True while the surrounding assistant message is still streaming in. */
export const StreamingContext = createContext(false);

const COLLAPSE_MIN_LINES = 14;

const CITE_RE = /【(\d{1,2})】|\[(\d{1,2})\]/g;

/* eslint-disable @typescript-eslint/no-explicit-any */
/** Rehype plugin: split text nodes on 【N】 / [N] into `<sup data-cite="N">`.
 *  Skips code/pre. A 【N】 past the sources cites nothing — a small model
 *  counting past what it was given (issue #14: 【4】【5】 under three sources)
 *  — so it is dropped rather than printed as a marker no chip answers to; an
 *  out-of-range [N] may be ordinary text and stays as it is. */
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
        const known = n >= 1 && n <= count;
        if (!known && m[1] === undefined) continue;
        if (m.index > last) out.push({ type: "text", value: value.slice(last, m.index) });
        if (known)
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
    <sup className="cite" onMouseEnter={(e) => fitPop(e.currentTarget)}>
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
export const STORAGE_SHIM = `<script>(function(){
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
 *  language label, and (optional, Settings → Chat) think-panel-style collapse
 *  for long blocks: while the block streams in it shows a small focus window
 *  pinned to the newest lines; finished blocks fold to a one-line header;
 *  clicking toggles the full code at any time. */
function CodeBlock({ children, ...props }: ComponentPropsWithoutRef<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  const focusRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);
  const [isHtml, setIsHtml] = useState(false);
  const [lines, setLines] = useState(0);
  // null = automatic (focus while streaming, folded when done); user clicks pin it.
  const [override, setOverride] = useState<boolean | null>(null);
  const { t } = useI18n();
  const openCanvas = useContext(CanvasOpenContext);
  const collapseEnabled = useContext(CodeCollapseContext);
  const streaming = useContext(StreamingContext);

  const lang = codeLang(children);

  useEffect(() => {
    const raw = ref.current?.textContent ?? "";
    const text = raw.trimStart().toLowerCase();
    const sniff = text.startsWith("<!doctype") || text.startsWith("<html") || text.startsWith("<svg");
    setIsHtml(lang === "html" || lang === "htm" || (lang === "" && sniff));
    // Live line count (grows while the block streams in).
    setLines(raw ? raw.split("\n").length - (raw.endsWith("\n") ? 1 : 0) : 0);
  });

  const foldable = collapseEnabled && lines >= COLLAPSE_MIN_LINES;
  const expanded = override ?? false;
  // Folded blocks are never fully hidden: while the message streams they show
  // a window pinned to the NEWEST lines (focus-follow); once finished they
  // show the FIRST few lines as a preview. Clicking either expands.
  const focusMode = foldable && !expanded && streaming && override === null;
  useEffect(() => {
    if (!focusRef.current) return;
    if (focusMode) {
      // streaming: pin the window to the newest lines
      focusRef.current.scrollTop = focusRef.current.scrollHeight;
    } else if (foldable && !expanded) {
      // finished preview: always show the HEAD of the block (the focus
      // window leaves its scroll position at the tail otherwise)
      focusRef.current.scrollTop = 0;
    }
  });

  // Mermaid diagrams render in place of the code block.
  if (lang === "mermaid") {
    return <Mermaid code={codeText(children)} />;
  }

  const copy = (e?: { stopPropagation: () => void }) => {
    e?.stopPropagation();
    const text = ref.current?.textContent ?? "";
    void copyToClipboard(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    });
  };

  const actions = (
    <div className="code-actions">
        {isHtml && openCanvas && (
          <button
            className="code-btn"
            onClick={(e) => {
              e.stopPropagation();
              openCanvas(ref.current?.textContent ?? "");
            }}
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
  );

  // One header for every block: the fold switch (or the language) on the
  // left, the actions on the right — where they never cover the code.
  return (
    <div className={`code-block ${foldable ? "foldable" : ""}`}>
      <div className="codeblock-head">
        {foldable ? (
          <button className="code-fold-toggle" type="button" onClick={() => setOverride(!expanded)}>
            <span className={`think-caret ${expanded ? "open" : ""}`}>▶</span>
            <span className="code-fold-lang">{lang || "code"}</span>
            <span className="code-fold-count">
              {lines} {t("codeLines")}
            </span>
          </button>
        ) : (
          <span className="code-lang">{lang || "text"}</span>
        )}
        {actions}
      </div>
      <div
        ref={focusRef}
        className={`code-fold-body ${foldable && !expanded ? (focusMode ? "focus" : "preview") : ""}`}
        onClick={foldable && !expanded ? () => setOverride(true) : undefined}
      >
        <pre ref={ref} {...props}>
          {children}
        </pre>
      </div>
    </div>
  );
}

const components = { pre: CodeBlock, sup: SupRenderer } as const;

/** One line of a code block still being written, highlighted on its own and
 *  never again: the lines above it do not change, so they do not re-render,
 *  and the block's DOM grows at its end instead of being rebuilt — which is
 *  what WebKit could not repaint fast enough. A construct spanning lines (a
 *  block comment, a template string) is coloured line by line until the fence
 *  closes; the block is then highlighted whole, once. */
const CodeLine = memo(function CodeLine({ text, lang, nl }: { text: string; lang: string; nl: boolean }) {
  const html = useMemo(() => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(text, { language: lang, ignoreIllegals: true }).value;
      } catch {
        /* plain below */
      }
    }
    return text.replace(/[&<>]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;"));
  }, [text, lang]);
  return (
    <>
      <span dangerouslySetInnerHTML={{ __html: html }} />
      {nl ? "\n" : null}
    </>
  );
});

/** The `code` of a fenced block still being written (see CodeLine). */
function StreamLinesCode({ className, children }: ComponentPropsWithoutRef<"code">) {
  const text = typeof children === "string" ? children : Array.isArray(children) ? children.join("") : "";
  const lang = /language-([\w+#.-]+)/.exec(className ?? "")?.[1]?.toLowerCase() ?? "";
  const lines = text.split("\n");
  return (
    <code className={`${className ?? ""} hljs`.trim()}>
      {lines.map((l, i) => (
        <CodeLine key={i} text={l} lang={lang} nl={i < lines.length - 1} />
      ))}
    </code>
  );
}

const streamingComponents = { pre: CodeBlock, sup: SupRenderer, code: StreamLinesCode } as const;

/** One top-level block, rendered on its own and re-rendered only when its own
 *  source changes. */
const Block = memo(function Block({
  src,
  plugins,
  openFence,
}: {
  src: string;
  plugins: unknown;
  openFence: boolean;
}) {
  return (
    <ReactMarkdown
      remarkPlugins={remarkPlugins}
      rehypePlugins={(openFence ? rehypeStreamingPlugins : plugins) as never}
      components={openFence ? streamingComponents : components}
    >
      {src}
    </ReactMarkdown>
  );
});

/** Markdown renderer with GFM tables, KaTeX math, and code highlighting.
 *  When `cites` is given, inline 【N】 markers become hoverable anchors.
 *
 *  `blocks`: render block by block (see mdBlocks) — for replies that stream
 *  in, where re-rendering the whole text on every frame is what stalled the
 *  page. Finished text renders identically either way. */
export function Markdown({
  children,
  cites,
  blocks,
}: {
  children: string;
  cites?: CiteSource[];
  blocks?: boolean;
}) {
  const count = cites?.length ?? 0;
  const plugins = useMemo(
    () => (count > 0 ? [...rehypePlugins, [rehypeCites, { count }]] : rehypePlugins),
    [count],
  );
  const cache = useRef<BlockCache | undefined>(undefined);
  const parts = useMemo(() => {
    if (!blocks) return null;
    try {
      const b = mdBlocks(children, cache.current);
      cache.current = { text: children, blocks: b };
      return b;
    } catch {
      cache.current = undefined;
      return null;
    }
  }, [blocks, children]);
  return (
    <CitesContext.Provider value={cites ?? []}>
      {parts ? (
        parts.map((b) => <Block key={b.start} src={b.src} plugins={plugins} openFence={!!b.openFence} />)
      ) : (
        <ReactMarkdown
          remarkPlugins={remarkPlugins}
          rehypePlugins={plugins as never}
          components={components}
        >
          {children}
        </ReactMarkdown>
      )}
    </CitesContext.Provider>
  );
}
