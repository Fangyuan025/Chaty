import {
  createContext,
  memo,
  useContext,
  useEffect,
  useLayoutEffect,
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
import { mdBlocks, type BlockCache } from "../lib/mdBlocks";
import { useI18n } from "../lib/i18n";
import { copyToClipboard } from "../lib/clipboard";
import { fitPop } from "../lib/popFit";
import { fadeTiming, rehypeStreamFade } from "../lib/streamFade";
import { stabilizeTail } from "../lib/streamMarkdown";
import { GrowingHighlight, highlightTyping, splitTyping } from "../lib/highlightLines";
import { diagramCandidates, finishedDiagram } from "../lib/mermaidStream";

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
let mermaidTheme = "";

/** Mermaid, loaded once and set to the theme the page is in. */
async function mermaidIn(dark: boolean) {
  mermaidReady ??= import("mermaid").then((m) => m.default);
  const mermaid = await mermaidReady;
  const theme = dark ? "dark" : "default";
  if (mermaidTheme !== theme) {
    // "loose" — strict rejects common model-generated labels (e.g. with
    // HTML entities) and silently blanked the diagram. We render into our
    // own sandboxed context, and the SVG is inert markup.
    mermaid.initialize({ startOnLoad: false, theme, securityLevel: "loose" });
    mermaidTheme = theme;
  }
  return mermaid;
}

/** One drawing at a time: Mermaid measures its text in the page and is not
 *  safe to run twice at once, and a drawing gone out of date while it waited
 *  (a newer line of its diagram, a diagram scrolled away and unmounted) is
 *  dropped when its turn comes. */
let mermaidQueue: Promise<unknown> = Promise.resolve();
function mermaidTurn(job: () => Promise<void>): Promise<void> {
  const turn = mermaidQueue.then(job, job);
  mermaidQueue = turn.catch(() => {});
  return turn;
}

/** A diagram being written is drawn again at most this often: six times what
 *  its last drawing cost (parsing, drawing, putting it on the page), and a
 *  quarter second at least — a small diagram still grows a line at a time, a
 *  large one, whose drawing takes a frame or three, every few lines, and the
 *  frames between are the reply's. */
let buildingCost = 0;
let buildingDone = 0;
const nextFrame = () => new Promise<void>((r) => setTimeout(r, 0));

/** Diagrams already drawn, by theme and source: a conversation opened again,
 *  or a reply's diagram once the reply is stored, shows at once instead of
 *  being drawn again. */
const mermaidDrawn = new Map<string, string>();
function drawnSvg(key: string): string | undefined {
  const svg = mermaidDrawn.get(key);
  if (svg !== undefined) {
    mermaidDrawn.delete(key);
    mermaidDrawn.set(key, svg);
  }
  return svg;
}
function keepSvg(key: string, svg: string) {
  mermaidDrawn.set(key, svg);
  if (mermaidDrawn.size > 64) mermaidDrawn.delete(mermaidDrawn.keys().next().value as string);
}

/** Whether the page is dark: the theme set, or under "system" the system's.
 *  (Read as "anything but light", a light system got dark diagrams.) */
function pageIsDark(): boolean {
  const theme = document.documentElement.dataset.theme ?? "";
  if (theme.includes("light")) return false;
  if (theme.includes("dark")) return true;
  return !window.matchMedia?.("(prefers-color-scheme: light)").matches;
}

function usePageDark(): boolean {
  const [dark, setDark] = useState(pageIsDark);
  useEffect(() => {
    const update = () => setDark(pageIsDark());
    const mo = new MutationObserver(update);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const mq = window.matchMedia?.("(prefers-color-scheme: light)");
    mq?.addEventListener?.("change", update);
    return () => {
      mo.disconnect();
      mq?.removeEventListener?.("change", update);
    };
  }, []);
  return dark;
}

/** A Mermaid diagram, lazily loaded.
 *
 *  While its fence is still being written (`building`) it is drawn as far as
 *  it goes, once a line (see mermaidStream): the longest start of its
 *  finished lines that parses, open subgraphs closed — so the flowchart is
 *  seen being put together node by node — and a line half-written is never an
 *  error, never a fall back to the code: the last drawing stays until the
 *  next one is ready. Once the fence closes it is drawn whole, and only a
 *  diagram that is wrong as written falls back to its source, with the
 *  reason. It follows the page's theme. */
function Mermaid({ code, building = false }: { code: string; building?: boolean }) {
  const { t } = useI18n();
  const dark = usePageDark();
  const source = building ? finishedDiagram(code) : code.trim();
  const tag = dark ? "d" : "l";
  // What is on screen, by key: a drawing the same as it is not drawn again.
  const shown = useRef("");
  const [svg, setSvg] = useState(() => {
    const hit = !building && source ? drawnSvg(`${tag}\n${source}`) : undefined;
    if (hit !== undefined) shown.current = `${tag}\n${source}`;
    return hit ?? "";
  });
  const [error, setError] = useState("");

  useEffect(() => {
    if (!source) return;
    let alive = true;
    if (!building) {
      const key = `${tag}\n${source}`;
      const hit = drawnSvg(key);
      if (hit !== undefined) {
        if (shown.current !== key) {
          shown.current = key;
          setSvg(hit);
        }
        setError("");
        return;
      }
    }
    void mermaidTurn(async () => {
      if (!alive) return;
      if (building) {
        const wait = buildingDone + Math.max(250, buildingCost * 6) - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        // Lines that came in meanwhile: the newest of them is drawn instead.
        if (!alive) return;
      }
      const began = performance.now();
      try {
        const mermaid = await mermaidIn(dark);
        let text = source;
        if (building) {
          let found: string | null = null;
          for (const candidate of diagramCandidates(source)) {
            try {
              await mermaid.parse(candidate);
              found = candidate;
              break;
            } catch {
              // a line Mermaid does not take; one fewer
            }
            if (!alive) return;
          }
          if (!found) return;
          text = found;
        } else {
          await mermaid.parse(text);
        }
        const key = `${tag}\n${text}`;
        if (!alive) return;
        if (shown.current === key) {
          setError("");
          return;
        }
        let out = drawnSvg(key);
        if (out === undefined) {
          // Parsing and drawing in tasks of their own: a frame between them.
          await nextFrame();
          if (!alive) return;
          out = (await mermaid.render(`mmd-${mermaidSeq++}`, text)).svg;
          if (!building) keepSvg(key, out);
        }
        if (alive) {
          shown.current = key;
          setSvg(out);
          setError("");
        }
        if (building) {
          // What putting it on the page costs is about half the drawing again.
          buildingCost = (performance.now() - began) * 1.5;
          buildingDone = performance.now();
        }
      } catch (e) {
        // Building, a failed drawing keeps the last one; written, it is the
        // diagram's own fault, said under its source.
        if (alive && !building) setError((e as Error)?.message ?? String(e));
      }
    });
    return () => {
      alive = false;
    };
  }, [source, building, dark, tag]);

  // Nothing drawn yet: being written, or drawing for the first time — the
  // source shows only for a diagram that failed as written.
  if (!svg && (building || !error)) {
    return (
      <div className="mermaid-diagram mermaid-pending">
        <span className="mermaid-pending-label">{t("mermaidDrawing")}</span>
      </div>
    );
  }
  if (svg && (building || !error)) {
    return <div className="mermaid-diagram" dangerouslySetInnerHTML={{ __html: svg }} />;
  }
  return (
    <pre className="mermaid-src">
      <code>{code}</code>
      {error ? <div className="mermaid-err">{`${t("mermaidFailed")}: ${error.slice(0, 300)}`}</div> : null}
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
  const fenceOpen = useContext(FenceOpen);

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

  // Mermaid diagrams render in place of the code block — drawn as far as they
  // go while the fence is still open and the reply still coming (see Mermaid).
  // A reply that ended inside the fence ended the diagram there: that is what
  // the model wrote, and it is judged as written.
  if (lang === "mermaid") {
    return <Mermaid code={codeText(children)} building={fenceOpen && streaming} />;
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

/** How far into the block being written its text has been shown (see
 *  streamFade): a word starting before this is not new. One per block. */
const FadeShown = createContext<{ upTo: number } | null>(null);

/** A span of the block being written. A word (`.sw`) fades in when it starts
 *  past what the block has shown, and then never again; its node is keyed by
 *  where it starts, so a component React hands another word gets a fresh node
 *  for it. Any other span (KaTeX's) renders as it is. */
const FadeSpan = memo(FadeSpanInner, (a, b) => {
  // react-markdown hands every element a fresh `node`; the word itself is
  // what matters. Unchanged, a word does not render again as the text grows.
  const ra = a as Record<string, unknown>;
  const rb = b as Record<string, unknown>;
  return a.className === b.className && ra["data-o"] === rb["data-o"] && a.children === b.children && a.style === b.style;
});
function FadeSpanInner({
  node: _node,
  className,
  style,
  children,
  ...rest
}: ComponentPropsWithoutRef<"span"> & { node?: unknown }) {
  const shown = useContext(FadeShown);
  const raw = (rest as Record<string, unknown>)["data-o"];
  const at = raw == null ? -1 : Number(raw);
  const word = className === "sw" && at >= 0 && shown != null;
  const fresh = useMemo(() => word && at >= shown!.upTo, [word, at, shown]);
  const timing = useMemo(() => (fresh ? fadeTiming() : null), [fresh, at]);
  const delay = timing?.delay ?? 0;
  const length = typeof children === "string" ? children.length : 0;
  useLayoutEffect(() => {
    if (word) shown!.upTo = Math.max(shown!.upTo, at + length);
  }, [word, at, length, shown]);
  if (!word) {
    return (
      <span className={className} style={style} {...rest}>
        {children}
      </span>
    );
  }
  delete (rest as Record<string, unknown>)["data-o"];
  return (
    <span
      key={at}
      className={fresh ? (timing?.quick ? "sw quick" : "sw") : undefined}
      style={fresh && delay > 0 ? { ...style, animationDelay: `${delay}ms` } : style}
      {...rest}
    >
      {children}
    </span>
  );
}
const fadeComponents = { ...components, span: FadeSpan } as const;

/** One line of a code block still being written, as markup: re-rendered only
 *  when its markup changes — the lines above the one being written do not,
 *  so the block's DOM grows at its end instead of being rebuilt, which is
 *  what WebKit could not repaint fast enough. */
const CodeLine = memo(function CodeLine({ html, nl }: { html: string; nl: boolean }) {
  return (
    <>
      <span dangerouslySetInnerHTML={{ __html: html }} />
      {nl ? "\n" : null}
    </>
  );
});

/** The `code` of a fenced block still being written: its finished lines
 *  highlighted together — once a line while that is cheap, less often for a
 *  long page (GrowingHighlight) — and the line being typed as they leave it,
 *  so an HTML page's script and style, or a comment over several lines, are
 *  coloured as they are while they stream (see highlightLines). */
function StreamLinesCode({ className, children }: ComponentPropsWithoutRef<"code">) {
  const text = typeof children === "string" ? children : Array.isArray(children) ? children.join("") : "";
  const lang = /language-([\w+#.-]+)/.exec(className ?? "")?.[1]?.toLowerCase() ?? "";
  const { finished, typing, trailing } = splitTyping(text);
  const growing = useRef<GrowingHighlight | null>(null);
  if (growing.current?.lang !== lang) growing.current = new GrowingHighlight(lang);
  const g = growing.current;
  const done = useMemo(() => g.update(finished), [g, finished]);
  const typingHtml = useMemo(() => highlightTyping(typing, done), [typing, done]);
  return (
    <code className={`${className ?? ""} hljs`.trim()}>
      {done.lines.map((html, i) => (
        <CodeLine key={i} html={html} nl />
      ))}
      <CodeLine key={done.lines.length} html={typingHtml} nl={trailing} />
    </code>
  );
}

const streamingComponents = { pre: CodeBlock, sup: SupRenderer, code: StreamLinesCode } as const;
const fadeStreamingComponents = { ...streamingComponents, span: FadeSpan } as const;

/** The block being written, as it is shown while it comes in: a fence whose
 *  opening line is not finished waits for it (`\`\`\`mer` was a code block of
 *  the language "mer" until "mermaid" came, and a diagram's place showed an
 *  empty code box first); any other block is held back where its markdown has
 *  not yet decided what it is (streamMarkdown). */
function liveSrc(b: { src: string; openFence?: unknown }): string {
  return b.openFence ? (b.src.includes("\n") ? b.src : "") : stabilizeTail(b.src);
}

/** The fenced block being rendered is still being written (no closing fence
 *  yet): a diagram's source is not parsed until it is whole. */
const FenceOpen = createContext(false);

/** One top-level block, rendered on its own and re-rendered only when its own
 *  source changes. */
const Block = memo(function Block({
  src,
  plugins,
  openFence,
  fade,
}: {
  src: string;
  plugins: unknown;
  openFence: boolean;
  /** The block being written while the reply streams: its words fade in. */
  fade?: boolean;
}) {
  const base = (openFence ? rehypeStreamingPlugins : plugins) as unknown[];
  const shown = useRef({ upTo: 0 }).current;
  return (
    <FenceOpen.Provider value={openFence}>
    <FadeShown.Provider value={fade ? shown : null}>
    <ReactMarkdown
      remarkPlugins={remarkPlugins}
      rehypePlugins={(fade ? [...base, rehypeStreamFade] : base) as never}
      components={
        openFence
          ? fade
            ? fadeStreamingComponents
            : streamingComponents
          : fade
            ? fadeComponents
            : components
      }
    >
      {src}
    </ReactMarkdown>
    </FadeShown.Provider>
    </FenceOpen.Provider>
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
  streaming,
}: {
  children: string;
  cites?: CiteSource[];
  blocks?: boolean;
  /** The text is still coming in: the block being written fades its words in
   *  (see streamFade). Defaults to the surrounding StreamingContext. */
  streaming?: boolean;
}) {
  const inStream = useContext(StreamingContext);
  const isStreaming = streaming ?? inStream;
  const live = isStreaming && !!blocks;
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
    <StreamingContext.Provider value={isStreaming}>
    <CitesContext.Provider value={cites ?? []}>
      {parts ? (
        parts.map((b, i) => (
          <Block
            key={b.start}
            // The block being written, held back where its markdown has not
            // yet decided what it is (see streamMarkdown).
            src={live && i === parts.length - 1 ? liveSrc(b) : b.src}
            plugins={plugins}
            openFence={!!b.openFence}
            fade={live && i === parts.length - 1}
          />
        ))
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
    </StreamingContext.Provider>
  );
}
