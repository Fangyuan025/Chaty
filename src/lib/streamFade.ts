/**
 * Words coming in while a reply streams: each arrives out of a soft blur, so
 * the newest words of the line still being written read as a gradient — the
 * text settles in rather than snapping onto the page a token at a time.
 *
 * `rehypeStreamFade` wraps the words of the block being written (only that
 * block; see Markdown) in spans that carry where they start in the block's
 * text (`data-o`). Whether a word fades is decided by that offset, not by
 * React's reuse of nodes: React keys a span by its place among its siblings,
 * and when markup closes — `**` becoming bold — the words after it take over
 * nodes that had already faded and would come in with no fade at all. A word
 * at an offset already shown stays as it is; one past it fades in.
 *
 * A frame can bring several words at once (a burst after a pause, a model
 * faster than the screen): `fadeDelay` gives each new word a start no earlier
 * than the one before it, a few milliseconds apart, so the line comes in in
 * order — never more than a fifth of a second behind the text itself.
 */

type HastNode = {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

/** Elements whose text is left as it is: code (a fenced block streams line by
 *  line on its own), math, citation anchors. */
const SKIP = new Set(["pre", "code", "sup", "svg", "math", "annotation", "script", "style"]);

/** One CJK character, or a run of anything else with the spaces after it.
 *  Chinese has no spaces to split at, and one span for a whole sentence would
 *  fade once and then grow without fading. */
const PIECE =
  /[⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]|[^\s⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]+\s*|\s+/g;

/** Split `text` into the pieces that fade on their own. */
export function fadePieces(text: string): string[] {
  return text.match(PIECE) ?? [];
}

function isKatex(node: HastNode): boolean {
  const cls = node.properties?.className;
  return Array.isArray(cls) && cls.some((c) => typeof c === "string" && c.startsWith("katex"));
}

/** Wrap the words under `node`, numbering them by where they start in the
 *  block's visible text; `at` is the offset so far, returned advanced. */
function wrap(node: HastNode, at: number): number {
  const kids = node.children;
  if (!kids) return at;
  const out: HastNode[] = [];
  for (const child of kids) {
    if (child.type === "element") {
      if (!SKIP.has(child.tagName ?? "") && !isKatex(child)) at = wrap(child, at);
      else at += textLength(child);
      out.push(child);
    } else if (child.type === "text" && child.value && /\S/.test(child.value)) {
      for (const piece of fadePieces(child.value)) {
        out.push({ type: "element", tagName: "span", properties: { className: ["sw"], dataO: at }, children: [{ type: "text", value: piece }] });
        at += piece.length;
      }
    } else {
      if (child.type === "text") at += child.value?.length ?? 0;
      out.push(child);
    }
  }
  node.children = out;
  return at;
}

function textLength(node: HastNode): number {
  if (node.type === "text") return node.value?.length ?? 0;
  return (node.children ?? []).reduce((n, c) => n + textLength(c), 0);
}

/** Rehype plugin: the block's words as `<span class="sw" data-o>`. */
export function rehypeStreamFade() {
  return (tree: HastNode) => {
    wrap(tree, 0);
  };
}

/** When the next new word may start fading in. */
let nextAt = -Infinity;
const STEP_MS = 12;
const MAX_LAG_MS = 220;

/** The delay before a word appearing now starts its fade: no earlier than the
 *  word before it plus a step, and no later than a little behind now. */
export function fadeDelay(now: number = performance.now()): number {
  const start = Math.min(Math.max(now, nextAt), now + MAX_LAG_MS);
  nextAt = start + STEP_MS;
  return Math.round(start - now);
}

/** For tests: forget the schedule. */
export function resetFadeSchedule(): void {
  nextAt = -Infinity;
}
