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
 * faster than the screen): `fadeTiming` staggers them a few milliseconds
 * apart so the line comes in in order, and past a hundred-odd words a second
 * lets each in quickly instead of queueing them.
 */

type HastNode = {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

/** Elements whose text is left as it is: code (a fenced block streams line by
 *  line on its own), math, citation anchors, tables. */
const SKIP = new Set(["pre", "code", "sup", "svg", "math", "annotation", "script", "style", "table"]);
/** Where words are plain text — the only words that fade. Text anywhere else
 *  (a table's cells, a diagram, a formula) streams as it always has. */
const PROSE = new Set(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "dd", "dt"]);

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
 *  block's visible text; `at` is the offset so far, returned advanced. Only
 *  text within prose (`inProse`) is wrapped; the rest is counted. */
function wrap(node: HastNode, at: number, inProse = false): number {
  const kids = node.children;
  if (!kids) return at;
  const out: HastNode[] = [];
  for (const child of kids) {
    if (child.type === "element") {
      if (!SKIP.has(child.tagName ?? "") && !isKatex(child)) at = wrap(child, at, inProse || PROSE.has(child.tagName ?? ""));
      else at += textLength(child);
      out.push(child);
    } else if (inProse && child.type === "text" && child.value && /\S/.test(child.value)) {
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

/** When the next new word may start fading in, and when the last words came. */
let nextAt = -Infinity;
const recent: number[] = [];
const STEP_MS = 8;
const MAX_LAG_MS = 60;
/** This many words inside FAST_MS is a stream faster than a fade can follow. */
const FAST_COUNT = 16;
const FAST_MS = 120;

/** How a word appearing now comes in: after what delay, and whether quickly.
 *  At a model's usual pace each word starts no earlier than the one before it
 *  plus a step, and no later than a little behind now — the line comes in in
 *  order. A stream faster than that (well over a hundred words a second) had
 *  its words queue up behind the schedule, a stretch of the newest text
 *  invisible and blurred at once; there each word fades in at once, quickly
 *  and without the blur (`.sw.quick`). */
export function fadeTiming(now: number = performance.now()): { delay: number; quick: boolean } {
  recent.push(now);
  if (recent.length > FAST_COUNT) recent.shift();
  if (recent.length === FAST_COUNT && now - recent[0] < FAST_MS) {
    nextAt = now;
    return { delay: 0, quick: true };
  }
  const start = Math.min(Math.max(now, nextAt), now + MAX_LAG_MS);
  nextAt = start + STEP_MS;
  return { delay: Math.round(start - now), quick: false };
}

/** For tests: forget the schedule. */
export function resetFadeSchedule(): void {
  nextAt = -Infinity;
  recent.length = 0;
}
