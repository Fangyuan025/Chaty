/**
 * A code block being written, highlighted as the whole it is so far — but
 * shown a line at a time.
 *
 * Each line used to be highlighted on its own, so that the lines above the
 * one being written never changed and never re-rendered (WebKit could not
 * repaint a block rebuilt every frame). Alone, though, a line does not know
 * where it is: a line of an HTML page's <script> or <style> is not HTML, and
 * the inside of a comment or a string that spans lines is not code — so an
 * HTML block showed nearly plain until its fence closed.
 *
 * Now the finished lines are highlighted together, once each time a line is
 * finished, and the result is split back into lines: a span still open at a
 * line's end is closed there and opened again on the next, so each line is
 * whole markup on its own, and a line that did not change is the same string
 * as before (and is not re-rendered). The line still being typed is
 * highlighted as what the finished lines leave it in: the script's or the
 * style's own language, or inside the comment or string still open.
 */
import hljs from "highlight.js/lib/common";

const escapeHtml = (s: string) => s.replace(/[&<>]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;"));

/** highlight.js's markup split at its newlines, each line closing the spans
 *  still open at its end and opening them again at its start; `open` is
 *  what is still open after the last line. */
export function splitHighlighted(html: string): { lines: string[]; open: string[] } {
  const lines: string[] = [];
  const stack: string[] = [];
  let line = "";
  const token = /<span[^>]*>|<\/span>|\n|[^<\n]+|</g;
  for (let m = token.exec(html); m; m = token.exec(html)) {
    const t = m[0];
    if (t === "\n") {
      lines.push(line + "</span>".repeat(stack.length));
      line = stack.join("");
    } else if (t === "</span>") {
      stack.pop();
      line += t;
    } else {
      if (t.startsWith("<span")) stack.push(t);
      line += t;
    }
  }
  lines.push(line + "</span>".repeat(stack.length));
  return { lines, open: [...stack] };
}

/** highlight.js's class for a scope (its own scopeToCSSClass). */
function scopeClass(scope: string): string {
  if (scope.startsWith("language:")) return scope.replace("language:", "language-");
  if (scope.includes(".")) {
    const [first, ...rest] = scope.split(".");
    return [`hljs-${first}`, ...rest.map((x, i) => `${x}${"_".repeat(i + 1)}`)].join(" ");
  }
  return `hljs-${scope}`;
}

type Mode = { scope?: unknown; subLanguage?: string | string[]; parent?: Mode };

/** The finished lines of a block, highlighted together. */
export interface FinishedLines {
  /** One line of markup per finished line. */
  lines: string[];
  /** What the last one ends inside — a comment, a string, a tag — as the
   *  spans that open it. (highlight.js closes everything at the end of its
   *  input, so this is read from the mode it ends in, not its markup.) */
  context: string[];
  /** The language the last one ends in when it is not the block's own: an
   *  HTML page's script or style. */
  sub: string | null;
  /** The block's language, when highlight.js knows it. */
  lang: string | null;
}

/** `text`: the finished lines, joined; null when there are none yet. */
export function highlightFinished(text: string | null, lang: string): FinishedLines {
  const known = lang && hljs.getLanguage(lang) ? lang : null;
  if (text === null) return { lines: [], context: [], sub: null, lang: known };
  if (!known) return { lines: text.split("\n").map(escapeHtml), context: [], sub: null, lang: null };
  try {
    const r = hljs.highlight(text, { language: known, ignoreIllegals: true });
    const { lines } = splitHighlighted(r.value);
    const top = (r as unknown as { _top?: Mode })._top;
    // Where the text ends. Inside a <script> or <style>, the mode highlight.js
    // is in hands what follows to another language; inside a comment or a
    // string, the modes up to it are the spans the next line is in.
    const s = top?.subLanguage;
    const sub = (Array.isArray(s) ? s : s ? [s] : []).find((x) => !!hljs.getLanguage(x)) ?? null;
    const context: string[] = [];
    for (let m = top; m?.parent; m = m.parent) {
      if (typeof m.scope === "string") context.unshift(`<span class="${scopeClass(m.scope)}">`);
    }
    return { lines, context, sub, lang: known };
  } catch {
    return { lines: text.split("\n").map(escapeHtml), context: [], sub: null, lang: known };
  }
}

/** The line being typed, as the finished lines leave it. */
export function highlightTyping(line: string, before: FinishedLines): string {
  if (!line) return "";
  // In a comment or a string still open: plain, inside it. Otherwise in the
  // language the finished lines end in.
  const as = before.sub ?? (before.context.length ? null : before.lang);
  let inner = escapeHtml(line);
  if (as) {
    try {
      inner = hljs.highlight(line, { language: as, ignoreIllegals: true }).value;
    } catch {
      /* plain */
    }
  }
  return before.context.join("") + inner + "</span>".repeat(before.context.length);
}

/** A fenced block's code as it reaches the renderer while it is still being
 *  written: what is finished (whole lines) and the line being typed. The
 *  newline markdown adds at the end of a block's code is not one the model
 *  wrote. */
export function splitTyping(code: string): { finished: string | null; typing: string; trailing: boolean } {
  const trailing = code.endsWith("\n");
  const body = trailing ? code.slice(0, -1) : code;
  const cut = body.lastIndexOf("\n");
  return { finished: cut >= 0 ? body.slice(0, cut) : null, typing: body.slice(cut + 1), trailing };
}

/**
 * The finished lines of a block that keeps growing, highlighted whole each
 * time a line is finished — while that is cheap. A long page costs a whole
 * highlight a dozen milliseconds and more (JavaScriptCore: 21 KB 5 ms, 62 KB
 * 12 ms, 124 KB 24 ms), which every finished line would turn into a hitch;
 * then the whole is highlighted again at most every eight times what it
 * cost (a fifth of a second at least), and the lines finished meanwhile are
 * highlighted in the place the last whole one left off — the script or style
 * they are in — until the next whole highlight sets them right.
 */
export class GrowingHighlight {
  private last: { text: string; result: FinishedLines; at: number; cost: number } | null = null;

  constructor(
    readonly lang: string,
    private readonly now: () => number = () => performance.now(),
  ) {}

  update(text: string | null): FinishedLines {
    if (text === null) return highlightFinished(null, this.lang);
    const last = this.last;
    if (last && last.text === text) return last.result;
    const t = this.now();
    if (
      last &&
      last.cost >= 4 &&
      text.startsWith(last.text + "\n") &&
      t - last.at < Math.max(200, last.cost * 8)
    ) {
      const added = text.slice(last.text.length + 1).split("\n");
      return { ...last.result, lines: [...last.result.lines, ...added.map((l) => highlightTyping(l, last.result))] };
    }
    const result = highlightFinished(text, this.lang);
    const end = this.now();
    this.last = { text, result, at: end, cost: end - t };
    return result;
  }
}
