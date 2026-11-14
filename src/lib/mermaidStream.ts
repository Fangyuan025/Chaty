/**
 * A Mermaid diagram being written, as it can be drawn so far.
 *
 * It is drawn once a line, not once a token: the line being typed is left out
 * (the code of a fence as it reaches the renderer ends with a newline markdown
 * adds, so "up to the last newline" was the whole text, and every token drew
 * the diagram again — four parses and a render each, which a large diagram
 * could not keep up with). A subgraph, or a sequence diagram's loop or alt,
 * that is not closed yet is closed for the drawing, so what is inside it is
 * drawn as it comes instead of the diagram standing still until its `end`.
 */
import { splitTyping } from "./highlightLines";

/** The finished lines of a diagram's code while its fence is open. */
export function finishedDiagram(code: string): string {
  return (splitTyping(code).finished ?? "").trim();
}

const OPENS = /^\s*(subgraph|loop|alt|opt|par|critical|break|rect|box)\b/;
/** A flowchart's only block is a subgraph: a node called "loop" or "box" at
 *  the start of a line opens nothing. */
const FLOW_OPENS = /^\s*subgraph\b/;
const END = /^\s*end\s*;?\s*$/;
const COMMENT = /^\s*%%(?!\{)/;

/** What opens a block in the diagram these lines are: its first real line
 *  says which kind it is. */
function opensOf(lines: string[]): RegExp {
  const head = lines.find((l) => l.trim() !== "" && !COMMENT.test(l)) ?? "";
  return /^\s*(?:graph|flowchart)\b/i.test(head) ? FLOW_OPENS : OPENS;
}

const indentOf = (l: string) => {
  let n = 0;
  for (const c of l) {
    if (c === " ") n++;
    else if (c === "\t") n += 4;
    else break;
  }
  return n;
};

/**
 * Lines with their blocks balanced: an `end` that closes nothing taken out,
 * and an `end` put in for each block left open — where the indentation says
 * it ends, when the block's lines are indented deeper than its opening line
 * and a later line comes back out to that depth (a model that forgot one
 * `end` meant the lines it wrote after the block to be outside it); else at
 * the end.
 */
function balanceBlocks<T>(lines: T[], textOf: (l: T) => string, make: (text: string) => T, opens: RegExp): { out: T[]; dropped: T[] } {
  const out: T[] = [];
  const dropped: T[] = [];
  const stack: { indent: number; deeper: boolean | null }[] = [];
  lines.forEach((l, i) => {
    const text = textOf(l);
    if (text.trim() === "") {
      out.push(l);
      return;
    }
    const ind = indentOf(text);
    const isEnd = END.test(text);
    for (;;) {
      const top = stack[stack.length - 1];
      if (!top) break;
      if (top.deeper === null) {
        // The block's first line: deeper than its opener, or not indented.
        top.deeper = ind > top.indent;
        break;
      }
      // Back out to the opener's depth (or past it, for an `end` that closes
      // a block further out) without an `end`: the block ended here.
      if (top.deeper && (isEnd ? ind < top.indent && stack.length > 1 : ind <= top.indent)) {
        out.push(make(`${" ".repeat(top.indent)}end`));
        stack.pop();
        continue;
      }
      break;
    }
    if (isEnd) {
      if (!stack.length) {
        dropped.push(l);
        return;
      }
      stack.pop();
    } else if (i > 0 && opens.test(text)) stack.push({ indent: ind, deeper: null });
    out.push(l);
  });
  while (stack.length) out.push(make(`${" ".repeat(stack.pop()!.indent)}end`));
  return { out, dropped };
}

/** `lines` with an `end` for each block still open (where the indentation
 *  says it ends), and none that closes nothing. */
export function closeBlocks(lines: string[]): string[] {
  return balanceBlocks(lines, (l) => l, (t) => t, opensOf(lines)).out;
}

/** A diagram drawn without the lines that stop it, and those lines. */
export interface Salvaged {
  text: string;
  /** The lines left out, as written. */
  skipped: string[];
}

/** The line a Mermaid parse error names (1-based), if it names one. */
export function errorLine(message: string): number | null {
  const m = /\bline (\d+)/i.exec(message);
  return m ? Number(m[1]) : null;
}

/**
 * As much of a diagram as can be drawn, when it cannot be drawn whole.
 *
 * A diagram a model wrote is right almost everywhere and wrong in a line or
 * two — a subgraph never closed, an arrow with nothing after it, a label
 * Mermaid cannot read. Wrong anywhere, Mermaid draws none of it; so the
 * lines it stops at are left out, one at a time, until the rest draws: an
 * `end` that closes nothing goes, a block still open is closed, and a line a
 * parse error names is left out (or the one before it, when that gets
 * further — an error is often only noticed on the line after the mistake).
 * Mermaid takes comments, leading blank lines and front matter off before
 * it numbers lines, so they are taken off here too: the line an error names
 * is the line it means.
 *
 * `parse` is Mermaid's; null when not even the diagram's first line draws.
 */
export async function salvageDiagram(
  code: string,
  parse: (text: string) => Promise<unknown>,
  budget = 48,
): Promise<Salvaged | null> {
  const all = code.split("\n");
  // Front matter is kept whole and is not counted (Mermaid numbers after it).
  let start = 0;
  if (/^---\s*$/.test(all[0] ?? "")) {
    const close = all.findIndex((l, i) => i > 0 && /^---\s*$/.test(l));
    if (close > 0) start = close + 1;
  }
  const front = all.slice(0, start);
  const body = all.slice(start).filter((l) => !COMMENT.test(l));
  while (body.length && body[0].trim() === "") body.shift();
  if (!body.length) return null;
  const opens = opensOf(body);
  const skipped = new Set<number>();

  /** The text with what is skipped left out and its blocks balanced, and
   *  for each of its lines the body line it is (-1: an `end` put in). */
  const build = () =>
    balanceBlocks(
      body.map((text, at) => ({ text, at })).filter((l) => !skipped.has(l.at)),
      (l) => l.text,
      (text) => ({ text, at: -1 }),
      opens,
    ).out;
  const textOf = (lines: { text: string }[]) => [...front, ...lines.map((l) => l.text)].join("\n");
  /** Where a try stops: the body line the error names, or Infinity drawn. */
  const tryParse = async (lines: { text: string; at: number }[]): Promise<{ ok: true } | { ok: false; at: number | null }> => {
    try {
      await parse(textOf(lines));
      return { ok: true };
    } catch (e) {
      const n = errorLine((e as Error)?.message ?? String(e));
      const hit = n != null ? lines[n - 1] : undefined;
      return { ok: false, at: hit && hit.at >= 0 ? hit.at : null };
    }
  };
  /** The last line still in, not the first (which names the diagram). */
  const lastKept = () => {
    for (let i = body.length - 1; i > 0; i--) if (!skipped.has(i) && body[i].trim() !== "") return i;
    return -1;
  };
  const done = (lines: { text: string; at: number }[]): Salvaged => ({
    text: textOf(lines),
    skipped: body.filter((l, i) => (skipped.has(i) || (END.test(l) && !lines.some((x) => x.at === i))) && l.trim() !== ""),
  });

  let lines = build();
  let res = await tryParse(lines);
  for (let n = 0; n < budget && !res.ok; n++) {
    if (res.at == null || res.at === 0) {
      // No line named, or the first: the last line in goes.
      const at = lastKept();
      if (at < 0) return null;
      skipped.add(at);
    } else {
      // The named line, or the one before it if leaving that out gets
      // further (or draws).
      const named = res.at;
      skipped.add(named);
      const a = build();
      const ra = await tryParse(a);
      if (ra.ok) return done(a);
      let prev = named - 1;
      while (prev > 0 && (skipped.has(prev) || body[prev].trim() === "")) prev--;
      if (prev > 0) {
        skipped.delete(named);
        skipped.add(prev);
        const b = build();
        const rb = await tryParse(b);
        if (rb.ok) return done(b);
        const further = (rb.at ?? -1) > (ra.at ?? -1);
        if (!further) {
          skipped.delete(prev);
          skipped.add(named);
          lines = a;
          res = ra;
          continue;
        }
        lines = b;
        res = rb;
        continue;
      }
      lines = a;
      res = ra;
      continue;
    }
    lines = build();
    res = await tryParse(lines);
  }
  return res.ok ? done(lines) : null;
}

/** The longest start of a diagram that draws, its open blocks closed and its
 *  spelling put right where that is what stops it — what a diagram being
 *  written shows. Null when none does. */
export async function longestDrawableStart(
  code: string,
  parse: (text: string) => Promise<unknown>,
): Promise<{ text: string; lines: number } | null> {
  const lines = code.split("\n");
  for (let n = lines.length; n >= 1; n--) {
    const candidate = closeBlocks(lines.slice(0, n)).join("\n");
    const fixed = repairDiagram(candidate);
    for (const attempt of fixed === candidate ? [candidate] : [candidate, fixed]) {
      try {
        await parse(attempt);
        return { text: attempt, lines: n };
      } catch {
        // not this one
      }
    }
  }
  return null;
}

/** What to try drawing, best first: all the finished lines, then one fewer at
 *  a time (a finished line can still be one Mermaid does not take), each with
 *  its open blocks closed. */
export function diagramCandidates(finished: string): string[] {
  const lines = finished.split("\n");
  const out: string[] = [];
  for (let n = lines.length; n >= 1 && n > lines.length - 4; n--) {
    out.push(closeBlocks(lines.slice(0, n)).join("\n"));
  }
  return out;
}

/**
 * A flowchart as models tend to write it, made to parse.
 *
 * What stops a diagram is almost always a label: `A[call foo(bar)]`,
 * `B{price > 5?}`, `C[say "hi"]`, `D["a \"quoted\" word"]` — brackets,
 * quotes and Mermaid's own punctuation inside text that was not quoted, or
 * was quoted the way another language quotes. The diagram is right; its
 * spelling is not. So a diagram that fails to parse is tried again with its
 * labels quoted and what is inside them escaped Mermaid's way (`#quot;`),
 * a subgraph's spaced title quoted, a literal "\n" made a line break and an
 * arrow typed as a character (→, —>) made `-->`. The text of every label is
 * kept as written. Used only when the diagram as written does not parse, and
 * only if the result does.
 */
export function repairDiagram(code: string): string {
  const lines = code.split("\n");
  const head = lines.findIndex((l) => l.trim() !== "" && !l.trim().startsWith("%%"));
  const isFlow = head >= 0 && /^\s*(?:graph|flowchart)\b/i.test(lines[head]);
  return lines
    .map((line, i) => {
      if (i <= head || line.trim().startsWith("%%")) return line;
      const arrows = line.replace(/[—–]+>/g, "-->").replace(/\s→\s/g, " --> ");
      return isFlow ? repairFlowLine(arrows) : arrows;
    })
    .join("\n");
}

/** Text that cannot stand unquoted in a label. */
const RISKY = /[()[\]{}"'`:;#|<>]|\\n|^\s*(?:end|subgraph|graph|style|class|click)\s*$/i;

/** A label's text, quoted for Mermaid: its own quotes as entities, "#" too
 *  (it opens one), a literal "\n" as a line break. */
function quoteLabel(text: string): string {
  let inner = text.trim();
  if (inner.length >= 2 && inner.startsWith('"') && inner.endsWith('"')) inner = inner.slice(1, -1);
  inner = inner
    .replace(/\\"/g, '"')
    .replace(/#(?![\w]+;)/g, "#35;")
    .replace(/"/g, "#quot;")
    .replace(/\\n/g, "<br/>");
  return `"${inner}"`;
}

/** Whether a label needs the repair: unquoted with risky text, or quoted
 *  with quotes of its own inside. */
function labelNeedsQuoting(text: string): boolean {
  const t = text.trim();
  if (t.length >= 2 && t.startsWith('"') && t.endsWith('"')) return /"|\\n/.test(t.slice(1, -1));
  return RISKY.test(t);
}

/** Node shapes, longest opener first, with the closer each takes. */
const SHAPES: [string, string][] = [
  ["(((", ")))"],
  ["((", "))"],
  ["([", "])"],
  ["[[", "]]"],
  ["[(", ")]"],
  ["{{", "}}"],
  ["[", "]"],
  ["(", ")"],
  ["{", "}"],
];

/** What may follow a node: the line's end, a link, another node (&), a class. */
const AFTER_NODE = /^\s*(?:$|;|&|:::|--|==|-\.|<--|<==|~~~|\|)/;
const ID_CHAR = /[\wÀ-￿]/;

function repairFlowLine(line: string): string {
  const t = line.trim();
  if (/^(?:direction\s|classDef\s|class\s|style\s|linkStyle\s|click\s)/.test(t) || /^end;?$/.test(t)) return line;
  // A subgraph's title: `subgraph Input stage` reads as an id followed by junk.
  const sub = /^(\s*subgraph\s+)(.+?)\s*$/.exec(line);
  if (sub) {
    const title = sub[2];
    // An id in any script (`subgraph 上升期 [Plateau Phase]`): read as one
    // spaced title, the brackets were drawn as part of it.
    if (/^[\p{L}\p{N}_-]+$/u.test(title) || /^[\p{L}\p{N}_-]+\s*\[.*\]$/u.test(title) || /^".*"$/.test(title)) {
      const m = /^([\p{L}\p{N}_-]+\s*)\[(.*)\]$/u.exec(title);
      return m && labelNeedsQuoting(m[2]) ? `${sub[1]}${m[1]}[${quoteLabel(m[2])}]` : line;
    }
    return `${sub[1]}${quoteLabel(title)}`;
  }
  // A node called "end": the word closes a subgraph, and as an id it stops
  // the parse. Mermaid's own advice is a capital letter.
  line = line.replace(/(^|[\s>|&;-])end(?=\s*(?:[[({]|$|;|&|--|==|-\.))/g, "$1End");
  // Link text between bars: -->|text|
  let s = line.replace(/(-->|---|==>|-\.->|<-->|--o|--x|===|-\.-)(\s*)\|([^|]*)\|/g, (m, link: string, gap: string, text: string) =>
    labelNeedsQuoting(text) ? `${link}${gap}|${quoteLabel(text)}|` : m,
  );
  // Node labels.
  let out = "";
  let i = 0;
  while (i < s.length) {
    const shape = i > 0 && ID_CHAR.test(s[i - 1]) ? SHAPES.find(([open]) => s.startsWith(open, i)) : undefined;
    if (!shape) {
      out += s[i++];
      continue;
    }
    const [open, close] = shape;
    // The nearest closer after which the line goes on as a line does.
    let end = -1;
    for (let j = s.indexOf(close, i + open.length); j !== -1; j = s.indexOf(close, j + 1)) {
      if (AFTER_NODE.test(s.slice(j + close.length))) {
        end = j;
        break;
      }
    }
    if (end === -1) {
      out += s[i++];
      continue;
    }
    const text = s.slice(i + open.length, end);
    out += open + (labelNeedsQuoting(text) ? quoteLabel(text) : text) + close;
    i = end + close.length;
  }
  s = out;
  return s;
}
