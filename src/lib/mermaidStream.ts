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
const END = /^\s*end\s*$/;

/** `lines` with an `end` for each block still open. */
export function closeBlocks(lines: string[]): string[] {
  let depth = 0;
  for (const l of lines) {
    if (END.test(l)) depth = Math.max(0, depth - 1);
    else if (OPENS.test(l)) depth++;
  }
  return depth ? [...lines, ...Array<string>(depth).fill("end")] : lines;
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
    if (/^[\w-]+$/.test(title) || /^[\w-]+\s*\[.*\]$/.test(title) || /^".*"$/.test(title)) {
      const m = /^([\w-]+\s*)\[(.*)\]$/.exec(title);
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
