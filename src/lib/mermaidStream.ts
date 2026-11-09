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
