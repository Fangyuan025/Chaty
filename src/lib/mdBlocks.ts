/** A markdown text as its top-level blocks — paragraph, list, table, fenced
 *  code… — parsed exactly as react-markdown parses it, so each block can be
 *  rendered, and memoised, on its own.
 *
 *  Why: a streaming reply was re-rendered whole on every animation frame — the
 *  whole text re-parsed, every code block in it re-highlighted, thousands of
 *  spans reconciled — and the cost grew with the reply. Past a few hundred
 *  lines of code a frame took longer than the tokens arriving, the page never
 *  got to repaint, and WebKit showed the invalidated code block as blank until
 *  the stream ended (owner report: at a fast model's pace, an unfolded code
 *  block vanished and flickered, and came back only when generation finished).
 *  With blocks, only the one being written re-renders.
 *
 *  Parsing is incremental: a text that grows by appending re-parses from the
 *  block before the last — appending can still change the last two (a line of
 *  `===` turns a paragraph into a heading, an indented line continues a list
 *  across a blank line), never what came before. */
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

export interface MdBlock {
  /** Offset of the block's first line in the text; stable while it streams,
   *  which makes it the block's key. */
  start: number;
  /** Its source, trailing blank lines included. */
  src: string;
  /** A fenced code block whose closing fence has not been written yet. */
  openFence?: { lang: string };
}

const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath);

type Node = { type: string; lang?: string | null; position?: { start: { offset?: number } } };

const lineStart = (text: string, at: number) => text.lastIndexOf("\n", at - 1) + 1;

/** Blocks of `text[from..]`, with offsets in `text`. Null when the text holds
 *  something that ties blocks together — a link reference definition or a
 *  footnote, which resolve across the whole document. */
function parseFrom(text: string, from: number): MdBlock[] | null {
  const tree = parser.parse(text.slice(from)) as unknown as { children: Node[] };
  const starts: { at: number; node: Node }[] = [];
  for (const node of tree.children) {
    if (node.type === "definition" || node.type === "footnoteDefinition") return null;
    const off = node.position?.start.offset;
    if (off === undefined) continue;
    const at = lineStart(text, from + off);
    // Two nodes on one line (rare) stay one block.
    if (starts.length && starts[starts.length - 1].at === at) continue;
    starts.push({ at, node });
  }
  return starts.map(({ at, node }, i) => {
    const end = i + 1 < starts.length ? starts[i + 1].at : text.length;
    const src = text.slice(at, end);
    const block: MdBlock = { start: at, src };
    if (node.type === "code") {
      const open = /^ {0,3}(`{3,}|~{3,})/.exec(src);
      if (open && !fenceClosed(src, open[1])) block.openFence = { lang: (node.lang ?? "").toLowerCase() };
    }
    return block;
  });
}

/** Does a fenced block's source end with a fence that closes it? */
function fenceClosed(src: string, opener: string): boolean {
  const lines = src.replace(/\s+$/, "").split("\n");
  if (lines.length < 2) return false;
  const last = lines[lines.length - 1].trim();
  return last.length >= opener.length && [...last].every((c) => c === opener[0]);
}

export interface BlockCache {
  text: string;
  blocks: MdBlock[] | null;
}

/** The blocks of `text`, re-using what an earlier parse of a prefix of it
 *  found. Null: render the text whole (see parseFrom). */
export function mdBlocks(text: string, cache?: BlockCache): MdBlock[] | null {
  // Footnote markers resolve against definitions that may not have streamed
  // in yet; a text with any is rendered whole from the start.
  if (/\[\^[^\]\n]+\]/.test(text)) return null;
  const prev = cache?.blocks;
  // A fence still being written stays one until a line that could close it
  // arrives; nothing before it can change meanwhile. The common case while
  // code streams, and the one where re-parsing the block on every frame cost
  // the most.
  const open = prev?.[prev.length - 1];
  if (cache && prev && open?.openFence && text.startsWith(cache.text)) {
    const opener = text.indexOf("\n", open.start);
    if (opener !== -1) {
      const from = Math.max(cache.text.lastIndexOf("\n") + 1, opener + 1);
      if (!/^ {0,3}(?:`{3,}|~{3,})[ \t]*$/m.test(text.slice(from))) {
        return [...prev.slice(0, -1), { ...open, src: text.slice(open.start) }];
      }
    }
  }
  if (cache && prev && prev.length >= 2 && text.startsWith(cache.text)) {
    const from = prev[prev.length - 2].start;
    const tail = parseFrom(text, from);
    return tail && [...prev.slice(0, -2), ...tail];
  }
  return parseFrom(text, 0);
}
