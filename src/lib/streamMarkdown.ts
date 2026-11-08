/**
 * The block being written, as it may be shown while it is still coming in.
 *
 * Markdown read a character at a time changes its mind. A paragraph followed
 * by a line that so far holds one "-" is a heading — the paragraph jumps to a
 * heading's size, with the rule under it — until the next character makes the
 * "-" a list item and it jumps back. "---" being typed is a rule for a moment
 * before the words after it; a table's header row is a line of pipes until
 * its delimiter row comes; "**bold" shows its asterisks until they close.
 * Each of those is a flash in the reply as it streams.
 *
 * `stabilizeTail` holds back what has not yet decided what it is: a last line
 * made only of markdown's marks (a fence's first backticks, display math's
 * first dollar among them), a
 * numbered item's number with nothing after
 * it, a table without its delimiter row, display math not yet closed. It
 * closes what is open — bold, strikethrough, inline code — so a phrase is
 * bold from its first word, not after its last. It changes only the end of
 * the text, and only while it streams: the finished reply is rendered from
 * what the model wrote.
 */

/** A line of nothing but marks — a rule, a setext underline, a heading's or
 *  list's or quote's start with no words yet. */
const MARKS_ONLY = /^\s{0,3}(?:[-=*_+#>`~$]\s*)+$/;
/** A numbered item's number, its words not come yet. */
const NUMBER_ONLY = /^\s{0,3}\d{1,9}[.)]?\s*$/;

export function stabilizeTail(src: string): string {
  // A fenced block is code, written or being written: its lines are not
  // markdown's marks, and its backticks are not inline code. (Its closing
  // fence read as a line of marks and was dropped, and its opening backticks
  // were "closed" with one more — a diagram failed on a stray backtick.)
  if (/^ {0,3}(?:`{3,}|~{3,})/.test(src)) return src;
  let s = src;
  // A table before its delimiter row is a paragraph of pipes: wait for it.
  // (A second line with anything but pipes, colons and dashes says it is not
  // one; one still being typed may yet be the delimiter.)
  const lines = s.split("\n");
  if (/^\s{0,3}\|/.test(lines[0])) {
    const second = lines[1];
    const maybeDelimiter = second !== undefined && !/[^|:\-\s]/.test(second);
    if (second === undefined || (maybeDelimiter && lines.length < 3)) return "";
  }
  // The last line, while it has not said what it is.
  const cut = s.lastIndexOf("\n");
  const tail = s.slice(cut + 1);
  if (tail && (MARKS_ONLY.test(tail) || NUMBER_ONLY.test(tail))) s = s.slice(0, Math.max(0, cut));
  // Display math not yet closed: its source would show, then the formula.
  const dollars = s.split("$$").length - 1;
  if (dollars % 2 === 1) s = s.slice(0, s.lastIndexOf("$$"));
  // Inline code, bold and strikethrough left open.
  s = closeOpen(s);
  return s;
}

/** Close an open `code`, **bold** or ~~strike~~ at the end — or drop the mark
 *  when nothing follows it yet. */
function closeOpen(s: string): string {
  const ticks = (s.match(/`/g) ?? []).length;
  if (ticks % 2 === 1) {
    const at = s.lastIndexOf("`");
    return s.slice(at + 1).trim() ? s + "`" : s.slice(0, at);
  }
  for (const mark of ["**", "~~"]) {
    // Outside code spans, and not in a line of marks: "***" is a rule, not
    // an open bold (read as one, it was cut to "*" — an empty list item).
    const plain = s
      .replace(/`[^`]*`/g, (m) => " ".repeat(m.length))
      .split("\n")
      .map((line) => (MARKS_ONLY.test(line) ? " ".repeat(line.length) : line))
      .join("\n");
    const n = plain.split(mark).length - 1;
    if (n % 2 === 1) {
      const at = plain.lastIndexOf(mark);
      s = s.slice(at + mark.length).trim() ? s + mark : s.slice(0, at);
    }
  }
  return s;
}
