import hljs from "highlight.js/lib/common";
import { describe, expect, it } from "vitest";
import { GrowingHighlight, highlightFinished, highlightTyping, splitHighlighted, splitTyping } from "./highlightLines";

const PAGE = [
  "<!DOCTYPE html>",
  "<html>",
  "<head>",
  "  <style>",
  "    .card { color: #333; }",
  "  </style>",
  "</head>",
  "<body>",
  "  <!-- a comment",
  "       over two lines -->",
  "  <script>",
  "    const n = 1;",
  "    console.log(`n=${n}`);",
  "  </script>",
  "</body>",
  "</html>",
].join("\n");

describe("highlighted markup, a line at a time", () => {
  it("closes what is open at a line's end and opens it again on the next", () => {
    const { lines, open } = splitHighlighted('<span class="a">x\ny<span class="b">z\nw</span></span>v');
    expect(lines).toEqual(['<span class="a">x</span>', '<span class="a">y<span class="b">z</span></span>', '<span class="a"><span class="b">w</span></span>v']);
    expect(open).toEqual([]);
    expect(splitHighlighted('<span class="c">/* a\nb').open).toEqual(['<span class="c">']);
  });

  it("each line is the same markup, as text, as the whole block highlighted at once", () => {
    const whole = hljs.highlight(PAGE, { language: "html", ignoreIllegals: true }).value;
    const { lines } = highlightFinished(PAGE, "html");
    expect(lines).toHaveLength(PAGE.split("\n").length);
    const text = (h: string) => h.replace(/<[^>]+>/g, "");
    expect(lines.map(text).join("\n")).toBe(text(whole));
  });

  it("a page's script and style are coloured as script and style, not as HTML", () => {
    const { lines } = highlightFinished(PAGE, "html");
    // "const" in the script is a keyword; ".card" in the style a selector.
    expect(lines[11]).toContain('<span class="hljs-keyword">const</span>');
    expect(lines[4]).toContain('<span class="hljs-selector-class">.card</span>');
    // The comment's second line is still the comment.
    expect(lines[9]).toMatch(/^<span class="hljs-comment">/);
    // Alone, as the lines were highlighted before, the script's line is plain.
    expect(hljs.highlight(PAGE.split("\n")[11], { language: "html", ignoreIllegals: true }).value).not.toContain("hljs-keyword");
  });
});

describe("the line being typed", () => {
  it("inside a script, is the script's language", () => {
    const before = highlightFinished(PAGE.split("\n").slice(0, 11).join("\n"), "html");
    expect(before.sub).toBe("javascript");
    expect(highlightTyping("    const m = 2", before)).toContain('<span class="hljs-keyword">const</span>');
  });

  it("inside a comment that spans lines, is the comment", () => {
    const before = highlightFinished(PAGE.split("\n").slice(0, 9).join("\n"), "html");
    expect(highlightTyping("       over two", before)).toBe('<span class="hljs-comment">       over two</span>');
  });

  it("otherwise, in the block's language — and plain for one highlight.js does not know", () => {
    expect(highlightTyping("let x = 1", highlightFinished("fn main() {", "rust"))).toContain("hljs-keyword");
    expect(highlightTyping("a < b", highlightFinished("x", "no-such-lang"))).toBe("a &lt; b");
  });

  it("is told from the finished lines; the newline markdown adds is not the model's", () => {
    expect(splitTyping("a\nb\nc\n")).toEqual({ finished: "a\nb", typing: "c", trailing: true });
    expect(splitTyping("a\n")).toEqual({ finished: null, typing: "a", trailing: true });
    expect(splitTyping("\nb\n")).toEqual({ finished: "", typing: "b", trailing: true });
    expect(splitTyping("")).toEqual({ finished: null, typing: "", trailing: false });
    // A first line left empty is a line.
    expect(highlightFinished("", "html").lines).toEqual([""]);
    expect(highlightFinished(null, "html").lines).toEqual([]);
  });
});

describe("a long block, highlighted whole only as often as it can afford", () => {
  it("cheap: whole every line; dear: the lines since in the place the last whole one left off, whole again after its gap", () => {
    let clock = 0;
    // Each whole highlight "costs" what the clock says: 1 ms while cheap, 10 ms once dear.
    let cost = 1;
    const now = () => clock;
    const g = new GrowingHighlight("html", () => {
      const t = now();
      clock += cost / 2;
      return t;
    });
    const head = PAGE.split("\n").slice(0, 12);
    // Cheap: every line is a whole highlight (the script's line is coloured).
    expect(g.update(head.slice(0, 11).join("\n")).lines).toHaveLength(11);
    clock += 5;
    expect(g.update(head.join("\n")).lines[11]).toContain('<span class="hljs-keyword">const</span>');
    // Dear: the next line, 30 ms on, is highlighted in the script's language
    // without a whole highlight...
    cost = 10;
    clock += 30;
    g.update(PAGE.split("\n").slice(0, 13).join("\n")); // whole (the last one was cheap) — now known dear
    clock += 30;
    const soon = g.update(PAGE.split("\n").slice(0, 14).join("\n"));
    expect(soon.lines).toHaveLength(14);
    expect(soon.lines[13]).toContain("&lt;/script&gt;"); // as script text, until the next whole one
    // ...and after the gap, whole again: the closing tag is a tag.
    clock += 200;
    const later = g.update(PAGE.split("\n").slice(0, 15).join("\n"));
    expect(later.lines[13]).toContain('<span class="hljs-tag">');
  });
});
