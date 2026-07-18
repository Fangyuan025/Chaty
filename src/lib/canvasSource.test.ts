import { describe, expect, test } from "vitest";
import { annotate, highlightLines } from "./canvasSource";

describe("annotate", () => {
  test("tags real elements with data-cv and records their lines", () => {
    const src = `<html>\n<body>\n<div class="a">\n<p>hi</p>\n</div>\n</body>\n</html>`;
    const { html, lineOf, tagOf } = annotate(src);
    expect(html).toContain('<html data-cv="0">');
    expect(html).toContain('<div class="a" data-cv="2">');
    expect(html).toContain('<p data-cv="3">');
    expect(lineOf[2]).toBe(2); // div on line 2 (0-based)
    expect(tagOf[3]).toBe("p");
    // closing tags untouched
    expect(html).toContain("</div>");
  });

  test("never rewrites tags inside script/style bodies or comments", () => {
    const src = [
      "<script>",
      'const s = "<div>not a tag</div>";',
      "</script>",
      "<style>",
      "/* <p> inside comment */ body { color: red; }",
      "</style>",
      "<!-- <span>commented out</span> -->",
      "<p>real</p>",
    ].join("\n");
    const { html, tagOf } = annotate(src);
    expect(html).toContain('const s = "<div>not a tag</div>";');
    expect(html).toContain("/* <p> inside comment */");
    expect(html).toContain("<!-- <span>commented out</span> -->");
    expect(html).toContain('<p data-cv="2">real</p>');
    expect(tagOf).toEqual(["script", "style", "p"]);
  });

  test("self-closing and attribute-heavy tags stay intact", () => {
    const src = `<img src="x.png" alt="a > b" />\n<input value='q>'>`;
    const { html } = annotate(src);
    expect(html).toContain('<img src="x.png" alt="a > b" data-cv="0" />');
    expect(html).toContain(`<input value='q>' data-cv="1">`);
  });

  test("multi-line opening tags advance the line counter", () => {
    const src = `<div\n  class="x"\n>\n<p>t</p>`;
    const { lineOf } = annotate(src);
    expect(lineOf[0]).toBe(0);
    expect(lineOf[1]).toBe(3); // p after the 3-line div tag
  });
});

describe("highlightLines", () => {
  test("splits into one entry per source line", () => {
    const src = `<div>\n  <p>hi</p>\n</div>`;
    const lines = highlightLines(src);
    expect(lines).toHaveLength(3);
  });

  test("re-opens spans across line breaks so each line is standalone", () => {
    const src = `<script>\nconst x = 1;\n</script>`;
    const lines = highlightLines(src);
    for (const l of lines) {
      const opens = (l.match(/<span/g) ?? []).length;
      const closes = (l.match(/<\/span>/g) ?? []).length;
      expect(opens).toBe(closes);
    }
  });

  test("escapes plain text when highlighting fails", () => {
    const lines = highlightLines("a & b < c");
    expect(lines.join("")).toContain("&amp;");
  });
});
