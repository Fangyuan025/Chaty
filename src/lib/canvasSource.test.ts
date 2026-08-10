import { describe, expect, it, test } from "vitest";
import { annotate, buildFixPayload, fixInstruction, highlightLines, instrumentHtml, precheckScripts } from "./canvasSource";

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

describe("buildFixPayload", () => {
  it("single error stays byte-compatible with the old banner payload", () => {
    expect(buildFixPayload("ReferenceError: x is not defined (app.js:3:1)", [])).toBe(
      "ReferenceError: x is not defined (app.js:3:1)",
    );
  });

  it("bundles the banner AND every console error, numbered", () => {
    const p = buildFixPayload("TypeError: a is null (a.js:1:1)", [
      "TypeError: a is null (a.js:1:1)", // dup of the banner → dropped
      "ReferenceError: b is not defined (b.js:2:2)",
      "Failed to load resource: http://localhost/x.png",
    ]);
    expect(p.split("\n")).toHaveLength(3);
    expect(p).toContain("1. TypeError: a is null");
    expect(p).toContain("2. ReferenceError: b");
    expect(p).toContain("3. Failed to load resource");
  });

  it("dedupes re-thrown errors and bounds count and size", () => {
    const errs = Array.from({ length: 40 }, (_, i) => `Error ${i % 5}: same thing (f.js:${i % 5}:1)`);
    const p = buildFixPayload("Error 0: same thing (f.js:0:1)", errs);
    expect(p.split("\n").length).toBeLessThanOrEqual(12);
    const long = Array.from({ length: 12 }, (_, i) => `E${i} ${"y".repeat(900)}`);
    expect(buildFixPayload("banner " + "z".repeat(900), long).length).toBeLessThanOrEqual(6400);
  });

  it("ignores blank lines and survives an empty console", () => {
    expect(buildFixPayload("", [])).toBe("");
    expect(buildFixPayload("  ", ["only console error (x.js:1:1)"])).toBe("only console error (x.js:1:1)");
  });
});

describe("precheckScripts", () => {
  it("recovers the real SyntaxError WebKit muzzles, with the block index", () => {
    const html = '<script>let a = 1;</script><script>sayHello(</script><script src="x.js"></script>';
    const errs = precheckScripts(html, "en");
    expect(errs).toHaveLength(1);
    expect(errs[0]).toContain("script block 2");
    expect(errs[0]).toContain("SyntaxError");
  });
  it("clean pages and src-only scripts yield nothing", () => {
    expect(precheckScripts('<script>console.log(1)</script>', "en")).toHaveLength(0);
    expect(precheckScripts('<script src="a.js"></script>', "en")).toHaveLength(0);
    expect(precheckScripts("<p>no scripts</p>", "en")).toHaveLength(0);
  });
  it("non-classic script types are never compiled — valid pages, not faults", () => {
    expect(precheckScripts('<script type="module">import * as t from "./x.js"; export const a = 1;</script>', "en")).toHaveLength(0);
    expect(precheckScripts('<script type="application/json">{"rows":[1,2,3]}</script>', "en")).toHaveLength(0);
    expect(precheckScripts('<script type="importmap">{"imports":{"a":"./a.js"}}</script>', "en")).toHaveLength(0);
    expect(precheckScripts("<script type='text/template'><div>{{name}}</div></script>", "en")).toHaveLength(0);
  });
  it("explicit classic types still compile, and skipped blocks keep their index", () => {
    const bad = '<script type="text/javascript">sayHello(</script>';
    expect(precheckScripts(bad, "en")[0]).toContain("SyntaxError");
    // json block is block 1, broken classic block is block 2 — the index the
    // model counts in the source must survive the skip.
    const mixed = '<script type="application/json">{"a":1}</script><script>oops(</script>';
    const errs = precheckScripts(mixed, "en");
    expect(errs).toHaveLength(1);
    expect(errs[0]).toContain("script block 2");
  });
});

describe("buildFixPayload muzzle note", () => {
  it("counts anonymized errors and says they are likely several bugs", () => {
    const p = buildFixPayload(
      "Script error. (:0)",
      ["Script error. (:0)", "Script error. (:0)", "Failed to load resource: x.js"],
      "zh",
    );
    expect(p).toContain("3 次");
    expect(p).toContain("不要只修一处");
  });
  it("no note when nothing was muzzled", () => {
    expect(buildFixPayload("TypeError: x", [], "zh")).toBe("TypeError: x");
  });
});

describe("fixInstruction", () => {
  it("single error keeps the old singular phrasing", () => {
    const s = fixInstruction("TypeError: x is null", "zh", "（how）");
    expect(s).toContain("修复这个错误");
    expect(s).not.toContain("一次性");
  });
  it("a numbered list gets the all-N contract with self-check", () => {
    const p = "1. A\n2. B\n3. C\nNote: extra tail";
    const zh = fixInstruction(p, "zh", "（how）");
    expect(zh).toContain("3 个问题");
    expect(zh).toContain("一次性修复全部 3 个问题");
    expect(zh).toContain("不要只修其中一项");
    expect(zh).toContain("完整修正后的 HTML");
    const en = fixInstruction(p, "en", "(how)");
    expect(en).toContain("ALL 3");
    expect(en).not.toMatch(/[一-鿿]/);
  });
  it("two items demand both but skip the rewrite nudge", () => {
    const s = fixInstruction("1. A\n2. B", "zh", "");
    expect(s).toContain("全部 2 个问题");
    expect(s).not.toContain("完整修正后的 HTML 更稳妥");
  });
});

describe("instrumentHtml (muzzle-defeating source rewrite)", () => {
  it("wraps a classic script body without adding lines", () => {
    const html = `<div>x</div>\n<script>\nconst a = 1;\nboom();\n</script>`;
    const out = instrumentHtml(html);
    expect(out.split("\n").length).toBe(html.split("\n").length);
    expect(out).toContain("<script>try{");
    expect(out).toContain("throw __cvE}</script>");
    // the body itself is untouched between the wrap
    expect(out).toContain("const a = 1;");
  });

  it("rewrites inline on*= handlers in both quote styles, outside scripts only", () => {
    const html = `<button onclick="go()">x</button>\n<a onmouseover='hover()'>y</a>\n<script>var once = "a"; var conf = { online: 1 };</script>`;
    const out = instrumentHtml(html);
    expect(out).toContain(`onclick="try{go()}catch(__cvE){`);
    expect(out).toContain(`onmouseover='try{hover()}catch(__cvE){`);
    // script-body text that merely LOOKS like an attribute stays untouched
    expect(out).toContain(`var once = "a"; var conf = { online: 1 };`);
  });

  it("skips 'use strict' scripts and non-classic blocks", () => {
    const strict = `<script>\n'use strict';\nlet a = 1;\n</script>`;
    expect(instrumentHtml(strict)).toBe(strict);
    const module = `<script type="module">import x from 'y';</script>`;
    expect(instrumentHtml(module)).toBe(module);
    const json = `<script type="application/json">{"a":1}</script>`;
    expect(instrumentHtml(json)).toBe(json);
  });

  it("on MULTI-script pages, scripts with top-level lexical declarations stay unwrapped", () => {
    const html = `<script>\nconst SHARED = 1;\n</script>\n<script>\nuse(SHARED);\n</script>`;
    const out = instrumentHtml(html);
    // first script (const at top level) untouched; second script wrapped
    expect(out).toContain(`<script>\nconst SHARED = 1;\n</script>`);
    expect(out).toContain(`<script>try{\nuse(SHARED);\n`);
    // single-script pages wrap even with top-level const
    const single = `<script>\nconst A = 1;\nboom();\n</script>`;
    expect(instrumentHtml(single)).toContain("<script>try{");
  });

  it("empty handlers and empty scripts stay untouched", () => {
    const html = `<button onclick="">x</button><script></script>`;
    expect(instrumentHtml(html)).toBe(html);
  });
});
