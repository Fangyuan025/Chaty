/** A page's stylesheets from elsewhere, made not to hold its picture up.
 *  Evidence: pages asking Google Fonts for their fonts, somewhere that cannot
 *  reach it, never loaded — the stylesheet held the page's scripts and its
 *  load event, and the folded page's card stayed grey for good. */
import { describe, expect, it } from "vitest";
import { sheetsAside } from "./Markdown";

describe("sheetsAside", () => {
  it("makes a stylesheet from elsewhere wait for nothing", () => {
    const out = sheetsAside(
      '<head><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet"></head>',
    );
    expect(out).toContain(`media="print" onload="this.media='all'" data-chaty-aside>`);
    expect(out).toContain("family=Inter:wght@400;600&display=swap");
  });

  it("turns an @import from elsewhere into such a link, its semicolons and all", () => {
    const out = sheetsAside(
      "<style>@import url('https://fonts.googleapis.com/css2?family=A:wght@3;4&display=swap'); body{color:red}</style>",
    );
    expect(out).toMatch(/^<link rel="stylesheet" href="https:\/\/fonts\.googleapis\.com\/css2\?family=A:wght@3;4&display=swap" media="print"/);
    expect(out).toContain("<style> body{color:red}</style>");
    expect(out).not.toContain("@import");
  });

  it("leaves the page's own sheets, and its print sheet, as they are", () => {
    const own = '<link rel="stylesheet" href="style.css"><link rel="stylesheet" media="print" href="https://x.cdn/p.css"><style>@import "local.css"; a{}</style>';
    expect(sheetsAside(own)).toBe(own);
  });
});
