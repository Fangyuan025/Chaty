import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LangProvider } from "../lib/i18n";
import { Markdown, type CiteSource } from "./Markdown";

const three: CiteSource[] = [1, 2, 3].map((i) => ({
  title: `Source ${i}`,
  url: `https://example.com/${i}`,
  snippet: `snippet ${i}`,
}));

const render = (text: string, cites?: CiteSource[]) =>
  renderToStaticMarkup(createElement(LangProvider, null, createElement(Markdown, { cites, children: text })));

const anchors = (html: string) => (html.match(/class="cite"/g) ?? []).length;

describe("citation markers", () => {
  // Issue #14: three sources, and the answer cited 【4】【5】 — printed as bare
  // markers no chip under the answer answered to.
  it("drop a 【N】 past the sources and anchor the ones in range", () => {
    const html = render("黑棘星很危险【1】【3】。它会吞噬一切【4】【5】。", three);
    expect(anchors(html)).toBe(2);
    expect(html).not.toContain("【4】");
    expect(html).not.toContain("【5】");
    expect(html).toContain("它会吞噬一切。");
  });

  it("leave an out-of-range [N] alone — it may be ordinary text", () => {
    const html = render("取数组的第 [7] 项", three);
    expect(html).toContain("[7]");
    expect(anchors(html)).toBe(0);
  });

  it("are left untouched in a reply with no sources", () => {
    expect(render("标题【4】")).toContain("【4】");
  });
});
