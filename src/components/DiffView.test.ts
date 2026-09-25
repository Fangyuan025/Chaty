import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LangProvider } from "../lib/i18n";
import { DiffView } from "./DiffView";

const render = (props: Parameters<typeof DiffView>[0]) =>
  renderToStaticMarkup(createElement(LangProvider, null, createElement(DiffView, props)));

const lines = (n: number, tag: string) => Array.from({ length: n }, (_, i) => `${tag} ${i}`).join("\n");

describe("a long diff", () => {
  it("offers the rest as a button, not a dead line of text", () => {
    const html = render({ before: lines(300, "old"), after: lines(300, "new"), total: 600 });
    expect(html).toContain('<button type="button" class="cm-dl ctx cm-dl-more"');
    expect((html.match(/class="cm-dl (add|del)"/g) ?? []).length).toBe(400);
  });

  it("offers the rest of a capped copy even when what is on hand fits", () => {
    const html = render({ before: "a", after: "b", total: 9000, loadFull: async () => null });
    expect(html).toContain("cm-dl-more");
  });

  it("says nothing more when everything is shown", () => {
    const html = render({ before: "a\nb", after: "a\nc", total: 2 });
    expect(html).not.toContain("cm-dl-more");
  });
});
