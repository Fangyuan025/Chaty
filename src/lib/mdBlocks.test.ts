/** Block-by-block rendering must look exactly like rendering the text whole,
 *  at every point of a stream — and the incremental parse must find the same
 *  blocks as parsing from scratch. Checked on every prefix of texts carrying
 *  the constructs that span or re-shape blocks as they arrive. */
import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { mdBlocks, type BlockCache } from "./mdBlocks";

const g = globalThis as Record<string, unknown>;
g.window ??= globalThis;
g.localStorage ??= { getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0 };
g.navigator ??= { userAgent: "chaty-test" };
const { Markdown } = await import("../components/Markdown");
const { LangProvider } = await import("./i18n");

const TEXTS = [
  [
    "# Title",
    "",
    "Intro paragraph with `inline` code and **bold**.",
    "",
    "Setext heading",
    "==============",
    "",
    "- one",
    "- two",
    "",
    "  still two, after a blank line",
    "- three",
    "",
    "1. first",
    "2. second",
    "",
    "| a | b |",
    "|---|---|",
    "| 1 | 2 |",
    "",
    "> quoted",
    "> more",
    "",
    "```ts",
    "const x: number = 1;",
    "",
    "function f() {",
    "  return `multi",
    "line`;",
    "}",
    "```",
    "",
    "$$",
    "a^2 + b^2 = c^2",
    "$$",
    "",
    "Closing words.",
  ].join("\n"),
  ["Before", "", "~~~~python", "print('hi')", "~~~", "still code", "~~~~", "", "after"].join("\n"),
  ["1. step", "", "   ```sh", "   ls -la", "   ```", "", "2. next"].join("\n"),
];

const raw = (text: string, blocks: boolean) =>
  renderToStaticMarkup(
    createElement(LangProvider as never, {} as never, createElement(Markdown as never, { blocks } as never, text)),
  );
// Whitespace between block elements is not rendered; inside a <pre> it is,
// which is why the fence still being written is compared on its raw text.
const render = (text: string, blocks: boolean) => raw(text, blocks).replace(/>\s+</g, "><");
const plain = (html: string) => html.replace(/<[^>]+>/g, "");
const last = <T,>(a: T[] | null) => (a ? a[a.length - 1] : undefined);

describe("markdown blocks", () => {
  it("renders every prefix of a stream as the whole text would", () => {
    for (const text of TEXTS) {
      for (let n = 1; n <= text.length; n += 3) {
        const prefix = text.slice(0, n);
        const blocks = mdBlocks(prefix);
        const whole = render(prefix, false);
        const split = render(prefix, true);
        if (last(blocks)?.openFence) {
          // The fence still being written is highlighted line by line; what
          // it says is the same.
          const at = JSON.stringify(prefix.slice(-30));
          expect(plain(split).replace(/\s+/g, ""), at).toBe(plain(whole).replace(/\s+/g, ""));
          // …and inside the <pre>, down to every newline.
          const code = (html: string) => [...html.matchAll(/<pre[^>]*>([\s\S]*?)<\/pre>/g)].map((m) => plain(m[1]));
          expect(code(raw(prefix, true)), at).toEqual(code(raw(prefix, false)));
        } else {
          expect(split, JSON.stringify(prefix.slice(-30))).toBe(whole);
        }
      }
    }
    // Renders hundreds of prefixes twice each: under a second here, over five
    // on a busy CI runner — work, not a hang.
  }, 30_000);

  it("finds the same blocks incrementally as from scratch", () => {
    for (const text of TEXTS) {
      let cache: BlockCache | undefined;
      for (let n = 1; n <= text.length; n++) {
        const prefix = text.slice(0, n);
        const inc = mdBlocks(prefix, cache);
        cache = { text: prefix, blocks: inc };
        expect(inc, JSON.stringify(prefix.slice(-30))).toEqual(mdBlocks(prefix));
      }
    }
  });

  it("marks only a fence that has not closed", () => {
    expect(last(mdBlocks("x\n\n```js\nlet a\n"))?.openFence).toEqual({ lang: "js" });
    expect(last(mdBlocks("x\n\n```js\nlet a\n```"))?.openFence).toBeUndefined();
    // A shorter fence inside does not close a longer one.
    expect(last(mdBlocks("````\n```\n"))?.openFence).toEqual({ lang: "" });
  });

  it("renders whole a text whose references resolve across blocks", () => {
    expect(mdBlocks("See [the docs][d].\n\n[d]: https://example.com\n")).toBeNull();
    expect(mdBlocks("A claim.[^1]\n\n[^1]: The source.")).toBeNull();
  });
});
