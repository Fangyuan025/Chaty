import { describe, expect, it } from "vitest";
import { fadePieces, fadeTiming, rehypeStreamFade, resetFadeSchedule } from "./streamFade";

type Node = { type: string; tagName?: string; value?: string; properties?: Record<string, unknown>; children?: Node[] };
const text = (value: string): Node => ({ type: "text", value });
const el = (tagName: string, children: Node[], properties: Record<string, unknown> = {}): Node => ({ type: "element", tagName, properties, children });

describe("the pieces that fade in", () => {
  it("one per CJK character, a word with its spaces otherwise — and nothing lost", () => {
    const s = "Rust 的生命周期 is a borrow's span.";
    expect(fadePieces(s)).toEqual(["Rust ", "的", "生", "命", "周", "期", " ", "is ", "a ", "borrow's ", "span."]);
    expect(fadePieces(s).join("")).toBe(s);
  });
});

describe("the block being written, as words", () => {
  it("wraps text in spans numbered by where they start; code and math stay whole", () => {
    const tree = el("root", [
      el("p", [text("引用 is "), el("strong", [text("safe")]), text(" now "), el("code", [text("fn a()")]), text(" ok")]),
      el("span", [text("x+y")], { className: ["katex"] }),
    ]);
    rehypeStreamFade()(tree as never);
    const p = tree.children![0];
    const spans = (n: Node): { o: unknown; t: string }[] =>
      (n.children ?? []).flatMap((c) =>
        c.type === "element" && c.tagName === "span" && (c.properties?.className as string[])?.includes("sw")
          ? [{ o: c.properties!.dataO, t: c.children![0].value! }]
          : spans(c),
      );
    expect(spans(p)).toEqual([
      { o: 0, t: "引" },
      { o: 1, t: "用" },
      { o: 2, t: " " },
      { o: 3, t: "is " },
      { o: 6, t: "safe" },
      { o: 10, t: " " },
      { o: 11, t: "now " },
      // `fn a()` (6 characters) is counted but not wrapped.
      { o: 21, t: " " },
      { o: 22, t: "ok" },
    ]);
    // Math is left exactly as it was.
    expect(tree.children![1].children).toEqual([text("x+y")]);
  });
});

describe("only plain text fades", () => {
  it("a table's cells, a list's words and a heading: the table streams as it always has", () => {
    const tree = el("root", [
      el("h2", [text("小结")]),
      el("ul", [el("li", [text("one")])]),
      el("table", [el("tbody", [el("tr", [el("td", [text("cell")])])])]),
    ]);
    rehypeStreamFade()(tree as never);
    const swCount = (n: Node): number =>
      (n.children ?? []).reduce((k, c) => k + ((c.properties?.className as string[] | undefined)?.includes("sw") ? 1 : 0) + swCount(c), 0);
    expect(swCount(tree.children![0])).toBe(2);
    expect(swCount(tree.children![1])).toBe(1);
    expect(swCount(tree.children![2])).toBe(0);
    expect(tree.children![2].children![0].children![0].children![0].children).toEqual([text("cell")]);
  });
});

describe("when each new word starts", () => {
  it("in order, a step apart, never more than a little behind the text", () => {
    resetFadeSchedule();
    // A model's pace: a word every 30 ms, two arriving in one frame.
    expect(fadeTiming(1000)).toEqual({ delay: 0, quick: false });
    expect(fadeTiming(1000)).toEqual({ delay: 8, quick: false });
    expect(fadeTiming(1030)).toEqual({ delay: 0, quick: false });
    // A burst after a pause: in order, capped at 60 ms behind.
    resetFadeSchedule();
    const burst = Array.from({ length: 10 }, () => fadeTiming(5000).delay);
    for (let i = 1; i < burst.length; i++) expect(burst[i]).toBeGreaterThanOrEqual(burst[i - 1]);
    expect(Math.max(...burst)).toBe(60);
  });

  it("a stream faster than the fade lets each word in at once, quickly", () => {
    resetFadeSchedule();
    let last = { delay: 0, quick: false };
    // Twenty words in 40 ms.
    for (let i = 0; i < 20; i++) last = fadeTiming(9000 + i * 2);
    expect(last).toEqual({ delay: 0, quick: true });
    // Back to a usual pace, the fade comes back.
    for (let i = 0; i < 20; i++) last = fadeTiming(20000 + i * 40);
    expect(last.quick).toBe(false);
  });
});
