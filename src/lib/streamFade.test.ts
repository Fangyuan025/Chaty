import { describe, expect, it } from "vitest";
import { fadeDelay, fadePieces, rehypeStreamFade, resetFadeSchedule } from "./streamFade";

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

describe("when each new word starts", () => {
  it("in order, a step apart, never more than a little behind the text", () => {
    resetFadeSchedule();
    const now = 1000;
    const burst = Array.from({ length: 30 }, () => fadeDelay(now));
    expect(burst.slice(0, 3)).toEqual([0, 12, 24]);
    // Each no earlier than the one before it…
    for (let i = 1; i < burst.length; i++) expect(burst[i]).toBeGreaterThanOrEqual(burst[i - 1]);
    // …and never more than 220 ms behind.
    expect(Math.max(...burst)).toBe(220);
    // Later, with the schedule caught up, a word starts at once.
    expect(fadeDelay(now + 2000)).toBe(0);
  });
});
