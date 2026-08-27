import { describe, expect, test } from "vitest";
import { type Compacted, historyPrint, standingTail } from "./ctxBudget";

/** Chat mode's compaction summary rides in the system message, so rewriting it
 *  every turn moves every token behind it and the engine re-prefills the whole
 *  window. These cover when the one already written may stand. */
const msg = (content: string) => ({ role: "user", content });
const conv = (n: number, size = 40) =>
  Array.from({ length: n }, (_, i) => msg(`m${i}`.padEnd(size, "x")));
const memoFor = (msgs: { content: string }[], covered: number): Compacted => ({
  summary: "earlier: they discussed hash maps",
  covered,
  print: historyPrint(msgs, covered),
});

describe("a summary already written stands until the tail outgrows its room", () => {
  test("the turns after the covered stretch are what gets sent", () => {
    const msgs = conv(10);
    const tail = standingTail(memoFor(msgs, 4), msgs, 100_000, "summary");
    expect(tail).toEqual(msgs.slice(4));
  });

  test("it keeps standing as the conversation grows, so the prompt stays an append", () => {
    const msgs = conv(10);
    const memo = memoFor(msgs, 4);
    const first = standingTail(memo, msgs, 100_000, "summary");
    const grown = [...msgs, msg("a new question"), msg("and the answer")];
    const second = standingTail(memo, grown, 100_000, "summary");
    expect(first).not.toBeNull();
    expect(second).not.toBeNull();
    // The same leading messages are elided both times: everything the engine
    // computed for the earlier prompt is still a prefix of the later one.
    expect(second!.slice(0, first!.length)).toEqual(first);
  });

  test("no memo yet — nothing to stand on", () => {
    expect(standingTail(null, conv(10), 100_000, "summary")).toBeNull();
  });

  test("editing a message inside the covered stretch retires it", () => {
    const msgs = conv(10);
    const memo = memoFor(msgs, 4);
    const edited = msgs.map((m, i) => (i === 2 ? msg("rewritten entirely, at some length") : m));
    expect(standingTail(memo, edited, 100_000, "summary")).toBeNull();
  });

  test("a regenerate that truncated history behind it retires it", () => {
    const msgs = conv(10);
    const memo = memoFor(msgs, 8);
    expect(standingTail(memo, msgs.slice(0, 3), 100_000, "summary")).toBeNull();
  });

  test("once the tail no longer leaves room, a new summary is due", () => {
    const msgs = conv(40, 4000);
    const memo = memoFor(msgs, 4);
    expect(standingTail(memo, msgs, 1024, "summary")).toBeNull();
  });
});
