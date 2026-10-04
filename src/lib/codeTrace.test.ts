import { describe, expect, test } from "vitest";
import { buildCodeTrace, type TraceMsg } from "./codeTrace";

const session = { id: "s1", title: "Fix the lexer", workspace: "/w/p", model: "Qwen3.6-35B", app: "Chaty 2.3.5" };

const msgs: TraceMsg[] = [
  { id: "u1", role: "user", text: "Fix the lexer", steps: [], images: ["/tmp/a.png"] },
  {
    id: "a1",
    role: "assistant",
    text: "Fixed.",
    thinking: "look first",
    plan: [{ content: "read", status: "done" }],
    steps: [
      { id: "st1", call: { name: "read_file", args: { path: "src/lexer.rs" } }, status: "done", result: "fn tok…", fullText: true, thinking: "read it" },
      {
        id: "st2",
        call: { name: "edit_file", args: { path: "src/lexer.rs" } },
        status: "done",
        result: "Edited",
        diff: { path: "src/lexer.rs", before: "a", after: "b" },
        fullDiff: true,
      },
      { id: "st3", call: { name: "bash", args: { command: "cargo test" } }, status: "error", result: "exit 101" },
    ],
    prompt: [{ role: "user", content: "Fix the lexer" }],
  },
  { id: "u2", role: "user", text: "Thanks", steps: [] },
  { id: "a2", role: "assistant", text: "You're welcome.", steps: [] },
];

describe("a Code session as a JSONL trace", () => {
  test("every turn, step and answer, in order, with what the host kept in full", async () => {
    const kept: Record<string, string> = {
      st1: "fn tokenize(src: &str) { /* the whole file */ }",
      "st2:diff": JSON.stringify({ path: "src/lexer.rs", before: "a\nmore", after: "b\nmore" }),
    };
    const asked: string[] = [];
    const out = await buildCodeTrace(session, msgs, async (k) => (asked.push(k), kept[k] ?? null), new Date("2026-10-03T00:00:00Z"));
    expect(out.endsWith("\n")).toBe(true);
    const lines = out.trimEnd().split("\n").map((l) => JSON.parse(l));
    expect(lines.map((l) => l.type)).toEqual(["session", "user", "tool_call", "tool_call", "tool_call", "assistant", "context", "user", "assistant"]);
    expect(lines[0]).toMatchObject({ ...session, exportedAt: "2026-10-03T00:00:00.000Z" });
    expect(lines[1]).toMatchObject({ turn: 1, text: "Fix the lexer", images: ["/tmp/a.png"] });
    // The card's trimmed copy gives way to what the model was given.
    expect(lines[2]).toMatchObject({ turn: 1, step: 1, name: "read_file", args: { path: "src/lexer.rs" }, thinking: "read it", result: kept.st1 });
    expect(lines[3].diff).toEqual({ path: "src/lexer.rs", before: "a\nmore", after: "b\nmore" });
    expect(lines[4]).toMatchObject({ status: "error", result: "exit 101" });
    expect(lines[5]).toMatchObject({ turn: 1, text: "Fixed.", thinking: "look first", plan: [{ content: "read", status: "done" }] });
    expect(lines[6]).toMatchObject({ turn: 1, messages: [{ role: "user", content: "Fix the lexer" }] });
    expect(lines[8]).toMatchObject({ turn: 2, text: "You're welcome." });
    // Only steps with something kept apart are looked up.
    expect(asked).toEqual(["st1", "st2:diff"]);
  });

  test("a lookup that fails keeps the card's copy", async () => {
    const out = await buildCodeTrace(session, msgs.slice(0, 2), async () => {
      throw new Error("gone");
    });
    const step = JSON.parse(out.split("\n")[2]);
    expect(step.result).toBe("fn tok…");
  });
});
