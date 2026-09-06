import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { replayableTail, storeAssistantTurn } = await import("./agentLoop");

type Tail = NonNullable<Parameters<typeof replayableTail>[0][number]["prompt"]>;
type Msg = { role: string; prompt?: Tail };
const tail = (n: number): Tail =>
  Array.from({ length: n }, (_, i) => ({ role: "user" as const, content: `m${i}` }));

describe("a turn continues from what the last one actually sent", () => {
  test("the newest recorded tail is replayed", () => {
    const t = tail(6);
    const msgs: Msg[] = [{ role: "user" }, { role: "assistant", prompt: t }];
    expect(replayableTail(msgs)).toBe(t);
  });

  test("only the newest — an older turn's record is behind", () => {
    const old = tail(2);
    const fresh = tail(9);
    const msgs: Msg[] = [
      { role: "user" },
      { role: "assistant", prompt: old },
      { role: "user" },
      { role: "assistant", prompt: fresh },
    ];
    expect(replayableTail(msgs)).toBe(fresh);
  });

  test("a user message after the record means the record is stale", () => {
    // Cannot normally happen — a turn answering that message would record its
    // own tail — so seeing it means something rewrote the conversation.
    const msgs: Msg[] = [{ role: "assistant", prompt: tail(4) }, { role: "user" }];
    expect(replayableTail(msgs)).toBeNull();
  });

  test("locally injected assistant text after it is harmless", () => {
    // /help appends an assistant bubble the model never produced.
    const t = tail(4);
    const msgs: Msg[] = [{ role: "assistant", prompt: t }, { role: "assistant" }];
    expect(replayableTail(msgs)).toBe(t);
  });

  test("a conversation with no record falls back", () => {
    expect(replayableTail([{ role: "user" }, { role: "assistant" }])).toBeNull();
    expect(replayableTail([])).toBeNull();
  });

  test("an empty record is not a record", () => {
    expect(replayableTail([{ role: "assistant", prompt: [] }])).toBeNull();
  });
});

describe("a stored turn reproduces what the model generated", () => {
  test("the tool-call closer trimmed by the stop sequence is put back", () => {
    const msgs: { role: string; content: string; reasoning_content?: string }[] = [];
    // What the app receives: generation stopped AT `</tool_call>`, so the
    // closer is not in the text even though the model produced it.
    storeAssistantTurn(msgs as never, '<think>\nplan\n</think>\n\n<tool_call>{"name":"ls"}', false);
    expect(msgs[0].content.endsWith("</tool_call>")).toBe(true);
  });

  test("a turn that already closes its call is left alone", () => {
    const msgs: { role: string; content: string }[] = [];
    const turn = '<tool_call>{"name":"ls"}</tool_call>';
    storeAssistantTurn(msgs as never, turn, false);
    expect(msgs[0].content).toBe(turn);
  });

  test("prose with no call is untouched", () => {
    const msgs: { role: string; content: string }[] = [];
    storeAssistantTurn(msgs as never, "all done", false);
    expect(msgs[0].content).toBe("all done");
  });
});
