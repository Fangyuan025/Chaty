import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { replayableTail } = await import("./agentLoop");

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
