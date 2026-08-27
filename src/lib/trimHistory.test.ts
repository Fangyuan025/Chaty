import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { trimHistory } = await import("./agentLoop");
const { messageTokens: estimateTokens } = await import("./ctxBudget");

/** Cross-turn compaction in code mode. What matters is not only that it frees
 *  space but that it frees ENOUGH: a trim that stops the moment it slips under
 *  the trigger is over the trigger again next turn, and every trim rewrites the
 *  note at the front of the prompt — which is the whole prompt, as far as KV
 *  reuse is concerned. */
const NCTX = 8192;
const turn = (i: number) => [
  { role: "user" as const, content: `question ${i} `.padEnd(1200, "x") },
  { role: "assistant" as const, content: `answer ${i} `.padEnd(1200, "y") },
];
const convo = (n: number) => Array.from({ length: n }, (_, i) => turn(i)).flat();

describe("code mode's cross-turn trim leaves room to grow", () => {
  test("a history under the trigger is handed back untouched", async () => {
    const h = convo(2);
    const out = await trimHistory(h, NCTX);
    expect(out.trimmed).toBe(false);
    expect(out.history).toBe(h);
  });

  test("trimming goes well under the trigger, not just under it", async () => {
    const out = await trimHistory(convo(30), NCTX);
    expect(out.trimmed).toBe(true);
    const trigger = Math.floor(NCTX * 0.4);
    expect(estimateTokens(out.history)).toBeLessThanOrEqual(Math.floor(trigger * 0.6));
  });

  test("so the next turn's growth does not immediately re-trim", async () => {
    const first = await trimHistory(convo(30), NCTX);
    // One more exchange lands on top of what was kept.
    const next = [...first.history, ...turn(99)];
    const second = await trimHistory(next, NCTX);
    expect(second.trimmed).toBe(false);
    // Untouched means the prompt still extends the one before it.
    expect(second.history).toBe(next);
  });

  test("the most recent exchange is never what gets dropped", async () => {
    const h = convo(30);
    const out = await trimHistory(h, NCTX);
    expect(out.history[out.history.length - 1]).toEqual(h[h.length - 1]);
  });

  test("the kept slice never opens on an assistant turn", async () => {
    const out = await trimHistory(convo(30), NCTX);
    // [0] is the compaction note; what follows it starts an exchange.
    expect(out.history[1]?.role).not.toBe("assistant");
  });
});
