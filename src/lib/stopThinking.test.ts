/** A turn that was stopped stops SAYING things. Tokens already on their way
 *  kept arriving after the user hit stop, and reporting them put the thinking
 *  panel back up — spinner and all — over a turn that had been told to stop. */
import { expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");

type Ev = { type: string; [k: string]: unknown };
type Chan = { onmessage?: (ev: Ev) => void };

it("says nothing more once the turn is stopped", async () => {
  // A stream that keeps coming after the stop, as a real engine's in-flight
  // tokens do.
  const pieces = ["<think>\n想第一步", "…想第二步", "…想第三步", "…想第四步", "</think>\n\n答案。"];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd !== "generate") return null;
    const ch = a.onEvent as Chan;
    for (const text of pieces) {
      ch.onmessage?.({ type: "token", text });
      await new Promise((r) => setTimeout(r, 1));
    }
    ch.onmessage?.({ type: "done", stats: { completionTokens: 5, tokensPerSecond: 9, promptTokens: 10 } });
    return null;
  });

  const signal = { cancelled: false };
  const thoughts: string[] = [];
  const texts: string[] = [];
  await runAgentTurn(
    "do it",
    [],
    "/tmp/ws",
    "zh",
    { thinkMode: "normal", maxSteps: 2, signal, approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }) } as never,
    {
      onThinking: (t: string) => {
        thoughts.push(t);
        // The user hits stop as the second thought arrives.
        if (thoughts.length === 2) signal.cancelled = true;
      },
      onAssistantText: (t: string) => texts.push(t),
      onStep: () => {}, onStats: () => {}, onPlan: () => {}, onCompacted: () => {},
      onError: () => {}, onFinal: () => {}, onStepText: () => {},
      onLiveStep: () => {}, onLiveStepGone: () => {},
    } as never,
  );

  // Two reports — the one that arrived before the stop and the one during
  // which it happened — and nothing after.
  expect(thoughts).toHaveLength(2);
  expect(thoughts[1]).not.toContain("第三步");
  expect(texts.every((t) => !t.includes("答案"))).toBe(true);
});
