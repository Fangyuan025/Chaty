/** Settings → Sampling reaches Code mode. The agent used to send a hard-coded
 *  top-p and repeat penalty on every step, so the sampling page did nothing
 *  there; temperature is the one value Code keeps as its own. Driven through
 *  the REAL loop. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");

type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };
type Params = Record<string, unknown>;

async function stepParams(opts: Record<string, unknown>): Promise<Params[]> {
  const seen: Params[] = [];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "generate") {
      seen.push(((a.request as Record<string, unknown>).params ?? {}) as Params);
      const ch = a.onEvent as Chan;
      ch.onmessage?.({ type: "token", text: "All done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 3, tokensPerSecond: 50, promptTokens: 50 } });
      return null;
    }
    return null;
  });
  await runAgentTurn(
    "say hi",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off", maxSteps: 2,
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
      ...opts,
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onStats: () => {}, onPlan: () => {}, onCompacted: () => {},
      onError: (e: string) => { throw new Error(e); },
      onFinal: () => {}, onStepText: () => {}, onLiveStep: () => {}, onLiveStepGone: () => {},
    } as never,
  );
  return seen;
}

describe("Code mode sampling", () => {
  it("uses the sampling the user set", async () => {
    const [p] = await stepParams({
      temperature: 0.35,
      sampling: { topP: 0.8, topK: 20, minP: 0.1, repeatPenalty: 1.02 },
    });
    expect(p).toMatchObject({ temperature: 0.35, topP: 0.8, topK: 20, minP: 0.1, repeatPenalty: 1.02 });
  });

  it("keeps its old values when no settings are passed", async () => {
    const [p] = await stepParams({});
    expect(p).toMatchObject({ temperature: 0.3, topP: 0.9, repeatPenalty: 1.05 });
  });
});

describe("Code mode shares the sampling page's length and stops", () => {
  it("ends a step at the user's stop sequences as well as at a call's closer", async () => {
    const [p] = await stepParams({
      sampling: { topP: 0.9, topK: 40, minP: 0.05, repeatPenalty: 1, stop: ["###"] },
    });
    expect(p.stop).toEqual(expect.arrayContaining(["</tool_call>", "###"]));
  });

  it("caps each step at the max length", async () => {
    const [p] = await stepParams({ maxGenTokens: 3000, nCtx: 16384 });
    expect(p.maxTokens).toBe(3000);
  });
});
