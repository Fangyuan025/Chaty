/** The think limit in Code: given to the engine, which closes a thought still
 *  open there in the model's own markup and goes on — the step acts in the
 *  same generation, as chat's limit and Skip do. Nothing is cut from here
 *  and no message tells the model to act (that is how it used to work: the
 *  step was stopped, and a second one asked for). */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");

type Ev = { type: string; [k: string]: unknown };
type Chan = { onmessage?: (ev: Ev) => void };

describe("per-round generation budget (maxGenTokens)", () => {
  afterEach(() => clearMocks());

  async function capturedMaxTokens(opts: Record<string, unknown>): Promise<number> {
    let seen = 0;
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const a = args as { request: { params: { maxTokens: number } }; onEvent: Chan };
        seen = a.request.params.maxTokens;
        a.onEvent.onmessage?.({ type: "token", text: "Done." });
        a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } });
        return null;
      }
      return null;
    });
    await runAgentTurn(
      "quick", [], "/tmp/ws", "en",
      {
        thinkMode: "off", maxSteps: 4,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
        ...opts,
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: () => {}, onError: (m) => { throw new Error(m); },
      },
    );
    return seen;
  }

  it("a user value rides straight into the request", async () => {
    expect(await capturedMaxTokens({ maxGenTokens: 2048, nCtx: 16384 })).toBe(2048);
  });

  it("zero means no ceiling of our own — the context window is the only bound", async () => {
    // Mirrors the think budget's 0: nothing is imposed beyond what the
    // window can hold (the per-thinkMode 4096/6144/8192 defaults are gone,
    // they silently truncated long reasoning and big writes).
    expect(await capturedMaxTokens({ maxGenTokens: 0, nCtx: 16384 })).toBe(12288);
    expect(await capturedMaxTokens({ maxGenTokens: 0, nCtx: 8192 })).toBe(6144);
    // …and the think depth no longer changes it.
    expect(await capturedMaxTokens({ maxGenTokens: 0, nCtx: 16384, thinkMode: "deep" })).toBe(12288);
    expect(await capturedMaxTokens({ maxGenTokens: 0, nCtx: 16384, thinkMode: "low" })).toBe(12288);
  });

  it("an oversized value is clamped to the context window", async () => {
    expect(await capturedMaxTokens({ maxGenTokens: 30000, nCtx: 8192 })).toBe(6144);
  });

  it("a tiny value is floored so a tool call still fits", async () => {
    expect(await capturedMaxTokens({ maxGenTokens: 100, nCtx: 16384 })).toBe(512);
  });
});

describe("think budget", () => {
  afterEach(() => clearMocks());

  async function sentBudget(opts: Record<string, unknown>): Promise<number | undefined> {
    let seen: number | undefined;
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const a = args as { request: { params: { thinkBudget?: number } }; onEvent: Chan };
        seen = a.request.params.thinkBudget;
        a.onEvent.onmessage?.({ type: "token", text: "Done." });
        a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } });
        return null;
      }
      return null;
    });
    await runAgentTurn(
      "quick", [], "/tmp/ws", "en",
      {
        thinkMode: "normal", maxSteps: 4,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
        ...opts,
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: () => {}, onError: (m) => { throw new Error(m); },
      },
    );
    return seen;
  }

  it("the user's limit is the engine's to keep", async () => {
    expect(await sentBudget({ thinkBudget: 40, nCtx: 8192 })).toBe(40);
    expect(await sentBudget({ thinkBudget: 2000, nCtx: 16384 })).toBe(2000);
  });

  it("with none set, a thought is closed short of the step's length cap, so the step can act", async () => {
    // 8192 window → a 6144-token step → closed a quarter short.
    expect(await sentBudget({ nCtx: 8192 })).toBe(4608);
    // A big window: at most 4096 short.
    expect(await sentBudget({ nCtx: 65536 })).toBe(49152 - 4096);
    // The user's step length counts as the cap.
    expect(await sentBudget({ nCtx: 65536, maxGenTokens: 2048 })).toBe(1536);
    // A limit past the cap is held to it.
    expect(await sentBudget({ thinkBudget: 60000, nCtx: 8192 })).toBe(4608);
  });

  it("reasoning off sends no limit", async () => {
    expect(await sentBudget({ thinkMode: "off", thinkBudget: 40 })).toBe(0);
  });

  it("a thought the engine closed goes on to act in the same step, nothing cut or added", async () => {
    // What the engine streams when the limit closes the thought: the close in
    // the model's markup, then the model's own words after it.
    const rounds: Ev[][] = [
      [
        { type: "token", text: "<think>" },
        ...Array.from({ length: 100 }, (_, i) => ({ type: "token", text: `step${i} ` })),
        { type: "token", text: "\n</think>\n\n" },
        { type: "token", text: "Acting on the plan: the answer is 42." },
        { type: "done", stats: { completionTokens: 110, tokensPerSecond: 50, promptTokens: 10 } },
      ],
    ];
    let cancels = 0;
    let gens = 0;
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        gens++;
        const ch = (args as { onEvent: Chan }).onEvent;
        for (const ev of rounds.shift() ?? [{ type: "token", text: "Done." }, { type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } }]) {
          ch.onmessage?.(ev);
        }
        return null;
      }
      if (cmd === "cancel_generation") { cancels++; return null; }
      return null;
    });
    const injects: string[] = [];
    let final = "";
    let thought = "";
    await runAgentTurn(
      "solve it", [], "/tmp/ws", "en",
      {
        thinkMode: "normal", thinkBudget: 40, maxSteps: 8,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: (t, th) => { final = t; thought = th ?? ""; },
        onError: (m) => { throw new Error(m); },
        onTrace: (ev) => { if (ev.kind === "inject") injects.push(ev.text); },
      },
    );
    expect(gens).toBe(1);
    expect(cancels).toBe(0);
    expect(injects).toEqual([]);
    expect(final).toContain("42");
    expect(thought).toContain("step99");
  });

  it("long thinking under the limit streams uncut", async () => {
    const rounds: Ev[][] = [
      [
        { type: "token", text: "<think>" },
        ...Array.from({ length: 3200 }, (_, i) => ({ type: "token", text: `t${i} ` })),
        { type: "token", text: "</think>Considered everything: shipping the answer." },
        { type: "done", stats: { completionTokens: 3202, tokensPerSecond: 50, promptTokens: 10 } },
      ],
    ];
    let cancels = 0;
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const ch = (args as { onEvent: Chan }).onEvent;
        for (const ev of rounds.shift() ?? []) ch.onmessage?.(ev);
        return null;
      }
      if (cmd === "cancel_generation") { cancels++; return null; }
      return null;
    });
    const injects: string[] = [];
    let final = "";
    await runAgentTurn(
      "solve it", [], "/tmp/ws", "en",
      {
        thinkMode: "normal", thinkBudget: 4000, maxSteps: 8,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: (t) => { final = t; },
        onError: (m) => { throw new Error(m); },
        onTrace: (ev) => { if (ev.kind === "inject") injects.push(ev.text); },
      },
    );
    expect(cancels).toBe(0);
    expect(injects).toEqual([]);
    expect(final).toContain("shipping the answer");
  });
});
