/** The user think budget: over the ceiling the think block CLOSES gracefully —
 *  reasoning kept in context, model told to act — instead of the runaway
 *  gate's discard. Owner call: a 35B at low temperature loops in thought;
 *  cutting must not cost coherence. */
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

describe("think budget", () => {
  afterEach(() => clearMocks());

  it("over budget: block closed, reasoning kept, model acts next round", async () => {
    // Round 1: 100 think tokens against a 40-token budget → tripped at the
    // 48-token check. Round 2: the model acts.
    const rounds: Ev[][] = [
      [
        { type: "token", text: "<think>" },
        ...Array.from({ length: 100 }, (_, i) => ({ type: "token", text: `step${i} ` })),
        { type: "done", stats: { completionTokens: 101, tokensPerSecond: 50, promptTokens: 10 } },
      ],
      [
        { type: "token", text: "Acting on the plan: the answer is 42." },
        { type: "done", stats: { completionTokens: 9, tokensPerSecond: 50, promptTokens: 10 } },
      ],
    ];
    let cancels = 0;
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
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
    await runAgentTurn(
      "solve it", [], "/tmp/ws", "en",
      {
        thinkMode: "normal", thinkBudget: 40, maxSteps: 8,
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
    expect(cancels).toBeGreaterThanOrEqual(1);
    expect(injects.some((i) => i.includes("budget reached"))).toBe(true);
    expect(final).toContain("42");
  });

  it("long thinking under budget streams uncut (the built-in runaway cap is gone)", async () => {
    // 3200 think tokens against a 4000 budget: the deleted built-in gate used
    // to behead this at 3000 and discard reasoning the user had allowed.
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
    expect(injects.some((i) => i.includes("budget reached"))).toBe(false);
    expect(final).toContain("shipping the answer");
  });

  it("no budget set: thinking is never cut mid-stream at all (owner call — cap is opt-in)", async () => {
    const rounds: Ev[][] = [
      [
        { type: "token", text: "<think>" },
        ...Array.from({ length: 3500 }, (_, i) => ({ type: "token", text: `x${i} ` })),
        { type: "token", text: "</think>Long thought, clean landing." },
        { type: "done", stats: { completionTokens: 3502, tokensPerSecond: 50, promptTokens: 10 } },
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
    let final = "";
    await runAgentTurn(
      "solve it", [], "/tmp/ws", "en",
      {
        thinkMode: "normal", maxSteps: 8,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: (t) => { final = t; },
        onError: (m) => { throw new Error(m); },
      },
    );
    expect(cancels).toBe(0);
    expect(final).toContain("clean landing");
  });

  it("budget 0 (auto) never trips the graceful close on ordinary thinking", async () => {
    const rounds: Ev[][] = [
      [
        { type: "token", text: "<think>" },
        ...Array.from({ length: 100 }, () => ({ type: "token", text: "hm " })),
        { type: "token", text: "</think>All finished: result ready." },
        { type: "done", stats: { completionTokens: 102, tokensPerSecond: 50, promptTokens: 10 } },
      ],
    ];
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const ch = (args as { onEvent: Chan }).onEvent;
        for (const ev of rounds.shift() ?? []) ch.onmessage?.(ev);
        return null;
      }
      return null;
    });
    const injects: string[] = [];
    let final = "";
    await runAgentTurn(
      "solve it", [], "/tmp/ws", "en",
      {
        thinkMode: "normal", maxSteps: 8,
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
    expect(injects.some((i) => i.includes("budget reached"))).toBe(false);
    expect(final).toContain("result ready");
  });
});
