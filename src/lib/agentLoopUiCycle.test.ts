/** A cycle is not a repeat. Alternating between two dead controls — click A,
 *  click B, click A … — never produces two identical calls in a row, so the
 *  consecutive-repeat counter never sees it. Evidence: a live browsing run
 *  spent 80 steps rotating three clicks on a page that never moved, and every
 *  one of them came back "clicked". What gives it away is the page: the
 *  results keep landing back on the same handful of states. */
import { describe, expect, it } from "vitest";

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

/** Drive the loop with a fixed script of tool calls; `pages` decides what the
 *  browser answers for each click, keyed by the clicked label. */
async function run(script: string[], pages: (label: string, n: number) => string) {
  const calls = [...script];
  let n = 0;
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const a = args as { onEvent: Chan };
      a.onEvent.onmessage?.({ type: "token", text: calls.shift() ?? "All done." });
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 50, promptTokens: 10 } });
      return null;
    }
    if (cmd === "browser_click") {
      const label = (args as { text?: string }).text ?? "";
      return pages(label, n++);
    }
    return null;
  });
  let final = "";
  let reason: string | undefined;
  let steps = 0;
  await runAgentTurn(
    "work the page", [], "/tmp/ws", "en",
    {
      thinkMode: "off", maxSteps: 40, temperature: 0.2,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: (s) => { if (s.status !== "running") steps++; },
      onFinal: (t, _th, r) => { final = t; reason = r; },
      onError: (m) => { throw new Error(m); },
    },
  );
  return { final, reason, steps };
}

const click = (label: string) =>
  `<tool_call>{"name":"browser_click","arguments":{"text":"${label}"}}</tool_call>`;

describe("a UI cycle that never moves the page is stopped", () => {
  it("pauses when alternating clicks keep returning the same states", async () => {
    // Two labels, two answers, forever — the shape of the real failure.
    const script = Array.from({ length: 20 }, (_, i) => click(i % 2 ? "START" : "OPEN"));
    const { reason, final, steps } = await run(script, (label) => `Clicked: ${label}\nPage: home`);
    expect(reason).toBe("steps");
    expect(final).toMatch(/same one or two states/i);
    // Caught in the window, not at the step limit.
    expect(steps).toBeLessThan(14);
  });

  it("leaves a flow alone while every step actually changes the page", async () => {
    // The user's case: clicking CONTINUE through a story, over and over, with
    // the page genuinely advancing each time. This must never be intercepted.
    const script = Array.from({ length: 12 }, () => click("CONTINUE"));
    const { reason } = await run(script, (label, n) => `Clicked: ${label}\nPage: line ${n} of the story`);
    expect(reason).not.toBe("steps");
  });
});
