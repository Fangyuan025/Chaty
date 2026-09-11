/** A code-mode turn must extend the prompt the previous turn left in the
 *  engine's cache.
 *
 *  Each turn hands the next everything it sent, and the engine still holds
 *  exactly that. Rewriting any of it — above all at the front — costs the
 *  whole conversation's prefill again, and on the hybrid architectures (the
 *  Qwen3.5 family) there is no partial resume to soften it. What used to do the
 *  rewriting was a start-of-turn trim with a budget of 40% of the window: one
 *  working turn of tool traffic is more than that on its own, so once a session
 *  got long every single turn trimmed, wrote a fresh summary at the front, and
 *  re-read everything. These drive the real loop against a stub engine and look
 *  at the prompt it was actually handed. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");
const { contextLimit, messageTokens, rawMessageTokens, resetCalibration } = await import("./ctxBudget");

type Msg = { role: string; content: string };
type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };

const NCTX = 16384;
const ASK = "now add tests for it";

/** A finished working turn, the way a replayed tail holds it: the request,
 *  a run of read_file calls each followed by its result, and the answer. */
function workedTurn(rounds: number): Msg[] {
  const out: Msg[] = [{ role: "user", content: "build the ledger module" }];
  for (let i = 0; i < rounds; i++) {
    out.push({
      role: "assistant",
      content: `<tool_call>{"name":"read_file","arguments":{"path":"src/mod${i}.py"}}</tool_call>`,
    });
    out.push({
      role: "tool",
      content: `<tool_result name="read_file">\n${`def f${i}(x):\n    return x + ${i}\n`.repeat(120)}</tool_result>`,
    });
  }
  out.push({ role: "assistant", content: "Done — the module is built." });
  return out;
}

/** Run one turn on top of `history` and return every prompt the engine saw. */
async function prompts(history: Msg[]): Promise<Msg[][]> {
  const seen: Msg[][] = [];
  mockIPC(async (cmd, args) => {
    if (cmd !== "generate") return null;
    const a = args as { request: { messages: Msg[] }; onEvent: Chan };
    const msgs = a.request.messages.map((m) => ({ role: m.role, content: m.content }));
    seen.push(msgs);
    a.onEvent.onmessage?.({ type: "token", text: "Done." });
    // Charge exactly what was predicted, so the calibration stays at 1 and the
    // sizes below mean what they say.
    a.onEvent.onmessage?.({
      type: "done",
      stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: rawMessageTokens(msgs) },
    });
    return null;
  });
  await runAgentTurn(
    ASK, history as never, "/tmp/ws", "en",
    {
      thinkMode: "off", maxSteps: 4, nCtx: NCTX, toolRole: true,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: () => {}, onError: (m) => { throw new Error(m); },
    },
  );
  return seen;
}

/** The prompt of the turn itself, as opposed to a summarisation pass. */
const turnPrompt = (all: Msg[][]) => all.find((p) => p[p.length - 1]?.content.startsWith(ASK))!;

describe("a new turn extends the cached prompt", () => {
  beforeEach(() => resetCalibration());
  afterEach(() => clearMocks());

  it("a long working history that fits the window is sent exactly as it was", async () => {
    const history = workedTurn(8);
    // The shape the owner hit: well past 40% of the window, well inside it.
    expect(messageTokens(history)).toBeGreaterThan(NCTX * 0.4);
    expect(messageTokens(history)).toBeLessThan(contextLimit(NCTX) * 0.8);
    const all = await prompts(history);
    // No summarisation pass — nothing needed condensing.
    expect(all).toHaveLength(1);
    const p = turnPrompt(all);
    // [system, ...history, this turn's request]: a pure append.
    expect(p.slice(1, 1 + history.length)).toEqual(history);
    expect(p).toHaveLength(history.length + 2);
  });

  it("a history too big for the window is still brought inside it", async () => {
    const history = workedTurn(40);
    expect(messageTokens(history)).toBeGreaterThan(NCTX);
    const p = turnPrompt(await prompts(history));
    expect(messageTokens(p)).toBeLessThanOrEqual(contextLimit(NCTX));
    // The request this turn is about is never what gets cut.
    expect(p[p.length - 1].content.startsWith(ASK)).toBe(true);
  });

  it("an earlier turn's result that gets compacted still says what it was", async () => {
    // Just over the window: the oldest results are stubbed, not dropped.
    const history = workedTurn(14);
    expect(messageTokens(history)).toBeGreaterThan(contextLimit(NCTX));
    const p = turnPrompt(await prompts(history));
    const stubbed = p.filter(
      (m) => m.content.startsWith('<tool_result name="read_file">') && m.content.length < 200,
    );
    expect(stubbed.length).toBeGreaterThan(0);
    // The stub names the file, so the model knows what it read and can read
    // it again, instead of a bare "elided".
    expect(stubbed[0].content).toContain("src/mod0.py");
  });
});
