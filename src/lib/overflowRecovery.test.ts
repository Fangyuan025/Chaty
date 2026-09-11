/** A prompt too long for the window must not end the session.
 *
 *  Measured: a 4B model wrote a sixteen-thousand-token write_file whose JSON did
 *  not parse. Stored verbatim — as a failed call is, to keep the cache — it
 *  filled the window on its own; compaction's tiers all hold the newest write
 *  and the working thread back, so nothing could be freed; the engine refused
 *  the prompt, the turn ended, and the next turn inherited the same transcript
 *  and was refused too. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn, compactMessages } = await import("./agentLoop");
const { contextLimit, messageTokens, rawMessageTokens, resetCalibration } = await import("./ctxBudget");

type Msg = { role: string; content: string };
type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };

const NCTX = 16384;
/** A write the model botched: the JSON never closes, so it cannot be stubbed. */
const botched = (lines: number) =>
  `<tool_call>{"name":"write_file","arguments":{"path":"t.py","content":"${"x = 1\\n".repeat(lines)}`;

describe("a turn too big for the window", () => {
  beforeEach(() => resetCalibration());
  afterEach(() => clearMocks());

  it("compaction cuts the model's own oversized turn when nothing else is left", async () => {
    const msgs = [
      { role: "system", content: "sys" },
      { role: "user", content: "write the tests" },
      { role: "assistant", content: botched(9000) },
      { role: "user", content: "that call was not valid" },
    ];
    expect(messageTokens(msgs)).toBeGreaterThan(contextLimit(NCTX));
    await compactMessages(msgs as never, NCTX);
    expect(messageTokens(msgs)).toBeLessThanOrEqual(contextLimit(NCTX));
    // Head and tail survive: the model still sees what it was doing.
    expect(msgs[2].content.startsWith('<tool_call>{"name":"write_file"')).toBe(true);
  });

  async function turn(refuse: "llama" | "mlx") {
    const sizes: number[] = [];
    const errors: string[] = [];
    let finals = 0;
    mockIPC(async (cmd, args) => {
      if (cmd !== "generate") return null;
      const a = args as { request: { messages: Msg[] }; onEvent: Chan };
      const raw = rawMessageTokens(a.request.messages);
      sizes.push(raw);
      if (sizes.length === 1) {
        // The engine counts more than we estimated, and more than fits.
        const counted = Math.round(raw * 1.3);
        if (refuse === "llama") {
          throw new Error(`提示词 ${counted} tokens 超出上下文窗口 ${NCTX}，请新建对话或缩短输入。`);
        }
        a.onEvent.onmessage?.({
          type: "done",
          stats: { promptTokens: counted, completionTokens: 0, tokensPerSecond: 0, stopReason: "context" },
        });
        return null;
      }
      a.onEvent.onmessage?.({ type: "token", text: "Done." });
      a.onEvent.onmessage?.({
        type: "done",
        stats: { promptTokens: raw, completionTokens: 1, tokensPerSecond: 1, stopReason: "eos" },
      });
      return null;
    });
    // Under the limit by our estimate — with room left for the system prompt
    // the step adds — so nothing compacts up front.
    const history = [
      { role: "user", content: "write the tests" },
      { role: "assistant", content: botched(4500) },
      { role: "user", content: "that call was not valid" },
      { role: "assistant", content: "I will write it again, smaller." },
    ];
    expect(messageTokens(history)).toBeLessThan(contextLimit(NCTX) - 3000);
    await runAgentTurn(
      "go on", history as never, "/tmp/ws", "en",
      {
        thinkMode: "off", maxSteps: 6, nCtx: NCTX,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: () => { finals++; }, onError: (m) => { errors.push(m); },
      },
    );
    return { sizes, errors, finals };
  }

  it("a prompt llama.cpp refuses is compacted and the step runs again", async () => {
    const { sizes, errors, finals } = await turn("llama");
    expect(errors).toEqual([]);
    expect(finals).toBe(1);
    // The refused step, a summary of what compaction dropped, then the retry.
    expect(sizes.length).toBeLessThanOrEqual(3);
    expect(sizes[sizes.length - 1]).toBeLessThan(sizes[0]);
  });

  it("so is one MLX turns away with a context stop", async () => {
    const { sizes, errors, finals } = await turn("mlx");
    expect(errors).toEqual([]);
    expect(finals).toBe(1);
    // The refused step, a summary of what compaction dropped, then the retry.
    expect(sizes.length).toBeLessThanOrEqual(3);
    expect(sizes[sizes.length - 1]).toBeLessThan(sizes[0]);
  });
});
