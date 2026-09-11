/** A model that keeps re-emitting the same unparseable tool call must not be
 *  allowed to spend the whole turn on it. Measured: a 4B model wrote a bash
 *  command with bare double quotes inside its JSON string and sent that exact
 *  call twenty times in a row, each answered with the same "not valid" note at
 *  the same temperature, until the step limit ran out. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");

type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };

// The shape from the measured run: the inner quotes end the JSON string early.
const BROKEN =
  '<think>\n验证所有命令。\n</think>\n\n<tool_call>{"name":"bash","arguments":{"command":"python3 cli.py total && echo "---" && python3 cli.py largest"}}';

async function run(replies: (n: number) => string) {
  const temps: number[] = [];
  let final: { text: string; reason?: string } | null = null;
  mockIPC(async (cmd, args) => {
    if (cmd !== "generate") return null;
    const a = args as { request: { params: { temperature: number } }; onEvent: Chan };
    temps.push(a.request.params.temperature);
    a.onEvent.onmessage?.({ type: "token", text: replies(temps.length) });
    a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } });
    return null;
  });
  await runAgentTurn(
    "verify the cli", [], "/tmp/ws", "zh",
    {
      thinkMode: "normal", maxSteps: 30, temperature: 0.3,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: (text: string, _u: unknown, reason?: string) => { final = { text, reason }; },
      onError: (m) => { throw new Error(m); },
    },
  );
  return { temps, final: final as { text: string; reason?: string } | null };
}

describe("the same broken tool call, over and over", () => {
  afterEach(() => clearMocks());

  it("pauses the turn within a few rounds instead of spending every step", async () => {
    const { temps, final } = await run(() => BROKEN);
    expect(temps.length).toBeLessThanOrEqual(5);
    expect(final?.reason).toBe("steps");
  });

  it("samples hotter once the plain note has not helped", async () => {
    const { temps } = await run(() => BROKEN);
    expect(temps[0]).toBe(0.3);
    expect(temps[2]).toBeGreaterThanOrEqual(0.7);
  });

  it("a different broken call does not count toward the pause", async () => {
    // Two distinct broken calls alternating never repeat back to back.
    const other = BROKEN.replace("largest", "total");
    const { temps } = await run((n) => (n <= 8 ? (n % 2 ? BROKEN : other) : "Done."));
    expect(temps.length).toBe(9);
  });
});
