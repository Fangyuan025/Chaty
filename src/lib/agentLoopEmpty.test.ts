/** Empty-completion breaker: zero tokens is never an answer. Evidence: the
 *  quick15@3.6 baseline (sympy-12419) — one empty raw after a read_file was
 *  accepted as the final answer and the task died at 8 of 120 steps. */
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

async function runRounds(rounds: string[]) {
  const script = [...rounds];
  const temps: number[] = [];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const a = args as { request: { params: { temperature: number } }; onEvent: Chan };
      temps.push(a.request.params.temperature);
      a.onEvent.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 50, promptTokens: 10 } });
      return null;
    }
    if (cmd === "agent_bash") return { stdout: "ok", stderr: "", code: 0, timedOut: false, bgId: null };
    return null;
  });
  let final = "";
  let reason: string | undefined;
  const raws: string[] = [];
  await runAgentTurn(
    "do the thing", [], "/tmp/ws", "en",
    {
      thinkMode: "off", maxSteps: 10, temperature: 0.2,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: (t, _th, r) => { final = t; reason = r; },
      onError: (m) => { throw new Error(m); },
      onTrace: (ev) => { if (ev.kind === "raw") raws.push(ev.text); },
    },
  );
  return { final, reason, raws, temps };
}

describe("empty-completion breaker", () => {
  afterEach(() => clearMocks());

  it("retries an empty round hotter instead of ending the task", async () => {
    const { final, reason, raws, temps } = await runRounds(["", "Recovered — here is the answer."]);
    expect(final).toContain("Recovered");
    expect(reason).not.toBe("steps");
    expect(raws).toEqual(["", "Recovered — here is the answer."]);
    // Retry after the empty round samples hotter than the base 0.2.
    expect(temps[1]).toBeGreaterThanOrEqual(0.7);
  });

  it("three empties in a row pause for the user — never a silent empty final", async () => {
    const { final, reason } = await runRounds(["", "", ""]);
    expect(reason).toBe("steps");
    expect(final).toContain("empty output");
  });

  it("an empty THINK BLOCK with nothing else is empty too — never an onFinal('')", async () => {
    // Dev repro: thinking off pre-fills the block, 3.6 stops right after it.
    const { final, reason, temps } = await runRounds(["<think>\n\n</think>\n", "Recovered for real."]);
    expect(final).toContain("Recovered");
    expect(reason).not.toBe("steps");
    expect(temps[1]).toBeGreaterThanOrEqual(0.7);
  });

  it("three empty-think rounds pause with a MESSAGE — silence is not an ending", async () => {
    const { final, reason } = await runRounds(["<think></think>", "<think>\n</think>", "<think></think>"]);
    expect(reason).toBe("steps");
    expect(final.length).toBeGreaterThan(10);
  });

  it("a non-empty round resets the streak (empties around real work never sum to a pause)", async () => {
    const tool = '<tool_call>{"name":"bash","arguments":{"command":"echo hi"}}</tool_call>';
    const { final, reason } = await runRounds(["", tool, "", tool, "", "All finished now."]);
    expect(final).toContain("finished");
    expect(reason).not.toBe("steps");
  });
});
