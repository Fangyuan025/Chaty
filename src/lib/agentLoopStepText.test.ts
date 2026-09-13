/** A step card, opened, shows exactly what the model was given for that step.
 *  The card's own copy (`step.result`) is trimmed for the renderer and lacks
 *  the notes appended for the model; where it differs, the loop hands the
 *  model's text to the host (`onStepText`) and marks the step `fullText`.
 *  Proved through the REAL loop, against the tool results it actually sent. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn, toolResultBody } = await import("./agentLoop");
type ToolStep = import("./agentLoop").ToolStep;

type Ev = { type: string; [k: string]: unknown };
type Chan = { onmessage?: (ev: Ev) => void };

const call = (name: string, args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;

// Far past what a card keeps (8,000 characters).
const BIG = Array.from({ length: 2000 }, (_, i) => `line ${i + 1} of a large source file`).join("\n");

async function run(rounds: string[]) {
  const script = [...rounds];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "generate") {
      const ch = a.onEvent as Chan;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "agent_read_file") return String(a.path) === "big.txt" ? BIG : "hello\nworld";
    if (cmd === "agent_bash") return { stdout: "hi", stderr: "", code: 0, timedOut: false, bgId: null };
    if (cmd === "agent_list_dir") return [{ name: "big.txt", isDir: false, size: BIG.length }];
    return null;
  });
  const order: string[] = [];
  const finished = new Map<string, number>(); // onStep reports of a step past "running"
  const steps = new Map<string, ToolStep>();
  const texts = new Map<string, string>();
  const results: string[] = [];
  await runAgentTurn(
    "look around",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off", maxSteps: 12,
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s) => {
        if (!steps.has(s.id)) order.push(s.id);
        if (s.status !== "running") finished.set(s.id, (finished.get(s.id) ?? 0) + 1);
        steps.set(s.id, { ...s });
      },
      onStepText: (id, text) => { texts.set(id, text); },
      onFinal: () => {},
      onError: (m) => { throw new Error(`loop errored: ${m}`); },
      onTrace: (ev) => {
        if (ev.kind === "inject" && ev.text.startsWith("<tool_result")) results.push(ev.text);
      },
    },
  );
  return { order, finished, steps, texts, results };
}

describe("an opened step card shows what the model was given", () => {
  afterEach(() => clearMocks());

  it("for every step — a large read, a small one, a command, a listing", async () => {
    const { order, steps, texts, results } = await run([
      call("read_file", { path: "big.txt" }),
      call("read_file", { path: "small.txt" }),
      call("bash", { command: "echo hi" }),
      call("list_dir", { path: "." }),
      "Done.",
    ]);
    expect(order).toHaveLength(4);
    expect(results).toHaveLength(4);
    order.forEach((id, i) => {
      const step = steps.get(id)!;
      const shown = texts.has(id) ? texts.get(id) : step.result;
      expect(shown, `${step.call.name} #${i}`).toBe(toolResultBody(results[i]));
    });
    // The large read is the case the card's copy cannot cover.
    const big = steps.get(order[0])!;
    expect(texts.get(big.id)).toContain("line 2000 of a large source file");
    expect(big.result!.length).toBeLessThan(texts.get(big.id)!.length);
    // A step's text is stored only where the card's copy differs.
    for (const [id, text] of texts) expect(text).not.toBe(steps.get(id)!.result);
  });

  it("a step is reported finished once — storing its text is not another step", async () => {
    const { finished, texts } = await run([call("read_file", { path: "big.txt" }), "Done."]);
    expect(texts.size).toBe(1);
    expect([...finished.values()]).toEqual([1]);
  });
});
