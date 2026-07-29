/** The run-check wrap-up note, end to end through the REAL loop: write code,
 *  try to deliver without running anything → exactly one nudge; run it for
 *  real → silence; fake a check with a read-only command → still nudged.
 *  Threshold behavior itself is unit-tested in wrapupGate.test.ts — this
 *  file proves the LOOP feeds the gate honest state. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

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

const BIG_PY = Array.from({ length: 40 }, (_, i) => `print(${i})`).join("\n");
const call = (name: string, args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;

async function runRounds(rounds: string[]): Promise<{ injects: string[]; final: string }> {
  const script = [...rounds];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const ch = (args as { onEvent: Chan }).onEvent;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "agent_write_file") return "written";
    if (cmd === "agent_bash") return { stdout: "ok", stderr: "", code: 0, timedOut: false, bgId: null };
    if (cmd === "agent_read_file") throw new Error("no such file");
    return null;
  });
  const injects: string[] = [];
  let final = "";
  await runAgentTurn(
    "build the tool",
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
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: (t) => { final = t; },
      onError: (m) => { throw new Error(`loop errored: ${m}`); },
      onTrace: (ev) => { if (ev.kind === "inject") injects.push(ev.text); },
    },
  );
  return { injects, final };
}

const RUN_MARK = "read-only commands don't count";

describe("run-check through the real loop", () => {
  beforeEach(() => {});
  afterEach(() => clearMocks());

  it("write big code, deliver without running → one nudge, then the answer stands", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      "All done, tool.py is ready.",
      "Final: shipped without a run, as instructed.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
    expect(final).toContain("Final");
  });

  it("write big code, actually run it → no nudge", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      call("bash", { command: "python3 tool.py" }),
      "All done, ran clean.",
    ]);
    expect(injects.some((i) => i.includes(RUN_MARK))).toBe(false);
    expect(final).toContain("All done");
  });

  it("a read-only command is not verification → still nudged", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      call("bash", { command: "ls -la" }),
      "All done.",
      "Final.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
  });

  it("small single edit stays frictionless — no nudge below the bar", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tiny.py", content: "print('hi')" }),
      "Done, one-liner written.",
    ]);
    expect(injects.some((i) => i.includes(RUN_MARK))).toBe(false);
    expect(final).toContain("one-liner");
  });
});
