/** A write_file call with no content field at all used to write an empty
 *  string — which, on a path that already existed, is a delete with extra
 *  steps. An explicit empty string still means "make this file empty".
 *  Driven through the REAL loop. */
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
type Step = { name: string; status: string; result: string };

const call = (args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name: "write_file", arguments: args })}</tool_call>`;

async function run(rounds: string[]) {
  const script = [...rounds];
  const wrote: { path: string; content: string }[] = [];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "generate") {
      const ch = a.onEvent as Chan;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "agent_read_file" || cmd === "agent_read_file_raw") return "ORIGINAL SOURCE";
    if (cmd === "agent_write_file") {
      wrote.push({ path: String(a.path), content: String(a.content) });
      return `wrote ${String(a.path)}`;
    }
    return null;
  });
  const steps: Step[] = [];
  await runAgentTurn(
    "write the file",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off", maxSteps: 4,
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s: { call: { name: string }; status: string; result?: string }) => {
        if (s.status !== "running") steps.push({ name: s.call.name, status: s.status, result: s.result ?? "" });
      },
      onStats: () => {}, onPlan: () => {}, onCompacted: () => {},
      onError: (e: string) => { throw new Error(e); },
      onFinal: () => {},
      onStepText: () => {}, onLiveStep: () => {}, onLiveStepGone: () => {},
    } as never,
  );
  return { wrote, steps };
}

describe("write_file without content", () => {
  it("does not empty the file, and says what is missing", async () => {
    const { wrote, steps } = await run([call({ path: "src/app.ts" }), "Understood."]);
    expect(wrote).toEqual([]);
    expect(steps[0].status).toBe("error");
    expect(steps[0].result).toContain("content");
  });

  it("still writes an empty file when that is what was asked", async () => {
    const { wrote, steps } = await run([call({ path: "src/app.ts", content: "" }), "Done."]);
    expect(wrote).toEqual([{ path: "src/app.ts", content: "" }]);
    expect(steps[0].status).toBe("done");
  });

  it("writes what it is given", async () => {
    const { wrote } = await run([call({ path: "notes.md", content: "hello" }), "Done."]);
    expect(wrote).toEqual([{ path: "notes.md", content: "hello" }]);
  });
});
