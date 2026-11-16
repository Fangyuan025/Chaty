/** Two steps taken by turns, nothing moving. Qwen3 8B, on a long task, went
 *  back and forth — a call, something else, the call again — and the repeat
 *  breaker, which compares a call only with the one right before it, never
 *  saw anything repeated. And `note`, which the rules ask for as an argument,
 *  was called as a tool of its own, came back "unknown tool", and was called
 *  again two steps on. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn, stepCycle, settledResult, resumeNudge } = await import("./agentLoop");

type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };

const call = (name: string, args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name, arguments: { note: "继续处理。", ...args } })}</tool_call>`;
const edit = call("edit_file", { path: "a.py", old_string: "x = 1", new_string: "x = 2" });
const test = call("bash", { command: "python3 -m pytest -q" });

async function run(outputs: string[], bash: () => string = () => "1 failed") {
  const results: string[] = [];
  const said: string[] = [];
  let bashRuns = 0;
  let final: { text: string; reason?: string; stuck?: { kind: string; steps?: string } } | null = null;
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const a = args as { request: { messages: { role: string; content: string }[] }; onEvent: Chan };
      const last = a.request.messages[a.request.messages.length - 1];
      if (/tool_result/.test(last.content)) results.push(last.content);
      a.onEvent.onmessage?.({ type: "token", text: outputs.shift() ?? "完成。" });
      a.onEvent.onmessage?.({ type: "done", stats: { promptTokens: 100, completionTokens: 1, tokensPerSecond: 1, stopReason: "eos" } });
      return null;
    }
    if (cmd === "agent_edit_file") throw new Error("old_string 在 a.py 中没有找到");
    if (cmd === "agent_bash") {
      bashRuns++;
      return { stdout: bash(), stderr: "", code: 1, timedOut: false, bgId: null };
    }
    return null;
  });
  await runAgentTurn(
    "修好测试", [], "/tmp/ws", "zh",
    {
      thinkMode: "off", maxSteps: 16, nCtx: 32768,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onSaid: (t: string) => said.push(t),
      onFinal: (text: string, _t: unknown, reason?: string, stuck?: { kind: string; steps?: string }) => { final = { text, reason, stuck }; },
      onError: (m: string) => { throw new Error(m); },
    } as never,
  );
  return { results, said, bashRuns, final: final as { text: string; reason?: string; stuck?: { kind: string; steps?: string } } | null };
}

describe("two steps taken by turns", () => {
  afterEach(() => clearMocks());

  it("is a cycle only when both results stayed exactly as they were", () => {
    const A = { key: "edit_file:{}", result: "ERROR: not found" };
    const B = { key: "bash:{}", result: "1 failed" };
    expect(stepCycle([A, B, A, B], A.key)).toEqual({ keys: [A.key, B.key] });
    // The other one next is not a third round of A yet.
    expect(stepCycle([A, B, A, B], B.key)).toBeNull();
    // A result that moved is work going on.
    expect(stepCycle([A, B, A, { ...B, result: "2 failed" }], A.key)).toBeNull();
    // One call over and over is the repeat breaker's.
    expect(stepCycle([A, A, A, A], A.key)).toBeNull();
    expect(stepCycle([A, B, A], A.key)).toBeNull();
    // Two listings by turns belong to the observation streak.
    const L = { key: "list_dir:{\"path\":\"x\"}", result: "a", observe: true };
    const M = { key: "list_dir:{\"path\":\"y\"}", result: "b", observe: true };
    expect(stepCycle([L, M, L, M], L.key)).toBeNull();
    expect(stepCycle([L, B, L, B], L.key)).toEqual({ keys: [L.key, B.key] });
    // Three steps round and round, a job id moving in one result.
    const C = { key: "bash:{\"command\":\"brew install pytest\"}", result: "Error: not writable\n[后台 #1099511627783]" };
    const C2 = { ...C, result: "Error: not writable\n[后台 #1099511627784]" };
    expect(stepCycle([A, B, C, A, B, C2], A.key)).toEqual({ keys: [A.key, B.key, C.key] });
    // How long a run took is not a change; how many failed is.
    expect(settledResult("1 failed in 0.05s")).toBe(settledResult("1 failed in 0.07s"));
    expect(settledResult("1 failed in 0.05s")).not.toBe(settledResult("2 failed in 0.05s"));
  });

  it("is pointed out on the third round, unrun, and paused if it goes on", async () => {
    const { results, bashRuns, final } = await run([edit, test, edit, test, edit, test, edit, test, edit]);
    const told = results.find((r) => r.includes("原样走了两轮")) ?? "";
    expect(told).toContain("edit_file a.py");
    expect(told).toContain("bash `python3 -m pytest -q`");
    // edit, test, edit, test ran; the fifth (an edit) was held; the test after it ran once more.
    expect(bashRuns).toBe(3);
    expect(final?.reason).toBe("steps");
    expect(final?.text).toContain("重复同样几步");
    // "Continue" tells the model what it was going round, not "identical calls".
    expect(final?.stuck?.kind).toBe("cycle");
    const nudge = resumeNudge(final!.stuck as never, true);
    expect(nudge).toContain("同样几步之间反复");
    expect(nudge).toContain("edit_file a.py");
  });

  it("catches three steps round and round, a job id moving in one of them", async () => {
    const chmod = call("bash", { command: "chmod u+w /opt/homebrew" });
    const brew = call("bash", { command: "brew install pytest" });
    let job = 1099511627780;
    const { results, final } = await run(
      [chmod, brew, edit, chmod, brew, edit, chmod, brew, edit, chmod, brew],
      () => `Error: /opt/homebrew/Cellar is not writable\n(job #${job++})`,
    );
    const told = results.find((r) => r.includes("原样走了两轮")) ?? "";
    expect(told).toContain("这 3 步");
    expect(told).toContain("brew install pytest");
    expect(final?.reason).toBe("steps");
  });

  it("leaves a pair alone whose results keep changing", async () => {
    let n = 0;
    const { results, final } = await run([edit, test, edit, test, edit, test, "改不动了,需要你看一下 a.py。"], () => `${++n} failed`);
    expect(results.some((r) => r.includes("原样走了两轮"))).toBe(false);
    expect(final?.reason).not.toBe("steps");
  });
});

describe("note called as a tool", () => {
  afterEach(() => clearMocks());

  it("says what it meant to the user and is told where the word goes", async () => {
    const noteCall = `<tool_call>${JSON.stringify({ name: "note", arguments: { message: "测试还没装上,我先换个办法装 pytest。" } })}</tool_call>`;
    const { results, said } = await run([noteCall, "好的,已经说明情况。"]);
    expect(said).toContain("测试还没装上,我先换个办法装 pytest。");
    const back = results.find((r) => r.includes('name="note"')) ?? "";
    expect(back).toContain("note 不是单独的工具");
    expect(back).not.toContain("未知工具");
  });
});

describe("a command that finished and left something running", () => {
  afterEach(() => clearMocks());

  it("is not called a dev server", async () => {
    const results: string[] = [];
    const outputs = [call("bash", { command: "brew install pytest" }), "装不上,需要你处理一下权限。"];
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const a = args as { request: { messages: { role: string; content: string }[] }; onEvent: Chan };
        const last = a.request.messages[a.request.messages.length - 1];
        if (/tool_result/.test(last.content)) results.push(last.content);
        a.onEvent.onmessage?.({ type: "token", text: outputs.shift() ?? "完成。" });
        a.onEvent.onmessage?.({ type: "done", stats: { promptTokens: 100, completionTokens: 1, tokensPerSecond: 1, stopReason: "eos" } });
        return null;
      }
      if (cmd === "agent_bash")
        return { stdout: "", stderr: "Error: /opt/homebrew/Cellar is not writable.", code: 1, timedOut: false, bgId: 7, detached: true };
      return null;
    });
    await runAgentTurn(
      "装 pytest", [], "/tmp/ws", "zh",
      { thinkMode: "off", maxSteps: 6, nCtx: 32768, signal: { cancelled: false }, approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }) } as never,
      { onThinking: () => {}, onAssistantText: () => {}, onStep: () => {}, onFinal: () => {}, onError: (m: string) => { throw new Error(m); } } as never,
    );
    const r = results.find((x) => x.includes('name="bash"')) ?? "";
    expect(r).toContain("[exit 1]");
    expect(r).toContain("命令本身已经结束");
    expect(r).not.toContain("dev server");
    expect(r).not.toContain("browser_navigate");
  });
});
