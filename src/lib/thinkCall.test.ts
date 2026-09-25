/** Tool calls written inside the reasoning. Qwen3.5 does it: the call goes
 *  out mid-thought, generation stops at its closer, and the thought never
 *  closes. It is run — the model acted — but its markup is not thought to
 *  show a person, and where the model went on to call something after its
 *  thought, that later call is the one it settled on. Driven through the
 *  REAL loop, with the engine mocked at the IPC boundary. */
import { describe, expect, it } from "vitest";
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= { getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0 };
g.navigator ??= { userAgent: "chaty-test" };
const { mockIPC } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");
type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };

async function run(replies: string[], opts: Record<string, unknown> = {}) {
  const sent: { role: string; content: string }[][] = [];
  const cmds: string[] = [];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "generate") {
      const req = a.request as { messages: { role: string; content: string }[]; stop?: string[] };
      sent.push(req.messages);
      const ch = a.onEvent as Chan;
      const reply = replies[sent.length - 1] ?? "Done.";
      ch.onmessage?.({ type: "token", text: reply });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 10, tokensPerSecond: 50, promptTokens: 50 } });
      return null;
    }
    if (cmd.endsWith("_reap")) return [];
    if (cmd.startsWith("agent_")) { cmds.push(cmd + " " + JSON.stringify(a).slice(0, 80)); return cmd === "agent_read_file" ? "1\tfn a() {}" : "ok"; }
    return null;
  });
  const steps: string[] = [];
  const thinking: string[] = [];
  let final = "";
  await runAgentTurn("read a.rs", [], "/tmp/ws", "en", {
    thinkMode: "normal", maxSteps: 4, toolFormat: "xml", supportsThinking: true,
    signal: { cancelled: false }, approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }), ...opts,
  } as never, {
    onThinking: (t: string) => thinking.push(t), onAssistantText: () => {},
    onStep: (s: { status: string; result?: string; call: { name: string; args: unknown } }) => { if (s.status !== "running") steps.push(`${s.call.name} ${JSON.stringify(s.call.args)} ${s.status}`); },
    onStats: () => {}, onPlan: () => {}, onCompacted: () => {},
    onError: (e: string) => { throw new Error(e); },
    onFinal: (t: string) => { final = t; }, onStepText: () => {}, onTrace: () => {},
  } as never);
  return { sent, cmds, steps, thinking, final };
}

describe("a call inside the reasoning", () => {
  it("runs when the thought stopped at it, and is not shown as thought", async () => {
    const r = await run(["<think>\nI should read the file.\n<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function>\n"]);
    expect(r.steps).toEqual(['read_file {"path":"a.rs"} done']);
    const shown = r.thinking.filter(Boolean);
    expect(shown.length).toBeGreaterThan(0);
    for (const t of shown) expect(t).not.toMatch(/<tool_call>|<function=|<parameter=/);
    expect(shown).toContain("I should read the file.");
  });

  it("gives way to the call made after the thought", async () => {
    const r = await run([
      "<think>\nplan: <tool_call>\n<function=list_dir>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>\n</think>\n<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function>\n",
    ]);
    expect(r.steps).toEqual(['read_file {"path":"a.rs"} done']);
    expect(r.cmds.some((c) => c.startsWith("agent_list_dir"))).toBe(false);
  });

  it("does not stand in for a call after the thought that does not read", async () => {
    const r = await run([
      "<think>\n<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>\n</think>\n<tool_call>\n<function=read_file>\n<parameter=path>",
    ]);
    expect(r.cmds.some((c) => c.startsWith("agent_list_dir"))).toBe(false);
  });

  it("is not run when the thought only quoted it and went on to answer", async () => {
    const r = await run([
      "<think>\nI could call <tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call> but the question is already answered.\n</think>\nThe file is fine.",
    ]);
    expect(r.steps).toEqual([]);
    expect(r.final).toBe("The file is fine.");
  });
});
