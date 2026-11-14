/** The same plan, sent again and again. Evidence: a Qwen3.5 4B asked for a
 *  web page made "了解当前工作区状态" its first item, thought "I'll use
 *  understand_repo" at every step, and sent the plan it had just sent nine
 *  times until the turn paused. The refusals told it to "write the first file,
 *  or mkdir" — not what its first item was — and left its context as it was,
 *  so every next sample was the same one. */
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
type Step = { call: { name: string }; status: string; result?: string };

const plan = (first: string) =>
  `<tool_call>{"name":"update_plan","arguments":{"todos":[{"content":"${first}","status":"in_progress"},{"content":"创建主页面 HTML 和 CSS","status":"pending"},{"content":"测试并优化体验","status":"pending"}]}}</tool_call>`;

async function run(script: string[]) {
  const calls = [...script];
  const prompts: string[] = [];
  let overviews = 0;
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      prompts.push(JSON.stringify(args));
      const a = args as { onEvent: Chan };
      a.onEvent.onmessage?.({ type: "token", text: calls.shift() ?? "做好了。" });
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 50, promptTokens: 10 } });
      return null;
    }
    if (cmd === "agent_understand_repo") {
      overviews++;
      return "Workspace: /tmp/ws\nFiles: (empty)\nNo README, no manifest.";
    }
    return null;
  });
  const steps: Step[] = [];
  let final = "";
  let reason: string | undefined;
  await runAgentTurn(
    "做一个0到3岁的早教网页", [], "/tmp/ws", "zh",
    {
      thinkMode: "off", maxSteps: 30, temperature: 0.2,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s) => { if (s.status !== "running") steps.push({ call: { name: s.call.name }, status: s.status, result: s.result }); },
      onFinal: (t, _th, r) => { final = t; reason = r; },
      onError: (m) => { throw new Error(m); },
    },
  );
  return { steps, final, reason, prompts, overviews };
}

const thought = "<think>根据我的计划,第一步应该是使用 understand_repo 来了解当前工作区的情况。</think>\n";

describe("a plan sent again before anything was looked at", () => {
  it("is answered with the look the model set out to take, once", async () => {
    const looped = Array.from({ length: 9 }, () => thought + plan("了解当前工作区状态"));
    const { steps, prompts, overviews, reason } = await run(looped);
    // The look ran, once, and shows as what it is.
    expect(overviews).toBe(1);
    const looked = steps.filter((s) => s.call.name === "understand_repo");
    expect(looked).toHaveLength(1);
    expect(looked[0].status).toBe("done");
    // On the first repeat — not after a string of refusals.
    expect(steps[0].call.name).toBe("understand_repo");
    // The model was given the workspace, and pointed past the item it answers.
    const after = prompts[2];
    expect(after).toContain("No README, no manifest.");
    expect(after).toContain("现在做下一项「创建主页面 HTML 和 CSS」");
    // A model that still will not move is paused as before.
    expect(reason).toBe("steps");
  });

  it("moves on once the model acts on what it was given", async () => {
    const write = `<tool_call>{"name":"write_file","arguments":{"path":"index.html","content":"<!doctype html><title>早教</title>"}}</tool_call>`;
    const { steps, reason, overviews } = await run([
      thought + plan("了解当前工作区状态"),
      thought + plan("了解当前工作区状态"),
      write,
    ]);
    expect(overviews).toBe(1);
    expect(steps.map((s) => s.call.name)).toEqual(["understand_repo", "write_file"]);
    expect(reason).not.toBe("steps");
  });

  it("does not look for a model whose plan starts by making something", async () => {
    const looped = Array.from({ length: 3 }, () => plan("创建主页面 HTML 和 CSS"));
    const { overviews, prompts } = await run(looped);
    expect(overviews).toBe(0);
    // The move it is shown is the one its item names: a file.
    const refusal = prompts[2].slice(prompts[2].lastIndexOf("调用被拦截"));
    expect(refusal).toMatch(/格式如:[\s\S]*write_file[\s\S]*index\.html/);
    expect(refusal).not.toContain("understand_repo");
    expect(refusal).not.toContain("mkdir");
  });

  it("names the tool the model's own thinking settled on", async () => {
    const think = "<think>我应该用 write_file 先写出 index.html。</think>\n";
    const looped = Array.from({ length: 3 }, () => think + plan("规划早教网页功能结构"));
    const { prompts } = await run(looped);
    expect(prompts[2]).toContain("你在思考里已经说了要用 write_file");
  });
});
