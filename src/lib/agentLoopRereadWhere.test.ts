/** A file read again for something that is not in it. Issue #25: a 9B read
 *  the whole of a 22-line game.js, then — "to find startGame" — read it again
 *  and again until the turn paused, told each time only that the file had not
 *  changed. What it lacked was where startGame is. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn, codeNames } = await import("./agentLoop");

type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };

const read = `<tool_call>{"name":"read_file","arguments":{"path":"game.js","note":"读取 game.js 中 startGame 函数附近,找到合适的位置初始化预览画布。"}}</tool_call>`;

async function run(grep: (pattern: string) => string) {
  const calls = [read, read, read, "好的,我去 index.html 里改。"];
  const results: string[] = [];
  const patterns: string[] = [];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const a = args as { request: { messages: { role: string; content: string }[] }; onEvent: Chan };
      const last = a.request.messages[a.request.messages.length - 1];
      if (/tool_result/.test(last.content)) results.push(last.content);
      a.onEvent.onmessage?.({ type: "token", text: calls.shift() ?? "完成。" });
      a.onEvent.onmessage?.({ type: "done", stats: { promptTokens: 100, completionTokens: 1, tokensPerSecond: 1, stopReason: "eos" } });
      return null;
    }
    if (cmd === "agent_read_file") return Array.from({ length: 22 }, (_, i) => `${i + 1}\tconst x${i} = ${i};`).join("\n");
    if (cmd === "agent_grep") {
      const p = (args as { pattern: string }).pattern;
      patterns.push(p);
      return grep(p);
    }
    return null;
  });
  await runAgentTurn(
    "加一个预览画布", [], "/tmp/ws", "zh",
    {
      thinkMode: "off", maxSteps: 8, nCtx: 32768,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    { onThinking: () => {}, onAssistantText: () => {}, onStep: () => {}, onFinal: () => {}, onError: (m) => { throw new Error(m); } },
  );
  return { results, patterns };
}

describe("a file read again for what is not in it", () => {
  afterEach(() => clearMocks());

  it("names code the way code spells it", () => {
    expect(codeNames("读取 game.js 中 startGame 函数附近,初始化 preview_canvas,用 read_file")).toEqual(["startGame", "preview_canvas"]);
    expect(codeNames("看看页面的标题")).toEqual([]);
  });

  it("is told where the name it is after actually is", async () => {
    const { results, patterns } = await run(() => "index.html:120:    function startGame() {\n");
    expect(patterns.some((p) => p.includes("startGame"))).toBe(true);
    const refusal = results.find((r) => r.includes("调用被拦截")) ?? "";
    expect(refusal).toContain("game.js 里没有 startGame");
    expect(refusal).toContain("index.html:120:");
  });

  it("is told when the name is nowhere — something to write, not to find", async () => {
    const { results } = await run(() => "(无匹配 / no matches)");
    expect(results.find((r) => r.includes("调用被拦截")) ?? "").toContain("工作区里没有任何地方出现 startGame");
  });
});
