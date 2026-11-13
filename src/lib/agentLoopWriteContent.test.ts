/** A write_file that arrives without its content. A 35B writing a page sent
 *  write_file with only a path, got "missing content", sent it again, met the
 *  repeat breaker — which said the identical write was "already on disk" —
 *  went looking for a file that was never written, and spun. The slip is now
 *  answered without being run, with corrections of its own, and nothing
 *  claims a file exists that does not. */
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

const noContent = '<tool_call>\n<function=write_file>\n<parameter=path>\nindex.html\n</parameter>\n</function>\n</tool_call>';
const withContent =
  '<tool_call>\n<function=write_file>\n<parameter=path>\nindex.html\n</parameter>\n<parameter=content>\n<!DOCTYPE html>\n<p>hi</p>\n</parameter>\n</function>\n</tool_call>';

async function run(rounds: string[]) {
  const script = [...rounds];
  const writes: { path: string; content: string }[] = [];
  const injected: string[] = [];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const a = args as { onEvent: Chan };
      a.onEvent.onmessage?.({ type: "token", text: script.shift() ?? "Done: index.html is written." });
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 50, promptTokens: 10 } });
      return null;
    }
    if (cmd === "agent_write_file") {
      const a = args as { path: string; content: string };
      writes.push({ path: a.path, content: a.content });
      return "wrote index.html";
    }
    return null;
  });
  let final = "";
  await runAgentTurn(
    "make a page", [], "/tmp/ws", "zh",
    {
      thinkMode: "off", maxSteps: 10, temperature: 0.2, toolFormat: "xml",
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: (t) => { final = t; },
      onError: (m) => { throw new Error(m); },
      onTrace: (ev) => { if (ev.kind === "inject") injected.push(ev.text); },
    },
  );
  return { final, writes, injected };
}

describe("a write_file without content", () => {
  afterEach(() => clearMocks());

  it("is not run, is told where the text goes, and the real write after it lands", async () => {
    const { writes, injected } = await run([noContent, noContent, noContent, withContent]);
    expect(writes).toHaveLength(1);
    expect(writes[0].content).toContain("<p>hi</p>");
    const notes = injected.join("\n");
    expect(notes).toContain("没有 content");
    // The third slip suggests a shorter first version, not another tool.
    expect(notes).toContain("精简的完整版本");
    // Never that a file never written is on disk.
    expect(notes).not.toContain("已经原样写入过了");
  });
});
