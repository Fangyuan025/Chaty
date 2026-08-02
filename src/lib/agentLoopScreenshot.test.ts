/** Segmented full-page screenshots through the real loop: multiple tile
 *  paths must ALL ride the next turn as images, with the top-to-bottom note
 *  (and the snapshot steer); single-path captures keep the old shape. */
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

async function runShot(shotReply: string) {
  const rounds = [
    '<tool_call>{"name":"browser_screenshot","arguments":{}}</tool_call>',
    "Looks right, done.",
  ];
  let attached = null as { content: string } | null;
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const ch = (args as { onEvent: Chan }).onEvent;
      ch.onmessage?.({ type: "token", text: rounds.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 4, tokensPerSecond: 50, promptTokens: 10 } });
      return null;
    }
    if (cmd === "browser_screenshot") return shotReply;
    return null;
  });
  await runAgentTurn(
    "look at the page", [], "/tmp/ws", "en",
    {
      thinkMode: "off", maxSteps: 6, visionReady: true,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: () => {}, onError: (m) => { throw new Error(m); },
      onTrace: (ev) => {
        if (ev.kind === "inject" && ev.text.includes("browser_screenshot")) {
          attached = { content: ev.text };
        }
      },
    },
  );
  return attached;
}

describe("segmented full-page screenshots", () => {
  afterEach(() => clearMocks());

  it("three tiles → note says 3 segments, nothing omitted, snapshot steer present", async () => {
    const a = await runShot("/tmp/t0.png\n/tmp/t1.png\n/tmp/t2.png");
    expect(a).not.toBeNull();
    expect(a!.content).toContain("3 segments");
    expect(a!.content).toContain("nothing omitted");
    expect(a!.content).toContain("browser_snapshot");
  });

  it("single capture keeps the plain wording", async () => {
    const a = await runShot("/tmp/only.png");
    expect(a).not.toBeNull();
    expect(a!.content).toContain("Screenshot of the current page");
    expect(a!.content).not.toContain("segments");
  });
});
