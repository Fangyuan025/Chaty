/** How long a screenshot stays in the prompt, driven through the real loop.
 *
 *  Both engines resume a media prefill across a NEW picture now, so an older
 *  screenshot costs its tokens and nothing else and is worth keeping. The one
 *  case that is not about cost: a model that accepts a single image per prompt
 *  answers a second live one with `imageTokenCountMismatch` and the turn fails,
 *  so there the old picture always goes. */
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
type Msg = { role: string; content: string; images?: string[] };

/** Two screenshot rounds, then a plain answer. Returns the messages of the
 *  LAST prompt — the one carrying the second screenshot. */
async function twoShots(multiImage: boolean) {
  const rounds = [
    '<tool_call>{"name":"browser_screenshot","arguments":{}}</tool_call>',
    '<tool_call>{"name":"browser_screenshot","arguments":{}}</tool_call>',
    "Both look right, done.",
  ];
  let shot = 0;
  let lastPrompt: Msg[] = [];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const a = args as { request: { messages: Msg[] }; onEvent: { onmessage?: (ev: Ev) => void } };
      lastPrompt = a.request.messages.map((m) => ({ ...m, images: [...(m.images ?? [])] }));
      a.onEvent.onmessage?.({ type: "token", text: rounds.shift() ?? "Done." });
      a.onEvent.onmessage?.({
        type: "done",
        stats: { completionTokens: 4, tokensPerSecond: 50, promptTokens: 10 },
      });
      return null;
    }
    if (cmd === "browser_screenshot") return `/tmp/shot${shot++}.png`;
    return null;
  });
  await runAgentTurn(
    "look at the page", [], "/tmp/ws", "en",
    {
      thinkMode: "off", maxSteps: 8, visionReady: true, multiImage,
      mediaPrefixReuse: true, mediaChunked: false,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: () => {}, onError: (m) => { throw new Error(m); }, onTrace: () => {},
    },
  );
  return lastPrompt;
}

const live = (m: Msg[]) => m.reduce((n, x) => n + (x.images?.length ?? 0), 0);

describe("a second screenshot in the same turn", () => {
  afterEach(() => clearMocks());

  it("stays alongside the first when the model can hold both", async () => {
    const prompt = await twoShots(true);
    expect(live(prompt)).toBe(2);
    expect(prompt.some((m) => m.content.includes("stale screenshot"))).toBe(false);
  });

  it("replaces the first when the model takes one picture at a time", async () => {
    const prompt = await twoShots(false);
    expect(live(prompt)).toBe(1);
    expect(prompt.find((m) => m.images?.length)!.images![0]).toBe("/tmp/shot1.png");
    expect(prompt.some((m) => m.content.includes("stale screenshot"))).toBe(true);
  });
});
