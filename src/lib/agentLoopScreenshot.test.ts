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

let lastStepImage: string | undefined;

async function runShot(shotReply: string, mediaChunked = false) {
  lastStepImage = undefined;
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
      thinkMode: "off", maxSteps: 6, visionReady: true, mediaChunked,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s: { image?: string }) => {
        if (s.image) lastStepImage = s.image;
      },
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

describe("how many tiles ride in one prompt", () => {
  afterEach(() => clearMocks());

  // A prompt carrying pictures is one forward pass over everything up to the
  // last image — it cannot be chunked the way text is, so the tile count is
  // the part of that pass this code controls. Thirteen of them cost 7922
  // prompt tokens and 147 seconds on Qwen3.6 35B with an otherwise empty
  // conversation; behind a real transcript the same round drove the machine
  // into swap.
  it("a very tall page attaches only the first few, and says so", async () => {
    const a = await runShot(
      Array.from({ length: 13 }, (_, i) => `/tmp/t${i}.png`).join("\n"),
    );
    expect(a).not.toBeNull();
    // It still reports what the page actually needed…
    expect(a!.content).toContain("13 segments");
    // …and is explicit that it did not send all of them, with the way on.
    expect(a!.content).toContain("Only the first 4 segments");
    expect(a!.content).toContain("browser_scroll");
  });

  it("an engine that feeds media incrementally gets the whole page", async () => {
    // llama.cpp pays one chunk per tile, so the count barely moves the cost of
    // the round; capping would lose coverage to solve a problem it does not
    // have. The engine that takes the span in a single pass still caps, even
    // though it now resumes the transcript beneath the pictures.
    const a = await runShot(
      Array.from({ length: 13 }, (_, i) => `/tmp/t${i}.png`).join("\n"),
      true,
    );
    expect(a).not.toBeNull();
    expect(a!.content).toContain("13 segments");
    expect(a!.content).not.toContain("Only the first");
  });

  it("the step card gets the whole page, not the first segment", async () => {
    // The two readers of one capture want different things: the model is fed
    // segments because a picture is one forward pass, and the card is not.
    const a = await runShot(
      ["full:/tmp/whole.png", "/tmp/t0.png", "/tmp/t1.png", "/tmp/t2.png"].join("\n"),
    );
    expect(lastStepImage).toBe("/tmp/whole.png");
    // …and the marker never reaches the model as if it were a picture.
    expect(a!.content).toContain("3 segments");
    expect(a!.content).not.toContain("whole.png");
  });

  it("a capture with no full-page marker still shows something", async () => {
    // browser_snapshot returns a bare path, and so did screenshot before the
    // marker existed.
    await runShot("/tmp/only.png");
    expect(lastStepImage).toBe("/tmp/only.png");
  });

  it("a page inside the cap sends everything, with no apology", async () => {
    const a = await runShot("/tmp/t0.png\n/tmp/t1.png\n/tmp/t2.png\n/tmp/t3.png");
    expect(a).not.toBeNull();
    expect(a!.content).toContain("4 segments");
    expect(a!.content).not.toContain("Only the first");
  });
});
