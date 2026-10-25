/** A turn goes on the way its first call went: a first call with no word to
 *  the user is sent back once to be made with a `note`, and a note becomes the
 *  step's words — said to the user, kept off the tool's arguments. */
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
type Step = { call: { name: string; args: Record<string, unknown> }; said?: string; status: string };

const call = (name: string, args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;

async function run(rounds: string[], askFirstNote = true) {
  const script = [...rounds];
  const prompts: string[] = [];
  const dirArgs: unknown[] = [];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const a = args as { request: { messages: { content: string }[] }; onEvent: Chan };
      prompts.push(a.request.messages[a.request.messages.length - 1].content);
      a.onEvent.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 50, promptTokens: 10 } });
      return null;
    }
    if (cmd === "agent_list_dir") {
      dirArgs.push(args);
      return "src/\nREADME.md";
    }
    return null;
  });
  const steps = new Map<string, Step>();
  await runAgentTurn(
    "look around", [], "/tmp/ws", "en",
    {
      thinkMode: "off", maxSteps: 10, temperature: 0.2, askFirstNote,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s) => steps.set(s.id, s as never),
      onFinal: () => {},
      onError: (m) => { throw new Error(m); },
    },
  );
  return { steps: [...steps.values()], prompts, dirArgs };
}

describe("a call's note", () => {
  afterEach(() => clearMocks());

  it("a silent first call is sent back once, and the noted one runs", async () => {
    const { steps, prompts, dirArgs } = await run([
      call("list_dir", {}),
      call("list_dir", { note: "Looking at what is in the project first." }),
      call("list_dir", { path: "src" }),
      "All there.",
    ]);
    // The first call never ran; what came back asked for a note.
    expect(prompts[1]).toMatch(/Not run: the user can't see your thinking/);
    expect(dirArgs).toHaveLength(2);
    expect(steps).toHaveLength(2);
    expect(steps[0].said).toBe("Looking at what is in the project first.");
    // The note is said, not passed on.
    expect(steps[0].call.args).not.toHaveProperty("note");
    // Once a turn: the later silent call runs as it is.
    expect(steps[1].said).toBeUndefined();
  });

  it("after two quiet calls, the third quiet one is sent back for a note", async () => {
    const { steps, prompts, dirArgs } = await run([
      call("list_dir", { note: "Starting with the layout." }),
      call("list_dir", { path: "a" }),
      call("list_dir", { path: "b" }),
      call("list_dir", { path: "c" }),
      call("list_dir", { path: "c", note: "a and b are empty; c has the code. Reading it next." }),
      "Done.",
    ]);
    expect(prompts[4]).toMatch(/Not run: two calls in a row without a word/);
    expect(dirArgs).toHaveLength(4);
    expect(steps.map((s) => s.said ?? "")).toEqual([
      "Starting with the layout.",
      "",
      "",
      "a and b are empty; c has the code. Reading it next.",
    ]);
  });

  it("a call made again just as it was runs — one send-back per call", async () => {
    const { steps, prompts, dirArgs } = await run([call("list_dir", {}), call("list_dir", {}), "Done."]);
    expect(prompts[1]).toMatch(/Not run/);
    expect(prompts[2]).not.toMatch(/Not run/);
    expect(dirArgs).toHaveLength(1);
    expect(steps).toHaveLength(1);
  });

  it("a write is never sent back — it would be written again whole", async () => {
    const { steps, prompts } = await run([
      call("write_file", { path: "x.txt", content: "hello" }),
      "Done.",
    ]);
    expect(steps).toHaveLength(1);
    expect(prompts[1]).not.toMatch(/Not run/);
  });

  it("a first call with words before it runs as it is", async () => {
    const { steps, prompts } = await run(["Checking the layout.\n" + call("list_dir", {}), "Fine."]);
    expect(steps).toHaveLength(1);
    expect(steps[0].said).toBe("Checking the layout.");
    expect(prompts[1]).not.toMatch(/Not run/);
  });

  it("off, nothing is sent back", async () => {
    const { steps } = await run([call("list_dir", {}), "Fine."], false);
    expect(steps).toHaveLength(1);
  });
});
