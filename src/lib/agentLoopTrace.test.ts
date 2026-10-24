/** The onTrace diagnostic instrument: raw model output per round + injected
 *  corrections must reach the bench transcript, because failed calls that
 *  never execute (missing-arg ladder rungs 1-2, parse retries) produce NO
 *  step card — without onTrace they are invisible to spin forensics. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

// Same shim the bench runner uses: mockIPC + the loop expect a window global.
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

/** Stream a canned raw string as one generation round, then finish. */
function streamRound(ch: Chan, raw: string) {
  ch.onmessage?.({ type: "token", text: raw });
  ch.onmessage?.({
    type: "done",
    stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 },
  });
}

describe("agentLoop onTrace instrument", () => {
  const rounds: string[] = [];
  beforeEach(() => {
    rounds.length = 0;
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const ch = (args as { onEvent: Chan }).onEvent;
        streamRound(ch, rounds.shift() ?? "all done here.");
        return null;
      }
      return null; // agent_set_lang and friends: accept quietly
    });
  });
  afterEach(() => clearMocks());

  it("captures raw rounds and correction injects that produce no step card", async () => {
    rounds.push(
      // Round 1: the classic spin shape — empty args on a required-arg tool.
      // The ladder corrects WITHOUT executing: no step, only an inject.
      '<tool_call>{"name":"search_code","arguments":{}}</tool_call>',
      // Round 2: clean finish.
      "Everything is finished.",
    );
    const traces: { kind: string; text: string }[] = [];
    const steps: string[] = [];
    await runAgentTurn(
      "find the config parser",
      [],
      "/tmp/ws",
      "en",
      {
        thinkMode: "off",
        maxSteps: 10,
        signal: { cancelled: false },
        approve: async () => true,
        approveDir: async () => false,
        approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {},
        onAssistantText: () => {},
        onStep: (s) => steps.push(`${s.call.name}:${s.status}`),
        onFinal: () => {},
        onError: (m) => {
          throw new Error(`loop errored: ${m}`);
        },
        onTrace: (ev) => traces.push(ev),
      },
    );

    const raws = traces.filter((t) => t.kind === "raw");
    const injects = traces.filter((t) => t.kind === "inject");
    // Both generation rounds captured, verbatim.
    expect(raws).toHaveLength(2);
    expect(raws[0].text).toContain('"search_code"');
    // The ladder's correction is an inject (rung 1: concrete example) and the
    // empty-args call was never executed as a step.
    expect(injects.length).toBeGreaterThanOrEqual(1);
    expect(injects.some((i) => i.text.includes("search_code"))).toBe(true);
    expect(steps.some((s) => s.startsWith("search_code"))).toBe(false);
  });

  it("ladder cooldown: real progress with other tools re-arms a slipped tool at rung 1", async () => {
    const bash = (n: number) => `<tool_call>{"name":"bash","arguments":{"command":"echo ${n}"}}</tool_call>`;
    const empty = '<tool_call>{"name":"search_code","arguments":{}}</tool_call>';
    rounds.push(empty, empty, bash(1), bash(2), bash(3), bash(4), empty, "All wrapped up.");
    const traces: { kind: string; text: string }[] = [];
    await runAgentTurn(
      "explore the repo", [], "/tmp/ws", "en",
      {
        thinkMode: "off", maxSteps: 12,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: () => {},
        onError: (m) => { throw new Error(`loop errored: ${m}`); },
        onTrace: (ev) => traces.push(ev),
      },
    );
    const slips = traces.filter((t) => t.kind === "inject" && t.text.includes("search_code"));
    // Third slip lands AFTER four productive rounds → rung 1 again (concrete
    // example), not the disable notice.
    const last = slips[slips.length - 1].text;
    expect(last).toContain("re-issue");
    expect(last).not.toContain("temporarily disabled");
  });
});

describe("a step keeps what the model said with its call", () => {
  afterEach(() => clearMocks());

  it("the words beside a call land on its step, the last answer on the turn", async () => {
    const rounds = [
      // Words, then the call — what a model writes with thinking off.
      'The file is in place; now I run it.\n<tool_call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>',
      // A summary written before a last call: the words a turn ended on, in effect.
      '<think>all done</think>\n\nSummary: hello.py prints hi.\n<tool_call>{"name":"update_plan","arguments":{"todos":[{"content":"run","status":"completed"}]}}</tool_call>',
      "Done.",
    ];
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        streamRound((args as { onEvent: Chan }).onEvent, rounds.shift() ?? "Done.");
        return null;
      }
      if (cmd === "agent_list_dir") return "hello.py";
      return null;
    });
    const steps = new Map<string, { name: string; said?: string }>();
    const saidWithoutCard: string[] = [];
    let final = "";
    await runAgentTurn(
      "make hello.py and run it",
      [],
      "/tmp/ws",
      "en",
      {
        thinkMode: "off",
        maxSteps: 10,
        signal: { cancelled: false },
        autoRunReadOnly: true,
        approve: async () => true,
        approveDir: async () => false,
        approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {},
        onAssistantText: () => {},
        onStep: (s) => steps.set(s.id, { name: s.call.name, said: s.said }),
        onSaid: (t) => saidWithoutCard.push(t),
        onFinal: (f) => {
          final = f;
        },
        onError: (m) => {
          throw new Error(`loop errored: ${m}`);
        },
      },
    );
    const byName = Object.fromEntries([...steps.values()].map((s) => [s.name, s.said]));
    expect(byName.list_dir).toBe("The file is in place; now I run it.");
    // update_plan has no card: its words come through on their own.
    expect(byName.update_plan).toBeUndefined();
    expect(saidWithoutCard).toEqual(["Summary: hello.py prints hi."]);
    expect(final).toBe("Done.");
  });
});
