/** A change while planning is stopped as soon as its name is written — not
 *  after the whole file it carries has been generated, only to be sent back. */
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
type Step = { call: { name: string; args: Record<string, unknown> }; status: string; result?: string };

/** Each round as the chunks it streams in; the engine stops when told to. */
async function run(
  rounds: string[][],
  blocked?: (name: string) => "plan" | "planAsk" | "planTodo" | null,
  planMode = false,
) {
  const script = rounds.map((r) => [...r]);
  const cmds: string[] = [];
  const streamed: string[] = [];
  const asked: string[] = [];
  const prompts: string[] = [];
  let final = "";
  let reason: string | undefined;
  const plans: unknown[] = [];
  let stop = false;
  mockIPC(async (cmd, args) => {
    cmds.push(cmd);
    if (cmd === "cancel_generation") {
      stop = true;
      return null;
    }
    if (cmd === "generate") {
      const a = args as { onEvent: Chan; request: { messages: { content: string }[] } };
      prompts.push(a.request.messages[a.request.messages.length - 1].content);
      stop = false;
      let out = "";
      for (const chunk of script.shift() ?? ["Done."]) {
        if (stop) break;
        out += chunk;
        a.onEvent.onmessage?.({ type: "token", text: chunk });
        await Promise.resolve();
      }
      streamed.push(out);
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 50, promptTokens: 10 } });
      return null;
    }
    return null;
  });
  const steps = new Map<string, Step>();
  await runAgentTurn(
    "plan it", [], "/tmp/ws", "en",
    {
      thinkMode: "off", maxSteps: 8, temperature: 0.2, toolFormat: "xml", blocked, planMode,
      signal: { cancelled: false },
      approve: async () => "plan",
      approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onCalling: () => {},
      onPlan: (p: unknown) => plans.push(p),
      onAskUser: async (q: string) => {
        asked.push(q);
        return "ok";
      },
      onStep: (s) => steps.set(s.id, s as never),
      onFinal: (t, _th, r) => {
        final = t;
        reason = r;
      },
      onError: (m) => { throw new Error(m); },
    },
  );
  return { steps: [...steps.values()], cmds, streamed, asked, prompts, final, reason, plans };
}

const WRITE = [
  "<tool_call>\n<function=write_file>\n",
  "<parameter=path>\nnotes.md\n</parameter>\n",
  "<parameter=content>\n",
  ...Array.from({ length: 50 }, (_, i) => `line ${i}\n`),
  "</parameter>\n</function>\n</tool_call>",
];

describe("a change while planning", () => {
  afterEach(() => clearMocks());

  it("is stopped as its name is written, and sent back unrun", async () => {
    const { steps, cmds, streamed } = await run([WRITE, ["The plan: …"]], (n) => (n === "write_file" ? "plan" : null));
    expect(cmds).toContain("cancel_generation");
    expect(cmds).not.toContain("agent_write_file");
    // Nothing past the name was generated.
    expect(streamed[0]).not.toContain("line 0");
    expect(steps).toHaveLength(1);
    expect(steps[0].call.name).toBe("write_file");
    expect(steps[0].status).toBe("denied");
  });

  it("a question that would stand in for the plan is stopped too, and never asked", async () => {
    const ASK = [
      "<tool_call>\n<function=ask_user>\n",
      "<parameter=question>\nHere is my plan: … Shall I go ahead?\n</parameter>\n</function>\n</tool_call>",
    ];
    const { steps, cmds, streamed, asked, prompts } = await run([ASK, ["The plan: …"]], (n) =>
      n === "ask_user" ? "planAsk" : null,
    );
    expect(cmds).toContain("cancel_generation");
    expect(asked).toEqual([]);
    expect(streamed[0]).not.toContain("Shall I go ahead");
    expect(steps[0].call.name).toBe("ask_user");
    expect(steps[0].status).toBe("denied");
    expect(prompts[1]).toMatch(/don't use ask_user to hand in the plan/);
  });

  it("tried again and again, it is told so, then the turn pauses instead of spinning", async () => {
    const { steps, prompts, final, reason } = await run([WRITE, WRITE, WRITE, WRITE, WRITE, ["never reached"]], (n) =>
      n === "write_file" ? "plan" : null,
    );
    expect(prompts[2]).toMatch(/This is attempt 2/);
    expect(steps).toHaveLength(5);
    expect(reason).toBe("steps");
    expect(final).toMatch(/tried 5 times/);
  });

  it("a todo list is stopped too — it would hold the plan back as unfinished", async () => {
    const TODO = [
      "<tool_call>\n<function=update_plan>\n",
      '<parameter=todos>\n[{"content":"fix it","status":"pending"}]\n</parameter>\n</function>\n</tool_call>',
    ];
    const { steps, plans, prompts } = await run([TODO, ["The plan: …"]], (n) => (n === "update_plan" ? "planTodo" : null), true);
    expect(plans).toEqual([]);
    expect(steps[0].status).toBe("denied");
    expect(prompts[1]).toMatch(/don't lay out todos with update_plan/);
  });

  it("a planning turn's answer is not held back by the wrap-up gate", async () => {
    // The todo list already there, unfinished — what held a plan back.
    const TODO = [
      '<tool_call>\n<function=update_plan>\n<parameter=todos>\n[{"content":"fix it","status":"pending"}]\n</parameter>\n</function>\n</tool_call>',
    ];
    const { final, prompts } = await run([TODO, ["The plan: change line_total."], ["(a nudged second answer)"]], undefined, true);
    expect(final).toBe("The plan: change line_total.");
    expect(prompts).toHaveLength(2);
  });

  it("going round two files, the third read of one is answered from what was read", async () => {
    const rd = (p: string) => [`<tool_call>\n<function=read_file>\n<parameter=path>\n${p}\n</parameter>\n</function>\n</tool_call>`];
    const { cmds, prompts, final } = await run(
      [rd("a.py"), rd("b.py"), rd("a.py"), rd("b.py"), rd("a.py"), ["The plan: …"]],
      undefined,
      true,
    );
    expect(cmds.filter((c) => c === "agent_read_file")).toHaveLength(4);
    expect(prompts[5]).toMatch(/You have read a\.py 2 times this turn/);
    expect(prompts[5]).toMatch(/write the plan as your answer/);
    expect(final).toBe("The plan: …");
  });

  it("without the check, the whole call is written and then sent back", async () => {
    const { streamed, steps } = await run([WRITE, ["The plan: …"]]);
    expect(streamed[0]).toContain("line 49");
    expect(steps[0].status).toBe("denied");
  });
});
