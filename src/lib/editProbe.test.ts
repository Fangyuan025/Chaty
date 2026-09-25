/** An edit whose old_string is not in the file is stopped while it is being
 *  written — before new_string, which is usually the larger half — and the
 *  next step is handed the report the finished call would have earned. One
 *  that can land is never stopped. Driven through the REAL loop, with the
 *  engine and the file check mocked at the IPC boundary. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");

type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };
const FILE = "fn a() {\n    let x = 1;\n    let y = 2;\n}\n";

/** Split the way a tokenizer would: a few characters at a time. */
function tokens(s: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < s.length; i += 4) out.push(s.slice(i, i + 4));
  return out;
}

async function run(firstReply: string) {
  const sent: string[][] = [];
  let cancelled = 0;
  const checks: { old: string; done: boolean }[] = [];
  let emitted = "";
  let edits = 0;
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "generate") {
      const req = a.request as { messages: { content: string }[] };
      sent.push(req.messages.map((m) => m.content));
      const ch = a.onEvent as Chan;
      const reply = sent.length === 1 ? firstReply : "Done.";
      const before = cancelled;
      for (const t of tokens(reply)) {
        if (cancelled > before) break;
        emitted += t;
        ch.onmessage?.({ type: "token", text: t });
        // Let the check's promise settle between tokens, as a real stream does.
        await new Promise((r) => setTimeout(r, 0));
      }
      ch.onmessage?.({ type: "done", stats: { completionTokens: 10, tokensPerSecond: 50, promptTokens: 50 } });
      return null;
    }
    if (cmd === "cancel_generation") {
      cancelled++;
      return null;
    }
    if (cmd === "agent_edit_check") {
      const old = String(a.oldString);
      checks.push({ old, done: a.done === true });
      // Stand-in for the real judgement: a line not in the file is a miss.
      const lines = old.split("\n").map((l) => l.trim()).filter(Boolean);
      const bad = lines.find((l) => !FILE.includes(l));
      if (bad) throw `old_string not found — closest place is lines 1-4 (report for ${bad})`;
      return null;
    }
    if (cmd === "agent_edit_file" || cmd === "agent_multi_edit") {
      edits++;
      return "edited a.rs";
    }
    if (cmd === "agent_read_file_raw") return FILE;
    return null;
  });
  const steps: { status: string; result?: string; name: string }[] = [];
  await runAgentTurn(
    "change x",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off",
      maxSteps: 4,
      toolFormat: "xml",
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s: { status: string; result?: string; call: { name: string } }) => {
        if (s.status !== "running") steps.push({ status: s.status, result: s.result, name: s.call.name });
      },
      onStats: () => {}, onPlan: () => {}, onCompacted: () => {},
      onError: (e: string) => { throw new Error(e); },
      onFinal: () => {}, onStepText: () => {},
    } as never,
  );
  return { sent, cancelled, checks, emitted, steps, edits };
}

const BIG_NEW = Array.from({ length: 60 }, (_, i) => `    let v${i} = ${i};`).join("\n");

describe("edit probe", () => {
  it("stops an edit whose old_string is not in the file, before new_string", async () => {
    const call =
      "<tool_call>\n<function=edit_file>\n<parameter=path>\na.rs\n</parameter>\n<parameter=old_string>\n" +
      "class Widget extends Base {\n  render(props) {\n    return null;\n  }\n</parameter>\n" +
      `<parameter=new_string>\n${BIG_NEW}\n</parameter>\n</function>\n</tool_call>`;
    const r = await run(call);
    expect(r.cancelled).toBe(1);
    // Cut while old_string was still being written — none of new_string went out.
    expect(r.emitted).not.toContain("let v0");
    expect(r.edits).toBe(0);
    expect(r.steps[0]).toMatchObject({ status: "error", name: "edit_file" });
    expect(r.steps[0].result).toContain("not run");
    // The next step sees the report, and the call as it was written.
    const second = r.sent[1].join("\n");
    expect(second).toContain("closest place is lines 1-4");
    expect(second).toContain("class Widget extends Base {");
  });

  it("never stops an edit that can land", async () => {
    const call =
      "<tool_call>\n<function=edit_file>\n<parameter=path>\na.rs\n</parameter>\n<parameter=old_string>\n" +
      "fn a() {\n    let x = 1;\n    let y = 2;\n</parameter>\n" +
      `<parameter=new_string>\n${BIG_NEW}\n</parameter>\n</function>\n</tool_call>`;
    const r = await run(call);
    expect(r.cancelled).toBe(0);
    expect(r.edits).toBe(1);
    // Judged as the lines arrived, and once more when old_string was finished.
    expect(r.checks.some((c) => !c.done)).toBe(true);
    expect(r.checks.some((c) => c.done)).toBe(true);
  });

  it("never runs an edit that says nothing about what to put in", async () => {
    // Read as "", a missing new_string deleted what old_string matched.
    const single =
      "<tool_call>\n<function=edit_file>\n<parameter=path>\na.rs\n</parameter>\n<parameter=old_string>\n    let x = 1;\n</parameter>\n</function>\n</tool_call>";
    let r = await run(single);
    expect(r.edits).toBe(0);
    expect(r.sent[1].join("\n")).toContain("has no new_string");
    const items =
      '<tool_call>\n<function=edit_file>\n<parameter=path>\na.rs\n</parameter>\n<parameter=edits>\n[{"old_string": "let x = 1;", "new_string": "let x = 2;"}, {"old_string": "let y = 2;", "text": "let y = 3;"}]\n</parameter>\n</function>\n</tool_call>';
    r = await run(items);
    expect(r.edits).toBe(0);
    expect(r.sent[1].join("\n")).toContain("edit 2 of 2 has no new_string");
    // An empty one is a deletion, and runs.
    r = await run(single.replace("</function>", "<parameter=new_string>\n</parameter>\n</function>"));
    expect(r.edits).toBe(1);
  });

  it("says an edits array did not parse, not that old_string is missing", async () => {
    const call =
      '<tool_call>\n<function=edit_file>\n<parameter=path>\na.rs\n</parameter>\n<parameter=edits>\n[{"old_string": "let x = 1;", "new_string": "let s = "x";"}]\n</parameter>\n</function>\n</tool_call>';
    const r = await run(call);
    expect(r.edits).toBe(0);
    const next = r.sent[1].join("\n");
    expect(next).toContain("edits could not be read");
    expect(next).not.toContain('missing "old_string"');
  });
});
