/** Long edits burned whole rounds: a Qwen3.5/3.6 model trained on its own
 *  XML tool-call format fell back to it (or broke the one-line JSON Chaty
 *  asked for), the call was rejected as "not valid", and the model wrote the
 *  whole edit out again — a different way each time, so the identical-repeat
 *  breaker never tripped. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { parseToolCall, describeInvalidCall, runAgentTurn, systemPrompt } = await import("./agentLoop");

describe("the XML tool format, when a turn asks for it", () => {
  afterEach(() => clearMocks());

  it("is what the system prompt teaches — and JSON is left as it was", () => {
    const xml = systemPrompt("/ws", true, "off", undefined, false, false, undefined, undefined, "xml");
    expect(xml).toContain("<function=工具名>");
    expect(xml).toContain("<parameter=参数名>");
    expect(xml).not.toContain('只输出一行 <tool_call>{"name"');
    const json = systemPrompt("/ws", true, "off");
    expect(json).toContain('只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call>');
    expect(json).not.toContain("<function=");
  });

  it("is what a missing-argument correction shows", async () => {
    const script = ["<tool_call>\n<function=read_file>\n</function>\n</tool_call>", "Done."];
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const ch = (args as { onEvent: { onmessage?: (e: unknown) => void } }).onEvent;
        ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
        ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      }
      return null;
    });
    const injects: string[] = [];
    await runAgentTurn(
      "read it", [], "/tmp/ws", "en",
      {
        thinkMode: "off", maxSteps: 6, toolFormat: "xml",
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {}, onFinal: () => {},
        onError: (m) => { throw new Error(m); },
        onTrace: (ev) => { if (ev.kind === "inject") injects.push(ev.text); },
      },
    );
    const note = injects.find((t) => t.includes('missing "path"'));
    expect(note).toContain("<parameter=path>\nsrc/app.ts\n</parameter>");
    expect(note).not.toContain('arguments: {"path"');
  });
});

describe("the model's own XML tool-call format", () => {
  it("is read, values as written — quotes, braces and newlines untouched", () => {
    const raw = [
      "I'll fix the counter.",
      "<tool_call>",
      "<function=edit_file>",
      "<parameter=path>",
      "index.html",
      "</parameter>",
      "<parameter=old_string>",
      '    store.setItem("visits", String(n));',
      "    return n;",
      "</parameter>",
      "<parameter=new_string>",
      '    store.setItem("visits", String(n + 1)); // {"a": 1}',
      "    return n + 1;",
      "</parameter>",
      "</function>",
      "</tool_call>",
    ].join("\n");
    expect(parseToolCall(raw)).toEqual({
      name: "edit_file",
      args: {
        path: "index.html",
        old_string: '    store.setItem("visits", String(n));\n    return n;',
        new_string: '    store.setItem("visits", String(n + 1)); // {"a": 1}\n    return n + 1;',
      },
    });
  });

  it("reads JSON-valued and boolean parameters, and keeps text parameters text", () => {
    const raw =
      "<tool_call>\n<function=multi_edit>\n<parameter=path>\na.py\n</parameter>\n" +
      '<parameter=edits>\n[{"old_string": "x = 1", "new_string": "x = 2"}]\n</parameter>\n' +
      "</function>\n</tool_call>";
    expect(parseToolCall(raw)?.args.edits).toEqual([{ old_string: "x = 1", new_string: "x = 2" }]);
    const b = parseToolCall(
      "<tool_call>\n<function=edit_file>\n<parameter=path>\na.json\n</parameter>\n<parameter=old_string>\n[1, 2]\n</parameter>\n" +
        "<parameter=new_string>\ntrue\n</parameter>\n<parameter=replace_all>\ntrue\n</parameter>\n</function>\n</tool_call>",
    );
    expect(b?.args).toEqual({ path: "a.json", old_string: "[1, 2]", new_string: "true", replace_all: true });
  });

  it("leaves a JSON call a JSON call, even when its content mentions <function=", () => {
    const raw = '<tool_call>{"name":"write_file","arguments":{"path":"n.md","content":"use <function=x> tags"}}</tool_call>';
    expect(parseToolCall(raw)).toEqual({ name: "write_file", args: { path: "n.md", content: "use <function=x> tags" } });
  });
});

describe("an invalid call is described", () => {
  it("as cut off mid-string", () => {
    const d = describeInvalidCall('<tool_call>{"name":"edit_file","arguments":{"path":"a","old_string":"abc', 1, "en");
    expect(d).toMatch(/middle of a string/);
    expect(d).not.toMatch(/in a row/);
  });

  it("with the place the JSON broke", () => {
    const d = describeInvalidCall('<tool_call>{"name":"bash","arguments":{"command":"echo "hi""}}</tool_call>', 1, "en");
    expect(d).toMatch(/near the error: …/);
    expect(d).toContain("echo");
  });

  it("with a smaller way to make the change, from the second failure", () => {
    expect(describeInvalidCall("<tool_call>{", 2, "zh")).toMatch(/连续 2 次/);
    expect(describeInvalidCall("<tool_call>\n<function=edit_file>\n<parameter=path>\na", 1, "zh")).toMatch(/<\/parameter>/);
  });
});

describe("invalid calls in a row", () => {
  afterEach(() => clearMocks());

  it("pause the turn after four, even when each is broken differently", async () => {
    const script = [1, 2, 3, 4, 5].map(
      (i) => `<tool_call>{"name":"edit_file","arguments":{"path":"a.py","old_string":"line ${i}`,
    );
    let calls = 0;
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        calls++;
        const ch = (args as { onEvent: { onmessage?: (e: unknown) => void } }).onEvent;
        ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
        ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      }
      return null;
    });
    let final = "";
    let reason: string | undefined;
    const injects: string[] = [];
    await runAgentTurn(
      "edit a.py",
      [],
      "/tmp/ws",
      "en",
      {
        thinkMode: "off", maxSteps: 12,
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
        onFinal: (t, _th, r) => { final = t; reason = r; },
        onError: (m) => { throw new Error(m); },
        onTrace: (ev) => { if (ev.kind === "inject") injects.push(ev.text); },
      },
    );
    expect(calls).toBe(4);
    expect(reason).toBe("steps");
    expect(final).toMatch(/paused/);
    expect(injects.filter((t) => /could not be parsed/.test(t))).toHaveLength(3);
    expect(injects.some((t) => /in a row/.test(t))).toBe(true);
  });
});
