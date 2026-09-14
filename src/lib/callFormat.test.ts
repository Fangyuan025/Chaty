/** Each model writes tool calls the way its chat template trained it; a family
 *  whose template names no format falls back to the one the user picked. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { resolveToolFormat, callRule, argsExample, setCallFormat, oneCall } = await import("./callFormat");
const { parseToolCall, parseGemmaToolCall, systemPrompt } = await import("./agentLoop");

describe("the format a turn uses", () => {
  it("is the model's own under auto, the fallback for an unknown family, and a manual pick otherwise", () => {
    expect(resolveToolFormat("auto", "gemma", "xml")).toBe("gemma");
    expect(resolveToolFormat("auto", "json", "xml")).toBe("json");
    expect(resolveToolFormat("auto", null, "xml")).toBe("xml");
    expect(resolveToolFormat("auto", undefined, "json")).toBe("json");
    expect(resolveToolFormat("lfm", "gemma", "xml")).toBe("lfm");
  });
});

describe("each format is taught in its own shape", () => {
  afterEach(() => setCallFormat("json"));

  it("in the system prompt", () => {
    const rule = (f: "json" | "xml" | "gemma" | "lfm") => systemPrompt("/ws", false, "off", undefined, false, false, undefined, undefined, f);
    expect(rule("gemma")).toContain('<|tool_call>call:tool_name{argument_name:<|"|>text value<|"|>');
    expect(rule("lfm")).toContain("<|tool_call_start|>[tool_name(");
    expect(rule("xml")).toContain("<function=tool_name>");
    expect(rule("json")).toContain('<tool_call>{"name":"tool","arguments":{...}}</tool_call>');
    for (const f of ["gemma", "lfm", "xml"] as const) expect(rule(f)).not.toContain('<tool_call>{"name"');
    expect(callRule(true, "gemma")).toContain("<|\"|>");
  });

  it("in a correction's example", () => {
    const ex = '{"path":"src/app.ts","limit":20,"edits":[{"old_string":"a"}]}';
    setCallFormat("gemma");
    expect(argsExample(ex)).toBe('path:<|"|>src/app.ts<|"|>,limit:20,edits:[{old_string:<|"|>a<|"|>}]');
    expect(oneCall(false)).toContain("<|tool_call>call:");
    setCallFormat("lfm");
    expect(argsExample('{"path":"src/app.ts","limit":20}')).toBe('path="src/app.ts", limit=20');
    setCallFormat("xml");
    expect(argsExample('{"path":"src/app.ts"}')).toBe("<parameter=path>\nsrc/app.ts\n</parameter>");
    setCallFormat("json");
    expect(argsExample('{"path":"src/app.ts"}')).toBe('arguments: {"path":"src/app.ts"}');
  });
});

describe("Gemma 4's own tool call", () => {
  it("is read — text as written, however many lines and quotes", () => {
    const raw =
      "<|channel>thought\nfix it<channel|>" +
      '<|tool_call>call:edit_file{new_string:<|"|>    store.setItem("visits", String(n + 1));\n    return n + 1;<|"|>,' +
      'old_string:<|"|>    return n;<|"|>,path:<|"|>index.html<|"|>}<tool_call|>';
    expect(parseToolCall(raw)).toEqual({
      name: "edit_file",
      args: {
        new_string: '    store.setItem("visits", String(n + 1));\n    return n + 1;',
        old_string: "    return n;",
        path: "index.html",
      },
    });
  });

  it("with numbers, booleans, arrays and objects as the template writes them", () => {
    const raw =
      '<|tool_call>call:multi_edit{edits:[{new_string:<|"|>b<|"|>,old_string:<|"|>a<|"|>,replace_all:true}],path:<|"|>x.py<|"|>}<tool_call|>';
    expect(parseGemmaToolCall(raw)?.args).toEqual({
      edits: [{ new_string: "b", old_string: "a", replace_all: true }],
      path: "x.py",
    });
    expect(parseGemmaToolCall('<|tool_call>call:read_file{limit:40,offset:10,path:<|"|>a.ts<|"|>}<tool_call|>')?.args).toEqual({
      limit: 40,
      offset: 10,
      path: "a.ts",
    });
  });

  it("an empty call reads as one — with no arguments — and a broken one as nothing", () => {
    expect(parseGemmaToolCall("<|tool_call>call:list_dir{}<tool_call|>")).toEqual({ name: "list_dir", args: {} });
    expect(parseGemmaToolCall('<|tool_call>call:write_file{path:<|"|>a.txt<|"|>,content:<|"|>never closed')).toBeNull();
  });
});
