/** XML tool calls as models actually write them. The case that mattered: a
 *  Qwen3.5 4B wrote `<parameter>old_string>` for the second argument of every
 *  edit — the `=` turned into `>` — and the argument was dropped without a
 *  word. It was told "old_string is missing", wrote the same thing again, and
 *  every edit in Code mode failed. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { parseXmlToolCall, parseToolCall, parseBareJsonCall, xmlRunsOn } = await import("./agentLoop");

describe("parseXmlToolCall", () => {
  it("reads the template's own form", () => {
    const c = parseXmlToolCall(
      "<tool_call>\n<function=edit_file>\n<parameter=path>\na.ts\n</parameter>\n<parameter=old_string>\nx\n</parameter>\n</function>\n</tool_call>",
    );
    expect(c).toEqual({ name: "edit_file", args: { path: "a.ts", old_string: "x" } });
  });

  it("keeps an argument opened with > instead of = (verbatim model output)", () => {
    const raw = `<tool_call>
<function=edit_file>
<parameter=path>
config.json</parameter>
<parameter>old_string>
{
  "name": "demo",
  "port": 3000
}</parameter>
<parameter=new_string>
{
  "name": "demo",
  "port": 8080
}</parameter>
</function>
</tool_call>`;
    const c = parseXmlToolCall(raw)!;
    expect(c.name).toBe("edit_file");
    expect(c.args.path).toBe("config.json");
    expect(c.args.old_string).toBe('{\n  "name": "demo",\n  "port": 3000\n}');
    expect(c.args.new_string).toBe('{\n  "name": "demo",\n  "port": 8080\n}');
  });

  it("reads the other spellings models use", () => {
    expect(parseXmlToolCall('<function=write_file><parameter name="path">a.md</parameter><parameter name=content>hi</parameter></function>')?.args)
      .toEqual({ path: "a.md", content: "hi" });
    expect(parseXmlToolCall('<function="read_file"><parameter="path">a.md</parameter></function>')?.args).toEqual({ path: "a.md" });
    expect(parseXmlToolCall("<function>read_file>\n<parameter=path>a.md</parameter></function>")?.name).toBe("read_file");
  });

  it("never reads a value as a name", () => {
    // `<parameter>` followed by a line of text is not a name opening.
    expect(parseXmlToolCall("<function=write_file>\n<parameter=path>a</parameter>\n<parameter>\ncontent</parameter></function>")?.args)
      .toEqual({ path: "a" });
  });
});

describe("other ways calls are written", () => {
  it("reads one element per argument (Qwen2 7B, verbatim)", () => {
    const raw = '<tool_call>\n<function=edit_file>\n<path>config.json</path>\n<old_string>{"port":</old_string>\n<new_string>{"port": "8080"</new_string>\n</edit_file>\n</tool_call>';
    expect(parseToolCall(raw)).toEqual({
      name: "edit_file",
      args: { path: "config.json", old_string: '{"port":', new_string: '{"port": "8080"' },
    });
  });

  it("does not take a markup value apart when the arguments are <parameter>s", () => {
    const raw = "<function=write_file>\n<parameter=path>i.html</parameter>\n<parameter=content>\n<p>hi</p>\n</parameter>\n</function>";
    expect(parseXmlToolCall(raw)?.args).toEqual({ path: "i.html", content: "<p>hi</p>" });
  });

  it("takes an unnamed arguments object when the words name one tool (EXAONE 4, verbatim)", () => {
    const raw = '<tool_call>用write_file工具新建文件 hello.txt,内容一行为"hi from chaty"。args:{ "path": "hello.txt", "content": "hi from chaty" }</tool_call>';
    expect(parseToolCall(raw)).toEqual({ name: "write_file", args: { path: "hello.txt", content: "hi from chaty" } });
  });

  it("does not guess when the words name two tools", () => {
    const raw = '<tool_call>先用 read_file 再用 write_file。args:{ "path": "a" }</tool_call>';
    expect(parseToolCall(raw)).toBeNull();
  });
});

describe("a call with no tags", () => {
  it("is a call when the reply is nothing but one (QwQ-32B, verbatim)", () => {
    const raw = '<think>\nThe user wants a file.\n</think>\n{"name":"write_file","arguments":{"path":"hello.txt","content":"hi from chaty"}}';
    expect(parseToolCall(raw)).toEqual({ name: "write_file", args: { path: "hello.txt", content: "hi from chaty" } });
    expect(parseBareJsonCall('```json\n{"name":"read_file","arguments":{"path":"a"}}\n```')?.name).toBe("read_file");
  });

  it("is prose when it sits inside an explanation, names no tool, or has no arguments", () => {
    expect(parseBareJsonCall('You could call {"name":"read_file","arguments":{"path":"a"}} next.')).toBeNull();
    expect(parseBareJsonCall('{"name":"launch_rocket","arguments":{}}')).toBeNull();
    expect(parseBareJsonCall('{"name":"read_file"}')).toBeNull();
    expect(parseBareJsonCall('{"port": 8080}')).toBeNull();
  });
});

describe("xmlRunsOn", () => {
  it("does not cut a value opened with the > spelling", () => {
    const open = "<tool_call>\n<function=write_file>\n<parameter=path>i.html</parameter>\n<parameter>content>\n<html><body><div></div></body></html></div></span></p>";
    expect(xmlRunsOn(open)).toBe(false);
  });

  it("still cuts a finished call that keeps writing closers", () => {
    const done = "<tool_call>\n<function=read_file>\n<parameter=path>a</parameter>\n</function>\n</tool_call></parameter></function></tool_call></x>";
    expect(xmlRunsOn(done)).toBe(true);
  });
});
