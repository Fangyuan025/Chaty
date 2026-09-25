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

const { parseXmlToolCall, parseToolCall, parseBareJsonCall, xmlRunsOn, unreadArg } = await import("./agentLoop");
const { liveFileCall } = await import("./liveCall");
const { missingArgLadder } = await import("./jitHints");

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

describe("calls that mix the two XML styles", () => {
  // Verbatim, Qwen3.5 4B at the app's Code defaults. Read one way only, the
  // path was dropped, the call refused as "missing path", and the model —
  // reading its own call back — wrote it again until edit_file was disabled.
  it("reads an element-style path next to <parameter> arguments", () => {
    const raw =
      "<tool_call>\n<function=edit_file>\n<path>cart.ts</path>\n<parameter=old_string>export interface Line {\n  qty: number;\n}</parameter>\n<parameter=new_string>export interface Line {\n  quantity: number;\n}</parameter>\n</function>\n</tool_call>";
    expect(parseToolCall(raw)).toEqual({
      name: "edit_file",
      args: {
        path: "cart.ts",
        old_string: "export interface Line {\n  qty: number;\n}",
        new_string: "export interface Line {\n  quantity: number;\n}",
      },
    });
  });

  it("reads an element closed the way a parameter closes", () => {
    const raw = "<tool_call>\n<function=read_file>\n</parameter>\n<path>cart.ts</parameter>\n</parameter>\n</parameter>\n</";
    expect(parseToolCall(raw)).toEqual({ name: "read_file", args: { path: "cart.ts" } });
    const edits =
      '<tool_call>\n<function=edit_file>\n<path>cart.ts</path>\n<parameter=edits>\n[{"old_string": "a", "new_string": "b"}]\n</parameter>\n</parameter>\n</';
    expect(parseToolCall(edits)?.args).toEqual({ path: "cart.ts", edits: [{ old_string: "a", new_string: "b" }] });
  });

  it("ends a parameter at its element-style closer where that is plainly the end (verbatim)", () => {
    const raw =
      '<tool_call>\n<function=edit_file>\n<parameter=path>Weather.swift</path>\n<parameter=old_string>    case .mild: return "☀️"</parameter>\n<parameter=new_string>    case .mild: return "🌤️"</parameter>\n</function>\n</tool_call>';
    expect(parseToolCall(raw)?.args).toEqual({
      path: "Weather.swift",
      old_string: '    case .mild: return "☀️"',
      new_string: '    case .mild: return "🌤️"',
    });
    // A feed whose text holds its own </content> is still one value.
    const feed = "<function=write_file>\n<parameter=path>a.xml</parameter>\n<parameter=content>\n<entry><content>x</content>\n<author>y</author></entry>\n</parameter>\n</function>";
    expect(parseXmlToolCall(feed)?.args.content).toBe("<entry><content>x</content>\n<author>y</author></entry>");
  });

  it("reads an argument opened with a closing tag (verbatim, Qwen3.5 4B)", () => {
    const read = "<tool_call>\n<function=read_file>\n<parameter=path>\nWeather.swift\n</parameter>\n</limit>\n50\n</limit>\n</read_file>";
    expect(parseToolCall(read)?.args).toEqual({ path: "Weather.swift", limit: "50" });
    const check = '<tool_call>\n<function=validate_change>\n</parameter>\n</files>\n["Weather.swift"]\n</parameter>\n</function>\n';
    expect(parseToolCall(check)?.args).toEqual({ files: ["Weather.swift"] });
    // A run of stray closers is not an argument.
    expect(parseToolCall("<tool_call>\n<function=list_dir>\n</parameter>\n</path>\n</parameter>\n</function>")?.args).toEqual({});
  });

  it("never takes markup inside a value for arguments", () => {
    // Closed value holding elements.
    const html =
      "<function=write_file>\n<parameter=path>i.html</parameter>\n<parameter=content>\n<title>x</title>\n<body>hi</body>\n</parameter>\n</function>";
    expect(parseXmlToolCall(html)?.args).toEqual({ path: "i.html", content: "<title>x</title>\n<body>hi</body>" });
    // A value still being written.
    const open = "<function=write_file>\n<parameter=path>i.html</parameter>\n<parameter=content>\n<title>x</title>\n<body>h";
    expect(parseXmlToolCall(open)?.args).toEqual({ path: "i.html" });
  });
});

describe("MiniCPM5's own form", () => {
  // Verbatim from a MiniCPM5 2B run: every call was read with no arguments,
  // refused as "missing path", and the model spent the turn doubting its own
  // parameter names.
  it("reads <param name=…> arguments and CDATA values", () => {
    expect(parseToolCall('<function name="read_file"><param name="path">greet.py</param></function>')).toEqual({
      name: "read_file",
      args: { path: "greet.py" },
    });
    expect(parseToolCall('<function name="bash"><param name="command"><![CDATA[cat greet.py && grep -n "Welcome" greet.py]]></param></function>')).toEqual({
      name: "bash",
      args: { command: 'cat greet.py && grep -n "Welcome" greet.py' },
    });
    const edit =
      '<function name="edit_file"><param name="path">a.html</param><param name="old_string"><![CDATA[<p>a</p>\n<p>b</p>]]></param><param name="new_string"><![CDATA[<p>a</p>]]></param></function>';
    expect(parseToolCall(edit)?.args).toEqual({ path: "a.html", old_string: "<p>a</p>\n<p>b</p>", new_string: "<p>a</p>" });
  });
});

describe("an argument written as its own opener", () => {
  // Assistant Pepe 32B (a Qwen2.5 tune), verbatim: `<path=…>` read as nothing,
  // so read_file went out with no path, twice, and the model then guessed at
  // the file it never saw.
  const PEPE_READ = "<tool_call>\n<function=read_file>\n<path=inventory.js>\n<symbol=applyDiscount</symbol>\n</function>\n</tool_call>";

  it("is read, with or without its closing >", () => {
    expect(parseToolCall(PEPE_READ)).toEqual({ name: "read_file", args: { path: "inventory.js", symbol: "applyDiscount" } });
    expect(parseXmlToolCall('<tool_call>\n<function=read_file>\n<path="a b.ts"></path>\n<limit=40>\n</function>')?.args).toEqual(
      parseXmlToolCall("<tool_call>\n<function=read_file>\n<path>a b.ts</path>\n<limit>40</limit>\n</function>")?.args,
    );
  });

  it("mixes with the other forms, which keep their meaning", () => {
    const c = parseXmlToolCall(
      "<tool_call>\n<function=edit_file>\n<path>Makefile</path>\n<old_string>go test ./...</old_string>\n<new_string>a <b=c> d</new_string>\n<replace_all=false>\n</function>\n</tool_call>",
    );
    expect(c).toEqual({
      name: "edit_file",
      args: { path: "Makefile", old_string: "go test ./...", new_string: "a <b=c> d", replace_all: false },
    });
    // `<parameter=…>` is still the parameter form, not an argument named "parameter".
    expect(parseXmlToolCall("<function=read_file>\n<parameter=path>\na.ts\n</parameter>\n</function>")).toEqual({
      name: "read_file",
      args: { path: "a.ts" },
    });
  });

  it("gives a live card its path", () => {
    const v = liveFileCall("<tool_call>\n<function=edit_file>\n<path=cart.ts>\n<old_string>\n  qty");
    expect(v?.path).toBe("cart.ts");
  });

  it("is shown back to a model when it still is not read", () => {
    const raw = "<tool_call>\n<function=read_file>\n<file path: src/a.ts>\n</function>\n</tool_call>";
    expect(unreadArg(raw, ["path"])).toBe("<file path: src/a.ts>");
    expect(missingArgLadder("read_file", "path", '{"path":"a.ts"}', 1, "en", unreadArg(raw, ["path"]))).toContain(
      "`<file path: src/a.ts>` is not how an argument is written",
    );
    // A word inside some other value is not an attempt at the argument.
    expect(unreadArg('{"name":"read_file","arguments":{"file_path":"a"}}', ["path"])).toBeUndefined();
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

describe("argument names", () => {
  it("lose the stray spaces a model put around them, in edits items too", () => {
    const raw =
      '<tool_call>\n<function=edit_file>\n<path>\ncart.ts\n</path>\n<edits>\n[\n  { "old_string": "a", "new_string": "b" },\n  { "old_string": "c", " new_string": "d" }\n]\n</edits>\n</function>\n</tool_call>';
    expect(parseToolCall(raw)?.args).toEqual({
      path: "cart.ts",
      edits: [
        { old_string: "a", new_string: "b" },
        { old_string: "c", new_string: "d" },
      ],
    });
    expect(parseToolCall('<tool_call>{"name":"read_file","arguments":{" path ":"a.ts"}}</tool_call>')?.args).toEqual({ path: "a.ts" });
  });
});

describe("a structured argument written as a JS or Python literal", () => {
  it("is read as the JSON it means", async () => {
    const { looseLiteral } = await import("./agentLoop");
    // Assistant Pepe 32B's plan, verbatim.
    const plan = parseToolCall(
      '<tool_call>\n<function=update_plan>\n<parameter=todos>\n[\n  { content: "查看 inventory.js 的 applyDiscount 函数定义", status: pending },\n  { content: "修改 gold 会员折扣为 0.8", status: in_progress }\n]\n</parameter>\n</function>',
    );
    expect(plan?.args.todos).toEqual([
      { content: "查看 inventory.js 的 applyDiscount 函数定义", status: "pending" },
      { content: "修改 gold 会员折扣为 0.8", status: "in_progress" },
    ]);
    expect(looseLiteral("['a.ts', 'it\\'s \"b\".ts',]")).toEqual(["a.ts", 'it\'s "b".ts']);
    expect(looseLiteral("{'old_string': 'x = 1\\n', 'replace_all': True, n: -2.5e1, v: None}")).toEqual({
      old_string: "x = 1\n",
      replace_all: true,
      n: -25,
      v: null,
    });
    expect(looseLiteral("[src/a.ts, lib/b.test.ts]")).toEqual(["src/a.ts", "lib/b.test.ts"]);
    expect(looseLiteral("{path: src/a.ts, limit: 20}")).toEqual({ path: "src/a.ts", limit: 20 });
    // Text that is neither stays text.
    expect(looseLiteral("[see the notes above]")).toBeUndefined();
    expect(parseToolCall("<function=write_file>\n<parameter=path>\na.md\n</parameter>\n<parameter=content>\n[draft, v2]\n</parameter>\n</function>")?.args.content).toBe("[draft, v2]");
  });
});

describe("a parameter with no name", () => {
  it("is read by what its value is", () => {
    // Qwen3.5 4B, verbatim in shape: the edits array under a bare <parameter>.
    const raw =
      '<tool_call>\n<function=edit_file>\n<parameter=path>\ncart.ts\n</parameter>\n<parameter>\n[{"old_string": "  qty: number;", "new_string": "  quantity: number;"}]\n</parameter>\n</function>\n</tool_call>';
    expect(parseToolCall(raw)?.args).toEqual({
      path: "cart.ts",
      edits: [{ old_string: "  qty: number;", new_string: "  quantity: number;" }],
    });
    // An object of arguments.
    expect(parseToolCall('<tool_call>\n<function=read_file>\n<parameter>\n{"path": "a.ts", "limit": 20}\n</parameter>\n</function>')?.args).toEqual({
      path: "a.ts",
      limit: 20,
    });
    // Plain text: the one argument the call is still without.
    expect(parseToolCall("<tool_call>\n<function=read_file>\n<parameter>\nsrc/a.ts\n</parameter>\n</function>")?.args).toEqual({ path: "src/a.ts" });
    // Named arguments are never overwritten by it.
    expect(parseToolCall("<tool_call>\n<function=read_file>\n<parameter=path>\nb.ts\n</parameter>\n<parameter>\nsrc/a.ts\n</parameter>\n</function>")?.args).toEqual({ path: "b.ts" });
  });
});
