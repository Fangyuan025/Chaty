import { describe, expect, it } from "vitest";
import { parseToolCall, repairUnclosedJson, repairXmlBleed } from "./agentLoop";

describe("repairXmlBleed — Qwen3.6 name= attractor (quick15 baseline, pytest-7571 raw dumps)", () => {
  it('parses {"name="tool","arguments":{…}} — colon written as equals', () => {
    const c = parseToolCall(
      '<tool_call>{"name="search_code","arguments":{"query":"caplog fixture log level restoration teardown"}}\n',
    )!;
    expect(c.name).toBe("search_code");
    expect(c.args.query).toContain("caplog");
  });

  it('parses the bare no-args variant {"name="tool"}', () => {
    const c = parseToolCall('<tool_call>{"name="search_code"}')!;
    expect(c.name).toBe("search_code");
    expect(Object.keys(c.args)).toHaveLength(0);
  });

  it('parses the XML-tag variant {"name="tool">', () => {
    const c = parseToolCall('<tool_call>{"name="search_code">')!;
    expect(c.name).toBe("search_code");
  });

  it('parses the fused variant {"name="tool">arguments": {…}}', () => {
    const c = parseToolCall(
      '<tool_call>{"name="search_code">arguments": {"query": "caplog fixture log level restore teardown"}}\n',
    )!;
    expect(c.name).toBe("search_code");
    expect(c.args.query).toContain("caplog");
  });

  it("inner name-valued arguments stay untouched (use_skill investigate-first)", () => {
    const c = parseToolCall(
      '<tool_call>{"name="use_skill","arguments":{"name":"investigate-first"}}',
    )!;
    expect(c.name).toBe("use_skill");
    expect(c.args.name).toBe("investigate-first");
  });

  it('parses {"name":"bash","arguments>{…}} — the bleed landing on the arguments key (sympy-23950 dump)', () => {
    const c = parseToolCall(
      '<tool_call>{"name":"bash","arguments>{"command": "ls /Users/stevenlin/Desktop/Chaty-repo/bench/coder/sympy__sympy-23950/bin/python"}}\n',
    )!;
    expect(c.name).toBe("bash");
    expect(c.args.command).toContain("bin/python");
  });

  it("parses both bleeds stacked (name= plus arguments>)", () => {
    const c = parseToolCall('<tool_call>{"name="bash","arguments>{"command": "echo hi"}}')!;
    expect(c.name).toBe("bash");
    expect(c.args.command).toBe("echo hi");
  });

  it('parses the stuttered opener {"name{"name":… (django-13925 dump)', () => {
    const c = parseToolCall(
      '<tool_call>{"name{"name":"bash","arguments":{"command": "cd /var/folders/_v/_6qzxmld2rq3dckt3rps_9km0000gn/T/chaty-bench-django__django-13925-2vniqS && /Users/stevenlin/Desktop/Chaty-repo/bench/coder/swebench/envs/django__django-13925/bin/python -m django test check_framework.test_model_checks --settings=test_sqlite -v 2 2>&1 | tail -40"}}',
    )!;
    expect(c.name).toBe("bash");
    expect(c.args.command).toContain("test_model_checks");
  });

  it('parses the dropped-quote key {"name":"x",arguments":{…} (django-13925 dump)', () => {
    const c = parseToolCall(
      '<tool_call>{"name":"search_code",arguments":{"query":"models.W042 auto-created primary key check warning"}}\n',
    )!;
    expect(c.name).toBe("search_code");
    expect(c.args.query).toContain("W042");
  });

  it('parses args as a separate object after the tag: {"name="grep">\\n{"pattern":…} (django-13925 dump)', () => {
    const c = parseToolCall('<tool_call>{"name="grep">\n{"pattern":"W042"}\n')!;
    expect(c.name).toBe("grep");
    expect(c.args.pattern).toBe("W042");
  });

  it("recovers an extra trailing brace via balancedSlice (update_plan raw ended ]}} )", () => {
    const c = parseToolCall(
      '<tool_call>{"name":"update_plan","todos":[{"content":"Implement the fix","status":"pending"},{"content":"Run tests to verify","status":"pending"}]}}',
    )!;
    expect(c.name).toBe("update_plan");
    expect(Array.isArray(c.args.todos)).toBe(true);
  });

  it('parses the dropped arguments KEY: {"name":"grep", {"pattern":…}} (postfix rerun dumps)', () => {
    const c = parseToolCall('<tool_call>{"name":"grep", {"pattern": "def urlize"}}\n')!;
    expect(c.name).toBe("grep");
    expect(c.args.pattern).toBe("def urlize");
    const c2 = parseToolCall('<tool_call>{"name":"search_files", {"query": "urlize"}}\n')!;
    expect(c2.name).toBe("search_files");
    expect(c2.args.query).toBe("urlize");
  });

  it('parses the dropped value-quote: {"name=read_file"… (postfix 15814 dump)', () => {
    const c = parseToolCall(
      '<tool_call>{"name=read_file","arguments":{"path": "django/db/models/sql/compiler.py"}}',
    )!;
    expect(c.name).toBe("read_file");
    expect(c.args.path).toContain("compiler.py");
  });

  it('parses the value-fused stutter: {"name{"bash",… (postfix 15814 dump)', () => {
    const c = parseToolCall('<tool_call>{"name{"bash","command": "echo ok"}}')!;
    expect(c.name).toBe("bash");
    expect(c.args.command).toBe("echo ok");
  });

  it("name= combined with dropped arguments key parses too (postfix 7571 dump)", () => {
    const c = parseToolCall('<tool_call>{"name="read_file", {"path": "src/_pytest/logging.py"}}\n')!;
    expect(c.name).toBe("read_file");
    expect(c.args.path).toBe("src/_pytest/logging.py");
  });

  it("never rewrites content that merely CONTAINS the pattern mid-string", () => {
    const html = '<div name="x">ok</div>';
    const c = parseToolCall(
      `<tool_call>{"name":"write_file","arguments":{"path":"a.html","content":"${html.replace(/"/g, '\\"')}"}}</tool_call>`,
    )!;
    expect(c.name).toBe("write_file");
    expect(c.args.content).toContain('name="x"');
    // And the repair itself is a no-op on well-formed bodies.
    const fine = '{"name":"bash","arguments":{"command":"ls"}}';
    expect(repairXmlBleed(fine)).toBe(fine);
  });
});

/** The write-stall autopsy (35B + html): ten straight wasted rounds because
 *  the parser mishandled the shapes the model actually emits. These fixtures
 *  are lifted from the probe's raw dumps. */
describe("parseToolCall — real local-model shapes", () => {
  it("canonical nested-arguments shape (E4B et al.) — unchanged", () => {
    const c = parseToolCall(
      '<tool_call>{"name":"write_file","arguments":{"path":"index.html","content":"<h1>hi</h1>"}}</tool_call>',
    )!;
    expect(c.name).toBe("write_file");
    expect(c.args.path).toBe("index.html");
  });

  it("an EMPTY arguments object must not shadow flat fields (the 35B stall)", () => {
    const c = parseToolCall(
      '<tool_call>{"name":"write_file","path":"index.html","content":"<html>…</html>","arguments":{}}</tool_call>',
    )!;
    expect(c.name).toBe("write_file");
    expect(c.args.path).toBe("index.html");
    expect(c.args.content).toBe("<html>…</html>");
    expect(c.args.arguments).toBeUndefined();
  });

  it("missing </tool_call> with complete JSON still parses", () => {
    const c = parseToolCall(
      '<tool_call>{"name":"write_file","arguments":{"path":"a.html","content":"x"}}',
    )!;
    expect(c.args.path).toBe("a.html");
  });

  it("model ends its turn without the outer brace — repaired", () => {
    // Content string IS terminated; only closers are missing.
    const c = parseToolCall(
      '<tool_call>{"name":"write_file","arguments":{"path":"a.html","content":"<html></html>"',
    )!;
    expect(c.name).toBe("write_file");
    expect(c.args.content).toBe("<html></html>");
  });

  it("payload cut off MID-STRING is never repaired (no corrupt writes)", () => {
    expect(
      parseToolCall('<tool_call>{"name":"write_file","arguments":{"path":"a.html","content":"<html><bo'),
    ).toBeNull();
  });

  it("genuinely empty arguments on a no-arg tool still work", () => {
    const c = parseToolCall('<tool_call>{"name":"browser_read","arguments":{}}</tool_call>')!;
    expect(c.name).toBe("browser_read");
    expect(Object.keys(c.args)).toHaveLength(0);
  });
});

describe("repairUnclosedJson", () => {
  it("appends missing closers when all strings are terminated", () => {
    expect(repairUnclosedJson('{"a":{"b":[1,2]')).toBe('{"a":{"b":[1,2]}}');
  });
  it("refuses mid-string truncation", () => {
    expect(repairUnclosedJson('{"a":"unterminated')).toBeNull();
  });
  it("refuses mismatched closers", () => {
    expect(repairUnclosedJson('{"a":[}')).toBeNull();
  });
  it("handles escaped quotes inside strings", () => {
    expect(repairUnclosedJson('{"a":"say \\"hi\\""')).toBe('{"a":"say \\"hi\\""}');
  });
  it("returns null when nothing is missing", () => {
    expect(repairUnclosedJson('{"a":1}')).toBeNull();
  });
});
