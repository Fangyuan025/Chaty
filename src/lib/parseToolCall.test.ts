import { describe, expect, it } from "vitest";
import { parseToolCall, repairUnclosedJson } from "./agentLoop";

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
