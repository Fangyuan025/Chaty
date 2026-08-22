import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { parseNativeToolCall, parseToolCall } = await import("./agentLoop");

describe("LFM2 emits its own tool-call syntax whatever the prompt asks for", () => {
  test("the exact shape LFM2.5-2.6B produced against Chaty's own system prompt", () => {
    const call = parseNativeToolCall(
      "<|tool_call_start|>[read_file(path='src/main.py')]<|tool_call_end|>",
    );
    expect(call).toEqual({ name: "read_file", args: { path: "src/main.py" } });
  });

  test("and the double-quoted spelling LFM2.5-8B produced", () => {
    const call = parseNativeToolCall(
      '<|tool_call_start|>[read_file(path="src/main.py")]<|tool_call_end|>',
    );
    expect(call).toEqual({ name: "read_file", args: { path: "src/main.py" } });
  });

  test("parseToolCall reaches it, so the call actually fires", () => {
    // Before this, the markers rendered empty, the text read as prose, and
    // nothing ran.
    expect(parseToolCall("<|tool_call_start|>[list_dir(path='src')]<|tool_call_end|>")).toEqual({
      name: "list_dir",
      args: { path: "src" },
    });
  });

  test("Chaty's own JSON form still wins when both could match", () => {
    const both = '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>';
    expect(parseToolCall(both)).toEqual({ name: "read_file", args: { path: "a.py" } });
  });

  test("a comma inside a quoted argument is not a separator", () => {
    expect(
      parseNativeToolCall("<|tool_call_start|>[bash(command='ls a, b', timeout=30)]<|tool_call_end|>"),
    ).toEqual({ name: "bash", args: { command: "ls a, b", timeout: 30 } });
  });

  test("the template's escapes come back as the characters they stand for", () => {
    const call = parseNativeToolCall(
      "<|tool_call_start|>[write_file(content='line1\\nline2', path='it\\'s.py')]<|tool_call_end|>",
    );
    expect(call!.args.content).toBe("line1\nline2");
    expect(call!.args.path).toBe("it's.py");
  });

  test("numbers, booleans and None arrive as themselves, not as words", () => {
    const call = parseNativeToolCall(
      "<|tool_call_start|>[t(n=42, f=1.5, yes=True, no=False, nil=None)]<|tool_call_end|>",
    );
    expect(call!.args).toEqual({ n: 42, f: 1.5, yes: true, no: false, nil: null });
  });

  test("an embedded object argument survives", () => {
    const call = parseNativeToolCall(
      '<|tool_call_start|>[edit(spec={"path": "a.py", "line": 3})]<|tool_call_end|>',
    );
    expect(call!.args.spec).toEqual({ path: "a.py", line: 3 });
  });

  test("several calls listed — Chaty runs one tool per step, so the first wins", () => {
    expect(
      parseNativeToolCall("<|tool_call_start|>[read_file(path='a'), read_file(path='b')]<|tool_call_end|>"),
    ).toEqual({ name: "read_file", args: { path: "a" } });
  });

  test("no arguments at all", () => {
    expect(parseNativeToolCall("<|tool_call_start|>[list_dir()]<|tool_call_end|>")).toEqual({
      name: "list_dir",
      args: {},
    });
  });

  test("an unterminated call still yields the tool and what it had", () => {
    expect(parseNativeToolCall("<|tool_call_start|>[read_file(path='src/main.py'")).toEqual({
      name: "read_file",
      args: { path: "src/main.py" },
    });
  });

  test("ordinary prose is not mistaken for a call", () => {
    expect(parseNativeToolCall("I will read src/main.py next.")).toBeNull();
    expect(parseNativeToolCall("<|tool_call_start|>[]<|tool_call_end|>")).toBeNull();
    expect(parseNativeToolCall("<|tool_call_start|>[not a call]<|tool_call_end|>")).toBeNull();
  });
});

const { normalizeChannels } = await import("./voiceText");

describe("the markers are stripped for display, kept for the transcript", () => {
  test("a person does not see LFM2's control markers", () => {
    expect(
      normalizeChannels("<|tool_call_start|>[read_file(path='a.py')]<|tool_call_end|>"),
    ).toBe("[read_file(path='a.py')]");
  });

  test("the tool-list markers go too", () => {
    expect(normalizeChannels("<|tool_list_start|>read_file<|tool_list_end|>")).toBe("read_file");
  });

  test("stripping is display-only — the parser still reads the raw text", () => {
    const raw = "<|tool_call_start|>[read_file(path='a.py')]<|tool_call_end|>";
    expect(parseToolCall(raw)).toEqual({ name: "read_file", args: { path: "a.py" } });
  });

  test("prose without markers is untouched", () => {
    expect(normalizeChannels("just a sentence")).toBe("just a sentence");
  });
});
