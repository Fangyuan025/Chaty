/** Inkling's turns, as its chat template writes them: blocks opened by
 *  `<|message_model|>` and closed by `<|end_message|>` — a thought, text, or a
 *  call (`NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}`). Every piece
 *  of the app that reads, writes, hides or shows a turn has to know them: a
 *  format one of them misses is a call run as prose, or protocol shown as the
 *  reply. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { parseToolCall } = await import("./agentLoop");
const { renderCall, closeOpenCalls, callStart, formatOf, withoutCalls, callRule } = await import("./callFormat");
const { liveFileCall, callingName } = await import("./liveCall");
const { withoutToolCallSpans, stripThink, normalizeChannels, answerOnly } = await import("./voiceText");

const THOUGHT = "<|message_model|><|content_thinking|>The file is small; read it.<|end_message|>";
// Exactly what the template writes for a call: keys sorted, no spaces.
const CALL = '<|message_model|>edit_file<|content_invoke_tool_json|>{"name":"edit_file","args":{"new_string":"  quantity: number;","old_string":"  qty: number;","path":"cart.ts"}}<|end_message|>';

describe("Inkling calls", () => {
  it("are read as the template writes them", () => {
    expect(parseToolCall(`${THOUGHT}${CALL}`)).toEqual({
      name: "edit_file",
      args: { new_string: "  quantity: number;", old_string: "  qty: number;", path: "cart.ts" },
    });
    // Without the header in front (a turn stored after its header was trimmed).
    expect(parseToolCall('read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"a.ts"}}')).toEqual({
      name: "read_file",
      args: { path: "a.ts" },
    });
  });

  it("round-trip through the app's own rendering", () => {
    const args = { path: "a.md", content: "line 1\nline 2 \"quoted\"" };
    const call = renderCall("write_file", args, "inkling");
    expect(call).toBe(
      '<|message_model|>write_file<|content_invoke_tool_json|>{"name":"write_file","args":{"path":"a.md","content":"line 1\\nline 2 \\"quoted\\""}}<|end_message|>',
    );
    expect(parseToolCall(call)).toEqual({ name: "write_file", args });
    expect(formatOf(call)).toBe("inkling");
  });

  it("are found where their block begins, and closed back as written", () => {
    const turn = `${THOUGHT}<|message_model|><|content_text|>Fixing it.<|end_message|>${CALL}`;
    expect(callStart(turn)).toBe(turn.indexOf(CALL));
    expect(withoutCalls(turn)).toBe(turn.slice(0, turn.indexOf(CALL)));
    expect(closeOpenCalls(turn)).toBe(turn);
    // Cut off mid-call: closed the way the block closes.
    const cut = CALL.slice(0, -"<|end_message|>".length);
    expect(closeOpenCalls(cut)).toBe(CALL);
  });

  it("stream into a live card", () => {
    const partial =
      THOUGHT + '<|message_model|>write_file<|content_invoke_tool_json|>{"name":"write_file","args":{"path":"notes.md","content":"# Title\\nfirst li';
    expect(callingName(partial)).toBe("write_file");
    const v = liveFileCall(partial);
    expect(v?.name).toBe("write_file");
    expect(v?.path).toBe("notes.md");
    // Still thinking: no card.
    expect(liveFileCall("<|message_model|><|content_thinking|>I could write")).toBeNull();
  });

  it("are not shown as the reply", () => {
    expect(withoutToolCallSpans(`<|message_model|><|content_text|>Let me fix it.<|end_message|>${CALL}`)).toBe(
      "<|message_model|><|content_text|>Let me fix it.<|end_message|>",
    );
    expect(stripThink(`${THOUGHT}<|message_model|><|content_text|>Done.<|end_message|>`)).toBe("Done.");
  });

  it("teach the block without writing its tokens into the prompt", () => {
    // Its markers are special tokens: written into the system prompt they
    // would be read as tokens, a block opened inside the system message.
    for (const zh of [true, false]) {
      const rule = callRule(zh, "inkling");
      expect(rule).not.toMatch(/<\|/);
      expect(rule).toContain('"args"');
    }
  });
});

describe("Inkling turns on screen", () => {
  it("show the thought as reasoning and the text as the answer", () => {
    const turn = `${THOUGHT}<|message_model|><|content_text|>391<|end_message|>`;
    expect(normalizeChannels(turn)).toBe("<think>The file is small; read it.</think>391");
    expect(answerOnly(turn)).toBe("391");
  });

  it("keep a thought open while it streams", () => {
    expect(normalizeChannels("<|message_model|><|content_thinking|>17 times 2")).toBe("<think>17 times 2");
  });

  it("hold back a block's header until it says what it is", () => {
    // A call's name arrives before its marker — never shown as text.
    expect(normalizeChannels(`${THOUGHT}<|message_model|>edit_fi`)).toBe("<think>The file is small; read it.</think>");
    expect(normalizeChannels(`${THOUGHT}<|message_mo`)).toBe("<think>The file is small; read it.</think>");
  });
});
