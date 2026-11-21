/** Olala's turns, as its chat template writes them: named channels, each
 *  `<|channel_start|>NAME<|content|>` to `<|channel_end|>` — its reasoning in
 *  `analysis`, the answer in `final`, and calls in `tools`, as
 *  `<toolcalls><call id="…"><name>…</name><arguments>{…}</arguments></call></toolcalls>`.
 *  Every piece of the app that reads, writes, hides or shows a turn has to
 *  know them. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { parseToolCall, parseBareJsonCall, codeNames } = await import("./agentLoop");
const { renderCall, closeOpenCalls, callStart, formatOf, withoutCalls, callRule } = await import("./callFormat");
const { liveFileCall, callingName } = await import("./liveCall");
const { withoutToolCallSpans, stripThink, normalizeChannels, answerOnly } = await import("./voiceText");
const { parseThinking } = await import("./reasoning");

const THOUGHT = "<|channel_start|>analysis<|content|>The file is small; read it.<|channel_end|>";
const CALL =
  '<|channel_start|>tools<|content|><toolcalls><call id="call_1"><name>edit_file</name><arguments>{"path": "cart.ts", "old_string": "  qty: number;", "new_string": "  quantity: number;"}</arguments></call></toolcalls><|channel_end|>';

describe("Olala calls", () => {
  it("are read as the model writes them", () => {
    expect(parseToolCall(`${THOUGHT}<|channel_start|>final<|content|><|channel_end|>${CALL}`)).toEqual({
      name: "edit_file",
      args: { path: "cart.ts", old_string: "  qty: number;", new_string: "  quantity: number;" },
    });
    // Without the channel around it.
    expect(parseToolCall('<toolcalls><call id="c"><name>read_file</name><arguments>{"path":"a.ts"}</arguments></call></toolcalls>')).toEqual({
      name: "read_file",
      args: { path: "a.ts" },
    });
  });

  it("round-trip through the app's own rendering", () => {
    const args = { path: "a.md", content: 'line 1\nline 2 "quoted"' };
    const call = renderCall("write_file", args, "olala");
    expect(parseToolCall(call)).toEqual({ name: "write_file", args });
    expect(formatOf(call)).toBe("olala");
  });

  it("are found where their channel begins, and taken out as a whole", () => {
    const turn = `${THOUGHT}<|channel_start|>final<|content|>Fixing it.<|channel_end|>${CALL}`;
    expect(callStart(turn)).toBe(turn.indexOf(CALL));
    expect(withoutCalls(turn)).toBe(turn.slice(0, turn.indexOf(CALL)));
    expect(closeOpenCalls(turn)).toBe(turn);
  });

  it("stream into a live card", () => {
    const partial =
      THOUGHT +
      '<|channel_start|>tools<|content|><toolcalls><call id="call_1"><name>write_file</name><arguments>{"path": "notes.md", "content": "# Title\\nfirst li';
    expect(callingName(partial)).toBe("write_file");
    const v = liveFileCall(partial);
    expect(v?.name).toBe("write_file");
    expect(v?.path).toBe("notes.md");
    expect(liveFileCall("<|channel_start|>analysis<|content|>I could write")).toBeNull();
  });

  it("are not shown as the reply", () => {
    expect(withoutToolCallSpans(`<|channel_start|>final<|content|>Let me fix it.<|channel_end|>${CALL}`)).toBe(
      "<|channel_start|>final<|content|>Let me fix it.<|channel_end|>",
    );
    expect(stripThink(`${THOUGHT}<|channel_start|>final<|content|>Done.<|channel_end|>`)).toBe("Done.");
  });

  it("teach the channel without writing its tokens into the prompt", () => {
    for (const zh of [true, false]) {
      const rule = callRule(zh, "olala");
      expect(rule).not.toMatch(/<\|/);
      expect(rule).toContain("<toolcalls>");
    }
  });

  it("written as the tool's name and its arguments are a call", () => {
    // In place of its own format, in its final channel.
    const turn = '<|channel_start|>analysis<|content|><|channel_end|><|channel_start|>final<|content|>{\n  "bash": {\n    "command": "ls"\n  }\n}<|channel_end|>';
    expect(parseBareJsonCall(turn)).toEqual({ name: "bash", args: { command: "ls" } });
    expect(parseBareJsonCall('{"name":"read_file","arguments":{"path":"a"}}')?.name).toBe("read_file");
    // A JSON answer is not a call: two keys, or a key that names no tool.
    expect(parseBareJsonCall('{"bash": {"command": "ls"}, "note": "x"}')).toBeNull();
    expect(parseBareJsonCall('{"port": {"value": 8080}}')).toBeNull();
  });
});

describe("Olala turns on screen", () => {
  it("show the analysis as reasoning and the final channel as the answer", () => {
    const turn = `${THOUGHT}<|channel_start|>final<|content|>391.<|channel_end|>`;
    expect(normalizeChannels(turn)).toBe("<think>The file is small; read it.</think>391.");
    expect(answerOnly(turn)).toBe("391.");
  });

  it("show no reasoning for the empty analysis a reply opens with when thinking is off", () => {
    const turn = "<|channel_start|>analysis<|content|><|channel_end|><|channel_start|>final<|content|>399<|channel_end|>";
    const t = parseThinking(turn);
    expect(t.reasoning).toBe("");
    expect(t.thinking).toBe(false);
    expect(t.answer).toBe("399");
  });

  it("keep the analysis open while it streams", () => {
    expect(normalizeChannels("<|channel_start|>analysis<|content|>17 times 2")).toBe("<think>17 times 2");
  });

  it("hold back a channel's name until it says which it is", () => {
    expect(normalizeChannels(`${THOUGHT}<|channel_start|>fin`)).toBe("<think>The file is small; read it.</think>");
    expect(normalizeChannels(`${THOUGHT}<|channel_sta`)).toBe("<think>The file is small; read it.</think>");
  });
});

describe("Names looked up for a model", () => {
  it("leave out the loop's own protocol words", () => {
    expect(codeNames("then I get the tool_result back and call startGame")).toEqual(["startGame"]);
  });
});

describe("Unlimited-OCR pages on screen", () => {
  it("show the text of each block, a title as a heading, without boxes", async () => {
    const { ocrPages } = await import("./voiceText");
    const page = "<PAGE><BLOCK>title[40, 64, 507, 165]<FORK>Invoice No. 2026-1121\n<BLOCK>text[40, 348, 281, 436]<FORK>Total: $142.50\n<BLOCK>text[41, 519, 709, 609]<FORK>收货地址：北京市\n";
    expect(ocrPages(page)).toBe("## Invoice No. 2026-1121\n\nTotal: $142.50\n\n收货地址：北京市\n");
    // A second page after a rule; a block still being opened held back.
    expect(ocrPages("<PAGE><BLOCK>text[1, 2, 3, 4]<FORK>a\n<PAGE><BLOCK>text[5, 6")).toBe("a\n\n\n---\n\n");
    expect(normalizeChannels("<PAGE><BLOCK>text[1, 2, 3, 4]<FORK>hello")).toBe("hello");
    // Text that merely mentions a block is left alone.
    expect(ocrPages("Use <PAGE> tags")).toBe("Use <PAGE> tags");
  });
});
