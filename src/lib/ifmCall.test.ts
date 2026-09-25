/** K2 Horizon's own tool-call format, as its chat template writes it. Every
 *  piece of the app that reads, writes, closes or hides a call has to know it:
 *  a format one of them misses is a call run as prose, or prose run as a call. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { parseToolCall } = await import("./agentLoop");
const { renderCall, closeOpenCalls, callStart, formatOf, CALL_CLOSERS } = await import("./callFormat");
const { liveFileCall } = await import("./liveCall");
const { withoutToolCallSpans, stripThink } = await import("./voiceText");

// Exactly what the template's render_tool_calls_block produces (xml format).
const TEMPLATE_CALL =
  "<ifm|tool_calls>\n<ifm|tool_call>edit_file\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_value>cart.ts</ifm|arg_value>\n<ifm|arg_key>old_string</ifm|arg_key>\n<ifm|arg_value>  qty: number;</ifm|arg_value>\n<ifm|arg_key>new_string</ifm|arg_key>\n<ifm|arg_value>  quantity: number;</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>";

describe("K2 Horizon calls", () => {
  it("are read as the template writes them", () => {
    expect(parseToolCall(`<ifm|think>\nplan\n</ifm|think>\n${TEMPLATE_CALL}`)).toEqual({
      name: "edit_file",
      args: { path: "cart.ts", old_string: "  qty: number;", new_string: "  quantity: number;" },
    });
    // The xml_typed variant, and the JSON variant.
    const typed =
      "<ifm|tool_calls>\n<ifm|tool_call>read_file\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n<ifm|arg_value>a.ts</ifm|arg_value>\n<ifm|arg_key>edits</ifm|arg_key>\n<ifm|arg_value>[1,2]</ifm|arg_value>\n</ifm|tool_call>";
    expect(parseToolCall(typed)).toEqual({ name: "read_file", args: { path: "a.ts", edits: [1, 2] } });
    const json = '<ifm|tool_calls>\n<ifm|tool_call>{"name": "list_dir", "arguments": {"path": "src"}}</ifm|tool_call>\n</ifm|tool_calls>';
    expect(parseToolCall(json)).toEqual({ name: "list_dir", args: { path: "src" } });
  });

  it("round-trip through the app's own rendering", () => {
    const args = { path: "a.md", content: "line 1\nline 2", replace_all: true };
    const call = renderCall("write_file", args, "ifm");
    expect(call).toBe(
      "<ifm|tool_calls>\n<ifm|tool_call>write_file\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_value>a.md</ifm|arg_value>\n<ifm|arg_key>content</ifm|arg_key>\n<ifm|arg_value>line 1\nline 2</ifm|arg_value>\n<ifm|arg_key>replace_all</ifm|arg_key>\n<ifm|arg_value>true</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
    );
    expect(parseToolCall(call)).toEqual({ name: "write_file", args });
  });

  it("stop at the call's closer and are closed back as the template closes them", () => {
    expect(CALL_CLOSERS).toContain("</ifm|tool_call>");
    // What the app receives when generation stops at the closer.
    const cut = TEMPLATE_CALL.slice(0, TEMPLATE_CALL.indexOf("</ifm|tool_call>"));
    expect(closeOpenCalls(cut)).toBe(TEMPLATE_CALL);
    expect(closeOpenCalls(TEMPLATE_CALL)).toBe(TEMPLATE_CALL);
    expect(callStart(`prose\n${TEMPLATE_CALL}`)).toBe(6);
    expect(formatOf(TEMPLATE_CALL)).toBe("ifm");
  });

  it("stream into a live card", () => {
    const partial = "<ifm|think>\nok\n</ifm|think>\n<ifm|tool_calls>\n<ifm|tool_call>edit_file\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_value>cart.ts</ifm|arg_value>\n<ifm|arg_key>old_string</ifm|arg_key>\n<ifm|arg_value>  qty: num";
    const v = liveFileCall(partial);
    expect(v?.name).toBe("edit_file");
    expect(v?.path).toBe("cart.ts");
    expect(v?.edits?.[0]).toMatchObject({ old: "  qty: num", oldDone: false });
    // The name not finished yet: no card yet.
    expect(liveFileCall("<ifm|think>\nI will call <ifm|tool_call>edit_fi")).toBeNull();
  });

  it("are not shown as the reply", () => {
    expect(withoutToolCallSpans(`Let me fix it.\n${TEMPLATE_CALL}`)).toBe("Let me fix it.\n");
    expect(stripThink(`<think>\nplan</ifm|think>\nDone.`)).toBe("Done.");
  });
});
