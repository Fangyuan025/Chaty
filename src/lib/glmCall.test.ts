/** GLM-4.5/4.6/4.7's own tool-call format, as its chat template writes it:
 *  K2 Horizon's arg_key/arg_value pairs without a namespace, after a bare tool
 *  name in the same `<tool_call>` the JSON and XML forms open with. Read as
 *  JSON — the fallback for a template it did not recognise — a GLM call was
 *  no call at all. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { parseToolCall, describeInvalidCall } = await import("./agentLoop");
const { renderCall, closeOpenCalls, callStart, formatOf, setCallFormat, callExample } = await import("./callFormat");
const { liveFileCall } = await import("./liveCall");

// GLM-4.7-Flash writes the pairs back to back; GLM-4.5 a line each.
const GLM47 =
  "<tool_call>edit_file<arg_key>path</arg_key><arg_value>cart.ts</arg_value><arg_key>old_string</arg_key><arg_value>  qty: number;</arg_value><arg_key>new_string</arg_key><arg_value>  quantity: number;</arg_value></tool_call>";
const GLM45 =
  "<tool_call>edit_file\n<arg_key>path</arg_key>\n<arg_value>cart.ts</arg_value>\n<arg_key>old_string</arg_key>\n<arg_value>  qty: number;</arg_value>\n<arg_key>new_string</arg_key>\n<arg_value>  quantity: number;</arg_value>\n</tool_call>";
const ARGS = { path: "cart.ts", old_string: "  qty: number;", new_string: "  quantity: number;" };

describe("GLM calls", () => {
  it("are read as either template writes them", () => {
    expect(parseToolCall(`<think>plan</think>${GLM47}`)).toEqual({ name: "edit_file", args: ARGS });
    expect(parseToolCall(`\n<think>plan</think>\n${GLM45}`)).toEqual({ name: "edit_file", args: ARGS });
    // JSON values as the template writes them (tojson for anything not text).
    expect(parseToolCall("<tool_call>read_file<arg_key>path</arg_key><arg_value>a.ts</arg_value><arg_key>limit</arg_key><arg_value>40</arg_value></tool_call>")?.args)
      .toEqual(parseToolCall("<tool_call>\n<function=read_file>\n<parameter=path>\na.ts\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>")?.args);
    // A call with no arguments.
    expect(parseToolCall("<tool_call>understand_repo</tool_call>")).toEqual({ name: "understand_repo", args: {} });
    // The other forms that open with <tool_call> are still theirs.
    expect(parseToolCall('<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>')).toEqual({ name: "read_file", args: { path: "a" } });
    expect(formatOf(GLM47)).toBe("glm");
    expect(formatOf('<tool_call>{"name":"x"}')).toBe("json");
  });

  it("round-trip through the app's own rendering and close as the template closes them", () => {
    const args = { path: "a.md", content: "line 1\nline 2", replace_all: true };
    const call = renderCall("write_file", args, "glm");
    expect(parseToolCall(call)).toEqual({ name: "write_file", args });
    const cut = GLM45.slice(0, GLM45.indexOf("</tool_call>"));
    expect(closeOpenCalls(cut)).toBe(GLM45);
    expect(callStart(`ok\n${GLM47}`)).toBe(3);
    setCallFormat("glm");
    expect(callExample("read_file", '{"path":"src/app.ts"}')).toBe("<tool_call>read_file<arg_key>path</arg_key><arg_value>src/app.ts</arg_value></tool_call>");
    setCallFormat("json");
  });

  it("stream into a live card", () => {
    const v = liveFileCall(`<think>ok</think>${GLM47.slice(0, GLM47.indexOf("quantity"))}`);
    expect(v?.name).toBe("edit_file");
    expect(v?.path).toBe("cart.ts");
    expect(v?.edits?.[0]).toMatchObject({ old: "  qty: number;", oldDone: true });
  });

  it("are corrected in their own terms when broken", () => {
    const note = describeInvalidCall("<tool_call>edit_file<arg_key>path</arg_key><arg_value>a.ts", 1, "en");
    expect(note).toContain("<arg_key>name</arg_key>");
  });
});
