import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { takeNote } = await import("./agentLoop");

describe("a call's note — the model's word to the user", () => {
  test("is taken off the arguments before the tool sees them", () => {
    const call = { name: "bash" as never, args: { note: "  Found it. Running the tests next. ", command: "npm test" } };
    expect(takeNote(call)).toBe("Found it. Running the tests next.");
    expect(call.args).toEqual({ command: "npm test" });
  });

  test("a call without one is left as it is", () => {
    const call = { name: "read_file" as never, args: { path: "a.py" } };
    expect(takeNote(call)).toBe("");
    expect(call.args).toEqual({ path: "a.py" });
  });

  test("the placeholder of a call shown back, copied as it was, is no note", () => {
    const call = { name: "read_file" as never, args: { note: "…", path: "a.py" } };
    expect(takeNote(call)).toBe("");
    expect(call.args).toEqual({ path: "a.py" });
  });

  test("a note that is not words is dropped, not passed on", () => {
    const call = { name: "bash" as never, args: { note: 3, command: "ls" } as Record<string, unknown> };
    expect(takeNote(call as never)).toBe("");
    expect(call.args).toEqual({ command: "ls" });
  });
});
