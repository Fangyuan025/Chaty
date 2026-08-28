import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { stubWrittenBodies } = await import("./agentLoop");

/** A file the model wrote is the biggest and most recoverable thing in a code
 *  transcript: it is on disk, and read_file brings it back. */
const body = "x".repeat(20000);
const call = (name: string, args: Record<string, unknown>) =>
  `<think>writing it</think>\n<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;

describe("reclaiming what the model wrote", () => {
  test("a write_file body goes, the call and the path stay", () => {
    const out = stubWrittenBodies(call("write_file", { path: "src/app.js", content: body }), "en");
    expect(out).not.toContain(body);
    expect(out).toContain("write_file");
    expect(out).toContain("src/app.js");
    expect(out).toContain("read_file");
    expect(out.length).toBeLessThan(1200);
  });

  test("the turn around it is untouched — it still reads as the turn it was", () => {
    const out = stubWrittenBodies(call("write_file", { path: "a.ts", content: body }), "en");
    expect(out.startsWith("<think>writing it</think>")).toBe(true);
  });

  test("edit_file's before/after both go", () => {
    const out = stubWrittenBodies(
      call("edit_file", { path: "a.ts", old: body, new: body }),
      "en",
    );
    expect(out).not.toContain(body);
    expect(out).toContain("a.ts");
  });

  test("a small body is left alone — nothing to gain, and it may be the point", () => {
    const src = call("write_file", { path: "a.ts", content: "export const x = 1;\n" });
    expect(stubWrittenBodies(src, "en")).toBe(src);
  });

  test("calls that are not writes are never rewritten", () => {
    const src = call("bash", { command: "npm test -- " + "y".repeat(2000) });
    expect(stubWrittenBodies(src, "en")).toBe(src);
  });

  test("unparseable markup is left exactly as it is", () => {
    const src = "<tool_call>{not json at all</tool_call>";
    expect(stubWrittenBodies(src, "en")).toBe(src);
  });

  test("it says how much it took out, in either language", () => {
    expect(stubWrittenBodies(call("write_file", { path: "a", content: body }), "en")).toContain("20000 chars");
    expect(stubWrittenBodies(call("write_file", { path: "a", content: body }), "zh")).toContain("20000 字符");
  });
});
