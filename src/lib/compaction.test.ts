import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { digestForCall, compactionStub, digestHistory } = await import("./agentLoop");

describe("digestForCall", () => {
  test("read_file carries path, symbol, and range", () => {
    const d = digestForCall("read_file", { path: "src/foo.py", offset: 1, limit: 400 }, "zh");
    expect(d).toContain("src/foo.py");
    expect(d).toContain("offset=1");
    expect(d).toContain("limit=400");
    expect(digestForCall("read_file", { path: "a.ts", symbol: "main" }, "en")).toContain("symbol=main");
  });
  test("bash keeps the command head", () => {
    expect(digestForCall("bash", { command: "pytest -x tests/test_foo.py" }, "en")).toBe(
      "pytest -x tests/test_foo.py",
    );
  });
  test("searches keep the query, edits keep the path, fetch keeps the url", () => {
    expect(digestForCall("search_code", { query: "token cache" }, "zh")).toBe("token cache");
    expect(digestForCall("grep", { pattern: "fn main" }, "zh")).toBe("fn main");
    expect(digestForCall("edit_file", { path: "src/x.rs", old_string: "aaa" }, "en")).toBe("src/x.rs");
    expect(digestForCall("web_fetch", { url: "https://ex.com/doc" }, "en")).toBe("https://ex.com/doc");
  });
  test("unknown tools fall back to trimmed JSON args", () => {
    const d = digestForCall("mystery", { a: 1, b: "x".repeat(200) }, "en");
    expect(d.length).toBeLessThanOrEqual(80);
  });
});

describe("compactionStub", () => {
  const meta = { name: "read_file", args: { path: "src/lib/agentLoop.ts", offset: 1, limit: 400 } };
  test("keeps the <tool_result envelope and stays under 180 chars", () => {
    const stub = compactionStub("read_file", meta, "<tool_result …big…>", "zh");
    expect(stub.startsWith('<tool_result name="read_file">')).toBe(true);
    expect(stub.endsWith("</tool_result>")).toBe(true);
    expect(stub.length).toBeLessThanOrEqual(180);
    expect(stub).toContain("src/lib/agentLoop.ts");
  });
  test("bash stubs carry the original exit status", () => {
    const bmeta = { name: "bash", args: { command: "pytest -x" } };
    const original = '<tool_result name="bash">\nboom\n[exit 1]\n</tool_result>';
    const zh = compactionStub("bash", bmeta, original, "zh");
    expect(zh).toContain("[exit 1]");
    const en = compactionStub("bash", bmeta, original, "en");
    expect(en).toContain("[exit 1]");
    expect(en).not.toContain("已压缩");
  });
  test("no meta falls back to a plain elision note, still enveloped", () => {
    const stub = compactionStub("grep", undefined, "xxx", "en");
    expect(stub.startsWith('<tool_result name="grep">')).toBe(true);
    expect(stub.length).toBeLessThanOrEqual(180);
  });
  test("over-long digests are truncated to fit", () => {
    const big = { name: "bash", args: { command: "x".repeat(300) } };
    const stub = compactionStub("bash", big, "[exit 0]", "zh");
    expect(stub.length).toBeLessThanOrEqual(180);
  });
});

describe("digestHistory", () => {
  test("user and assistant turns become capped bullets", () => {
    const digest = digestHistory(
      [
        { role: "user", content: "帮我修复登录页面的重定向 bug,具体表现是……" + "长".repeat(100) },
        { role: "assistant", content: "(tools run: read_file ×2, edit_file)\n已修复:重定向改为相对路径。" },
        { role: "user", content: "<tool_result name=\"bash\">\nnoise\n</tool_result>" },
      ] as never,
      "zh",
    );
    expect(digest).toContain("- 用户: ");
    expect(digest).toContain("- 助手: ");
    expect(digest).toContain("(tools run: read_file ×2, edit_file)");
    expect(digest).not.toContain("noise");
  });
  test("stays under 700 chars by dropping oldest bullets", () => {
    const many = Array.from({ length: 40 }, (_, i) => ({
      role: "user" as const,
      content: `任务 ${i}: ` + "内容".repeat(30),
    }));
    const digest = digestHistory(many as never, "zh");
    expect(digest.length).toBeLessThanOrEqual(700);
    expect(digest).toContain("任务 39"); // newest survives
  });
});
