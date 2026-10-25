import { describe, expect, test } from "vitest";
import type { ToolStep } from "./agentLoop";
import { activityOf, deniedCount, failedCount, segmentsOf, workSummary } from "./toolGroups";

const step = (id: string, name: string, args: Record<string, unknown> = {}, extra: Partial<ToolStep> = {}): ToolStep => ({
  id,
  call: { name: name as never, args },
  status: "done",
  ...extra,
});

describe("a turn as runs of steps between what the model said", () => {
  test("words with a call, or with a plan update, end one run and open the next", () => {
    const steps = [
      step("a", "read_file", { path: "x.py" }),
      step("b", "grep", { pattern: "foo" }),
      step("c", "write_file", { path: "y.py" }, { said: "Now the tests." }),
      step("d", "bash", { command: "python3 -m unittest" }),
    ];
    const segs = segmentsOf(steps, [{ after: 4, text: "All green." }]);
    expect(segs.map((s) => (s.kind === "say" ? `say:${s.text}` : `tools:${s.steps.map((x) => x.id).join("")}`))).toEqual([
      "tools:ab",
      "say:Now the tests.",
      "tools:cd",
      "say:All green.",
    ]);
  });

  test("a plan update's words between steps split there", () => {
    const steps = [step("a", "read_file", { path: "x" }), step("b", "bash", { command: "ls" })];
    const segs = segmentsOf(steps, [{ after: 1, text: "Plan set." }]);
    expect(segs.map((s) => s.kind)).toEqual(["tools", "say", "tools"]);
  });

  test("no words: one run", () => {
    const segs = segmentsOf([step("a", "bash"), step("b", "bash")]);
    expect(segs).toHaveLength(1);
    expect(segs[0].kind).toBe("tools");
  });
});

describe("what a run did", () => {
  test("by kind in first-seen order, files counted once", () => {
    const sum = workSummary([
      step("1", "read_file", { path: "a.py" }),
      step("2", "multi_read", { paths: ["a.py", "b.py", "c.py"] }),
      step("3", "bash", { command: "ls" }),
      step("4", "edit_file", { path: "a.py" }),
      step("5", "edit_file", { path: "a.py" }),
      step("6", "bash", { command: "pytest" }),
      step("7", "browser_click", { text: "Go" }),
      step("8", "browser_eval", { expression: "1" }),
      step("9", "mcp_thing"),
    ]);
    expect(sum).toEqual([
      { kind: "read", n: 3 },
      { kind: "command", n: 2 },
      { kind: "edit", n: 1 },
      { kind: "browser", n: 2 },
      { kind: "other", n: 1 },
    ]);
  });

  test("a file written where none was is created; lines summed; failed file steps not counted", () => {
    const counts: Record<string, { added: number; removed: number }> = {
      w1: { added: 40, removed: 0 },
      w2: { added: 12, removed: 0 },
      w3: { added: 5, removed: 9 },
      e1: { added: 3, removed: 1 },
    };
    const sum = workSummary(
      [
        step("w1", "write_file", { path: "a.py" }, { diff: { path: "a.py", before: "", after: "x" } }),
        step("w2", "write_file", { path: "b.py" }, { diff: { path: "b.py", before: "", after: "y" } }),
        step("w3", "write_file", { path: "c.py" }, { diff: { path: "c.py", before: "old", after: "new" } }),
        step("e1", "edit_file", { path: "a.py" }, { diff: { path: "a.py", before: "x", after: "z" } }),
        step("e2", "edit_file", { path: "d.py" }, { status: "error" }),
      ],
      (s) => counts[s.id] ?? null,
    );
    expect(sum).toEqual([
      { kind: "create", n: 2, added: 52, removed: 0 },
      { kind: "edit", n: 2, added: 8, removed: 10 },
    ]);
  });

  test("steps not run are left out of what the run did", () => {
    const steps = [
      step("1", "edit_file", { path: "a" }, { status: "denied" }),
      step("2", "bash", { command: "npm test" }, { status: "denied" }),
      step("3", "read_file", { path: "b" }),
    ];
    expect(workSummary(steps)).toEqual([{ kind: "read", n: 1 }]);
    expect(deniedCount(steps)).toBe(2);
    expect(workSummary([steps[0], steps[1]])).toEqual([]);
  });

  test("a run that only failed sums up as the tools it used", () => {
    expect(workSummary([step("1", "edit_file", { path: "a" }, { status: "error" })])).toEqual([{ kind: "other", n: 1 }]);
  });

  test("failures are counted — a command that exited non-zero among them", () => {
    expect(failedCount([step("1", "bash", {}, { status: "error" }), step("2", "bash")])).toBe(1);
    expect(
      failedCount([
        step("1", "bash", {}, { result: "FAIL src/a.test.ts\n[exit 1]" }),
        step("2", "bash", {}, { result: "ok\n[exit 0]" }),
        step("3", "bash", {}, { status: "denied" }),
      ]),
    ).toBe(1);
  });
});

describe("what a running step is doing", () => {
  test("names its object", () => {
    expect(activityOf(step("1", "write_file", { path: "src/app.ts" }))).toEqual({ key: "actWriting", x: "src/app.ts" });
    expect(activityOf(step("2", "bash", { command: "npm test" }))).toEqual({ key: "actRunning", x: "npm test" });
    expect(activityOf(step("3", "search_code", { query: "login" }))).toEqual({ key: "actSearchCode", x: "login" });
    expect(activityOf(step("4", "browser_eval", { expression: "x" }))).toEqual({ key: "actPageScript" });
    expect(activityOf(step("5", "ask_user", { question: "?" }))).toEqual({ key: "actAsking" });
    expect(activityOf(step("6", "some_tool"))).toEqual({ key: "actTool", x: "some_tool" });
  });

  test("every tool has words of its own — none falls to the generic line", () => {
    const names = [
      "read_file", "multi_read", "write_file", "edit_file", "edit_lines", "multi_edit", "outline", "list_dir", "glob", "grep",
      "search_files", "search_code", "search_docs", "search_history", "read_history", "bash", "bash_bg", "bg_output", "bg_input",
      "bg_kill", "understand_repo", "validate_change", "web_search", "web_fetch", "web_download", "update_plan", "ask_user",
      "view_image", "use_skill", "remember", "browser_navigate", "browser_tabs", "browser_refresh", "browser_read",
      "browser_screenshot", "browser_snapshot", "browser_scroll", "browser_close", "browser_console", "browser_click",
      "browser_type", "browser_key", "browser_eval",
    ];
    for (const n of names) expect(activityOf(step("x", n)).key, n).not.toBe("actTool");
    // An MCP server's tool names the tool and its server.
    expect(activityOf(step("m", "github__create_issue"))).toEqual({ key: "actMcp", x: "create_issue · github" });
  });
});
