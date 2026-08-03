/** The run-check wrap-up note, end to end through the REAL loop: write code,
 *  try to deliver without running anything → exactly one nudge; run it for
 *  real → silence; fake a check with a read-only command → still nudged.
 *  Threshold behavior itself is unit-tested in wrapupGate.test.ts — this
 *  file proves the LOOP feeds the gate honest state. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");

type Ev = { type: string; [k: string]: unknown };
type Chan = { onmessage?: (ev: Ev) => void };

const BIG_PY = Array.from({ length: 40 }, (_, i) => `print(${i})`).join("\n");
const call = (name: string, args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;

async function runRounds(rounds: string[]): Promise<{ injects: string[]; final: string }> {
  const script = [...rounds];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const ch = (args as { onEvent: Chan }).onEvent;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "agent_write_file") return "written";
    if (cmd === "agent_bash") {
      // Commands containing "fail" exit 1 — lets scripts exercise the
      // red-build path without a second mock.
      const c = String((args as { command?: string }).command ?? "");
      return c.includes("fail")
        ? { stdout: "", stderr: "error: build failed", code: 1, timedOut: false, bgId: null }
        : { stdout: "ok", stderr: "", code: 0, timedOut: false, bgId: null };
    }
    if (cmd === "agent_validate_change") {
      const f = String(((args as { files?: string[] }).files ?? []).join(","));
      if (f.includes("red")) return "验证目标: red.py\n\n$ pytest red\n✗ 失败 (exit 1)\nFAILED red";
      if (f.includes("green")) return "验证目标: green.py\n\n$ pytest green\n✓ 通过\n";
      return "验证目标: tool.py\n\n没有发现与改动相关的测试(按 test_*.py 约定查找)。";
    }
    if (cmd === "agent_read_file") throw new Error("no such file");
    return null;
  });
  const injects: string[] = [];
  let final = "";
  await runAgentTurn(
    "build the tool",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off", maxSteps: 12,
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {},
      onFinal: (t) => { final = t; },
      onError: (m) => { throw new Error(`loop errored: ${m}`); },
      onTrace: (ev) => { if (ev.kind === "inject") injects.push(ev.text); },
    },
  );
  return { injects, final };
}

const RUN_MARK = "read-only commands don't count";

describe("run-check through the real loop", () => {
  beforeEach(() => {});
  afterEach(() => clearMocks());

  it("write big code, deliver without running → one nudge, then the answer stands", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      "All done, tool.py is ready.",
      "Final: shipped without a run, as instructed.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
    expect(final).toContain("Final");
  });

  it("write big code, actually run it → no nudge", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      call("bash", { command: "python3 tool.py" }),
      "All done, ran clean.",
    ]);
    expect(injects.some((i) => i.includes(RUN_MARK))).toBe(false);
    expect(final).toContain("All done");
  });

  it("a read-only command is not verification → still nudged", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      call("bash", { command: "ls -la" }),
      "All done.",
      "Final.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
  });

  it("small single edit stays frictionless — no nudge below the bar", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tiny.py", content: "print('hi')" }),
      "Done, one-liner written.",
    ]);
    expect(injects.some((i) => i.includes(RUN_MARK))).toBe(false);
    expect(final).toContain("one-liner");
  });

  // ── CalendarApp audit scenarios: the exploits that shipped a project ──
  // ── that didn't compile must each still end in a nudge.              ──

  it("failed build + symbolic checks (the audit's exact combo) → red-build nudge", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "app.swift", content: BIG_PY }),
      call("bash", { command: "xcodebuild build # fail" }),
      call("bash", { command: "swift --version" }),
      call("bash", { command: "cd CalendarApp && swiftc -parse a.swift b.swift" }),
      "All done, project delivered.",
      "Final: shipping anyway.",
      "Final final.",
    ]);
    const red = injects.filter((i) => i.includes("FAILED") && i.includes("until it passes"));
    expect(red.length).toBeGreaterThanOrEqual(1);
    expect(red.length).toBeLessThanOrEqual(2); // one extra push-back max
  });

  it("symbolic probe after edits → in-flight hint pointing at validate_change, twice max", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "app.swift", content: BIG_PY }),
      call("bash", { command: "swift --version" }),
      call("bash", { command: "swiftc -parse app.swift" }),
      call("bash", { command: "swiftc -parse app.swift 2>&1 | head -5" }),
      "All done.",
      "Final.",
    ]);
    const hints = injects.filter((i) => i.includes("version/syntax probe"));
    expect(hints).toHaveLength(2);
  });

  it("failed build then a real green run → silence", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "app.swift", content: BIG_PY }),
      call("bash", { command: "xcodebuild build # fail" }),
      call("bash", { command: "xcodebuild build" }),
      "All done, build is green.",
    ]);
    expect(injects.some((i) => i.includes(RUN_MARK) || i.includes("FAILED"))).toBe(false);
    expect(final).toContain("green");
  });

  it("validate_change that found nothing to run is a no-op → still nudged", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      call("validate_change", { files: ["tool.py"] }),
      "All done.",
      "Final.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
  });

  it("wind-down warning fires once, two steps before the ceiling", async () => {
    const rounds = Array.from({ length: 11 }, (_, i) =>
      call("write_file", { path: `f${i}.py`, content: "print(1)" }),
    );
    const { injects } = await runRounds([...rounds, "Done."]);
    expect(injects.filter((i) => i.includes("step warning"))).toHaveLength(1);
  });

  it("validate_change reporting ✗ → red-build nudge; ✓ → silence", async () => {
    const failed = await runRounds([
      call("write_file", { path: "red.py", content: BIG_PY }),
      call("validate_change", { files: ["red.py"] }),
      "All done.",
      "Final.",
      "Final final.",
    ]);
    expect(failed.injects.some((i) => i.includes("FAILED"))).toBe(true);
    const passed = await runRounds([
      call("write_file", { path: "green.py", content: BIG_PY }),
      call("validate_change", { files: ["green.py"] }),
      "All done.",
    ]);
    expect(passed.injects.some((i) => i.includes(RUN_MARK) || i.includes("FAILED"))).toBe(false);
  });
});
