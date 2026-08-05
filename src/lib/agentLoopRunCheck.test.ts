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
  const files = new Map<string, string>();
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const ch = (args as { onEvent: Chan }).onEvent;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "agent_write_file") {
      const a = args as { path?: string; content?: string };
      files.set(String(a.path), String(a.content ?? ""));
      return "written";
    }
    if (cmd === "agent_bash") {
      // Commands containing "fail" exit 1; "eperm" produces a permission
      // error — lets scripts exercise those paths without a second mock.
      const c = String((args as { command?: string }).command ?? "");
      if (c.includes("eperm"))
        return { stdout: "", stderr: "rm: /Users/x/.npm: Operation not permitted", code: 1, timedOut: false, bgId: null };
      if (c.includes("pipeswallow"))
        return { stdout: "error: Invalid manifest\nsandbox-exec: sandbox_apply: Operation not permitted", stderr: "", code: 0, timedOut: false, bgId: null };
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
    if (cmd === "agent_read_file") {
      const p = String((args as { path?: string }).path);
      if (files.has(p)) return files.get(p);
      throw new Error("no such file");
    }
    if (cmd === "agent_list_files") {
      const q = String((args as { query?: string }).query ?? "");
      if (q.includes(".app/Contents")) return []; // no packaged bundle in tests
      return ["kept.py"];
    }
    if (cmd === "agent_list_dir") {
      const p = (args as { path?: string }).path;
      if (!p) return [{ name: "CalendarApp", isDir: true, size: 0 }];
      return [
        { name: "App.swift", isDir: false, size: 100 },
        { name: "Views", isDir: true, size: 0 },
      ];
    }
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

  it("write big code, keep delivering without a single run → two nudges (2nd sharpened), then stands", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      "All done, tool.py is ready.",
      "Final: shipped without a run, as instructed.",
      "Truly final.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
    expect(injects.filter((i) => i.includes("Second reminder"))).toHaveLength(1);
    expect(final).toContain("Truly final");
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

  it("a read-only command is not verification → still nudged (both shots)", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      call("bash", { command: "ls -la" }),
      "All done.",
      "Final.",
      "Final final.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
    expect(injects.filter((i) => i.includes("Second reminder"))).toHaveLength(1);
  });

  it("nudge answered with a real green run → no second reminder", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      "All done.",
      call("bash", { command: "python3 tool.py" }),
      "Verified and done.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
    expect(injects.some((i) => i.includes("Second reminder"))).toBe(false);
    expect(final).toContain("Verified");
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
      "Final final.",
    ]);
    expect(injects.filter((i) => i.includes(RUN_MARK))).toHaveLength(1);
    expect(injects.filter((i) => i.includes("Second reminder"))).toHaveLength(1);
  });

  it("identical read_file repeats → soft-locked act-instead notes, turn survives", async () => {
    const rd = call("read_file", { path: "kept.py" });
    const { injects, final } = await runRounds([
      call("write_file", { path: "kept.py", content: "print(1)" }),
      rd,
      rd,
      rd,
      rd,
      call("bash", { command: "python3 kept.py" }),
      "Acted and done.",
    ]);
    expect(injects.filter((i) => i.includes("reading again reveals nothing new"))).toHaveLength(2);
    expect(final).toContain("Acted and done");
  });

  it("hand-writing project.pbxproj → one scaffold steer to SwiftPM", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "X.xcodeproj/project.pbxproj", content: "// !$*UTF8*$!\n{}" }),
      call("write_file", { path: "X.xcodeproj/project.pbxproj", content: "// !$*UTF8*$!\n{ objects = {}; }" }),
      "Done.",
    ]);
    expect(injects.filter((i) => i.includes("[scaffold]"))).toHaveLength(1);
  });

  it("minesweeper shape: edit after release build, package+launch old binary → stale warning, ledger stays dirty", async () => {
    const swiftApp = "import SwiftUI\n@main\nstruct M: App { var body: some Scene { WindowGroup { Text(\"x\") } } }";
    const { injects } = await runRounds([
      call("write_file", { path: "Sources/M/App.swift", content: swiftApp }),
      call("write_file", { path: "Sources/M/GameLogic.swift", content: BIG_PY }),
      call("bash", { command: "swift build -c release" }),
      call("bash", { command: 'mkdir -p M.app/Contents/MacOS && cp .build/release/M M.app/Contents/MacOS/' }),
      call("edit_file", { path: "Sources/M/GameLogic.swift", old_string: "print(0)", new_string: "print(9)" }),
      call("bash", { command: "swift test" }),
      call("bash", { command: 'cp .build/release/M M.app/Contents/MacOS/ && ./M.app/Contents/MacOS/M' }),
      "All done, LAUNCH OK.",
      "Final.",
      "Final final.",
    ]);
    expect(injects.some((i) => i.includes("[stale artifact]"))).toBe(true);
    expect(injects.some((i) => i.includes("predates your last source edits"))).toBe(true);
  });

  it("edit then rebuild then repackage → fresh, silent", async () => {
    const swiftApp = "import SwiftUI\n@main\nstruct M: App { var body: some Scene { WindowGroup { Text(\"x\") } } }";
    const { injects } = await runRounds([
      call("write_file", { path: "Sources/M/App.swift", content: swiftApp }),
      call("bash", { command: "swift build -c release" }),
      call("edit_file", { path: "Sources/M/App.swift", old_string: "x", new_string: "y" }),
      call("bash", { command: "swift build -c release" }),
      call("bash", { command: "swift test" }),
      call("bash", { command: 'cp .build/release/M M.app/Contents/MacOS/ && ./M.app/Contents/MacOS/M' }),
      "All done.",
    ]);
    expect(injects.some((i) => i.includes("[stale artifact]"))).toBe(false);
    expect(injects.some((i) => i.includes("predates"))).toBe(false);
  });

  it("test-file edits don't stale the artifact", async () => {
    const swiftApp = "import SwiftUI\n@main\nstruct M: App { var body: some Scene { WindowGroup { Text(\"x\") } } }";
    const { injects } = await runRounds([
      call("write_file", { path: "Sources/M/App.swift", content: swiftApp }),
      call("bash", { command: "swift build -c release" }),
      call("bash", { command: 'mkdir -p M.app/Contents/MacOS && cp .build/release/M M.app/Contents/MacOS/' }),
      call("write_file", { path: "Tests/MTests/CoreTests.swift", content: BIG_PY }),
      call("bash", { command: "swift test" }),
      call("bash", { command: './M.app/Contents/MacOS/M' }),
      "All done.",
    ]);
    expect(injects.some((i) => i.includes("[stale artifact]"))).toBe(false);
  });

  it("app-scale build green but zero executed functions → functional bar nudge, twice max", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "a.py", content: BIG_PY }),
      call("write_file", { path: "b.py", content: BIG_PY }),
      call("write_file", { path: "c.py", content: BIG_PY }),
      call("bash", { command: "make build" }),
      "All done.",
      "Final.",
      "Final final.",
    ]);
    const bar = injects.filter((i) => i.includes("entry ticket"));
    expect(bar.length).toBeGreaterThanOrEqual(1);
    expect(bar.length).toBeLessThanOrEqual(2);
  });

  it("single-file html app with zero walkthrough → both web and functional notes fire", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "index.html", content: "<html><body><input><button>add</button></body></html>" }),
      "All done, todo app delivered.",
      "Final.",
      "Final final.",
    ]);
    expect(injects.some((i) => i.includes("entry ticket"))).toBe(true);
    expect(injects.some((i) => i.includes("browser"))).toBe(true);
  });

  it("html app followed by a browser walkthrough → silent", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "index.html", content: "<html><body>app</body></html>" }),
      call("browser_navigate", { url: "http://127.0.0.1:8000/index.html" }),
      call("browser_click", { text: "add" }),
      "All done, walked through.",
    ]);
    expect(injects.some((i) => i.includes("entry ticket"))).toBe(false);
  });

  it("a green test run is a functional receipt → silent", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "a.py", content: BIG_PY }),
      call("write_file", { path: "b.py", content: BIG_PY }),
      call("write_file", { path: "c.py", content: BIG_PY }),
      call("bash", { command: "pytest -q" }),
      "All done.",
    ]);
    expect(injects.some((i) => i.includes("entry ticket"))).toBe(false);
  });

  it("really invoking the built thing (CLI run / curl) is a functional receipt → silent", async () => {
    const invoked = await runRounds([
      call("write_file", { path: "a.js", content: BIG_PY }),
      call("write_file", { path: "b.js", content: BIG_PY }),
      call("write_file", { path: "c.js", content: BIG_PY }),
      call("bash", { command: "node a.js --input sample.txt" }),
      "All done.",
    ]);
    expect(invoked.injects.some((i) => i.includes("entry ticket"))).toBe(false);
    const curled = await runRounds([
      call("write_file", { path: "a.py", content: BIG_PY }),
      call("write_file", { path: "b.py", content: BIG_PY }),
      call("write_file", { path: "s.py", content: BIG_PY }),
      call("bash", { command: "curl -s http://127.0.0.1:8123/api/books" }),
      "All done.",
    ]);
    expect(curled.injects.some((i) => i.includes("entry ticket"))).toBe(false);
  });

  it("exit 0 with compiler-failure output (pipe-swallowed code) is not a receipt", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "a.swift", content: BIG_PY }),
      call("write_file", { path: "b.swift", content: BIG_PY }),
      call("bash", { command: "swift build 2>&1 | tail -5 # pipeswallow" }),
      "All done.",
      "Final.",
      "Final final.",
    ]);
    expect(injects.some((i) => i.includes("FAILED") && i.includes("until it passes"))).toBe(true);
  });

  it("failed bash repeated 4x → soft-locked (not paused), turn survives to fix and finish", async () => {
    const fb = call("bash", { command: "swift build # fail" });
    const { injects, final } = await runRounds([
      call("write_file", { path: "a.swift", content: BIG_PY }),
      fb,
      fb,
      fb,
      fb,
      call("edit_file", { path: "a.swift", old_string: "print(0)", new_string: "print(9)" }),
      call("bash", { command: "swift build" }),
      call("bash", { command: "swift test" }),
      "Fixed and done.",
    ]);
    expect(injects.filter((i) => i.includes("Command LOCKED"))).toHaveLength(2);
    expect(final).toContain("Fixed and done");
  });

  it("re-sending a failed build verbatim → code-fix advice, not args advice", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "a.swift", content: BIG_PY }),
      call("bash", { command: "swift build # fail" }),
      call("bash", { command: "swift build # fail" }),
      call("edit_file", { path: "a.swift", old_string: "x", new_string: "y" }),
      "Done.",
      "Final.",
      "Final final.",
    ]);
    const advice = injects.find((i) => i.includes("cannot go green"));
    expect(advice).toBeTruthy();
    expect(advice).toContain("edit_file");
  });

  it("plan-prose 'final' is intercepted once; a real summary passes through", async () => {
    const { injects, final } = await runRounds([
      call("write_file", { path: "a.py", content: "print(1)" }),
      "Let me check the project structure first, I need to...",
      "Done: wrote a.py, verified by running it.",
    ]);
    expect(injects.some((i) => i.includes("planning/inner monologue"))).toBe(true);
    expect(final).toContain("Done: wrote a.py");
  });

  it("permission error → sandbox-vs-TCC attribution note, twice max", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "a.py", content: "print(1)" }),
      call("bash", { command: "rm -rf ~/.npm # eperm" }),
      call("bash", { command: "screencapture x.png # eperm" }),
      call("bash", { command: "codesign y # eperm" }),
      "Done.",
      "Final.",
    ]);
    expect(injects.filter((i) => i.includes("[permissions]"))).toHaveLength(2);
  });

  it("second parallel app stack → one stack warning naming both", async () => {
    const swiftApp = "import SwiftUI\n@main\nstruct A: App { var body: some Scene { WindowGroup { Text(\"x\") } } }";
    const { injects } = await runRounds([
      call("write_file", { path: "App.swift", content: swiftApp }),
      call("write_file", { path: "calculator.py", content: "import webview\nwebview.create_window('c','x')" }),
      call("write_file", { path: "calc2.py", content: "import webview\n# more pywebview" }),
      "Done.",
      "Final.",
    ]);
    const warn = injects.filter((i) => i.includes("[stack warning]"));
    expect(warn).toHaveLength(1);
    expect(warn[0]).toContain("pywebview");
    expect(warn[0]).toContain("swift");
  });

  it("mac-app entry write → immediate use_skill hint", async () => {
    const swiftApp = "import SwiftUI\n@main\nstruct A: App { var body: some Scene { WindowGroup { Text(\"x\") } } }";
    const { injects } = await runRounds([
      call("write_file", { path: "App.swift", content: swiftApp }),
      "Done.",
    ]);
    expect(injects.filter((i) => i.includes("[skill hint]"))).toHaveLength(1);
  });

  it("byte-identical write_file repeats → soft-locked with a move-on order, turn survives", async () => {
    const w = call("write_file", { path: "same.py", content: "print(1)" });
    const { injects, final } = await runRounds([
      w,
      w,
      w,
      w,
      call("bash", { command: "python3 same.py" }),
      "Recovered and done.",
    ]);
    expect(injects.filter((i) => i.includes("already on disk"))).toHaveLength(2);
    expect(final).toContain("Recovered");
  });

  it("update_plan pattern-lock → soft-locked rejections, turn survives to a real final", async () => {
    const plan = { todos: [{ content: "step one", status: "pending" }] };
    const { injects, final } = await runRounds([
      call("update_plan", plan),
      call("update_plan", plan),
      call("update_plan", plan),
      call("update_plan", plan),
      call("write_file", { path: "a.py", content: "print(1)" }),
      "Recovered and done.",
      "Truly done.", // the unfinished-todo wrap-up note consumes one answer
    ]);
    expect(injects.filter((i) => i.includes("LOCKED"))).toHaveLength(2);
    expect(final).toContain("Truly done");
  });

  it("near-identical full rewrite is accepted with a tip; a real partial regen still bounces", async () => {
    const tweaked = BIG_PY.replace("print(0)", "print(999)");
    const partial = BIG_PY.split("\n").map((l, i) => (i % 6 === 0 ? l + " # x" : l)).join("\n");
    const { injects } = await runRounds([
      call("write_file", { path: "tool.py", content: BIG_PY }),
      call("write_file", { path: "tool.py", content: tweaked }),
      call("write_file", { path: "tool.py", content: partial }),
      call("bash", { command: "python3 tool.py" }),
      "Done.",
    ]);
    const tip = injects.find((i) => i.includes("prefer edit_file next time"));
    expect(tip).toBeTruthy();
    const bounce = injects.find((i) => i.includes("not written"));
    expect(bounce).toBeTruthy();
  });

  it("five consecutive pure-observation steps → act-now breaker", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "a.py", content: "print(1)" }),
      call("list_dir", { path: "x" }),
      call("list_dir", { path: "y" }),
      call("list_dir", { path: "x" }),
      call("list_dir", { path: "y" }),
      call("list_dir", { path: "x" }),
      "Done.",
    ]);
    expect(injects.filter((i) => i.includes("[act now]"))).toHaveLength(1);
  });

  it("4th unverified source file → one incremental-cadence hint", async () => {
    const rounds = [1, 2, 3, 4, 5].map((i) => call("write_file", { path: `f${i}.py`, content: "print(1)" }));
    const { injects } = await runRounds([...rounds, call("bash", { command: "python3 f1.py" }), "Done."]);
    expect(injects.filter((i) => i.includes("[incremental]"))).toHaveLength(1);
  });

  it("red result after a green point → regression hint naming the edits since green", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "core.py", content: BIG_PY }),
      call("bash", { command: "python3 core.py" }),
      call("write_file", { path: "extra.py", content: BIG_PY }),
      call("bash", { command: "python3 fail.py # fail" }),
      "Done.",
      "Final.",
      "Final final.",
    ]);
    const reg = injects.find((i) => i.includes("[regression]"));
    expect(reg).toBeTruthy();
    expect(reg).toContain("extra.py");
    expect(reg).not.toContain("core.py,");
  });

  it("mac-app entry written but no packaged .app → delivery demands the bundle, twice max", async () => {
    const swiftApp = "import SwiftUI\n@main\nstruct CalApp: App { var body: some Scene { WindowGroup { Text(\"hi\") } } }\n" + BIG_PY;
    const { injects, final } = await runRounds([
      call("write_file", { path: "Sources/App/CalApp.swift", content: swiftApp }),
      call("write_file", { path: "Sources/App/Views.swift", content: BIG_PY }),
      call("bash", { command: "swift build" }),
      "All done, app is ready.",
      "Final: delivering without packaging.",
      "Final final.",
    ]);
    const demands = injects.filter((i) => i.includes("Contents/MacOS") && i.includes("mac-app"));
    expect(demands.length).toBeGreaterThanOrEqual(1);
    expect(demands.length).toBeLessThanOrEqual(2);
    expect(final).toContain("Final final");
  });

  it("rm that swallowed own-written files → immediate accounting warning", async () => {
    const { injects } = await runRounds([
      call("write_file", { path: "kept.py", content: BIG_PY }),
      call("write_file", { path: "gone.py", content: BIG_PY }),
      call("bash", { command: "rm -rf gone.py" }),
      "Done.",
      "Final.",
      "Final final.",
    ]);
    const warn = injects.find((i) => i.includes("[warning]"));
    expect(warn).toBeTruthy();
    expect(warn).toContain("gone.py");
    expect(warn).not.toContain("kept.py,");
  });

  it("lonely-folder listing auto-descends one level — the repeat bait is gone", async () => {
    const { injects } = await runRounds([
      call("list_dir", {}),
      "Done looking.",
    ]);
    const listing = injects.find((i) => i.includes("list_dir") && i.includes("CalendarApp/"));
    expect(listing).toBeTruthy();
    expect(listing).toContain("App.swift");
    expect(listing).toContain("Views/");
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
