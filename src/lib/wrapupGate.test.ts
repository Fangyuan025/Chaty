import { describe, expect, it } from "vitest";
import {
  devServerUrlFrom,
  isSourceCodeFile,
  isWebSourceFile,
  planEcho,
  wrapupNudge,
  type WrapupState,
} from "./wrapupGate";

const base: WrapupState = {
  plan: [],
  lastWebEditStep: -1,
  lastBrowserActionStep: -1,
  serverCtx: false,
  devServerUrl: undefined,
  codeEditsSinceExec: { files: [], lines: 0 },
  nudged: false,
};

describe("wrapupNudge", () => {
  it("stays silent on a clean turn (no plan, no web edits)", () => {
    expect(wrapupNudge(base, "zh")).toBeNull();
    expect(wrapupNudge(base, "en")).toBeNull();
  });

  it("fires on unfinished todos and names them", () => {
    const n = wrapupNudge(
      {
        ...base,
        plan: [
          { content: "加按钮", status: "done" },
          { content: "接后端", status: "in_progress" },
          { content: "写样式", status: "pending" },
        ],
      },
      "zh",
    );
    expect(n).toContain("收尾检查");
    expect(n).toContain("2 项未完成");
    expect(n).toContain("接后端");
    expect(n).toContain("写样式");
    expect(n).not.toContain("加按钮");
  });

  it("stays silent when every todo is done", () => {
    const n = wrapupNudge(
      { ...base, plan: [{ content: "a", status: "done" }] },
      "zh",
    );
    expect(n).toBeNull();
  });

  it("fires when web files changed after the last browser check (server up)", () => {
    const n = wrapupNudge(
      {
        ...base,
        lastWebEditStep: 7,
        lastBrowserActionStep: 3,
        serverCtx: true,
        devServerUrl: "http://localhost:5173/",
      },
      "zh",
    );
    expect(n).toContain("浏览器走查");
    expect(n).toContain("http://localhost:5173/");
  });

  it("stays silent when the browser was checked AFTER the last edit", () => {
    const n = wrapupNudge(
      { ...base, lastWebEditStep: 3, lastBrowserActionStep: 7, serverCtx: true },
      "zh",
    );
    expect(n).toBeNull();
  });

  it("stays silent on web edits when there is nothing to open (no server, no browser use)", () => {
    const n = wrapupNudge({ ...base, lastWebEditStep: 5 }, "zh");
    expect(n).toBeNull();
  });

  it("fires without a server when the browser WAS in use earlier this turn", () => {
    const n = wrapupNudge({ ...base, lastWebEditStep: 5, lastBrowserActionStep: 2 }, "en");
    expect(n).toContain("browser_navigate");
  });

  it("fires at most once per turn", () => {
    const st = { ...base, plan: [{ content: "x", status: "pending" }] };
    expect(wrapupNudge(st, "zh")).not.toBeNull();
    expect(wrapupNudge({ ...st, nudged: true }, "zh")).toBeNull();
  });

  it("combines both notes into one nudge", () => {
    const n = wrapupNudge(
      {
        ...base,
        plan: [{ content: "todo", status: "pending" }],
        lastWebEditStep: 4,
        serverCtx: true,
      },
      "en",
    )!;
    expect(n).toContain("todo list");
    expect(n).toContain("browser_navigate");
  });

  it("is single-language per lang (repo rule: model-visible text)", () => {
    const st = { ...base, plan: [{ content: "x", status: "pending" }] };
    expect(wrapupNudge(st, "zh")).not.toMatch(/todo list|wrap-up/i);
    expect(wrapupNudge(st, "en")).not.toMatch(/[一-鿿]/);
  });

  // ── The run-check note (non-web sibling of the browser walk) ──

  it("stays silent below the bar: one small unrun edit keeps its flow", () => {
    const n = wrapupNudge(
      { ...base, codeEditsSinceExec: { files: ["util.py"], lines: 12 } },
      "zh",
    );
    expect(n).toBeNull();
  });

  it("fires on a screenful of unrun code and names the file", () => {
    const n = wrapupNudge(
      { ...base, codeEditsSinceExec: { files: ["cli.py"], lines: 80 } },
      "zh",
    )!;
    expect(n).toContain("运行验证");
    expect(n).toContain("cli.py");
    expect(n).toContain("validate_change");
  });

  it("fires on multiple unrun files even when each is small", () => {
    const n = wrapupNudge(
      { ...base, codeEditsSinceExec: { files: ["a.rs", "b.rs"], lines: 14 } },
      "en",
    )!;
    expect(n).toContain("read-only commands don't count");
    expect(n).toContain("a.rs");
  });

  it("does not double-bark: files the browser note covers are excluded", () => {
    const n = wrapupNudge(
      {
        ...base,
        lastWebEditStep: 6,
        lastBrowserActionStep: 2,
        serverCtx: true,
        codeEditsSinceExec: { files: ["src/App.tsx"], lines: 90 },
      },
      "en",
    )!;
    expect(n).toContain("browser_navigate");
    expect(n).not.toContain("read-only commands");
  });

  it("still names genuinely non-web files alongside the browser note", () => {
    const n = wrapupNudge(
      {
        ...base,
        lastWebEditStep: 6,
        lastBrowserActionStep: 2,
        serverCtx: true,
        codeEditsSinceExec: { files: ["src/App.tsx", "tools/gen.py"], lines: 90 },
      },
      "en",
    )!;
    expect(n).toContain("browser_navigate");
    expect(n).toContain("tools/gen.py");
    expect(n).not.toContain("App.tsx, ");
  });
});

describe("isSourceCodeFile", () => {
  it("counts code, skips docs and config", () => {
    for (const f of ["a.py", "b.rs", "c.sh", "d.go", "e.tsx"]) expect(isSourceCodeFile(f), f).toBe(true);
    for (const f of ["README.md", "conf.yaml", "data.json", "notes.txt", "Cargo.toml"])
      expect(isSourceCodeFile(f), f).toBe(false);
  });
});

describe("planEcho", () => {
  it("re-injects statuses (done / in-progress / pending counts)", () => {
    const s = planEcho(
      [
        { content: "one", status: "done" },
        { content: "two", status: "in_progress" },
        { content: "three", status: "pending" },
      ],
      "zh",
    );
    expect(s).toContain("1/3 完成");
    expect(s).toContain("进行中:two");
    expect(s).toContain("待办 1 项");
    expect(s).toContain("不要再调 update_plan");
  });

  it("plain when everything is done", () => {
    expect(planEcho([{ content: "a", status: "done" }], "en")).toBe(
      "Plan updated (recorded — no need to re-send): 1/1 done.",
    );
  });
});

describe("isWebSourceFile", () => {
  it("always counts page files", () => {
    for (const f of ["index.html", "app.css", "Btn.tsx", "view.jsx", "App.vue", "x.svelte", "a.scss"]) {
      expect(isWebSourceFile(f, false), f).toBe(true);
    }
  });
  it("counts plain ts/js only when a dev server is around", () => {
    expect(isWebSourceFile("src/main.ts", false)).toBe(false);
    expect(isWebSourceFile("src/main.ts", true)).toBe(true);
    expect(isWebSourceFile("cli.js", false)).toBe(false);
  });
  it("never counts non-web files", () => {
    expect(isWebSourceFile("src/agent.rs", true)).toBe(false);
    expect(isWebSourceFile("README.md", true)).toBe(false);
  });
});

describe("devServerUrlFrom", () => {
  it("extracts local origins from typical banners", () => {
    expect(devServerUrlFrom("  ➜  Local:   http://localhost:5173/")).toBe("http://localhost:5173/");
    expect(devServerUrlFrom("Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...")).toBe(
      "http://0.0.0.0:8000/",
    );
    expect(devServerUrlFrom("Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)")).toBe(
      "http://127.0.0.1:8000",
    );
  });
  it("ignores non-local URLs", () => {
    expect(devServerUrlFrom("see https://example.com/docs")).toBeUndefined();
  });
});

describe("wrapupNudge · red build escalation", () => {
  const edited = { files: ["a.swift", "b.swift"], lines: 200 };
  it("an outstanding failed run replaces the note with the red-build demand", () => {
    const n = wrapupNudge(
      { ...base, codeEditsSinceExec: edited, lastFailedRun: "xcodebuild -project X build" },
      "en",
    );
    expect(n).toContain("FAILED");
    expect(n).toContain("xcodebuild -project X build");
    expect(n).toContain("until it passes");
    const zh = wrapupNudge(
      { ...base, codeEditsSinceExec: edited, lastFailedRun: "xcodebuild -project X build" },
      "zh",
    );
    expect(zh).toContain("失败");
    expect(zh).toContain("不允许带着编译/构建错误交付");
  });
  it("no failed run → the standard run-check wording, not the red-build one", () => {
    const n = wrapupNudge({ ...base, codeEditsSinceExec: edited }, "en");
    expect(n).toContain("read-only commands don't count");
    expect(n).not.toContain("FAILED");
  });
  it("failed run but empty ledger (green run followed) → silent", () => {
    expect(
      wrapupNudge({ ...base, lastFailedRun: "cargo build" }, "en"),
    ).toBeNull();
  });
});

describe("wrapupNudge · second-attempt sharpening", () => {
  const edited = { files: ["a.py", "b.py"], lines: 100 };
  it("attempt 2 with untouched ledger → the sharpened order, not the verbatim note", () => {
    const n = wrapupNudge({ ...base, codeEditsSinceExec: edited, attempt: 2 }, "en");
    expect(n).toContain("Second reminder");
    expect(n).not.toContain("read-only commands don't count");
    const zh = wrapupNudge({ ...base, codeEditsSinceExec: edited, attempt: 2 }, "zh");
    expect(zh).toContain("第二次提醒");
  });
  it("attempt 1 keeps the standard wording", () => {
    const n = wrapupNudge({ ...base, codeEditsSinceExec: edited, attempt: 1 }, "en");
    expect(n).toContain("read-only commands don't count");
  });
});
