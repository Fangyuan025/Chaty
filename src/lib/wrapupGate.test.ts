import { describe, expect, it } from "vitest";
import { devServerUrlFrom, isWebSourceFile, planEcho, wrapupNudge, type WrapupState } from "./wrapupGate";

const base: WrapupState = {
  plan: [],
  lastWebEditStep: -1,
  lastBrowserActionStep: -1,
  serverCtx: false,
  devServerUrl: undefined,
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
