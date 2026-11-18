import { describe, expect, it } from "vitest";
import {
  devServerUrlFrom,
  isLocalPageUrl,
  isSourceCodeFile,
  loadedUrlFrom,
  isWebSourceFile,
  planEcho,
  wrapupNudge,
  normPath,
  fileUrlPath,
  dirOf,
  isPageScript,
  answerAfterGate,
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

  it("shows the first move its first item names, with a note to fill in", () => {
    const look = planEcho([{ content: "了解当前工作区状态", status: "in_progress" }, { content: "写首页", status: "pending" }], "zh");
    expect(look).toContain("先看工作区");
    expect(look).toContain("understand_repo");
    expect(look).not.toContain("mkdir");
    const make = planEcho([{ content: "创建主页面 HTML", status: "in_progress" }], "zh");
    expect(make).toContain("mkdir -p");
    for (const s of [look, make]) expect(s).toContain("把 note 的 … 换成你对用户说的一句话");
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

describe("loadedUrlFrom", () => {
  it("reads the loaded URL off a navigate or refresh result, in either language", () => {
    expect(loadedUrlFrom("Loaded: http://localhost:5173/app\nTitle: x\n\nbody")).toBe("http://localhost:5173/app");
    expect(loadedUrlFrom("Reloaded (cache ignored): http://127.0.0.1:8000/\nTitle: y")).toBe("http://127.0.0.1:8000/");
    expect(loadedUrlFrom("已打开:https://example.com/\n标题:e")).toBe("https://example.com/");
    expect(loadedUrlFrom("已刷新(忽略缓存):file:///tmp/a.html\n标题:a")).toBe("file:///tmp/a.html");
  });

  it("is null for anything else — a click result, an error", () => {
    expect(loadedUrlFrom("Clicked 'Add'.\n\n[page]")).toBeNull();
    expect(loadedUrlFrom("ERROR: 导航失败 (navigation failed): net::ERR_CONNECTION_REFUSED")).toBeNull();
    expect(loadedUrlFrom("")).toBeNull();
  });
});

describe("isLocalPageUrl", () => {
  it("counts the page being built: dev servers and local files", () => {
    for (const u of [
      "http://localhost:5173/",
      "https://127.0.0.1:8443/app?x=1",
      "http://0.0.0.0:3000",
      "http://[::1]:8080/#/list",
      "http://app.localhost/",
      "localhost:5173",
      "127.0.0.1:8000/index.html",
      "file:///Users/me/site/index.html",
    ])
      expect(isLocalPageUrl(u), u).toBe(true);
  });

  it("does not count the web, blank pages or paths it can't place", () => {
    for (const u of [
      "https://developer.mozilla.org/en-US/docs/Web/API/fetch",
      "http://localhost.evil.com/",
      "https://example.com/localhost",
      "about:blank",
      "data:text/html,<p>hi</p>",
      "index.html",
      "docs.rs/tokio",
      "",
    ])
      expect(isLocalPageUrl(u), u).toBe(false);
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

// ── Pages opened as files (Qwen3.5 4B, a single-page Y2K site opened from
// disk: walked twice, held back twice, its answer lost) ──

describe("page files without a server", () => {
  it("normPath gives one spelling for relative, absolute, Windows and dotted paths", () => {
    expect(normPath("js/y2k.js", "/Users/x/1")).toBe("/users/x/1/js/y2k.js");
    expect(normPath("/Users/x/1/js/y2k.js", "/elsewhere")).toBe("/users/x/1/js/y2k.js");
    expect(normPath("./js/../js/y2k.js", "/Users/x/1/")).toBe("/users/x/1/js/y2k.js");
    expect(normPath("js\\app.js", "C:\\Users\\x\\site")).toBe("c:/users/x/site/js/app.js");
    expect(normPath("C:\\Users\\x\\site\\js\\app.js", "D:\\other")).toBe("c:/users/x/site/js/app.js");
  });

  it("fileUrlPath reads file:// URLs, POSIX and Windows, and nothing else", () => {
    expect(fileUrlPath("file:///Users/x/1/index.html")).toBe("/Users/x/1/index.html");
    expect(fileUrlPath("file:///Users/x/my%20site/index.html?v=2#top")).toBe("/Users/x/my site/index.html");
    expect(fileUrlPath("file:///C:/Users/x/site/index.html")).toBe("C:/Users/x/site/index.html");
    expect(fileUrlPath("http://localhost:5173/")).toBeNull();
    expect(fileUrlPath("about:blank")).toBeNull();
  });

  it("isPageScript: plain scripts under a page's folder, nothing else", () => {
    const dirs = [dirOf(normPath("index.html", "/Users/x/1"))];
    expect(isPageScript("js/y2k.js", "/Users/x/1", dirs)).toBe(true);
    expect(isPageScript("/Users/x/1/js/y2k.js", "/Users/x/1", dirs)).toBe(true);
    expect(isPageScript("app.mjs", "/Users/x/1", dirs)).toBe(true);
    // Not a plain script: built or run some other way.
    expect(isPageScript("src/app.ts", "/Users/x/1", dirs)).toBe(false);
    expect(isPageScript("gen.py", "/Users/x/1", dirs)).toBe(false);
    // A script outside the page's folder (a sibling folder sharing a prefix).
    expect(isPageScript("/Users/x/10/app.js", "/Users/x/1", dirs)).toBe(false);
    expect(isPageScript("js/y2k.js", "/Users/x/1", [])).toBe(false);
  });

  it("the browser note owns page scripts: no run-check note for them after it", () => {
    const n = wrapupNudge(
      {
        ...base,
        htmlEdited: true,
        lastWebEditStep: 5,
        lastBrowserActionStep: -1,
        codeEditsSinceExec: { files: ["js/y2k.js"], lines: 120 },
        pageFiles: ["js/y2k.js"],
      },
      "zh",
    )!;
    expect(n).toContain("浏览器走查");
    expect(n).not.toContain("运行验证");
  });

  it("a script that is not page code still gets the run-check note beside the browser note", () => {
    const n = wrapupNudge(
      {
        ...base,
        htmlEdited: true,
        lastWebEditStep: 5,
        codeEditsSinceExec: { files: ["js/y2k.js", "tools/build.js"], lines: 120 },
        pageFiles: ["js/y2k.js"],
      },
      "en",
    )!;
    expect(n).toContain("browser_navigate");
    expect(n).toContain("tools/build.js");
    expect(n).not.toContain("y2k.js");
  });
});

describe("the closing line asks for the user's answer, not a check report", () => {
  it("both languages", () => {
    const st = { ...base, plan: [{ content: "x", status: "pending" }] };
    expect(wrapupNudge(st, "zh")).toContain("完整回答用户原本的需求");
    expect(wrapupNudge(st, "en")).toContain("a complete answer to what the user asked");
  });
});

describe("answerAfterGate", () => {
  // The repro's two answers, as the model wrote them.
  const held =
    "完成！已创建酷酷的千禧女孩风格网页：\n\n**项目文件：**\n- `index.html` - 主页面（包含完整 HTML+CSS）\n- `styles/y2k.css` - 分离的 CSS 样式文件\n- `js/y2k.js` - 交互脚本（滚动进度、Emoji 雨等）\n\n**设计特色：**\n- 🌈 Y2K 配色：霓虹粉→电光紫→青色的渐变背景\n- ⚡ 互动功能：点击按钮触发 Emoji 雨、鼠标跟随星星、滚动进度条";
  const report =
    "**运行验证完成：**\n\n1. ✅ **HTML 渲染正常** - 页面完整显示所有内容和样式\n2. ✅ **JavaScript 无错误** - 控制台清空\n3. ✅ **交互功能正常** - 点击按钮成功触发 Emoji 雨效果\n\n所有代码改动已通过浏览器运行验证，确认能正常工作。";

  it("no answer was held → the latest stands as written", () => {
    expect(answerAfterGate(null, "Done.")).toBe("Done.");
    expect(answerAfterGate(undefined, report)).toBe(report);
  });

  it("a check report after the held answer → the answer first, the report after it", () => {
    const out = answerAfterGate(held, report);
    expect(out.startsWith(held)).toBe(true);
    expect(out).toContain("运行验证完成");
    expect(answerAfterGate(held, "已经通过所有检查✅")).toBe(`${held}\n\n已经通过所有检查✅`);
  });

  it("the answer given again (rewritten or extended) → the latest alone, nothing doubled", () => {
    const again = held.replace("完成！", "已完成并验证！") + "\n\n**验证：** 浏览器打开无报错，Emoji 雨可触发。";
    expect(answerAfterGate(held, again)).toBe(again);
    expect(answerAfterGate(held, `${held}\n\nVerified in the browser.`)).toBe(`${held}\n\nVerified in the browser.`);
    const en = "Built the todo app: index.html holds the markup, app.js adds and removes items, style.css themes it.";
    const enAgain = "Built the todo app: index.html holds the markup, app.js adds, edits and removes items, style.css themes it. Walked it in the browser: no console errors.";
    expect(answerAfterGate(en, enAgain)).toBe(enAgain);
  });

  it("a condensed retelling counts as the answer again; an English check report does not", () => {
    const condensed =
      "已创建千禧女孩风格网页(index.html、styles/y2k.css、js/y2k.js):Y2K 霓虹粉→电光紫→青色渐变背景,点击按钮触发 Emoji 雨、鼠标跟随星星、滚动进度条。浏览器验证无报错。";
    expect(answerAfterGate(held, condensed)).toBe(condensed);
    const en =
      "Done! I built a todo app. Files: index.html (markup), app.js (adding, completing and deleting tasks, saved to localStorage), style.css (dark theme). Open index.html in a browser to use it; tasks persist across reloads.";
    const enReport =
      "Verification complete: the page renders, the console shows no errors, clicking Add creates a task and Delete removes it. All checks passed.";
    expect(answerAfterGate(en, enReport)).toBe(`${en}\n\n${enReport}`);
  });

  it("a placeholder held back (\"Done.\") is not kept beside a real answer", () => {
    expect(answerAfterGate("All done.", report)).toBe(report);
  });

  it("an empty or contained latest falls back to the held answer", () => {
    expect(answerAfterGate(held, "   ")).toBe(held);
    expect(answerAfterGate(held, "**项目文件：**")).toBe(held);
  });
});
