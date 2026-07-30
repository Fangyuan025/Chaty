import { describe, expect, test } from "vitest";
import { jitHintFor, missingArgLadder, type HintKey } from "./jitHints";

describe("jitHintFor", () => {
  test("first browser_* result gets the hint once per turn", () => {
    const shown = new Set<HintKey>();
    const h1 = jitHintFor("browser_navigate", "…page text…", "zh", shown);
    expect(h1).toContain("[浏览器提示]");
    expect(jitHintFor("browser_click", "…", "zh", shown)).toBe("");
    // next turn: fresh set → re-arms
    expect(jitHintFor("browser_read", "…", "en", new Set())).toContain("[Browser hint]");
  });

  test("edit failure triggers the recovery hint, success does not", () => {
    const shown = new Set<HintKey>();
    expect(jitHintFor("edit_file", "已编辑 src/x.rs(替换 1 处)", "zh", shown)).toBe("");
    const h = jitHintFor("edit_file", "未找到 old_string(需与文件内容逐字匹配)", "zh", shown);
    expect(h).toContain("[编辑提示]");
    expect(jitHintFor("multi_edit", "old_string not found", "en", new Set())).toContain("[Edit hint]");
    expect(jitHintFor("edit_file", "old_string is not unique (3 matches)", "en", new Set())).toContain(
      "[Edit hint]",
    );
  });

  test("hints stay bounded and single-language", () => {
    for (const lang of ["zh", "en"] as const) {
      const h = jitHintFor("browser_navigate", "", lang, new Set());
      expect(h.length).toBeLessThanOrEqual(1100);
      if (lang === "en") expect(h).not.toMatch(/[一-鿿]/);
    }
  });

  test("non-matching tools yield nothing", () => {
    expect(jitHintFor("bash", "[exit 1]", "zh", new Set())).toBe("");
    expect(jitHintFor("read_file", "…", "en", new Set())).toBe("");
  });

  test("a local server URL in bash output triggers the webapp-flow hint once", () => {
    const shown = new Set<HintKey>();
    const banner = "VITE ready in 300ms\n➜ Local: http://localhost:5173/\n[已自动转入后台 #3]";
    const h = jitHintFor("bash", banner, "zh", shown);
    expect(h).toContain("[Webapp 提示]");
    expect(h).toContain("bash_bg");
    // Once per turn, across the whole bash family.
    expect(jitHintFor("bg_output", "#3 running… http://localhost:5173/", "zh", shown)).toBe("");
    expect(jitHintFor("bash_bg", "listening on http://127.0.0.1:8000", "en", new Set())).toContain(
      "[Webapp hint]",
    );
  });

  test("bash output without a local URL does not trigger the webapp hint", () => {
    expect(jitHintFor("bash", "compiled 12 files, done\n[exit 0]", "zh", new Set())).toBe("");
    expect(jitHintFor("bash", "fetched https://registry.npmjs.org/react", "en", new Set())).toBe("");
  });

  test("anchored read_file appends the edit_lines bridge once", () => {
    const shown = new Set<Parameters<typeof jitHintFor>[3] extends Set<infer K> ? K : never>();
    const anchored = '1:gaj→"""HTML utilities."""\n2:ddg→\n3:vua→import html';
    const h = jitHintFor("read_file", anchored, "en", shown);
    expect(h).toContain("edit_lines");
    expect(h).toContain('"anchor"');
    expect(jitHintFor("read_file", anchored, "en", shown)).toBe("");
    // Plain (non-anchored) reads stay hint-free.
    expect(jitHintFor("read_file", "import html\nimport json", "en", new Set())).toBe("");
  });

  test("understand_repo appends the concrete-arguments nudge once", () => {
    const shown = new Set<Parameters<typeof jitHintFor>[3] extends Set<infer K> ? K : never>();
    const h = jitHintFor("understand_repo", "[directory, top 2 levels]…", "en", shown);
    expect(h).toContain('search_code {"query"');
    expect(h).not.toMatch(/[一-鿿]/);
    expect(jitHintFor("understand_repo", "…", "en", shown)).toBe("");
  });
});

describe("missingArgLadder", () => {
  const at = (n: number, lang: "zh" | "en" = "zh") =>
    missingArgLadder("search_code", "query", '{"query":"where login auth is handled"}', n, lang);

  test("every rung adds NEW information — three distinct texts", () => {
    const [a, b, c] = [at(1), at(2), at(3)];
    expect(new Set([a, b, c]).size).toBe(3);
    // 1: fill the argument, with the example
    expect(a).toContain('{"query":"where login auth is handled"}');
    // 2: break the tool fixation — concrete ALTERNATIVE actions
    expect(b).toContain("list_dir");
    expect(b).toContain("read_file");
    expect(b).toContain("grep");
    // 3: the tool is off the table
    expect(c).toContain("停用");
    expect(c).not.toContain("重发");
  });

  test("rung 4 keeps the disable framing", () => {
    expect(at(4)).toContain("停用");
    expect(at(4)).toContain("4");
  });

  test("single-language per lang", () => {
    expect(at(2, "en")).not.toMatch(/[一-鿿]/);
    expect(at(3, "en")).toContain("temporarily disabled");
    expect(at(2, "zh")).not.toMatch(/Stop using/);
  });
});
