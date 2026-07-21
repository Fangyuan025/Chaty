import { describe, expect, test } from "vitest";
import { jitHintFor, type HintKey } from "./jitHints";

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
