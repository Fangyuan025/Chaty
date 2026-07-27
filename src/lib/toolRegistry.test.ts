import { describe, expect, test } from "vitest";
import {
  allToolSpecs,
  ARG_EXAMPLE,
  buildToolsDoc,
  capKeepsTail,
  MUTATING_TOOLS,
  NATIVE_TOOL_NAMES,
  registerTool,
  REPEAT_EXEMPT,
  REQUIRED_ARGS,
  resultCap,
  toolSpec,
  UNTRUSTED_TOOLS,
  unregisterTool,
} from "./toolRegistry";

// ── M0 contract: the derived views equal the pre-refactor literals ──────────
// These memberships are copied verbatim from the agentLoop.ts literals the
// registry replaced. A mismatch here means a flag was mis-transcribed onto a
// ToolSpec — model-visible behavior would change silently.

describe("derived sets match the pre-M0 literals", () => {
  test("MUTATING_TOOLS", () => {
    expect(new Set(MUTATING_TOOLS)).toEqual(
      new Set([
        "write_file",
        "edit_file",
        "edit_lines",
        "multi_edit",
        "bash",
        "bash_bg",
        "web_download",
        "validate_change",
      ]),
    );
  });

  test("REPEAT_EXEMPT", () => {
    expect(new Set(REPEAT_EXEMPT)).toEqual(
      new Set([
        "browser_refresh",
        "browser_scroll",
        "browser_screenshot",
        "browser_snapshot",
        "browser_read",
        "browser_console",
        "bg_output",
      ]),
    );
  });

  test("UNTRUSTED_TOOLS", () => {
    expect(new Set(UNTRUSTED_TOOLS)).toEqual(
      new Set([
        "web_fetch",
        "web_search",
        "browser_navigate",
        "browser_refresh",
        "browser_read",
        "browser_console",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_eval",
      ]),
    );
  });

  test("REQUIRED_ARGS", () => {
    expect(REQUIRED_ARGS).toEqual({
      search_code: ["query"],
      search_docs: ["query"],
      search_files: ["query"],
      web_search: ["query"],
      web_fetch: ["url"],
      browser_navigate: ["url"],
      web_download: ["url", "path"],
      read_file: ["path"],
      write_file: ["path"],
      edit_file: ["path"],
      edit_lines: ["path"],
      multi_edit: ["path"],
      view_image: ["path"],
      grep: ["pattern"],
      bash: ["command"],
      bash_bg: ["command"],
    });
  });

  test("ARG_EXAMPLE keys mirror REQUIRED_ARGS", () => {
    expect(new Set(Object.keys(ARG_EXAMPLE))).toEqual(new Set(Object.keys(REQUIRED_ARGS)));
  });

  test("result caps match the pre-M0 toolResultMsg branches", () => {
    expect(resultCap("read_file")).toBe(400000);
    expect(resultCap("web_fetch")).toBe(48000);
    expect(resultCap("bash")).toBe(12000);
    expect(resultCap("no_such_tool")).toBe(12000);
    for (const n of ["bash", "bash_bg", "bg_output"]) expect(capKeepsTail(n)).toBe(true);
    expect(capKeepsTail("read_file")).toBe(false);
  });
});

describe("registry invariants", () => {
  test("every native name has a spec and vice versa", () => {
    const specNames = allToolSpecs()
      .filter((s) => s.source === "native")
      .map((s) => s.name);
    expect(new Set(specNames)).toEqual(new Set(NATIVE_TOOL_NAMES));
    expect(specNames.length).toBe(NATIVE_TOOL_NAMES.length);
  });

  test("every spec has a permission class", () => {
    for (const s of allToolSpecs()) {
      expect(["read", "write", "exec", "network", "ui"]).toContain(s.perm);
    }
  });

  test("only multi_edit is undocumented; only edit_lines is doc-hidden", () => {
    const undoc = allToolSpecs().filter((s) => !s.docLine).map((s) => s.name);
    expect(undoc).toEqual(["multi_edit"]);
    const hidden = allToolSpecs().filter((s) => s.docHidden).map((s) => s.name);
    expect(hidden).toEqual(["edit_lines"]);
  });

  test("natives cannot be shadowed or removed", () => {
    expect(
      registerTool({ name: "bash", source: "mcp", suite: "core", perm: "exec", tier: "core" }),
    ).toBe(false);
    expect(unregisterTool("bash")).toBe(false);
    expect(toolSpec("bash")?.source).toBe("native");
  });
});

// ── Deferred tier: the budget mechanism MCP tools ride in on (M1) ───────────

describe("deferred tier (inert in production, mechanism live)", () => {
  test("a deferred tool collapses into the index line and unregisters cleanly", () => {
    const before = buildToolsDoc("en", { vision: true });
    expect(
      registerTool({
        name: "mcp_github_search",
        source: "mcp",
        suite: "core",
        perm: "network",
        tier: "deferred",
        docLine: {
          zh: "- mcp_github_search: 全文文档(按需加载)",
          en: "- mcp_github_search: full doc (loaded on demand)",
        },
        hint: { zh: "搜 GitHub", en: "search GitHub" },
      }),
    ).toBe(true);
    try {
      const withDeferred = buildToolsDoc("en", { vision: true });
      // Index line present, full doc absent, core/browser docs untouched.
      expect(withDeferred).toContain("mcp_github_search(search GitHub)");
      expect(withDeferred).not.toContain("full doc (loaded on demand)");
      expect(withDeferred.startsWith(before)).toBe(true);
    } finally {
      expect(unregisterTool("mcp_github_search")).toBe(true);
    }
    // Registry back to byte-identical output after unregister.
    expect(buildToolsDoc("en", { vision: true })).toBe(before);
  });

  test("a core-tier runtime tool joins the doc block directly", () => {
    registerTool({
      name: "mcp_sqlite_query",
      source: "mcp",
      suite: "core",
      perm: "read",
      tier: "core",
      docLine: { zh: "- mcp_sqlite_query: 查询", en: "- mcp_sqlite_query: query" },
    });
    try {
      expect(buildToolsDoc("en", {})).toContain("- mcp_sqlite_query: query");
    } finally {
      unregisterTool("mcp_sqlite_query");
    }
  });
});
