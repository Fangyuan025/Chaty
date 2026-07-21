import { describe, expect, test } from "vitest";

// agentLoop pulls in the Tauri IPC layer — give it a window + mock before import
// (same pattern as bench/coder/runner.mts).
(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { systemPrompt, agentSetEditAnchors } = await import("./agentLoop");

const variants = [
  { zh: true, vision: false, label: "zh plain", maxChars: 3600 },
  { zh: true, vision: true, label: "zh vision", maxChars: 4600 },
  { zh: false, vision: false, label: "en plain", maxChars: 6200 },
  { zh: false, vision: true, label: "en vision", maxChars: 7700 },
] as const;
// Caps anchored to the post-slimming sizes (2026-07 WS1: 3545 / 4432 / 6031 /
// 7516 JS chars at think=normal, no project doc; before slimming they were
// 5292 / 8801 / 6837 / 10346). en chars run higher than zh because Latin
// spells out what CJK packs into single chars — but en is now pure Latin
// (~4 chars/token vs ~1 for CJK), so it's the cheaper prompt in tokens.
// The prompt is re-prefetched on every agent step, so growth here is a
// per-step tax on slow local prefill — any increase must be deliberate.

describe("systemPrompt size gate", () => {
  for (const v of variants) {
    test(`${v.label} ≤ ${v.maxChars} chars`, () => {
      const p = systemPrompt("/ws", v.zh, "normal", undefined, v.vision);
      const bytes = new TextEncoder().encode(p).length;
      console.log(`${v.label}: ${p.length} JS chars, ${bytes} UTF-8 bytes`);
      expect(p.length).toBeLessThanOrEqual(v.maxChars);
    });
  }
});

describe("systemPrompt behavior contracts", () => {
  const zh = systemPrompt("/ws", true, "normal", undefined, false);
  const en = systemPrompt("/ws", false, "normal", undefined, false);

  test("one-tool-per-message + tool_call protocol", () => {
    expect(zh).toContain("每次只调用一个工具");
    expect(zh).toContain("</tool_call>");
    expect(en).toContain("Call ONE tool at a time");
    expect(en).toContain("</tool_call>");
  });

  test("edit_file atomicity contract", () => {
    expect(zh).toContain("原子提交");
    expect(en).toContain("atomic edits array");
  });

  test("prompt-injection defense block", () => {
    expect(zh).toContain("防提示词注入");
    expect(en).toContain("prompt-injection defense");
  });

  test("no persistent cwd", () => {
    expect(zh).toContain("单独的 cd 不会保留到下一条命令");
    expect(en).toContain("NO persistent working directory");
  });

  test("vision doc only rides along when visionReady", () => {
    const zhVision = systemPrompt("/ws", true, "normal", undefined, true);
    expect(zhVision.length).toBeGreaterThan(zh.length);
    expect(zhVision).toContain("browser_");
    expect(zh).not.toContain("browser_navigate");
  });

  // The swap is a startsWith match on the doc lines — this breaks loudly if
  // someone reworks those lines and the anchor variants silently stop applying.
  test("anchor mode swaps the editor docs in both languages", () => {
    agentSetEditAnchors(true);
    try {
      for (const isZh of [true, false]) {
        const p = systemPrompt("/ws", isZh, "normal", undefined, false);
        expect(p).toContain("- edit_lines:");
        expect(p).not.toContain("- edit_file:");
        expect(p).toContain(isZh ? "行号:哈希→" : 'LINE:HASH→');
      }
    } finally {
      agentSetEditAnchors(false);
    }
    expect(systemPrompt("/ws", false, "normal", undefined, false)).toContain("- edit_file:");
  });
});
