/**
 * Capture the CURRENT systemPrompt output for every mode combination into a
 * golden fixture. Run BEFORE a prompt-affecting refactor; the vitest golden
 * test (src/lib/promptGolden.test.ts) then proves the refactor is byte-exact.
 *
 *   npx tsx scripts/capture-prompt-golden.mts
 *
 * The date/time line is the prompt's only nondeterminism — it is neutralized
 * with the same regexes the test uses (keep them in sync).
 */
(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { systemPrompt, agentSetEditAnchors } = await import("../src/lib/agentLoop");
const { writeFileSync, mkdirSync } = await import("node:fs");

export function neutralizeDate(p: string): string {
  return p
    .replace(/当前日期时间:[^\n]+/g, "当前日期时间:<DATE>")
    .replace(/Current date & time: [^\n]+/g, "Current date & time: <DATE>");
}

const out: Record<string, string> = {};
for (const zh of [true, false]) {
  for (const mode of ["off", "deep"] as const) {
    // (vision, browserText): the three real suites — none / vision / text-browser.
    for (const [vision, browserText] of [
      [false, false],
      [true, false],
      [false, true],
    ] as const) {
      for (const anchors of [false, true]) {
        agentSetEditAnchors(anchors);
        const key = [
          zh ? "zh" : "en",
          mode,
          vision ? "vision" : browserText ? "textbrowser" : "plain",
          anchors ? "anchors" : "exact",
        ].join(".");
        out[key] = neutralizeDate(systemPrompt("/ws", zh, mode, undefined, vision, browserText));
      }
    }
  }
}
agentSetEditAnchors(false);

mkdirSync("src/lib/__fixtures__", { recursive: true });
writeFileSync("src/lib/__fixtures__/systemPrompt.golden.json", JSON.stringify(out, null, 1) + "\n");
console.log(`captured ${Object.keys(out).length} variants`);
