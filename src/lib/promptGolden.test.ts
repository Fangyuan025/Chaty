import { describe, expect, test } from "vitest";
import golden from "./__fixtures__/systemPrompt.golden.json";

// The M0 ToolRegistry refactor must be BYTE-EXACT: the system prompt is the
// model's entire world, so "pure refactor" is provable by string equality
// against fixtures captured from the pre-refactor code
// (scripts/capture-prompt-golden.mts). If a variant fails here, the refactor
// changed model-visible behavior — fix the regression; only regenerate the
// fixture for a change that is deliberate, reviewed, and bench-gated.
(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { systemPrompt, agentSetEditAnchors } = await import("./agentLoop");

// Keep in sync with the capture script.
function neutralizeDate(p: string): string {
  return p
    .replace(/当前日期时间:[^\n]+/g, "当前日期时间:<DATE>")
    .replace(/Current date & time: [^\n]+/g, "Current date & time: <DATE>");
}

describe("systemPrompt golden (byte-exact across the registry refactor)", () => {
  for (const [key, want] of Object.entries(golden as Record<string, string>)) {
    test(key, () => {
      const [lang, mode, suite, anchor] = key.split(".");
      agentSetEditAnchors(anchor === "anchors");
      try {
        const got = neutralizeDate(
          systemPrompt(
            "/ws",
            lang === "zh",
            mode as "off" | "deep",
            undefined,
            suite === "vision",
            suite === "textbrowser",
          ),
        );
        expect(got).toBe(want);
      } finally {
        agentSetEditAnchors(false);
      }
    });
  }
});
