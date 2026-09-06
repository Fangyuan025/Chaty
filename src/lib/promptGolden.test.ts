import { describe, expect, test } from "vitest";
import golden from "./__fixtures__/systemPrompt.golden.json";

// The M0 ToolRegistry refactor must be BYTE-EXACT: the system prompt is the
// model's entire world, so "pure refactor" is provable by string equality
// against fixtures captured from the pre-refactor code
// (scripts/capture-prompt-golden.mts). If a variant fails here, the refactor
// changed model-visible behavior — fix the regression; only regenerate the
// fixture for a change that is deliberate, reviewed, and bench-gated.
//
// The comparison is byte-exact with nothing neutralized, which also makes this
// the guard against putting anything time-varying back into the prompt. The
// clock used to live here, and every turn that began in a new minute re-read
// the whole conversation because of it.
(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { systemPrompt, agentSetEditAnchors } = await import("./agentLoop");

describe("systemPrompt golden (byte-exact across the registry refactor)", () => {
  for (const [key, want] of Object.entries(golden as Record<string, string>)) {
    test(key, () => {
      const [lang, mode, suite, anchor] = key.split(".");
      agentSetEditAnchors(anchor === "anchors");
      try {
        const got = systemPrompt(
          "/ws",
          lang === "zh",
          mode as "off" | "deep",
          undefined,
          suite === "vision",
          suite === "textbrowser",
        );
        expect(got).toBe(want);
      } finally {
        agentSetEditAnchors(false);
      }
    });
  }
});
