import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { thinkSuffix } = await import("./agentLoop");

/** The thinking rung rides on every user-role turn, because that is where the
 *  model decides both whether to think and how much. */
describe("what a user turn carries for the thinking rung", () => {
  test("off sends the soft switch to a model that has one", () => {
    expect(thinkSuffix("off", false, true)).toBe("\n/no_think");
  });

  test("off sends nothing to a model without one — the engine flag does that", () => {
    expect(thinkSuffix("off", false, false)).toBe("");
    expect(thinkSuffix("off", false, undefined)).toBe("");
  });

  test("standard adds nothing at all", () => {
    for (const zh of [true, false]) {
      for (const sw of [true, false, undefined]) {
        expect(thinkSuffix("normal", zh, sw)).toBe("");
        expect(thinkSuffix("low", zh, sw)).toBe("");
      }
    }
  });

  test("deep asks for it, in the interface language", () => {
    expect(thinkSuffix("deep", false).toLowerCase()).toContain("thoroughly");
    expect(thinkSuffix("deep", true)).toContain("充分思考");
  });

  test("deep never carries the soft switch — that would turn thinking off", () => {
    expect(thinkSuffix("deep", false, true)).not.toContain("/no_think");
    expect(thinkSuffix("deep", true, true)).not.toContain("/no_think");
  });

  test("it is a suffix, so it starts its own line", () => {
    for (const s of [thinkSuffix("deep", true), thinkSuffix("deep", false), thinkSuffix("off", false, true)]) {
      expect(s.startsWith("\n")).toBe(true);
    }
  });
});
