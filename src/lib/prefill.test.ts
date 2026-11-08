import { describe, expect, it } from "vitest";
import { prefillPct } from "./prefill";

describe("prompt processing as a percent", () => {
  it("is what was reported, never ahead of it", () => {
    expect(prefillPct(0)).toBe(0);
    expect(prefillPct(0.374)).toBe(37);
    expect(prefillPct(0.379)).toBe(37);
    expect(prefillPct(0.5)).toBe(50);
  });
  it("reaches 100 only when the whole prompt is read", () => {
    expect(prefillPct(0.996)).toBe(99);
    expect(prefillPct(1)).toBe(100);
    expect(prefillPct(512 / 512)).toBe(100);
    expect(prefillPct(0.29 * 100 / 100)).toBe(29);
  });
  it("stays in range on odd input", () => {
    expect(prefillPct(1.2)).toBe(100);
    expect(prefillPct(-0.1)).toBe(0);
    expect(prefillPct(NaN)).toBe(0);
  });
});
