import { describe, expect, it } from "vitest";
import { popShift } from "./popFit";

describe("popShift", () => {
  it("leaves a preview that already fits", () => {
    expect(popShift(100, 320, 0, 800)).toBe(0);
  });

  it("pulls a preview that would cross the right edge back inside", () => {
    // Issue #14: the third source chip sits near the column's right edge.
    const s = popShift(700, 320, 0, 900);
    expect(s).toBeGreaterThan(0);
    expect(700 - s + 320).toBeLessThanOrEqual(900 - 8);
  });

  it("keeps the left edge in bounds when the column is narrower than the preview", () => {
    const s = popShift(60, 320, 0, 300);
    expect(60 - s).toBe(8);
  });
});
