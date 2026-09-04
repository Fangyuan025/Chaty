import { describe, expect, it } from "vitest";
import { contentHeight } from "./autoGrow";

describe("a composer follows its content", () => {
  it("takes the height its content needs", () => {
    expect(contentHeight(22, 200)).toBe(22);
    expect(contentHeight(66, 200)).toBe(66);
  });

  it("stops at the ceiling and scrolls from there", () => {
    expect(contentHeight(960, 200)).toBe(200);
  });
});
