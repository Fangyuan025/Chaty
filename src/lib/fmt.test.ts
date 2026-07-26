import { describe, expect, test } from "vitest";
import { fmtBytes, fmtCount } from "./fmt";

describe("fmt — the ONE size/count formatter", () => {
  test("bytes are DECIMAL (the Settings-vs-store 7% split is dead)", () => {
    expect(fmtBytes(34_941_000_000)).toBe("34.9 GB"); // not 32.5 GB (binary)
    expect(fmtBytes(1e9)).toBe("1.0 GB");
    expect(fmtBytes(250e6)).toBe("250.0 MB");
    expect(fmtBytes(999)).toBe("999 B");
    expect(fmtBytes(0)).toBe("");
    expect(fmtBytes(Number.NaN)).toBe("");
  });
  test("counts compact", () => {
    expect(fmtCount(1_234_567)).toBe("1.2M");
    expect(fmtCount(45_600)).toBe("45.6k");
    expect(fmtCount(230)).toBe("230");
  });
});
