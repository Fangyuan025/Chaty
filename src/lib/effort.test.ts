import { describe, expect, test } from "vitest";

import { effortLabel, intensityOf, thinkTabActive } from "./effort";
import { lookup, type TKey } from "./i18n";

const zh = (key: TKey) => lookup(key, "zh");
const en = (key: TKey) => lookup(key, "en");

describe("effortLabel", () => {
  test("gives `high` and `xhigh` names that can be told apart", () => {
    // A four-rung ladder rendered them both as 高, so the top two rungs of
    // Muse Glimmer's ladder were indistinguishable in the menu.
    expect(zh("effortHigh" as TKey)).not.toBe(zh("effortXhigh" as TKey));
    expect(effortLabel("high", zh)).toBe("高");
    expect(effortLabel("xhigh", zh)).toBe("最高");
    expect(effortLabel("high", en)).toBe("High");
    expect(effortLabel("xhigh", en)).toBe("Highest");
  });

  test("names every rung of both ladders in use", () => {
    for (const ladder of [
      ["low", "medium", "xhigh"],
      ["low", "medium", "high", "xhigh"],
    ]) {
      const labels = ladder.map((r) => effortLabel(r, zh));
      expect(new Set(labels).size).toBe(ladder.length);
      expect(labels).not.toContain("");
    }
  });

  test("shows a rung it has no name for verbatim", () => {
    // Better read as the model wrote it than mislabelled as a rung it isn't.
    expect(effortLabel("ultra", zh)).toBe("ultra");
  });
});

describe("thinkTabActive", () => {
  test("lights the chosen tab on a model with no ladder", () => {
    // The ladder rework read `rung` for every model, and a model without one
    // has no rung — so Normal and Deep stopped lighting up at all.
    for (const mode of ["off", "normal", "deep"]) {
      const lit = ["off", "normal", "deep"].filter((tab) =>
        thinkTabActive(tab, { nativeEffort: false, thinkMode: mode, rung: "" }),
      );
      expect(lit).toEqual([mode]);
    }
  });

  test("lights the chosen rung on a model with one", () => {
    const tabs = ["off", "low", "medium", "high", "xhigh"];
    const lit = tabs.filter((tab) =>
      thinkTabActive(tab, { nativeEffort: true, thinkMode: "normal", rung: "high" }),
    );
    expect(lit).toEqual(["high"]);
  });

  test("lights `off` alone when thinking is off, whatever rung is remembered", () => {
    const tabs = ["off", "low", "medium", "high", "xhigh"];
    const lit = tabs.filter((tab) =>
      thinkTabActive(tab, { nativeEffort: true, thinkMode: "off", rung: "high" }),
    );
    expect(lit).toEqual(["off"]);
  });
});

describe("intensityOf", () => {
  test("puts a three-rung ladder exactly where the old fixed mapping did", () => {
    // Qwen3.8's ladder was hardcoded as low→low, medium→normal, xhigh→deep.
    // Reading position instead of name must not move it.
    const qwen = ["low", "medium", "xhigh"];
    expect(intensityOf(qwen, "low")).toBe("low");
    expect(intensityOf(qwen, "medium")).toBe("normal");
    expect(intensityOf(qwen, "xhigh")).toBe("deep");
  });

  test("spreads a four-rung ladder across the same three intensities", () => {
    const four = ["low", "medium", "high", "xhigh"];
    expect(four.map((r) => intensityOf(four, r))).toEqual(["low", "normal", "normal", "deep"]);
  });
});
