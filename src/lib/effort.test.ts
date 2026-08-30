import { describe, expect, test } from "vitest";

import { effortLabel } from "./effort";
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
