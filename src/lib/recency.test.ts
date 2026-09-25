import { describe, expect, it } from "vitest";
import { recencyGroups } from "./recency";

describe("recencyGroups", () => {
  it("groups by the viewer's calendar days, pinned first, order kept", () => {
    const now = new Date(2026, 8, 24, 9, 0).getTime(); // 09:00 local
    const at = (d: number, h = 12) => new Date(2026, 8, 24 - d, h).getTime();
    const items = [
      { id: "a", updatedAt: at(0, 8) },
      { id: "p", updatedAt: at(40), pinned: true },
      { id: "b", updatedAt: at(1, 23) }, // last night
      { id: "c", updatedAt: at(3) },
      { id: "d", updatedAt: at(20) },
      { id: "e", updatedAt: at(90) },
      { id: "f", updatedAt: at(0, 1) },
    ];
    const groups = recencyGroups(items, now);
    expect(groups.map((g) => [g.key, g.items.map((i) => i.id)])).toEqual([
      ["pinned", ["p"]],
      ["today", ["a", "f"]],
      ["yesterday", ["b"]],
      ["week", ["c"]],
      ["month", ["d"]],
      ["older", ["e"]],
    ]);
  });

  it("leaves out empty groups", () => {
    expect(recencyGroups([], 0)).toEqual([]);
  });
});
