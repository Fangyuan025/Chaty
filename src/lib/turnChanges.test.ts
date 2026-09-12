import { describe, expect, it } from "vitest";
import { KEEP_CHARS, changeTotals, keptRestore, startsFolded, toTurnChange } from "./turnChanges";

describe("the changes card's fold", () => {
  it("starts open up to five files, folded past that", () => {
    expect(startsFolded(1)).toBe(false);
    expect(startsFolded(5)).toBe(false);
    expect(startsFolded(6)).toBe(true);
  });

  it("totals every file's lines when folded", () => {
    const a = { path: "/a", rel: "a", added: 3, removed: 1 };
    const b = { path: "/b", rel: "b", added: 2, removed: 4 };
    expect(changeTotals([a, b])).toEqual({ added: 5, removed: 5 });
    expect(changeTotals([])).toEqual({ added: 0, removed: 0 });
  });
});

const change = (over: Partial<Parameters<typeof toTurnChange>[0]>) => ({
  path: "/ws/a.txt",
  rel: "a.txt",
  before: "one\ntwo\n",
  after: "one\nTWO\nthree\n",
  created: false,
  deleted: false,
  binary: false,
  ...over,
});

describe("the changes card under a turn's answer", () => {
  it("counts lines exactly and keeps both versions for the diff", () => {
    const c = toTurnChange(change({}));
    expect([c.added, c.removed]).toEqual([2, 1]);
    expect(c.before).toBe("one\ntwo\n");
    expect(c.after).toBe("one\nTWO\nthree\n");
  });

  it("marks a created file, and undoes it by removing it", () => {
    const c = toTurnChange(change({ before: null, created: true, after: "new\n" }));
    expect(c.created).toBe(true);
    expect(c.added).toBe(1);
    expect(keptRestore(c)).toBeNull();
  });

  it("puts a changed file back from its kept copy", () => {
    expect(keptRestore(toTurnChange(change({})))).toBe("one\ntwo\n");
  });

  it("does not carry a huge file's contents — and so cannot undo it without the checkpoint", () => {
    const big = "x\n".repeat(KEEP_CHARS);
    const c = toTurnChange(change({ before: big, after: big + "y\n" }));
    expect(c.added).toBe(1);
    expect(c.before).toBeUndefined();
    expect(keptRestore(c)).toBeUndefined();
  });

  it("shows a binary file without a diff", () => {
    const c = toTurnChange(change({ before: null, after: null, binary: true }));
    expect(c.binary).toBe(true);
    expect([c.added, c.removed]).toEqual([0, 0]);
    expect(keptRestore(c)).toBeUndefined();
  });
});
