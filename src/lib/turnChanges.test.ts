import { describe, expect, it } from "vitest";
import { KEEP_CHARS, keptRestore, toTurnChange } from "./turnChanges";

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
