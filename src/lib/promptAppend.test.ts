/** A step's prompt must be the last one with something added: the engine
 *  resumes from it, and a cache that cannot be rewound throws everything away
 *  when it is not. These are the shapes that break it. */
import { describe, expect, it } from "vitest";
import { firstDivergence, shapeOf, type SentShape } from "./agentLoop";

const msgs = (...pairs: [string, string][]) =>
  shapeOf(pairs.map(([role, content]) => ({ role, content })) as never);

describe("is this prompt an append of the last one", () => {
  it("says nothing when messages were only added", () => {
    const before = msgs(["system", "rules"], ["user", "do it"]);
    const after = msgs(["system", "rules"], ["user", "do it"], ["assistant", "<tool_call>…"], ["user", "<tool_result …"]);
    expect(firstDivergence(before, after)).toBeNull();
  });

  it("names the message that was rewritten", () => {
    const before = msgs(["system", "rules"], ["assistant", "a long answer, as generated"], ["user", "next"]);
    const after = msgs(["system", "rules"], ["assistant", "a long answer, t…"], ["user", "next"]);
    const off = firstDivergence(before, after);
    expect(off?.at).toBe(1);
    expect(off?.was.len).toBe("a long answer, as generated".length);
    expect(off?.is?.len).toBe("a long answer, t…".length);
  });

  it("names a message that disappeared (compaction dropping a span)", () => {
    const before = msgs(["system", "rules"], ["user", "q"], ["assistant", "a"]);
    const after = msgs(["system", "rules"], ["user", "q"]);
    expect(firstDivergence(before, after)).toEqual({ at: 2, was: { role: "assistant", len: 1, head: "a" } });
  });

  it("catches a role change, which re-renders everything after it", () => {
    const before = msgs(["system", "rules"], ["user", "<tool_result name=\"bash\">ok"]);
    const after = msgs(["system", "rules"], ["tool", "<tool_result name=\"bash\">ok"]);
    expect(firstDivergence(before, after)?.at).toBe(1);
  });

  it("catches a changed system prompt, wherever the change sits", () => {
    const before: SentShape[] = msgs(["system", "rules v1"], ["user", "q"]);
    const after = msgs(["system", "rules v2 with a memory line"], ["user", "q"]);
    expect(firstDivergence(before, after)?.at).toBe(0);
  });
});
