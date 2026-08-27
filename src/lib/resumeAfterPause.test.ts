import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { resumeNudge } = await import("./agentLoop");
type Stuck = Parameters<typeof resumeNudge>[0];

/** A pause is not a reset. What "Continue" tells the model it must do
 *  differently, on the turn itself — the previous behaviour was to say nothing
 *  and re-run the same transcript at base temperature. */
describe("what a continue-after-pause tells the model", () => {
  const slip: Stuck = { kind: "argslip", tool: "write_file", count: 5 };
  const rep: Stuck = { kind: "repeat", tool: "list_dir", key: 'list_dir:{"path":"."}', count: 6 };

  test("an argument slip names the tool, the count, and what to do instead", () => {
    for (const zh of [true, false]) {
      const n = resumeNudge(slip, zh);
      expect(n).toContain("write_file");
      expect(n).toContain("5");
      // It must point somewhere else, not just repeat the prohibition.
      expect(n).toMatch(/list_dir|read_file|grep/);
    }
  });

  test("a repeat says re-sending cannot help", () => {
    expect(resumeNudge(rep, false)).toContain("list_dir");
    expect(resumeNudge(rep, false).toLowerCase()).toContain("unchanged");
    expect(resumeNudge(rep, true)).toContain("原样重发");
  });

  test("both start their own line, so they append to the user's turn", () => {
    for (const st of [slip, rep]) for (const zh of [true, false]) {
      expect(resumeNudge(st, zh).startsWith("\n")).toBe(true);
    }
  });

  test("the count reaches the model — it is why 'try again' is not enough", () => {
    expect(resumeNudge({ kind: "argslip", tool: "edit_file", count: 9 }, false)).toContain("9");
  });
});
