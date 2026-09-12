import { describe, expect, it } from "vitest";
import { bgStatus, bgTitle, fmtElapsed } from "./bgTasks";

describe("the background tasks panel's labels", () => {
  it("tells running, completed, stopped and failed apart", () => {
    expect(bgStatus({ running: true, code: null, killed: false })).toBe("running");
    expect(bgStatus({ running: false, code: 0, killed: false })).toBe("done");
    // Killed on request exits non-zero — that is not a failure.
    expect(bgStatus({ running: false, code: -1, killed: true })).toBe("stopped");
    expect(bgStatus({ running: false, code: 1, killed: false })).toBe("failed");
  });

  it("shows running time in the unit that reads at a glance", () => {
    expect(fmtElapsed(0)).toBe("0s");
    expect(fmtElapsed(45)).toBe("45s");
    expect(fmtElapsed(200)).toBe("3m 20s");
    expect(fmtElapsed(3900)).toBe("1h 05m");
  });

  it("names a job by its command's first line", () => {
    expect(bgTitle("\n  npm run dev\n# port 5173")).toBe("npm run dev");
    expect(bgTitle("cargo build")).toBe("cargo build");
  });
});
