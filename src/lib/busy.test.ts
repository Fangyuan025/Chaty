/** Long work shows on the Dock icon / taskbar button and keeps the machine
 *  awake while it runs; a long one ending, or an agent stopping to ask, nudges
 *  a window that is in the background — and only then. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
let focused = false;
g.document = { hasFocus: () => focused };

const { mockIPC, mockWindows, clearMocks } = await import("@tauri-apps/api/mocks");
const { startJob, holdAwake, askAttention } = await import("./busy");

let calls: { cmd: string; args: Record<string, unknown> }[] = [];
beforeEach(() => {
  vi.useFakeTimers();
  focused = false;
  calls = [];
  mockWindows("main");
  mockIPC((cmd, args) => {
    calls.push({ cmd, args: (args ?? {}) as Record<string, unknown> });
    return null;
  });
});
afterEach(() => {
  vi.useRealTimers();
  clearMocks();
});

const bars = () =>
  calls.filter((c) => c.cmd === "plugin:window|set_progress_bar").map((c) => c.args.value as { status: string; progress?: number });
const awake = () => calls.filter((c) => c.cmd === "keep_awake").map((c) => c.args.on);
const nudges = () => calls.filter((c) => c.cmd === "plugin:window|request_user_attention").length;
const last = <T,>(xs: T[]): T | undefined => xs[xs.length - 1];
const flush = async () => {
  await vi.advanceTimersByTimeAsync(500);
};

describe("long work", () => {
  it("shows its progress, averaged over jobs, and keeps the machine awake", async () => {
    const a = startJob();
    const b = startJob();
    a.update(0.5);
    b.update(0.1);
    await flush();
    expect(last(bars())).toMatchObject({ status: "normal", progress: 30 });
    expect(awake()).toEqual([true]);
    a.end();
    b.end();
    await flush();
    expect(last(bars())).toMatchObject({ status: "none" });
    expect(awake()).toEqual([true, false]);
  });

  it("is indeterminate while how far is not known", async () => {
    const j = startJob();
    await flush();
    expect(last(bars())).toMatchObject({ status: "indeterminate" });
    j.end(false);
    await flush();
  });

  it("nudges when a long job ends with the window elsewhere — not a short one, not a focused window", async () => {
    const short = startJob();
    await vi.advanceTimersByTimeAsync(2_000);
    short.end();
    expect(nudges()).toBe(0);
    const long = startJob();
    await vi.advanceTimersByTimeAsync(20_000);
    long.end();
    expect(nudges()).toBe(1);
    focused = true;
    const seen = startJob();
    await vi.advanceTimersByTimeAsync(20_000);
    seen.end();
    expect(nudges()).toBe(1);
    await flush();
  });

  it("a chat reply puts no bar on the icon, and a step of an agent's run does not nudge", async () => {
    const reply = startJob({ show: false });
    await flush();
    expect(last(bars())).toMatchObject({ status: "none" });
    expect(last(awake())).toBe(true);
    holdAwake("code-agent", true);
    await vi.advanceTimersByTimeAsync(20_000);
    reply.end();
    expect(nudges()).toBe(0);
    holdAwake("code-agent", false);
    await flush();
    expect(last(awake())).toBe(false);
    askAttention();
    expect(nudges()).toBe(1);
  });
});
