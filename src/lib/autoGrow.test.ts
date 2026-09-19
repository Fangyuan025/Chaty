import { describe, expect, it } from "vitest";
import { contentHeight, watchContentHeight } from "./autoGrow";

describe("a composer follows its content", () => {
  it("takes the height its content needs", () => {
    expect(contentHeight(22, 200)).toBe(22);
    expect(contentHeight(66, 200)).toBe(66);
  });

  it("stops at the ceiling and scrolls from there", () => {
    expect(contentHeight(960, 200)).toBe(200);
  });
});

/** The composer refits on the next frame, not inside the observation pass —
 *  resizing from within a ResizeObserver callback is what made WebKit report
 *  "loop completed with undelivered notifications" into the error log. */
describe("a width change refits outside the observation pass", () => {
  it("does not touch the box while the callback runs", () => {
    const frames: (() => void)[] = [];
    const g = globalThis as Record<string, unknown>;
    const prevDoc = g.document;
    const prevWin = g.window;
    g.document = { fonts: undefined } as never;
    g.window = { addEventListener: () => {}, removeEventListener: () => {} } as never;
    const prevRaf = g.requestAnimationFrame;
    const prevCancel = g.cancelAnimationFrame;
    g.requestAnimationFrame = ((fn: () => void) => frames.push(fn)) as never;
    g.cancelAnimationFrame = (() => {}) as never;
    let observed: null | (() => void) = null;
    const fire = () => observed?.();
    const prevRO = g.ResizeObserver;
    g.ResizeObserver = class {
      constructor(fn: () => void) { observed = fn; }
      observe() {}
      disconnect() {}
    } as never;
    let fits = 0;
    let width = 100;
    const el = { style: {}, scrollHeight: 20, value: "", clientHeight: 20 } as unknown as HTMLTextAreaElement;
    const box = { getBoundingClientRect: () => ({ width }) } as unknown as Element;
    Object.defineProperty(el, "style", { value: { height: "", overflowY: "" }, writable: true });
    const stop = watchContentHeight(el, box, 200);
    fits = frames.length;
    width = 300;
    fire();
    // Nothing fitted yet — a frame was asked for instead.
    expect(frames.length).toBe(fits + 1);
    stop();
    g.requestAnimationFrame = prevRaf as never;
    g.cancelAnimationFrame = prevCancel as never;
    g.ResizeObserver = prevRO as never;
    g.document = prevDoc as never;
    g.window = prevWin as never;
  });
});
