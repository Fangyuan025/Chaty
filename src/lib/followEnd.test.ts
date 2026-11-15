import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Hold, movedByCommit, watchScrollInput } from "./followEnd";

const view = (scrollTop: number) => ({ scrollTop }) as HTMLElement;
const ev = (type: string, props: Record<string, unknown> = {}) => Object.assign(new Event(type), props);

describe("movedByCommit", () => {
  it("is the change's doing when the view was where the pin left it before it", () => {
    expect(movedByCommit(view(300), 400, 400)).toBe(true);
  });
  it("is not when the view had already moved up before the change (the user)", () => {
    expect(movedByCommit(view(200), 300, 400)).toBe(false);
  });
  it("is not without a snapshot, a pin, or a move", () => {
    expect(movedByCommit(view(300), null, 400)).toBe(false);
    expect(movedByCommit(view(300), 400, -1)).toBe(false);
    expect(movedByCommit(view(395), 400, 400)).toBe(false);
  });
});

describe("Hold", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("puts the view back if nothing moves it on", () => {
    const back = vi.fn();
    const h = new Hold(back, 200);
    h.start(view(500));
    expect(h.active).toBe(true);
    vi.advanceTimersByTime(199);
    expect(back).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(back).toHaveBeenCalledOnce();
    expect(h.active).toBe(false);
  });
  it("knows the view going on up, and clears without putting it back", () => {
    const back = vi.fn();
    const h = new Hold(back, 200);
    h.start(view(500));
    expect(h.further(view(499))).toBe(false);
    expect(h.further(view(480))).toBe(true);
    h.clear();
    vi.advanceTimersByTime(500);
    expect(back).not.toHaveBeenCalled();
  });
  it("keeps where it was first held", () => {
    const h = new Hold(() => {}, 200);
    h.start(view(500));
    h.start(view(300));
    expect(h.further(view(450))).toBe(true);
  });
});

describe("watchScrollInput", () => {
  const win = new EventTarget();
  beforeEach(() => {
    (globalThis as { window?: unknown }).window = win;
  });
  afterEach(() => {
    delete (globalThis as { window?: unknown }).window;
  });

  it("counts a wheel turned up, not down", () => {
    const el = new EventTarget() as HTMLElement;
    const w = watchScrollInput(el);
    el.dispatchEvent(ev("wheel", { deltaY: 40 }));
    expect(w.recent()).toBe(false);
    el.dispatchEvent(ev("wheel", { deltaY: -40 }));
    expect(w.recent()).toBe(true);
    w.dispose();
  });
  it("counts keys that scroll up, not typing", () => {
    const el = new EventTarget() as HTMLElement;
    const w = watchScrollInput(el);
    const box = { tagName: "TEXTAREA", isContentEditable: false };
    const typed = ev("keydown", { key: "ArrowUp" });
    Object.defineProperty(typed, "target", { value: box });
    win.dispatchEvent(typed);
    expect(w.recent()).toBe(false);
    win.dispatchEvent(ev("keydown", { key: "PageUp" }));
    expect(w.recent()).toBe(true);
    w.dispose();
  });
  it("counts a press for as long as it is held", () => {
    vi.useFakeTimers();
    const el = new EventTarget() as HTMLElement;
    const w = watchScrollInput(el);
    el.dispatchEvent(ev("pointerdown"));
    vi.advanceTimersByTime(5000);
    expect(w.recent()).toBe(true);
    win.dispatchEvent(ev("pointerup"));
    vi.advanceTimersByTime(1000);
    expect(w.recent()).toBe(false);
    w.dispose();
    vi.useRealTimers();
  });
});
