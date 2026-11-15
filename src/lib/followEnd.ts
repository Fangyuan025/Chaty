// Following a transcript's end as it is written, and letting the user take
// the view away — by what they do, not by where the view happens to be.
//
// A view kept at the end and found higher than it was left is either the
// user scrolling — their wheel moves the page on WebKit's scrolling thread,
// its event reaching the page a frame or several later, behind the reply
// being drawn — or WebKit moving it: taking out a block that holds anything
// positioned or animated on its own layer (a reply's copy button, words
// fading in) clamps the view to the page without that block, there and
// then, before what replaces it is put in — a block moved from one place to
// another, a card's body changed — and the view stays where it was clamped
// to, the end out of sight. Read as the user, that ended the following: in
// Code mode with tool calls folded, every time the model went from speaking
// to calling a tool.
//
// So the view is let go only for the user's input, and otherwise put back:
// - moved by a change React made (`BeforeCommit` notes where it was before
//   the change; see `movedByCommit`), it is put back before it is drawn;
// - moved with no input seen, it is held where it is a moment (`Hold`) for
//   the wheel's event to arrive; if it does, or the view goes on up (a
//   scrollbar dragged), the user has it; if not, it goes back to the end.
import { Component, type ReactNode } from "react";

/** Calls `onBefore` as React is about to change the page under it, every
 *  time it does (put it where it renders whenever the transcript does). */
export class BeforeCommit extends Component<{ onBefore: () => void; children?: ReactNode }> {
  getSnapshotBeforeUpdate(): null {
    this.props.onBefore();
    return null;
  }
  componentDidUpdate(): void {}
  render(): ReactNode {
    return this.props.children;
  }
}

/** At `before` (taken just before a change), as far down as the last pin
 *  left it (`pinned`), and higher now: the change moved it, not the user —
 *  their scrolling is applied between tasks, never inside one. */
export function movedByCommit(el: HTMLElement, before: number | null, pinned: number): boolean {
  return before != null && pinned >= 0 && before >= pinned - 12 && el.scrollTop < before - 12;
}

const NAV_UP = new Set(["PageUp", "ArrowUp", "Home"]);

function editable(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  return !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
}

/** When the user last did something that can scroll `el` up: a wheel or
 *  trackpad turned up, a key that scrolls, a press on it (its scrollbar,
 *  or a drag that selects past its edge), a touch. */
export function watchScrollInput(el: HTMLElement): { recent: (ms?: number) => boolean; dispose: () => void } {
  // Never, to begin with: not "at time 0", which an app opened under a
  // second ago is still close to.
  let at = Number.NEGATIVE_INFINITY;
  let pressed = false;
  const mark = () => {
    at = performance.now();
  };
  const onWheel = (e: WheelEvent) => {
    if (e.deltaY < 0) mark();
  };
  const onKey = (e: KeyboardEvent) => {
    if (!editable(e.target) && (NAV_UP.has(e.key) || (e.key === " " && e.shiftKey))) mark();
  };
  const onDown = () => {
    pressed = true;
    mark();
  };
  const onUp = () => {
    pressed = false;
  };
  const onMove = () => {
    if (pressed) mark();
  };
  el.addEventListener("wheel", onWheel, { passive: true });
  el.addEventListener("pointerdown", onDown, { passive: true });
  el.addEventListener("touchstart", mark, { passive: true });
  el.addEventListener("touchmove", mark, { passive: true });
  window.addEventListener("pointermove", onMove, { passive: true });
  window.addEventListener("pointerup", onUp, { passive: true });
  window.addEventListener("keydown", onKey, true);
  return {
    recent: (ms = 800) => pressed || performance.now() - at < ms,
    dispose: () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("touchstart", mark);
      el.removeEventListener("touchmove", mark);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("keydown", onKey, true);
    },
  };
}

/** The view held where it was found, not pinned, while it is learned
 *  whether the user moved it; `back` puts it at the end if nothing says so. */
export class Hold {
  private top = -1;
  private timer: ReturnType<typeof setTimeout> | 0 = 0;
  constructor(
    private back: () => void,
    private ms = 220,
  ) {}
  get active(): boolean {
    return this.timer !== 0;
  }
  start(el: HTMLElement): void {
    if (this.timer) return;
    this.top = el.scrollTop;
    this.timer = setTimeout(() => {
      this.timer = 0;
      this.back();
    }, this.ms);
  }
  /** Gone on up since it was held: someone is moving it. */
  further(el: HTMLElement): boolean {
    return this.timer !== 0 && el.scrollTop < this.top - 2;
  }
  clear(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = 0;
  }
}
