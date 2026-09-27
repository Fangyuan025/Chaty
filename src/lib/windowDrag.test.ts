import { beforeAll, describe, expect, it } from "vitest";

// Just enough of an element for the walk: a tag and attributes. The tests run
// without a DOM, so the global HTMLElement is this.
class El {
  constructor(
    public tagName: string,
    private attrs: Record<string, string> = {},
  ) {}
  getAttribute(name: string): string | null {
    return name in this.attrs ? this.attrs[name] : null;
  }
}

let pressKind: typeof import("./windowDrag").pressKind;
beforeAll(async () => {
  (globalThis as unknown as { HTMLElement: typeof El }).HTMLElement = El;
  ({ pressKind } = await import("./windowDrag"));
});

const path = (...els: El[]) => els as unknown as EventTarget[];
const div = (attrs?: Record<string, string>) => new El("DIV", attrs);
const drag = { "data-tauri-drag-region": "" };

describe("pressKind", () => {
  it("leaves a press on a drag region itself to Tauri", () => {
    const row = div(drag);
    expect(pressKind(path(row, div(), div()))).toBe("tauri");
  });

  it("moves the window from a backdrop that covers the title row", () => {
    // Live mode, a dialog's backdrop: plain surface over the top band.
    expect(pressKind(path(div(), div(), new El("BODY")))).toBe("free");
  });

  it("moves it from plain surface inside the row, where Tauri declines", () => {
    // A wrapper div inside the row: Tauri only takes presses on the region
    // element itself.
    expect(pressKind(path(div(), div(drag), div()))).toBe("free");
  });

  it("never takes a press meant for a control", () => {
    expect(pressKind(path(new El("BUTTON"), div(drag)))).toBe("block");
    expect(pressKind(path(new El("svg" as string), new El("BUTTON"), div()))).toBe("block");
    expect(pressKind(path(div({ role: "separator" }), div()))).toBe("block");
    expect(pressKind(path(div({ tabindex: "0" }), div()))).toBe("block");
    expect(pressKind(path(new El("INPUT")))).toBe("block");
  });

  it("finds a control above a region Tauri declined", () => {
    // A popover's plain text inside a button that sits in the row.
    expect(pressKind(path(div(), div(drag), new El("BUTTON")))).toBe("block");
  });

  it("respects an explicit opt-out", () => {
    expect(pressKind(path(div(), div({ "data-tauri-drag-region": "false" }), div(drag)))).toBe("block");
  });

  it("takes a deep region's whole subtree as Tauri's", () => {
    expect(pressKind(path(div(), div({ "data-tauri-drag-region": "deep" })))).toBe("tauri");
  });
});
