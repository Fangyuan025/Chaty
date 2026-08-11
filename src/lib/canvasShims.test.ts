import { describe, expect, it } from "vitest";
import { type ConsoleEntry, foldConsoleEntry, PREVIEW_SHIMS, withShims } from "../components/CanvasPanel";

/** The generated shim code must PARSE. A template-escape slip ('\n' in the
 *  TS source becomes a real newline inside the shim's single-quoted string)
 *  produced a SyntaxError that silently killed the console and error shims —
 *  the "canvas console shows nothing" incident. new Function() is a pure
 *  syntax gate: it compiles without executing. */
describe("canvas preview shims", () => {
  for (const [name, tag] of Object.entries(PREVIEW_SHIMS)) {
    it(`${name} generates syntactically valid JS`, () => {
      const bodies = [...tag.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
      // Style-only shims (the scrollbar paint) carry no script — they must at
      // least carry a style block; script-bearing shims must all compile.
      if (bodies.length === 0) {
        expect(tag).toMatch(/<style[\s>]/);
        return;
      }
      for (const body of bodies) {
        expect(() => new Function(body)).not.toThrow();
      }
    });
  }
});

const body = (tag: string) => /<script>([\s\S]*?)<\/script>/.exec(tag)![1];

/** Minimal ErrorEvent for the stubbed window. */
class FakeErrorEvent {
  type: string;
  message = "";
  filename = "";
  lineno = 0;
  colno = 0;
  error: unknown = null;
  constructor(type: string, init: Record<string, unknown> = {}) {
    this.type = type;
    Object.assign(this, init);
  }
}

describe("TRAP_SHIM behavior (WebKit muzzle workaround)", () => {
  function boot(lineOff = 10) {
    const timers: ((...a: unknown[]) => unknown)[] = [];
    const dispatched: FakeErrorEvent[] = [];
    const listeners: { type: string; fn: unknown }[] = [];
    const removed: { type: string; fn: unknown }[] = [];
    class ET {}
    (ET.prototype as Record<string, unknown>).addEventListener = function (type: string, fn: unknown) {
      listeners.push({ type, fn });
    };
    (ET.prototype as Record<string, unknown>).removeEventListener = function (type: string, fn: unknown) {
      removed.push({ type, fn });
    };
    const win: Record<string, unknown> = {
      __CV_LINEOFF: lineOff,
      setTimeout: (fn: (...a: unknown[]) => unknown) => {
        timers.push(fn);
        return 7;
      },
      dispatchEvent: (e: FakeErrorEvent) => {
        dispatched.push(e);
        return true;
      },
      EventTarget: ET,
    };
    new Function("window", "ErrorEvent", body(PREVIEW_SHIMS.TRAP_SHIM))(win, FakeErrorEvent);
    return { win, timers, dispatched, listeners, removed, ET };
  }

  it("a throwing timer callback replays a DETAILED synthetic error and rethrows", () => {
    const { win, timers, dispatched } = boot(10);
    const boom = () => {
      const err = new Error("boom");
      err.stack = "boom@about:srcdoc:52:7\ncaller@about:srcdoc:60:1";
      throw err;
    };
    const id = (win.setTimeout as (fn: unknown, ms: number) => number)(boom, 0);
    expect(id).toBe(7);
    expect(timers).toHaveLength(1);
    expect(() => timers[0]()).toThrow("boom"); // semantics preserved
    expect(dispatched).toHaveLength(1);
    const e = dispatched[0];
    // srcdoc line 52 minus the 10 shim lines = user-source line 42.
    expect(e.lineno).toBe(42);
    expect(e.colno).toBe(7);
    expect(e.filename).toBe("canvas");
    expect((e.error as Error).stack).toContain("canvas:42:7");
    expect((e.error as Error).stack).toContain("canvas:50:1");
    expect(win.__CV_TRAP_MSG).toBe("boom");
    expect(typeof win.__CV_TRAP_AT).toBe("number");
  });

  it("removeEventListener unhooks the SAME wrapped listener addEventListener installed", () => {
    const { listeners, removed, ET } = boot();
    const target = Object.create(ET.prototype) as {
      addEventListener: (t: string, fn: unknown) => void;
      removeEventListener: (t: string, fn: unknown) => void;
    };
    const fn = () => {};
    target.addEventListener("click", fn);
    target.removeEventListener("click", fn);
    expect(listeners).toHaveLength(1);
    expect(removed).toHaveLength(1);
    // The wrapper is a DIFFERENT function (that's the guard), but add/remove
    // must agree on its identity or listeners leak forever.
    expect(listeners[0].fn).not.toBe(fn);
    expect(removed[0].fn).toBe(listeners[0].fn);
    // A function never added passes through untouched.
    const stranger = () => {};
    target.removeEventListener("click", stranger);
    expect(removed[1].fn).toBe(stranger);
  });
});

describe("withShims line offset invariant", () => {
  // The whole line-calibration scheme rests on ONE invariant: user-source
  // line L sits at srcdoc line L + __CV_LINEOFF. The storage shim used to be
  // injected OUTSIDE this block (separate wrapper) and broke it — the
  // owner's 4-line repro reported canvas:16.
  const shapes: Record<string, string> = {
    "bare fragment": `<button onclick="setTimeout(()=>{ boom() },0)">点我</button>\n<script>\nMARKER_LINE_3();\n</script>`,
    "doctyped document": `<!doctype html>\n<html>\n<head><title>t</title></head>\n<body>\nMARKER_LINE_5\n</body>\n</html>`,
  };
  for (const [name, html] of Object.entries(shapes)) {
    it(`${name}: marker line maps to source line + __CV_LINEOFF`, () => {
      const out = withShims(html, "n");
      const K = Number(/__CV_LINEOFF=(\d+)/.exec(out)![1]);
      expect(K).toBeGreaterThan(10);
      const srcLine = html.split("\n").findIndex((l) => l.includes("MARKER_LINE") || l.includes("boom(")) + 1;
      const outLine = out.split("\n").findIndex((l) => l.includes("MARKER_LINE") || l.includes("boom(")) + 1;
      expect(outLine).toBe(srcLine + K);
      // The marker deeper in the document obeys the same offset.
      const deepSrc = html.split("\n").findIndex((l) => l.includes("MARKER_LINE")) + 1;
      const deepOut = out.split("\n").findIndex((l) => l.includes("MARKER_LINE")) + 1;
      expect(deepOut).toBe(deepSrc + K);
      // Storage shim rides INSIDE the counted block now.
      expect(out.slice(0, out.indexOf("MARKER_LINE")).includes("localStorage.getItem")).toBe(true);
    });
  }
});

describe("CONSOLE_SHIM line mapping and duplicate-drop", () => {
  function boot(lineOff = 10) {
    const posted: { level: string; text: string }[] = [];
    const handlers: Record<string, (e: Record<string, unknown>) => void> = {};
    const win: Record<string, unknown> = {
      __CV_LINEOFF: lineOff,
      __CV_NONCE: "n",
      addEventListener: (type: string, fn: (e: Record<string, unknown>) => void) => {
        handlers[type] = fn;
      },
    };
    const parent = {
      postMessage: (m: { __chatyCvConsole?: { level: string; text: string } }) => {
        if (m.__chatyCvConsole) posted.push(m.__chatyCvConsole);
      },
    };
    const fakeConsole: Record<string, unknown> = {};
    new Function("window", "parent", "console", body(PREVIEW_SHIMS.CONSOLE_SHIM))(win, parent, fakeConsole);
    return { win, posted, handlers };
  }

  it("organic detailed errors report USER-source lines and rewritten stacks", () => {
    const { posted, handlers } = boot(10);
    handlers.error({
      message: "x is not defined",
      filename: "about:srcdoc",
      lineno: 52,
      colno: 3,
      error: { stack: "handler@about:srcdoc:52:3" },
    });
    expect(posted).toHaveLength(1);
    expect(posted[0].text).toContain("(canvas:42)");
    expect(posted[0].text).toContain("canvas:42");
    expect(posted[0].text).not.toContain("srcdoc:52");
  });

  it("drops the browser's follow-up after a trap replay (muzzled AND duplicate)", () => {
    const { win, posted, handlers } = boot(10);
    win.__CV_TRAP_AT = Date.now();
    win.__CV_TRAP_MSG = "boom";
    // WebKit's anonymized follow-up.
    handlers.error({ message: "Script error.", filename: "", lineno: 0 });
    // Chromium/WebView2's detailed follow-up for the same throw.
    handlers.error({ message: "boom", filename: "about:srcdoc", lineno: 52 });
    expect(posted).toHaveLength(0);
    // The trap's own synthetic event (filename 'canvas') must pass.
    handlers.error({ message: "boom", filename: "canvas", lineno: 42, error: { stack: "boom@canvas:42:7" } });
    expect(posted).toHaveLength(1);
    expect(posted[0].text).toContain("(canvas:42)");
    // Long after the trap window, an anonymized error still reports (residual).
    win.__CV_TRAP_AT = Date.now() - 5000;
    handlers.error({ message: "Script error.", filename: "", lineno: 0 });
    expect(posted).toHaveLength(2);
    expect(posted[1].text).toBe("Script error.");
  });
});

describe("foldConsoleEntry (devtools-style duplicate folding)", () => {
  it("repeats bump a counter instead of appending", () => {
    let log = foldConsoleEntry([], { level: "error", text: "boom (canvas:3)" });
    log = foldConsoleEntry(log, { level: "error", text: "boom (canvas:3)" });
    log = foldConsoleEntry(log, { level: "error", text: "boom (canvas:3)" });
    expect(log).toHaveLength(1);
    expect(log[0].count).toBe(3);
  });

  it("different text or level stays a separate line", () => {
    let log = foldConsoleEntry([], { level: "error", text: "boom (canvas:3)" });
    log = foldConsoleEntry(log, { level: "error", text: "boom (canvas:5)" });
    log = foldConsoleEntry(log, { level: "warn", text: "boom (canvas:3)" });
    expect(log).toHaveLength(3);
    expect(log.every((c) => (c.count ?? 1) === 1)).toBe(true);
  });

  it("folding still works once the cap is reached (only NEW lines are dropped)", () => {
    let log: ConsoleEntry[] = Array.from({ length: 300 }, (_, i) => ({ level: "log", text: `line ${i}` }));
    log = foldConsoleEntry(log, { level: "log", text: "line 5" });
    expect(log).toHaveLength(300);
    expect(log[5].count).toBe(2);
    log = foldConsoleEntry(log, { level: "log", text: "brand new" });
    expect(log).toHaveLength(300);
  });
});
