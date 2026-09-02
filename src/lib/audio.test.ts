import { beforeEach, describe, expect, test, vi } from "vitest";

/** A context whose `resume()` can behave the three ways a real one does:
 *  answer, refuse, or never answer at all — the last being the failure with no
 *  error to catch, and the one that used to kill speech until an app restart. */
class FakeContext {
  static behaviour: "ok" | "reject" | "hang" = "ok";
  static created: FakeContext[] = [];

  state = "suspended";
  destination = {};

  constructor() {
    FakeContext.created.push(this);
  }
  createBuffer() {
    return { copyToChannel() {} };
  }
  createBufferSource() {
    return {
      buffer: null as unknown,
      onended: null as null | (() => void),
      connect() {},
      disconnect() {},
      start(this: { onended: null | (() => void) }) {
        this.onended?.();
      },
      stop() {},
    };
  }
  createAnalyser() {
    return { fftSize: 0, connect() {}, disconnect() {} };
  }
  close() {
    this.state = "closed";
    return Promise.resolve();
  }
  resume() {
    if (FakeContext.behaviour === "hang") return new Promise<void>(() => {});
    if (FakeContext.behaviour === "reject") return Promise.reject(new Error("interrupted"));
    this.state = "running";
    return Promise.resolve();
  }
}

(globalThis as { AudioContext?: unknown }).AudioContext = FakeContext;

/** The shared output context lives in module scope, so each case gets its own
 *  copy of the module — otherwise one test's context answers the next one. */
async function freshAudio() {
  vi.resetModules();
  const { playAudio } = await import("./audio");
  return () => playAudio(new Float32Array(8), 24_000);
}

describe("audio output recovers from a wedged context", () => {
  beforeEach(() => {
    FakeContext.behaviour = "ok";
    FakeContext.created = [];
    vi.useRealTimers();
  });

  test("plays through one context while it keeps answering", async () => {
    const speak = await freshAudio();
    await speak().done;
    await speak().done;
    expect(FakeContext.created.length).toBe(1);
  });

  test("a resume that never answers does not hang the utterance", async () => {
    // The real symptom: no audio, no error, and the promise never settling
    // would have left the caller waiting forever.
    const speak = await freshAudio();
    await speak().done; // establish the shared context
    FakeContext.behaviour = "hang";
    vi.useFakeTimers();
    const pending = speak().done;
    await vi.advanceTimersByTimeAsync(2_000);
    await expect(pending).resolves.toBeUndefined();
  });

  test("the next utterance starts over on a fresh context", async () => {
    const speak = await freshAudio();
    await speak().done;
    expect(FakeContext.created.length).toBe(1);

    FakeContext.behaviour = "hang";
    vi.useFakeTimers();
    const pending = speak().done;
    await vi.advanceTimersByTimeAsync(2_000);
    await pending;
    vi.useRealTimers();

    // Speech that stopped used to stay stopped: the dead context was reused
    // for every later utterance because it was not `closed`.
    FakeContext.behaviour = "ok";
    await speak().done;
    expect(FakeContext.created.length).toBe(2);
  });

  test("a refused resume is treated the same way", async () => {
    const speak = await freshAudio();
    await speak().done;
    FakeContext.behaviour = "reject";
    await speak().done;
    FakeContext.behaviour = "ok";
    await speak().done;
    expect(FakeContext.created.length).toBe(2);
  });

  test("an interrupted context is replaced before it is spoken into", async () => {
    const speak = await freshAudio();
    await speak().done;
    // WebKit's own non-standard state, which `closed` alone never matched.
    FakeContext.created[0].state = "interrupted";
    await speak().done;
    expect(FakeContext.created.length).toBe(2);
  });
});
