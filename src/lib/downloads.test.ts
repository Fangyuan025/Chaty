import { afterEach, describe, expect, it, vi } from "vitest";

const cancelled: string[] = [];
vi.mock("./ipc", () => ({ cancelDownload: async (key: string) => void cancelled.push(key) }));

const dl = await import("./downloads");

describe("downloads, the app's", () => {
  afterEach(() => {
    cancelled.length = 0;
  });

  it("goes from begun to done, telling whoever listens once", () => {
    const heard: string[] = [];
    const off = dl.onDownloadDone((t) => heard.push(`${t.name}:${t.path}`));
    expect(dl.dlBegin({ id: "a", name: "Qwen3.5-9B", variant: "Q4_K_M", repo: "unsloth/Qwen3.5-9B-GGUF", total: 100 })).toBe(true);
    // The same thing is not fetched twice at once.
    expect(dl.dlBegin({ id: "a", name: "Qwen3.5-9B", variant: "Q4_K_M", repo: "x", total: 100 })).toBe(false);
    dl.dlProgress("a", 40);
    expect(dl.dlGet("a")?.done).toBe(40);
    expect(dl.dlOverall([dl.dlGet("a")!])).toBeCloseTo(0.4);
    dl.dlDone("a", "/models/Qwen3.5-9B/q.gguf");
    expect(dl.dlGet("a")?.state).toBe("done");
    expect(heard).toEqual(["Qwen3.5-9B:/models/Qwen3.5-9B/q.gguf"]);
    // Finished: no longer counted as running, and put away on request.
    expect(dl.dlOverall([dl.dlGet("a")!])).toBeNull();
    dl.dlDismiss("a");
    expect(dl.dlGet("a")).toBeUndefined();
    off();
  });

  it("stops the file in flight, and the files after it, when cancelled", async () => {
    dl.dlBegin({ id: "b", name: "Gemma", variant: "Q8_0", repo: "r", total: 10 });
    dl.dlFile("b", "gemma-Q8_0.gguf");
    await dl.dlCancel("b");
    expect(cancelled).toEqual(["gemma-Q8_0.gguf"]);
    expect(dl.dlStopping("b")).toBe(true);
    dl.dlCancelled("b");
    // A cancel leaves nothing to show.
    expect(dl.dlGet("b")).toBeUndefined();
    expect(dl.dlStopping("b")).toBe(false);
  });

  it("keeps a failure to be read until put away", () => {
    dl.dlBegin({ id: "c", name: "M", variant: "Q4", repo: "r", total: 10 });
    dl.dlFailed("c", "HTTP 404");
    expect(dl.dlGet("c")?.state).toBe("error");
    expect(dl.dlGet("c")?.error).toBe("HTTP 404");
    dl.dlDismiss("c");
    expect(dl.dlGet("c")).toBeUndefined();
  });
});
