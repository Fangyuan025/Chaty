/** The two switches added for issues #12 and #13 must actually reach the
 *  backend — a setting that is stored and then not passed is indistinguishable
 *  from a setting that does nothing, and neither of these can be checked by
 *  hand on the machine that reported the bug. */
import { describe, expect, it, afterEach } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { ragAddDocument, setModelsRoot, getModelsRoot } = await import("./ipc");
const { defaultSettings } = await import("../components/SettingsPanel");

afterEach(() => clearMocks());

/** Record what the backend was asked, and stream a "done" so the call ends. */
function capture() {
  const seen: { cmd: string; args: Record<string, unknown> }[] = [];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    seen.push({ cmd, args: a });
    const ch = a.onProgress as { onmessage?: (p: unknown) => void } | undefined;
    ch?.onmessage?.({ phase: "done", frac: 1 });
    if (cmd === "get_models_root")
      return { custom: "D:\\ChatyModels", effective: "D:\\ChatyModels", available: true };
    return null;
  });
  return seen;
}

describe("knowledge-base image captioning switch (#13)", () => {
  it("passes the choice through on every import", async () => {
    const seen = capture();
    await ragAddDocument("/tmp/book.pdf", () => {}, undefined, false);
    await ragAddDocument("/tmp/book.pdf", () => {}, undefined, true);
    expect(seen.map((s) => s.args.captionImages)).toEqual([false, true]);
  });

  it("leaves the decision to the backend when the caller says nothing", async () => {
    const seen = capture();
    await ragAddDocument("/tmp/book.pdf", () => {});
    // null, not undefined: an omitted key would not survive the IPC boundary
    // as an explicit "unset", and the backend's default is what should win.
    expect(seen[0].args.captionImages).toBeNull();
  });

  it("is on by default, because that is what indexing has always done", () => {
    expect(defaultSettings.kbCaptionImages).toBe(true);
  });
});

describe("models folder (#12)", () => {
  it("sends the chosen path, and null to restore the default", async () => {
    const seen = capture();
    await setModelsRoot("D:\\ChatyModels");
    await setModelsRoot(null);
    expect(seen.map((s) => s.args.path)).toEqual(["D:\\ChatyModels", null]);
  });

  it("reports what the backend says about reachability", async () => {
    capture();
    const info = await getModelsRoot();
    expect(info.custom).toBe("D:\\ChatyModels");
    expect(info.available).toBe(true);
  });
});
