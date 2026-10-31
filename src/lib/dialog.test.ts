/** On Linux a picker asks the desktop's portal first (issue #29) and shows
 *  the plugin's GTK dialog only when no portal answers; a cancelled portal
 *  dialog is a cancel, not a second dialog. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
// plugin-os reads the platform from an injected global at import time.
g.__TAURI_OS_PLUGIN_INTERNALS__ = { platform: "linux", os_type: "linux", family: "unix" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { open, save } = await import("./dialog");

type Picked = { kind: "unavailable" } | { kind: "cancelled" } | { kind: "paths"; paths: string[] };

function answer(picked: Picked) {
  const calls: { cmd: string; args: Record<string, unknown> }[] = [];
  mockIPC((cmd, args) => {
    calls.push({ cmd, args: (args ?? {}) as Record<string, unknown> });
    if (cmd === "portal_pick") return picked;
    if (cmd === "plugin:dialog|open") return "/home/me/from-gtk";
    if (cmd === "plugin:dialog|save") return "/home/me/saved-by-gtk.md";
    return null;
  });
  return calls;
}

afterEach(() => clearMocks());

describe("pickers on Linux", () => {
  it("take the portal's folder", async () => {
    const calls = answer({ kind: "paths", paths: ["/home/me/project"] });
    expect(await open({ directory: true })).toBe("/home/me/project");
    expect(calls.map((c) => c.cmd)).toEqual(["portal_pick"]);
    expect(calls[0].args.request).toMatchObject({ directory: true, multiple: false, save: false });
  });

  it("give every file when several are asked for", async () => {
    answer({ kind: "paths", paths: ["/a.md", "/b.md"] });
    expect(await open({ multiple: true, filters: [{ name: "Markdown", extensions: ["md"] }] })).toEqual(["/a.md", "/b.md"]);
  });

  it("show the GTK dialog when no portal answers", async () => {
    const calls = answer({ kind: "unavailable" });
    expect(await open({ directory: true })).toBe("/home/me/from-gtk");
    expect(calls.map((c) => c.cmd)).toEqual(["portal_pick", "plugin:dialog|open"]);
    expect(await save({ defaultPath: "chat.md" })).toBe("/home/me/saved-by-gtk.md");
  });

  it("stop at a cancel", async () => {
    const calls = answer({ kind: "cancelled" });
    expect(await open({ directory: true })).toBeNull();
    expect(await save({ defaultPath: "chat.md" })).toBeNull();
    expect(calls.map((c) => c.cmd)).toEqual(["portal_pick", "portal_pick"]);
  });

  it("save where the portal says", async () => {
    const calls = answer({ kind: "paths", paths: ["/home/me/notes/chat.md"] });
    expect(await save({ defaultPath: "chat.md", filters: [{ name: "Markdown", extensions: ["md"] }] })).toBe(
      "/home/me/notes/chat.md",
    );
    expect(calls[0].args.request).toMatchObject({ save: true, defaultPath: "chat.md" });
  });
});
