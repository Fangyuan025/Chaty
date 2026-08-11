/** use_skill on a directory-shaped official skill must materialize its
 *  support files into the workspace exactly once (rev-keyed), substitute
 *  {SKILL_ROOT}, and stay hands-off when a user skill shadows the name. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");
const { officialSkills, parseSkill } = await import("./skillFiles");

type Ev = { type: string; [k: string]: unknown };
type Chan = { onmessage?: (ev: Ev) => void };

const call = (name: string, args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;

async function runTurn(
  rounds: string[],
  skills: ReturnType<typeof officialSkills>,
  files: Map<string, string>,
): Promise<{ writes: string[]; results: string[] }> {
  const script = [...rounds];
  const writes: string[] = [];
  const results: string[] = [];
  mockIPC(async (cmd, args) => {
    if (cmd === "generate") {
      const ch = (args as { onEvent: Chan }).onEvent;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "agent_write_file") {
      const a = args as { path?: string; content?: string };
      writes.push(String(a.path));
      files.set(String(a.path), String(a.content ?? ""));
      return "written";
    }
    if (cmd === "agent_read_file") {
      const p = String((args as { path?: string }).path);
      if (files.has(p)) return files.get(p);
      throw new Error("no such file");
    }
    if (cmd === "agent_bash") return { stdout: "ok", stderr: "", code: 0, timedOut: false, bgId: null };
    if (cmd === "agent_list_files") return [];
    return null;
  });
  await runAgentTurn(
    "make me a short video",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off", maxSteps: 8,
      skills,
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: (s: { result?: string }) => { if (s.result) results.push(s.result); },
      onFinal: () => {},
      onError: (m: string) => { throw new Error(`loop errored: ${m}`); },
      onTrace: () => {},
    },
  );
  return { writes, results };
}

describe("directory-shaped skill materialization", () => {
  afterEach(() => clearMocks());

  it("first use writes scripts + rev, substitutes {SKILL_ROOT}; second use is zero writes", async () => {
    const files = new Map<string, string>();
    const skills = officialSkills();

    const first = await runTurn([call("use_skill", { name: "tiktok-video" }), "Done."], skills, files);
    const skillWrites = first.writes.filter((p) => p.startsWith(".chaty/skills/tiktok-video/"));
    expect(skillWrites).toContain(".chaty/skills/tiktok-video/scripts/pipeline.py");
    expect(skillWrites).toContain(".chaty/skills/tiktok-video/scripts/setup.sh");
    expect(skillWrites).toContain(".chaty/skills/tiktok-video/.bundle-rev");
    expect(files.get(".chaty/skills/tiktok-video/scripts/pipeline.py")).toContain("One-shot pipeline");

    const body = first.results.find((r) => r.includes("tiktok-video"))!;
    expect(body).toContain(".chaty/skills/tiktok-video/scripts/setup.sh");
    expect(body).not.toContain("{SKILL_ROOT}");

    // Same bundle rev already on disk → the second use is read-only.
    const second = await runTurn([call("use_skill", { name: "tiktok-video" }), "Done."], skills, files);
    expect(second.writes.filter((p) => p.startsWith(".chaty/skills/"))).toHaveLength(0);
    expect(second.results.find((r) => r.includes("tiktok-video"))).toContain("scripts/pipeline.py");
  });

  it("bridge read semantics (missing file resolves undefined) still materializes", async () => {
    const files = new Map<string, string>();
    const writes: string[] = [];
    const script = [call("use_skill", { name: "tiktok-video" }), "Done."];
    mockIPC(async (cmd, args) => {
      if (cmd === "generate") {
        const ch = (args as { onEvent: { onmessage?: (ev: { type: string; [k: string]: unknown }) => void } }).onEvent;
        ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
        ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
        return null;
      }
      if (cmd === "agent_write_file") {
        const a = args as { path?: string; content?: string };
        writes.push(String(a.path));
        files.set(String(a.path), String(a.content ?? ""));
        return "written";
      }
      // The real bench bridge RESOLVES undefined for a missing file instead
      // of rejecting — the shape that cost run #1 all fourteen files.
      if (cmd === "agent_read_file") return files.get(String((args as { path?: string }).path));
      if (cmd === "agent_bash") return { stdout: "ok", stderr: "", code: 0, timedOut: false, bgId: null };
      if (cmd === "agent_list_files") return [];
      return null;
    });
    await runAgentTurn(
      "make me a short video", [], "/tmp/ws", "en",
      { thinkMode: "off", maxSteps: 8, skills: officialSkills(), signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }) } as never,
      { onThinking: () => {}, onAssistantText: () => {}, onStep: () => {}, onFinal: () => {},
        onError: (m: string) => { throw new Error(`loop errored: ${m}`); }, onTrace: () => {} },
    );
    expect(writes).toContain(".chaty/skills/tiktok-video/scripts/pipeline.py");
    expect(writes).toContain(".chaty/skills/tiktok-video/.bundle-rev");
  });

  it("a shadowing user skill keeps its own files — no materialization", async () => {
    const files = new Map<string, string>();
    const mine = parseSkill(
      "---\nname: tiktok-video\ndescription: my own\n---\nMy own steps, my own scripts.",
      ".chaty/skills/tiktok-video.md",
      "project",
    )!;
    const merged = [mine, ...officialSkills().filter((s) => s.name !== "tiktok-video")];
    const run = await runTurn([call("use_skill", { name: "tiktok-video" }), "Done."], merged, files);
    expect(run.writes.filter((p) => p.startsWith(".chaty/skills/"))).toHaveLength(0);
    expect(run.results.find((r) => r.includes("My own steps"))).toBeTruthy();
  });
});

describe("skill sync live layer", () => {
  afterEach(() => clearMocks());

  async function turnWithLive(
    live: unknown,
    files: Map<string, string>,
  ): Promise<string[]> {
    const writes: string[] = [];
    mockIPC(async (cmd, args) => {
      if (cmd === "skill_live_support") return live;
      if (cmd === "generate") {
        const ch = (args as { onEvent: Chan }).onEvent;
        ch.onmessage?.({ type: "token", text: call("use_skill", { name: "tiktok-video" }) });
        ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
        return null;
      }
      if (cmd === "agent_write_file") {
        const a = args as { path?: string; content?: string };
        writes.push(String(a.path));
        files.set(String(a.path), String(a.content ?? ""));
        return "written";
      }
      if (cmd === "agent_read_file") {
        const p = String((args as { path?: string }).path);
        if (files.has(p)) return files.get(p);
        throw new Error("no such file");
      }
      if (cmd === "agent_bash") return { stdout: "ok", stderr: "", code: 0, timedOut: false, bgId: null };
      if (cmd === "agent_list_files") return [];
      return null;
    });
    const skills = officialSkills().filter((sk) => sk.name === "tiktok-video");
    await new Promise<void>((resolve) => {
      void runAgentTurn(
        "make a video",
        [] as never,
        "/ws",
        "zh",
        {
          thinkMode: "off", nCtx: 8192, maxSteps: 2, temperature: 0.2, bashTimeout: 5,
          browserTextMode: true, signal: { cancelled: false } as never,
          approve: async () => true, approveDir: async () => false,
          approveSudo: async () => ({ ok: false }),
          skills,
        } as never,
        {
          onThinking: () => {}, onAssistantText: () => {}, onPlan: () => {}, onStep: () => {},
          onFinal: () => resolve(), onError: () => resolve(),
          onAskUser: async (_q: string, o: string[]) => o[0] ?? "continue",
        } as never,
      );
    });
    return writes;
  }

  it("a live layer replaces the bundled support set and stamps a combined rev", async () => {
    const files = new Map<string, string>();
    await turnWithLive(
      { rev: "deadbee", files: [{ path: "scripts/pipeline.py", text: "print('LIVE')" }] },
      files,
    );
    const root = [...files.keys()].find((k) => k.endsWith("scripts/pipeline.py"))!;
    expect(files.get(root)).toBe("print('LIVE')");
    const revFile = [...files.keys()].find((k) => k.endsWith(".bundle-rev"))!;
    expect(files.get(revFile)).toMatch(/\+deadbee\n$/);
    // bundled-only files are NOT written when the live layer is in effect
    expect([...files.keys()].some((k) => k.endsWith("scripts/tts.py"))).toBe(false);
  });

  it("undefined/reject from the backend falls back to bundled files (bridge semantics)", async () => {
    for (const live of [undefined, null]) {
      const files = new Map<string, string>();
      await turnWithLive(live, files);
      expect([...files.keys()].some((k) => k.endsWith("scripts/pipeline.py"))).toBe(true);
      const revFile = [...files.keys()].find((k) => k.endsWith(".bundle-rev"))!;
      expect(files.get(revFile)).not.toContain("+");
      clearMocks();
    }
  });
});
