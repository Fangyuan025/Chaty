import { beforeEach, describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");

// In-memory localStorage for the config round-trip.
const store = new Map<string, string>();
(globalThis as Record<string, unknown>).localStorage = {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => void store.set(k, v),
  removeItem: (k: string) => void store.delete(k),
};

const mcp = await import("./mcp");
const { isUntrusted, needsApproval, toolSpec, buildToolsDoc } = await import("./toolRegistry");

const TOOLS = (n: number) =>
  Array.from({ length: n }, (_, i) => ({
    name: `tool_${i}`,
    description: `Does thing ${i}. And a very long second sentence that should never reach the prompt because docs are synthesized lean.`,
    input_schema: {
      type: "object",
      properties: { query: { type: "string" }, page: { type: "number" } },
      required: ["query"],
    },
  }));

let calls: { cmd: string; args: Record<string, unknown> }[] = [];
beforeEach(() => {
  calls = [];
  mockIPC((cmd, args) => {
    calls.push({ cmd, args: args as Record<string, unknown> });
    if (cmd === "mcp_connect") return Promise.resolve(TOOLS(3));
    if (cmd === "mcp_call") return Promise.resolve("result-text");
    return Promise.resolve(null);
  });
});

describe("lean-doc synthesis", () => {
  test("first sentence only, capped", () => {
    expect(mcp.leanDescription("One. Two. Three.")).toBe("One.");
    expect(mcp.leanDescription("x".repeat(300)).length).toBeLessThanOrEqual(110);
  });

  test("args signature marks optionals", () => {
    expect(mcp.leanArgs(TOOLS(1)[0].input_schema)).toBe('{ "query": string, "page"?: number }');
    expect(mcp.leanArgs({})).toBe("{}");
  });

  test("doc line is one line, name-prefixed", () => {
    const line = mcp.leanDocLine("gh__tool_0", TOOLS(1)[0]).en;
    expect(line).toContain("- gh__tool_0: Does thing 0.");
    expect(line).not.toContain("second sentence");
    expect(line.split("\n").length).toBe(1);
  });
});

describe("registration lifecycle", () => {
  test("small server registers core-tier, untrusted, approval-gated", async () => {
    const res = await mcp.syncMcpServers([
      { name: "gh", enabled: true, transport: "stdio", command: "x" },
    ]);
    expect(res).toEqual([{ server: "gh", tools: 3 }]);
    const spec = toolSpec("gh__tool_0");
    expect(spec?.source).toBe("mcp");
    expect(spec?.tier).toBe("core");
    expect(isUntrusted("gh__tool_0")).toBe(true);
    expect(needsApproval("gh__tool_0")).toBe(true); // untrusted server ⇒ approval per call
    expect(buildToolsDoc("en", {})).toContain("- gh__tool_0: Does thing 0.");
    // Disable → tools unregister and the prompt is clean again.
    await mcp.syncMcpServers([{ name: "gh", enabled: false, transport: "stdio", command: "x" }]);
    expect(toolSpec("gh__tool_0")).toBeUndefined();
    expect(buildToolsDoc("en", {})).not.toContain("gh__tool_0");
  });

  test("trusted server skips per-call approval; big server goes deferred", async () => {
    mockIPC((cmd) =>
      Promise.resolve(cmd === "mcp_connect" ? TOOLS(9) : cmd === "mcp_call" ? "ok" : null),
    );
    await mcp.syncMcpServers([
      { name: "big", enabled: true, trusted: true, transport: "stdio", command: "x" },
    ]);
    try {
      expect(needsApproval("big__tool_0")).toBe(false);
      expect(toolSpec("big__tool_0")?.tier).toBe("deferred");
      const doc = buildToolsDoc("en", {});
      expect(doc).not.toContain("- big__tool_0:"); // no doc line…
      expect(doc).toContain("big__tool_0"); // …but indexed
    } finally {
      await mcp.syncMcpServers([]);
    }
  });
});

describe("dispatch", () => {
  test("missing required arg returns the full doc (deferred load path)", async () => {
    await mcp.syncMcpServers([{ name: "gh", enabled: true, transport: "stdio", command: "x" }]);
    const out = await mcp.callMcpTool("gh__tool_1", {});
    expect(out).toContain('ERROR: missing "query"');
    expect(out).toContain("args schema:");
    expect(out).toContain("second sentence"); // FULL description rides along
    // Valid call routes server+tool through the backend.
    const ok = await mcp.callMcpTool("gh__tool_1", { query: "hi" });
    expect(ok).toBe("result-text");
    const call = calls.find((c) => c.cmd === "mcp_call");
    expect(call?.args).toMatchObject({ server: "gh", tool: "tool_1" });
    await mcp.syncMcpServers([]);
  });

  test("unknown tool fails soft (server disabled mid-session)", async () => {
    const out = await mcp.callMcpTool("ghost__x", {});
    expect(out).toContain("not connected");
  });
});
