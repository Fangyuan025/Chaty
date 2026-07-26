/**
 * ChatyMCP-Bench runner: the real agent loop against real MCP servers, with
 * tools registered through the real registry. Mirrors bench/web/runner.mts —
 * same Bridge to chaty-headless, same budgets, same per-task isolation — but
 * the world under test is an MCP server's state instead of a web fixture.
 *
 *   CHATY_BENCH_MODEL=/path/to/model npx tsx bench/mcp/runner.mts [--only id]
 */
import { spawn, type ChildProcess } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Bridge, type Json, norm } from "../lib/bridge.mts";

const DIR = path.dirname(fileURLToPath(import.meta.url));

// agentLoop pulls in the Tauri IPC layer, which wants a window; the MCP
// bridge wants localStorage. Both must exist BEFORE any of that is imported.
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
const lsStore = new Map<string, string>();
g.localStorage = {
  getItem: (k: string) => lsStore.get(k) ?? null,
  setItem: (k: string, v: string) => void lsStore.set(k, v),
  removeItem: (k: string) => void lsStore.delete(k),
};

interface Task {
  id: string;
  server: "filesystem" | "memory" | "everything";
  type: "state" | "answer";
  instruction: string;
  seedFiles?: Record<string, string>;
  grade:
    | { kind: "file"; path: string; contains: string; absent?: string }
    | { kind: "graph"; entity: string; contains: string }
    | { kind: "answer"; expect: string };
}



async function main() {
  const argv = process.argv.slice(2);
  const flag = (n: string) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : undefined; };
  const only = flag("--only");
  const bin = process.env.CHATY_HEADLESS_BIN ?? path.join(DIR, "../../src-tauri/target/debug/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  if (!existsSync(bin)) throw new Error(`chaty-headless not found at ${bin} — build it first`);

  const tasks: Task[] = JSON.parse(readFileSync(path.join(DIR, "tasks.json"), "utf8"));
  const names = only ? tasks.filter((t) => t.id === only) : tasks;
  if (!names.length) throw new Error(`no tasks match --only ${only}`);

  const bridge = new Bridge(bin);
  const info = (await bridge.call("load_model", { path: model, nCtx: 16384 })) as Json;
  console.log(`model loaded: ${JSON.stringify(info).slice(0, 160)}`);

  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC((cmd: string, args?: Json) => bridge.ipc(cmd, args));
  const { runAgentTurn } = await import("../../src/lib/agentLoop");
  const { syncMcpServers } = await import("../../src/lib/mcp");

  const runsDir = path.join(DIR, "runs");
  mkdirSync(runsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  const outFile = path.join(runsDir, `mcp-${stamp}.jsonl`);
  console.log(`${names.length} task(s): ${names.map((t) => t.id).join(", ")}`);

  let resolved = 0;
  for (const task of names) {
    const t0 = Date.now();
    // Fresh state per task: a temp dir the filesystem server is scoped to and
    // the memory server persists into.
    const stateDir = mkdtempSync(path.join(tmpdir(), "chaty-mcpbench-"));
    const ws = mkdtempSync(path.join(tmpdir(), "chaty-mcpbench-ws-"));
    for (const [name, body] of Object.entries(task.seedFiles ?? {})) {
      writeFileSync(path.join(stateDir, name), body);
    }
    const memFile = path.join(stateDir, "memory.json");
    const cfg = {
      filesystem: { name: "filesystem", enabled: true, transport: "stdio" as const, command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem@2026.7.10", stateDir], trusted: true },
      memory: { name: "memory", enabled: true, transport: "stdio" as const, command: "npx", args: ["-y", "@modelcontextprotocol/server-memory@2026.7.4"], env: { MEMORY_FILE_PATH: memFile }, trusted: true },
      everything: { name: "everything", enabled: true, transport: "stdio" as const, command: "npx", args: ["-y", "@modelcontextprotocol/server-everything@2026.7.4"], trusted: true },
    }[task.server];

    let steps = 0, turns = 0, error: string | undefined, finalText = "";
    try {
      const status = await syncMcpServers([cfg]);
      if (status[0]?.error) throw new Error(`connect: ${status[0].error}`);
      await bridge.call("agent_set_workspace", { path: ws });
      const instruction = task.instruction.replaceAll("{DIR}", stateDir);
      const history: { role: string; content: string; images: string[] }[] = [];
      let prompt = instruction;
      const signal = { cancelled: false };
      const seenSteps = new Set<string>();
      for (turns = 1; turns <= 2; turns++) {
        let paused = false;
        await new Promise<void>((resolve) => {
          // Same option/callback split as the web bench: options object, then
          // the callbacks object (the loop reads onPrefill et al. off the
          // second argument — passing one merged object silently breaks it).
          runAgentTurn(prompt, history as never, ws, "en", {
            thinkMode: "off",
            nCtx: 16384,
            maxSteps: 30,
            temperature: 0.2,
            bashTimeout: 120,
            browserTextMode: true,
            signal: signal as never,
            approve: async () => true,
            approveDir: async () => false,
            approveSudo: async () => ({ ok: false }),
          } as never, {
            onThinking: () => {},
            onAssistantText: () => {},
            onStep: (st: { id: string }) => {
              if (!seenSteps.has(st.id)) { seenSteps.add(st.id); steps++; }
            },
            onFinal: (text: string, _think: unknown, reason?: string) => {
              finalText = text;
              paused = reason === "steps";
              resolve();
            },
            onError: (m: string) => { error = m; resolve(); },
            onAskUser: async (_q: string, options: string[]) => options[0] ?? "continue",
          } as never);
        });
        if (!paused) break;
        prompt = "Continue.";
      }
    } catch (e) {
      error = String(e);
    }

    // ── Grade from server state, not the transcript ──
    let ok = false;
    try {
      const g = task.grade;
      if (g.kind === "file") {
        const body = readFileSync(path.join(stateDir, g.path), "utf8");
        ok = body.includes(g.contains) && (!g.absent || !body.includes(g.absent));
      } else if (g.kind === "graph") {
        const raw = existsSync(memFile) ? readFileSync(memFile, "utf8") : "";
        ok = raw.includes(g.entity) && raw.includes(g.contains);
      } else {
        ok = norm(finalText).includes(norm(g.expect));
      }
    } catch (e) {
      error ??= `grade: ${e}`;
    }

    await syncMcpServers([]); // disconnect before the next task
    const row = { task: task.id, server: task.server, resolved: ok, steps, turns, seconds: Math.round((Date.now() - t0) / 1000), why: error };
    appendFileSync(outFile, JSON.stringify(row) + "\n");
    console.log(`${ok ? "✓" : "✗"} ${task.id}  ${row.seconds}s  ${steps} steps${error ? `  — ${error.slice(0, 90)}` : ""}`);
    if (ok) resolved++;
    rmSync(stateDir, { recursive: true, force: true });
    rmSync(ws, { recursive: true, force: true });
  }

  console.log(`\nMCP bench: ${resolved}/${names.length}`);
  bridge.kill();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
