/**
 * ChatyCoder-Bench runner — drives the REAL production agent loop
 * (src/lib/agentLoop.ts) headless in Node, so a benchmark task exercises the
 * full Coder tool layer: fused search_code, symbol reads, syntax gate,
 * validate_change, understand_repo, doc reading — everything a user gets.
 *
 * Architecture:
 *   runAgentTurn (real agentLoop.ts)
 *     └─ ipc.ts invoke()  ── intercepted by @tauri-apps/api/mocks mockIPC
 *          └─ Bridge ── stdio JSON-lines ──> chaty-headless (Rust bin:
 *               agent.rs tools + inference engine, streaming generate)
 *
 * Task layout (one dir per task):
 *   tasks/<name>/task.md      the instruction given to the agent
 *   tasks/<name>/workspace/   fixture copied to a temp dir per run
 *   tasks/<name>/grade.sh     run inside the workspace after the agent
 *                             finishes; exit 0 = resolved
 *
 * Run:  npx tsx bench/coder/runner.mts [--tasks bench/coder/tasks] [--only name]
 * Env:  CHATY_HEADLESS_BIN  path to the chaty-headless binary
 *       CHATY_BENCH_MODEL   model path/dir forwarded to the bin's load command
 */

// agentLoop's imports expect a browser-ish global before anything Tauri loads.
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-bench" };

import { spawn, execFileSync, type ChildProcess } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import { mkdtempSync, cpSync, readFileSync, appendFileSync, mkdirSync, readdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

type Json = Record<string, unknown>;

/** stdio JSON-lines client for the chaty-headless Rust binary. */
class Bridge {
  private proc: ChildProcess;
  private rl: Interface;
  private nextId = 1;
  // reject carries the RAW error string (not an Error) — Tauri command
  // rejections are plain strings and agentLoop matches markers like
  // NEED_DIR_GRANT against them verbatim.
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: unknown) => void; onEvent?: (ev: unknown) => void }>();

  constructor(bin: string) {
    this.proc = spawn(bin, [], { stdio: ["pipe", "pipe", "inherit"] });
    this.rl = createInterface({ input: this.proc.stdout! });
    this.rl.on("line", (line) => {
      let msg: Json;
      try { msg = JSON.parse(line); } catch { return; }
      const p = this.pending.get(msg.id as number);
      if (!p) return;
      if (msg.type === "event") { p.onEvent?.(msg.event); return; }
      this.pending.delete(msg.id as number);
      if (msg.type === "error") p.reject(msg.message);
      else p.resolve(msg.result);
    });
  }

  call(cmd: string, args: Json = {}, onEvent?: (ev: unknown) => void): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
      this.proc.stdin!.write(JSON.stringify({ id, cmd, args }) + "\n");
    });
  }

  kill() { this.proc.kill(); }
}

interface TaskResult {
  task: string; resolved: boolean; steps: number; turns: number;
  seconds: number; error?: string;
}

async function main() {
  const argv = process.argv.slice(2);
  const flag = (name: string) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
  const here = path.dirname(new URL(import.meta.url).pathname);
  const tasksDir = path.resolve(flag("--tasks") ?? path.join(here, "tasks"));
  const only = flag("--only");
  const bin = process.env.CHATY_HEADLESS_BIN ?? path.join(here, "../../src-tauri/target/release/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  if (!existsSync(bin)) throw new Error(`chaty-headless not found at ${bin} — build it first`);

  const bridge = new Bridge(bin);
  const info = (await bridge.call("load_model", { path: model, nCtx: 16384 })) as Json;
  console.log(`model loaded: ${JSON.stringify(info)}`);

  // The real production loop + IPC layer, loaded AFTER window/mock are ready.
  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC(async (cmd: string, args?: Json) => {
    if (cmd === "generate") {
      const ch = (args as Json)?.onEvent as { onmessage?: (ev: unknown) => void } | undefined;
      return bridge.call("generate", { request: (args as Json)?.request }, (ev) => ch?.onmessage?.(ev));
    }
    return bridge.call(cmd, args ?? {});
  });
  const { runAgentTurn } = await import("../../src/lib/agentLoop");

  const runsDir = path.join(here, "runs");
  mkdirSync(runsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  const outFile = path.join(runsDir, `${stamp}.jsonl`);

  const names = readdirSync(tasksDir).filter(
    (n) => (!only || n === only) && existsSync(path.join(tasksDir, n, "task.md")),
  );
  console.log(`${names.length} task(s): ${names.join(", ")}`);

  let resolvedCount = 0;
  for (const name of names) {
    const t0 = Date.now();
    const taskDir = path.join(tasksDir, name);
    const ws = mkdtempSync(path.join(tmpdir(), `chaty-bench-${name}-`));
    cpSync(path.join(taskDir, "workspace"), ws, { recursive: true });
    await bridge.call("agent_set_workspace", { path: ws });

    const instruction = readFileSync(path.join(taskDir, "task.md"), "utf8");
    let steps = 0, turns = 0, error: string | undefined;
    const signal = { cancelled: false };

    // One turn; if the loop pauses at the step limit, nudge it on (max 3 turns)
    // — mirrors a user clicking Continue.
    let prompt = instruction;
    const history: unknown[] = [];
    for (turns = 1; turns <= 3; turns++) {
      let pausedAtSteps = false;
      await new Promise<void>((resolve) => {
        runAgentTurn(prompt, history as never, ws, "en", {
          thinkMode: "off",
          nCtx: 16384,
          maxSteps: 40,
          temperature: 0.2,
          bashTimeout: 120,
          signal: signal as never,
          approve: async () => true,          // benchmark = bypass mode
          approveDir: async () => false,      // stay inside the workspace
          approveSudo: async () => ({ ok: false }),
        }, {
          onThinking: () => {},
          onAssistantText: () => {},
          onStep: () => { steps++; },
          onFinal: (_text, _think, reason) => { pausedAtSteps = reason === "steps"; resolve(); },
          onError: (m) => { error = m; resolve(); },
          onAskUser: async (_q, options) => options[0] ?? "continue",
        });
      });
      if (error || !pausedAtSteps) break;
      prompt = "继续";
    }

    let resolved = false;
    if (!error) {
      try { execFileSync("bash", [path.join(taskDir, "grade.sh")], { cwd: ws, timeout: 600_000, stdio: "pipe" }); resolved = true; }
      catch { resolved = false; }
    }
    if (resolved) resolvedCount++;
    const r: TaskResult = { task: name, resolved, steps, turns, seconds: Math.round((Date.now() - t0) / 1000), error };
    appendFileSync(outFile, JSON.stringify(r) + "\n");
    console.log(`${resolved ? "✓" : "✗"} ${name}  ${r.seconds}s  ${steps} steps${error ? `  ERROR: ${error}` : ""}`);
  }

  console.log(`\n${resolvedCount}/${names.length} resolved — results: ${outFile}`);
  bridge.kill();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
