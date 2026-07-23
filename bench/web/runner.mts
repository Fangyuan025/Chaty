/**
 * ChatyWeb-Bench runner — drives the REAL production agent loop
 * (src/lib/agentLoop.ts) against local deterministic web fixtures, so a task
 * exercises the full browser tool chain end to end: browser_navigate /
 * browser_read digests / click / type / eval through agent.rs + browser.rs
 * into a real headless Chrome.
 *
 * Text-browser mode: the bench model has no vision encoder, so the loop runs
 * with `browserTextMode: true` — the browser suite minus the two screenshot
 * tools; browser_read's digest is the model's eyes.
 *
 * Architecture (mirrors bench/coder/runner.mts):
 *   runAgentTurn (real agentLoop.ts)
 *     └─ ipc.ts invoke() ── mockIPC ── Bridge ── stdio ──> chaty-headless
 *   fixture server (server.mts, in-process) ── graded via getState()
 *
 * Run:  npx tsx bench/web/runner.mts [--only task-id]
 * Env:  CHATY_HEADLESS_BIN     path to chaty-headless (default: target/debug)
 *       CHATY_BENCH_MODEL      model path (required)
 *       CHATY_BENCH_TRANSCRIPT per-task transcript dir (optional)
 *       CHATY_BENCH_LANG       zh | en (default en)
 */
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-bench" };

import { spawn, type ChildProcess } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import { mkdtempSync, rmSync, readFileSync, appendFileSync, mkdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer, resetState, getState, DEFAULT_PORT } from "./server.mts";
import { GRADERS } from "./graders.mts";

type Json = Record<string, unknown>;
const DIR = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.CHATY_WEBBENCH_PORT) || DEFAULT_PORT;
const BASE = `http://127.0.0.1:${PORT}`;

class Bridge {
  private proc: ChildProcess;
  private rl: Interface;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: unknown) => void; onEvent?: (ev: unknown) => void }>();
  constructor(bin: string) {
    this.proc = spawn(bin, [], {
      stdio: ["pipe", "pipe", "inherit"],
      env: { ...process.env, CHATY_BROWSER_HEADLESS: "1" },
    });
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

type Task = {
  id: string;
  site: string;
  type: "state" | "answer";
  instruction: string;
};

async function main() {
  const argv = process.argv.slice(2);
  const flag = (name: string) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
  const only = flag("--only");
  const bin = process.env.CHATY_HEADLESS_BIN ?? path.join(DIR, "../../src-tauri/target/debug/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  if (!existsSync(bin)) throw new Error(`chaty-headless not found at ${bin} — build it first`);

  const tasks: Task[] = JSON.parse(readFileSync(path.join(DIR, "tasks.json"), "utf8"));
  const names = only ? tasks.filter((t) => t.id === only) : tasks;
  if (!names.length) throw new Error(`no tasks match --only ${only}`);

  const srv = await startServer(PORT);
  const bridge = new Bridge(bin);
  const info = (await bridge.call("load_model", { path: model, nCtx: 16384 })) as Json;
  console.log(`model loaded: ${JSON.stringify(info).slice(0, 160)}`);

  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC(async (cmd: string, args?: Json) => {
    if (cmd === "generate") {
      const ch = (args as Json)?.onEvent as { onmessage?: (ev: unknown) => void } | undefined;
      return bridge.call("generate", { request: (args as Json)?.request }, (ev) => ch?.onmessage?.(ev));
    }
    return bridge.call(cmd, args ?? {});
  });
  const { runAgentTurn } = await import("../../src/lib/agentLoop");

  const runsDir = path.join(DIR, "runs");
  mkdirSync(runsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  const outFile = path.join(runsDir, `web-${stamp}.jsonl`);
  console.log(`${names.length} task(s): ${names.map((t) => t.id).join(", ")}`);

  let resolvedCount = 0;
  for (const task of names) {
    const t0 = Date.now();
    resetState();
    // Browser tasks need no repo, but the loop wants a real workspace dir.
    const ws = mkdtempSync(path.join(tmpdir(), `chaty-webbench-`));
    try {
      await bridge.call("agent_set_workspace", { path: ws });
      const instruction = task.instruction.replaceAll("{BASE}", BASE);

      let steps = 0, turns = 0, error: string | undefined;
      const seenSteps = new Set<string>();
      const trDir = process.env.CHATY_BENCH_TRANSCRIPT;
      const trFile = trDir ? path.join(trDir, `${task.id}.jsonl`) : null;
      if (trDir) mkdirSync(trDir, { recursive: true });
      const trace = (ev: Json) => { if (trFile) appendFileSync(trFile, JSON.stringify(ev) + "\n"); };
      const signal = { cancelled: false };

      let prompt = instruction;
      const history: { role: string; content: string; images: string[] }[] = [];
      let finalText = "";
      for (turns = 1; turns <= 2; turns++) {
        let pausedAtSteps = false;
        await new Promise<void>((resolve) => {
          runAgentTurn(prompt, history as never, ws, (process.env.CHATY_BENCH_LANG === "zh" ? "zh" : "en"), {
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
            onAssistantText: (t) => { trace({ ev: "text", t }); },
            onStep: (s) => {
              if (!seenSteps.has(s.id)) { seenSteps.add(s.id); steps++; }
              if (s.status !== "running") trace({ ev: "step", call: s.call, status: s.status, result: s.result?.slice(0, 4000) });
            },
            onFinal: (text, _think, reason) => { trace({ ev: "final", reason: reason ?? "done", text }); finalText = text; pausedAtSteps = reason === "steps"; resolve(); },
            onError: (m) => { error = m; resolve(); },
            onAskUser: async (_q, options) => options[0] ?? "continue",
          });
        });
        if (error || !pausedAtSteps) break;
        history.push({ role: "user", content: prompt, images: [] });
        history.push({ role: "assistant", content: finalText, images: [] });
        prompt = process.env.CHATY_BENCH_LANG === "zh" ? "继续" : "Continue.";
      }

      const grader = GRADERS[task.id];
      const verdict = error
        ? { pass: false, why: `agent error: ${error}` }
        : grader
          ? grader(getState(), finalText)
          : { pass: false, why: "NO GRADER" };
      if (verdict.pass) resolvedCount++;
      const r = {
        task: task.id, resolved: verdict.pass, steps, turns,
        seconds: Math.round((Date.now() - t0) / 1000),
        ...(verdict.pass ? {} : { why: verdict.why }), ...(error ? { error } : {}),
      };
      appendFileSync(outFile, JSON.stringify(r) + "\n");
      console.log(`${verdict.pass ? "✓" : "✗"} ${task.id}  ${r.seconds}s  ${steps} steps${verdict.pass ? "" : `  — ${verdict.why}`}`);
    } finally {
      try { await bridge.call("browser_close", {}); } catch { /* already down */ }
      rmSync(ws, { recursive: true, force: true });
    }
  }

  console.log(`\n${resolvedCount}/${names.length} resolved — results: ${outFile}`);
  bridge.kill();
  srv.close();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
