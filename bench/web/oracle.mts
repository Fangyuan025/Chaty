/**
 * ChatyWeb-Bench oracle — proves every task is solvable and every grader is
 * sound, with NO model involved (the web-bench analogue of SWE-bench's
 * gold-patch validation).
 *
 * For each task it resets the fixture world, replays the task's known-good
 * action sequence through the REAL tool chain (chaty-headless stdio →
 * agent.rs → browser.rs → headless Chrome), then runs the task's grader
 * against the fixture server's state (+ the oracle's canned final answer for
 * answer tasks). A task that can't pass its own oracle has a broken fixture,
 * oracle, or grader — never enters a scored run.
 *
 *   npx tsx oracle.mts [--only task-id]
 */
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import { startServer, resetState, getState, DEFAULT_PORT } from "./server.mts";
import { GRADERS } from "./graders.mts";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(DIR, "../..");
const PORT = Number(process.env.CHATY_WEBBENCH_PORT) || DEFAULT_PORT;
const BASE = `http://127.0.0.1:${PORT}`;

type Json = Record<string, unknown>;
type Task = {
  id: string;
  site: string;
  type: "state" | "answer";
  instruction: string;
  oracle: { actions: { cmd: string; args: Json }[]; finalText?: string };
};

function headlessBin(): string {
  const env = process.env.CHATY_HEADLESS_BIN;
  if (env) return env;
  for (const p of ["debug", "release"]) {
    const bin = path.join(REPO, "src-tauri", "target", p, "chaty-headless");
    if (existsSync(bin)) return bin;
  }
  throw new Error("chaty-headless not built — run: cargo build --bin chaty-headless");
}

class Bridge {
  private proc: ChildProcess;
  private nextId = 1;
  private pending = new Map<number, (v: { result?: unknown; error?: string }) => void>();
  constructor(bin: string) {
    this.proc = spawn(bin, [], {
      stdio: ["pipe", "pipe", "inherit"],
      env: { ...process.env, CHATY_BROWSER_HEADLESS: "1" },
    });
    let buf = "";
    this.proc.stdout!.on("data", (d: Buffer) => {
      buf += d.toString();
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        try {
          const m = JSON.parse(line);
          const cb = this.pending.get(m.id);
          if (cb && (m.type === "result" || m.error !== undefined)) {
            this.pending.delete(m.id);
            cb(m);
          }
        } catch {
          /* progress noise */
        }
      }
    });
  }
  call(cmd: string, args: Json = {}, timeoutMs = 60_000): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${cmd} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      this.pending.set(id, (m) => {
        clearTimeout(t);
        m.error !== undefined ? reject(new Error(String(m.error))) : resolve(m.result);
      });
      this.proc.stdin!.write(JSON.stringify({ id, cmd, args }) + "\n");
    });
  }
  kill() {
    this.proc.kill();
  }
}

function withBase(v: unknown): unknown {
  if (typeof v === "string") return v.replaceAll("{BASE}", BASE);
  if (Array.isArray(v)) return v.map(withBase);
  if (v && typeof v === "object")
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, withBase(x)]));
  return v;
}

async function main() {
  const only = process.argv.includes("--only")
    ? process.argv[process.argv.indexOf("--only") + 1]
    : undefined;
  const tasks: Task[] = JSON.parse(readFileSync(path.join(DIR, "tasks.json"), "utf8"));
  const run = only ? tasks.filter((t) => t.id === only) : tasks;
  if (!run.length) throw new Error(`no tasks match --only ${only}`);

  const srv = await startServer(PORT);
  const bridge = new Bridge(headlessBin());
  let passed = 0;
  const failures: string[] = [];

  for (const task of run) {
    resetState();
    let err: string | undefined;
    for (const [i, a] of task.oracle.actions.entries()) {
      try {
        await bridge.call(a.cmd, withBase(a.args) as Json);
      } catch (e) {
        err = `action ${i + 1} (${a.cmd}) failed: ${e instanceof Error ? e.message : e}`;
        break;
      }
    }
    // Fresh page state settles through the API before grading.
    await new Promise((r) => setTimeout(r, 250));
    const grader = GRADERS[task.id];
    const verdict = err
      ? { pass: false, why: err }
      : grader
        ? grader(getState(), task.oracle.finalText ?? "")
        : { pass: false, why: "NO GRADER for this task id" };
    console.log(`${verdict.pass ? "✓" : "✗"} ${task.id}${verdict.pass ? "" : `  — ${verdict.why}`}`);
    if (verdict.pass) passed++;
    else failures.push(task.id);
    try {
      await bridge.call("browser_close", {});
    } catch {
      /* browser may already be down */
    }
  }

  bridge.kill();
  srv.close();
  console.log(`\noracle: ${passed}/${run.length} tasks validated`);
  if (failures.length) {
    console.log(`failing: ${failures.join(", ")}`);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
