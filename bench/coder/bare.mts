/**
 * Ablation baseline — a deliberately minimal bash-only agent on the SAME
 * model, inference engine, sampling params and task set as runner.mts.
 * The score delta (runner − bare) isolates the value of Chaty Coder's tool
 * layer + loop engineering, which is the number the product claims.
 *
 * Run:  npx tsx bench/coder/bare.mts [--tasks …] [--only name]
 * Env:  CHATY_HEADLESS_BIN, CHATY_BENCH_MODEL   (same as runner.mts)
 */
import { spawn, execFileSync, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import { mkdtempSync, cpSync, readFileSync, appendFileSync, mkdirSync, readdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

type Json = Record<string, unknown>;

const SYS = `You are a coding agent working in a repository checkout. You have ONE tool:
- bash: run a shell command in the workspace. args: {"command": string}

Rules:
- To call the tool, output exactly one line:
  <tool_call>{"name":"bash","arguments":{"command":"..."}}</tool_call>
  and stop. You'll get the output back, then continue.
- Commands run with the workspace as the current directory.
- Write files with heredocs (cat > file <<'EOF' ... EOF).
- When the task is fully done, reply with a short summary and NO tool call.`;

class Bridge {
  private proc: ChildProcess;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: unknown) => void; onEvent?: (ev: Json) => void }>();
  constructor(bin: string) {
    this.proc = spawn(bin, [], { stdio: ["pipe", "pipe", "inherit"] });
    createInterface({ input: this.proc.stdout! }).on("line", (line) => {
      let m: Json; try { m = JSON.parse(line); } catch { return; }
      const p = this.pending.get(m.id as number);
      if (!p) return;
      if (m.type === "event") { p.onEvent?.(m.event as Json); return; }
      this.pending.delete(m.id as number);
      m.type === "error" ? p.reject(m.message) : p.resolve(m.result);
    });
  }
  call(cmd: string, args: Json = {}, onEvent?: (ev: Json) => void): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
      this.proc.stdin!.write(JSON.stringify({ id, cmd, args }) + "\n");
    });
  }
  kill() { this.proc.kill(); }
}

const TOOL_RE = /<tool_call>\s*(\{[\s\S]*?\})\s*(?:<\/tool_call>|$)/;
const THINK_RE = /<think>[\s\S]*?(?:<\/think>|$)/g;

async function main() {
  const argv = process.argv.slice(2);
  const flag = (n: string) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : undefined; };
  const here = path.dirname(new URL(import.meta.url).pathname);
  const tasksDir = path.resolve(flag("--tasks") ?? path.join(here, "tasks"));
  const only = flag("--only");
  const bin = process.env.CHATY_HEADLESS_BIN ?? path.join(here, "../../src-tauri/target/release/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");

  const bridge = new Bridge(bin);
  await bridge.call("load_model", { path: model, nCtx: 16384 });
  console.log("model loaded (bare agent)");

  mkdirSync(path.join(here, "runs"), { recursive: true });
  const outFile = path.join(here, "runs", `bare-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19)}.jsonl`);
  const names = readdirSync(tasksDir).filter((n) => (!only || n === only) && existsSync(path.join(tasksDir, n, "task.md")));
  console.log(`${names.length} task(s)`);

  let resolvedCount = 0;
  for (const name of names) {
    const t0 = Date.now();
    const taskDir = path.join(tasksDir, name);
    const ws = mkdtempSync(path.join(tmpdir(), `chaty-bare-${name}-`));
    cpSync(path.join(taskDir, "workspace"), ws, { recursive: true });
    const cfgPath = path.join(taskDir, "grade_config.json");
    if (existsSync(cfgPath)) {
      const envBin = (JSON.parse(readFileSync(cfgPath, "utf8")) as { env_bin: string }).env_bin;
      try { execFileSync("uv", ["pip", "install", "-q", "-e", ws, "--python", path.join(envBin, "python")], { stdio: "pipe" }); }
      catch { /* best-effort; grade.py PYTHONPATH is the backstop */ }
    }
    await bridge.call("agent_set_workspace", { path: ws });

    const messages: Json[] = [
      { role: "system", content: SYS, images: [] },
      { role: "user", content: readFileSync(path.join(taskDir, "task.md"), "utf8"), images: [] },
    ];
    let steps = 0, error: string | undefined, lastCmd = "", repeats = 0;
    for (let step = 0; step < 40; step++) {
      let raw = "";
      try {
        await bridge.call("generate", {
          request: { messages, params: { temperature: 0.2, topP: 0.9, repeatPenalty: 1.05, maxTokens: 3072, think: false, stop: ["</tool_call>"] } },
        }, (ev) => { if (ev.type === "token") raw += ev.text; });
      } catch (e) { error = String(e); break; }
      const body = raw.replace(THINK_RE, "");
      const m = TOOL_RE.exec(body);
      if (!m) break; // final answer
      let cmd = "";
      try { cmd = String(JSON.parse(m[1]).arguments?.command ?? ""); } catch { /* malformed */ }
      steps++;
      let result: string;
      if (!cmd) result = "ERROR: malformed tool call";
      else if (cmd === lastCmd && ++repeats >= 2) break;
      else {
        if (cmd !== lastCmd) { lastCmd = cmd; repeats = 0; }
        try {
          const r = (await bridge.call("agent_bash", { command: cmd, timeoutSecs: 120 })) as Json;
          result = String(r.output ?? JSON.stringify(r));
        } catch (e) { result = `ERROR: ${String(e)}`; }
        if (result.length > 8000) result = result.slice(0, 2400) + "\n… (truncated) …\n" + result.slice(-5600);
      }
      messages.push({ role: "assistant", content: body.includes("</tool_call>") ? body : body + "</tool_call>", images: [] });
      messages.push({ role: "user", content: `<tool_result>\n${result}\n</tool_result>`, images: [] });
    }

    let resolved = false;
    if (!error) {
      try { execFileSync("bash", [path.join(taskDir, "grade.sh")], { cwd: ws, timeout: 600_000, stdio: "pipe" }); resolved = true; }
      catch { resolved = false; }
    }
    if (resolved) resolvedCount++;
    appendFileSync(outFile, JSON.stringify({ task: name, resolved, steps, seconds: Math.round((Date.now() - t0) / 1000), error }) + "\n");
    console.log(`${resolved ? "✓" : "✗"} ${name}  ${steps} steps`);
  }
  console.log(`\nbare agent: ${resolvedCount}/${names.length} resolved — ${outFile}`);
  bridge.kill();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
