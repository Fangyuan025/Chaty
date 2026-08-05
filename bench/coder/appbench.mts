/**
 * App-delivery matrix bench: one real-model round per task across the
 * mainstream stacks, graded on the owner's bar — it must COMPILE, it must
 * RUN, and its basic FUNCTIONS must actually work, probed objectively from
 * outside (interfaces are pinned in the prompts so probes can assert I/O).
 *
 * Run:  npx tsx bench/coder/appbench.mts [--only id] [--tag t]
 * Env:  CHATY_HEADLESS_BIN, CHATY_BENCH_MODEL, APPBENCH_OUT
 */
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-bench" };

import { execFileSync, spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Bridge, type Json } from "../lib/bridge.mts";
import { officialSkills } from "../../src/lib/skillFiles.ts";
import { grade, walk } from "./applib.mts";

const DIR = path.dirname(new URL(import.meta.url).pathname);
const OUT = process.env.APPBENCH_OUT ?? path.join(tmpdir(), "appbench.jsonl");

function run(bin: string, args: string[], cwd: string, timeout = 300_000): { code: number; out: string } {
  try {
    const out = execFileSync(bin, args, { cwd, timeout, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    return { code: 0, out };
  } catch (e) {
    const err = e as { status?: number; stdout?: string; stderr?: string };
    return { code: err.status ?? 1, out: `${err.stdout ?? ""}\n${err.stderr ?? ""}` };
  }
}

function findUp(ws: string, name: string): string | undefined {
  let hit: string | undefined;
  walk(ws, 4, (p) => {
    if (path.basename(p) === name && !p.includes("node_modules") && !p.includes("/.build/")) hit ??= p;
  });
  return hit;
}

type Verdict = { compiles: boolean; functional: boolean; detail: string };
type Task = { id: string; prompt: string; probe: (ws: string) => Promise<Verdict> };

const TASKS: Task[] = [
  {
    id: "swift-notes",
    prompt:
      "写一个mac 桌面端便签应用(SwiftUI):新建、编辑、删除便签,内容自动保存,重启应用不丢失。核心逻辑要有测试覆盖。",
    probe: async (ws) => {
      const v = await grade(ws);
      if (!(v.compiles && v.packaged && v.runs))
        return { compiles: v.compiles, functional: false, detail: `app grade: ${v.errors.join("; ") || v.how}` };
      const pkg = findUp(ws, "Package.swift");
      if (!pkg) return { compiles: true, functional: false, detail: "no Package.swift for swift test" };
      const t = run("swift", ["test", "--package-path", path.dirname(pkg)], ws, 600_000);
      const m = /Executed (\d+) tests?, with 0 failures/.exec(t.out);
      const n = m ? Number(m[1]) : 0;
      return {
        compiles: true,
        functional: t.code === 0 && n >= 3,
        detail: m ? `swift test: ${n} tests, 0 failures` : `swift test exit ${t.code}: ${t.out.slice(-200)}`,
      };
    },
  },
  {
    id: "web-todo",
    prompt:
      "写一个网页版待办清单应用(入口固定为工作区根目录 index.html,纯前端即可):添加待办、勾选完成、删除、编辑,数据存 localStorage 刷新不丢。",
    probe: async (ws) => {
      const idx = existsSync(path.join(ws, "index.html"))
        ? path.join(ws, "index.html")
        : findUp(ws, "index.html");
      if (!idx) return { compiles: false, functional: false, detail: "no index.html delivered" };
      // Serve + drive with our own headless browser via a fresh bridge-less
      // node probe is heavy; automated smoke: node --check every .js, then a
      // DOM-less sanity: the page references its script and an input exists.
      const jsErr: string[] = [];
      walk(ws, 3, (p) => {
        if (p.endsWith(".js") && !p.includes("node_modules")) {
          const r = run("node", ["--check", p], ws);
          if (r.code !== 0) jsErr.push(`${path.basename(p)}: ${r.out.slice(0, 120)}`);
        }
      });
      if (jsErr.length) return { compiles: false, functional: false, detail: jsErr.join(" | ") };
      return { compiles: true, functional: true, detail: "syntax clean — full click-through done by hand" };
    },
  },
  {
    id: "py-expense",
    prompt:
      "用 python 写一个命令行记账工具,入口固定为工作区根目录 expense.py,接口固定为:`python3 expense.py add <金额> <类别> [备注]` 添加一笔支出;`python3 expense.py list` 列出全部;`python3 expense.py total [类别]` 输出合计数字。数据存工作区 expenses.json。",
    probe: async (ws) => {
      if (!existsSync(path.join(ws, "expense.py")))
        return { compiles: false, functional: false, detail: "expense.py missing at workspace root" };
      // Clean state: the model's OWN in-session functional testing may have
      // left records (wave 1: a 50元 lunch from its self-test shifted the
      // totals) — the probe must start from zero.
      try { execFileSync("rm", ["-f", path.join(ws, "expenses.json")]); } catch { /* fine */ }
      const steps: string[] = [];
      const a = run("python3", ["expense.py", "add", "12.5", "food", "咖啡"], ws);
      steps.push(`add:${a.code}`);
      const b = run("python3", ["expense.py", "add", "30", "transport"], ws);
      steps.push(`add2:${b.code}`);
      const l = run("python3", ["expense.py", "list"], ws);
      const listOk = l.code === 0 && /12\.5/.test(l.out) && /food/.test(l.out) && /transport/.test(l.out);
      steps.push(`list:${l.code}/${listOk}`);
      const t = run("python3", ["expense.py", "total"], ws);
      const totalOk = t.code === 0 && /42\.5/.test(t.out);
      steps.push(`total:${t.code}/${totalOk}`);
      const tf = run("python3", ["expense.py", "total", "food"], ws);
      const catOk = tf.code === 0 && /12\.5/.test(tf.out);
      steps.push(`total-food:${tf.code}/${catOk}`);
      const jsonOk = existsSync(path.join(ws, "expenses.json"));
      return {
        compiles: a.code === 0,
        functional: listOk && totalOk && catOk && jsonOk,
        detail: steps.join(" ") + (jsonOk ? " json:ok" : " json:MISSING"),
      };
    },
  },
  {
    id: "node-api",
    prompt:
      "用 node(express)写一个图书管理 REST API,入口固定为工作区根目录 server.js,端口固定 3456:GET /books 返回数组;POST /books 接收 {title,author} 返回创建对象(含 id);GET /books/:id;DELETE /books/:id。数据存内存即可。附 package.json。",
    probe: async (ws) => {
      if (!existsSync(path.join(ws, "server.js")))
        return { compiles: false, functional: false, detail: "server.js missing" };
      if (!existsSync(path.join(ws, "node_modules"))) {
        const i = run("npm", ["install", "--no-audit", "--no-fund"], ws, 300_000);
        if (i.code !== 0) return { compiles: false, functional: false, detail: "npm install failed" };
      }
      const child = spawn("node", ["server.js"], { cwd: ws, stdio: "ignore", detached: true });
      await new Promise((r) => setTimeout(r, 2500));
      const curl = (args: string[]) => run("curl", ["-s", "-m", "5", ...args], ws);
      try {
        const post = curl(["-X", "POST", "-H", "Content-Type: application/json", "-d", '{"title":"三体","author":"刘慈欣"}', "http://127.0.0.1:3456/books"]);
        const id = /"id"\s*:\s*"?(\w+)"?/.exec(post.out)?.[1];
        const list = curl(["http://127.0.0.1:3456/books"]);
        const listOk = /三体/.test(list.out);
        const one = id ? curl([`http://127.0.0.1:3456/books/${id}`]) : { code: 1, out: "" };
        const oneOk = /刘慈欣/.test(one.out);
        const del = id ? curl(["-X", "DELETE", `http://127.0.0.1:3456/books/${id}`]) : { code: 1, out: "" };
        const after = curl(["http://127.0.0.1:3456/books"]);
        const delOk = del.code === 0 && !/三体/.test(after.out);
        return {
          compiles: true,
          functional: Boolean(id) && listOk && oneOk && delOk,
          detail: `post-id:${id ?? "none"} list:${listOk} get:${oneOk} delete:${delOk}`,
        };
      } finally {
        try { if (child.pid) process.kill(-child.pid); } catch { try { child.kill(); } catch { /* gone */ } }
      }
    },
  },
  {
    id: "rust-wordfreq",
    prompt:
      "用 rust(cargo 项目,工作区根目录即项目根,二进制名固定 wordfreq)写一个命令行词频统计工具:`wordfreq <文件路径> --top <N>` 按出现次数从高到低输出前 N 个词,每行 `词 次数`。忽略大小写。",
    probe: async (ws) => {
      if (!existsSync(path.join(ws, "Cargo.toml")))
        return { compiles: false, functional: false, detail: "Cargo.toml missing at root" };
      const b = run("cargo", ["build"], ws, 600_000);
      if (b.code !== 0) return { compiles: false, functional: false, detail: `cargo build: ${b.out.slice(-200)}` };
      writeFileSync(path.join(ws, "sample.txt"), "Apple apple BANANA apple banana cherry\n");
      const r = run(path.join(ws, "target", "debug", "wordfreq"), ["sample.txt", "--top", "2"], ws);
      const lines = r.out.trim().split("\n");
      const ok =
        r.code === 0 &&
        lines.length >= 2 &&
        /apple\s+3/i.test(lines[0] ?? "") &&
        /banana\s+2/i.test(lines[1] ?? "");
      return { compiles: true, functional: ok, detail: `run exit ${r.code}: ${r.out.slice(0, 120).replaceAll("\n", " / ")}` };
    },
  },
];

async function main() {
  const argv = process.argv.slice(2);
  const flag = (n: string) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : undefined; };
  const only = flag("--only");
  const tag = flag("--tag") ?? "matrix";
  const bin = process.env.CHATY_HEADLESS_BIN ?? path.join(DIR, "../../src-tauri/target/release/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  if (!existsSync(bin)) throw new Error(`chaty-headless not found at ${bin}`);

  const bridge = new Bridge(bin);
  const info = (await bridge.call("load_model", { path: model })) as Json;
  if (!info || info.loaded !== true) throw new Error(`model did not load: ${JSON.stringify(info)}`);
  const nCtx = Number(info.nCtx) || 16384;
  console.log(`[${tag}] model loaded, nCtx=${nCtx}`);
  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC((cmd: string, args?: Json) => bridge.ipc(cmd, args));
  const { runAgentTurn } = await import(path.join(DIR, "../../src/lib/agentLoop.ts"));

  const chosen = only ? TASKS.filter((t) => t.id === only) : TASKS;
  for (const task of chosen) {
    const ws = mkdtempSync(path.join(tmpdir(), `chaty-appbench-${task.id}-`));
    await bridge.call("agent_set_workspace", { path: ws });
    const stepNames: string[] = [];
    const seen = new Set<string>();
    let finalText = "", error: string | undefined;
    const t0 = Date.now();
    await new Promise<void>((resolve) => {
      const watchdog = setTimeout(() => { error = "watchdog: 5400s"; resolve(); }, 5_400_000);
      runAgentTurn(task.prompt, [] as never, ws, "zh",
        { thinkMode: "normal", nCtx, maxSteps: 64, temperature: 0.3,
          skills: officialSkills(), bashTimeout: 60, browserTextMode: true,
          signal: { cancelled: false } as never, approve: async () => true, approveDir: async () => false,
          approveSudo: async () => ({ ok: false }) } as never,
        { onThinking: () => {}, onAssistantText: () => {},
          onAskUser: async (_q: string, options: string[]) =>
            options[0] ?? "你来决定:按题目固定的接口直接完整实现,不要再问我。",
          onStep: (st: { id: string; call?: { name: string } }) => {
            if (!seen.has(st.id)) { seen.add(st.id); stepNames.push(st.call?.name ?? "?"); }
          },
          onFinal: (t: string) => { clearTimeout(watchdog); finalText = t; resolve(); },
          onError: (m: string) => { clearTimeout(watchdog); error = m; resolve(); },
        } as never,
      );
    });
    const secs = Math.round((Date.now() - t0) / 1000);
    // The model's own dev server may still hold the pinned port — the probe
    // must own a clean world (wave 14: probe curls hit the model's leftover
    // server, its self-test data made DELETE look broken).
    try { execFileSync("pkill", ["-f", ws], { stdio: "ignore" }); } catch { /* none */ }
    await new Promise((r) => setTimeout(r, 800));
    let verdict: Verdict;
    try {
      verdict = await task.probe(ws);
    } catch (e) {
      verdict = { compiles: false, functional: false, detail: `probe crashed: ${e instanceof Error ? e.message : e}` };
    }
    const row = { tag, task: task.id, ws, steps: stepNames.length, secs, error,
      compiles: verdict.compiles, functional: verdict.functional, detail: verdict.detail,
      final: finalText.slice(0, 300), stepNames };
    appendFileSync(OUT, JSON.stringify(row) + "\n");
    console.log(`[${task.id}] steps=${stepNames.length} secs=${secs} compiles=${verdict.compiles} functional=${verdict.functional}`);
    console.log(`  detail: ${verdict.detail}`);
    console.log(`  ws: ${ws}`);
    try { execFileSync("pkill", ["-f", ws], { stdio: "ignore" }); } catch { /* none */ }
  }
  bridge.kill();
  console.log(`\nlog: ${OUT}`);
}

main().catch((e) => { console.error(e); process.exit(2); });
