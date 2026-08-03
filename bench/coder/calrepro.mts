/**
 * CalendarApp delivery repro — the owner's exact prompt ("写一个mac 桌面端日历应用")
 * through the REAL agent loop with app-default settings, graded the only way
 * that matters: does the delivered project COMPILE. Zero compile errors =
 * pass (logic bugs allowed); anything else fails the round.
 *
 * Grading is objective and external: xcodebuild on the delivered .xcodeproj
 * (falls back to whole-set `swiftc -typecheck` when no project file), the
 * model's claims are never trusted.
 *
 * Run:  npx tsx bench/coder/calrepro.mts [--tag t]
 * Env:  CHATY_HEADLESS_BIN, CHATY_BENCH_MODEL
 */
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-bench" };

import { execFileSync } from "node:child_process";
import { appendFileSync, existsSync, mkdtempSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Bridge, type Json } from "../lib/bridge.mts";

const DIR = path.dirname(new URL(import.meta.url).pathname);
const OUT = process.env.CALREPRO_OUT ?? path.join(tmpdir(), "calrepro.jsonl");

function walk(dir: string, depth: number, hit: (p: string) => void) {
  if (depth < 0) return;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.name.startsWith(".")) continue;
    const p = path.join(dir, e.name);
    hit(p);
    if (e.isDirectory()) walk(p, depth - 1, hit);
  }
}

function grade(ws: string): { compiles: boolean; errors: string[]; how: string } {
  let xcodeproj: string | undefined;
  let hasPackage = false;
  let hasSources = false;
  const swifts: string[] = [];
  walk(ws, 4, (p) => {
    if (p.endsWith(".xcodeproj") && statSync(p).isDirectory()) xcodeproj ??= p;
    if (path.basename(p) === "Package.swift") hasPackage = true;
    if (path.basename(p) === "Sources" && statSync(p).isDirectory()) hasSources = true;
    // Package manifests need the SwiftPM toolchain; bare swiftc reports a
    // phantom "no such module 'PackageDescription'" on fine projects.
    if (p.endsWith(".swift") && !/Tests\//.test(p) && !/\/Package(@swift-[^/]*)?\.swift$/.test(p))
      swifts.push(p);
  });
  const run = (bin: string, args: string[]): { code: number; out: string } => {
    try {
      const out = execFileSync(bin, args, { cwd: ws, timeout: 300_000, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
      return { code: 0, out };
    } catch (e) {
      const err = e as { status?: number; stdout?: string; stderr?: string };
      return { code: err.status ?? 1, out: `${err.stdout ?? ""}\n${err.stderr ?? ""}` };
    }
  };
  const errLines = (out: string) =>
    [...new Set(out.split("\n").filter((l) => / error: |^error: /.test(l)))].slice(0, 20);
  if (xcodeproj) {
    const r = run("xcodebuild", [
      "-project", xcodeproj, "-alltargets", "-configuration", "Debug",
      "build", "CODE_SIGNING_ALLOWED=NO",
    ]);
    return { compiles: r.code === 0 && r.out.includes("BUILD SUCCEEDED"), errors: errLines(r.out), how: "xcodebuild" };
  }
  if (hasPackage && hasSources) {
    const r = run("swift", ["build"]);
    return { compiles: r.code === 0, errors: errLines(r.out), how: "swift build" };
  }
  if (swifts.length) {
    const r = run("swiftc", ["-typecheck", ...swifts]);
    return { compiles: r.code === 0, errors: errLines(r.out), how: "swiftc -typecheck" };
  }
  // Node/TS delivery (the model may legitimately pick Electron/Tauri for a
  // "mac desktop app"): compile-check every tsconfig, then the bundler build.
  if (existsSync(path.join(ws, "package.json"))) {
    const tsconfigs = readdirSync(ws).filter((f) => /^tsconfig.*\.json$/.test(f));
    const tsErr = (out: string) => [...new Set(out.split("\n").filter((l) => /error TS\d+:/.test(l)))].slice(0, 20);
    if (!existsSync(path.join(ws, "node_modules"))) {
      const i = run("npm", ["install", "--no-audit", "--no-fund"]);
      if (i.code !== 0) return { compiles: false, errors: ["npm install failed", ...i.out.split("\n").slice(-5)], how: "npm install" };
    }
    for (const tc of tsconfigs) {
      const r = run("npx", ["tsc", "--noEmit", "-p", tc]);
      if (r.code !== 0) return { compiles: false, errors: tsErr(r.out), how: `tsc -p ${tc}` };
    }
    if (existsSync(path.join(ws, "vite.config.ts")) || existsSync(path.join(ws, "vite.config.js"))) {
      const r = run("npx", ["vite", "build"]);
      if (r.code !== 0) return { compiles: false, errors: errLines(r.out).concat(tsErr(r.out)).slice(0, 20), how: "vite build" };
    }
    return { compiles: true, errors: [], how: `tsc ×${tsconfigs.length}+vite` };
  }
  return { compiles: false, errors: ["no compilable sources delivered at all"], how: "none" };
}

async function main() {
  const argv = process.argv.slice(2);
  const tag = argv.includes("--tag") ? argv[argv.indexOf("--tag") + 1] : "cur";
  const bin = process.env.CHATY_HEADLESS_BIN ?? path.join(DIR, "../../src-tauri/target/release/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  if (!existsSync(bin)) throw new Error(`chaty-headless not found at ${bin}`);

  const bridge = new Bridge(bin);
  // No explicit nCtx — the app's contextLength=0 (auto) loads at the model's
  // trained cap (262k for Qwen3.6), so a fixed 16384 here triggers mid-turn
  // context compaction the real app never sees (round-3 artifact).
  const info = (await bridge.call("load_model", { path: model })) as Json;
  if (!info || info.loaded !== true) throw new Error(`model did not load: ${JSON.stringify(info)}`);
  const nCtx = Number(info.nCtx) || 16384;
  console.log(`[${tag}] model loaded, nCtx=${nCtx}`);

  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC((cmd: string, args?: Json) => bridge.ipc(cmd, args));
  const { runAgentTurn } = await import(path.join(DIR, "../../src/lib/agentLoop.ts"));

  const ws = mkdtempSync(path.join(tmpdir(), "chaty-calrepro-"));
  await bridge.call("agent_set_workspace", { path: ws });

  type Row = { i: number; name: string; args: string; result: string };
  const stepLog: Row[] = [];
  const byId = new Map<string, Row>();
  const injects: string[] = [];
  let steps = 0;
  let finalText = "";
  let error: string | undefined;
  const t0 = Date.now();

  await new Promise<void>((resolve) => {
    const watchdog = setTimeout(() => { error = "watchdog: 2700s"; resolve(); }, 2_700_000);
    runAgentTurn(
      "写一个mac 桌面端日历应用",
      [] as never,
      ws,
      "zh",
      {
        // App-default settings — the exact conditions of the audited session.
        thinkMode: "normal", nCtx, maxSteps: 32, temperature: 0.3,
        bashTimeout: 60, browserTextMode: true,
        signal: { cancelled: false } as never,
        approve: async () => true,
        approveDir: async () => false,
        approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {},
        onAssistantText: () => {},
        onStep: (st: { id: string; call?: { name: string; args?: Json }; status?: string; result?: string }) => {
          let row = byId.get(st.id);
          if (!row) {
            steps++;
            row = { i: steps, name: st.call?.name ?? "?", args: JSON.stringify(st.call?.args ?? {}).slice(0, 120), result: "" };
            byId.set(st.id, row);
            stepLog.push(row);
          }
          if (st.result) row.result = String(st.result).slice(0, 200);
        },
        onFinal: (t: string) => { clearTimeout(watchdog); finalText = t; resolve(); },
        onError: (m: string) => { clearTimeout(watchdog); error = m; resolve(); },
        onTrace: (ev: { kind: string; text?: string }) => { if (ev.kind === "inject") injects.push(ev.text ?? ""); },
      },
    );
  });

  const secs = Math.round((Date.now() - t0) / 1000);
  console.log(`\n[${tag}] turn done: ${steps} steps, ${secs}s, error=${error ?? "none"}`);
  for (const r of stepLog) console.log(`  ${String(r.i).padStart(2)} ${r.name.padEnd(18)} ${r.args}\n     → ${r.result.replaceAll("\n", " / ")}`);
  console.log(`\ninjects (${injects.length}):`);
  for (const i of injects) console.log(`  · ${i.slice(0, 160).replaceAll("\n", " ")}`);

  console.log(`\n[grade] compiling delivered project…`);
  const verdict = grade(ws);
  console.log(`[grade:${verdict.how}] compiles=${verdict.compiles}`);
  for (const e of verdict.errors) console.log(`  ✗ ${e}`);

  appendFileSync(OUT, JSON.stringify({ tag, ws, steps, secs, error, compiles: verdict.compiles, how: verdict.how, errors: verdict.errors, injects: injects.length, final: finalText.slice(0, 400), stepNames: stepLog.map((s) => s.name) }) + "\n");
  console.log(`\nworkspace kept at: ${ws}\nlog: ${OUT}`);
  bridge.kill();
  // Reap anything the turn left running against this workspace (dev servers,
  // an Electron window) — detached groups outlive the bridge teardown.
  try { execFileSync("pkill", ["-f", ws], { stdio: "ignore" }); } catch { /* none alive */ }
  process.exit(verdict.compiles ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(2); });
