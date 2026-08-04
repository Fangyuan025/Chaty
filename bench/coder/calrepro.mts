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

import { execFileSync, spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdtempSync, readFileSync, readdirSync, statSync } from "node:fs";
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

/** Owner bar v2: compile green is only step one — the deliverable is a
 *  packaged .app that LAUNCHES and stays alive for 4 seconds. */
async function grade(ws: string): Promise<{ compiles: boolean; packaged: boolean; runs: boolean; errors: string[]; how: string }> {
  const c = gradeCompile(ws);
  if (!c.compiles) return { ...c, packaged: false, runs: false };
  // Find delivered .app bundles — excluding node_modules (the stock
  // Electron.app inside the dependency is NOT the delivery) and .build.
  const bundles: string[] = [];
  walk(ws, 6, (p) => {
    if (
      p.endsWith(".app") &&
      !p.includes("node_modules") &&
      !p.includes("/.build/") &&
      statSync(p).isDirectory() &&
      existsSync(path.join(p, "Contents", "MacOS"))
    )
      bundles.push(p);
  });
  if (!bundles.length) {
    return { ...c, packaged: false, runs: false, errors: ["compile green but no packaged .app bundle delivered"], how: c.how + "+no-bundle" };
  }
  // Launch check: the bundle's main binary must survive 4 seconds.
  const macos = path.join(bundles[0], "Contents", "MacOS");
  const bin = readdirSync(macos).find((f) => { try { return (statSync(path.join(macos, f)).mode & 0o111) !== 0; } catch { return false; } });
  if (!bin) return { ...c, packaged: true, runs: false, errors: ["bundle has no executable in Contents/MacOS"], how: c.how + "+launch" };
  const child = spawn(path.join(macos, bin), [], { stdio: "ignore", detached: true });
  let dead = false;
  child.on("exit", () => { dead = true; });
  await new Promise((r) => setTimeout(r, 4000));
  const runs = !dead;
  try { if (child.pid) process.kill(-child.pid); } catch { try { child.kill(); } catch { /* already gone */ } }
  return {
    ...c,
    packaged: true,
    runs,
    errors: runs ? c.errors : [`launch check failed: ${path.basename(bundles[0])} exited within 4s`],
    how: c.how + "+launch",
  };
}

function gradeCompile(ws: string): { compiles: boolean; errors: string[]; how: string } {
  let xcodeproj: string | undefined;
  let packageDir: string | undefined;
  const swifts: string[] = [];
  walk(ws, 4, (p) => {
    if (p.endsWith(".xcodeproj") && statSync(p).isDirectory()) xcodeproj ??= p;
    // Track the manifest's own directory: `swift build` must run THERE.
    // Running it at the ws root walks UP the parent chain for a manifest —
    // round 17 "passed" by silently building a stale package that round 1
    // had left in the shared temp ROOT, two directories above the delivery.
    if (path.basename(p) === "Package.swift") packageDir ??= path.dirname(p);
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
  // Substance floor (round 13: a cleanup rm -rf left ONE typecheck-clean
  // file and the compile check waved the hollow tree through): an app
  // delivery needs an entry point and more than a fragment of source.
  const hollow = (why: string) => ({ compiles: false, errors: [why], how: "substance" });
  if (swifts.length) {
    const hasMain = swifts.some((p) => {
      try { return readFileSync(p, "utf8").includes("@main"); } catch { return false; }
    });
    if (!hasMain) return hollow("no @main entry point in delivered Swift sources");
    if (swifts.length < 3) return hollow(`only ${swifts.length} Swift source file(s) delivered — not an app`);
  }
  if (xcodeproj) {
    const r = run("xcodebuild", [
      "-project", xcodeproj, "-alltargets", "-configuration", "Debug",
      "build", "CODE_SIGNING_ALLOWED=NO",
    ]);
    return { compiles: r.code === 0 && r.out.includes("BUILD SUCCEEDED"), errors: errLines(r.out), how: "xcodebuild" };
  }
  if (packageDir && existsSync(path.join(packageDir, "Sources"))) {
    const r = run("swift", ["build", "--package-path", packageDir]);
    return { compiles: r.code === 0, errors: errLines(r.out), how: "swift build" };
  }
  if (swifts.length) {
    const r = run("swiftc", ["-typecheck", ...swifts]);
    return { compiles: r.code === 0, errors: errLines(r.out), how: "swiftc -typecheck" };
  }
  // Node/TS delivery (the model may legitimately pick Electron/Vue/Tauri for
  // a "mac desktop app"): find the project dir (round 18 shipped in a
  // subdirectory and the root-only check called a green delivery "nothing"),
  // then prefer the project's OWN build script — that is the honest "does it
  // compile" for whatever stack it chose; tsc/vite are the fallback.
  let pkgDir: string | undefined;
  walk(ws, 3, (p) => {
    if (path.basename(p) === "package.json" && !p.includes("node_modules")) pkgDir ??= path.dirname(p);
  });
  if (pkgDir) {
    const dir = pkgDir;
    const runIn = (bin: string, args: string[]) => {
      try {
        const out = execFileSync(bin, args, { cwd: dir, timeout: 300_000, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
        return { code: 0, out };
      } catch (e) {
        const err = e as { status?: number; stdout?: string; stderr?: string };
        return { code: err.status ?? 1, out: `${err.stdout ?? ""}\n${err.stderr ?? ""}` };
      }
    };
    const tsErr = (out: string) => [...new Set(out.split("\n").filter((l) => /error TS\d+:/.test(l)))].slice(0, 20);
    if (!existsSync(path.join(dir, "node_modules"))) {
      const i = runIn("npm", ["install", "--no-audit", "--no-fund"]);
      if (i.code !== 0) return { compiles: false, errors: ["npm install failed", ...i.out.split("\n").slice(-5)], how: "npm install" };
    }
    const scripts = (JSON.parse(readFileSync(path.join(dir, "package.json"), "utf8")).scripts ?? {}) as Record<string, string>;
    if (scripts.build) {
      const r = runIn("npm", ["run", "build"]);
      return { compiles: r.code === 0, errors: errLines(r.out).concat(tsErr(r.out)).slice(0, 20), how: "npm run build" };
    }
    const tsconfigs = readdirSync(dir).filter((f) => /^tsconfig.*\.json$/.test(f));
    for (const tc of tsconfigs) {
      const r = runIn("npx", ["tsc", "--noEmit", "-p", tc]);
      if (r.code !== 0) return { compiles: false, errors: tsErr(r.out), how: `tsc -p ${tc}` };
    }
    if (existsSync(path.join(dir, "vite.config.ts")) || existsSync(path.join(dir, "vite.config.js"))) {
      const r = runIn("npx", ["vite", "build"]);
      if (r.code !== 0) return { compiles: false, errors: errLines(r.out).concat(tsErr(r.out)).slice(0, 20), how: "vite build" };
    }
    return { compiles: tsconfigs.length > 0, errors: tsconfigs.length ? [] : ["no build script, no tsconfig — nothing verifiable"], how: `tsc ×${tsconfigs.length}` };
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
    // 90 min: a 48-step turn with thinking + real xcodebuild cycles can pass
    // 45 min while converging (round 11 was cut mid-repair at 2700s with one
    // error left) — the app has no wall clock, so neither should the bench.
    const watchdog = setTimeout(() => { error = "watchdog: 5400s"; resolve(); }, 5_400_000);
    runAgentTurn(
      "写一个mac 桌面端日历应用",
      [] as never,
      ws,
      "zh",
      {
        // App-default settings — the exact conditions of the audited session.
        // maxSteps 64 mirrors the raised default (32 starved rounds 7/8;
        // 48 cut round 16 one error from green).
        thinkMode: "normal", nCtx, maxSteps: 64, temperature: 0.3,
        bashTimeout: 60, browserTextMode: true,
        signal: { cancelled: false } as never,
        approve: async () => true,
        approveDir: async () => false,
        approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {},
        onAssistantText: () => {},
        // A hands-off user: the audited session got zero guidance, so the
        // bench answers stack questions decisively instead of orphaning the
        // turn (round 15 died in 2 steps re-asking its own question).
        onAskUser: async (_q: string, options: string[]) =>
          options[0] ?? "你来决定:选你认为最合适的方案,直接完整实现整个应用,不要再问我。",
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
  const verdict = await grade(ws);
  const pass = verdict.compiles && verdict.packaged && verdict.runs;
  console.log(`[grade:${verdict.how}] compiles=${verdict.compiles} packaged=${verdict.packaged} runs=${verdict.runs} PASS=${pass}`);
  for (const e of verdict.errors) console.log(`  ✗ ${e}`);

  appendFileSync(OUT, JSON.stringify({ tag, ws, steps, secs, error, compiles: verdict.compiles, packaged: verdict.packaged, runs: verdict.runs, how: verdict.how, errors: verdict.errors, injects: injects.length, final: finalText.slice(0, 400), stepNames: stepLog.map((s) => s.name) }) + "\n");
  console.log(`\nworkspace kept at: ${ws}\nlog: ${OUT}`);
  bridge.kill();
  // Reap anything the turn left running against this workspace (dev servers,
  // an Electron window) — detached groups outlive the bridge teardown.
  try { execFileSync("pkill", ["-f", ws], { stdio: "ignore" }); } catch { /* none alive */ }
  process.exit(pass ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(2); });
