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
import { appendFileSync, existsSync, mkdtempSync, readFileSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Bridge, type Json } from "../lib/bridge.mts";
import { officialSkills } from "../../src/lib/skillFiles.ts";
import { grade } from "./applib.mts";

const DIR = path.dirname(new URL(import.meta.url).pathname);
const OUT = process.env.CALREPRO_OUT ?? path.join(tmpdir(), "calrepro.jsonl");

async function main() {
  const argv = process.argv.slice(2);
  const tag = argv.includes("--tag") ? argv[argv.indexOf("--tag") + 1] : "cur";
  const prompt = argv.includes("--prompt")
    ? argv[argv.indexOf("--prompt") + 1]
    : "写一个mac 桌面端日历应用";
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
      prompt,
      [] as never,
      ws,
      "zh",
      {
        // App-default settings — the exact conditions of the audited session.
        // maxSteps 64 mirrors the raised default (32 starved rounds 7/8;
        // 48 cut round 16 one error from green).
        thinkMode: "normal", nCtx, maxSteps: 64, temperature: 0.3,
          skills: officialSkills(),
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
