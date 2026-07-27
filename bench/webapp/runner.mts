/**
 * ChatyWebapp-Bench: do the webapp-flow app-layer fixes change what the agent
 * actually ships? Four scenarios = the four audited failure modes (foreground
 * dev server stall, console blindness, abandoned todos, unverified delivery).
 *
 * A/B by CODE SIDE, not by flag: point --repo (agentLoop source) and
 * CHATY_HEADLESS_BIN (matching binary) at either the current tree or the
 * baseline worktree — everything else (model, scenarios, grading) identical.
 *
 *   CHATY_BENCH_MODEL=…  npx tsx bench/webapp/runner.mts \
 *       [--repo /path/to/tree] [--only id] [--tag oldside]
 *
 * Grading is objective only: file contents, an end-of-run HTTP probe against
 * the dev server, and step-log facts (was there a browser action after the
 * last edit). The model's own claims are never trusted.
 */
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
const lsStore = new Map<string, string>();
g.localStorage = {
  getItem: (k: string) => lsStore.get(k) ?? null,
  setItem: (k: string, v: string) => void lsStore.set(k, v),
  removeItem: (k: string) => void lsStore.delete(k),
};

import { Bridge, type Json } from "../lib/bridge.mts";
import { SCENARIOS, type WebappScenario } from "./scenarios.mts";

const EDIT_TOOLS = new Set(["write_file", "edit_file", "edit_lines", "multi_edit"]);

async function serverAlive(port: number): Promise<boolean> {
  if (!port) return false;
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 3000);
    const res = await fetch(`http://127.0.0.1:${port}/`, { signal: ctl.signal });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const flag = (n: string) => {
    const i = argv.indexOf(n);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const only = flag("--only");
  const repo = path.resolve(flag("--repo") ?? path.join(DIR, "../.."));
  const tag = flag("--tag") ?? path.basename(repo);
  const bin =
    process.env.CHATY_HEADLESS_BIN ?? path.join(repo, "src-tauri/target/release/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  if (!existsSync(bin)) throw new Error(`chaty-headless not found at ${bin}`);

  const chosen = only ? SCENARIOS.filter((s) => s.id === only) : SCENARIOS;

  const bridge = new Bridge(bin);
  const info = (await bridge.call("load_model", { path: model, nCtx: 16384 })) as Json;
  if (!info || typeof info !== "object") {
    throw new Error(`load_model returned ${JSON.stringify(info)} — stale sidecar / OOM? pkill chaty-mlx and retry`);
  }
  console.log(`[${tag}] model loaded: ${JSON.stringify(info).slice(0, 100)}`);
  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC((cmd: string, args?: Json) => bridge.ipc(cmd, args));

  const { runAgentTurn } = await import(path.join(repo, "src/lib/agentLoop.ts"));

  const runsDir = path.join(DIR, "runs");
  mkdirSync(runsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  const outFile = path.join(runsDir, `webapp-${tag}-${stamp}.jsonl`);

  let portSeq = 21000 + Math.floor(Math.random() * 3000);
  for (const s of chosen as WebappScenario[]) {
    // Fresh port per RUN: dev servers spawned by a previous side survive that
    // side's process teardown (own process groups), so fixed ports would let
    // a stale server answer this side's liveness probe.
    const port = s.port === -1 ? portSeq++ : s.port;
    const sub = (t: string) => t.replaceAll("{PORT}", String(port));
    const instruction = sub(s.instruction);
    const expectAnswer = s.expectAnswer ? sub(s.expectAnswer) : undefined;
    const ws = mkdtempSync(path.join(tmpdir(), `chaty-webapp-${s.id}-`));
    for (const [rel, body] of Object.entries(s.files)) {
      const p = path.join(ws, rel);
      mkdirSync(path.dirname(p), { recursive: true });
      writeFileSync(p, sub(body));
    }
    await bridge.call("agent_set_workspace", { path: ws });

    const stepLog: { i: number; name: string; head: string }[] = [];
    let steps = 0;
    let finalText = "";
    let error: string | undefined;
    let lastPlan: { content: string; status: string }[] = [];
    const seen = new Set<string>();
    const t0 = Date.now();

    try {
      await new Promise<void>((resolve) => {
        const watchdog = setTimeout(() => {
          error = "watchdog: 600s";
          resolve();
        }, 600_000);
        runAgentTurn(
          instruction,
          [] as never,
          ws,
          "en",
          {
            thinkMode: "off",
            nCtx: 16384,
            maxSteps: 40,
            temperature: 0.2,
            bashTimeout: 120,
            browserTextMode: true,
            signal: { cancelled: false } as never,
            approve: async () => true,
            approveDir: async () => false,
            approveSudo: async () => ({ ok: false }),
          } as never,
          {
            onThinking: () => {},
            onAssistantText: () => {},
            onPlan: (todos: { content: string; status: string }[]) => {
              lastPlan = todos;
            },
            onStep: (st: { id: string; call?: { name: string; args?: Json }; status?: string; result?: string }) => {
              if (!seen.has(st.id)) {
                seen.add(st.id);
                steps++;
                stepLog.push({
                  i: steps,
                  name: st.call?.name ?? "?",
                  head: JSON.stringify(st.call?.args ?? {}).slice(0, 90),
                });
              }
            },
            onFinal: (text: string) => {
              clearTimeout(watchdog);
              finalText = text;
              resolve();
            },
            onError: (m: string) => {
              clearTimeout(watchdog);
              error = m;
              resolve();
            },
            onAskUser: async (_q: string, options: string[]) => options[0] ?? "continue",
          } as never,
        );
      });
    } catch (e) {
      error = String(e);
    }
    const seconds = Math.round((Date.now() - t0) / 1000);

    // ── Objective grading, before the next set_workspace kills the server ──
    const alive = await serverAlive(port);
    const readWs = (rel: string): string => {
      try {
        return readFileSync(path.join(ws, rel), "utf8");
      } catch {
        return "";
      }
    };
    const fileChecks = [
      ...(s.expectFiles ?? []).map(([rel, needle]) => ({
        rel,
        needle,
        ok: readWs(rel).includes(needle),
      })),
      ...(s.expectFilesAbsent ?? []).map(([rel, needle]) => ({
        rel,
        needle: `!${needle}`,
        ok: !readWs(rel).includes(needle),
      })),
      ...(s.expectFilesAny ?? []).map(([rel, needles]) => {
        const joined = rel.split("|").map(readWs).join("\n");
        return { rel, needle: needles.join("|"), ok: needles.some((n) => joined.includes(n)) };
      }),
    ];
    const lastEditIdx = stepLog.reduce((a, st) => (EDIT_TOOLS.has(st.name) ? st.i : a), -1);
    const lastBrowserIdx = stepLog.reduce((a, st) => (st.name.startsWith("browser_") ? st.i : a), -1);
    const browserAfterEdit = lastEditIdx >= 0 && lastBrowserIdx > lastEditIdx;

    const clicked = stepLog.some((st) => st.name === "browser_click");
    let ok = !error;
    if (ok && expectAnswer) ok = finalText.includes(expectAnswer);
    if (ok && fileChecks.length) ok = fileChecks.every((c) => c.ok);
    if (ok && s.expectServerAlive) ok = alive;
    if (ok && s.expectBrowserAfterEdit) ok = browserAfterEdit;
    if (ok && s.expectClicked) ok = clicked;

    const row = {
      side: tag,
      scenario: s.id,
      resolved: ok,
      steps,
      seconds,
      serverAlive: alive,
      browserAfterEdit,
      lastEditIdx,
      lastBrowserIdx,
      planCount: lastPlan.length,
      planDone: lastPlan.filter((t) => t.status === "done").length,
      fileChecks,
      finalHead: finalText.slice(0, 160),
      error,
      stepNames: stepLog.map((st) => st.name),
    };
    appendFileSync(outFile, JSON.stringify(row) + "\n");
    // Tear down THIS scenario's bg jobs + browser inside the backend before
    // the workspace dir vanishes (set_workspace's changed-path branch).
    const scratchWs = mkdtempSync(path.join(tmpdir(), "chaty-webapp-gap-"));
    await bridge.call("agent_set_workspace", { path: scratchWs });
    console.log(
      `${ok ? "✓" : "✗"} [${tag}] ${s.id}  ${seconds}s  ${steps} steps  server=${alive ? "up" : "—"}  walkthru=${browserAfterEdit ? "yes" : "no"}${error ? `  — ${String(error).slice(0, 60)}` : ""}`,
    );
    rmSync(ws, { recursive: true, force: true });
  }

  // Point the workspace at scratch so set_workspace teardown kills any
  // leftover dev server / browser from the last scenario.
  const scratch = mkdtempSync(path.join(tmpdir(), "chaty-webapp-end-"));
  await bridge.call("agent_set_workspace", { path: scratch });
  bridge.kill();
  console.log(`rows → ${outFile}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
