/**
 * ChatyMemory-Bench: does a fact left in project memory help the next session?
 *
 * The clean isolation of M4's promise — the ONLY variable is whether the
 * relevant fact is already in .chaty/memory/. Two arms per scenario:
 *
 *   control — empty memory: the agent must find the answer in the code.
 *   seeded  — the fact pre-written to memory: the agent can read the index,
 *             read the one fact, and answer.
 *
 * Graded on the final answer (server-of-truth here is the seeded fact vs the
 * buried source). We measure BOTH resolve and steps — the interesting signal,
 * as with skills, is usually effort, not the pass/fail flip.
 *
 *   CHATY_BENCH_MODEL=… npx tsx bench/memory/runner.mts [--only id] [--arm seeded|control]
 */
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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

import { Bridge, norm, type Json } from "../lib/bridge.mts";

interface Scenario {
  id: string;
  instruction: string;
  files: Record<string, string>;
  seedFact: { title: string; fact: string };
  grade: { kind: "answer" | "answer-all"; expect: string | string[] };
}

async function main() {
  const argv = process.argv.slice(2);
  const flag = (n: string) => {
    const i = argv.indexOf(n);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const only = flag("--only");
  const armFilter = flag("--arm");
  const bin = process.env.CHATY_HEADLESS_BIN ?? path.join(DIR, "../../src-tauri/target/release/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  if (!existsSync(bin)) throw new Error(`chaty-headless not found at ${bin}`);

  const scenarios: Scenario[] = JSON.parse(
    (await import("node:fs")).readFileSync(path.join(DIR, "scenarios.json"), "utf8"),
  );
  const chosen = only ? scenarios.filter((s) => s.id === only) : scenarios;

  const bridge = new Bridge(bin);
  const info = (await bridge.call("load_model", { path: model, nCtx: 16384 })) as Json;
  console.log(`model loaded: ${JSON.stringify(info).slice(0, 120)}`);
  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC((cmd: string, args?: Json) => bridge.ipc(cmd, args));

  const { runAgentTurn } = await import("../../src/lib/agentLoop");
  const { rememberFact, MEMORY_INDEX } = await import("../../src/lib/memoryFiles");

  const runsDir = path.join(DIR, "runs");
  mkdirSync(runsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  const outFile = path.join(runsDir, `memory-${stamp}.jsonl`);

  const grade = (s: Scenario, finalText: string): boolean => {
    const t = norm(finalText);
    if (s.grade.kind === "answer-all") {
      return (s.grade.expect as string[]).every((n) => t.includes(norm(n)));
    }
    return t.includes(norm(s.grade.expect as string));
  };

  const arms = (armFilter ? [armFilter] : ["control", "seeded"]) as ("control" | "seeded")[];

  for (const s of chosen) {
    for (const arm of arms) {
      const ws = mkdtempSync(path.join(tmpdir(), `chaty-membench-${s.id}-`));
      for (const [rel, body] of Object.entries(s.files)) {
        const p = path.join(ws, rel);
        mkdirSync(path.dirname(p), { recursive: true });
        writeFileSync(p, body);
      }
      if (arm === "seeded") {
        // Pre-populate memory exactly as a prior session's remember() would.
        const fs = {
          readFile: async (rel: string) => {
            const p = path.join(ws, rel);
            if (!existsSync(p)) throw new Error("missing");
            return (await import("node:fs")).readFileSync(p, "utf8");
          },
          writeFile: async (rel: string, content: string) => {
            const p = path.join(ws, rel);
            mkdirSync(path.dirname(p), { recursive: true });
            writeFileSync(p, content);
          },
        };
        await rememberFact(fs, s.seedFact.title, s.seedFact.fact, "en");
      }
      // memoryIndex must be "" for control (feature dormant) and the real
      // index for seeded — that single value is the whole experiment.
      let memoryIndex = "";
      if (arm === "seeded") {
        memoryIndex = (await import("node:fs")).readFileSync(path.join(ws, MEMORY_INDEX), "utf8");
      }

      await bridge.call("agent_set_workspace", { path: ws });
      let steps = 0,
        finalText = "",
        error: string | undefined;
      const seen = new Set<string>();
      const t0 = Date.now();
      try {
        await new Promise<void>((resolve) => {
          runAgentTurn(
            s.instruction,
            [] as never,
            ws,
            "en",
            {
              thinkMode: "off",
              nCtx: 16384,
              maxSteps: 25,
              temperature: 0.2,
              bashTimeout: 120,
              memoryIndex,
              signal: { cancelled: false } as never,
              approve: async () => true,
              approveDir: async () => false,
              approveSudo: async () => ({ ok: false }),
            } as never,
            {
              onThinking: () => {},
              onAssistantText: () => {},
              onStep: (st: { id: string }) => {
                if (!seen.has(st.id)) {
                  seen.add(st.id);
                  steps++;
                }
              },
              onFinal: (text: string) => {
                finalText = text;
                resolve();
              },
              onError: (m: string) => {
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
      const ok = !error && grade(s, finalText);
      const row = {
        scenario: s.id,
        arm,
        resolved: ok,
        steps,
        seconds: Math.round((Date.now() - t0) / 1000),
        why: error,
      };
      appendFileSync(outFile, JSON.stringify(row) + "\n");
      console.log(`${ok ? "✓" : "✗"} ${s.id} [${arm}]  ${row.seconds}s  ${steps} steps${error ? `  — ${error.slice(0, 70)}` : ""}`);
      rmSync(ws, { recursive: true, force: true });
    }
  }

  bridge.kill();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
