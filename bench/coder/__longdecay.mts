/** Does a long Code-mode session slow down, or degenerate, as it grows?
 *
 *  Drives the REAL agent loop through a multi-turn project that keeps adding
 *  to the context, and for every `generate` records: prompt size, how much the
 *  engine resumed, decode speed, the wall time around it (prefill is the
 *  difference), whether the reply degenerated (a run of one character — the
 *  "!!!!!!!!" a model emits when its logits go NaN), the sidecar's physical
 *  footprint and the machine's swap. With CHATY_MLX_MEMLOG=1 the sidecar adds
 *  MLX's own active / cache / peak memory per call on stderr.
 *
 *    CHATY_BENCH_MODEL=<dir> [NCTX=…] [TURNS=8] [MAXSTEPS=20] npx tsx bench/coder/__longdecay.mts
 */
const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
  clear: () => {},
  key: () => null,
  length: 0,
};
g.navigator ??= { userAgent: "chaty-bench" };

import { execSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Bridge } from "../lib/bridge.mts";

const bin =
  process.env.CHATY_HEADLESS_BIN ??
  "/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless";
const model = process.env.CHATY_BENCH_MODEL as string;
const nCtx = process.env.NCTX ? Number(process.env.NCTX) : undefined;

const bridge = new Bridge(bin, { CHATY_MLX_MEMLOG: "1" });
const info = (await bridge.call("load_model", { path: model, ...(nCtx ? { nCtx } : {}) })) as Record<string, unknown>;
if (!info?.loaded) throw new Error("model did not load");
console.log(`model: ${info.modelName} backend=${info.backend} nCtx=${info.nCtx} toolFormat=${info.toolFormat}`);

function sidecarFootprintMb(): number | null {
  try {
    const pid = execSync("pgrep -f chaty-mlx-aarch64 | head -1").toString().trim();
    if (!pid) return null;
    const out = execSync(`footprint -p ${pid} 2>/dev/null | grep -m1 "Footprint:"`).toString();
    const m = out.match(/Footprint:\s*([\d.]+)\s*(KB|MB|GB)/);
    if (!m) return null;
    const v = Number(m[1]);
    return Math.round(m[2] === "GB" ? v * 1024 : m[2] === "KB" ? v / 1024 : v);
  } catch {
    return null;
  }
}
function swapUsedMb(): number {
  const s = execSync("sysctl -n vm.swapusage").toString();
  const m = s.match(/used = ([\d.]+)M/);
  return m ? Math.round(Number(m[1])) : -1;
}
/** A reply that degenerated: one character, repeated, where words belong. */
function degenerate(text: string): string | null {
  const m = text.match(/(.)\1{23,}/su);
  if (!m) return null;
  if (/[\s=\-_*#.~`]/.test(m[1])) return null; // rules, indents and fences repeat legitimately
  return `${JSON.stringify(m[1])}×${m[0].length}`;
}

let call = 0;
let turn = 0;
const t0 = Date.now();
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(async (cmd: string, args?: Record<string, never>) => {
  if (cmd !== "generate") return bridge.ipc(cmd, args);
  const a = args as Record<string, unknown>;
  const n = ++call;
  const started = Date.now();
  let reply = "";
  let stats: { promptTokens?: number; reused?: number; completionTokens?: number; tokensPerSecond?: number; stopReason?: string } = {};
  const ch = a.onEvent as { onmessage?: (e: unknown) => void } | undefined;
  const result = await bridge.call("generate", { request: a.request }, (ev: Record<string, unknown>) => {
    if (ev?.type === "token") reply += String(ev.text ?? "");
    if (ev?.type === "done") stats = (ev.stats ?? {}) as typeof stats;
    ch?.onmessage?.(ev);
  });
  const wall = (Date.now() - started) / 1000;
  const gen = stats.completionTokens ?? 0;
  const tps = stats.tokensPerSecond ?? 0;
  const decode = tps > 0 ? gen / tps : 0;
  const prefill = Math.max(0, wall - decode);
  const fresh = (stats.promptTokens ?? 0) - (stats.reused ?? 0);
  const deg = degenerate(reply);
  console.log(
    `#${String(n).padStart(3)} t${turn} +${Math.round((Date.now() - t0) / 1000)}s` +
      ` prompt=${String(stats.promptTokens ?? "?").padStart(6)} fresh=${String(fresh).padStart(6)}` +
      ` gen=${String(gen).padStart(5)} tps=${tps.toFixed(1).padStart(5)}` +
      ` prefill≈${prefill.toFixed(1).padStart(5)}s (${fresh > 0 && prefill > 0.05 ? Math.round(fresh / prefill) : "-"} tok/s)` +
      ` stop=${stats.stopReason ?? "?"} footprint=${sidecarFootprintMb() ?? "?"}MB swap=${swapUsedMb()}MB` +
      (deg ? `  *** DEGENERATE ${deg} ***` : ""),
  );
  if (deg && process.env.DUMP_DEGEN !== "0") console.log(`    reply tail: ${JSON.stringify(reply.slice(-300))}`);
  return result;
});

const loop = await import("../../src/lib/agentLoop");
const { runAgentTurn, replayableTail } = loop;

const ws = mkdtempSync(path.join(tmpdir(), "chaty-decay-"));
mkdirSync(path.join(ws, "src"), { recursive: true });
writeFileSync(path.join(ws, "README.md"), "# ledger\n\n一个命令行记账工具（待实现）。\n");
await bridge.call("agent_set_workspace", { path: ws });
console.log(`workspace: ${ws}`);

const TURNS_TEXT = [
  "在这个工作区里实现一个命令行记账工具 ledger：src/ledger/ 下分 models.py（交易记录数据类，含日期、金额、类别、备注）、" +
    "storage.py（JSON 文件读写）、report.py（按月、按类别汇总）、cli.py（argparse：add/list/report/delete 子命令）；" +
    "tests/ 下为每个模块写测试；用 python3 跑测试并修掉失败。写完把每个文件都读一遍检查。",
  "给 report 增加按周汇总和导出 CSV，补测试并跑通。",
  "把 storage 改成同时支持 SQLite 后端（保留 JSON 后端，用参数选择），补测试并跑通。",
  "通读所有源码和测试，找出潜在 bug 并修复，跑全部测试。",
  "加一个预算功能：每个类别可以设月预算，report 时标出超支，补测试并跑通。",
  "写一份详细的 README，包含每个子命令的用法示例；再把 cli 的帮助信息完善。",
  "重构：把重复的日期解析逻辑提成 src/ledger/utils.py，更新所有引用，跑测试。",
  "再通读一遍整个项目的每个文件，确认测试全部通过，总结目前的功能和代码结构。",
];
const TURNS = Number(process.env.TURNS ?? TURNS_TEXT.length);

type Card = { role: string; text: string; steps: unknown[]; prompt?: { role: string; content: string }[] };
const cards: Card[] = [];
for (let t = 1; t <= TURNS; t++) {
  turn = t;
  const replay = replayableTail(cards as never);
  const history = (replay ?? cards.map((c) => ({ role: c.role, content: c.text }))) as never;
  const input = TURNS_TEXT[t - 1] ?? "继续";
  cards.push({ role: "user", text: input, steps: [] });
  const asst: Card = { role: "assistant", text: "", steps: [] };
  cards.push(asst);
  console.log(`\n--- turn ${t}: ${input.slice(0, 40)}… ---`);
  let steps = 0;
  const signal = new loop.AgentSignal();
  await runAgentTurn(
    input,
    history,
    ws,
    "zh",
    {
      // The owner's Code-mode setup: deep thinking, 0.7 / 0.95 / 20 / 0 / 1.1.
      thinkMode: (process.env.THINK ?? "deep") as never,
      toolFormat: ((info as { toolFormat?: string }).toolFormat ?? "xml") as never,
      supportsThinking: !!info.supportsThinking,
      thinkSwitch: !!info.thinkSwitch,
      toolRole: !!info.toolRole,
      reasoningField: !!info.reasoningField,
      visionReady: !!info.visionReady,
      sessionId: "decay-session",
      nCtx: info.nCtx,
      maxSteps: Number(process.env.MAXSTEPS ?? 20),
      temperature: Number(process.env.TEMP ?? 0.7),
      sampling: {
        topP: Number(process.env.TOP_P ?? 0.95),
        topK: Number(process.env.TOP_K ?? 20),
        minP: Number(process.env.MIN_P ?? 0),
        repeatPenalty: Number(process.env.REPEAT ?? 1.1),
      },
      signal,
      autoApproveEdits: true,
      autoRunReadOnly: true,
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
      allowedCommands: ["python3", "pytest", "ls", "cat"],
    } as never,
    {
      onThinking: () => {},
      onAssistantText: () => {},
      onStats: () => {},
      onContext: () => {},
      onPrefill: () => {},
      onPlan: () => {},
      onCompacted: () => console.log("    [compacted]"),
      onAskUser: async () => "ok",
      onStep: (s: { status: string; call?: { name?: string } }) => {
        if (s.status !== "running") steps++;
      },
      onFinal: (final: string) => {
        asst.text = final;
      },
      onError: (m: string) => console.log(`  ERROR: ${m}`),
      onTrace: () => {},
      onTranscript: (tail: { role: string; content: string }[]) => {
        for (const c of cards) c.prompt = undefined;
        asst.prompt = tail;
      },
    } as never,
  );
  console.log(`  turn ${t} done: ${steps} steps`);
}
try {
  await bridge.call("eject_model", {});
} catch {
  /* going anyway */
}
bridge.kill();
