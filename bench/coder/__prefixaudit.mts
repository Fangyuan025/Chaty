/** Is every code-mode round a pure append onto the last?
 *
 *  Drives the REAL agent loop (the runner's wiring) and intercepts every
 *  `generate` on the way to the engine, so what is measured is the prompt the
 *  model actually received rather than a reconstruction of it. Per call it
 *  records the message array and, from the done event, how much of the prompt
 *  the engine resumed from cache.
 *
 *  A round that appends diverges from the last one at its END. A round that
 *  rewrote history diverges in the MIDDLE, and every token after the
 *  divergence is prefill the machine paid for twice.
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

import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync, mkdirSync, cpSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Bridge } from "../lib/bridge.mts";

type Msg = { role: string; content: string };
type Call = { turn: number; msgs: Msg[]; promptTokens?: number; reused?: number; reply?: string };

const bin =
  process.env.CHATY_HEADLESS_BIN ??
  "/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless";
const model = process.env.CHATY_BENCH_MODEL as string;
const nCtx = Number(process.env.CHATY_BENCH_NCTX ?? 32768);

const bridge = new Bridge(bin);
const info = (await bridge.call("load_model", { path: model, nCtx })) as Record<string, unknown>;
if (!info?.loaded) throw new Error("model did not load");
console.log(
  `model: ${info.modelName} backend=${info.backend} toolRole=${info.toolRole} ` +
    `reasoningField=${info.reasoningField} nCtx=${info.nCtx}`,
);

const calls: Call[] = [];
let turn = 0;

const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(async (cmd: string, args?: Record<string, never>) => {
  const a = args as Record<string, never> & Record<string, unknown>;
  if (cmd !== "generate") return bridge.ipc(cmd, args);
  const req = a.request as { messages: Record<string, unknown>[] };
  const msgs: Msg[] = (req?.messages ?? []).map((m) => ({
    role: String(m.role),
    content: String(m.content ?? "") + (m.reasoning_content ? ` R:${m.reasoning_content}` : ""),
  }));
  const rec: Call = { turn, msgs, reply: "" };
  calls.push(rec);
  // `generate` streams through `onEvent` — the same key bridge.ipc reads.
  // Getting this wrong hands the loop an empty reply and every measurement
  // after it describes a model that said nothing.
  const ch = a.onEvent as { onmessage?: (e: unknown) => void } | undefined;
  return bridge.call("generate", { request: a.request }, (ev: Record<string, unknown>) => {
    if (ev?.type === "token") rec.reply += String(ev.text ?? "");
    if (ev?.type === "done") {
      const s = ev.stats as { promptTokens?: number; reused?: number };
      rec.promptTokens = s?.promptTokens;
      rec.reused = s?.reused;
    }
    ch?.onmessage?.(ev);
  });
});

const loop = await import("../../src/lib/agentLoop");
const { runAgentTurn, replayableTail } = loop;

const ws = mkdtempSync(path.join(tmpdir(), "chaty-prefix-"));
// A bench task directory (task.md + workspace/) when one is named, so the
// audit runs the same work a graded run does; otherwise a scratch project.
const taskDir = process.env.TASK_DIR;
if (taskDir) {
  cpSync(path.join(taskDir, "workspace"), ws, { recursive: true });
} else {
  mkdirSync(path.join(ws, "src"), { recursive: true });
  writeFileSync(
    path.join(ws, "README.md"),
    "# Scratch project\n\nA tiny project used to exercise a long agent run.\n",
  );
  writeFileSync(
    path.join(ws, "src", "main.py"),
    "def add(a, b):\n    return a + b\n\n\nif __name__ == '__main__':\n    print(add(1, 2))\n",
  );
}
await bridge.call("agent_set_workspace", { path: ws });
console.log(`workspace: ${ws}`);

const TASK =
  (taskDir ? readFileSync(path.join(taskDir, "task.md"), "utf8") : undefined) ??
  process.env.TASK ??
  "在这个工作区里:先读 README.md 和 src/main.py,然后给 src/main.py 增加 mul(a, b) 和 div(a, b)" +
    "(除数为 0 时抛 ValueError),在 tests/test_main.py 写覆盖四个函数(含报错分支)的测试," +
    "用 python3 -m pytest 跑起来并修掉失败,最后更新 README.md 说明这四个函数。一步一步做。";

// The app's own cross-turn rule, so the audit sees the history a real
// continuation would see rather than an idealised one.
type Card = { role: string; text: string; steps: unknown[]; prompt?: Msg[] };
const cards: Card[] = [];
const TURNS = Number(process.env.TURNS ?? 3);

for (let t = 1; t <= TURNS; t++) {
  turn = t;
  const replay = replayableTail(cards as never);
  const history = (replay ?? cards.map((c) => ({ role: c.role, content: c.text }))) as never;
  const input = t === 1 ? TASK : "继续";
  cards.push({ role: "user", text: input, steps: [] });
  const asst: Card = { role: "assistant", text: "", steps: [] };
  cards.push(asst);
  console.log(
    `\n--- turn ${t} (history: ${replay ? "replayed tail" : "summary fallback"}, ` +
      `${(history as unknown[]).length} msgs) ---`,
  );
  let steps = 0;
  await runAgentTurn(
    input,
    history,
    ws,
    "zh",
    {
      thinkMode: "normal",
      supportsThinking: !!info.supportsThinking,
      thinkSwitch: !!info.thinkSwitch,
      toolRole: !!info.toolRole,
      reasoningField: !!info.reasoningField,
      visionReady: !!info.visionReady,
      nCtx: info.nCtx,
      maxSteps: Number(process.env.MAXSTEPS ?? 14),
      temperature: 0.3,
      signal: new loop.AgentSignal(),
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
        if (s.status !== "running") {
          steps++;
          console.log(`    step ${steps}: ${s.call?.name ?? "?"} [${s.status}]`);
        }
      },
      onFinal: (final: string) => {
        asst.text = final;
      },
      onError: (m: string) => console.log(`  ERROR: ${m}`),
      onTrace: (ev: { kind: string; text: string }) =>
        console.log(`    trace[${ev.kind}] ${JSON.stringify(ev.text.slice(0, 160))}`),
      onTranscript: (tail: Msg[]) => {
        for (const c of cards) c.prompt = undefined;
        asst.prompt = tail;
      },
    } as never,
  );
  console.log(`  steps=${steps}`);
}

// ---- the audit ------------------------------------------------------------
const h = (s: string) => createHash("sha1").update(s).digest("hex").slice(0, 8);
console.log(`\n===== ${calls.length} generate calls =====`);
let prev: Msg[] | null = null;
for (let i = 0; i < calls.length; i++) {
  const c = calls[i];
  let note = "first";
  if (prev) {
    let same = 0;
    while (
      same < prev.length &&
      same < c.msgs.length &&
      prev[same].role === c.msgs[same].role &&
      prev[same].content === c.msgs[same].content
    )
      same++;
    const appended = c.msgs.length - same;
    const dropped = prev.length - same;
    note =
      dropped === 0
        ? `append(+${appended})`
        : `REWROTE at #${same}: dropped ${dropped}, added ${appended}` +
          (same < prev.length
            ? `  [was ${prev[same].role}/${h(prev[same].content)}` +
              ` now ${same < c.msgs.length ? c.msgs[same].role + "/" + h(c.msgs[same].content) : "-"}]`
            : "");
  }
  // Did the loop store the previous reply verbatim? Anything it rewrote is a
  // token the engine's cache no longer matches — and on a hybrid model that is
  // not a partial resume, it is the whole cache.
  let stored = "";
  if (prev && c.msgs.length > prev.length) {
    const added = c.msgs.slice(prev.length - (prev.length ? 0 : 0));
    const asst = c.msgs.slice(prev.length).find((m) => m.role === "assistant");
    const raw = (calls[i - 1].reply ?? "").trim();
    if (asst && raw) {
      const kept = asst.content.replace(/ R:.*$/s, "").trim();
      const inRaw = raw.includes(kept.slice(0, Math.min(80, kept.length)));
      if (!inRaw) stored = "  STORED≠GENERATED";
    }
    void added;
  }
  const pct = c.promptTokens ? Math.round((100 * (c.reused ?? 0)) / c.promptTokens) : 0;
  console.log(
    `t${c.turn} #${String(i).padStart(2)} msgs=${String(c.msgs.length).padStart(3)}` +
      ` prompt=${String(c.promptTokens ?? "?").padStart(6)}` +
      ` reused=${String(c.reused ?? "?").padStart(6)} (${String(pct).padStart(3)}%)  ${note}${stored}`,
  );
  if (process.env.SHOW_REPLIES === "1") {
    const r = c.reply ?? "";
    console.log(`      reply[${r.length}] head: ${JSON.stringify(r.slice(0, 120))}`);
    console.log(`      reply     tail: ${JSON.stringify(r.slice(-260))}`);
  }
  prev = c.msgs;
}
console.log(`\nworkspace kept at: ${ws}`);
bridge.kill();
process.exit(0);
