/** Decode speed at a long context, before, during and after memory pressure
 *  from another process — the machine as it is when the app, a browser and
 *  the rest are open beside the model.
 *
 *    CHATY_BENCH_MODEL=<dir> [CTX_TOKENS=40000] [EXTRA_GB=3] [GEN=64] npx tsx bench/coder/__pressure.mts
 */
import { execSync, spawn, type ChildProcess } from "node:child_process";
import { Bridge, type Json } from "../lib/bridge.mts";

const bin = process.env.CHATY_HEADLESS_BIN ?? "/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless";
const model = process.env.CHATY_BENCH_MODEL as string;
const ctxTokens = Number(process.env.CTX_TOKENS ?? 40000);
const extraGb = Number(process.env.EXTRA_GB ?? 3);
const bridge = new Bridge(bin);

const info = (await bridge.call("load_model", { path: model })) as Json & { nCtx?: number };
console.log(`loaded nCtx=${info.nCtx}`);

// About ctxTokens tokens of code-like text, every line different.
const lines: string[] = [];
for (let i = 0; lines.join("\n").length < ctxTokens * 3.2; i++) {
  lines.push(`def helper_${i}(value, scale=${(i % 9) + 1}):\n    return value * scale + ${i % 13}  # step ${i}`);
}
const big = `下面是一个模块的源码，先读完：\n\n${lines.join("\n")}\n\n读完后，按顺序解释前面几个函数在做什么。`;
const messages: { role: string; content: string }[] = [{ role: "user", content: big }];

function availableGb(): number {
  const v = execSync("vm_stat").toString();
  const page = Number(v.match(/page size of (\d+)/)?.[1] ?? 16384);
  const n = (k: string) => Number(v.match(new RegExp(`${k}:\\s+(\\d+)`))?.[1] ?? 0);
  return ((n("Pages free") + n("Pages inactive") + n("Pages speculative")) * page) / 2 ** 30;
}
function swapMb(): number {
  return Math.round(Number(execSync("sysctl -n vm.swapusage").toString().match(/used = ([\d.]+)M/)?.[1] ?? 0));
}

async function round(label: string) {
  let reply = "";
  let stats: { promptTokens?: number; completionTokens?: number; tokensPerSecond?: number } = {};
  const t0 = Date.now();
  // A round that crawls is cut off once it is decoding: what is being measured
  // is how slow it got. Not while it reads the prompt — cut there, the cache
  // goes and every round after re-reads everything.
  let cap: ReturnType<typeof setTimeout> | undefined;
  await bridge.call(
    "generate",
    { request: { messages, params: { temperature: 0, maxTokens: Number(process.env.GEN ?? 64), think: false } } },
    (ev: Record<string, unknown>) => {
      if (ev?.type === "token") {
        reply += String(ev.text ?? "");
        cap ??= setTimeout(() => void bridge.call("cancel_generation", {}).catch(() => {}), 120_000);
      }
      if (ev?.type === "done") stats = (ev.stats ?? {}) as typeof stats;
    },
  );
  clearTimeout(cap);
  const wall = (Date.now() - t0) / 1000;
  console.log(
    `${label.padEnd(10)} prompt=${stats.promptTokens} gen=${stats.completionTokens} ` +
      `tps=${(stats.tokensPerSecond ?? 0).toFixed(1)} wall=${wall.toFixed(1)}s swap=${swapMb()}MB avail=${availableGb().toFixed(1)}GB`,
  );
  messages.push({ role: "assistant", content: reply }, { role: "user", content: "继续，解释接下来的几个函数。" });
}

await round("prefill");
for (let i = 1; i <= 3; i++) await round(`calm ${i}`);
const take = Math.max(1, Math.round(availableGb() + extraGb));
console.log(`--- pressure: another process takes ${take} GB ---`);
// Random bytes, so the compressor cannot fold them away, and touched over and
// over, so they stay active the way a working app's memory does.
const hog: ChildProcess = spawn("python3", [
  "-c",
  [
    "import os, time",
    "block = os.urandom(64 * 1024 * 1024)",
    `b = bytearray(${take} * 1024**3)`,
    "for o in range(0, len(b), len(block)): b[o:o+len(block)] = block[: len(b) - o]",
    'print("held", flush=True)',
    "while True:",
    "    if os.getppid() == 1: break  # the bench is gone: so is the pressure",
    "    s = 0",
    "    for i in range(0, len(b), 16384): s ^= b[i]",
    "    time.sleep(0.5)",
  ].join("\n"),
]);
await new Promise<void>((r) => hog.stdout!.once("data", () => r()));
for (let i = 1; i <= 4; i++) await round(`pressed ${i}`);
hog.kill("SIGKILL");
await new Promise((r) => setTimeout(r, 3000));
console.log("--- pressure released ---");
for (let i = 1; i <= 2; i++) await round(`after ${i}`);
try {
  await bridge.call("eject_model", {});
} catch {
  /* going anyway */
}
bridge.kill();
