/** Real-model proof for the tiktok-video official skill: one autonomous
 *  session — the model must discover the skill via the index, use_skill,
 *  run setup + pipeline, review, and deliver final.mp4. Probe = ffprobe. */
import { execFileSync } from "node:child_process";
import { appendFileSync, mkdtempSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-bench" };

const REPO = path.join(import.meta.dirname, "../..");
const { Bridge } = await import(path.join(REPO, "bench/lib/bridge.mts"));
type Json = Record<string, unknown> | unknown[] | string | number | boolean | null;

const PROMPT = "帮我做一个关于「为什么猫总在凌晨疯跑」的抖音短视频,30到40秒,中文配音。";

function walk(dir: string, depth: number, hit: (p: string) => void) {
  if (depth < 0) return;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    hit(p);
    if (e.isDirectory() && e.name !== "node_modules") walk(p, depth - 1, hit);
  }
}

async function main() {
  const bin = path.join(REPO, "src-tauri/target/release/chaty-headless");
  const model = process.env.CHATY_BENCH_MODEL;
  if (!model) throw new Error("set CHATY_BENCH_MODEL");
  const bridge = new Bridge(bin);
  const info = (await bridge.call("load_model", { path: model })) as Record<string, unknown>;
  if (!info || info.loaded !== true) throw new Error(`model did not load: ${JSON.stringify(info)}`);
  const nCtx = Number(info.nCtx) || 16384;
  console.log(`[tv] model loaded, nCtx=${nCtx}`);
  const { mockIPC } = await import("@tauri-apps/api/mocks");
  mockIPC((cmd: string, args?: Json) => bridge.ipc(cmd, args));
  const { runAgentTurn } = await import(path.join(REPO, "src/lib/agentLoop.ts"));
  const { officialSkills } = await import(path.join(REPO, "src/lib/skillFiles.ts"));

  const ws = mkdtempSync(path.join(tmpdir(), "chaty-tvbench-"));
  await bridge.call("agent_set_workspace", { path: ws });
  const stepNames: string[] = [];
  const seen = new Set<string>();
  let finalText = "", error: string | undefined;
  const t0 = Date.now();
  await new Promise<void>((resolve) => {
    const watchdog = setTimeout(() => { error = "watchdog: 5400s"; resolve(); }, 5_400_000);
    runAgentTurn(PROMPT, [] as never, ws, "zh",
      { thinkMode: "normal", nCtx, maxSteps: 64, temperature: 0.3,
        skills: officialSkills(), bashTimeout: 300, browserTextMode: true,
        signal: { cancelled: false } as never, approve: async () => true, approveDir: async () => false,
        approveSudo: async () => ({ ok: false }) } as never,
      { onThinking: () => {}, onAssistantText: () => {},
        onAskUser: async (_q: string, options: string[]) =>
          options[0] ?? "你来决定:按技能给的流程直接做完,不要再问我。",
        onStep: (st: { id: string; call?: { name: string; args?: Record<string, unknown> }; result?: string }) => {
          if (!seen.has(st.id)) {
            seen.add(st.id);
            stepNames.push(st.call?.name ?? "?");
            const brief = st.call?.name === "bash" ? String(st.call?.args?.command ?? "").slice(0, 120) : "";
            console.log(`  step#${stepNames.length} ${st.call?.name}${brief ? " $ " + brief : ""}`);
          }
        },
        onFinal: (t: string) => { clearTimeout(watchdog); finalText = t; resolve(); },
        onError: (m: string) => { clearTimeout(watchdog); error = m; resolve(); },
      } as never,
    );
  });
  const secs = Math.round((Date.now() - t0) / 1000);
  try { execFileSync("pkill", ["-f", ws], { stdio: "ignore" }); } catch { /* none */ }

  // Probe: a delivered final.mp4 that ffprobe confirms is a real vertical video.
  const mp4s: string[] = [];
  walk(ws, 6, (p) => { if (p.endsWith("final.mp4") && statSync(p).isFile()) mp4s.push(p); });
  let verdict = "NO final.mp4";
  if (mp4s.length) {
    try {
      const probe = JSON.parse(execFileSync("ffprobe", ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", mp4s[0]], { encoding: "utf8" }));
      const v = probe.streams.find((s: Record<string, unknown>) => s.codec_type === "video");
      const a = probe.streams.find((s: Record<string, unknown>) => s.codec_type === "audio");
      const dur = Number(probe.format.duration);
      const ok = v?.codec_name === "h264" && v?.width === 1080 && v?.height === 1920 && !!a && dur >= 15 && dur <= 95;
      verdict = `${ok ? "FUNCTIONAL" : "BROKEN"} ${v?.codec_name} ${v?.width}x${v?.height} audio=${!!a} dur=${dur?.toFixed(1)}s`;
    } catch (e) {
      verdict = `ffprobe failed: ${e instanceof Error ? e.message : e}`;
    }
  }
  const row = { ws, steps: stepNames.length, secs, error, verdict, mp4: mp4s[0] ?? null, final: finalText.slice(0, 400), stepNames };
  appendFileSync(path.join(tmpdir(), "tvbench.jsonl"), JSON.stringify(row) + "\n");
  console.log(`\n[tv] steps=${stepNames.length} secs=${secs} error=${error ?? "none"}`);
  console.log(`[tv] verdict: ${verdict}`);
  console.log(`[tv] ws: ${ws}`);
  bridge.kill();
}

main().catch((e) => { console.error(e); process.exit(2); });
