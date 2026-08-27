/** Does the deep / standard switch still change how much a model reasons in
 *  code mode, on models with no native effort ladder? For those, the rung is
 *  carried by ONE line of the system prompt, so this drives the real
 *  systemPrompt() rather than a paraphrase of it. */
import { Bridge } from "../lib/bridge.mts";
import { systemPrompt, thinkSuffix } from "../../src/lib/agentLoop";

const path = process.argv[2];
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path, nCtx: 8192 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
const native = (i.effortLevels?.length ?? 0) > 0;
console.error(`### ${(i.modelName ?? "").slice(0, 34)}  原生档位=${native ? "有(不适用)" : "无(正是要测的)"}`);

const norm = (s: string) => s.replace(/<\|channel>/g, "<think>").replace(/<channel\|>/g, "</think>");
const thinkOf = (r: string) => {
  const s = norm(r);
  const o = s.indexOf("<think>"), c = s.indexOf("</think>");
  if (c !== -1 && (o === -1 || c < o)) return s.slice(0, c);
  return o === -1 ? "" : s.slice(o + 7, c === -1 ? undefined : c);
};

const TASKS = [
  "Add a --dry-run flag to the CLI in src/main.rs. Start by looking at what is there.",
  "The test suite is failing on Windows only. Work out why.",
  "Rename the `parse` function to `parseConfig` everywhere it is used.",
  "Our JSON parser is 3x slower than serde_json on nested input. Find out where the time goes.",
  "Add retry-with-backoff to the HTTP client, and make sure the existing tests still cover it.",
];

const SYS = {
  normal: systemPrompt("/Users/stevenlin/Desktop/Chaty-repo", false, "normal"),
  deep: systemPrompt("/Users/stevenlin/Desktop/Chaty-repo", false, "deep"),
};
async function one(mode: "normal" | "deep", task: string) {
  let text = "";
  await b.call("generate", {
    request: {
      messages: [
        { role: "system", content: SYS[mode] },
        // The rung rides on the user turn too, exactly as the agent loop sends it.
        { role: "user", content: task + thinkSuffix(mode, false) },
      ],
      params: { temperature: 0.3, topP: 0.9, maxTokens: 1400, think: true },
    },
  }, (ev: any) => { if (ev.type === "token") text += ev.text; });
  return thinkOf(text).trim().length;
}

// Paired by task — reasoning length swings hugely between questions, so the
// only honest comparison is the same question under both settings — and
// alternating which goes first so warm-up cannot favour one of them.
let sumN = 0, sumD = 0, deepWins = 0;
for (const [k, task] of TASKS.entries()) {
  const first = k % 2 === 0 ? "normal" : "deep";
  const a = await one(first as "normal" | "deep", task);
  const bb = await one((first === "normal" ? "deep" : "normal") as "normal" | "deep", task);
  const n = first === "normal" ? a : bb;
  const d = first === "normal" ? bb : a;
  sumN += n; sumD += d; if (d > n) deepWins++;
  console.error(`  第${k + 1}题  标准 ${String(n).padStart(5)} 字   深入 ${String(d).padStart(5)} 字   ${d > n ? "深入更长" : "深入更短"} (${n ? (d / n).toFixed(2) : "?"}×)`);
}
console.error(`  合计   标准 ${sumN} 字   深入 ${sumD} 字   深入/标准 = ${(sumD / Math.max(1, sumN)).toFixed(2)}×   深入更长的题数 ${deepWins}/${TASKS.length}`);
process.exit(0);
