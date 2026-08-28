/** The degeneration case, end to end: a long conversation, then a screenshot.
 *
 *  This is the shape that produced nothing for ninety minutes — the vision
 *  model evaluates the span up to the last image in ONE pass, and that span
 *  used to be the whole transcript. Walked here as a real conversation, so the
 *  cache is warm when the picture arrives, and timed against the same prompt
 *  read cold. */
import { readFileSync } from "node:fs";
import { Bridge } from "../lib/bridge.mts";

const tiles = readFileSync(process.argv[3], "utf8").trim().split("\n");
const COLD = process.argv[4] === "cold";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 65536 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
const per = i.multiImage === false ? 1 : 4;
console.error(`### ${(i.modelName ?? "").slice(0, 30)} 引擎=${i.backend} ${COLD ? "冷" : "热"} 每轮${per}张`);

const LINE = "  const handler = (req, res) => { res.status(200).json({ ok: true, id: req.params.id }); };\n";
const msgs: any[] = [{ role: "system", content: "You are a coding agent. Answer in one short sentence." }];

async function step(content: string, images?: string[]) {
  const m: any = { role: "user", content };
  if (images?.length) m.images = images;
  msgs.push(m);
  const t0 = Date.now();
  let text = "", st: any = null;
  await b.call("generate", {
    request: { messages: msgs, params: { temperature: 0.3, topP: 0.9, maxTokens: 48, think: false } },
  }, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") st = ev.stats; });
  msgs.push({ role: "assistant", content: text });
  return { secs: (Date.now() - t0) / 1000, prompt: st?.promptTokens ?? 0, reused: st?.reused ?? 0 };
}

// Eight tool results, ~4k tokens each: a working session, not a synthetic wall.
for (let k = 0; k < 8; k++) {
  const r = await step(`<tool_result name="read_file">\n${LINE.repeat(180)}</tool_result>`);
  if (COLD) msgs.splice(1); // cold: keep only the system turn, rebuild below
  else console.error(`  文本轮 ${k + 1}  prompt=${String(r.prompt).padStart(6)} 复用 ${Math.round(100 * r.reused / Math.max(1, r.prompt))}%  ${r.secs.toFixed(1)}s`);
}
if (COLD) {
  // Same transcript, never evaluated: the shape after a restart.
  for (let k = 0; k < 8; k++) {
    msgs.push({ role: "user", content: `<tool_result name="read_file">\n${LINE.repeat(180)}</tool_result>` });
    msgs.push({ role: "assistant", content: "Noted." });
  }
}
const r = await step('<tool_result name="browser_screenshot">Captured the page.</tool_result>', tiles.slice(0, per));
console.error(
  `  ▶ 截图轮  prompt=${r.prompt}  复用 ${r.reused} tok (${Math.round(100 * r.reused / Math.max(1, r.prompt))}%)  用时 ${r.secs.toFixed(1)}s`,
);
process.exit(0);
