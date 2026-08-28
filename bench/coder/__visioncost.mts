/** A prompt carrying pixels is evaluated by the VLM's prepare() in ONE forward
 *  pass over everything up to the last image — no chunking is possible, the
 *  model computes the images' positions from the text before them. So the cost
 *  of a screenshot round is set by how much transcript precedes it. This walks
 *  that up and times the pass. */
import { readFileSync } from "node:fs";
import { Bridge } from "../lib/bridge.mts";

const tiles = readFileSync(process.argv[3], "utf8").trim().split("\n").slice(0, Number(process.argv[4] ?? 13));
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 65536 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${(i.modelName ?? "").slice(0, 30)}  图 ${tiles.length} 张  nCtx=${i.nCtx}`);

// Filler that tokenizes like a transcript rather than like one repeated token.
const LINE = "  const handler = (req, res) => { res.status(200).json({ ok: true, id: req.params.id }); };\n";
const fillerFor = (approxTokens: number) => LINE.repeat(Math.max(1, Math.round(approxTokens / 22)));

for (const want of [1000, 4000, 8000, 16000, 32000]) {
  const t0 = Date.now();
  let stats: any = null, err = "", text = "";
  try {
    await b.call("generate", {
      request: {
        messages: [
          { role: "system", content: "You are a code reviewer." },
          { role: "user", content: "Earlier work:\n" + fillerFor(want) },
          { role: "assistant", content: "Understood." },
          { role: "user", content: "Here is the page. One sentence on the layout.", images: tiles },
        ],
        params: { temperature: 0.3, topP: 0.9, maxTokens: 40, think: false },
      },
    }, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") stats = ev.stats; });
  } catch (e: any) { err = String(e?.message ?? e).slice(0, 80); }
  const secs = (Date.now() - t0) / 1000;
  console.error(
    `  文本约 ${String(want).padStart(5)} tok → prompt=${String(stats?.promptTokens ?? "?").padStart(6)}  用时 ${secs.toFixed(0).padStart(4)}s` +
    (err ? `  ❌ ${err}` : ""),
  );
}
process.exit(0);
