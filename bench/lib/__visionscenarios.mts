/** Two images in one turn, then a text follow-up: does the prefix survive? */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 32768 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${i.modelName} 引擎=${i.backend} 视觉=${i.visionReady} 多图=${i.multiImage}`);
const imgs = process.argv.slice(3);
const msgs: any[] = [
  { role: "user", content: "Describe the colours in each image, one line each.", images: imgs },
];
const rounds = ["Describe the colours in each image, one line each.", "Which image is warmer?"];
for (const [n, ask] of rounds.entries()) {
  if (n > 0) msgs.push({ role: "user", content: ask });
  let text = "", stats: any = null;
  await b.call("generate", {
    request: { messages: msgs, params: { temperature: 0, maxTokens: 90, think: true, effort: "low" } },
  }, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") stats = ev.stats; });
  const pct = stats?.promptTokens ? Math.round((100 * (stats.reused ?? 0)) / stats.promptTokens) : 0;
  console.error(`round ${n + 1}: prompt=${stats?.promptTokens} 复用=${stats?.reused} (${pct}%)`);
  console.error(`  ${text.replace(/\n+/g, " ").slice(0, 220)}`);
  msgs.push({ role: "assistant", content: text });
}
process.exit(0);
