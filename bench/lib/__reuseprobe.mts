/** Multi-turn KV reuse: every round after the first must be a pure append. */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 32768 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${i.modelName} 引擎=${i.backend} reasonField=${i.reasoningField} toolRole=${i.toolRole}`);
const asks = [
  "Name one primary colour.",
  "Now name a second one.",
  "And a third?",
  "Which of the three is warmest?",
];
const msgs: any[] = [{ role: "system", content: "You are terse." }];
for (const [n, ask] of asks.entries()) {
  msgs.push({ role: "user", content: ask });
  let text = "", reasoning = "", stats: any = null;
  await b.call("generate", {
    request: { messages: msgs, params: { temperature: 0, maxTokens: 64, think: true, effort: "low" } },
  }, (ev: any) => {
    if (ev.type === "token") text += ev.text;
    if (ev.type === "reasoning") reasoning += ev.text;
    if (ev.type === "done") stats = ev.stats;
  });
  const pct = stats?.promptTokens ? Math.round((100 * (stats.reused ?? 0)) / stats.promptTokens) : 0;
  console.error(`round ${n + 1}: prompt=${stats?.promptTokens} 复用=${stats?.reused} (${pct}%)  ${JSON.stringify(text.trim().slice(0, 46))}`);
  const turn: any = { role: "assistant", content: text };
  if (reasoning) turn.reasoning_content = reasoning;
  msgs.push(turn);
}
process.exit(0);
