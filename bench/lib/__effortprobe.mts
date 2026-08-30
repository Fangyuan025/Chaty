/** Does the requested rung actually reach the template? */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 8192 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${i.modelName} 档=${JSON.stringify(i.effortLevels ?? [])}`);
for (const effort of ["low", "medium", "high", "xhigh"]) {
  let stats: any = null;
  await b.call("generate", {
    request: {
      messages: [{ role: "user", content: "hi" }],
      params: { temperature: 0, maxTokens: 1, think: true, effort },
    },
  }, (ev: any) => { if (ev.type === "done") stats = ev.stats; });
  console.error(`${effort.padEnd(7)} promptTokens=${stats?.promptTokens}`);
}
process.exit(0);
