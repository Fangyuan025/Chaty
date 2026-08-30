/** One generation against a freshly loaded model — is the output coherent? */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 8192 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${i.modelName} 引擎=${i.backend} 思考档=${JSON.stringify(i.effortLevels ?? [])}`);
let text = "", stats: any = null;
await b.call("generate", {
  request: {
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: process.argv[3] ?? "Name three primary colours, then count from 1 to 5." },
    ],
    params: { temperature: 0.3, topP: 0.9, maxTokens: 120, think: false },
  },
}, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") stats = ev.stats; });
console.error("── 输出 ──\n" + text.slice(0, 700));
console.error(`\nprompt=${stats?.promptTokens} 复用=${stats?.reused} tps=${stats?.tokensPerSecond?.toFixed(1)}`);
process.exit(0);
