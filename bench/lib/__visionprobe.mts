import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 8192 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${i.modelName} 引擎=${i.backend} 视觉=${i.visionReady} 多图=${i.multiImage}`);
let text = "", stats: any = null;
await b.call("generate", {
  request: {
    messages: [{ role: "user", content: process.argv[4] ?? "What colours do you see in this image? Answer in one short sentence.", images: [process.argv[3]] }],
    params: { temperature: 0, maxTokens: 100, think: true, effort: "low" },
  },
}, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") stats = ev.stats; });
console.error("── 输出 ──\n" + text.slice(0, 600));
console.error(`\nprompt=${stats?.promptTokens} 复用=${stats?.reused}`);
process.exit(0);
