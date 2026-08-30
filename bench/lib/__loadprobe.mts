/** Load one model and print what comes back — the error path included. */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
try {
  const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 4096 });
  console.log("loaded:", i?.loaded, "|", i?.modelName ?? "", "| vision:", i?.visionReady);
} catch (e: any) {
  console.log("ERROR →", String(e?.message ?? e));
}
process.exit(0);
