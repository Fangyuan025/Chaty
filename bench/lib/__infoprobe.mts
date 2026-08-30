import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 8192 });
console.error(JSON.stringify(i, null, 1));
process.exit(0);
