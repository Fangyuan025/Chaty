/** Behavioural audit: does each bad input produce the outcome the code says
 *  it intends? Each case names the intended behaviour, not just "an error". */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const S = process.argv[2];

async function probe(name: string, expect: string, fn: () => Promise<unknown>) {
  let outcome: string;
  try {
    const r = await fn();
    outcome = "OK: " + JSON.stringify(r).slice(0, 150);
  } catch (e) {
    outcome = "ERR: " + String((e as Error).message ?? e).slice(0, 170);
  }
  console.error(`\n[${name}]\n  期望: ${expect}\n  实际: ${outcome}`);
}

await probe("不存在的路径", "说清楚文件找不到", () =>
  b.call("load_model", { path: `${S}/definitely-not-here.gguf`, nCtx: 2048 }));

await probe("文本文件伪装成 .gguf", "说明这不是 GGUF 文件", () =>
  b.call("load_model", { path: `${S}/fake.gguf`, nCtx: 2048 }));

await probe("空目录当 MLX 模型", "说明缺少 config.json / 不是模型目录", () =>
  b.call("load_model", { path: `${S}/emptydir`, nCtx: 2048 }));

await probe("架构未知的 GGUF", "指出该架构不被这个 llama.cpp 支持", () =>
  b.call("load_model", { path: `${S}/unknown-arch.gguf`, nCtx: 2048 }));

process.exit(0);
