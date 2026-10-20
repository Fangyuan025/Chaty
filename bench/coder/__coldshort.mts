/** Decode speed right after a load, on short chat turns — the app's first
 *  replies, before a long prompt has brought the weights in off the disk.
 *
 *    CHATY_BENCH_MODEL=<dir> npx tsx bench/coder/__coldshort.mts
 */
import { Bridge, type Json } from "../lib/bridge.mts";

const bin = process.env.CHATY_HEADLESS_BIN ?? "/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless";
const bridge = new Bridge(bin);
const t0 = Date.now();
const info = (await bridge.call("load_model", { path: process.env.CHATY_BENCH_MODEL })) as Json & { nCtx?: number };
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s nCtx=${info.nCtx}`);
const messages: { role: string; content: string }[] = [];
for (const q of ["用一句话介绍你自己。", "再用三句话讲讲 Rust 的所有权。", "给我写一个 Python 的快速排序。", "解释一下它的时间复杂度。", "谢谢，总结一下我们聊了什么。"]) {
  messages.push({ role: "user", content: q });
  let reply = "";
  let stats: { promptTokens?: number; completionTokens?: number; tokensPerSecond?: number } = {};
  const s = Date.now();
  await bridge.call("generate", { request: { messages, params: { temperature: 0, maxTokens: 160, think: false } } }, (ev: Record<string, unknown>) => {
    if (ev?.type === "token") reply += String(ev.text ?? "");
    if (ev?.type === "done") stats = (ev.stats ?? {}) as typeof stats;
  });
  console.log(`prompt=${stats.promptTokens} gen=${stats.completionTokens} tps=${(stats.tokensPerSecond ?? 0).toFixed(1)} wall=${((Date.now() - s) / 1000).toFixed(1)}s`);
  messages.push({ role: "assistant", content: reply });
}
try { await bridge.call("eject_model", {}); } catch { /* going anyway */ }
bridge.kill();
