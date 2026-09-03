/** Behavioural audit of the generate path: does each edge case do what the
 *  code says it intends, and is the engine still usable afterwards? */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const info: any = await b.call("load_model", { path: process.argv[2], nCtx: 2048 });
if (!info?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${info.modelName} nCtx=${info.nCtx}`);

async function run(label: string, req: any, opts: { cancelAfterMs?: number } = {}) {
  let text = "", stats: any = null, err = "";
  const p = b.call("generate", { request: req }, (ev: any) => {
    if (ev.type === "token") text += ev.text;
    if (ev.type === "done") stats = ev.stats;
    if (ev.type === "error") err = ev.message;
  });
  if (opts.cancelAfterMs) {
    setTimeout(() => void b.call("cancel_generation", {}).catch(() => {}), opts.cancelAfterMs);
  }
  try { await p; } catch (e) { err = String((e as Error).message ?? e); }
  console.error(
    `\n[${label}]\n  tokens=${stats?.completionTokens ?? "?"} stop=${stats?.stopReason ?? "?"}` +
    ` 输出=${JSON.stringify(text.slice(0, 60))}${err ? `\n  错误: ${err.slice(0, 120)}` : ""}`,
  );
  return { text, stats, err };
}

const ask = (c: string) => [{ role: "user", content: c }];
const P = { temperature: 0, topP: 1 };

await run("maxTokens=1", { messages: ask("Count from 1 to 20."), params: { ...P, maxTokens: 1 } });
await run("stop 序列命中", { messages: ask("Say: alpha bravo charlie delta"), params: { ...P, maxTokens: 60, stop: ["bravo"] } });
await run("空消息列表", { messages: [], params: { ...P, maxTokens: 10 } });
await run("取消(200ms 后)", { messages: ask("Write a 500-word essay about the sea."), params: { ...P, maxTokens: 500 } }, { cancelAfterMs: 200 });
await run("取消后引擎仍可用", { messages: ask("Reply with the single word: ok"), params: { ...P, maxTokens: 10 } });
await run("提示词超出上下文", { messages: ask("x ".repeat(6000)), params: { ...P, maxTokens: 20 } });
await run("超限后引擎仍可用", { messages: ask("Reply with the single word: fine"), params: { ...P, maxTokens: 10 } });
process.exit(0);
