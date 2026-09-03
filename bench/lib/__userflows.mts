/** Real-usage sweep: sequences a person actually performs, checked for things
 *  they would notice — lost context, ignored switches, a dead session. */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const info: any = await b.call("load_model", { path: process.argv[2], nCtx: 8192 });
if (!info?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(`### ${info.modelName} nCtx=${info.nCtx} think=${info.supportsThinking} switch=${info.thinkSwitch}\n`);

type Msg = { role: string; content: string };
async function say(msgs: Msg[], p: Record<string, unknown> = {}) {
  let text = "", stats: any = null, err = "";
  try {
    await b.call("generate", { request: { messages: msgs, params: { temperature: 0, topP: 1, maxTokens: 200, ...p } } },
      (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") stats = ev.stats; if (ev.type === "error") err = ev.message; });
  } catch (e) { err = String((e as Error).message ?? e); }
  return { text, stats, err };
}
const strip = (s: string) => s.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<think>[\s\S]*$/, "").trim();
const check = (name: string, ok: boolean, detail: string) =>
  console.error(`${ok ? "✓" : "✗ 缺陷"} ${name}\n    ${detail}`);

// ── 1. 多轮记忆:第三轮能不能引用第一轮说过的事 ──────────────────
const conv: Msg[] = [{ role: "user", content: "My cat is called Mochi. Just acknowledge." }];
let r = await say(conv, { maxTokens: 60 });
conv.push({ role: "assistant", content: r.text });
conv.push({ role: "user", content: "I also have a dog named Rex. Acknowledge." });
r = await say(conv, { maxTokens: 60 });
conv.push({ role: "assistant", content: r.text });
conv.push({ role: "user", content: "What is my cat's name? One word." });
r = await say(conv, { maxTokens: 60, think: false });
check("三轮后仍记得第一轮的信息", /mochi/i.test(strip(r.text)), `回答: ${JSON.stringify(strip(r.text).slice(0, 70))}`);

// ── 2. 关闭思考是否立即生效 ─────────────────────────────────
const q = [{ role: "user", content: "What is 6 times 7? Answer with the number only." }];
const on = await say(q, { think: true, maxTokens: 150 });
const off = await say(q, { think: false, maxTokens: 150 });
check("关闭思考后不再输出思考块",
  !/<think>/.test(off.text) || /<think>\s*<\/think>/.test(off.text),
  `开: ${JSON.stringify(on.text.slice(0, 40))}\n    关: ${JSON.stringify(off.text.slice(0, 40))}`);

// ── 3. 取消之后同一会话继续,上下文还在吗 ───────────────────────
const long = [{ role: "user", content: "My secret word is PLATYPUS. Acknowledge, then write a long essay about oceans." }];
const p = say(long, { maxTokens: 400 });
setTimeout(() => void b.call("cancel_generation", {}).catch(() => {}), 300);
const cancelled = await p;
long.push({ role: "assistant", content: cancelled.text });
long.push({ role: "user", content: "What was my secret word? One word." });
r = await say(long, { maxTokens: 60, think: false });
check("取消后继续聊,先前上下文仍在",
  /platypus/i.test(strip(r.text)),
  `取消于 ${cancelled.stats?.completionTokens ?? "?"} tokens,续问答: ${JSON.stringify(strip(r.text).slice(0, 60))}`);

process.exit(0);
