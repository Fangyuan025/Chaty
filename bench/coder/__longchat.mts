/** Ten turns, watching for a turn that quietly stops behaving like the others:
 *  reasoning that disappears mid-conversation, an empty reply, a stop reason
 *  that changes. Per-model differences only show up with length. */
import { Bridge } from "../lib/bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 16384 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
const THINK = process.argv[3] !== "nothink";
console.error(`### ${(i.modelName ?? "").slice(0,32)} arch=${i.arch} think=${THINK}`);
const REASONED = /<think>|<\/think>|<\|channel>|<channel\|>/;
const qs = [
  "What is 17 times 3?", "Subtract 11 from that.", "Is the result prime?",
  "Name the next prime after it.", "And the one after that?",
  "What is their sum?", "Is that sum even?", "Halve it.",
  "Is the half a prime?", "Name one prime larger than it.",
];
const msgs: any[] = [{ role: "system", content: "You are a helpful assistant. Answer in one short sentence." }];
const rows: string[] = [];
for (const q of qs) {
  msgs.push({ role: "user", content: q });
  let text = "", st: any = null;
  await b.call("generate", { request: { messages: msgs,
    params: { temperature: 0.3, topP: 0.9, maxTokens: 1200, think: THINK } } },
    (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") st = ev.stats; });
  const reasoned = REASONED.test(text);
  const answer = text.replace(/<\|channel>[\s\S]*?<channel\|>/g, "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  rows.push(`${reasoned ? "T" : "-"}${answer ? "" : "!"}`);
  if (THINK && !reasoned) console.error(`  ⚠ 第${rows.length}轮无思考  stop=${st.stopReason} 生成=${st.completionTokens}  答:${answer.slice(0,60)}`);
  if (!answer) console.error(`  ⚠ 第${rows.length}轮空答复  stop=${st.stopReason} 生成=${st.completionTokens}`);
  if (i.reasoningField) {
    const m = /<think>([\s\S]*?)<\/think>\s*/.exec(text);
    msgs.push(m ? { role: "assistant", content: text.replace(m[0], "").trim(), reasoning_content: m[1].trim() }
                : { role: "assistant", content: text });
  } else msgs.push({ role: "assistant", content: text });
}
console.error(`  十轮: ${rows.join(" ")}   (T=有思考 -=无 !=空答复)`);
process.exit(0);
