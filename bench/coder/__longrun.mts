/** A long conversation the way chat mode actually runs one: answers long enough
 *  to fill the window, reasoning kept in history where the template reads it
 *  from its own field, and the same compaction chat mode performs once the
 *  prompt passes 85% of the budget. Reports per-turn anomalies rather than a
 *  score — the point is to see WHICH turn stops behaving, per model. */
import { Bridge } from "../lib/bridge.mts";
import { calibrate, contextLimit, messageTokens, fitTranscript } from "../../src/lib/ctxBudget";

const path = process.argv[2];
const NCTX = Number(process.argv[3] ?? 8192);
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path, nCtx: NCTX });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
const name = (i.modelName ?? path.split("/").pop()).slice(0, 36);
console.error(`### ${name}  arch=${i.arch} nCtx=${i.nCtx} switch=${i.thinkSwitch} reasonField=${!!i.reasoningField}`);

const REASONED = /<think>|<\/think>/;
const norm = (s: string) => s.replace(/<\|channel>/g, "<think>").replace(/<channel\|>/g, "</think>");
const thinkPart = (r: string) => { const s = norm(r); const o = s.indexOf("<think>"), c = s.indexOf("</think>");
  if (c !== -1 && (o === -1 || c < o)) return s.slice(0, c);
  return o === -1 ? "" : s.slice(o + 7, c === -1 ? undefined : c); };
const answerPart = (r: string) => { let s = norm(r); const o = s.indexOf("<think>"), c = s.indexOf("</think>");
  if (c !== -1 && (o === -1 || c < o)) s = s.slice(c + 8);
  return s.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<think>[\s\S]*$/, "").replace(/<\/?think>/g, "").trim(); };

const noThink = i.supportsThinking && !i.thinkSwitch ? false : undefined;
async function gen(messages: any[], think: boolean | undefined, maxTokens: number) {
  let text = "", stats: any = null, err = "";
  try {
    await b.call("generate", { request: { messages, params: { temperature: 0.4, topP: 0.9, maxTokens, think } } },
      (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") stats = ev.stats; });
  } catch (e: any) { err = String(e?.message ?? e).slice(0, 160); }
  return { text, stats, err };
}

let memo: { summary: string; covered: number; print: string } | null = null;
/** What the engine charged for the last prompt — chat mode calibrates its
 *  estimator against this every turn, and refuses to let a standing summary
 *  keep growing past it. A harness that skips both measures a budget the app
 *  does not use. */
let charged = 0;
const printOf = (msgs: any[], upto: number) =>
  `${upto}:${msgs.slice(0, upto).reduce((n: number, m: any) => n + m.content.length, 0)}`;

/** Chat mode's composeContext, same thresholds. */
async function compose(msgs: any[]) {
  if (msgs.length < 6) return null;
  const budget = Math.max(1024, contextLimit(NCTX, 2048));
  if (charged > budget) memo = null;
  if (memo) {
    if (memo.covered <= msgs.length && printOf(msgs, memo.covered) === memo.print) {
      const tail = msgs.slice(memo.covered);
      const summary = "【前情摘要】" + memo.summary;
      if (messageTokens([{ content: summary }]) + messageTokens(tail) <= budget * 0.85)
        return { summary: memo.summary, tail, kept: tail.length, headLen: memo.covered, reused: true };
    } else memo = null;
  }
  if (messageTokens(msgs) <= budget * 0.85) return null;
  const tailCap = budget * 0.4;
  let acc = 0, keep = 0;
  for (let k = msgs.length - 1; k >= 0; k--) { acc += messageTokens([msgs[k]]); if (acc > tailCap && keep >= 2) break; keep++; }
  const splitAt = Math.max(1, msgs.length - keep);
  const head = msgs.slice(0, splitAt), tail = msgs.slice(splitAt);
  if (!head.length) return null;
  const transcript = fitTranscript(head.map((m: any) => `${m.role === "user" ? "用户" : "助手"}: ${m.content}`),
    Math.max(1500, Math.floor(budget * 0.6)), "zh");
  const r = await gen([{ role: "system", content: "请把下面这段较早的对话压缩成简洁的要点摘要。只输出摘要正文，不要解释或思考过程。" },
    { role: "user", content: transcript + (i.thinkSwitch ? "\n/no_think" : "") }], noThink, 400);
  const summary = answerPart(r.text).trim();
  if (!summary) return null;
  memo = { summary, covered: splitAt, print: printOf(msgs, splitAt) };
  return { summary, tail, kept: keep, headLen: head.length, reused: false };
}

// Questions that pull LONG answers — the window has to actually fill up.
const QS = [
  "Explain how a hash map handles collisions. Give the two main strategies with their trade-offs.",
  "Now walk through what happens when the load factor is exceeded.",
  "Compare that with how a B-tree stays balanced.",
  "Which of the two would you pick for an on-disk index, and why?",
  "What changes if the keys are highly skewed — say 90% of lookups hit 10 keys?",
  "Write a short Python function implementing open addressing with linear probing.",
  "What is the worst case of that function, and how would you fix it?",
  "Explain the same fix in terms of the memory hierarchy.",
  "Summarize everything you have told me so far as a bulleted list.",
  "Finally: name the single most important trade-off in this whole discussion.",
];

const sys = { role: "system", content: "You are a helpful assistant. Answer thoroughly, with examples." };
let ui: any[] = [];             // full history, like the UI keeps it
const flags: string[] = [];
for (let n = 0; n < QS.length; n++) {
  ui.push({ role: "user", content: QS[n] });
  // The app's historyForModel: split reasoning into its own field where the template reads it there.
  const hist = ui.map((m: any) => {
    if (m.role !== "assistant") return { role: m.role, content: m.content };
    const r = i.reasoningField ? thinkPart(m.content).trim() : "";
    return r ? { role: m.role, content: answerPart(m.content), reasoning_content: r } : { role: m.role, content: m.content };
  });
  let comp: any = null;
  if (process.argv[4] !== "nocompact") { try { comp = await compose(hist); } catch { /* the app swallows this too */ } }
  const sent = [{ role: "system", content: sys.content + (comp ? "\n\n【前情摘要】" + comp.summary : "") },
                ...(comp ? comp.tail : hist)];
  const sentRaw = messageTokens(sent);
  const r = await gen(sent, true, 2048);
  if (r.stats?.promptTokens) { calibrate(sentRaw, r.stats.promptTokens); charged = r.stats.promptTokens; }
  const reasoned = REASONED.test(norm(r.text));
  const ans = answerPart(r.text);
  const mark = r.err ? "E" : !ans ? "!" : reasoned ? "T" : "-";
  flags.push(mark);
  const tag = comp ? ` [压缩 头${comp.headLen}→摘要, 留${comp.kept}${comp.reused ? ", 沿用" : ", 新写"}]` : "";
  if (r.err) console.error(`  ⚠ 第${n + 1}轮 报错: ${r.err}${tag}`);
  else if (!ans) console.error(`  ⚠ 第${n + 1}轮 空答复 stop=${r.stats?.stopReason} 生成=${r.stats?.completionTokens} 思考=${reasoned}${tag}`);
  else if (!reasoned) console.error(`  ⚠ 第${n + 1}轮 无思考 stop=${r.stats?.stopReason} 生成=${r.stats?.completionTokens}${tag} 答:${ans.replace(/\s+/g," ").slice(0,60)}`);
  else console.error(`  · 第${n + 1}轮${tag} prompt=${r.stats?.promptTokens} 复用=${r.stats?.reused ?? "?"} (${Math.round(100 * (r.stats?.reused ?? 0) / Math.max(1, r.stats?.promptTokens ?? 1))}%)`);
  ui.push({ role: "assistant", content: r.text });
}
console.error(`  十轮: ${flags.join(" ")}   (T=有思考 -=无思考 !=空答复 E=报错)`);
process.exit(0);
