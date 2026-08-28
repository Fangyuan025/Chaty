/** Several pictures of DIFFERENT sizes, added one turn at a time — what
 *  `view_image` does when a model walks a folder of screenshots.
 *
 *  Equal-sized tiles cut from one page cannot catch a per-picture pixel
 *  mix-up: every image's placeholder run is the same length, so the wrong
 *  one still fits. These are the owner's own screenshots, no two alike. */
import { readFileSync } from "node:fs";
import { Bridge } from "../lib/bridge.mts";

const shots = readFileSync(process.argv[3], "utf8").trim().split("\n");
const THINK = process.argv[4] === "think";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 32768 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
console.error(
  `### ${(i.modelName ?? "").slice(0, 30)} 引擎=${i.backend} 多图=${i.multiImage}`
  + ` toolRole=${i.toolRole} reasonField=${i.reasoningField} 思考=${THINK}`,
);

// A tool result never opens a conversation — in the app it answers something
// the user asked, and a template that looks for the user's question says so.
// The shape the failing session had: tool results on the TOOL role where the
// model has one, THEN the pictures. That order is the point — a tool→assistant
// boundary sitting before every picture is what the block search can land on.
const msgs: any[] = [
  { role: "system", content: "You are a coding agent. One short sentence per answer." },
  { role: "user", content: "Look at every screenshot in this workspace and tell me what each shows." },
  { role: "assistant", content: '<tool_call>{"name":"understand_repo","arguments":{}}</tool_call>' },
  {
    role: i.toolRole ? "tool" : "user",
    content: '<tool_result name="understand_repo">A French course folder: four PDFs and four screenshots.</tool_result>',
  },
  { role: "assistant", content: '<tool_call>{"name":"ask_user","arguments":{"question":"Shall I read the screenshots?"}}</tool_call>' },
  {
    role: i.toolRole ? "tool" : "user",
    content: '<tool_result name="ask_user">Yes, read them all.</tool_result>',
  },
];
let bad = 0;
for (let k = 0; k < shots.length; k++) {
  // `view_image` pushes the picture on the USER role even where the model has
  // a tool role — it does not go through the loop's pushUser at all.
  const m: any = {
    role: "user",
    content:
      `<tool_result name="view_image">Loaded image ${shots[k].split("/").pop()};`
      + ` its contents are below — look and continue.</tool_result>`,
    images: [shots[k]],
  };
  // The loop keeps every live picture on an engine that resumes across a new
  // one; only a model that takes a single image per prompt swaps.
  if (i.multiImage === false) {
    for (const x of msgs.filter((y: any) => y.images?.length)) x.images = [];
  }
  msgs.push(m);
  let text = "", st: any = null, err = "";
  try {
    await b.call("generate", {
      request: { messages: msgs, params: { temperature: 0.3, topP: 0.9, maxTokens: 48, think: THINK } },
    }, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") st = ev.stats; });
  } catch (e: any) { err = String(e?.message ?? e).slice(0, 90); bad++; }
  // Stored the way the loop stores it: verbatim, except where the template
  // reads thinking from its own field, which the loop splits out — leaving it
  // inline renders a different turn, and a different turn is a different
  // prefix.
  const norm = text.replace(/<\|channel>/g, "<think>").replace(/<channel\|>/g, "</think>");
  const o = norm.indexOf("<think>"), c = norm.indexOf("</think>");
  const reasoning = i.reasoningField
    ? (c !== -1 && (o === -1 || c < o) ? norm.slice(0, c) : o === -1 ? "" : norm.slice(o + 7, c === -1 ? undefined : c)).trim()
    : "";
  // The loop's stripThink, including its last clause: a thought the token cap
  // cut off has no closer, and leaving it in `content` stores the reasoning
  // twice — which renders a turn the model never produced.
  let answer = norm;
  if (c !== -1 && (o === -1 || c < o)) answer = answer.slice(c + 8);
  answer = answer
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<think>[\s\S]*$/, "")
    .trim();
  msgs.push(
    reasoning
      ? { role: "assistant", content: answer, reasoning_content: reasoning }
      : { role: "assistant", content: text },
  );
  const pct = st?.promptTokens ? Math.round((100 * (st.reused ?? 0)) / st.promptTokens) : 0;
  console.error(
    `  第${k + 1}张 prompt=${String(st?.promptTokens ?? 0).padStart(6)} 复用 ${String(pct).padStart(3)}%` +
    (err ? `  ❌ ${err}` : `  ${text.replace(/\s+/g, " ").slice(0, 40)}`),
  );
}
console.error(`  判定: ${bad === 0 ? "✓ 全部读入" : `❌ ${bad} 张失败`}`);
process.exit(0);
