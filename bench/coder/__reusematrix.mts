/** KV reuse across the shapes a real code-mode turn actually takes.
 *
 *  Faithful to the agent loop, because a harness that is not proves nothing:
 *  tool results go in the TOOL role where the model has one (the app's
 *  `pushUser` does), the thinking rung rides on every user-role turn
 *  (`thinkSuffix`), an assistant turn is stored raw with its tool-call markup,
 *  and reasoning is split into its own field for templates that read it there —
 *  each of which changes how a turn renders, which is exactly what prefix
 *  reuse is made of.
 *
 *  What is NOT faithful, deliberately: the system message is one line, where
 *  the app's carries the whole tool manual. That moves the percentages (a
 *  bigger fixed head makes every round look better) but not the structure
 *  being measured, which is whether a round is a pure append onto the last. */
import { readFileSync } from "node:fs";
import { Bridge } from "../lib/bridge.mts";
import { thinkSuffix } from "../../src/lib/agentLoop";

const allTiles = readFileSync(process.argv[3], "utf8").trim().split("\n");
const THINK = process.argv[4] === "think";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 32768 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
const name = (i.modelName ?? process.argv[2].split("/").pop()).slice(0, 30);
console.error(
  `### ${name}  引擎=${i.backend ?? "?"} 多图=${i.multiImage} toolRole=${i.toolRole} reasonField=${i.reasoningField} 思考=${THINK}`,
);

// The loop sends ONE tile to a model that cannot take several — sending more
// fails the round outright, which is not what the app does and would measure
// nothing.
const per = i.multiImage === false ? 1 : 3;
const tiles = allTiles;
const suffix = thinkSuffix(THINK ? "normal" : "off", false, i.thinkSwitch);
const norm = (s: string) => s.replace(/<\|channel>/g, "<think>").replace(/<channel\|>/g, "</think>");
const thinkPart = (r: string) => {
  const s = norm(r); const o = s.indexOf("<think>"), c = s.indexOf("</think>");
  if (c !== -1 && (o === -1 || c < o)) return s.slice(0, c);
  return o === -1 ? "" : s.slice(o + 7, c === -1 ? undefined : c);
};
const stripThink = (r: string) => {
  let s = norm(r); const o = s.indexOf("<think>"), c = s.indexOf("</think>");
  if (c !== -1 && (o === -1 || c < o)) s = s.slice(c + 8);
  return s.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<think>[\s\S]*$/, "").trim();
};

const msgs: any[] = [{ role: "system", content: "You are a coding agent. Answer in one short sentence." }];
const rows: { label: string; pct: number; prompt: number; reused: number }[] = [];

/** The app's pushUser: a tool result takes the tool role where there is one. */
function pushUser(content: string, images?: string[]) {
  const isResult = content.trimStart().startsWith("<tool_result");
  const role = isResult && i.toolRole ? "tool" : "user";
  const m: any = { role, content: content + suffix };
  if (images?.length) m.images = images;
  msgs.push(m);
}

async function round(label: string, content: string, images?: string[]) {
  pushUser(content, images);
  // Evicted BEFORE the step runs, exactly where the loop does it — and on the
  // same condition: only a model that can hold ONE picture drops the old one
  // now that both engines resume across a new image. Doing it after the
  // generate measured a prompt the app never sends.
  if (i.multiImage === false) {
    const withImages = msgs.filter((m: any) => m.images?.length);
    for (const m of withImages.slice(0, Math.max(0, withImages.length - 1))) {
      m.images = [];
      if (!m.content.includes("[stale screenshot")) {
        m.content += "\n[stale screenshot evicted from context — retake if needed]";
      }
    }
  }
  let text = "", st: any = null;
  await b.call("generate", {
    request: { messages: msgs, params: { temperature: 0.3, topP: 0.9, maxTokens: 96, think: THINK } },
  }, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") st = ev.stats; });
  const pct = Math.round((100 * (st?.reused ?? 0)) / Math.max(1, st?.promptTokens ?? 1));
  rows.push({ label, pct, prompt: st?.promptTokens ?? 0, reused: st?.reused ?? 0 });
  // Stored the way the loop stores it: raw turn, reasoning split out only where
  // the template reads it from its own field.
  const split = i.reasoningField ? thinkPart(text).trim() : "";
  msgs.push(
    split
      ? { role: "assistant", content: stripThink(text), reasoning_content: split }
      : { role: "assistant", content: text },
  );
  return pct;
}

await round("1 冷启动", "List the files you would look at first.");
await round("2 文本", '<tool_result name="list_dir">index.html styles.css app.js</tool_result>');
await round("3 截图(首次带图)", '<tool_result name="browser_screenshot">Captured the page.</tool_result>', tiles.slice(0, per));
await round("4 文本(图仍在)", '<tool_result name="bash">exit 0</tool_result>');
await round("5 文本", '<tool_result name="read_file">const a = 1;</tool_result>');
await round("6 截图(换新图)", '<tool_result name="browser_screenshot">Captured again.</tool_result>', tiles.slice(3, 3 + per));
await round("7 文本(新图仍在)", '<tool_result name="bash">exit 0</tool_result>');

for (const r of rows) {
  console.error(`  ${r.label.padEnd(20)} prompt=${String(r.prompt).padStart(6)}  复用 ${String(r.pct).padStart(3)}%`);
}
// Round 1 is cold and round 3 is the first pixels, which nothing can resume.
// Round 2 is the text baseline: its prompt is tiny, so its percentage is
// dominated by the fixed system block and says little. Everything after pixels
// entered must be an append — including round 6, which adds a SECOND
// screenshot: the transcript beneath it is still in the cache and only the new
// picture needs evaluating.
//
// Except on a model that can hold one picture at a time. There the old
// screenshot has to be evicted from the transcript, which rewrites it at that
// point, so round 6 resumes only as far as the picture it replaced — a limit
// of the model, not of the cache.
const bad = rows.filter((r, k) => [3, 4, 6].includes(k) && r.pct < 60);
// Round 6 is judged in tokens, not percent: it carries a brand-new picture
// that nobody can reuse, so the share of the prompt it reuses is set by how
// big that picture is next to the transcript. What must hold is that
// EVERYTHING BELOW the picture came from the cache — the round-5 prompt and
// the reply that followed it.
if (i.multiImage !== false && rows.length > 5 && rows[5].reused < rows[4].prompt * 0.85) {
  bad.push(rows[5]);
}
console.error(
  `  新图轮 复用 ${rows[5]?.reused ?? 0} tok / 上一轮 prompt ${rows[4]?.prompt ?? 0} tok`,
);
console.error(`  判定: ${bad.length === 0 ? "✓ 应复用的轮次全部复用" : "❌ 未复用: " + bad.map((r) => r.label).join(", ")}`);
process.exit(0);
