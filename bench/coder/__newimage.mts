/** The round that ADDS a screenshot. Two things have to hold at once: the
 *  transcript below the new image must be reused, and the model must still
 *  read that image correctly — a resumed span whose positions are wrong would
 *  reuse beautifully and answer nonsense.
 *
 *  So the same prompt is answered twice, warm (resuming the cache built by the
 *  earlier round) and cold (fresh cache, the path that was always correct).
 *  Greedy sampling makes the two comparable token for token: if they agree,
 *  the resumed span is positioned exactly where the full pass would put it. */
import { readFileSync, writeFileSync } from "node:fs";
import { Bridge } from "../lib/bridge.mts";
import { thinkSuffix } from "../../src/lib/agentLoop";

const tiles = readFileSync(process.argv[3], "utf8").trim().split("\n");
const THINK = process.argv[4] === "think";
const MODE = process.argv[5] ?? "warm";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const i: any = await b.call("load_model", { path: process.argv[2], nCtx: 32768 });
if (!i?.loaded) { console.error("LOAD FAIL"); process.exit(1); }
const per = i.multiImage === false ? 1 : 2;
const suffix = thinkSuffix(THINK ? "normal" : "off", false, i.thinkSwitch);
console.error(`### ${(i.modelName ?? "").slice(0, 30)} 引擎=${i.backend} 每轮${per}张 思考=${THINK} ${MODE}`);

const msgs: any[] = [{ role: "system", content: "You are a coding agent looking at screenshots of a web page." }];
const strip = (r: string) => r.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/^[\s\S]*?<\/think>/, "").trim();
const norm = (r: string) => r.replace(/<\|channel>/g, "<think>").replace(/<channel\|>/g, "</think>");
/** The reasoning the app would move into its own field, for templates that
 *  read it from there — storing it inline instead renders a different turn,
 *  and a different turn is a different prefix. */
const thinkPart = (r: string) => {
  const t = norm(r); const o = t.indexOf("<think>"), c = t.indexOf("</think>");
  if (c !== -1 && (o === -1 || c < o)) return t.slice(0, c);
  return o === -1 ? "" : t.slice(o + 7, c === -1 ? undefined : c);
};
const store = (text: string) => {
  const split = i.reasoningField ? thinkPart(text).trim() : "";
  return split
    ? { role: "assistant", content: strip(norm(text)), reasoning_content: split }
    : { role: "assistant", content: text };
};

async function round(content: string, images?: string[]) {
  const m: any = { role: "user", content: content + suffix };
  if (images?.length) m.images = images;
  msgs.push(m);
  // The loop's own eviction, mirrored: a model that can hold only one picture
  // never sees two, so measuring two would measure a prompt the app never
  // sends. Before the generate, exactly where the loop does it.
  if (i.multiImage === false) {
    const withImages = msgs.filter((x: any) => x.images?.length);
    for (const x of withImages.slice(0, -1)) {
      x.images = [];
      if (!x.content.includes("[stale screenshot")) {
        x.content += "\n[stale screenshot evicted from context — retake if needed]";
      }
    }
  }
  let text = "", stats: any = null;
  await b.call("generate", {
    request: {
      messages: msgs.map((x) => ({ ...x })),
      params: { temperature: 0, topP: 1, maxTokens: 220, think: THINK },
    },
  }, (ev: any) => { if (ev.type === "token") text += ev.text; if (ev.type === "done") stats = ev.stats; });
  msgs.push(store(text));
  const pct = stats?.promptTokens ? Math.round((100 * (stats.reused ?? 0)) / stats.promptTokens) : 0;
  return { text: strip(text), pct, prompt: stats?.promptTokens ?? 0 };
}

// The earlier round exists only to leave a cache behind. In cold mode its
// turns are still in the transcript — the prompt is identical either way —
// but nothing has been evaluated when the second round starts.
const FIRST = process.argv[6] ?? "/tmp/chaty-newimage-first.txt";
const firstQ = "Here is the top of the page. One sentence.";
if (MODE === "warm") {
  const r1 = await round(firstQ, tiles.slice(0, per));
  // The reply the cold run must replay verbatim: a transcript that differs by
  // one token is a different prompt, and would measure nothing.
  writeFileSync(FIRST, JSON.stringify(msgs[msgs.length - 1]));
  console.error(`  第一轮(建缓存)  复用 ${r1.pct}%  prompt=${r1.prompt}`);
} else {
  msgs.push(
    { role: "user", content: firstQ + suffix, images: tiles.slice(0, per) },
    JSON.parse(readFileSync(FIRST, "utf8")),
  );
}

const r2 = await round(
  "Here is a NEW screenshot of a different part of the same page. List EVERY line of text you can see in it, one per line, in order, exactly as written.",
  // A model that takes one picture at a time gets the tile with text on it;
  // tile 4 alone is a blank band, which measures nothing.
  per === 1 ? tiles.slice(5, 6) : tiles.slice(4, 4 + per),
);
console.error(`  新图轮          复用 ${r2.pct}%  prompt=${r2.prompt}`);
console.log(JSON.stringify({ mode: MODE, pct: r2.pct, prompt: r2.prompt, text: r2.text }));
await b.close?.();
process.exit(0);
