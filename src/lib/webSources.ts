import { textTokens } from "./ctxBudget";
import type { SearchResult, WebResearch } from "./ipc";

/** What a chat turn's web sources may take of the context window: a quarter
 *  of it (the rest is the conversation, the question and the answer), and
 *  never less than a few pages' worth. */
export function webSourceBudget(nCtx: number | null | undefined): number {
  return Math.max(2500, Math.floor((nCtx ?? 0) * 0.25));
}

/** A web search's results as numbered source blocks for the prompt, and the
 *  sources to show under the answer. No set number of them: every result on
 *  topic, in the engine's order, with its page's text where the page could be
 *  read and its snippet where not, for as long as they fit `budget` tokens. A
 *  page too long for what is left gives way to its snippet. Numbering starts
 *  at `first`; URLs in `seen` are already sources and are skipped. */
export function webSources(
  research: WebResearch,
  budget: number,
  first: number,
  seen: Set<string> = new Set(),
): { blocks: string[]; sources: SearchResult[] } {
  const blocks: string[] = [];
  const sources: SearchResult[] = [];
  const read = new Map(research.pages.map((p) => [p.url, p.text]));
  const taken = new Set(seen);
  let left = budget;
  for (const r of research.results) {
    if (taken.has(r.url)) continue;
    const page = read.get(r.url) ?? "";
    for (const body of [page.length >= 40 ? page : "", r.snippet.trim()]) {
      if (!body) continue;
      const block = `【${first + blocks.length}】 ${r.title}\n${body}`;
      const cost = textTokens(block);
      if (cost > left) continue;
      blocks.push(block);
      sources.push({ title: r.title, url: r.url, snippet: body.slice(0, 360) });
      taken.add(r.url);
      left -= cost;
      break;
    }
  }
  return { blocks, sources };
}
