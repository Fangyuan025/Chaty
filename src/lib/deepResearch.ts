// Deep Research: given a topic, plan search queries, run multiple rounds of web
// search interleaved with model reasoning, then synthesize a long, cited report.
// Orchestrated entirely on the frontend over the existing web tools + generate(),
// so the model only ever has to do two natural things — propose search queries
// and write prose — which any instruct model handles reliably.

import { cancelGeneration, generate, webResearch, type ChatMessage } from "./ipc";

export interface DRSource {
  n: number;
  title: string;
  url: string;
  snippet: string;
}

export type DRPhase = "planning" | "searching" | "reasoning" | "writing" | "done";

export interface DRCallbacks {
  onPhase: (phase: DRPhase, round: number, rounds: number) => void;
  onQuery: (query: string) => void;
  onSources: (sources: DRSource[]) => void;
  onReasoning: (text: string) => void;
  onReportToken: (full: string) => void;
  onDone: (report: string, sources: DRSource[]) => void;
  onError: (message: string) => void;
}

export class DRSignal {
  private _cancelled = false;
  get cancelled() {
    return this._cancelled;
  }
  cancel() {
    this._cancelled = true;
    void cancelGeneration().catch(() => {});
  }
}

export interface DROptions {
  topic: string;
  /** Number of search rounds (each round runs a few queries). */
  rounds?: number;
  lang: "zh" | "en";
  think?: boolean | null;
  thinkSwitch?: boolean;
  /** Loaded context window, to size how much source text we feed the writer. */
  nCtx?: number;
  signal: DRSignal;
}

function stripThink(s: string): string {
  return s.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, "");
}

/** Pull search queries out of a model reply (numbered/bulleted/plain lines). */
function parseQueries(raw: string, max: number): string[] {
  return stripThink(raw)
    .split(/\r?\n/)
    .map((l) => l.replace(/^\s*(?:[-*+•]|\d+[.)、]|查询|query)\s*[:：.]?\s*/i, "").trim())
    .map((l) => l.replace(/^["“'']|["”'']$/g, "").trim())
    .filter((l) => l.length >= 2 && l.length <= 200 && !/^done$/i.test(l))
    .slice(0, max);
}

/** One non-streaming completion → text. */
async function ask(messages: ChatMessage[], opts: DROptions, maxTokens: number): Promise<string> {
  let out = "";
  await generate(
    { messages, params: { temperature: 0.4, topP: 0.9, maxTokens, think: opts.think } },
    (ev) => {
      if (ev.type === "token") out += ev.text;
    },
  );
  return stripThink(out).trim();
}

const sys = (body: string): ChatMessage => ({ role: "system", content: body });

export async function deepResearch(opts: DROptions, cb: DRCallbacks): Promise<void> {
  const rounds = Math.max(1, Math.min(5, opts.rounds ?? 3));
  const suffix = opts.thinkSwitch ? "\n/no_think" : "";
  const zh = opts.lang === "zh";
  const topic = opts.topic.trim();

  const sources: DRSource[] = [];
  const byUrl = new Set<string>();
  const seenQueries = new Set<string>();
  const addSource = (title: string, url: string, snippet: string) => {
    if (!url || byUrl.has(url)) return;
    byUrl.add(url);
    sources.push({ n: sources.length + 1, title: title || url, url, snippet: snippet.slice(0, 600) });
  };

  try {
    // ---- 1. plan initial queries ----
    cb.onPhase("planning", 0, rounds);
    const planMsg: ChatMessage[] = [
      sys(
        zh
          ? "你是一名严谨的研究员。请把用户给出的主题拆解为 3-4 个高质量的网络搜索查询词，覆盖不同侧面。每行一个查询，不要编号、不要解释。"
          : "You are a rigorous researcher. Break the user's topic into 3-4 high-quality web search queries covering different angles. One query per line, no numbering, no explanation.",
      ),
      { role: "user", content: `${zh ? "主题" : "Topic"}: ${topic}${suffix}` },
    ];
    let queries = parseQueries(await ask(planMsg, opts, 300), 4);
    if (opts.signal.cancelled) return;
    if (queries.length === 0) queries = [topic];

    // ---- 2. search rounds, interleaved with reasoning ----
    for (let round = 1; round <= rounds; round++) {
      if (opts.signal.cancelled) return;
      cb.onPhase("searching", round, rounds);

      for (const q of queries) {
        if (opts.signal.cancelled) return;
        const key = q.toLowerCase();
        if (seenQueries.has(key)) continue;
        seenQueries.add(key);
        cb.onQuery(q);
        try {
          const research = await webResearch(q);
          for (const p of research.pages) addSource(p.title, p.url, p.text);
          for (const r of research.results) addSource(r.title, r.url, r.snippet);
        } catch {
          /* a failed query shouldn't abort the whole run */
        }
        cb.onSources([...sources]);
      }

      if (round >= rounds || sources.length === 0) break;

      // Reason about gaps → next-round queries (or stop early).
      cb.onPhase("reasoning", round, rounds);
      const digest = sources
        .slice(0, 40)
        .map((s) => `[${s.n}] ${s.title} — ${s.snippet.slice(0, 200)}`)
        .join("\n");
      const reasonMsg: ChatMessage[] = [
        sys(
          zh
            ? "你在做深度调研。根据已收集的资料，判断还缺哪些关键信息。若需要继续检索，给出至多 3 个新的搜索查询（每行一个，不要重复已有角度）；若资料已足够，只回复 DONE。"
            : "You are doing deep research. From the gathered material, decide what key information is still missing. If more search is needed, output up to 3 new search queries (one per line, avoid repeating covered angles); if the material is sufficient, reply with just DONE.",
        ),
        {
          role: "user",
          content: `${zh ? "主题" : "Topic"}: ${topic}\n\n${zh ? "已有资料" : "Gathered so far"}:\n${digest}${suffix}`,
        },
      ];
      const reasonOut = await ask(reasonMsg, opts, 400);
      if (opts.signal.cancelled) return;
      cb.onReasoning(reasonOut);
      if (/\bDONE\b/i.test(reasonOut) && parseQueries(reasonOut, 3).length === 0) break;
      const next = parseQueries(reasonOut, 3).filter((q) => !seenQueries.has(q.toLowerCase()));
      if (next.length === 0) break;
      queries = next;
    }

    if (opts.signal.cancelled) return;
    if (sources.length === 0) {
      cb.onError(zh ? "未能检索到相关资料。" : "No sources could be retrieved.");
      return;
    }

    // ---- 3. synthesize the report ----
    cb.onPhase("writing", rounds, rounds);
    // Budget the source text fed to the writer to the context window.
    const budgetChars = Math.max(6000, Math.min(24000, ((opts.nCtx ?? 8192) - 2400) * 3));
    let used = 0;
    const corpus: string[] = [];
    for (const s of sources) {
      const block = `[${s.n}] ${s.title}\nURL: ${s.url}\n${s.snippet}`;
      if (used + block.length > budgetChars) break;
      corpus.push(block);
      used += block.length;
    }
    const writeMsg: ChatMessage[] = [
      sys(
        zh
          ? "你是一名专业的深度报道作者。请基于下面带编号的资料，撰写一篇结构清晰、深入、客观的长篇中文报告。要求：使用 Markdown（含标题层级、要点列表）；在引用事实的句子后用 [n] 角标标注对应资料编号；包含引言、若干主体章节和结论；不要编造资料中没有的信息；不要自行编写参考文献列表（系统会自动附上）。"
          : "You are a professional deep-dive writer. Using the numbered material below, write a clear, in-depth, objective long-form report in English. Requirements: use Markdown (heading levels, bullet lists); after sentences citing a fact, add a [n] marker for the source; include an introduction, several body sections, and a conclusion; do not invent anything not in the material; do not write a references list yourself (the system appends one).",
      ),
      {
        role: "user",
        content: `${zh ? "主题" : "Topic"}: ${topic}\n\n${zh ? "资料" : "Material"}:\n${corpus.join("\n\n")}${suffix}`,
      },
    ];

    let report = "";
    await generate(
      { messages: writeMsg, params: { temperature: 0.6, topP: 0.95, maxTokens: 4096, think: opts.think } },
      (ev) => {
        if (ev.type === "token") {
          report += ev.text;
          cb.onReportToken(stripThink(report));
        }
      },
    );
    if (opts.signal.cancelled) return;

    const refsHead = zh ? "## 参考来源" : "## References";
    const refs = sources.map((s) => `${s.n}. [${s.title}](${s.url})`).join("\n");
    const full = `${stripThink(report).trim()}\n\n${refsHead}\n${refs}\n`;
    cb.onPhase("done", rounds, rounds);
    cb.onDone(full, sources);
  } catch (e) {
    if (!opts.signal.cancelled) cb.onError(e instanceof Error ? e.message : String(e));
  }
}