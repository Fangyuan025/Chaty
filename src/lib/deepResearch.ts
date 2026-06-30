// Deep Research: given a topic, plan search queries, run multiple rounds of web
// search interleaved with model reasoning, then synthesize a long, cited report.
// Orchestrated entirely on the frontend over the existing web tools + generate(),
// so the model only ever has to do two natural things — propose search queries
// and write prose — which any instruct model handles reliably.

import {
  cancelGeneration,
  generate,
  ragCorpusDocs,
  ragListDocuments,
  webResearch,
  type ChatMessage,
} from "./ipc";

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
async function ask(
  messages: ChatMessage[],
  opts: { think?: boolean | null },
  maxTokens: number,
): Promise<string> {
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

/** Shared writer for both web Deep Research and the knowledge-base report:
 *  feed the numbered sources (budgeted to the context window) to the model, then
 *  renumber [n] markers to a contiguous, cited-only reference list. `kb` switches
 *  the grounding wording, drops the URL line, and renders refs as plain titles
 *  (knowledge-base sources have no URL). */
async function writeReport(
  sources: DRSource[],
  opts: { lang: "zh" | "en"; think?: boolean | null; thinkSwitch?: boolean; nCtx?: number; signal: DRSignal },
  cb: DRCallbacks,
  phaseTotal: number,
  kind: { kb: boolean; topic?: string; tree?: string },
): Promise<void> {
  const zh = opts.lang === "zh";
  const suffix = opts.thinkSwitch ? "\n/no_think" : "";
  const kb = kind.kb;
  const topic = kind.topic?.trim() ?? "";
  // No topic on a KB report → NotebookLM-style overview of the whole knowledge base.
  const overview = kb && !topic;

  cb.onPhase("writing", phaseTotal, phaseTotal);
  // Budget the source text fed to the writer to the context window.
  const budgetChars = Math.max(6000, Math.min(24000, ((opts.nCtx ?? 8192) - 2400) * 3));
  let used = 0;
  const corpus: string[] = [];
  for (const s of sources) {
    const block = kb
      ? `[${s.n}] ${s.title}\n${s.snippet}`
      : `[${s.n}] ${s.title}\nURL: ${s.url}\n${s.snippet}`;
    if (used + block.length > budgetChars) break;
    corpus.push(block);
    used += block.length;
  }

  const sysPrompt = overview
    ? zh
      ? `你是一名严谨的分析师。下面是来自用户本地知识库的带编号资料（每条对应一个文件）。请仅依据这些资料，生成一篇结构清晰、客观的中文综述报告，帮助读者快速把握整个知识库的内容。要求：以一个能概括主题的一级标题（# 标题）开头；使用 Markdown（标题层级、要点列表）；涵盖整体概览、关键主题/要点、文件之间的关联，以及一个简短结论；在引用事实的句子后用 [n] 角标标注对应文件编号；若提供了文件结构，可据此说明项目/资料的组织方式；严格基于资料、绝不编造；不要自行编写参考来源列表（系统会自动附上）。`
      : `You are a rigorous analyst. Below is numbered material from the user's local knowledge base (each entry is one file). Using ONLY this material, generate a clear, objective overview report that helps a reader quickly grasp the whole knowledge base. Requirements: begin with a single top-level title heading (# Title) that captures the subject; use Markdown (heading levels, bullet lists); cover an overall summary, the key themes/points, how the files relate, and a short conclusion; after sentences citing a fact, add a [n] marker for the file; if a file structure is provided, use it to explain how the project/material is organized; stay strictly grounded and never invent; do not write a references list yourself (the system appends one).`
    : kb
      ? zh
        ? `你是一名严谨的分析师。请仅依据下面这份来自用户本地知识库的带编号资料，围绕主题《${topic}》撰写一篇结构清晰、客观的中文报告。要求：使用 Markdown（含标题层级、要点列表）；在引用事实的句子后用 [n] 角标标注对应资料编号；包含引言、若干主体章节和结论；严格基于资料、绝不编造资料中没有的信息；若资料不足以支撑该主题，请如实说明；不要自行编写参考来源列表（系统会自动附上）。`
        : `You are a rigorous analyst. Using ONLY the numbered material below — drawn from the user's local knowledge base — write a clear, objective English report on the topic "${topic}". Requirements: Markdown (heading levels, bullet lists); after sentences citing a fact, add a [n] marker; include an introduction, several body sections, and a conclusion; stay strictly grounded in the material and never invent anything not present; if the material is insufficient to support the topic, say so honestly; do not write a references list yourself (the system appends one).`
      : zh
        ? `你是一名专业的深度报道作者。请围绕主题《${topic}》，基于下面带编号的资料撰写一篇结构清晰、深入、客观的长篇中文报告。要求：使用 Markdown（含标题层级、要点列表）；在引用事实的句子后用 [n] 角标标注对应资料编号；包含引言、若干主体章节和结论；不要编造资料中没有的信息；不要自行编写参考文献列表（系统会自动附上）。重要：若某条资料与主题无关，请直接忽略它，绝不要把无关内容硬塞进报告或牵强地与主题关联；若与主题相关的资料严重不足，请如实说明。`
        : `You are a professional deep-dive writer. Write a clear, in-depth, objective long-form English report ON THE TOPIC "${topic}", using the numbered material below. Requirements: Markdown (heading levels, bullet lists); after sentences citing a fact, add a [n] marker; include an introduction, several body sections, and a conclusion; do not invent anything not in the material; do not write a references list yourself (the system appends one). IMPORTANT: ignore any source that is not relevant to the topic — never force unrelated material into the report or contrive a connection to the topic; if there is little relevant material, say so honestly.`;

  const treeBlock = kind.tree
    ? `${zh ? "文件结构" : "File structure"}:\n${kind.tree}\n\n`
    : "";
  const userContent = overview
    ? `${treeBlock}${zh ? "资料" : "Material"}:\n${corpus.join("\n\n")}${suffix}`
    : `${zh ? "主题" : "Topic"}: ${topic}\n\n${treeBlock}${zh ? "资料" : "Material"}:\n${corpus.join("\n\n")}${suffix}`;
  const writeMsg: ChatMessage[] = [sys(sysPrompt), { role: "user", content: userContent }];

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

  const clean = stripThink(report).trim();
  // Only list sources the report actually cited ([n] / 【n】) — the model is told
  // to ignore irrelevant material, so uncited sources must not pollute the
  // references. Fall back to the first few collected sources if nothing was
  // cited. Renumber so the list is contiguous and the markers still line up.
  const cited = new Set<number>();
  for (const m of clean.matchAll(/[[【](\d{1,3})[\]】]/g)) {
    const n = parseInt(m[1], 10);
    if (n >= 1 && n <= sources.length) cited.add(n);
  }
  const refSources = cited.size > 0 ? sources.filter((s) => cited.has(s.n)) : sources.slice(0, 8);
  const remap = new Map(refSources.map((s, i) => [s.n, i + 1]));
  const renumbered = clean.replace(/([[【])(\d{1,3})([\]】])/g, (whole, _l, d) => {
    const nn = remap.get(parseInt(d, 10));
    return nn ? `[${nn}]` : whole;
  });
  const refsHead = zh ? "## 参考来源" : "## References";
  const refs = refSources
    .map((s, i) => (kb || !s.url ? `${i + 1}. ${s.title}` : `${i + 1}. [${s.title}](${s.url})`))
    .join("\n");
  const full = `${renumbered}\n\n${refsHead}\n${refs}\n`;
  cb.onPhase("done", phaseTotal, phaseTotal);
  cb.onDone(full, refSources);
}

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
          ? "你是一名严谨的研究员。请把用户给出的主题拆解为 3-4 个高质量的网络搜索查询词，覆盖不同侧面。每个查询都必须紧扣该主题本身，不要偏离到无关话题。每行一个查询，不要编号、不要解释。"
          : "You are a rigorous researcher. Break the user's topic into 3-4 high-quality web search queries covering different angles. Every query MUST stay strictly on the given topic — do not drift to unrelated subjects. One query per line, no numbering, no explanation.",
      ),
      { role: "user", content: `${zh ? "主题" : "Topic"}: ${topic}${suffix}` },
    ];
    const modelQueries = parseQueries(await ask(planMsg, opts, 300), 4);
    if (opts.signal.cancelled) return;
    // ALWAYS search the verbatim topic first. Some models (esp. uncensored
    // finetunes on sensitive subjects) derail and propose unrelated queries;
    // anchoring on the topic keeps the sources on-topic regardless.
    const tl = topic.toLowerCase();
    let queries = [topic, ...modelQueries.filter((q) => q.toLowerCase() !== tl)].slice(0, 4);

    // ---- 2. search rounds, interleaved with reasoning ----
    for (let round = 1; round <= rounds; round++) {
      if (opts.signal.cancelled) return;
      cb.onPhase("searching", round, rounds);

      let first = true;
      for (const q of queries) {
        if (opts.signal.cancelled) return;
        const key = q.toLowerCase();
        if (seenQueries.has(key)) continue;
        seenQueries.add(key);
        // Space out requests a little so the search providers don't rate-limit
        // us mid-run (which forced fragile fallbacks before).
        if (!first) await new Promise((r) => setTimeout(r, 350));
        first = false;
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
    await writeReport(sources, opts, cb, rounds, { kb: false, topic });
  } catch (e) {
    if (!opts.signal.cancelled) cb.onError(e instanceof Error ? e.message : String(e));
  }
}

export interface KBReportOptions {
  lang: "zh" | "en";
  think?: boolean | null;
  thinkSwitch?: boolean;
  /** Loaded context window, to size how much source text we feed the writer. */
  nCtx?: number;
  signal: DRSignal;
}

/** Render an indented file tree from relative-path document names (folder import
 *  stores e.g. `myproject/src/lib/ipc.ts`), so the writer can describe how the
 *  project/material is organized. Returns "" when names carry no structure. */
function buildFileTree(names: string[]): string {
  interface Node {
    children: Map<string, Node>;
    file: boolean;
  }
  const root: Node = { children: new Map(), file: false };
  for (const name of names) {
    const parts = name.split("/").filter(Boolean);
    let cur = root;
    parts.forEach((p, i) => {
      let next = cur.children.get(p);
      if (!next) {
        next = { children: new Map(), file: false };
        cur.children.set(p, next);
      }
      if (i === parts.length - 1) next.file = true;
      cur = next;
    });
  }
  const lines: string[] = [];
  const walk = (node: Node, depth: number) => {
    const entries = [...node.children.entries()].sort((a, b) => {
      const ad = a[1].children.size > 0;
      const bd = b[1].children.size > 0;
      if (ad !== bd) return ad ? -1 : 1; // directories first
      return a[0].localeCompare(b[0]);
    });
    for (const [seg, child] of entries) {
      const isDir = child.children.size > 0;
      lines.push(`${"  ".repeat(depth)}${isDir ? `${seg}/` : seg}`);
      if (isDir) walk(child, depth + 1);
    }
  };
  walk(root, 0);
  return lines.join("\n").slice(0, 2000);
}

/** Knowledge-base report (NotebookLM-style): no topic needed — clicking generate
 *  immediately synthesizes a cited overview of the whole knowledge base. Reuses
 *  the Deep Research writer; grounds on per-file document text (one citation per
 *  file) plus the folder structure, entirely offline. */
export async function knowledgeReport(opts: KBReportOptions, cb: DRCallbacks): Promise<void> {
  const zh = opts.lang === "zh";
  try {
    // ---- 1. read the knowledge base (file tree for structure context) ----
    cb.onPhase("planning", 0, 1);
    let tree = "";
    try {
      const docs = await ragListDocuments();
      const names = docs.filter((d) => d.enabled).map((d) => d.name);
      if (names.some((n) => n.includes("/"))) tree = buildFileTree(names);
    } catch {
      /* tree is optional context */
    }
    if (opts.signal.cancelled) return;

    // ---- 2. gather per-file content (fair budget per document) ----
    cb.onPhase("searching", 1, 1);
    const budget = Math.max(8000, Math.min(40000, ((opts.nCtx ?? 8192) - 2400) * 3));
    let docTexts;
    try {
      docTexts = await ragCorpusDocs(budget);
    } catch (e) {
      cb.onError(e instanceof Error ? e.message : String(e));
      return;
    }
    if (opts.signal.cancelled) return;
    const sources: DRSource[] = docTexts
      .filter((d) => d.text.trim())
      .map((d, i) => ({ n: i + 1, title: d.name, url: "", snippet: d.text }));
    if (sources.length === 0) {
      cb.onError(zh ? "知识库为空。" : "The knowledge base is empty.");
      return;
    }
    cb.onSources([...sources]);

    // ---- 3. synthesize the overview report (shared writer) ----
    await writeReport(sources, opts, cb, 1, { kb: true, tree });
  } catch (e) {
    if (!opts.signal.cancelled) cb.onError(e instanceof Error ? e.message : String(e));
  }
}