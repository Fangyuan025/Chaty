import { useEffect, useRef, useState } from "react";
import { agentLang, useI18n } from "../lib/i18n";
import { exportTextFile, openHtmlReport, type ModelInfo } from "../lib/ipc";
import { Markdown } from "./Markdown";
import { Select } from "./Select";
import { IconResearch, IconSearch, IconDownload, IconPlay, IconStop } from "./icons";
import {
  deepResearch,
  knowledgeReport,
  DRSignal,
  type DRPhase,
  type DROptions,
  type DRSource,
  type KBReportOptions,
} from "../lib/deepResearch";

const PHASE_KEY = {
  planning: "drPhasePlanning",
  searching: "drPhaseSearching",
  reasoning: "drPhaseReasoning",
  writing: "drPhaseWriting",
  done: "drPhaseDone",
} as const satisfies Record<DRPhase, string>;

// KB report reads local documents — never the web — so it gets its own phase
// wording (no "searching the web", no search-round counters).
const KB_PHASE_KEY = {
  planning: "kbReportPhasePlan",
  searching: "kbReportPhaseRead",
  reasoning: "drPhaseReasoning",
  writing: "drPhaseWriting",
  done: "drPhaseDone",
} as const satisfies Record<DRPhase, string>;

/** Deep Research: multi-round web search + reasoning → a long cited report,
 *  exportable to PDF / Markdown. In `kb` mode the same panel instead grounds the
 *  report in the local knowledge base (retrieval over indexed docs, no web). */
export function DeepResearchPanel({
  model,
  onClose,
  onLockChange,
  mode = "web",
}: {
  model: ModelInfo | null;
  onClose: () => void;
  onLockChange: (locked: boolean) => void;
  mode?: "web" | "kb";
}) {
  const { t, lang } = useI18n();
  const kb = mode === "kb";
  const [topic, setTopic] = useState("");
  const [depth, setDepth] = useState(3);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<{ phase: DRPhase; round: number; rounds: number } | null>(null);
  const [queries, setQueries] = useState<string[]>([]);
  const [sources, setSources] = useState<DRSource[]>([]);
  const [reasoning, setReasoning] = useState("");
  const [report, setReport] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");

  const signalRef = useRef<DRSignal | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const printRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    onLockChange(running);
    return () => onLockChange(false);
  }, [running, onLockChange]);
  useEffect(() => () => signalRef.current?.cancel(), []);
  useEffect(() => {
    if (running) bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
  }, [report, sources, reasoning, phase, running]);

  // KB report is NotebookLM-style: generate immediately on open, no topic prompt.
  const startedRef = useRef(false);
  useEffect(() => {
    if (kb && !startedRef.current) {
      startedRef.current = true;
      if (model) void start();
      else setError(t("kbReportNeedModel"));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function start() {
    // The KB report is topic-free (NotebookLM-style overview); web Deep Research
    // needs a topic.
    if (!model || running || (!kb && !topic.trim())) return;
    setQueries([]);
    setSources([]);
    setReasoning("");
    setReport("");
    setDone(false);
    setError("");
    setRunning(true);
    const signal = new DRSignal();
    signalRef.current = signal;

    const think = model.supportsThinking && !model.thinkSwitch ? false : undefined;
    const callbacks = {
      onPhase: (phase: DRPhase, round: number, rounds: number) => setPhase({ phase, round, rounds }),
      onQuery: (q: string) => setQueries((cur) => [...cur, q]),
      onSources: (s: DRSource[]) => setSources(s),
      onReasoning: (r: string) => setReasoning(r),
      onReportToken: (full: string) => setReport(full),
      onDone: (full: string) => {
        setReport(full);
        setDone(true);
        setRunning(false);
        setPhase(null);
      },
      onError: (msg: string) => {
        setError(msg);
        setRunning(false);
        setPhase(null);
      },
    };

    if (kb) {
      const opts: KBReportOptions = {
        lang: agentLang(lang),
        think,
        thinkSwitch: model.thinkSwitch,
        nCtx: model.nCtx ?? undefined,
        signal,
      };
      await knowledgeReport(opts, callbacks);
    } else {
      const opts: DROptions = {
        topic: topic.trim(),
        rounds: depth,
        lang: agentLang(lang),
        think,
        thinkSwitch: model.thinkSwitch,
        nCtx: model.nCtx ?? undefined,
        signal,
      };
      await deepResearch(opts, callbacks);
    }
    setRunning(false);
  }

  function stop() {
    signalRef.current?.cancel();
    setRunning(false);
    setPhase(null);
  }

  async function exportMd() {
    const safe =
      (topic.trim() || (kb ? t("kbReportTitle") : "")).slice(0, 40).replace(/[/\\:*?"<>|]/g, "_") ||
      (kb ? "knowledge-base-report" : "deep-research");
    // Don't add a topic title if the report already opens with its own heading
    // (otherwise the title shows up twice).
    const md = /^\s*#{1,2}\s/.test(report) ? report : `# ${topic.trim()}\n\n${report}`;
    try {
      await exportTextFile(`${safe}.md`, md, "md");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function exportPdf() {
    // WKWebView can't print itself, so render the report to a styled HTML file
    // and open it in the system browser, which auto-fires the print dialog
    // ("Save as PDF"). System fonts handle CJK, so Chinese reports render fine.
    const node = printRef.current;
    if (!node) return;
    const css = `
      @page { margin: 18mm 16mm; }
      body { font: 15px/1.75 -apple-system, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif; color: #111; max-width: 760px; margin: 0 auto; padding: 24px; }
      h1 { font-size: 24px; } h2 { font-size: 19px; margin-top: 1.5em; } h3 { font-size: 16px; }
      a { color: #1155cc; word-break: break-all; }
      pre, code { font-family: ui-monospace, Menlo, monospace; background: #f4f4f5; }
      pre { padding: 10px; border-radius: 6px; white-space: pre-wrap; overflow-wrap: anywhere; }
      sup { font-size: 0.7em; }
      img { max-width: 100%; }
      @media print { .dr-print-hint { display: none; } }`;
    const hint =
      lang === "zh"
        ? '<div class="dr-print-hint" style="background:#fffae6;border:1px solid #f0e0a0;padding:8px 12px;border-radius:8px;margin-bottom:16px;font-size:13px;">如果未自动弹出打印窗口，请按 ⌘P，然后在「目标」中选择「存储为 PDF」。</div>'
        : '<div class="dr-print-hint" style="background:#fffae6;border:1px solid #f0e0a0;padding:8px 12px;border-radius:8px;margin-bottom:16px;font-size:13px;">If the print dialog didn\'t open, press ⌘P and choose "Save as PDF".</div>';
    const html = `<!doctype html><html lang="${lang}"><head><meta charset="utf-8"><title>${topic.trim() || (kb ? t("kbReportTitle") : "")}</title><style>${css}</style></head><body>${hint}${node.innerHTML}<script>window.addEventListener("load",function(){setTimeout(function(){window.print();},400);});<\/script></body></html>`;
    try {
      await openHtmlReport(html, "deep-research");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const phaseLabel = phase ? t((kb ? KB_PHASE_KEY : PHASE_KEY)[phase.phase]) : "";

  return (
    <div className="dr-view">
        <div className="dr-panel">
          <div className="setup-head dr-head">
            <div className="setup-title"><IconResearch size={18} /> {kb ? t("kbReportTitle") : t("drTitle")}</div>
            <button className="preview-close" onClick={onClose} disabled={running} title={t("drBackToChat")}>
              ×
            </button>
          </div>

          <div className={`dr-input-row${kb ? " kb" : ""}`}>
            {!kb && (
              <textarea
                className="dr-input"
                placeholder={t("drTopicPh")}
                value={topic}
                disabled={running}
                onChange={(e) => setTopic(e.target.value)}
                rows={2}
              />
            )}
            {kb && <span className="dr-kb-hint">{t("kbReportRunning")}</span>}
            <div className="dr-controls">
              {!kb && (
                <span className="dr-depth">
                  {t("drDepth")}
                  <Select
                    className="dr-depth-select"
                    value={depth}
                    disabled={running}
                    ariaLabel={t("drDepth")}
                    onChange={(v) => setDepth(v)}
                    options={[
                      { value: 2, label: t("drDepthQuick") },
                      { value: 3, label: t("drDepthStd") },
                      { value: 4, label: t("drDepthDeep") },
                    ]}
                  />
                </span>
              )}
              {running ? (
                <button className="setup-dl dr-stop" onClick={stop}>
                  <IconStop size={13} style={{ marginRight: 6 }} /> {t("drStop")}
                </button>
              ) : (
                <button
                  className="setup-dl ready"
                  onClick={() => void start()}
                  disabled={(!kb && !topic.trim()) || !model}
                >
                  <IconPlay size={13} style={{ marginRight: 6 }} /> {kb ? t("kbReportRegen") : t("drRun")}
                </button>
              )}
            </div>
          </div>

          <div className="dr-body" ref={bodyRef}>
            {!report && !running && !error && (
              <div className="dr-empty">
                <IconResearch
                  size={40}
                  style={{ display: "block", margin: "0 auto 16px", color: "var(--accent)", opacity: 0.85 }}
                />
                {kb ? t("kbReportEmpty") : t("drEmpty")}
              </div>
            )}

            {phase && (
              <div className="dr-status">
                <span className="dr-spin" />
                <span className="dr-status-text">
                  {phaseLabel}
                  {!kb && (phase.phase === "searching" || phase.phase === "reasoning")
                    ? ` · ${t("drRound")} ${phase.round}/${phase.rounds}`
                    : ""}
                </span>
                <span className="dr-status-meta">
                  {kb
                    ? `${sources.length} ${t("kbReportFiles")}`
                    : `${queries.length} ${t("drQueries")} · ${sources.length} ${t("drSources")}`}
                </span>
              </div>
            )}

            {queries.length > 0 && !done && (
              <div className="dr-queries">
                {queries.map((q, i) => (
                  <span key={i} className="dr-query">
                    <IconSearch size={12} style={{ marginRight: 5 }} />
                    {q}
                  </span>
                ))}
              </div>
            )}

            {reasoning && running && phase?.phase === "reasoning" && (
              <div className="dr-reasoning">{reasoning}</div>
            )}

            {report && (
              <div className="dr-report">
                <Markdown>{report}</Markdown>
                {running && <span className="cursor" />}
              </div>
            )}

            {error && <div className="setup-err">{error}</div>}
          </div>

          {done && report && (
            <div className="dr-export">
              <span className="dr-export-meta">
                {sources.length} {t("drSources")}
              </span>
              <button className="setup-dl ready" onClick={() => void exportPdf()}>
                <IconDownload size={14} style={{ marginRight: 6 }} /> {t("drExportPdf")}
              </button>
              <button className="setup-dl" onClick={() => void exportMd()}>
                <IconDownload size={14} style={{ marginRight: 6 }} /> {t("drExportMd")}
              </button>
            </div>
          )}
        </div>

        {/* Off-screen container; its rendered HTML is what the PDF export prints. */}
        {done && report && (
          <div className="dr-print-root" ref={printRef}>
            {!/^\s*#{1,2}\s/.test(report) && <h1>{topic.trim()}</h1>}
            <Markdown>{report}</Markdown>
          </div>
        )}
    </div>
  );
}
