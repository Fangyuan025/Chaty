import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { exportTextFile, type ModelInfo } from "../lib/ipc";
import { Markdown } from "./Markdown";
import {
  deepResearch,
  DRSignal,
  type DRPhase,
  type DROptions,
  type DRSource,
} from "../lib/deepResearch";

const PHASE_KEY = {
  planning: "drPhasePlanning",
  searching: "drPhaseSearching",
  reasoning: "drPhaseReasoning",
  writing: "drPhaseWriting",
  done: "drPhaseDone",
} as const satisfies Record<DRPhase, string>;

/** Deep Research: multi-round web search + reasoning → a long cited report,
 *  exportable to PDF / Markdown. */
export function DeepResearchPanel({
  model,
  onClose,
  onLockChange,
}: {
  model: ModelInfo | null;
  onClose: () => void;
  onLockChange: (locked: boolean) => void;
}) {
  const { t, lang } = useI18n();
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

  useEffect(() => {
    onLockChange(running);
    return () => onLockChange(false);
  }, [running, onLockChange]);
  useEffect(() => () => signalRef.current?.cancel(), []);
  useEffect(() => {
    if (running) bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
  }, [report, sources, reasoning, phase, running]);

  async function start() {
    if (!model || !topic.trim() || running) return;
    setQueries([]);
    setSources([]);
    setReasoning("");
    setReport("");
    setDone(false);
    setError("");
    setRunning(true);
    const signal = new DRSignal();
    signalRef.current = signal;

    const opts: DROptions = {
      topic: topic.trim(),
      rounds: depth,
      lang,
      think: model.supportsThinking && !model.thinkSwitch ? false : undefined,
      thinkSwitch: model.thinkSwitch,
      nCtx: model.nCtx ?? undefined,
      signal,
    };
    await deepResearch(opts, {
      onPhase: (phase, round, rounds) => setPhase({ phase, round, rounds }),
      onQuery: (q) => setQueries((cur) => [...cur, q]),
      onSources: (s) => setSources(s),
      onReasoning: (r) => setReasoning(r),
      onReportToken: (full) => setReport(full),
      onDone: (full) => {
        setReport(full);
        setDone(true);
        setRunning(false);
        setPhase(null);
      },
      onError: (msg) => {
        setError(msg);
        setRunning(false);
        setPhase(null);
      },
    });
    setRunning(false);
  }

  function stop() {
    signalRef.current?.cancel();
    setRunning(false);
    setPhase(null);
  }

  async function exportMd() {
    const safe = topic.trim().slice(0, 40).replace(/[/\\:*?"<>|]/g, "_") || "deep-research";
    const md = `# ${topic.trim()}\n\n${report}`;
    try {
      await exportTextFile(`${safe}.md`, md, "md");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function exportPdf() {
    // Print the off-screen report container; macOS print panel → "Save as PDF".
    document.body.classList.add("dr-printing");
    const cleanup = () => {
      document.body.classList.remove("dr-printing");
      window.removeEventListener("afterprint", cleanup);
    };
    window.addEventListener("afterprint", cleanup);
    setTimeout(() => window.print(), 60);
  }

  const phaseLabel = phase ? t(PHASE_KEY[phase.phase]) : "";

  return createPortal(
    <>
      <div className="preview-overlay" onMouseDown={running ? undefined : onClose}>
        <div className="setup-modal dr-modal" onMouseDown={(e) => e.stopPropagation()}>
          <div className="setup-head">
            <div>
              <div className="setup-title">🔬 {t("drTitle")}</div>
              <div className="setup-hw">{t("drSub")}</div>
            </div>
            {!running && (
              <button className="preview-close" onClick={onClose}>
                ×
              </button>
            )}
          </div>

          <div className="dr-input-row">
            <textarea
              className="dr-input"
              placeholder={t("drTopicPh")}
              value={topic}
              disabled={running}
              onChange={(e) => setTopic(e.target.value)}
              rows={2}
            />
            <div className="dr-controls">
              <label className="dr-depth">
                {t("drDepth")}
                <select
                  value={depth}
                  disabled={running}
                  onChange={(e) => setDepth(Number(e.target.value))}
                >
                  <option value={2}>{t("drDepthQuick")}</option>
                  <option value={3}>{t("drDepthStd")}</option>
                  <option value={4}>{t("drDepthDeep")}</option>
                </select>
              </label>
              {running ? (
                <button className="setup-dl dr-stop" onClick={stop}>
                  ⏹ {t("drStop")}
                </button>
              ) : (
                <button
                  className="setup-dl ready"
                  onClick={() => void start()}
                  disabled={!topic.trim() || !model}
                >
                  ▶ {t("drRun")}
                </button>
              )}
            </div>
          </div>

          <div className="dr-body" ref={bodyRef}>
            {!report && !running && !error && <div className="dr-empty">{t("drEmpty")}</div>}

            {phase && (
              <div className="dr-status">
                <span className="dr-spin" />
                <span className="dr-status-text">
                  {phaseLabel}
                  {phase.phase === "searching" || phase.phase === "reasoning"
                    ? ` · ${t("drRound")} ${phase.round}/${phase.rounds}`
                    : ""}
                </span>
                <span className="dr-status-meta">
                  {queries.length} {t("drQueries")} · {sources.length} {t("drSources")}
                </span>
              </div>
            )}

            {queries.length > 0 && !done && (
              <div className="dr-queries">
                {queries.map((q, i) => (
                  <span key={i} className="dr-query">
                    🔍 {q}
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
              <button className="setup-dl ready" onClick={exportPdf}>
                ⬇ {t("drExportPdf")}
              </button>
              <button className="setup-dl" onClick={() => void exportMd()}>
                ⬇ {t("drExportMd")}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Off-screen container printed to PDF (vector, full text). */}
      {done && report && (
        <div className="dr-print-root">
          <h1>{topic.trim()}</h1>
          <Markdown>{report}</Markdown>
        </div>
      )}
    </>,
    document.body,
  );
}
