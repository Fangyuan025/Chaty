import { useEffect, useRef, useState, type RefObject } from "react";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { useConfirm } from "./ConfirmModal";
import { agentBgClearFinished, agentBgKill, agentBgLog, type AgentBgInfo } from "../lib/ipc";
import { bgStatus, bgTitle, fmtElapsed } from "../lib/bgTasks";

/** The background tasks panel: every command the agent left running or ran in
 *  the background, what became of it, and — once opened — its output, live.
 *  The header pill used to offer one thing, "kill them all"; what a dev server
 *  was printing, or why a build had failed, was nowhere to be seen. */
export function BgTasksPanel({
  jobs,
  onClose,
  onChanged,
  anchorRef,
}: {
  jobs: AgentBgInfo[];
  onClose: () => void;
  /** Ask the owner to refresh `jobs` now rather than at its next poll. */
  onChanged: () => void;
  /** The control that opened the panel: it hangs from it. */
  anchorRef?: RefObject<HTMLElement | null>;
}) {
  const { t } = useI18n();
  const confirm = useConfirm();
  // Hung under the control that opened it, and kept off the window's edge —
  // pinned to the right edge it sat flush against it. Re-placed on resize.
  const [, setResized] = useState(0);
  useEffect(() => {
    const onResize = () => setResized((n) => n + 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const EDGE = 32;
  const anchor = anchorRef?.current?.getBoundingClientRect();
  const width = Math.min(460, window.innerWidth - 2 * EDGE);
  const place = anchor
    ? {
        top: anchor.bottom + 8,
        left: Math.max(EDGE, Math.min(anchor.left, window.innerWidth - width - EDGE)),
        right: "auto",
        width,
        maxHeight: Math.max(240, window.innerHeight - anchor.bottom - 8 - EDGE),
      }
    : undefined;
  const [open, setOpen] = useState<Set<number>>(() => new Set());
  const [logs, setLogs] = useState<Record<number, string>>({});
  const [showFinished, setShowFinished] = useState(true);
  const running = jobs.filter((j) => j.running);
  const finished = jobs.filter((j) => !j.running);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Output of every opened job: fetched when it opens, every second while it
  // runs, and once more when it ends (the running set changes then).
  const runningIds = new Set(running.map((j) => j.id));
  const openKey = [...open].sort((a, b) => a - b).join(",");
  const runningKey = [...runningIds].sort((a, b) => a - b).join(",");
  useEffect(() => {
    const ids = [...open];
    if (!ids.length) return;
    let alive = true;
    const pull = (id: number) =>
      agentBgLog(id)
        .then((j) => {
          if (alive) setLogs((cur) => ({ ...cur, [id]: j.tail }));
        })
        .catch(() => {});
    ids.forEach(pull);
    const live = ids.filter((id) => runningIds.has(id));
    const timer = live.length ? window.setInterval(() => live.forEach(pull), 1000) : undefined;
    return () => {
      alive = false;
      if (timer !== undefined) window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openKey, runningKey]);

  const toggle = (id: number) =>
    setOpen((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const stop = async (id: number) => {
    await agentBgKill(id).catch(() => {});
    onChanged();
  };

  const stopAll = async () => {
    const ok = await confirm({
      message: t("cmBgKillConfirm", { n: String(running.length) }),
      confirmLabel: t("cmBgKill"),
      danger: true,
    });
    if (!ok) return;
    await Promise.all(running.map((j) => agentBgKill(j.id).catch(() => {})));
    onChanged();
  };

  const clearFinished = async () => {
    await agentBgClearFinished().catch(() => 0);
    const gone = new Set(finished.map((j) => j.id));
    setOpen((cur) => new Set([...cur].filter((id) => !gone.has(id))));
    onChanged();
  };

  const statusLabel = (j: AgentBgInfo) => {
    const s = bgStatus(j);
    if (s === "failed") return t("bgFailed", { code: String(j.code ?? "?") });
    return t(s === "running" ? "bgRunningNow" : s === "done" ? "bgDone" : "bgStopped");
  };

  const item = (j: AgentBgInfo) => {
    const s = bgStatus(j);
    const isOpen = open.has(j.id);
    return (
      <div key={j.id} className={`bg-item ${isOpen ? "open" : ""}`}>
        <div className="bg-item-row">
          <button className="bg-item-head" onClick={() => toggle(j.id)} aria-expanded={isOpen}>
            <Icon name={isOpen ? "chevron-down" : "chevron-right"} size={13} />
            <span className="bg-item-text">
              <span className="bg-item-title">{bgTitle(j.command)}</span>
              <span className={`bg-item-meta ${s}`}>
                {s === "running" && <span className="cm-spin" />}
                Bash · {statusLabel(j)} · {fmtElapsed(j.elapsedSecs)}
              </span>
            </span>
          </button>
          {j.running && (
            <button className="bg-stop" onClick={() => void stop(j.id)}>
              {t("bgStop")}
            </button>
          )}
        </div>
        {isOpen && (
          <div className="bg-item-body">
            <pre className="cm-out bg-cmd">$ {j.command}</pre>
            <LogView text={logs[j.id]} empty={t("bgNoOutput")} />
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      <div className="popover-backdrop" onClick={onClose} />
      <div
        className="bg-panel"
        style={place}
        role="dialog"
        aria-label={t("bgTitle")}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="bg-panel-head">
          <span className="settings-title">{t("bgTitle")}</span>
          <div className="bg-panel-actions">
            {running.length > 0 && (
              <button className="bg-act danger" onClick={() => void stopAll()}>
                {t("cmBgKill")}
              </button>
            )}
            {finished.length > 0 && (
              <button className="bg-act" onClick={() => void clearFinished()}>
                {t("bgClearFinished")}
              </button>
            )}
            <button className="bg-act icon" onClick={onClose} title={t("bgClose")} aria-label={t("bgClose")}>
              <Icon name="x" size={14} />
            </button>
          </div>
        </div>
        <div className="bg-panel-list">
          {jobs.length === 0 && <div className="hw-loading">{t("bgEmpty")}</div>}
          {running.length > 0 && (
            <>
              <div className="bg-group">
                {t("bgRunning")} {running.length}
              </div>
              {running.map(item)}
            </>
          )}
          {finished.length > 0 && (
            <>
              <button className="bg-group toggle" onClick={() => setShowFinished((v) => !v)}>
                {t("bgFinished")} {finished.length}
                <Icon name={showFinished ? "chevron-down" : "chevron-right"} size={12} />
              </button>
              {showFinished && finished.map(item)}
            </>
          )}
        </div>
      </div>
    </>
  );
}

/** A job's output, kept scrolled to the newest line while the reader is at
 *  the bottom — and left alone once they scroll up to read something. */
function LogView({ text, empty }: { text?: string; empty: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const pinned = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [text]);
  return (
    <pre
      ref={ref}
      className="cm-out bg-log"
      onScroll={(e) => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
    >
      {text === undefined ? "…" : text.trim() ? text : empty}
    </pre>
  );
}
