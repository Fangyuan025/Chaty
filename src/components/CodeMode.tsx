import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { agentLang, useI18n } from "../lib/i18n";
import { diffLines } from "../lib/diff";
import { useConfirm } from "./ConfirmModal";
import { BUILTIN_SKILLS } from "../lib/skills";
import { copyToClipboard } from "../lib/clipboard";
import { cleanTitle } from "../lib/voiceText";
import { Icon } from "./Icon";
import { Markdown } from "./Markdown";
import {
  agentBgKill,
  agentBgList,
  agentDlList,
  type AgentDlInfo,
  agentCheckpointBegin,
  agentCheckpointRevertTo,
  agentGetWorkspace,
  agentGlob,
  agentListFiles,
  agentReadFile,
  agentSetWorkspace,
  agentGrantDir,
  agentRevokeDir,
  agentListGrants,
  agentClearGrants,
  imageThumb,
  imageDataUrl,
  saveImageAs,
  pickAttachmentFile,
  readAttachment,
  isVisionImagePath,
  type Attachment,
  type AgentBgInfo,
  codeSessionDelete,
  codeSessionList,
  codeSessionLoad,
  codeSessionSave,
  type ChatMessage,
  type CodeSessionMeta,
  type ModelInfo,
  generate,
} from "../lib/ipc";
import {
  AgentSignal,
  argContent,
  argEdits,
  argNew,
  argOld,
  argPath,
  runAgentTurn,
  IS_WINDOWS,
  type PlanItem,
  type ThinkMode,
  type ToolCall,
  type ToolStep,
} from "../lib/agentLoop";
import { isReadOnlyCommand } from "../lib/readOnlyCmd";
import { syncMcpServers } from "../lib/mcp";
import { loadSkills } from "../lib/skillFiles";
import { loadMemoryIndex } from "../lib/memoryFiles";
import { homeDir } from "@tauri-apps/api/path";
import { fmtBytes } from "../lib/fmt";

interface CodeMsg {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Attached image paths (vision models) shown in the user bubble. */
  images?: string[];
  /** All attachments (docs + images) shown as chips in the user bubble. */
  attachments?: { name: string; kind: string; path?: string }[];
  /** Reasoning shown before the final answer (collapsible). */
  thinking?: string;
  /** Live reasoning streaming for the in-flight step. */
  liveThinking?: string;
  steps: ToolStep[];
  /** The agent's current task plan (todo list), updated in place. */
  plan?: PlanItem[];
  /** Context was auto-compacted during this turn. */
  compacted?: boolean;
  /** The turn paused at the step limit (offer a Continue button). */
  paused?: boolean;
  /** Checkpoint opened before this user message's turn — enables rewind. */
  checkpointId?: number;
}

const THINK_MODES: ThinkMode[] = ["off", "normal", "deep"];
/** Models with a native effort ladder (Qwen3.8) show the model's own rungs
 *  instead of Chaty's generic intensities — off still means enable_thinking
 *  false, which the ladder itself has no rung for. */
const NATIVE_THINK_MODES: ThinkMode[] = ["off", "low", "normal", "deep"];
/** thinkMode → the native rung it requests. */
const EFFORT_OF: Partial<Record<ThinkMode, string>> = {
  low: "low",
  normal: "medium",
  deep: "xhigh",
};

const RAIL_DEFAULT = 240;
const RAIL_MIN = 180;
const RAIL_MAX = 420;

const uid = () => Math.random().toString(36).slice(2);


const TOOL_ICON: Record<string, string> = {
  read_file: "M9 2h6l4 4v14a0 0 0 0 1 0 0H5V2z",
  write_file: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M12 12v6M9 15h6",
  edit_file: "M17 3a2.85 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5z",
  multi_edit: "M17 3a2.85 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5zM14 8l2 2",
  outline: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
  list_dir: "M3 6h18M3 12h18M3 18h18",
  glob: "M3 6h18M3 12h18M3 18h18",
  grep: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-4-4",
  search_files: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-4-4M8 8h6M8 11h4",
  search_code: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-4-4M8.5 9.5L7 11l1.5 1.5M13.5 9.5L15 11l-1.5 1.5",
  search_docs: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M11 11a3 3 0 102.2 5.1L16 19",
  bash: "M4 5l6 7-6 7M13 19h7",
  bash_bg: "M4 5l6 7-6 7M13 5h7M13 12h7M13 19h7",
  bg_output: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 7v5l3 3",
  bg_kill: "M12 3a9 9 0 100 18 9 9 0 000-18zM9 9l6 6M15 9l-6 6",
  web_search: "M12 3a9 9 0 100 18 9 9 0 000-18zM3 12h18M12 3c2.5 2.5 3.8 5.6 3.8 9s-1.3 6.5-3.8 9c-2.5-2.5-3.8-5.6-3.8-9S9.5 5.5 12 3z",
  web_fetch: "M12 3a9 9 0 100 18 9 9 0 000-18zM3 12h18M12 3c2.5 2.5 3.8 5.6 3.8 9s-1.3 6.5-3.8 9c-2.5-2.5-3.8-5.6-3.8-9S9.5 5.5 12 3z",
  web_download: "M12 3v12M6 9l6 6 6-6M4 21h16",
  ask_user: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 8v5M12 16h.01",
  view_image: "M3 3h18v18H3zM3 15l5-5 4 4 3-3 6 6",
  browser_navigate: "M12 3a9 9 0 100 18 9 9 0 000-18zM3 12h18M12 3c2.5 2.5 3.8 5.6 3.8 9s-1.3 6.5-3.8 9",
  browser_screenshot: "M3 7h4l2-2h6l2 2h4v12H3zM12 17a3.5 3.5 0 100-7 3.5 3.5 0 000 7z",
  browser_snapshot: "M4 5h16v11H4zM8 20h8M12 16v4",
  browser_scroll: "M12 4v16M6 14l6 6 6-6M6 10l6-6 6 6",
  browser_close: "M18 6L6 18M6 6l12 12",
  browser_console: "M4 4h16v16H4zM7 9l3 3-3 3M13 15h4",
  browser_read: "M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7zM12 9a3 3 0 100 6 3 3 0 000-6z",
  browser_click: "M7 3v6l2-1 2 4 2-1-2-4h3z",
  browser_type: "M4 7h16M4 12h16M4 17h10",
  browser_eval: "M8 3H5a2 2 0 00-2 2v14a2 2 0 002 2h3M16 3h3a2 2 0 012 2v14a2 2 0 01-2 2h-3M10 8l-2 4 2 4M14 8l2 4-2 4",
};

function toolSummary(call: ToolCall): string {
  const a = call.args as Record<string, string>;
  switch (call.name) {
    case "read_file":
      return `read ${argPath(call.args) || "?"}`;
    case "write_file":
      return `write ${argPath(call.args) || "?"}`;
    case "edit_file":
    case "multi_edit": {
      const n = argEdits(call.args).length;
      const p = argPath(call.args) || "?";
      return n > 1 ? `edit ×${n} ${p}` : `edit ${p}`;
    }
    case "outline":
      return `outline ${argPath(call.args) || "?"}`;
    case "list_dir":
      // NOT "ls" — that read as a successful shell `ls` on Windows (where
      // cmd.exe has no ls) and sent users chasing a phantom bug.
      return `list ${a.path ?? "."}`;
    case "glob":
      return `glob ${a.pattern ?? ""}`;
    case "grep":
      return `grep ${a.pattern ?? ""}`;
    case "search_files":
      return `find ${a.query ?? ""}`;
    case "understand_repo":
      return "understand repo";
    case "validate_change":
      return `validate ${Array.isArray(call.args.files) ? (call.args.files as string[]).join(" ") : "changes"}`;
    case "search_code":
      return `code? ${a.query ?? ""}`;
    case "search_docs":
      return `docs? ${a.query ?? ""}`;
    case "bash":
      return `$ ${a.command ?? ""}`;
    case "bash_bg":
      return `bg $ ${a.command ?? ""}`;
    case "bg_output":
      return `bg output #${a.id ?? "?"}`;
    case "bg_kill":
      return `bg kill #${a.id ?? "?"}`;
    case "web_search":
      return a.site ? `search ${a.site}: ${a.query ?? ""}` : `search ${a.query ?? ""}`;
    case "web_fetch":
      return `fetch ${a.url ?? ""}`;
    case "web_download":
      return `download → ${argPath(call.args) || a.path || "?"}`;
    case "ask_user":
      return a.question ?? "ask user";
    case "view_image":
      return `view ${argPath(call.args) || a.path || "image"}`;
    case "browser_navigate":
      return `open ${a.url ?? ""}`;
    case "browser_screenshot":
      return "screenshot";
    case "browser_snapshot":
      return "snapshot";
    case "browser_scroll":
      return `scroll ${(a.to as string) ?? (a.by ? a.by + "px" : "")}`;
    case "browser_close":
      return "close browser";
    case "browser_console":
      return "console";
    case "browser_read":
      return "read page";
    case "browser_click":
      return `click ${a.text ?? a.selector ?? ""}`;
    case "browser_type":
      return `type → ${a.label ?? a.selector ?? ""}`;
    case "browser_eval":
      return `eval ${(a.expression ?? "").slice(0, 40)}`;
    default:
      return call.name;
  }
}

/** "Always allow" grant key for a call: a two-token command prefix for shells
 *  (`npm test`, `cargo build`), the tool name for file edits. */
function allowKeyFor(call: ToolCall): string {
  if (call.name === "bash" || call.name === "bash_bg") {
    const cmd = String((call.args as Record<string, unknown>).command ?? "").trim();
    return `cmd:${cmd.split(/\s+/).slice(0, 2).join(" ")}`;
  }
  return `tool:${call.name}`;
}

/** A short result badge for a finished step (exit code, line count).
 *  Edits render a red/green diffstat instead — see StepCard. */
function stepMeta(step: ToolStep, linesLabel: string): { text: string; tone: "add" | "warn" | "muted" } | null {
  if (step.status !== "done") return null;
  if (step.diff) return null;
  const r = step.result ?? "";
  const m = r.match(/\[exit (\d+)/);
  if (m) return { text: `exit ${m[1]}`, tone: m[1] === "0" ? "muted" : "warn" };
  const lines = r ? r.split("\n").length : 0;
  if (lines > 1) return { text: `${lines} ${linesLabel}`, tone: "muted" };
  return null;
}

/** One tool step: a compact header (icon + summary + status) that expands to the
 *  result or a diff. Image steps (screenshot / view_image) preview on click. */
function StepCard({ step, onPreview }: { step: ToolStep; onPreview?: (path: string) => void }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(step.status === "error");
  const diff = step.diff;
  const hasImage = !!step.image && step.status === "done";
  const hasBody = !!(step.result || diff);
  const meta = stepMeta(step, t("cmLines"));
  // A command that ran but exited non-zero: "done" (the result went back to
  // the model) but visually a failure — a green check here read as "ls
  // worked" when cmd.exe had actually rejected the command.
  const cmdFailed = step.status === "done" && meta?.tone === "warn";
  // Compute the diff once; the +N/−M badge uses the EXACT totals, never the
  // render-capped rows, so big edits are counted correctly.
  const d = useMemo(() => (diff ? diffLines(diff.before, diff.after) : null), [diff]);
  // Clicking an image step opens the preview directly; otherwise toggle the body.
  const onHead = () => {
    if (hasImage && step.image) onPreview?.(step.image);
    else if (hasBody) setOpen((o) => !o);
  };
  return (
    <div className={`cm-step ${step.status}${cmdFailed ? " cmd-failed" : ""}`}>
      <button className="cm-step-head" onClick={onHead}>
        <svg className="cm-step-ico" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d={TOOL_ICON[step.call.name] ?? "M4 6h16M4 12h16M4 18h16"} />
        </svg>
        <span className="cm-step-sum">{toolSummary(step.call)}</span>
        {d && step.status === "done" && (
          <span className="cm-step-diffstat">
            <em className="plus">+{d.added}</em>
            <em className="minus">-{d.removed}</em>
          </span>
        )}
        {hasImage && <span className="cm-step-meta muted">{t("cmClickPreview")}</span>}
        {meta && <span className={`cm-step-meta ${meta.tone}`}>{meta.text}</span>}
        <span className="cm-step-status">
          {step.status === "running" ? <span className="cm-spin" /> : null}
          {step.status === "done" ? (
            <Icon name={cmdFailed ? "x" : "check"} size={12} strokeWidth={2.2} />
          ) : null}
          {step.status === "error" ? <Icon name="x" size={12} strokeWidth={2.2} /> : null}
          {step.status === "denied" ? <Icon name="ban" size={12} strokeWidth={2} /> : null}
        </span>
      </button>
      {open && hasBody && (
        <div className="cm-step-body">
          {d ? (
            <pre className="cm-diff">
              {d.rows.map((l, i) => (
                <div key={i} className={`cm-dl ${l.kind}`}>
                  <span className="cm-dl-mark">{l.kind === "add" ? "+" : l.kind === "del" ? "-" : " "}</span>
                  {l.text}
                </div>
              ))}
              {d.truncated && (
                <div className="cm-dl ctx cm-dl-more">
                  {t("cmDiffMore").replace("{n}", String(d.added + d.removed))}
                </div>
              )}
            </pre>
          ) : (
            <pre className="cm-out">{(step.result ?? "").slice(0, 6000)}</pre>
          )}
        </div>
      )}
    </div>
  );
}

/** Collapsible reasoning panel. Streams open while `live`; collapsed once done. */
function ThinkPanel({ text, live, label }: { text: string; live?: boolean; label: string }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  return (
    <div className={`cm-think ${live ? "live" : ""}`}>
      <button className="cm-think-head" onClick={() => setOpen((o) => !o)}>
        {live && <span className="cm-spin" />}
        <span className="cm-think-label">{label}</span>
        <span className={`cm-think-caret ${open || live ? "open" : ""}`}>
          <Icon name="chevron-right" size={11} strokeWidth={2} />
        </span>
      </button>
      {(open || live) && <div className="cm-think-body">{text}</div>}
    </div>
  );
}

/** The agent's task plan as a live checklist with a progress count. */
function PlanPanel({ plan, label }: { plan: PlanItem[]; label: string }) {
  if (!plan.length) return null;
  const done = plan.filter((p) => p.status === "done").length;
  return (
    <div className="cm-plan">
      <div className="cm-plan-head">
        <Icon name="lines" size={13} />
        <span className="cm-plan-label">{label}</span>
        <span className="cm-plan-count">{done}/{plan.length}</span>
      </div>
      <ul className="cm-plan-list">
        {plan.map((p, i) => (
          <li key={i} className={`cm-plan-item ${p.status}`}>
            <span className="cm-plan-box">
              {p.status === "done" ? (
                <Icon name="check" size={11} strokeWidth={2.6} />
              ) : p.status === "in_progress" ? (
                <span className="cm-spin" />
              ) : (
                ""
              )}
            </span>
            <span className="cm-plan-text">{p.content}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Small self-loading thumbnail for a local image path (attach previews). */
const cmThumbCache = new Map<string, string>();
function ImgThumb({ path }: { path: string }) {
  const [src, setSrc] = useState<string | null>(cmThumbCache.get(path) ?? null);
  // Connect enabled MCP servers once per session — their tools join the
  // registry before the first agent turn builds its prompt.
  useEffect(() => {
    void syncMcpServers().catch(() => {});
  }, []);

  useEffect(() => {
    let live = true;
    if (!cmThumbCache.has(path)) {
      imageThumb(path, 256)
        .then((d) => {
          cmThumbCache.set(path, d);
          if (live) setSrc(d);
        })
        .catch(() => live && setSrc(""));
    }
    return () => {
      live = false;
    };
  }, [path]);
  if (src === "") return <span className="cm-attach-ph" />;
  return src ? <img src={src} alt="" /> : <span className="cm-attach-ph" />;
}

/** Full-size image preview modal with a "save to local" action. */
function ImagePreview({ path, onClose }: { path: string; onClose: () => void }) {
  const { t } = useI18n();
  const [src, setSrc] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    let live = true;
    // Full-resolution (crisp) — not a downscaled thumbnail.
    imageDataUrl(path)
      .then((d) => live && setSrc(d))
      .catch(() => live && setSrc(""));
    return () => {
      live = false;
    };
  }, [path]);
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const save = async () => {
    try {
      const name = path.split(/[/\\]/).pop() || "screenshot.png";
      const dest = await saveImageAs(path, name);
      if (dest) setSaved(true);
    } catch (e) {
      console.error(e);
    }
  };
  return createPortal(
    <div className="preview-overlay" onMouseDown={onClose}>
      <div className="cm-preview" onMouseDown={(e) => e.stopPropagation()}>
        <div className="cm-preview-bar">
          <span className="cm-preview-title">{t("cmScreenshot")}</span>
          <button className="cm-preview-btn" onClick={() => void save()}>
            <Icon name="download" size={13} strokeWidth={2} />
            {saved ? t("cmSaved") : t("cmSaveImage")}
          </button>
          <button className="cm-preview-close" onClick={onClose} title={t("closePreview")}>
            <Icon name="x" size={14} strokeWidth={2.2} />
          </button>
        </div>
        <div className="cm-preview-stage">
          {src ? <img src={src} alt="" /> : <span className="cm-spin" />}
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** Circular prompt-processing progress: a small ring that fills 0→100% while a
 *  long prompt is prefilling (the silent gap before tokens stream), plus the
 *  live percentage. Replaces the generic spinner whenever progress is known. */
function PrefillRing({ frac, size = 16 }: { frac: number; size?: number }) {
  const r = (size - 3) / 2; // stroke 2.5 + hairline padding
  const c = 2 * Math.PI * r;
  // Progress events arrive per decode batch / per media segment — seconds
  // apart on big prompts, which read as jumps. Smooth the DISPLAYED value:
  // chase a new target quickly, and between events creep gently ahead (capped
  // a few percent past the last report) so the ring visibly keeps moving
  // through long silent stretches like image encoding.
  const [shown, setShown] = useState(frac);
  const targetRef = useRef(frac);
  targetRef.current = frac;
  useEffect(() => {
    const timer = setInterval(() => {
      setShown((s) => {
        const t = targetRef.current;
        if (t < s - 0.08) return t; // new turn restarted — snap back
        if (t > s) return Math.min(t, s + Math.max((t - s) * 0.3, 0.005)); // chase
        return Math.min(s + 0.0015, t + 0.05, 0.99); // creep between events
      });
    }, 80);
    return () => clearInterval(timer);
  }, []);
  const pct = Math.round(Math.min(1, Math.max(0, shown)) * 100);
  return (
    <span className="cm-prefill" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border-strong)" strokeWidth="2.5" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct / 100)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className="cm-prefill-arc"
        />
      </svg>
      <span className="cm-prefill-pct">{pct}%</span>
    </span>
  );
}

export function CodeMode({
  model,
  active,
  maxSteps,
  bashTimeout,
  temperature,
  thinkBudget = 0,
  maxGenTokens = 0,
  autoApproveEdits = false,
  autoRunReadOnly = true,
  skills = [],
  disabledSkills = [],
  memoryEnabled = true,
  allowedCommands = [],
  sendKey = "enter",
  autoTitle = true,
}: {
  model: ModelInfo | null;
  active: boolean;
  /** Generate a session title with the model after the first turn (Settings → General). */
  autoTitle?: boolean;
  /** Max agent steps per turn (Settings → Code). */
  maxSteps?: number;
  /** Default bash timeout in seconds (Settings → Code). */
  bashTimeout?: number;
  /** Sampling temperature for agent steps (Settings → Code). */
  temperature?: number;
  /** Hard per-round think-token ceiling, 0 = auto (Settings → Code). */
  thinkBudget?: number;
  /** Per-round generation budget in tokens, 0 = auto (Settings → Code). */
  maxGenTokens?: number;
  /** Auto-approve file edits — write/edit/multi_edit run without asking
   *  (Settings → Code; checkpoints still allow rollback). */
  autoApproveEdits?: boolean;
  /** Auto-run obviously read-only bash commands without asking (Settings → Code). */
  autoRunReadOnly?: boolean;
  /** User-defined skills: /name inserts the prompt template (Settings → Code). */
  skills?: { name: string; prompt: string }[];
  /** Names of built-in skills the user turned off (Settings → Code). */
  disabledSkills?: string[];
  /** Project memory on (default): load the index + offer `remember`
   *  (Settings → Code). Off ⇒ neither, byte-identical prompt. */
  memoryEnabled?: boolean;
  /** Persistent command prefixes that never need approval (Settings → Code). */
  allowedCommands?: string[];
  /** Composer send shortcut (Settings → General). */
  sendKey?: "enter" | "modEnter";
}) {
  const { t, lang } = useI18n();
  const confirm = useConfirm();
  const [sessions, setSessions] = useState<CodeSessionMeta[]>([]);
  const [sid, setSid] = useState<string>(() => uid());
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<CodeMsg[]>([]);
  const [input, setInput] = useState("");
  // Files the user attached to the next Code turn — same as chat: documents
  // (PDF/Word/Excel/text/code, extracted to text), and images (vision models
  // see the pixels; text-only models get OCR text).
  const [codeAttachments, setCodeAttachments] = useState<Attachment[]>([]);
  const [attachErr, setAttachErr] = useState("");
  // Full-size image preview (screenshot / view_image steps).
  const [previewImg, setPreviewImg] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [bypass, setBypass] = useState(false);
  /** The model exposes a native reasoning-effort ladder (Qwen3.8) — the think
   *  switch then shows the model's own rungs instead of Chaty's intensities. */
  const nativeEffort = (model?.effortLevels?.length ?? 0) > 0;
  const [thinkMode, setThinkMode] = useState<ThinkMode>(() => {
    const v = localStorage.getItem("chaty.code.think");
    return v === "off" || v === "low" || v === "normal" || v === "deep" ? v : "normal";
  });
  const [approval, setApproval] = useState<{ call: ToolCall; resolve: (ok: boolean) => void } | null>(null);
  /** Out-of-workspace access request from the agent (grant persists this session). */
  const [dirAsk, setDirAsk] = useState<{ dir: string; resolve: (ok: boolean) => void } | null>(null);
  /** High-risk sudo command awaiting explicit user permission (always asks). */
  const [sudoAsk, setSudoAsk] = useState<{ cmd: string; resolve: (r: { ok: boolean; password?: string }) => void } | null>(null);
  /** Password typed into the sudo dialog (masked; cleared the moment it resolves). */
  const [sudoPw, setSudoPw] = useState("");
  /** Directories granted beyond the workspace this session (header chips). */
  const [dirGrants, setDirGrants] = useState<string[]>([]);
  const [ask, setAsk] = useState<{ question: string; options: string[]; resolve: (choice: string) => void } | null>(null);
  const [askText, setAskText] = useState("");
  const [slashSel, setSlashSel] = useState(0);
  const [atFiles, setAtFiles] = useState<string[]>([]);
  const [atSel, setAtSel] = useState(0);
  const [atHidden, setAtHidden] = useState(false);
  const [railW, setRailW] = useState(() => {
    try {
      const v = Number(localStorage.getItem("chaty.code.railW"));
      if (Number.isFinite(v) && v >= RAIL_MIN && v <= RAIL_MAX) return v;
    } catch {
      /* ignore */
    }
    return RAIL_DEFAULT;
  });
  const [stats, setStats] = useState<{ tokens: number; tps: number } | null>(null);
  const [ctxUsed, setCtxUsed] = useState(0);
  /** Messages typed while the agent was running — auto-sent one by one after. */
  const [queue, setQueue] = useState<string[]>([]);
  const queueRef = useRef(queue);
  queueRef.current = queue;
  /** Running background jobs (dev servers …) — header indicator. */
  const [bgJobs, setBgJobs] = useState<AgentBgInfo[]>([]);
  /** Active background downloads (header progress badge). */
  const [downloads, setDownloads] = useState<AgentDlInfo[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const signalRef = useRef<AgentSignal | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const bodyRef = useRef({ msgs, workspace, sid });
  bodyRef.current = { msgs, workspace, sid };
  // The approve callback closes over send()-time state — read bypass through a
  // ref so flipping the toggle MID-RUN takes effect immediately.
  const bypassRef = useRef(bypass);
  bypassRef.current = bypass;

  // Session-scoped "always allow" grants (from the approval dialog's third
  // button). Keyed "cmd:<two-token prefix>" or "tool:<name>"; reset per session.
  const sessionAllowsRef = useRef<Set<string>>(new Set());
  const allowedCommandsRef = useRef(allowedCommands);
  allowedCommandsRef.current = allowedCommands;
  const autoEditsRef = useRef(autoApproveEdits);
  autoEditsRef.current = autoApproveEdits;
  const autoRunReadOnlyRef = useRef(autoRunReadOnly);
  autoRunReadOnlyRef.current = autoRunReadOnly;
  /** Prompt-processing progress (0..1) for the current step, null = not prefilling. */
  const [prefill, setPrefill] = useState<number | null>(null);

  /** Toggle bypass; turning it ON also releases any approval that's waiting. */
  const toggleBypass = useCallback(() => {
    setBypass((b) => !b);
    if (!bypassRef.current) {
      // was off → now on: release the pending approval, if any
      setApproval((cur) => {
        cur?.resolve(true);
        return null;
      });
    }
  }, []);

  const refreshSessions = useCallback(() => {
    codeSessionList().then(setSessions).catch(() => {});
  }, []);

  useEffect(() => {
    if (active) {
      refreshSessions();
      agentGetWorkspace().then((w) => setWorkspace((cur) => cur ?? w)).catch(() => {});
    }
  }, [active, refreshSessions]);

  // Keep the background-jobs indicator fresh (a dev server the agent started
  // keeps running after the turn — the user needs to see and stop it).
  useEffect(() => {
    if (!active || !workspace) return;
    const tick = () => {
      agentBgList().then(setBgJobs).catch(() => {});
      agentDlList()
        .then((all) => setDownloads(all.filter((d) => !d.done)))
        .catch(() => {});
    };
    tick();
    const timer = setInterval(tick, running ? 1000 : 5000);
    return () => clearInterval(timer);
  }, [active, workspace, running]);

  // Follow-the-stream is an *intent*, not a position: any upward wheel motion
  // releases it immediately (a distance check alone loses to the next stream
  // frame re-pinning the bottom before the user escapes the threshold), and
  // parking back at the bottom re-arms it.
  const followRef = useRef(true);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const dist = () => el.scrollHeight - el.scrollTop - el.clientHeight;
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) followRef.current = false;
      else if (dist() < 40) followRef.current = true;
    };
    const onScroll = () => {
      // Covers scrollbar drags and keyboard scrolling; programmatic pins land
      // at the bottom, so they only ever re-arm.
      const d = dist();
      if (d < 4) followRef.current = true;
      else if (d > 240) followRef.current = false;
    };
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("scroll", onScroll);
    };
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && followRef.current) el.scrollTo({ top: el.scrollHeight });
  }, [msgs]);

  // Keyboard shortcuts: approval Enter/Esc, ask-user number keys, Esc to stop.
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (sudoAsk) {
        // High-risk: require an explicit click to allow; Esc denies.
        if (e.key === "Escape") { e.preventDefault(); sudoAsk.resolve({ ok: false }); setSudoAsk(null); setSudoPw(""); }
        return;
      }
      if (dirAsk) {
        if (e.key === "Enter") { e.preventDefault(); dirAsk.resolve(true); setDirAsk(null); }
        else if (e.key === "Escape") { e.preventDefault(); dirAsk.resolve(false); setDirAsk(null); }
        return;
      }
      if (approval) {
        if (e.key === "Enter") { e.preventDefault(); approval.resolve(true); setApproval(null); }
        else if (e.key === "Escape") { e.preventDefault(); approval.resolve(false); setApproval(null); }
        return;
      }
      if (ask) {
        if (e.key === "Escape") { e.preventDefault(); stop(); return; } // never trap the user
        if ((e.target as HTMLElement | null)?.tagName === "INPUT") return; // typing a custom answer
        const n = Number(e.key);
        if (Number.isInteger(n) && n >= 1 && n <= ask.options.length) {
          e.preventDefault();
          ask.resolve(ask.options[n - 1]);
          setAsk(null);
        }
        return;
      }
      if (running && e.key === "Escape") stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, approval, ask, dirAsk, sudoAsk, running]);

  /** Model-generated session titles by session id — once one exists it wins
   *  over the first-message fallback on every later persist. */
  const titlesRef = useRef(new Map<string, string>());

  /** Sessions already warned about a failing save — one alert, not a storm. */
  const saveFailWarnedRef = useRef(new Set<string>());

  const persist = useCallback((next: CodeMsg[], ws: string | null, id: string) => {
    const firstUser = next.find((m) => m.role === "user");
    const fallback =
      (firstUser?.text ?? "New session").replace(/\s+/g, " ").trim().slice(0, 48) || "New session";
    const title = titlesRef.current.get(id) ?? fallback;
    codeSessionSave(id, title, ws, JSON.stringify(next))
      .then(refreshSessions)
      .catch((e) => {
        // A silently-swallowed save failure is invisible data loss — the
        // calculator-session audit ended with a transcript the owner thought
        // was kept and no row in the database. Log loudly; once per session,
        // tell the user their transcript is not persisting.
        console.error("code session save FAILED", id, e);
        if (!saveFailWarnedRef.current.has(id)) {
          saveFailWarnedRef.current.add(id);
          alert(
            "会话保存失败——当前对话记录没有写入磁盘,重启后会丢失。请检查磁盘空间/权限。\n(Session save failed — this transcript is NOT persisting to disk.)",
          );
        }
      });
  }, [refreshSessions]);

  /** Debounced mid-run persist: trailing 2s, drops when the session moved. */
  const persistTimerRef = useRef<number | null>(null);
  const persistSoon = useCallback(
    (id: string) => {
      if (persistTimerRef.current !== null) return;
      persistTimerRef.current = window.setTimeout(() => {
        persistTimerRef.current = null;
        if (bodyRef.current.sid !== id) return;
        setMsgs((cur) => {
          persist(cur, bodyRef.current.workspace, id);
          return cur;
        });
      }, 2000);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );


  /** Ask the model for a concise session title after the first turn —
   *  mirrors the chat side's makeTitle (no-think, low temperature). */
  async function makeSessionTitle(id: string, firstMsg: string) {
    if (!autoTitle || !model) return;
    try {
      let acc = "";
      await generate(
        {
          messages: [
            {
              role: "system",
              content:
                lang === "zh"
                  ? "请用一个不超过12个汉字的简短短语，概括下面这条消息的主题，作为对话标题。只输出标题本身，不要引号、标点、解释或思考过程。"
                  : "Summarize the topic of the following message as a short chat title (max ~5 words). Output only the title — no quotes, punctuation, explanation, or reasoning.",
            },
            { role: "user", content: `${firstMsg}${model.thinkSwitch ? "\n/no_think" : ""}` },
          ],
          params: {
            temperature: 0.2,
            topP: 0.9,
            maxTokens: 512,
            think: model.supportsThinking && !model.thinkSwitch ? false : undefined,
          },
        },
        (ev) => {
          if (ev.type === "token") acc += ev.text;
        },
      );
      const title = cleanTitle(acc);
      // The session may have been deleted or switched away while the title
      // generated — saving then would resurrect / mislabel it.
      if (!title || bodyRef.current.sid !== id) return;
      titlesRef.current.set(id, title);
      setMsgs((cur) => {
        persist(cur, bodyRef.current.workspace, id);
        return cur;
      });
    } catch (e) {
      console.error(e);
    }
  }

  // Drag the rail's right edge to resize (rAF-throttled, persisted on release,
  // double-click resets) — mirrors the chat sidebar.
  function startRailResize(e: React.PointerEvent) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = railW;
    let frame: number | null = null;
    let latest = startW;
    document.body.classList.add("resizing-x");
    const onMove = (ev: PointerEvent) => {
      latest = Math.min(RAIL_MAX, Math.max(RAIL_MIN, startW + (ev.clientX - startX)));
      if (frame == null)
        frame = requestAnimationFrame(() => {
          frame = null;
          setRailW(latest);
        });
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      if (frame != null) cancelAnimationFrame(frame);
      document.body.classList.remove("resizing-x");
      setRailW(latest);
      try {
        localStorage.setItem("chaty.code.railW", String(latest));
      } catch {
        /* ignore */
      }
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }

  function resetRailW() {
    setRailW(RAIL_DEFAULT);
    try {
      localStorage.setItem("chaty.code.railW", String(RAIL_DEFAULT));
    } catch {
      /* ignore */
    }
  }

  async function pickWorkspace() {
    const dir = await open({ directory: true });
    if (!dir || Array.isArray(dir)) return;
    try {
      const abs = await agentSetWorkspace(dir);
      setWorkspace(abs);
      agentListGrants().then(setDirGrants).catch(() => {});
    } catch (e) {
      // window.alert doesn't render inside WKWebView — use the in-app modal.
      void confirm({
        message: e instanceof Error ? e.message : String(e),
        confirmLabel: t("confirm"),
      });
    }
  }

  /** Manually grant one more directory for this session (folder picker). */
  async function addGrantDir() {
    const dir = await open({ directory: true });
    if (!dir || Array.isArray(dir)) return;
    try {
      await agentGrantDir(dir);
      setDirGrants(await agentListGrants());
    } catch (e) {
      void confirm({ message: e instanceof Error ? e.message : String(e), confirmLabel: t("confirm") });
    }
  }

  /** One-click revoke of a granted directory. */
  async function revokeDir(d: string) {
    await agentRevokeDir(d).catch(() => {});
    setDirGrants(await agentListGrants().catch(() => [] as string[]));
  }

  function newSession() {
    if (running) return;
    setSid(uid());
    setMsgs([]);
    setInput("");
    setCtxUsed(0);
    sessionAllowsRef.current = new Set();
    void agentClearGrants().catch(() => {});
    setDirGrants([]);
  }

  async function openSession(id: string) {
    if (running || id === sid) return;
    const raw = await codeSessionLoad(id).catch(() => null);
    if (raw == null) return;
    try {
      const parsed = JSON.parse(raw) as CodeMsg[];
      setSid(id);
      setMsgs(parsed);
      setCtxUsed(0);
      setStats(null);
      sessionAllowsRef.current = new Set();
      void agentClearGrants().catch(() => {});
      setDirGrants([]);
      const meta = sessions.find((s) => s.id === id);
      if (meta?.workspace) {
        setWorkspace(meta.workspace);
        agentSetWorkspace(meta.workspace).catch(() => {});
      }
    } catch {
      /* ignore corrupt */
    }
  }

  async function deleteSession(id: string) {
    const ok = await confirm({
      message: t("confirmDeleteSession"),
      confirmLabel: t("confirmDelete"),
      danger: true,
    });
    if (!ok) return;
    await codeSessionDelete(id).catch(() => {});
    if (id === sid) {
      // Deleting the session you're viewing must reset the view even
      // mid-run: newSession()'s running-guard is for the "+" button (don't
      // silently abandon a run), not for a destructive delete. Stop the
      // agent first, then wake any dialog it is parked on — the loop only
      // notices the cancel once the pending promise resolves.
      if (running) {
        stop();
        approval?.resolve(false);
        setApproval(null);
        dirAsk?.resolve(false);
        setDirAsk(null);
        sudoAsk?.resolve({ ok: false });
        setSudoAsk(null);
        ask?.resolve("");
        setAsk(null);
      }
      setQueue([]);
      titlesRef.current.delete(id);
      setSid(uid());
      setMsgs([]);
      setInput("");
      setCtxUsed(0);
      setStats(null);
      sessionAllowsRef.current = new Set();
      void agentClearGrants().catch(() => {});
      setDirGrants([]);
    }
    refreshSessions();
  }

  /** Rewind to before `m`: restore journaled files and drop later messages.
   *  The message text lands back in the composer for editing & re-sending. */
  async function rewindTo(m: CodeMsg) {
    if (running || m.checkpointId == null) return;
    const ok = await confirm({
      message: t("cmRewindConfirm"),
      confirmLabel: t("cmRewind"),
      danger: true,
    });
    if (!ok) return;
    try {
      await agentCheckpointRevertTo(m.checkpointId);
    } catch (e) {
      void confirm({ message: e instanceof Error ? e.message : String(e), confirmLabel: t("confirm") });
      return;
    }
    setMsgs((cur) => {
      const i = cur.findIndex((x) => x.id === m.id);
      const next = i === -1 ? cur : cur.slice(0, i);
      persist(next, bodyRef.current.workspace, bodyRef.current.sid);
      return next;
    });
    setInput(m.text);
  }

  function stop() {
    signalRef.current?.cancel();
    approval?.resolve(false);
    setApproval(null);
    sudoAsk?.resolve({ ok: false });
    setSudoAsk(null);
    setSudoPw("");
    dirAsk?.resolve(false);
    setDirAsk(null);
    ask?.resolve("");
    setAsk(null);
    setQueue([]);
    setRunning(false);
  }

  // ── @-mention: reference a workspace file from the composer ──
  // Active while the input ends with an "@token" (start of a whitespace-split word).
  const atMatch = /(?:^|\s)@([^\s@]*)$/.exec(input);
  const atQuery = !running && workspace && atMatch ? atMatch[1] : null;

  useEffect(() => {
    if (atQuery == null) {
      setAtFiles([]);
      return;
    }
    const timer = setTimeout(() => {
      agentListFiles(atQuery || undefined, 12)
        .then((fs) => {
          setAtFiles(fs);
          setAtSel(0);
        })
        .catch(() => setAtFiles([]));
    }, 120);
    return () => clearTimeout(timer);
  }, [atQuery]);

  const atMenu = atQuery != null && !atHidden ? atFiles : [];

  function pickAtFile(path: string) {
    setInput((cur) => cur.replace(/(^|\s)@[^\s@]*$/, `$1${path} `));
    setAtFiles([]);
  }

  // ── Slash commands (local, never sent to the model) ──
  // Built-in skills the user hasn't disabled; a custom skill shadows a builtin
  // of the same name.
  const activeBuiltins = BUILTIN_SKILLS.filter(
    (b) => !disabledSkills.includes(b.name) && !skills.some((s) => s.name === b.name),
  );
  const slashMenu = (() => {
    const q = input.trimStart().toLowerCase();
    if (!q.startsWith("/") || q.includes("\n") || running) return [];
    const all = [
      { cmd: "/clear", desc: t("cmSlashClear") },
      { cmd: "/think off", desc: `${t("cmSlashThink")} · ${t("cmThinkOff")}` },
      { cmd: "/think normal", desc: `${t("cmSlashThink")} · ${t("cmThinkNormal")}` },
      { cmd: "/think deep", desc: `${t("cmSlashThink")} · ${t("cmThinkDeep")}` },
      { cmd: "/bypass", desc: t("cmSlashBypass") },
      { cmd: "/help", desc: t("cmSlashHelp") },
      ...activeBuiltins.map((b) => ({
        cmd: `/${b.name}`,
        desc: lang === "zh" ? b.desc.zh : b.desc.en,
      })),
      ...skills.map((s) => ({
        cmd: `/${s.name}`,
        desc: s.prompt.replace(/\s+/g, " ").slice(0, 60),
      })),
    ];
    return all.filter((e) => e.cmd.toLowerCase().startsWith(q));
  })();

  function runSlash(cmd: string) {
    // Skill (custom shadows builtin) → insert its prompt for editing.
    const skill = skills.find((s) => `/${s.name}` === cmd);
    if (skill) {
      setInput(skill.prompt);
      return;
    }
    const builtin = activeBuiltins.find((b) => `/${b.name}` === cmd);
    if (builtin) {
      setInput(lang === "zh" ? builtin.prompt.zh : builtin.prompt.en);
      return;
    }
    setInput("");
    if (cmd === "/clear") {
      newSession();
    } else if (cmd.startsWith("/think ")) {
      const arg = cmd.slice(7).trim();
      if (arg === "off" || arg === "low" || arg === "normal" || arg === "deep") {
        setThinkMode(arg);
        localStorage.setItem("chaty.code.think", arg);
      }
    } else if (cmd === "/bypass") {
      toggleBypass();
    } else if (cmd === "/help") {
      const allSkills = [
        ...activeBuiltins.map((b) => ({ name: b.name, d: lang === "zh" ? b.desc.zh : b.desc.en })),
        ...skills.map((s) => ({ name: s.name, d: s.prompt.replace(/\s+/g, " ").slice(0, 48) })),
      ];
      const skillLines = allSkills.length
        ? (lang === "zh" ? "\n\n**技能**(设置 → Code 可管理):\n" : "\n\n**Skills** (manage in Settings → Code):\n") +
          allSkills.map((s) => `- \`/${s.name}\` — ${s.d}`).join("\n")
        : "";
      const text =
        (lang === "zh"
          ? "**可用命令**\n\n- `/clear` — 新建会话(清空上下文)\n- `/think off|normal|deep` — 思考强度\n- `/bypass` — 切换自动批准\n- `/help` — 显示本帮助\n\n**快捷键**:运行中按 `Esc` 中断;`Shift+Tab` 切换自动批准;审批弹窗 `Enter` 允许、`Esc` 拒绝;选择题可按数字键;输入 `@` 可引用工作区文件。"
          : "**Commands**\n\n- `/clear` — new session (clears context)\n- `/think off|normal|deep` — reasoning depth\n- `/bypass` — toggle auto-approve\n- `/help` — this help\n\n**Shortcuts**: `Esc` interrupts a run; `Shift+Tab` toggles auto-approve; approval dialog `Enter` allows / `Esc` denies; number keys answer choice questions; type `@` to reference a workspace file.") +
        skillLines;
      setMsgs((cur) => {
        const next = [...cur, { id: uid(), role: "assistant" as const, text, steps: [] }];
        persist(next, bodyRef.current.workspace, bodyRef.current.sid);
        return next;
      });
    }
  }

  /** Project guide auto-load: AGENTS.md (the emerging standard) > PROJECT.md
   *  (what /init writes) > CLAUDE.md > Cursor/Copilot rule files, so projects
   *  configured for other agents work here unchanged. Re-read each turn —
   *  /init may have just written it. Capped so it can't crowd the context. */
  async function loadProjectDoc(): Promise<{ name: string; text: string } | undefined> {
    // read_file may be in hashline-anchor mode — rule text must go into the
    // system prompt clean, not with "12:abc→" prefixes.
    const deanchor = (s: string) => s.replace(/^\d+:[a-z]{2,4}→/gm, "");
    for (const name of [
      "AGENTS.md",
      "PROJECT.md",
      "CLAUDE.md",
      ".cursorrules",
      ".github/copilot-instructions.md",
    ]) {
      try {
        const text = deanchor(await agentReadFile(name));
        if (text.trim()) return { name, text: text.slice(0, 6000) };
      } catch {
        /* try the next candidate */
      }
    }
    try {
      // Cursor's split rules: concatenate the first few .mdc files.
      const files = (await agentGlob(".cursor/rules/*.mdc")).filter((f) => f.endsWith(".mdc")).slice(0, 3);
      const parts: string[] = [];
      for (const f of files) {
        try {
          parts.push(deanchor(await agentReadFile(f)));
        } catch {
          /* skip unreadable rule */
        }
      }
      const text = parts.join("\n\n").trim();
      if (text) return { name: ".cursor/rules", text: text.slice(0, 6000) };
    } catch {
      /* no cursor rules */
    }
    return undefined;
  }

  /** Attach a document or image to the next Code turn (mirrors chat). */
  async function attachCodeFile() {
    setAttachErr("");
    const path = await pickAttachmentFile();
    if (!path) return;
    try {
      if (model?.visionReady && isVisionImagePath(path)) {
        const name = path.split(/[/\\]/).pop() ?? "image";
        setCodeAttachments((cur) => [
          ...cur,
          { name, kind: "vision", text: "", chars: 0, truncated: false, path },
        ]);
      } else {
        // Documents (extracted) and non-vision images (OCR) both go here.
        const att = await readAttachment(path);
        setCodeAttachments((cur) => [...cur, { ...att, path }]);
      }
    } catch (e) {
      setAttachErr(typeof e === "string" ? e : t("readAttachFailed"));
    }
  }

  async function send(textArg?: string) {
    const text = (textArg ?? input).trim();
    if (!textArg && text.startsWith("/") && slashMenu.some((e) => e.cmd === text)) {
      runSlash(text);
      return;
    }
    if (!text || running || !model || !workspace) return;
    if (!textArg) setInput("");
    // Sending expresses interest in the newest output — re-arm the follow.
    followRef.current = true;
    // Snapshot point: everything the agent writes/edits this turn is journaled
    // so the user can rewind to "before this message".
    const checkpointId = await agentCheckpointBegin().catch(() => undefined);
    const projectDoc = await loadProjectDoc();
    // Skills: official + ~/.chaty/skills + <workspace>/.chaty/skills. Re-read
    // each turn so a skill the user just wrote is live immediately. A missing
    // home dir (or a glob that can't reach it) just means no global skills.
    const home = await homeDir().catch(() => undefined);
    const memoryIndex = memoryEnabled
      ? await loadMemoryIndex({ readFile: (fp) => agentReadFile(fp) }).catch(() => "")
      : undefined;
    const skills = await loadSkills(
      agentGlob,
      (fp) => agentReadFile(fp),
      home ? home.replace(/[/\\]$/, "") : undefined,
    ).catch(() => []);
    const visionImgs = codeAttachments.flatMap((a) =>
      a.kind === "vision" && a.path ? [a.path] : (a.images ?? []),
    );
    const docAtts = codeAttachments.filter((a) => a.kind !== "vision" && a.text.trim());
    // Document / OCR text becomes context prepended to the model's prompt, but
    // the visible bubble keeps just the typed text (+ attachment chips).
    const attachCtx = docAtts
      .map((a) => `【${t("attachContextLabel")} ${a.name}】\n${a.text.slice(0, 9000)}`)
      .join("\n\n");
    const userMsg: CodeMsg = {
      id: uid(),
      role: "user",
      text,
      steps: [],
      checkpointId,
      images: visionImgs.length ? visionImgs : undefined,
      attachments: codeAttachments.length
        ? codeAttachments.map((a) => ({ name: a.name, kind: a.kind, path: a.path }))
        : undefined,
    };
    const asst: CodeMsg = { id: uid(), role: "assistant", text: "", steps: [] };
    // Cross-turn history keeps only the text, but assistant turns carry a
    // compact record of the tools they ran — so "continue" resumes from the
    // actual progress instead of re-exploring the workspace from scratch.
    const history: ChatMessage[] = msgs.map((m) => {
      if (m.role !== "assistant" || m.steps.length === 0) return { role: m.role, content: m.text };
      const done = m.steps
        .filter((s) => s.status === "done")
        .map((s) => toolSummary(s.call))
        .join("; ");
      const prefix = done ? (lang === "zh" ? `(已执行:${done})\n` : `(tools run: ${done})\n`) : "";
      return { role: m.role, content: prefix + m.text };
    });
    const base = [...msgs, userMsg, asst];
    const isFirstTurn = msgs.length === 0;
    setMsgs(base);
    setRunning(true);
    setStats(null);
    // The session this turn belongs to — if the user deletes it mid-run the
    // live sid moves on, and the turn's results must not be written anywhere.
    const turnSid = bodyRef.current.sid;
    // First message = the session EXISTS: on disk, in the sidebar, named
    // (fallback title from the message text; the model-polished title still
    // lands after the turn). Persisting only at turn end meant a paused or
    // crashed first turn left ZERO rows — the calculator and minesweeper
    // audits both lost their transcripts to exactly that.
    persist(base, bodyRef.current.workspace, turnSid);

    const signal = new AgentSignal();
    signalRef.current = signal;
    const update = (fn: (a: CodeMsg) => CodeMsg) =>
      setMsgs((cur) => cur.map((m) => (m.id === asst.id ? fn(m) : m)));

    const turnImages = visionImgs;
    setCodeAttachments([]);
    const modelInput = attachCtx ? `${attachCtx}\n\n${text}` : text;
    await runAgentTurn(modelInput, history, workspace, agentLang(lang), {
      thinkMode,
      supportsThinking: model.supportsThinking,
      thinkSwitch: model.thinkSwitch,
      effort: nativeEffort ? EFFORT_OF[thinkMode] : undefined,
      nCtx: model.nCtx ?? undefined,
      maxSteps,
      temperature,
      thinkBudget,
      maxGenTokens,
      bashTimeout,
      projectDoc,
      skills,
      memoryIndex,
      visionReady: model.visionReady,
      toolRole: model.toolRole ?? false,
      // No vision encoder → still expose the browser suite, minus the two
      // screenshot tools: browser_read's digest is the model's eyes.
      // (ChatyWeb-Bench: 22/23 web tasks on a text-only 35B-A3B in this mode.)
      browserTextMode: !model.visionReady,
      images: turnImages.length ? turnImages : undefined,
      signal,
      // Out-of-workspace access always asks — even in bypass mode: bypass
      // covers per-step approvals, not widening the sandbox boundary itself.
      approveDir: (dir: string) => new Promise<boolean>((resolve) => setDirAsk({ dir, resolve })),
      // `sudo` is privileged/dangerous — ALWAYS ask with a dedicated dialog
      // (with optional secure password entry), even under bypass/allowlist.
      approveSudo: (cmd: string) =>
        new Promise<{ ok: boolean; password?: string }>((resolve) => setSudoAsk({ cmd, resolve })),
      approve: (call: ToolCall) => {
        if (bypassRef.current) return Promise.resolve(true);
        if (sessionAllowsRef.current.has(allowKeyFor(call))) return Promise.resolve(true);
        // Settings → Code: file edits run without asking (checkpoints cover rollback).
        if (
          autoEditsRef.current &&
          (call.name === "write_file" || call.name === "edit_file" || call.name === "multi_edit")
        ) {
          return Promise.resolve(true);
        }
        if (call.name === "bash" || call.name === "bash_bg") {
          const cmd = String(call.args.command ?? "").trim();
          // Settings → Code: obviously read-only commands skip the dialog —
          // their approval prompt has only one sane answer. Fail-closed:
          // anything uncertain falls through to the allowlist / dialog.
          if (autoRunReadOnlyRef.current && isReadOnlyCommand(cmd, { windows: IS_WINDOWS })) {
            return Promise.resolve(true);
          }
          if (allowedCommandsRef.current.some((p) => cmd === p || cmd.startsWith(p + " "))) {
            return Promise.resolve(true);
          }
        }
        return new Promise<boolean>((resolve) => setApproval({ call, resolve }));
      },
    }, {
      // DEV forensics: keep the last 200 raw rounds + injected corrections on
      // window.__agentTrace — `copy(window.__agentTrace)` in devtools dumps
      // the evidence the bench transcripts get via the same instrument.
      ...(import.meta.env.DEV
        ? {
            onTrace: (ev: { kind: string; text: string }) => {
              const w = window as unknown as { __agentTrace?: { kind: string; text: string }[] };
              (w.__agentTrace ??= []).push({ kind: ev.kind, text: ev.text.slice(0, 4000) });
              if (w.__agentTrace.length > 200) w.__agentTrace.shift();
            },
          }
        : {}),
      onThinking: (t) => update((m) => ({ ...m, liveThinking: t })),
      onStats: (tokens, tps) => setStats({ tokens, tps }),
      onContext: (used) => setCtxUsed(used),
      onPrefill: setPrefill,
      onDirGrants: setDirGrants,
      onPlan: (todos) => update((m) => ({ ...m, plan: todos })),
      onCompacted: () => update((m) => ({ ...m, compacted: true })),
      onAskUser: (question, options) =>
        new Promise<string>((resolve) => {
          setAskText("");
          setAsk({ question, options, resolve });
        }),
      onAssistantText: (full) => update((m) => ({ ...m, text: full })),
      onStep: (step) => {
        update((m) => {
          const steps = [...m.steps];
          const i = steps.findIndex((s) => s.id === step.id);
          if (i >= 0) steps[i] = step;
          else steps.push(step);
          // the reasoning is now captured on the step → clear the live buffer
          return { ...m, steps, liveThinking: "" };
        });
        // Mid-run durability: every step lands on disk (debounced), so a
        // pause + quit — or a crash — loses at most the last two seconds,
        // not the whole transcript.
        persistSoon(turnSid);
      },
      onFinal: (final, thinking, reason) =>
        update((m) => ({
          ...m,
          text: final,
          thinking: thinking || m.thinking,
          liveThinking: "",
          paused: reason === "steps",
        })),
      onError: (msg) => update((m) => ({ ...m, text: (m.text ? m.text + "\n\n" : "") + `**${msg}**` })),
    });

    setRunning(false);
    setApproval(null);
    setPrefill(null);
    // Persist only while this turn's session is still the active one — after
    // a mid-run delete, writing here would resurrect the file the user just
    // removed (and clobber the fresh empty session).
    if (bodyRef.current.sid === turnSid) {
      setMsgs((cur) => {
        persist(cur, bodyRef.current.workspace, turnSid);
        return cur;
      });
      // First completed turn of a fresh session → give it a real title (the
      // engine is idle again; the helper no-op's if the session goes away).
      if (isFirstTurn && !signal.cancelled) {
        void makeSessionTitle(turnSid, text);
      }
    }

    // Messages queued while the agent was working → run them in order.
    const next = queueRef.current[0];
    if (next !== undefined && !signal.cancelled && bodyRef.current.sid === turnSid) {
      setQueue((q) => q.slice(1));
      void send(next);
    }
  }

  const wsName = workspace ? workspace.split("/").filter(Boolean).pop() : null;

  return (
    <div className="code-mode" style={active ? undefined : { display: "none" }}>
      <aside className="code-rail" style={{ width: railW }}>
        <button className="cm-new" onClick={newSession} disabled={running}>
          <Icon name="plus" size={13} strokeWidth={2} /> {t("cmNewSession")}
        </button>
        <div className="cm-sessions">
          {sessions.length === 0 ? (
            <div className="cm-empty-list">{t("cmNoSessions")}</div>
          ) : (
            sessions.map((s) => (
              <div
                key={s.id}
                className={`cm-session ${s.id === sid ? "active" : ""}`}
                onClick={() => void openSession(s.id)}
              >
                <span className="cm-session-title">{s.title}</span>
                <button
                  className="cm-session-del"
                  title={t("deleteConv")}
                  onClick={(e) => { e.stopPropagation(); void deleteSession(s.id); }}
                >
                  <Icon name="x" size={11} strokeWidth={2.4} />
                </button>
              </div>
            ))
          )}
        </div>
        <div className="side-status" title={model ? model.name : ""}>
          <span className="ss-dot" />
          <span className="ss-meta">v{__APP_VERSION__}</span>
        </div>
        <div
          className="sidebar-resizer"
          role="separator"
          aria-orientation="vertical"
          title={t("resizeSidebar")}
          onPointerDown={startRailResize}
          onDoubleClick={resetRailW}
        />
      </aside>

      <main className="code-main">
        <div className="code-head">
          <button className="cm-ws" onClick={() => void pickWorkspace()} disabled={running} title={workspace ?? ""}>
            <Icon name="folder" size={14} />
            {wsName ? <span className="cm-ws-name">{wsName}</span> : <span className="cm-ws-pick">{t("cmOpenFolder")}</span>}
          </button>
          {(dirGrants.length > 0 || workspace) && (
            <div className="cm-grants">
              {dirGrants.map((d) => (
                <span key={d} className="cm-grant-chip" title={d}>
                  <Icon name="folder" size={11} />
                  <span className="cm-grant-name">{d.split("/").filter(Boolean).pop()}</span>
                  <button className="cm-grant-del" title={t("cmGrantRevoke")} onClick={() => void revokeDir(d)}>
                    <Icon name="x" size={10} strokeWidth={2.2} />
                  </button>
                </span>
              ))}
              {workspace && (
                <button className="cm-grant-add" title={t("cmGrantAddTip")} onClick={() => void addGrantDir()}>
                  +
                </button>
              )}
            </div>
          )}
          <span className="cm-head-spacer" />
          {downloads.length > 0 && (
            <span
              className="cm-bgjobs cm-dl-badge"
              title={downloads.map((d) => `#${d.id} ${d.url} → ${d.path}`).join("\n")}
            >
              <span className="cm-spin" /> ⬇ {downloads.length}
              {downloads[0].total
                ? ` · ${Math.min(100, Math.round((downloads[0].downloaded / downloads[0].total) * 100))}%`
                : ` · ${fmtBytes(downloads[0].downloaded)}`}
            </span>
          )}
          {bgJobs.length > 0 && (
            <button
              className="cm-bgjobs"
              title={bgJobs.map((j) => `#${j.id} · ${j.command}`).join("\n") + "\n" + t("cmBgKillHint")}
              onClick={async () => {
                const ok = await confirm({
                  message: t("cmBgKillConfirm", { n: String(bgJobs.length) }),
                  confirmLabel: t("cmBgKill"),
                  danger: true,
                });
                if (!ok) return;
                await Promise.all(bgJobs.map((j) => agentBgKill(j.id).catch(() => {})));
                agentBgList().then(setBgJobs).catch(() => {});
              }}
            >
              <span className="cm-spin" /> {bgJobs.length} {t("cmBgJobs")}
            </button>
          )}
          {ctxUsed > 0 && (model?.nCtx ?? 0) > 0 && (() => {
            const nCtx = model!.nCtx!;
            const pct = Math.min(100, Math.round((ctxUsed / nCtx) * 100));
            const r = 7;
            const circ = 2 * Math.PI * r;
            const tone = pct >= 90 ? "hot" : pct >= 70 ? "warn" : "";
            return (
              <div className={`cm-ctx ${tone}`} title={`${ctxUsed.toLocaleString()} / ${nCtx.toLocaleString()} tokens · ${t("cmContext")}`}>
                <svg width="18" height="18" viewBox="0 0 18 18">
                  <circle cx="9" cy="9" r={r} fill="none" stroke="var(--border-strong)" strokeWidth="2.2" />
                  <circle
                    cx="9" cy="9" r={r} fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
                    strokeDasharray={circ} strokeDashoffset={circ * (1 - pct / 100)}
                    transform="rotate(-90 9 9)"
                  />
                </svg>
                <span className="cm-ctx-pct">{pct}%</span>
              </div>
            );
          })()}
          <div className="cm-think-switch" title={t(nativeEffort ? "effortHint" : "cmThinkHint")}>
            {(nativeEffort ? NATIVE_THINK_MODES : THINK_MODES).map((mode) => (
              <button
                key={mode}
                className={`cm-think-tab ${thinkMode === mode ? "active" : ""}`}
                onClick={() => {
                  setThinkMode(mode);
                  localStorage.setItem("chaty.code.think", mode);
                }}
                disabled={running}
              >
                {t(
                  mode === "off"
                    ? "cmThinkOff"
                    : nativeEffort
                      ? mode === "low"
                        ? "effortLow"
                        : mode === "normal"
                          ? "effortMedium"
                          : "effortXhigh"
                      : mode === "normal"
                        ? "cmThinkNormal"
                        : "cmThinkDeep",
                )}
              </button>
            ))}
          </div>
          <button
            className={`cm-bypass ${bypass ? "on" : ""}`}
            onClick={toggleBypass}
            title={t("cmBypassHint")}
          >
            <span className="cm-bypass-dot" /> {t("cmBypass")}
          </button>
        </div>

        <div className="code-scroll" ref={scrollRef}>
          {msgs.length === 0 ? (
            <div className="cm-welcome">
              <div className="cm-welcome-title">{t("cmWelcome")}</div>
              <div className="cm-welcome-sub">
                {!model ? t("cmWelcomeNoModel") : workspace ? t("cmWelcomeReady") : t("cmWelcomePick")}
              </div>
              {!workspace && (
                <button className="cm-welcome-pick" onClick={() => void pickWorkspace()}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" /></svg>
                  {t("cmOpenFolder")}
                </button>
              )}
              {workspace && model && (
                <div className="cm-welcome-egs">
                  {[t("cmEg1"), t("cmEg2"), t("cmEg3")].map((eg) => (
                    <button key={eg} className="cm-welcome-eg" onClick={() => setInput(eg)}>
                      {eg}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="cm-thread">
            {msgs.map((m) =>
              m.role === "user" ? (
                <div key={m.id} className="cm-user-row">
                  {m.checkpointId != null && !running && (
                    <button
                      className="cm-rewind"
                      title={t("cmRewindHint")}
                      onClick={() => void rewindTo(m)}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 109-9 9.75 9.75 0 00-6.74 2.74L3 8" /><path d="M3 3v5h5" /></svg>
                    </button>
                  )}
                  <div className="cm-user">
                    {m.images && m.images.length > 0 && (
                      <span className="cm-user-images">
                        {m.images.map((p) => (
                          <span key={p} className="cm-attach-img" onClick={() => setPreviewImg(p)} style={{ cursor: "zoom-in" }}>
                            <ImgThumb path={p} />
                          </span>
                        ))}
                      </span>
                    )}
                    {m.attachments && m.attachments.some((a) => a.kind !== "vision") && (
                      <span className="cm-user-docs">
                        {m.attachments.filter((a) => a.kind !== "vision").map((a, i) => (
                          <span key={a.name + i} className="cm-attach-doc">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6" /></svg>
                            <span className="cm-attach-doc-name">{a.name}</span>
                          </span>
                        ))}
                      </span>
                    )}
                    {m.text}
                  </div>
                </div>
              ) : (
                <div key={m.id} className="cm-asst">
                  {m.compacted && (
                    <div className="cm-compacted">
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M13 2L4 14h6l-1 8 9-12h-6z" /></svg>
                      {t("cmCompacted")}
                    </div>
                  )}
                  {m.plan && m.plan.length > 0 && <PlanPanel plan={m.plan} label={t("cmPlan")} />}
                  {m.steps.map((s) => (
                    <div key={s.id} className="cm-block">
                      {s.thinking && <ThinkPanel text={s.thinking} label={t("cmThought")} />}
                      <StepCard step={s} onPreview={setPreviewImg} />
                    </div>
                  ))}
                  {m.liveThinking && <ThinkPanel text={m.liveThinking} live label={t("cmThinking")} />}
                  {m.thinking && <ThinkPanel text={m.thinking} label={t("cmThought")} />}
                  {m.text && (
                    <div className="cm-asst-text answer">
                      <Markdown>{m.text}</Markdown>
                      {!running && (
                        <button
                          className="cm-copy"
                          title={t("copy")}
                          onClick={() => {
                            void copyToClipboard(m.text);
                            setCopiedId(m.id);
                            setTimeout(() => setCopiedId((c) => (c === m.id ? null : c)), 1200);
                          }}
                        >
                          {copiedId === m.id ? (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
                          ) : (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="12" height="12" rx="2" /><path d="M5 15V5a2 2 0 012-2h10" /></svg>
                          )}
                        </button>
                      )}
                    </div>
                  )}
                  {m.paused && !running && m === msgs[msgs.length - 1] && (
                    <button
                      className="cm-continue"
                      onClick={() => void send(lang === "zh" ? "继续" : "continue")}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                      {t("cmContinue")}
                    </button>
                  )}
                  {running && m === msgs[msgs.length - 1] && !m.text && !m.liveThinking && m.steps.length === 0 && (
                    prefill != null ? <PrefillRing frac={prefill} size={18} /> : <span className="cm-spin cm-working" />
                  )}
                </div>
              ),
            )}
            </div>
          )}
        </div>

        {running && (() => {
          const cur = msgs[msgs.length - 1];
          const curStep = cur?.steps?.[cur.steps.length - 1];
          const label =
            prefill != null
              ? t("cmPrefill")
              : curStep && curStep.status === "running"
                ? toolSummary(curStep.call)
                : t("cmRunning");
          return (
            <div className="cm-runbar">
              {prefill != null ? <PrefillRing frac={prefill} /> : <span className="cm-spin" />}
              <span className="cm-runbar-label">{label}</span>
              {stats && (
                <span className="cm-runbar-stats">
                  {stats.tokens} tok{stats.tps > 0 ? ` · ${stats.tps.toFixed(1)} tok/s` : ""}
                </span>
              )}
            </div>
          );
        })()}
        <div className="code-composer">
          {queue.length > 0 && (
            <div className="cm-queue">
              {queue.map((q, i) => (
                <span key={i} className="cm-queue-chip" title={q}>
                  <span className="cm-queue-text">{q}</span>
                  <button
                    className="cm-queue-del"
                    onClick={() => setQueue((cur) => cur.filter((_, j) => j !== i))}
                  >
                    <Icon name="x" size={10} strokeWidth={2.2} />
                  </button>
                </span>
              ))}
            </div>
          )}
          {(codeAttachments.length > 0 || attachErr) && (
            <div className="cm-attach-row">
              {codeAttachments.map((a, i) =>
                a.kind === "vision" && a.path ? (
                  <span key={a.path} className="cm-attach-img">
                    <ImgThumb path={a.path} />
                    <button
                      className="cm-attach-del"
                      title={t("removeAttach")}
                      onClick={() => setCodeAttachments((cur) => cur.filter((_, j) => j !== i))}
                    >
                      <Icon name="x" size={10} strokeWidth={2.2} />
                    </button>
                  </span>
                ) : (
                  <span key={a.name + i} className="cm-attach-doc" title={a.name}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6" /></svg>
                    <span className="cm-attach-doc-name">{a.name}</span>
                    <button
                      className="cm-attach-del inline"
                      title={t("removeAttach")}
                      onClick={() => setCodeAttachments((cur) => cur.filter((_, j) => j !== i))}
                    >
                      <Icon name="x" size={10} strokeWidth={2.2} />
                    </button>
                  </span>
                ),
              )}
              {attachErr && <span className="cm-attach-err">{attachErr}</span>}
            </div>
          )}
          <div className="cm-input-row">
            {slashMenu.length > 0 && (
              <div className="cm-slash">
                {slashMenu.map((entry, i) => (
                  <button
                    key={entry.cmd}
                    className={`cm-slash-item ${i === slashSel ? "sel" : ""}`}
                    onMouseEnter={() => setSlashSel(i)}
                    onClick={() => runSlash(entry.cmd)}
                  >
                    <span className="cm-slash-cmd">{entry.cmd}</span>
                    <span className="cm-slash-desc">{entry.desc}</span>
                  </button>
                ))}
              </div>
            )}
            {slashMenu.length === 0 && atMenu.length > 0 && (
              <div className="cm-slash">
                {atMenu.map((f, i) => (
                  <button
                    key={f}
                    className={`cm-slash-item ${i === atSel ? "sel" : ""}`}
                    onMouseEnter={() => setAtSel(i)}
                    onClick={() => pickAtFile(f)}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6" /></svg>
                    <span className="cm-slash-desc">{f}</span>
                  </button>
                ))}
              </div>
            )}
            <textarea
              className="cm-input"
              placeholder={
                !model
                  ? t("cmPlaceholderNoModel")
                  : !workspace
                    ? t("cmPlaceholderNoWs")
                    : running
                      ? t("cmQueuePlaceholder")
                      : sendKey === "modEnter"
                        ? t("cmPlaceholderMod")
                        : t("cmPlaceholder")
              }
              value={input}
              disabled={!model || !workspace}
              onChange={(e) => { setInput(e.target.value); setSlashSel(0); setAtHidden(false); }}
              onKeyDown={(e) => {
                // Shift+Tab toggles auto-approve, mirroring Claude Code.
                if (e.key === "Tab" && e.shiftKey) {
                  e.preventDefault();
                  toggleBypass();
                  return;
                }
                if (atMenu.length > 0) {
                  if (e.key === "ArrowDown") { e.preventDefault(); setAtSel((s) => (s + 1) % atMenu.length); return; }
                  if (e.key === "ArrowUp") { e.preventDefault(); setAtSel((s) => (s - 1 + atMenu.length) % atMenu.length); return; }
                  if (e.key === "Enter" || e.key === "Tab") {
                    e.preventDefault();
                    pickAtFile(atMenu[Math.min(atSel, atMenu.length - 1)]);
                    return;
                  }
                  if (e.key === "Escape") { setAtHidden(true); return; }
                }
                if (slashMenu.length > 0) {
                  if (e.key === "ArrowDown") { e.preventDefault(); setSlashSel((s) => (s + 1) % slashMenu.length); return; }
                  if (e.key === "ArrowUp") { e.preventDefault(); setSlashSel((s) => (s - 1 + slashMenu.length) % slashMenu.length); return; }
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    runSlash(slashMenu[Math.min(slashSel, slashMenu.length - 1)].cmd);
                    return;
                  }
                  if (e.key === "Escape") { setInput(""); return; }
                }
                // Send combo per Settings → General; with ⌘/Ctrl+Enter chosen,
                // plain Enter falls through and inserts a newline.
                const sends =
                  e.key === "Enter" &&
                  (sendKey === "modEnter" ? e.metaKey || e.ctrlKey : !e.shiftKey);
                if (sends) {
                  e.preventDefault();
                  if (running) {
                    // Queue it — sent automatically once the current turn ends.
                    const q = input.trim();
                    if (q) {
                      setQueue((cur) => [...cur, q]);
                      setInput("");
                    }
                    return;
                  }
                  void send();
                }
              }}
              rows={1}
            />
            <button
              className="cm-attach-btn"
              title={t("cmAttachFile")}
              disabled={!workspace || running}
              onClick={() => void attachCodeFile()}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d="M21.4 11.05l-8.5 8.5a5 5 0 01-7.07-7.07l8.49-8.49a3 3 0 014.24 4.24l-8.49 8.49a1 1 0 01-1.42-1.42l7.8-7.79" /></svg>
            </button>
            {running ? (
              <button className="cm-send stop" onClick={stop} title={t("drStop")}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5" /></svg>
              </button>
            ) : (
              <button className="cm-send" onClick={() => void send()} disabled={!input.trim() || !model || !workspace}>
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
              </button>
            )}
          </div>
        </div>
      </main>

      {previewImg && <ImagePreview path={previewImg} onClose={() => setPreviewImg(null)} />}
      {sudoAsk && (() => {
        const deny = () => { sudoAsk.resolve({ ok: false }); setSudoAsk(null); setSudoPw(""); };
        const allow = () => { sudoAsk.resolve({ ok: true, password: sudoPw || undefined }); setSudoAsk(null); setSudoPw(""); };
        return (
          <div className="cm-approve-backdrop" onMouseDown={deny}>
            <div className="cm-approve cm-approve-danger" onMouseDown={(e) => e.stopPropagation()}>
              <div className="cm-approve-title">{t("cmSudoTitle")}</div>
              <pre className="cm-approve-cmd cm-approve-cmd-danger">{sudoAsk.cmd}</pre>
              <div className="cm-approve-detail">{t("cmSudoHint")}</div>
              <input
                type="password"
                className="cm-sudo-pw"
                autoFocus
                autoComplete="off"
                placeholder={t("cmSudoPwPlaceholder")}
                value={sudoPw}
                onChange={(e) => setSudoPw(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); allow(); } }}
              />
              <div className="cm-approve-detail cm-sudo-pw-note">{t("cmSudoPwNote")}</div>
              <div className="cm-approve-actions">
                <button className="cm-allow" onClick={deny}>{t("cmDeny")}</button>
                <button className="cm-deny" onClick={allow}>{t("cmSudoAllow")}</button>
              </div>
            </div>
          </div>
        );
      })()}
      {dirAsk && (
        <div className="cm-approve-backdrop" onMouseDown={() => { dirAsk.resolve(false); setDirAsk(null); }}>
          <div className="cm-approve" onMouseDown={(e) => e.stopPropagation()}>
            <div className="cm-approve-title">{t("cmDirAskTitle")}</div>
            <pre className="cm-approve-cmd">{dirAsk.dir}</pre>
            <div className="cm-approve-detail">{t("cmDirAskHint")}</div>
            <div className="cm-approve-actions">
              <button className="cm-deny" onClick={() => { dirAsk.resolve(false); setDirAsk(null); }}>{t("cmDeny")}</button>
              <button className="cm-allow" onClick={() => { dirAsk.resolve(true); setDirAsk(null); }}>{t("cmDirAllow")}</button>
            </div>
          </div>
        </div>
      )}
      {approval && (
        <div className="cm-approve-backdrop" onMouseDown={() => { approval.resolve(false); setApproval(null); }}>
          <div className="cm-approve" onMouseDown={(e) => e.stopPropagation()}>
            <div className="cm-approve-title">
              {approval.call.name === "bash" || approval.call.name === "bash_bg"
                ? t("cmApproveBash")
                : t("cmApproveWrite")}
            </div>
            <pre className="cm-approve-cmd">{toolSummary(approval.call)}</pre>
            {(approval.call.name === "bash" || approval.call.name === "bash_bg") && (
              <div className="cm-approve-detail">{String(approval.call.args.command ?? "")}</div>
            )}
            {approval.call.name === "web_download" && (
              <div className="cm-approve-detail">{String(approval.call.args.url ?? "")}</div>
            )}
            {(approval.call.name === "edit_file" || approval.call.name === "multi_edit") && (() => {
              const edits = argEdits(approval.call.args);
              // Multi-spot edit → group each change; single edit → one diff.
              const groups =
                edits.length > 0
                  ? edits.map((e) => ({ old: e.old_string, new: e.new_string }))
                  : [{ old: argOld(approval.call.args), new: argNew(approval.call.args) }];
              return (
                <pre className="cm-diff cm-approve-diff">
                  {groups.flatMap((g, ei) => [
                    ...(groups.length > 1
                      ? [
                          <div key={`h${ei}`} className="cm-dl ctx">
                            <span className="cm-dl-mark"> </span>
                            {`— ${t("cmEditN").replace("{n}", String(ei + 1))} —`}
                          </div>,
                        ]
                      : []),
                    ...diffLines(g.old, g.new).rows.map((l, i) => (
                      <div key={`${ei}-${i}`} className={`cm-dl ${l.kind}`}>
                        <span className="cm-dl-mark">{l.kind === "add" ? "+" : l.kind === "del" ? "-" : " "}</span>
                        {l.text}
                      </div>
                    )),
                  ])}
                </pre>
              );
            })()}
            {approval.call.name === "write_file" && (
              <pre className="cm-diff cm-approve-diff">
                {argContent(approval.call.args).split("\n").slice(0, 40).map((text, i) => (
                  <div key={i} className="cm-dl add">
                    <span className="cm-dl-mark">+</span>
                    {text}
                  </div>
                ))}
              </pre>
            )}
            <div className="cm-approve-actions">
              <button
                className="cm-allow-session"
                onClick={() => {
                  sessionAllowsRef.current.add(allowKeyFor(approval.call));
                  approval.resolve(true);
                  setApproval(null);
                }}
              >
                {approval.call.name === "bash" || approval.call.name === "bash_bg"
                  ? t("cmAllowAlwaysCmd", { cmd: allowKeyFor(approval.call).slice(4) })
                  : t("cmAllowAlwaysEdits")}
              </button>
              <button className="cm-deny" onClick={() => { approval.resolve(false); setApproval(null); }}>{t("cmDeny")}</button>
              <button className="cm-allow" onClick={() => { approval.resolve(true); setApproval(null); }}>{t("cmAllow")}</button>
            </div>
          </div>
        </div>
      )}

      {ask && (
        <div className="cm-approve-backdrop">
          <div className="cm-ask" onMouseDown={(e) => e.stopPropagation()}>
            <div className="cm-ask-label">{t("cmAskUser")}</div>
            <div className="cm-ask-question">{ask.question}</div>
            <div className="cm-ask-options">
              {ask.options.map((opt, i) => (
                <button
                  key={i}
                  className="cm-ask-opt"
                  onClick={() => { ask.resolve(opt); setAsk(null); }}
                >
                  <span className="cm-ask-opt-key">{i + 1}</span>
                  <span className="cm-ask-opt-text">{opt}</span>
                </button>
              ))}
            </div>
            <div className="cm-ask-custom">
              <input
                className="cm-ask-input"
                placeholder={t("cmAskCustom")}
                value={askText}
                onChange={(e) => setAskText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && askText.trim()) { ask.resolve(askText.trim()); setAsk(null); }
                }}
              />
              <button
                className="cm-ask-go"
                disabled={!askText.trim()}
                onClick={() => { ask.resolve(askText.trim()); setAsk(null); }}
              >→</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
