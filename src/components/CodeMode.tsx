import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { RECENCY_LABEL, recencyGroups } from "../lib/recency";
import { ImagePreview } from "./ImagePreview";
import { resolveToolFormat, type CallFormat } from "../lib/callFormat";
import { open } from "@tauri-apps/plugin-dialog";
import { agentLang, useI18n, type TKey } from "../lib/i18n";
import { watchContentHeight } from "../lib/autoGrow";
/** Matches `.cm-input` max-height. */
const CM_COMPOSER_MAX_H = 200;
import { effortLabel, intensityOf, thinkTabActive } from "../lib/effort";
import { diffLines } from "../lib/diff";
import {
  activityOf,
  deniedCount,
  failedCount,
  segmentsOf,
  workSummary,
  type DiffCounter,
  type WorkKind,
} from "../lib/toolGroups";
import { liveScan, type LiveScan } from "../lib/liveCall";
import { useChoice, useConfirm } from "./ConfirmModal";
import { BgTasksPanel } from "./BgTasksPanel";
import { groupSessionsByWorkspace, workspaceName } from "../lib/sessionGroups";
import { keptRestore, toTurnChange, type TurnChange } from "../lib/turnChanges";
import { TurnChangesCard } from "./TurnChangesCard";
import { BUILTIN_SKILLS } from "../lib/skills";
import { copyToClipboard } from "../lib/clipboard";
import { cleanTitle } from "../lib/voiceText";
import { DiffView } from "./DiffView";
import { Icon } from "./Icon";
import { PermModePicker } from "./PermModePicker";
import { SelectBar, SelectCheck, SelectToggle, selectClick, useMultiSelect } from "./MultiSelect";
import { Markdown } from "./Markdown";
import {
  agentBgAll,
  agentSetSession,
  agentDlList,
  type AgentDlInfo,
  agentCheckpointBegin,
  agentCheckpointRevertTo,
  agentCheckpointChanges,
  agentCheckpointRevertFile,
  agentRestoreFile,
  type CpChange,
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
  pickAttachmentFile,
  readAttachment,
  isVisionImagePath,
  type Attachment,
  type AgentBgInfo,
  codeSessionDelete,
  codeSessionFind,
  codeStepTextGet,
  codeStepTextPut,
  codeSessionList,
  codeSessionSearch,
  codeSessionLoad,
  codeSessionSave,
  sessionStatsGet,
  sessionStatsSet,
  type ChatMessage,
  type CodeSessionMeta,
  type ModelInfo,
  generate,
  exportTextFile,
} from "../lib/ipc";
import { buildCodeTrace } from "../lib/codeTrace";
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
  type StuckState,
  type ThinkMode,
  type ToolCall,
  type ToolStep,
  replayableTail,
} from "../lib/agentLoop";
import { PERM_MODES, isFileEdit, isPermMode, permVerdict, type PermMode } from "../lib/permMode";
import { syncMcpServers } from "../lib/mcp";
import { localSave } from "../lib/localSave";
import { loadSkills } from "../lib/skillFiles";
import { loadMemoryIndex, sessionMemoryIndex } from "../lib/memoryFiles";
import { homeDir } from "@tauri-apps/api/path";
import { fmtBytes } from "../lib/fmt";
import { JumpBottom } from "./JumpBottom";
import { PrefillRing } from "./PrefillRing";
import { ContextRing } from "./ContextRing";

interface CodeMsg {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Attached image paths (vision models) shown in the user bubble. */
  images?: string[];
  /** All attachments (docs + images) shown as chips in the user bubble. */
  attachments?: { name: string; kind: string; path?: string }[];
  /** The exact message tail the turn ended with, as the model saw it — kept so
   *  the NEXT turn continues from it verbatim. Only the newest assistant turn
   *  holds one; older turns drop it, since only the tail is ever replayed. */
  prompt?: ChatMessage[];
  /** Reasoning shown before the final answer (collapsible). */
  thinking?: string;
  /** Live reasoning streaming for the in-flight step. */
  liveThinking?: string;
  steps: ToolStep[];
  /** The agent's current task plan (todo list), updated in place. */
  plan?: PlanItem[];
  /** Words the model said with a call that has no step card (update_plan),
   *  after how many steps — kept for the session's trace. */
  said?: { after: number; text: string }[];
  /** Sent in plan mode: its answer is a plan to carry out or keep working on. */
  planned?: boolean;
  /** Context was auto-compacted during this turn. */
  compacted?: boolean;
  /** The turn paused at the step limit (offer a Continue button). */
  paused?: boolean;
  /** Checkpoint opened before this user message's turn — enables rewind. */
  checkpointId?: number;
  /** Assistant turns: every file the turn changed (the card under its answer). */
  changes?: TurnChange[];
  /** The checkpoint those changes are undone through, while the app holds it. */
  changesCp?: number;
}

const THINK_MODES: ThinkMode[] = ["off", "normal", "deep"];
/** thinkMode → the native rung it maps onto, for a model whose ladder happens
 *  to carry those names. Only a starting point now: a model with its own
 *  ladder shows that ladder, however many rungs it has. */
const EFFORT_OF: Partial<Record<ThinkMode, string>> = {
  low: "low",
  normal: "medium",
  deep: "xhigh",
};


const uid = () => Math.random().toString(36).slice(2);


const TOOL_ICON: Record<string, string> = {
  read_file: "M9 2h6l4 4v14a0 0 0 0 1 0 0H5V2z",
  multi_read: "M7 3h5l3 3v10H7zM4 7v12h9M15 6h-3V3",
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
  search_history: "M3 12a9 9 0 109-9 9 9 0 00-6.4 2.7L3 8M3 4v4h4M12 7v5l3.5 2",
  read_history: "M3 12a9 9 0 109-9 9 9 0 00-6.4 2.7L3 8M3 4v4h4M9 12h7M9 15h5",
  bash: "M4 5l6 7-6 7M13 19h7",
  bash_bg: "M4 5l6 7-6 7M13 5h7M13 12h7M13 19h7",
  bg_output: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 7v5l3 3",
  bg_kill: "M12 3a9 9 0 100 18 9 9 0 000-18zM9 9l6 6M15 9l-6 6",
  bg_input: "M3 6h18v12H3zM7 10h.01M11 10h.01M15 10h.01M7 14h10",
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
  browser_tabs: "M3 8h18v12H3zM3 8V5h7l2 3",
};

function toolSummary(call: ToolCall): string {
  const a = call.args as Record<string, string>;
  // Tools that exist only at runtime (history, skills, MCP) aren't in the
  // native name union, so they are named before the switch.
  if ((call.name as string) === "search_history") {
    return `history? ${a.query ?? a.session ?? ""}`;
  }
  if ((call.name as string) === "read_history") {
    if (a.step) return `history step ${a.step}`;
    return a.turn ? `history #${a.turn}` : "history (whole session)";
  }
  switch (call.name) {
    case "read_file":
      return `read ${argPath(call.args) || "?"}`;
    case "multi_read": {
      // Named as the model wrote it — the tool takes the list under any of
      // these, so the card must read them all too.
      const raw = call.args.paths ?? call.args.files ?? call.args.path;
      const paths = (Array.isArray(raw) ? raw.map(String) : typeof raw === "string" ? raw.split(/\s*,\s*/) : [])
        .map((p) => p.trim())
        .filter(Boolean);
      // Counted, the way multi_edit counts its edits: several files have no
      // one name to show.
      return paths.length > 1 ? `read ×${paths.length}` : `read ${paths[0] ?? "?"}`;
    }
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
    case "bg_input": {
      const keys = Array.isArray(call.args.keys) ? (call.args.keys as unknown[]).map(String) : [];
      const typed = a.text ? ` ${a.text}` : "";
      return `input #${a.id ?? "?"}${typed}${keys.length ? ` [${keys.join(" ")}]` : ""}`;
    }
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
    case "browser_tabs":
      return `tabs ${a.action ?? "list"}${a.index != null ? ` ${a.index}` : ""}`;
    default:
      return call.name;
  }
}

/** File tools name what is happening to the file: writing → wrote,
 *  editing → edited. */
const FILE_VERBS: Record<string, [string, string]> = {
  write_file: ["writing", "wrote"],
  edit_file: ["editing", "edited"],
  multi_edit: ["editing", "edited"],
};

/** A step card's (and the run bar's) summary: `toolSummary`, with a file
 *  tool's verb for where the step is. */
function stepSummary(step: ToolStep): string {
  const verbs = FILE_VERBS[step.call.name];
  const word = verbs && (step.status === "running" ? verbs[0] : step.status === "done" ? verbs[1] : null);
  if (!word) return toolSummary(step.call);
  // The path is the first argument most formats write, but not all (Gemma
  // orders them by name): until it arrives there is nothing to name.
  if (!argPath(step.call.args)) return `${word} …`;
  return toolSummary(step.call).replace(/^\S+/, word);
}

/** A saved session keeps no card for a call that had not been accepted to
 *  run, and no half-written arguments on the rest. */
function forDisk(msgs: CodeMsg[]): CodeMsg[] {
  if (!msgs.some((m) => m.steps?.some((st) => st.live))) return msgs;
  return msgs.map((m) =>
    m.steps?.some((st) => st.live)
      ? { ...m, steps: m.steps.filter((st) => !st.live?.pending).map((st) => (st.live ? { ...st, live: undefined } : st)) }
      : m,
  );
}

/** A live card's rows are one fixed height, so only those around the line
 *  the model is at are drawn — a file can be thousands of lines long. */
const LIVE_ROW_H = 19;
const LIVE_OVERSCAN = 60;

/** A write or edit as it is being written, drawn the way Canvas draws a patch
 *  being applied: the whole file, the changes made so far in place, and a
 *  head on the line the model is at — the block it is copying out to
 *  replace, then the replacement as it arrives, then the next block. The
 *  view follows the head until the user scrolls. */
function LiveScanView({ scan }: { scan: LiveScan }) {
  const { t } = useI18n();
  const ref = useRef<HTMLDivElement | null>(null);
  const [follow, setFollow] = useState(true);
  const [win, setWin] = useState(0);
  const head = scan.head ?? scan.rows.length - 1;
  useEffect(() => {
    if (follow) setWin(Math.max(0, head - LIVE_OVERSCAN));
  }, [head, follow]);
  useEffect(() => {
    const el = ref.current;
    if (!follow || !el || scan.head === null) return;
    const target = Math.max(0, scan.head * LIVE_ROW_H + 8 - el.clientHeight / 2);
    if (Math.abs(el.scrollTop - target) > 2) el.scrollTo({ top: target, behavior: "smooth" });
  }, [scan, follow, win]);
  const onScroll = () => {
    // Off-follow, the drawn window tracks what the user is looking at.
    if (follow || !ref.current) return;
    const first = Math.floor(ref.current.scrollTop / LIVE_ROW_H);
    setWin((w) => {
      const next = Math.max(0, first - LIVE_OVERSCAN);
      return Math.abs(next - w) > LIVE_OVERSCAN / 2 ? next : w;
    });
  };
  const end = Math.min(scan.rows.length, win + LIVE_OVERSCAN * 2);
  return (
    <div
      className="cm-step-body cm-live-scan"
      ref={ref}
      onWheel={() => setFollow(false)}
      onTouchMove={() => setFollow(false)}
      onScroll={onScroll}
    >
      <div className="cm-diff">
        {win > 0 && <div style={{ height: win * LIVE_ROW_H }} aria-hidden="true" />}
        {scan.rows.slice(win, end).map((r, k) => {
          const i = win + k;
          return (
            <div
              key={i}
              className={`cm-dl ${r.kind === "pending" ? "ctx cvp-pending" : r.kind}${r.found ? " cm-dl-found" : ""}${i === scan.head ? " cvp-scanhead" : ""}`}
            >
              <span className="cm-dl-mark">{r.kind === "add" ? "+" : r.kind === "del" ? "-" : " "}</span>
              {r.text}
            </div>
          );
        })}
        {scan.rows.length > end && <div style={{ height: (scan.rows.length - end) * LIVE_ROW_H }} aria-hidden="true" />}
      </div>
      {!follow && (
        <button className="cvp-follow" onClick={() => setFollow(true)}>
          {t("canvasFollow")}
        </button>
      )}
    </div>
  );
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
function StepCard({
  step,
  sessionId,
  onPreview,
  liveOpen = true,
}: {
  step: ToolStep;
  sessionId?: string | null;
  onPreview?: (path: string) => void;
  /** A live card opens while its call is written (Settings → Code). */
  liveOpen?: boolean;
}) {
  const { t } = useI18n();
  const live = step.live;
  const [open, setOpen] = useState(step.status === "error" || (!!live && liveOpen));
  // A live card opens while its call is written — when the setting says so —
  // and folds away once the call has run, unless the user has opened or
  // closed it by hand.
  const touched = useRef(false);
  const wasLive = useRef(!!live);
  useEffect(() => {
    if (touched.current) return;
    if (live) {
      wasLive.current = true;
      if (liveOpen) setOpen(true);
    } else if (wasLive.current && step.status !== "running") {
      wasLive.current = false;
      setOpen(step.status === "error");
    }
  }, [live, liveOpen, step.status]);
  // Opened, the card shows exactly what the model was given for this step.
  // The session keeps only a trimmed copy (`step.result`); the model's is
  // fetched while the card is open and let go when it closes, so a session
  // never holds every large read at once.
  const [modelText, setModelText] = useState<string | null>(null);
  useEffect(() => {
    if (!open) {
      setModelText(null);
      return;
    }
    if (!step.fullText || !sessionId) return;
    let live = true;
    codeStepTextGet(sessionId, step.id)
      .then((text) => {
        if (live && text != null) setModelText(text);
      })
      .catch((e) => console.error("step text load failed", e));
    return () => {
      live = false;
    };
  }, [open, step.fullText, step.id, sessionId]);
  const diff = step.diff;
  const liveS = useMemo(() => (live ? liveScan(live) : null), [live]);
  const hasImage = !!step.image && step.status === "done";
  const hasBody = !!(step.result || diff || (liveS && liveS.rows.length > 0));
  const meta = stepMeta(step, t("cmLines"));
  // A command that ran but exited non-zero: "done" (the result went back to
  // the model) but visually a failure — a green check here read as "ls
  // worked" when cmd.exe had actually rejected the command.
  const cmdFailed = step.status === "done" && meta?.tone === "warn";
  // Compute the diff once; the +N/−M badge uses the EXACT totals, never the
  // render-capped rows, so big edits are counted correctly.
  const d = useMemo(() => (diff ? diffLines(diff.before, diff.after) : null), [diff]);
  // Exact totals when the contents were capped for the card; otherwise the
  // ones just computed, which are exact anyway.
  const added = step.diffCounts?.added ?? d?.added ?? 0;
  const removed = step.diffCounts?.removed ?? d?.removed ?? 0;
  // Clicking an image step opens the preview directly; otherwise toggle the body.
  const onHead = () => {
    if (hasImage && step.image) onPreview?.(step.image);
    else if (hasBody) {
      touched.current = true;
      setOpen((o) => !o);
    }
  };
  return (
    <div className={`cm-step ${step.status}${cmdFailed ? " cmd-failed" : ""}`}>
      <button className="cm-step-head" onClick={onHead}>
        <svg className="cm-step-ico" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d={TOOL_ICON[step.call.name] ?? "M4 6h16M4 12h16M4 18h16"} />
        </svg>
        <span className="cm-step-sum">{stepSummary(step)}</span>
        {liveS ? (
          <span className="cm-step-diffstat">
            <em className="plus">+{liveS.added}</em>
            <em className="minus">-{liveS.removed}</em>
          </span>
        ) : (
          d &&
          step.status === "done" && (
            <span className="cm-step-diffstat">
              <em className="plus">+{added}</em>
              <em className="minus">-{removed}</em>
            </span>
          )
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
      {open && hasBody && liveS && <LiveScanView scan={liveS} />}
      {open && hasBody && !liveS && (
        <div className="cm-step-body">
          {d && diff ? (
            <DiffView
              before={diff.before}
              after={diff.after}
              total={added + removed}
              // A big file's card holds a capped copy; the whole of it was
              // kept aside when the step ran.
              loadFull={
                step.fullDiff && sessionId
                  ? () =>
                      codeStepTextGet(sessionId, `${step.id}:diff`).then((text) =>
                        text ? (JSON.parse(text) as { before: string; after: string }) : null,
                      )
                  : undefined
              }
            />
          ) : (
            <pre className="cm-out">{modelText ?? step.result ?? ""}</pre>
          )}
        </div>
      )}
    </div>
  );
}

/** Collapsible reasoning panel. Streams open while `live`; collapsed once done. */
/** Each Code session's permission mode, by session id. */
const PERM_MODES_KEY = "chaty.code.permModes";

function loadPermModes(): Record<string, PermMode> {
  try {
    const raw = JSON.parse(localStorage.getItem(PERM_MODES_KEY) ?? "{}") as Record<string, unknown>;
    return Object.fromEntries(Object.entries(raw).filter((e): e is [string, PermMode] => isPermMode(e[1])));
  } catch {
    return {};
  }
}

function savePermModes(m: Record<string, PermMode>): void {
  try {
    localStorage.setItem(PERM_MODES_KEY, JSON.stringify(m));
  } catch {
    /* storage off: the mode lasts as long as the app is open */
  }
}

/** Marks a code run as in flight so an interrupted one can be reported. */
const RUN_INFLIGHT_KEY = "chaty.code.runInflight";

function ThinkPanel({ text, live, label }: { text: string; live?: boolean; label: string }) {
  const [open, setOpen] = useState(false);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  // Focus-follows-generation, as chat mode has had: while reasoning streams,
  // hold a short window pinned to the newest line and fade what is above it.
  // Code mode let the whole trace grow instead, so a long think pushed the
  // steps below it off the screen and the newest words were wherever the page
  // happened to be scrolled. Opening it by hand releases the window.
  const focus = !!live && !open;
  useEffect(() => {
    if (focus && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [text, focus]);
  if (!text) return null;
  return (
    <div className={`cm-think ${live ? "live" : ""}`}>
      <button className="cm-think-head" onClick={() => setOpen((o) => !o)}>
        {live ? (
          <span className="cm-spin" />
        ) : (
          <span className={`cm-think-caret ${open ? "open" : ""}`}>
            <Icon name="chevron-right" size={11} strokeWidth={2} />
          </span>
        )}
        <span className="cm-think-label">{label}</span>
      </button>
      {(open || live) && (
        <div ref={bodyRef} className={`cm-think-body ${focus ? "focus" : ""}`}>
          {text}
        </div>
      )}
    </div>
  );
}

/** Lines a step's diff adds and removes — the card's own count when the
 *  contents were capped, else counted once per diff and kept. */
const diffCountCache = new WeakMap<object, { added: number; removed: number }>();
const countStepDiff: DiffCounter = (s) => {
  if (s.diffCounts) return s.diffCounts;
  if (!s.diff) return null;
  let c = diffCountCache.get(s.diff);
  if (!c) {
    const d = diffLines(s.diff.before, s.diff.after);
    c = { added: d.added, removed: d.removed };
    diffCountCache.set(s.diff, c);
  }
  return c;
};

const SUM_KEY: Record<WorkKind, TKey> = {
  read: "sumRead",
  create: "sumCreate",
  edit: "sumEdit",
  command: "sumCommand",
  search: "sumSearch",
  websearch: "sumWebSearch",
  fetch: "sumFetch",
  browser: "sumBrowser",
  validate: "sumValidate",
  image: "sumImage",
  other: "sumOther",
};

/** One kind of work as its words: the count, singular or plural (en carries
 *  both, "one|many"). */
function sumPhrase(t: (k: TKey, v?: Record<string, string | number>) => string, kind: WorkKind, n: number): string {
  const forms = t(SUM_KEY[kind], { n }).split("|");
  return forms.length > 1 && n !== 1 ? forms[1] : forms[0];
}

/** A run of tool calls folded to one line (Settings → Code → collapse tool
 *  calls). While it is the run going on, the line says what its latest call
 *  is doing; once the model has spoken again, or the turn is over, what the
 *  run did — "Read 3 files, ran 2 commands, edited 1 file +12 −3". Opened,
 *  the steps as they always show. */
function ToolRun({
  steps,
  activity,
  doneLabel,
  children,
}: {
  steps: ToolStep[];
  /** What the run going on is doing now; absent for a run that is over. */
  activity?: Activity | null;
  /** The line of a run that is over and called no tool — only thought. */
  doneLabel?: string;
  children?: React.ReactNode;
}) {
  const { t, lang } = useI18n();
  const [open, setOpen] = useState(false);
  const active = !!activity;
  const openable = steps.length > 0 || !!children;
  let label: React.ReactNode;
  if (activity) {
    label = <ActivityLabel activity={activity} />;
  } else if (doneLabel) {
    label = <span className="cm-run-label">{doneLabel}</span>;
  } else {
    const items = workSummary(steps, countStepDiff);
    const failed = failedCount(steps);
    const denied = deniedCount(steps);
    const sep = lang === "zh" ? "，" : ", ";
    // The tail after what was done: "1 failed", "2 not run" — leading the
    // line when nothing ran at all.
    const tails = [
      ...(failed > 0 ? [{ cls: "cm-run-failed", text: t("sumFailed", { n: failed }) }] : []),
      ...(denied > 0 ? [{ cls: "cm-run-denied", text: t("sumDenied", { n: denied }) }] : []),
    ];
    label = (
      <span className="cm-run-label">
        {items.map((it, i) => {
          let words = sumPhrase(t, it.kind, it.n);
          if (i > 0 && lang !== "zh") words = words.charAt(0).toLowerCase() + words.slice(1);
          return (
            <span key={it.kind}>
              {i > 0 && sep}
              {words}
              {(it.added || it.removed) ? (
                <span className="cm-step-diffstat cm-run-diff">
                  {it.added ? <em className="plus">+{it.added}</em> : null}
                  {it.removed ? <em className="minus">-{it.removed}</em> : null}
                </span>
              ) : null}
            </span>
          );
        })}
        {tails.map((x, i) => (
          <span key={x.cls} className={x.cls}>
            {items.length > 0 || i > 0 ? sep : ""}
            {x.text}
          </span>
        ))}
      </span>
    );
  }
  return (
    <div className={`cm-run${active ? " active" : ""}${open ? " open" : ""}`}>
      <button
        type="button"
        className="cm-run-head"
        onClick={() => openable && setOpen((o) => !o)}
        aria-expanded={openable ? open : undefined}
        disabled={!openable}
      >
        {/* Keyed by the state, not its words: a new state fades in, while
            the same one updating — a path arriving, the prompt's percent —
            changes in place. */}
        <span key={activity ? `a:${activity.key}` : doneLabel ? "done:thought" : "done"} className="cm-run-state">
          {label}
        </span>
        {openable && (
          <span className={`cm-run-caret${open ? " open" : ""}`}>
            <Icon name="chevron-right" size={11} strokeWidth={2} />
          </span>
        )}
      </button>
      {open && openable && <div className="cm-run-body">{children}</div>}
    </div>
  );
}

type Activity = { key: string; x?: string };

/** What is happening now, in words: "Running npm test", "Thinking". */
function ActivityLabel({ activity }: { activity: Activity }) {
  const { t } = useI18n();
  const x = activity.x ? (activity.x.length > 80 ? activity.x.slice(0, 80) + "…" : activity.x) : "…";
  return <span className="cm-run-label">{t(activity.key as TKey, { x })}</span>;
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


export function CodeMode({
  model,
  active,
  headSlot,
  railW,
  onRailResize,
  onRailReset,
  maxSteps,
  bashTimeout,
  ragTopK,
  temperature,
  sampling,
  thinkBudget = 0,
  maxGenTokens = 0,
  toolFormat = "auto",
  toolFallback = "xml",
  autoRunReadOnly = true,
  skills = [],
  disabledSkills = [],
  memoryEnabled = true,
  groupByWorkspace = false,
  liveCardsOpen = true,
  collapseTools = false,
  allowedCommands = [],
  sendKey = "enter",
  autoTitle = true,
  onExportable,
  actionsRef,
}: {
  model: ModelInfo | null;
  active: boolean;
  /** The window's top row, over this panel: the header renders there. */
  headSlot: HTMLElement | null;
  /** The sidebar's width, shared with chat and the image studio so the
   *  top row's mode switch stays put when the mode changes. */
  railW: number;
  onRailResize: (e: React.PointerEvent) => void;
  onRailReset: () => void;
  /** Generate a session title with the model after the first turn (Settings → General). */
  autoTitle?: boolean;
  /** Max agent steps per turn (Settings → Code). */
  maxSteps?: number;
  /** Knowledge-base excerpts `search_docs` may return. */
  ragTopK?: number;
  /** Default bash timeout in seconds (Settings → Code). */
  bashTimeout?: number;
  /** Sampling temperature for agent steps (Settings → Code). */
  temperature?: number;
  /** Top-p / top-k / min-p / repeat penalty from Settings → Sampling. */
  sampling?: { topP: number; topK: number; minP: number; repeatPenalty: number; stop?: string[] };
  /** Hard per-round think-token ceiling, 0 = auto (Settings → Code). */
  thinkBudget?: number;
  /** Per-round generation budget in tokens, 0 = none of its own (Settings → Sampling → Max length). */
  maxGenTokens?: number;
  /** How the model writes tool calls: "auto" = its template's own format
   *  (Settings → Code). */
  toolFormat?: "auto" | CallFormat;
  /** The format for a model whose template names none (Settings → Code). */
  toolFallback?: "xml" | "json";
  /** Auto-run obviously read-only bash commands without asking (Settings → Code). */
  autoRunReadOnly?: boolean;
  /** User-defined skills: /name inserts the prompt template (Settings → Code). */
  skills?: { name: string; prompt: string }[];
  /** Names of built-in skills the user turned off (Settings → Code). */
  disabledSkills?: string[];
  /** Project memory on (default): load the index + offer `remember`
   *  (Settings → Code). Off ⇒ neither, byte-identical prompt. */
  memoryEnabled?: boolean;
  /** Group the session rail by workspace (Settings → Code). */
  groupByWorkspace?: boolean;
  /** Settings → Code: a write/edit card is open while its call is written. */
  liveCardsOpen?: boolean;
  /** Settings → Code: the tool calls between two things the model says fold
   *  into one line (see ToolRun). */
  collapseTools?: boolean;
  /** Persistent command prefixes that never need approval (Settings → Code). */
  allowedCommands?: string[];
  /** Composer send shortcut (Settings → General). */
  sendKey?: "enter" | "modEnter";
  /** The top row's export button: handed the open session's trace export
   *  while this mode is showing a session with turns, null otherwise. */
  onExportable?: (exportTrace: (() => Promise<boolean>) | null) => void;
  /** What the command palette does here: "New" is a new Code session. */
  actionsRef?: React.MutableRefObject<{ newSession: () => void } | null>;
}) {
  const { t, lang } = useI18n();
  const confirm = useConfirm();
  const choose = useChoice();
  const [sessions, setSessions] = useState<CodeSessionMeta[]>([]);
  const [sid, setSid] = useState<string>(() => uid());
  /** The session on screen, for callbacks that outlive a render: a run's
   *  numbers belong to the session it started in, not whichever is open. */
  const sidRef = useRef(sid);
  sidRef.current = sid;
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<CodeMsg[]>([]);
  const [input, setInput] = useState("");
  /** The composer follows its content — see `watchContentHeight`. Without it a
   *  one-row textarea keeps a single line's height and scrolls the rest, which
   *  at a larger UI scale is the placeholder itself. */
  const inputRef = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = inputRef.current;
    const row = el?.parentElement;
    if (!el || !row) return;
    return watchContentHeight(el, row, CM_COMPOSER_MAX_H);
  }, [input]);
  // Files the user attached to the next Code turn — same as chat: documents
  // (PDF/Word/Excel/text/code, extracted to text), and images (vision models
  // see the pixels; text-only models get OCR text).
  const [codeAttachments, setCodeAttachments] = useState<Attachment[]>([]);
  /** Other sessions the user referenced with @ for the next turn. */
  const [refSessions, setRefSessions] = useState<{ id: string; title: string }[]>([]);
  const [attachErr, setAttachErr] = useState("");
  // Full-size image preview (screenshot / view_image steps).
  const [previewImg, setPreviewImg] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  // Each session's permission mode, its own and kept across restarts; a
  // session never set is in auto.
  const [permModes, setPermModes] = useState<Record<string, PermMode>>(loadPermModes);
  const modesRef = useRef(permModes);
  modesRef.current = permModes;
  const mode: PermMode = permModes[sid] ?? "auto";
  /** The session a running turn belongs to — whose mode its calls answer to. */
  const runSidRef = useRef<string | null>(null);
  /** The model exposes a native reasoning-effort ladder (Qwen3.8) — the think
   *  switch then shows the model's own rungs instead of Chaty's intensities. */
  const effortLevels = useMemo(() => model?.effortLevels ?? [], [model?.effortLevels]);
  const nativeEffort = effortLevels.length > 0;
  /** Set when a turn pauses on a loop the model could not get out of; consumed
   *  by the next turn so "Continue" picks the escape up where it stopped. */
  const stuckRef = useRef<StuckState | null>(null);
  const [thinkMode, setThinkMode] = useState<ThinkMode>(() => {
    const v = localStorage.getItem("chaty.code.think");
    return v === "off" || v === "low" || v === "normal" || v === "deep" ? v : "normal";
  });
  /** The rung chosen on the model's own ladder. Kept apart from `thinkMode`
   *  because that is Chaty's intensity, which also survives models with no
   *  ladder at all. */
  const [codeEffort, setCodeEffort] = useState<string>(() => {
    try {
      return localStorage.getItem("chaty.code.effort") || "";
    } catch {
      return "";
    }
  });
  /** The rung to show as chosen: the stored one while this model still offers
   *  it, otherwise whatever Chaty's current intensity maps onto, otherwise the
   *  ladder's second rung — which is where the old three-rung default sat. */
  const rung = useMemo(() => {
    if (effortLevels.includes(codeEffort)) return codeEffort;
    const mapped = EFFORT_OF[thinkMode];
    if (mapped && effortLevels.includes(mapped)) return mapped;
    return effortLevels[Math.min(1, effortLevels.length - 1)] ?? "";
  }, [effortLevels, codeEffort, thinkMode]);
  const [approval, setApproval] = useState<{ call: ToolCall; resolve: (ok: boolean | "plan") => void } | null>(null);
  /** The model is writing a tool call: the tool's name once it shows, "" until
   *  then, null when it is not (see onCalling). */
  const [callingTool, setCallingTool] = useState<string | null>(null);
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
  const [stats, setStats] = useState<{ tokens: number; tps: number } | null>(null);
  const [ctxUsed, setCtxUsed] = useState(0);
  /** The context window `ctxUsed` was counted against: the ring is the last
   *  turn's, and stays as it was through a model or context-size change until
   *  the next turn counts again. */
  const [ctxTotal, setCtxTotal] = useState(0);
  /** Messages typed while the agent was running — auto-sent one by one after. */
  const [queue, setQueue] = useState<string[]>([]);
  /** The plan the user chose to keep working on — its choices put away. */
  const [keptPlanning, setKeptPlanning] = useState<string | null>(null);
  /** The turn in flight, settled once it has wound down (see sendWhileRunning). */
  const turnDone = useRef<Promise<void> | null>(null);
  // Whether a turn runs, as of now rather than as of the last render: a
  // message sent right after a stop must not be turned away by a `running`
  // that rendering has not caught up with.
  const runningRef = useRef(false);
  const setRunningNow = (v: boolean) => {
    runningRef.current = v;
    setRunning(v);
  };
  const sendRef = useRef(send);
  sendRef.current = send;
  const queueRef = useRef(queue);
  queueRef.current = queue;
  /** Background jobs (dev servers …), running and finished — header pill and
   *  the tasks panel it opens. */
  const [bgJobs, setBgJobs] = useState<AgentBgInfo[]>([]);
  const [showBg, setShowBg] = useState(false);
  const bgPillRef = useRef<HTMLButtonElement | null>(null);
  /** Workspace groups folded shut in the session rail (by path; "" = none). */
  const [collapsedWs, setCollapsedWs] = useState<Set<string>>(() => new Set());
  /** Selecting several sessions to delete them together. */
  const sessSel = useMultiSelect();
  /** The sidebar's search: titles as typed, the sessions' words from the
   *  database (debounced), as the chat's sidebar does. */
  const [sessQuery, setSessQuery] = useState("");
  const [sessMatches, setSessMatches] = useState<Set<string>>(() => new Set());
  useEffect(() => {
    const q = sessQuery.trim();
    if (q.length < 2) {
      setSessMatches(new Set());
      return;
    }
    let cancelled = false;
    const id = window.setTimeout(() => {
      codeSessionFind(q)
        .then((ids) => {
          if (!cancelled) setSessMatches(new Set(ids));
        })
        .catch(() => {});
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(id);
    };
  }, [sessQuery]);
  /** Active background downloads (header progress badge). */
  const [downloads, setDownloads] = useState<AgentDlInfo[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const signalRef = useRef<AgentSignal | null>(null);
  // Connect enabled MCP servers — their tools join the registry before the
  // first agent turn builds its prompt. This lived in the attachment
  // thumbnail's mount effect, where a session without a picture in it never
  // reached the call at all and every picture in one reconnected the lot.
  useEffect(() => {
    void syncMcpServers().catch(() => {});
  }, []);
  // Left behind when a run does not reach its own cleanup — i.e. the page went
  // away underneath it. Reported once on the next mount, then cleared.
  useEffect(() => {
    if (!localStorage.getItem(RUN_INFLIGHT_KEY)) return;
    localStorage.removeItem(RUN_INFLIGHT_KEY);
    setMsgs((cur) =>
      cur.length
        ? [...cur, { id: uid(), role: "assistant" as const, text: t("cmRunInterrupted"), steps: [] }]
        : cur,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const prevSidRef = useRef<string | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);
  const showJumpRef = useRef(false);
  const [showJump, setShowJump] = useState(false);
  const bodyRef = useRef({ msgs, workspace, sid });
  bodyRef.current = { msgs, workspace, sid };
  // The memory index each session started with — see sessionMemoryIndex.
  const memoryPinsRef = useRef(new Map<string, string>());
  // The approve callback closes over send()-time state — read the mode
  // through a ref so a change MID-RUN takes effect immediately.
  const modeRef = useRef(mode);
  modeRef.current = mode;

  // Session-scoped "always allow" grants (from the approval dialog's third
  // button). Keyed "cmd:<two-token prefix>" or "tool:<name>"; reset per session.
  const sessionAllowsRef = useRef<Set<string>>(new Set());
  const allowedCommandsRef = useRef(allowedCommands);
  allowedCommandsRef.current = allowedCommands;
  const autoRunReadOnlyRef = useRef(autoRunReadOnly);
  autoRunReadOnlyRef.current = autoRunReadOnly;
  /** Prompt-processing progress (0..1) for the current step, null = not prefilling. */
  const [prefill, setPrefill] = useState<number | null>(null);

  /** What the mode makes of a call, with this session's allowances. */
  const verdictFor = useCallback(
    (m: PermMode, call: ToolCall) =>
      permVerdict(m, call, {
        readOnlyAuto: autoRunReadOnlyRef.current,
        allowlist: allowedCommandsRef.current,
        sessionAllowed: sessionAllowsRef.current.has(allowKeyFor(call)),
        windows: IS_WINDOWS,
      }),
    [],
  );

  /** Switch the open session's permission mode, and keep it. A call of
   *  this session's turn waiting on the user is judged again under the new
   *  mode: switching to accept edits lets the edit on screen through, as
   *  turning bypass on always did. */
  const pickMode = useCallback(
    (m: PermMode) => {
      const at = sidRef.current;
      const next = { ...modesRef.current, [at]: m };
      modesRef.current = next;
      modeRef.current = m;
      setPermModes(next);
      savePermModes(next);
      setApproval((cur) => {
        if (!cur || runSidRef.current !== at) return cur;
        const v = verdictFor(m, cur.call);
        if (v === "ask") return cur;
        cur.resolve(v === "plan" ? "plan" : true);
        return null;
      });
    },
    [verdictFor],
  );
  const cycleMode = useCallback(() => {
    pickMode(PERM_MODES[(PERM_MODES.indexOf(modeRef.current) + 1) % PERM_MODES.length]);
  }, [pickMode]);
  /** `/bypass`: into bypass, or back to the mode it was entered from. */
  const preBypass = useRef<PermMode>("manual");
  const toggleBypass = useCallback(() => {
    if (modeRef.current === "bypass") pickMode(preBypass.current);
    else {
      preBypass.current = modeRef.current;
      pickMode("bypass");
    }
  }, [pickMode]);

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
      agentBgAll().then(setBgJobs).catch(() => {});
      agentDlList()
        .then((all) => setDownloads(all.filter((d) => !d.done)))
        .catch(() => {});
    };
    tick();
    const timer = setInterval(tick, running || showBg ? 1000 : 5000);
    return () => clearInterval(timer);
  }, [active, workspace, running, showBg]);

  // Background jobs belong to the session that started them: tell the agent
  // which session it is working for, and show that session's jobs.
  useEffect(() => {
    setBgJobs([]);
    agentSetSession(sid)
      .then(() => agentBgAll())
      .then(setBgJobs)
      .catch(() => {});
  }, [sid]);

  // Follow-the-stream is an *intent*, not a position: any upward wheel motion
  // releases it immediately (a distance check alone loses to the next stream
  // frame re-pinning the bottom before the user escapes the threshold), and
  // parking back at the bottom re-arms it.
  const followRef = useRef(true);
  /** How far the end is from view, and whether that is far enough to offer the
   *  way back. Recomputed from the CONTENT as well as from scrolling: a
   *  transcript shrinks — a thinking panel folds itself away, a live card is
   *  withdrawn — without any scroll event, and the button was left over a
   *  conversation whose end was already on screen. */
  const syncJump = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const jump = el.scrollHeight - el.scrollTop - el.clientHeight > 320;
    if (jump !== showJumpRef.current) {
      showJumpRef.current = jump;
      setShowJump(jump);
    }
  }, []);
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
      syncJump();
    };
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("scroll", onScroll);
    };
  }, [syncJump]);

  // BEFORE the browser paints: a session that opens at its end must never be
  // drawn anywhere else first. Done after the paint (useEffect), switching
  // sessions showed one frame at the top and then jumped — which is what it
  // looked like.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Switching re-arms the follow and drops the pill, which no scroll event
    // would do on its own.
    const switched = sid !== prevSidRef.current;
    prevSidRef.current = sid;
    if (switched) {
      followRef.current = true;
      showJumpRef.current = false;
      setShowJump(false);
    }
    if (followRef.current) el.scrollTop = el.scrollHeight;
    syncJump();
  }, [msgs, sid, syncJump]);

  // Whatever arrives late — a picture, a diff card, a code block that only
  // knows its height once it is laid out — keeps the view at the end while
  // the follow is armed. Without it, opening a session pinned the bottom of
  // the content as it was at that instant and everything that grew afterwards
  // pushed the end back out of sight.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const content = threadRef.current;
    if (!el || !content) return;
    const obs = new ResizeObserver(() => {
      if (followRef.current) el.scrollTop = el.scrollHeight;
      syncJump();
    });
    obs.observe(content);
    return () => obs.disconnect();
  }, [msgs.length > 0, syncJump]);

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
    codeSessionSave(id, title, ws, JSON.stringify(forDisk(next)))
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
  // The session the pending save is for: the latest to ask. A stopped turn's
  // step scheduled a save for its own session, and the new session's steps
  // inside those two seconds found the timer taken and were never saved.
  const persistIdRef = useRef("");
  const persistSoon = useCallback(
    (id: string) => {
      persistIdRef.current = id;
      if (persistTimerRef.current !== null) return;
      persistTimerRef.current = window.setTimeout(() => {
        persistTimerRef.current = null;
        const id = persistIdRef.current;
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
            // Enough to name the topic, within the title's small context.
            { role: "user", content: `${firstMsg.slice(0, 1200)}${model.thinkSwitch ? "\n/no_think" : ""}` },
          ],
          params: {
            temperature: 0.2,
            topP: 0.9,
            maxTokens: 512,
            think: model.supportsThinking && !model.thinkSwitch ? false : undefined,
            // The title's own cache: the system prompt every session shares
            // stays warm for the session's first turn.
            scratch: true,
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


  async function pickWorkspace() {
    if (running) return;
    const dir = await open({ directory: true });
    if (!dir || Array.isArray(dir)) return;
    try {
      const abs = await agentSetWorkspace(dir);
      // A session belongs to one workspace. One that already holds a
      // conversation keeps its own, and the new folder gets a session of its
      // own; an empty one — or one whose folder is gone — just takes it.
      if (msgs.length > 0 && workspace && abs !== workspace) newSession();
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
    // A pause belongs to the session it happened in. Carried over, the next
    // session's first turn opened with "the previous attempt was paused after
    // N identical calls…" and hotter sampling — the model then reasoned about
    // another session's steps (owner report: context bleeding across sessions).
    stuckRef.current = null;
    const fresh = uid();
    sidRef.current = fresh;
    setSid(fresh);
    setMsgs([]);
    setInput("");
    setRefSessions([]);
    setCtxUsed(0);
    setStats(null);
    sessionAllowsRef.current = new Set();
    void agentClearGrants().catch(() => {});
    setDirGrants([]);
  }

  if (actionsRef) actionsRef.current = { newSession };

  async function openSession(id: string) {
    if (running || id === sid) return;
    const raw = await codeSessionLoad(id).catch(() => null);
    if (raw == null) return;
    try {
      const parsed = forDisk(JSON.parse(raw) as CodeMsg[]);
      stuckRef.current = null;
      sidRef.current = id;
      setSid(id);
      // The draft survives a switch, but a reference to the session now open
      // would only quote it back to itself.
      setRefSessions((cur) => cur.filter((r) => r.id !== id));
      setMsgs(parsed);
      setCtxUsed(0);
      setStats(null);
      // The context ring this session last showed, kept with it — after a
      // restart too, and whichever model is loaded now, until its next turn.
      void sessionStatsGet("code", id)
        .then((raw) => {
          const kept = JSON.parse(raw || "{}") as { ctxUsed?: number; nCtx?: number };
          if (sidRef.current !== id || typeof kept.ctxUsed !== "number") return;
          setCtxUsed(kept.ctxUsed);
          setCtxTotal(kept.nCtx ?? 0);
        })
        .catch(() => {});
      sessionAllowsRef.current = new Set();
      void agentClearGrants().catch(() => {});
      setDirGrants([]);
      // A session opens in its own workspace. Setting it used to fail in
      // silence — a folder since moved or deleted left the header showing
      // one workspace while the agent kept working in another.
      const meta = sessions.find((s) => s.id === id);
      if (meta?.workspace) {
        try {
          setWorkspace(await agentSetWorkspace(meta.workspace));
        } catch {
          setWorkspace(null);
          void confirm({
            message: t("cmSessionWsMissing", { path: meta.workspace }),
            confirmLabel: t("confirm"),
          });
        }
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
    try {
      await removeSession(id);
    } catch (e) {
      // The row is still on disk, so the view must not act as though it had
      // gone: it used to clear itself either way, and the session came back
      // on the next refresh with nothing said.
      await confirm({
        message: `${t("cmDeleteSessionFailed")}\n${e instanceof Error ? e.message : String(e)}`,
        confirmLabel: t("close"),
        hideCancel: true,
      });
    }
    refreshSessions();
  }

  /** Delete the ticked sessions after one confirmation. */
  async function deleteSelectedSessions() {
    const ids = sessSel.picked();
    if (ids.length === 0) return;
    const ok = await confirm({
      message: t("confirmDeleteSessions", { n: ids.length }),
      confirmLabel: t("confirmDelete"),
      danger: true,
    });
    if (!ok) return;
    const failed: string[] = [];
    for (const id of ids) {
      try {
        await removeSession(id);
      } catch (e) {
        failed.push(e instanceof Error ? e.message : String(e));
      }
    }
    sessSel.exit();
    refreshSessions();
    if (failed.length) {
      await confirm({
        message: `${t("cmDeleteSessionFailed")}\n${failed[0]}`,
        confirmLabel: t("close"),
        hideCancel: true,
      });
    }
  }

  /** Delete one session, and reset the view when it is the one open. Throws
   *  when the row could not be removed; the callers confirm and refresh. */
  async function removeSession(id: string) {
    await codeSessionDelete(id);
    if (id in modesRef.current) {
      const { [id]: _gone, ...rest } = modesRef.current;
      modesRef.current = rest;
      setPermModes(rest);
      savePermModes(rest);
    }
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
      // Belongs to the turn that was paused in the session being deleted —
      // left behind, the next send in the fresh session resumed with the old
      // session's escape (newSession/openSession already clear it).
      stuckRef.current = null;
      const fresh = uid();
      sidRef.current = fresh;
      setSid(fresh);
      setMsgs([]);
      setInput("");
      setRefSessions([]);
      setCtxUsed(0);
      setStats(null);
      setPrefill(null);
      sessionAllowsRef.current = new Set();
      void agentClearGrants().catch(() => {});
      setDirGrants([]);
    }
  }

  /** Rewind to before `m`: restore journaled files and drop later messages.
   *  The message text lands back in the composer for editing & re-sending. */
  /** Put files a turn changed back the way they were before it: through the
   *  turn's checkpoint while the app holds it (byte for byte), otherwise from
   *  the copy the card kept. */
  async function undoChanges(m: CodeMsg, paths: string[]) {
    if (running || !m.changes) return;
    const targets = m.changes.filter((x) => paths.includes(x.path) && !x.undone);
    if (!targets.length) return;
    // Irreversible: once a file is put back, what the turn wrote to it is gone.
    const ok = await confirm({
      message:
        targets.length === 1
          ? t("cmUndoConfirmOne", { file: targets[0].rel })
          : t("cmUndoConfirmMany", { n: String(targets.length) }),
      confirmLabel: t("cmChangeUndo"),
      danger: true,
    });
    if (!ok) return;
    const done = new Set<string>();
    const failed: string[] = [];
    for (const c of targets) {
      try {
        if (m.changesCp == null) throw new Error("no checkpoint");
        await agentCheckpointRevertFile(m.changesCp, c.path);
        done.add(c.path);
      } catch {
        const kept = keptRestore(c);
        if (kept === undefined) {
          failed.push(c.rel);
          continue;
        }
        try {
          await agentRestoreFile(c.rel, kept);
          done.add(c.path);
        } catch {
          failed.push(c.rel);
        }
      }
    }
    if (done.size) {
      setMsgs((cur) => {
        const next = cur.map((x) =>
          x.id === m.id
            ? { ...x, changes: x.changes?.map((c) => (done.has(c.path) ? { ...c, undone: true } : c)) }
            : x,
        );
        persist(next, bodyRef.current.workspace, bodyRef.current.sid);
        return next;
      });
    }
    if (failed.length) {
      void confirm({ message: t("cmUndoFailed", { files: failed.join(", ") }), confirmLabel: t("confirm") });
    }
  }

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
    // Whatever it was thinking when the user stopped it: kept, but no longer
    // "thinking". Left live, the panel kept its spinner turning over a turn
    // that had already been told to stop.
    setMsgs((cur) =>
      cur.map((m) =>
        m.liveThinking
          ? { ...m, thinking: m.thinking ? `${m.thinking}\n\n${m.liveThinking}` : m.liveThinking, liveThinking: "" }
          : m,
      ),
    );
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
    setRunningNow(false);
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

  /** @ offers the other sessions as well as the workspace's files: a session
   *  the user points at is context for this turn, and the agent can search
   *  the rest of it with search_history. */
  type AtEntry = { kind: "file"; path: string } | { kind: "session"; id: string; title: string };
  const atMenu: AtEntry[] = (() => {
    if (atQuery == null || atHidden) return [];
    const q = atQuery.toLowerCase();
    const mine = sessions
      .filter((s) => s.id !== sid && !refSessions.some((r) => r.id === s.id))
      .filter((s) => (q ? s.title.toLowerCase().includes(q) : true))
      .slice(0, q ? 4 : 2)
      .map((s): AtEntry => ({ kind: "session", id: s.id, title: s.title }));
    return [...mine, ...atFiles.map((path): AtEntry => ({ kind: "file", path }))];
  })();

  function pickAt(entry: AtEntry) {
    if (entry.kind === "file") {
      setInput((cur) => cur.replace(/(^|\s)@[^\s@]*$/, `$1${entry.path} `));
    } else {
      // The reference is a chip, not text: the message keeps the words the
      // user typed, and the session rides with it.
      setInput((cur) => cur.replace(/(^|\s)@[^\s@]*$/, "$1"));
      setRefSessions((cur) =>
        cur.some((r) => r.id === entry.id) ? cur : [...cur, { id: entry.id, title: entry.title }],
      );
    }
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
        localSave("chaty.code.think", arg);
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
          ? "**可用命令**\n\n- `/clear` — 新建会话(清空上下文)\n- `/think off|normal|deep` — 思考强度\n- `/bypass` — 切换绕过权限\n- `/help` — 显示本帮助\n\n**快捷键**:运行中按 `Esc` 中断;`Shift+Tab` 切换权限模式(自动、手动、接受编辑、计划、绕过权限);审批弹窗 `Enter` 允许、`Esc` 拒绝;选择题可按数字键;输入 `@` 可引用工作区文件。"
          : "**Commands**\n\n- `/clear` — new session (clears context)\n- `/think off|normal|deep` — reasoning depth\n- `/bypass` — toggle bypass permissions\n- `/help` — this help\n\n**Shortcuts**: `Esc` interrupts a run; `Shift+Tab` cycles the permission modes (auto, manual, accept edits, plan, bypass); approval dialog `Enter` allows / `Esc` denies; number keys answer choice questions; type `@` to reference a workspace file.") +
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

  /** Sent while a turn runs: ask whether it waits for the turn (queued, sent
   *  once the turn ends) or cuts in — the turn stopped, and the message sent
   *  once it has wound down, so what the stopped turn did is in the record
   *  the new one starts from. Escape leaves it in the input. */
  async function sendWhileRunning() {
    const q = input.trim();
    if (!q) return;
    // After the key that sent it is done with: opened within it, the dialog's
    // default button took focus in time for the same Enter to press it.
    await new Promise((r) => setTimeout(r, 0));
    const pick = await choose({
      title: t("cmBusyTitle"),
      message: t("cmBusyMsg"),
      confirmLabel: t("cmBusyQueue"),
      altLabel: t("cmBusyInterrupt"),
    });
    if (!pick) return;
    setInput("");
    if (pick === "ok" || !runningRef.current) {
      // Queued — or the turn ended while the dialog was up: the queue sends
      // it next either way.
      if (runningRef.current) setQueue((cur) => [...cur, q]);
      else void sendRef.current(q);
      return;
    }
    const winding = turnDone.current;
    stop();
    // Shown as waiting while the stopped turn winds down; taking it off the
    // queue meanwhile takes the message back.
    setQueue([q]);
    await winding?.catch(() => {});
    if (!queueRef.current.includes(q)) return;
    setQueue((cur) => cur.filter((x) => x !== q));
    void sendRef.current(q);
  }

  async function send(textArg?: string) {
    const text = (textArg ?? input).trim();
    if (!textArg && text.startsWith("/") && slashMenu.some((e) => e.cmd === text)) {
      runSlash(text);
      return;
    }
    if (!text || runningRef.current || !model || !workspace) return;
    if (!textArg) setInput("");
    let finish = () => {};
    turnDone.current = new Promise<void>((r) => (finish = r));
    try {
      await runTurn(text, textArg);
    } finally {
      finish();
    }
  }

  async function runTurn(text: string, textArg?: string) {
    if (!model || !workspace) return;
    // The session this turn belongs to, fixed BEFORE the first await below.
    // Preparing a turn reads the project doc, the memory index and the skill
    // files, and none of that blocks the session list: switching sessions in
    // that window left the turn writing the OLD session's messages under the
    // NEW session's id and workspace.
    const turnSid = bodyRef.current.sid;
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
      ? await sessionMemoryIndex(memoryPinsRef.current, bodyRef.current.sid, () =>
          loadMemoryIndex({ readFile: (fp) => agentReadFile(fp) }).catch(() => ""),
        )
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
    // Sessions the user pointed at with @: how each opened and where it got
    // to rides with the message, and its id lets the agent read the rest with
    // search_history rather than carrying a whole transcript it may not need.
    const refCtx = (
      await Promise.all(
        refSessions.map(async (r) => {
          const hits = await codeSessionSearch("", r.id, 4).catch(() => []);
          const body = hits.map((h) => `${h.role}: ${h.text}`).join("\n");
          return `【${t("cmRefSession")}「${r.title}」 id=${r.id}】\n${body}\n(${t("cmRefHint")})`;
        }),
      )
    ).join("\n\n");
    const userMsg: CodeMsg = {
      id: uid(),
      role: "user",
      text,
      steps: [],
      checkpointId,
      images: visionImgs.length ? visionImgs : undefined,
      attachments:
        codeAttachments.length || refSessions.length
          ? [
              ...refSessions.map((r) => ({ name: r.title, kind: "session" })),
              ...codeAttachments.map((a) => ({ name: a.name, kind: a.kind, path: a.path })),
            ]
          : undefined,
    };
    // A planning turn ends on a plan the user can carry out from under it.
    const asst: CodeMsg = {
      id: uid(),
      role: "assistant",
      text: "",
      steps: [],
      ...((modesRef.current[turnSid] ?? "auto") === "plan" ? { planned: true } : {}),
    };
    // The previous turn hands back exactly what it sent, so this one continues
    // from it instead of from a summary. Replaying that tail is what keeps the
    // model's own work in front of it — a follow-up used to arrive with the
    // tool results gone and only "(tools run: read_file, bash)" in their place,
    // so the model re-read files it had just read, and the prompt could not
    // extend the previous one either. The tail is only the truth while it is
    // still the end of the conversation: anything edited or deleted after it
    // falls back to the text record.
    // The newest turn that recorded a tail. Locally-injected assistant text
    // (/help) can sit after it without invalidating anything; a USER message
    // after it cannot, because a turn that answered one would have recorded a
    // tail of its own — so seeing one means this record is not the truth.
    const replay = replayableTail(msgs);
    const history: ChatMessage[] =
      replay ??
      msgs.map((m) => {
        if (m.role !== "assistant" || m.steps.length === 0)
          return { role: m.role, content: m.text };
        const done = m.steps
          .filter((s) => s.status === "done")
          .map((s) => toolSummary(s.call))
          .join("; ");
        const prefix = done
          ? lang === "zh"
            ? `(已执行:${done})\n`
            : `(tools run: ${done})\n`
          : "";
        return { role: m.role, content: prefix + m.text };
      });
    // Moved on while the turn was being prepared: everything below belongs to
    // a session that is no longer on screen, so the turn is dropped and the
    // text handed back if the composer is still empty.
    if (bodyRef.current.sid !== turnSid) {
      if (!textArg) setInput((cur) => cur || text);
      return;
    }
    const base = [...msgs, userMsg, asst];
    const isFirstTurn = msgs.length === 0;
    setMsgs(base);
    setRunningNow(true);
    // A run in flight, recorded outside React state. The webview reloads on its
    // own sometimes and takes the turn with it; without this the session simply
    // reappeared idle and there was nothing to say what had happened.
    localSave(RUN_INFLIGHT_KEY, String(Date.now()));
    const resumeFrom = stuckRef.current;
    stuckRef.current = null;
    setStats(null);
    // First message = the session EXISTS: on disk, in the sidebar, named
    // (fallback title from the message text; the model-polished title still
    // lands after the turn). Persisting only at turn end meant a paused or
    // crashed first turn left ZERO rows — the calculator and minesweeper
    // audits both lost their transcripts to exactly that.
    persist(base, bodyRef.current.workspace, turnSid);

    const signal = new AgentSignal();
    signalRef.current = signal;
    // A stopped turn still winds down after the controls are handed back, and
    // by then the user may be running another turn — in this session or a new
    // one. Whatever this turn reports past that point is about itself, not
    // the run on screen.
    const runSid = sid;
    runSidRef.current = runSid;
    const runCtx = model?.nCtx ?? 0;
    const owns = () => signalRef.current === signal;
    // A run's numbers are its session's: once that session is deleted (or
    // another is open) they no longer reach the screen — a stopped run's last
    // report used to land after the reset and put the old ring back.
    const shown = () => owns() && sidRef.current === runSid;
    const update = (fn: (a: CodeMsg) => CodeMsg) =>
      setMsgs((cur) => cur.map((m) => (m.id === asst.id ? fn(m) : m)));

    const turnImages = visionImgs;
    setCodeAttachments([]);
    setRefSessions([]);
    const modelInput = [refCtx, attachCtx, text].filter(Boolean).join("\n\n");
    // A fresh session's title, asked for BEFORE its first turn. The engine keeps
    // one cache, and a title is a prompt of its own: asked after the turn, as it
    // used to be, it replaced everything the first turn had built, and the
    // second turn re-read the whole of it — the one turn boundary in a session
    // that never resumed. Before the first turn there is nothing yet to lose.
    if (isFirstTurn) await makeSessionTitle(turnSid, text);
    await runAgentTurn(modelInput, history, workspace, agentLang(lang), {
      thinkMode,
      // Consumed once: this turn resumes the escape, the one after it starts clean.
      resume: resumeFrom ?? undefined,
      supportsThinking: model.supportsThinking,
      thinkSwitch: model.thinkSwitch,
      effort: nativeEffort && thinkMode !== "off" ? rung : undefined,
      nCtx: model.nCtx ?? undefined,
      maxSteps,
      temperature,
      sampling,
      thinkBudget,
      maxGenTokens,
      bashTimeout,
      projectDoc,
      skills,
      memoryIndex,
      // The session's transcript outlives its context window: search_history
      // reads back what compaction dropped, and the sessions the user @-ed.
      sessionId: turnSid,
      visionReady: model.visionReady,
      // Both engines keep an already-encoded image whose identity still
      // prefixes the new prompt, so dropping a stale screenshot only costs.
      // They differ in HOW pixels are fed: llama.cpp takes one chunk per tile,
      // MLX takes each picture's span in a single pass, and that is what
      // decides how many tiles a tall page may send.
      mediaPrefixReuse: true,
      mediaChunked: model.backend === "llama.cpp",
      multiImage: model.multiImage !== false,
      ragTopK,
      toolRole: model.toolRole ?? false,
      reasoningField: model.reasoningField ?? false,
      // The format this model was trained to write calls in (its template
      // says which); a family whose template names none gets the fallback.
      toolFormat: resolveToolFormat(toolFormat, model.toolFormat, toolFallback),
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
      // The permission mode decides, as it stands when the call comes —
      // with the allowlist, read-only commands and "always allow" in it.
      approve: (call: ToolCall) => {
        const v = verdictFor(modesRef.current[runSid] ?? "auto", call);
        if (v === "allow") return Promise.resolve(true);
        if (v === "plan") return Promise.resolve("plan" as const);
        return new Promise<boolean | "plan">((resolve) => setApproval({ call, resolve }));
      },
      planMode: (modesRef.current[runSid] ?? "auto") === "plan",
      // While planning, a change is known not to run from its name — as the
      // mode stands then, so switching out of planning lets the next through.
      // And a question would stand in for the plan, which belongs in the
      // answer, where the choice to carry it out goes under it.
      blocked: (name: string) => {
        if ((modesRef.current[runSid] ?? "auto") !== "plan") return null;
        if (isFileEdit(name) || name === "web_download") return "plan";
        return name === "ask_user" ? "planAsk" : null;
      },
      askFirstNote: true,
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
      onStats: (tokens, tps) => shown() && setStats({ tokens, tps }),
      onContext: (used) => {
        if (!shown()) return;
        setCtxUsed(used);
        setCtxTotal(runCtx);
        void sessionStatsSet("code", runSid, JSON.stringify({ ctxUsed: used, nCtx: runCtx })).catch(() => {});
      },
      onPrefill: (frac) => shown() && setPrefill(frac),
      onCalling: (name) => shown() && setCallingTool(name),
      onDirGrants: (grants) => owns() && setDirGrants(grants),
      onPlan: (todos) => update((m) => ({ ...m, plan: todos })),
      onSaid: (text) =>
        update((m) => ({
          ...m,
          said: [...(m.said ?? []), { after: m.steps.length, text }],
          text: m.text.trim() === text.trim() ? "" : m.text,
        })),
      onCompacted: () => update((m) => ({ ...m, compacted: true })),
      onAskUser: (question, options) =>
        new Promise<string>((resolve) => {
          setAskText("");
          setAsk({ question, options, resolve });
        }),
      onAssistantText: (full) => update((m) => ({ ...m, text: full })),
      onLiveStep: (step) =>
        update((m) => {
          const steps = [...m.steps];
          const i = steps.findIndex((s) => s.id === step.id);
          if (i >= 0) steps[i] = step;
          else steps.push(step);
          // The reasoning behind the call now rides on its card, and so do
          // the words before it — not twice.
          const said = step.said?.trim();
          const text = said && m.text.trim() === said ? "" : m.text;
          return { ...m, steps, liveThinking: "", text };
        }),
      onLiveStepGone: (id) => update((m) => ({ ...m, steps: m.steps.filter((s) => s.id !== id) })),
      onStep: (step) => {
        update((m) => {
          const steps = [...m.steps];
          const i = steps.findIndex((s) => s.id === step.id);
          if (i >= 0) steps[i] = step;
          else steps.push(step);
          // the reasoning is now captured on the step → clear the live buffer;
          // and the words it said, which streamed into the answer box, now
          // sit on the step — not twice.
          const said = step.said?.trim();
          const text = said && m.text.trim() === said ? "" : m.text;
          return { ...m, steps, liveThinking: "", text };
        });
        // Mid-run durability: every step lands on disk (debounced), so a
        // pause + quit — or a crash — loses at most the last two seconds,
        // not the whole transcript.
        persistSoon(turnSid);
      },
      onStepDiff: (stepId, diff) => {
        codeStepTextPut(turnSid, `${stepId}:diff`, JSON.stringify(diff)).catch((e) =>
          console.error("step diff save failed", e),
        );
      },
      onStepText: (stepId, text) => {
        codeStepTextPut(turnSid, stepId, text).catch((e) => console.error("step text save failed", e));
        update((m) => ({
          ...m,
          steps: m.steps.map((s) => (s.id === stepId ? { ...s, fullText: true } : s)),
        }));
        persistSoon(turnSid);
      },
      onFinal: (final, thinking, reason, stuck) => {
        // What the turn was stuck on, so "Continue" resumes the escape instead
        // of restarting it — see AgentOptions.resume.
        if (owns()) stuckRef.current = reason === "steps" ? (stuck ?? null) : null;
        update((m) => ({
          ...m,
          text: final,
          thinking: thinking || m.thinking,
          liveThinking: "",
          paused: reason === "steps",
        }));
      },
      onError: (msg) => update((m) => ({ ...m, text: (m.text ? m.text + "\n\n" : "") + `**${msg}**` })),
      // Keep the tail on the newest assistant turn only — it is the only one
      // ever replayed, and every earlier copy would be dead weight in the
      // session file, which stores the whole transcript verbatim.
      // Only in this turn's own transcript: after a switch, clearing "every
      // other" tail wiped the one the open session needed to resume from.
      onTranscript: (tail) =>
        setMsgs((cur) =>
          !cur.some((m) => m.id === asst.id)
            ? cur
            : cur.map((m) =>
                m.id === asst.id
                  ? { ...m, prompt: tail }
                  : m.prompt
                    ? { ...m, prompt: undefined }
                    : m,
              ),
        ),
    });

    // What the turn left changed in the workspace, file by file, for the card
    // under its answer — read before the next turn can touch anything.
    if (checkpointId != null) {
      const changes = await agentCheckpointChanges(checkpointId).catch(() => [] as CpChange[]);
      if (changes.length) {
        update((m) => ({ ...m, changes: changes.map(toTurnChange), changesCp: checkpointId }));
      }
    }

    // Stop hands the controls back at once, but a stopped turn can take a while
    // to wind down (a command it was running still has to end). By then the
    // user may have sent the next turn — or opened a new session — and this
    // cleanup used to reset THAT run: marked it not running, cleared its
    // approval dialog (leaving the run waiting on an answer that could never
    // come) and its in-flight marker. Only the run that still owns the
    // controls resets them.
    if (owns()) {
      setRunningNow(false);
      localStorage.removeItem(RUN_INFLIGHT_KEY);
      setApproval(null);
      setPrefill(null);
      setCallingTool(null);
    }
    // Persist only while this turn's session is still the active one — after
    // a mid-run delete, writing here would resurrect the file the user just
    // removed (and clobber the fresh empty session).
    if (bodyRef.current.sid === turnSid) {
      setMsgs((cur) => {
        persist(cur, bodyRef.current.workspace, turnSid);
        return cur;
      });
    }

    // Messages queued while the agent was working → run them in order.
    const next = queueRef.current[0];
    if (next !== undefined && !signal.cancelled && bodyRef.current.sid === turnSid) {
      setQueue((q) => q.slice(1));
      void send(next);
    }
  }

  /** The folded view of an assistant turn (Settings → Code → collapse tool
   *  calls): its words and runs, which run is going on, what is happening,
   *  and whether that needs a line of its own under what was said last. */
  const foldOf = (m: CodeMsg, isLast: boolean) => {
    // A run is keyed by its place among the turn's runs, not by its first
    // step: the line for a run not yet begun — the thinking before it, its
    // call still being written — then carries the key of the run it turns
    // into, and stays the same element. Keyed by step, the line was dropped
    // for a new one as the step arrived, and a run opened by hand shut.
    let runs = 0;
    const segs = segmentsOf(m.steps, m.said).map((x) => (x.kind === "tools" ? { ...x, key: `run-${runs++}` } : x));
    const tail = segs[segs.length - 1];
    const turnLive = running && isLast;
    // Words in the answer box: the model is speaking — a step's words before
    // its call, or the answer. Thinking is not what is happening then.
    const speaking = !!m.text.trim();
    const lastStep = m.steps[m.steps.length - 1];
    // What is happening, most specific first: a call waiting on the user, a
    // step running, the prompt being read, a call being written (named once
    // its tool is), thinking.
    const activity: Activity | null = !turnLive
      ? null
      : approval
        ? { key: "actAwaiting" }
        : lastStep?.status === "running"
          ? activityOf(lastStep)
          : prefill != null
            ? { key: "actPrefill", x: `${Math.round(Math.min(1, Math.max(0, prefill)) * 100)}%` }
            : callingTool !== null
              ? callingTool
                ? activityOf({ id: "", call: { name: callingTool as never, args: {} }, status: "running" })
                : { key: "actUsingTools" }
              : speaking
                ? null
                : { key: "actThinking" };
    const activeKey = turnLive && !speaking && tail?.kind === "tools" ? tail.key : null;
    // The thinking that is not a step's — before the answer, or before a call
    // still being written — goes in the last run, out of sight like the rest.
    // A turn with no run shows it as any reply does.
    const lastRun = [...segs].reverse().find((s) => s.kind === "tools")?.key ?? null;
    return { segs, activeKey, activity, bottom: activity && !activeKey ? activity : null, lastRun, nextRun: `run-${runs}`, turnLive };
  };

  // Either separator: split on "/" alone, a Windows workspace showed its whole
  // path where its folder name belongs.
  const wsName = workspace ? workspaceName(workspace) : null;

  // The whole session as a JSONL trace (the top row's export button). It
  // reads the session as it is when pressed, not as it was when handed over.
  const traceOf = useRef({ sid, msgs, workspace, title: "", model: model?.name ?? null });
  traceOf.current = {
    sid,
    msgs,
    workspace,
    title: sessions.find((s) => s.id === sid)?.title ?? msgs.find((m) => m.role === "user")?.text.slice(0, 60) ?? "session",
    model: model?.name ?? null,
  };
  const exportTrace = useCallback(async () => {
    const cur = traceOf.current;
    if (cur.msgs.length === 0) return false;
    const content = await buildCodeTrace(
      { id: cur.sid, title: cur.title, workspace: cur.workspace, model: cur.model, app: `Chaty ${__APP_VERSION__}` },
      cur.msgs,
      (key) => codeStepTextGet(cur.sid, key),
    );
    const safe = cur.title.replace(/[\\/:*?"<>|\n]+/g, "_").slice(0, 60) || "session";
    return exportTextFile(`${safe}.jsonl`, content, "jsonl");
  }, []);
  const hasTurns = msgs.length > 0;
  useEffect(() => {
    onExportable?.(active && hasTurns ? exportTrace : null);
  }, [active, hasTurns, exportTrace, onExportable]);

  const sessionRow = (s: (typeof sessions)[number]) => (
    <div
      key={s.id}
      className={`cm-session ${s.id === sid && !sessSel.selecting ? "active" : ""} ${
        sessSel.selecting ? "selecting" : ""
      } ${sessSel.selected.has(s.id) ? "picked" : ""}`}
      onClick={(e) => {
        if (selectClick(sessSel, s.id, e)) return;
        void openSession(s.id);
      }}
    >
      {sessSel.selecting && <SelectCheck on={sessSel.selected.has(s.id)} />}
      <span className="cm-session-title">{s.title}</span>
      {!sessSel.selecting && (
        <button
          className="cm-session-del"
          title={t("deleteConv")}
          onClick={(e) => { e.stopPropagation(); void deleteSession(s.id); }}
        >
          <Icon name="x" size={11} strokeWidth={2.4} />
        </button>
      )}
    </div>
  );
  const sq = sessQuery.trim().toLowerCase();
  const shownSessions = sq
    ? sessions.filter((x) => x.title.toLowerCase().includes(sq) || sessMatches.has(x.id))
    : sessions;
  // The order the rows are shown in, for Shift-ranges and "all".
  sessSel.order(
    (sq
      ? shownSessions
      : groupByWorkspace
        ? groupSessionsByWorkspace(sessions).flatMap((g) =>
            collapsedWs.has(g.path ?? "") ? [] : recencyGroups(g.sessions).flatMap((r) => r.items),
          )
        : recencyGroups(sessions).flatMap((g) => g.items)
    ).map((x) => x.id),
  );

  const toggleWsGroup = (key: string) =>
    setCollapsedWs((cur) => {
      const next = new Set(cur);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  return (
    <div className="code-mode" style={active ? undefined : { display: "none" }}>
      <aside className="code-rail" style={{ width: railW }}>
        <button className="cm-new" onClick={newSession} disabled={running}>
          <Icon name="plus" size={13} strokeWidth={2} /> {t("cmNewSession")}
        </button>
        {sessions.length > 0 && (
          <div className="conv-search">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <circle cx="11" cy="11" r="7" />
              <path d="M21 21l-4.3-4.3" strokeLinecap="round" />
            </svg>
            <input
              type="text"
              placeholder={t("cmSearchSessions")}
              value={sessQuery}
              onChange={(e) => setSessQuery(e.target.value)}
            />
            {sessQuery && (
              <button className="conv-search-clear" onClick={() => setSessQuery("")} title={t("cancel")}>
                <Icon name="x" size={11} strokeWidth={2.2} />
              </button>
            )}
            <SelectToggle ms={sessSel} />
          </div>
        )}
        <div className="cm-sessions">
          {sessions.length === 0 ? (
            <div className="cm-empty-list">{t("cmNoSessions")}</div>
          ) : sq ? (
            shownSessions.length === 0 ? (
              <div className="cm-empty-list">{t("noMatches")}</div>
            ) : (
              shownSessions.map(sessionRow)
            )
          ) : groupByWorkspace ? (
            groupSessionsByWorkspace(sessions).map((g) => {
              const key = g.path ?? "";
              const folded = collapsedWs.has(key);
              return (
                <div key={key || "(none)"} className="cm-ws-block">
                  <button
                    className={`cm-ws-group ${g.path && g.path === workspace ? "current" : ""}`}
                    title={g.path ?? ""}
                    aria-expanded={!folded}
                    onClick={() => toggleWsGroup(key)}
                  >
                    <Icon name={folded ? "chevron-right" : "chevron-down"} size={13} />
                    <Icon name="folder" size={15} />
                    <span className="cm-ws-group-name">{g.path ? g.name : t("cmNoWorkspace")}</span>
                    <span className="cm-ws-group-count">{g.sessions.length}</span>
                  </button>
                  {/* Inside a workspace, by when they were last touched — as
                      the rail is ungrouped. */}
                  {!folded &&
                    recencyGroups(g.sessions).map((r) => (
                      <div key={r.key} className="conv-group cm-ws-recency">
                        <div className="conv-group-label">{t(RECENCY_LABEL[r.key])}</div>
                        {r.items.map(sessionRow)}
                      </div>
                    ))}
                </div>
              );
            })
          ) : (
            // Grouped by when they were last touched, as the chat sidebar is.
            recencyGroups(sessions).map((g) => (
              <div key={g.key} className="conv-group">
                <div className="conv-group-label">{t(RECENCY_LABEL[g.key])}</div>
                {g.items.map(sessionRow)}
              </div>
            ))
          )}
        </div>
        <SelectBar ms={sessSel} onDelete={() => void deleteSelectedSessions()} />
        <div className="side-status" title={model ? model.name : ""}>
          <span className="ss-meta">v{__APP_VERSION__}</span>
        </div>
        <div
          className="sidebar-resizer"
          role="separator"
          aria-orientation="vertical"
          title={t("resizeSidebar")}
          onPointerDown={onRailResize}
          onDoubleClick={onRailReset}
        />
      </aside>

      <main className="code-main">
        {active &&
          headSlot &&
          createPortal(
            <div className="code-head" data-tauri-drag-region>
              <button
                className="cm-ws"
                onClick={() => void pickWorkspace()}
                disabled={running}
                title={workspace ?? t("cmOpenFolder")}
                aria-label={workspace ? undefined : t("cmOpenFolder")}
              >
                <Icon name="folder" size={14} />
                {wsName ? <span className="cm-ws-name">{wsName}</span> : <span className="cm-ws-pick">{t("cmOpenFolder")}</span>}
                <Icon name="chevron-down" size={11} strokeWidth={2} className="cm-ws-caret" />
              </button>
              {(dirGrants.length > 0 || workspace) && (
                <div className="cm-grants">
                  {dirGrants.map((d) => (
                    <span key={d} className="cm-grant-chip" title={d}>
                      <Icon name="folder" size={11} />
                      <span className="cm-grant-name">{workspaceName(d)}</span>
                      <button className="cm-grant-del" title={t("cmGrantRevoke")} onClick={() => void revokeDir(d)}>
                        <Icon name="x" size={10} strokeWidth={2.2} />
                      </button>
                    </span>
                  ))}
                  {workspace && (
                    <button className="cm-grant-add" title={t("cmGrantAddTip")} aria-label={t("cmGrantAddTip")} onClick={() => void addGrantDir()}>
                      <Icon name="plus" size={13} strokeWidth={2} />
                    </button>
                  )}
                </div>
              )}
              <span className="cm-head-spacer" data-tauri-drag-region />
              {downloads.length > 0 && (
                <span
                  className="cm-bgjobs cm-dl-badge"
                  title={downloads.map((d) => `#${d.id} ${d.url} → ${d.path}`).join("\n")}
                >
                  <span className="cm-spin" /> <Icon name="download" size={12} /> {downloads.length}
                  {downloads[0].total
                    ? ` · ${Math.min(100, Math.round((downloads[0].downloaded / downloads[0].total) * 100))}%`
                    : ` · ${fmtBytes(downloads[0].downloaded)}`}
                </span>
              )}
              {bgJobs.length > 0 &&
                (() => {
                  const live = bgJobs.filter((j) => j.running).length;
                  return (
                    <button
                      ref={bgPillRef}
                      className={`cm-bgjobs ${showBg ? "active" : ""}`}
                      title={t("bgOpenHint")}
                      onClick={() => setShowBg((v) => !v)}
                    >
                      {live > 0 ? <span className="cm-spin" /> : <Icon name="check" size={12} strokeWidth={2.4} />}
                      {live > 0 ? (
                        <>
                          <span className="cm-bg-n">{live}</span>
                          <span className="cm-head-label">{t("cmBgCount", { n: String(live) })}</span>
                        </>
                      ) : (
                        <span className="cm-head-label">{t("bgTitle")}</span>
                      )}
                    </button>
                  );
                })()}
              {showBg && (
                <BgTasksPanel
                  anchorRef={bgPillRef}
                  jobs={bgJobs}
                  onClose={() => setShowBg(false)}
                  onChanged={() => agentBgAll().then(setBgJobs).catch(() => {})}
                />
              )}
            </div>,
            headSlot,
          )}

        <div className="code-wrap">
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
            <div className="cm-thread" ref={threadRef}>
            {msgs.map((m, mi) => {
              const f = collapseTools && m.role !== "user" ? foldOf(m, mi === msgs.length - 1) : null;
              return m.role === "user" ? (
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
                  <div className="cm-user" data-copy={m.text}>
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
                          <span key={a.name + i} className="cm-attach-doc" title={a.kind === "session" ? t("cmRefSession") : a.name}>
                            {a.kind === "session" ? (
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 109-9 9 9 0 00-6.4 2.7L3 8M3 4v4h4M12 7v5l3.5 2" /></svg>
                            ) : (
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6" /></svg>
                            )}
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
                  {f ? (
                    // Folded: the model's words in the open, the runs of steps
                    // between them one line each — the run going on says what
                    // is happening, and every thought sits in a run. One list,
                    // so a line becomes its run without being replaced.
                    [
                    ...f.segs.map((seg) =>
                      seg.kind === "say" ? (
                        <div key={seg.key} className="cm-asst-text answer cm-said">
                          <Markdown>{seg.text}</Markdown>
                        </div>
                      ) : (
                        <ToolRun
                          key={seg.key}
                          steps={seg.steps}
                          activity={seg.key === f.activeKey ? f.activity : null}
                        >
                          {seg.steps.map((s) => (
                            <div key={s.id} className="cm-block">
                              {s.thinking && <ThinkPanel text={s.thinking} label={t("cmThought")} />}
                              <StepCard step={s} sessionId={sid} onPreview={setPreviewImg} liveOpen={liveCardsOpen} />
                            </div>
                          ))}
                          {seg.key === f.lastRun && m.thinking && <ThinkPanel text={m.thinking} label={t("cmThought")} />}
                          {/* The live thought, unless the run not yet begun holds it. */}
                          {seg.key === f.lastRun && m.liveThinking && !f.bottom && (
                            <ThinkPanel text={m.liveThinking} live label={t("cmThinking")} />
                          )}
                        </ToolRun>
                      ),
                    ),
                    ...(f.bottom
                      ? [
                          // The words before a call still being written, then
                          // the run that call begins.
                          ...(m.text.trim()
                            ? [
                                <div key="said-pending" className="cm-asst-text answer cm-said">
                                  <Markdown>{m.text}</Markdown>
                                </div>,
                              ]
                            : []),
                          <ToolRun key={f.nextRun} steps={[]} activity={f.bottom}>
                            {m.liveThinking ? <ThinkPanel text={m.liveThinking} live label={t("cmThinking")} /> : null}
                          </ToolRun>,
                        ]
                      : // A turn that called no tool keeps its thought in a run
                        // of its own: one line, opened by hand. Not while the
                        // turn goes on — words then may be the ones before a
                        // call, whose run comes under them, not over.
                        !f.lastRun && !f.turnLive && (m.thinking || m.liveThinking)
                        ? [
                            <ToolRun key={f.nextRun} steps={[]} doneLabel={t("cmThought")}>
                              <div className="cm-think-body">
                                {[m.thinking, m.liveThinking].filter(Boolean).join("\n\n")}
                              </div>
                            </ToolRun>,
                          ]
                        : []),
                    ]
                  ) : (
                    <>
                    {/* What the model said between its steps stays where it said it:
                        with a step's call, or with a plan update (which has no card). */}
                    {m.steps.map((s, i) => (
                      <Fragment key={s.id}>
                        {(m.said ?? [])
                          .filter((w) => w.after === i)
                          .map((w, j) => (
                            <div key={`said-${j}`} className="cm-asst-text answer cm-said">
                              <Markdown>{w.text}</Markdown>
                            </div>
                          ))}
                        <div className="cm-block">
                          {s.thinking && <ThinkPanel text={s.thinking} label={t("cmThought")} />}
                          {s.said && (
                            <div className="cm-asst-text answer cm-said">
                              <Markdown>{s.said}</Markdown>
                            </div>
                          )}
                          <StepCard step={s} sessionId={sid} onPreview={setPreviewImg} liveOpen={liveCardsOpen} />
                        </div>
                      </Fragment>
                    ))}
                    {(m.said ?? [])
                      .filter((w) => w.after >= m.steps.length)
                      .map((w, j) => (
                        <div key={`said-end-${j}`} className="cm-asst-text answer cm-said">
                          <Markdown>{w.text}</Markdown>
                        </div>
                      ))}
                    {m.liveThinking && <ThinkPanel text={m.liveThinking} live label={t("cmThinking")} />}
                    </>
                  )}
                  {!f && m.thinking && <ThinkPanel text={m.thinking} label={t("cmThought")} />}
                  {m.text && !f?.bottom && (
                    <div className="cm-asst-text answer" data-copy={m.text}>
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
                  {m.changes && m.changes.length > 0 && (
                    <TurnChangesCard
                      changes={m.changes}
                      disabled={running}
                      onUndo={(paths) => void undoChanges(m, paths)}
                    />
                  )}
                  {m.planned && !m.paused && !running && mode === "plan" && m === msgs[msgs.length - 1] && m.text.trim() && keptPlanning !== m.id && (
                    // The plan is written: carry it out — edits accepted, or
                    // each one asked — or say what to change and keep planning.
                    <div className="cm-plan-go">
                      <span className="cm-plan-go-q">{t("planReady")}</span>
                      <button
                        className="cm-plan-go-btn primary"
                        onClick={() => {
                          pickMode("acceptEdits");
                          void send(t("planGoMsg"));
                        }}
                      >
                        {t("planGoAccept")}
                      </button>
                      <button
                        className="cm-plan-go-btn"
                        onClick={() => {
                          pickMode("manual");
                          void send(t("planGoMsg"));
                        }}
                      >
                        {t("planGoManual")}
                      </button>
                      <button
                        className="cm-plan-go-btn"
                        onClick={() => {
                          // Still planning: the choice goes, and the input
                          // asks for what to change.
                          setKeptPlanning(m.id);
                          inputRef.current?.focus();
                        }}
                      >
                        {t("planKeep")}
                      </button>
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
                  {!collapseTools && running && m === msgs[msgs.length - 1] && !m.text && !m.liveThinking && m.steps.length === 0 && (
                    prefill != null ? <PrefillRing frac={prefill} size={18} /> : <span className="cm-spin cm-working" />
                  )}
                </div>
              );
            })}
            </div>
          )}
        </div>
        <JumpBottom
          shown={showJump && msgs.length > 0}
          title={t("jumpLatest")}
          onClick={() => {
            followRef.current = true;
            scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
          }}
        />
        </div>

        <div className="code-composer">
          {/* What the agent is doing sits inside the composer, as chat's
              status line does: above it, the transcript's fade covered it. */}
          {running && (() => {
            // The same words the folded run uses — what is going on now, the
            // tool it is by name — whichever way the steps are shown.
            const cur = msgs[msgs.length - 1];
            const act = cur && cur.role !== "user" ? foldOf(cur, true).activity : null;
            const label =
              prefill != null ? t("cmPrefill") : act ? <ActivityLabel activity={act} /> : t(cur?.role === "user" ? "cmRunning" : "actReplying");
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
          {(codeAttachments.length > 0 || refSessions.length > 0 || attachErr) && (
            <div className="cm-attach-row">
              {refSessions.map((r) => (
                <span key={r.id} className="cm-attach-doc" title={`${t("cmRefSession")}: ${r.title}`}>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 109-9 9 9 0 00-6.4 2.7L3 8M3 4v4h4M12 7v5l3.5 2" /></svg>
                  <span className="cm-attach-doc-name">{r.title}</span>
                  <button
                    className="cm-attach-del inline"
                    title={t("cmRefRemove")}
                    onClick={() => setRefSessions((cur) => cur.filter((x) => x.id !== r.id))}
                  >
                    <Icon name="x" size={10} strokeWidth={2.2} />
                  </button>
                </span>
              ))}
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
                {atMenu.map((e, i) => (
                  <button
                    key={e.kind === "file" ? e.path : e.id}
                    className={`cm-slash-item ${i === atSel ? "sel" : ""}`}
                    onMouseEnter={() => setAtSel(i)}
                    onClick={() => pickAt(e)}
                  >
                    {e.kind === "file" ? (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6" /></svg>
                    ) : (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 109-9 9 9 0 00-6.4 2.7L3 8M3 4v4h4M12 7v5l3.5 2" /></svg>
                    )}
                    <span className="cm-slash-desc">{e.kind === "file" ? e.path : e.title}</span>
                    {e.kind === "session" && <span className="cm-slash-cmd">{t("cmRefSession")}</span>}
                  </button>
                ))}
              </div>
            )}
            <textarea
              ref={inputRef}
              className="cm-input"
              placeholder={
                !model
                  ? t("cmPlaceholderNoModel")
                  : !workspace
                    ? t("cmPlaceholderNoWs")
                    : running
                      ? t("cmQueuePlaceholder")
                      : keptPlanning && keptPlanning === msgs[msgs.length - 1]?.id && mode === "plan"
                        ? t("planKeepHint")
                        : sendKey === "modEnter"
                        ? t("cmPlaceholderMod")
                        : t("cmPlaceholder")
              }
              value={input}
              disabled={!model || !workspace}
              onChange={(e) => { setInput(e.target.value); setSlashSel(0); setAtHidden(false); }}
              onKeyDown={(e) => {
                // Shift+Tab cycles the permission modes, as in Claude Code.
                if (e.key === "Tab" && e.shiftKey) {
                  e.preventDefault();
                  cycleMode();
                  return;
                }
                if (atMenu.length > 0) {
                  if (e.key === "ArrowDown") { e.preventDefault(); setAtSel((s) => (s + 1) % atMenu.length); return; }
                  if (e.key === "ArrowUp") { e.preventDefault(); setAtSel((s) => (s - 1 + atMenu.length) % atMenu.length); return; }
                  if (e.key === "Enter" || e.key === "Tab") {
                    e.preventDefault();
                    pickAt(atMenu[Math.min(atSel, atMenu.length - 1)]);
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
                    void sendWhileRunning();
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
              <button className="cm-send" onClick={() => void send()} disabled={!input.trim() || !model || !workspace} title={t("sendTitle")}>
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
              </button>
            )}
          </div>
          {/* How the next message runs, under the input as the image
              studio's settings are: thinking and bypass on the left, the
              context in use on the right. They lived in a header bar that
              is gone now. */}
          <div className="cm-status">
            <div className="cm-think-switch" title={t(nativeEffort ? "effortHint" : "cmThinkHint")}>
              {/* A model with its own ladder shows that ladder, whatever its
                  length; `off` is Chaty's, because a ladder has no rung for
                  not thinking at all. */}
              {(nativeEffort ? ["off", ...effortLevels] : THINK_MODES).map((tab) => {
                const active = thinkTabActive(tab, { nativeEffort, thinkMode, rung });
                return (
                  <button
                    key={tab}
                    className={`cm-think-tab ${active ? "active" : ""}`}
                    onClick={() => {
                      if (tab === "off") {
                        setThinkMode("off");
                        localSave("chaty.code.think", "off");
                        return;
                      }
                      if (!nativeEffort) {
                        setThinkMode(tab as ThinkMode);
                        localSave("chaty.code.think", tab);
                        return;
                      }
                      const mode = intensityOf(effortLevels, tab);
                      setCodeEffort(tab);
                      setThinkMode(mode);
                      localSave("chaty.code.effort", tab);
                      localSave("chaty.code.think", mode);
                    }}
                    disabled={running}
                  >
                    {tab === "off"
                      ? t("cmThinkOff")
                      : nativeEffort
                        ? effortLabel(tab, t)
                        : t(tab === "normal" ? "cmThinkNormal" : "cmThinkDeep")}
                  </button>
                );
              })}
            </div>
            <PermModePicker mode={mode} onPick={pickMode} />
            <span className="cm-status-spacer" />
            {ctxUsed > 0 && ctxTotal > 0 && <ContextRing used={ctxUsed} total={ctxTotal} />}
          </div>
        </div>
      </main>

      {previewImg && <ImagePreview path={previewImg} onClose={() => setPreviewImg(null)} />}
      {sudoAsk && (() => {
        const deny = () => { sudoAsk.resolve({ ok: false }); setSudoAsk(null); setSudoPw(""); };
        const allow = () => { sudoAsk.resolve({ ok: true, password: sudoPw || undefined }); setSudoAsk(null); setSudoPw(""); };
        return (
          <div className="cm-approve-backdrop">
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
        <div className="cm-approve-backdrop">
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
        <div className="cm-approve-backdrop">
          <div className="cm-approve" onMouseDown={(e) => e.stopPropagation()}>
            <div className="cm-approve-title">
              {approval.call.name === "bash" || approval.call.name === "bash_bg" || approval.call.name === "bg_input"
                ? t("cmApproveBash")
                : t("cmApproveWrite")}
            </div>
            {/* For a command this is the whole command ("$ …"). It used to be
                followed by the command again, which read as an explanation
                that said nothing (issue #18). */}
            <pre className="cm-approve-cmd">{toolSummary(approval.call)}</pre>
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
