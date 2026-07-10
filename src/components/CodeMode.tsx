import { useCallback, useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useI18n } from "../lib/i18n";
import { useConfirm } from "./ConfirmModal";
import { BUILTIN_SKILLS } from "../lib/skills";
import { copyToClipboard } from "../lib/clipboard";
import { Icon } from "./Icon";
import { Markdown } from "./Markdown";
import {
  agentBgKill,
  agentBgList,
  agentCheckpointBegin,
  agentCheckpointRevertTo,
  agentGetWorkspace,
  agentListFiles,
  agentReadFile,
  agentSetWorkspace,
  type AgentBgInfo,
  codeSessionDelete,
  codeSessionList,
  codeSessionLoad,
  codeSessionSave,
  type ChatMessage,
  type CodeSessionMeta,
  type ModelInfo,
} from "../lib/ipc";
import {
  AgentSignal,
  argContent,
  argEdits,
  argNew,
  argOld,
  argPath,
  runAgentTurn,
  type PlanItem,
  type ThinkMode,
  type ToolCall,
  type ToolStep,
} from "../lib/agentLoop";

interface CodeMsg {
  id: string;
  role: "user" | "assistant";
  text: string;
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

const RAIL_DEFAULT = 240;
const RAIL_MIN = 180;
const RAIL_MAX = 420;

const uid = () => Math.random().toString(36).slice(2);

/** Minimal line diff: trim the common prefix/suffix, show the changed middle as
 *  removals + additions with a couple of context lines. */
function lineDiff(before: string, after: string): { kind: "ctx" | "add" | "del"; text: string }[] {
  const a = before.split("\n");
  const b = after.split("\n");
  let s = 0;
  while (s < a.length && s < b.length && a[s] === b[s]) s++;
  let ea = a.length;
  let eb = b.length;
  while (ea > s && eb > s && a[ea - 1] === b[eb - 1]) {
    ea--;
    eb--;
  }
  const out: { kind: "ctx" | "add" | "del"; text: string }[] = [];
  for (let i = Math.max(0, s - 2); i < s; i++) out.push({ kind: "ctx", text: a[i] });
  for (let i = s; i < ea; i++) out.push({ kind: "del", text: a[i] });
  for (let i = s; i < eb; i++) out.push({ kind: "add", text: b[i] });
  for (let i = ea; i < Math.min(a.length, ea + 2); i++) out.push({ kind: "ctx", text: a[i] });
  return out.slice(0, 60);
}

const TOOL_ICON: Record<string, string> = {
  read_file: "M9 2h6l4 4v14a0 0 0 0 1 0 0H5V2z",
  write_file: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M12 12v6M9 15h6",
  edit_file: "M17 3a2.85 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5z",
  multi_edit: "M17 3a2.85 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5zM14 8l2 2",
  outline: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
  list_dir: "M3 6h18M3 12h18M3 18h18",
  glob: "M3 6h18M3 12h18M3 18h18",
  grep: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-4-4",
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
};

function toolSummary(call: ToolCall): string {
  const a = call.args as Record<string, string>;
  switch (call.name) {
    case "read_file":
      return `read ${argPath(call.args) || "?"}`;
    case "write_file":
      return `write ${argPath(call.args) || "?"}`;
    case "edit_file":
      return `edit ${argPath(call.args) || "?"}`;
    case "multi_edit":
      return `edit ×${argEdits(call.args).length} ${argPath(call.args) || "?"}`;
    case "outline":
      return `outline ${argPath(call.args) || "?"}`;
    case "list_dir":
      return `ls ${a.path ?? "."}`;
    case "glob":
      return `glob ${a.pattern ?? ""}`;
    case "grep":
      return `grep ${a.pattern ?? ""}`;
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
 *  result or a diff. */
function StepCard({ step }: { step: ToolStep }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(step.status === "error");
  const diff = step.diff;
  const hasBody = !!(step.result || diff);
  const meta = stepMeta(step, t("cmLines"));
  return (
    <div className={`cm-step ${step.status}`}>
      <button className="cm-step-head" onClick={() => hasBody && setOpen((o) => !o)}>
        <svg className="cm-step-ico" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d={TOOL_ICON[step.call.name] ?? "M4 6h16M4 12h16M4 18h16"} />
        </svg>
        <span className="cm-step-sum">{toolSummary(step.call)}</span>
        {diff && step.status === "done" && (() => {
          const d = lineDiff(diff.before, diff.after);
          const add = d.filter((l) => l.kind === "add").length;
          const del = d.filter((l) => l.kind === "del").length;
          return (
            <span className="cm-step-diffstat">
              <em className="plus">+{add}</em>
              <em className="minus">-{del}</em>
            </span>
          );
        })()}
        {meta && <span className={`cm-step-meta ${meta.tone}`}>{meta.text}</span>}
        <span className="cm-step-status">
          {step.status === "running" ? <span className="cm-spin" /> : null}
          {step.status === "done" ? <Icon name="check" size={12} strokeWidth={2.2} /> : null}
          {step.status === "error" ? <Icon name="x" size={12} strokeWidth={2.2} /> : null}
          {step.status === "denied" ? <Icon name="ban" size={12} strokeWidth={2} /> : null}
        </span>
      </button>
      {open && hasBody && (
        <div className="cm-step-body">
          {diff ? (
            <pre className="cm-diff">
              {lineDiff(diff.before, diff.after).map((l, i) => (
                <div key={i} className={`cm-dl ${l.kind}`}>
                  <span className="cm-dl-mark">{l.kind === "add" ? "+" : l.kind === "del" ? "-" : " "}</span>
                  {l.text}
                </div>
              ))}
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
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01" /></svg>
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

export function CodeMode({
  model,
  active,
  maxSteps,
  bashTimeout,
  skills = [],
  disabledSkills = [],
  allowedCommands = [],
  sendKey = "enter",
}: {
  model: ModelInfo | null;
  active: boolean;
  /** Max agent steps per turn (Settings → Code). */
  maxSteps?: number;
  /** Default bash timeout in seconds (Settings → Code). */
  bashTimeout?: number;
  /** User-defined skills: /name inserts the prompt template (Settings → Code). */
  skills?: { name: string; prompt: string }[];
  /** Names of built-in skills the user turned off (Settings → Code). */
  disabledSkills?: string[];
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
  const [running, setRunning] = useState(false);
  const [bypass, setBypass] = useState(false);
  const [thinkMode, setThinkMode] = useState<ThinkMode>(() => {
    const v = localStorage.getItem("chaty.code.think");
    return v === "off" || v === "normal" || v === "deep" ? v : "normal";
  });
  const [approval, setApproval] = useState<{ call: ToolCall; resolve: (ok: boolean) => void } | null>(null);
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
    const tick = () => agentBgList().then(setBgJobs).catch(() => {});
    tick();
    const timer = setInterval(tick, 5000);
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
  }, [active, approval, ask, running]);

  const persist = useCallback((next: CodeMsg[], ws: string | null, id: string) => {
    const firstUser = next.find((m) => m.role === "user");
    const title = (firstUser?.text ?? "New session").replace(/\s+/g, " ").trim().slice(0, 48) || "New session";
    codeSessionSave(id, title, ws, JSON.stringify(next)).then(refreshSessions).catch(() => {});
  }, [refreshSessions]);

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
    } catch (e) {
      // window.alert doesn't render inside WKWebView — use the in-app modal.
      void confirm({
        message: e instanceof Error ? e.message : String(e),
        confirmLabel: t("confirm"),
      });
    }
  }

  function newSession() {
    if (running) return;
    setSid(uid());
    setMsgs([]);
    setInput("");
    setCtxUsed(0);
    sessionAllowsRef.current = new Set();
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
      message: t("confirmDeleteConv"),
      confirmLabel: t("confirmDelete"),
      danger: true,
    });
    if (!ok) return;
    await codeSessionDelete(id).catch(() => {});
    refreshSessions();
    if (id === sid) newSession();
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
      if (arg === "off" || arg === "normal" || arg === "deep") {
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
   *  (what /init writes) > CLAUDE.md. Re-read each turn — /init may have just
   *  written it. Capped so it can't crowd the context window. */
  async function loadProjectDoc(): Promise<{ name: string; text: string } | undefined> {
    for (const name of ["AGENTS.md", "PROJECT.md", "CLAUDE.md"]) {
      try {
        const text = await agentReadFile(name);
        if (text.trim()) return { name, text: text.slice(0, 6000) };
      } catch {
        /* try the next candidate */
      }
    }
    return undefined;
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
    const userMsg: CodeMsg = { id: uid(), role: "user", text, steps: [], checkpointId };
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
    setMsgs(base);
    setRunning(true);
    setStats(null);

    const signal = new AgentSignal();
    signalRef.current = signal;
    const update = (fn: (a: CodeMsg) => CodeMsg) =>
      setMsgs((cur) => cur.map((m) => (m.id === asst.id ? fn(m) : m)));

    await runAgentTurn(text, history, workspace, lang, {
      thinkMode,
      supportsThinking: model.supportsThinking,
      thinkSwitch: model.thinkSwitch,
      nCtx: model.nCtx ?? undefined,
      maxSteps,
      bashTimeout,
      projectDoc,
      signal,
      approve: (call: ToolCall) => {
        if (bypassRef.current) return Promise.resolve(true);
        if (sessionAllowsRef.current.has(allowKeyFor(call))) return Promise.resolve(true);
        if (call.name === "bash" || call.name === "bash_bg") {
          const cmd = String(call.args.command ?? "").trim();
          if (allowedCommandsRef.current.some((p) => cmd === p || cmd.startsWith(p + " "))) {
            return Promise.resolve(true);
          }
        }
        return new Promise<boolean>((resolve) => setApproval({ call, resolve }));
      },
    }, {
      onThinking: (t) => update((m) => ({ ...m, liveThinking: t })),
      onStats: (tokens, tps) => setStats({ tokens, tps }),
      onContext: (used) => setCtxUsed(used),
      onPlan: (todos) => update((m) => ({ ...m, plan: todos })),
      onCompacted: () => update((m) => ({ ...m, compacted: true })),
      onAskUser: (question, options) =>
        new Promise<string>((resolve) => {
          setAskText("");
          setAsk({ question, options, resolve });
        }),
      onAssistantText: (full) => update((m) => ({ ...m, text: full })),
      onStep: (step) =>
        update((m) => {
          const steps = [...m.steps];
          const i = steps.findIndex((s) => s.id === step.id);
          if (i >= 0) steps[i] = step;
          else steps.push(step);
          // the reasoning is now captured on the step → clear the live buffer
          return { ...m, steps, liveThinking: "" };
        }),
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
    setMsgs((cur) => {
      persist(cur, bodyRef.current.workspace, bodyRef.current.sid);
      return cur;
    });

    // Messages queued while the agent was working → run them in order.
    const next = queueRef.current[0];
    if (next !== undefined && !signal.cancelled) {
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
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>
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
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" /></svg>
            {wsName ? <span className="cm-ws-name">{wsName}</span> : <span className="cm-ws-pick">{t("cmOpenFolder")}</span>}
          </button>
          <span className="cm-head-spacer" />
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
          <div className="cm-think-switch" title={t("cmThinkHint")}>
            {THINK_MODES.map((mode) => (
              <button
                key={mode}
                className={`cm-think-tab ${thinkMode === mode ? "active" : ""}`}
                onClick={() => {
                  setThinkMode(mode);
                  localStorage.setItem("chaty.code.think", mode);
                }}
                disabled={running}
              >
                {t(mode === "off" ? "cmThinkOff" : mode === "normal" ? "cmThinkNormal" : "cmThinkDeep")}
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
                  <div className="cm-user">{m.text}</div>
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
                      <StepCard step={s} />
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
                    <span className="cm-spin cm-working" />
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
          const label = curStep && curStep.status === "running" ? toolSummary(curStep.call) : t("cmRunning");
          return (
            <div className="cm-runbar">
              <span className="cm-spin" />
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
            {approval.call.name === "edit_file" && (
              <pre className="cm-diff cm-approve-diff">
                {lineDiff(argOld(approval.call.args), argNew(approval.call.args)).map((l, i) => (
                  <div key={i} className={`cm-dl ${l.kind}`}>
                    <span className="cm-dl-mark">{l.kind === "add" ? "+" : l.kind === "del" ? "-" : " "}</span>
                    {l.text}
                  </div>
                ))}
              </pre>
            )}
            {approval.call.name === "multi_edit" && (
              <pre className="cm-diff cm-approve-diff">
                {argEdits(approval.call.args).flatMap((e, ei) => [
                  <div key={`h${ei}`} className="cm-dl ctx">
                    <span className="cm-dl-mark"> </span>
                    {`— 修改 ${ei + 1} —`}
                  </div>,
                  ...lineDiff(e.old_string, e.new_string).map((l, i) => (
                    <div key={`${ei}-${i}`} className={`cm-dl ${l.kind}`}>
                      <span className="cm-dl-mark">{l.kind === "add" ? "+" : l.kind === "del" ? "-" : " "}</span>
                      {l.text}
                    </div>
                  )),
                ])}
              </pre>
            )}
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
