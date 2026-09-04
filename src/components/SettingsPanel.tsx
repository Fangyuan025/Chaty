import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { LANGS, useI18n } from "../lib/i18n";
import { useExitTransition } from "../lib/useExit";
import { Icon } from "./Icon";
import {
  openDataDir,
  clearAllConversations,
  checkUpdate,
  runUpdate,
  dataStats,
  listModels,
  ragStatus,
  ragClearAll,
  openModelsDir,
  openErrorLog,
  clearErrorLog,
  openExternal,
  synthesize,
  type UpdateInfo,
} from "../lib/ipc";
import { decodeAudio, playAudio, primeAudioPlayback } from "../lib/audio";
import { CODE_THEMES, type CodeTheme } from "../lib/codeTheme";
import { useConfirm } from "./ConfirmModal";
import { Select } from "./Select";
import { BUILTIN_SKILLS } from "../lib/skills";
import { loadMcpServers, saveMcpServers, syncMcpServers, type McpServerCfg } from "../lib/mcp";
import catalog from "../lib/mcpStore.catalog.json";
import { disabledSkills, officialSkills, setDisabledSkills } from "../lib/skillFiles";
import { fmtBytes } from "../lib/fmt";
import logoUrl from "../assets/logo.png";

export interface PromptPreset {
  name: string;
  prompt: string;
}

export type Theme = "dark" | "light" | "system";

export interface GenSettings {
  theme: Theme;
  systemPrompt: string;
  temperature: number;
  topP: number;
  /** Whether to cap the per-reply length at `maxTokens` (off = unlimited). */
  limitTokens: boolean;
  maxTokens: number;
  topK: number;
  minP: number;
  repeatPenalty: number;
  /** Newline/comma-separated stop sequences. */
  stop: string;
  /** Saved system-prompt presets. */
  presets: PromptPreset[];
  /** Kokoro voice id (0–10). */
  voiceSid: number;
  /** Speech rate multiplier (0.5–2.0). */
  voiceSpeed: number;
  /** Recognise and speak Chinese. Defaults on in the Chinese interface and
   *  off elsewhere, and is a plain setting either way — the Chinese
   *  interface can turn it off. */
  chineseVoice: boolean;
  /** Speaker index into VOICES_ZH, independent of `voiceSid`. */
  voiceSidZh: number;
  /** GPU offload: -1 = auto‑tune by VRAM, 0 = CPU only, >0 = that many layers. */
  gpuLayers: number;
  /** Context window to load the model with: 0 = memory-friendly default (≤8192),
   *  >0 = that many tokens (clamped to the model's trained length). */
  contextLength: number;
  /** How many knowledge-base chunks a question may cite. Was hardcoded at 6
   *  with a ceiling of 12, so a large library answered out of six chunks no
   *  matter how much of it was relevant. */
  ragTopK: number;
  /** Code mode: max agent steps per turn before it pauses. */
  codeMaxSteps: number;
  /** Code mode: default bash-command timeout in seconds. */
  codeBashTimeout: number;
  /** Code mode: sampling temperature for agent steps (0–1). */
  codeTemperature: number;
  /** Code mode: hard ceiling on thinking tokens per agent round (0 = auto).
   *  Over budget the think block closes gracefully — reasoning kept, model
   *  told to act on it. */
  codeThinkBudget: number;
  /** Code mode: per-round generation budget in tokens (0 = auto by think
   *  depth; always clamped to the context window). */
  codeMaxTokens: number;
  /** Code mode: file edits (write/edit/multi_edit) run without approval. */
  codeAutoApproveEdits: boolean;
  /** Code mode: obviously read-only bash commands run without approval. */
  codeAutoRunReadOnly: boolean;
  /** Code mode: run the agent's browser hidden (headless). */
  codeBrowserHeadless: boolean;
  codeMemory: boolean;
  /** Code mode: user-defined skills (named prompt templates, invoked via /). */
  codeSkills: PromptPreset[];
  /** Code mode: names of built-in skills the user turned off. */
  codeDisabledSkills: string[];
  /** Code mode: command prefixes that never need approval (e.g. "npm test"). */
  codeAllowedCommands: string[];
  /** Dark palette: warm charcoal (v1.5) or the cooler pre-v1.5 charcoal. */
  darkScheme: "warm" | "cool";
  /** Light palette: paper white or the softer warm cream. */
  lightScheme: "paper" | "cream";
  /** Code-block highlight palette (chat markdown). */
  codeTheme: "github-dark" | "atom-one-dark" | "monokai" | "nord";
  /** Chat: collapse long code blocks to a header, focus-follow while streaming. */
  chatCollapseCode: boolean;
  /** Canvas iterations: search/replace patches (fast, needs verbatim SEARCH
   *  echoes) or full-document rewrite (the system diffs the stream live —
   *  the reliable choice for smaller models). */
  canvasEditMode: "patch" | "rewrite";
  /** UI zoom (0.9–1.2). Applied via the native webview page zoom. */
  uiScale: number;
  /** Composer send key: plain Enter, or ⌘/Ctrl+Enter (Enter = newline). */
  sendKey: "enter" | "modEnter";
  /** Disable in-app animations regardless of the OS setting. */
  reduceMotion: boolean;
  /** Reading size for model answers. */
  answerSize: "sm" | "md" | "lg";
  /** Auto-generate conversation titles after the first reply. */
  autoTitle: boolean;
  /** Load the last-used model automatically on startup. */
  autoLoadLast: boolean;
  /** HuggingFace endpoint for search/downloads — the official host or a
   *  path-compatible mirror (e.g. https://hf-mirror.com for mainland China). */
  hfEndpoint: string;
}

export const defaultSettings: GenSettings = {
  theme: "dark",
  systemPrompt: "",
  temperature: 0.7,
  topP: 0.95,
  limitTokens: false,
  maxTokens: 1024,
  topK: 40,
  minP: 0.05,
  repeatPenalty: 1.1,
  stop: "",
  presets: [],
  voiceSid: 0,
  voiceSidZh: 0,
  voiceSpeed: 1.0,
  // Seeded from the interface language on first run (see loadSettings) —
  // stored from then on, so the Chinese interface can turn it off.
  chineseVoice: false,
  gpuLayers: -1,
  contextLength: 0,
  ragTopK: 8,
  // 64, was 32: the CalendarApp repro showed 32 starves app-scale one-shots,
  // and 48 still cut the model off ONE error from green (round 16: mid-fix
  // on a duplicate-struct error at the buzzer). Simple tasks end early
  // anyway, so the higher ceiling only costs anything when real work is left.
  codeMaxSteps: 64,
  codeBashTimeout: 60,
  codeTemperature: 0.3,
  codeThinkBudget: 0,
  codeMaxTokens: 0,
  codeAutoApproveEdits: false,
  codeAutoRunReadOnly: true,
  codeBrowserHeadless: false,
  codeMemory: true,
  codeSkills: [],
  codeDisabledSkills: [],
  codeAllowedCommands: [],
  darkScheme: "warm",
  lightScheme: "paper",
  codeTheme: "github-dark",
  chatCollapseCode: true,
  canvasEditMode: "patch",
  uiScale: 1,
  sendKey: "enter",
  reduceMotion: false,
  answerSize: "md",
  autoTitle: true,
  autoLoadLast: true,
  hfEndpoint: "https://huggingface.co",
};

/** The well-known HF endpoints offered as one-click choices. */
export const HF_ENDPOINT_OFFICIAL = "https://huggingface.co";
export const HF_ENDPOINT_MIRROR = "https://hf-mirror.com";

/** kokoro-en-v0_19 speakers, in sid order (the array index IS the speaker id,
 *  so the order must not change). This pack ships exactly these 11 voices;
 *  the larger Kokoro-82M set (af_heart, am_fenrir, …) needs a different model.
 *  Labels carry the Kokoro VOICES.md overall grade so users can pick good ones;
 *  ★ marks the best in each gender. */
/** The five speakers `sherpa-onnx-vits-zh-ll` declares in its own
 *  `G_multisperaker_latest.json` (`speakers`: suyingxue 0, gunian 1,
 *  fushiyu 2, bingjiao 3, bazong 4). Named as the model names them rather
 *  than translated — these are the ids it was trained with. */
export const VOICES_ZH = ["suyingxue", "gunian", "fushiyu", "bingjiao", "bazong"];

export const VOICES = [
  "af · warm female · C+",
  "★ af_bella · female · A-",
  "af_nicole · female · B-",
  "af_sarah · female · C+",
  "af_sky · female · C-",
  "am_adam · male · F+",
  "★ am_michael · male · C+",
  "bf_emma · UK female · B-",
  "bf_isabella · UK female · C",
  "bm_george · UK male · C",
  "bm_lewis · UK male · D+",
];

/** Parse the stop-sequence textarea into a clean array for the backend. */
export function parseStops(raw: string): string[] {
  return raw
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

type CatId = "general" | "chat" | "sampling" | "model" | "code" | "voice" | "data" | "about";

const CAT_ICONS: Record<CatId, string> = {
  general: "M12 3a9 9 0 100 18 9 9 0 000-18zM3 12h18",
  chat: "M21 12a8 8 0 01-8 8H5l-2 2V12a8 8 0 018-8h2a8 8 0 018 8z",
  sampling: "M4 20V10M10 20V4M16 20v-8M22 20H2",
  model: "M4 7l8-4 8 4v10l-8 4-8-4zM4 7l8 4m0 0l8-4m-8 4v10",
  code: "M8 6l-6 6 6 6M16 6l6 6-6 6",
  voice: "M12 3a3 3 0 013 3v6a3 3 0 11-6 0V6a3 3 0 013-3zM19 11a7 7 0 11-14 0M12 18v3",
  data: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  about: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 8h.01M12 12v5",
};

/** Modifier-key glyph for the current platform (send-shortcut labels). */
const MOD_KEY = /mac/i.test(navigator.platform ?? navigator.userAgent) ? "⌘" : "Ctrl +";



/** Aggregated numbers behind the Data-category statistics tiles. */
interface StatsView {
  convs: number;
  msgs: number;
  code: number;
  db: number;
  models: number;
  modelBytes: number;
  kbDocs: number;
  kbChunks: number;
}

/** LM-Studio-style row: label (+hint) left, control right. */
function SetRow({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="field field-row">
      <div className="field-row-text">
        <span>{label}</span>
        {hint && <span className="field-row-hint">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

/**
 * A number that can be switched off entirely: the off/custom pair, with the
 * slider showing only while it is on.
 *
 * Three settings are this control — Sampling's max length and Code's step and
 * timeout ceilings — and each had grown its own copy inline. They had drifted:
 * two said "Limit / No limit" in that order against the third's "No limit /
 * Custom", and their sliders lived in a bordered card of their own holding
 * nothing but a bare bar. One component, so the next one cannot drift either.
 */
function LimitField({
  label,
  tip,
  offLabel,
  onLabel,
  off,
  onOff,
  value,
  children,
}: {
  label: string;
  tip?: string;
  offLabel: string;
  onLabel: string;
  off: boolean;
  onOff: (off: boolean) => void;
  /** Shown beside the label while the setting is on. */
  value?: React.ReactNode;
  /** The slider — rendered only while the setting is on. */
  children: React.ReactNode;
}) {
  return (
    <label className="field">
      <span>
        {tip ? (
          <em className="has-tip" data-tip={tip}>
            {label}
          </em>
        ) : (
          label
        )}
        {!off && value != null && <> <b>{value}</b></>}
      </span>
      <div className="lang-switch">
        <button type="button" className={off ? "active" : ""} onClick={() => onOff(true)}>
          {offLabel}
        </button>
        <button type="button" className={off ? "" : "active"} onClick={() => onOff(false)}>
          {onLabel}
        </button>
      </div>
      {!off && children}
    </label>
  );
}

function Switch({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button type="button" role="switch" aria-checked={on} className={`set-switch ${on ? "on" : ""}`} onClick={onToggle}>
      <span className="set-knob" />
    </button>
  );
}

export function SettingsPanel({
  open,
  value,
  onChange,
  onClose,
  maxTokensLimit = 4096,
  ctxTrainLimit,
  layersLimit,
  onReloadModel,
  reloading = false,
  onDataCleared,
}: {
  open: boolean;
  value: GenSettings;
  onChange: (next: GenSettings) => void;
  onClose: () => void;
  /** Upper bound for the max-length slider — adapts to the loaded model's context. */
  maxTokensLimit?: number;
  /** The loaded model's layer count — ceiling for the GPU-offload slider
   *  (a hardcoded 80 clipped every deeper model). */
  layersLimit?: number | null;
  /** The loaded model's trained context length, used as the slider ceiling. */
  ctxTrainLimit?: number | null;
  /** Reload the current model so context/GPU changes take effect. Absent = no model. */
  onReloadModel?: () => void;
  reloading?: boolean;
  /** Called after the user clears all conversations, so the app can reset. */
  onDataCleared?: () => void;
}) {
  const { t, lang, setLang } = useI18n();
  const confirm = useConfirm();
  const [cat, setCat] = useState<CatId>("general");
  const [presetName, setPresetName] = useState("");
  const [upd, setUpd] = useState<UpdateInfo | null>(null);
  const [checking, setChecking] = useState(false);
  const set = <K extends keyof GenSettings>(key: K, v: GenSettings[K]) =>
    onChange({ ...value, [key]: v });

  /** The hover bubble for a `data-tip` label: its text and where to draw it.
   *
   *  This used to be a `::after` on the label, positioned `fixed` and clamped
   *  to the window. It could not work. A fixed box is positioned against — and
   *  clipped by — the nearest ancestor carrying a transform, and the panel's
   *  open animation (`ui-pop-in`, `both`) leaves one on `.settings-modal`,
   *  which is also `overflow: hidden`. So the bubble was measured against the
   *  window, drawn against a `.field` 362px away, and then cut off by the
   *  panel's own edge — the "sometimes hangs off the window" report, and why
   *  clamping harder never fixed it. A real element under `document.body` has
   *  no such ancestor: viewport coordinates mean the viewport, and nothing
   *  clips it. It also has a measurable height, so the placement below reads
   *  the box it is actually placing instead of estimating from text length. */
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);
  const tipRef = useRef<HTMLDivElement | null>(null);
  const tipAnchor = useRef<DOMRect | null>(null);

  useEffect(() => {
    if (!open) {
      setTip(null);
      return;
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    const onOver = (e: globalThis.MouseEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.("[data-tip]");
      const text = el instanceof HTMLElement ? el.getAttribute("data-tip") : null;
      if (!text || !(el instanceof HTMLElement)) return;
      clearTimeout(timer);
      // The same short delay the CSS transition used to carry, so sweeping the
      // pointer across a column of labels does not strobe.
      timer = setTimeout(() => {
        tipAnchor.current = el.getBoundingClientRect();
        // Provisional: placed for real once the box below has been measured.
        setTip({ text, x: -9999, y: -9999 });
      }, 150);
    };
    const onOut = (e: globalThis.MouseEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.("[data-tip]");
      if (!el) return;
      clearTimeout(timer);
      setTip(null);
    };
    document.addEventListener("mouseover", onOver, true);
    document.addEventListener("mouseout", onOut, true);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("mouseover", onOver, true);
      document.removeEventListener("mouseout", onOut, true);
    };
  }, [open]);

  // Place the bubble once it has a box: above its label, below when there is no
  // room above, and clamped into the window either way.
  useLayoutEffect(() => {
    const box = tipRef.current;
    const anchor = tipAnchor.current;
    if (!box || !anchor || !tip || tip.x !== -9999) return;
    const MARGIN = 12;
    const { width: w, height: h } = box.getBoundingClientRect();
    const x = Math.min(Math.max(MARGIN, anchor.left), window.innerWidth - w - MARGIN);
    const above = anchor.top - 7 - h;
    const want = above >= MARGIN ? above : anchor.bottom + 7;
    const y = Math.max(MARGIN, Math.min(want, window.innerHeight - h - MARGIN));
    setTip({ text: tip.text, x: Math.round(x), y: Math.round(y) });
  }, [tip]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Mounted through the exit animation; fully unmounted after.
  const { mounted, closing } = useExitTransition(open);

  // ---- Data-category statistics ----
  const [stats, setStats] = useState<StatsView | null>(null);
  const refreshStats = () => {
    Promise.all([
      dataStats(),
      listModels().catch(() => []),
      ragStatus().catch(() => null),
    ])
      .then(([ds, models, rs]) =>
        setStats({
          convs: ds.conversations,
          msgs: ds.messages,
          code: ds.codeSessions,
          db: ds.dbBytes,
          models: models.length,
          // sizeMb is mebibytes (bytes / 1024²) — scaling it by 1e6 quietly
          // shaved ~4.6% off every model folder the panel reported.
          modelBytes: models.reduce((a, m) => a + (m.sizeMb ?? 0) * 1024 * 1024, 0),
          kbDocs: rs?.docs ?? 0,
          kbChunks: rs?.chunks ?? 0,
        }),
      )
      .catch(console.error);
  };
  useEffect(() => {
    if (open && cat === "data") refreshStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, cat]);

  // ---- Voice preview ----
  /** Which sample is playing — the two voices are separate models, so each
   *  gets its own button and its own busy state. */
  const [voiceTesting, setVoiceTesting] = useState<"" | "en" | "zh">("");
  const [voiceTestError, setVoiceTestError] = useState("");
  const SAMPLE = {
    zh: "你好，这是我的中文声音。很高兴认识你。",
    en: "Hi! This is how I sound. Nice to meet you.",
  };
  // Routing is by the TEXT: Chinese in it reaches the Chinese voice, English
  // never does. So the flag rides along unchanged and the sample decides.
  const testVoice = (which: "en" | "zh") => {
    if (voiceTesting) return;
    // Must happen synchronously inside the click handler. Waiting until the
    // async native synthesis returns loses Safari/WKWebView user activation.
    primeAudioPlayback();
    setVoiceTestError("");
    setVoiceTesting(which);
    synthesize(
      SAMPLE[which],
      value.voiceSpeed,
      value.voiceSid,
      value.chineseVoice,
      value.voiceSidZh,
    )
      .then((a) => playAudio(decodeAudio(a.audio), a.sampleRate).done)
      .catch((e) => {
        console.error(e);
        setVoiceTestError(typeof e === "string" ? e : ((e as Error)?.message ?? String(e)));
      })
      .finally(() => setVoiceTesting(""));
  };

  const savePreset = () => {
    const name = presetName.trim();
    const prompt = value.systemPrompt.trim();
    if (!name || !prompt) return;
    const presets = [...value.presets.filter((p) => p.name !== name), { name, prompt }];
    onChange({ ...value, presets });
    setPresetName("");
  };
  const deletePreset = (name: string) =>
    onChange({ ...value, presets: value.presets.filter((p) => p.name !== name) });

  const [skillName, setSkillName] = useState("");
  const [skillPrompt, setSkillPrompt] = useState("");
  const saveSkill = () => {
    // Skill names become /slash entries — keep them single-token.
    const name = skillName.trim().replace(/^\/+/, "").replace(/\s+/g, "-");
    const prompt = skillPrompt.trim();
    if (!name || !prompt) return;
    const codeSkills = [...value.codeSkills.filter((s) => s.name !== name), { name, prompt }];
    onChange({ ...value, codeSkills });
    setSkillName("");
    setSkillPrompt("");
  };
  const deleteSkill = (name: string) =>
    onChange({ ...value, codeSkills: value.codeSkills.filter((s) => s.name !== name) });

  const [allowInput, setAllowInput] = useState("");
  const [mcpServers, setMcpServers] = useState<McpServerCfg[]>(loadMcpServers);
  const [mcpStatus, setMcpStatus] = useState<Record<string, string>>({});
  const [mcpName, setMcpName] = useState("");
  const [mcpCmd, setMcpCmd] = useState("");
  const [mcpToken, setMcpToken] = useState("");
  /** Store entry pending placeholder/token input before it can be added. */
  const [skillOff, setSkillOff] = useState<string[]>(disabledSkills);
  const toggleSkill = (name: string) => {
    const next = skillOff.includes(name) ? skillOff.filter((n) => n !== name) : [...skillOff, name];
    setSkillOff(next);
    setDisabledSkills(next);
  };
  const [storeOpen, setStoreOpen] = useState<string | null>(null);
  const [storeInput, setStoreInput] = useState<Record<string, string>>({});
  const addFromStore = (entry: (typeof catalog)["entries"][number]) => {
    const values = storeInput;
    if (entry.transport === "http") {
      const token = (values.token ?? "").trim();
      if ((entry as { needsToken?: boolean }).needsToken && !token) return;
      applyMcp([
        ...mcpServers,
        {
          name: entry.id,
          enabled: true,
          transport: "http",
          url: (entry as { url?: string }).url ?? "",
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        },
      ]);
    } else {
      const args = ((entry as { args?: string[] }).args ?? []).map((a) => {
        let out = a;
        for (const ph of (entry as { placeholders?: { key: string }[] }).placeholders ?? []) {
          out = out.split(`{${ph.key}}`).join((values[ph.key] ?? "").trim());
        }
        return out;
      });
      if (args.some((a) => /\{[a-z]+\}/.test(a))) return; // unfilled placeholder
      applyMcp([
        ...mcpServers,
        { name: entry.id, enabled: true, transport: "stdio", command: (entry as { command?: string }).command ?? "npx", args },
      ]);
    }
    setStoreOpen(null);
    setStoreInput({});
  };
  const applyMcp = (list: McpServerCfg[]) => {
    setMcpServers(list);
    saveMcpServers(list);
    void syncMcpServers(list).then((rs) => {
      setMcpStatus((prev) => {
        const st: Record<string, string> = { ...prev };
        for (const r of rs) st[r.server] = r.error ? `✗ ${r.error.slice(0, 90)}` : `✓ ${r.tools} tools`;
        return st;
      });
    });
  };
  const addMcp = () => {
    const name = mcpName.trim().replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 16);
    const cmd = mcpCmd.trim();
    if (!name || !cmd || mcpServers.some((x) => x.name === name)) return;
    const token = mcpToken.trim();
    const cfg: McpServerCfg = /^https?:\/\//.test(cmd)
      ? { name, enabled: true, transport: "http", url: cmd, headers: token ? { Authorization: `Bearer ${token}` } : {} }
      : { name, enabled: true, transport: "stdio", command: cmd.split(/\s+/)[0], args: cmd.split(/\s+/).slice(1) };
    applyMcp([...mcpServers, cfg]);
    setMcpName("");
    setMcpCmd("");
    setMcpToken("");
  };
  const addAllow = () => {
    const p = allowInput.trim();
    if (!p) return;
    if (!value.codeAllowedCommands.includes(p)) {
      set("codeAllowedCommands", [...value.codeAllowedCommands, p]);
    }
    setAllowInput("");
  };

  const cats: { id: CatId; label: string }[] = [
    { id: "general", label: t("setCatGeneral") },
    { id: "chat", label: t("setCatChat") },
    { id: "sampling", label: t("setCatSampling") },
    { id: "model", label: t("setCatModel") },
    { id: "code", label: "Code" },
    { id: "voice", label: t("setCatVoice") },
    { id: "data", label: t("setCatData") },
    { id: "about", label: t("setCatAbout") },
  ];

  if (!mounted) return null;

  return createPortal(
    <>
    {tip && (
      // Outside the overlay on purpose: the overlay's descendants establish
      // containing blocks and clip, which is the whole reason this is not a
      // pseudo-element on the label any more.
      <div
        ref={tipRef}
        className={`settings-tip ${tip.x === -9999 ? "" : "placed"}`}
        style={{ left: tip.x, top: tip.y }}
      >
        {tip.text}
      </div>
    )}
    <div className={`settings-overlay ${closing ? "closing" : ""}`} onMouseDown={onClose}>
      <div className="settings-modal" onMouseDown={(e) => e.stopPropagation()}>
        <aside className="settings-nav">
          <div className="settings-nav-title">{t("settingsTitle")}</div>
          {cats.map((c) => (
            <button
              key={c.id}
              className={`settings-nav-item ${cat === c.id ? "active" : ""}`}
              onClick={() => setCat(c.id)}
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <path d={CAT_ICONS[c.id]} />
              </svg>
              {c.label}
            </button>
          ))}
        </aside>

        <div className="settings-pane">
          {/* Fixed pane header: category title + close. Only the body scrolls. */}
          <div className="settings-pane-head">
            <span className="settings-pane-title">{cats.find((c) => c.id === cat)?.label}</span>
            <button className="settings-close" onClick={onClose} aria-label={t("cancel")}>
              <Icon name="x" size={11} strokeWidth={2.2} />
            </button>
          </div>

          <div className="settings-pane-body">
          {cat === "general" && (
            <>
              <SetRow label={t("language")}>
                <div className="lang-switch">
                  {LANGS.map((l) => (
                    <button key={l.id} type="button" className={lang === l.id ? "active" : ""} onClick={() => setLang(l.id)}>
                      {l.label}
                    </button>
                  ))}
                </div>
              </SetRow>
              <SetRow label={t("theme")}>
                <div className="lang-switch">
                  <button type="button" className={value.theme === "system" ? "active" : ""} onClick={() => set("theme", "system")}>{t("themeSystem")}</button>
                  <button type="button" className={value.theme === "light" ? "active" : ""} onClick={() => set("theme", "light")}>{t("themeLight")}</button>
                  <button type="button" className={value.theme === "dark" ? "active" : ""} onClick={() => set("theme", "dark")}>{t("themeDark")}</button>
                </div>
              </SetRow>
              <SetRow label={t("setDarkScheme")} hint={t("setDarkSchemeHint")}>
                <div className="lang-switch">
                  <button type="button" className={value.darkScheme === "warm" ? "active" : ""} onClick={() => set("darkScheme", "warm")}>
                    <span className="scheme-dot" style={{ background: "#201f1d" }} />
                    {t("schemeWarmCharcoal")}
                  </button>
                  <button type="button" className={value.darkScheme === "cool" ? "active" : ""} onClick={() => set("darkScheme", "cool")}>
                    <span className="scheme-dot" style={{ background: "#0e0f12" }} />
                    {t("schemeCoolCharcoal")}
                  </button>
                </div>
              </SetRow>
              <SetRow label={t("setLightScheme")} hint={t("setLightSchemeHint")}>
                <div className="lang-switch">
                  <button type="button" className={value.lightScheme === "paper" ? "active" : ""} onClick={() => set("lightScheme", "paper")}>
                    <span className="scheme-dot" style={{ background: "#faf9f7" }} />
                    {t("schemePaper")}
                  </button>
                  <button type="button" className={value.lightScheme === "cream" ? "active" : ""} onClick={() => set("lightScheme", "cream")}>
                    <span className="scheme-dot" style={{ background: "#f3eee4" }} />
                    {t("schemeCream")}
                  </button>
                </div>
              </SetRow>
              <SetRow label={t("setUiScale")} hint={t("setUiScaleHint")}>
                <div className="lang-switch">
                  {[0.9, 1, 1.1, 1.2].map((s) => (
                    <button key={s} type="button" className={Math.abs(value.uiScale - s) < 0.01 ? "active" : ""} onClick={() => set("uiScale", s)}>
                      {Math.round(s * 100)}%
                    </button>
                  ))}
                </div>
              </SetRow>
              <SetRow label={t("setSendKey")} hint={t("setSendKeyHint")}>
                <div className="lang-switch">
                  <button type="button" className={value.sendKey === "enter" ? "active" : ""} onClick={() => set("sendKey", "enter")}>Enter</button>
                  <button type="button" className={value.sendKey === "modEnter" ? "active" : ""} onClick={() => set("sendKey", "modEnter")}>{MOD_KEY} Enter</button>
                </div>
              </SetRow>
              <SetRow label={t("setReduceMotion")} hint={t("setReduceMotionHint")}>
                <Switch on={value.reduceMotion} onToggle={() => set("reduceMotion", !value.reduceMotion)} />
              </SetRow>
              <SetRow label={t("errorLog")} hint={t("errorLogHint")}>
                {/* Open and clear sit one above the other: they act on the same
                    file, so a row each would read as two unrelated settings. */}
                <div className="log-actions">
                  <div className="lang-switch">
                    <button type="button" onClick={() => { void openErrorLog().catch(() => {}); }}>
                      {t("errorLogOpen")}
                    </button>
                  </div>
                  <button
                    type="button"
                    className="data-btn danger"
                    onClick={async () => {
                      if (
                        !(await confirm({
                          message: t("confirmClearErrorLog"),
                          title: t("errorLogClear"),
                          confirmLabel: t("errorLogClear"),
                          danger: true,
                        }))
                      ) {
                        return;
                      }
                      void clearErrorLog().catch(console.error);
                    }}
                  >
                    {t("errorLogClear")}
                  </button>
                </div>
              </SetRow>
            </>
          )}

          {cat === "chat" && (
            <>
              <label className="field">
                <span>{t("systemPrompt")}</span>
                <textarea
                  rows={4}
                  placeholder={t("systemPromptPh")}
                  value={value.systemPrompt}
                  onChange={(e) => set("systemPrompt", e.target.value)}
                />
              </label>
              <div className="field">
                <span>{t("presets")}</span>
                {value.presets.length > 0 && (
                  <div className="preset-chips">
                    {value.presets.map((p) => (
                      <span key={p.name} className="preset-chip">
                        <button type="button" className="preset-apply" title={p.prompt} onClick={() => set("systemPrompt", p.prompt)}>
                          {p.name}
                        </button>
                        <button type="button" className="preset-del" title={t("cancel")} onClick={() => deletePreset(p.name)}><Icon name="x" size={11} strokeWidth={2.2} /></button>
                      </span>
                    ))}
                  </div>
                )}
                <div className="preset-add">
                  <input
                    type="text"
                    placeholder={t("presetNamePh")}
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        savePreset();
                      }
                    }}
                  />
                  <button type="button" onClick={savePreset} disabled={!presetName.trim() || !value.systemPrompt.trim()}>
                    {t("presetSave")}
                  </button>
                </div>
              </div>
              <SetRow label={t("setCodeTheme")} hint={t("setCodeThemeHint")}>
                <div className="lang-switch">
                  {(Object.keys(CODE_THEMES) as CodeTheme[]).map((k) => (
                    <button key={k} type="button" className={value.codeTheme === k ? "active" : ""} onClick={() => set("codeTheme", k)}>
                      {CODE_THEMES[k].label}
                    </button>
                  ))}
                </div>
              </SetRow>
              <SetRow label={t("chatCollapseCode")} hint={t("chatCollapseCodeHint")}>
                <Switch
                  on={value.chatCollapseCode}
                  onToggle={() => set("chatCollapseCode", !value.chatCollapseCode)}
                />
              </SetRow>
              <SetRow label={t("canvasEditModeLabel")} hint={t("canvasEditModeHint")}>
                <div className="lang-switch">
                  <button type="button" className={value.canvasEditMode === "patch" ? "active" : ""} onClick={() => set("canvasEditMode", "patch")}>{t("canvasEditModePatch")}</button>
                  <button type="button" className={value.canvasEditMode === "rewrite" ? "active" : ""} onClick={() => set("canvasEditMode", "rewrite")}>{t("canvasEditModeRewrite")}</button>
                </div>
              </SetRow>
              <SetRow label={t("setAnswerSize")} hint={t("setAnswerSizeHint")}>
                <div className="lang-switch">
                  <button type="button" className={value.answerSize === "sm" ? "active" : ""} onClick={() => set("answerSize", "sm")}>{t("sizeSm")}</button>
                  <button type="button" className={value.answerSize === "md" ? "active" : ""} onClick={() => set("answerSize", "md")}>{t("sizeMd")}</button>
                  <button type="button" className={value.answerSize === "lg" ? "active" : ""} onClick={() => set("answerSize", "lg")}>{t("sizeLg")}</button>
                </div>
              </SetRow>
              <SetRow label={t("setAutoTitle")} hint={t("setAutoTitleHint")}>
                <Switch on={value.autoTitle} onToggle={() => set("autoTitle", !value.autoTitle)} />
              </SetRow>

              <label className="field">
                <span>
                  <em className="has-tip" data-tip={t("tipRagTopK")}>{t("ragTopK")}</em> <b>{value.ragTopK}</b>
                </span>
                <input
                  type="range"
                  min={1}
                  max={32}
                  step={1}
                  value={value.ragTopK}
                  onChange={(e) => set("ragTopK", Number(e.target.value))}
                />
              </label>
              <div className="settings-hint">{t("ragTopKHint")}</div>
            </>
          )}

          {cat === "sampling" && (
            <>
              <label className="field">
                <span>
                  <em className="has-tip" data-tip={t("tipTemperature")}>{t("temperature")}</em> <b>{value.temperature.toFixed(2)}</b>
                </span>
                <input type="range" min={0} max={1.5} step={0.05} value={value.temperature} onChange={(e) => set("temperature", Number(e.target.value))} />
              </label>
              <label className="field">
                <span>
                  <em className="has-tip" data-tip={t("tipTopP")}>Top-P</em> <b>{value.topP.toFixed(2)}</b>
                </span>
                <input type="range" min={0.1} max={1} step={0.01} value={value.topP} onChange={(e) => set("topP", Number(e.target.value))} />
              </label>
              <LimitField
                label={t("maxTokens")}
                tip={t("tipMaxTokens")}
                offLabel={t("noLimit")}
                onLabel={t("gpuCustom")}
                off={!value.limitTokens}
                onOff={(o) => set("limitTokens", !o)}
                value={Math.min(value.maxTokens, maxTokensLimit)}
              >
                <input
                  type="range"
                  min={128}
                  max={maxTokensLimit}
                  step={128}
                  value={Math.min(value.maxTokens, maxTokensLimit)}
                  onChange={(e) => set("maxTokens", Number(e.target.value))}
                />
              </LimitField>
              <label className="field">
                <span>
                  <em className="has-tip" data-tip={t("tipTopK")}>Top-K</em> <b>{value.topK === 0 ? t("off") : value.topK}</b>
                </span>
                <input type="range" min={0} max={100} step={1} value={value.topK} onChange={(e) => set("topK", Number(e.target.value))} />
              </label>
              <label className="field">
                <span>
                  <em className="has-tip" data-tip={t("tipMinP")}>Min-P</em> <b>{value.minP.toFixed(2)}</b>
                </span>
                <input type="range" min={0} max={0.5} step={0.01} value={value.minP} onChange={(e) => set("minP", Number(e.target.value))} />
              </label>
              <label className="field">
                <span>
                  <em className="has-tip" data-tip={t("tipRepeatPenalty")}>{t("repeatPenalty")}</em> <b>{value.repeatPenalty.toFixed(2)}</b>
                </span>
                <input type="range" min={1} max={1.5} step={0.01} value={value.repeatPenalty} onChange={(e) => set("repeatPenalty", Number(e.target.value))} />
              </label>
              <label className="field">
                <span><em className="has-tip" data-tip={t("tipStopSeqs")}>{t("stopSeqs")}</em></span>
                <textarea rows={2} placeholder={t("stopSeqsPh")} value={value.stop} onChange={(e) => set("stop", e.target.value)} />
              </label>
              <button
                className="settings-reset"
                onClick={() =>
                  onChange({
                    ...value,
                    temperature: defaultSettings.temperature,
                    topP: defaultSettings.topP,
                    limitTokens: defaultSettings.limitTokens,
                    maxTokens: defaultSettings.maxTokens,
                    topK: defaultSettings.topK,
                    minP: defaultSettings.minP,
                    repeatPenalty: defaultSettings.repeatPenalty,
                    stop: defaultSettings.stop,
                  })
                }
              >
                {t("resetDefaults")}
              </button>
            </>
          )}

          {cat === "model" && (
            <>
              <label className="field">
                <span><em className="has-tip" data-tip={t("tipGpuAccel")}>{t("gpuAccel")}</em></span>
                <div className="lang-switch">
                  <button type="button" className={value.gpuLayers < 0 ? "active" : ""} onClick={() => set("gpuLayers", -1)}>{t("gpuAuto")}</button>
                  <button type="button" className={value.gpuLayers === 0 ? "active" : ""} onClick={() => set("gpuLayers", 0)}>{t("gpuOff")}</button>
                  <button
                    type="button"
                    className={value.gpuLayers > 0 ? "active" : ""}
                    onClick={() => set("gpuLayers", value.gpuLayers > 0 ? value.gpuLayers : 20)}
                  >
                    {t("gpuCustom")}
                  </button>
                </div>
              </label>
              {value.gpuLayers > 0 && (
                <label className="field">
                  <span>
                    {t("gpuLayersLabel")} <b>{value.gpuLayers}</b>
                  </span>
                  <input
                    type="range"
                    min={1}
                    max={Math.max(1, layersLimit ?? 80)}
                    step={1}
                    value={value.gpuLayers}
                    onChange={(e) => set("gpuLayers", Number(e.target.value))}
                  />
                </label>
              )}
              <div className="settings-hint">{t("gpuHint")}</div>

              <label className="field">
                <span><em className="has-tip" data-tip={t("tipCtxLength")}>{t("ctxLength")}</em></span>
                <div className="lang-switch">
                  <button type="button" className={value.contextLength <= 0 ? "active" : ""} onClick={() => set("contextLength", 0)}>{t("ctxAuto")}</button>
                  <button
                    type="button"
                    className={value.contextLength > 0 ? "active" : ""}
                    onClick={() => set("contextLength", value.contextLength > 0 ? value.contextLength : 8192)}
                  >
                    {t("gpuCustom")}
                  </button>
                </div>
              </label>
              {value.contextLength > 0 &&
                (() => {
                  // Slider ceiling = the loaded model's trained context (fallback
                  // 32768 when nothing is loaded) — no point offering more.
                  const ctxMax = Math.max(4096, ctxTrainLimit ?? 32768);
                  return (
                    <label className="field">
                      <span>
                        {t("ctxTokens")} <b>{Math.min(value.contextLength, ctxMax)}</b>
                      </span>
                      <input
                        type="range"
                        min={2048}
                        max={ctxMax}
                        step={2048}
                        value={Math.min(value.contextLength, ctxMax)}
                        onChange={(e) => set("contextLength", Number(e.target.value))}
                      />
                    </label>
                  );
                })()}
              <div className="settings-hint">{t("ctxHint")}</div>
              {onReloadModel && (
                <button className="settings-reload" onClick={onReloadModel} disabled={reloading}>
                  {reloading ? "…" : t("reloadApply")}
                </button>
              )}
              <SetRow label={t("setAutoLoadLast")} hint={t("setAutoLoadLastHint")}>
                <Switch on={value.autoLoadLast} onToggle={() => set("autoLoadLast", !value.autoLoadLast)} />
              </SetRow>
              <SetRow label={t("modelsFolder")} hint={t("modelsFolderHint")}>
                <button type="button" className="data-btn" onClick={() => void openModelsDir().catch(console.error)}>
                  {t("openModelsDir")}
                </button>
              </SetRow>

              <label className="field">
                <span><em className="has-tip" data-tip={t("tipHfEndpoint")}>{t("hfEndpoint")}</em></span>
                <div className="lang-switch">
                  <button
                    type="button"
                    className={value.hfEndpoint === HF_ENDPOINT_OFFICIAL ? "active" : ""}
                    onClick={() => set("hfEndpoint", HF_ENDPOINT_OFFICIAL)}
                  >
                    {t("hfEndpointOfficial")}
                  </button>
                  <button
                    type="button"
                    className={value.hfEndpoint === HF_ENDPOINT_MIRROR ? "active" : ""}
                    onClick={() => set("hfEndpoint", HF_ENDPOINT_MIRROR)}
                  >
                    hf-mirror.com
                  </button>
                  <button
                    type="button"
                    className={
                      value.hfEndpoint !== HF_ENDPOINT_OFFICIAL && value.hfEndpoint !== HF_ENDPOINT_MIRROR
                        ? "active"
                        : ""
                    }
                    onClick={() => {
                      if (value.hfEndpoint === HF_ENDPOINT_OFFICIAL || value.hfEndpoint === HF_ENDPOINT_MIRROR) {
                        set("hfEndpoint", "https://");
                      }
                    }}
                  >
                    {t("hfEndpointCustom")}
                  </button>
                </div>
              </label>
              {value.hfEndpoint !== HF_ENDPOINT_OFFICIAL && value.hfEndpoint !== HF_ENDPOINT_MIRROR && (
                <div className="preset-add">
                  <input
                    type="text"
                    placeholder="https://…"
                    value={value.hfEndpoint}
                    onChange={(e) => set("hfEndpoint", e.target.value)}
                    spellCheck={false}
                  />
                </div>
              )}
              <div className="settings-hint">{t("hfEndpointHint")}</div>
            </>
          )}

          {cat === "code" && (
            <>
              <LimitField
                label={t("cmMaxSteps")}
                offLabel={t("noLimit")}
                onLabel={t("gpuCustom")}
                off={value.codeMaxSteps <= 0}
                onOff={(o) => set("codeMaxSteps", o ? 0 : 64)}
                value={value.codeMaxSteps}
              >
                <input
                  type="range"
                  min={8}
                  max={256}
                  step={4}
                  value={value.codeMaxSteps}
                  onChange={(e) => set("codeMaxSteps", Number(e.target.value))}
                />
              </LimitField>
              <div className="settings-hint">{t("cmMaxStepsHint")}</div>

              <LimitField
                label={t("cmBashTimeout")}
                offLabel={t("noLimit")}
                onLabel={t("gpuCustom")}
                off={value.codeBashTimeout <= 0}
                onOff={(o) => set("codeBashTimeout", o ? 0 : 60)}
                value={`${value.codeBashTimeout}s`}
              >
                <input
                  type="range"
                  min={10}
                  max={1800}
                  step={10}
                  value={value.codeBashTimeout}
                  onChange={(e) => set("codeBashTimeout", Number(e.target.value))}
                />
              </LimitField>
              <div className="settings-hint">{t("cmBashTimeoutHint")}</div>

              <label className="field">
                <span>
                  {t("cmTemp")} <b>{value.codeTemperature.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={value.codeTemperature}
                  onChange={(e) => set("codeTemperature", Number(e.target.value))}
                />
              </label>
              <div className="settings-hint">{t("cmTempHint")}</div>

              <label className="field">
                <span>
                  {t("cmThinkBudget")}{" "}
                  <b>{value.codeThinkBudget > 0 ? value.codeThinkBudget : t("cmThinkBudgetOff")}</b>
                </span>
                <input
                  type="range"
                  min={0}
                  max={maxTokensLimit}
                  step={250}
                  value={value.codeThinkBudget}
                  onChange={(e) => set("codeThinkBudget", Number(e.target.value))}
                />
              </label>
              <div className="settings-hint">{t("cmThinkBudgetHint")}</div>

              <label className="field">
                <span>
                  {t("cmMaxTokens")}{" "}
                  <b>{value.codeMaxTokens > 0 ? value.codeMaxTokens : t("cmThinkBudgetOff")}</b>
                </span>
                <input
                  type="range"
                  min={0}
                  max={maxTokensLimit}
                  step={512}
                  value={value.codeMaxTokens}
                  onChange={(e) => set("codeMaxTokens", Number(e.target.value))}
                />
              </label>
              <div className="settings-hint">{t("cmMaxTokensHint")}</div>

              <SetRow label={t("cmAutoEdits")} hint={t("cmAutoEditsHint")}>
                <Switch
                  on={value.codeAutoApproveEdits}
                  onToggle={() => set("codeAutoApproveEdits", !value.codeAutoApproveEdits)}
                />
              </SetRow>
              <SetRow label={t("cmAutoReadOnly")} hint={t("cmAutoReadOnlyHint")}>
                <Switch
                  on={value.codeAutoRunReadOnly}
                  onToggle={() => set("codeAutoRunReadOnly", !value.codeAutoRunReadOnly)}
                />
              </SetRow>
              <SetRow label={t("cmHeadless")} hint={t("cmHeadlessHint")}>
                <Switch
                  on={value.codeBrowserHeadless}
                  onToggle={() => set("codeBrowserHeadless", !value.codeBrowserHeadless)}
                />
              </SetRow>
              <SetRow label={t("cmMemory")} hint={t("cmMemoryHint")}>
                <Switch
                  on={value.codeMemory}
                  onToggle={() => set("codeMemory", !value.codeMemory)}
                />
              </SetRow>

              <div className="field">
                <span>{t("cmAllowlist")}</span>
                {value.codeAllowedCommands.length > 0 && (
                  <div className="preset-chips">
                    {value.codeAllowedCommands.map((p) => (
                      <span key={p} className="preset-chip">
                        <span className="preset-apply allow-chip">{p}</span>
                        <button
                          type="button"
                          className="preset-del"
                          title={t("cancel")}
                          onClick={() =>
                            set("codeAllowedCommands", value.codeAllowedCommands.filter((x) => x !== p))
                          }
                        ><Icon name="x" size={11} strokeWidth={2.2} /></button>
                      </span>
                    ))}
                  </div>
                )}
                <div className="preset-add">
                  <input
                    type="text"
                    placeholder={t("cmAllowlistPh")}
                    value={allowInput}
                    onChange={(e) => setAllowInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        addAllow();
                      }
                    }}
                  />
                  <button type="button" onClick={addAllow} disabled={!allowInput.trim()}>
                    {t("presetSave")}
                  </button>
                </div>
              </div>
              <div className="settings-hint">{t("cmAllowlistHint")}</div>

              <div className="field">
                <span>{t("cmBuiltinSkills")}</span>
                <div className="skill-rows">
                  {BUILTIN_SKILLS.map((s) => {
                    const enabled = !value.codeDisabledSkills.includes(s.name);
                    return (
                      <button
                        key={s.name}
                        type="button"
                        className={`skill-row ${enabled ? "on" : ""}`}
                        onClick={() =>
                          set(
                            "codeDisabledSkills",
                            enabled
                              ? [...value.codeDisabledSkills, s.name]
                              : value.codeDisabledSkills.filter((n) => n !== s.name),
                          )
                        }
                      >
                        <span className="skill-row-name">/{s.name}</span>
                        <span className="skill-row-desc">{lang === "zh" ? s.desc.zh : s.desc.en}</span>
                        <span className="skill-row-toggle" aria-hidden="true">
                          <span className="skill-row-knob" />
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="field">
                <span>{t("cmSkills")}</span>
                {value.codeSkills.length > 0 && (
                  <div className="preset-chips">
                    {value.codeSkills.map((s) => (
                      <span key={s.name} className="preset-chip">
                        <button
                          type="button"
                          className="preset-apply"
                          title={s.prompt}
                          onClick={() => {
                            setSkillName(s.name);
                            setSkillPrompt(s.prompt);
                          }}
                        >
                          /{s.name}
                        </button>
                        <button type="button" className="preset-del" title={t("cancel")} onClick={() => deleteSkill(s.name)}><Icon name="x" size={11} strokeWidth={2.2} /></button>
                      </span>
                    ))}
                  </div>
                )}
                <div className="preset-add">
                  <input
                    type="text"
                    placeholder={t("cmSkillNamePh")}
                    value={skillName}
                    onChange={(e) => setSkillName(e.target.value)}
                  />
                  <button type="button" onClick={saveSkill} disabled={!skillName.trim() || !skillPrompt.trim()}>
                    {t("presetSave")}
                  </button>
                </div>
                <textarea
                  rows={3}
                  placeholder={t("cmSkillPromptPh")}
                  value={skillPrompt}
                  onChange={(e) => setSkillPrompt(e.target.value)}
                />
              </div>
              <div className="settings-hint">{t("cmSkillsHint")}</div>

              <div className="field">
                <span>{t("cmMcp")}</span>
                {mcpServers.length > 0 && (
                  <div className="skill-rows">
                    {mcpServers.map((sv) => (
                      <div key={sv.name} className={`skill-row mcp-row ${sv.enabled ? "on" : ""}`}>
                        <button
                          type="button"
                          className="skill-row-toggle"
                          title={sv.enabled ? "on" : "off"}
                          onClick={() =>
                            applyMcp(mcpServers.map((x) => (x.name === sv.name ? { ...x, enabled: !x.enabled } : x)))
                          }
                        >
                          <span className="skill-row-knob" />
                        </button>
                        <span className="skill-row-name">{sv.name}</span>
                        <span className="skill-row-desc">
                          {sv.transport === "http" ? sv.url : [sv.command, ...(sv.args ?? [])].join(" ")}
                          {mcpStatus[sv.name] ? ` · ${mcpStatus[sv.name]}` : ""}
                        </span>
                        <label className="mcp-trust" title={t("cmMcpHint")}>
                          <input
                            type="checkbox"
                            checked={sv.trusted === true}
                            onChange={(e) =>
                              applyMcp(mcpServers.map((x) => (x.name === sv.name ? { ...x, trusted: e.target.checked } : x)))
                            }
                          />
                          {t("cmMcpTrusted")}
                        </label>
                        <button type="button" className="preset-del" title={t("cancel")} onClick={() => applyMcp(mcpServers.filter((x) => x.name !== sv.name))}>
                          <Icon name="x" size={11} strokeWidth={2.2} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="preset-add">
                  <input type="text" placeholder={t("cmMcpNamePh")} value={mcpName} onChange={(e) => setMcpName(e.target.value)} style={{ maxWidth: 120 }} />
                  <input type="text" placeholder={t("cmMcpCmdPh")} value={mcpCmd} onChange={(e) => setMcpCmd(e.target.value)} />
                  <button type="button" onClick={addMcp} disabled={!mcpName.trim() || !mcpCmd.trim()}>
                    {t("presetSave")}
                  </button>
                </div>
                {/^https?:\/\//.test(mcpCmd.trim()) && (
                  <input type="password" placeholder={t("cmMcpTokenPh")} value={mcpToken} onChange={(e) => setMcpToken(e.target.value)} />
                )}
              </div>
              <div className="settings-hint">{t("cmMcpHint")}</div>

              <div className="field">
                <span>{t("cmMcpStore")}</span>
                <div className="skill-rows">
                  {catalog.entries.map((en) => {
                    const installed = mcpServers.some((x) => x.name === en.id);
                    const needsInput =
                      (en as { needsToken?: boolean }).needsToken === true ||
                      ((en as { placeholders?: unknown[] }).placeholders?.length ?? 0) > 0;
                    const open = storeOpen === en.id;
                    return (
                      <div key={en.id} className="skill-row mcp-row on">
                        <span className="skill-row-name">{lang === "zh" ? en.title.zh : en.title.en}</span>
                        <span className="skill-row-desc" title={lang === "zh" ? en.permNote.zh : en.permNote.en}>
                          {lang === "zh" ? en.desc.zh : en.desc.en}
                        </span>
                        {en.probe != null && <span className="mcp-cert">✓ {t("cmMcpCertified")}</span>}
                        <button
                          type="button"
                          className="preset-apply"
                          disabled={installed}
                          onClick={() => {
                            if (installed) return;
                            if (needsInput && !open) {
                              setStoreOpen(en.id);
                              setStoreInput({});
                              return;
                            }
                            addFromStore(en);
                          }}
                        >
                          {installed ? t("cmMcpAdded") : t("cmMcpAddBtn")}
                        </button>
                        {open && !installed && (
                          <div className="mcp-store-inputs">
                            {((en as { placeholders?: { key: string; label: { zh: string; en: string } }[] }).placeholders ?? []).map((ph) => (
                              <input
                                key={ph.key}
                                type="text"
                                placeholder={lang === "zh" ? ph.label.zh : ph.label.en}
                                value={storeInput[ph.key] ?? ""}
                                onChange={(ev) => setStoreInput((v) => ({ ...v, [ph.key]: ev.target.value }))}
                              />
                            ))}
                            {(en as { needsToken?: boolean }).needsToken && (
                              <input
                                type="password"
                                placeholder={t("cmMcpTokenPh")}
                                value={storeInput.token ?? ""}
                                onChange={(ev) => setStoreInput((v) => ({ ...v, token: ev.target.value }))}
                              />
                            )}
                            <button type="button" className="preset-apply" onClick={() => addFromStore(en)}>
                              {t("cmMcpAddBtn")}
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="settings-hint">{t("cmMcpStoreHint")}</div>

              <div className="field">
                <span>{t("cmSkillFiles")}</span>
                <div className="skill-rows">
                  {officialSkills().map((sk) => {
                    const on = !skillOff.includes(sk.name);
                    return (
                      <button
                        key={sk.name}
                        type="button"
                        className={`skill-row ${on ? "on" : ""}`}
                        onClick={() => toggleSkill(sk.name)}
                      >
                        <span className="skill-row-name">{sk.name}</span>
                        <span className="skill-row-desc">{sk.description}</span>
                        <span className="skill-row-toggle" aria-hidden="true">
                          <span className="skill-row-knob" />
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
              <div className="settings-hint">{t("cmSkillFilesHint")}</div>
            </>
          )}

          {cat === "voice" && (
            <>
              <SetRow label={t("chineseVoice")} hint={t("chineseVoiceHint")}>
                <Switch
                  on={value.chineseVoice}
                  onToggle={() => set("chineseVoice", !value.chineseVoice)}
                />
              </SetRow>
              <label className="field">
                <span>{value.chineseVoice ? t("voiceEn") : t("voice")}</span>
                <Select
                  className="field-select"
                  value={value.voiceSid}
                  ariaLabel={value.chineseVoice ? t("voiceEn") : t("voice")}
                  onChange={(v) => set("voiceSid", v)}
                  options={VOICES.map((name, i) => ({ value: i, label: name }))}
                />
              </label>
              {value.chineseVoice && (
                <label className="field">
                  <span>{t("voiceZh")}</span>
                  <Select
                    className="field-select"
                    value={value.voiceSidZh}
                    ariaLabel={t("voiceZh")}
                    onChange={(v) => set("voiceSidZh", v)}
                    options={VOICES_ZH.map((name, i) => ({ value: i, label: name }))}
                  />
                </label>
              )}
              {value.chineseVoice && <div className="settings-hint">{t("voiceZhHint")}</div>}
              <label className="field">
                <span>
                  {t("voiceSpeed")} <b>{value.voiceSpeed.toFixed(2)}×</b>
                </span>
                <input type="range" min={0.5} max={2} step={0.05} value={value.voiceSpeed} onChange={(e) => set("voiceSpeed", Number(e.target.value))} />
              </label>
              <SetRow label={t("voicePreview")} hint={t("voicePreviewHint")}>
                {value.chineseVoice ? (
                  <div className="row-btns">
                    <button
                      type="button"
                      className="data-btn"
                      disabled={!!voiceTesting}
                      onClick={() => testVoice("zh")}
                    >
                      {voiceTesting === "zh" ? "…" : t("voicePreviewZh")}
                    </button>
                    <button
                      type="button"
                      className="data-btn"
                      disabled={!!voiceTesting}
                      onClick={() => testVoice("en")}
                    >
                      {voiceTesting === "en" ? "…" : t("voicePreviewEn")}
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="data-btn"
                    disabled={!!voiceTesting}
                    onClick={() => testVoice(lang === "zh" ? "zh" : "en")}
                  >
                    {voiceTesting ? "…" : t("voicePreviewBtn")}
                  </button>
                )}
              </SetRow>
              {voiceTestError && <div className="settings-hint settings-error">{voiceTestError}</div>}
              <div className="settings-hint">{t("voiceEngineHint")}</div>
            </>
          )}

          {cat === "data" && (
            <>
              <div className="stats-grid">
                <div className="stat-tile">
                  <span className="stat-num">{stats ? stats.convs : "–"}</span>
                  <span className="stat-label">{t("statConvs")}</span>
                </div>
                <div className="stat-tile">
                  <span className="stat-num">{stats ? stats.msgs : "–"}</span>
                  <span className="stat-label">{t("statMsgs")}</span>
                </div>
                <div className="stat-tile">
                  <span className="stat-num">{stats ? stats.code : "–"}</span>
                  <span className="stat-label">{t("statCodeSessions")}</span>
                </div>
                <div className="stat-tile">
                  <span className="stat-num">
                    {stats ? stats.models : "–"}
                    {stats && stats.modelBytes > 0 && <em className="stat-sub">{fmtBytes(stats.modelBytes)}</em>}
                  </span>
                  <span className="stat-label">{t("statModels")}</span>
                </div>
                <div className="stat-tile">
                  <span className="stat-num">{stats ? stats.kbDocs : "–"}</span>
                  <span className="stat-label">{t("statKbDocs")}</span>
                </div>
                <div className="stat-tile">
                  <span className="stat-num">{stats ? stats.kbChunks : "–"}</span>
                  <span className="stat-label">{t("statKbChunks")}</span>
                </div>
              </div>
              <div className="settings-hint stats-db-line">
                {t("statDbSize")}: {stats ? fmtBytes(stats.db) : "–"}
              </div>
              <SetRow label={t("dataFolder")} hint={t("dataHint")}>
                <button type="button" className="data-btn" onClick={() => void openDataDir().catch(console.error)}>
                  {t("openDataDir")}
                </button>
              </SetRow>
              <SetRow label={t("clearAllChats")} hint={t("clearChatsHint")}>
                <button
                  type="button"
                  className="data-btn danger"
                  onClick={async () => {
                    if (
                      !(await confirm({
                        message: t("confirmClearChats"),
                        title: t("clearAllChats"),
                        confirmLabel: t("clearAllChats"),
                        danger: true,
                      }))
                    ) {
                      return;
                    }
                    clearAllConversations()
                      .then(() => {
                        onDataCleared?.();
                        refreshStats();
                      })
                      .catch(console.error);
                  }}
                >
                  {t("clearAllChats")}
                </button>
              </SetRow>
              <SetRow label={t("clearKb")} hint={t("clearKbHint")}>
                <button
                  type="button"
                  className="data-btn danger"
                  onClick={async () => {
                    if (
                      !(await confirm({
                        message: t("clearKbConfirm"),
                        title: t("clearKb"),
                        confirmLabel: t("clearKb"),
                        danger: true,
                      }))
                    ) {
                      return;
                    }
                    ragClearAll().then(refreshStats).catch(console.error);
                  }}
                >
                  {t("clearKb")}
                </button>
              </SetRow>
            </>
          )}

          {cat === "about" && (
            <div className="about-card">
              <img className="about-logo" src={logoUrl} alt="Chaty" draggable={false} />
              <div className="about-name">Chaty</div>
              <div className="about-version">v{__APP_VERSION__}</div>
              <div className="about-tagline">{t("aboutTagline")}</div>
              <div className="about-actions">
                <button
                  type="button"
                  className="data-btn"
                  disabled={checking}
                  onClick={() => {
                    setChecking(true);
                    setUpd(null);
                    checkUpdate()
                      .then(setUpd)
                      .catch(() => setUpd({ available: false, current: __APP_VERSION__, latest: __APP_VERSION__ }))
                      .finally(() => setChecking(false));
                  }}
                >
                  {checking ? "…" : t("aboutCheckUpdate")}
                </button>
                {upd?.available && upd.url && (
                  <button
                    type="button"
                    className="data-btn accent"
                    onClick={() => void runUpdate(upd.url!).catch(console.error)}
                  >
                    {t("aboutUpdateNow")}
                  </button>
                )}
              </div>
              {upd && (
                <div className="about-status">
                  {upd.available
                    ? `${t("aboutNewVersion")}: v${upd.latest}`
                    : t("aboutUpToDate")}
                </div>
              )}
              <div className="about-links">
                <button type="button" onClick={() => void openExternal("https://chaty.ca").catch(console.error)}>
                  chaty.ca
                </button>
                <span aria-hidden="true">·</span>
                <button type="button" onClick={() => void openExternal("https://github.com/Fangyuan025/Chaty").catch(console.error)}>
                  GitHub
                </button>
              </div>
            </div>
          )}
          </div>
        </div>
      </div>
    </div>
    </>,
    document.body,
  );
}
