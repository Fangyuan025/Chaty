import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
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
  openExternal,
  synthesize,
  type UpdateInfo,
} from "../lib/ipc";
import { decodeAudio, playAudio } from "../lib/audio";
import { CODE_THEMES, type CodeTheme } from "../lib/codeTheme";
import { useConfirm } from "./ConfirmModal";
import { Select } from "./Select";
import { BUILTIN_SKILLS } from "../lib/skills";
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
  /** GPU offload: -1 = auto‑tune by VRAM, 0 = CPU only, >0 = that many layers. */
  gpuLayers: number;
  /** Context window to load the model with: 0 = memory-friendly default (≤8192),
   *  >0 = that many tokens (clamped to the model's trained length). */
  contextLength: number;
  /** Code mode: max agent steps per turn before it pauses. */
  codeMaxSteps: number;
  /** Code mode: default bash-command timeout in seconds. */
  codeBashTimeout: number;
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
  voiceSpeed: 1.0,
  gpuLayers: -1,
  contextLength: 0,
  codeMaxSteps: 32,
  codeBashTimeout: 60,
  codeSkills: [],
  codeDisabledSkills: [],
  codeAllowedCommands: [],
  darkScheme: "warm",
  lightScheme: "paper",
  codeTheme: "github-dark",
  uiScale: 1,
  sendKey: "enter",
  reduceMotion: false,
  answerSize: "md",
  autoTitle: true,
  autoLoadLast: true,
};

/** kokoro-en-v0_19 speakers, in sid order (the array index IS the speaker id,
 *  so the order must not change). This pack ships exactly these 11 voices;
 *  the larger Kokoro-82M set (af_heart, am_fenrir, …) needs a different model.
 *  Labels carry the Kokoro VOICES.md overall grade so users can pick good ones;
 *  ★ marks the best in each gender. */
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

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

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
          modelBytes: models.reduce((a, m) => a + (m.sizeMb ?? 0) * 1e6, 0),
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
  const [voiceTesting, setVoiceTesting] = useState(false);
  const testVoice = () => {
    if (voiceTesting) return;
    setVoiceTesting(true);
    synthesize("Hi! This is how I sound. Nice to meet you.", value.voiceSpeed, value.voiceSid)
      .then((a) => playAudio(decodeAudio(a.audio), a.sampleRate).done)
      .catch(console.error)
      .finally(() => setVoiceTesting(false));
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
    // TTS is English-only, so the voice section only exists in English UI.
    ...(lang === "en" ? [{ id: "voice" as CatId, label: t("setCatVoice") }] : []),
    { id: "data", label: t("setCatData") },
    { id: "about", label: t("setCatAbout") },
  ];

  if (!mounted) return null;

  return createPortal(
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
                  <button type="button" className={lang === "zh" ? "active" : ""} onClick={() => setLang("zh")}>中文</button>
                  <button type="button" className={lang === "en" ? "active" : ""} onClick={() => setLang("en")}>English</button>
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
              <label className="field">
                <span><em className="has-tip" data-tip={t("tipMaxTokens")}>{t("maxTokens")}</em></span>
                <div className="lang-switch">
                  <button type="button" className={!value.limitTokens ? "active" : ""} onClick={() => set("limitTokens", false)}>{t("noLimit")}</button>
                  <button type="button" className={value.limitTokens ? "active" : ""} onClick={() => set("limitTokens", true)}>{t("gpuCustom")}</button>
                </div>
              </label>
              {value.limitTokens && (
                <label className="field">
                  <span>
                    {t("maxTokens")} <b>{value.maxTokens}</b>
                  </span>
                  <input
                    type="range"
                    min={128}
                    max={maxTokensLimit}
                    step={128}
                    value={Math.min(value.maxTokens, maxTokensLimit)}
                    onChange={(e) => set("maxTokens", Number(e.target.value))}
                  />
                </label>
              )}
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
                  <input type="range" min={1} max={80} step={1} value={value.gpuLayers} onChange={(e) => set("gpuLayers", Number(e.target.value))} />
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
            </>
          )}

          {cat === "code" && (
            <>
              <label className="field">
                <span>
                  {t("cmMaxSteps")} <b>{value.codeMaxSteps}</b>
                </span>
                <input
                  type="range"
                  min={8}
                  max={96}
                  step={4}
                  value={value.codeMaxSteps}
                  onChange={(e) => set("codeMaxSteps", Number(e.target.value))}
                />
              </label>
              <div className="settings-hint">{t("cmMaxStepsHint")}</div>

              <label className="field">
                <span>
                  {t("cmBashTimeout")} <b>{value.codeBashTimeout}s</b>
                </span>
                <input
                  type="range"
                  min={10}
                  max={300}
                  step={10}
                  value={value.codeBashTimeout}
                  onChange={(e) => set("codeBashTimeout", Number(e.target.value))}
                />
              </label>
              <div className="settings-hint">{t("cmBashTimeoutHint")}</div>

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
            </>
          )}

          {cat === "voice" && lang === "en" && (
            <>
              <label className="field">
                <span>{t("voice")}</span>
                <Select
                  className="field-select"
                  value={value.voiceSid}
                  ariaLabel={t("voice")}
                  onChange={(v) => set("voiceSid", v)}
                  options={VOICES.map((name, i) => ({ value: i, label: name }))}
                />
              </label>
              <label className="field">
                <span>
                  {t("voiceSpeed")} <b>{value.voiceSpeed.toFixed(2)}×</b>
                </span>
                <input type="range" min={0.5} max={2} step={0.05} value={value.voiceSpeed} onChange={(e) => set("voiceSpeed", Number(e.target.value))} />
              </label>
              <SetRow label={t("voicePreview")} hint={t("voicePreviewHint")}>
                <button type="button" className="data-btn" disabled={voiceTesting} onClick={testVoice}>
                  {voiceTesting ? "…" : t("voicePreviewBtn")}
                </button>
              </SetRow>
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
    </div>,
    document.body,
  );
}
