import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { openDataDir, clearAllConversations, checkUpdate, runUpdate, type UpdateInfo } from "../lib/ipc";
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

export function SettingsPanel({
  value,
  onChange,
  onClose,
  maxTokensLimit = 4096,
  ctxTrainLimit,
  onReloadModel,
  reloading = false,
  onDataCleared,
}: {
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
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

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

  return createPortal(
    <div className="settings-overlay" onMouseDown={onClose}>
      <div className="settings-modal" onMouseDown={(e) => e.stopPropagation()}>
        <aside className="settings-nav">
          <div className="settings-nav-title">{t("settingsTitle")}</div>
          {cats.map((c) => (
            <button
              key={c.id}
              className={`settings-nav-item ${cat === c.id ? "active" : ""}`}
              onClick={() => setCat(c.id)}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <path d={CAT_ICONS[c.id]} />
              </svg>
              {c.label}
            </button>
          ))}
        </aside>

        <div className="settings-pane">
          <button className="settings-close" onClick={onClose} aria-label={t("cancel")}>×</button>

          {cat === "general" && (
            <>
              <div className="settings-sec">{t("setCatGeneral")}</div>
              <label className="field">
                <span>{t("language")}</span>
                <div className="lang-switch">
                  <button type="button" className={lang === "zh" ? "active" : ""} onClick={() => setLang("zh")}>中文</button>
                  <button type="button" className={lang === "en" ? "active" : ""} onClick={() => setLang("en")}>English</button>
                </div>
              </label>
              <label className="field">
                <span>{t("theme")}</span>
                <div className="lang-switch">
                  <button type="button" className={value.theme === "system" ? "active" : ""} onClick={() => set("theme", "system")}>{t("themeSystem")}</button>
                  <button type="button" className={value.theme === "light" ? "active" : ""} onClick={() => set("theme", "light")}>{t("themeLight")}</button>
                  <button type="button" className={value.theme === "dark" ? "active" : ""} onClick={() => set("theme", "dark")}>{t("themeDark")}</button>
                </div>
              </label>
            </>
          )}

          {cat === "chat" && (
            <>
              <div className="settings-sec">{t("setCatChat")}</div>
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
                        <button type="button" className="preset-del" title={t("cancel")} onClick={() => deletePreset(p.name)}>×</button>
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
            </>
          )}

          {cat === "sampling" && (
            <>
              <div className="settings-sec">{t("setCatSampling")}</div>
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
            </>
          )}

          {cat === "model" && (
            <>
              <div className="settings-sec">{t("setCatModel")}</div>
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
            </>
          )}

          {cat === "code" && (
            <>
              <div className="settings-sec">Code</div>
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
                        <button type="button" className="preset-del" title={t("cancel")} onClick={() => deleteSkill(s.name)}>×</button>
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
              <div className="settings-sec">{t("setCatVoice")}</div>
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
            </>
          )}

          {cat === "data" && (
            <>
              <div className="settings-sec">{t("setCatData")}</div>
              <div className="settings-data-row">
                <button type="button" className="data-btn" onClick={() => void openDataDir().catch(console.error)}>
                  {t("openDataDir")}
                </button>
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
                      .then(() => onDataCleared?.())
                      .catch(console.error);
                  }}
                >
                  {t("clearAllChats")}
                </button>
              </div>
              <div className="settings-hint">{t("dataHint")}</div>
              <button className="settings-reset" onClick={() => onChange(defaultSettings)}>
                {t("resetDefaults")}
              </button>
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
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
