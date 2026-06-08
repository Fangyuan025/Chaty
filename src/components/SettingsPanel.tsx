import { useState } from "react";
import { useI18n } from "../lib/i18n";

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
}

export const defaultSettings: GenSettings = {
  theme: "dark",
  systemPrompt: "",
  temperature: 0.7,
  topP: 0.95,
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
};

/** Kokoro-en-v0_19 speakers (sid order). */
export const VOICES = [
  "af · warm female",
  "af_bella · female",
  "af_nicole · female",
  "af_sarah · female",
  "af_sky · female",
  "am_adam · male",
  "am_michael · male",
  "bf_emma · UK female",
  "bf_isabella · UK female",
  "bm_george · UK male",
  "bm_lewis · UK male",
];

/** Parse the stop-sequence textarea into a clean array for the backend. */
export function parseStops(raw: string): string[] {
  return raw
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function SettingsPanel({
  value,
  onChange,
  onClose,
  maxTokensLimit = 4096,
  ctxTrainLimit,
}: {
  value: GenSettings;
  onChange: (next: GenSettings) => void;
  onClose: () => void;
  /** Upper bound for the max-length slider — adapts to the loaded model's context. */
  maxTokensLimit?: number;
  /** The loaded model's trained context length, used as the slider ceiling. */
  ctxTrainLimit?: number | null;
}) {
  const { t, lang, setLang } = useI18n();
  const [presetName, setPresetName] = useState("");
  const set = <K extends keyof GenSettings>(key: K, v: GenSettings[K]) =>
    onChange({ ...value, [key]: v });

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

  return (
    <>
      <div className="popover-backdrop" onClick={onClose} />
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-title">{t("settingsTitle")}</div>

        <label className="field">
          <span>{t("language")}</span>
          <div className="lang-switch">
            <button
              type="button"
              className={lang === "zh" ? "active" : ""}
              onClick={() => setLang("zh")}
            >
              中文
            </button>
            <button
              type="button"
              className={lang === "en" ? "active" : ""}
              onClick={() => setLang("en")}
            >
              English
            </button>
          </div>
        </label>

        <label className="field">
          <span>{t("theme")}</span>
          <div className="lang-switch">
            <button
              type="button"
              className={value.theme === "system" ? "active" : ""}
              onClick={() => set("theme", "system")}
            >
              {t("themeSystem")}
            </button>
            <button
              type="button"
              className={value.theme === "light" ? "active" : ""}
              onClick={() => set("theme", "light")}
            >
              {t("themeLight")}
            </button>
            <button
              type="button"
              className={value.theme === "dark" ? "active" : ""}
              onClick={() => set("theme", "dark")}
            >
              {t("themeDark")}
            </button>
          </div>
        </label>

        <label className="field">
          <span>{t("systemPrompt")}</span>
          <textarea
            rows={3}
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
                  <button
                    type="button"
                    className="preset-apply"
                    title={p.prompt}
                    onClick={() => set("systemPrompt", p.prompt)}
                  >
                    {p.name}
                  </button>
                  <button
                    type="button"
                    className="preset-del"
                    title={t("cancel")}
                    onClick={() => deletePreset(p.name)}
                  >
                    ×
                  </button>
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

        <label className="field">
          <span>
            {t("temperature")} <b>{value.temperature.toFixed(2)}</b>
          </span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={value.temperature}
            onChange={(e) => set("temperature", Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span>
            Top-P <b>{value.topP.toFixed(2)}</b>
          </span>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.01}
            value={value.topP}
            onChange={(e) => set("topP", Number(e.target.value))}
          />
        </label>

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

        <label className="field">
          <span>
            Top-K <b>{value.topK === 0 ? t("off") : value.topK}</b>
          </span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={value.topK}
            onChange={(e) => set("topK", Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span>
            Min-P <b>{value.minP.toFixed(2)}</b>
          </span>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={value.minP}
            onChange={(e) => set("minP", Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span>
            {t("repeatPenalty")} <b>{value.repeatPenalty.toFixed(2)}</b>
          </span>
          <input
            type="range"
            min={1}
            max={1.5}
            step={0.01}
            value={value.repeatPenalty}
            onChange={(e) => set("repeatPenalty", Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span>{t("stopSeqs")}</span>
          <textarea
            rows={2}
            placeholder={t("stopSeqsPh")}
            value={value.stop}
            onChange={(e) => set("stop", e.target.value)}
          />
        </label>

        {lang === "en" && (
          <>
            <label className="field">
              <span>{t("voice")}</span>
              <select
                className="field-select"
                value={value.voiceSid}
                onChange={(e) => set("voiceSid", Number(e.target.value))}
              >
                {VOICES.map((name, i) => (
                  <option key={i} value={i}>
                    {name}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>
                {t("voiceSpeed")} <b>{value.voiceSpeed.toFixed(2)}×</b>
              </span>
              <input
                type="range"
                min={0.5}
                max={2}
                step={0.05}
                value={value.voiceSpeed}
                onChange={(e) => set("voiceSpeed", Number(e.target.value))}
              />
            </label>
          </>
        )}

        <label className="field">
          <span>{t("gpuAccel")}</span>
          <div className="lang-switch">
            <button
              type="button"
              className={value.gpuLayers < 0 ? "active" : ""}
              onClick={() => set("gpuLayers", -1)}
            >
              {t("gpuAuto")}
            </button>
            <button
              type="button"
              className={value.gpuLayers === 0 ? "active" : ""}
              onClick={() => set("gpuLayers", 0)}
            >
              {t("gpuOff")}
            </button>
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
              max={80}
              step={1}
              value={value.gpuLayers}
              onChange={(e) => set("gpuLayers", Number(e.target.value))}
            />
          </label>
        )}
        <div className="settings-hint">{t("gpuHint")}</div>

        <label className="field">
          <span>{t("ctxLength")}</span>
          <div className="lang-switch">
            <button
              type="button"
              className={value.contextLength <= 0 ? "active" : ""}
              onClick={() => set("contextLength", 0)}
            >
              {t("ctxAuto")}
            </button>
            <button
              type="button"
              className={value.contextLength > 0 ? "active" : ""}
              onClick={() =>
                set("contextLength", value.contextLength > 0 ? value.contextLength : 8192)
              }
            >
              {t("gpuCustom")}
            </button>
          </div>
        </label>
        {value.contextLength > 0 && (
          <label className="field">
            <span>
              {t("ctxTokens")} <b>{value.contextLength}</b>
            </span>
            <input
              type="range"
              min={2048}
              max={Math.max(8192, ctxTrainLimit ?? 32768)}
              step={2048}
              value={Math.min(value.contextLength, Math.max(8192, ctxTrainLimit ?? 32768))}
              onChange={(e) => set("contextLength", Number(e.target.value))}
            />
          </label>
        )}
        <div className="settings-hint">{t("ctxHint")}</div>

        <button className="settings-reset" onClick={() => onChange(defaultSettings)}>
          {t("resetDefaults")}
        </button>
      </div>
    </>
  );
}
