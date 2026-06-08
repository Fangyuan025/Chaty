import { useI18n } from "../lib/i18n";

export interface GenSettings {
  systemPrompt: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  /** GPU offload: -1 = auto‑tune by VRAM, 0 = CPU only, >0 = that many layers. */
  gpuLayers: number;
  /** Context window to load the model with: 0 = memory-friendly default (≤8192),
   *  >0 = that many tokens (clamped to the model's trained length). */
  contextLength: number;
}

export const defaultSettings: GenSettings = {
  systemPrompt: "",
  temperature: 0.7,
  topP: 0.95,
  maxTokens: 1024,
  gpuLayers: -1,
  contextLength: 0,
};

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
  const set = <K extends keyof GenSettings>(key: K, v: GenSettings[K]) =>
    onChange({ ...value, [key]: v });

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
          <span>{t("systemPrompt")}</span>
          <textarea
            rows={3}
            placeholder={t("systemPromptPh")}
            value={value.systemPrompt}
            onChange={(e) => set("systemPrompt", e.target.value)}
          />
        </label>

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
