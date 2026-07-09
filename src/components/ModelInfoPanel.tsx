import { useI18n } from "../lib/i18n";
import type { ModelInfo } from "../lib/ipc";

/** Top-right popover with the probed GGUF metadata for the loaded model. */
export function ModelInfoPanel({
  model,
  onClose,
}: {
  model: ModelInfo | null;
  onClose: () => void;
}) {
  const { t } = useI18n();

  const cap = (on: boolean) =>
    on ? <span className="mi-cap on">✓</span> : <span className="mi-cap off">—</span>;

  return (
    <>
      <div className="popover-backdrop" onClick={onClose} />
      <div className="titlebar-pop hw-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-title">{t("miTitle")}</div>
        {!model ? (
          <div className="hw-loading">{t("miNoModel")}</div>
        ) : (
          <div className="hw-list">
            <div className="hw-row">
              <span className="hw-k">{t("miName")}</span>
              <span className="hw-v">{model.modelName || model.name}</span>
            </div>
            {model.arch && (
              <div className="hw-row">
                <span className="hw-k">{t("miArch")}</span>
                <span className="hw-v">{model.arch}</span>
              </div>
            )}
            {model.paramsB != null && (
              <div className="hw-row">
                <span className="hw-k">{t("miParams")}</span>
                <span className="hw-v">{model.paramsB.toFixed(2)} B</span>
              </div>
            )}
            {model.quant && (
              <div className="hw-row">
                <span className="hw-k">{t("miQuant")}</span>
                <span className="hw-v">{model.quant}</span>
              </div>
            )}
            {model.sizeMb != null && (
              <div className="hw-row">
                <span className="hw-k">{t("miSize")}</span>
                <span className="hw-v">{(model.sizeMb / 1024).toFixed(2)} GB</span>
              </div>
            )}
            <div className="hw-row">
              <span className="hw-k">{t("miContext")}</span>
              <span className="hw-v">
                {model.nCtx?.toLocaleString() ?? "?"}
                {model.nCtxTrain ? (
                  <small> / {model.nCtxTrain.toLocaleString()} {t("miTrained")}</small>
                ) : null}
              </span>
            </div>
            {model.nLayer != null && (
              <div className="hw-row">
                <span className="hw-k">{t("miLayers")}</span>
                <span className="hw-v">{model.nLayer}</span>
              </div>
            )}
            {model.nEmbd != null && (
              <div className="hw-row">
                <span className="hw-k">{t("miEmbed")}</span>
                <span className="hw-v">{model.nEmbd}</span>
              </div>
            )}
            <div className="hw-divider" />
            <div className="hw-row">
              <span className="hw-k">{t("miTemplate")}</span>
              {cap(model.hasChatTemplate)}
            </div>
            <div className="hw-row">
              <span className="hw-k">{t("miThinking")}</span>
              {cap(model.supportsThinking)}
            </div>
            <div className="hw-row">
              <span className="hw-k">{t("miTools")}</span>
              {cap(model.supportsTools)}
            </div>
            <div className="hw-row">
              <span className="hw-k">{t("miMultimodal")}</span>
              {cap(model.multimodal)}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
