import { useEffect, useState } from "react";
import { useI18n } from "../lib/i18n";
import { getHardwareInfo, type HardwareInfo, type ModelInfo } from "../lib/ipc";

/** Top-right popover showing the machine's CPU / RAM / GPU and the current
 *  model's GPU‑acceleration status. */
export function HardwarePanel({
  model,
  onClose,
}: {
  model: ModelInfo | null;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [hw, setHw] = useState<HardwareInfo | null>(null);

  useEffect(() => {
    getHardwareInfo()
      .then(setHw)
      .catch(() => {});
  }, []);

  const accel = () => {
    if (!model) return <span className="hw-accel off">{t("hwNoModel")}</span>;
    if (model.gpuLayers > 0) {
      return (
        <span className="hw-accel on">
          {t("hwLayersOn", { a: model.gpuLayers, b: model.nLayer ?? "?" })}
          {model.gpuName ? <small> · {model.gpuName}</small> : null}
        </span>
      );
    }
    return <span className="hw-accel off">{t("hwCpuOnly")}</span>;
  };

  return (
    <>
      <div className="popover-backdrop" onClick={onClose} />
      <div className="settings-panel hw-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-title">{t("hwTitle")}</div>
        {hw ? (
          <div className="hw-list">
            <div className="hw-row">
              <span className="hw-k">{t("hwCpu")}</span>
              <span className="hw-v">
                {hw.cpu}
                <small> · {t("hwThreads", { n: hw.cpuThreads })}</small>
              </span>
            </div>
            <div className="hw-row">
              <span className="hw-k">{t("hwRam")}</span>
              <span className="hw-v">{(hw.ramMb / 1024).toFixed(1)} GB</span>
            </div>
            <div className="hw-row">
              <span className="hw-k">{t("hwGpu")}</span>
              <span className="hw-v">
                {hw.gpu ? `${hw.gpu.name} · ${(hw.gpu.vramMb / 1024).toFixed(1)} GB` : t("hwNoGpu")}
              </span>
            </div>
            <div className="hw-row">
              <span className="hw-k">{t("hwBackend")}</span>
              <span className="hw-v">{hw.gpuBackend}</span>
            </div>
            <div className="hw-divider" />
            <div className="hw-row">
              <span className="hw-k">{t("hwAccel")}</span>
              {accel()}
            </div>
          </div>
        ) : (
          <div className="hw-loading">…</div>
        )}
      </div>
    </>
  );
}
