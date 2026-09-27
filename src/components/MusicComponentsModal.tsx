import { useCallback, useEffect, useRef, useState } from "react";

import { etaSeconds as dlEta, fmtTime, type EtaSample } from "../lib/eta";
import { fmtBytes } from "../lib/fmt";
import { useI18n, type TKey } from "../lib/i18n";
import { DOWNLOAD_CANCELLED, cancelDownload, musicModelProbe, type MusicComponent, type MusicProbe } from "../lib/ipc";
import { downloadMusicParts } from "../lib/musicDownload";
import { Icon } from "./Icon";

const ROLE_KEY: Record<MusicComponent["role"], TKey> = {
  vae: "musRoleVae",
  config: "musRoleConfig",
  tokenizer: "musRoleTokenizer",
  weights: "musRoleWeights",
};

function basename(p: string): string {
  return p.split(/[/\\]/).pop() || p;
}

/** The parts of a music model: what it has, and a one-click download of the
 *  rest into its folder. Shown before loading a model that lacks something,
 *  and from Settings → Music model. */
export function MusicComponentsModal({
  path,
  onLoad,
  onClose,
}: {
  /** The music model (the file picked). */
  path: string;
  /** Load the model — offered once nothing is missing. */
  onLoad?: () => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [probe, setProbe] = useState<MusicProbe | null>(null);
  const [error, setError] = useState("");
  const [active, setActive] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [eta, setEta] = useState<number | null>(null);
  const etaStore = useRef<EtaSample[]>([]);
  const cancelKey = useRef<string | null>(null);

  const refresh = useCallback(() => {
    musicModelProbe(path)
      .then((p) => setProbe(p))
      .catch((e) => setError(String(e)));
  }, [path]);
  useEffect(refresh, [refresh]);

  const download = async () => {
    if (!probe || active) return;
    const todo = probe.suggestions;
    const total = todo.reduce((a, s) => a + s.size, 0);
    setActive(true);
    setError("");
    etaStore.current = [];
    setProgress({ done: 0, total });
    try {
      await downloadMusicParts(
        probe.path,
        todo,
        (done) => {
          setProgress({ done, total });
          setEta(dlEta(etaStore.current, done, total));
        },
        (name) => (cancelKey.current = name),
        (msg) => msg !== DOWNLOAD_CANCELLED && setError(msg),
      );
    } catch (e) {
      const msg = typeof e === "string" ? e : String((e as Error)?.message ?? e);
      if (msg !== DOWNLOAD_CANCELLED) setError(msg || t("dlFailed"));
    } finally {
      refresh();
      setActive(false);
      cancelKey.current = null;
    }
  };

  const need = probe?.suggestions.reduce((a, s) => a + s.size, 0) ?? 0;
  const missing = probe?.missing ?? [];
  const pct = progress.total > 0 ? Math.min(100, (progress.done / progress.total) * 100) : 0;

  return (
    <>
      <div className="popover-backdrop" onClick={active ? undefined : onClose} style={{ zIndex: 90 }} />
      <div className="dl-modal is-comp-modal">
        <div className="dl-head">
          <span className="dl-title">{t("musCompTitle")}</span>
          {!active && (
            <button className="dl-close" onClick={onClose} title={t("cancel")}>
              <Icon name="x" size={12} strokeWidth={2.2} />
            </button>
          )}
        </div>
        {!probe ? (
          <div className="is-comp-loading">{error || <span className="cm-spin" />}</div>
        ) : (
          <>
            <div className="is-comp-model">
              <b>{basename(probe.path)}</b>
              <span>
                {probe.familyName}
                {probe.paramsB ? ` · ${probe.paramsB}B` : ""}
                {probe.quant ? ` · ${probe.quant}` : ""}
              </span>
            </div>
            {!probe.supported ? (
              <div className="dl-error">{t("musUnsupportedFamily", { family: probe.familyName })}</div>
            ) : (
              <div className="is-comp-hint">{missing.length ? t("musCompNeedHint") : t("musCompReady")}</div>
            )}
            <div className="is-comp-list">
              {probe.components.length === 0 && missing.length === 0 && <div className="is-comp-row">{t("musCompSingle")}</div>}
              {probe.components.map((c) => (
                <div key={c.file} className="is-comp-row ok">
                  <span className="is-comp-state">
                    <Icon name="check" size={12} strokeWidth={2.4} />
                  </span>
                  <span className="is-comp-main">
                    <span className="is-comp-role">{t(ROLE_KEY[c.role])}</span>
                    <span className="is-comp-file" title={c.file}>
                      {c.file} · {fmtBytes(c.size)}
                    </span>
                  </span>
                </div>
              ))}
              {probe.suggestions.map((s) => {
                const required = missing.includes(s.dest);
                return (
                  <div key={s.dest} className={`is-comp-row ${required ? "missing" : ""}`}>
                    <span className="is-comp-state">{required ? "!" : "—"}</span>
                    <span className="is-comp-main">
                      <span className="is-comp-role">
                        {t(ROLE_KEY[s.role])}
                        {!required && <em>{t("imgOptional")}</em>}
                      </span>
                      <span className="is-comp-file" title={`${s.repo}/${s.file}`}>
                        {t("imgWillDownload")} {s.dest} · {fmtBytes(s.size)}
                      </span>
                    </span>
                  </div>
                );
              })}
            </div>
            {active && (
              <div className="dl-progress store-progress">
                <div className="dl-bar">
                  <div className="dl-bar-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="dl-pct">
                  {fmtBytes(progress.done)} / {fmtBytes(progress.total)}
                  {eta !== null ? ` · ${t("etaLeft")} ~${fmtTime(eta)}` : ""}
                </span>
                <button
                  className="dl-cancel"
                  title={t("cancel")}
                  onClick={() => {
                    if (cancelKey.current) void cancelDownload(cancelKey.current).catch(() => {});
                  }}
                >
                  <Icon name="x" size={12} strokeWidth={2.2} />
                </button>
              </div>
            )}
            {error && <div className="dl-error">{error}</div>}
            <div className="is-comp-foot">
              {probe.suggestions.length > 0 && (
                <button className="dl-get" disabled={active} onClick={() => void download()}>
                  {t("imgDownloadMissing", { size: fmtBytes(need) })}
                </button>
              )}
              {onLoad && (
                <button className="dl-get is-comp-load" disabled={active || missing.length > 0 || !probe.supported} onClick={onLoad}>
                  {t("imgLoadModel")}
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
}
