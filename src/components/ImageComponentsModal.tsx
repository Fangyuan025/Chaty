import { useCallback, useEffect, useRef, useState } from "react";
import { open as openDialog } from "@tauri-apps/plugin-dialog";

import { etaSeconds as dlEta, fmtTime, type EtaSample } from "../lib/eta";
import { fmtBytes, fmtGbFromMb } from "../lib/fmt";
import { useI18n, type TKey } from "../lib/i18n";
import {
  DOWNLOAD_CANCELLED,
  cancelDownload,
  downloadModel,
  hfResolveUrl,
  imageModelProbe,
  type ImageProbe,
  type ImageRole,
} from "../lib/ipc";
import { Icon } from "./Icon";

export const ROLE_KEY: Record<ImageRole, TKey> = {
  vae: "imgRoleVae",
  llm: "imgRoleLlm",
  llmVision: "imgRoleLlmVision",
  clipL: "imgRoleClipL",
  clipG: "imgRoleClipG",
  t5xxl: "imgRoleT5",
};

function basename(p: string): string {
  return p.split(/[/\\]/).pop() || p;
}
function dirname(p: string): string {
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return i > 0 ? p.slice(0, i) : p;
}

/** The companion files of an image model: what it needs, what was found, and
 *  the way to get the rest — a one-click download into the model's folder, or
 *  a file picked by hand. Shown before loading a model that lacks something,
 *  and from Settings → Image model. */
export function ImageComponentsModal({
  path,
  overrides,
  onOverrides,
  onLoad,
  onClose,
}: {
  /** The image model (its denoiser file). */
  path: string;
  /** Files picked by hand for this model, per role ("" = none). */
  overrides: Record<string, string>;
  onOverrides: (next: Record<string, string>) => void;
  /** Load the model — offered once nothing required is missing. */
  onLoad?: () => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [probe, setProbe] = useState<ImageProbe | null>(null);
  const [error, setError] = useState("");
  const [active, setActive] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [eta, setEta] = useState<number | null>(null);
  const etaStore = useRef<EtaSample[]>([]);
  const cancelKey = useRef<string | null>(null);

  const refresh = useCallback(() => {
    imageModelProbe(path, overrides)
      .then((p) => setProbe(p))
      .catch((e) => setError(String(e)));
  }, [path, overrides]);
  useEffect(refresh, [refresh]);

  const pick = async (role: ImageRole) => {
    const picked = await openDialog({
      multiple: false,
      defaultPath: dirname(path),
      filters: [{ name: t("imgWeightsFiles"), extensions: ["gguf", "safetensors", "sft"] }],
    });
    if (typeof picked === "string") onOverrides({ ...overrides, [role]: picked });
  };
  const unset = (role: ImageRole) => {
    const next = { ...overrides };
    delete next[role];
    onOverrides(next);
  };

  const download = async () => {
    if (!probe || active) return;
    const todo = probe.suggestions;
    const total = todo.reduce((a, s) => a + s.size, 0);
    setActive(true);
    setError("");
    etaStore.current = [];
    setProgress({ done: 0, total });
    let base = 0;
    try {
      for (const s of todo) {
        const name = basename(s.file);
        cancelKey.current = name;
        let got = 0;
        await downloadModel(
          hfResolveUrl(s.repo, s.file),
          name,
          (p) => {
            if (p.type === "progress") {
              got = p.downloaded;
              setProgress({ done: base + got, total });
              setEta(dlEta(etaStore.current, base + got, total));
            } else if (p.type === "error" && p.message !== DOWNLOAD_CANCELLED) {
              setError(p.message);
            }
          },
          undefined,
          dirname(probe.path),
        );
        base += Math.max(got, s.size);
      }
      refresh();
    } catch (e) {
      const msg = typeof e === "string" ? e : String((e as Error)?.message ?? e);
      if (msg !== DOWNLOAD_CANCELLED) setError(msg || t("dlFailed"));
      refresh();
    } finally {
      setActive(false);
      cancelKey.current = null;
    }
  };

  const roles: ImageRole[] = probe ? [...probe.requires, ...probe.optional.filter((r) => !probe.requires.includes(r))] : [];
  const need = probe?.suggestions.reduce((a, s) => a + s.size, 0) ?? 0;
  const missing = probe?.missing ?? [];
  const pct = progress.total > 0 ? Math.min(100, (progress.done / progress.total) * 100) : 0;

  return (
    <>
      <div className="popover-backdrop" onClick={active ? undefined : onClose} style={{ zIndex: 90 }} />
      <div className="dl-modal is-comp-modal">
        <div className="dl-head">
          <span className="dl-title">{t("imgCompTitle")}</span>
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
            <div className="is-comp-hint">{missing.length ? t("imgCompNeedHint") : t("imgCompReady")}</div>
            <div className="is-comp-list">
              {roles.length === 0 && <div className="is-comp-row">{t("imgCompNone")}</div>}
              {roles.map((role) => {
                const c = probe.components.find((x) => x.role === role);
                const required = probe.requires.includes(role);
                const sug = probe.suggestions.find((s) => s.role === role);
                return (
                  <div key={role} className={`is-comp-row ${c ? "ok" : required ? "missing" : ""}`}>
                    <span className="is-comp-state">
                      {c ? <Icon name="check" size={12} strokeWidth={2.4} /> : required ? "!" : "—"}
                    </span>
                    <span className="is-comp-main">
                      <span className="is-comp-role">
                        {t(ROLE_KEY[role])}
                        {!required && <em>{t("imgOptional")}</em>}
                      </span>
                      <span className="is-comp-file" title={c?.path ?? sug?.file}>
                        {c
                          ? `${basename(c.path)} · ${fmtGbFromMb(c.sizeMb)} · ${t(c.source === "override" ? "imgSrcOverride" : c.source === "shared" ? "imgSrcShared" : "imgSrcFolder")}`
                          : sug
                            ? `${t("imgWillDownload")} ${sug.repo}/${basename(sug.file)} · ${fmtBytes(sug.size)}`
                            : required
                              ? t("imgPickManually")
                              : t("imgNotUsed")}
                      </span>
                    </span>
                    <span className="is-comp-acts">
                      <button className="data-btn" disabled={active} onClick={() => void pick(role)}>
                        {t("imgPickFile")}
                      </button>
                      {overrides[role] !== undefined && (
                        <button className="data-btn" disabled={active} onClick={() => unset(role)} title={t("imgAutoDetect")}>
                          {t("imgAutoDetect")}
                        </button>
                      )}
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
                <button className="dl-get is-comp-load" disabled={active || missing.length > 0} onClick={onLoad}>
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
