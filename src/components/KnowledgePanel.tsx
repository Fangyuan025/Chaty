import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { useI18n } from "../lib/i18n";
import { etaSeconds, fmtTime, type EtaSample } from "../lib/eta";
import { IconKb, IconDoc, IconMic } from "./icons";
import {
  DOWNLOAD_CANCELLED,
  ragAddDocument,
  ragCancelDownload,
  ragDownloadModel,
  ragListDocuments,
  ragRemoveDocument,
  ragSetDocEnabled,
  ragStatus,
  type RagDoc,
  type RagStatus,
} from "../lib/ipc";

/** Local knowledge-base manager: embedding model, documents, indexing. */
export function KnowledgePanel({
  onClose,
  onPodcast,
}: {
  onClose: () => void;
  onPodcast?: () => void;
}) {
  const { t } = useI18n();
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [docs, setDocs] = useState<RagDoc[]>([]);
  const [dl, setDl] = useState<{ pct: number; eta: number | null } | null>(null);
  const dlEtaStore = useRef<EtaSample[]>([]);
  const [indexing, setIndexing] = useState<{ name: string; pct: number } | null>(null);
  const [error, setError] = useState("");

  const refresh = () => {
    ragStatus().then(setStatus).catch(() => {});
    ragListDocuments().then(setDocs).catch(() => {});
  };
  useEffect(refresh, []);

  async function downloadModel() {
    setError("");
    setDl({ pct: 0, eta: null });
    dlEtaStore.current = [];
    try {
      await ragDownloadModel((p) => {
        if (p.type === "progress" && p.total > 0) {
          setDl({
            pct: Math.round((p.downloaded / p.total) * 100),
            eta: etaSeconds(dlEtaStore.current, p.downloaded, p.total),
          });
        } else if (p.type === "error" && p.message !== DOWNLOAD_CANCELLED) {
          setError(p.message);
        }
      });
      setDl(null);
      refresh();
    } catch (e) {
      setDl(null);
      const msg = e instanceof Error ? e.message : String(e);
      if (msg !== DOWNLOAD_CANCELLED) setError(msg);
    }
  }

  async function addDocuments() {
    setError("");
    const picked = await open({
      multiple: true,
      filters: [
        {
          name: "Documents / Images",
          extensions: [
            "pdf", "txt", "md", "markdown", "html", "csv", "json", "log",
            "png", "jpg", "jpeg", "webp", "bmp", "gif",
          ],
        },
      ],
    });
    const paths = Array.isArray(picked) ? picked : picked ? [picked] : [];
    for (const path of paths) {
      const name = path.split(/[/\\]/).pop() ?? path;
      setIndexing({ name, pct: 0 });
      try {
        await ragAddDocument(path, (p) => {
          setIndexing({ name, pct: Math.round(p.frac * 100) });
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }
    setIndexing(null);
    refresh();
  }

  const ready = status?.modelReady ?? false;

  return createPortal(
    <div className="preview-overlay" onMouseDown={onClose}>
      <div className="setup-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="setup-head">
          <div>
            <div className="setup-title"><IconKb size={18} /> {t("kbTitle")}</div>
            <div className="setup-hw">
              {status
                ? `${status.docs} ${t("kbDocs")} · ${status.chunks} ${t("kbChunks")}`
                : "…"}
            </div>
          </div>
          <button className="preview-close" onClick={onClose}>
            ×
          </button>
        </div>

        {!ready ? (
          <div className="kb-setup">
            <p className="kb-note">{t("kbModelNote")}</p>
            {dl ? (
              <div className="setup-progress-row">
                <div className="setup-progress">
                  <div className="setup-progress-fill" style={{ width: `${dl.pct}%` }} />
                  <span>
                    {dl.pct}%{dl.eta !== null ? ` · ${t("etaLeft")} ~${fmtTime(dl.eta)}` : ""}
                  </span>
                </div>
                <button
                  className="dl-cancel"
                  title={t("cancel")}
                  onClick={() => void ragCancelDownload().catch(() => {})}
                >
                  ×
                </button>
              </div>
            ) : (
              <button className="setup-dl ready" onClick={() => void downloadModel()}>
                {t("kbDownloadModel")}
              </button>
            )}
          </div>
        ) : (
          <>
            <div className="kb-list">
              {docs.length === 0 ? (
                <div className="model-menu-empty">{t("kbEmpty")}</div>
              ) : (
                docs.map((d) => (
                  <div key={d.id} className={`kb-doc ${d.enabled ? "" : "off"}`}>
                    <label className="kb-doc-scope" title={t("kbScopeTip")}>
                      <input
                        type="checkbox"
                        checked={d.enabled}
                        onChange={(e) => {
                          const enabled = e.target.checked;
                          setDocs((cur) =>
                            cur.map((x) => (x.id === d.id ? { ...x, enabled } : x)),
                          );
                          void ragSetDocEnabled(d.id, enabled).catch(() => refresh());
                        }}
                      />
                      <span className="kb-doc-name">
                        <IconDoc size={13} style={{ marginRight: 5 }} />
                        {d.name}
                      </span>
                    </label>
                    <span className="kb-doc-meta">
                      {d.chunks} {t("kbChunks")}
                    </span>
                    <button
                      className="kb-doc-del"
                      title={t("kbRemove")}
                      onClick={() => {
                        void ragRemoveDocument(d.id).then(refresh);
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))
              )}
            </div>
            {docs.length > 0 && <div className="kb-scope-hint">{t("kbScopeHint")}</div>}
            <div className="kb-foot-actions">
              {indexing ? (
                <div className="setup-progress">
                  <div
                    className="setup-progress-fill"
                    style={{ width: `${indexing.pct}%` }}
                  />
                  <span>
                    {t("kbIndexing")} {indexing.name} · {indexing.pct}%
                  </span>
                </div>
              ) : (
                <button className="setup-dl" onClick={() => void addDocuments()}>
                  + {t("kbAdd")}
                </button>
              )}
              {onPodcast && docs.length > 0 && !indexing && (
                <button className="setup-dl kb-podcast" onClick={onPodcast}>
                  <IconMic size={15} style={{ marginRight: 6 }} /> {t("kbPodcast")}
                </button>
              )}
            </div>
          </>
        )}

        {error && <div className="setup-err">{error.slice(0, 240)}</div>}
      </div>
    </div>,
    document.body,
  );
}
