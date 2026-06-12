import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { useI18n } from "../lib/i18n";
import {
  ragAddDocument,
  ragDownloadModel,
  ragListDocuments,
  ragRemoveDocument,
  ragStatus,
  type RagDoc,
  type RagStatus,
} from "../lib/ipc";

/** Local knowledge-base manager: embedding model, documents, indexing. */
export function KnowledgePanel({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [docs, setDocs] = useState<RagDoc[]>([]);
  const [dl, setDl] = useState<{ pct: number } | null>(null);
  const [indexing, setIndexing] = useState<{ name: string; pct: number } | null>(null);
  const [error, setError] = useState("");

  const refresh = () => {
    ragStatus().then(setStatus).catch(() => {});
    ragListDocuments().then(setDocs).catch(() => {});
  };
  useEffect(refresh, []);

  async function downloadModel() {
    setError("");
    setDl({ pct: 0 });
    try {
      await ragDownloadModel((p) => {
        if (p.type === "progress" && p.total > 0) {
          setDl({ pct: Math.round((p.downloaded / p.total) * 100) });
        }
      });
      setDl(null);
      refresh();
    } catch (e) {
      setDl(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function addDocuments() {
    setError("");
    const picked = await open({
      multiple: true,
      filters: [
        {
          name: "Documents",
          extensions: ["pdf", "txt", "md", "markdown", "html", "csv", "json", "log"],
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
            <div className="setup-title">📚 {t("kbTitle")}</div>
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
              <div className="setup-progress">
                <div className="setup-progress-fill" style={{ width: `${dl.pct}%` }} />
                <span>{dl.pct}%</span>
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
                  <div key={d.id} className="kb-doc">
                    <span className="kb-doc-name">📄 {d.name}</span>
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
          </>
        )}

        {error && <div className="setup-err">{error.slice(0, 240)}</div>}
        <div className="setup-foot">{t("kbFoot")}</div>
      </div>
    </div>,
    document.body,
  );
}
