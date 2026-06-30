import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { useI18n } from "../lib/i18n";
import { etaSeconds, fmtTime, type EtaSample } from "../lib/eta";
import { IconKb, IconDoc, IconMic, IconResearch } from "./icons";
import { useConfirm } from "./ConfirmModal";
import {
  DOWNLOAD_CANCELLED,
  ragAddDocument,
  ragCancelDownload,
  ragClearAll,
  ragDownloadModel,
  ragListDocuments,
  ragListSupportedFiles,
  ragRemoveDocument,
  ragSetDocEnabled,
  ragStatus,
  type RagDoc,
  type RagStatus,
} from "../lib/ipc";

// Mirrors SUPPORTED_EXTS in src-tauri/src/rag.rs — documents, images, and a
// broad set of text/code/markup/config files.
const KB_EXTS = [
  "pdf", "docx", "xlsx", "png", "jpg", "jpeg", "webp", "bmp", "gif",
  "txt", "md", "markdown", "mdx", "rst", "org", "tex", "log", "csv", "tsv", "json", "jsonl",
  "ndjson", "yaml", "yml", "toml", "ini", "cfg", "conf", "properties", "env",
  "html", "htm", "xml", "css", "scss", "sass", "less", "vue", "svelte", "astro",
  "js", "mjs", "cjs", "jsx", "ts", "tsx", "py", "pyi", "rs", "go", "java", "kt", "kts", "c", "h",
  "cpp", "cc", "cxx", "hpp", "hh", "cs", "rb", "php", "swift", "scala", "sh", "bash", "zsh",
  "fish", "ps1", "bat", "sql", "lua", "r", "jl", "pl", "pm", "dart", "ex", "exs", "erl", "hs",
  "clj", "cljs", "elm", "ml", "fs", "vb", "gradle", "groovy", "m", "mm",
];

/** Local knowledge-base manager: embedding model, documents, indexing. */
export function KnowledgePanel({
  onClose,
  onPodcast,
  onReport,
}: {
  onClose: () => void;
  onPodcast?: () => void;
  onReport?: () => void;
}) {
  const { t } = useI18n();
  const confirm = useConfirm();
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

  /** Ingest a list of file paths one by one, streaming per-file progress.
   *  When more than one file is queued the label shows its position (3/20).
   *  `root` (folder import) preserves each file's path relative to the folder. */
  async function ingestPaths(paths: string[], root?: string) {
    const total = paths.length;
    for (let i = 0; i < total; i++) {
      const path = paths[i];
      const base = path.split(/[/\\]/).pop() ?? path;
      const name = total > 1 ? `${base} · ${i + 1}/${total}` : base;
      setIndexing({ name, pct: 0 });
      try {
        await ragAddDocument(
          path,
          (p) => {
            setIndexing({ name, pct: Math.round(p.frac * 100) });
          },
          root,
        );
      } catch (e) {
        // Folder import (root set): a file with no extractable text — or that's
        // otherwise unreadable — is silently skipped, like a single-file add
        // would just be ignored. Manual file picks still surface the error.
        if (root) console.warn("KB: skipped", path, e);
        else setError(e instanceof Error ? e.message : String(e));
      }
    }
    setIndexing(null);
    refresh();
  }

  async function addDocuments() {
    setError("");
    const picked = await open({
      multiple: true,
      filters: [{ name: "Documents / Images", extensions: KB_EXTS }],
    });
    const paths = Array.isArray(picked) ? picked : picked ? [picked] : [];
    if (paths.length) await ingestPaths(paths);
  }

  /** Pick a folder and ingest every supported file inside it (and its
   *  subdirectories). Confirms first when a lot of files are found. */
  async function addFolder() {
    setError("");
    const dir = await open({ directory: true });
    if (!dir || Array.isArray(dir)) return;
    let files: string[];
    try {
      files = await ragListSupportedFiles(dir);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return;
    }
    if (files.length === 0) {
      setError(t("kbFolderEmpty"));
      return;
    }
    if (
      files.length > 20 &&
      !(await confirm({
        title: t("kbAddFolder"),
        message: t("kbFolderConfirm", { n: files.length }),
        confirmLabel: t("kbFolderConfirmGo"),
      }))
    ) {
      return;
    }
    await ingestPaths(files, dir);
  }

  async function clearAll() {
    if (
      !(await confirm({
        title: t("kbClear"),
        message: t("kbClearConfirm"),
        confirmLabel: t("kbClear"),
        danger: true,
      }))
    ) {
      return;
    }
    try {
      await ragClearAll();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
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
            {docs.length > 0 && (
              <div className="kb-scope-row">
                <span className="kb-scope-hint">{t("kbScopeHint")}</span>
                {!indexing && (
                  <button className="kb-clear" onClick={() => void clearAll()}>
                    {t("kbClear")}
                  </button>
                )}
              </div>
            )}
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
                <>
                  <button className="setup-dl" onClick={() => void addDocuments()}>
                    + {t("kbAdd")}
                  </button>
                  <button className="setup-dl" onClick={() => void addFolder()}>
                    {t("kbAddFolder")}
                  </button>
                </>
              )}
              {onReport && docs.length > 0 && !indexing && (
                <button className="setup-dl kb-report" onClick={onReport}>
                  <IconResearch size={15} style={{ marginRight: 6 }} /> {t("kbReport")}
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
