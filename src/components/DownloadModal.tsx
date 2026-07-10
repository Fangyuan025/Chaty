import { useEffect, useRef, useState } from "react";
import { etaSeconds, fmtTime, type EtaSample } from "../lib/eta";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import {
  listHfGgufs,
  downloadModel,
  cancelDownload,
  DOWNLOAD_CANCELLED,
  type HfFile,
} from "../lib/ipc";

function fmtSize(n: number): string {
  if (!n) return "";
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(0)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

export function DownloadModal({
  onClose,
  onDownloaded,
  initialRepo,
  initialFile,
}: {
  onClose: () => void;
  onDownloaded: () => void;
  /** Pre-fill the repo (e.g. from a chaty:// deep link) and list its files. */
  initialRepo?: string;
  /** Auto-start this file's download once the repo's files are listed. */
  initialFile?: string;
}) {
  const { t } = useI18n();
  const [repo, setRepo] = useState("");
  const [files, setFiles] = useState<HfFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [active, setActive] = useState<string | null>(null); // file being downloaded
  const [progress, setProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [eta, setEta] = useState<number | null>(null);
  const etaStore = useRef<EtaSample[]>([]);

  const search = async () => {
    if (!repo.trim() || loading) return;
    setError("");
    setFiles([]);
    setLoading(true);
    try {
      setFiles(await listHfGgufs(repo));
    } catch (e) {
      setError(typeof e === "string" ? e : t("dlSearchFailed"));
    } finally {
      setLoading(false);
    }
  };

  const download = async (f: HfFile) => {
    if (active) return;
    setError("");
    setActive(f.name);
    setProgress({ done: 0, total: f.size });
    setEta(null);
    etaStore.current = [];
    try {
      await downloadModel(f.url, f.name, (p) => {
        if (p.type === "progress") {
          const total = p.total || f.size;
          setProgress({ done: p.downloaded, total });
          setEta(etaSeconds(etaStore.current, p.downloaded, total));
        } else if (p.type === "error" && p.message !== DOWNLOAD_CANCELLED) setError(p.message);
      });
      onDownloaded();
      onClose();
    } catch (e) {
      const msg = typeof e === "string" ? e : "";
      if (msg !== DOWNLOAD_CANCELLED) setError(msg || t("dlFailed"));
      setActive(null);
    }
  };

  // Deep link (chaty://open_from_hf): pre-load the repo and, if a specific file
  // was named, start it automatically.
  useEffect(() => {
    if (!initialRepo) return;
    setRepo(initialRepo);
    (async () => {
      setError("");
      setFiles([]);
      setLoading(true);
      try {
        const found = await listHfGgufs(initialRepo);
        setFiles(found);
        if (initialFile) {
          const f = found.find((x) => x.name === initialFile);
          if (f) void download(f);
        }
      } catch (e) {
        setError(typeof e === "string" ? e : t("dlSearchFailed"));
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialRepo, initialFile]);

  const pct = progress.total > 0 ? Math.min(100, (progress.done / progress.total) * 100) : 0;

  return (
    <>
      <div className="popover-backdrop" onClick={active ? undefined : onClose} style={{ zIndex: 90 }} />
      <div className="dl-modal">
        <div className="dl-head">
          <span className="dl-title">{t("dlTitle")}</span>
          {!active && (
            <button className="dl-close" onClick={onClose} title={t("cancel")}>
              <Icon name="x" size={12} strokeWidth={2.2} />
            </button>
          )}
        </div>

        <div className="dl-search">
          <input
            type="text"
            placeholder={t("dlRepoPh")}
            value={repo}
            disabled={!!active}
            onChange={(e) => setRepo(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") search();
            }}
          />
          <button onClick={search} disabled={loading || !repo.trim() || !!active}>
            {loading ? t("dlSearching") : t("dlSearch")}
          </button>
        </div>

        <div className="dl-hint">{t("dlHint")}</div>
        {error && <div className="dl-error">{error}</div>}

        <div className="dl-list">
          {files.map((f) => (
            <div key={f.name} className="dl-item">
              <div className="dl-item-info">
                <span className="dl-item-name">{f.name}</span>
                <span className="dl-item-size">{fmtSize(f.size)}</span>
              </div>
              {active === f.name ? (
                <div className="dl-progress">
                  <div className="dl-bar">
                    <div className="dl-bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="dl-pct">
                    {fmtSize(progress.done)}
                    {progress.total ? ` / ${fmtSize(progress.total)}` : ""}
                    {eta !== null ? ` · ${t("etaLeft")} ~${fmtTime(eta)}` : ""}
                  </span>
                  <button
                    className="dl-cancel"
                    title={t("cancel")}
                    onClick={() => void cancelDownload(f.name).catch(() => {})}
                  >
                    <Icon name="x" size={12} strokeWidth={2.2} />
                  </button>
                </div>
              ) : (
                <button className="dl-get" onClick={() => download(f)} disabled={!!active}>
                  {t("dlGet")}
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
