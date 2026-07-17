import { useCallback, useEffect, useRef, useState } from "react";
import { platform } from "@tauri-apps/plugin-os";
import { etaSeconds, fmtTime, type EtaSample } from "../lib/eta";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { Markdown } from "./Markdown";
import { OrgAvatar } from "./VendorIcon";
import {
  hfSearch,
  hfModelDetail,
  hfResolveUrl,
  hfBase,
  downloadModel,
  downloadMlxRepo,
  cancelDownload,
  modelFolderFor,
  DOWNLOAD_CANCELLED,
  type HfModelHit,
  type HfModelDetail,
} from "../lib/ipc";

// MLX repos are only usable on macOS (Apple-Silicon sidecar).
const IS_MACOS = (() => {
  try {
    return platform() === "macos";
  } catch {
    return false;
  }
})();

function fmtSize(n: number): string {
  if (!n) return "";
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(0)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function daysAgo(iso: string, t: ReturnType<typeof useI18n>["t"]): string {
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return "";
  const d = Math.max(0, Math.floor((Date.now() - ms) / 86_400_000));
  if (d === 0) return t("storeToday");
  return t("storeDaysAgo", { n: String(d) });
}

/**
 * HF READMEs mix raw HTML into markdown. react-markdown (rightly) escapes
 * HTML, which showed as source code — so: relative image/link URLs become
 * absolute `resolve/main` URLs, `<img>` tags become markdown images, and any
 * remaining HTML tags are stripped (their text content stays).
 */
function cleanReadme(md: string, repo: string): string {
  const abs = (u: string) =>
    /^(https?:)?\/\//.test(u) || u.startsWith("#")
      ? u
      : `${hfBase()}/${repo}/resolve/main/${u.replace(/^\.?\//, "")}`;
  let s = md;
  // <img src=…> → ![](abs)
  s = s.replace(/<img[^>]*?src=["']([^"']+)["'][^>]*>/gi, (_m, src) => `\n\n![](${abs(src)})\n\n`);
  // markdown images/links with relative targets → absolute
  s = s.replace(/(!?\[[^\]]*\]\()([^)\s]+)(\))/g, (_m, pre, url, post) => pre + abs(url) + post);
  // drop every other tag, keep inner text
  s = s.replace(/<\/?[a-zA-Z][^>]*>/g, "\n");
  // drop html comments and collapse blank runs
  s = s.replace(/<!--[\s\S]*?-->/g, "");
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}

/** owner/name or a pasted HF URL → repo id; null when it's just a keyword. */
function asRepoId(input: string): string | null {
  const s = input
    .trim()
    .replace(/^https?:\/\//, "")
    .replace(/^huggingface\.co\//, "")
    .replace(/\/(tree|blob)\/.*$/, "")
    .replace(/^\/+|\/+$/g, "");
  return /^[\w.-]+\/[\w.-]+$/.test(s) ? s : null;
}

export function DownloadModal({
  onClose,
  onDownloaded,
  initialRepo,
  initialFile,
}: {
  onClose: () => void;
  onDownloaded: () => void;
  /** Pre-open this repo's detail (e.g. from a chaty:// deep link). */
  initialRepo?: string;
  /** Auto-start the quant containing this file once the detail loads. */
  initialFile?: string;
}) {
  const { t } = useI18n();
  // -- browse state --
  const [query, setQuery] = useState("");
  const [format, setFormat] = useState<"gguf" | "mlx">("gguf");
  const [sort, setSort] = useState<"trending" | "downloads" | "likes" | "updated">("trending");
  const [hits, setHits] = useState<HfModelHit[]>([]);
  const [listLoading, setListLoading] = useState(false);
  // -- detail state --
  const [detail, setDetail] = useState<HfModelDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [quantIdx, setQuantIdx] = useState(0);
  // -- download state --
  const [active, setActive] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [eta, setEta] = useState<number | null>(null);
  const etaStore = useRef<EtaSample[]>([]);
  const cancelKey = useRef<string | null>(null);
  const [error, setError] = useState("");
  const seq = useRef(0);

  const runSearch = useCallback(
    async (q: string, f: "gguf" | "mlx", s: typeof sort) => {
      const my = ++seq.current;
      setListLoading(true);
      setError("");
      try {
        const found = await hfSearch(q, f, s);
        if (my === seq.current) setHits(found);
      } catch (e) {
        if (my === seq.current) {
          setHits([]);
          setError(typeof e === "string" ? e : t("dlSearchFailed"));
        }
      } finally {
        if (my === seq.current) setListLoading(false);
      }
    },
    [t],
  );

  // storefront on open + debounced live search
  useEffect(() => {
    const id = setTimeout(() => void runSearch(query, format, sort), query ? 350 : 0);
    return () => clearTimeout(id);
  }, [query, format, sort, runSearch]);

  const openDetail = useCallback(
    async (repo: string, f: "gguf" | "mlx" = format) => {
      setDetailLoading(true);
      setDetail(null);
      setError("");
      try {
        const d = await hfModelDetail(repo, f);
        setDetail(d);
        // default to a mid-size quant — the classic "Q4-ish" pick
        setQuantIdx(Math.min(Math.floor(d.quants.length / 2), d.quants.length - 1));
        return d;
      } catch (e) {
        setError(typeof e === "string" ? e : t("dlSearchFailed"));
        return null;
      } finally {
        setDetailLoading(false);
      }
    },
    [format, t],
  );

  const startDownload = useCallback(
    async (d: HfModelDetail, qi: number) => {
      if (active) return;
      const quant = d.quants[qi];
      if (!quant) return;
      setActive(true);
      setError("");
      etaStore.current = [];
      const grandTotal =
        d.format === "mlx" ? quant.size : quant.size + (d.mmproj ? d.mmprojSize : 0);
      setProgress({ done: 0, total: grandTotal });
      setEta(null);
      const tick = (done: number) => {
        setProgress({ done, total: grandTotal });
        setEta(etaSeconds(etaStore.current, done, grandTotal));
      };
      try {
        if (d.format === "mlx") {
          cancelKey.current = d.id.split("/").pop() ?? d.id;
          await downloadMlxRepo(d.id, (p) => {
            if (p.type === "progress") tick(p.downloaded);
            else if (p.type === "error" && p.message !== DOWNLOAD_CANCELLED) setError(p.message);
          });
        } else {
          const subdir = modelFolderFor(d.id);
          const files = [...quant.files, ...(d.mmproj ? [d.mmproj] : [])];
          let doneBase = 0;
          for (const path of files) {
            const name = path.split("/").pop() ?? path;
            cancelKey.current = name;
            let thisFile = 0;
            await downloadModel(
              hfResolveUrl(d.id, path),
              name,
              (p) => {
                if (p.type === "progress") {
                  thisFile = p.downloaded;
                  tick(doneBase + thisFile);
                } else if (p.type === "error" && p.message !== DOWNLOAD_CANCELLED) {
                  setError(p.message);
                }
              },
              subdir,
            );
            doneBase += thisFile;
          }
        }
        onDownloaded();
        onClose();
      } catch (e) {
        const msg = typeof e === "string" ? e : "";
        if (msg !== DOWNLOAD_CANCELLED) setError(msg || t("dlFailed"));
      } finally {
        setActive(false);
        cancelKey.current = null;
      }
    },
    [active, onClose, onDownloaded, t],
  );

  // Deep link: open the repo directly; a named file selects + starts its quant.
  useEffect(() => {
    if (!initialRepo) return;
    void (async () => {
      const d = await openDetail(initialRepo, "gguf"); // backend auto-detects MLX
      if (!d) return;
      if (d.format === "mlx" && IS_MACOS) {
        void startDownload(d, 0);
      } else if (initialFile) {
        const qi = d.quants.findIndex((q) => q.files.some((f) => f.endsWith(initialFile)));
        if (qi >= 0) {
          setQuantIdx(qi);
          void startDownload(d, qi);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialRepo, initialFile]);

  const onSearchEnter = () => {
    const repo = asRepoId(query);
    if (repo) void openDetail(repo);
  };

  const quant = detail?.quants[quantIdx];
  const needBytes = quant ? quant.size + (detail?.format === "gguf" && detail.mmproj ? detail.mmprojSize : 0) : 0;
  const fitsRam = detail ? needBytes * 1.15 < detail.totalRamMb * 1024 * 1024 : false;
  const pct = progress.total > 0 ? Math.min(100, (progress.done / progress.total) * 100) : 0;

  return (
    <>
      <div className="popover-backdrop" onClick={active ? undefined : onClose} style={{ zIndex: 90 }} />
      <div className="dl-modal store-modal">
        <div className="dl-head">
          <span className="dl-title">{t("dlTitle")}</span>
          {!active && (
            <button className="dl-close" onClick={onClose} title={t("cancel")}>
              <Icon name="x" size={12} strokeWidth={2.2} />
            </button>
          )}
        </div>

        <div className="store-body">
          {/* ---- left: search + browse ---- */}
          <div className="store-list-pane">
            <div className="dl-search">
              <input
                type="text"
                placeholder={t("storeSearchPh")}
                value={query}
                disabled={active}
                autoFocus
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onSearchEnter();
                }}
              />
            </div>
            <div className="store-filters">
              <select
                value={format}
                disabled={active}
                onChange={(e) => setFormat(e.target.value as "gguf" | "mlx")}
                title={t("storeFormat")}
              >
                <option value="gguf">GGUF</option>
                {IS_MACOS && <option value="mlx">MLX</option>}
              </select>
              <select
                value={sort}
                disabled={active}
                onChange={(e) => setSort(e.target.value as typeof sort)}
                title={t("storeSort")}
              >
                <option value="trending">{t("storeTrending")}</option>
                <option value="downloads">{t("storeDownloads")}</option>
                <option value="likes">{t("storeLikes")}</option>
                <option value="updated">{t("storeUpdated")}</option>
              </select>
            </div>
            <div className="store-hits">
              {listLoading ? (
                Array.from({ length: 6 }, (_, i) => (
                  <div key={i} className="store-hit store-skel" style={{ animationDelay: `${i * 60}ms` }}>
                    <span className="store-avatar skel-block" />
                    <span className="store-hit-main">
                      <span className="skel-block skel-line" style={{ width: `${70 - i * 6}%` }} />
                      <span className="skel-block skel-line thin" style={{ width: `${45 + (i % 3) * 8}%` }} />
                    </span>
                  </div>
                ))
              ) : hits.length === 0 ? (
                <div className="store-empty">{t("storeNoResults")}</div>
              ) : (
                hits.map((h) => (
                  <button
                    key={h.id}
                    className={`store-hit ${detail?.id === h.id ? "active" : ""}`}
                    disabled={active}
                    onClick={() => void openDetail(h.id)}
                    title={h.id}
                  >
                    <span className="store-avatar">
                      <OrgAvatar author={h.author} size={22} />
                    </span>
                    <span className="store-hit-main">
                      <span className="store-hit-name">
                        {h.name}
                        {h.vision && <span className="mm-vision">{t("visionBadge")}</span>}
                      </span>
                      <span className="store-hit-sub">
                        {h.author}
                        {h.paramsB ? ` · ${h.paramsB}B` : ""} · ↓{fmtCount(h.downloads)} ·{" "}
                        {daysAgo(h.updatedAt, t)}
                      </span>
                    </span>
                  </button>
                ))
              )}
            </div>
            <div className="dl-hint">{t("dlHint")}</div>
          </div>

          {/* ---- right: detail ---- */}
          <div className="store-detail-pane">
            {detailLoading ? (
              <div className="store-detail-skel">
                <span className="skel-block skel-line" style={{ width: "55%", height: 22 }} />
                <span className="skel-block skel-line" style={{ width: "35%" }} />
                <span className="skel-block" style={{ width: "100%", height: 92, borderRadius: 12 }} />
                <span className="skel-block skel-line" style={{ width: "90%" }} />
                <span className="skel-block skel-line" style={{ width: "80%" }} />
                <span className="skel-block skel-line" style={{ width: "85%" }} />
              </div>
            ) : !detail ? (
              <div className="store-empty">{t("storePickModel")}</div>
            ) : (
              <>
                <div className="store-detail-title">{detail.id.split("/").pop()}</div>
                <div className="store-detail-author">
                  <OrgAvatar author={detail.id.split("/")[0]} size={15} /> {detail.id.split("/")[0]}
                </div>
                <div className="store-badges">
                  {detail.paramsB ? <span className="store-badge">{detail.paramsB}B</span> : null}
                  {detail.arch ? <span className="store-badge">{detail.arch}</span> : null}
                  <span className="store-badge store-badge-fmt">{detail.format.toUpperCase()}</span>
                  {detail.vision && (
                    <span className="store-badge store-badge-vision">{t("visionBadge")}</span>
                  )}
                </div>

                <div className="store-dl-box">
                  <div className="store-quant-row">
                    <select
                      value={quantIdx}
                      disabled={active || detail.quants.length <= 1}
                      onChange={(e) => setQuantIdx(Number(e.target.value))}
                    >
                      {detail.quants.map((q, i) => (
                        <option key={q.label} value={i}>
                          {q.label} · {fmtSize(q.size)}
                        </option>
                      ))}
                    </select>
                    {!active ? (
                      <button
                        className="dl-get store-get"
                        onClick={() => void startDownload(detail, quantIdx)}
                        disabled={detail.format === "mlx" && !IS_MACOS}
                      >
                        {t("dlGet")} {quant ? fmtSize(needBytes) : ""}
                      </button>
                    ) : (
                      <button
                        className="dl-cancel"
                        title={t("cancel")}
                        onClick={() => {
                          if (cancelKey.current) void cancelDownload(cancelKey.current).catch(() => {});
                        }}
                      >
                        <Icon name="x" size={12} strokeWidth={2.2} />
                      </button>
                    )}
                  </div>
                  {active && (
                    <div className="dl-progress store-progress">
                      <div className="dl-bar">
                        <div className="dl-bar-fill" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="dl-pct">
                        {fmtSize(progress.done)}
                        {progress.total ? ` / ${fmtSize(progress.total)}` : ""}
                        {eta !== null ? ` · ${t("etaLeft")} ~${fmtTime(eta)}` : ""}
                      </span>
                    </div>
                  )}
                  <div className={`store-fit ${fitsRam ? "ok" : "warn"}`}>
                    {detail.format === "mlx" && !IS_MACOS
                      ? t("storeMlxMacOnly")
                      : (fitsRam ? t("storeFitsRam") : t("storeOverRam")) +
                        (detail.format === "gguf" && detail.mmproj ? ` · ${t("storeVisionIncluded")}` : "")}
                  </div>
                </div>

                <div
                  className="store-readme"
                  onErrorCapture={(e) => {
                    // README images resolve against the HF CDN, which some
                    // networks block — hide broken images instead of showing
                    // the browser's broken-image glyph.
                    const el = e.target as HTMLElement;
                    if (el.tagName === "IMG") el.style.display = "none";
                  }}
                >
                  {detail.readme ? (
                    <Markdown>{cleanReadme(detail.readme, detail.id)}</Markdown>
                  ) : (
                    <div className="store-empty">{t("storeReadmeEmpty")}</div>
                  )}
                </div>
              </>
            )}
            {error && <div className="dl-error">{error}</div>}
          </div>
        </div>
      </div>
    </>
  );
}
