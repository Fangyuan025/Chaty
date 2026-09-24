import { useMemo, useState } from "react";

import { useI18n } from "../lib/i18n";
import type { ImageRecord } from "../lib/ipc";
import type { ImageStudioState } from "../lib/useImageStudio";
import { Icon } from "./Icon";
import { ImageThumb } from "./ImageThumb";
import { useConfirm } from "./ConfirmModal";

type Group = { key: "imgToday" | "imgYesterday" | "imgEarlier"; items: ImageRecord[] };

function groups(list: ImageRecord[]): Group[] {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const today = start.getTime();
  const yesterday = today - 86_400_000;
  const out: Group[] = [
    { key: "imgToday", items: [] },
    { key: "imgYesterday", items: [] },
    { key: "imgEarlier", items: [] },
  ];
  for (const r of list) {
    out[r.createdAt >= today ? 0 : r.createdAt >= yesterday ? 1 : 2].items.push(r);
  }
  return out.filter((g) => g.items.length > 0);
}

/** The sidebar in the image studio: every generation, newest first. */
export function ImageSidebar({
  studio,
  busy,
  notify,
}: {
  studio: ImageStudioState;
  busy: boolean;
  notify: (kind: "warn" | "error", text: string) => void;
}) {
  const { t, lang } = useI18n();
  const confirm = useConfirm();
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const list = useMemo(
    () => (q ? studio.history.filter((r) => r.prompt.toLowerCase().includes(q)) : studio.history),
    [studio.history, q],
  );
  const time = (ms: number) =>
    new Date(ms).toLocaleTimeString(lang === "zh" ? "zh-CN" : "en-US", { hour: "2-digit", minute: "2-digit" });

  const del = async (r: ImageRecord) => {
    const ok = await confirm({
      title: t("imgDeleteTitle"),
      message: t("imgDeleteConfirm"),
      confirmLabel: t("confirmDelete"),
      danger: true,
    });
    if (ok) await studio.remove(r.id, true).catch((e) => notify("error", String(e)));
  };

  return (
    <>
      <button className="new-chat" onClick={studio.startNew} disabled={busy}>
        <Icon name="plus" size={13} strokeWidth={2} /> {t("imgNew")}
      </button>
      {studio.history.length > 0 && (
        <div className="conv-search">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="M21 21l-4.3-4.3" strokeLinecap="round" />
          </svg>
          <input type="text" placeholder={t("imgSearch")} value={query} onChange={(e) => setQuery(e.target.value)} />
          {query && (
            <button className="conv-search-clear" onClick={() => setQuery("")} title={t("cancel")}>
              <Icon name="x" size={11} strokeWidth={2.2} />
            </button>
          )}
        </div>
      )}
      <div className="conv-list">
        {list.length === 0 ? (
          <div className="conv-empty">{studio.history.length === 0 ? t("imgNoHistory") : t("noMatches")}</div>
        ) : (
          groups(list).map((g) => (
            <div key={g.key} className="is-group">
              <div className="is-group-title">{t(g.key)}</div>
              {g.items.map((r) => (
                <div
                  key={r.id}
                  className={`conv-item is-hist ${r.id === studio.selectedId ? "active" : ""}`}
                  onClick={() => !busy && studio.select(r.id)}
                  title={r.prompt}
                >
                  {r.images[0] ? <ImageThumb path={r.images[0].path} size={38} className="is-hist-thumb" /> : <span className="is-hist-thumb" />}
                  <span className="is-hist-text">
                    <span className="conv-title">{r.prompt}</span>
                    <span className="is-hist-meta">
                      {time(r.createdAt)}
                      {r.images.length > 1 ? ` · ${t("imgCountN", { n: r.images.length })}` : ""}
                    </span>
                  </span>
                  <div className="conv-actions">
                    <button
                      className="conv-del"
                      title={t("imgDelete")}
                      onClick={(e) => {
                        e.stopPropagation();
                        void del(r);
                      }}
                    >
                      <Icon name="x" size={11} strokeWidth={2.2} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ))
        )}
      </div>
    </>
  );
}
