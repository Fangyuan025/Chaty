import { useEffect, useState } from "react";

import { useI18n } from "../lib/i18n";
import { imageSessionSearch, type ImageSession } from "../lib/ipc";
import type { ImageStudioState } from "../lib/useImageStudio";
import { Icon } from "./Icon";
import { IconEdit, IconPin, IconPinFilled } from "./icons";
import { useConfirm } from "./ConfirmModal";
import { SelectBar, SelectCheck, SelectToggle, selectClick, useMultiSelect } from "./MultiSelect";

/** The sidebar in the image studio: its sessions, the way the chat lists its
 *  conversations — pinned first, then most recent, searchable, renameable. */
export function ImageSidebar({
  studio,
  busy,
  notify,
}: {
  studio: ImageStudioState;
  busy: boolean;
  notify: (kind: "warn" | "error", text: string) => void;
}) {
  const { t } = useI18n();
  const confirm = useConfirm();
  const [query, setQuery] = useState("");
  const [contentMatches, setContentMatches] = useState<Set<string>>(new Set());
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  /** Selecting several sessions to delete them together. */
  const sel = useMultiSelect();

  // Titles match here at once; prompts inside the sessions are searched in
  // the database, debounced — the chat sidebar's search, for images.
  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) {
      setContentMatches(new Set());
      return;
    }
    let cancelled = false;
    const id = window.setTimeout(() => {
      imageSessionSearch(q)
        .then((ids) => {
          if (!cancelled) setContentMatches(new Set(ids));
        })
        .catch(() => {});
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(id);
    };
  }, [query]);

  const q = query.trim().toLowerCase();
  const visible = q
    ? studio.sessions.filter((s) => s.title.toLowerCase().includes(q) || contentMatches.has(s.id))
    : studio.sessions;
  sel.order(visible.map((s) => s.id));

  const del = async (s: ImageSession) => {
    const ok = await confirm({
      title: t("imgDeleteSession"),
      message: t("imgDeleteSessionConfirm"),
      confirmLabel: t("confirmDelete"),
      danger: true,
    });
    if (ok) await studio.deleteSession(s.id).catch((e) => notify("error", String(e)));
  };

  /** Delete the ticked sessions after one confirmation. */
  const delSelected = async () => {
    const ids = sel.picked();
    if (ids.length === 0) return;
    const ok = await confirm({
      title: t("imgDeleteSession"),
      // The pictures go with them, as with one session.
      message: t("imgDeleteSessionsConfirm", { n: ids.length }),
      confirmLabel: t("confirmDelete"),
      danger: true,
    });
    if (!ok) return;
    for (const id of ids) await studio.deleteSession(id).catch((e) => notify("error", String(e)));
    sel.exit();
  };

  const commitRename = async () => {
    const id = renamingId;
    const title = renameDraft.trim();
    setRenamingId(null);
    if (!id || !title) return;
    await studio.rename(id, title).catch(console.error);
  };

  return (
    <>
      <button className="new-chat" onClick={studio.startNew} disabled={busy}>
        <Icon name="plus" size={13} strokeWidth={2} /> {t("imgNew")}
      </button>
      {studio.sessions.length > 0 && (
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
          <SelectToggle ms={sel} />
        </div>
      )}
      <div className="conv-list">
        {visible.length === 0 ? (
          <div className="conv-empty">{studio.sessions.length === 0 ? t("imgNoHistory") : t("noMatches")}</div>
        ) : (
          visible.map((s) => (
            <div
              key={s.id}
              className={`conv-item ${s.id === studio.sessionId && !sel.selecting ? "active" : ""} ${
                s.pinned ? "pinned" : ""
              } ${sel.selecting ? "selecting" : ""} ${sel.selected.has(s.id) ? "picked" : ""}`}
              onClick={(e) => {
                if (renamingId === s.id) return;
                if (selectClick(sel, s.id, e)) return;
                if (!busy) void studio.openSession(s.id);
              }}
            >
              {sel.selecting && <SelectCheck on={sel.selected.has(s.id)} />}
              {renamingId === s.id ? (
                <input
                  className="conv-rename"
                  autoFocus
                  value={renameDraft}
                  onChange={(e) => setRenameDraft(e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void commitRename();
                    } else if (e.key === "Escape") {
                      e.preventDefault();
                      setRenamingId(null);
                    }
                  }}
                  onBlur={() => void commitRename()}
                />
              ) : (
                <>
                  <span className="conv-title">{s.title}</span>
                  {!sel.selecting && <div className="conv-actions">
                    <button
                      className={`conv-act ${s.pinned ? "on" : ""}`}
                      title={s.pinned ? t("unpinConv") : t("pinConv")}
                      onClick={(e) => {
                        e.stopPropagation();
                        void studio.togglePin(s).catch(console.error);
                      }}
                    >
                      {s.pinned ? <IconPinFilled size={13} /> : <IconPin size={13} />}
                    </button>
                    <button
                      className="conv-act"
                      title={t("renameConv")}
                      onClick={(e) => {
                        e.stopPropagation();
                        setRenamingId(s.id);
                        setRenameDraft(s.title);
                      }}
                    >
                      <IconEdit size={13} />
                    </button>
                    <button
                      className="conv-del"
                      title={t("imgDeleteSession")}
                      onClick={(e) => {
                        e.stopPropagation();
                        void del(s);
                      }}
                    >
                      <Icon name="x" size={11} strokeWidth={2.2} />
                    </button>
                  </div>}
                </>
              )}
            </div>
          ))
        )}
      </div>
      <SelectBar ms={sel} onDelete={() => void delSelected()} busy={busy} />
    </>
  );
}
