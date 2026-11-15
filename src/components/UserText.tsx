import { useState } from "react";

import { smoothly } from "../lib/smoothHeight";
import { copyToClipboard } from "../lib/clipboard";
import { useI18n } from "../lib/i18n";

/** User-message text with a clamp for pasted walls of text: over ~15 lines or
 *  1200 chars it renders a 220px preview with a fade + expand pill. */
export function UserText({ content, expandLabel, collapseLabel }: { content: string; expandLabel: string; collapseLabel: string }) {
  const long = content.length > 1200 || content.split("\n").length > 15;
  const [open, setOpen] = useState(false);
  if (!long) return <span className="user-text">{content}</span>;
  return (
    <>
      <span className={`user-text ${open ? "" : "clamped"}`}>{content}</span>
      <button
        className="user-expand"
        type="button"
        onClick={(e) => smoothly(e.currentTarget.closest<HTMLElement>(".bubble"), () => setOpen(!open))}
      >
        {open ? collapseLabel : expandLabel}
      </button>
    </>
  );
}

/** Hover copy button on user messages (mirrors the edit pencil). */
export function UserCopy({ content, title }: { content: string; title: string }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      className="user-edit user-copy"
      title={title}
      onClick={() =>
        void copyToClipboard(content).then(() => {
          setOk(true);
          setTimeout(() => setOk(false), 1400);
        })
      }
    >
      {ok ? (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
          <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15V5a2 2 0 0 1 2-2h8" strokeLinecap="round" />
        </svg>
      )}
    </button>
  );
}

/** When a message was sent, by the copy and edit buttons under it: "just
 *  now", "5 minutes ago", an earlier day's date and time — the whole of it
 *  on hover. In the interface's language. */
export function SentAt({ at }: { at: number }) {
  const { lang: ui } = useI18n();
  const lang = ui === "zh" ? "zh-CN" : ui === "pt" ? "pt-BR" : ui;
  const now = Date.now();
  const secs = Math.round((now - at) / 1000);
  const rel = new Intl.RelativeTimeFormat(lang, { numeric: "auto" });
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(new Date(now)) - day(new Date(at))) / 86400000);
  const time = new Date(at).toLocaleTimeString(lang, { hour: "2-digit", minute: "2-digit" });
  const label =
    secs < 45
      ? ui === "zh" ? "刚刚" : ui === "pt" ? "agora mesmo" : "just now"
      : secs < 3600
        ? rel.format(-Math.round(secs / 60), "minute")
        : days === 0
          ? rel.format(-Math.round(secs / 3600), "hour")
          : days === 1
            ? `${rel.format(-1, "day")} ${time}`
            : new Date(at).toLocaleDateString(lang, { month: "short", day: "numeric", ...(days > 300 ? { year: "numeric" } : {}) }) + ` ${time}`;
  const full = new Date(at).toLocaleString(lang, { dateStyle: "full", timeStyle: "short" });
  return (
    <span className="user-time" title={full}>
      {label}
    </span>
  );
}
