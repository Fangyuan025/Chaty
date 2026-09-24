import { useState } from "react";

import { copyToClipboard } from "../lib/clipboard";

/** User-message text with a clamp for pasted walls of text: over ~15 lines or
 *  1200 chars it renders a 220px preview with a fade + expand pill. */
export function UserText({ content, expandLabel, collapseLabel }: { content: string; expandLabel: string; collapseLabel: string }) {
  const long = content.length > 1200 || content.split("\n").length > 15;
  const [open, setOpen] = useState(false);
  if (!long) return <span className="user-text">{content}</span>;
  return (
    <>
      <span className={`user-text ${open ? "" : "clamped"}`}>{content}</span>
      <button className="user-expand" type="button" onClick={() => setOpen(!open)}>
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
