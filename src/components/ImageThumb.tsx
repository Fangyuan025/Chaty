import { useEffect, useState } from "react";
import { imageThumb } from "../lib/ipc";

/** Module-level thumbnail cache: path → data URL (survives re-renders). */
const thumbCache = new Map<string, string>();

/** Self-loading thumbnail for a local image path; hides itself if unreadable.
 *  Shared by chat attachments and the image studio's history. */
export function ImageThumb({
  path,
  size = 168,
  onOpen,
  className,
}: {
  path: string;
  size?: number;
  /** Clicking the picture opens it full size (issue #18: it did nothing). */
  onOpen?: () => void;
  className?: string;
}) {
  const [src, setSrc] = useState<string | null>(thumbCache.get(path) ?? null);
  useEffect(() => {
    let live = true;
    const hit = thumbCache.get(path);
    if (hit) {
      setSrc(hit);
    } else {
      setSrc(null);
      imageThumb(path, 512)
        .then((d) => {
          thumbCache.set(path, d);
          if (live) setSrc(d);
        })
        .catch(() => {
          if (live) setSrc("");
        });
    }
    return () => {
      live = false;
    };
  }, [path]);
  if (src === "") return null; // moved/deleted — degrade quietly
  return (
    <span
      className={`${onOpen ? "img-thumb img-thumb-open" : "img-thumb"}${className ? ` ${className}` : ""}`}
      style={{ maxWidth: size, maxHeight: size }}
      role={onOpen ? "button" : undefined}
      onClick={onOpen}
    >
      {src ? <img src={src} alt="" /> : <span className="img-thumb-ph" />}
    </span>
  );
}
