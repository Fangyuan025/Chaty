import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { imageDataUrl, saveImageAs } from "../lib/ipc";
import { Icon } from "./Icon";

/** Full-size image preview modal with a "save to local" action. Shared by the
 *  code agent's screenshots and the pictures attached to chat messages. */
export function ImagePreview({
  path,
  title,
  onClose,
}: {
  path: string;
  /** Heading of the preview; the code agent's default is "Screenshot". */
  title?: string;
  onClose: () => void;
}) {
  const { t } = useI18n();
  // The picture, decoded, with its size — or "failed". The window opens once
  // it is ready, at its size: it opened as a spinner and then grew to the
  // picture's height, both ways from the middle, which read as a jolt.
  const [img, setImg] = useState<{ src: string; w: number; h: number } | "failed" | null>(null);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    let live = true;
    // Full-resolution (crisp) — not a downscaled thumbnail.
    imageDataUrl(path)
      .then(async (src) => {
        const el = new Image();
        el.src = src;
        try {
          await el.decode();
        } catch {
          /* drawn as it comes, then */
        }
        if (live) setImg(el.naturalWidth ? { src, w: el.naturalWidth, h: el.naturalHeight } : "failed");
      })
      .catch(() => live && setImg("failed"));
    return () => {
      live = false;
    };
  }, [path]);
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const save = async () => {
    try {
      const name = path.split(/[/\\]/).pop() || "screenshot.png";
      const dest = await saveImageAs(path, name);
      if (dest) setSaved(true);
    } catch (e) {
      console.error(e);
    }
  };
  if (img === null) {
    return createPortal(
      <div className="preview-overlay" onMouseDown={onClose}>
        <span className="cm-spin" />
      </div>,
      document.body,
    );
  }
  return createPortal(
    <div className="preview-overlay" onMouseDown={onClose}>
      <div className="cm-preview" onMouseDown={(e) => e.stopPropagation()}>
        <div className="cm-preview-bar">
          <span className="cm-preview-title">{title ?? t("cmScreenshot")}</span>
          <button className="cm-preview-btn" onClick={() => void save()}>
            <Icon name="download" size={13} strokeWidth={2} />
            {saved ? t("cmSaved") : t("cmSaveImage")}
          </button>
          <button className="cm-preview-close" onClick={onClose} title={t("closePreview")}>
            <Icon name="x" size={14} strokeWidth={2.2} />
          </button>
        </div>
        <div className="cm-preview-stage">
          {img === "failed" ? (
            <span className="muted">{t("imagePreviewFailed")}</span>
          ) : (
            <img src={img.src} width={img.w} height={img.h} alt="" />
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
