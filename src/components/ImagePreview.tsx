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
  const [src, setSrc] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    let live = true;
    // Full-resolution (crisp) — not a downscaled thumbnail.
    imageDataUrl(path)
      .then((d) => live && setSrc(d))
      .catch(() => live && setSrc(""));
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
          {src ? <img src={src} alt="" /> : <span className="cm-spin" />}
        </div>
      </div>
    </div>,
    document.body,
  );
}
