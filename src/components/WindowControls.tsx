import { useEffect, useMemo, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useI18n } from "../lib/i18n";

export function WindowControls() {
  const { t } = useI18n();
  const [maximized, setMaximized] = useState(false);
  // Resolved lazily inside the component: at module scope this reads Tauri
  // internals that don't exist in a plain-browser preview and crashes the
  // whole module graph before anything renders.
  const appWindow = useMemo(() => {
    try {
      return getCurrentWindow();
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    if (!appWindow) return;
    let unlisten: (() => void) | undefined;
    const sync = () => appWindow.isMaximized().then(setMaximized).catch(() => {});
    sync();
    appWindow
      .onResized(sync)
      .then((u) => {
        unlisten = u;
      })
      .catch(() => {});
    return () => unlisten?.();
  }, [appWindow]);

  return (
    <div className="win-controls">
      <button className="win-btn" title={t("minimize")} onClick={() => appWindow?.minimize()}>
        <svg width="15" height="15" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M2.5 6h7" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
      </button>
      <button
        className="win-btn"
        title={maximized ? t("restore") : t("maximize")}
        onClick={() => appWindow?.toggleMaximize()}
      >
        {maximized ? (
          <svg
            width="15"
            height="15"
            viewBox="0 0 12 12"
            aria-hidden="true"
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
          >
            <rect x="2.5" y="4" width="5.5" height="5.5" rx="1" />
            <path d="M4.5 4V3a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1" strokeLinecap="round" />
          </svg>
        ) : (
          <svg
            width="15"
            height="15"
            viewBox="0 0 12 12"
            aria-hidden="true"
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
          >
            <rect x="2.5" y="2.5" width="7" height="7" rx="1" />
          </svg>
        )}
      </button>
      <button className="win-btn close" title={t("close")} onClick={() => appWindow?.close()}>
        <svg width="15" height="15" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}
