import React from "react";
import ReactDOM from "react-dom/client";
// Anthropic-style reading serif for model output & suggestion cards (bundled offline).
import "@fontsource/source-serif-4/400.css";
import "@fontsource/source-serif-4/400-italic.css";
import "@fontsource/source-serif-4/600.css";
import "@fontsource/source-serif-4/700.css";
import { LangProvider } from "./lib/i18n";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ConfirmProvider } from "./components/ConfirmModal";
import { noteFrontendReady } from "./lib/ipc";

// One call per page load. The backend counts them: a second one without the app
// restarting is the webview having been reloaded under us, which is how an
// unattended code run disappears without leaving an error anywhere.
void noteFrontendReady().catch(() => {});

// Don't let async failures vanish silently — at least surface them in the
// console/devtools. (Nothing leaves the machine.)
window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled promise rejection:", e.reason);
});
window.addEventListener("error", (e) => {
  console.error("Uncaught error:", e.error ?? e.message);
});

async function bootstrap() {
  // Plain-browser dev (vite, no Tauri webview): install the IPC mock so every
  // surface renders with rich fixtures for visual work. Gated by an explicit
  // env flag because `vite build` pins DEV=false even in development mode;
  // production builds never set the flag, so this stays out of releases.
  if (
    (import.meta.env.DEV || import.meta.env.VITE_UI_PREVIEW === "1") &&
    !("__TAURI_INTERNALS__" in window)
  ) {
    const { installDevMock } = await import("./lib/devMock");
    installDevMock();
  }
  // Deferred so App's module-level platform detection runs after the mock
  // (a static import would evaluate it before bootstrap).
  const { default: App } = await import("./App");
  ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
      <ErrorBoundary>
        <LangProvider>
          <ConfirmProvider>
            <App />
          </ConfirmProvider>
        </LangProvider>
      </ErrorBoundary>
    </React.StrictMode>,
  );
}
void bootstrap();
