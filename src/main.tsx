import React from "react";
import ReactDOM from "react-dom/client";
// Anthropic-style reading serif for model output & suggestion cards (bundled offline).
import "@fontsource/source-serif-4/400.css";
import "@fontsource/source-serif-4/400-italic.css";
import "@fontsource/source-serif-4/600.css";
import "@fontsource/source-serif-4/700.css";
import App from "./App";
import { LangProvider } from "./lib/i18n";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ConfirmProvider } from "./components/ConfirmModal";

// Don't let async failures vanish silently — at least surface them in the
// console/devtools. (Nothing leaves the machine.)
window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled promise rejection:", e.reason);
});
window.addEventListener("error", (e) => {
  console.error("Uncaught error:", e.error ?? e.message);
});

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
