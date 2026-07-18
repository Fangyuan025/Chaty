// Visual/e2e harness for the rewritten Canvas: split preview|code, element↔
// line correspondence (Inspect), and the diff view on version iterations.
// Served by plain vite at /canvas-harness.html — no Tauri needed.
import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import "../App.css";
import { LangProvider } from "../lib/i18n";
import { ConfirmProvider } from "../components/ConfirmModal";
import { CanvasPanel, type CanvasVersion } from "../components/CanvasPanel";
import { applyCodeTheme } from "../lib/codeTheme";

const V1 = `<!doctype html>
<html>
<head>
<style>
  body { font-family: sans-serif; background: #f6f4ef; margin: 0; padding: 32px; }
  .card { background: #fff; border-radius: 12px; padding: 24px; max-width: 420px; box-shadow: 0 4px 16px rgba(0,0,0,.08); }
  h1 { margin: 0 0 8px; font-size: 22px; color: #1f2937; }
  p { margin: 0 0 16px; color: #6b7280; }
  button { background: #357557; color: #fff; border: none; border-radius: 8px; padding: 10px 18px; cursor: pointer; }
</style>
</head>
<body>
  <div class="card">
    <h1>Hello Canvas</h1>
    <p>A tiny demo card for the inspect test.</p>
    <button id="cta" onclick="this.textContent='Clicked!'">Click me</button>
  </div>
  <script>
    // APIs that work on a real page but used to throw in the sandboxed frame:
    history.replaceState({}, "", "#home");
    document.cookie = "demo=1";
    if (navigator.clipboard) navigator.clipboard.writeText("hi");
    console.log("hello from canvas", { n: 42 });
    console.warn("demo warning");
  </script>
</body>
</html>`;

const V2 = V1.replace("Hello Canvas", "Hello Canvas v2")
  .replace("#357557", "#7c3aed")
  .replace(
    `<p>A tiny demo card for the inspect test.</p>`,
    `<p>A tiny demo card for the inspect test.</p>\n    <p class="sub">Second paragraph added in v2.</p>`,
  );

// A patch stream like the model emits: prose → SEARCH copy → REPLACE growth.
const PATCH_STREAM = [
  "好的,我来修改。\n",
  "<<<<<<< SEARCH\n  <h1>Hello Canvas</h1>\n",
  "=======\n",
  "  <h1>Hello ",
  "Canvas — scanned!</h1>\n",
  ">>>>>>> REPLACE\n",
  "<<<<<<< SEARCH\n  <button id=\"cta\" onclick=\"this.textContent='Clicked!'\">Click me</button>\n",
  "=======\n  <button id=\"cta\" ",
  "style=\"background:#e11d48\" ",
  "onclick=\"this.textContent='Clicked!'\">Click me</button>\n>>>>>>> REPLACE\n",
].join("");

function Harness() {
  const [versions, setVersions] = useState<CanvasVersion[]>([{ html: V1, note: "初始" }]);
  const [index, setIndex] = useState(0);
  const [open, setOpen] = useState(true);
  const [stream, setStream] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Deterministic stepper — browser-pane tabs throttle timers, so the sim is
  // driven from outside: __simTo(chars) renders the stream at that offset,
  // __simFinish() lands the version like the real completion path.
  const startScanSim = () => {
    setBusy(true);
    setStream("");
  };
  (window as unknown as Record<string, unknown>).__simTo = (chars: number) => {
    setStream(PATCH_STREAM.slice(0, Math.min(PATCH_STREAM.length, chars)));
  };
  (window as unknown as Record<string, unknown>).__simLen = PATCH_STREAM.length;
  (window as unknown as Record<string, unknown>).__simFinish = () => {
    setStream(null);
    setBusy(false);
    setVersions((vs) => [
      ...vs,
      {
        html: V1.replace("Hello Canvas", "Hello Canvas — scanned!").replace(
          `<button id="cta" `,
          `<button id="cta" style="background:#e11d48" `,
        ),
        note: "修改:扫描模拟",
      },
    ]);
    setIndex(1);
  };

  return (
    <>
      <div id="controls" style={{ position: "fixed", top: 4, left: 4, zIndex: 99999, display: "flex", gap: 6 }}>
        <button id="add-v2" onClick={() => { setVersions([...versions, { html: V2, note: "修改:主题色+副段落" }]); setIndex(versions.length); }}>
          模拟迭代→v2
        </button>
        <button id="reopen" onClick={() => setOpen(true)}>重开</button>
        <button id="scan-sim" onClick={startScanSim}>模拟扫描</button>
        <button
          id="toggle-appearance"
          onClick={() => {
            const light = document.documentElement.dataset.theme !== "light";
            document.documentElement.dataset.theme = light ? "light" : "dark";
            applyCodeTheme("github-dark", light);
          }}
        >
          明暗切换
        </button>
      </div>
      <CanvasPanel
        open={open}
        versions={versions}
        index={index}
        busy={busy}
        streamText={stream}
        onSelectVersion={setIndex}
        onReset={() => { setVersions((vs) => (vs.length ? [vs[0]] : vs)); setIndex(0); }}
        onManualEdit={(html) => { setVersions((vs) => [...vs, { html, note: "手动编辑" }]); setIndex(versions.length); }}
        onIterate={(ins) => { setVersions([...versions, { html: V2, note: `修改:${ins}` }]); setIndex(versions.length); }}
        onFix={() => {}}
        onExport={() => {}}
        onOpenExternal={() => {}}
        onClose={() => setOpen(false)}
      />
    </>
  );
}

document.documentElement.dataset.theme = "dark";
applyCodeTheme("github-dark", false);
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LangProvider>
      <ConfirmProvider>
        <Harness />
      </ConfirmProvider>
    </LangProvider>
  </React.StrictMode>,
);
