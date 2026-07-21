// Visual/e2e harness for the rewritten Canvas: split preview|code, element↔
// line correspondence (Inspect), and the diff view on version iterations.
// Served by plain vite at /canvas-harness.html — no Tauri needed.
import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import "../App.css";
import { LangProvider } from "../lib/i18n";
import { ConfirmProvider } from "../components/ConfirmModal";
import { CanvasPanel, type CanvasVersion } from "../components/CanvasPanel";
import { applyCodeTheme } from "../lib/codeTheme";

// A page the model could plausibly have generated — the same demo drives the
// interactive harness and the release screenshots, so it has to look real.
const V1 = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Aurora Roasters</title>
<style>
  :root {
    --bg: #faf6ef;
    --ink: #2b2119;
    --muted: #8a7a68;
    --card: #ffffff;
    --accent: #b4552d;
    --accent-ink: #fff7f0;
    --line: rgba(43, 33, 25, 0.1);
  }
  * { box-sizing: border-box; margin: 0; }
  body { font-family: "Avenir Next", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
  nav {
    position: sticky; top: 0;
    display: flex; align-items: center; gap: 22px;
    padding: 16px 36px;
    background: color-mix(in srgb, var(--bg) 90%, transparent);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
  }
  .logo { font-weight: 700; letter-spacing: 0.04em; }
  .logo::before { content: "●"; color: var(--accent); margin-right: 8px; }
  nav a { color: var(--muted); text-decoration: none; font-size: 14px; }
  nav a:hover { color: var(--ink); }
  .spacer { flex: 1; }
  .pill { background: var(--accent); color: var(--accent-ink); padding: 7px 14px; border-radius: 999px; font-size: 13px; }
  header { padding: 76px 36px 56px; max-width: 680px; }
  .eyebrow { color: var(--accent); font-size: 13px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; }
  h1 { font-family: Georgia, "Times New Roman", serif; font-size: 46px; line-height: 1.1; margin: 14px 0 16px; }
  .sub { color: var(--muted); font-size: 16px; line-height: 1.65; max-width: 52ch; }
  .cta-row { display: flex; gap: 12px; margin-top: 26px; }
  .btn { padding: 11px 20px; border-radius: 10px; font-size: 14px; border: 1px solid transparent; cursor: pointer; }
  .btn.primary { background: var(--ink); color: var(--bg); }
  .btn.ghost { border-color: var(--line); background: transparent; color: var(--ink); }
  .menu { padding: 0 36px 60px; }
  .menu h2 { font-family: Georgia, serif; font-size: 26px; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .card { background: var(--card); border-radius: 14px; padding: 20px; box-shadow: 0 2px 12px rgba(20, 12, 6, 0.07); }
  .card .icon { font-size: 24px; }
  .card h3 { margin: 10px 0 6px; font-size: 16px; }
  .card p { color: var(--muted); font-size: 13.5px; line-height: 1.5; }
  .card .price { display: block; margin-top: 12px; font-weight: 700; color: var(--accent); }
  footer { padding: 26px 36px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; display: flex; gap: 22px; }
</style>
</head>
<body>
  <nav>
    <span class="logo">Aurora Roasters</span>
    <a href="#menu">Menu</a>
    <a href="#visit">Visit</a>
    <span class="spacer"></span>
    <span class="pill">Open · until 6 pm</span>
  </nav>
  <header>
    <p class="eyebrow">Small-batch coffee, roasted on site</p>
    <h1>Mornings taste better under the aurora.</h1>
    <p class="sub">We roast twelve kilos at a time, pull shots on a lever machine, and bake the cardamom buns before sunrise. Come early — the window seats go first.</p>
    <div class="cta-row">
      <button class="btn primary">See the menu</button>
      <button class="btn ghost">Find the shop</button>
    </div>
  </header>
  <section class="menu" id="menu">
    <h2>On the bar this week</h2>
    <div class="grid">
      <div class="card"><span class="icon">☕️</span><h3>Polar Espresso</h3><p>Honey-processed Huila, pulled ristretto. Cherry &amp; cocoa.</p><span class="price">$4</span></div>
      <div class="card"><span class="icon">🫐</span><h3>Blueberry Cortado</h3><p>Single-origin Sidamo under silky oat milk.</p><span class="price">$5.5</span></div>
      <div class="card"><span class="icon">🥐</span><h3>Cardamom Bun</h3><p>Laminated overnight, glazed while warm.</p><span class="price">$4.5</span></div>
    </div>
  </section>
  <footer id="visit">
    <span>14 Harbor Lane, Tromsø</span>
    <span>Mon–Sun · 7:30–18:00</span>
    <span>hello@auroraroasters.no</span>
  </footer>
</body>
</html>`;

// The "evening" iteration the demo model produces: palette swap, later hours,
// reworked hero copy, and a testimonials section before the footer.
const V2 = V1.replace(
  `    --bg: #faf6ef;
    --ink: #2b2119;
    --muted: #8a7a68;
    --card: #ffffff;
    --accent: #b4552d;
    --accent-ink: #fff7f0;
    --line: rgba(43, 33, 25, 0.1);`,
  `    color-scheme: dark;
    --bg: #171210;
    --ink: #f3ece2;
    --muted: #a3927f;
    --card: #211a15;
    --accent: #e0a458;
    --accent-ink: #171210;
    --line: rgba(243, 236, 226, 0.12);`,
)
  .replace("Open · until 6 pm", "Open late · until 11 pm")
  .replace(
    "Mornings taste better under the aurora.</h1>",
    "Evenings glow longer under the aurora.</h1>",
  )
  .replace(
    "Come early — the window seats go first.",
    "Come late — the lamps come on at six.",
  )
  .replace(
    "  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }",
    "  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }\n  .grid.two { grid-template-columns: repeat(2, 1fr); }",
  )
  .replace(
    "  <footer id=\"visit\">",
    `  <section class="menu" id="voices">
    <h2>What the regulars say</h2>
    <div class="grid two">
      <div class="card"><p>“The cortado converted me. I don't even like oat milk.”</p><span class="price">— Mari, every morning</span></div>
      <div class="card"><p>“Half my thesis was written at the window seat.”</p><span class="price">— Jonas, table four</span></div>
    </div>
  </section>
  <footer id="visit">`,
  );

const NOTE_V2 = "Edit：an evening look + what the regulars say";

// A patch stream like the model emits: prose → SEARCH copy → REPLACE growth.
// The hero copy lands first, then the palette — the freeze point for the
// "scan" screenshot sits mid-REPLACE of the second block.
const PATCH_STREAM = [
  "Shifting the page into an evening mood: warmer copy first, then the palette, and a small section of regulars' quotes before the footer.\n\n",
  "<<<<<<< SEARCH\n    <h1>Mornings taste better under the aurora.</h1>\n",
  "=======\n    <h1>Evenings glow longer",
  " under the aurora.</h1>\n>>>>>>> REPLACE\n",
  "<<<<<<< SEARCH\n    <p class=\"sub\">We roast twelve kilos at a time, pull shots on a lever machine, and bake the cardamom buns before sunrise. Come early — the window seats go first.</p>\n",
  "=======\n    <p class=\"sub\">We roast twelve kilos at a time, pull shots on a lever machine, and bake the cardamom buns before sunrise. Come late — the lamps",
  " come on at six.</p>\n>>>>>>> REPLACE\n",
  "<<<<<<< SEARCH\n    --bg: #faf6ef;\n    --ink: #2b2119;\n",
  "=======\n    --bg: #171210;\n    --ink: #f3ece2;\n>>>>>>> REPLACE\n",
].join("");
const SCAN_FREEZE = PATCH_STREAM.indexOf("Come late — the lamps") + "Come late — the lamps".length;

// Screenshot automation: ?shot=hero|diff|scan&theme=light&chrome=hide boots a
// deterministic state (used by the release screenshot pipeline). Shots force
// English UI and the fullscreen layout so captures match the in-app look.
const params = new URLSearchParams(location.search);
const shotMode = params.get("shot");
if (shotMode) {
  localStorage.setItem("chaty.lang", params.get("lang") ?? "en");
  localStorage.setItem("chaty.canvasLayout", JSON.stringify({ railW: 200, codePct: 46, full: true }));
}

function Harness() {
  const [versions, setVersions] = useState<CanvasVersion[]>(() =>
    shotMode === "hero" || shotMode === "diff"
      ? [{ html: V1, note: "Initial" }, { html: V2, note: NOTE_V2 }]
      : [{ html: V1, note: shotMode ? "Initial" : "初始" }],
  );
  const [index, setIndex] = useState(shotMode === "hero" || shotMode === "diff" ? 1 : 0);
  const [open, setOpen] = useState(true);
  const [stream, setStream] = useState<string | null>(() =>
    shotMode === "scan" ? PATCH_STREAM.slice(0, SCAN_FREEZE) : null,
  );
  const [busy, setBusy] = useState(shotMode === "scan");

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
    setVersions((vs) => [...vs, { html: V2, note: shotMode ? NOTE_V2 : "修改:扫描模拟" }]);
    setIndex(1);
  };

  // The diff shot mounts with v2 already landed (a second srcdoc load never
  // paints under headless Chrome's virtual time) and then takes the user
  // path into the diff view: clicking the Changes tab.
  const bootShot = useRef(false);
  useEffect(() => {
    if (bootShot.current) return;
    bootShot.current = true;
    if (shotMode === "diff") {
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          const tab = [...document.querySelectorAll("button")].find(
            (b) => b.textContent?.trim() === "Changes" || b.textContent?.trim() === "变更",
          );
          tab?.click();
        }),
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <div id="controls" style={{ position: "fixed", top: 4, left: 4, zIndex: 99999, display: "flex", gap: 6 }}>
        <button id="add-v2" onClick={() => { setVersions([...versions, { html: V2, note: "修改:黄昏主题+口碑区" }]); setIndex(versions.length); }}>
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
        onStop={() => { setBusy(false); setStream(null); }}
        onExport={() => {}}
        onOpenExternal={() => {}}
        onClose={() => setOpen(false)}
      />
    </>
  );
}

const bootLight = params.get("theme") === "light";
document.documentElement.dataset.theme = bootLight ? "light" : "dark";
applyCodeTheme("github-dark", bootLight);
if (params.get("chrome") === "hide") {
  const style = document.createElement("style");
  style.textContent = "#controls { display: none !important; }";
  document.head.appendChild(style);
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LangProvider>
      <ConfirmProvider>
        <Harness />
      </ConfirmProvider>
    </LangProvider>
  </React.StrictMode>,
);
