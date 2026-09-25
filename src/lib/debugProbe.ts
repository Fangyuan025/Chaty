/** Development builds only (the commands answer nothing in a release): send
 *  the message `CHATY_AUTORUN` names once the page is up, and measure how the
 *  page keeps up while the reply streams — how late the main thread runs, how
 *  far apart painted frames are, and whether a code block goes missing or is
 *  rebuilt from scratch. Written to `CHATY_REPORT` for the harness to read.
 *
 *  Exists because the webview is where rendering problems live: the browser
 *  pane is another engine, and a background window takes no clicks. */
import { invoke } from "@tauri-apps/api/core";

export async function maybeAutorun(send: (text: string) => void, toChat: () => void): Promise<void> {
  let text: string | null = null;
  try {
    text = await invoke<string | null>("debug_autorun");
  } catch {
    return;
  }
  if (!text) return;
  toChat();
  await new Promise((r) => setTimeout(r, 800));
  const secs = 40;
  const lags: number[] = [];
  const frames: number[] = [];
  let vanished = 0;
  let rebuilt = 0;
  let collapsed = 0;
  let seen = 0;
  let lastPre: Element | null = null;
  let lastTick = performance.now();
  const tick = setInterval(() => {
    const now = performance.now();
    lags.push(now - lastTick - 16);
    lastTick = now;
  }, 16);
  let lastFrame = performance.now();
  let running = true;
  const frame = (t: number) => {
    frames.push(t - lastFrame);
    lastFrame = t;
    const pres = document.querySelectorAll(".msg.assistant .code-block pre");
    if (pres.length > 0) seen = Math.max(seen, pres.length);
    else if (seen > 0) vanished++;
    const pre = pres[pres.length - 1] ?? null;
    if (pre && lastPre && pre !== lastPre && pres.length === seen) rebuilt++;
    if (pre && pre.getBoundingClientRect().height < 4) collapsed++;
    lastPre = pre;
    if (running) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  send(text);
  await new Promise((r) => setTimeout(r, secs * 1000));
  running = false;
  clearInterval(tick);
  const pct = (a: number[], p: number) => {
    const s = [...a].sort((x, y) => x - y);
    return Math.round(s[Math.min(s.length - 1, Math.floor(s.length * p))] ?? 0);
  };
  const report = {
    frames: frames.length,
    frameP50: pct(frames, 0.5),
    frameP95: pct(frames, 0.95),
    frameMax: Math.round(Math.max(0, ...frames)),
    framesOver50: frames.filter((f) => f > 50).length,
    lagP95: pct(lags, 0.95),
    lagMax: Math.round(Math.max(0, ...lags)),
    codeBlockVanished: vanished,
    codeBlockRebuilt: rebuilt,
    codeBlockCollapsed: collapsed,
    textLength: document.querySelector(".msg.assistant:last-child .answer")?.textContent?.length ?? 0,
  };
  await invoke("debug_report", { text: JSON.stringify(report, null, 2) }).catch(() => {});
}
