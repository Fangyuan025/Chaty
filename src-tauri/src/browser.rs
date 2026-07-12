//! Browser automation for Code mode, over the Chrome DevTools Protocol (CDP).
//!
//! We spawn the user's installed Chrome headless with a fresh profile and a
//! remote-debugging port, then drive one page target over a raw CDP WebSocket
//! (sync `tungstenite`, no async runtime — the whole browser lives on one
//! dedicated actor thread, mirroring the inference worker's design). The agent
//! gets: navigate, screenshot (→ the vision pipeline), read the JS console /
//! exceptions (the client-side "backend errors" pipeline), click, type, and a
//! general `eval` escape hatch — enough to drive and visually verify a web app,
//! or automate a browsing task for the user.
//!
//! No heavyweight browser crate: CDP is line-oriented JSON over a localhost
//! WebSocket, so `tungstenite` + `serde_json` is all it takes.

use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};

type Ws = WebSocket<MaybeTlsStream<TcpStream>>;

/// One request to the browser actor.
enum BrowserCmd {
    Navigate { url: String, reply: Sender<Result<String, String>> },
    /// Full-page screenshot (auto-scrolls first to trigger lazy content).
    Screenshot { reply: Sender<Result<Vec<u8>, String>> },
    /// Snapshot of just the CURRENT viewport (immediate; for lazy-load pages
    /// the model scrolls then snapshots what's now visible).
    Snapshot { reply: Sender<Result<Vec<u8>, String>> },
    Scroll { to: Option<String>, by: Option<f64>, reply: Sender<Result<String, String>> },
    Eval { expr: String, reply: Sender<Result<String, String>> },
    Click { selector: Option<String>, text: Option<String>, reply: Sender<Result<String, String>> },
    Type { selector: Option<String>, label: Option<String>, text: String, reply: Sender<Result<String, String>> },
    Console { reply: Sender<Result<String, String>> },
    Read { reply: Sender<Result<String, String>> },
    Close,
}

/// JS that returns a compact list of the page's interactive elements, so the
/// model clicks/types against real visible text rather than guessed selectors.
const PAGE_DIGEST_JS: &str = r#"(function(){
  function vis(e){var r=e.getBoundingClientRect();return r.width>1&&r.height>1;}
  var out=[];
  var nodes=document.querySelectorAll("a,button,[role=button],input,textarea,select,summary");
  for(var i=0;i<nodes.length&&out.length<70;i++){
    var e=nodes[i];if(!vis(e))continue;
    var tag=e.tagName.toLowerCase();
    var t=((e.innerText||e.value||e.getAttribute('aria-label')||e.placeholder||'')+'').trim().replace(/\s+/g,' ').slice(0,70);
    if(tag==='a'){ if(t) out.push('链接/link: "'+t+'"'); }
    else if(tag==='button'||e.getAttribute('role')==='button'||e.type==='submit'||e.type==='button'){ if(t) out.push('按钮/button: "'+t+'"'); }
    else if(tag==='input'||tag==='textarea'){ var h=e.placeholder||e.name||e.getAttribute('aria-label')||e.type||'text'; out.push('输入框/input ['+(e.type||'text')+']: '+h); }
    else if(tag==='select'){ out.push('下拉/select: '+(e.name||e.id||'')); }
  }
  return out.length? out.join("\n") : "(未发现明显的可交互元素 / no obvious interactive elements)";
})()"#;

/// Auto-scroll through the whole page (triggering lazy-loaded content) and
/// return to the top — run before a full-page screenshot. Resolves a promise so
/// `Runtime.evaluate` with awaitPromise waits for it.
const AUTOSCROLL_JS: &str = r#"new Promise(function(done){
  var y=0,h=document.body.scrollHeight,step=Math.max(200,window.innerHeight*0.9),ticks=0;
  var timer=setInterval(function(){
    window.scrollTo(0,y);y+=step;ticks++;
    if(y>=document.body.scrollHeight||ticks>40){clearInterval(timer);window.scrollTo(0,0);setTimeout(done,200);}
  },80);
})"#;

/// Process-wide handle to the browser actor thread. Lazily started.
static BROWSER: Mutex<Option<Sender<BrowserCmd>>> = Mutex::new(None);
/// Persistent profile dir for the interactive browser (set once at startup, so
/// the user's logins survive across runs). `None` → throwaway profile (tests).
static PROFILE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Point the interactive browser at a persistent profile directory so cookies /
/// logins persist. Called once from app setup with the app-data path.
pub fn set_profile_dir(dir: PathBuf) {
    *PROFILE_DIR.lock().unwrap() = Some(dir);
}

/// User preference: run the agent's interactive browser hidden (headless).
/// Settings → Code; applies the next time the browser starts.
static HEADLESS_PREF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_headless(on: bool) {
    HEADLESS_PREF.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// PID of the launched Chrome, for a synchronous kill at app exit (the exit
/// handler `_exit()`s to dodge a ggml teardown crash, skipping destructors, so
/// the browser child would otherwise be orphaned).
static CHROME_PID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Kill the launched Chrome immediately by PID. Safe to call from the exit
/// handler (does not touch the actor thread or run destructors).
pub fn kill_now() {
    let pid = CHROME_PID.swap(0, std::sync::atomic::Ordering::SeqCst);
    if pid != 0 {
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .output();
        }
    }
}

/// Candidate Chrome/Chromium executables by platform.
fn chrome_path() -> Option<PathBuf> {
    let candidates: &[&str] = if cfg!(target_os = "macos") {
        &[
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    } else if cfg!(target_os = "windows") {
        &[
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
    } else {
        &["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]
    };
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
        // bare name on PATH (Linux)
        if !c.contains('/') && !c.contains('\\') {
            if let Ok(out) = std::process::Command::new("which").arg(c).output() {
                if out.status.success() {
                    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if !s.is_empty() {
                        return Some(PathBuf::from(s));
                    }
                }
            }
        }
    }
    None
}

/// Test-only: is a browser available? (chrome_path is private.)
#[cfg(test)]
pub fn chrome_path_pub() -> Option<PathBuf> { chrome_path() }

/// Ensure the actor is running; returns a sender to talk to it.
fn ensure() -> Result<Sender<BrowserCmd>, String> {
    let mut guard = BROWSER.lock().unwrap();
    if let Some(tx) = guard.as_ref() {
        return Ok(tx.clone());
    }
    let (tx, rx) = std::sync::mpsc::channel::<BrowserCmd>();
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::Builder::new()
        .name("chaty-browser".into())
        .spawn(move || actor(rx, init_tx))
        .map_err(|e| format!("无法启动浏览器线程 (failed to start browser thread): {e}"))?;
    match init_rx.recv() {
        Ok(Ok(())) => {
            *guard = Some(tx.clone());
            Ok(tx)
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err("浏览器线程初始化失败 (browser thread failed to init)".into()),
    }
}

/// Drop the actor (kills Chrome). Called on workspace switch / app teardown.
pub fn shutdown() {
    if let Some(tx) = BROWSER.lock().unwrap().take() {
        let _ = tx.send(BrowserCmd::Close);
    }
}

/// Does an error mean the CDP session/browser is gone (user closed the window,
/// Chrome quit)? Such commands are retried against a freshly relaunched browser.
fn is_dead_session(e: &str) -> bool {
    let e = e.to_lowercase();
    e.contains("session with given id not found")
        || e.contains("connection closed")
        || e.contains("cdp read")
        || e.contains("cdp send")
        || e.contains("timed out")
        || e.contains("target closed")
        || e.contains("no target")
        || e.contains("inspected target")
        || e.contains("cannot find context")
}

/// The Chrome-owning actor: launches the browser, attaches to one page, then
/// serves commands until told to close (or the channel drops). Auto-recovers by
/// relaunching if the user manually closes the window mid-session.
fn actor(rx: Receiver<BrowserCmd>, init: Sender<Result<(), String>>) {
    // Foreground (headful) by default so the user WATCHES the agent drive the
    // browser; tests / headless envs set CHATY_BROWSER_HEADLESS, and the user
    // can prefer a hidden browser in Settings → Code.
    let headless = std::env::var("CHATY_BROWSER_HEADLESS").is_ok()
        || HEADLESS_PREF.load(std::sync::atomic::Ordering::Relaxed);
    let mut session = match BrowserSession::launch(headless, true) {
        Ok(s) => s,
        Err(e) => {
            let _ = init.send(Err(e));
            return;
        }
    };
    let _ = init.send(Ok(()));

    // Run `f`; if it fails because the browser is gone (or the child has
    // exited), relaunch a fresh session and retry once.
    fn run<T>(
        session: &mut BrowserSession,
        headless: bool,
        mut f: impl FnMut(&mut BrowserSession) -> Result<T, String>,
    ) -> Result<T, String> {
        let dead = session.child.try_wait().map(|s| s.is_some()).unwrap_or(true);
        if !dead {
            match f(session) {
                Ok(v) => return Ok(v),
                Err(e) if !is_dead_session(&e) => return Err(e),
                Err(_) => {} // fall through to relaunch + retry
            }
        }
        session.kill();
        *session = BrowserSession::launch(headless, true)?;
        f(session)
    }

    while let Ok(cmd) = rx.recv() {
        match cmd {
            BrowserCmd::Navigate { url, reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.navigate(&url)));
            }
            BrowserCmd::Screenshot { reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.screenshot()));
            }
            BrowserCmd::Snapshot { reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.snapshot()));
            }
            BrowserCmd::Scroll { to, by, reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.scroll(to.as_deref(), by)));
            }
            BrowserCmd::Eval { expr, reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.eval(&expr)));
            }
            BrowserCmd::Click { selector, text, reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.click(selector.as_deref(), text.as_deref())));
            }
            BrowserCmd::Type { selector, label, text, reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.type_text(selector.as_deref(), &text, label.as_deref())));
            }
            BrowserCmd::Console { reply } => {
                let _ = reply.send(run(&mut session, headless, |s| Ok(s.drain_console())));
            }
            BrowserCmd::Read { reply } => {
                let _ = reply.send(run(&mut session, headless, |s| s.digest()));
            }
            BrowserCmd::Close => break,
        }
    }
    session.kill();
}

/// A launched Chrome + an attached page session over one CDP WebSocket.
struct BrowserSession {
    child: std::process::Child,
    ws: Ws,
    session_id: String,
    next_id: i64,
    /// Buffered console API calls + exceptions since the last `drain_console`.
    console: Vec<String>,
    /// Throwaway profile guard (deletes on drop); `None` for a persistent profile.
    _profile: Option<tempdir::Guard>,
}

impl BrowserSession {
    /// `track_pid`: register the child in CHROME_PID for exit-time cleanup (the
    /// shared interactive browser); one-shot headless captures pass false.
    fn launch(headless: bool, track_pid: bool) -> Result<Self, String> {
        let exe = chrome_path().ok_or(
            "未找到 Chrome/Chromium,请先安装 Chrome。(No Chrome/Chromium found — install Google Chrome.)",
        )?;
        // The interactive browser (track_pid) uses a PERSISTENT profile so the
        // user's logins survive across runs; one-shot captures use a throwaway.
        let persistent = if track_pid { PROFILE_DIR.lock().unwrap().clone() } else { None };
        let (profile_path, _guard) = match persistent {
            Some(dir) => {
                std::fs::create_dir_all(&dir).ok();
                (dir, None)
            }
            None => {
                let g = tempdir::Guard::new("chaty-cdp");
                (g.path().to_path_buf(), Some(g))
            }
        };
        // Chrome only writes DevToolsActivePort AFTER init; a stale one from a
        // previous run of a persistent profile would be read as the wrong port.
        let port_file = profile_path.join("DevToolsActivePort");
        let _ = std::fs::remove_file(&port_file);

        let mut cmd = std::process::Command::new(&exe);
        if headless {
            cmd.arg("--headless=new").arg("--hide-scrollbars");
        } else {
            // Bring the automation window to the front so it's clearly visible.
            cmd.arg("--new-window").arg("--start-maximized");
        }
        let mut child = cmd
            .arg("--remote-debugging-port=0")
            .arg(format!("--user-data-dir={}", profile_path.display()))
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            // Crisp captures (2× device pixels) for both the screenshot the model
            // reads and the preview the user opens.
            .arg("--force-device-scale-factor=2")
            .arg("--window-size=1280,900")
            .arg("--disable-background-networking")
            .arg("about:blank")
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("启动 Chrome 失败 (failed to launch Chrome): {e}"))?;
        if track_pid {
            CHROME_PID.store(child.id(), std::sync::atomic::Ordering::SeqCst);
        }

        // Chrome writes ws endpoint info to <profile>/DevToolsActivePort:
        // line 1 = port, line 2 = /devtools/browser/<uuid>.
        let deadline = Instant::now() + Duration::from_secs(15);
        let (port, ws_path) = loop {
            if Instant::now() > deadline {
                let _ = child.kill();
                return Err("Chrome 未在预期时间内就绪 (Chrome did not become ready in time)".into());
            }
            if let Ok(content) = std::fs::read_to_string(&port_file) {
                let mut lines = content.lines();
                if let (Some(p), Some(path)) = (lines.next(), lines.next()) {
                    if let Ok(port) = p.trim().parse::<u16>() {
                        break (port, path.trim().to_string());
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(60));
        };

        // Connect to the browser-level endpoint, open a page target, attach.
        let url = format!("ws://127.0.0.1:{port}{ws_path}");
        let (mut ws, _) = connect(&url)
            .map_err(|e| format!("连接 CDP 失败 (failed to connect CDP): {e}"))?;
        set_read_timeout(&ws, Duration::from_secs(30));

        let mut next_id = 1i64;
        // Create a real page target and attach with a flat session.
        let created = cdp_call(&mut ws, &mut next_id, None, "Target.createTarget", json!({"url":"about:blank"}))?;
        let target_id = created["targetId"].as_str().unwrap_or_default().to_string();
        let attached = cdp_call(
            &mut ws,
            &mut next_id,
            None,
            "Target.attachToTarget",
            json!({"targetId": target_id, "flatten": true}),
        )?;
        let session_id = attached["sessionId"].as_str().unwrap_or_default().to_string();
        if session_id.is_empty() {
            let _ = child.kill();
            return Err("CDP 会话附加失败 (failed to attach CDP session)".into());
        }

        let mut s = BrowserSession { child, ws, session_id, next_id, console: Vec::new(), _profile: _guard };
        // Enable the domains we consume. Runtime.enable surfaces console API
        // calls + uncaught exceptions; Log.enable surfaces browser log entries.
        let sid = s.session_id.clone();
        let _ = s.call(Some(&sid), "Page.enable", json!({}));
        let _ = s.call(Some(&sid), "Runtime.enable", json!({}));
        let _ = s.call(Some(&sid), "Log.enable", json!({}));
        Ok(s)
    }

    /// Send a CDP method to the page session and pump events until its reply.
    fn call(&mut self, session: Option<&str>, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let mut msg = json!({"id": id, "method": method, "params": params});
        if let Some(sid) = session {
            msg["sessionId"] = json!(sid);
        }
        self.ws
            .send(Message::Text(msg.to_string().into()))
            .map_err(|e| format!("CDP send failed: {e}"))?;
        self.pump_until(id)
    }

    /// Read frames until the response with `id` arrives, buffering events.
    fn pump_until(&mut self, id: i64) -> Result<Value, String> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if Instant::now() > deadline {
                return Err("CDP 响应超时 (CDP response timed out)".into());
            }
            let frame = match self.ws.read() {
                Ok(Message::Text(t)) => t.to_string(),
                Ok(Message::Binary(_)) | Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
                Ok(Message::Close(_)) => return Err("CDP 连接已关闭 (CDP connection closed)".into()),
                Ok(Message::Frame(_)) => continue,
                Err(tungstenite::Error::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    return Err("CDP 读取超时 (CDP read timed out)".into());
                }
                Err(e) => return Err(format!("CDP read failed: {e}")),
            };
            let v: Value = match serde_json::from_str(&frame) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v.get("id").and_then(|x| x.as_i64()) == Some(id) {
                if let Some(err) = v.get("error") {
                    return Err(format!("CDP error: {}", err["message"].as_str().unwrap_or("unknown")));
                }
                return Ok(v.get("result").cloned().unwrap_or(Value::Null));
            }
            // Not our reply — record console-worthy events.
            self.record_event(&v);
        }
    }

    /// Collect console API calls, uncaught exceptions and log entries.
    fn record_event(&mut self, v: &Value) {
        let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let p = v.get("params").cloned().unwrap_or(Value::Null);
        match method {
            "Runtime.consoleAPICalled" => {
                let level = p["type"].as_str().unwrap_or("log");
                let args: Vec<String> = p["args"]
                    .as_array()
                    .map(|a| a.iter().map(remote_object_to_string).collect())
                    .unwrap_or_default();
                self.push_console(format!("[{level}] {}", args.join(" ")));
            }
            "Runtime.exceptionThrown" => {
                let d = &p["exceptionDetails"];
                let text = d["exception"]["description"]
                    .as_str()
                    .or_else(|| d["text"].as_str())
                    .unwrap_or("Uncaught exception");
                self.push_console(format!("[exception] {text}"));
            }
            "Log.entryAdded" => {
                let e = &p["entry"];
                let level = e["level"].as_str().unwrap_or("info");
                if matches!(level, "error" | "warning") {
                    self.push_console(format!("[{level}] {}", e["text"].as_str().unwrap_or("")));
                }
            }
            _ => {}
        }
    }

    fn push_console(&mut self, line: String) {
        if self.console.len() < 200 {
            self.console.push(line);
        }
    }

    fn drain_console(&mut self) -> String {
        // Also pump any pending frames (non-blocking-ish) so freshly-logged
        // messages are included even without an intervening command.
        self.pump_pending();
        if self.console.is_empty() {
            return "（控制台无输出 / console is empty）".into();
        }
        let out = self.console.join("\n");
        self.console.clear();
        out
    }

    /// Drain frames already waiting on the socket (short read timeout).
    fn pump_pending(&mut self) {
        set_read_timeout(&self.ws, Duration::from_millis(120));
        for _ in 0..500 {
            match self.ws.read() {
                Ok(Message::Text(t)) => {
                    if let Ok(v) = serde_json::from_str::<Value>(&t) {
                        self.record_event(&v);
                    }
                }
                _ => break,
            }
        }
        set_read_timeout(&self.ws, Duration::from_secs(30));
    }

    fn navigate(&mut self, url: &str) -> Result<String, String> {
        let sid = self.session_id.clone();
        let r = self.call(Some(&sid), "Page.navigate", json!({"url": url}))?;
        if let Some(err) = r.get("errorText").and_then(|e| e.as_str()) {
            if !err.is_empty() {
                return Err(format!("导航失败 (navigation failed): {err}"));
            }
        }
        // Give the page a moment to load + settle (SPA JS, first paint).
        std::thread::sleep(Duration::from_millis(1200));
        self.pump_pending();
        let title = self.eval("document.title").unwrap_or_default().trim_matches('"').to_string();
        let final_url = self.eval("location.href").unwrap_or_default().trim_matches('"').to_string();
        let digest = self.eval(PAGE_DIGEST_JS).unwrap_or_default();
        Ok(format!(
            "已打开 (loaded): {final_url}\n标题 (title): {title}\n\n页面上可交互的元素 (interactive elements — click by text / type into these):\n{digest}"
        ))
    }

    /// A compact digest of the page's interactive elements (used after navigate
    /// and by the `browser_read` tool). Keeps the model grounded in what's
    /// actually there, so it clicks by real visible text instead of guessing
    /// selectors.
    fn digest(&mut self) -> Result<String, String> {
        self.eval(PAGE_DIGEST_JS)
    }

    /// Full-page screenshot. First auto-scrolls through the page (triggering
    /// lazy-loaded images/sections), then returns to the top and captures the
    /// whole document — so nothing below the fold is missed or blank.
    fn screenshot(&mut self) -> Result<Vec<u8>, String> {
        let _ = self.eval(AUTOSCROLL_JS); // best-effort; ignore if it errors
        std::thread::sleep(Duration::from_millis(300));
        self.capture(true)
    }

    /// Snapshot of just the current viewport (no scrolling) — pairs with
    /// `scroll` for lazy-loaded pages the model wants to walk section by section.
    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        self.capture(false)
    }

    fn capture(&mut self, beyond_viewport: bool) -> Result<Vec<u8>, String> {
        let sid = self.session_id.clone();
        let r = self.call(
            Some(&sid),
            "Page.captureScreenshot",
            json!({"format": "png", "captureBeyondViewport": beyond_viewport}),
        )?;
        let b64 = r["data"].as_str().ok_or("截图无数据 (no screenshot data)")?;
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("截图解码失败 (screenshot decode failed): {e}"))
    }

    /// Scroll the page: to "bottom"/"top", or by `by` pixels (default one
    /// viewport). Triggers lazy loading; returns the new scroll position.
    fn scroll(&mut self, to: Option<&str>, by: Option<f64>) -> Result<String, String> {
        let js = match to {
            Some("bottom") => "window.scrollTo(0, document.body.scrollHeight)".to_string(),
            Some("top") => "window.scrollTo(0, 0)".to_string(),
            _ => format!("window.scrollBy(0, {})", by.unwrap_or(0.0).max(1.0).max(0.0)),
        };
        // `by` defaults to one viewport when neither bottom/top nor an explicit
        // amount is given.
        let js = if to.is_none() && by.is_none() {
            "window.scrollBy(0, window.innerHeight*0.9)".to_string()
        } else {
            js
        };
        // Also dispatch a scroll event — some lazy-load listeners don't fire on
        // programmatic scrollTo in headless Chrome.
        self.eval(&format!("{js};window.dispatchEvent(new Event('scroll'))"))?;
        std::thread::sleep(Duration::from_millis(700)); // let lazy content load
        self.pump_pending();
        let pos = self
            .eval("Math.round(window.scrollY)+' / '+Math.round(document.body.scrollHeight)")
            .unwrap_or_default();
        // Surface any newly-revealed elements (lazy-load) so the model knows
        // whether scrolling accomplished anything and what's now on screen.
        let d = self.digest().unwrap_or_default();
        Ok(format!(
            "已滚动 (scrolled), 位置 scrollY: {}\n当前可交互元素:\n{d}\n(要看这一屏的视觉效果用 browser_snapshot)",
            pos.trim_matches('"')
        ))
    }

    fn eval(&mut self, expr: &str) -> Result<String, String> {
        let sid = self.session_id.clone();
        let r = self.call(
            Some(&sid),
            "Runtime.evaluate",
            json!({"expression": expr, "returnByValue": true, "awaitPromise": true}),
        )?;
        if let Some(exc) = r.get("exceptionDetails") {
            let text = exc["exception"]["description"]
                .as_str()
                .or_else(|| exc["text"].as_str())
                .unwrap_or("evaluation error");
            return Err(format!("JS 报错 (JS error): {text}"));
        }
        Ok(remote_object_to_string(&r["result"]))
    }

    /// Click by visible text (preferred — robust) or by CSS selector.
    fn click(&mut self, selector: Option<&str>, text: Option<&str>) -> Result<String, String> {
        let js = if let Some(txt) = text.filter(|t| !t.is_empty()) {
            format!(
                r#"(function(){{
                    var t={txt}.trim().toLowerCase();
                    var els=[].slice.call(document.querySelectorAll("a,button,[role=button],input[type=submit],input[type=button],[onclick],summary,label"));
                    var hit=els.find(function(e){{var s=((e.innerText||e.textContent||e.value||e.getAttribute('aria-label')||'')).trim().toLowerCase();return s===t||s.includes(t);}});
                    if(!hit)return 'NOT_FOUND';
                    hit.scrollIntoView({{block:'center'}});hit.click();return 'OK';
                }})()"#,
                txt = json!(txt)
            )
        } else {
            let sel = selector.unwrap_or("");
            format!(
                "(function(){{var el=document.querySelector({sel});if(!el)return 'NOT_FOUND';el.scrollIntoView({{block:'center'}});el.click();return 'OK';}})()",
                sel = json!(sel)
            )
        };
        let label = text.filter(|t| !t.is_empty()).unwrap_or_else(|| selector.unwrap_or(""));
        if label.is_empty() {
            let d = self.digest().unwrap_or_default();
            return Err(format!(
                "browser_click 需要 \"text\"(优先,按可见文字)或 \"selector\"。当前页面可点击的元素:\n{d}"
            ));
        }
        std::thread::sleep(Duration::from_millis(150)); // let a prior nav settle
        match self.eval(&js)?.trim_matches('"') {
            "OK" => {
                // A click may navigate — let it settle before the next step.
                std::thread::sleep(Duration::from_millis(700));
                self.pump_pending();
                // Hand back the post-click page state so the model verifies the
                // result instead of guessing what the click did. Include title +
                // url so a navigation is obvious.
                let where_ = self
                    .eval("document.title+' — '+location.href")
                    .unwrap_or_default();
                let d = self.digest().unwrap_or_default();
                Ok(format!(
                    "已点击 (clicked): {label}\n点击后当前页面:{where_}\n可交互元素:\n{d}\n(若要确认视觉渲染,再用 browser_snapshot)",
                    where_ = where_.trim_matches('"')
                ))
            }
            _ => {
                let d = self.digest().unwrap_or_default();
                Err(format!(
                    "未找到可点击的元素 (no match): {label}。用下面清单里的准确文字重试:\n{d}"
                ))
            }
        }
    }

    /// Type into a field matched by CSS selector OR by its label/placeholder text.
    fn type_text(&mut self, selector: Option<&str>, text: &str, label: Option<&str>) -> Result<String, String> {
        let finder = if let Some(sel) = selector.filter(|s| !s.is_empty()) {
            format!("document.querySelector({})", json!(sel))
        } else if let Some(lbl) = label.filter(|l| !l.is_empty()) {
            format!(
                r#"(function(){{
                    var l={lbl}.trim().toLowerCase();
                    var fields=[].slice.call(document.querySelectorAll("input,textarea,select"));
                    return fields.find(function(f){{
                        var hints=[f.placeholder,f.name,f.id,f.getAttribute('aria-label')];
                        var lab=f.labels&&f.labels[0];if(lab)hints.push(lab.textContent);
                        return hints.some(function(h){{return h&&h.trim().toLowerCase().includes(l);}});
                    }})||null;
                }})()"#,
                lbl = json!(lbl)
            )
        } else {
            "null".to_string()
        };
        let js = format!(
            r#"(function(){{
                var el={finder};
                if(!el)return 'NOT_FOUND';
                el.scrollIntoView({{block:'center'}});el.focus();
                el.value={val};
                el.dispatchEvent(new Event('input',{{bubbles:true}}));
                el.dispatchEvent(new Event('change',{{bubbles:true}}));
                return 'OK';
            }})()"#,
            val = json!(text)
        );
        let what = selector.or(label).unwrap_or("");
        if what.is_empty() && selector.is_none() && label.is_none() {
            let d = self.digest().unwrap_or_default();
            return Err(format!(
                "browser_type 需要 \"label\"(输入框的占位符/字段名)或 \"selector\",以及 \"text\"。当前页面的输入框:\n{d}"
            ));
        }
        match self.eval(&js)?.trim_matches('"') {
            "OK" => Ok(format!("已输入 (typed) → {what}")),
            _ => {
                let d = self.digest().unwrap_or_default();
                Err(format!("未找到输入框 (no input matched): {what}。可用输入框见清单:\n{d}"))
            }
        }
    }

    fn kill(&mut self) {
        CHROME_PID
            .compare_exchange(
                self.child.id(),
                0,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .ok();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn set_read_timeout(ws: &Ws, d: Duration) {
    if let MaybeTlsStream::Plain(tcp) = ws.get_ref() {
        let _ = tcp.set_read_timeout(Some(d));
    }
}

/// Send a CDP method on a fresh WS (used during setup before we have a session).
fn cdp_call(
    ws: &mut Ws,
    next_id: &mut i64,
    session: Option<&str>,
    method: &str,
    params: Value,
) -> Result<Value, String> {
    let id = *next_id;
    *next_id += 1;
    let mut msg = json!({"id": id, "method": method, "params": params});
    if let Some(sid) = session {
        msg["sessionId"] = json!(sid);
    }
    ws.send(Message::Text(msg.to_string().into()))
        .map_err(|e| format!("CDP send failed: {e}"))?;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if Instant::now() > deadline {
            return Err("CDP setup timed out".into());
        }
        match ws.read() {
            Ok(Message::Text(t)) => {
                if let Ok(v) = serde_json::from_str::<Value>(&t) {
                    if v.get("id").and_then(|x| x.as_i64()) == Some(id) {
                        if let Some(err) = v.get("error") {
                            return Err(format!("CDP error: {}", err["message"].as_str().unwrap_or("?")));
                        }
                        return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                    }
                }
            }
            Ok(Message::Close(_)) => return Err("CDP closed during setup".into()),
            Ok(_) => continue,
            Err(e) => return Err(format!("CDP read failed: {e}")),
        }
    }
}

/// Best-effort human string for a CDP RemoteObject (console args / eval result).
fn remote_object_to_string(o: &Value) -> String {
    if let Some(v) = o.get("value") {
        return match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
    }
    if let Some(d) = o.get("description").and_then(|d| d.as_str()) {
        return d.to_string();
    }
    if let Some(u) = o.get("unserializableValue").and_then(|u| u.as_str()) {
        return u.to_string();
    }
    o.get("type").and_then(|t| t.as_str()).unwrap_or("undefined").to_string()
}

// ---- public API used by the agent tool commands ----

fn dispatch<T, F>(build: F) -> Result<T, String>
where
    F: FnOnce(Sender<Result<T, String>>) -> BrowserCmd,
{
    let tx = ensure()?;
    let (reply, rx) = std::sync::mpsc::channel();
    tx.send(build(reply)).map_err(|_| "浏览器已关闭 (browser closed)".to_string())?;
    rx.recv().map_err(|_| "浏览器无响应 (browser did not respond)".to_string())?
}

pub fn navigate(url: &str) -> Result<String, String> {
    let url = normalize_url(url);
    dispatch(|reply| BrowserCmd::Navigate { url, reply })
}

pub fn screenshot() -> Result<Vec<u8>, String> {
    dispatch(|reply| BrowserCmd::Screenshot { reply })
}

pub fn snapshot() -> Result<Vec<u8>, String> {
    dispatch(|reply| BrowserCmd::Snapshot { reply })
}

pub fn scroll_page(to: Option<String>, by: Option<f64>) -> Result<String, String> {
    dispatch(|reply| BrowserCmd::Scroll { to, by, reply })
}

pub fn eval(expr: &str) -> Result<String, String> {
    // Wrap ordinary statement bodies (with `return`, `const`/`let`/`var` + `;`)
    // in an IIFE so the model can write multi-line JS, not just one expression.
    // A body already starting with `(` is assumed self-contained.
    let trimmed = expr.trim();
    let looks_like_statements = !trimmed.starts_with('(')
        && (trimmed.contains("return ")
            || trimmed.contains("return;")
            || (trimmed.contains(';')
                && (trimmed.contains("const ") || trimmed.contains("let ") || trimmed.contains("var "))));
    let expr = if looks_like_statements {
        format!("(function(){{{trimmed}}})()")
    } else {
        expr.to_string()
    };
    dispatch(|reply| BrowserCmd::Eval { expr, reply })
}

pub fn click(selector: Option<String>, text: Option<String>) -> Result<String, String> {
    dispatch(|reply| BrowserCmd::Click { selector, text, reply })
}

pub fn type_text(selector: Option<String>, label: Option<String>, text: String) -> Result<String, String> {
    dispatch(|reply| BrowserCmd::Type { selector, label, text, reply })
}

pub fn console() -> Result<String, String> {
    dispatch(|reply| BrowserCmd::Console { reply })
}

pub fn read_page() -> Result<String, String> {
    dispatch(|reply| BrowserCmd::Read { reply })
}

/// One-shot, always-HEADLESS render of a URL → (PNG bytes, console text). Uses
/// a dedicated throwaway Chrome that never shows a window — for Canvas's
/// "see the current page" capture, which must not pop up the interactive
/// browser the user is watching in Code mode.
pub fn capture_headless(url: &str) -> Result<(Vec<u8>, String), String> {
    let mut s = BrowserSession::launch(true, false)?;
    let r = (|| {
        s.navigate(url)?;
        let png = s.screenshot()?;
        let console = s.drain_console();
        Ok((png, console))
    })();
    s.kill();
    r
}

/// Accept bare hosts and local file paths; default to https for schemeless hosts.
fn normalize_url(u: &str) -> String {
    let u = u.trim();
    if u.starts_with("http://")
        || u.starts_with("https://")
        || u.starts_with("file://")
        || u.starts_with("about:")
        || u.starts_with("data:")
    {
        return u.to_string();
    }
    // An existing local file → file:// URL.
    let p = std::path::Path::new(u);
    if p.exists() {
        if let Ok(abs) = p.canonicalize() {
            return format!("file://{}", abs.display());
        }
    }
    format!("https://{u}")
}

/// A self-cleaning temp directory (Chrome's throwaway profile).
mod tempdir {
    use std::path::{Path, PathBuf};

    pub struct Guard(PathBuf);
    impl Guard {
        pub fn new(prefix: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "{prefix}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            let _ = std::fs::create_dir_all(&p);
            Guard(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Full CDP round-trip against the real Chrome. Ignored by default (needs a
    // browser). Run: cargo test -p chaty browser_cdp -- --ignored --nocapture
    #[test]
    #[ignore]
    fn browser_cdp_end_to_end() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        let html = "<!doctype html><html><head><title>Chaty Test</title></head>\
            <body style='background:#0a7'><h1 id='h'>Hello Chaty</h1>\
            <button id='b' onclick=\"document.getElementById('h').textContent='Clicked'\">Go</button>\
            <script>console.error('boom-42');console.log('ok-hi');</script></body></html>";
        let path = std::env::temp_dir().join(format!("chaty-browser-test-{}.html", std::process::id()));
        std::fs::write(&path, html).unwrap();
        let url = format!("file://{}", path.display());

        let nav = navigate(&url).expect("navigate");
        eprintln!("nav: {nav}");
        assert!(nav.contains("Chaty Test"), "title should appear: {nav}");

        let title = eval("document.title").expect("eval");
        assert_eq!(title, "Chaty Test");

        let shot = screenshot().expect("screenshot");
        assert!(shot.len() > 1000 && &shot[1..4] == b"PNG", "expected a PNG, got {} bytes", shot.len());

        let con = console().expect("console");
        eprintln!("console: {con}");
        assert!(con.contains("boom-42"), "console.error should be captured: {con}");

        let clicked = click(Some("#b".into()), None).expect("click");
        eprintln!("{clicked}");
        let h = eval("document.getElementById('h').textContent").expect("eval2");
        assert_eq!(h, "Clicked", "click handler should have run");

        let typed = type_text(Some("#h".into()), None, "ignored".into());
        assert!(typed.is_ok());
        // click by visible text (the robust path)
        let by_text = click(None, Some("Go".into()));
        assert!(by_text.is_ok(), "click-by-text should work: {by_text:?}");
        // a statement-body eval with `return` must not error
        let ev = eval("const x = 40 + 2; return x").expect("eval stmt");
        assert_eq!(ev.trim_matches('"'), "42");
        // element digest lists the button by text (still on the first page)
        let dig = read_page().expect("digest");
        assert!(dig.contains("Go"), "digest should list the button: {dig}");

        // lazy-load: a page that injects a marker only after scrolling near the
        // bottom. viewport snapshot at top must NOT see it; scroll + snapshot must.
        let lazy = "<!doctype html><title>Lazy</title>            <body style='margin:0'><div style='height:3000px'>top</div>            <div id='late'></div>            <script>addEventListener('scroll',function(){if(window.scrollY>1500){document.getElementById('late').textContent='LAZY-LOADED-CONTENT';}})</script></body>";
        let lp = std::env::temp_dir().join(format!("chaty-lazy-{}.html", std::process::id()));
        std::fs::write(&lp, lazy).unwrap();
        navigate(&format!("file://{}", lp.display())).expect("nav lazy");
        assert!(eval("document.getElementById('late').textContent").unwrap().trim_matches('"').is_empty(), "marker absent before scroll");
        let _sc = scroll_page(Some("bottom".into()), None).expect("scroll");
        let marker = eval("document.getElementById('late').textContent").unwrap();
        assert!(marker.contains("LAZY-LOADED-CONTENT"), "scroll should trigger lazy content, got {marker:?}");
        let snap = snapshot().expect("snapshot");
        assert!(snap.len() > 1000 && &snap[1..4] == b"PNG");
        let _ = std::fs::remove_file(&lp);

        // ---- edge cases ----
        // click nonexistent text → an error (with the digest), not a panic.
        assert!(click(None, Some("This Text Does Not Exist Anywhere".into())).is_err());
        // type into a nonexistent field → error, not panic.
        assert!(type_text(None, Some("no_such_field".into()), "x".into()).is_err());
        // scroll on a short page is harmless.
        assert!(scroll_page(Some("bottom".into()), None).is_ok());
        // navigate to a nonexistent local file → Chrome loads an error page but
        // the tool must not hang/panic.
        let _ = navigate("file:///no/such/file/anywhere.html");

        // ---- auto-recovery: simulate the user closing the browser ----
        crate::browser::kill_now(); // kill the tracked Chrome out-of-band
        std::thread::sleep(Duration::from_millis(400));
        // the next command must relaunch a fresh browser and succeed.
        let recovered = navigate(&url).expect("should auto-recover after the browser is killed");
        assert!(recovered.contains("Chaty Test"), "recovered nav should load: {recovered}");

        // ---- browser_close, then reuse ----
        shutdown();
        std::thread::sleep(Duration::from_millis(300));
        // a call after close starts a fresh actor + browser.
        let reused = navigate(&url).expect("should start a fresh browser after close");
        assert!(reused.contains("Chaty Test"));

        shutdown();
        let _ = std::fs::remove_file(&path);
    }

    // Canvas's headless one-shot capture: a PNG + the page's console, with no
    // interactive-browser window involved.
    #[test]
    #[ignore]
    fn capture_headless_returns_png_and_console() {
        if chrome_path().is_none() { eprintln!("SKIP: no Chrome"); return; }
        let html = "<!doctype html><title>Cap</title><body style='background:#123'>            <h1>Hi</h1><script>console.error('cap-err-7')</script></body>";
        let path = std::env::temp_dir().join(format!("chaty-cap-{}.html", std::process::id()));
        std::fs::write(&path, html).unwrap();
        let url = format!("file://{}", path.display());
        let (png, console) = capture_headless(&url).expect("capture");
        assert!(png.len() > 1000 && &png[1..4] == b"PNG");
        assert!(console.contains("cap-err-7"), "console should be captured: {console}");
        let _ = std::fs::remove_file(&path);
    }
}
