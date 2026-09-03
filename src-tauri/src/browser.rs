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
    ClickSeq { steps: Vec<(Option<String>, Option<String>)>, reply: Sender<Result<String, String>> },
    Type { selector: Option<String>, label: Option<String>, text: String, reply: Sender<Result<String, String>> },
    TypeSeq { steps: Vec<(Option<String>, Option<String>, String)>, reply: Sender<Result<String, String>> },
    Console { reply: Sender<Result<String, String>> },
    Refresh { reply: Sender<Result<String, String>> },
    Read { reply: Sender<Result<String, String>> },
    Close(Sender<()>),
}

// Model-visible browser strings pick their language per session (WS2 单语化;
// this module was the one surface that missed the v1.8.5 pass). The
// formatting itself comes from agent.rs's `trf!` — this file used to declare
// `btr!`, a byte-identical second copy of the same three lines.

/// JS that returns a compact list of the page's interactive elements, so the
/// model clicks/types against real visible text rather than guessed selectors.
/// Inputs/textareas/contenteditables also report their CURRENT value, so the
/// model sees what's typed without a screenshot.
const PAGE_DIGEST_JS: &str = r#"(function(){
  function vis(e){var r=e.getBoundingClientRect();if(r.width<2||r.height<2)return false;var s=getComputedStyle(e);return s.visibility!=='hidden'&&s.display!=='none';}
  var out=[];
  // Standard ARIA roles a choice control actually uses — a quiz answer is a
  // [role=radio] far more often than a button, and one that is not listed is
  // one the agent cannot be told about. `[tabindex]` catches the rest: an
  // unlabelled div a site made focusable and clickable. That last one needs
  // cursor:pointer to stay out, or every scroll container joins the list.
  var nodes=document.querySelectorAll("a,button,[role=button],[role=link],[role=menuitem],[role=tab],[role=radio],[role=checkbox],[role=switch],[role=option],[role=menuitemradio],[role=menuitemcheckbox],[role=treeitem],input,textarea,select,summary,[tabindex],[contenteditable=''],[contenteditable=true]");
  var all=[];
  for(var i=0;i<nodes.length;i++){
    var e=nodes[i];if(!vis(e))continue;
    if(!e.matches("a,button,[role],input,textarea,select,summary,[contenteditable=''],[contenteditable=true]")
       && getComputedStyle(e).cursor!=='pointer') continue;
    all.push(e);
  }
  // What is on screen comes first. Taking the first N in document order gave
  // a long page's list to whatever happened to be at the top of the DOM, so a
  // dialog or a card the user just opened — the only thing they can act on —
  // fell off the end.
  var inView=[],off=[];
  for(var i=0;i<all.length;i++){
    var r=all[i].getBoundingClientRect();
    (r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth?inView:off).push(all[i]);
  }
  var ordered=inView.concat(off), CAP=120;
  for(var i=0;i<ordered.length&&out.length<CAP;i++){
    var e=ordered[i];
    var tag=e.tagName.toLowerCase();
    var t=((e.innerText||e.value||e.getAttribute('aria-label')||e.placeholder||'')+'').trim().replace(/\s+/g,' ').slice(0,80);
    // Glyph-only controls (▶ ✕ ☰ …) are unclickable-by-text for a text agent
    // when the page repeats them per row/card — surface the aria-label, which
    // authors write exactly to disambiguate ("Move \"Fix login bug\" right").
    var al=(e.getAttribute('aria-label')||'').trim().replace(/\s+/g,' ');
    if(al&&al!==t&&t.replace(/[^\w一-鿿]/g,'').length<3){t=al.slice(0,80);}
    // State, not just label. A greyed-out submit, a chosen answer, an
    // already-matched tile all read exactly like their untouched selves
    // without this — so an agent clicks a dead control forever, or loses
    // track of what it has already picked.
    var st='';
    if(e.disabled||e.getAttribute('aria-disabled')==='true') st=' __L_DISABLED__';
    else if(e.getAttribute('aria-selected')==='true'||e.getAttribute('aria-checked')==='true'
            ||e.getAttribute('aria-pressed')==='true'||(e.checked===true&&tag==='input')) st=' __L_SELECTED__';
    else if(e.getAttribute('aria-expanded')==='true') st=' __L_EXPANDED__';
    if(e.isContentEditable){ out.push('__L_EDITABLE__: '+(e.getAttribute('aria-label')||e.id||'')+' = "'+((e.innerText||'').trim().replace(/\s+/g,' ').slice(0,120))+'"'); }
    else if(tag==='a'){ if(t) out.push('__L_LINK__: "'+t+'"'+st); }
    else if(tag==='button'||e.type==='submit'||e.type==='button'||/^(button|radio|checkbox|switch|option|tab|menuitem|menuitemradio|menuitemcheckbox|treeitem)$/.test(e.getAttribute('role')||'')||e.hasAttribute('tabindex')){ if(t) out.push('__L_BUTTON__: "'+t+'"'+st); }
    else if(tag==='input'||tag==='textarea'){ var h=e.placeholder||e.name||e.getAttribute('aria-label')||e.type||'text'; var v=(e.value||'').trim().replace(/\s+/g,' ').slice(0,120); out.push('__L_INPUT__ ['+(e.type||'text')+']: '+h+(v?(' = "'+v+'"'):'')+st); }
    else if(tag==='select'){ out.push('__L_SELECT__: '+(e.name||e.id||'')+' = "'+((e.options[e.selectedIndex]||{}).text||'')+'"'); }
  }
  if(ordered.length>out.length){ out.push('__L_MORE__ '+(ordered.length-out.length)); }
  return out.length? out.join("\n") : "__L_NONE__";
})()"#;

/// PAGE_DIGEST_JS with its labels in the session language.
fn digest_js() -> String {
    let en = crate::agent::lang_is_en();
    PAGE_DIGEST_JS
        .replace("__L_EDITABLE__", if en { "editable" } else { "可编辑区" })
        .replace("__L_LINK__", if en { "link" } else { "链接" })
        .replace("__L_BUTTON__", if en { "button" } else { "按钮" })
        .replace("__L_INPUT__", if en { "input" } else { "输入框" })
        .replace("__L_SELECT__", if en { "select" } else { "下拉" })
        .replace("__L_DISABLED__", if en { "[disabled]" } else { "[已禁用]" })
        .replace("__L_SELECTED__", if en { "[selected]" } else { "[已选中]" })
        .replace("__L_EXPANDED__", if en { "[expanded]" } else { "[已展开]" })
        .replace(
            "__L_MORE__",
            if en {
                "(+ more not listed — scroll to bring them into view:)"
            } else {
                "(还有若干未列出,滚动到视口内即可看到:)"
            },
        )
        .replace(
            "__L_NONE__",
            if en { "(no obvious interactive elements)" } else { "(未发现明显的可交互元素)" },
        )
}

/// JS returning the page's VISIBLE text (what a person / the vision model would
/// read), rendered in document order via innerText (which already respects
/// visibility, display and layout, and de-duplicates container/child text).
/// Blank lines are collapsed and the result is capped by the caller-supplied
/// `__CAP__`. This is the text substitute for a screenshot: dynamically shown
/// content (game rules, validation messages, results) is read as text, so the
/// model never has to screenshot to learn what just appeared.
const PAGE_TEXT_JS: &str = r#"(function(){
  var t='';
  try{ t=(document.body&&document.body.innerText)||''; }catch(e){ t=''; }
  t=t.replace(/[ \t ]+/g,' ').replace(/\n{3,}/g,'\n\n').split('\n').map(function(l){return l.trim();}).join('\n').replace(/\n{3,}/g,'\n\n').trim();
  var cap=__CAP__;
  if(t.length>cap){ t=t.slice(0,cap)+'\n__L_TRUNC__'; }
  return t||'__L_EMPTY__';
})()"#;

/// Separates the parts of a `rich_digest` inside the single string the browser
/// hands back. Control characters, so it cannot collide with page text.
const DIGEST_SEP: &str = "\u{0}\u{1}chaty\u{1}\u{0}";
/// The same sentinel spelled for a JavaScript string literal.
const SEP_JS: &str = "\\u0000\\u0001chaty\\u0001\\u0000";

/// PAGE_TEXT_JS with the cap substituted and notes in the session language.
fn page_text_js(cap: usize) -> String {
    let en = crate::agent::lang_is_en();
    PAGE_TEXT_JS
        .replace("__CAP__", &cap.to_string())
        .replace("__L_TRUNC__", if en { "…(text truncated)" } else { "…(文字过长已截断)" })
        .replace("__L_EMPTY__", if en { "(no visible text)" } else { "(页面无可见文字)" })
}

/// Watch for asynchronous page work so an action's result digest reflects what
/// the page became, not what it was. Without this, an AJAX form submit returns
/// the OLD page (the form, no "Thank you") because `readyState` never left
/// "complete" — the model concludes the submit failed and clicks again, posting
/// duplicates. Patches fetch/XHR to count in-flight requests and watches DOM
/// mutations for a quiet period. Idempotent; re-installed after navigation
/// (the patch dies with the document).
const SETTLE_INSTALL_JS: &str = r#"(function(){
  if(window.__chatyWatch)return 'ok';
  var w={pending:0,last:Date.now(),n:0};
  window.__chatyWatch=w;
  function touch(){w.last=Date.now();w.n++;}
  try{
    var of=window.fetch;
    if(of){window.fetch=function(){w.pending++;touch();
      var p=of.apply(this,arguments);
      try{return p.then(function(r){w.pending--;touch();return r;},
                        function(e){w.pending--;touch();throw e;});}
      catch(e){w.pending--;touch();return p;}};}
  }catch(e){}
  try{
    var os=XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send=function(){w.pending++;touch();
      this.addEventListener('loadend',function(){w.pending--;touch();});
      return os.apply(this,arguments);};
  }catch(e){}
  try{
    // attributes matter: real sites reveal a success banner with
    // `el.style.display='block'` or a class toggle, which changes NO text
    // nodes — without attribute mutations we'd call the page "unchanged".
    new MutationObserver(touch).observe(document.documentElement,
      {subtree:true,childList:true,characterData:true,attributes:true,
       attributeFilter:['style','class','hidden','aria-hidden','disabled','value']});
  }catch(e){}
  return 'ok';
})()"#;

/// Returns "<in-flight requests>:<ms since last change>:<change counter>", or
/// "none" when the watcher isn't installed (fresh document — caller reinstalls).
const SETTLE_POLL_JS: &str = r#"(function(){
  var w=window.__chatyWatch;
  if(!w)return 'none';
  return w.pending+':'+(Date.now()-w.last)+':'+w.n;
})()"#;

/// Text of the region around the element the agent last clicked or typed into.
/// A long page's visible text is capped from the TOP, so anything that appears
/// next to the control — a form's success banner, an inline error, a cart total
/// — falls outside the window and the agent concludes nothing happened. (Real
/// report: a contact form near the bottom of a long single-page site revealed
/// "Thank you" on success; the agent never saw it and kept submitting.)
const NEAR_TEXT_JS: &str = r#"(function(){
  var el=window.__chatyLast;
  if(!el||!el.isConnected)return '';
  var box=el.closest('form,[role=dialog],dialog,section,article,main')||el.parentElement;
  for(var i=0;i<3&&box&&((box.innerText||'').trim().length<40)&&box.parentElement;i++){
    box=box.parentElement; // tiny wrapper — widen until there's something to read
  }
  if(!box)return '';
  var t=(box.innerText||'').replace(/[ \t ]+/g,' ').split('\n')
        .map(function(l){return l.trim();}).filter(function(l,i,a){return l||a[i-1];})
        .join('\n').trim();
  var cap=__NEARCAP__;
  if(t.length>cap)t=t.slice(0,cap)+'…';
  return t;
})()"#;

/// Why did a click do nothing? The most common answer on real sites is native
/// form validation: `required` / `type=email` fields block submission before
/// any handler runs, and the browser's own bubble is invisible to both the
/// page text and the element list — the agent sees an unchanged page, assumes
/// the click missed, and clicks forever. Report the blocking fields instead.
/// Scoped to the form containing the element the agent just used, so a click on
/// an unrelated link never reports someone else's empty field.
const FORM_BLOCKERS_JS: &str = r#"(function(){
  var out=[];
  try{
    var el=window.__chatyLast;
    if(!el||!el.isConnected)return '';
    // Only a control that SUBMITS can be blocked by validation. Clicking a
    // checkbox or a plain button inside a half-filled form is not blocked, and
    // saying "the submit never fired" there would be a lie.
    var tag=(el.tagName||'').toUpperCase(), ty=(el.getAttribute('type')||'').toLowerCase();
    var submits=(tag==='BUTTON'&&(ty===''||ty==='submit'))||
                (tag==='INPUT'&&(ty==='submit'||ty==='image'));
    if(!submits)return '';
    if(el.hasAttribute('formnovalidate'))return '';
    var f=el.closest&&el.closest('form');
    // novalidate forms validate in JS, if at all — the browser lets them submit.
    if(!f||f.noValidate)return '';
    if(typeof f.checkValidity!=='function'||f.checkValidity())return '';
    var fields=[].slice.call(f.querySelectorAll('input,select,textarea'));
    for(var j=0;j<fields.length&&out.length<8;j++){
      var x=fields[j];
      if(x.disabled||x.type==='hidden')continue;
      if(x.willValidate===false||x.checkValidity())continue;
      var name=(x.labels&&x.labels[0]&&x.labels[0].innerText)||x.placeholder||
               x.getAttribute('aria-label')||x.name||x.id||x.type;
      out.push(((name+'')).trim().replace(/\s+/g,' ').slice(0,40)+': '+
               (x.validationMessage||'invalid'));
    }
  }catch(e){}
  return out.join(' | ');
})()"#;

/// Auto-scroll through the whole page (triggering lazy-loaded content) and
/// return to the top — run before a full-page screenshot. Resolves a promise so
/// `Runtime.evaluate` with awaitPromise waits for it.
const AUTOSCROLL_JS: &str = r#"new Promise(function(done){
  var y=0,h=document.body.scrollHeight,step=Math.max(200,window.innerHeight*0.9),ticks=0;
  var timer=setInterval(function(){
    window.scrollTo(0,y);y+=step;ticks++;
    if(y>=document.body.scrollHeight||ticks>40){clearInterval(timer);window.scrollTo(0,0);setTimeout(done,200);}
  },50);
})"#;

/// Freeze CSS motion at its END state before a full-page capture: reveal-on-
/// scroll pages (IntersectionObserver + transition) raced the shot — the
/// autoscroll passes a 6-screen page in ~0.4s, every staggered 0.5-0.8s
/// transition is mid-flight or unstarted, and whole sections captured BLANK
/// (owner walkthrough: the Hello-Kitty featured-products grid photographed
/// empty; the model faithfully reported products that "weren't there").
/// Zeroing durations makes any triggered reveal land instantly; idempotent.
const FREEZE_ANIMATIONS_JS: &str = r#"(function(){
  if(document.getElementById('__chaty_freeze'))return 'ok';
  var s=document.createElement('style');s.id='__chaty_freeze';
  s.textContent='*,*::before,*::after{animation-duration:0s!important;animation-delay:0s!important;transition-duration:0s!important;transition-delay:0s!important}';
  (document.head||document.documentElement).appendChild(s);return 'ok';
})()"#;

/// Process-wide handle to the browser actor thread. Lazily started.
/// The live actor's handle, tagged with which actor installed it. The tag is
/// what makes teardown safe: an actor that exits must forget ITS OWN handle
/// and never a successor's.
static BROWSER: Mutex<Option<(u64, Sender<BrowserCmd>)>> = Mutex::new(None);
/// Hands out actor tags. Wraps after 2^64 actors, which is not a concern.
static ACTOR_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Persistent profile dir for the interactive browser (set once at startup, so
/// the user's logins survive across runs). `None` → throwaway profile (tests).
static PROFILE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Point the interactive browser at a persistent profile directory so cookies /
/// logins persist. Called once from app setup with the app-data path.
pub fn set_profile_dir(dir: PathBuf) {
    *PROFILE_DIR.lock().unwrap() = Some(dir);
}

/// User preference: run the agent's interactive browser hidden (headless).
/// Settings → Code. Toggling it while a browser is open closes that browser so
/// the next tool call relaunches in the mode the user just picked — otherwise
/// the setting looked ignored until the session ended.
static HEADLESS_PREF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_headless(on: bool) {
    let was = HEADLESS_PREF.swap(on, std::sync::atomic::Ordering::Relaxed);
    if was != on {
        shutdown();
    }
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
            let mut cmd = std::process::Command::new("taskkill");
            cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
            let _ = crate::agent::hide_console(&mut cmd).output();
        }
    }
}

/// Profile dirs under `root` whose creator process is gone. The dir name is
/// `chaty-cdp-<creator pid>-<nanos>` (tempdir::Guard), so liveness of that
/// pid decides ownership — a concurrent Chaty/headless keeps its own.
fn orphan_cdp_dirs(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(rd) = std::fs::read_dir(root) else { return Vec::new() };
    rd.flatten()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(rest) = name.strip_prefix("chaty-cdp-") else { return false };
            // Unparsable pid = malformed debris → orphan.
            rest.split('-').next().and_then(|s| s.parse::<u32>().ok()).is_none_or(|pid| !pid_alive(pid))
        })
        .map(|e| e.path())
        .collect()
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    // Creators are always our own uid, so ESRCH is the only "gone" signal;
    // EPERM would mean someone else's process — never ours — count it alive.
    if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}
#[cfg(windows)]
fn pid_alive(pid: u32) -> bool {
    let mut cmd = std::process::Command::new("tasklist");
    cmd.args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"]);
    crate::agent::hide_console(&mut cmd)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&format!(",\"{pid}\"")))
        .unwrap_or(true) // can't tell ⇒ assume alive, never kill blind
}

/// Startup sweep for browsers that outlived their Chaty. Every exit path
/// that skips destructors — the `_exit()` in the app's exit handler, a
/// SIGKILLed bench bridge, a crash — leaves the headless Chrome tree running
/// and its profile dir behind (16 helpers + 14 dirs stood on the author's
/// machine the day this was written). Kill by profile path, then remove.
pub fn sweep_orphan_browsers() {
    for dir in orphan_cdp_dirs(&std::env::temp_dir()) {
        #[cfg(unix)]
        {
            let pat = dir.to_string_lossy().to_string();
            let _ = std::process::Command::new("pkill").args(["-9", "-f", &pat]).status();
        }
        // Windows: no safe kill-by-cmdline; a live Chrome holds the profile
        // lock so the remove fails and the dir simply waits for the next try.
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// PIDs of browsers currently holding `profile` as their user-data-dir.
///
/// Chrome records its debugging endpoint inside the profile, which is how a
/// leftover browser is normally found again — but that file is gone whenever
/// the browser exited uncleanly, or an older build of this app deleted it on
/// the way to a launch that could never succeed. The process list still knows.
#[cfg(unix)]
fn browsers_holding(profile: &std::path::Path) -> Vec<u32> {
    let needle = format!("--user-data-dir={}", profile.display());
    let Ok(out) = std::process::Command::new("ps").args(["-Ao", "pid,args"]).output() else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.contains(&needle) && !l.contains("--type="))
        .filter_map(|l| l.split_whitespace().next()?.parse().ok())
        .collect()
}
#[cfg(not(unix))]
fn browsers_holding(_profile: &std::path::Path) -> Vec<u32> {
    // No safe way to match a command line here; the timeout message explains
    // the situation instead.
    Vec::new()
}

/// Candidate Chrome/Chromium executables by platform.
fn chrome_path() -> Option<PathBuf> {
    #[cfg_attr(not(target_os = "windows"), allow(unused_mut))]
    let mut candidates: Vec<String> = if cfg!(target_os = "macos") {
        [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
        .map(String::from)
        .to_vec()
    } else if cfg!(target_os = "windows") {
        [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        ]
        .map(String::from)
        .to_vec()
    } else {
        ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]
            .map(String::from)
            .to_vec()
    };
    // Windows per-user installs (Chrome's default for non-admin setups) live
    // under %LOCALAPPDATA% — a very common miss on personal machines.
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        candidates.insert(0, format!(r"{local}\Google\Chrome\Application\chrome.exe"));
        candidates.push(format!(r"{local}\Chromium\Application\chrome.exe"));
        candidates.push(format!(r"{local}\BraveSoftware\Brave-Browser\Application\brave.exe"));
    }
    for c in &candidates {
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

/// Clears the cached sender when the actor thread leaves, by any route — but
/// only if the cache still holds THIS actor's handle.
///
/// Clearing unconditionally is a race with a fatal shape: a closed browser's
/// actor finishes its teardown a moment after the next one has already been
/// launched and cached, and wipes the newcomer's handle. Nothing detects that,
/// because the sender still works — so every later call built a whole new
/// browser, ran one command in it, and lost it again. What the user saw was
/// browsing dying permanently after a single `browser_close`: navigation
/// reported success while every page read came back blank, since the read
/// happened in a brand-new window that had never been navigated anywhere.
struct ForgetOnExit(u64);
impl Drop for ForgetOnExit {
    fn drop(&mut self) {
        if let Ok(mut guard) = BROWSER.lock() {
            if guard.as_ref().is_some_and(|(id, _)| *id == self.0) {
                *guard = None;
            }
        }
    }
}

/// Ensure the actor is running; returns a sender to talk to it.
fn ensure() -> Result<Sender<BrowserCmd>, String> {
    let mut guard = BROWSER.lock().unwrap();
    if let Some((_, tx)) = guard.as_ref() {
        return Ok(tx.clone());
    }
    let id = ACTOR_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    let (tx, rx) = std::sync::mpsc::channel::<BrowserCmd>();
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::Builder::new()
        .name("chaty-browser".into())
        .spawn(move || {
            // Whatever ends the actor — a clean Close, a failure, or a panic —
            // takes the handle it was reached through with it. A sender left
            // behind by a dead actor is not detectably dead: every later call
            // fails on the send, the handle stays cached, and the browser is
            // gone for the rest of the session with nothing to relaunch it.
            let _forget = ForgetOnExit(id);
            actor(rx, init_tx)
        })
        .map_err(|e| trf!("无法启动浏览器线程:{}", "failed to start the browser thread: {}", e))?;
    match init_rx.recv() {
        Ok(Ok(())) => {
            *guard = Some((id, tx.clone()));
            Ok(tx)
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err(trf!("浏览器线程初始化失败", "the browser thread failed to initialize")),
    }
}

/// Drop the actor (kills Chrome). Called on workspace switch / app teardown.
pub fn shutdown() {
    let Some((_, tx)) = BROWSER.lock().unwrap().take() else { return };
    // Wait for Chrome to actually be gone. Returning while it is still dying
    // meant the next navigate launched a second Chrome on the SAME profile
    // directory — and a second Chrome hands its command line to the instance
    // that already owns the profile and exits, so what we attached to was a
    // browser on its way out: a blank page, and every page after it blank too,
    // for the rest of the run. One close used to end browsing for good.
    let (done, wait) = std::sync::mpsc::channel();
    if tx.send(BrowserCmd::Close(done)).is_ok() {
        // Bounded: a wedged actor must not hang the caller.
        let _ = wait.recv_timeout(Duration::from_secs(5));
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
        // An adopted browser has no process of ours to check; if it has gone
        // away, the command below fails and the CDP error says so.
        let dead = session
            .child
            .as_mut()
            .map(|c| c.try_wait().map(|s| s.is_some()).unwrap_or(true))
            .unwrap_or(false);
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

    // Text-returning interactions carry any NEW error-class console lines —
    // the model shouldn't have to remember to ask whether the page it just
    // touched blew up. Cursor-based: each line rides along exactly once.
    // Screenshot/snapshot (image payloads) and browser_console (the explicit
    // full view) stay untouched.
    fn with_console_errors(
        session: &mut BrowserSession,
        r: Result<String, String>,
    ) -> Result<String, String> {
        let mut text = r?;
        // Always advance the cursor (mark lines as seen), but only ATTACH on
        // pages the developer owns — localhost / local files. On someone
        // else's website the console is third-party noise; it stays available
        // via an explicit browser_console call.
        let mut errs = session.unsurfaced_errors();
        if !BrowserSession::is_local_page_url(&session.current_url) {
            return Ok(text);
        }
        if errs.is_empty() {
            return Ok(text);
        }
        if errs.len() > 8 {
            let dropped = errs.len() - 8;
            errs.drain(..dropped);
            errs.insert(0, trf!("(另有 {dropped} 条更早的报错)", "({dropped} earlier errors omitted)"));
        }
        let mut block = errs.join("\n");
        if block.len() > 1500 {
            let cut = block.len() - 1500;
            block = block[block.char_indices().map(|(i, _)| i).find(|&i| i >= cut).unwrap_or(0)..].to_string();
        }
        // A dialog fired by the page (alert after a click, confirm on submit)
        // is expected behavior, not an error — labeling it "报错" sent the
        // model (and the user) hunting for a bug that isn't there.
        let has_error = errs.iter().any(|l| !l.starts_with("[dialog]"));
        let has_dialog = errs.iter().any(|l| l.starts_with("[dialog]"));
        let head = if has_error && has_dialog {
            trf!(
                "\n\n[console] 页面在这次操作前后新增的报错与弹窗(自动附带):\n",
                "\n\n[console] page errors and dialogs since the last action (auto-attached):\n"
            )
        } else if has_dialog {
            trf!(
                "\n\n[console] 页面弹窗(已自动处理,附带告知,非报错):\n",
                "\n\n[console] page dialog (auto-handled, attached for awareness — not an error):\n"
            )
        } else {
            trf!(
                "\n\n[console] 页面在这次操作前后新增的报错(自动附带):\n",
                "\n\n[console] page errors since the last action (auto-attached):\n"
            )
        };
        text.push_str(&head);
        text.push_str(&block);
        Ok(text)
    }

    while let Ok(cmd) = rx.recv() {
        match cmd {
            BrowserCmd::Navigate { url, reply } => {
                let r = run(&mut session, headless, |s| s.navigate(&url));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::Screenshot { reply } => {
                let r = run(&mut session, headless, |s| {
                    if !page_loaded(&s.current_url) {
                        return Err(no_page_error());
                    }
                    s.screenshot()
                });
                let _ = reply.send(r);
            }
            BrowserCmd::Snapshot { reply } => {
                let r = run(&mut session, headless, |s| {
                    if !page_loaded(&s.current_url) {
                        return Err(no_page_error());
                    }
                    s.snapshot()
                });
                let _ = reply.send(r);
            }
            BrowserCmd::Scroll { to, by, reply } => {
                let r = run(&mut session, headless, |s| s.scroll(to.as_deref(), by));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::Eval { expr, reply } => {
                let r = run(&mut session, headless, |s| s.eval(&expr));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::Click { selector, text, reply } => {
                let r = run(&mut session, headless, |s| s.click(selector.as_deref(), text.as_deref()));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::ClickSeq { steps, reply } => {
                let r = run(&mut session, headless, |s| s.click_seq(&steps));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::Type { selector, label, text, reply } => {
                let r = run(&mut session, headless, |s| s.type_text(selector.as_deref(), &text, label.as_deref()));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::TypeSeq { steps, reply } => {
                let r = run(&mut session, headless, |s| s.type_seq(&steps));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::Refresh { reply } => {
                let r = run(&mut session, headless, |s| s.refresh());
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::Console { reply } => {
                let _ = reply.send(run(&mut session, headless, |s| Ok(s.drain_console())));
            }
            BrowserCmd::Read { reply } => {
                let r = run(&mut session, headless, |s| s.rich_digest(12000));
                let _ = reply.send(with_console_errors(&mut session, r));
            }
            BrowserCmd::Close(done) => {
                session.kill();
                let _ = done.send(());
                return;
            }
        }
    }
    session.kill();
}

/// A launched Chrome + an attached page session over one CDP WebSocket.
/// Console lines kept for `browser_console`. Enough to hold a page's whole
/// startup noise plus the error that matters, bounded so a chatty page cannot
/// grow the buffer without limit across a long session.
const CONSOLE_KEEP: usize = 400;

struct BrowserSession {
    /// `None` when this session ADOPTED a browser someone else started —
    /// there is no process of ours to wait on or signal.
    child: Option<std::process::Child>,
    ws: Ws,
    session_id: String,
    next_id: i64,
    /// Buffered console API calls + exceptions since the last `drain_console`.
    console: Vec<String>,
    /// How many buffered lines were already auto-attached to an interaction
    /// result (the cursor keeps repeats out; `drain_console` resets it).
    surfaced: usize,
    /// Sessions that attached while we were pumping — cross-origin iframes,
    /// popups, workers. Each needs Runtime/Log enabled before it reports
    /// anything, and that call cannot be made from inside the pump loop, so
    /// they queue here and are drained after it.
    pending_sessions: Vec<String>,
    /// What intercepted the last click that never reached its target, so the
    /// error can name the overlay instead of just saying the click failed.
    last_click_blocker: Option<String>,
    /// Main-frame URL, kept fresh by navigate() and Page.frameNavigated —
    /// gates console auto-attach to LOCAL pages only (real websites are full
    /// of third-party console noise the model must not drown in).
    current_url: String,
    /// Throwaway profile guard (deletes on drop); `None` for a persistent profile.
    _profile: Option<tempdir::Guard>,
}

impl BrowserSession {
    /// `track_pid`: register the child in CHROME_PID for exit-time cleanup (the
    /// shared interactive browser); one-shot headless captures pass false.
    fn launch(headless: bool, track_pid: bool) -> Result<Self, String> {
        Self::launch_once(headless, track_pid, true)
    }

    /// `may_recover` is false on the retry, so a browser that cannot be
    /// cleared reports the failure instead of looping.
    fn launch_once(headless: bool, track_pid: bool, may_recover: bool) -> Result<Self, String> {
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
        let port_file = profile_path.join("DevToolsActivePort");
        // A browser of ours may STILL BE OPEN on this profile — the app was
        // force-quit, crashed, or reloaded in dev while its window stayed up.
        // Chrome will not start a second browser on one profile: the new
        // process hands its command line to the one already there and exits,
        // so the launch below would wait out its deadline for a port that
        // never appears, and browsing would stay broken until the user hunted
        // down the stray window themselves. If the endpoint from last time
        // still answers, that browser is ours to drive.
        let adopted = std::fs::read_to_string(&port_file).ok().and_then(|c| {
            let mut lines = c.lines();
            let port = lines.next()?.trim().parse::<u16>().ok()?;
            let path = lines.next()?.trim().to_string();
            connect(&format!("ws://127.0.0.1:{port}{path}")).ok().map(|(ws, _)| ws)
        });
        // Chrome only writes DevToolsActivePort AFTER init; a stale one from a
        // previous run of a persistent profile would be read as the wrong port.
        if adopted.is_none() {
            let _ = std::fs::remove_file(&port_file);
        }

        let mut cmd = std::process::Command::new(&exe);
        if headless {
            cmd.arg("--headless=new").arg("--hide-scrollbars");
        } else {
            // Bring the automation window to the front so it's clearly visible.
            cmd.arg("--new-window").arg("--start-maximized");
        }
        let spawn_one = |cmd: &mut std::process::Command| cmd
            .arg("--remote-debugging-port=0")
            .arg(format!("--user-data-dir={}", profile_path.display()))
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            // Crisp captures (2× device pixels) for both the screenshot the model
            // reads and the preview the user opens.
            .arg("--force-device-scale-factor=2")
            .arg("--window-size=1280,900")
            .arg("--disable-background-networking")
            // New-headless Chrome (≥~150) parks frame production on static /
            // occluded pages; a later Page.captureScreenshot then waits for a
            // frame that never comes, times out, and the dead-session
            // recovery relaunches into a blank tab. The standard automation
            // trio (same defaults Puppeteer ships) keeps frames alive.
            .arg("--disable-background-timer-throttling")
            .arg("--disable-backgrounding-occluded-windows")
            .arg("--disable-renderer-backgrounding")
            .arg("about:blank")
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .spawn();

        let (mut ws, child) = match adopted {
            Some(ws) => (ws, None),
            None => {
                let mut child = spawn_one(&mut cmd)
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
                        let _ = child.wait();
                        // A browser already owning this profile is the reason
                        // this happens: ours never really started — Chrome
                        // handed our command line to the one already there and
                        // exited. Close that one and take the profile back,
                        // rather than leaving the user with a browser that can
                        // never open again.
                        let stray = browsers_holding(&profile_path);
                        if may_recover && !stray.is_empty() {
                            for pid in &stray {
                                // TERM, not KILL: it gets to save its session.
                                let _ = std::process::Command::new("kill")
                                    .args(["-TERM", &pid.to_string()])
                                    .status();
                            }
                            let gone = Instant::now() + Duration::from_secs(6);
                            while Instant::now() < gone
                                && !browsers_holding(&profile_path).is_empty()
                            {
                                std::thread::sleep(Duration::from_millis(120));
                            }
                            return Self::launch_once(headless, track_pid, false);
                        }
                        return Err(trf!(
                            "Chrome 未在预期时间内就绪。多半是另一个 Chrome 正占着同一个浏览器数据目录({});关掉那个窗口再试。",
                            "Chrome did not become ready in time. Most likely another Chrome already has this browser profile open ({}) — close that window and try again.",
                            profile_path.display()
                        ));
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
                let (ws, _) = connect(&url)
                    .map_err(|e| format!("连接 CDP 失败 (failed to connect CDP): {e}"))?;
                (ws, Some(child))
            }
        };
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
            if let Some(mut c) = child {
                let _ = c.kill();
            }
            return Err("CDP 会话附加失败 (failed to attach CDP session)".into());
        }

        let mut s = BrowserSession { child, ws, session_id, next_id, console: Vec::new(), surfaced: 0, pending_sessions: Vec::new(), last_click_blocker: None, current_url: String::new(), _profile: _guard };
        // Enable the domains we consume. Runtime.enable surfaces console API
        // calls + uncaught exceptions; Log.enable surfaces browser log entries.
        let sid = s.session_id.clone();
        let _ = s.call(Some(&sid), "Page.enable", json!({}));
        let _ = s.call(Some(&sid), "Runtime.enable", json!({}));
        // New-headless parks rendering for unfocused pages — a later
        // captureScreenshot then waits on a frame that never comes and hits
        // the CDP read timeout (observed on Chrome 150 after a plain
        // scroll). Emulating focus keeps the compositor producing frames;
        // Puppeteer ships the same call for exactly this reason.
        let _ = s.call(Some(&sid), "Emulation.setFocusEmulationEnabled", json!({"enabled": true}));
        let _ = s.call(Some(&sid), "Log.enable", json!({}));
        // Everything the page spawns reports on its OWN session: a cross-origin
        // iframe, a window it opens, a worker. Without this they are simply not
        // attached, and their errors appear in Chrome's console — where the user
        // sees them — while `browser_console` comes back empty, which is exactly
        // the shape of "the browser shows an error the tool cannot find".
        let _ = s.call(
            Some(&sid),
            "Target.setAutoAttach",
            json!({"autoAttach": true, "waitForDebuggerOnStart": false, "flatten": true}),
        );
        // And at the browser level, for targets the page did not create.
        let _ = s.call(
            None,
            "Target.setAutoAttach",
            json!({"autoAttach": true, "waitForDebuggerOnStart": false, "flatten": true}),
        );
        s.enable_pending_sessions();
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
                Ok(Message::Close(_)) => return Err(trf!("CDP 连接已关闭", "the CDP connection closed")),
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
            "Page.frameNavigated" => {
                // Main frame only (no parentId): clicks and redirects move the
                // page without a navigate() call.
                let frame = &p["frame"];
                if frame.get("parentId").and_then(|v| v.as_str()).is_none() {
                    if let Some(u) = frame["url"].as_str() {
                        self.current_url = u.to_string();
                    }
                }
            }
            "Target.attachedToTarget" => {
                if let Some(sid) = p["sessionId"].as_str() {
                    self.pending_sessions.push(sid.to_string());
                }
            }
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
            // A native dialog FREEZES the page's JS engine — an in-flight
            // Runtime.evaluate never replies and every later call times out.
            // Handle it right here in the event pump: OK an alert (its only
            // button), DECLINE confirm/prompt/beforeunload (auto-accepting
            // could trigger destructive paths). Fire-and-forget: waiting for
            // the reply would recurse into pump_until and eat the pending
            // response; the stray ack is dropped by the id filter later.
            // The dialog text goes to the console so the model SEES why the
            // page paused (e.g. a form backend's alert on failure).
            "Page.javascriptDialogOpening" => {
                let dtype = p["type"].as_str().unwrap_or("alert");
                let text = p["message"].as_str().unwrap_or("");
                let accept = dtype == "alert";
                let id = self.next_id;
                self.next_id += 1;
                let cmd = json!({
                    "id": id,
                    "sessionId": self.session_id,
                    "method": "Page.handleJavaScriptDialog",
                    "params": { "accept": accept },
                });
                let _ = self.ws.send(Message::Text(cmd.to_string().into()));
                self.push_console(format!(
                    "[dialog] {dtype}: {text} ({})",
                    if accept { "auto-accepted" } else { "auto-dismissed" }
                ));
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

    /// Is this a page whose console the developer OWNS — a dev server or a
    /// local file — as opposed to someone else's website? Console auto-attach
    /// is gated on this: real sites routinely spew third-party errors
    /// (analytics, CSP, ad scripts) that would pollute every interaction.
    fn is_local_page_url(url: &str) -> bool {
        let u = url.trim().to_ascii_lowercase();
        if u.starts_with("file://") || u.starts_with("data:") || u.starts_with("about:") {
            return true;
        }
        let rest = match u.strip_prefix("http://").or_else(|| u.strip_prefix("https://")) {
            Some(r) => r,
            None => return false,
        };
        let host = rest.split(['/', '?', '#']).next().unwrap_or("");
        let host = host.strip_prefix('[').map(|h| h.split(']').next().unwrap_or(h)).unwrap_or_else(
            || host.split(':').next().unwrap_or(""),
        );
        host == "localhost"
            || host.ends_with(".localhost")
            || host == "127.0.0.1"
            || host == "0.0.0.0"
            || host == "::1"
    }

    /// Error-class console lines the model hasn't seen yet ([error] /
    /// [exception] / [dialog]) — auto-attached to interaction results so a
    /// broken page is impossible to miss. Advances the cursor; info/log lines
    /// stay buffered for an explicit browser_console call.
    fn unsurfaced_errors(&mut self) -> Vec<String> {
        self.pump_pending();
        let from = self.surfaced.min(self.console.len());
        let errs: Vec<String> = self.console[from..]
            .iter()
            .filter(|l| {
                l.starts_with("[error]") || l.starts_with("[exception]") || l.starts_with("[dialog]")
            })
            .cloned()
            .collect();
        self.surfaced = self.console.len();
        errs
    }

    fn push_console(&mut self, line: String) {
        if self.console.len() < 200 {
            self.console.push(line);
        }
    }

    /// The console as Chrome would show it. Reading does NOT empty it: a model
    /// debugging a page looks more than once, and a second look answering
    /// "console is empty" while the browser still shows the error is worse than
    /// repeating a line. The buffer is trimmed to a bound instead.
    fn drain_console(&mut self) -> String {
        // Also pump any pending frames (non-blocking-ish) so freshly-logged
        // messages are included even without an intervening command.
        self.pump_pending();
        self.enable_pending_sessions();
        // Enabling a session can produce a burst of buffered entries.
        self.pump_pending();
        if self.console.len() > CONSOLE_KEEP {
            let cut = self.console.len() - CONSOLE_KEEP;
            self.console.drain(..cut);
            self.surfaced = self.surfaced.saturating_sub(cut);
        }
        if self.console.is_empty() {
            return "（控制台无输出 / console is empty）".into();
        }
        // Everything has now been shown, so nothing here is "unsurfaced" any
        // more — a later interaction attaches only what arrives after this.
        self.surfaced = self.console.len();
        self.console.join("\n")
    }

    /// Drain frames already waiting on the socket (short read timeout).
    /// Turn on the domains we read for every session that attached while we
    /// were pumping. Called outside the pump loop, which cannot send.
    fn enable_pending_sessions(&mut self) {
        while let Some(sid) = self.pending_sessions.pop() {
            let _ = self.call(Some(&sid), "Runtime.enable", json!({}));
            let _ = self.call(Some(&sid), "Log.enable", json!({}));
            // A target this one spawns in turn reports the same way.
            let _ = self.call(
                Some(&sid),
                "Target.setAutoAttach",
                json!({"autoAttach": true, "waitForDebuggerOnStart": false, "flatten": true}),
            );
        }
    }

    fn pump_pending(&mut self) {
        // Drain what has already arrived; don't sit waiting for more. Anything
        // sent while this returns is still in the socket and the next drain
        // takes it — and every one of these is called in a loop. A blocking
        // window here is paid on EVERY call, and it is empty nearly every
        // time, which put more than a hundred milliseconds on each poll of
        // `wait_settled` and so on every click, type and scroll the agent made.
        set_read_timeout(&self.ws, Duration::from_millis(2));
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
        let (final_url, title, rich) = self.settle_and_digest()?;
        Ok(trf!(
            "已打开:{}\n标题:{}\n\n{}",
            "Loaded: {}\nTitle: {}\n\n{}",
            final_url,
            title,
            rich
        ))
    }

    /// True reload of the current page, cache ignored — the local-dev verb.
    /// The webapp walkthrough caught the 35B "refreshing" by re-screenshotting
    /// the STALE render after editing files; without a reload the page can't
    /// show the new code, no matter how it is observed.
    fn refresh(&mut self) -> Result<String, String> {
        let sid = self.session_id.clone();
        self.call(Some(&sid), "Page.reload", json!({"ignoreCache": true}))?;
        let (final_url, title, rich) = self.settle_and_digest()?;
        Ok(trf!(
            "已刷新(忽略缓存):{}\n标题:{}\n\n{}",
            "Reloaded (cache ignored): {}\nTitle: {}\n\n{}",
            final_url,
            title,
            rich
        ))
    }

    /// Shared post-load tail: settle, re-arm the async watcher, read back
    /// url/title, and build the text digest.
    fn settle_and_digest(&mut self) -> Result<(String, String, String), String> {
        // Give the page a moment to load + settle (SPA JS, first paint), then
        // arm the async-work watcher for the interactions that follow.
        std::thread::sleep(Duration::from_millis(1200));
        self.pump_pending();
        self.install_settle();
        let title = self.eval("document.title").unwrap_or_default().trim_matches('"').to_string();
        let final_url = self.eval("location.href").unwrap_or_default().trim_matches('"').to_string();
        self.current_url = final_url.clone();
        let rich = self.rich_digest(4000)?;
        Ok((final_url, title, rich))
    }

    /// Install the async-work watcher (idempotent, best-effort).
    fn install_settle(&mut self) {
        let _ = self.eval(SETTLE_INSTALL_JS);
    }

    /// Wait for the page to finish reacting to the action we just performed,
    /// in two phases:
    ///   1. up to `work_ms`, wait for work to START — an in-flight fetch/XHR or
    ///      a DOM change past the baseline. A submit whose confirmation arrives
    ///      a second later (network round-trip, or a plain `setTimeout` render
    ///      with no request at all) is caught here; a genuinely dead click pays
    ///      this wait once and reports "nothing happened".
    ///   2. then wait for quiet — no requests in flight and no change for
    ///      `quiet_ms` — capped by `max_ms` overall.
    /// Returns whether any work was observed.
    /// Wait for the page to finish reacting — without waiting on a page that
    /// is merely alive.
    ///
    /// The rule used to be "no DOM mutation for `quiet_ms`", but the observer
    /// watches `class` and `style` across the whole document, so any site with
    /// an animation or a polling timer never goes quiet and every action spent
    /// the entire budget. Measured against a lesson page: clicking a control
    /// that changed nothing took 6.5 seconds, and a plain choice 1.5.
    ///
    /// Requests in flight are the signal that survives an animated page, so
    /// they decide how long to wait. Mutations only answer "did anything
    /// happen at all", which is worth a brief look when nothing went out on
    /// the wire. Returning early on a page that is still working is not the
    /// hazard it sounds like: the caller re-reads the page, and the repeat
    /// gate above tolerates a single unchanged result for exactly this reason.
    fn wait_settled(&mut self, max_ms: u64, _quiet_ms: u64) -> bool {
        // Each poll costs an eval round trip of its own — on the order of a
        // hundred milliseconds — so a budget counted in accumulated SLEEP runs
        // about three times longer in real time than it reads. Budgets here
        // are wall clock, and the sleep is short because the round trip
        // already paces the loop.
        const POLL: u64 = 40;
        /// Nothing on the wire: how long to keep looking before calling it done.
        const IDLE_LOOK: u64 = 700;
        /// After the last response, time for the render it causes.
        const RENDER_GRACE: u64 = 180;
        /// Quiet means nothing pending AND no new mutations. Once the page has
        /// been quiet this long it is done, and waiting out the rest of the
        /// look window is pure latency on every click the agent makes.
        const QUIET_EXIT: u64 = 130;
        /// ...but look at least this long first, or a click whose effect starts
        /// on the next frame reads as "already finished".
        const MIN_LOOK: u64 = 200;
        fn parse(s: &str) -> Option<(u32, u64, u64)> {
            let mut it = s.split(':');
            Some((
                it.next()?.parse().ok()?,
                it.next()?.parse().ok()?,
                it.next()?.parse().ok()?,
            ))
        }
        let raw = self.eval(SETTLE_POLL_JS).unwrap_or_default();
        let base = raw.trim_matches('"').to_string();
        if base == "none" {
            self.install_settle();
            return true;
        }
        let base_n = parse(&base).map(|(_, _, n)| n).unwrap_or(0);
        let started_at = std::time::Instant::now();
        let mut saw_net = false;
        let mut changed = false;
        let mut last_n = base_n;
        let mut quiet_since: Option<std::time::Instant> = None;
        while (started_at.elapsed().as_millis() as u64) < max_ms {
            std::thread::sleep(Duration::from_millis(POLL));
            self.pump_pending();
            let raw = self.eval(SETTLE_POLL_JS).unwrap_or_default();
            let s = raw.trim_matches('"');
            if s == "none" {
                self.install_settle();
                return true;
            }
            let Some((pending, _since, n)) = parse(s) else { break };
            if pending > 0 {
                saw_net = true;
            }
            if n > base_n {
                changed = true;
            }
            if saw_net && pending == 0 {
                std::thread::sleep(Duration::from_millis(RENDER_GRACE));
                return true;
            }
            // Nothing on the wire. Leave as soon as the DOM stops moving —
            // most clicks are local (a selection, a class, a panel) and finish
            // in a frame or two, and sitting out the whole look window put the
            // better part of a second on every one of them. Mutations only
            // hold the door open, never extend it: a page with a carousel or a
            // clock never goes quiet, so the look window still ends it.
            if !saw_net {
                if n == last_n {
                    let q = *quiet_since.get_or_insert_with(std::time::Instant::now);
                    let elapsed = started_at.elapsed().as_millis() as u64;
                    if elapsed >= MIN_LOOK && (q.elapsed().as_millis() as u64) >= QUIET_EXIT {
                        break;
                    }
                } else {
                    quiet_since = None;
                }
                if (started_at.elapsed().as_millis() as u64) >= IDLE_LOOK {
                    break;
                }
            }
            last_n = n;
        }
        changed
    }

    /// A compact digest of the page's interactive elements (used after navigate
    /// and by the `browser_read` tool). Keeps the model grounded in what's
    /// actually there, so it clicks by real visible text instead of guessing
    /// selectors.
    fn digest(&mut self) -> Result<String, String> {
        self.eval(&digest_js())
    }

    /// The text substitute for a screenshot: the page's VISIBLE TEXT plus the
    /// interactive-element list (with current input values). Lets the model
    /// read everything that just appeared — dynamic rules, messages, results —
    /// as text, so it doesn't have to screenshot to "see" the page.
    fn rich_digest(&mut self, text_cap: usize) -> Result<String, String> {
        // All three parts in ONE evaluation. Every `eval` is a round trip to
        // the browser — on the order of a hundred milliseconds — and this
        // digest is what every page tool returns, so gathering it in three
        // calls made each click, type and scroll pay three of them. The parts
        // are joined with a control-character sentinel rather than JSON so the
        // page text passes through byte for byte (a region that legitimately
        // starts with a quote keeps it), and each part is guarded on its own so
        // one of them failing still leaves the others, as separate calls did.
        //
        // The page text is capped from the top. On a long page that hides
        // whatever appeared next to the control the agent just used, so the
        // third part adds the enclosing region's text — that is where
        // confirmations and inline errors live. Only when the cap actually bit;
        // the region script itself returns nothing until an interaction has
        // set the anchor.
        let js = format!(
            "(function(){{var g=function(f){{try{{return f()||'';}}catch(e){{return '';}}}};\
             var t=g(function(){{return {text};}});\
             var e=g(function(){{return {els};}});\
             var n=t.length>{cap}?g(function(){{return {near};}}):'';\
             return [t,e,n].join('{SEP_JS}');}})()",
            text = page_text_js(text_cap),
            els = digest_js(),
            cap = text_cap,
            near = NEAR_TEXT_JS.replace("__NEARCAP__", "1400"),
        );
        let raw = self.eval(&js).unwrap_or_default();
        let mut parts = raw.split(DIGEST_SEP);
        let text = parts.next().unwrap_or_default();
        let els = parts.next().unwrap_or_default();
        let region = parts.next().unwrap_or_default().trim();
        let mut near = String::new();
        if region.len() > 1 {
            near = trf!(
                "\n\n刚操作的元素所在区域(长页面已截断,这里是重点):\n{}",
                "\n\nThe region around the element you just used (the page text above was truncated — this is the part that matters):\n{}",
                region
            );
        }
        Ok(trf!(
            "页面可见文字(替代截图,直接读这个):\n{}{}\n\n可交互元素(按可见文字点击/向这些输入):\n{}",
            "Visible text (read this instead of screenshotting):\n{}{}\n\nInteractive elements (click by text / type into these):\n{}",
            text,
            near,
            els
        ))
    }

    /// Fields whose native validation is blocking the form the agent just used
    /// (empty `required`, malformed `type=email`, …) — invisible to page text.
    fn form_blockers(&mut self) -> Option<String> {
        let t = self.eval(FORM_BLOCKERS_JS).unwrap_or_default();
        (!t.trim().is_empty()).then(|| t.trim().to_string())
    }

    /// Full-page screenshot. First auto-scrolls through the page (triggering
    /// lazy-loaded images/sections), then returns to the top and captures the
    /// whole document — so nothing below the fold is missed or blank.
    fn screenshot(&mut self) -> Result<Vec<u8>, String> {
        // Order matters: freeze first, so reveals triggered by the scroll
        // land at their end state instantly instead of racing the capture.
        let _ = self.eval(FREEZE_ANIMATIONS_JS);
        let _ = self.eval(AUTOSCROLL_JS); // best-effort; ignore if it errors
        // The scroll pulls in lazy images and staged inserts; wait for those to
        // finish rather than for a fixed guess. CSS is already frozen, so this
        // only has to cover fetches and setTimeout chains.
        self.wait_settled(1500, 250);
        // "Full page" has to stop somewhere. An endless-scroll feed or a game
        // map is tens of thousands of pixels tall: decoding one costs hundreds
        // of megabytes before a single tile exists, and it splits into dozens
        // of pictures no model can be shown — a screenshot of a page like that
        // took the whole app down rather than answering. Capture a bounded
        // window from wherever the page is now, which is the part being worked
        // on; the rest is reachable by scrolling and capturing again.
        let m = self
            .eval("[Math.round(scrollY),innerWidth,Math.round(document.documentElement.scrollHeight)].join(',')")
            .unwrap_or_default();
        let n: Vec<f64> = m
            .trim_matches('"')
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        if let [top, width, doc_h] = n[..] {
            // Tiling cuts at width*0.72; six of those is already more than any
            // model is shown in one prompt.
            let cap = (width * 0.72 * 6.0).max(2000.0);
            if doc_h > cap && width > 0.0 {
                let y = top.min((doc_h - cap).max(0.0));
                return self.capture_clip(0.0, y, width, cap);
            }
        }
        self.capture(true)
    }

    /// Capture one rectangle of the page (page coordinates, CSS pixels).
    fn capture_clip(&mut self, x: f64, y: f64, w: f64, h: f64) -> Result<Vec<u8>, String> {
        let sid = self.session_id.clone();
        let r = self.call(
            Some(&sid),
            "Page.captureScreenshot",
            json!({
                "format": "png",
                "captureBeyondViewport": true,
                "clip": {"x": x, "y": y, "width": w, "height": h, "scale": 1},
            }),
        )?;
        let b64 = r["data"].as_str().ok_or("截图无数据 (no screenshot data)")?;
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("截图解码失败 (screenshot decode failed): {e}"))
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
        // Lazy-loaded sections arrive over the wire, so wait for the wire
        // rather than for a fixed guess: a slow one gets the time it needs and
        // a page with nothing to load costs a couple of polls.
        self.wait_settled(1500, 250);
        let pos = self
            .eval("Math.round(window.scrollY)+' / '+Math.round(document.body.scrollHeight)")
            .unwrap_or_default();
        // Surface any newly-revealed text/elements (lazy-load) so the model
        // reads what appeared without a screenshot.
        let rich = self.rich_digest(3500).unwrap_or_default();
        Ok(trf!(
            "已滚动,位置 scrollY: {}\n\n{}",
            "Scrolled, scrollY: {}\n\n{}",
            pos.trim_matches('"'),
            rich
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

    /// Dispatch a REAL mouse click at viewport coordinates over CDP — the same
    /// event stream a human produces (move → press → release), so components
    /// that listen for mousedown/pointer events (React/Vue widgets, custom
    /// dropdowns) respond where a synthetic `el.click()` silently did nothing.
    fn mouse_click(&mut self, x: f64, y: f64) -> Result<(), String> {
        let sid = self.session_id.clone();
        self.call(
            Some(&sid),
            "Input.dispatchMouseEvent",
            json!({"type": "mouseMoved", "x": x, "y": y, "button": "none"}),
        )?;
        self.call(
            Some(&sid),
            "Input.dispatchMouseEvent",
            json!({"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1}),
        )?;
        self.call(
            Some(&sid),
            "Input.dispatchMouseEvent",
            json!({"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1}),
        )?;
        Ok(())
    }

    /// Insert text the way the browser does for a keystroke, over CDP.
    ///
    /// Assigning `el.value` and firing an `input` event does not reach a React
    /// component: React installs a value tracker on the node, so a direct
    /// assignment is recorded as already-seen and `onChange` never runs. The
    /// page then still believes the field is empty — its submit stays disabled
    /// and an agent retypes forever. `Input.insertText` goes through the same
    /// path a keypress does, which no framework can miss.
    fn insert_text(&mut self, text: &str) -> Result<(), String> {
        let sid = self.session_id.clone();
        self.call(Some(&sid), "Input.insertText", json!({ "text": text }))?;
        Ok(())
    }

    /// Where the element found by the last locate actually is, measured as
    /// late as possible and checked against what is really under the point.
    ///
    /// Measuring and clicking are two round trips. On a page that scrolls or
    /// animates in between — a card sliding in, a popover settling, a long
    /// list smooth-scrolling to the target — the element has moved by the time
    /// the mouse event is dispatched, so the click lands on nothing and still
    /// reports success. The agent then sees an unchanged page and tries again
    /// forever. Re-measure just before dispatching, confirm the point really
    /// hits the element, and give an animation a couple of chances to finish.
    /// Dispatch the click and check that the intended element actually got it.
    ///
    /// A coordinate measured a moment ago can be wrong by the time the event
    /// is sent: the page reflows, a panel finishes animating in, a scroll
    /// container snaps back. The event then lands on whatever now occupies
    /// that spot, and the tool reports a success that did something else
    /// entirely — the failure that leaves an agent clicking the same button
    /// forever because it is told, every time, that the click worked. So arm
    /// the target with a listener, click, and ask whether it fired.
    fn click_confirmed(&mut self, pre_blockers: &mut Option<String>) -> Result<&'static str, String> {
        /// Records both whether the target got the click and what did, so a
        /// failure can say what is in the way instead of just "no".
        const ARM: &str = r#"(function(){
            var e=window.__chatyLast;
            if(!e||!e.isConnected) return 'GONE';
            window.__chatyHit=0; window.__chatyGot='';
            var on=function(){ window.__chatyHit=1; e.removeEventListener('click',on,true); };
            e.addEventListener('click',on,true);
            var doc=function(ev){
                var t=ev.target||{};
                var c=(t.className&&(t.className+'').split(' ')[0])||'';
                window.__chatyGot=(t.tagName||'?')+(c?('.'+c):'');
                document.removeEventListener('click',doc,true);
            };
            document.addEventListener('click',doc,true);
            return 'ARMED';
        })()"#;
        // A full navigation throws the page's globals away, so `undefined` is
        // not a miss — it is the click having worked well enough to leave.
        const CHECK: &str = r#"(function(){
            if(typeof window.__chatyHit==='undefined') return 'NAV';
            return window.__chatyHit?'HIT':('MISS:'+(window.__chatyGot||'nothing'));
        })()"#;
        let mut missed = String::new();
        for attempt in 0..3 {
            let Some((x, y)) = self.settled_click_point() else {
                return Ok("NOT_FOUND");
            };
            if self.eval(ARM).unwrap_or_default().contains("GONE") {
                return Ok("NOT_FOUND");
            }
            if attempt == 0 {
                // Validity BEFORE the click decides whether a submit could
                // even fire. Checking after is wrong: a successful submit
                // often calls form.reset(), which makes required fields
                // empty (invalid) again and would fake a "blocked" report.
                *pre_blockers = self.form_blockers();
            }
            self.mouse_click(x, y)?;
            let got = self.eval(CHECK).unwrap_or_default();
            let got = got.trim_matches('"');
            if got == "HIT" || got == "NAV" {
                return Ok("OK");
            }
            missed = got.trim_start_matches("MISS:").to_string();
        }
        self.last_click_blocker = (!missed.is_empty()).then_some(missed);
        Ok("BLOCKED")
    }

    fn settled_click_point(&mut self) -> Option<(f64, f64)> {
        const MEASURE: &str = r#"(function(){
            var e=window.__chatyLast; if(!e) return 'NOT_FOUND';
            var r=e.getBoundingClientRect();
            if(r.width<2||r.height<2) return 'NOT_FOUND';
            if(r.top<0||r.bottom>innerHeight||r.left<0||r.right>innerWidth){
                // 'instant': a site with scroll-behavior:smooth would otherwise
                // hand back the rect from before the scroll had happened.
                try{ e.scrollIntoView({block:'center',behavior:'instant'}); }
                catch(_){ e.scrollIntoView({block:'center'}); }
                r=e.getBoundingClientRect();
            }
            var x=Math.round(r.left+r.width/2), y=Math.round(r.top+r.height/2);
            var at=document.elementFromPoint(x,y);
            var ok=!!at&&(at===e||e.contains(at)||at.contains(e));
            return JSON.stringify({x:x,y:y,ok:ok});
        })()"#;
        // Two things have to hold before a coordinate is worth clicking: the
        // point must hit the element, and it must have stopped moving. A
        // popover, modal or toast that is still animating in reports a
        // perfectly valid rect that is stale by the time the event is
        // dispatched — the click then lands on whatever occupies that spot,
        // which is how a "successful" click ends up dismissing the very panel
        // it was aiming at. Same point twice in a row means the motion is over.
        const TOL: f64 = 2.0;
        let mut prev: Option<(f64, f64)> = None;
        let mut moving: Option<(f64, f64)> = None;
        for attempt in 0..6 {
            std::thread::sleep(Duration::from_millis(if attempt == 0 { 60 } else { 90 }));
            let Ok(raw) = self.eval(MEASURE) else { continue };
            let raw = raw.trim_matches('"').replace("\\\"", "\"");
            let Ok(v) = serde_json::from_str::<Value>(&raw) else { continue };
            let (Some(x), Some(y)) = (v.get("x").and_then(|n| n.as_f64()), v.get("y").and_then(|n| n.as_f64()))
            else { continue };
            let on_target = v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false);
            if on_target {
                if let Some((px, py)) = prev {
                    if (px - x).abs() <= TOL && (py - y).abs() <= TOL {
                        return Some((x, y));
                    }
                }
                moving = Some((x, y));
            }
            prev = Some((x, y));
        }
        // Never fall back to a point the hit test rejected. Clicking a
        // coordinate known to belong to something else is worse than saying so:
        // it reports success while doing something the agent never asked for,
        // and the agent, told it worked, has no reason to try another route.
        // A target that hits but never settles is still worth a try.
        moving
    }

    /// Click by visible text or CSS selector, then hand back the fresh page
    /// state (visible text + elements). Single-step wrapper over `click_once`.
    fn click(&mut self, selector: Option<&str>, text: Option<&str>) -> Result<String, String> {
        let label = self.click_once(selector, text)?;
        let where_ = self.eval("document.title+' — '+location.href").unwrap_or_default();
        let rich = self.rich_digest(3500).unwrap_or_default();
        Ok(trf!(
            "已点击:{}\n点击后当前页面:{}\n\n{}",
            "Clicked: {}\nPage after the click: {}\n\n{}",
            label,
            where_.trim_matches('"'),
            rich
        ))
    }

    /// Click a SEQUENCE of targets in one call (form flows, multi-step wizards)
    /// — real mouse events with a short settle between each. Stops at the first
    /// failure and reports how far it got; on success returns ONE fresh page
    /// state at the end (not after every click) to keep the result compact.
    fn click_seq(&mut self, steps: &[(Option<String>, Option<String>)]) -> Result<String, String> {
        let mut done: Vec<String> = Vec::new();
        for (i, (sel, text)) in steps.iter().enumerate() {
            match self.click_once(sel.as_deref(), text.as_deref()) {
                Ok(label) => done.push(label),
                Err(e) => {
                    let rich = self.rich_digest(3000).unwrap_or_default();
                    return Ok(trf!(
                        "顺序点击:成功 {} 步 [{}],第 {} 步失败:{}\n\n{}",
                        "Click sequence: {} succeeded [{}], step {} failed: {}\n\n{}",
                        done.len(),
                        done.join(" → "),
                        i + 1,
                        e,
                        rich
                    ));
                }
            }
        }
        let rich = self.rich_digest(3500).unwrap_or_default();
        Ok(trf!(
            "已依次点击 {} 处:{}\n\n{}",
            "Clicked {} targets in order: {}\n\n{}",
            done.len(),
            done.join(" → "),
            rich
        ))
    }

    /// Locate + real-mouse-click a single element. Returns the matched label on
    /// success. Two-phase: JS locates the element (exact visible-text match
    /// first, then prefix, then substring — visible elements only, so "Save"
    /// hits the button labelled exactly "Save", not "Save All"), scrolls it into
    /// view and reports its center; then a REAL mouse click lands on that point.
    fn click_once(&mut self, selector: Option<&str>, text: Option<&str>) -> Result<String, String> {
        let js = if let Some(txt) = text.filter(|t| !t.is_empty()) {
            format!(
                r#"(function(){{
                    var t={txt}.trim().replace(/\s+/g,' ').toLowerCase();
                    var els=[].slice.call(document.querySelectorAll("a,button,[role=button],[role=link],[role=menuitem],[role=tab],[role=radio],[role=checkbox],[role=switch],[role=option],[role=menuitemradio],[role=menuitemcheckbox],[role=treeitem],input,textarea,select,summary,[tabindex],[contenteditable=''],[contenteditable=true],input[type=submit],input[type=button],[onclick],label"));
                    function vis(e){{var r=e.getBoundingClientRect();if(r.width<2||r.height<2)return false;var s=getComputedStyle(e);return s.visibility!=='hidden'&&s.display!=='none'&&s.pointerEvents!=='none';}}
                    // Match the visible text OR the aria-label: the page
                    // digest surfaces aria-labels for glyph-only buttons, so
                    // the model clicks by the label it was shown.
                    function txts(e){{var a=[];function ad(s){{s=((s||'')+'').trim().replace(/\s+/g,' ').toLowerCase();if(s)a.push(s);}}ad(e.innerText||e.textContent||e.value);ad(e.getAttribute('aria-label'));return a;}}
                    // Priority when several elements share the same text (e.g. a
                    // nav "Login" LINK vs the form's "Login" SUBMIT button): the
                    // actionable control wins over a plain link, so clicking
                    // "Login"/"Submit"/"Search" fires the form, not a same-named
                    // link that just reloads.
                    function rank(e){{var tag=e.tagName.toLowerCase(),ty=(e.type||'').toLowerCase();
                        if(ty==='submit')return 0;
                        if(tag==='button'||e.getAttribute('role')==='button'||ty==='button')return 1;
                        return 2;}}
                    var cand=els.filter(vis);
                    // How much text an element carries beyond what was asked
                    // for. Anything WRAPPING the control matches the same
                    // substring and carries the whole panel with it, so
                    // without this the click lands on a container and the
                    // control the agent named never hears about it.
                    function tight(e){{var b=1e9;txts(e).forEach(function(s){{
                        if(s.indexOf(t)>=0&&s.length<b)b=s.length;}});return b;}}
                    function pick(pred){{var m=cand.filter(pred);if(!m.length)return null;
                        m.sort(function(a,b){{return (tight(a)-tight(b))||(rank(a)-rank(b));}});return m[0];}}
                    // A choice in a list usually wears a badge the page put
                    // there — "1", "2)", "3." — while the page TEXT the agent
                    // read shows only the words. Comparing with the badge off
                    // both sides makes "jolie" find the row labelled "2 jolie",
                    // and "2 jolie" find a row labelled just "jolie".
                    function bare(s){{return s.replace(/^\s*\d{{1,2}}\s*[.)\]:、,]?\s+/,'');}}
                    var tb=bare(t);
                    var hit=pick(function(e){{return txts(e).some(function(s){{return s===t;}});}})
                          ||pick(function(e){{return txts(e).some(function(s){{return bare(s)===tb;}});}})
                          ||pick(function(e){{return txts(e).some(function(s){{return s.lastIndexOf(t,0)===0;}});}})
                          ||pick(function(e){{return txts(e).some(function(s){{return s.indexOf(t)>=0;}});}});
                    if(!hit)return 'NOT_FOUND';
                    // A disabled control swallows the click and changes
                    // nothing. Reported as a success it reads as "it worked",
                    // and the agent clicks it again, and again.
                    if(hit.disabled||hit.getAttribute('aria-disabled')==='true') return 'DISABLED';
                    window.__chatyLast=hit; // anchor for the "near this element" digest
                    // Only scroll when the target is actually out of view, and
                    // never smoothly: scrolling a page that does not need it
                    // sets off scroll-snap and reveal animations, which move
                    // the target out from under the coordinate we are about to
                    // click. Left alone, a visible element stays put.
                    var rr=hit.getBoundingClientRect();
                    if(rr.top<0||rr.bottom>innerHeight||rr.left<0||rr.right>innerWidth){{
                        try{{ hit.scrollIntoView({{block:'center',behavior:'instant'}}); }}
                        catch(_){{ hit.scrollIntoView({{block:'center'}}); }}
                    }}
                    return 'FOUND';
                }})()"#,
                txt = json!(txt)
            )
        } else {
            let sel = selector.unwrap_or("");
            format!(
                r#"(function(){{
                    var el=document.querySelector({sel});
                    if(!el)return 'NOT_FOUND';
                    if(el.tagName==='SELECT')return 'IS_SELECT';
                    window.__chatyLast=el;
                    el.scrollIntoView({{block:'center'}});
                    return 'FOUND';
                }})()"#,
                sel = json!(sel)
            )
        };
        let label = text.filter(|t| !t.is_empty()).unwrap_or_else(|| selector.unwrap_or(""));
        if label.is_empty() {
            let d = self.digest().unwrap_or_default();
            return Err(trf!(
                "browser_click 需要 \"text\"(优先,按可见文字)或 \"selector\"。当前页面可点击的元素:\n{}",
                "browser_click needs \"text\" (preferred — the visible label) or \"selector\". Clickable elements on this page:\n{}",
                d
            ));
        }
        // A navigation left over from the previous tool call must not eat this
        // click. Ask the page whether it is ready instead of assuming a fixed
        // wait: ready is the normal case and now costs one cheap round trip.
        for _ in 0..8 {
            if self.eval("document.readyState").unwrap_or_default().contains("complete") {
                break;
            }
            std::thread::sleep(Duration::from_millis(80));
        }
        self.install_settle(); // so we can tell when this click's work finishes
        let mut pre_blockers: Option<String> = None;
        let found = self.eval(&js)?;
        let found = found.trim_matches('"');
        let clicked = if found == "NOT_FOUND" {
            "NOT_FOUND"
        } else if found == "IS_SELECT" {
            "IS_SELECT"
        } else if found == "DISABLED" {
            "DISABLED"
        } else {
            self.click_confirmed(&mut pre_blockers)?
        };
        match clicked {
            "OK" => {
                // A click may navigate (submit/login/next). Let it settle, then
                // if a navigation is in flight, wait for the destination to be
                // ready — otherwise the digest would show the OLD page and the
                // model would wrongly re-click (over-clicking Login/Next).
                // Just enough for the event to reach the page's handlers; the
                // real waiting is `wait_settled` below, which has its own floor.
                std::thread::sleep(Duration::from_millis(40));
                self.pump_pending();
                for _ in 0..12 {
                    let ready = self
                        .eval("document.readyState")
                        .unwrap_or_default();
                    if ready.contains("complete") {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(150));
                    self.pump_pending();
                }
                // Same-document work (AJAX submit → "Thank you", SPA route
                // change, spinner → result) never touches readyState, so wait
                // for the page to actually go quiet before the caller reads it.
                self.wait_settled(6000, 400);
                // The clicked control sat in a form that already failed native
                // validation, so no submit could have fired — name the fields.
                // (Not gated on "did the page react?": that is a false positive
                // on sites with scroll-reveal animations, whose class toggles
                // fire on every scroll.)
                if let Some(bad) = pre_blockers {
                    return Ok(trf!(
                        "{}(注意:该表单未通过浏览器校验,提交没有真正发出。先修正这些字段再点提交 → {})",
                        "{} (note: this form fails browser validation, so the submit never fired. Fix these fields, then click submit again → {})",
                        label,
                        bad
                    ));
                }
                Ok(label.to_string())
            }
            "DISABLED" => {
                let rich = self.rich_digest(2500).unwrap_or_default();
                Err(trf!(
                    "\"{}\" 现在是禁用状态,点它不会有任何反应。它通常要等某个前置条件满足才会变亮——先把该填的填完/该选的选上(下面是当前页面状态),再点它:\n\n{}",
                    "\"{}\" is disabled right now — clicking it does nothing. Something has to happen before it goes live: finish the field or choice it is waiting on (the page state is below), then click it:\n\n{}",
                    label,
                    rich
                ))
            }
            "BLOCKED" => {
                let what = self.last_click_blocker.take().unwrap_or_default();
                let d = self.digest().unwrap_or_default();
                Err(trf!(
                    "点了 \"{}\",但事件没有到达它{}。它多半被浮层/弹窗/Cookie 横幅挡住了,或者页面在点击时还在动。先关掉挡住的东西,或换个目标:\n{}",
                    "Clicked \"{}\" but the event never reached it{}. Something is covering it — an overlay, dialog or cookie banner — or the page was still moving. Dismiss what's on top, or pick a different target:\n{}",
                    label,
                    if what.is_empty() { String::new() } else { trf!("(实际收到点击的是 {})", " (it landed on {} instead)", what) },
                    d
                ))
            }
            "IS_SELECT" => Err(trf!(
                "这是下拉框,点击不会展开选项。改用 browser_type 选择:{{\"selector\":\"{}\",\"text\":\"<选项的可见文字>\"}}",
                "That's a <select> — clicking won't open it. Choose with browser_type instead: {{\"selector\":\"{}\",\"text\":\"<the option's visible label>\"}}",
                label
            )),
            _ => {
                let d = self.digest().unwrap_or_default();
                Err(trf!(
                    "未找到可点击的元素:{}。用下面清单里的准确文字重试:\n{}",
                    "No clickable element matched: {}. Retry with the exact text from this list:\n{}",
                    label,
                    d
                ))
            }
        }
    }

    /// Type into a field, then hand back the fresh page state (validation /
    /// rules that just appeared). Single-step wrapper over `type_once`.
    fn type_text(&mut self, selector: Option<&str>, text: &str, label: Option<&str>) -> Result<String, String> {
        let what = self.type_once(selector, text, label)?;
        let rich = self.rich_digest(3500).unwrap_or_default();
        Ok(trf!("已输入 → {}\n\n{}", "Typed → {}\n\n{}", what, rich))
    }

    /// Fill a SEQUENCE of fields in one call (a whole form at once). Stops at the
    /// first failure; returns ONE fresh page state at the end.
    fn type_seq(&mut self, steps: &[(Option<String>, Option<String>, String)]) -> Result<String, String> {
        let mut done: Vec<String> = Vec::new();
        for (i, (sel, label, text)) in steps.iter().enumerate() {
            match self.type_once(sel.as_deref(), text, label.as_deref()) {
                Ok(what) => done.push(what),
                Err(e) => {
                    let rich = self.rich_digest(3000).unwrap_or_default();
                    return Ok(trf!(
                        "顺序输入:成功 {} 个字段 [{}],第 {} 个失败:{}\n\n{}",
                        "Type sequence: {} fields filled [{}], field {} failed: {}\n\n{}",
                        done.len(),
                        done.join(", "),
                        i + 1,
                        e,
                        rich
                    ));
                }
            }
        }
        let rich = self.rich_digest(3500).unwrap_or_default();
        Ok(trf!(
            "已依次填写 {} 个字段:{}\n\n{}",
            "Filled {} fields in order: {}\n\n{}",
            done.len(),
            done.join(", "),
            rich
        ))
    }

    /// Set one field's value (no digest). Returns the field label on success.
    /// Matches by CSS selector, label/placeholder text, or the page's single
    /// obvious text field; supports contenteditable.
    fn type_once(&mut self, selector: Option<&str>, text: &str, label: Option<&str>) -> Result<String, String> {
        // The finder also considers contenteditable regions (rich editors,
        // some game/note inputs) alongside form fields.
        let finder = if let Some(sel) = selector.filter(|s| !s.is_empty()) {
            format!("document.querySelector({})", json!(sel))
        } else if let Some(lbl) = label.filter(|l| !l.is_empty()) {
            format!(
                r#"(function(){{
                    var l={lbl}.trim().toLowerCase();
                    var fields=[].slice.call(document.querySelectorAll("input,textarea,select,[contenteditable=''],[contenteditable=true]"));
                    return fields.find(function(f){{
                        var hints=[f.placeholder,f.name,f.id,f.getAttribute('aria-label')];
                        var lab=f.labels&&f.labels[0];if(lab)hints.push(lab.textContent);
                        return hints.some(function(h){{return h&&h.trim().toLowerCase().includes(l);}});
                    }})||null;
                }})()"#,
                lbl = json!(lbl)
            )
        } else {
            // No selector/label: the single obvious text field (many single-input
            // pages/games — one textarea/input/editable — need no locator).
            r#"(function(){var f=[].slice.call(document.querySelectorAll("textarea,input[type=text],input:not([type]),[contenteditable=''],[contenteditable=true]")).filter(function(e){var r=e.getBoundingClientRect();return r.width>1&&r.height>1;});return f.length===1?f[0]:null;})()"#.to_string()
        };
        let js = format!(
            r#"(function(){{
                var el={finder};
                if(!el)return 'NOT_FOUND';
                window.__chatyLast=el; // anchor for the "near this element" digest
                el.scrollIntoView({{block:'center'}});el.focus();
                var want=({val}+'').trim().toLowerCase();
                if(el.tagName==='SELECT'){{
                    // Dropdown: pick the option whose visible text or value
                    // matches (exact first, then prefix, then substring). You
                    // can't "type" into a <select>.
                    var opts=[].slice.call(el.options);
                    function t(o){{return (o.textContent||'').trim().toLowerCase();}}
                    var hit=opts.find(function(o){{return t(o)===want||(o.value||'').toLowerCase()===want;}})
                          ||opts.find(function(o){{return t(o).lastIndexOf(want,0)===0;}})
                          ||opts.find(function(o){{return want&&t(o).indexOf(want)>=0;}});
                    if(!hit)return 'NO_OPTION:'+opts.map(function(o){{return t(o);}}).filter(Boolean).slice(0,20).join(' | ');
                    el.value=hit.value;
                    el.dispatchEvent(new Event('input',{{bubbles:true}}));
                    el.dispatchEvent(new Event('change',{{bubbles:true}}));
                }} else if(el.isContentEditable){{
                    // Select what is there so the insertion replaces it.
                    var rg=document.createRange();rg.selectNodeContents(el);
                    var sl=window.getSelection();sl.removeAllRanges();sl.addRange(rg);
                    return 'FOCUSED';
                }} else {{
                    try{{el.setSelectionRange(0,(el.value||'').length);}}catch(_){{try{{el.select();}}catch(__){{}}}}
                    return 'FOCUSED';
                }}
                return 'OK';
            }})()"#,
            val = json!(text)
        );
        let what = selector.or(label).unwrap_or("(the page's single text field)");
        let r = self.eval(&js)?;
        let r = r.trim_matches('"');
        if r == "FOCUSED" {
            self.insert_text(text)?;
        }
        if r == "OK" || r == "FOCUSED" {
            std::thread::sleep(Duration::from_millis(250));
            self.pump_pending();
            // Typing can trigger async validation / autocomplete — let it land
            // so the digest shows the rule the model just tripped.
            self.wait_settled(2500, 300);
            Ok(what.to_string())
        } else if let Some(opts) = r.strip_prefix("NO_OPTION:") {
            // A <select> was found but no option matched — list the options so
            // the model retries with an exact one.
            Err(trf!(
                "下拉框「{}」里没有匹配「{}」的选项。可选项:{}",
                "Dropdown \"{}\" has no option matching \"{}\". Options: {}",
                what,
                text,
                opts.replace("\\|", "|")
            ))
        } else {
            let d = self.digest().unwrap_or_default();
            Err(trf!(
                "未找到输入框:{}。可用输入框见清单:\n{}",
                "No input field matched: {}. Available fields:\n{}",
                what,
                d
            ))
        }
    }

    fn kill(&mut self) {
        let Some(child) = self.child.as_mut() else {
            // An adopted browser is not ours to signal — ask it to close
            // itself the way any CDP client would.
            let _ = self.call(None, "Browser.close", json!({}));
            return;
        };
        CHROME_PID
            .compare_exchange(
                child.id(),
                0,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .ok();
        let _ = child.kill();
        let _ = child.wait();
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
    tx.send(build(reply)).map_err(|_| trf!("浏览器已关闭", "the browser is closed"))?;
    rx.recv().map_err(|_| "浏览器无响应 (browser did not respond)".to_string())?
}

pub fn navigate(url: &str) -> Result<String, String> {
    let ws = crate::agent::agent_get_workspace().map(std::path::PathBuf::from);
    let url = normalize_url(url, ws.as_deref())?;
    dispatch(|reply| BrowserCmd::Navigate { url, reply })
}

pub fn refresh() -> Result<String, String> {
    dispatch(|reply| BrowserCmd::Refresh { reply })
}

/// A capture only makes sense once a page is loaded. Models reach for
/// browser_screenshot to "verify" NATIVE app windows (session audit: a
/// calculator delivery tried it as a system-level screenshot) — the lazy
/// blank session must teach, not hand back an empty white capture.
pub(crate) fn page_loaded(url: &str) -> bool {
    !(url.is_empty() || url == "about:blank")
}

fn no_page_error() -> String {
    crate::agent::tr(
        "浏览器还没有打开任何页面。浏览器截图/快照只能拍到内嵌浏览器里的网页,拍不到系统屏幕或原生应用窗口——原生 GUI 的验证用「启动 + 存活检查」(见 mac-app 技能);网页则先 browser_navigate 打开页面再截。",
        "The browser has no page open. Browser screenshot/snapshot capture ONLY the embedded browser's web page — never the system screen or native app windows. Verify a native GUI with a launch + stay-alive check (see the mac-app skill); for web pages, browser_navigate first, then capture.",
    )
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

/// Click a sequence of targets in one call: Vec of (selector, text).
pub fn click_seq(steps: Vec<(Option<String>, Option<String>)>) -> Result<String, String> {
    dispatch(|reply| BrowserCmd::ClickSeq { steps, reply })
}

pub fn type_text(selector: Option<String>, label: Option<String>, text: String) -> Result<String, String> {
    dispatch(|reply| BrowserCmd::Type { selector, label, text, reply })
}

/// Fill a sequence of fields in one call: Vec of (selector, label, text).
pub fn type_seq(steps: Vec<(Option<String>, Option<String>, String)>) -> Result<String, String> {
    dispatch(|reply| BrowserCmd::TypeSeq { steps, reply })
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
/// Extensions that mark an input as a FILE reference, never a domain guess —
/// `index.html` must not become `https://index.html` (a DNS error the model
/// retries forever); `example.com` must keep becoming a website.
const WEB_FILE_EXTS: &[&str] = &[
    "html", "htm", "xhtml", "svg", "pdf", "png", "jpg", "jpeg", "gif", "webp", "css", "js", "mjs",
    "json", "txt", "md", "csv", "mp4", "webm", "ico",
];

fn web_file_ext(u: &str) -> bool {
    let seg = u.rsplit(['/', '\\']).next().unwrap_or(u);
    match seg.rsplit_once('.') {
        Some((_, ext)) => WEB_FILE_EXTS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

/// Scheme-less local dev hosts (`localhost:8000`, `127.0.0.1:3000/app`) speak
/// plain http — an `https://` guess dies on TLS and loops the model.
fn is_local_host(u: &str) -> bool {
    let host_port = u.split('/').next().unwrap_or("");
    let host = if host_port.starts_with('[') {
        host_port.split(']').next().map(|h| format!("{h}]")).unwrap_or_default()
    } else {
        match host_port.rsplit_once(':') {
            Some((h, p)) if !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => h.to_string(),
            _ => host_port.to_string(),
        }
    };
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "0.0.0.0" | "[::1]")
        || host.ends_with(".localhost")
}

/// file:// URL with the handful of characters that break URL parsing escaped.
fn file_url(p: &std::path::Path) -> String {
    let s = p
        .display()
        .to_string()
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    format!("file://{s}")
}

/// Bounded workspace walk for a unique basename match: the model says
/// `index.html`, the file lives at `dist/index.html` — one hit resolves it,
/// several hits produce a disambiguation error instead of a wrong guess.
fn find_by_name(ws: &std::path::Path, name: &str) -> Vec<std::path::PathBuf> {
    const SKIP: &[&str] = &[".git", "node_modules", ".venv", "__pycache__", "target"];
    let mut stack = vec![ws.to_path_buf()];
    let mut hits = Vec::new();
    let mut visited = 0usize;
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            visited += 1;
            if visited > 20_000 || hits.len() >= 5 {
                return hits;
            }
            let p = e.path();
            let fname = e.file_name();
            let fname = fname.to_string_lossy();
            if p.is_dir() {
                if !SKIP.contains(&fname.as_ref()) {
                    stack.push(p);
                }
            } else if fname == name {
                hits.push(p);
            }
        }
    }
    hits
}

/// Turn a model-supplied navigation target into a real URL. Resolution order
/// (each step preserved from the previous behavior unless noted):
/// scheme'd URLs pass through → bare localhost gets http:// → an existing
/// path (absolute or CWD-relative, the old rule) → WORKSPACE-relative (new:
/// the agent's cwd is the workspace, not the app process's) → a unique
/// basename match inside the workspace (new) → a file-looking name that
/// resolved nowhere is a plain-language error (new — it used to become
/// `https://index.html` and loop the model on DNS) → https:// guess.
fn normalize_url(u: &str, workspace: Option<&std::path::Path>) -> Result<String, String> {
    let u = u.trim();
    if u.starts_with("http://")
        || u.starts_with("https://")
        || u.starts_with("file://")
        || u.starts_with("about:")
        || u.starts_with("data:")
    {
        return Ok(u.to_string());
    }
    if is_local_host(u) {
        return Ok(format!("http://{u}"));
    }
    // An existing local file → file:// URL.
    let p = std::path::Path::new(u);
    if p.exists() {
        if let Ok(abs) = p.canonicalize() {
            return Ok(file_url(&abs));
        }
    }
    if let Some(ws) = workspace {
        if !p.is_absolute() {
            let joined = ws.join(u);
            if joined.exists() {
                if let Ok(abs) = joined.canonicalize() {
                    // `../`-escapes stay jailed: resolve only inside the workspace.
                    if abs.starts_with(ws) {
                        return Ok(file_url(&abs));
                    }
                }
            }
            if !u.contains(['/', '\\']) && web_file_ext(u) {
                let hits = find_by_name(ws, u);
                match hits.len() {
                    1 => {
                        if let Ok(abs) = hits[0].canonicalize() {
                            return Ok(file_url(&abs));
                        }
                    }
                    n if n > 1 => {
                        let shown = hits
                            .iter()
                            .filter_map(|h| h.strip_prefix(ws).ok())
                            .map(|h| h.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Err(crate::agent::tr(
                            &format!("工作区里有多个 {u}:{shown}。请用相对工作区的完整路径指明要打开哪一个。"),
                            &format!("Multiple files named {u} in the workspace: {shown}. Pass the full workspace-relative path of the one to open."),
                        ));
                    }
                    _ => {}
                }
            }
        }
    }
    if web_file_ext(u) {
        let ws_shown = workspace.map(|w| w.display().to_string()).unwrap_or_else(|| "-".into());
        return Err(crate::agent::tr(
            &format!(
                "找不到文件 {u}(工作区:{ws_shown})。请传工作区相对路径(如 dist/index.html)或绝对路径;网页地址请带 http(s)://。"
            ),
            &format!(
                "File not found: {u} (workspace: {ws_shown}). Pass a workspace-relative path (e.g. dist/index.html) or an absolute path; for websites include http(s)://."
            ),
        ));
    }
    Ok(format!("https://{u}"))
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

    /// The model-supplied navigation target resolver: relative file paths
    /// resolve against the WORKSPACE (the agent's world), never the app
    /// process's cwd — the old rule turned `index.html` into
    /// `https://index.html` and looped the model on a DNS error.
    #[test]
    fn normalize_url_resolves_files_hosts_and_teaches() {
        let ws = std::env::temp_dir().join(format!("chaty-navtest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(ws.join("dist")).unwrap();
        std::fs::create_dir_all(ws.join("src")).unwrap();
        std::fs::write(ws.join("index.html"), "<p>hi</p>").unwrap();
        std::fs::write(ws.join("dist/app.html"), "<p>app</p>").unwrap();
        std::fs::write(ws.join("dist/dup.html"), "x").unwrap();
        std::fs::write(ws.join("src/dup.html"), "x").unwrap();
        std::fs::write(ws.join("my page.html"), "x").unwrap();
        let ws = ws.canonicalize().unwrap();
        let w = Some(ws.as_path());

        // Scheme'd URLs pass through untouched.
        for u in ["https://example.com/a?b=1", "http://x.dev", "about:blank", "data:text/html,hi"] {
            assert_eq!(normalize_url(u, w).unwrap(), u);
        }
        // Bare local dev hosts speak http, not an https guess that dies on TLS.
        assert_eq!(normalize_url("localhost:8000", w).unwrap(), "http://localhost:8000");
        assert_eq!(normalize_url("127.0.0.1:3000/app", w).unwrap(), "http://127.0.0.1:3000/app");
        assert_eq!(normalize_url("app.localhost:5173", w).unwrap(), "http://app.localhost:5173");
        // Websites keep working.
        assert_eq!(normalize_url("example.com", w).unwrap(), "https://example.com");
        // Workspace-relative resolution: bare name and subdir path.
        assert_eq!(normalize_url("index.html", w).unwrap(), file_url(&ws.join("index.html")));
        assert_eq!(normalize_url("dist/app.html", w).unwrap(), file_url(&ws.join("dist/app.html")));
        // Unique basename rescue: `app.html` lives only in dist/.
        assert_eq!(normalize_url("app.html", w).unwrap(), file_url(&ws.join("dist/app.html")));
        // Ambiguous basename → a disambiguation error, not a wrong guess.
        let e = normalize_url("dup.html", w).unwrap_err();
        assert!(e.contains("dup.html"), "err should name the file: {e}");
        // Missing file-looking target → plain-language error, not https://.
        let e = normalize_url("nope.html", w).unwrap_err();
        assert!(e.contains("nope.html") && e.contains(ws.display().to_string().as_str()), "{e}");
        // `../` cannot escape the workspace jail (parent exists but is outside).
        assert!(normalize_url("../outside-escape.html", w).is_err());
        // Spaces in resolved paths are escaped for the URL.
        let got = normalize_url("my page.html", w).unwrap();
        assert!(got.contains("my%20page.html"), "{got}");
        // No workspace: files-looking names still teach instead of guessing DNS.
        assert!(normalize_url("index.html", None).is_err());
        assert_eq!(normalize_url("example.com", None).unwrap(), "https://example.com");

        let _ = std::fs::remove_dir_all(&ws);
    }

    /// The orphan sweep must claim dirs whose creator pid is gone (and
    /// malformed debris), and must NEVER claim a live process's profile —
    /// the first design pass nearly killed a concurrently-running bench's
    /// browser.
    #[test]
    #[cfg(unix)]
    fn orphan_sweep_respects_live_creators() {
        let root = std::env::temp_dir().join(format!("chaty-sweeptest-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        // A dead creator: spawn+reap a child so the pid is real but gone.
        let dead = std::process::Command::new("true").spawn().map(|mut c| {
            let pid = c.id();
            let _ = c.wait();
            pid
        });
        let dead = dead.unwrap();
        let mine = std::process::id();
        std::fs::create_dir_all(root.join(format!("chaty-cdp-{dead}-111"))).unwrap();
        std::fs::create_dir_all(root.join(format!("chaty-cdp-{mine}-222"))).unwrap();
        std::fs::create_dir_all(root.join("chaty-cdp-garbage-333")).unwrap();
        std::fs::create_dir_all(root.join("unrelated-dir")).unwrap();
        let got: Vec<String> = orphan_cdp_dirs(&root)
            .into_iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(got.contains(&format!("chaty-cdp-{dead}-111")), "dead creator must be swept: {got:?}");
        assert!(got.contains(&"chaty-cdp-garbage-333".to_string()), "malformed debris must be swept");
        assert!(!got.iter().any(|n| n.contains(&mine.to_string())), "live creator must be kept: {got:?}");
        assert!(!got.contains(&"unrelated-dir".to_string()), "non-chaty dirs are untouchable");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Windows twin of the sweep test: pid_alive must see the current
    /// process via tasklist, treat a spawned-and-reaped cmd as gone, and the
    /// dir selection must obey it. Runs only on the Windows CI job — the dev
    /// Mac can't execute this path at all.
    #[test]
    #[cfg(windows)]
    fn orphan_sweep_respects_live_creators_windows() {
        let root = std::env::temp_dir().join(format!("chaty-sweeptest-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let dead = {
            let mut c = std::process::Command::new("cmd").args(["/C", "exit"]).spawn().unwrap();
            let pid = c.id();
            let _ = c.wait();
            pid
        };
        let mine = std::process::id();
        assert!(pid_alive(mine), "tasklist must see the current process");
        std::fs::create_dir_all(root.join(format!("chaty-cdp-{dead}-111"))).unwrap();
        std::fs::create_dir_all(root.join(format!("chaty-cdp-{mine}-222"))).unwrap();
        let got: Vec<String> = orphan_cdp_dirs(&root)
            .into_iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(got.contains(&format!("chaty-cdp-{dead}-111")), "dead creator must be swept: {got:?}");
        assert!(!got.iter().any(|n| n.contains(&mine.to_string())), "live creator must be kept: {got:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Blank lazy sessions must not hand a "capture" of nothing back to a
    /// model that thinks it is taking a system screenshot.
    #[test]
    fn capture_requires_a_loaded_page() {
        assert!(!page_loaded(""));
        assert!(!page_loaded("about:blank"));
        assert!(page_loaded("http://localhost:5173/"));
        assert!(page_loaded("https://example.com"));
    }

    /// Discovery must agree with the file system: when any known browser is
    /// installed (incl. Windows per-user %LOCALAPPDATA% Chrome — the usual
    /// non-admin install this used to miss), chrome_path finds one.
    #[test]
    fn chrome_discovery_matches_filesystem() {
        #[cfg(target_os = "windows")]
        {
            let mut known: Vec<std::path::PathBuf> = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            ]
            .iter()
            .map(std::path::PathBuf::from)
            .collect();
            if let Ok(local) = std::env::var("LOCALAPPDATA") {
                known.push(std::path::PathBuf::from(format!(
                    r"{local}\Google\Chrome\Application\chrome.exe"
                )));
            }
            if known.iter().any(|p| p.exists()) {
                let found = chrome_path();
                assert!(found.is_some(), "a browser exists on disk but chrome_path found none");
                assert!(found.unwrap().exists());
            }
        }
        // On any platform a returned path must actually exist.
        if let Some(p) = chrome_path() {
            assert!(p.exists());
        }
    }

    // A page alert() freezes the JS engine; the event-pump handler must
    // auto-dismiss it, unblock the in-flight eval, and surface the text in
    /// Console auto-attach must stay inside the developer's own world:
    /// dev servers and local files yes, other people's websites no.
    #[test]
    fn local_page_gate_matches_dev_origins_only() {
        for u in [
            "http://localhost:5173/",
            "http://localhost/app",
            "https://localhost:8443/x?q=1",
            "http://127.0.0.1:8000/index.html",
            "http://0.0.0.0:3000",
            "http://[::1]:9000/page",
            "http://app.localhost:3000/",
            "file:///Users/dev/site/index.html",
            "data:text/html,<h1>x</h1>",
            "about:blank",
        ] {
            assert!(BrowserSession::is_local_page_url(u), "{u} should be local");
        }
        for u in [
            "https://example.com/",
            "https://github.com/Fangyuan025/Chaty",
            "http://192.168.1.20:8080/",
            "https://localhost.evil.com/phish",
            "https://mylocalhost.com/",
            "",
            "chrome://settings",
        ] {
            assert!(!BrowserSession::is_local_page_url(u), "{u} should NOT be local");
        }
    }

    // Same contract over HTTP (the real dev-server flow, python http.server
    // with zero cache headers): edit the served file on disk → refresh must
    // show the new content. Proves ignoreCache reaches CDP and the reload is
    // a true hard refresh, not a cache read.
    // Run: cargo test -p chaty refresh_hard -- --ignored --nocapture
    #[test]
    #[ignore]
    fn refresh_hard_reloads_over_http() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        let dir = std::env::temp_dir().join(format!("chaty-httprefresh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"), "<title>h</title><body><h1>HTTP-VER-ONE-55</h1></body>").unwrap();
        let mut srv = std::process::Command::new("python3")
            .args(["-m", "http.server", "29513", "--directory"])
            .arg(&dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn http.server");
        std::thread::sleep(Duration::from_millis(1200));
        let nav = navigate("http://127.0.0.1:29513/").expect("navigate");
        assert!(nav.contains("HTTP-VER-ONE-55"), "{nav}");
        std::fs::write(dir.join("index.html"), "<title>h</title><body><h1>HTTP-VER-TWO-66</h1></body>").unwrap();
        let re = refresh().expect("refresh");
        let _ = srv.kill();
        let _ = srv.wait();
        assert!(re.contains("HTTP-VER-TWO-66"), "hard refresh must fetch the new file: {re}");
        assert!(!re.contains("HTTP-VER-ONE-55"), "stale content must be gone: {re}");
        shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    // The exact walkthrough failure: edit a local file, "refresh" by
    // re-screenshotting → stale render. browser_refresh must show the NEW
    // content. Real Chrome; ignored by default.
    // Run: cargo test -p chaty refresh_shows -- --ignored --nocapture
    #[test]
    #[ignore]
    fn refresh_shows_edited_local_file() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        let dir = std::env::temp_dir().join(format!("chaty-refresh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("page.html");
        std::fs::write(&f, "<title>r</title><body><h1>VERSION-ONE-77</h1></body>").unwrap();
        let url = format!("file://{}", f.display());
        let nav = navigate(&url).expect("navigate");
        assert!(nav.contains("VERSION-ONE-77"), "{nav}");
        // Edit on disk — the loaded DOM is now stale.
        std::fs::write(&f, "<title>r</title><body><h1>VERSION-TWO-88</h1></body>").unwrap();
        let re = refresh().expect("refresh");
        assert!(re.contains("VERSION-TWO-88"), "refresh must show the new content: {re}");
        assert!(!re.contains("VERSION-ONE-77"), "stale content must be gone: {re}");
        shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Interaction results must auto-attach NEW error-class console lines
    // (the model kept shipping broken pages because it never asked), exactly
    // once each — and info-level lines must stay out of the way but remain
    // available to an explicit browser_console. Real Chrome; ignored by
    // default. Run: cargo test -p chaty console_autoattach -- --ignored --nocapture
    #[test]
    #[ignore]
    fn console_autoattach_errors_ride_interactions_once() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        // Page whose script throws at load AND logs an info line.
        // Markers are CONCATENATED in the page script so the URL echo in the
        // navigate result can never contain them verbatim.
        let broken = "data:text/html,<title>boom</title><body>x</body><script>console.log('fyi'+'-note-77');throw new Error('boot'+'-crash-77')</script>";
        let nav = navigate(broken).expect("navigate");
        assert!(
            nav.contains("[console]") && nav.contains("boot-crash-77"),
            "load-time exception must ride the navigate result: {nav}"
        );
        assert!(!nav.contains("fyi-note-77"), "info lines must NOT auto-attach: {nav}");

        // A follow-up interaction on a now-quiet page: no repeat of old errors.
        let out = eval("1+1").expect("eval");
        assert!(
            !out.contains("boot-crash-77"),
            "already-surfaced errors must not repeat: {out}"
        );

        // A fresh error caused BY the interaction rides ITS result.
        let out = eval("console.error('click'+'-broke-77'); 7").expect("eval2");
        assert!(
            out.contains("[console]") && out.contains("click-broke-77"),
            "new error must ride the interaction that caused it: {out}"
        );

        // A dialog caused by the interaction rides along labeled as a DIALOG,
        // not an error (expected page behavior, e.g. alert on button click).
        let out = eval("alert('ding'+'-77'); 1").expect("eval3");
        assert!(out.contains("[console]") && out.contains("[dialog]"), "{out}");
        assert!(
            out.contains("非报错") || out.contains("not an error"),
            "dialog-only attach must be labeled as a dialog: {out}"
        );
        assert!(
            !out.contains("新增的报错") && !out.contains("page errors since"),
            "dialog-only attach must NOT be labeled as errors: {out}"
        );

        // The explicit console view still holds the full history (incl. info).
        let full = console().unwrap_or_default();
        assert!(
            full.contains("fyi-note-77") && full.contains("boot-crash-77"),
            "browser_console must keep the full buffer: {full}"
        );
        // And looking twice still shows it. Reading used to EMPTY the buffer,
        // so a model that checked the console a second time — which is what
        // debugging a page looks like — was told it was empty while Chrome went
        // on showing the error.
        let again = console().unwrap_or_default();
        assert!(
            again.contains("boot-crash-77"),
            "a second look must not come back empty: {again}"
        );
        shutdown();
    }

    // the console. Run: cargo test -p chaty dialog_ -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dialog_auto_dismiss_unblocks_eval() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        let nav = navigate("data:text/html,<title>dlg</title><body>hi</body>").expect("navigate");
        assert!(nav.contains("dlg"), "{nav}");
        // Without the handler this call times out at 30s — alert() blocks the
        // engine and Runtime.evaluate never returns.
        let t0 = std::time::Instant::now();
        let out = eval("alert('boom-dialog'); 40+2").expect("eval must not wedge");
        assert!(out.contains("42"), "eval result after alert: {out}");
        assert!(t0.elapsed() < Duration::from_secs(10), "eval unblocked late: {:?}", t0.elapsed());
        let console = console().unwrap_or_default();
        assert!(console.contains("[dialog] alert: boom-dialog"), "console: {console}");
        shutdown();
    }

    /// A browser left open on our profile gets adopted, not fought with.
    ///
    /// The regression: force-quitting the app (or reloading it in dev) leaves
    /// its browser window up, still holding the profile. Chrome will not start
    /// a second browser on one profile — the new process hands its command
    /// line to the one already running and exits — so the launcher waited out
    /// its deadline and every browser tool failed with "Chrome did not become
    /// ready in time", for good, until the user found the stray window and
    /// closed it themselves.
    #[test]
    #[ignore]
    fn a_browser_left_open_on_our_profile_is_adopted() {
        let Some(exe) = chrome_path() else {
            eprintln!("SKIP: no Chrome found");
            return;
        };
        let dir = std::env::temp_dir().join(format!("chaty-adopt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("profile dir");
        set_headless(true);
        set_profile_dir(dir.clone());

        // The window the last run left behind.
        let mut stray = std::process::Command::new(&exe)
            .arg("--headless=new")
            .arg("--remote-debugging-port=0")
            .arg(format!("--user-data-dir={}", dir.display()))
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("about:blank")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("stray chrome");
        let port_file = dir.join("DevToolsActivePort");
        let deadline = Instant::now() + Duration::from_secs(15);
        while !port_file.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(60));
        }
        assert!(port_file.exists(), "the stray browser never came up");

        let out = navigate("data:text/html,<title>adopt</title><body><button>ADOPTED</button></body>");
        shutdown();
        let _ = stray.kill();
        let _ = stray.wait();
        let _ = std::fs::remove_dir_all(&dir);

        let page = out.expect("navigating with a browser already on the profile");
        assert!(page.contains("ADOPTED"), "adopted browser did not load the page: {page}");
    }

    /// ...and if the endpoint record is gone, the stray browser is cleared.
    ///
    /// Chrome deletes that record when it exits cleanly, and an older build of
    /// this app deleted it on the way to a launch that could never succeed —
    /// so "a browser is holding the profile" and "there is nothing to adopt"
    /// happen together, which is precisely the state that used to be
    /// unrecoverable without the user hunting down a window.
    #[test]
    #[ignore]
    fn a_stray_browser_with_no_endpoint_record_is_cleared() {
        let Some(exe) = chrome_path() else {
            eprintln!("SKIP: no Chrome found");
            return;
        };
        let dir = std::env::temp_dir().join(format!("chaty-stray-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("profile dir");
        set_headless(true);
        set_profile_dir(dir.clone());

        let mut stray = std::process::Command::new(&exe)
            .arg("--headless=new")
            .arg("--remote-debugging-port=0")
            .arg(format!("--user-data-dir={}", dir.display()))
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("about:blank")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("stray chrome");
        let port_file = dir.join("DevToolsActivePort");
        let deadline = Instant::now() + Duration::from_secs(15);
        while !port_file.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(60));
        }
        assert!(port_file.exists(), "the stray browser never came up");
        // The record an older build would already have thrown away.
        std::fs::remove_file(&port_file).expect("remove endpoint record");

        let out = navigate("data:text/html,<title>stray</title><body><button>RECOVERED</button></body>");
        shutdown();
        let _ = stray.kill();
        let _ = stray.wait();
        let _ = std::fs::remove_dir_all(&dir);

        let page = out.expect("navigating with an unreachable browser holding the profile");
        assert!(page.contains("RECOVERED"), "did not recover the profile: {page}");
    }

    /// Clicking by the words a person can actually read.
    ///
    /// The regression: a choice in a list wears a badge the page adds — "1",
    /// "2)", "3." — so its element text is "2 jolie" while the page reads
    /// "jolie". Asking for "jolie" fell through to a substring match, and the
    /// PANEL wrapping all the choices contains that substring too. The panel
    /// won, the click landed on it, and the tool reported success while
    /// nothing at all was selected — the agent then tried every option in turn
    /// and never got anywhere.
    #[test]
    #[ignore]
    fn a_choice_is_clickable_by_the_words_on_screen() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        // The prompt sits above the choices, as it does on a real quiz, so the
        // wrapper's midpoint is empty space — a click there selects nothing at
        // all, which is what the failure looked like.
        let html = "<!doctype html><title>choices</title><body style='margin:0'>\
            <div id='panel' tabindex='0' style='cursor:pointer'\
                 onclick='window.picked=window.picked||\"PANEL\"'>\
              <p style='height:420px;margin:0'>Fill in the blank</p>\
              <div role='radio' aria-checked='false' style='height:44px'\
                   onclick='window.picked=\"one\"'><span>1</span> <span>mechante</span></div>\
              <div role='radio' aria-checked='false' style='height:44px'\
                   onclick='window.picked=\"two\"'><span>2</span> <span>jolie</span></div>\
              <div role='radio' aria-checked='false' style='height:44px'\
                   onclick='window.picked=\"three\"'><span>3</span> <span>tante</span></div>\
            </div></body>";
        let url = format!("data:text/html,{}", html.replace('#', "%23"));

        // The words as they appear on screen, with no badge.
        navigate(&url).expect("navigate");
        click(None, Some("jolie".into())).expect("click by the visible word");
        let picked = eval("String(window.picked)").unwrap_or_default();
        assert!(picked.contains("two"), "clicked \"jolie\" and got {picked}");

        // And the badge spelled out, which is how the element list shows it.
        navigate(&url).expect("navigate");
        click(None, Some("3 tante".into())).expect("click by the listed label");
        let picked = eval("String(window.picked)").unwrap_or_default();
        assert!(picked.contains("three"), "clicked \"3 tante\" and got {picked}");
        shutdown();
    }

    /// Browsing has to survive `browser_close`.
    ///
    /// The regression: the closed browser's actor cleared the cached handle as
    /// it exited — but by then the NEXT browser had already been launched and
    /// cached, so the newcomer's handle was wiped instead. Nothing looked
    /// broken (the sender still worked), yet every call after that built a
    /// fresh window, ran one command in it, and lost it. Navigation reported
    /// success while every read came back blank, because the read ran in a
    /// window that had never been navigated anywhere.
    #[test]
    #[ignore]
    fn browsing_survives_a_close() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        let url = "data:text/html,<title>after-close</title><body><button>HELLO</button></body>";
        for round in 1..=3 {
            navigate(url).expect("navigate");
            let page = read_page().unwrap_or_default();
            assert!(
                page.contains("HELLO"),
                "round {round}: the page read back blank after a close — {page}"
            );
            shutdown();
        }
    }

    /// A click must reach the element it names, or say it did not.
    ///
    /// The regression: the coordinate is measured, and by the time the event
    /// is dispatched something else owns that spot — a cookie banner, a modal,
    /// a panel that finished animating in. The event lands on the intruder,
    /// and the tool reported success, so the agent had no way to know it had
    /// clicked something else entirely and went on clicking forever.
    #[test]
    #[ignore]
    fn click_lands_on_the_named_element_or_admits_it_did_not() {
        if chrome_path().is_none() {
            eprintln!("SKIP: no Chrome found");
            return;
        }
        // TARGET is real, visible and correctly labelled — and completely
        // covered by a banner, exactly as a consent overlay covers a page.
        let html = "<!doctype html><title>covered</title><body style='margin:0;height:900px'>\
            <button id='t' style='position:absolute;top:100px;left:0;width:300px;height:60px'\
                 onclick='window.hit=1'>TARGET</button>\
            <div id='cover' style='position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:9'\
                 onclick='window.cover=(window.cover||0)+1'>consent</div>\
            </body>";
        let url = format!("data:text/html,{}", html.replace('#', "%23"));
        navigate(&url).expect("navigate");
        let outcome = click(None, Some("TARGET".into()));
        let hit = eval("String(window.hit||0)").unwrap_or_default();
        let cover = eval("String(window.cover||0)").unwrap_or_default();
        // Missing is allowed — the banner really is in the way. Claiming the
        // miss worked is not.
        if outcome.is_ok() {
            panic!("reported success while the banner took the click (target fired: {hit}, banner: {cover})");
        }
        // And the failure has to be useful: it names what is in the way.
        let msg = outcome.unwrap_err();
        assert!(
            msg.contains("TARGET"),
            "the error should name the target the agent asked for: {msg}"
        );
        shutdown();
    }

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
            <button id='md'>Save</button><button id='md2'>Save All</button>\
            <script>console.error('boom-42');console.log('ok-hi');\
            document.getElementById('md').addEventListener('mousedown',function(){document.getElementById('h').dataset.md='exact';});\
            document.getElementById('md2').addEventListener('mousedown',function(){document.getElementById('h').dataset.md='all';});</script></body></html>";
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
        // REAL mouse events: a mousedown-only listener (React-style widgets)
        // must fire — a synthetic el.click() never triggered these. And exact
        // text ("Save") must win over a substring container ("Save All").
        click(None, Some("Save".into())).expect("click Save");
        let md = eval("document.getElementById('h').dataset.md||''").expect("md");
        assert_eq!(md.trim_matches('"'), "exact", "real mousedown should fire on the EXACT 'Save' button");
        click(None, Some("Save All".into())).expect("click Save All");
        let md2 = eval("document.getElementById('h').dataset.md||''").expect("md2");
        assert_eq!(md2.trim_matches('"'), "all", "exact match 'Save All' should hit the second button");
        // a statement-body eval with `return` must not error
        let ev = eval("const x = 40 + 2; return x").expect("eval stmt");
        assert_eq!(ev.trim_matches('"'), "42");
        // rich read = visible TEXT + interactive elements. The dynamic rule
        // case: JS injects new text; a plain read must surface it (no screenshot
        // needed). Also verifies input VALUES show up in the digest.
        eval("var d=document.createElement('p');d.textContent='RULE: your password must include a month';document.body.appendChild(d);").expect("inject");
        eval("var i=document.createElement('input');i.id='pw';i.placeholder='password';document.body.appendChild(i);i.value='hunter2';").expect("inject input");
        let dig = read_page().expect("digest");
        assert!(dig.contains("Go"), "digest should list the button: {dig}");
        assert!(dig.contains("must include a month"), "rich read must surface dynamically-injected TEXT (the vision substitute): {dig}");
        assert!(dig.contains("hunter2"), "rich read must show the current input VALUE: {dig}");
        // typing returns the fresh visible text so the model reads changes.
        let typed = type_text(Some("#pw".into()), None, "December1".into()).expect("type");
        assert!(typed.contains("December1"), "type result should echo the new page state incl. the value: {typed}");

        // BATCH: fill a whole form + click a sequence of buttons in ONE call each.
        eval("document.body.innerHTML='<input id=n placeholder=name><input id=e placeholder=email><textarea id=m placeholder=message></textarea><button id=b1>Step1</button><button id=b2>Step2</button><p id=log></p>';var l=document.getElementById('log');document.getElementById('b1').onclick=function(){l.textContent+='1';};document.getElementById('b2').onclick=function(){l.textContent+='2';};").expect("build form");
        let ts = type_seq(vec![
            (Some("#n".into()), None, "Alice".into()),
            (Some("#e".into()), None, "a@b.com".into()),
            (Some("#m".into()), None, "hello world".into()),
        ]).expect("type_seq");
        assert!(ts.contains("3 个字段"), "type_seq should report 3 filled: {ts}");
        assert_eq!(eval("document.getElementById('n').value").unwrap().trim_matches('"'), "Alice");
        assert_eq!(eval("document.getElementById('e').value").unwrap().trim_matches('"'), "a@b.com");
        assert_eq!(eval("document.getElementById('m').value").unwrap().trim_matches('"'), "hello world");
        let cs = click_seq(vec![
            (None, Some("Step1".into())),
            (None, Some("Step2".into())),
        ]).expect("click_seq");
        assert!(cs.contains("2 处"), "click_seq should report 2 clicks: {cs}");
        assert_eq!(eval("document.getElementById('log').textContent").unwrap().trim_matches('"'), "12", "both buttons clicked in order");

        // Ambiguous text: a nav "Login" LINK and a form submit "Login" button.
        // Clicking "Login" must fire the FORM (submit control wins over link),
        // not follow the same-named link — the real quotes.toscrape login bug.
        eval("document.body.innerHTML='<a href=\"/login\">Login</a><form onsubmit=\"event.preventDefault();document.body.dataset.submitted=1;return false;\"><input name=u><input type=submit value=Login></form>';").expect("build login");
        click(None, Some("Login".into())).expect("click Login");
        assert_eq!(eval("document.body.dataset.submitted||'0'").unwrap().trim_matches('"'), "1", "click 'Login' must submit the form, not follow the nav link of the same text");

        // <select> dropdown: browser_type selects the option by visible text.
        eval("document.body.innerHTML='<select id=sel><option value=\"\">--</option><option value=\"e\">Albert Einstein</option><option value=\"m\">Marilyn Monroe</option></select>';").expect("build select");
        let seld = type_text(Some("#sel".into()), None, "Marilyn Monroe".into()).expect("select by text");
        // The result string is bilingual and language-state dependent —
        // assert the act, not one language's phrasing.
        assert!(seld.contains("typed") || seld.contains("已输入"), "select result: {seld}");
        assert_eq!(eval("document.getElementById('sel').value").unwrap().trim_matches('"'), "m", "select set to the matching option");
        // A non-existent option returns the option list, not a silent no-op.
        let miss = type_text(Some("#sel".into()), None, "Nobody".into());
        assert!(miss.is_err() && format!("{miss:?}").contains("没有匹配"), "missing option should error with the list: {miss:?}");

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
        // KNOWN IN-TEST FLAKE (audited 2026-08-08): at THIS point in the
        // accumulated session (≈15 DOM-heavy sections deep), Chrome 150's
        // captureScreenshot can wedge past the CDP read timeout. The real
        // product path is healthy — the identical nav→scroll→snapshot
        // sequence (including an http→file cross-process hop) captures in
        // <100ms through chaty-headless, verified twice during the audit.
        // Anti-throttling launch flags + focus emulation were added as
        // standard hardening; if this expect ever fires, suspect the test's
        // own session accumulation before suspecting the capture path.
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

    // Toggling Settings → Code's hidden-browser preference must close the
    // running browser, or the setting silently applies "next session".
    #[test]
    fn headless_toggle_closes_the_open_browser() {
        let start = HEADLESS_PREF.load(std::sync::atomic::Ordering::Relaxed);
        let (tx, _rx) = std::sync::mpsc::channel::<BrowserCmd>();
        *BROWSER.lock().unwrap() = Some((0, tx));
        // Same value → no restart (don't kill a browser for a no-op write).
        set_headless(start);
        assert!(BROWSER.lock().unwrap().is_some(), "no-op toggle must keep the browser");
        // Changed value → the actor is dropped so the next call relaunches.
        set_headless(!start);
        assert!(BROWSER.lock().unwrap().is_none(), "changed toggle must close the browser");
        set_headless(start); // restore the process-wide default
        *BROWSER.lock().unwrap() = None;
    }

    /// A browser thread that dies takes its handle with it. Left cached, the
    /// dead sender failed every later call with "the browser is closed" and
    /// nothing ever replaced it — the tool was gone until the app restarted.
    #[test]
    fn a_dead_browser_thread_clears_its_handle() {
        let (tx, rx) = std::sync::mpsc::channel::<BrowserCmd>();
        *BROWSER.lock().unwrap() = Some((0, tx));

        std::thread::spawn(move || {
            let _forget = super::ForgetOnExit(0);
            let _rx = rx;
            panic!("the actor gave up");
        })
        .join()
        .expect_err("the fixture must actually panic");

        assert!(
            BROWSER.lock().unwrap().is_none(),
            "a dead actor must not leave a sender nothing can reach"
        );
    }

    /// ...and it must not take the NEXT actor's handle with it. A close and a
    /// relaunch overlap: the outgoing actor finishes its teardown after the
    /// incoming one is already cached. Wiping the cache blindly there left
    /// every later call building a browser of its own — navigation "worked"
    /// and every read came back blank.
    #[test]
    fn a_dead_browser_thread_leaves_its_successor_alone() {
        let (old_tx, old_rx) = std::sync::mpsc::channel::<BrowserCmd>();
        let (new_tx, _new_rx) = std::sync::mpsc::channel::<BrowserCmd>();
        *BROWSER.lock().unwrap() = Some((1, old_tx));

        // The successor registers while the outgoing actor is still winding up.
        *BROWSER.lock().unwrap() = Some((2, new_tx));
        std::thread::spawn(move || {
            let _forget = super::ForgetOnExit(1);
            let _rx = old_rx;
        })
        .join()
        .expect("fixture");

        assert!(
            matches!(*BROWSER.lock().unwrap(), Some((2, _))),
            "the outgoing actor cleared the browser its successor had just installed"
        );
        *BROWSER.lock().unwrap() = None;
    }

    // The async-settle watcher is injected as text; a botched edit would break
    // AJAX-confirmation detection silently (the whole point of the fix).
    #[test]
    fn settle_scripts_keep_their_contract() {
        assert!(SETTLE_INSTALL_JS.contains("window.__chatyWatch"));
        assert!(SETTLE_INSTALL_JS.contains("MutationObserver"));
        assert!(SETTLE_INSTALL_JS.contains("XMLHttpRequest.prototype.send"));
        assert!(SETTLE_INSTALL_JS.contains("window.fetch"));
        // Poll must report both halves the waiter parses: "pending:sinceMs".
        assert!(SETTLE_POLL_JS.contains("w.pending+':'+"));
        assert!(SETTLE_POLL_JS.contains("'none'"));
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
