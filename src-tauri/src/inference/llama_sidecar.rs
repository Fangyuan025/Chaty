//! llama.cpp in its own process: the `chaty-llama` sidecar and the app's side
//! of it.
//!
//! The engine used to run inside the app. Moving it out does two things: the
//! engine can be replaced without an app release (a runtime pack, see
//! runtimes.rs), and a crash inside llama.cpp — a GPU driver abort, a bad file
//! — ends the sidecar instead of the window.
//!
//! The protocol is JSON lines, one object each way per line. Commands carry
//! the UI language (`lang`), so what the engine says follows it.
//!
//!   → {"cmd":"load","path","gpuLayers","nCtx","speculative"}
//!       ← {"event":"progress","frac"}…  {"event":"loaded","info":ModelInfo} | {"event":"error","message"}
//!   → {"cmd":"generate","id","req":GenRequest}   ← {"event":"ev","id","data":StreamEvent}… {"event":"end","id","error"?}
//!   → {"cmd":"collect","id","req":GenRequest}    ← {"event":"end","id","text"|"error"}
//!   → {"cmd":"cancel","id"}
//!   → {"cmd":"embed_load","path"}                ← {"event":"embed_ready"} | {"event":"error","message"}
//!   → {"cmd":"embed","id","texts"}               ← {"event":"embedded","id","vectors":[base64 f32 LE]} | {"event":"end","id","error"}
//!   → {"cmd":"quit"}
//!
//! It starts with {"event":"ready","protocol":PROTOCOL}. A closed stdin ends it.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{json, Value};
use tauri::ipc::Channel;

use super::{GenRequest, InferenceBackend, ModelInfo, StreamEvent};

/// Bump with any change an older app or an older binary would misread: a
/// runtime pack is only used when it speaks the app's (runtimes.rs).
pub const PROTOCOL: &str = "1";

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const LOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const QUIT_TIMEOUT: Duration = Duration::from_secs(10);

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "chaty-llama.exe"
    } else {
        "chaty-llama"
    }
}

/// Locate the bundled `chaty-llama`: the test override, next to the app (the
/// installed app, `tauri dev`), or a `cargo build` of it in this checkout —
/// the newest, for tests and the headless binary.
pub fn find_sidecar() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHATY_LLAMA_SIDECAR") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(p) = std::env::current_exe().ok().and_then(|me| me.parent().map(|d| d.join(exe_name()))) {
        if p.is_file() {
            return Some(p);
        }
    }
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    ["release", "debug"]
        .iter()
        .map(|profile| target.join(profile).join(exe_name()))
        .filter(|p| p.is_file())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
}

fn lang() -> &'static str {
    if crate::agent::lang_is_en() {
        "en"
    } else {
        "zh"
    }
}

/// The sidecar ended while something was waiting on it.
#[derive(Debug)]
pub(crate) struct SidecarExited;

impl std::fmt::Display for SidecarExited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&trf!(
            "对话引擎进程意外退出(可能是显存或内存不足、驱动出错或模型文件损坏)",
            "the chat engine process exited unexpectedly (out of GPU or system memory, a driver fault, or a damaged model file)"
        ))
    }
}

impl std::error::Error for SidecarExited {}

// ---- the app's side --------------------------------------------------------

/// One running sidecar: its pipes, and where each line it writes goes.
struct Proc {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    pid: u32,
    /// Events naming a request, routed to whoever sent it.
    routes: Arc<Mutex<HashMap<String, tokio::sync::mpsc::UnboundedSender<Value>>>>,
    next_id: AtomicU64,
}

impl Proc {
    /// Start `exe` and wait for its handshake. The receiver gets every event
    /// that names no request, and `{"event":"exited"}` when the process ends.
    fn spawn(exe: &Path) -> Result<(Arc<Proc>, Receiver<Value>)> {
        let mut cmd = Command::new(exe);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        crate::agent::hide_console(&mut cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!(trf!("无法启动对话引擎 {}:{e}", "could not start the chat engine {}: {e}", exe.display())))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        let stdin = child.stdin.take();
        let pid = child.id();
        super::SIDECAR_PIDS.lock().unwrap().push(pid);
        let routes: Arc<Mutex<HashMap<String, tokio::sync::mpsc::UnboundedSender<Value>>>> = Arc::default();
        let (control_tx, control) = std::sync::mpsc::channel::<Value>();
        let reader_routes = routes.clone();
        std::thread::Builder::new()
            .name("chaty-llama-reader".into())
            .spawn(move || {
                let reader = std::io::BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let Ok(ev) = serde_json::from_str::<Value>(&line) else {
                        // Not ours: something in the engine printed to stdout.
                        eprintln!("chaty-llama: {line}");
                        continue;
                    };
                    let id = ev.get("id").and_then(Value::as_str).map(str::to_string);
                    let routed = id.and_then(|id| reader_routes.lock().unwrap().get(&id).cloned());
                    match routed {
                        Some(route) => {
                            let _ = route.send(ev);
                        }
                        None => {
                            let _ = control_tx.send(ev);
                        }
                    }
                }
                // Gone: everyone still waiting hears it.
                for (_, route) in reader_routes.lock().unwrap().drain() {
                    let _ = route.send(json!({ "event": "exited" }));
                }
                let _ = control_tx.send(json!({ "event": "exited" }));
            })?;
        let proc = Arc::new(Proc {
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(stdin),
            pid,
            routes,
            next_id: AtomicU64::new(1),
        });
        let ready = match control.recv_timeout(READY_TIMEOUT) {
            Ok(ev) => ev,
            Err(_) => {
                proc.kill();
                bail!(trf!("对话引擎启动后没有应答", "the chat engine started but did not answer"));
            }
        };
        if ready["event"] != "ready" {
            proc.kill();
            if ready["event"] == "exited" {
                return Err(SidecarExited.into());
            }
            bail!(trf!("对话引擎的应答不对:{ready}", "the chat engine answered something else: {ready}"));
        }
        let spoke = ready["protocol"].as_str().unwrap_or("");
        if spoke != PROTOCOL {
            proc.kill();
            bail!(trf!(
                "对话引擎的协议版本是 {spoke},这个 Chaty 需要 {PROTOCOL}",
                "the chat engine speaks protocol {spoke}; this Chaty needs {PROTOCOL}"
            ));
        }
        Ok((proc, control))
    }

    fn send(&self, mut cmd: Value) -> Result<()> {
        cmd["lang"] = json!(lang());
        let mut guard = self.stdin.lock().unwrap();
        let stdin = guard.as_mut().ok_or(SidecarExited)?;
        let line = serde_json::to_string(&cmd)?;
        stdin
            .write_all(line.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
            .map_err(|_| anyhow::Error::from(SidecarExited))
    }

    fn new_id(&self) -> String {
        format!("r{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// A request's own events, from now until it is forgotten.
    fn route(&self, id: &str) -> tokio::sync::mpsc::UnboundedReceiver<Value> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.routes.lock().unwrap().insert(id.to_string(), tx);
        rx
    }

    fn forget(&self, id: &str) {
        self.routes.lock().unwrap().remove(id);
    }

    /// Ask it to leave, and make sure it has — its exit is what gives the
    /// model's memory back.
    fn quit(&self) {
        let _ = self.send(json!({ "cmd": "quit" }));
        drop(self.stdin.lock().unwrap().take());
        let deadline = Instant::now() + QUIT_TIMEOUT;
        loop {
            let done = match self.child.lock().unwrap().as_mut() {
                Some(c) => !matches!(c.try_wait(), Ok(None)),
                None => true,
            };
            if done || Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        self.kill();
    }

    fn kill(&self) {
        drop(self.stdin.lock().unwrap().take());
        if let Some(mut c) = self.child.lock().unwrap().take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        super::SIDECAR_PIDS.lock().unwrap().retain(|p| *p != self.pid);
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Start the engine for `start`, from a runtime pack when one is in use. On
/// Windows a process that died inside a load most likely took the GPU driver
/// with it (issue #5): the crash guard turns the marker that load left into a
/// cap on the offload, and it is tried once more under it.
fn start_engine<T>(start: impl Fn(&Path) -> Result<T>) -> Result<T> {
    crate::runtimes::start_with(
        crate::runtimes::Engine::Llama,
        find_sidecar(),
        || anyhow!(trf!("未找到对话引擎组件 chaty-llama,请重新安装应用", "the chat engine (chaty-llama) is missing; please reinstall")),
        |exe| match start(exe) {
            Err(e) if cfg!(windows) && e.is::<SidecarExited>() => {
                if super::llama::apply_gpu_crash_guard().is_some() {
                    start(exe)
                } else {
                    Err(e)
                }
            }
            r => r,
        },
    )
}

/// A chat model loaded in a sidecar.
pub struct LlamaSidecar {
    proc: Arc<Proc>,
}

impl LlamaSidecar {
    /// Load a GGUF file: `gpu_pref` and `n_ctx_pref` as `LlamaEngine::load`
    /// takes them, `progress` the weights' share loaded so far.
    pub fn load(
        path: &str,
        gpu_pref: Option<i32>,
        n_ctx_pref: Option<u32>,
        speculative: bool,
        progress: impl Fn(f32) + Send + Sync + 'static,
    ) -> Result<(Self, ModelInfo)> {
        let progress: Arc<dyn Fn(f32) + Send + Sync> = Arc::new(progress);
        start_engine(|exe| Self::load_with(exe, path, gpu_pref, n_ctx_pref, speculative, &progress))
    }

    fn load_with(
        exe: &Path,
        path: &str,
        gpu_pref: Option<i32>,
        n_ctx_pref: Option<u32>,
        speculative: bool,
        progress: &Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Result<(Self, ModelInfo)> {
        let (proc, control) = Proc::spawn(exe)?;
        proc.send(json!({
            "cmd": "load", "path": path, "gpuLayers": gpu_pref, "nCtx": n_ctx_pref, "speculative": speculative,
        }))?;
        let deadline = Instant::now() + LOAD_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let ev = control.recv_timeout(left).map_err(|_| {
                proc.kill();
                anyhow!(trf!("加载模型超时", "loading the model timed out"))
            })?;
            match ev["event"].as_str() {
                Some("progress") => progress(ev["frac"].as_f64().unwrap_or(0.0) as f32),
                Some("loaded") => {
                    let info: ModelInfo = serde_json::from_value(ev["info"].clone())?;
                    return Ok((Self { proc }, info));
                }
                Some("error") => {
                    proc.kill();
                    bail!("{}", ev["message"].as_str().unwrap_or("load failed"));
                }
                Some("exited") => {
                    proc.kill();
                    return Err(SidecarExited.into());
                }
                _ => {}
            }
        }
    }

    /// Send a request and follow its events to the end: each `ev` to `on_event`,
    /// the `end` returned. A cancel is passed on once.
    async fn request(&self, cmd: &str, req: GenRequest, cancel: &AtomicBool, on_event: impl Fn(StreamEvent)) -> Result<Value> {
        let id = self.proc.new_id();
        let mut rx = self.proc.route(&id);
        if let Err(e) = self.proc.send(json!({ "cmd": cmd, "id": id, "req": req })) {
            self.proc.forget(&id);
            return Err(e);
        }
        let mut cancelled = false;
        let mut tick = tokio::time::interval(Duration::from_millis(50));
        let end: Result<Value> = loop {
            tokio::select! {
                ev = rx.recv() => {
                    let Some(ev) = ev else { break Err(SidecarExited.into()) };
                    match ev["event"].as_str() {
                        Some("ev") => {
                            if let Ok(e) = serde_json::from_value::<StreamEvent>(ev["data"].clone()) {
                                on_event(e);
                            }
                        }
                        Some("end") => break Ok(ev),
                        Some("exited") => break Err(SidecarExited.into()),
                        _ => {}
                    }
                }
                _ = tick.tick(), if !cancelled => {
                    if cancel.load(Ordering::SeqCst) {
                        cancelled = true;
                        let _ = self.proc.send(json!({ "cmd": "cancel", "id": id }));
                    }
                }
            }
        };
        self.proc.forget(&id);
        let end = end?;
        match end["error"].as_str() {
            Some(msg) => Err(anyhow!(msg.to_string())),
            None => Ok(end),
        }
    }
}

impl Drop for LlamaSidecar {
    fn drop(&mut self) {
        self.proc.kill();
    }
}

#[async_trait]
impl InferenceBackend for LlamaSidecar {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn unload(&self) {
        self.proc.quit();
    }

    async fn generate(&self, req: GenRequest, sink: Channel<StreamEvent>, cancel: Arc<AtomicBool>) -> Result<()> {
        // A listener that has gone away is not a reason to stop (see the
        // EventSink for Channel in llama.rs).
        self.request("generate", req, &cancel, |e| {
            let _ = sink.send(e);
        })
        .await
        .map(|_| ())
    }

    async fn generate_collect(&self, req: GenRequest, cancel: Arc<AtomicBool>) -> Result<String> {
        let end = self.request("collect", req, &cancel, |_| {}).await?;
        Ok(end["text"].as_str().unwrap_or_default().to_string())
    }
}

/// The knowledge base's embedding model, loaded in a sidecar of its own so it
/// runs beside the chat model and goes when it is unloaded.
pub struct EmbedClient {
    proc: Arc<Proc>,
}

impl EmbedClient {
    pub fn start(model: &Path) -> Result<Self, String> {
        start_engine(|exe| Self::start_with(exe, model)).map_err(|e| format!("{e:#}"))
    }

    fn start_with(exe: &Path, model: &Path) -> Result<Self> {
        let (proc, control) = Proc::spawn(exe)?;
        proc.send(json!({ "cmd": "embed_load", "path": model }))?;
        loop {
            let ev = control.recv_timeout(LOAD_TIMEOUT).map_err(|_| {
                proc.kill();
                anyhow!(trf!("加载嵌入模型超时", "loading the embedding model timed out"))
            })?;
            match ev["event"].as_str() {
                Some("embed_ready") => return Ok(Self { proc }),
                Some("error") => {
                    proc.kill();
                    bail!("{}", ev["message"].as_str().unwrap_or("embed load failed"));
                }
                Some("exited") => {
                    proc.kill();
                    return Err(SidecarExited.into());
                }
                _ => {}
            }
        }
    }

    /// One L2-normalized vector per text. Blocking.
    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let id = self.proc.new_id();
        let mut rx = self.proc.route(&id);
        if let Err(e) = self.proc.send(json!({ "cmd": "embed", "id": id, "texts": texts })) {
            self.proc.forget(&id);
            return Err(format!("{e:#}"));
        }
        let ev = rx.blocking_recv();
        self.proc.forget(&id);
        let ev = ev.ok_or_else(|| SidecarExited.to_string())?;
        match ev["event"].as_str() {
            Some("embedded") => ev["vectors"]
                .as_array()
                .map(|vs| vs.iter().map(|v| decode_vector(v.as_str().unwrap_or_default())).collect())
                .ok_or_else(|| "no vectors".to_string()),
            Some("exited") => Err(SidecarExited.to_string()),
            _ => Err(ev["error"].as_str().unwrap_or("embed failed").to_string()),
        }
    }
}

impl Drop for EmbedClient {
    fn drop(&mut self) {
        self.proc.quit();
    }
}

fn encode_vector(v: &[f32]) -> String {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn decode_vector(s: &str) -> Vec<f32> {
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ---- the sidecar's side ----------------------------------------------------

/// Stdout, one event per line.
struct Out(Mutex<std::io::Stdout>);

impl Out {
    fn emit(&self, ev: &Value) {
        let line = serde_json::to_string(ev).unwrap_or_default();
        let mut out = self.0.lock().unwrap();
        if out.write_all(line.as_bytes()).and_then(|_| out.write_all(b"\n")).and_then(|_| out.flush()).is_err() {
            // Nobody is reading: the app is gone.
            leave();
        }
    }
}

/// A turn's events, written as they come.
struct LineSink {
    id: String,
    out: Arc<Out>,
}

impl super::llama::EventSink for LineSink {
    fn emit(&self, ev: StreamEvent) -> Result<()> {
        self.out.emit(&json!({ "event": "ev", "id": self.id, "data": ev }));
        Ok(())
    }
}

/// Leave without running teardown: ggml's at-exit handlers crash when a
/// model is still loaded (the app quits the same way, lib.rs).
fn leave() -> ! {
    unsafe { libc::_exit(0) }
}

/// The sidecar: serve commands from stdin until it closes or says quit.
pub fn serve() -> ! {
    crate::errlog::install_panic_hook();
    crate::agent::load_saved_lang();
    #[cfg(target_os = "linux")]
    crate::gpu::configure_linux_vulkan_device();
    // As the app does before the first Metal init (lib.rs), for a sidecar
    // started on its own.
    #[cfg(target_os = "macos")]
    if std::env::var_os("CHATY_METAL_RESIDENCY").is_none() {
        std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");
    }

    let out = Arc::new(Out(Mutex::new(std::io::stdout())));
    out.emit(&json!({ "event": "ready", "protocol": PROTOCOL }));

    let engine: Arc<Mutex<Option<Arc<super::llama::LlamaEngine>>>> = Arc::default();
    let embedder: Arc<Mutex<Option<super::llama_embed::LocalEmbedder>>> = Arc::default();
    let cancels: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>> = Arc::default();

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(cmd) = serde_json::from_str::<Value>(&line) else {
            out.emit(&json!({ "event": "error", "message": format!("bad command: {line}") }));
            continue;
        };
        if let Some(l) = cmd["lang"].as_str() {
            crate::agent::set_lang(l);
        }
        let id = cmd["id"].as_str().unwrap_or_default().to_string();
        match cmd["cmd"].as_str().unwrap_or_default() {
            "load" => {
                let (out, engine) = (out.clone(), engine.clone());
                std::thread::spawn(move || {
                    if let Some(old) = engine.lock().unwrap().take() {
                        old.unload();
                    }
                    let path = cmd["path"].as_str().unwrap_or_default().to_string();
                    let gpu = cmd["gpuLayers"].as_i64().map(|n| n as i32);
                    let n_ctx = cmd["nCtx"].as_u64().map(|n| n as u32);
                    let speculative = cmd["speculative"].as_bool().unwrap_or(false);
                    let last = Mutex::new(-1.0f32);
                    let tick = out.clone();
                    let progress = Arc::new(move |f: f32| {
                        let mut last = last.lock().unwrap();
                        if f >= *last + 0.01 || f < *last {
                            *last = f;
                            tick.emit(&json!({ "event": "progress", "frac": f }));
                        }
                    });
                    match super::llama::LlamaEngine::load_with_progress(&path, gpu, n_ctx, speculative, progress) {
                        Ok((e, info)) => {
                            *engine.lock().unwrap() = Some(Arc::new(e));
                            out.emit(&json!({ "event": "loaded", "info": info }));
                        }
                        Err(e) => out.emit(&json!({ "event": "error", "message": format!("{e:#}") })),
                    }
                });
            }
            c @ ("generate" | "collect") => {
                let Some(e) = engine.lock().unwrap().clone() else {
                    out.emit(&json!({ "event": "end", "id": id, "error": trf!("模型已卸载", "model unloaded") }));
                    continue;
                };
                let req: GenRequest = match serde_json::from_value(cmd["req"].clone()) {
                    Ok(r) => r,
                    Err(err) => {
                        out.emit(&json!({ "event": "end", "id": id, "error": format!("bad request: {err}") }));
                        continue;
                    }
                };
                let cancel = Arc::new(AtomicBool::new(false));
                cancels.lock().unwrap().insert(id.clone(), cancel.clone());
                let (out, cancels, collect) = (out.clone(), cancels.clone(), c == "collect");
                std::thread::spawn(move || {
                    let end = if collect {
                        match tauri::async_runtime::block_on(e.collect(req, cancel)) {
                            Ok(text) => json!({ "event": "end", "id": id, "text": text }),
                            Err(err) => json!({ "event": "end", "id": id, "error": format!("{err:#}") }),
                        }
                    } else {
                        let sink = Box::new(LineSink { id: id.clone(), out: out.clone() });
                        match tauri::async_runtime::block_on(e.generate_to(req, sink, cancel)) {
                            Ok(()) => json!({ "event": "end", "id": id }),
                            Err(err) => json!({ "event": "end", "id": id, "error": format!("{err:#}") }),
                        }
                    };
                    cancels.lock().unwrap().remove(&id);
                    out.emit(&end);
                });
            }
            "cancel" => {
                if let Some(c) = cancels.lock().unwrap().get(&id) {
                    c.store(true, Ordering::SeqCst);
                }
            }
            "embed_load" => {
                let (out, embedder) = (out.clone(), embedder.clone());
                std::thread::spawn(move || {
                    drop(embedder.lock().unwrap().take());
                    let path = PathBuf::from(cmd["path"].as_str().unwrap_or_default());
                    match super::llama_embed::LocalEmbedder::start(&path, None) {
                        Ok(e) => {
                            *embedder.lock().unwrap() = Some(e);
                            out.emit(&json!({ "event": "embed_ready" }));
                        }
                        Err(e) => out.emit(&json!({ "event": "error", "message": e })),
                    }
                });
            }
            "embed" => {
                let (out, embedder) = (out.clone(), embedder.clone());
                let texts: Vec<String> = serde_json::from_value(cmd["texts"].clone()).unwrap_or_default();
                std::thread::spawn(move || {
                    let got = match embedder.lock().unwrap().as_ref() {
                        Some(e) => e.embed(texts),
                        None => Err(trf!("嵌入模型未加载", "the embedding model is not loaded")),
                    };
                    out.emit(&match got {
                        Ok(vs) => json!({ "event": "embedded", "id": id, "vectors": vs.iter().map(|v| encode_vector(v)).collect::<Vec<_>>() }),
                        Err(e) => json!({ "event": "end", "id": id, "error": e }),
                    });
                });
            }
            "quit" => leave(),
            other => out.emit(&json!({ "event": "error", "id": id, "message": format!("unknown command: {other}") })),
        }
    }
    leave()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vector crosses the pipe bit for bit.
    #[test]
    fn a_vector_survives_the_pipe() {
        let v = vec![0.0f32, -1.5, 3.25e-7, f32::MAX, 1.0 / 3.0];
        assert_eq!(decode_vector(&encode_vector(&v)), v);
    }

    /// What the app reads back from the sidecar is what the engine reported,
    /// and an older or newer engine's extra or missing fields do not stop it.
    #[test]
    fn model_info_crosses_the_pipe() {
        let info = ModelInfo { name: "m".into(), n_ctx: Some(4096), gpu_layers: 28, kind: "chat".into(), ..Default::default() };
        let back: ModelInfo = serde_json::from_value(serde_json::to_value(&info).unwrap()).unwrap();
        assert_eq!((back.name.as_str(), back.n_ctx, back.gpu_layers), ("m", Some(4096), 28));
        let sparse: ModelInfo = serde_json::from_value(json!({ "name": "x", "somethingNew": 1 })).unwrap();
        assert_eq!(sparse.name, "x");
    }

    fn pid_alive(pid: u32) -> bool {
        let mut sys = sysinfo::System::new();
        let p = sysinfo::Pid::from_u32(pid);
        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[p]), true);
        sys.process(p).is_some()
    }

    fn ask(text: &str, max_tokens: u32) -> GenRequest {
        serde_json::from_value(json!({
            "messages": [{ "role": "user", "content": text }],
            "params": { "maxTokens": max_tokens, "temperature": 0.0, "think": false },
        }))
        .unwrap()
    }

    /// End to end through the real sidecar, as the app runs it: the weights
    /// load with progress, a reply streams, a cancel stops one, a one-shot
    /// answer comes back whole, and unloading ends the process. Build it first:
    ///   cargo build --bin chaty-llama && CHATY_TEST_GGUF=<a .gguf> \
    ///   cargo test --lib llama_sidecar -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore]
    fn a_model_loads_answers_cancels_and_leaves_in_its_own_process() {
        let path = std::env::var("CHATY_TEST_GGUF").expect("set CHATY_TEST_GGUF");
        let ticks = Arc::new(Mutex::new(Vec::<f32>::new()));
        let t = ticks.clone();
        let (engine, info) = LlamaSidecar::load(&path, None, Some(4096), false, move |f| t.lock().unwrap().push(f)).expect("load");
        let ticks = ticks.lock().unwrap().clone();
        eprintln!("loaded {} ({:?} gpu layers {}), {} progress ticks", info.name, info.arch, info.gpu_layers, ticks.len());
        assert_eq!(info.kind, "chat");
        assert!(info.n_ctx.unwrap_or(0) >= 1024, "{info:?}");
        assert!(!ticks.is_empty() && ticks.iter().all(|f| (0.0..=1.0).contains(f)), "{ticks:?}");
        let pid = engine.proc.pid;
        assert!(pid_alive(pid) && pid != std::process::id());

        let rt = tokio::runtime::Runtime::new().unwrap();
        let events = Arc::new(Mutex::new(Vec::<StreamEvent>::new()));
        let into = events.clone();
        let sink: Channel<StreamEvent> = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
            if let tauri::ipc::InvokeResponseBody::Json(text) = &body {
                if let Ok(e) = serde_json::from_str::<StreamEvent>(text) {
                    into.lock().unwrap().push(e);
                }
            }
            Ok(())
        });
        rt.block_on(engine.generate(ask("What is 2+3? Answer with the number only.", 32), sink.clone(), Arc::new(AtomicBool::new(false))))
            .expect("generate");
        let seen = std::mem::take(&mut *events.lock().unwrap());
        let text: String = seen.iter().filter_map(|e| if let StreamEvent::Token { text } = e { Some(text.as_str()) } else { None }).collect();
        eprintln!("answer: {text:?}");
        assert!(matches!(seen.first(), Some(StreamEvent::Started)), "{seen:?}");
        assert!(matches!(seen.last(), Some(StreamEvent::Done { .. })), "{seen:?}");
        assert!(text.contains('5'), "{text:?}");

        // A cancel mid-reply ends it early, as cancelled.
        let cancel = Arc::new(AtomicBool::new(false));
        let c = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400));
            c.store(true, Ordering::SeqCst);
        });
        rt.block_on(engine.generate(ask("Write a long story about a lighthouse.", 2000), sink, cancel)).expect("generate");
        let seen = std::mem::take(&mut *events.lock().unwrap());
        match seen.last() {
            Some(StreamEvent::Done { stats }) => assert_eq!(stats.stop_reason, "cancelled", "{stats:?}"),
            other => panic!("ended with {other:?}"),
        }

        let whole = rt
            .block_on(engine.generate_collect(ask("Say hello in one word.", 16), Arc::new(AtomicBool::new(false))))
            .expect("collect");
        eprintln!("collected: {whole:?}");
        assert!(!whole.trim().is_empty());

        engine.unload();
        assert!(!pid_alive(pid), "the sidecar must be gone after unload");
        let after = rt.block_on(engine.generate_collect(ask("hi", 4), Arc::new(AtomicBool::new(false))));
        assert!(after.is_err(), "a request after unload fails instead of hanging");
    }
}
