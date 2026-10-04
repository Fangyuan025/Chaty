//! The music engine: audio.cpp in a sidecar process (`chaty-audio`, built from
//! `src-tauri/audio-sidecar`), driven the way the image engine drives its own
//! — JSON lines over stdin/stdout.
//!
//! A separate process because audio.cpp carries its own ggml, which cannot
//! share a binary with llama.cpp's. It also makes stopping possible: audio.cpp
//! runs a song to its end with no way in, so a stop ends the process, and the
//! next song starts a fresh one. That costs little — audio.cpp reads a model's
//! weights when a song needs them, not when the model is loaded, and hands
//! them back as each stage ends.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tauri::ipc::Channel;

use super::{GenRequest, InferenceBackend, StreamEvent};

const READY_TIMEOUT: Duration = Duration::from_secs(60);
const LOAD_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// Longest a song may go without a word from the engine. A stage reports only
/// when it ends, and the music tokens of a long song on a CPU are an hour's
/// work; a GPU does the same in minutes.
const IDLE_TIMEOUT: Duration = Duration::from_secs(4 * 60 * 60);
/// Lines of the engine's own log kept for the error report.
const STDERR_TAIL: usize = 40;

/// Locate the `chaty-audio` sidecar. In order: an explicit override (tests),
/// beside the app executable (installed, and `tauri dev`), the staged build in
/// `src-tauri/binaries`, and the raw CMake output (local development).
pub fn find_sidecar() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHATY_AUDIO_SIDECAR") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let exe = if cfg!(windows) { "chaty-audio.exe" } else { "chaty-audio" };
    let mut candidates = Vec::new();
    if let Ok(me) = std::env::current_exe() {
        if let Some(dir) = me.parent() {
            candidates.push(dir.join(exe));
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Staged as `chaty-audio-<target-triple>[.exe]` for Tauri's externalBin.
    if let Ok(rd) = std::fs::read_dir(manifest.join("binaries")) {
        let mut staged: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("chaty-audio-") && (!cfg!(windows) || n.ends_with(".exe")))
            })
            .collect();
        staged.sort();
        candidates.extend(staged);
    }
    candidates.push(manifest.join("audio-sidecar/build/bin").join(exe));
    candidates.into_iter().find(|p| p.is_file())
}

/// One piece, as the engine takes it: the request text, an earlier piece to
/// work from, and audio.cpp's request options by name.
#[derive(Debug, Clone, Default)]
pub struct Job {
    pub text: String,
    /// A WAV the family works from (a repaint, a cover, a variation).
    pub audio_path: Option<String>,
    pub options: BTreeMap<String, String>,
    /// Negative = a random one, chosen by the engine and reported back.
    pub seed: i64,
}

/// One of a family's options, as the engine describes it.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct EngineOption {
    pub name: String,
    /// "int", "float", "bool", "string", "path", or the choices `a|b|c`.
    #[serde(rename(deserialize = "type"))]
    pub kind: String,
    pub description: String,
    pub default: String,
    pub min: String,
    pub max: String,
    pub required: bool,
}

/// What a piece reports while it is made.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum MusicEvent {
    /// A stage began: "prepare", then the family's own ("score", "tokens",
    /// "render", "decode"), then "save".
    #[serde(rename_all = "camelCase")]
    Stage { stage: String, seed: Option<i64> },
    /// `done` of `total` within the stage.
    #[serde(rename_all = "camelCase")]
    Progress { stage: String, done: f64, total: f64 },
    /// Something learned on the way (`seconds`: the length composed).
    #[serde(rename_all = "camelCase")]
    Info { key: String, value: f64 },
    /// The finished piece, already on disk.
    #[serde(rename_all = "camelCase")]
    Audio { audio: MusicAudio },
}

/// A finished piece and what came with it.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MusicAudio {
    pub path: String,
    pub seconds: f64,
    pub sample_rate: u32,
    pub channels: u32,
    pub seed: i64,
    /// The score the model wrote for itself (YuE2), an ABC file beside it.
    pub score_path: Option<String>,
    /// Its music tokens (YuE2), what a later round continues from.
    pub tokens_path: Option<String>,
}

/// A piece that ran to its end — or was stopped.
#[derive(Debug, Clone, Default)]
pub struct MusicOutcome {
    pub audio: Option<MusicAudio>,
    /// A run that stops once the score is written (YuE2 `stop_after=abc`):
    /// the ABC file, and whether the model ran out of room writing it.
    pub score: Option<(String, bool)>,
    pub cancelled: bool,
    pub elapsed_ms: u64,
}

/// What `load` learned from the engine.
#[derive(Debug, Clone, Default)]
pub struct Loaded {
    pub version: String,
    /// The first GPU the engine sees, "" on the CPU.
    pub device: String,
    pub family: String,
    /// The family's options, as the engine lists them.
    pub request_options: Vec<EngineOption>,
    pub session_options: Vec<EngineOption>,
}

/// One running sidecar.
struct Proc {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    lines: Mutex<Receiver<String>>,
    tail: Arc<Mutex<VecDeque<String>>>,
}

impl Proc {
    fn spawn(sidecar: &Path) -> Result<Self> {
        let mut command = Command::new(sidecar);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        crate::agent::hide_console(&mut command);
        let mut child = command
            .spawn()
            .with_context(|| trf!("无法启动音乐引擎 {}", "failed to start the music engine {}", sidecar.display()))?;
        super::SIDECAR_PIDS.lock().unwrap().push(child.id());
        let stdin = child.stdin.take().context("sidecar stdin unavailable")?;
        let stdout = child.stdout.take().context("sidecar stdout unavailable")?;
        let stderr = child.stderr.take().context("sidecar stderr unavailable")?;
        // The engine's own log: kept (the tail) for when something goes
        // wrong, and drained always — a full pipe would stall the engine.
        let tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL)));
        {
            let tail = tail.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(|l| l.ok()) {
                    if let Ok(mut t) = tail.lock() {
                        if t.len() == STDERR_TAIL {
                            t.pop_front();
                        }
                        t.push_back(line);
                    }
                }
            });
        }
        let (tx, rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(Some(stdin)),
            lines: Mutex::new(rx),
            tail,
        })
    }

    fn send(&self, cmd: &Value) -> Result<()> {
        let mut guard = self.stdin.lock().map_err(|_| anyhow!("stdin poisoned"))?;
        let pipe = guard
            .as_mut()
            .ok_or_else(|| anyhow!(trf!("音乐引擎已卸载", "the music engine was unloaded")))?;
        writeln!(pipe, "{cmd}")?;
        pipe.flush()?;
        Ok(())
    }

    fn recv(rx: &Receiver<String>, timeout: Duration) -> Result<Option<Value>> {
        match rx.recv_timeout(timeout) {
            Ok(line) => Ok(serde_json::from_str::<Value>(&line).ok().filter(|v| v.get("event").is_some())),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(anyhow!("sidecar exited")),
        }
    }

    fn died(&self) -> anyhow::Error {
        // Give the reader a moment to collect the last words.
        std::thread::sleep(Duration::from_millis(150));
        let tail = self
            .tail
            .lock()
            .map(|t| t.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();
        let base = trf!("音乐引擎意外退出", "the music engine exited unexpectedly");
        anyhow::Error::new(super::sd::Crashed(if tail.trim().is_empty() { base } else { format!("{base}\n{tail}") }))
    }

    /// The handshake, then `cmd` (a `load`). Returns what the engine said.
    fn handshake_and_load(&self, cmd: &Value, progress: &dyn Fn(f32)) -> Result<Loaded> {
        let rx = self.lines.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        let started = Instant::now();
        let ready = loop {
            if started.elapsed() > READY_TIMEOUT {
                bail!(trf!("音乐引擎启动超时", "the music engine did not start in time"));
            }
            match Self::recv(&rx, Duration::from_millis(500)) {
                Ok(Some(ev)) => break ev,
                Ok(None) => continue,
                Err(_) => return Err(self.died()),
            }
        };
        match ready["event"].as_str() {
            Some("ready") => {}
            Some("fatal") => bail!(
                "{}",
                trf!(
                    "音乐引擎无法在这台电脑上运行:{}",
                    "the music engine cannot run on this computer: {}",
                    ready["message"].as_str().unwrap_or("?")
                )
            ),
            _ => bail!("unexpected first event from the music engine: {ready}"),
        }
        progress(0.3);
        let device = ready["devices"]
            .as_array()
            .and_then(|a| {
                a.iter()
                    .filter(|d| d["type"] != "cpu")
                    .filter_map(|d| {
                        let name = d["name"].as_str().unwrap_or("");
                        (!name.is_empty() && !name.eq_ignore_ascii_case("cpu"))
                            .then(|| d["description"].as_str().filter(|s| !s.is_empty()).unwrap_or(name).to_string())
                    })
                    .next()
            })
            .unwrap_or_default();

        self.send(cmd)?;
        let deadline = Instant::now() + LOAD_TIMEOUT;
        loop {
            if Instant::now() > deadline {
                bail!(trf!("音乐模型加载超时", "loading the music model timed out"));
            }
            let ev = match Self::recv(&rx, Duration::from_millis(500)) {
                Ok(Some(ev)) => ev,
                Ok(None) => continue,
                Err(_) => return Err(self.died()),
            };
            match ev["event"].as_str() {
                Some("loaded") => {
                    let options = |k: &str| -> Vec<EngineOption> {
                        serde_json::from_value(ev[k].clone()).unwrap_or_default()
                    };
                    return Ok(Loaded {
                        version: ready["version"].as_str().unwrap_or_default().to_string(),
                        device: if cmd["backend"] == "cpu" { String::new() } else { device },
                        family: ev["family"].as_str().unwrap_or_default().to_string(),
                        request_options: options("request_options"),
                        session_options: options("session_options"),
                    });
                }
                Some("error") => {
                    let msg = ev["message"].as_str().unwrap_or("unknown error").to_string();
                    bail!("{}", explain(&msg));
                }
                _ => {}
            }
        }
    }

    fn kill(&self) {
        // Closing stdin lets a healthy sidecar exit by itself; the kill after
        // is for one that is busy, and wait() is what guarantees its memory
        // is back before this returns.
        drop(self.stdin.lock().map(|mut s| s.take()));
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut c) = guard.take() {
                let pid = c.id();
                let _ = c.kill();
                let _ = c.wait();
                super::SIDECAR_PIDS.lock().unwrap().retain(|p| *p != pid);
            }
        }
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        self.kill();
    }
}

pub struct AudioEngine {
    sidecar: PathBuf,
    /// The `load` command, sent again to every sidecar started after a stop.
    load_cmd: Value,
    proc: Mutex<Option<Arc<Proc>>>,
    busy: AtomicBool,
    /// A stop was asked for: the sidecar going away is not a crash.
    stopping: AtomicBool,
    seq: AtomicU64,
}

impl AudioEngine {
    /// Start the sidecar and load a model. `cmd` is the sidecar's `load`
    /// command; `progress` receives a rough fraction (there is little to
    /// report: the weights are read when a song needs them).
    pub fn load(sidecar: &Path, cmd: Value, progress: impl Fn(f32)) -> Result<(Self, Loaded)> {
        let proc = Proc::spawn(sidecar)?;
        let loaded = match proc.handshake_and_load(&cmd, &progress) {
            Ok(l) => l,
            Err(e) => {
                proc.kill();
                return Err(e);
            }
        };
        let engine = Self {
            sidecar: sidecar.to_path_buf(),
            load_cmd: cmd,
            proc: Mutex::new(Some(Arc::new(proc))),
            busy: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            seq: AtomicU64::new(0),
        };
        Ok((engine, loaded))
    }

    /// The running sidecar, or a new one when the last was stopped.
    fn live_proc(&self) -> Result<Arc<Proc>> {
        let mut slot = self.proc.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        if let Some(p) = slot.as_ref() {
            return Ok(p.clone());
        }
        let proc = Proc::spawn(&self.sidecar)?;
        if let Err(e) = proc.handshake_and_load(&self.load_cmd, &|_| {}) {
            proc.kill();
            return Err(e);
        }
        let proc = Arc::new(proc);
        *slot = Some(proc.clone());
        Ok(proc)
    }

    /// Make one piece, reporting as it goes. Blocking: call it from a
    /// blocking thread. The piece is written to `out_path` (a WAV).
    pub fn generate(&self, job: &Job, out_path: &Path, mut on_event: impl FnMut(MusicEvent)) -> Result<MusicOutcome> {
        if self.busy.swap(true, Ordering::SeqCst) {
            bail!(trf!("上一首还在生成", "a piece is already being made"));
        }
        struct Idle<'a>(&'a AtomicBool);
        impl Drop for Idle<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _idle = Idle(&self.busy);
        self.stopping.store(false, Ordering::SeqCst);
        let started = Instant::now();

        let proc = self.live_proc()?;
        let id = format!("m{}", self.seq.fetch_add(1, Ordering::SeqCst) + 1);
        let mut cmd = json!({
            "cmd": "generate",
            "id": id,
            "text": job.text,
            "seed": job.seed,
            "options": job.options,
            "out_path": out_path.to_string_lossy(),
        });
        if let Some(a) = job.audio_path.as_deref().filter(|a| !a.is_empty()) {
            cmd["audio_path"] = json!(a);
        }

        let rx = proc.lines.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        // Anything left over from a job that ended abnormally is not ours.
        while let Ok(Some(_)) = Proc::recv(&rx, Duration::from_millis(0)) {}
        proc.send(&cmd)?;

        let mut out = MusicOutcome::default();
        let mut last = Instant::now();
        loop {
            let ev = match Proc::recv(&rx, Duration::from_millis(500)) {
                Ok(Some(ev)) => ev,
                Ok(None) => {
                    // A stop is on its way (the channel closes next), or the
                    // engine is quiet between reports.
                    if !self.stopping.load(Ordering::SeqCst) && last.elapsed() > IDLE_TIMEOUT {
                        bail!(trf!("音乐引擎长时间无响应", "the music engine stopped responding"));
                    }
                    continue;
                }
                Err(_) => {
                    if self.stopping.load(Ordering::SeqCst) {
                        out.cancelled = true;
                        out.elapsed_ms = started.elapsed().as_millis() as u64;
                        return Ok(out);
                    }
                    // A crash: the next piece starts a new sidecar.
                    let err = proc.died();
                    if let Ok(mut slot) = self.proc.lock() {
                        if slot.as_ref().is_some_and(|s| Arc::ptr_eq(s, &proc)) {
                            *slot = None;
                        }
                    }
                    return Err(err);
                }
            };
            last = Instant::now();
            if ev.get("id").and_then(|v| v.as_str()).is_some_and(|v| v != id) {
                continue;
            }
            let text = |k: &str| ev[k].as_str().unwrap_or_default().to_string();
            match ev["event"].as_str() {
                Some("stage") => on_event(MusicEvent::Stage { stage: text("stage"), seed: ev["seed"].as_i64() }),
                Some("progress") => on_event(MusicEvent::Progress {
                    stage: text("stage"),
                    done: ev["done"].as_f64().unwrap_or(0.0),
                    total: ev["total"].as_f64().unwrap_or(0.0),
                }),
                Some("info") => on_event(MusicEvent::Info { key: text("key"), value: ev["value"].as_f64().unwrap_or(0.0) }),
                Some("audio") => {
                    let a = MusicAudio {
                        path: text("path"),
                        seconds: ev["seconds"].as_f64().unwrap_or(0.0),
                        sample_rate: ev["sample_rate"].as_u64().unwrap_or(0) as u32,
                        channels: ev["channels"].as_u64().unwrap_or(0) as u32,
                        seed: ev["seed"].as_i64().unwrap_or(0),
                        score_path: ev["score_path"].as_str().map(str::to_string),
                        tokens_path: ev["tokens_path"].as_str().map(str::to_string),
                    };
                    on_event(MusicEvent::Audio { audio: a.clone() });
                    out.audio = Some(a);
                }
                Some("score") => out.score = Some((text("score_path"), ev["truncated"].as_bool().unwrap_or(false))),
                Some("done") => {
                    out.elapsed_ms = ev["elapsed_ms"].as_u64().unwrap_or_else(|| started.elapsed().as_millis() as u64);
                    return Ok(out);
                }
                Some("error") => {
                    let msg = ev["message"].as_str().unwrap_or("unknown error").to_string();
                    bail!("{}", explain(&msg));
                }
                _ => {}
            }
        }
    }

    /// Stop the song being made: the sidecar is ended (audio.cpp cannot be
    /// interrupted), and the next song starts another.
    pub fn cancel(&self) {
        if !self.busy.load(Ordering::SeqCst) {
            return;
        }
        self.stopping.store(true, Ordering::SeqCst);
        let proc = self.proc.lock().ok().and_then(|mut s| s.take());
        if let Some(p) = proc {
            // Off this thread: wait() on a process mid-GPU-call can take a
            // moment, and the caller is the UI's stop button.
            std::thread::spawn(move || p.kill());
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    fn kill(&self) {
        if let Some(p) = self.proc.lock().ok().and_then(|mut s| s.take()) {
            p.kill();
        }
    }
}

/// Put the engine's terse errors in words a user can act on; keep the
/// original after it for the report.
fn explain(msg: &str) -> String {
    let m = msg.to_lowercase();
    if m.contains("out of memory") || m.contains("failed to allocate") || m.contains("alloc") && m.contains("fail") {
        return format!(
            "{} ({msg})",
            trf!(
                "显存或内存不足。可以在 设置 → 音乐模型 里改用 CPU、开启省显存,或换更小的量化",
                "out of GPU or system memory. Settings → Music model can move it to the CPU or save memory — or use a smaller quantization"
            )
        );
    }
    if m.contains("missing ") || m.contains("no such file") || m.contains("not found") {
        return format!(
            "{} ({msg})",
            trf!(
                "模型文件夹里缺少文件,请在模型菜单里补全配套文件",
                "a file the model needs is missing from its folder — fetch its companion files from the model menu"
            )
        );
    }
    msg.to_string()
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        self.kill();
    }
}

#[async_trait]
impl InferenceBackend for AudioEngine {
    fn name(&self) -> &str {
        "audio.cpp"
    }

    fn unload(&self) {
        self.kill();
    }

    fn as_music(&self) -> Option<&AudioEngine> {
        Some(self)
    }

    async fn generate(&self, _req: GenRequest, sink: Channel<StreamEvent>, _cancel: Arc<AtomicBool>) -> Result<()> {
        let msg = trf!(
            "当前加载的是文生音乐模型,不能对话。请换一个对话模型",
            "the loaded model makes music and cannot chat — load a chat model"
        );
        let _ = sink.send(StreamEvent::Error { message: msg.clone() });
        Err(anyhow!(msg))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn fresh_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("chaty-audio-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The whole plumbing against a scripted fake sidecar: handshake, load
    /// (with the family's options), a piece's stages and progress, the audio,
    /// and the end.
    #[test]
    fn a_scripted_sidecar_loads_and_makes_a_piece() {
        let dir = fresh_dir("mock");
        let s = script(
            &dir,
            "fake-audio.sh",
            r#"#!/bin/sh
echo "engine chatter on stderr" >&2
echo '{"event":"ready","protocol":"1","version":"x","devices":[{"name":"CPU","description":"cpu","type":"cpu"},{"name":"Vulkan0","description":"Test GPU","type":"gpu"}]}'
read load
echo '{"event":"loaded","family":"yue2","description":"YuE2","request_options":[{"name":"guidance_scale","type":"float","description":"g","default":"1.0","min":"0.0","max":"20.0","required":false}],"session_options":[]}'
read gen
id=$(echo "$gen" | sed 's/.*"id":"\([^"]*\)".*/\1/')
echo "$gen" > "$(dirname "$0")/gen.json"
echo '{"event":"stage","id":"'$id'","stage":"prepare","seed":7}'
echo '{"event":"stage","id":"'$id'","stage":"tokens"}'
echo '{"event":"progress","id":"'$id'","stage":"tokens","done":100,"total":9000}'
echo '{"event":"info","id":"'$id'","key":"seconds","value":10}'
echo '{"event":"stage","id":"'$id'","stage":"render"}'
echo '{"event":"progress","id":"'$id'","stage":"render","done":4,"total":8}'
echo '{"event":"audio","id":"'$id'","path":"/tmp/x.wav","seconds":10.5,"sample_rate":44100,"channels":2,"seed":7,"score_path":"/tmp/x.abc","tokens_path":"/tmp/x.tokens.json"}'
echo '{"event":"done","id":"'$id'","elapsed_ms":12}'
read rest
"#,
        );
        let (engine, loaded) = AudioEngine::load(&s, json!({"cmd": "load", "backend": "best"}), |_| {}).unwrap();
        assert_eq!(loaded.device, "Test GPU");
        assert_eq!(loaded.family, "yue2");
        assert_eq!(loaded.request_options.len(), 1);
        assert_eq!(loaded.request_options[0].kind, "float");
        assert_eq!(loaded.request_options[0].max, "20.0");

        let mut events = Vec::new();
        let job = Job {
            text: String::new(),
            audio_path: Some("/tmp/parent.wav".into()),
            options: [("style".to_string(), "pop".to_string()), ("lyrics".to_string(), String::new())].into(),
            seed: -1,
        };
        let out = engine.generate(&job, &dir.join("x.wav"), |e| events.push(e)).unwrap();
        let sent: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("gen.json")).unwrap()).unwrap();
        assert_eq!(sent["options"]["style"], "pop");
        assert_eq!(sent["options"]["lyrics"], "", "an empty value is sent: it means instrumental");
        assert_eq!(sent["audio_path"], "/tmp/parent.wav");
        assert_eq!(sent["seed"], -1);
        let a = out.audio.expect("audio");
        assert_eq!((a.path.as_str(), a.sample_rate, a.channels, a.seed), ("/tmp/x.wav", 44100, 2, 7));
        assert_eq!(a.score_path.as_deref(), Some("/tmp/x.abc"));
        assert_eq!(a.tokens_path.as_deref(), Some("/tmp/x.tokens.json"));
        assert!(!out.cancelled);
        let kinds: Vec<String> = events
            .iter()
            .map(|e| match e {
                MusicEvent::Stage { stage, .. } => stage.clone(),
                MusicEvent::Progress { stage, done, total } => format!("{stage} {done}/{total}"),
                MusicEvent::Info { key, value } => format!("{key}={value}"),
                MusicEvent::Audio { .. } => "audio".into(),
            })
            .collect();
        assert_eq!(kinds, vec!["prepare", "tokens", "tokens 100/9000", "seconds=10", "render", "render 4/8", "audio"]);
        assert!(!engine.is_busy());
        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A stop ends the sidecar; the piece reports cancelled, and the next one
    /// starts a new sidecar and loads the model into it again.
    #[test]
    fn a_stopped_song_ends_the_sidecar_and_the_next_one_starts_another() {
        let dir = fresh_dir("stop");
        let s = script(
            &dir,
            "slow-audio.sh",
            r#"#!/bin/sh
echo '{"event":"ready","devices":[]}'
read load
echo "spawn" >> "$(dirname "$0")/spawns"
echo '{"event":"loaded","family":"yue2"}'
read gen
id=$(echo "$gen" | sed 's/.*"id":"\([^"]*\)".*/\1/')
echo '{"event":"stage","id":"'$id'","stage":"tokens"}'
if [ -f "$(dirname "$0")/fast" ]; then
  echo '{"event":"done","id":"'$id'","elapsed_ms":1}'
  read rest
fi
exec sleep 30
"#,
        );
        let (engine, _) = AudioEngine::load(&s, json!({"cmd": "load"}), |_| {}).unwrap();
        let engine = Arc::new(engine);
        let e2 = engine.clone();
        let stopper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400));
            e2.cancel();
        });
        let t0 = Instant::now();
        let out = engine.generate(&Job::default(), &dir.join("a.wav"), |_| {}).unwrap();
        stopper.join().unwrap();
        assert!(out.cancelled);
        assert!(out.audio.is_none());
        assert!(t0.elapsed() < Duration::from_secs(10), "the stop did not end the song");

        std::fs::write(dir.join("fast"), b"").unwrap();
        let out = engine.generate(&Job::default(), &dir.join("b.wav"), |_| {}).unwrap();
        assert!(!out.cancelled);
        let spawns = std::fs::read_to_string(dir.join("spawns")).unwrap();
        assert_eq!(spawns.lines().count(), 2, "the second song ran in a new sidecar");
        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A sidecar that dies says so, with the last lines it wrote.
    #[test]
    fn a_dying_sidecar_reports_its_last_words() {
        let dir = fresh_dir("die");
        let s = script(
            &dir,
            "die.sh",
            "#!/bin/sh\necho '{\"event\":\"ready\",\"devices\":[]}'\nread load\necho 'ggml_vulkan: device lost' >&2\nexit 1\n",
        );
        let err = match AudioEngine::load(&s, json!({"cmd": "load"}), |_| {}) {
            Err(e) => format!("{e:#}"),
            Ok(_) => panic!("load should fail"),
        };
        assert!(err.contains("device lost"), "{err}");
        assert!(matches!(AudioEngine::load(&s, json!({"cmd": "load"}), |_| {}), Err(e) if e.downcast_ref::<super::super::sd::Crashed>().is_some()));
        std::fs::remove_dir_all(&dir).ok();
    }
}
