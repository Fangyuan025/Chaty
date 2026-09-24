//! The image engine: stable-diffusion.cpp in a sidecar process (`chaty-sd`,
//! built from `src-tauri/sd-sidecar`), driven the way the MLX engine drives
//! its own — JSON lines over stdin/stdout.
//!
//! A separate process because llama.cpp and stable-diffusion.cpp each carry
//! their own ggml, and two in one binary do not link. It also means ejecting
//! the model is killing the process (all of its memory comes back at once), and
//! a GPU driver that aborts mid-generation takes the sidecar down, not the app.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tauri::ipc::Channel;

use super::{GenRequest, InferenceBackend, StreamEvent};

const READY_TIMEOUT: Duration = Duration::from_secs(60);
const LOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);
/// Longest silence a generation may keep. One step of a large model at a
/// large size on a CPU is minutes, never this.
const IDLE_TIMEOUT: Duration = Duration::from_secs(45 * 60);
/// Lines of the engine's own log kept for the error report.
const STDERR_TAIL: usize = 40;

/// Locate the `chaty-sd` sidecar. In order: an explicit override (tests),
/// beside the app executable (installed, and `tauri dev`), the staged build in
/// `src-tauri/binaries`, and the raw CMake output (local development).
pub fn find_sidecar() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHATY_SD_SIDECAR") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let exe = if cfg!(windows) { "chaty-sd.exe" } else { "chaty-sd" };
    let mut candidates = Vec::new();
    if let Ok(me) = std::env::current_exe() {
        if let Some(dir) = me.parent() {
            candidates.push(dir.join(exe));
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Staged as `chaty-sd-<target-triple>[.exe]` for Tauri's externalBin.
    if let Ok(rd) = std::fs::read_dir(manifest.join("binaries")) {
        let mut staged: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("chaty-sd-") && (!cfg!(windows) || n.ends_with(".exe")))
            })
            .collect();
        staged.sort();
        candidates.extend(staged);
    }
    candidates.push(manifest.join("sd-sidecar/build/bin").join(exe));
    candidates.into_iter().find(|p| p.is_file())
}

/// One generation, as the sidecar takes it.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerateParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f32,
    /// Distilled guidance (FLUX-dev); ignored by models without it.
    pub guidance: Option<f32>,
    /// Empty = the model's default.
    pub sampler: String,
    pub scheduler: String,
    /// Negative = a random one, chosen by the engine and reported back.
    pub seed: i64,
    pub batch_count: u32,
    /// 0 = the model's default.
    pub flow_shift: f32,
    pub vae_tiling: bool,
    pub clip_skip: i32,
    /// "proj" (fast latent preview), "vae" (exact, slow) or "none".
    pub preview: String,
    pub preview_interval: u32,
    /// "png" (with the settings embedded) or "jpg".
    pub format: String,
    /// img2img: the picture to start from, and how far to move away from it.
    pub init_image: Option<String>,
    pub strength: f32,
    /// Reference pictures for editing models.
    pub ref_images: Vec<String>,
    /// Sampling acceleration: "off", "balanced" or "fast".
    pub accel: String,
    /// The step cache it resolved to for the loaded model (method and
    /// reuse threshold, 0 = the engine's default) — set by the caller.
    #[serde(skip)]
    pub cache: Option<(String, f32)>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 20,
            cfg_scale: 7.0,
            guidance: None,
            sampler: String::new(),
            scheduler: String::new(),
            seed: -1,
            batch_count: 1,
            flow_shift: 0.0,
            vae_tiling: false,
            clip_skip: -1,
            preview: "proj".into(),
            preview_interval: 1,
            format: "png".into(),
            init_image: None,
            strength: 0.75,
            ref_images: Vec::new(),
            accel: String::new(),
            cache: None,
        }
    }
}

/// What a generation reports while it runs.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SdEvent {
    /// A phase began: "encode" (reading the prompt), "weights" (weights not
    /// yet resident are loading), "sample" (denoising image `index` of
    /// `count`), "decode" (latents to pixels).
    #[serde(rename_all = "camelCase")]
    Stage { stage: String, index: u32, count: u32, seed: Option<i64> },
    /// `step` of `steps` within the current phase; `secs` is the last step's
    /// duration.
    #[serde(rename_all = "camelCase")]
    Progress { stage: String, step: u32, steps: u32, secs: f32 },
    /// The denoised estimate so far, as a small JPEG data URL.
    #[serde(rename_all = "camelCase")]
    Preview { step: u32, width: u32, height: u32, data_url: String },
    /// A finished picture, already on disk.
    #[serde(rename_all = "camelCase")]
    Image { index: u32, path: String, width: u32, height: u32, seed: i64 },
    /// A cache saved work: "conditioning" (the prompt's encoding was reused)
    /// or "steps" (`skipped` of `total` denoising steps were reused).
    #[serde(rename_all = "camelCase")]
    Cache { kind: String, skipped: u32, total: u32 },
}

/// A generation that ran to its end (or was stopped with pictures made).
#[derive(Debug, Clone)]
pub struct GenerateOutcome {
    pub images: Vec<(String, u32, u32, i64)>,
    pub cancelled: bool,
    pub elapsed_ms: u64,
}

/// What `load` learned from the engine.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub version: String,
    pub default_sampler: String,
    pub default_scheduler: String,
    /// The first GPU the engine sees, "" on CPU.
    pub device: String,
    /// Non-fatal warnings the engine logged while loading.
    pub warnings: Vec<String>,
}

pub struct SdEngine {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    lines: Mutex<Receiver<String>>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    busy: AtomicBool,
    seq: std::sync::atomic::AtomicU64,
}

fn send(stdin: &Arc<Mutex<Option<ChildStdin>>>, cmd: &Value) -> Result<()> {
    let mut guard = stdin.lock().map_err(|_| anyhow!("stdin poisoned"))?;
    let pipe = guard
        .as_mut()
        .ok_or_else(|| anyhow!(trf!("图像引擎已卸载", "the image engine was unloaded")))?;
    writeln!(pipe, "{cmd}")?;
    pipe.flush()?;
    Ok(())
}

impl SdEngine {
    /// Spawn the sidecar and load a model. `cmd` is the sidecar's `load`
    /// command (paths and options); `progress` receives the weight-loading
    /// fraction.
    pub fn load(sidecar: &Path, cmd: Value, progress: impl Fn(f32)) -> Result<(Self, Loaded)> {
        let mut command = Command::new(sidecar);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        crate::agent::hide_console(&mut command);
        let mut child = command
            .spawn()
            .with_context(|| trf!("无法启动图像引擎 {}", "failed to start the image engine {}", sidecar.display()))?;
        super::SIDECAR_PIDS.lock().unwrap().push(child.id());
        let stdin = child.stdin.take().context("sidecar stdin unavailable")?;
        let stdout = child.stdout.take().context("sidecar stdout unavailable")?;
        let stderr = child.stderr.take().context("sidecar stderr unavailable")?;

        // The engine's own log: kept (the tail) for when something goes wrong,
        // and drained always — a full pipe would stall the engine.
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

        let engine = Self {
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(Some(stdin))),
            lines: Mutex::new(rx),
            stderr_tail: tail,
            busy: AtomicBool::new(false),
            seq: std::sync::atomic::AtomicU64::new(0),
        };
        let loaded = engine.handshake_and_load(cmd, progress);
        match loaded {
            Ok(l) => Ok((engine, l)),
            Err(e) => {
                engine.kill();
                Err(e)
            }
        }
    }

    fn died(&self) -> anyhow::Error {
        // Give the reader a moment to collect the last words.
        std::thread::sleep(Duration::from_millis(150));
        let tail = self
            .stderr_tail
            .lock()
            .map(|t| t.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();
        let base = trf!("图像引擎意外退出", "the image engine exited unexpectedly");
        anyhow::Error::new(Crashed(if tail.trim().is_empty() { base } else { format!("{base}\n{tail}") }))
    }

    fn recv(&self, rx: &Receiver<String>, timeout: Duration) -> Result<Option<Value>> {
        match rx.recv_timeout(timeout) {
            Ok(line) => Ok(serde_json::from_str::<Value>(&line).ok().filter(|v| v.get("event").is_some())),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(SidecarDied.into()),
        }
    }

    fn handshake_and_load(&self, cmd: Value, progress: impl Fn(f32)) -> Result<Loaded> {
        let rx = self.lines.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        let started = Instant::now();
        let ready = loop {
            if started.elapsed() > READY_TIMEOUT {
                bail!(trf!("图像引擎启动超时", "the image engine did not start in time"));
            }
            match self.recv(&rx, Duration::from_millis(500)) {
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
                    "图像引擎无法在这台电脑上运行:{}",
                    "the image engine cannot run on this computer: {}",
                    ready["message"].as_str().unwrap_or("?")
                )
            ),
            _ => bail!("unexpected first event from the image engine: {ready}"),
        }
        let device = ready["devices"]
            .as_array()
            .and_then(|a| {
                a.iter()
                    .filter_map(|d| {
                        let name = d["name"].as_str().unwrap_or("");
                        (!name.is_empty() && !name.eq_ignore_ascii_case("cpu"))
                            .then(|| d["description"].as_str().filter(|s| !s.is_empty()).unwrap_or(name).to_string())
                    })
                    .next()
            })
            .unwrap_or_default();

        send(&self.stdin, &cmd)?;
        let deadline = Instant::now() + LOAD_TIMEOUT;
        let mut warnings = Vec::new();
        loop {
            if Instant::now() > deadline {
                bail!(trf!("图像模型加载超时", "loading the image model timed out"));
            }
            let ev = match self.recv(&rx, Duration::from_millis(500)) {
                Ok(Some(ev)) => ev,
                Ok(None) => continue,
                Err(_) => return Err(self.died()),
            };
            match ev["event"].as_str() {
                Some("load_progress") => {
                    let (s, n) = (ev["step"].as_f64().unwrap_or(0.0), ev["steps"].as_f64().unwrap_or(0.0));
                    if n > 0.0 {
                        progress(((s / n) as f32 * 0.98).clamp(0.0, 0.98));
                    }
                }
                Some("log") if ev["level"] == "warn" => {
                    if let Some(t) = ev["text"].as_str() {
                        warnings.push(t.to_string());
                    }
                }
                Some("loaded") => {
                    if ev["supports_image"] == false {
                        bail!(trf!(
                            "这是视频模型,Chaty 目前只支持文生图模型",
                            "this is a video model; Chaty generates still images only"
                        ));
                    }
                    return Ok(Loaded {
                        version: ev["version"].as_str().unwrap_or_default().to_string(),
                        default_sampler: ev["default_sampler"].as_str().unwrap_or_default().to_string(),
                        default_scheduler: ev["default_scheduler"].as_str().unwrap_or_default().to_string(),
                        device: if cmd["backend"] == "cpu" { String::new() } else { device },
                        warnings,
                    });
                }
                Some("error") if ev["scope"] == "load" => {
                    let msg = ev["message"].as_str().unwrap_or("unknown error").to_string();
                    bail!("{}", explain(&msg));
                }
                _ => {}
            }
        }
    }

    /// Run one generation to its end, reporting as it goes. Blocking: call it
    /// from a blocking thread. `out_dir`/`file_stem` name the files written;
    /// `metadata` is embedded in PNGs under "parameters".
    pub fn generate(
        &self,
        p: &GenerateParams,
        out_dir: &Path,
        file_stem: &str,
        metadata: &str,
        mut on_event: impl FnMut(SdEvent),
    ) -> Result<GenerateOutcome> {
        if self.busy.swap(true, Ordering::SeqCst) {
            bail!(trf!("上一张图还在生成", "an image is already being generated"));
        }
        struct Idle<'a>(&'a AtomicBool);
        impl Drop for Idle<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _idle = Idle(&self.busy);

        let id = format!("g{}", self.seq.fetch_add(1, Ordering::SeqCst) + 1);
        let mut cmd = json!({
            "cmd": "generate",
            "id": id,
            "prompt": p.prompt,
            "negative_prompt": p.negative_prompt,
            "width": p.width,
            "height": p.height,
            "steps": p.steps,
            "cfg_scale": p.cfg_scale,
            "sampler": p.sampler,
            "scheduler": p.scheduler,
            "seed": p.seed,
            "batch_count": p.batch_count.max(1),
            "flow_shift": p.flow_shift,
            "vae_tiling": p.vae_tiling,
            "clip_skip": p.clip_skip,
            "preview": if p.preview.is_empty() { "proj" } else { p.preview.as_str() },
            "preview_interval": p.preview_interval.max(1),
            "format": if p.format == "jpg" { "jpg" } else { "png" },
            "out_dir": out_dir.to_string_lossy(),
            "file_stem": file_stem,
            "metadata": metadata,
            "strength": p.strength,
            "ref_images": p.ref_images,
        });
        if let Some(g) = p.guidance {
            cmd["guidance"] = json!(g);
        }
        if let Some(init) = p.init_image.as_ref().filter(|s| !s.is_empty()) {
            cmd["init_image"] = json!(init);
        }
        if let Some((mode, threshold)) = &p.cache {
            cmd["cache_mode"] = json!(mode);
            if *threshold > 0.0 {
                cmd["cache_threshold"] = json!(threshold);
            }
        }

        let rx = self.lines.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        // Anything left over from a job that ended abnormally is not ours.
        while let Ok(Some(_)) = self.recv(&rx, Duration::from_millis(0)) {}
        send(&self.stdin, &cmd)?;

        let mut images = Vec::new();
        let mut last = Instant::now();
        loop {
            let ev = match self.recv(&rx, Duration::from_millis(500)) {
                Ok(Some(ev)) => ev,
                Ok(None) => {
                    if last.elapsed() > IDLE_TIMEOUT {
                        bail!(trf!("图像引擎长时间无响应", "the image engine stopped responding"));
                    }
                    continue;
                }
                Err(_) => return Err(self.died()),
            };
            last = Instant::now();
            if ev.get("id").and_then(|v| v.as_str()).is_some_and(|v| v != id) {
                continue;
            }
            let u = |k: &str| ev[k].as_u64().unwrap_or(0) as u32;
            match ev["event"].as_str() {
                Some("stage") => on_event(SdEvent::Stage {
                    stage: ev["stage"].as_str().unwrap_or_default().to_string(),
                    index: u("index"),
                    count: u("count").max(1),
                    seed: ev["seed"].as_i64(),
                }),
                Some("progress") => on_event(SdEvent::Progress {
                    stage: ev["stage"].as_str().unwrap_or_default().to_string(),
                    step: u("step"),
                    steps: u("steps"),
                    secs: ev["time"].as_f64().unwrap_or(0.0) as f32,
                }),
                Some("preview") => on_event(SdEvent::Preview {
                    step: u("step"),
                    width: u("width"),
                    height: u("height"),
                    data_url: ev["data"].as_str().unwrap_or_default().to_string(),
                }),
                Some("cache") => on_event(SdEvent::Cache {
                    kind: ev["kind"].as_str().unwrap_or_default().to_string(),
                    skipped: u("skipped"),
                    total: u("total"),
                }),
                Some("image") => {
                    let path = ev["path"].as_str().unwrap_or_default().to_string();
                    let (w, h, seed) = (u("width"), u("height"), ev["seed"].as_i64().unwrap_or(0));
                    images.push((path.clone(), w, h, seed));
                    on_event(SdEvent::Image { index: u("index"), path, width: w, height: h, seed });
                }
                Some("done") => {
                    return Ok(GenerateOutcome {
                        images,
                        cancelled: ev["cancelled"].as_bool().unwrap_or(false),
                        elapsed_ms: ev["elapsed_ms"].as_u64().unwrap_or(0),
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

    /// Stop the running generation: "all" now, or "after_current" once the
    /// picture being drawn is finished (the rest of a batch is skipped).
    pub fn cancel(&self, mode: &str) {
        let _ = send(&self.stdin, &json!({ "cmd": "cancel", "mode": mode }));
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    fn kill(&self) {
        // Closing stdin lets a healthy sidecar exit by itself; the kill after
        // is for one that is busy or wedged, and wait() is what guarantees its
        // memory is back before this returns.
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

/// The sidecar process died — not an error it reported, a crash. A GPU driver
/// that aborts is the usual cause, which is why a load that ends this way is
/// worth one more try on the CPU.
#[derive(Debug)]
pub struct Crashed(pub String);
impl std::fmt::Display for Crashed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Crashed {}

/// The sidecar's channel closed: the process is gone.
#[derive(Debug)]
struct SidecarDied;
impl std::fmt::Display for SidecarDied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("sidecar exited")
    }
}
impl std::error::Error for SidecarDied {}

/// Put the engine's terse errors in words a user can act on; keep the
/// original after it for the report.
fn explain(msg: &str) -> String {
    let m = msg.to_lowercase();
    if m.contains("out of memory") || m.contains("failed to allocate") || m.contains("alloc") && m.contains("fail") {
        return format!(
            "{} ({msg})",
            trf!(
                "显存或内存不足。可以在 设置 → 生图模型 里开启「省显存」、把文本编码器或 VAE 放到 CPU,或换更小的量化",
                "out of GPU or system memory. Settings → Image model can save VRAM, move the text encoder or the VAE to the CPU — or use a smaller quantization"
            )
        );
    }
    if m.contains("get sd version") || m.contains("unknown model") || m.contains("cannot detect") {
        return format!(
            "{} ({msg})",
            trf!(
                "图像引擎认不出这个模型的结构,可能是不支持的模型或文件已损坏",
                "the image engine does not recognise this model — an unsupported architecture, or a damaged file"
            )
        );
    }
    msg.to_string()
}

impl Drop for SdEngine {
    fn drop(&mut self) {
        self.kill();
    }
}

#[async_trait]
impl InferenceBackend for SdEngine {
    fn name(&self) -> &str {
        "sd.cpp"
    }

    fn unload(&self) {
        self.kill();
    }

    fn as_image(&self) -> Option<&SdEngine> {
        Some(self)
    }

    async fn generate(&self, _req: GenRequest, sink: Channel<StreamEvent>, _cancel: Arc<AtomicBool>) -> Result<()> {
        let msg = trf!(
            "当前加载的是文生图模型,不能对话。请换一个对话模型",
            "the loaded model generates images and cannot chat — load a chat model"
        );
        let _ = sink.send(StreamEvent::Error { message: msg.clone() });
        Err(anyhow!(msg))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// The whole plumbing against a scripted fake sidecar: handshake, load
    /// progress, a generation's phases, a picture, and the end.
    #[test]
    fn a_scripted_sidecar_loads_and_generates() {
        let dir = std::env::temp_dir().join(format!("chaty-sd-mock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake-sd.sh");
        std::fs::write(
            &script,
            r#"#!/bin/sh
echo "engine chatter on stderr" >&2
echo '{"event":"ready","protocol":"1","devices":[{"name":"Vulkan0","description":"Test GPU"}]}'
read load
echo '{"event":"load_stage","component":"diffusion"}'
echo '{"event":"load_progress","step":5,"steps":10}'
echo '{"event":"loaded","version":"Qwen Image 2.1","supports_image":true,"default_sampler":"euler","default_scheduler":"simple"}'
read gen
id=$(echo "$gen" | sed 's/.*"id":"\([^"]*\)".*/\1/')
echo "$gen" > "$(dirname "$0")/gen.json"
echo '{"event":"stage","id":"'$id'","stage":"encode","index":0,"count":1}'
echo '{"event":"cache","id":"'$id'","kind":"conditioning"}'
echo '{"event":"stage","id":"'$id'","stage":"sample","index":0,"count":1,"seed":7}'
echo '{"event":"progress","id":"'$id'","stage":"sample","step":1,"steps":2,"time":0.5}'
echo '{"event":"preview","id":"'$id'","step":1,"width":8,"height":8,"data":"data:image/jpeg;base64,AA=="}'
echo '{"event":"cache","id":"'$id'","kind":"steps","skipped":1,"total":2}'
echo '{"event":"image","id":"'$id'","index":0,"path":"/tmp/x.png","width":64,"height":64,"seed":7}'
echo '{"event":"done","id":"'$id'","count":1,"cancelled":false,"elapsed_ms":12}'
read rest
"#,
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let fracs = Mutex::new(Vec::new());
        let (engine, loaded) = SdEngine::load(&script, json!({"cmd": "load"}), |f| fracs.lock().unwrap().push(f)).unwrap();
        assert_eq!(loaded.version, "Qwen Image 2.1");
        assert_eq!(loaded.device, "Test GPU");
        assert_eq!(fracs.lock().unwrap().as_slice(), &[0.49]);

        let mut events = Vec::new();
        let params = GenerateParams { cache: Some(("easycache".into(), 0.35)), ..Default::default() };
        let out = engine.generate(&params, &dir, "t", "", |e| events.push(e)).unwrap();
        let sent: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("gen.json")).unwrap()).unwrap();
        assert_eq!(sent["cache_mode"], "easycache", "the step cache goes to the engine");
        assert!((sent["cache_threshold"].as_f64().unwrap() - 0.35).abs() < 1e-6);
        assert_eq!(out.images.len(), 1);
        assert_eq!(out.images[0].3, 7);
        assert!(!out.cancelled);
        let kinds: Vec<&str> = events
            .iter()
            .map(|e| match e {
                SdEvent::Stage { stage, .. } => stage.as_str(),
                SdEvent::Progress { .. } => "progress",
                SdEvent::Preview { .. } => "preview",
                SdEvent::Image { .. } => "image",
                SdEvent::Cache { kind, .. } => kind.as_str(),
            })
            .collect();
        assert_eq!(kinds, vec!["encode", "conditioning", "sample", "progress", "preview", "steps", "image"]);
        assert!(matches!(&events[5], SdEvent::Cache { skipped: 1, total: 2, .. }));
        assert!(!engine.is_busy());
        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A sidecar that dies says so, with the last lines it wrote.
    #[test]
    fn a_dying_sidecar_reports_its_last_words() {
        let dir = std::env::temp_dir().join(format!("chaty-sd-die-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("die.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\necho '{\"event\":\"ready\",\"devices\":[]}'\nread load\necho 'ggml_vulkan: device lost' >&2\nexit 1\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let err = match SdEngine::load(&script, json!({"cmd": "load"}), |_| {}) {
            Err(e) => format!("{e:#}"),
            Ok(_) => panic!("load should fail"),
        };
        assert!(err.contains("device lost"), "{err}");
        assert!(matches!(SdEngine::load(&script, json!({"cmd": "load"}), |_| {}), Err(e) if e.downcast_ref::<Crashed>().is_some()));
        std::fs::remove_dir_all(&dir).ok();
    }
}
