//! MLX engine — Apple-Silicon inference for folder-layout MLX models
//! (`models/<Name>/{config.json, *.safetensors, tokenizer…}`), the format
//! published by mlx-community on Hugging Face.
//!
//! The heavy lifting happens in a **sidecar process** (`chaty-mlx`, Swift,
//! built on mlx-swift-lm) driven over stdio JSON-lines. Process isolation is
//! deliberate: ejecting an MLX model kills the sidecar, so its memory comes
//! back unconditionally — no wired-memory balloons, no teardown races.
//!
//! Stop sequences are enforced here (boundary-safe holdback on the streamed
//! text, mirroring the llama.cpp engine); the sidecar is told to `cancel`
//! the moment a stop matches.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tauri::ipc::Channel;

use super::{GenRequest, GenStats, InferenceBackend, ModelInfo, StreamEvent};

/// A folder is an MLX model iff it has a `config.json` and at least one
/// `.safetensors` shard. (GGUF folders never contain either.)
pub fn is_mlx_dir(path: &Path) -> bool {
    if !path.is_dir() || !path.join("config.json").is_file() {
        return false;
    }
    std::fs::read_dir(path).map_or(false, |rd| {
        rd.flatten().any(|e| {
            e.path().extension().map_or(false, |x| x.eq_ignore_ascii_case("safetensors"))
        })
    })
}

/// The folder's config.json declares a vision tower (natively-multimodal
/// architectures like the Qwen3.5 family) — vision works once loaded.
pub fn mlx_dir_has_vision(path: &Path) -> bool {
    std::fs::read_to_string(path.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .map_or(false, |v| v.get("vision_config").is_some())
}

/// Sum of the safetensors shards, in MiB — the honest weight footprint.
pub fn mlx_dir_size_mb(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |x| x.eq_ignore_ascii_case("safetensors"))
                })
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum::<u64>()
        })
        .unwrap_or(0)
        / (1024 * 1024)
}

/// Locate the `chaty-mlx` sidecar binary. Checked in order: explicit env
/// override (tests), next to the app executable (bundled / `tauri dev`), the
/// staged build output, and the raw xcodebuild products dir (local dev).
pub fn find_sidecar() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHATY_MLX_SIDECAR") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("chaty-mlx"));
            candidates.push(dir.join("chaty-mlx-aarch64-apple-darwin"));
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest.join("binaries/chaty-mlx-aarch64-apple-darwin"));
    candidates
        .push(manifest.join("mlx-sidecar/.build/xcode/Build/Products/Release/chaty-mlx"));
    let existing: Vec<PathBuf> = candidates.into_iter().filter(|p| p.is_file()).collect();
    // The sidecar loads Metal kernels from `mlx-swift_Cmlx.bundle` NEXT TO
    // itself, so a copy without that bundle dies at startup with "failed to
    // load the default metallib" — which surfaces upstream as the unhelpful
    // "sidecar exited unexpectedly". `tauri build` stages a bare binary into
    // target/release/, and that copy sits next to the exe, so it would win the
    // search for anything run from there (dev builds, benches) while the
    // complete copy in binaries/ sat unused. Prefer a candidate whose bundle
    // is present; fall back to plain existence so an unknown-but-working
    // layout (the .app bundle, a future packaging) still resolves.
    existing
        .iter()
        .find(|p| p.with_file_name("mlx-swift_Cmlx.bundle").exists())
        .or_else(|| existing.first())
        .cloned()
}

// ---------------------------------------------------------------------------
// Boundary-safe stop-sequence holdback
// ---------------------------------------------------------------------------

/// Streams text through while holding back any suffix that could still grow
/// into a stop sequence. `feed` returns text safe to emit plus whether a stop
/// fully matched (matched text and everything after it is discarded).
struct StopWatch {
    stops: Vec<String>,
    pending: String,
}

impl StopWatch {
    fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            pending: String::new(),
        }
    }

    fn feed(&mut self, chunk: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (chunk.to_string(), false);
        }
        self.pending.push_str(chunk);
        // Full match → emit everything before the earliest one and stop.
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min();
        if let Some(i) = hit {
            let out = self.pending[..i].to_string();
            self.pending.clear();
            return (out, true);
        }
        // Otherwise hold back the longest tail that is a prefix of any stop.
        let keep = self
            .stops
            .iter()
            .map(|s| Self::partial_holdback(&self.pending, s))
            .max()
            .unwrap_or(0);
        let mut cut = self.pending.len() - keep;
        while cut > 0 && !self.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }

    /// Longest suffix of `pending` that equals a proper prefix of `stop`.
    fn partial_holdback(pending: &str, stop: &str) -> usize {
        let mut best = 0;
        for (i, _) in stop.char_indices().skip(1) {
            if i <= pending.len() && pending.ends_with(&stop[..i]) {
                best = best.max(i);
            }
        }
        best
    }

    /// Natural end of generation: nothing pending can complete a stop
    /// anymore, so release it.
    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

enum Job {
    Generate {
        req: GenRequest,
        sink: Channel<StreamEvent>,
        cancel: Arc<AtomicBool>,
        done: tokio::sync::oneshot::Sender<Result<()>>,
    },
    Collect {
        req: GenRequest,
        cancel: Arc<AtomicBool>,
        done: tokio::sync::oneshot::Sender<Result<String>>,
    },
}

pub struct MlxEngine {
    jobs: Sender<Job>,
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
}

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const LOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// Max silence mid-generation before we declare the sidecar wedged. Prefill
/// progress lines count as activity, so only a genuine stall trips this.
const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);

impl MlxEngine {
    /// Load `dir` (an MLX model folder) with the auto-located sidecar.
    pub fn load(
        dir: &str,
        n_ctx: Option<u32>,
        progress: impl Fn(f32),
    ) -> Result<(Self, ModelInfo)> {
        let sidecar = find_sidecar().ok_or_else(|| {
            anyhow!(
                "未找到 MLX 引擎组件 chaty-mlx，请重新安装应用 \
                 (the chaty-mlx sidecar is missing; please reinstall)"
            )
        })?;
        Self::load_with_sidecar(&sidecar, dir, n_ctx, progress)
    }

    pub fn load_with_sidecar(
        sidecar: &Path,
        dir: &str,
        n_ctx: Option<u32>,
        progress: impl Fn(f32),
    ) -> Result<(Self, ModelInfo)> {
        let dir_path = PathBuf::from(dir);
        if !is_mlx_dir(&dir_path) {
            bail!(
                "不是有效的 MLX 模型文件夹（需要 config.json 与 .safetensors） \
                 (not an MLX model folder: config.json + .safetensors required)"
            );
        }

        let mut child = Command::new(sidecar)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("无法启动 MLX 引擎 (failed to spawn sidecar) {sidecar:?}"))?;
        SIDECAR_PIDS.lock().unwrap().push(child.id());
        let mut stdin_pipe = child.stdin.take().context("sidecar stdin unavailable")?;
        let stdout = child.stdout.take().context("sidecar stdout unavailable")?;

        // Dedicated reader thread; everything downstream consumes parsed
        // lines through this channel and treats disconnect as sidecar death.
        let (line_tx, line_rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(|l| l.ok()) {
                if line_tx.send(line).is_err() {
                    break;
                }
            }
        });

        let recv_event = |timeout: Duration| -> Result<Value> {
            let deadline = Instant::now() + timeout;
            loop {
                let remain = deadline
                    .checked_duration_since(Instant::now())
                    .ok_or_else(|| anyhow!("MLX 引擎响应超时 (sidecar timed out)"))?;
                match line_rx.recv_timeout(remain) {
                    Ok(line) => {
                        if let Ok(v) = serde_json::from_str::<Value>(&line) {
                            if v.get("event").is_some() {
                                return Ok(v);
                            }
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        bail!("MLX 引擎响应超时 (sidecar timed out)")
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        bail!("MLX 引擎意外退出 (sidecar exited unexpectedly)")
                    }
                }
            }
        };

        // Handshake, then load.
        let ev = recv_event(READY_TIMEOUT)?;
        if ev["event"] != "ready" {
            bail!("MLX 引擎握手失败 (unexpected first event: {ev})");
        }
        writeln!(stdin_pipe, "{}", json!({ "cmd": "load", "path": dir, "nCtx": n_ctx }))?;
        stdin_pipe.flush()?;

        let loaded = {
            // The sidecar has no local-load progress callback, but weights
            // stream into ITS process roughly linearly — its RSS over the
            // shard bytes is a good fraction (same heuristic as the GGUF
            // path, one process over).
            let expected = mlx_dir_size_mb(&dir_path).saturating_mul(1024 * 1024).max(1);
            let pid = sysinfo::Pid::from_u32(child.id());
            let mut sys = sysinfo::System::new();
            let deadline = Instant::now() + LOAD_TIMEOUT;
            loop {
                let remain = deadline
                    .checked_duration_since(Instant::now())
                    .ok_or_else(|| anyhow!("MLX 模型加载超时 (load timed out)"))?;
                let ev = recv_event(remain.min(Duration::from_millis(250)).max(Duration::from_millis(50)))
                    .or_else(|e| {
                        // Distinguish "still loading" ticks from real death.
                        if e.to_string().contains("exited") {
                            Err(e)
                        } else {
                            Ok(json!({ "event": "tick" }))
                        }
                    })?;
                match ev["event"].as_str() {
                    Some("loadProgress") => {
                        progress(ev["frac"].as_f64().unwrap_or(0.0) as f32);
                    }
                    Some("loaded") => break ev["info"].clone(),
                    Some("error") => {
                        let msg = ev["message"].as_str().unwrap_or("unknown").to_string();
                        let _ = child.kill();
                        let _ = child.wait();
                        bail!("{msg}");
                    }
                    _ => {
                        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                        if let Some(p) = sys.process(pid) {
                            progress((p.memory() as f32 / expected as f32).min(0.99));
                        }
                    }
                }
            }
        };

        let name = dir_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "MLX model".into());
        let n_layer = loaded["nLayer"].as_u64().map(|v| v as u32);
        let info = ModelInfo {
            name: name.clone(),
            path: dir.to_string(),
            backend: "mlx".into(),
            loaded: true,
            arch: loaded["arch"].as_str().map(str::to_string),
            size_mb: loaded["sizeMb"].as_u64().or_else(|| Some(mlx_dir_size_mb(&dir_path))),
            params_b: loaded["paramsB"].as_f64(),
            n_ctx_train: loaded["nCtxTrain"].as_u64().map(|v| v as u32),
            n_ctx: loaded["nCtx"].as_u64().map(|v| v as u32),
            n_layer,
            // Unified memory: MLX always runs the whole model on the GPU.
            gpu_layers: n_layer.map(|v| v as i32).unwrap_or(-1),
            gpu_name: crate::gpu::detect_gpu().map(|g| g.name),
            model_name: Some(name),
            quant: loaded["quant"].as_str().map(str::to_string),
            n_embd: loaded["nEmbd"].as_u64().map(|v| v as u32),
            has_chat_template: loaded["hasChatTemplate"].as_bool().unwrap_or(false),
            supports_thinking: loaded["supportsThinking"].as_bool().unwrap_or(false),
            // MLX thinking control always goes through `GenParams.think`
            // (template kwarg or empty-<think> prefill in the sidecar) —
            // never the `/no_think` prompt-suffix switch.
            think_switch: false,
            supports_tools: loaded["supportsTools"].as_bool().unwrap_or(false),
            multimodal: loaded["multimodal"].as_bool().unwrap_or(false),
            // MLX VLMs carry their vision tower in the same weights — loaded
            // model ⇒ vision works; there is no separate mmproj to miss.
            vision_ready: loaded["multimodal"].as_bool().unwrap_or(false),
            mmproj: None,
            warning: None,
        };

        let child = Arc::new(Mutex::new(Some(child)));
        let stdin = Arc::new(Mutex::new(Some(stdin_pipe)));
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
        {
            let stdin = stdin.clone();
            std::thread::spawn(move || actor(jobs_rx, line_rx, stdin));
        }
        Ok((Self { jobs: jobs_tx, child, stdin }, info))
    }

    fn kill(&self) {
        // Closing stdin first lets a healthy sidecar exit on its own; the
        // kill after is for a wedged one. wait() is what guarantees the
        // memory is actually gone before we return.
        drop(self.stdin.lock().map(|mut s| s.take()));
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut c) = guard.take() {
                let pid = c.id();
                let _ = c.kill();
                let _ = c.wait();
                SIDECAR_PIDS.lock().unwrap().retain(|p| *p != pid);
            }
        }
    }
}

/// Live sidecar PIDs. The app's quit path exits via `libc::_exit` (dodging a
/// ggml teardown SIGABRT), which skips every destructor — without an explicit
/// reap, quitting while an MLX model is loaded orphans a sidecar that keeps
/// the entire model resident. lib.rs calls `kill_sidecars_now` on exit.
static SIDECAR_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

pub fn kill_sidecars_now() {
    let pids: Vec<u32> = std::mem::take(&mut *SIDECAR_PIDS.lock().unwrap());
    for pid in pids {
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
        #[cfg(not(unix))]
        let _ = pid;
    }
}

impl Drop for MlxEngine {
    fn drop(&mut self) {
        self.kill();
    }
}

#[async_trait]
impl InferenceBackend for MlxEngine {
    fn name(&self) -> &str {
        "mlx"
    }

    fn unload(&self) {
        self.kill();
    }

    async fn generate(
        &self,
        req: GenRequest,
        sink: Channel<StreamEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        let (done, rx) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Job::Generate { req, sink, cancel, done })
            .map_err(|_| anyhow!("MLX 引擎已停止 (engine stopped)"))?;
        rx.await.map_err(|_| anyhow!("MLX 引擎已停止 (engine stopped)"))?
    }

    async fn generate_collect(&self, req: GenRequest, cancel: Arc<AtomicBool>) -> Result<String> {
        let (done, rx) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Job::Collect { req, cancel, done })
            .map_err(|_| anyhow!("MLX 引擎已停止 (engine stopped)"))?;
        rx.await.map_err(|_| anyhow!("MLX 引擎已停止 (engine stopped)"))?
    }
}

// ---------------------------------------------------------------------------
// Actor: owns the sidecar conversation, one job at a time
// ---------------------------------------------------------------------------

fn actor(jobs: Receiver<Job>, lines: Receiver<String>, stdin: Arc<Mutex<Option<ChildStdin>>>) {
    for job in jobs {
        match job {
            Job::Generate { req, sink, cancel, done } => {
                let sink2 = sink.clone();
                let res = run_generation(&req, &lines, &stdin, &cancel, |ev| {
                    let _ = sink2.send(ev);
                });
                let _ = match res {
                    Ok(_) => done.send(Ok(())),
                    Err(e) => {
                        let _ = sink.send(StreamEvent::Error { message: format!("{e:#}") });
                        done.send(Err(e))
                    }
                };
            }
            Job::Collect { req, cancel, done } => {
                let text = Arc::new(Mutex::new(String::new()));
                let text2 = text.clone();
                let res = run_generation(&req, &lines, &stdin, &cancel, move |ev| {
                    if let StreamEvent::Token { text } = ev {
                        if let Ok(mut t) = text2.lock() {
                            t.push_str(&text);
                        }
                    }
                });
                let _ = done.send(res.map(|_| {
                    text.lock().map(|t| t.clone()).unwrap_or_default()
                }));
            }
        }
    }
}

fn send_cmd(stdin: &Arc<Mutex<Option<ChildStdin>>>, cmd: &Value) -> Result<()> {
    let mut guard = stdin
        .lock()
        .map_err(|_| anyhow!("stdin poisoned"))?;
    let pipe = guard
        .as_mut()
        .ok_or_else(|| anyhow!("MLX 引擎已卸载 (engine unloaded)"))?;
    writeln!(pipe, "{cmd}")?;
    pipe.flush()?;
    Ok(())
}

fn run_generation(
    req: &GenRequest,
    lines: &Receiver<String>,
    stdin: &Arc<Mutex<Option<ChildStdin>>>,
    cancel: &Arc<AtomicBool>,
    mut emit: impl FnMut(StreamEvent),
) -> Result<()> {
    emit(StreamEvent::Started);

    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|m| {
            json!({
                "role": match m.role {
                    super::Role::System => "system",
                    super::Role::User => "user",
                    super::Role::Assistant => "assistant",
                },
                "content": m.content,
                // Vision cap, TIGHTER than the GGUF engine's 2 MP: raw 2x
                // full-page screenshots (15+ MP) either blow past Metal
                // limits inside the sidecar's mlx_eval (fatalError →
                // "sidecar exited unexpectedly") or explode into more image
                // tokens than the context holds — and on a 48 GB box with a
                // 35 GB model resident, even the 2 MP encode's transient
                // activations are headroom the machine doesn't have (the
                // owner reproduced sidecar deaths reading its own
                // screenshots). 1 MP ≈ a thousand image tokens: screenshots
                // stay perfectly legible, the ViT transient halves twice.
                // Downscaled JPEGs are cached and keyed by
                // source path+size+mtime, so the sidecar's image KV cache
                // still hits across turns.
                "images": m.images.iter()
                    .map(|p| super::llama::downscale_for_vision_capped(p, 1_000_000))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let p = &req.params;
    send_cmd(
        stdin,
        &json!({
            "cmd": "generate",
            "messages": messages,
            "params": {
                "temperature": p.temperature,
                "topP": p.top_p,
                "topK": p.top_k,
                "minP": p.min_p,
                "repeatPenalty": p.repeat_penalty,
                "maxTokens": p.max_tokens,
                "seed": p.seed,
                "think": p.think,
            },
        }),
    )?;

    let mut watch = StopWatch::new(&p.stop);
    let mut stop_hit = false;
    let mut cancel_sent = false;
    let mut last_activity = Instant::now();

    loop {
        // The user hitting stop and a stop-sequence match funnel into the
        // same sidecar cancel; the sidecar answers with a final done event.
        if (cancel.load(Ordering::Relaxed) || stop_hit) && !cancel_sent {
            let _ = send_cmd(stdin, &json!({ "cmd": "cancel" }));
            cancel_sent = true;
        }
        let line = match lines.recv_timeout(Duration::from_millis(100)) {
            Ok(l) => l,
            Err(RecvTimeoutError::Timeout) => {
                if last_activity.elapsed() > IDLE_TIMEOUT {
                    bail!("MLX 生成超时无响应 (generation stalled)");
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                bail!("MLX 引擎意外退出 (sidecar exited unexpectedly)")
            }
        };
        last_activity = Instant::now();
        let Ok(ev) = serde_json::from_str::<Value>(&line) else { continue };
        match ev["event"].as_str() {
            Some("prefill") => {
                let processed = ev["processed"].as_u64().unwrap_or(0) as u32;
                let total = ev["total"].as_u64().unwrap_or(0) as u32;
                emit(StreamEvent::Prefill { processed, total });
            }
            Some("token") => {
                if stop_hit {
                    continue; // draining until the sidecar acks the cancel
                }
                let (out, hit) = watch.feed(ev["text"].as_str().unwrap_or(""));
                if !out.is_empty() {
                    emit(StreamEvent::Token { text: out });
                }
                if hit {
                    stop_hit = true;
                }
            }
            Some("done") => {
                if !stop_hit {
                    let tail = watch.flush();
                    if !tail.is_empty() {
                        emit(StreamEvent::Token { text: tail });
                    }
                }
                let reason = if stop_hit {
                    "stop".to_string()
                } else {
                    ev["stopReason"].as_str().unwrap_or("eos").to_string()
                };
                emit(StreamEvent::Done {
                    stats: GenStats {
                        prompt_tokens: ev["promptTokens"].as_u64().unwrap_or(0) as u32,
                        completion_tokens: ev["completionTokens"].as_u64().unwrap_or(0) as u32,
                        tokens_per_second: ev["tokensPerSecond"].as_f64().unwrap_or(0.0) as f32,
                        stop_reason: reason,
                        reused: ev["reused"].as_u64().unwrap_or(0) as u32,
                    },
                });
                return Ok(());
            }
            Some("error") => {
                bail!("{}", ev["message"].as_str().unwrap_or("unknown sidecar error"));
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    /// A sidecar copy without its Metal resource bundle dies at startup with
    /// an unhelpful "sidecar exited unexpectedly"; the search must prefer a
    /// complete copy over a bare one that merely sits closer to the exe.
    #[test]
    fn sidecar_search_prefers_the_copy_with_its_resource_bundle() {
        let tmp = std::env::temp_dir().join(format!("chaty-sidecar-{}", std::process::id()));
        let bare = tmp.join("staged");
        let full = tmp.join("shipped");
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::create_dir_all(&full).unwrap();
        std::fs::write(bare.join("chaty-mlx"), b"x").unwrap();
        std::fs::write(full.join("chaty-mlx"), b"x").unwrap();
        std::fs::create_dir_all(full.join("mlx-swift_Cmlx.bundle")).unwrap();

        let candidates = vec![bare.join("chaty-mlx"), full.join("chaty-mlx")];
        let existing: Vec<std::path::PathBuf> =
            candidates.into_iter().filter(|p| p.is_file()).collect();
        let picked = existing
            .iter()
            .find(|p| p.with_file_name("mlx-swift_Cmlx.bundle").exists())
            .or_else(|| existing.first())
            .cloned()
            .unwrap();
        assert_eq!(picked, full.join("chaty-mlx"), "must skip the bundle-less copy");

        // With no complete copy anywhere, fall back rather than find nothing.
        let only_bare = vec![bare.join("chaty-mlx")];
        let picked2 = only_bare
            .iter()
            .find(|p| p.with_file_name("mlx-swift_Cmlx.bundle").exists())
            .or_else(|| only_bare.first())
            .cloned()
            .unwrap();
        assert_eq!(picked2, bare.join("chaty-mlx"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    use super::*;
    use crate::inference::{ChatMessage, GenParams, Role};

    #[test]
    fn stopwatch_holds_back_and_trims() {
        let mut w = StopWatch::new(&["STOP".to_string()]);
        // "ST" could still become STOP → held back.
        let (out, hit) = w.feed("Hello ST");
        assert_eq!(out, "Hello ");
        assert!(!hit);
        // "STAR" breaks the partial match → releases everything safe.
        let (out, hit) = w.feed("AR");
        assert_eq!(out, "STAR");
        assert!(!hit);
        // Full match across chunks → trims the stop and everything after.
        let (out, hit) = w.feed(" go ST");
        assert_eq!(out, " go ");
        assert!(!hit);
        let (out, hit) = w.feed("OP tail");
        assert_eq!(out, "");
        assert!(hit);
    }

    #[test]
    fn stopwatch_flush_releases_partial() {
        let mut w = StopWatch::new(&["<END>".to_string()]);
        let (out, hit) = w.feed("done <EN");
        assert_eq!(out, "done ");
        assert!(!hit);
        assert_eq!(w.flush(), "<EN");
    }

    #[test]
    fn stopwatch_multibyte_safe() {
        let mut w = StopWatch::new(&["стоп".to_string()]);
        let (out, hit) = w.feed("привет ст");
        assert!(!hit);
        assert!(out.ends_with("привет ") || out == "привет ");
        let (out2, hit2) = w.feed("оп next");
        assert_eq!(out2, "");
        assert!(hit2);
    }

    #[test]
    fn detects_mlx_dirs() {
        let tmp = std::env::temp_dir().join(format!("chaty-mlx-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!is_mlx_dir(&tmp));
        std::fs::write(tmp.join("config.json"), "{}").unwrap();
        assert!(!is_mlx_dir(&tmp));
        std::fs::write(tmp.join("model.safetensors"), [0u8; 8]).unwrap();
        assert!(is_mlx_dir(&tmp));
        assert_eq!(mlx_dir_size_mb(&tmp), 0); // 8 bytes rounds to 0 MiB
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Full plumbing test against a scripted fake sidecar: handshake, load
    /// info mapping, token streaming, stop-sequence trimming.
    #[cfg(unix)]
    #[test]
    fn mock_sidecar_end_to_end() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = std::env::temp_dir().join(format!("chaty-mlx-mock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // A fake model folder…
        let model = tmp.join("TinyModel");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), "{}").unwrap();
        std::fs::write(model.join("model.safetensors"), [0u8; 16]).unwrap();

        // …and a fake sidecar that speaks the protocol.
        let script = tmp.join("mock-sidecar.sh");
        std::fs::write(
            &script,
            r#"#!/bin/bash
echo '{"event":"ready"}'
while IFS= read -r line; do
  case "$line" in
    *'"load"'*)
      echo '{"event":"loadProgress","frac":0.5}'
      echo '{"event":"loaded","info":{"arch":"qwen3","nLayer":2,"nEmbd":64,"nCtxTrain":4096,"nCtx":2048,"quant":"4-bit","sizeMb":123,"paramsB":0.6,"hasChatTemplate":true,"supportsThinking":true,"thinkArg":true,"multimodal":false}}'
      ;;
    *'"generate"'*)
      echo '{"event":"prefill","processed":512,"total":1000}'
      echo '{"event":"token","text":"Hello wo"}'
      echo '{"event":"token","text":"rld ST"}'
      echo '{"event":"token","text":"OP hidden tail"}'
      echo '{"event":"done","promptTokens":1000,"completionTokens":5,"tokensPerSecond":42.0,"stopReason":"eos"}'
      ;;
    *'"cancel"'*) : ;;
    *'"quit"'*) exit 0 ;;
  esac
done
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let progress = Arc::new(Mutex::new(Vec::<f32>::new()));
        let p2 = progress.clone();
        let (engine, info) = MlxEngine::load_with_sidecar(
            &script,
            model.to_str().unwrap(),
            Some(2048),
            move |f| p2.lock().unwrap().push(f),
        )
        .expect("mock load");

        assert_eq!(info.backend, "mlx");
        assert_eq!(info.arch.as_deref(), Some("qwen3"));
        assert_eq!(info.n_ctx, Some(2048));
        assert_eq!(info.quant.as_deref(), Some("4-bit"));
        assert!(info.supports_thinking);
        assert!(!info.think_switch);
        assert_eq!(info.gpu_layers, 2);
        assert!(!progress.lock().unwrap().is_empty());

        // Stop sequence "STOP" must trim mid-stream, boundary-safely.
        let req = GenRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hi".into(),
                images: vec![],
            }],
            params: GenParams { stop: vec!["STOP".into()], ..Default::default() },
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let text = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(engine.generate_collect(req, cancel))
            .expect("collect");
        assert_eq!(text, "Hello world ");

        engine.unload(); // kills the mock; must not hang
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

/// Real-model e2e — needs the compiled sidecar (Xcode build) and a local MLX
/// model folder:
///   CHATY_TEST_MLX=/path/to/models/Qwen3-4B-4bit \
///     cargo test --lib mlx_e2e -- --ignored --nocapture
#[cfg(test)]
mod mlx_e2e {
    use super::*;
    use crate::inference::{ChatMessage, GenParams, GenRequest, Role};

    fn model_dir() -> String {
        std::env::var("CHATY_TEST_MLX").expect("set CHATY_TEST_MLX=<mlx model dir>")
    }

    fn req(prompt: &str, params: GenParams) -> GenRequest {
        GenRequest {
            messages: vec![ChatMessage { role: Role::User, content: prompt.into(), images: vec![] }],
            params,
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    #[ignore]
    fn mlx_e2e_generate_stop_think_cancel_unload() {
        let dir = model_dir();
        let t0 = std::time::Instant::now();
        let (engine, info) =
            MlxEngine::load(&dir, Some(8192), |f| eprintln!("load {:.0}%", f * 100.0))
                .expect("load MLX model");
        eprintln!("loaded {} in {:?}: {info:?}", info.name, t0.elapsed());
        assert_eq!(info.backend, "mlx");
        assert!(info.loaded);
        assert!(info.n_ctx.unwrap_or(0) > 0);

        // 1. Plain generation answers a trivial question. (Kept to recall
        // knowledge, not arithmetic — tiny quants flub "2+2" with thinking
        // off, identically to the mlx-lm reference implementation.)
        let cancel = Arc::new(AtomicBool::new(false));
        let text = rt()
            .block_on(engine.generate_collect(
                req("Name the capital of France. Answer with one word.", GenParams {
                    max_tokens: 64,
                    think: Some(false),
                    ..Default::default()
                }),
                cancel.clone(),
            ))
            .expect("generate");
        eprintln!("answer: {text:?}");
        assert!(text.to_lowercase().contains("paris"), "expected 'Paris' in: {text}");
        assert!(!text.contains("<think>"), "think=false must suppress reasoning: {text}");

        // 2. Stop sequences trim mid-stream (engine-side holdback).
        let text = rt()
            .block_on(engine.generate_collect(
                req("Count from 1 to 20 as words, comma-separated: one, two, three,", GenParams {
                    max_tokens: 200,
                    stop: vec!["seven".into()],
                    think: Some(false),
                    ..Default::default()
                }),
                Arc::new(AtomicBool::new(false)),
            ))
            .expect("generate with stop");
        eprintln!("stopped: {text}");
        assert!(!text.contains("seven"), "stop sequence must trim: {text}");

        // 3. Second request on the same engine: KV template-prefix reuse must
        // not corrupt the context (coherent, clean output — knowledge-level
        // asserts are unreliable on 0.6B-class quants and belong to a 4B run).
        let text = rt()
            .block_on(engine.generate_collect(
                req("Name the capital of Japan. Answer with one word.", GenParams {
                    max_tokens: 64,
                    think: Some(false),
                    ..Default::default()
                }),
                Arc::new(AtomicBool::new(false)),
            ))
            .expect("turn 2");
        eprintln!("turn2: {text:?}");
        assert!(!text.trim().is_empty(), "turn 2 produced no text");
        assert!(!text.contains("<think>"), "think=false must hold on turn 2: {text}");

        // 4. Cancellation: flag pre-set → returns quickly with no panic.
        let cancelled = Arc::new(AtomicBool::new(true));
        let t1 = std::time::Instant::now();
        let _ = rt().block_on(engine.generate_collect(
            req("Write a 2000-word essay about oceans.", GenParams::default()),
            cancelled,
        ));
        assert!(t1.elapsed() < Duration::from_secs(30), "cancel must cut generation short");

        // 5. Unload kills the sidecar; further work must fail, not hang.
        engine.unload();
        let after = rt().block_on(engine.generate_collect(
            req("hello", GenParams::default()),
            Arc::new(AtomicBool::new(false)),
        ));
        assert!(after.is_err(), "generate after unload must error");
    }

    /// Streaming path: Started → (Prefill…) → Token… → Done with stats.
    /// Cross-conversation isolation: after generating in conversation A, a
    /// brand-new conversation B must never see A's content. Regression test
    /// for the MambaCache.offset bug — hybrid models reported offset 0, the
    /// engine concluded "nothing to trim", and B's prompt was appended on top
    /// of A's KV (the model visibly recalled A's prompt and derailed).
    #[test]
    #[ignore]
    fn mlx_e2e_no_cross_conversation_bleed() {
        let dir = model_dir();
        let (engine, info) = MlxEngine::load(&dir, Some(8192), |_| {}).expect("load");
        let hybrid = info.arch.as_deref().unwrap_or("").starts_with("qwen3_5");
        let rt = rt();

        let run = |content: &str| -> (String, u64) {
            let events = Arc::new(Mutex::new(Vec::<String>::new()));
            let ev2 = events.clone();
            let sink = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
                if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                    ev2.lock().unwrap().push(s);
                }
                Ok(())
            });
            let req = GenRequest {
                messages: vec![ChatMessage {
                    role: Role::User,
                    content: content.into(),
                    images: vec![],
                }],
                params: GenParams { max_tokens: 96, think: Some(false), temperature: 0.0, ..Default::default() },
            };
            rt.block_on(engine.generate(req, sink, Arc::new(AtomicBool::new(false))))
                .expect("generate");
            let evs = events.lock().unwrap();
            let mut text = String::new();
            let mut reused = 0;
            for e in evs.iter() {
                let v: serde_json::Value = serde_json::from_str(e).unwrap();
                match v["type"].as_str() {
                    Some("token") => text.push_str(v["text"].as_str().unwrap_or("")),
                    Some("done") => reused = v["stats"]["reused"].as_u64().unwrap_or(0),
                    _ => {}
                }
            }
            (text, reused)
        };

        // Conversation A: plant a distinctive canary.
        let (a, _) = run("The secret codeword is BANANA. Acknowledge with OK and nothing else.");
        eprintln!("conv A answer: {a:?}");

        // Conversation B: totally unrelated. Any trace of A = contamination.
        let (b, reused_b) =
            run("Name the capital of France. Answer with one word only.");
        eprintln!("conv B answer: {b:?}, reused: {reused_b} (hybrid={hybrid})");
        let lb = b.to_lowercase();
        assert!(lb.contains("paris"), "expected 'Paris' in: {b}");
        assert!(
            !lb.contains("banana") && !lb.contains("codeword") && !lb.contains("secret"),
            "conversation A leaked into B: {b}"
        );
        if hybrid {
            // Non-trimmable caches can never resume — anything else means the
            // old conversation's KV was reused (the contamination bug).
            assert_eq!(reused_b, 0, "hybrid model must re-prefill from scratch");
        }
        engine.unload();
    }

    #[test]
    #[ignore]
    fn mlx_e2e_stream_events() {
        let dir = model_dir();
        let (engine, _info) = MlxEngine::load(&dir, Some(8192), |_| {}).expect("load");
        let events = Arc::new(Mutex::new(Vec::<String>::new()));
        let ev2 = events.clone();
        let sink = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
            if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                ev2.lock().unwrap().push(s);
            }
            Ok(())
        });
        // A long prompt forces chunked prefill → Prefill events.
        let filler = "The quick brown fox jumps over the lazy dog. ".repeat(120);
        let req = GenRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: format!("{filler}\nSummarize the above in one short sentence."),
                images: vec![],
            }],
            params: GenParams { max_tokens: 64, think: Some(false), temperature: 0.0, ..Default::default() },
        };
        rt().block_on(engine.generate(req, sink, Arc::new(AtomicBool::new(false))))
            .expect("stream");
        let evs = events.lock().unwrap();
        let joined = evs.join("\n");
        eprintln!("--- events ---\n{joined}");
        assert!(joined.contains("\"started\""), "missing Started");
        assert!(joined.contains("\"prefill\""), "missing Prefill progress for a long prompt");
        assert!(joined.contains("\"token\""), "missing Token");
        assert!(joined.contains("\"done\""), "missing Done");
        assert!(joined.contains("tokensPerSecond"), "missing stats");
        engine.unload();
    }
}

/// MLX vision e2e — needs the compiled sidecar and a natively-multimodal MLX
/// model folder (e.g. mlx-community/Qwen3.5-2B-4bit):
///   CHATY_TEST_MLX_VLM=/path/to/models/Qwen3.5-2B-4bit \
///     cargo test --lib mlx_vlm_e2e -- --ignored --nocapture --test-threads=1
/// (serial: two parallel sidecars contend for Metal and answers get flaky)
#[cfg(test)]
mod mlx_vlm_e2e {
    use super::*;
    use crate::inference::{ChatMessage, GenParams, GenRequest, Role};

    #[test]
    #[ignore]
    fn mlx_vlm_e2e_sees_colors_then_chats() {
        let dir = std::env::var("CHATY_TEST_MLX_VLM").expect("set CHATY_TEST_MLX_VLM=<mlx vlm dir>");
        let (engine, info) = MlxEngine::load(&dir, Some(8192), |_| {}).expect("load VLM");
        assert!(info.multimodal, "config has a vision tower");
        assert!(info.vision_ready, "MLX VLM must report vision_ready");

        // A solid red square — unambiguous even for a 2B quant.
        let img = image::RgbImage::from_pixel(96, 96, image::Rgb([220, 20, 20]));
        let img_path = std::env::temp_dir().join("chaty-mlx-vlm-red.png");
        img.save(&img_path).expect("write test image");

        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = GenRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: "What is the dominant color of this image? Answer with one word."
                    .into(),
                images: vec![img_path.to_string_lossy().to_string()],
            }],
            params: GenParams { max_tokens: 160, think: Some(false), temperature: 0.0, ..Default::default() },
        };
        let text = rt
            .block_on(engine.generate_collect(req, Arc::new(AtomicBool::new(false))))
            .expect("vision generate");
        eprintln!("vision answer: {text:?}");
        assert!(text.to_lowercase().contains("red"), "expected 'red' in: {text}");

        // A text-only follow-up must still work (image turn resets the KV
        // cache; the next turn re-prefils clean).
        let req = GenRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: "Name the capital of France. One word.".into(),
                images: vec![],
            }],
            params: GenParams { max_tokens: 64, think: Some(false), temperature: 0.0, ..Default::default() },
        };
        let text = rt
            .block_on(engine.generate_collect(req, Arc::new(AtomicBool::new(false))))
            .expect("text follow-up");
        eprintln!("follow-up: {text:?}");
        assert!(text.to_lowercase().contains("paris"), "expected 'Paris' in: {text}");
        engine.unload();
    }

    /// Community quants sometimes ship a VLM checkpoint without its processor
    /// configs (no preprocessor_config.json / processor_config.json) — the
    /// sidecar heals the folder by synthesizing one from config.json's
    /// vision_config instead of refusing to load. Prove it on a symlink clone
    /// of the test model with those files stripped.
    // MLX runs through the Apple-Silicon sidecar; unix-only (symlink) so the
    // Windows `cargo test` build doesn't trip over it.
    #[cfg(unix)]
    #[test]
    #[ignore]
    fn mlx_vlm_e2e_heals_missing_processor_configs() {
        let dir = std::env::var("CHATY_TEST_MLX_VLM").expect("set CHATY_TEST_MLX_VLM=<mlx vlm dir>");
        let clone = std::env::temp_dir().join("chaty-mlx-heal-e2e");
        let _ = std::fs::remove_dir_all(&clone);
        std::fs::create_dir_all(&clone).expect("mkdir clone");
        for entry in std::fs::read_dir(&dir).expect("read model dir").flatten() {
            let name = entry.file_name();
            let n = name.to_string_lossy().to_string();
            if n.contains("preprocessor_config") || n == "processor_config.json" {
                continue; // the files the broken quants forgot to ship
            }
            std::os::unix::fs::symlink(entry.path(), clone.join(&name)).expect("symlink");
        }

        let (engine, info) =
            MlxEngine::load(clone.to_str().unwrap(), Some(8192), |_| {}).expect("healed load");
        assert!(info.vision_ready, "healed VLM must still be vision-ready");
        assert!(
            clone.join("preprocessor_config.json").is_file(),
            "sidecar must synthesize preprocessor_config.json"
        );

        // The synthesized processor must actually work end to end.
        let img = image::RgbImage::from_pixel(96, 96, image::Rgb([220, 20, 20]));
        let img_path = std::env::temp_dir().join("chaty-mlx-heal-red.png");
        img.save(&img_path).expect("write test image");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = GenRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: "What is the dominant color of this image? Answer with one word.".into(),
                images: vec![img_path.to_string_lossy().to_string()],
            }],
            params: GenParams { max_tokens: 160, think: Some(false), temperature: 0.0, ..Default::default() },
        };
        let text = rt
            .block_on(engine.generate_collect(req, Arc::new(AtomicBool::new(false))))
            .expect("vision generate on healed clone");
        eprintln!("healed-clone vision answer: {text:?}");
        assert!(text.to_lowercase().contains("red"), "expected 'red' in: {text}");
        engine.unload();
        let _ = std::fs::remove_dir_all(&clone);
    }

    /// A 2x full-page browser screenshot easily reaches 15+ megapixels. Fed
    /// raw, the vision tower blows past Metal's limits inside mlx_eval and
    /// mlx-swift's error handler fatalError()s the whole sidecar ("MLX 引擎
    /// 意外退出"). The engine must downscale oversized images to the shared
    /// vision cap before they cross the stdio boundary — same as GGUF.
    #[test]
    #[ignore]
    fn mlx_vlm_e2e_oversized_screenshot_survives() {
        let dir = std::env::var("CHATY_TEST_MLX_VLM").expect("set CHATY_TEST_MLX_VLM=<mlx vlm dir>");
        let (engine, _info) = MlxEngine::load(&dir, Some(8192), |_| {}).expect("load VLM");

        // Tall red "page" at 2x-screenshot proportions: 2200x7000 = 15.4 MP.
        let img = image::RgbImage::from_pixel(2200, 7000, image::Rgb([220, 20, 20]));
        let img_path = std::env::temp_dir().join("chaty-mlx-vlm-huge-red.png");
        img.save(&img_path).expect("write huge test image");

        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = GenRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: "What is the dominant color of this image? Answer with one word."
                    .into(),
                images: vec![img_path.to_string_lossy().to_string()],
            }],
            params: GenParams { max_tokens: 160, think: Some(false), temperature: 0.0, ..Default::default() },
        };
        let text = rt
            .block_on(engine.generate_collect(req, Arc::new(AtomicBool::new(false))))
            .expect("oversized image must not kill the sidecar");
        eprintln!("huge-image answer: {text:?}");
        assert!(text.to_lowercase().contains("red"), "expected 'red' in: {text}");

        // The sidecar must still be alive and conversational afterwards.
        let req = GenRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: "Name the capital of France. One word.".into(),
                images: vec![],
            }],
            params: GenParams { max_tokens: 64, think: Some(false), temperature: 0.0, ..Default::default() },
        };
        let text = rt
            .block_on(engine.generate_collect(req, Arc::new(AtomicBool::new(false))))
            .expect("text follow-up after huge image");
        assert!(text.to_lowercase().contains("paris"), "expected 'Paris' in: {text}");
        engine.unload();
    }

    /// Image-turn prefill progress + the GGUF media-cache analogue: a text
    /// follow-up in the same conversation must not re-encode the image on
    /// dense VLMs (qwen3_vl — trimmable cache resumes past the media prefix,
    /// so its first prefill event starts beyond zero), while hybrid ones
    /// (qwen3_5 — Mamba state can't rewind) re-prefill from zero, still with
    /// progress events. Both variants must answer correctly.
    #[test]
    #[ignore]
    fn mlx_vlm_e2e_prefill_progress_and_reuse() {
        let dir = std::env::var("CHATY_TEST_MLX_VLM").expect("set CHATY_TEST_MLX_VLM=<mlx vlm dir>");
        let (engine, info) = MlxEngine::load(&dir, Some(8192), |_| {}).expect("load VLM");
        assert!(info.vision_ready, "MLX VLM must report vision_ready");
        let arch = info.arch.clone().unwrap_or_default();

        let img = image::RgbImage::from_pixel(96, 96, image::Rgb([220, 20, 20]));
        let img_path = std::env::temp_dir().join("chaty-mlx-vlm-red-reuse.png");
        img.save(&img_path).expect("write test image");

        let rt = tokio::runtime::Runtime::new().unwrap();
        let run = |messages: Vec<ChatMessage>| -> (String, Vec<(u64, u64)>) {
            let events = Arc::new(Mutex::new(Vec::<String>::new()));
            let ev2 = events.clone();
            let sink = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
                if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                    ev2.lock().unwrap().push(s);
                }
                Ok(())
            });
            let req = GenRequest {
                messages,
                params: GenParams { max_tokens: 160, think: Some(false), temperature: 0.0, ..Default::default() },
            };
            rt.block_on(engine.generate(req, sink, Arc::new(AtomicBool::new(false))))
                .expect("generate");
            let evs = events.lock().unwrap();
            let mut text = String::new();
            let mut prefills = Vec::new();
            for e in evs.iter() {
                let v: serde_json::Value = serde_json::from_str(e).unwrap();
                match v["type"].as_str() {
                    Some("prefill") => prefills.push((
                        v["processed"].as_u64().unwrap_or(0),
                        v["total"].as_u64().unwrap_or(0),
                    )),
                    Some("token") => text.push_str(v["text"].as_str().unwrap_or("")),
                    _ => {}
                }
            }
            (text, prefills)
        };

        // Turn 1: the vision turn itself must show prefill progress (the
        // Code-mode ring) — media turns always emit, starting at zero.
        let img_msg = ChatMessage {
            role: Role::User,
            content: "What is the dominant color of this image? Answer with one word.".into(),
            images: vec![img_path.to_string_lossy().to_string()],
        };
        let (answer1, prefills1) = run(vec![img_msg.clone()]);
        eprintln!("turn1 answer: {answer1:?}, prefills: {prefills1:?}");
        assert!(answer1.to_lowercase().contains("red"), "expected 'red' in: {answer1}");
        assert!(!prefills1.is_empty(), "vision turn must emit prefill progress");
        assert_eq!(prefills1[0].0, 0, "fresh vision turn starts at zero");

        // Turn 2: same conversation grows by the reply and a long text
        // follow-up (the Code-mode tool-loop shape).
        let filler = "Notes: the quick brown fox jumps over the lazy dog. ".repeat(60);
        let (answer2, prefills2) = run(vec![
            img_msg,
            ChatMessage { role: Role::Assistant, content: answer1.clone(), images: vec![] },
            ChatMessage {
                role: Role::User,
                content: format!("{filler}\nNow name the capital of France. One word."),
                images: vec![],
            },
        ]);
        eprintln!("turn2 answer: {answer2:?}, prefills: {prefills2:?}");
        assert!(answer2.to_lowercase().contains("paris"), "expected 'Paris' in: {answer2}");
        assert!(!prefills2.is_empty(), "long follow-up must emit prefill progress");
        if arch == "qwen3_vl" {
            assert!(
                prefills2[0].0 > 0,
                "dense VLM must resume past the cached image prefix, got {:?}",
                prefills2
            );
        } else {
            assert_eq!(
                prefills2[0].0, 0,
                "hybrid VLM re-prefills from zero (Mamba state can't rewind)"
            );
        }
        engine.unload();
    }
}

/// Memory hygiene across repeated load/eject cycles — the sidecar must die
/// on every unload and the PARENT process must not accumulate memory (the
/// weights live in the child, so parent growth means a plumbing leak).
///   CHATY_TEST_MLX=<model dir> cargo test --lib mlx_e2e_memory_cycles -- --ignored --nocapture
#[cfg(test)]
mod mlx_mem_e2e {
    use super::*;
    use crate::inference::{ChatMessage, GenParams, GenRequest, Role};

    fn sidecar_pid(engine: &MlxEngine) -> Option<u32> {
        engine.child.lock().ok()?.as_ref().map(|c| c.id())
    }

    fn pid_alive(pid: u32) -> bool {
        std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "pid="])
            .output()
            .map_or(false, |o| !o.stdout.is_empty())
    }

    #[test]
    #[ignore]
    fn mlx_e2e_memory_cycles() {
        let dir = std::env::var("CHATY_TEST_MLX").expect("set CHATY_TEST_MLX=<mlx model dir>");
        let me = sysinfo::Pid::from_u32(std::process::id());
        let mut sys = sysinfo::System::new();
        let probe = |sys: &mut sysinfo::System| -> u64 {
            sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[me]), true);
            sys.process(me).map(|p| p.memory()).unwrap_or(0)
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let base = probe(&mut sys);

        for cycle in 0..5 {
            let (engine, _info) = MlxEngine::load(&dir, Some(4096), |_| {}).expect("load");
            let pid = sidecar_pid(&engine).expect("sidecar pid");
            assert!(pid_alive(pid), "cycle {cycle}: sidecar not running after load");

            let req = GenRequest {
                messages: vec![ChatMessage {
                    role: Role::User,
                    content: "Say OK.".into(),
                    images: vec![],
                }],
                params: GenParams { max_tokens: 8, think: Some(false), temperature: 0.0, ..Default::default() },
            };
            let text = rt
                .block_on(engine.generate_collect(req, Arc::new(AtomicBool::new(false))))
                .expect("generate");
            assert!(!text.trim().is_empty(), "cycle {cycle}: empty generation");

            engine.unload();
            assert!(!pid_alive(pid), "cycle {cycle}: sidecar survived unload — memory NOT released");
        }

        let growth_mb = probe(&mut sys).saturating_sub(base) / (1024 * 1024);
        eprintln!("parent RSS growth after 5 load/eject cycles: {growth_mb} MiB");
        assert!(growth_mb < 200, "parent process leaked {growth_mb} MiB across cycles");
    }
}
