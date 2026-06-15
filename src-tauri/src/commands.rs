//! Tauri command surface exposed to the frontend.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use base64::Engine;
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::menu::{Menu, MenuItem};
use tauri::{Manager, State};

use crate::inference::llama::LlamaEngine;
use crate::inference::{GenRequest, InferenceBackend, ModelInfo, StreamEvent};
use crate::state::AppState;

/// Phase + fraction streamed to the UI while a model loads.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadProgress {
    /// "eject" (freeing the old model) | "weights" (loading) | "ready".
    pub phase: &'static str,
    pub frac: f32,
}

/// Load a GGUF file and make it the active engine. Heavy and blocking, so the
/// actual load runs on a blocking thread.
#[tauri::command]
pub async fn load_model(
    state: State<'_, AppState>,
    path: String,
    gpu_layers: Option<i32>,
    n_ctx: Option<u32>,
    on_progress: Channel<LoadProgress>,
) -> Result<ModelInfo, String> {
    let _ = on_progress.send(LoadProgress { phase: "eject", frac: 0.0 });

    // Eject the old model SYNCHRONOUSLY before loading the new one. Merely
    // dropping the Arc races the worker thread's teardown — both models end
    // up resident at once, which on unified memory swap-freezes the machine.
    state.cancel.store(true, Ordering::SeqCst);
    // Size of the model being ejected — used to VERIFY its memory actually
    // came back before we start the next load.
    let old_size_mb = state.model.read().await.as_ref().and_then(|m| m.size_mb).unwrap_or(0);
    let old = state.engine.write().await.take();
    *state.model.write().await = None;
    if let Some(old) = old {
        // Baseline BEFORE the unload — with the synchronous (malloc) teardown
        // most memory is already back by the time unload() returns, so a
        // post-unload baseline made the "drop by half" target unreachable and
        // every first switch raised a false alarm.
        let pre = {
            let pid = sysinfo::Pid::from_u32(std::process::id());
            let mut sys = sysinfo::System::new();
            sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            sys.process(pid).map(|p| p.memory()).unwrap_or(0)
        };
        let _ = tokio::task::spawn_blocking(move || old.unload()).await;

        // Verify the eject: footprint must fall to at most `pre - old/2`
        // (or to a small absolute floor). Failing loudly here beats wiring a
        // second model on top of an unreleased one and freezing the machine.
        let verified = tokio::task::spawn_blocking(move || {
            let pid = sysinfo::Pid::from_u32(std::process::id());
            let mut sys = sysinfo::System::new();
            let probe = |sys: &mut sysinfo::System| -> u64 {
                sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                sys.process(pid).map(|p| p.memory()).unwrap_or(0)
            };
            let old_bytes = old_size_mb.saturating_mul(1024 * 1024);
            // Tiny models / unknown size: nothing meaningful to verify.
            if old_bytes < 2 * 1024 * 1024 * 1024 {
                return Ok(());
            }
            let target = pre.saturating_sub(old_bytes / 2);
            let floor = 4u64 * 1024 * 1024 * 1024; // app baseline + slack
            let mut now = probe(&mut sys);
            for _ in 0..100 {
                // up to ~20 s
                if now <= target || now <= floor {
                    eprintln!(
                        "eject ok: footprint {} -> {} MiB",
                        pre / (1024 * 1024),
                        now / (1024 * 1024)
                    );
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
                now = probe(&mut sys);
            }
            Err(format!(
                "旧模型的内存未能释放（{} MiB → {} MiB），为避免系统卡死已中止本次加载，请重试或重启应用。\
                 (The previous model's memory was not released ({} → {} MiB); load aborted to avoid freezing the system — retry or restart the app.)",
                pre / (1024 * 1024),
                now / (1024 * 1024),
                pre / (1024 * 1024),
                now / (1024 * 1024)
            ))
        })
        .await
        .map_err(|e| format!("卸载校验任务异常 (eject check failed): {e}"))?;
        verified?;
    }
    state.cancel.store(false, Ordering::SeqCst);

    // Progress: llama-cpp-2 doesn't expose llama.cpp's load callback, but on
    // unified memory the weights stream into the process roughly linearly, so
    // memory growth over the file size is a good fraction.
    let expected = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0).max(1);
    let done_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let done = done_flag.clone();
        let chan = on_progress.clone();
        std::thread::spawn(move || {
            let pid = sysinfo::Pid::from_u32(std::process::id());
            let mut sys = sysinfo::System::new();
            sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            let base = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
            while !done.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(250));
                sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                let now = sys.process(pid).map(|p| p.memory()).unwrap_or(base);
                let frac = (now.saturating_sub(base) as f32 / expected as f32).min(0.99);
                let _ = chan.send(LoadProgress { phase: "weights", frac });
            }
        });
    }

    let result =
        tokio::task::spawn_blocking(move || LlamaEngine::load(&path, gpu_layers, n_ctx)).await;
    done_flag.store(true, Ordering::Relaxed);
    let (engine, mut info) = result
        .map_err(|e| format!("加载任务异常 (load task panicked): {e}"))?
        .map_err(|e| format!("{e:#}"))?;
    let _ = on_progress.send(LoadProgress { phase: "ready", frac: 1.0 });

    let backend: Arc<dyn InferenceBackend> = Arc::new(engine);
    info.backend = backend.name().to_string();
    *state.engine.write().await = Some(backend);
    *state.model.write().await = Some(info.clone());
    Ok(info)
}

/// Current model metadata, or `null` if nothing is loaded.
#[tauri::command]
pub async fn get_model(state: State<'_, AppState>) -> Result<Option<ModelInfo>, String> {
    Ok(state.model.read().await.clone())
}

/// CPU / RAM / GPU info + the compiled GPU backend, for the hardware panel.
#[tauri::command]
pub fn get_hardware_info() -> crate::gpu::HardwareInfo {
    crate::gpu::hardware()
}

/// Live VRAM usage of the primary GPU (polled by the hardware panel).
#[tauri::command]
pub fn get_gpu_usage() -> Option<crate::gpu::GpuUsage> {
    crate::gpu::gpu_usage()
}

/// Write `content` to `path` (used by conversation export after a save dialog).
#[tauri::command]
pub fn write_text_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| format!("写入文件失败 (failed to write file): {e}"))
}

/// Write base64 little-endian f32 mono PCM to `path` as a 16-bit PCM WAV file
/// (used to export the generated deep-dive podcast audio).
#[tauri::command]
pub fn write_wav_file(path: String, audio: String, sample_rate: u32) -> Result<(), String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio.as_bytes())
        .map_err(|e| format!("音频解码失败 (audio decode failed): {e}"))?;
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // mono, 16-bit
    let data_len = num_samples * 2;
    let mut out: Vec<u8> = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in &samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, out).map_err(|e| format!("写入文件失败 (failed to write file): {e}"))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
}

/// Directories scanned for `.gguf` files: a `models/` folder next to the
/// executable (the install dir) and one under app-data (always writable).
fn model_dirs(app: &tauri::AppHandle) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
        }
    }
    if let Ok(data) = app.path().app_data_dir() {
        dirs.push(data.join("models"));
    }
    if let Ok(res) = app.path().resource_dir() {
        dirs.push(res.join("models"));
    }
    dirs
}

/// Make sure at least one `models/` folder exists so users have a place to
/// drop GGUF files. Best-effort.
pub fn ensure_models_dir(app: &tauri::AppHandle) {
    // macOS: only the app-data dir. Creating a folder next to the executable
    // would put it *inside* the .app bundle (Contents/MacOS/models) — invisible
    // to the user, and it breaks the code-signature seal.
    #[cfg(target_os = "macos")]
    {
        if let Ok(data) = app.path().app_data_dir() {
            let _ = std::fs::create_dir_all(data.join("models"));
        }
    }
    #[cfg(not(target_os = "macos"))]
    for dir in model_dirs(app) {
        if std::fs::create_dir_all(&dir).is_ok() {
            // The install-dir one may be read-only; app-data one always works.
        }
    }
}

/// Open a file path or URL in the OS default app WITHOUT forking.
///
/// The `open` crate (which `tauri_plugin_opener` uses for its detached variant)
/// does a manual `fork()` to detach the child. In this process — a multithreaded
/// WebKit host — a `fork()` followed by any allocation in the child trips the
/// libmalloc fork-child assertion on macOS, crashing the whole app
/// (non-deterministically, hence "sometimes slow, often crashes"). A plain
/// `Command::spawn` goes through `posix_spawn`, which is atomic and fork-free, so
/// it's both safe and fast. A short reaper thread `wait()`s the launcher child
/// (which exits in milliseconds) so we don't leak zombies.
fn open_default(target: &str) -> Result<(), String> {
    #[allow(unused_mut)]
    let mut cmd;
    #[cfg(target_os = "macos")]
    {
        cmd = std::process::Command::new("/usr/bin/open");
        cmd.arg(target);
    }
    #[cfg(target_os = "windows")]
    {
        cmd = std::process::Command::new("cmd");
        cmd.args(["/C", "start", "", target]);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        cmd = std::process::Command::new("xdg-open");
        cmd.arg(target);
    }
    match cmd.spawn() {
        Ok(mut child) => {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            Ok(())
        }
        Err(e) => Err(format!("无法打开 (failed to open): {e}")),
    }
}

/// Open a URL (or local file) in the user's default browser/app. Used by the
/// frontend for source links and previews; fork-free (see `open_default`).
#[tauri::command]
pub fn open_external(target: String) -> Result<(), String> {
    open_default(&target)
}

/// Reveal the writable models folder in the file manager (Finder/Explorer),
/// creating it first if needed — on macOS it lives under ~/Library, which
/// users can't easily browse to by hand.
#[tauri::command]
pub fn open_models_dir(app: tauri::AppHandle) -> Result<String, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.to_string_lossy().to_string();
    open_default(&path)?;
    Ok(path)
}

/// Write an HTML document to app-data and open it in the default browser. Used
/// by Deep Research to export a report as PDF: WKWebView's own `window.print()`
/// is a no-op, but the system browser prints (and saves as PDF, CJK included)
/// reliably. Returns the file path.
#[tauri::command]
pub fn open_html_report(app: tauri::AppHandle, html: String) -> Result<String, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("reports");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("deep-research-{ts}.html"));
    std::fs::write(&path, html).map_err(|e| format!("写入文件失败 (failed to write file): {e}"))?;
    let p = path.to_string_lossy().to_string();
    open_default(&p)?;
    Ok(p)
}

/// List `.gguf` models discovered in the scanned directories, for the in-app
/// hot-swap picker.
#[tauri::command]
pub fn list_models(app: tauri::AppHandle) -> Result<Vec<ModelEntry>, String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for dir in model_dirs(&app) {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path
                .extension()
                .map_or(false, |x| x.eq_ignore_ascii_case("gguf"))
            {
                let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
                if seen.insert(canon) {
                    let name = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    out.push(ModelEntry {
                        name,
                        path: path.to_string_lossy().to_string(),
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(out)
}

/// Rebuild the system-tray menu in the given UI language (`"zh"` | `"en"`).
#[tauri::command]
pub fn set_tray_language(app: tauri::AppHandle, lang: String) -> Result<(), String> {
    let (show, quit) = if lang == "zh" {
        ("显示 Chaty", "退出")
    } else {
        ("Show Chaty", "Quit")
    };
    let show_i = MenuItem::with_id(&app, "show", show, true, None::<&str>)
        .map_err(|e| e.to_string())?;
    let quit_i = MenuItem::with_id(&app, "quit", quit, true, None::<&str>)
        .map_err(|e| e.to_string())?;
    let menu = Menu::with_items(&app, &[&show_i, &quit_i]).map_err(|e| e.to_string())?;
    if let Some(tray) = app.tray_by_id("main-tray") {
        tray.set_menu(Some(menu)).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Stream a completion. Tokens arrive on `on_event` as [`StreamEvent`]s.
#[tauri::command]
pub async fn generate(
    state: State<'_, AppState>,
    request: GenRequest,
    on_event: Channel<StreamEvent>,
) -> Result<(), String> {
    let Some(backend) = state.backend().await else {
        let _ = on_event.send(StreamEvent::Error {
            message: "尚未加载模型".into(),
        });
        return Err("no model loaded".into());
    };

    // Clear any stale cancel request from a previous run, then hand a fresh
    // handle to the engine.
    state.cancel.store(false, Ordering::SeqCst);
    let cancel = state.cancel.clone();

    backend
        .generate(request, on_event.clone(), cancel)
        .await
        .map_err(|e| {
            let msg = format!("{e:#}");
            let _ = on_event.send(StreamEvent::Error {
                message: msg.clone(),
            });
            msg
        })
}

/// Ask the in-flight generation (if any) to stop early.
#[tauri::command]
pub fn cancel_generation(state: State<'_, AppState>) {
    state.cancel.store(true, Ordering::SeqCst);
}

// ---------- Voice (STT / TTS via sherpa-onnx, CPU) ----------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynthAudio {
    /// base64 of little-endian f32 PCM samples.
    pub audio: String,
    pub sample_rate: u32,
}

fn voice_models_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("voice-models"))
}

/// Transcribe base64-encoded f32 PCM audio to text (Whisper).
#[tauri::command]
pub async fn transcribe(
    app: tauri::AppHandle,
    audio: String,
    sample_rate: u32,
) -> Result<String, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio.as_bytes())
        .map_err(|e| e.to_string())?;
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let dir = voice_models_dir(&app)?;
    crate::voice::transcribe(dir, samples, sample_rate)
        .await
        .map_err(|e| format!("{e:#}"))
}

/// Synthesize speech for `text` (Kokoro). Returns base64 f32 PCM + sample rate.
#[tauri::command]
pub async fn synthesize(
    app: tauri::AppHandle,
    text: String,
    speed: Option<f32>,
    sid: Option<i32>,
) -> Result<SynthAudio, String> {
    let dir = voice_models_dir(&app)?;
    let (samples, sample_rate) =
        crate::voice::synthesize(dir, text, speed.unwrap_or(1.0), sid.unwrap_or(0))
            .await
            .map_err(|e| format!("{e:#}"))?;

    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in &samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let audio = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(SynthAudio { audio, sample_rate })
}
