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

/// Load a GGUF file and make it the active engine. Heavy and blocking, so the
/// actual load runs on a blocking thread.
#[tauri::command]
pub async fn load_model(
    state: State<'_, AppState>,
    path: String,
    gpu_layers: Option<i32>,
    n_ctx: Option<u32>,
) -> Result<ModelInfo, String> {
    // Free the currently-loaded model first, so switching models doesn't briefly
    // hold two of them in VRAM (which can OOM the new model's context).
    {
        *state.engine.write().await = None;
        *state.model.write().await = None;
    }

    let (engine, mut info) =
        tokio::task::spawn_blocking(move || LlamaEngine::load(&path, gpu_layers, n_ctx))
            .await
            .map_err(|e| format!("加载任务异常: {e}"))?
            .map_err(|e| format!("{e:#}"))?;

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
    std::fs::write(&path, content).map_err(|e| format!("写入文件失败: {e}"))
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
    for dir in model_dirs(app) {
        if std::fs::create_dir_all(&dir).is_ok() {
            // The install-dir one may be read-only; app-data one always works.
        }
    }
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
