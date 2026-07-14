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
    // Stored paths from before the folder-layout migration point at
    // `models/Foo.gguf`; the file now lives at `models/Foo/Foo.gguf`. Follow
    // it transparently — the returned ModelInfo carries the new path, which
    // the frontend persists, so this heals itself after one load.
    let path = {
        let p = std::path::Path::new(&path);
        if p.exists() {
            path
        } else {
            match (p.parent(), p.file_stem(), p.file_name()) {
                (Some(dir), Some(stem), Some(name)) => {
                    let migrated = dir.join(stem).join(name);
                    if migrated.exists() {
                        migrated.to_string_lossy().to_string()
                    } else {
                        path
                    }
                }
                _ => path,
            }
        }
    };
    // "Load from folder" is the one loading entry point for BOTH formats —
    // the canonical layout has been one-folder-per-model since the vision
    // pipeline. A GGUF folder resolves to its main weights file here; MLX
    // folders are routed to the sidecar below.
    let path = {
        let p = std::path::Path::new(&path);
        if p.is_dir() && !crate::inference::mlx::is_mlx_dir(p) {
            match main_gguf_in_dir(p) {
                Some(main) => main.to_string_lossy().to_string(),
                None => {
                    return Err(
                        "该文件夹里没有可加载的模型（需要 .gguf 权重，或 MLX 的 config.json + safetensors） \
                         (no loadable model in this folder — needs a .gguf, or an MLX config.json + safetensors)"
                            .into(),
                    )
                }
            }
        } else {
            path
        }
    };

    // MLX models are folders (config.json + safetensors) driven by the
    // Swift sidecar; GGUF stays on the in-process llama.cpp engine.
    let is_mlx = crate::inference::mlx::is_mlx_dir(std::path::Path::new(&path));
    #[cfg(not(target_os = "macos"))]
    if is_mlx {
        return Err(
            "MLX 模型仅支持 macOS (Apple Silicon)，请改用 GGUF 版本 \
             (MLX models run on macOS only — use a GGUF build instead)"
                .into(),
        );
    }

    let _ = on_progress.send(LoadProgress { phase: "eject", frac: 0.0 });

    // Eject the old model SYNCHRONOUSLY before loading the new one. Merely
    // dropping the Arc races the worker thread's teardown — both models end
    // up resident at once, which on unified memory swap-freezes the machine.
    state.cancel.store(true, Ordering::SeqCst);
    // Size of the model being ejected — used to VERIFY its memory actually
    // came back before we start the next load. MLX models live in a sidecar
    // process, so killing it *is* the release — nothing to verify in-process.
    let (old_size_mb, old_was_mlx) = {
        let guard = state.model.read().await;
        (
            guard.as_ref().and_then(|m| m.size_mb).unwrap_or(0),
            guard.as_ref().map_or(false, |m| m.backend == "mlx"),
        )
    };
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
            // Sidecar-hosted (MLX) models never occupied this process's
            // memory — the kill+wait in unload() already freed everything.
            if old_was_mlx {
                return Ok(());
            }
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
    // memory growth over the file size is a good fraction. (GGUF only — the
    // MLX sidecar loads in its own process and reports progress itself.)
    let done_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    if !is_mlx {
        let expected = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0).max(1);
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

    let result = if is_mlx {
        let chan = on_progress.clone();
        tokio::task::spawn_blocking(move || {
            crate::inference::mlx::MlxEngine::load(&path, n_ctx, |frac| {
                let _ = chan.send(LoadProgress { phase: "weights", frac });
            })
            .map(|(engine, info)| (Arc::new(engine) as Arc<dyn InferenceBackend>, info))
        })
        .await
    } else {
        tokio::task::spawn_blocking(move || {
            LlamaEngine::load(&path, gpu_layers, n_ctx)
                .map(|(engine, info)| (Arc::new(engine) as Arc<dyn InferenceBackend>, info))
        })
        .await
    };
    done_flag.store(true, Ordering::Relaxed);
    let (backend, mut info) = result
        .map_err(|e| format!("加载任务异常 (load task panicked): {e}"))?
        .map_err(|e| format!("{e:#}"))?;
    let _ = on_progress.send(LoadProgress { phase: "ready", frac: 1.0 });

    info.backend = backend.name().to_string();
    *state.engine.write().await = Some(backend);
    *state.model.write().await = Some(info.clone());
    Ok(info)
}

/// Eject the active model and return to the empty state, freeing its memory.
/// Same synchronous teardown as a model switch, minus the next load — there is
/// no second model going resident, so the post-eject memory verification (which
/// only guards against stacking two models) isn't needed here.
#[tauri::command]
pub async fn eject_model(state: State<'_, AppState>) -> Result<(), String> {
    // Ask any in-flight generation to stop, then drop the engine.
    state.cancel.store(true, Ordering::SeqCst);
    let old = state.engine.write().await.take();
    *state.model.write().await = None;
    if let Some(old) = old {
        // Block until the worker's memory is actually released (synchronous
        // teardown), mirroring the eject path in `load_model`.
        let _ = tokio::task::spawn_blocking(move || old.unload()).await;
    }
    // A full eject means "free everything" — also drop the cached embedding
    // model (bge-m3, ~730 MB) so the knowledge base doesn't keep it resident.
    let _ = tokio::task::spawn_blocking(crate::rag::embed_unload).await;
    state.cancel.store(false, Ordering::SeqCst);
    Ok(())
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
    /// On-disk size in MiB (for display + a hint at delete time).
    pub size_mb: Option<u64>,
    /// Paired vision encoder (mmproj) path — present for folder-layout vision
    /// models, so the picker can badge them before loading.
    pub mmproj: Option<String>,
    /// Weight format: "gguf" (llama.cpp) or "mlx" (Apple-Silicon sidecar).
    pub format: &'static str,
    /// Vision-capable once loaded (GGUF: paired mmproj; MLX: built-in tower).
    pub vision: bool,
}

/// The main weights inside a folder-layout GGUF model: the largest non-mmproj
/// `.gguf` (folders normally hold exactly one, plus an optional encoder).
fn main_gguf_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("gguf")))
        .filter(|p| {
            !p.file_name()
                .and_then(|s| s.to_str())
                .map_or(false, |n| n.to_lowercase().contains("mmproj"))
        })
        .collect();
    candidates.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    candidates.pop()
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

/// One-time layout migration: every loose `*.gguf` sitting directly in a
/// models root moves into its own folder (`models/Foo.gguf` →
/// `models/Foo/Foo.gguf`). The folder layout is the only one `list_models`
/// recognizes — one folder per model, with an optional `mmproj*.gguf` beside
/// the weights for vision models. Best-effort and idempotent; a same-volume
/// rename is instant even for 20 GB files.
pub fn migrate_models_layout(app: &tauri::AppHandle) {
    for dir in model_dirs(app) {
        migrate_models_dir(&dir);
    }
}

/// Loose MAIN model files sitting directly in a models root — the pre-folder
/// layout that `list_models` no longer shows. mmproj-only leftovers don't
/// count: they were never listed as models, so there's nothing user-visible
/// to organize.
fn loose_main_ggufs(dir: &std::path::Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension().map_or(false, |x| x.eq_ignore_ascii_case("gguf"))
                && !p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map_or(false, |n| n.to_lowercase().contains("mmproj"))
        })
        .collect()
}

/// Startup entry point for the models-folder migration.
///
/// Normal operation is unchanged: loose GGUFs are silently organized into the
/// folder layout on every launch (drop-in convenience). The ONE addition is a
/// friendly heads-up for users updating from an old version: on the FIRST
/// launch of a build that has this feature, if models are still sitting loose
/// in a models root, ask once with a native dialog and organize on click.
///
/// A marker file in app-data (survives updates) makes it strictly one-time.
/// Fresh installs and users with no loose files never see the dialog — the
/// marker is written silently on their first launch, and silent migration
/// takes over from then on.
pub fn migrate_or_prompt_models(app: &tauri::AppHandle) {
    let marker = app
        .path()
        .app_data_dir()
        .ok()
        .map(|d| d.join("models-migrate-prompted"));

    // Already past the one-time gate (or no app-data dir) → keep the existing
    // silent behavior.
    if marker.as_ref().map_or(true, |m| m.exists()) {
        migrate_models_layout(app);
        return;
    }
    let marker = marker.unwrap();

    let loose: usize = model_dirs(app).iter().map(|d| loose_main_ggufs(d).len()).sum();
    // Record that this build has done its one-time check, whatever happens next
    // (so a later drop-in on a fresh install never triggers the update dialog).
    let _ = std::fs::write(&marker, "1");

    if loose == 0 {
        // Fresh install / already organized — no dialog, just the usual silent
        // pass (also pairs any stray mmproj).
        migrate_models_layout(app);
        return;
    }

    prompt_models_migration(app, loose);
}

/// Show the one-time native dialog and organize loose models on confirmation.
fn prompt_models_migration(app: &tauri::AppHandle, n: usize) {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};

    let title = "整理模型文件 (Organize models)";
    let msg = format!(
        "检测到 {n} 个旧版本散放的模型文件。新版本按「一个模型一个文件夹」管理 —— 点击「立即整理」自动归位（只移动文件位置，不删除任何内容，20GB 大模型也是秒级完成）。\n选择「以后再说」则暂不整理。\n\nFound {n} loose model file(s) from an older version. This version keeps one folder per model — click Organize to move them into place (files are only relocated, never deleted; instant even for 20 GB models). Or choose Later to skip for now."
    );
    let handle = app.clone();
    app.dialog()
        .message(msg)
        .title(title)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "立即整理 (Organize)".into(),
            "以后再说 (Later)".into(),
        ))
        .show(move |organize| {
            if !organize {
                return;
            }
            migrate_models_layout(&handle);
            handle
                .dialog()
                .message(format!(
                    "已整理好 {n} 个模型，现在可以在模型列表里直接选用。\nOrganized {n} model(s) — they now appear in the model list."
                ))
                .title(title)
                .show(|_| {});
        });
}

fn migrate_models_dir(dir: &std::path::Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    let is_gguf =
        |p: &PathBuf| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("gguf"));
    let is_mmproj = |p: &PathBuf| {
        p.file_name()
            .and_then(|s| s.to_str())
            .map_or(false, |n| n.to_lowercase().contains("mmproj"))
    };

    // Pass 1: loose main GGUFs → their own folders.
    for p in &entries {
        if !p.is_file() || !is_gguf(p) || is_mmproj(p) {
            continue;
        }
        let (Some(stem), Some(name)) = (p.file_stem().and_then(|s| s.to_str()), p.file_name())
        else {
            continue;
        };
        let dest = dir.join(stem).join(name);
        if dest.exists() {
            eprintln!(
                "models migration: {} already exists; leaving {} in place",
                dest.display(),
                p.display()
            );
            continue;
        }
        if std::fs::create_dir_all(dir.join(stem)).is_err() {
            continue; // read-only dir (install/resource) — nothing to do
        }
        match std::fs::rename(p, &dest) {
            Ok(()) => eprintln!("models migration: {} -> {}", p.display(), dest.display()),
            Err(e) => eprintln!("models migration failed for {}: {e}", p.display()),
        }
    }

    // Pass 2: a loose mmproj joins the single name-affine model folder, if
    // exactly one matches. Ambiguous or unmatched ones stay in the root
    // (harmless — mmproj files are never listed as models).
    for p in &entries {
        if !p.is_file() || !is_gguf(p) || !is_mmproj(p) {
            continue;
        }
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
        let lower = name.to_lowercase();
        let mut hits: Vec<PathBuf> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for f in rd.flatten() {
                let fp = f.path();
                let Some(fstem) = fp.file_name().and_then(|s| s.to_str()) else { continue };
                if !fp.is_dir() {
                    continue;
                }
                let token = fstem
                    .to_lowercase()
                    .split(['-', '_', '.'])
                    .next()
                    .unwrap_or_default()
                    .to_string();
                if token.len() >= 3 && lower.contains(&token) {
                    hits.push(fp);
                }
            }
        }
        if let [folder] = hits.as_slice() {
            let dest = folder.join(name);
            if !dest.exists() && std::fs::rename(p, &dest).is_ok() {
                eprintln!("models migration: {} -> {}", p.display(), dest.display());
            }
        }
    }
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

/// Native webview page zoom (Settings → UI scale). CSS `zoom` reflows the
/// document but not the viewport, so fixed/vw elements overflow and the
/// backdrop shows through when shrinking; the platform zoom (WKWebView
/// pageZoom / WebView2 zoom factor) scales everything coherently instead.
#[tauri::command]
pub fn set_ui_zoom(window: tauri::WebviewWindow, factor: f64) -> Result<(), String> {
    window
        .set_zoom(factor.clamp(0.5, 2.0))
        .map_err(|e| e.to_string())
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

/// Reveal the app's data folder (conversation DB, models, KB indexes) in the
/// file manager so users can back it up or inspect what's stored on disk.
#[tauri::command]
pub fn open_data_dir(app: tauri::AppHandle) -> Result<String, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
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
pub fn open_html_report(
    app: tauri::AppHandle,
    html: String,
    name: Option<String>,
) -> Result<String, String> {
    let stem = name.unwrap_or_else(|| "page".into());
    // Keep Canvas pages out of the deep-research `reports` folder — they're a
    // different thing, so they get their own folder.
    let subdir = match stem.as_str() {
        "canvas" => "canvas",
        _ => "reports",
    };
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join(subdir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("{stem}-{ts}.html"));
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
    // A GGUF whose name mentions "mmproj" is a vision encoder, not a chat
    // model — paired via `find_mmproj`, never listed as its own entry.
    let is_mmproj = |p: &PathBuf| {
        p.file_name()
            .and_then(|s| s.to_str())
            .map_or(false, |n| n.to_lowercase().contains("mmproj"))
    };
    // Multi-part GGUFs: only the first shard is loadable (llama.cpp pulls in
    // the rest); later shards must not show up as their own models.
    let shard_of = |name: &str| -> Option<u32> {
        let stem = name.strip_suffix(".gguf").unwrap_or(name);
        if stem.len() > 15 {
            let tail = &stem[stem.len() - 15..];
            let tb = tail.as_bytes();
            if tb[0] == b'-'
                && tb[1..6].iter().all(|c| c.is_ascii_digit())
                && &tail[6..10] == "-of-"
                && tb[10..15].iter().all(|c| c.is_ascii_digit())
            {
                return tail[1..6].parse().ok();
            }
        }
        None
    };
    let push = |path: PathBuf, out: &mut Vec<ModelEntry>, seen: &mut HashSet<PathBuf>| {
        if !path
            .extension()
            .map_or(false, |x| x.eq_ignore_ascii_case("gguf"))
            || is_mmproj(&path)
        {
            return;
        }
        if let Some(n) = path
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|n| shard_of(&n.to_lowercase()))
        {
            if n != 1 {
                return;
            }
        }
        let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
        if seen.insert(canon) {
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let mmproj = crate::inference::llama::find_mmproj(&path.to_string_lossy())
                .map(|p| p.to_string_lossy().to_string());
            // Honest on-disk footprint: weights + paired vision encoder.
            let size_mb = std::fs::metadata(&path).ok().map(|m| {
                let extra = mmproj
                    .as_deref()
                    .and_then(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len())
                    .unwrap_or(0);
                (m.len() + extra) / (1024 * 1024)
            });
            out.push(ModelEntry {
                name,
                path: path.to_string_lossy().to_string(),
                size_mb,
                vision: mmproj.is_some(),
                mmproj,
                format: "gguf",
            });
        }
    };
    // MLX model folders (config.json + safetensors) sit directly under a
    // models root, same one-folder-per-model layout as GGUF. macOS only —
    // the sidecar is Apple-Silicon-specific, so don't tease them elsewhere.
    #[cfg(target_os = "macos")]
    let push_mlx = |path: &PathBuf, out: &mut Vec<ModelEntry>, seen: &mut HashSet<PathBuf>| {
        if !crate::inference::mlx::is_mlx_dir(path) {
            return;
        }
        let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
        if seen.insert(canon) {
            out.push(ModelEntry {
                name: path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                path: path.to_string_lossy().to_string(),
                size_mb: Some(crate::inference::mlx::mlx_dir_size_mb(path)),
                mmproj: None,
                format: "mlx",
                vision: crate::inference::mlx::mlx_dir_has_vision(path),
            });
        }
    };
    // Folder layout ONLY: models/<Name>/{model.gguf[, mmproj-*.gguf]} — one
    // folder per model. Loose GGUFs directly in a models root are migrated
    // into folders at startup (`migrate_models_layout`) and are deliberately
    // not listed, so the picker always reflects the canonical layout.
    for dir in model_dirs(&app) {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            #[cfg(target_os = "macos")]
            push_mlx(&path, &mut out, &mut seen);
            if let Ok(sub) = std::fs::read_dir(&path) {
                for e in sub.flatten() {
                    push(e.path(), &mut out, &mut seen);
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(out)
}

/// Permanently delete a GGUF file from the models folder. Guarded three ways:
/// the path must end in `.gguf`, must resolve to a location inside a known
/// models directory (no arbitrary file deletion), and must not be the model
/// currently loaded (eject it first).
#[tauri::command]
pub async fn delete_model_file(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    path: String,
) -> Result<(), String> {
    let target = PathBuf::from(&path);
    // MLX models are folders — remove the whole folder, behind the same
    // guards (must be a real MLX model, inside a models dir, not loaded).
    if target.is_dir() {
        if !crate::inference::mlx::is_mlx_dir(&target) {
            return Err(
                "只能删除模型文件夹 (only model folders can be deleted this way)".into()
            );
        }
        let canon = target
            .canonicalize()
            .map_err(|e| format!("文件夹不存在 (folder not found): {e}"))?;
        let in_models = model_dirs(&app)
            .iter()
            .any(|d| d.canonicalize().map_or(false, |dc| canon.starts_with(&dc)));
        if !in_models {
            return Err(
                "该文件夹不在模型文件夹内，已拒绝删除 (folder is outside the models folder; refusing to delete)"
                    .into(),
            );
        }
        if let Some(m) = state.model.read().await.as_ref() {
            let loaded = PathBuf::from(&m.path)
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(&m.path));
            if loaded == canon {
                return Err(
                    "无法删除正在使用的模型，请先卸载 (can't delete the model in use — eject it first)"
                        .into(),
                );
            }
        }
        return std::fs::remove_dir_all(&canon)
            .map_err(|e| format!("删除失败 (delete failed): {e}"));
    }
    if !target
        .extension()
        .map_or(false, |x| x.eq_ignore_ascii_case("gguf"))
    {
        return Err("只能删除 .gguf 模型文件 (only .gguf model files can be deleted)".into());
    }
    let canon = target
        .canonicalize()
        .map_err(|e| format!("文件不存在 (file not found): {e}"))?;
    let in_models = model_dirs(&app)
        .iter()
        .any(|d| d.canonicalize().map_or(false, |dc| canon.starts_with(&dc)));
    if !in_models {
        return Err(
            "该文件不在模型文件夹内，已拒绝删除 (file is outside the models folder; refusing to delete)"
                .into(),
        );
    }
    if let Some(m) = state.model.read().await.as_ref() {
        let loaded = PathBuf::from(&m.path)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(&m.path));
        if loaded == canon {
            return Err(
                "无法删除正在使用的模型，请先卸载 (can't delete the model in use — eject it first)"
                    .into(),
            );
        }
    }
    // Folder-layout vision models live in a dedicated subfolder together with
    // their mmproj — deleting the model removes the whole folder. A model
    // sitting directly in a models root is deleted alone (plus its paired
    // mmproj when nothing else would use it).
    let parent = canon.parent().map(PathBuf::from);
    let parent_is_models_root = parent.as_ref().map_or(true, |p| {
        model_dirs(&app)
            .iter()
            .any(|d| d.canonicalize().map_or(false, |dc| dc == *p))
    });
    if parent_is_models_root {
        let mmproj = crate::inference::llama::find_mmproj(&canon.to_string_lossy());
        std::fs::remove_file(&canon).map_err(|e| format!("删除失败 (delete failed): {e}"))?;
        // Remove a flat-dir mmproj orphaned by this delete (it only paired
        // because this was the folder's single main model).
        if let Some(p) = mmproj {
            if p.parent() == canon.parent() {
                let _ = std::fs::remove_file(p);
            }
        }
    } else {
        let dir = parent.expect("non-root parent");
        std::fs::remove_dir_all(&dir).map_err(|e| format!("删除失败 (delete failed): {e}"))?;
    }
    Ok(())
}

/// Copy a file from `src` to `dest` (used by "save screenshot to local…").
/// `dest` comes from the native save dialog, so it's a user-chosen location.
#[tauri::command]
pub async fn save_file(src: String, dest: String) -> Result<(), String> {
    tokio::task::spawn_blocking(move || std::fs::copy(&src, &dest).map(|_| ()))
        .await
        .map_err(|e| format!("保存任务异常 (save task failed): {e}"))?
        .map_err(|e| format!("保存失败 (save failed): {e}"))
}

/// Full-resolution image as a data URL (no downscaling) — for the crisp
/// screenshot preview modal. Only re-encodes if the file is huge.
#[tauri::command]
pub async fn image_data_url(path: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let bytes = std::fs::read(&path).map_err(|e| format!("读取失败 (read failed): {e}"))?;
        use base64::Engine as _;
        let lower = path.to_lowercase();
        // Serve PNG/JPEG/etc. verbatim so the preview is pixel-exact; downscale
        // only if the file is very large (> ~12 MB).
        if bytes.len() <= 12 * 1024 * 1024 {
            let mime = if lower.ends_with(".png") {
                "image/png"
            } else if lower.ends_with(".webp") {
                "image/webp"
            } else if lower.ends_with(".gif") {
                "image/gif"
            } else {
                "image/jpeg"
            };
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            return Ok(format!("data:{mime};base64,{b64}"));
        }
        let img = image::load_from_memory(&bytes).map_err(|e| format!("解码失败 (decode failed): {e}"))?;
        let scaled = img.thumbnail(2400, 2400);
        let mut buf = std::io::Cursor::new(Vec::new());
        scaled
            .to_rgb8()
            .write_to(&mut buf, image::ImageFormat::Jpeg)
            .map_err(|e| format!("编码失败 (encode failed): {e}"))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        Ok(format!("data:image/jpeg;base64,{b64}"))
    })
    .await
    .map_err(|e| format!("图片任务异常 (image task failed): {e}"))?
}

/// Downscaled data-URL thumbnail of a local image, for chat-bubble previews.
/// (WKWebView can't load arbitrary file paths without the asset protocol; a
/// small JPEG data URL sidesteps that and keeps the DOM light.)
#[tauri::command]
pub async fn image_thumb(path: String, max_dim: Option<u32>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let img = image::open(&path).map_err(|e| format!("无法读取图片 (failed to read image): {e}"))?;
        let max = max_dim.unwrap_or(512).clamp(64, 1024);
        let thumb = img.thumbnail(max, max);
        let mut buf = std::io::Cursor::new(Vec::new());
        // JPEG keeps photo thumbnails tiny; alpha is dropped (fine for previews).
        thumb
            .to_rgb8()
            .write_to(&mut buf, image::ImageFormat::Jpeg)
            .map_err(|e| format!("缩略图编码失败 (thumbnail encode failed): {e}"))?;
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        Ok(format!("data:image/jpeg;base64,{b64}"))
    })
    .await
    .map_err(|e| format!("缩略图任务异常 (thumbnail task failed): {e}"))?
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

/// One-shot vision analysis: run the loaded vision model on `images` + `prompt`
/// and return the full text (no streaming). Shared by Code mode (view_image /
/// browser screenshots), the knowledge base (image captions) and Canvas
/// (screenshot review). Errors clearly when the active model can't see images.
#[tauri::command]
pub async fn vision_query(
    state: State<'_, AppState>,
    images: Vec<String>,
    prompt: String,
    max_tokens: Option<u32>,
) -> Result<String, String> {
    let vision_ready = state
        .model
        .read()
        .await
        .as_ref()
        .map(|m| m.vision_ready)
        .unwrap_or(false);
    if !vision_ready {
        return Err(
            "当前模型不支持图像识别（未加载视觉编码器 mmproj）。(The active model can't see images — no vision encoder loaded.)"
                .into(),
        );
    }
    let Some(backend) = state.backend().await else {
        return Err("尚未加载模型 (no model loaded)".into());
    };
    for p in &images {
        if !std::path::Path::new(p).exists() {
            return Err(format!("图片不存在 (image not found): {p}"));
        }
    }
    state.cancel.store(false, Ordering::SeqCst);
    let req = crate::inference::GenRequest {
        messages: vec![crate::inference::ChatMessage {
            role: crate::inference::Role::User,
            content: prompt,
            images,
        }],
        params: crate::inference::GenParams {
            temperature: 0.3,
            max_tokens: max_tokens.unwrap_or(640),
            think: Some(false),
            ..Default::default()
        },
    };
    let text = backend
        .generate_collect(req, state.cancel.clone())
        .await
        .map_err(|e| format!("{e:#}"))?;
    // These callers want the answer, not any stray reasoning block.
    Ok(strip_think_blocks(&text).trim().to_string())
}

/// Remove `<think>…</think>` spans (and a dangling opener) from model output.
pub fn strip_think_blocks(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("<think>") {
        out.push_str(&rest[..i]);
        match rest[i..].find("</think>") {
            Some(j) => rest = &rest[i + j + "</think>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
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

#[cfg(test)]
mod tests {
    #[test]
    fn folder_resolves_to_main_gguf() {
        let tmp = std::env::temp_dir().join(format!("chaty-folder-load-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Empty folder → nothing to load.
        assert!(super::main_gguf_in_dir(&tmp).is_none());
        // mmproj alone is an encoder, not a model.
        std::fs::write(tmp.join("mmproj-F16.gguf"), [0u8; 4]).unwrap();
        assert!(super::main_gguf_in_dir(&tmp).is_none());
        // The largest non-mmproj gguf wins.
        std::fs::write(tmp.join("tiny-draft.gguf"), [0u8; 8]).unwrap();
        std::fs::write(tmp.join("model-Q4_K_M.gguf"), [0u8; 64]).unwrap();
        let main = super::main_gguf_in_dir(&tmp).expect("main gguf");
        assert_eq!(main.file_name().unwrap(), "model-Q4_K_M.gguf");
        let _ = std::fs::remove_dir_all(&tmp);
    }


    use super::{loose_main_ggufs, migrate_models_dir};
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("chaty-migrate-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn mk(dir: &PathBuf, name: &str) {
        std::fs::write(dir.join(name), b"x").unwrap();
    }

    #[test]
    fn migrates_loose_ggufs_into_folders() {
        let d = tmp("loose");
        mk(&d, "Qwen-7B-Chat.Q4_K_M.gguf");
        mk(&d, "notes.txt"); // non-gguf untouched
        mk(&d, "download.gguf.part"); // partial download untouched
        migrate_models_dir(&d);
        assert!(d.join("Qwen-7B-Chat.Q4_K_M/Qwen-7B-Chat.Q4_K_M.gguf").is_file());
        assert!(!d.join("Qwen-7B-Chat.Q4_K_M.gguf").exists());
        assert!(d.join("notes.txt").is_file());
        assert!(d.join("download.gguf.part").is_file());
        // idempotent: a second run changes nothing
        migrate_models_dir(&d);
        assert!(d.join("Qwen-7B-Chat.Q4_K_M/Qwen-7B-Chat.Q4_K_M.gguf").is_file());
    }

    #[test]
    fn loose_main_ggufs_counts_only_loose_main_files() {
        let d = tmp("loose-count");
        mk(&d, "Qwen-7B.Q4_K_M.gguf"); // loose main → counts
        mk(&d, "Gemma-4-E4B.gguf"); // loose main → counts
        mk(&d, "mmproj-Gemma-4-E4B.gguf"); // mmproj → never counts
        mk(&d, "readme.txt"); // non-gguf → never counts
        std::fs::create_dir_all(d.join("Already-Organized")).unwrap();
        std::fs::write(d.join("Already-Organized/model.gguf"), b"x").unwrap(); // in a folder → not loose
        assert_eq!(loose_main_ggufs(&d).len(), 2);
        // After migration nothing is loose → the dialog would not fire.
        migrate_models_dir(&d);
        assert_eq!(loose_main_ggufs(&d).len(), 0);
    }

    #[test]
    fn loose_mmproj_joins_unique_affine_folder() {
        let d = tmp("mmproj");
        mk(&d, "SmolVLM-500M-Instruct-Q8_0.gguf");
        mk(&d, "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf");
        migrate_models_dir(&d);
        let folder = d.join("SmolVLM-500M-Instruct-Q8_0");
        assert!(folder.join("SmolVLM-500M-Instruct-Q8_0.gguf").is_file());
        assert!(
            folder.join("mmproj-SmolVLM-500M-Instruct-Q8_0.gguf").is_file(),
            "affine mmproj should join the model's folder"
        );
    }

    #[test]
    fn unmatched_mmproj_stays_in_root() {
        let d = tmp("orphan");
        mk(&d, "SomeModel-Q4.gguf");
        mk(&d, "mmproj-F16.gguf"); // no shared stem token
        migrate_models_dir(&d);
        assert!(d.join("mmproj-F16.gguf").is_file(), "unmatched mmproj must not move");
    }

    #[test]
    fn collision_leaves_source_in_place() {
        let d = tmp("collide");
        std::fs::create_dir_all(d.join("model")).unwrap();
        std::fs::write(d.join("model/model.gguf"), b"existing").unwrap();
        std::fs::write(d.join("model.gguf"), b"loose").unwrap();
        migrate_models_dir(&d);
        assert_eq!(std::fs::read(d.join("model/model.gguf")).unwrap(), b"existing");
        assert!(d.join("model.gguf").is_file(), "loose file must survive a collision");
    }
}
