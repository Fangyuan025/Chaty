//! Text-to-image: diffusion GGUFs (Qwen-Image, Z-Image, FLUX, Stable
//! Diffusion …) run by the stable-diffusion.cpp sidecar.
//!
//! Loading one turns the whole app into an image studio. This module is the
//! backend of that: recognising the model and what it needs (`probe`,
//! `family`), loading it into the engine, running generations with live
//! progress, and keeping what they made.

pub mod family;
pub mod probe;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::ipc::Channel;
use tauri::{Manager, State};

use crate::inference::sd::{GenerateParams, SdEngine, SdEvent};
use crate::inference::{InferenceBackend, ModelInfo};
use crate::state::AppState;
use crate::store::{Db, ImageItem, ImageRecord};
use family::Role;
use probe::{Component, ImageProbe, LoadOptions};

/// What the UI knows about a loaded image model, beside the usual ModelInfo.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageModelInfo {
    pub family: String,
    pub family_name: String,
    /// What stable-diffusion.cpp identified the weights as ("Qwen Image 2.1").
    pub engine_version: String,
    pub components: Vec<Component>,
    pub defaults: family::Defaults,
    /// Reference-picture editing is available.
    pub edits: bool,
    pub default_sampler: String,
    pub default_scheduler: String,
    /// The GPU the engine runs on, "" on the CPU.
    pub device: String,
    pub on_cpu: bool,
    /// "sd.cpp" or "mlx" — what the studio can offer depends on it.
    pub engine: String,
    /// The engine's samplers and schedulers; empty = stable-diffusion.cpp's.
    pub samplers: Vec<String>,
    pub schedulers: Vec<String>,
}

fn sidecar_key(role: Role) -> &'static str {
    match role {
        Role::Vae => "vae",
        Role::Llm => "llm",
        Role::LlmVision => "llm_vision",
        Role::ClipL => "clip_l",
        Role::ClipG => "clip_g",
        Role::T5xxl => "t5xxl",
    }
}

/// The sidecar's `load` command for a probed model.
fn load_cmd(p: &ImageProbe, o: &LoadOptions, cpu: bool) -> Value {
    let mut c = json!({ "cmd": "load" });
    c[if p.all_in_one { "model" } else { "diffusion_model" }] = json!(p.path);
    for comp in &p.components {
        c[sidecar_key(comp.role)] = json!(comp.path);
    }
    c["diffusion_fa"] = json!(o.flash_attn);
    c["mmap"] = json!(o.mmap);
    c["clip_on_cpu"] = json!(o.text_encoder_on_cpu);
    c["vae_on_cpu"] = json!(o.vae_on_cpu);
    if cpu {
        c["backend"] = json!("cpu");
    } else {
        c["offload_to_cpu"] = json!(o.offload_to_cpu);
        if o.max_vram_gb > 0.0 {
            c["max_vram"] = json!(format!("{}", o.max_vram_gb));
        }
    }
    if o.threads > 0 {
        c["threads"] = json!(o.threads);
    }
    c
}

/// Name the roles a model is missing, for the one error that must be read.
fn roles_text(roles: &[Role]) -> String {
    roles
        .iter()
        .map(|r| match r {
            Role::Vae => "VAE".to_string(),
            Role::Llm => trf!("文本编码器", "text encoder"),
            Role::LlmVision => trf!("视觉编码器", "vision encoder"),
            Role::ClipL => "CLIP-L".into(),
            Role::ClipG => "CLIP-G".into(),
            Role::T5xxl => "T5-XXL".into(),
        })
        .collect::<Vec<_>>()
        .join(trf!("、", ", ").as_str())
}

/// Load the image model at `path` into a fresh sidecar. Blocking.
///
/// GPU by default, every part of the model on it and only what does not fit
/// spilled over — the way the chat engine offloads every layer it can. When
/// the engine dies while loading onto the GPU (a driver that aborts, as in
/// issue #5) it gets one more try on the CPU, with a warning to say so.
pub fn load(
    path: &Path,
    opts: &LoadOptions,
    roots: &[PathBuf],
    progress: impl Fn(f32),
) -> anyhow::Result<(Arc<dyn InferenceBackend>, ModelInfo)> {
    let probe = probe::probe_image_model(path, &opts.components, roots)
        .ok_or_else(|| anyhow::anyhow!(trf!("这不是文生图模型", "not an image generation model")))?;
    if probe.engine == "mlx" {
        return load_mlx(path, &probe, progress);
    }
    if !probe.missing.is_empty() {
        anyhow::bail!(trf!(
            "{} 还缺少配套文件:{}。请在模型菜单里下载配套文件,或在 设置 → 生图模型 中手动指定。",
            "{} is missing files it needs: {}. Download them from the model menu, or choose them in Settings → Image model.",
            probe.family_name,
            roles_text(&probe.missing)
        ));
    }
    let sidecar = crate::inference::sd::find_sidecar().ok_or_else(|| {
        anyhow::anyhow!(trf!(
            "未找到图像引擎组件 chaty-sd,请重新安装应用",
            "the image engine (chaty-sd) is missing; please reinstall"
        ))
    })?;

    let want_cpu = opts.device == "cpu";
    let mut warning = None;
    let (engine, loaded, on_cpu) = match SdEngine::load(&sidecar, load_cmd(&probe, opts, want_cpu), &progress) {
        Ok((e, l)) => (e, l, want_cpu),
        Err(e) if !want_cpu && e.downcast_ref::<crate::inference::sd::Crashed>().is_some() => {
            crate::errlog::append_error("image-model-load", &format!("path: {}\nGPU load crashed, retrying on CPU\n{e:#}", path.display()));
            warning = Some("image-gpu-crash-cpu".to_string());
            let (e, l) = SdEngine::load(&sidecar, load_cmd(&probe, opts, true), &progress)?;
            (e, l, true)
        }
        Err(e) => return Err(e),
    };

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image model").to_string();
    let device = if on_cpu { String::new() } else if !loaded.device.is_empty() { loaded.device.clone() } else {
        crate::gpu::detect_gpu().map(|g| g.name).unwrap_or_default()
    };
    let total_mb = std::fs::metadata(path).map(|m| m.len() / (1024 * 1024)).unwrap_or(0)
        + probe.components.iter().map(|c| c.size_mb).sum::<u64>();
    let image = ImageModelInfo {
        family: probe.family.clone(),
        family_name: probe.family_name.clone(),
        engine_version: loaded.version.clone(),
        components: probe.components.clone(),
        defaults: probe.defaults.clone(),
        edits: probe.edits,
        default_sampler: loaded.default_sampler.clone(),
        default_scheduler: loaded.default_scheduler.clone(),
        device: device.clone(),
        on_cpu,
        engine: "sd.cpp".into(),
        samplers: loaded.samplers.clone(),
        schedulers: loaded.schedulers.clone(),
    };
    let info = ModelInfo {
        name: name.clone(),
        path: path.to_string_lossy().to_string(),
        backend: "sd.cpp".into(),
        loaded: true,
        arch: Some(probe.family_name.clone()),
        size_mb: Some(total_mb),
        params_b: probe.params_b,
        n_ctx_train: None,
        n_ctx: None,
        n_layer: None,
        gpu_layers: if on_cpu { 0 } else { -1 },
        gpu_name: (!device.is_empty()).then_some(device),
        model_name: Some(name),
        quant: probe.quant.clone(),
        n_embd: None,
        has_chat_template: false,
        supports_thinking: false,
        think_switch: false,
        effort_levels: Vec::new(),
        tool_role: false,
        reasoning_field: false,
        tool_format: None,
        supports_tools: false,
        multimodal: false,
        vision_ready: false,
        multi_image: false,
        mmproj: None,
        speculative: false,
        speculative_on: false,
        warning,
        kind: "image".into(),
        image: Some(image),
    };
    Ok((Arc::new(engine), info))
}

/// Load an MLX image model folder into the MLX sidecar's image mode.
fn load_mlx(
    path: &Path,
    probe: &ImageProbe,
    progress: impl Fn(f32),
) -> anyhow::Result<(Arc<dyn InferenceBackend>, ModelInfo)> {
    if !cfg!(target_os = "macos") {
        anyhow::bail!(trf!(
            "MLX 模型仅支持 macOS (Apple Silicon),请改用 GGUF 版本",
            "MLX models run on macOS (Apple Silicon) only — use a GGUF build of this model"
        ));
    }
    if !probe::MLX_FAMILIES.contains(&probe.family.as_str()) {
        if probe.family == "generic" {
            anyhow::bail!(trf!(
                "Chaty 的 MLX 生图引擎还不支持这个模型的结构",
                "Chaty's MLX image engine does not run this model's architecture yet"
            ));
        }
        anyhow::bail!(trf!(
            "{} 的 MLX 版本 Chaty 还不支持,请改用它的 GGUF 版本",
            "Chaty cannot run the MLX build of {} yet — use its GGUF build",
            probe.family_name
        ));
    }
    let sidecar = crate::inference::mlx::find_sidecar().ok_or_else(|| {
        anyhow::anyhow!(trf!(
            "未找到 MLX 引擎组件 chaty-mlx,请重新安装应用",
            "the MLX engine (chaty-mlx) is missing; please reinstall"
        ))
    })?;
    // Always the GPU, whatever the image engine's device setting says: MLX's
    // CPU backend runs a quantized model on one core, and a picture that
    // takes seconds per step on the GPU did not finish two steps in twelve
    // minutes there.
    let cmd = json!({ "cmd": "load", "model": path });
    let (engine, loaded) = SdEngine::load_with(&sidecar, &["--image"], "mlx", cmd, &progress)?;
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("image model").to_string();
    let device = loaded.device.clone();
    let image = ImageModelInfo {
        family: probe.family.clone(),
        family_name: probe.family_name.clone(),
        engine_version: loaded.version.clone(),
        components: Vec::new(),
        defaults: probe.defaults.clone(),
        edits: probe.edits,
        default_sampler: loaded.default_sampler.clone(),
        default_scheduler: loaded.default_scheduler.clone(),
        device: device.clone(),
        on_cpu: false,
        engine: "mlx".into(),
        samplers: loaded.samplers.clone(),
        schedulers: loaded.schedulers.clone(),
    };
    let info = ModelInfo {
        name: name.clone(),
        path: path.to_string_lossy().to_string(),
        backend: "mlx".into(),
        loaded: true,
        arch: Some(probe.family_name.clone()),
        size_mb: Some(probe.size_mb),
        params_b: probe.params_b,
        n_ctx_train: None,
        n_ctx: None,
        n_layer: None,
        gpu_layers: -1,
        gpu_name: (!device.is_empty()).then_some(device),
        model_name: Some(name),
        quant: probe.quant.clone(),
        n_embd: None,
        has_chat_template: false,
        supports_thinking: false,
        think_switch: false,
        effort_levels: Vec::new(),
        tool_role: false,
        reasoning_field: false,
        tool_format: None,
        supports_tools: false,
        multimodal: false,
        vision_ready: false,
        multi_image: false,
        mmproj: None,
        speculative: false,
        speculative_on: false,
        warning: None,
        kind: "image".into(),
        image: Some(image),
    };
    Ok((Arc::new(engine), info))
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Everything about an image model before loading it: its family, what it
/// needs, what is there, and the downloads for what is not. `None` when the
/// file is not an image model.
#[tauri::command]
pub async fn image_model_probe(
    app: tauri::AppHandle,
    path: String,
    components: Option<probe::Overrides>,
) -> Result<Option<ImageProbe>, String> {
    let roots = crate::commands::model_dirs(&app);
    tauri::async_runtime::spawn_blocking(move || {
        let p = PathBuf::from(&path);
        let target = if p.is_dir() { probe::image_model_in_dir(&p) } else { Some(p) };
        target.and_then(|t| probe::probe_image_model(&t, &components.unwrap_or_default(), &roots))
    })
    .await
    .map_err(|e| e.to_string())
}

/// Where pictures are written when no folder is chosen: app-data/images,
/// one folder per day.
fn default_output_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("images"))
}

#[tauri::command]
pub fn image_output_dir(app: tauri::AppHandle) -> Result<String, String> {
    let dir = default_output_root(&app)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.to_string_lossy().to_string())
}

/// A generation request from the studio.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageRequest {
    #[serde(flatten)]
    pub params: GenerateParams,
    /// Folder the pictures go to; empty = app-data/images/<date>.
    #[serde(default)]
    pub out_dir: Option<String>,
    /// The session this round belongs to. Absent: a session of its own.
    #[serde(default)]
    pub session_id: Option<String>,
    /// The round whose picture this one starts from (multi-turn editing).
    #[serde(default)]
    pub parent_id: Option<String>,
}

/// What the studio hears while a generation runs.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ImageEvent {
    #[serde(rename_all = "camelCase")]
    Started { id: String, request: ImageRequest, started_at: i64 },
    #[serde(rename_all = "camelCase")]
    Engine { event: SdEvent },
    #[serde(rename_all = "camelCase")]
    Done { record: Option<ImageRecord>, cancelled: bool },
    #[serde(rename_all = "camelCase")]
    Error { message: String },
}

/// A generation in progress, held by the app so a page that reloads under it
/// can pick it back up (the webview can be replaced mid-run, as the chat side
/// learned).
struct LiveImage {
    id: String,
    request: ImageRequest,
    started_at: i64,
    listener: Arc<Mutex<Option<Channel<ImageEvent>>>>,
    /// The latest of each kind of report, replayed to a page that attaches.
    stage: Option<SdEvent>,
    progress: Option<SdEvent>,
    preview: Option<SdEvent>,
    images: Vec<SdEvent>,
}

static LIVE: Mutex<Option<LiveImage>> = Mutex::new(None);

/// A running generation, as a page that just arrived needs it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveImageInfo {
    pub id: String,
    pub request: ImageRequest,
    pub started_at: i64,
    /// Replay: stage, progress, preview and finished pictures so far.
    pub events: Vec<SdEvent>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn fire(listener: &Arc<Mutex<Option<Channel<ImageEvent>>>>, ev: ImageEvent) {
    if let Ok(l) = listener.lock() {
        if let Some(ch) = l.as_ref() {
            let _ = ch.send(ev);
        }
    }
}

/// The A1111-style parameters block every Stable Diffusion tool reads back
/// out of a PNG. The engine appends the seed of each picture.
fn metadata_text(p: &GenerateParams, model: &str) -> String {
    let mut s = p.prompt.clone();
    if !p.negative_prompt.trim().is_empty() {
        s.push_str(&format!("\nNegative prompt: {}", p.negative_prompt));
    }
    s.push_str(&format!("\nSteps: {}, CFG scale: {}", p.steps, p.cfg_scale));
    if let Some(g) = p.guidance {
        s.push_str(&format!(", Distilled CFG: {g}"));
    }
    if !p.sampler.is_empty() {
        s.push_str(&format!(", Sampler: {}", p.sampler));
    }
    if !p.scheduler.is_empty() {
        s.push_str(&format!(", Schedule type: {}", p.scheduler));
    }
    s.push_str(&format!(", Size: {}x{}, Model: {model}, Generator: Chaty", p.width, p.height));
    s
}

/// Generate pictures with the loaded image model. Progress, previews and each
/// finished picture stream on `on_event`; the finished generation is saved to
/// the history and returned.
#[tauri::command]
pub async fn image_generate(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    request: ImageRequest,
    on_event: Channel<ImageEvent>,
) -> Result<Option<ImageRecord>, String> {
    let backend = state.backend().await;
    let Some(backend) = backend.filter(|b| b.as_image().is_some()) else {
        let msg = trf!("尚未加载文生图模型", "no image model is loaded");
        let _ = on_event.send(ImageEvent::Error { message: msg.clone() });
        return Err(msg);
    };
    let (model_name, family) = state
        .model
        .read()
        .await
        .as_ref()
        .map(|m| (m.name.clone(), m.image.as_ref().map(|i| i.family.clone()).unwrap_or_default()))
        .unwrap_or_default();

    let started_at = now_ms();
    let stamp = chrono::Local::now();
    let out_dir = match request.out_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(d) => PathBuf::from(d),
        None => default_output_root(&app)?.join(stamp.format("%Y-%m-%d").to_string()),
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| {
        trf!("无法创建输出文件夹 {}:{}", "cannot create the output folder {}: {}", out_dir.display(), e)
    })?;
    let id = format!("{}-{:04x}", stamp.format("%Y%m%d%H%M%S%3f"), rand_u16());
    // The studio names its session before the first round, as the chat does
    // a conversation; any other caller gets one named after the prompt.
    let session_id = match request.session_id.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => s.to_string(),
        None => {
            let title: String = request.params.prompt.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(40).collect();
            crate::store::image_session_ensure(&app.state::<Db>(), &id, &title)?;
            id.clone()
        }
    };
    let stem = stamp.format("%Y%m%d-%H%M%S").to_string();
    let meta = metadata_text(&request.params, &model_name);

    let listener = Arc::new(Mutex::new(Some(on_event.clone())));
    if let Ok(mut live) = LIVE.lock() {
        *live = Some(LiveImage {
            id: id.clone(),
            request: request.clone(),
            started_at,
            listener: listener.clone(),
            stage: None,
            progress: None,
            preview: None,
            images: Vec::new(),
        });
    }
    fire(&listener, ImageEvent::Started { id: id.clone(), request: request.clone(), started_at });

    let mut params = request.params.clone();
    params.cache = family::step_cache(&family, &params.accel).map(|(m, t)| (m.to_string(), t));
    let dir = out_dir.clone();
    let l2 = listener.clone();
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let engine = backend.as_image().expect("checked above");
        engine.generate(&params, &dir, &stem, &meta, |ev| {
            if let Ok(mut live) = LIVE.lock() {
                if let Some(l) = live.as_mut() {
                    match &ev {
                        SdEvent::Stage { .. } => {
                            l.stage = Some(ev.clone());
                            l.progress = None;
                        }
                        SdEvent::Progress { .. } => l.progress = Some(ev.clone()),
                        SdEvent::Preview { .. } => l.preview = Some(ev.clone()),
                        SdEvent::Image { .. } => l.images.push(ev.clone()),
                        SdEvent::Cache { .. } => {}
                    }
                }
            }
            fire(&l2, ImageEvent::Engine { event: ev });
        })
    })
    .await
    .map_err(|e| e.to_string())?;

    let result = match outcome {
        Ok(out) => {
            let record = (!out.images.is_empty()).then(|| ImageRecord {
                id: id.clone(),
                prompt: request.params.prompt.clone(),
                negative_prompt: request.params.negative_prompt.clone(),
                params: serde_json::to_value(&request.params).unwrap_or(Value::Null),
                images: out
                    .images
                    .iter()
                    .map(|(path, width, height, seed)| ImageItem { path: path.clone(), width: *width, height: *height, seed: *seed })
                    .collect(),
                model: model_name.clone(),
                family: family.clone(),
                created_at: started_at,
                elapsed_ms: out.elapsed_ms as i64,
                session_id: session_id.clone(),
                parent_id: request.parent_id.clone(),
            });
            let record = match record {
                Some(r) => match crate::store::image_record_insert(&app.state::<Db>(), &r) {
                    Ok(true) => Some(r),
                    // The session was deleted while this round was drawn: like
                    // a chat reply to a deleted conversation, it is dropped —
                    // and so are its pictures, which nothing points to now.
                    Ok(false) => {
                        for im in &r.images {
                            let _ = std::fs::remove_file(&im.path);
                        }
                        None
                    }
                    Err(e) => {
                        crate::errlog::append_error("image-history-save", &e);
                        Some(r)
                    }
                },
                None => None,
            };
            fire(&listener, ImageEvent::Done { record: record.clone(), cancelled: out.cancelled });
            Ok(record)
        }
        Err(e) => {
            let msg = format!("{e:#}");
            crate::errlog::append_error("image-generate", &format!("model: {model_name}\n{msg}"));
            fire(&listener, ImageEvent::Error { message: msg.clone() });
            Err(msg)
        }
    };
    if let Ok(mut live) = LIVE.lock() {
        if live.as_ref().is_some_and(|l| l.id == id) {
            *live = None;
        }
    }
    result
}

fn rand_u16() -> u16 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish() as u16
}

/// Stop the running generation: "all" at once, or "after_current" when the
/// picture being drawn is done.
#[tauri::command]
pub async fn image_cancel(state: State<'_, AppState>, mode: Option<String>) -> Result<(), String> {
    if let Some(b) = state.backend().await {
        if let Some(engine) = b.as_image() {
            engine.cancel(mode.as_deref().unwrap_or("all"));
        }
    }
    Ok(())
}

/// Take over receiving a generation already running (after a page reload).
#[tauri::command]
pub fn image_attach(on_event: Channel<ImageEvent>) -> Option<LiveImageInfo> {
    let live = LIVE.lock().ok()?;
    let l = live.as_ref()?;
    if let Ok(mut slot) = l.listener.lock() {
        *slot = Some(on_event);
    }
    let mut events: Vec<SdEvent> = l.images.clone();
    events.extend(l.stage.iter().cloned());
    events.extend(l.progress.iter().cloned());
    events.extend(l.preview.iter().cloned());
    Some(LiveImageInfo { id: l.id.clone(), request: l.request.clone(), started_at: l.started_at, events })
}

/// Put a picture on the clipboard as an image (not a path).
#[tauri::command]
pub async fn image_copy(app: tauri::AppHandle, path: String) -> Result<(), String> {
    use tauri_plugin_clipboard_manager::ClipboardExt;
    let img = tauri::async_runtime::spawn_blocking(move || image::open(&path).map(|i| i.to_rgba8()))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| trf!("无法读取图片:{}", "cannot read the picture: {}", e))?;
    let (w, h) = img.dimensions();
    app.clipboard()
        .write_image(&tauri::image::Image::new_owned(img.into_raw(), w, h))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_parameters_read_like_every_other_tool_writes_them() {
        let p = GenerateParams {
            prompt: "a cat".into(),
            negative_prompt: "blurry".into(),
            steps: 20,
            cfg_scale: 6.0,
            sampler: "euler".into(),
            width: 1024,
            height: 768,
            ..Default::default()
        };
        let m = metadata_text(&p, "qwen-image-2.1-Q4_K_M");
        assert_eq!(
            m,
            "a cat\nNegative prompt: blurry\nSteps: 20, CFG scale: 6, Sampler: euler, Size: 1024x768, Model: qwen-image-2.1-Q4_K_M, Generator: Chaty"
        );
    }

    /// End to end on a real model and the real sidecar: probe the folder,
    /// load onto whatever device there is, draw one small picture.
    /// `CHATY_TEST_SD_MODEL=/path/to/model.gguf cargo test … -- --ignored`
    #[test]
    #[ignore]
    fn a_real_image_model_loads_and_draws() {
        let model = PathBuf::from(std::env::var("CHATY_TEST_SD_MODEL").expect("set CHATY_TEST_SD_MODEL"));
        let fracs = Mutex::new(Vec::<f32>::new());
        let (backend, info) = load(&model, &LoadOptions::default(), &[], |f| fracs.lock().unwrap().push(f)).expect("load");
        assert_eq!(info.kind, "image");
        eprintln!("loaded {} as {:?}; progress ticks {}", info.name, info.image.as_ref().map(|i| &i.engine_version), fracs.lock().unwrap().len());
        let engine = backend.as_image().expect("an image engine");
        let out = std::env::temp_dir().join(format!("chaty-sd-e2e-{}", std::process::id()));
        // Small and quick by default; `CHATY_TEST_SD_SIZE` / `CHATY_TEST_SD_STEPS`
        // (0 = the model's own) to see what it draws at real settings.
        let env_num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u32>().ok());
        let side = env_num("CHATY_TEST_SD_SIZE").unwrap_or(256);
        let steps = match env_num("CHATY_TEST_SD_STEPS") {
            Some(0) => info.image.as_ref().map(|i| i.defaults.steps).unwrap_or(20),
            Some(n) => n,
            None => info.image.as_ref().map(|i| i.defaults.steps.min(4)).unwrap_or(4),
        };
        let params = GenerateParams {
            prompt: std::env::var("CHATY_TEST_SD_PROMPT").unwrap_or_else(|_| "a lighthouse at dusk".into()),
            width: side,
            height: side,
            steps,
            cfg_scale: info.image.as_ref().map(|i| i.defaults.cfg_scale).unwrap_or(1.0),
            seed: 1,
            ..Default::default()
        };
        let mut stages = Vec::new();
        let result = engine
            .generate(&params, &out, "e2e", &metadata_text(&params, &info.name), |e| {
                if let SdEvent::Stage { stage, .. } = &e {
                    stages.push(stage.clone());
                }
            })
            .expect("generate");
        assert_eq!(result.images.len(), 1);
        assert!(Path::new(&result.images[0].0).is_file());
        assert!(stages.contains(&"sample".to_string()) && stages.contains(&"decode".to_string()), "{stages:?}");
        backend.unload();
        // `CHATY_TEST_SD_KEEP=1` keeps the picture, to look at.
        if std::env::var_os("CHATY_TEST_SD_KEEP").is_some() {
            eprintln!("kept {} ({} ms)", result.images[0].0, result.elapsed_ms);
            eprintln!("info {}", serde_json::to_string(&info).unwrap_or_default());
        } else {
            std::fs::remove_dir_all(&out).ok();
        }
    }

    #[test]
    fn the_load_command_puts_every_file_where_the_engine_expects_it() {
        let probe = ImageProbe {
            path: "/m/q.gguf".into(),
            family: "qwen-image-2.1".into(),
            family_name: "Qwen-Image 2.1".into(),
            all_in_one: false,
            params_b: None,
            quant: None,
            size_mb: 0,
            components: vec![
                Component { role: Role::Vae, path: "/m/vae.safetensors".into(), size_mb: 1, source: "folder" },
                Component { role: Role::Llm, path: "/m/llm.gguf".into(), size_mb: 1, source: "folder" },
            ],
            missing: vec![],
            suggestions: vec![],
            requires: vec![],
            optional: vec![],
            defaults: family::QWEN_IMAGE_21.defaults.clone(),
            edits: false,
            engine: "sd.cpp",
        };
        // Default: everything on the GPU.
        let c = load_cmd(&probe, &LoadOptions::default(), false);
        assert_eq!(c["diffusion_model"], "/m/q.gguf");
        assert_eq!(c["vae"], "/m/vae.safetensors");
        assert_eq!(c["llm"], "/m/llm.gguf");
        assert_eq!(c["diffusion_fa"], true);
        assert_eq!(c["offload_to_cpu"], false);
        assert_eq!(c["clip_on_cpu"], false);
        assert!(c.get("backend").is_none());
        let cpu = load_cmd(&probe, &LoadOptions::default(), true);
        assert_eq!(cpu["backend"], "cpu");
    }
}
