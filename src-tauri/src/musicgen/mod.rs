//! Text-to-music: audio.cpp's music GGUFs (YuE2, ACE-Step, HeartMuLa, Stable
//! Audio, MiniMax Music 3, MiDashengLM-Gen) run by the `chaty-audio` sidecar.
//!
//! Loading one turns the whole app into a music studio, the way an image
//! model turns it into an image studio. This module is the backend of that:
//! recognising the model and what it needs (`probe`, `family`), loading it
//! into the engine, making pieces with live progress — new ones, or edits of
//! an earlier round — and keeping what they made.

pub mod family;
pub mod probe;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::ipc::Channel;
use tauri::{Manager, State};

use crate::inference::audio::{AudioEngine, EngineOption, Job, MusicAudio, MusicEvent};
use crate::inference::{InferenceBackend, ModelInfo};
use crate::state::AppState;
use crate::store::{Db, MusicRecord};
use family::{EditKind, Family, Layout};
use probe::{Component, MusicProbe};

/// What the UI knows about a loaded music model, beside the usual ModelInfo.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MusicModelInfo {
    pub family: String,
    pub family_name: String,
    /// The audio.cpp revision the engine was built from.
    pub engine_version: String,
    /// The GPU the engine runs on, "" on the CPU.
    pub device: String,
    pub on_cpu: bool,
    pub components: Vec<Component>,
    /// The package files the engine was told to use (session option → file).
    pub files: BTreeMap<String, String>,
    /// How the studio drives this family (composer fields, length, edits,
    /// stages).
    pub spec: Family,
    /// The family's options, as the engine lists them.
    pub request_options: Vec<EngineOption>,
    pub session_options: Vec<EngineOption>,
}

/// How a music model is loaded (Settings → Music model).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MusicLoadOptions {
    /// "gpu" (default) or "cpu".
    pub device: String,
    /// 0 = as many as the machine has (up to 16).
    pub threads: u32,
    /// The engine's session options by family, then by name — weight types,
    /// memory saving, attention kernels. The loaded model's family's are
    /// used; they win over the package files chosen here.
    pub session_options: BTreeMap<String, BTreeMap<String, String>>,
}

/// Load the music model picked at `path` into a fresh sidecar. Blocking.
///
/// GPU by default. When the engine dies while loading onto the GPU (a driver
/// that aborts) it gets one more try on the CPU, with a warning to say so.
pub fn load(path: &Path, opts: &MusicLoadOptions, progress: impl Fn(f32)) -> anyhow::Result<(Arc<dyn InferenceBackend>, ModelInfo)> {
    let probe = probe::probe_music_model(path)
        .ok_or_else(|| anyhow::anyhow!(trf!("这不是文生音乐模型", "not a music generation model")))?;
    let Some(fam) = family::by_id(&probe.family) else {
        anyhow::bail!(trf!(
            "{} 不是 Chaty 的音乐引擎支持的模型家族(支持:YuE2、ACE-Step、HeartMuLa、Stable Audio、MiniMax Music 3、MiDashengLM-Gen)",
            "{} is not a model family Chaty's music engine runs (it runs YuE2, ACE-Step, HeartMuLa, Stable Audio, MiniMax Music 3 and MiDashengLM-Gen)",
            probe.family_name
        ));
    };
    if !probe.missing.is_empty() {
        anyhow::bail!(trf!(
            "{} 还缺少配套文件:{}。请在模型菜单里下载配套文件。",
            "{} is missing files it needs: {}. Download them from the model menu.",
            probe.family_name,
            probe.missing.join(trf!("、", ", ").as_str())
        ));
    }
    let sidecar = crate::inference::audio::find_sidecar().ok_or_else(|| {
        anyhow::anyhow!(trf!("未找到音乐引擎组件 chaty-audio,请重新安装应用", "the music engine (chaty-audio) is missing; please reinstall"))
    })?;

    let want_cpu = opts.device == "cpu";
    let mut warning = None;
    let (engine, loaded, on_cpu) = match AudioEngine::load(&sidecar, load_cmd(&probe, opts, want_cpu), &progress) {
        Ok((e, l)) => (e, l, want_cpu),
        Err(e) if !want_cpu && e.downcast_ref::<crate::inference::sd::Crashed>().is_some() => {
            crate::errlog::append_error("music-model-load", &format!("path: {}\nGPU load crashed, retrying on CPU\n{e:#}", path.display()));
            warning = Some("music-gpu-crash-cpu".to_string());
            let (e, l) = AudioEngine::load(&sidecar, load_cmd(&probe, opts, true), &progress)?;
            (e, l, true)
        }
        Err(e) => return Err(e),
    };

    let name = model_name(path, &probe);
    let device = if on_cpu {
        String::new()
    } else if !loaded.device.is_empty() {
        loaded.device.clone()
    } else {
        crate::gpu::detect_gpu().map(|g| g.name).unwrap_or_default()
    };
    let music = MusicModelInfo {
        family: probe.family.clone(),
        family_name: probe.family_name.clone(),
        engine_version: loaded.version.clone(),
        device: device.clone(),
        on_cpu,
        components: probe.components.clone(),
        files: probe.files.clone(),
        spec: fam.clone(),
        request_options: loaded.request_options.clone(),
        session_options: loaded.session_options.clone(),
    };
    let info = ModelInfo {
        name: name.clone(),
        path: path.to_string_lossy().to_string(),
        backend: "audio.cpp".into(),
        loaded: true,
        arch: Some(probe.family_name.clone()),
        size_mb: Some(probe.size_mb),
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
        kind: "music".into(),
        image: None,
        music: Some(music),
    };
    Ok((Arc::new(engine), info))
}

/// The name a model goes by: its file's, or for a package whose pick is a
/// generically named part (`language_model_q4_0.gguf`), its folder's and the
/// quantization.
fn model_name(path: &Path, probe: &MusicProbe) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("music model").to_string();
    if probe.layout == Some(Layout::MiniMax) {
        let folder = path.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str()).unwrap_or("MiniMax-Music3");
        return match &probe.quant {
            Some(q) => format!("{folder} ({q})"),
            None => folder.to_string(),
        };
    }
    stem
}

/// The sidecar's `load` command for a probed model.
fn load_cmd(p: &MusicProbe, o: &MusicLoadOptions, cpu: bool) -> Value {
    let mut session: BTreeMap<String, String> = p.files.clone();
    for (k, v) in o.session_options.get(&p.family).into_iter().flatten() {
        if !v.trim().is_empty() {
            session.insert(k.clone(), v.trim().to_string());
        }
    }
    json!({
        "cmd": "load",
        "model_path": p.model_path,
        "family": p.family,
        "backend": if cpu { "cpu" } else { "best" },
        "threads": o.threads,
        "session_options": session,
    })
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// An edit of an earlier round of the session.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MusicEdit {
    pub kind: EditKind,
    pub parent_id: String,
    /// Seconds into the parent: where a repainted or inpainted stretch
    /// starts, or how much of it a continuation keeps (0 = all of it).
    #[serde(default)]
    pub start: f32,
    /// Where a repainted or inpainted stretch ends; 0 = the end.
    #[serde(default)]
    pub end: f32,
    /// How far a variation may move from the parent (0..1); 0 = the
    /// recommended amount.
    #[serde(default)]
    pub strength: f32,
}

/// A piece asked for by the studio.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MusicRequest {
    /// The description: genre, instruments, mood, voice.
    pub prompt: String,
    pub lyrics: String,
    /// No singing (the lyrics are not sent).
    pub instrumental: bool,
    /// Length in seconds; 0 = the recommended one (or the model decides).
    pub seconds: f32,
    /// Negative = random.
    pub seed: i64,
    /// The family's request options the studio set, by name — only those
    /// moved off their recommended values.
    pub options: BTreeMap<String, String>,
    /// A score (ABC) the piece follows (YuE2).
    pub score_path: Option<String>,
    pub edit: Option<MusicEdit>,
    /// Folder the piece goes to; empty = app-data/music/<date>.
    pub out_dir: Option<String>,
    /// The session this round belongs to. Absent: a session of its own.
    pub session_id: Option<String>,
}

/// A number the way an engine option reads it.
fn num(v: f32) -> String {
    if v.fract() == 0.0 && v.abs() < 1e9 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// The engine job for a request: the composer's fields put where the family
/// takes them, and an edit turned into the options it needs from its parent.
pub fn build_job(fam: &Family, req: &MusicRequest, parent: Option<&MusicRecord>) -> Result<Job, String> {
    let mut o: BTreeMap<String, String> = fam.fixed.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    for (k, v) in &req.options {
        o.insert(k.clone(), v.clone());
    }
    let prompt = req.prompt.trim();
    if prompt.is_empty() && fam.prompt_option.is_some() && fam.id == family::YUE2.id {
        return Err(trf!("请描述曲风(如:流行、钢琴、温暖女声)", "describe the style (e.g. pop, piano, warm female vocal)"));
    }
    let text = if fam.prompt_text {
        match fam.prompt_template {
            Some(t) if !prompt.contains("<|") => t.replace("{}", prompt),
            _ => prompt.to_string(),
        }
    } else {
        String::new()
    };
    if let Some(opt) = fam.prompt_option {
        o.insert(opt.into(), prompt.to_string());
    }
    if let Some(opt) = fam.lyrics {
        if req.instrumental {
            match fam.instrumental_lyrics {
                Some(v) => {
                    o.insert(opt.into(), v.into());
                }
                None => {
                    return Err(trf!(
                        "{} 需要歌词,不能生成纯音乐",
                        "{} needs lyrics — it cannot make an instrumental",
                        fam.name
                    ))
                }
            }
        } else if req.lyrics.trim().is_empty() {
            if fam.lyrics_required {
                return Err(trf!("{} 需要填写歌词", "{} needs lyrics", fam.name));
            }
            if let Some(v) = fam.instrumental_lyrics {
                o.insert(opt.into(), v.into());
            }
        } else {
            o.insert(opt.into(), req.lyrics.trim().to_string());
        }
    }
    if let Some(len) = fam.length {
        if req.seconds > 0.0 {
            let s = req.seconds.clamp(len.min_s, len.max_s);
            o.insert(len.option.into(), num((s * len.per_second).round()));
        } else if let Some(auto) = len.auto_value {
            o.entry(len.option.into()).or_insert_with(|| auto.to_string());
        }
    }
    let mut audio_path = None;

    if fam.id == family::YUE2.id {
        // The tokens of every piece are kept: a later round may continue it.
        o.insert("export_semantic".into(), "true".into());
        // A score to follow asks for planning that reads it.
        let follow = |o: &mut BTreeMap<String, String>, path: &str| {
            o.insert("abc_file".into(), path.to_string());
            if o.get("cot").is_none_or(|c| c == "off") {
                o.insert("cot".into(), "melody".into());
            }
        };
        if let Some(score) = req.score_path.as_deref().filter(|s| !s.trim().is_empty()) {
            follow(&mut o, score.trim());
        }
        if let Some(edit) = &req.edit {
            let parent = parent.ok_or_else(|| trf!("找不到要修改的那一首", "the piece to edit is gone"))?;
            match edit.kind {
                EditKind::Rearrange => {
                    let score = parent.audio.score_path.as_deref().filter(|p| Path::new(p).is_file()).ok_or_else(|| {
                        trf!(
                            "那一首没有乐谱可循(生成它时关闭了「先写乐谱」)",
                            "that piece has no score to follow (it was made without planning)"
                        )
                    })?;
                    follow(&mut o, score);
                }
                EditKind::Continue => {
                    let tokens_path = parent
                        .audio
                        .tokens_path
                        .as_deref()
                        .filter(|p| Path::new(p).is_file())
                        .ok_or_else(|| trf!("那一首没有可续写的记录", "that piece has nothing to continue from"))?;
                    let tokens: Vec<i64> = std::fs::read_to_string(tokens_path)
                        .ok()
                        .and_then(|t| serde_json::from_str(&t).ok())
                        .ok_or_else(|| trf!("无法读取 {}", "cannot read {}", tokens_path))?;
                    let keep = if edit.start > 0.0 {
                        ((edit.start * 25.0).round() as usize).min(tokens.len())
                    } else {
                        tokens.len()
                    };
                    let prefix = &tokens[..keep];
                    o.insert("semantic_prefix".into(), serde_json::to_string(prefix).unwrap_or_else(|_| "[]".into()));
                    // The kept music was planned from the parent's score; the
                    // rest follows it too. Without one, no planning at all.
                    match parent.audio.score_path.as_deref().filter(|p| Path::new(p).is_file()) {
                        Some(score) if o.get("cot").is_none_or(|c| c != "off") => {
                            o.insert("abc_file".into(), score.to_string());
                        }
                        _ => {
                            o.insert("cot".into(), "off".into());
                            o.remove("abc_file");
                        }
                    }
                    // The budget counts the kept music too: it must leave room
                    // for more (ten seconds at least).
                    let need = keep as i64 + 250;
                    let max = o.get("semantic_max_tokens").and_then(|v| v.parse::<i64>().ok()).unwrap_or(9000);
                    if max < need {
                        o.insert("semantic_max_tokens".into(), need.to_string());
                    }
                }
                other => return Err(format!("{other:?} is not an edit {} makes", fam.name)),
            }
        }
        // The shortest piece may not be longer than the longest.
        if let (Some(max), Some(min)) = (
            o.get("semantic_max_tokens").and_then(|v| v.parse::<i64>().ok()),
            o.get("semantic_min_tokens").and_then(|v| v.parse::<i64>().ok()),
        ) {
            if min > max {
                o.insert("semantic_min_tokens".into(), max.to_string());
            }
        } else if let Some(max) = o.get("semantic_max_tokens").and_then(|v| v.parse::<i64>().ok()) {
            if max < 200 {
                o.insert("semantic_min_tokens".into(), max.to_string());
            }
        }
    } else if let Some(edit) = &req.edit {
        let parent = parent.ok_or_else(|| trf!("找不到要修改的那一首", "the piece to edit is gone"))?;
        if !fam.edits.contains(&edit.kind) {
            return Err(format!("{:?} is not an edit {} makes", edit.kind, fam.name));
        }
        if !Path::new(&parent.audio.path).is_file() {
            return Err(trf!("那一首的音频文件已不在了", "that piece's audio file is gone"));
        }
        audio_path = Some(parent.audio.path.clone());
        let parent_len = parent.audio.seconds as f32;
        let end = if edit.end > 0.0 { edit.end.min(parent_len) } else { parent_len };
        let start = edit.start.clamp(0.0, end);
        if edit.kind.is_range() && end - start < 0.5 {
            return Err(trf!("选择的片段太短", "the stretch chosen is too short"));
        }
        match edit.kind {
            EditKind::Repaint => {
                o.insert("route".into(), "repaint".into());
                o.insert("repainting_start".into(), num(start));
                o.insert("repainting_end".into(), num(end));
                // The source sets the length.
                if let Some(len) = fam.length {
                    o.remove(len.option);
                }
            }
            EditKind::Cover => {
                o.insert("route".into(), "cover".into());
                if let Some(len) = fam.length {
                    o.remove(len.option);
                }
            }
            EditKind::Variation => {
                o.insert("audio_input_kind".into(), "init_audio".into());
                o.insert("init_noise_level".into(), num(if edit.strength > 0.0 { edit.strength.clamp(0.05, 1.0) } else { 0.7 }));
                if req.seconds <= 0.0 {
                    if let Some(len) = fam.length {
                        o.insert(len.option.into(), num(parent_len.round()));
                    }
                }
            }
            EditKind::Inpaint => {
                o.insert("audio_input_kind".into(), "inpaint_audio".into());
                o.insert("inpaint_mask_start_seconds".into(), num(start));
                o.insert("inpaint_mask_end_seconds".into(), num(end));
                if let Some(len) = fam.length {
                    o.insert(len.option.into(), num(parent_len.round()));
                }
            }
            EditKind::Rearrange | EditKind::Continue => unreachable!("checked against the family's edits"),
        }
    }
    Ok(Job { text, audio_path, options: o, seed: req.seed })
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// What the studio hears while a piece is made.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum MusicStudioEvent {
    #[serde(rename_all = "camelCase")]
    Started { id: String, request: MusicRequest, started_at: i64 },
    #[serde(rename_all = "camelCase")]
    Engine { event: MusicEvent },
    #[serde(rename_all = "camelCase")]
    Done { record: Option<MusicRecord>, cancelled: bool },
    #[serde(rename_all = "camelCase")]
    Error { message: String },
}

/// A piece in progress, held by the app so a page that reloads under it can
/// pick it back up.
struct LiveMusic {
    id: String,
    request: MusicRequest,
    started_at: i64,
    listener: Arc<Mutex<Option<Channel<MusicStudioEvent>>>>,
    /// The latest of each kind of report, replayed to a page that attaches:
    /// every stage (the percentage is made from where each began), the last
    /// progress, what was learned.
    stages: Vec<MusicEvent>,
    progress: Option<MusicEvent>,
    info: Vec<MusicEvent>,
}

static LIVE: Mutex<Option<LiveMusic>> = Mutex::new(None);

/// A running piece, as a page that just arrived needs it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveMusicInfo {
    pub id: String,
    pub request: MusicRequest,
    pub started_at: i64,
    pub events: Vec<MusicEvent>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn fire(listener: &Arc<Mutex<Option<Channel<MusicStudioEvent>>>>, ev: MusicStudioEvent) {
    if let Ok(l) = listener.lock() {
        if let Some(ch) = l.as_ref() {
            let _ = ch.send(ev);
        }
    }
}

fn rand_u16() -> u16 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish() as u16
}

/// Where pieces are written when no folder is chosen: app-data/music, one
/// folder per day.
fn default_output_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("music"))
}

#[tauri::command]
pub fn music_output_dir(app: tauri::AppHandle) -> Result<String, String> {
    let dir = default_output_root(&app)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.to_string_lossy().to_string())
}

/// Everything about a music model before loading it: its family, what it
/// needs, what is there, and the downloads for what is not. `None` when the
/// file is not a music model.
#[tauri::command]
pub async fn music_model_probe(path: String) -> Result<Option<MusicProbe>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let p = PathBuf::from(&path);
        let target = if p.is_dir() { probe::music_model_in_dir(&p) } else { Some(p) };
        target.and_then(|t| probe::probe_music_model(&t))
    })
    .await
    .map_err(|e| e.to_string())
}

/// The outline of a WAV's waveform: `bars` peaks, 0..1, the loudest at 1.
pub fn wav_peaks(path: &Path, bars: usize) -> Vec<f32> {
    let Ok(bytes) = std::fs::read(path) else { return Vec::new() };
    let u16le = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Vec::new();
    }
    let (mut channels, mut bits, mut pos) = (0usize, 0usize, 12usize);
    while pos + 8 <= bytes.len() {
        let size = u32le(pos + 4) as usize;
        let body = pos + 8;
        if &bytes[pos..pos + 4] == b"fmt " && body + 16 <= bytes.len() {
            channels = u16le(body + 2) as usize;
            bits = u16le(body + 14) as usize;
        } else if &bytes[pos..pos + 4] == b"data" {
            if channels == 0 || bits != 16 {
                return Vec::new();
            }
            let data = &bytes[body..(body + size).min(bytes.len())];
            let frames = data.len() / (2 * channels);
            if frames == 0 || bars == 0 {
                return Vec::new();
            }
            let mut peaks = vec![0f32; bars];
            for (f, chunk) in data.chunks_exact(2 * channels).enumerate() {
                let bar = f * bars / frames;
                for c in 0..channels {
                    let v = (i16::from_le_bytes([chunk[2 * c], chunk[2 * c + 1]]) as f32).abs() / 32768.0;
                    if v > peaks[bar] {
                        peaks[bar] = v;
                    }
                }
            }
            let top = peaks.iter().cloned().fold(0f32, f32::max);
            if top > 0.0 {
                for p in &mut peaks {
                    *p = (*p / top * 1000.0).round() / 1000.0;
                }
            }
            return peaks;
        }
        pos = body + size + (size & 1);
    }
    Vec::new()
}

/// Make a piece with the loaded music model. Stages and progress stream on
/// `on_event`; the finished round is saved to its session and returned.
#[tauri::command]
pub async fn music_generate(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    request: MusicRequest,
    on_event: Channel<MusicStudioEvent>,
) -> Result<Option<MusicRecord>, String> {
    let backend = state.backend().await;
    let Some(backend) = backend.filter(|b| b.as_music().is_some()) else {
        let msg = trf!("尚未加载文生音乐模型", "no music model is loaded");
        let _ = on_event.send(MusicStudioEvent::Error { message: msg.clone() });
        return Err(msg);
    };
    let (model_name, fam_id) = state
        .model
        .read()
        .await
        .as_ref()
        .map(|m| (m.name.clone(), m.music.as_ref().map(|i| i.family.clone()).unwrap_or_default()))
        .unwrap_or_default();
    let fail = |msg: String| {
        let _ = on_event.send(MusicStudioEvent::Error { message: msg.clone() });
        Err(msg)
    };
    let Some(fam) = family::by_id(&fam_id) else {
        return fail(format!("unknown music family {fam_id}"));
    };
    let parent = match request.edit.as_ref() {
        Some(e) => match crate::store::music_record(&app.state::<Db>(), &e.parent_id) {
            Ok(p) => p,
            Err(e) => return fail(e),
        },
        None => None,
    };
    let job = match build_job(fam, &request, parent.as_ref()) {
        Ok(j) => j,
        Err(e) => return fail(e),
    };

    let started_at = now_ms();
    let stamp = chrono::Local::now();
    let out_dir = match request.out_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(d) => PathBuf::from(d),
        None => default_output_root(&app)?.join(stamp.format("%Y-%m-%d").to_string()),
    };
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        return fail(trf!("无法创建输出文件夹 {}:{}", "cannot create the output folder {}: {}", out_dir.display(), e));
    }
    let id = format!("{}-{:04x}", stamp.format("%Y%m%d%H%M%S%3f"), rand_u16());
    let session_id = match request.session_id.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => s.to_string(),
        None => {
            let title: String = request.prompt.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(40).collect();
            crate::store::music_session_ensure(&app.state::<Db>(), &id, &title)?;
            id.clone()
        }
    };
    let out_path = out_dir.join(format!("{}-{:04x}.wav", stamp.format("%Y%m%d-%H%M%S"), rand_u16()));

    let listener = Arc::new(Mutex::new(Some(on_event.clone())));
    if let Ok(mut live) = LIVE.lock() {
        *live = Some(LiveMusic {
            id: id.clone(),
            request: request.clone(),
            started_at,
            listener: listener.clone(),
            stages: Vec::new(),
            progress: None,
            info: Vec::new(),
        });
    }
    fire(&listener, MusicStudioEvent::Started { id: id.clone(), request: request.clone(), started_at });

    let l2 = listener.clone();
    let path2 = out_path.clone();
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let engine = backend.as_music().expect("checked above");
        let result = engine.generate(&job, &path2, |ev| {
            if let Ok(mut live) = LIVE.lock() {
                if let Some(l) = live.as_mut() {
                    match &ev {
                        MusicEvent::Stage { .. } => {
                            l.stages.push(ev.clone());
                            l.progress = None;
                        }
                        MusicEvent::Progress { .. } => l.progress = Some(ev.clone()),
                        MusicEvent::Info { .. } => l.info.push(ev.clone()),
                        MusicEvent::Audio { .. } => {}
                    }
                }
            }
            fire(&l2, MusicStudioEvent::Engine { event: ev });
        });
        // The outline for the player, read while still off the async runtime.
        let peaks = match &result {
            Ok(o) => o.audio.as_ref().map(|a| wav_peaks(Path::new(&a.path), 240)).unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        result.map(|o| (o, peaks))
    })
    .await
    .map_err(|e| e.to_string())?;

    let result = match outcome {
        Ok((out, peaks)) => {
            let record = out.audio.map(|audio: MusicAudio| MusicRecord {
                id: id.clone(),
                session_id: session_id.clone(),
                parent_id: request.edit.as_ref().map(|e| e.parent_id.clone()),
                prompt: request.prompt.clone(),
                lyrics: if request.instrumental { String::new() } else { request.lyrics.clone() },
                params: serde_json::to_value(&request).unwrap_or(Value::Null),
                audio,
                peaks,
                model: model_name.clone(),
                family: fam.id.to_string(),
                created_at: started_at,
                elapsed_ms: out.elapsed_ms as i64,
            });
            let record = match record {
                Some(r) => match crate::store::music_record_insert(&app.state::<Db>(), &r) {
                    Ok(true) => Some(r),
                    // The session was deleted while this round was made: like
                    // a chat reply to a deleted conversation, it is dropped —
                    // and so are its files, which nothing points to now.
                    Ok(false) => {
                        for p in [Some(&r.audio.path), r.audio.score_path.as_ref(), r.audio.tokens_path.as_ref()].into_iter().flatten() {
                            let _ = std::fs::remove_file(p);
                        }
                        None
                    }
                    Err(e) => {
                        crate::errlog::append_error("music-history-save", &e);
                        Some(r)
                    }
                },
                None => None,
            };
            fire(&listener, MusicStudioEvent::Done { record: record.clone(), cancelled: out.cancelled });
            Ok(record)
        }
        Err(e) => {
            let msg = format!("{e:#}");
            crate::errlog::append_error("music-generate", &format!("model: {model_name}\n{msg}"));
            fire(&listener, MusicStudioEvent::Error { message: msg.clone() });
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

/// Stop the piece being made.
#[tauri::command]
pub async fn music_cancel(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(b) = state.backend().await {
        if let Some(engine) = b.as_music() {
            engine.cancel();
        }
    }
    Ok(())
}

/// Take over receiving a piece already being made (after a page reload).
#[tauri::command]
pub fn music_attach(on_event: Channel<MusicStudioEvent>) -> Option<LiveMusicInfo> {
    let live = LIVE.lock().ok()?;
    let l = live.as_ref()?;
    if let Ok(mut slot) = l.listener.lock() {
        *slot = Some(on_event);
    }
    let mut events = l.stages.clone();
    events.extend(l.info.iter().cloned());
    events.extend(l.progress.iter().cloned());
    Some(LiveMusicInfo { id: l.id.clone(), request: l.request.clone(), started_at: l.started_at, events })
}

// ---------------------------------------------------------------------------
// The player's files
// ---------------------------------------------------------------------------

/// The URI scheme the studio's player reads pieces through
/// (`convertFileSrc(path, MEDIA_SCHEME)`): only files of the music history,
/// with byte ranges, which is what lets the player seek.
pub const MEDIA_SCHEME: &str = "chatymedia";

/// Answer one request of the player. `uri_path` is the request's path —
/// `/` and the file's path, percent-encoded.
pub fn serve_media(app: &tauri::AppHandle, uri_path: &str, range: Option<&str>) -> tauri::http::Response<Vec<u8>> {
    use tauri::http::{header, Response, StatusCode};
    let respond = |status: StatusCode| Response::builder().status(status).body(Vec::new()).unwrap();
    let raw = uri_path.trim_start_matches('/');
    let path = percent_encoding::percent_decode_str(raw).decode_utf8_lossy().to_string();
    if path.is_empty() || !crate::store::music_file_known(&app.state::<Db>(), &path) {
        return respond(StatusCode::NOT_FOUND);
    }
    let Ok(meta) = std::fs::metadata(&path) else { return respond(StatusCode::NOT_FOUND) };
    let len = meta.len();
    let mime = if path.to_lowercase().ends_with(".wav") { "audio/wav" } else { "text/plain; charset=utf-8" };
    // `bytes=a-b`, `bytes=a-`, `bytes=-n`.
    let parsed = range.and_then(|r| r.trim().strip_prefix("bytes=")).and_then(|r| {
        let (a, b) = r.split_once('-')?;
        let (a, b) = (a.trim(), b.trim().split(',').next().unwrap_or("").trim());
        match (a.parse::<u64>().ok(), b.parse::<u64>().ok()) {
            (Some(s), Some(e)) => Some((s, e.min(len.saturating_sub(1)))),
            (Some(s), None) => Some((s, len.saturating_sub(1))),
            (None, Some(n)) => Some((len.saturating_sub(n), len.saturating_sub(1))),
            (None, None) => None,
        }
    });
    let read = |start: u64, count: u64| -> std::io::Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(&path)?;
        f.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; count as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    };
    match parsed {
        Some((start, end)) if start <= end && end < len => {
            // Large pieces are served a few megabytes at a time; the player
            // asks for the rest as it plays.
            let end = end.min(start + 8 * 1024 * 1024 - 1);
            match read(start, end - start + 1) {
                Ok(body) => Response::builder()
                    .status(StatusCode::PARTIAL_CONTENT)
                    .header(header::CONTENT_TYPE, mime)
                    .header(header::ACCEPT_RANGES, "bytes")
                    .header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{len}"))
                    .header(header::CONTENT_LENGTH, body.len().to_string())
                    .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                    .body(body)
                    .unwrap(),
                Err(_) => respond(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
        Some(_) => Response::builder()
            .status(StatusCode::RANGE_NOT_SATISFIABLE)
            .header(header::CONTENT_RANGE, format!("bytes */{len}"))
            .body(Vec::new())
            .unwrap(),
        None => match std::fs::read(&path) {
            Ok(body) => Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime)
                .header(header::ACCEPT_RANGES, "bytes")
                .header(header::CONTENT_LENGTH, body.len().to_string())
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(body)
                .unwrap(),
            Err(_) => respond(StatusCode::INTERNAL_SERVER_ERROR),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(audio: MusicAudio) -> MusicRecord {
        MusicRecord {
            id: "p".into(),
            session_id: "s".into(),
            parent_id: None,
            prompt: String::new(),
            lyrics: String::new(),
            params: Value::Null,
            audio,
            peaks: Vec::new(),
            model: String::new(),
            family: String::new(),
            created_at: 0,
            elapsed_ms: 0,
        }
    }

    fn req() -> MusicRequest {
        MusicRequest { prompt: " indie pop, acoustic guitar ".into(), lyrics: "[Verse]\nHello".into(), seed: -1, ..Default::default() }
    }

    /// End to end on a real model and the real sidecar: probe the package,
    /// load it onto whatever device there is, make a short piece.
    /// `CHATY_TEST_MUSIC_MODEL=/path/to/yue2-3b-q4_0.gguf cargo test … -- --ignored`
    #[test]
    #[ignore]
    fn a_real_music_model_loads_and_plays() {
        let model = PathBuf::from(std::env::var("CHATY_TEST_MUSIC_MODEL").expect("set CHATY_TEST_MUSIC_MODEL"));
        let ticks = Mutex::new(0usize);
        let (backend, info) = load(&model, &MusicLoadOptions::default(), |_| *ticks.lock().unwrap() += 1).expect("load");
        assert_eq!(info.kind, "music");
        let music = info.music.clone().expect("what the engine loaded");
        eprintln!("loaded {} as {} on {:?}; progress ticks {}", info.name, music.family, music.device, ticks.lock().unwrap());
        let fam = family::by_id(&music.family).expect("a family the studio drives");
        let engine = backend.as_music().expect("a music engine");
        let out = std::env::temp_dir().join(format!("chaty-audio-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&out).unwrap();
        // Short by default (each family's shortest, 10 s for YuE2);
        // `CHATY_TEST_MUSIC_SECONDS` for a real length.
        let seconds = std::env::var("CHATY_TEST_MUSIC_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
        let request = MusicRequest {
            prompt: "indie pop, bright acoustic guitar, warm female vocal".into(),
            lyrics: if fam.lyrics.is_some() { "[verse]\nMorning light is on the window\n".into() } else { String::new() },
            seconds,
            seed: 7,
            ..Default::default()
        };
        let job = build_job(fam, &request, None).expect("job");
        let mut stages = Vec::new();
        let outcome = engine
            .generate(&job, &out.join("e2e.wav"), |e| {
                if let MusicEvent::Stage { stage, .. } = &e {
                    stages.push(stage.clone());
                }
            })
            .expect("generate");
        let audio = outcome.audio.expect("a piece");
        assert!(Path::new(&audio.path).is_file());
        assert!(audio.seconds > 1.0, "{audio:?}");
        assert_eq!(stages.first().map(String::as_str), Some("prepare"), "{stages:?}");
        assert_eq!(stages.last().map(String::as_str), Some("save"), "{stages:?}");
        assert!(wav_peaks(Path::new(&audio.path), 32).iter().any(|&p| p > 0.01), "silence");
        backend.unload();
        // `CHATY_TEST_MUSIC_KEEP=1` keeps the piece, to listen to.
        if std::env::var_os("CHATY_TEST_MUSIC_KEEP").is_some() {
            eprintln!("kept {} ({} ms, stages {stages:?})", audio.path, outcome.elapsed_ms);
        } else {
            std::fs::remove_dir_all(&out).ok();
        }
    }

    #[test]
    fn yue2_takes_style_and_lyrics_as_options_and_keeps_its_tokens() {
        let j = build_job(&family::YUE2, &req(), None).unwrap();
        assert_eq!(j.text, "");
        assert_eq!(j.options["style"], "indie pop, acoustic guitar");
        assert_eq!(j.options["lyrics"], "[Verse]\nHello");
        assert_eq!(j.options["export_semantic"], "true");
        assert!(!j.options.contains_key("semantic_max_tokens"), "no length set = the model's own limit");
        // A length is a token budget, 25 a second — and the minimum follows it.
        let j = build_job(&family::YUE2, &MusicRequest { seconds: 6.0, ..req() }, None).unwrap();
        assert_eq!(j.options["semantic_max_tokens"], "250", "clamped to the family's shortest");
        assert!(!j.options.contains_key("semantic_min_tokens"), "the model's own minimum fits");
        let j = build_job(&family::YUE2, &MusicRequest { seconds: 60.0, ..req() }, None).unwrap();
        assert_eq!(j.options["semantic_max_tokens"], "1500");
        // Instrumental: empty lyrics.
        let j = build_job(&family::YUE2, &MusicRequest { instrumental: true, ..req() }, None).unwrap();
        assert_eq!(j.options["lyrics"], "");
        // No style: YuE2 needs one.
        assert!(build_job(&family::YUE2, &MusicRequest { prompt: " ".into(), ..req() }, None).is_err());
    }

    #[test]
    fn the_studio_options_win_over_nothing_but_the_composer() {
        let mut r = req();
        r.options.insert("num_inference_steps".into(), "16".into());
        r.options.insert("style".into(), "overridden".into());
        let j = build_job(&family::YUE2, &r, None).unwrap();
        assert_eq!(j.options["num_inference_steps"], "16");
        assert_eq!(j.options["style"], "indie pop, acoustic guitar", "the composer's field wins");
    }

    #[test]
    fn other_families_put_the_description_where_they_read_it() {
        let j = build_job(&family::ACE_STEP, &req(), None).unwrap();
        assert_eq!(j.text, "indie pop, acoustic guitar");
        assert_eq!(j.options["route"], "text2music");
        assert_eq!(j.options["duration_seconds"], "-1", "ACE-Step decides the length by itself");
        let j = build_job(&family::ACE_STEP, &MusicRequest { instrumental: true, ..req() }, None).unwrap();
        assert_eq!(j.options["lyrics"], "[Instrumental]");

        let j = build_job(&family::HEARTMULA, &req(), None).unwrap();
        assert_eq!((j.text.as_str(), j.options["tags"].as_str()), ("indie pop, acoustic guitar", "indie pop, acoustic guitar"));

        let j = build_job(&family::STABLE_AUDIO, &MusicRequest { seconds: 30.0, ..req() }, None).unwrap();
        assert!(!j.options.contains_key("lyrics"), "Stable Audio sings nothing");
        assert_eq!(j.options["duration_seconds"], "30");

        let j = build_job(&family::MIDASHENGLM_GEN, &req(), None).unwrap();
        assert_eq!(j.text, "<|caption|> indie pop, acoustic guitar <|music|> indie pop, acoustic guitar");
        let tagged = MusicRequest { prompt: "<|music|> jazz".into(), ..req() };
        assert_eq!(build_job(&family::MIDASHENGLM_GEN, &tagged, None).unwrap().text, "<|music|> jazz");

        // MiniMax needs lyrics, and cannot make an instrumental.
        assert!(build_job(&family::MINIMAX_MUSIC3, &MusicRequest { lyrics: String::new(), ..req() }, None).is_err());
        assert!(build_job(&family::MINIMAX_MUSIC3, &MusicRequest { instrumental: true, ..req() }, None).is_err());
        assert!(build_job(&family::MINIMAX_MUSIC3, &req(), None).is_ok());
    }

    #[test]
    fn yue2_edits_follow_the_parent_score_or_continue_its_tokens() {
        let dir = std::env::temp_dir().join(format!("chaty-music-edit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let score = dir.join("a.abc");
        std::fs::write(&score, "X:1").unwrap();
        let tokens = dir.join("a.tokens.json");
        std::fs::write(&tokens, serde_json::to_string(&(0..500).collect::<Vec<i64>>()).unwrap()).unwrap();
        let parent = record(MusicAudio {
            path: dir.join("a.wav").to_string_lossy().into(),
            seconds: 20.0,
            score_path: Some(score.to_string_lossy().into()),
            tokens_path: Some(tokens.to_string_lossy().into()),
            ..Default::default()
        });
        let edit = |kind, start| MusicRequest {
            edit: Some(MusicEdit { kind, parent_id: "p".into(), start, end: 0.0, strength: 0.0 }),
            ..req()
        };
        let j = build_job(&family::YUE2, &edit(EditKind::Rearrange, 0.0), Some(&parent)).unwrap();
        assert_eq!(j.options["abc_file"], score.to_string_lossy());
        assert_eq!(j.options["cot"], "melody", "following a score is melody planning unless asked for full");

        let j = build_job(&family::YUE2, &edit(EditKind::Continue, 8.0), Some(&parent)).unwrap();
        let prefix: Vec<i64> = serde_json::from_str(&j.options["semantic_prefix"]).unwrap();
        assert_eq!(prefix.len(), 200, "eight seconds kept");
        assert_eq!(j.options["abc_file"], score.to_string_lossy(), "the rest follows the same score");
        // A budget shorter than what is kept grows to leave room for more.
        let mut short = edit(EditKind::Continue, 0.0);
        short.seconds = 10.0;
        let j = build_job(&family::YUE2, &short, Some(&parent)).unwrap();
        assert_eq!(j.options["semantic_max_tokens"], "750");

        // A parent made without planning continues without it.
        let mut plain = parent.clone();
        plain.audio.score_path = None;
        let j = build_job(&family::YUE2, &edit(EditKind::Continue, 0.0), Some(&plain)).unwrap();
        assert_eq!(j.options["cot"], "off");
        assert!(!j.options.contains_key("abc_file"));
        assert!(build_job(&family::YUE2, &edit(EditKind::Rearrange, 0.0), Some(&plain)).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn audio_edits_send_the_parent_and_the_stretch() {
        let dir = std::env::temp_dir().join(format!("chaty-music-edit2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("a.wav");
        std::fs::write(&wav, b"RIFF").unwrap();
        let parent = record(MusicAudio { path: wav.to_string_lossy().into(), seconds: 60.0, ..Default::default() });
        let edit = |kind, start, end| MusicRequest {
            edit: Some(MusicEdit { kind, parent_id: "p".into(), start, end, strength: 0.0 }),
            ..req()
        };
        let j = build_job(&family::ACE_STEP, &edit(EditKind::Repaint, 20.0, 35.5), Some(&parent)).unwrap();
        assert_eq!(j.audio_path.as_deref(), Some(wav.to_string_lossy().as_ref()));
        assert_eq!(j.options["route"], "repaint");
        assert_eq!((j.options["repainting_start"].as_str(), j.options["repainting_end"].as_str()), ("20", "35.5"));
        assert!(!j.options.contains_key("duration_seconds"), "the source sets the length");
        assert_eq!(build_job(&family::ACE_STEP, &edit(EditKind::Cover, 0.0, 0.0), Some(&parent)).unwrap().options["route"], "cover");

        let j = build_job(&family::STABLE_AUDIO, &edit(EditKind::Inpaint, 10.0, 0.0), Some(&parent)).unwrap();
        assert_eq!(j.options["audio_input_kind"], "inpaint_audio");
        assert_eq!(j.options["inpaint_mask_end_seconds"], "60", "0 = to the end");
        assert_eq!(j.options["duration_seconds"], "60");
        let j = build_job(&family::STABLE_AUDIO, &edit(EditKind::Variation, 0.0, 0.0), Some(&parent)).unwrap();
        assert_eq!((j.options["audio_input_kind"].as_str(), j.options["init_noise_level"].as_str()), ("init_audio", "0.7"));

        // An edit the family does not make, a stretch too short.
        assert!(build_job(&family::ACE_STEP, &edit(EditKind::Inpaint, 1.0, 5.0), Some(&parent)).is_err());
        assert!(build_job(&family::ACE_STEP, &edit(EditKind::Repaint, 5.0, 5.2), Some(&parent)).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_load_command_names_the_package_files_and_the_familys_own_options() {
        let probe = MusicProbe {
            path: "/m/yue2-3b-q4_0.gguf".into(),
            family: "yue2".into(),
            family_name: "YuE2".into(),
            supported: true,
            layout: Some(Layout::Yue2),
            quant: None,
            params_b: None,
            size_mb: 0,
            components: Vec::new(),
            missing: Vec::new(),
            suggestions: Vec::new(),
            model_path: "/m".into(),
            files: [("yue2.model_gguf".to_string(), "yue2-3b-q4_0.gguf".to_string())].into(),
        };
        let mut o = MusicLoadOptions::default();
        o.session_options.insert("yue2".into(), [("yue2.attention".to_string(), "eager".to_string()), ("yue2.vae_weight_type".to_string(), " ".to_string())].into());
        o.session_options.insert("ace_step".into(), [("ace_step.mem_saver".to_string(), "true".to_string())].into());
        let c = load_cmd(&probe, &o, false);
        assert_eq!(c["model_path"], "/m");
        assert_eq!(c["backend"], "best");
        assert_eq!(c["session_options"]["yue2.model_gguf"], "yue2-3b-q4_0.gguf");
        assert_eq!(c["session_options"]["yue2.attention"], "eager");
        assert!(c["session_options"].get("yue2.vae_weight_type").is_none(), "a blank value is not sent");
        assert!(c["session_options"].get("ace_step.mem_saver").is_none(), "another family's options stay out");
        assert_eq!(load_cmd(&probe, &o, true)["backend"], "cpu");
    }

    #[test]
    fn peaks_outline_a_wav() {
        let dir = std::env::temp_dir().join(format!("chaty-music-peaks-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wav");
        // 4 stereo frames, loud in the second half.
        let samples: [i16; 8] = [0, 0, 100, -100, 16000, 0, -32000, 5];
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&48000u32.to_le_bytes());
        b.extend_from_slice(&(48000u32 * 4).to_le_bytes());
        b.extend_from_slice(&4u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&data);
        std::fs::write(&p, b).unwrap();
        let peaks = wav_peaks(&p, 2);
        assert_eq!(peaks.len(), 2);
        assert!(peaks[0] < 0.01 && (peaks[1] - 1.0).abs() < 1e-6, "{peaks:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
