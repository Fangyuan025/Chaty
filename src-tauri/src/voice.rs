//! Voice engines: Whisper STT + Kokoro TTS via sherpa-onnx (ONNX Runtime, CPU).
//!
//! Runs entirely on CPU (`provider = "cpu"`) so it never touches the LLM's GPU
//! memory. Uses ONNX Runtime, fully isolated from llama.cpp's ggml. Models
//! auto-download to the app data dir on first use — see [`VoiceModel`].

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use sherpa_rs::tts::{CommonTtsConfig, KokoroTts, KokoroTtsConfig, VitsTts, VitsTtsConfig};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};
use sherpa_rs::OnnxConfig;

/// A pinned Hugging Face snapshot: its files, each with its exact size. A
/// pinned revision cannot change, so any other size is a bad copy.
struct HfSnapshot {
    repo: &'static str,
    rev: &'static str,
    files: &'static [(&'static str, u64)],
}

/// Where a voice model comes from. The Hugging Face snapshot is tried first,
/// through the user's HF endpoint, so a mirror such as hf-mirror.com reaches
/// it where huggingface.co and GitHub do not — issue #14: a user in mainland
/// China had set the mirror, yet voice downloads still went to huggingface.co
/// itself (or GitHub) and timed out. The sherpa-onnx GitHub release archive
/// is the fallback, and Kokoro's only source: no Hugging Face repo carries
/// that exact build.
struct VoiceModel {
    /// Folder under voice-models — also the archive's top-level folder.
    dir: &'static str,
    /// "stt" or "tts": which kind of model a download is, for the UI.
    what: &'static str,
    ready: fn(&Path) -> bool,
    hf: Option<HfSnapshot>,
    archive: &'static str,
}

const WHISPER_EN: VoiceModel = VoiceModel {
    dir: "sherpa-onnx-whisper-base.en",
    what: "stt",
    ready: whisper_model_ready,
    hf: Some(HfSnapshot {
        repo: "csukuangfj/sherpa-onnx-whisper-base.en",
        rev: "59eea950fc76df2453efb57e6c0fd334548e8ffe",
        files: &[
            ("base.en-encoder.int8.onnx", 29_120_534),
            ("base.en-decoder.int8.onnx", 130_669_978),
            ("base.en-tokens.txt", 835_554),
        ],
    }),
    archive: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2",
};

const WHISPER_MULTILINGUAL: VoiceModel = VoiceModel {
    dir: "sherpa-onnx-whisper-base",
    what: "stt",
    ready: whisper_model_ready,
    hf: Some(HfSnapshot {
        repo: "csukuangfj/sherpa-onnx-whisper-base",
        rev: "bb53ee204431c90d314c1cc08d28d23e5b7927cc",
        files: &[
            ("base-encoder.int8.onnx", 29_120_534),
            ("base-decoder.int8.onnx", 130_672_026),
            ("base-tokens.txt", 816_730),
        ],
    }),
    archive: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2",
};

const KOKORO: VoiceModel = VoiceModel {
    dir: "kokoro-en-v0_19",
    what: "tts",
    ready: kokoro_model_ready,
    hf: None,
    archive: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
};

const CHINESE_TTS: VoiceModel = VoiceModel {
    dir: "sherpa-onnx-vits-zh-ll",
    what: "tts",
    ready: chinese_tts_model_ready,
    hf: Some(HfSnapshot {
        repo: "csukuangfj/sherpa-onnx-vits-zh-ll",
        rev: "7ddf37bcacf05ed56afee360d96835be633a5265",
        files: &[
            ("model.onnx", 121_100_803),
            ("lexicon.txt", 376_868),
            ("tokens.txt", 331),
            ("dict/jieba.dict.utf8", 5_071_204),
            ("dict/hmm_model.utf8", 519_739),
            ("dict/user.dict.utf8", 49),
            ("dict/idf.utf8", 5_998_717),
            ("dict/stop_words.utf8", 8_974),
            ("dict/pos_dict/char_state_tab.utf8", 327_139),
            ("dict/pos_dict/prob_emit.utf8", 1_687_686),
            ("dict/pos_dict/prob_start.utf8", 4_347),
            ("dict/pos_dict/prob_trans.utf8", 124_159),
        ],
    }),
    archive: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2",
};

const CHINESE_TTS_SPEAKER_COUNT: i32 = 5;

/// Hugging Face and GitHub want to know who is calling (see http.rs).
const DOWNLOAD_UA: &str = "Chaty model downloader";

/// A voice model download in progress: which kind of model, bytes so far and
/// the whole (0 when the source did not say). `done` closes it, however it
/// ended. Issue #14: the first press of the mic fetched ~160 MB behind a
/// spinner with nothing to say it was moving, which on a slow line read as
/// hung.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceDownload {
    pub model: &'static str,
    /// Which model: its folder name (`kokoro-en-v0_19`, `sherpa-onnx-vits-zh-ll`, …),
    /// so the bar can say which voice is coming — issue #20 read a first
    /// download of the English voice as the Chinese one downloading again.
    pub voice: &'static str,
    pub downloaded: u64,
    pub total: u64,
    pub done: bool,
}

/// Where a download's progress goes — the command hands it to the UI.
pub type Progress<'a> = &'a (dyn Fn(VoiceDownload) + Send + Sync);

/// One model's download progress, reported at most every quarter second
/// except where it matters: a new source, a finished file, the end.
struct Meter<'a> {
    report: Progress<'a>,
    model: &'static str,
    voice: &'static str,
    downloaded: u64,
    total: u64,
    last: Option<std::time::Instant>,
}

impl<'a> Meter<'a> {
    fn new(report: Progress<'a>, model: &'static str, voice: &'static str) -> Self {
        Meter { report, model, voice, downloaded: 0, total: 0, last: None }
    }

    /// A new source: its whole size, and how much of it is already here.
    fn start(&mut self, total: u64, have: u64) {
        self.total = total;
        self.downloaded = have;
        self.send();
    }

    fn add(&mut self, n: u64) {
        self.downloaded += n;
        if self.last.is_none_or(|t| t.elapsed() >= std::time::Duration::from_millis(250)) {
            self.send();
        }
    }

    fn send(&mut self) {
        self.last = Some(std::time::Instant::now());
        (self.report)(VoiceDownload {
            model: self.model,
            voice: self.voice,
            downloaded: self.downloaded,
            total: self.total,
            done: false,
        });
    }

    fn finish(&self) {
        (self.report)(VoiceDownload {
            model: self.model,
            voice: self.voice,
            downloaded: self.downloaded,
            total: self.total,
            done: true,
        });
    }
}

static EN_STT: OnceLock<Mutex<WhisperRecognizer>> = OnceLock::new();
static MULTILINGUAL_STT: OnceLock<Mutex<WhisperRecognizer>> = OnceLock::new();
static EN_TTS: OnceLock<Mutex<KokoroTts>> = OnceLock::new();
static ZH_TTS: OnceLock<Mutex<VitsTts>> = OnceLock::new();

/// Trigger the macOS app-level microphone consent prompt (TCC) and wait for
/// the user's answer. WKWebView's permission delegate auto-grants the webview
/// layer, but the system dialog only appears once something in the process
/// requests capture access — which nothing does unless we ask here. Returns
/// whether access is (now) authorized; always true on other platforms.
#[tauri::command(async)]
pub fn request_mic_permission() -> bool {
    #[cfg(target_os = "macos")]
    unsafe {
        use block2::RcBlock;
        use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};

        let Some(media) = AVMediaTypeAudio else {
            return false;
        };
        match AVCaptureDevice::authorizationStatusForMediaType(media) {
            AVAuthorizationStatus::Authorized => return true,
            AVAuthorizationStatus::Denied | AVAuthorizationStatus::Restricted => return false,
            _ => {} // NotDetermined → ask
        }
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        let block = RcBlock::new(move |granted: objc2::runtime::Bool| {
            let _ = tx.send(granted.as_bool());
        });
        AVCaptureDevice::requestAccessForMediaType_completionHandler(media, &block);
        // Wait for the dialog. Fine only because the command is declared
        // `async`: a plain `#[tauri::command] fn` runs ON the main thread, and
        // this wait froze the whole window for as long as the system dialog
        // stayed unanswered.
        rx.recv_timeout(std::time::Duration::from_secs(300))
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "macos"))]
    true
}

/// Threads for the ONNX voice engines — use most cores for snappier STT/TTS,
/// but leave one for the rest of the app and cap to avoid oversubscribing the
/// CPU while the LLM is also generating.
fn voice_threads() -> i32 {
    // On Apple Silicon this is the performance-core count (minus one for the
    // UI, handled in the helper); elsewhere the logical CPU count. Clamp to a
    // sane range so we don't oversubscribe while the LLM is also generating.
    crate::gpu::cpu_worker_threads().clamp(2, 8) as i32
}

fn find_in(dir: &Path, suffix: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(suffix))
                .unwrap_or(false)
        })
}

fn file_at_least(path: &Path, bytes: u64) -> bool {
    path.metadata().map(|m| m.len() >= bytes).unwrap_or(false)
}

fn whisper_model_ready(dir: &Path) -> bool {
    dir.is_dir()
        && whisper_weights_ready(dir)
        && find_in(dir, "tokens.txt").is_some_and(|p| file_at_least(&p, 100 * 1024))
}

fn whisper_weights_ready(dir: &Path) -> bool {
    dir.is_dir()
        && (find_in(dir, "encoder.int8.onnx").is_some_and(|p| file_at_least(&p, 20 * 1024 * 1024))
            || find_in(dir, "encoder.onnx").is_some_and(|p| file_at_least(&p, 80 * 1024 * 1024)))
        && (find_in(dir, "decoder.int8.onnx").is_some_and(|p| file_at_least(&p, 100 * 1024 * 1024))
            || find_in(dir, "decoder.onnx").is_some_and(|p| file_at_least(&p, 150 * 1024 * 1024)))
}

fn kokoro_model_ready(dir: &Path) -> bool {
    dir.is_dir()
        && dir.join("model.onnx").is_file()
        && dir.join("voices.bin").is_file()
        && dir.join("tokens.txt").is_file()
        && dir.join("espeak-ng-data").is_dir()
}

fn chinese_tts_model_ready(dir: &Path) -> bool {
    dir.is_dir()
        && file_at_least(&dir.join("model.onnx"), 100 * 1024 * 1024)
        && dir.join("lexicon.txt").is_file()
        && dir.join("tokens.txt").is_file()
        && dir.join("dict/jieba.dict.utf8").is_file()
        && dir.join("dict/hmm_model.utf8").is_file()
        && dir.join("dict/user.dict.utf8").is_file()
        && dir.join("dict/idf.utf8").is_file()
        && dir.join("dict/stop_words.utf8").is_file()
}

fn host_of(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    rest.split('/').next().unwrap_or(rest)
}

/// One short line for a source that failed: which host, and what went wrong
/// in words a person can act on — not the nested request error, which named
/// only huggingface.co and "operation timed out" (issue #14).
fn failure_reason(url: &str, e: &anyhow::Error) -> String {
    let Some(re) = e.chain().find_map(|c| c.downcast_ref::<reqwest::Error>()) else {
        return format!("{}: {e}", host_of(url));
    };
    let host = re.url().and_then(|u| u.host_str()).unwrap_or_else(|| host_of(url));
    let what = if re.is_timeout() {
        "timed out".to_string()
    } else if re.is_connect() {
        "could not connect".to_string()
    } else if let Some(s) = re.status() {
        format!("HTTP {}", s.as_u16())
    } else {
        re.to_string()
    };
    format!("{host}: {what}")
}

/// Stream one file into place, checked against its exact size. Written to a
/// `.part` beside it and renamed, so a file that exists is a whole one; a
/// stalled connection ends in a minute (http.rs `download_client`) rather
/// than holding the request for its full size.
async fn fetch_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    size: u64,
    meter: &mut Meter<'_>,
) -> Result<()> {
    use std::io::Write;
    if file_complete(dest, size) {
        return Ok(());
    }
    eprintln!("voice: downloading {url}");
    let mut resp = client.get(url).send().await?.error_for_status()?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).context("create voice model directory")?;
    }
    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let part = dest.with_file_name(format!("{name}.part"));
    let mut file = std::fs::File::create(&part).context("write voice model file")?;
    let mut got = 0u64;
    while let Some(chunk) = resp.chunk().await? {
        got += chunk.len() as u64;
        if got > size {
            break;
        }
        file.write_all(&chunk).context("write voice model file")?;
        meter.add(chunk.len() as u64);
    }
    drop(file);
    if got != size {
        let _ = std::fs::remove_file(&part);
        bail!("{name} came back as {got} bytes, expected {size}");
    }
    std::fs::rename(&part, dest).context("install voice model file")?;
    meter.send();
    Ok(())
}

fn file_complete(path: &Path, size: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.len() == size)
}

/// Fetch every file of a snapshot that is missing or the wrong size. Files
/// already complete stay, so an interrupted download resumes where it stopped
/// — and its progress with it.
async fn fetch_snapshot(dir: &Path, hf: &HfSnapshot, base: &str, meter: &mut Meter<'_>) -> Result<()> {
    let client = crate::http::download_client(DOWNLOAD_UA).map_err(|e| anyhow!(e))?;
    let dest_of = |path: &str| path.split('/').fold(dir.to_path_buf(), |p, s| p.join(s));
    let total = hf.files.iter().map(|&(_, size)| size).sum();
    let have = hf
        .files
        .iter()
        .filter(|&&(path, size)| file_complete(&dest_of(path), size))
        .map(|&(_, size)| size)
        .sum();
    meter.start(total, have);
    for &(path, size) in hf.files {
        let url = format!("{base}/{}/resolve/{}/{path}", hf.repo, hf.rev);
        fetch_file(&client, &url, &dest_of(path), size, meter).await?;
    }
    Ok(())
}

/// A ready model's folder, downloading it first if need be: the Hugging Face
/// snapshot through `base` (the user's HF endpoint), then the release archive.
/// A download reports its progress to `report`, and always its end.
/// Wakes a voice model download to stop it (the × on its progress bar). A
/// stalled connection never returns another chunk, so the download is raced
/// against this rather than checking a flag between chunks.
static CANCEL: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// What a cancelled download fails with; the UI says nothing about it.
pub const DOWNLOAD_CANCELLED: &str = "VOICE_DOWNLOAD_CANCELLED";

/// Stop the voice model download in progress, if any. What already arrived
/// stays: the next attempt picks up the files that are complete.
pub fn cancel_download() {
    CANCEL.notify_waiters();
}

async fn ensure_model(models_dir: &Path, m: &VoiceModel, base: &str, report: Progress<'_>) -> Result<PathBuf> {
    let dir = models_dir.join(m.dir);
    if (m.ready)(&dir) {
        return Ok(dir);
    }
    let mut meter = Meter::new(report, m.what, m.dir);
    let got = tokio::select! {
        got = download_model(models_dir, m, base, &mut meter) => got,
        _ = CANCEL.notified() => Err(anyhow!(DOWNLOAD_CANCELLED)),
    };
    meter.finish();
    got
}

async fn download_model(models_dir: &Path, m: &VoiceModel, base: &str, meter: &mut Meter<'_>) -> Result<PathBuf> {
    let dir = models_dir.join(m.dir);
    let mut reasons = Vec::new();
    if let Some(hf) = &m.hf {
        match fetch_snapshot(&dir, hf, base, meter).await {
            Ok(()) if (m.ready)(&dir) => return Ok(dir),
            Ok(()) => reasons.push(format!("{}: the files arrived but the model is incomplete", host_of(base))),
            Err(e) => reasons.push(failure_reason(base, &e)),
        }
    }
    match install_archive(models_dir, m.archive, m.dir, m.ready, meter).await {
        Ok(dir) => Ok(dir),
        Err(e) => {
            reasons.push(failure_reason(m.archive, &e));
            Err(download_failed(models_dir, m, base, reasons))
        }
    }
}

/// The error for a model no source could deliver, as a payload the UI words
/// in the reader's language (lib/voiceError.ts): the folder the model belongs
/// in, what to fetch by hand and from where, and why each source failed.
fn download_failed(models_dir: &Path, m: &VoiceModel, base: &str, reasons: Vec<String>) -> anyhow::Error {
    let payload = serde_json::json!({
        "dir": models_dir.join(m.dir).to_string_lossy(),
        "modelsDir": models_dir.to_string_lossy(),
        "hfUrl": m.hf.as_ref().map(|hf| format!("{base}/{}/tree/{}", hf.repo, hf.rev)),
        "files": m.hf.as_ref().map(|hf| hf.files.iter().map(|(p, _)| *p).collect::<Vec<_>>()).unwrap_or_default(),
        "archive": m.archive,
        "reasons": reasons,
    });
    anyhow!("VOICE_MODEL_DOWNLOAD {payload}")
}

async fn install_archive(
    models_dir: &Path,
    url: &str,
    dir_name: &str,
    ready: fn(&Path) -> bool,
    meter: &mut Meter<'_>,
) -> Result<PathBuf> {
    let dir = models_dir.join(dir_name);
    std::fs::create_dir_all(models_dir).context("create voice models dir")?;

    eprintln!("voice: downloading model from {url}");
    let mut resp = crate::http::download_client(DOWNLOAD_UA)
        .map_err(|e| anyhow!(e))?
        .get(url)
        .send()
        .await?
        .error_for_status()?;
    meter.start(resp.content_length().unwrap_or(0), 0);
    let mut bytes = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        bytes.extend_from_slice(&chunk);
        meter.add(chunk.len() as u64);
    }
    meter.send();

    // Extract away from the live model directory. If the app is interrupted
    // halfway through unpacking a large archive, the next launch must not
    // mistake a lone early .onnx file for a complete model.
    let staging = models_dir.join(format!(".{dir_name}.download-{}", std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).context("remove stale voice-model staging dir")?;
    }
    std::fs::create_dir_all(&staging).context("create voice-model staging dir")?;
    let target = staging.clone();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let bz = bzip2::read::BzDecoder::new(&bytes[..]);
        let mut archive = tar::Archive::new(bz);
        archive.unpack(&target).context("解压语音模型失败")?;
        Ok(())
    })
    .await??;

    let extracted = staging.join(dir_name);
    if !ready(&extracted) {
        let _ = std::fs::remove_dir_all(&staging);
        bail!(
            "语音模型下载不完整或解压后结构异常: {}",
            extracted.display()
        );
    }
    if dir.exists() {
        std::fs::remove_dir_all(&dir).context("remove incomplete voice model")?;
    }
    std::fs::rename(&extracted, &dir).context("install downloaded voice model")?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(dir)
}

/// Take a voice engine's lock, recovering it when an earlier panic poisoned it.
///
/// Treating poisoning as fatal made one bad synthesis permanent: every later
/// call returned the same "lock poisoned" error for the life of the process,
/// so voice appeared to die on its own and only came back with a restart. What
/// the lock guards is a loaded model session, not a half-written structure, so
/// it is taken back and used — and the poisoning is recorded, because a model
/// that really is broken should leave a trace rather than a silence.
fn engine_lock<'a, T>(engine: &'a Mutex<T>, what: &str) -> std::sync::MutexGuard<'a, T> {
    engine.lock().unwrap_or_else(|poisoned| {
        crate::errlog::append_error(
            "voice-lock-recovered",
            &format!("{what} lock was poisoned by an earlier panic; recovered and continuing"),
        );
        poisoned.into_inner()
    })
}

fn stt_engine(dir: &Path, multilingual: bool) -> Result<&'static Mutex<WhisperRecognizer>> {
    let slot = if multilingual {
        &MULTILINGUAL_STT
    } else {
        &EN_STT
    };
    if let Some(e) = slot.get() {
        return Ok(e);
    }
    let encoder = find_in(dir, "encoder.int8.onnx")
        .or_else(|| find_in(dir, "encoder.onnx"))
        .ok_or_else(|| anyhow!("未找到 Whisper encoder"))?;
    let decoder = find_in(dir, "decoder.int8.onnx")
        .or_else(|| find_in(dir, "decoder.onnx"))
        .ok_or_else(|| anyhow!("未找到 Whisper decoder"))?;
    let tokens = find_in(dir, "tokens.txt").ok_or_else(|| anyhow!("未找到 tokens.txt"))?;

    let config = WhisperConfig {
        encoder: encoder.to_string_lossy().into_owned(),
        decoder: decoder.to_string_lossy().into_owned(),
        tokens: tokens.to_string_lossy().into_owned(),
        // Empty enables language auto-detection for multilingual Whisper.
        // English keeps base.en and an explicit language so English-only users
        // retain its accuracy and do not download another model.
        language: if multilingual {
            String::new()
        } else {
            "en".into()
        },
        // Silence appended after the audio so a final word spoken right up to
        // the cut is not lost. How much matters far more than it looks: swept
        // against this model, 100-300 makes base.en degenerate — "hello world,
        // this is a voice test" came back as "Hello" twelve times, and at 100
        // and 200 a sentence also lost its second half — while 0 and anything
        // from 400 up transcribe it correctly. 300 was the value that shipped.
        tail_paddings: Some(1000),
        num_threads: Some(voice_threads()),
        ..Default::default()
    };
    let rec = WhisperRecognizer::new(config).map_err(|e| anyhow!("创建 Whisper 失败: {e}"))?;
    let _ = slot.set(Mutex::new(rec));
    Ok(slot.get().unwrap())
}

fn english_tts_engine(dir: &Path) -> Result<&'static Mutex<KokoroTts>> {
    if let Some(e) = EN_TTS.get() {
        return Ok(e);
    }
    let model = find_in(dir, "model.onnx")
        .or_else(|| find_in(dir, ".onnx"))
        .ok_or_else(|| anyhow!("未找到 Kokoro 模型"))?;

    let config = KokoroTtsConfig {
        model: model.to_string_lossy().into_owned(),
        voices: dir.join("voices.bin").to_string_lossy().into_owned(),
        tokens: dir.join("tokens.txt").to_string_lossy().into_owned(),
        data_dir: dir.join("espeak-ng-data").to_string_lossy().into_owned(),
        length_scale: 1.0,
        lang: "en".into(),
        onnx_config: OnnxConfig {
            provider: "cpu".into(),
            debug: false,
            num_threads: voice_threads(),
        },
        ..Default::default()
    };
    let tts = KokoroTts::new(config);
    let _ = EN_TTS.set(Mutex::new(tts));
    Ok(EN_TTS.get().unwrap())
}

fn chinese_tts_engine(dir: &Path) -> Result<&'static Mutex<VitsTts>> {
    if let Some(e) = ZH_TTS.get() {
        return Ok(e);
    }
    // This model's published lexicon contains blank lines and rare entries
    // whose phonemes are absent from tokens.txt. Older sherpa-onnx releases
    // print a warning for each one, producing tens of thousands of log lines.
    let source_lexicon = dir.join("lexicon.txt");
    let clean_lexicon = dir.join("lexicon.chaty-v2.txt");
    if !clean_lexicon.is_file() {
        let source =
            std::fs::read_to_string(&source_lexicon).context("read Chinese TTS lexicon")?;
        let valid_tokens: std::collections::HashSet<String> =
            std::fs::read_to_string(dir.join("tokens.txt"))
                .context("read Chinese TTS tokens")?
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .map(str::to_owned)
                .collect();
        let cleaned = source
            .lines()
            .filter(|line| {
                let mut fields = line.split_whitespace();
                fields.next().is_some()
                    && fields.clone().next().is_some()
                    && fields.all(|token| valid_tokens.contains(token))
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&clean_lexicon, format!("{cleaned}\n"))
            .context("write cleaned Chinese TTS lexicon")?;
    }
    let config = VitsTtsConfig {
        model: dir.join("model.onnx").to_string_lossy().into_owned(),
        lexicon: clean_lexicon.to_string_lossy().into_owned(),
        tokens: dir.join("tokens.txt").to_string_lossy().into_owned(),
        dict_dir: dir.join("dict").to_string_lossy().into_owned(),
        length_scale: 1.0,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        silence_scale: 1.0,
        onnx_config: OnnxConfig {
            provider: "cpu".into(),
            debug: false,
            num_threads: voice_threads(),
        },
        tts_config: CommonTtsConfig {
            max_num_sentences: 1,
            silence_scale: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let tts = VitsTts::new(config);
    let _ = ZH_TTS.set(Mutex::new(tts));
    Ok(ZH_TTS.get().unwrap())
}

fn contains_cjk(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(
            c as u32,
            0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF
        )
    })
}

/// Which voice speaks `text`. With Chinese enabled, a Han character sends it
/// to VITS. A stretch without one inside a Chinese reply (a model name, a
/// line of code, "OK") stays with VITS too while the English voice has never
/// been downloaded: fetching a 300 MB model from GitHub mid-reply, for a few
/// words, is what issue #20 took for the Chinese voice downloading again.
fn use_chinese_tts(text: &str, chinese_enabled: bool, reply_is_chinese: bool, english_ready: bool) -> bool {
    chinese_enabled && (contains_cjk(text) || (reply_is_chinese && !english_ready))
}

/// Strip Whisper's non-speech annotations for noise — `(buzzing)`, `[BLANK_AUDIO]`,
/// `*wind*`, `（音乐）` etc. If nothing meaningful remains, returns an empty string
/// so the caller ignores it instead of "transcribing" the room tone.
fn clean_transcript(s: &str) -> String {
    let mut out = String::new();
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut star = false;
    for c in s.chars() {
        match c {
            '(' | '（' | '〔' => paren += 1,
            ')' | '）' | '〕' => paren = (paren - 1).max(0),
            '[' | '【' | '［' => brack += 1,
            ']' | '】' | '］' => brack = (brack - 1).max(0),
            '*' => star = !star, // drop *...* asides
            _ if paren == 0 && brack == 0 && !star => out.push(c),
            _ => {}
        }
    }
    let cleaned: String = out.split_whitespace().collect::<Vec<_>>().join(" ");
    // Treat punctuation-only / single-letter leftovers as nothing.
    let meaningful = cleaned.chars().filter(|c| c.is_alphanumeric()).count();
    if meaningful <= 1 {
        String::new()
    } else {
        cleaned
    }
}

/// Whisper writes Chinese in Traditional characters about as readily as in
/// Simplified — its training data is full of Traditional subtitles — so
/// Mandarin could come back as 這是什麼, word by word mixed with Simplified
/// (issue #14). Chaty's Chinese is Simplified, so a transcript is written in
/// it. Japanese is left as it is: kana gives it away, and its kanji are not
/// Traditional Chinese to be converted.
fn to_simplified(text: &str) -> String {
    let kana = text.chars().any(|c| matches!(c as u32, 0x3040..=0x30FF));
    if kana || !contains_cjk(text) {
        text.to_string()
    } else {
        fast2s::convert(text)
    }
}

/// Transcribe mono audio to text (Whisper). Resamples to 16 kHz as needed.
pub async fn transcribe(
    models_dir: PathBuf,
    samples: Vec<f32>,
    sample_rate: u32,
    multilingual: bool,
    endpoint: &str,
    report: Progress<'_>,
) -> Result<String> {
    let model = if multilingual { &WHISPER_MULTILINGUAL } else { &WHISPER_EN };
    let dir = ensure_model(&models_dir, model, endpoint, report).await?;
    eprintln!(
        "voice: transcribing {:.2}s of audio",
        samples.len() as f64 / sample_rate.max(1) as f64
    );
    tokio::task::spawn_blocking(move || -> Result<String> {
        let engine = stt_engine(&dir, multilingual)?;
        let audio = resample_to_16k(&samples, sample_rate);
        let mut rec = engine_lock(engine, "STT");
        let raw = rec.transcribe(16000, &audio).text;
        let text = clean_transcript(raw.trim());
        let text = if multilingual { to_simplified(&text) } else { text };
        eprintln!(
            "voice: transcription finished ({} chars)",
            text.chars().count()
        );
        Ok(text)
    })
    .await?
}

/// Synthesize speech using Chinese VITS for CJK text and English Kokoro
/// otherwise. Returns (samples, sample_rate).
#[allow(clippy::too_many_arguments)]
pub async fn synthesize(
    models_dir: PathBuf,
    text: String,
    speed: f32,
    sid: i32,
    sid_zh: i32,
    chinese_enabled: bool,
    reply_is_chinese: bool,
    endpoint: &str,
    report: Progress<'_>,
) -> Result<(Vec<f32>, u32)> {
    // Deliberately simple heuristic: when Chinese support is enabled, one Han
    // character routes the whole utterance to VITS. This avoids splitting and
    // stitching mixed-language audio, at the cost of English quality in a
    // mostly-English sentence containing one Chinese word.
    let english_ready = kokoro_model_ready(&models_dir.join(KOKORO.dir));
    if use_chinese_tts(&text, chinese_enabled, reply_is_chinese, english_ready) {
        let dir = ensure_model(&models_dir, &CHINESE_TTS, endpoint, report).await?;
        let latin_only = !contains_cjk(&text);
        tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32)> {
            let engine = chinese_tts_engine(&dir)?;
            let mut tts = engine_lock(engine, "中文 TTS");
            // The Chinese voice is chosen from its OWN list — the two models
            // share nothing but a slider, and folding the Kokoro index onto
            // five VITS speakers meant picking an English voice silently
            // moved the Chinese one.
            match tts.create(&text, sid_zh.rem_euclid(CHINESE_TTS_SPEAKER_COUNT), speed) {
                Ok(audio) => Ok((audio.samples, audio.sample_rate)),
                // A stretch without Chinese in a Chinese reply: the Chinese
                // voice has nothing to say for it ("OK" comes back as no
                // audio at all). A beat of silence, not an error in the
                // middle of a reply.
                Err(_) if latin_only => Ok((vec![0.0; 1600], 16000)),
                Err(e) => Err(anyhow!("中文语音合成失败: {e}")),
            }
        })
        .await?
    } else {
        let dir = ensure_model(&models_dir, &KOKORO, endpoint, report).await?;
        tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32)> {
            let engine = english_tts_engine(&dir)?;
            let mut tts = engine_lock(engine, "TTS");
            let audio = tts
                .create(&text, sid, speed)
                .map_err(|e| anyhow!("合成失败: {e}"))?;
            Ok((audio.samples, audio.sample_rate))
        })
        .await?
    }
}

/// Linear resample to 16 kHz (good enough for speech recognition).
fn resample_to_16k(samples: &[f32], sr: u32) -> Vec<f32> {
    if sr == 16000 || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = 16000f32 / sr as f32;
    let out_len = ((samples.len() as f32) * ratio).round() as usize;
    let last = samples.len() - 1;
    (0..out_len)
        .map(|i| {
            let src = i as f32 / ratio;
            let idx = src as usize;
            let frac = src - idx as f32;
            let a = samples[idx.min(last)];
            let b = samples[(idx + 1).min(last)];
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_model_readiness_requires_every_runtime_file() {
        let root = std::env::temp_dir().join(format!(
            "chaty-voice-ready-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let whisper = root.join("whisper");
        let kokoro = root.join("kokoro");
        std::fs::create_dir_all(&whisper).unwrap();
        std::fs::create_dir_all(&kokoro).unwrap();

        let encoder = std::fs::File::create(whisper.join("encoder.onnx")).unwrap();
        encoder.set_len(80 * 1024 * 1024).unwrap();
        let decoder = std::fs::File::create(whisper.join("decoder.int8.onnx")).unwrap();
        decoder.set_len(100 * 1024 * 1024).unwrap();
        assert!(!whisper_model_ready(&whisper));
        let tokens = std::fs::File::create(whisper.join("tokens.txt")).unwrap();
        tokens.set_len(100 * 1024).unwrap();
        assert!(whisper_model_ready(&whisper));

        std::fs::write(kokoro.join("model.onnx"), []).unwrap();
        std::fs::write(kokoro.join("voices.bin"), []).unwrap();
        std::fs::write(kokoro.join("tokens.txt"), []).unwrap();
        assert!(!kokoro_model_ready(&kokoro));
        std::fs::create_dir(kokoro.join("espeak-ng-data")).unwrap();
        assert!(kokoro_model_ready(&kokoro));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn speech_language_routing_detects_cjk() {
        assert!(contains_cjk("你好，世界"));
        assert!(contains_cjk("Hello 世界"));
        assert!(!contains_cjk("Hello, world."));
        assert!(use_chinese_tts("Hello 世界", true, false, true));
        assert!(!use_chinese_tts("Hello 世界", false, true, false));
        assert!(!use_chinese_tts("Hello, world.", true, false, true));
        // Issue #20: a stretch without Han inside a Chinese reply stays with
        // the Chinese voice while the English one was never downloaded…
        assert!(use_chinese_tts("Qwythos 9B.", true, true, false));
        // …and goes to the English voice once it is there, or when the
        // reply is English.
        assert!(!use_chinese_tts("Qwythos 9B.", true, true, true));
        assert!(!use_chinese_tts("Hello, world.", true, false, false));
    }

    /// Issue #20: a download stuck on a connection that never sends another
    /// byte stops when its × is pressed, and closes its progress bar.
    #[test]
    fn a_stalled_download_stops_when_cancelled() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        // Accept and say nothing, ever.
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for c in listener.incoming().flatten() {
                held.push(c);
            }
        });
        let url: &'static str = Box::leak(format!("http://127.0.0.1:{port}/model.tar.bz2").into_boxed_str());
        let m = VoiceModel { dir: "stalled-model", what: "tts", ready: |_| false, hf: None, archive: url };
        let root = std::env::temp_dir().join(format!("chaty-voice-cancel-{}", std::process::id()));
        let reports = std::sync::Mutex::new(Vec::<VoiceDownload>::new());
        let report = |d: VoiceDownload| reports.lock().unwrap().push(d);
        let started = std::time::Instant::now();
        let got = rt.block_on(async {
            let canceller = tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                cancel_download();
            });
            let got = ensure_model(&root, &m, OFFICIAL, &report).await;
            canceller.await.unwrap();
            got
        });
        let err = got.expect_err("a cancelled download fails").to_string();
        assert!(err.contains(DOWNLOAD_CANCELLED), "{err}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5), "took {:?}", started.elapsed());
        let r = reports.lock().unwrap();
        assert!(r.last().is_some_and(|d| d.done && d.voice == "stalled-model"), "{r:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    const OFFICIAL: &str = crate::download::HF_OFFICIAL;

    // Issue #14: a model no source can deliver says where it goes, what to
    // fetch and why each source failed — a payload the UI puts into words.
    #[test]
    fn unreachable_sources_report_folder_files_and_reasons() {
        const DEAD: VoiceModel = VoiceModel {
            dir: "dead-model",
            what: "stt",
            ready: whisper_model_ready,
            hf: Some(HfSnapshot {
                repo: "someone/dead-model",
                rev: "abc",
                files: &[("a.onnx", 10), ("sub/b.txt", 5)],
            }),
            archive: "http://127.0.0.1:9/dead-model.tar.bz2",
        };
        let root = std::env::temp_dir().join(format!("chaty-voice-dead-{}", std::process::id()));
        let rt = tokio::runtime::Runtime::new().unwrap();
        let events = Mutex::new(Vec::new());
        let report = |d: VoiceDownload| events.lock().unwrap().push(d);
        let err = rt
            .block_on(ensure_model(&root, &DEAD, "http://127.0.0.1:9", &report))
            .unwrap_err()
            .to_string();
        // A download that failed still ends, so the UI stops showing it.
        let events = events.into_inner().unwrap();
        assert!(events.last().is_some_and(|d| d.done), "{events:?}");
        let json = err.strip_prefix("VOICE_MODEL_DOWNLOAD ").expect("marked payload");
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["dir"], root.join("dead-model").to_string_lossy().as_ref());
        assert_eq!(v["modelsDir"], root.to_string_lossy().as_ref());
        assert_eq!(v["hfUrl"], "http://127.0.0.1:9/someone/dead-model/tree/abc");
        assert_eq!(v["files"], serde_json::json!(["a.onnx", "sub/b.txt"]));
        assert_eq!(v["archive"], DEAD.archive);
        let reasons: Vec<&str> = v["reasons"].as_array().unwrap().iter().map(|r| r.as_str().unwrap()).collect();
        assert_eq!(reasons, vec!["127.0.0.1: could not connect"; 2]);
        let _ = std::fs::remove_dir_all(root);
    }

    // The models' Hugging Face sources are real and their pinned sizes right.
    #[test]
    fn every_hf_snapshot_is_pinned_and_sized() {
        for m in [&WHISPER_EN, &WHISPER_MULTILINGUAL, &CHINESE_TTS] {
            let hf = m.hf.as_ref().unwrap();
            assert_eq!(hf.rev.len(), 40, "{} must pin a commit", hf.repo);
            assert!(hf.files.iter().all(|&(_, size)| size > 0));
        }
        assert!(KOKORO.hf.is_none() && KOKORO.archive.ends_with("kokoro-en-v0_19.tar.bz2"));
    }

    fn quiet(_: VoiceDownload) {}

    // Issue #14: Mandarin came back partly in Traditional characters.
    #[test]
    fn transcripts_are_written_in_simplified_chinese() {
        assert_eq!(to_simplified("這是什麼問題"), "这是什么问题");
        assert_eq!(to_simplified("那麼我們準備出發了"), "那么我们准备出发了");
        // Words, not characters: 著 stays in 著名 and becomes 着 in 看著.
        assert_eq!(to_simplified("他很著名，我正看著你"), "他很著名，我正看着你");
        assert_eq!(to_simplified("乾隆的衣服很乾淨"), "乾隆的衣服很干净");
        assert_eq!(to_simplified("已经是简体了"), "已经是简体了");
        assert_eq!(to_simplified("Hello, world."), "Hello, world.");
        // Japanese keeps its own kanji.
        assert_eq!(to_simplified("これは東京の電車です"), "これは東京の電車です");
    }

    fn two_files_ready(dir: &Path) -> bool {
        file_complete(&dir.join("a.bin"), 300_000) && file_complete(&dir.join("b.bin"), 300_000)
    }

    /// A local server that answers every request with `body`.
    fn serve(body: Vec<u8>) -> u16 {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut req = [0u8; 4096];
                let _ = s.read(&mut req);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&body);
            }
        });
        port
    }

    // Issue #14: a first download ran behind a spinner and read as hung.
    #[test]
    fn a_voice_download_reports_its_progress_and_its_end() {
        const SMALL: VoiceModel = VoiceModel {
            dir: "small-model",
            what: "stt",
            ready: two_files_ready,
            hf: Some(HfSnapshot {
                repo: "someone/small-model",
                rev: "abc",
                files: &[("a.bin", 300_000), ("b.bin", 300_000)],
            }),
            archive: "http://127.0.0.1:9/small-model.tar.bz2",
        };
        let port = serve(vec![7u8; 300_000]);
        let base = format!("http://127.0.0.1:{port}");
        let root = std::env::temp_dir().join(format!("chaty-voice-progress-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let rt = tokio::runtime::Runtime::new().unwrap();

        let events = Mutex::new(Vec::new());
        let report = |d: VoiceDownload| events.lock().unwrap().push((d.downloaded, d.total, d.done));
        rt.block_on(ensure_model(&root, &SMALL, &base, &report)).expect("download");
        let events = events.into_inner().unwrap();
        assert_eq!(events.first(), Some(&(0, 600_000, false)), "{events:?}");
        assert!(events.contains(&(300_000, 600_000, false)), "the first file's end is reported: {events:?}");
        assert_eq!(events[events.len() - 2], (600_000, 600_000, false), "{events:?}");
        assert_eq!(events.last(), Some(&(600_000, 600_000, true)), "{events:?}");
        assert!(events.windows(2).all(|w| w[0].0 <= w[1].0), "never goes backwards: {events:?}");

        // A resumed download starts from what is already there.
        std::fs::remove_file(root.join("small-model/b.bin")).unwrap();
        let events = Mutex::new(Vec::new());
        let report = |d: VoiceDownload| events.lock().unwrap().push((d.downloaded, d.total, d.done));
        rt.block_on(ensure_model(&root, &SMALL, &base, &report)).expect("resume");
        assert_eq!(events.into_inner().unwrap().first(), Some(&(300_000, 600_000, false)));

        // A model already in place reports nothing at all.
        let calls = Mutex::new(0);
        let count = |_: VoiceDownload| *calls.lock().unwrap() += 1;
        rt.block_on(ensure_model(&root, &SMALL, &base, &count)).expect("ready");
        assert_eq!(calls.into_inner().unwrap(), 0);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Issue #14 end to end: the Chinese voice pair fetched file by file
    /// through a mirror into an empty folder, then spoken and heard —
    ///   cargo test --release --lib voice_models_download_through_mirror -- --ignored --nocapture
    /// (CHATY_TEST_HF_ENDPOINT overrides https://hf-mirror.com)
    #[test]
    #[ignore]
    fn voice_models_download_through_mirror() {
        let endpoint = std::env::var("CHATY_TEST_HF_ENDPOINT").unwrap_or_else(|_| "https://hf-mirror.com".into());
        let dir = std::env::temp_dir().join(format!("chaty-voice-mirror-{}", std::process::id()));
        let rt = tokio::runtime::Runtime::new().unwrap();
        for m in [&CHINESE_TTS, &WHISPER_MULTILINGUAL] {
            let t = std::time::Instant::now();
            let target = dir.join(m.dir);
            rt.block_on(fetch_snapshot(&target, m.hf.as_ref().unwrap(), &endpoint, &mut Meter::new(&quiet, m.what, m.dir)))
                .unwrap_or_else(|e| panic!("{} through {endpoint}: {e:#}", m.dir));
            assert!((m.ready)(&target), "{} incomplete", m.dir);
            println!("{} via {endpoint} in {:.0?}", m.dir, t.elapsed());
        }
        let (samples, rate) = rt
            .block_on(synthesize(dir.clone(), "你好，这是一个中文语音测试。".into(), 1.0, 0, 0, true, true, &endpoint, &quiet))
            .expect("Chinese TTS");
        let text = rt
            .block_on(transcribe(dir.clone(), samples, rate, true, &endpoint, &quiet))
            .expect("multilingual Whisper");
        println!("heard: {text}");
        assert!(text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)), "no Chinese heard: {text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Full voice loop on the real CPU engines (Kokoro TTS → Whisper STT),
    /// using the app's downloaded voice models:
    ///   CHATY_TEST_VOICE_DIR="$HOME/Library/Application Support/com.chaty.desktop/voice-models" \
    ///   cargo test --lib voice_tts_stt_roundtrip -- --ignored
    #[test]
    #[ignore]
    fn voice_tts_stt_roundtrip() {
        let dir = std::path::PathBuf::from(
            std::env::var("CHATY_TEST_VOICE_DIR").expect("set CHATY_TEST_VOICE_DIR"),
        );
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (samples, rate) = rt
            .block_on(synthesize(
                dir.clone(),
                "hello world, this is a voice test".into(),
                1.0,
                0,
                0,
                false,
                false,
                OFFICIAL,
                &quiet,
            ))
            .expect("kokoro synthesis");
        assert!(rate >= 16000, "sane sample rate: {rate}");
        assert!(
            samples.len() as u32 > rate,
            "at least a second of audio: {}",
            samples.len()
        );
        let text = rt
            .block_on(transcribe(dir, samples, rate, false, OFFICIAL, &quiet))
            .expect("whisper transcription");
        let low = text.to_lowercase();
        assert!(
            low.contains("hello") && low.contains("world"),
            "roundtrip lost the words: {text}"
        );
    }

    /// Chinese VITS → multilingual Whisper round trip:
    ///   CHATY_TEST_VOICE_DIR="$HOME/Library/Application Support/com.chaty.desktop/voice-models" \
    ///   cargo test --release --lib voice_chinese_tts_stt_roundtrip -- --ignored --nocapture
    #[test]
    #[ignore]
    fn voice_chinese_tts_stt_roundtrip() {
        let dir = std::path::PathBuf::from(
            std::env::var("CHATY_TEST_VOICE_DIR").expect("set CHATY_TEST_VOICE_DIR"),
        );
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (samples, rate) = rt
            .block_on(synthesize(
                dir.clone(),
                "你好，这是一个中文语音测试。".into(),
                1.0,
                0,
                // The Chinese speaker comes from its own list now — 0 is
                // suyingxue, the first of the model's five.
                0,
                true,
                true,
                OFFICIAL,
                &quiet,
            ))
            .expect("Chinese VITS synthesis");
        assert!(rate >= 8000, "sane sample rate: {rate}");
        assert!(!samples.is_empty(), "Chinese TTS returned no samples");
        let text = rt
            .block_on(transcribe(dir, samples, rate, true, OFFICIAL, &quiet))
            .expect("multilingual Whisper transcription");
        let han = text
            .chars()
            .filter(|c| matches!(*c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF))
            .count();
        assert!(han >= 4, "Chinese roundtrip lost the language: {text}");
    }
}
