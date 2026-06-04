//! Voice engines: Whisper STT + Kokoro TTS via sherpa-onnx (ONNX Runtime, CPU).
//!
//! Runs entirely on CPU (`provider = "cpu"`) so it never touches the LLM's GPU
//! memory. Uses ONNX Runtime, fully isolated from llama.cpp's ggml. Models
//! auto-download (sherpa-onnx `.tar.bz2` bundles) to the app data dir on first use.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use sherpa_rs::tts::{KokoroTts, KokoroTtsConfig};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};
use sherpa_rs::OnnxConfig;

const WHISPER_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2";
const WHISPER_DIR: &str = "sherpa-onnx-whisper-base.en";
const KOKORO_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2";
const KOKORO_DIR: &str = "kokoro-en-v0_19";

static STT: OnceLock<Mutex<WhisperRecognizer>> = OnceLock::new();
static TTS: OnceLock<Mutex<KokoroTts>> = OnceLock::new();

/// Threads for the ONNX voice engines — use most cores for snappier STT/TTS,
/// but leave one for the rest of the app and cap to avoid oversubscribing the
/// CPU while the LLM is also generating.
fn voice_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    cores.saturating_sub(1).clamp(2, 8) as i32
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

async fn ensure_extracted(models_dir: &Path, url: &str, dir_name: &str) -> Result<PathBuf> {
    let dir = models_dir.join(dir_name);
    if dir.is_dir() && find_in(&dir, ".onnx").is_some() {
        return Ok(dir);
    }
    std::fs::create_dir_all(models_dir).context("create voice models dir")?;

    let bytes = reqwest::get(url)
        .await
        .with_context(|| format!("下载语音模型失败: {url}"))?
        .error_for_status()
        .with_context(|| format!("下载语音模型失败: {url}"))?
        .bytes()
        .await?;

    let target = models_dir.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let bz = bzip2::read::BzDecoder::new(&bytes[..]);
        let mut archive = tar::Archive::new(bz);
        archive.unpack(&target).context("解压语音模型失败")?;
        Ok(())
    })
    .await??;

    if !(dir.is_dir() && find_in(&dir, ".onnx").is_some()) {
        bail!("语音模型解压后结构异常: {}", dir.display());
    }
    Ok(dir)
}

fn stt_engine(dir: &Path) -> Result<&'static Mutex<WhisperRecognizer>> {
    if let Some(e) = STT.get() {
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
        language: "en".into(),
        num_threads: Some(voice_threads()),
        ..Default::default()
    };
    let rec = WhisperRecognizer::new(config).map_err(|e| anyhow!("创建 Whisper 失败: {e}"))?;
    let _ = STT.set(Mutex::new(rec));
    Ok(STT.get().unwrap())
}

fn tts_engine(dir: &Path) -> Result<&'static Mutex<KokoroTts>> {
    if let Some(e) = TTS.get() {
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
    let _ = TTS.set(Mutex::new(tts));
    Ok(TTS.get().unwrap())
}

/// Transcribe mono audio to text (Whisper). Resamples to 16 kHz as needed.
pub async fn transcribe(models_dir: PathBuf, samples: Vec<f32>, sample_rate: u32) -> Result<String> {
    let dir = ensure_extracted(&models_dir, WHISPER_URL, WHISPER_DIR).await?;
    tokio::task::spawn_blocking(move || -> Result<String> {
        let engine = stt_engine(&dir)?;
        let audio = resample_to_16k(&samples, sample_rate);
        let mut rec = engine.lock().map_err(|_| anyhow!("STT lock poisoned"))?;
        Ok(rec.transcribe(16000, &audio).text.trim().to_string())
    })
    .await?
}

/// Synthesize speech for `text` (Kokoro). Returns (samples, sample_rate).
pub async fn synthesize(
    models_dir: PathBuf,
    text: String,
    speed: f32,
    sid: i32,
) -> Result<(Vec<f32>, u32)> {
    let dir = ensure_extracted(&models_dir, KOKORO_URL, KOKORO_DIR).await?;
    tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32)> {
        let engine = tts_engine(&dir)?;
        let mut tts = engine.lock().map_err(|_| anyhow!("TTS lock poisoned"))?;
        let audio = tts.create(&text, sid, speed).map_err(|e| anyhow!("合成失败: {e}"))?;
        Ok((audio.samples, audio.sample_rate))
    })
    .await?
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
