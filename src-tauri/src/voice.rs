//! Voice engines: Whisper STT + Kokoro TTS via sherpa-onnx (ONNX Runtime, CPU).
//!
//! Runs entirely on CPU (`provider = "cpu"`) so it never touches the LLM's GPU
//! memory. Uses ONNX Runtime, fully isolated from llama.cpp's ggml. Models
//! auto-download (sherpa-onnx `.tar.bz2` bundles) to the app data dir on first use.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use sherpa_rs::tts::{CommonTtsConfig, KokoroTts, KokoroTtsConfig, VitsTts, VitsTtsConfig};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};
use sherpa_rs::OnnxConfig;

const WHISPER_EN_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2";
const WHISPER_EN_DIR: &str = "sherpa-onnx-whisper-base.en";
// Pin the Hugging Face snapshot: size checks detect truncation, not an
// upstream file being replaced with different valid bytes.
const WHISPER_MULTILINGUAL_REVISION: &str = "bb53ee204431c90d314c1cc08d28d23e5b7927cc";
const WHISPER_MULTILINGUAL_DIR: &str = "sherpa-onnx-whisper-base";
const KOKORO_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2";
const KOKORO_DIR: &str = "kokoro-en-v0_19";
const CHINESE_TTS_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2";
const CHINESE_TTS_DIR: &str = "sherpa-onnx-vits-zh-ll";
const CHINESE_TTS_SPEAKER_COUNT: i32 = 5;

static EN_STT: OnceLock<Mutex<WhisperRecognizer>> = OnceLock::new();
static MULTILINGUAL_STT: OnceLock<Mutex<WhisperRecognizer>> = OnceLock::new();
static EN_TTS: OnceLock<Mutex<KokoroTts>> = OnceLock::new();
static ZH_TTS: OnceLock<Mutex<VitsTts>> = OnceLock::new();

/// Trigger the macOS app-level microphone consent prompt (TCC) and wait for
/// the user's answer. WKWebView's permission delegate auto-grants the webview
/// layer, but the system dialog only appears once something in the process
/// requests capture access — which nothing does unless we ask here. Returns
/// whether access is (now) authorized; always true on other platforms.
#[tauri::command]
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
        // Wait for the dialog; commands run off the main thread, so blocking is fine.
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

fn download_client(timeout: std::time::Duration) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(timeout)
        .build()
        .context("create voice-model HTTP client")
}

async fn download_file(url: &str, dest: &Path, min_bytes: u64) -> Result<()> {
    eprintln!(
        "voice: downloading {}",
        dest.file_name().unwrap_or_default().to_string_lossy()
    );
    let bytes = download_client(std::time::Duration::from_secs(15 * 60))?
        .get(url)
        .send()
        .await
        .with_context(|| format!("下载语音模型文件失败: {url}"))?
        .error_for_status()
        .with_context(|| format!("下载语音模型文件失败: {url}"))?
        .bytes()
        .await
        .with_context(|| format!("读取语音模型文件失败: {url}"))?;
    if bytes.len() < min_bytes as usize {
        bail!(
            "下载的语音模型文件不完整: {}（{} 字节）",
            dest.display(),
            bytes.len()
        );
    }
    let tmp = dest.with_extension(format!(
        "{}.part",
        dest.extension()
            .and_then(|x| x.to_str())
            .unwrap_or_default()
    ));
    std::fs::write(&tmp, &bytes).context("write voice model file")?;
    std::fs::rename(&tmp, dest).context("install voice model file")?;
    Ok(())
}

/// Repair the old partially-extracted Whisper installation one file at a time,
/// avoiding a second download of files that are already complete.
async fn repair_whisper_model(dir: &Path) -> Result<bool> {
    std::fs::create_dir_all(dir).context("create multilingual Whisper directory")?;
    let mut repaired = false;
    let root = format!(
        "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/{WHISPER_MULTILINGUAL_REVISION}"
    );
    let encoder = dir.join("base-encoder.int8.onnx");
    if !file_at_least(&encoder, 20 * 1024 * 1024) {
        download_file(
            &format!("{root}/base-encoder.int8.onnx"),
            &encoder,
            20 * 1024 * 1024,
        )
        .await?;
        repaired = true;
    }
    let decoder = dir.join("base-decoder.int8.onnx");
    if !file_at_least(&decoder, 100 * 1024 * 1024) {
        download_file(
            &format!("{root}/base-decoder.int8.onnx"),
            &decoder,
            100 * 1024 * 1024,
        )
        .await?;
        repaired = true;
    }
    let tokens = dir.join("base-tokens.txt");
    if !file_at_least(&tokens, 100 * 1024) {
        download_file(&format!("{root}/base-tokens.txt"), &tokens, 100 * 1024).await?;
        repaired = true;
    }
    Ok(repaired)
}

async fn ensure_multilingual_whisper_model(models_dir: &Path) -> Result<PathBuf> {
    let dir = models_dir.join(WHISPER_MULTILINGUAL_DIR);
    repair_whisper_model(&dir).await?;
    if !whisper_model_ready(&dir) {
        bail!("多语言 Whisper 模型下载不完整: {}", dir.display());
    }
    Ok(dir)
}

async fn ensure_extracted(
    models_dir: &Path,
    url: &str,
    dir_name: &str,
    ready: fn(&Path) -> bool,
) -> Result<PathBuf> {
    let dir = models_dir.join(dir_name);
    if ready(&dir) {
        return Ok(dir);
    }
    std::fs::create_dir_all(models_dir).context("create voice models dir")?;

    eprintln!("voice: downloading model from {url}");
    let bytes = download_client(std::time::Duration::from_secs(30 * 60))?
        .get(url)
        .send()
        .await
        .with_context(|| format!("下载语音模型失败: {url}"))?
        .error_for_status()
        .with_context(|| format!("下载语音模型失败: {url}"))?
        .bytes()
        .await?;

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
        tail_paddings: Some(300),
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

fn use_chinese_tts(text: &str, chinese_enabled: bool) -> bool {
    chinese_enabled && contains_cjk(text)
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

/// Transcribe mono audio to text (Whisper). Resamples to 16 kHz as needed.
pub async fn transcribe(
    models_dir: PathBuf,
    samples: Vec<f32>,
    sample_rate: u32,
    multilingual: bool,
) -> Result<String> {
    let dir = if multilingual {
        ensure_multilingual_whisper_model(&models_dir).await?
    } else {
        ensure_extracted(
            &models_dir,
            WHISPER_EN_URL,
            WHISPER_EN_DIR,
            whisper_model_ready,
        )
        .await?
    };
    eprintln!(
        "voice: transcribing {:.2}s of audio",
        samples.len() as f64 / sample_rate.max(1) as f64
    );
    tokio::task::spawn_blocking(move || -> Result<String> {
        let engine = stt_engine(&dir, multilingual)?;
        let audio = resample_to_16k(&samples, sample_rate);
        let mut rec = engine.lock().map_err(|_| anyhow!("STT lock poisoned"))?;
        let raw = rec.transcribe(16000, &audio).text;
        let text = clean_transcript(raw.trim());
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
pub async fn synthesize(
    models_dir: PathBuf,
    text: String,
    speed: f32,
    sid: i32,
    sid_zh: i32,
    chinese_enabled: bool,
) -> Result<(Vec<f32>, u32)> {
    // Deliberately simple heuristic: when Chinese support is enabled, one Han
    // character routes the whole utterance to VITS. This avoids splitting and
    // stitching mixed-language audio, at the cost of English quality in a
    // mostly-English sentence containing one Chinese word.
    if use_chinese_tts(&text, chinese_enabled) {
        let dir = ensure_extracted(
            &models_dir,
            CHINESE_TTS_URL,
            CHINESE_TTS_DIR,
            chinese_tts_model_ready,
        )
        .await?;
        tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32)> {
            let engine = chinese_tts_engine(&dir)?;
            let mut tts = engine
                .lock()
                .map_err(|_| anyhow!("中文 TTS lock poisoned"))?;
            let audio = tts
                // The Chinese voice is chosen from its OWN list — the two
                // models share nothing but a slider, and folding the Kokoro
                // index onto five VITS speakers meant picking an English
                // voice silently moved the Chinese one.
                .create(&text, sid_zh.rem_euclid(CHINESE_TTS_SPEAKER_COUNT), speed)
                .map_err(|e| anyhow!("中文语音合成失败: {e}"))?;
            Ok((audio.samples, audio.sample_rate))
        })
        .await?
    } else {
        let dir = ensure_extracted(&models_dir, KOKORO_URL, KOKORO_DIR, kokoro_model_ready).await?;
        tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32)> {
            let engine = english_tts_engine(&dir)?;
            let mut tts = engine.lock().map_err(|_| anyhow!("TTS lock poisoned"))?;
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
        assert!(use_chinese_tts("Hello 世界", true));
        assert!(!use_chinese_tts("Hello 世界", false));
        assert!(!use_chinese_tts("Hello, world.", true));
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
            ))
            .expect("kokoro synthesis");
        assert!(rate >= 16000, "sane sample rate: {rate}");
        assert!(
            samples.len() as u32 > rate,
            "at least a second of audio: {}",
            samples.len()
        );
        let text = rt
            .block_on(transcribe(dir, samples, rate, false))
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
            ))
            .expect("Chinese VITS synthesis");
        assert!(rate >= 8000, "sane sample rate: {rate}");
        assert!(!samples.is_empty(), "Chinese TTS returned no samples");
        let text = rt
            .block_on(transcribe(dir, samples, rate, true))
            .expect("multilingual Whisper transcription");
        let han = text
            .chars()
            .filter(|c| matches!(*c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF))
            .count();
        assert!(han >= 4, "Chinese roundtrip lost the language: {text}");
    }
}
