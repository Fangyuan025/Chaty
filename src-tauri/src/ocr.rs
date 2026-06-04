//! Latin-script OCR via `ocrs` (pure-Rust, `rten` runtime). On first use the
//! detection + recognition models (~16 MB) are downloaded and cached to disk.
//!
//! Note: the default ocrs models are Latin-only — Chinese/CJK is not supported.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;

const DET_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten";
const REC_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten";

async fn ensure_models(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(dir).context("create models dir")?;
    let det = dir.join("text-detection.rten");
    let rec = dir.join("text-recognition.rten");
    if !det.exists() {
        download(DET_URL, &det).await?;
    }
    if !rec.exists() {
        download(REC_URL, &rec).await?;
    }
    Ok((det, rec))
}

async fn download(url: &str, to: &Path) -> Result<()> {
    let bytes = reqwest::get(url)
        .await
        .with_context(|| format!("下载 OCR 模型失败: {url}"))?
        .error_for_status()?
        .bytes()
        .await?;
    std::fs::write(to, &bytes).context("write model file")?;
    Ok(())
}

/// Extract Latin text from an image file. Returns empty string if no text found.
pub async fn ocr_image(models_dir: PathBuf, image_path: String) -> Result<String> {
    let (det, rec) = ensure_models(&models_dir).await?;

    // Model loading + inference is CPU-bound and synchronous.
    tokio::task::spawn_blocking(move || -> Result<String> {
        let detection_model =
            Model::load_file(&det).map_err(|e| anyhow::anyhow!("加载检测模型失败: {e}"))?;
        let recognition_model =
            Model::load_file(&rec).map_err(|e| anyhow::anyhow!("加载识别模型失败: {e}"))?;

        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection_model),
            recognition_model: Some(recognition_model),
            ..Default::default()
        })?;

        let img = image::open(&image_path)
            .context("打开图片失败")?
            .into_rgb8();
        let (w, h) = img.dimensions();
        let source = ImageSource::from_bytes(img.as_raw(), (w, h))
            .map_err(|e| anyhow::anyhow!("图像预处理失败: {e:?}"))?;
        let input = engine.prepare_input(source)?;
        engine.get_text(&input)
    })
    .await
    .context("OCR 任务异常")?
}
