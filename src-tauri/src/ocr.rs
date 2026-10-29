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
    std::fs::create_dir_all(dir).context(trf!("无法创建模型文件夹", "could not create the models folder"))?;
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
        .with_context(|| trf!("下载 OCR 模型失败:{url}", "could not download the OCR model: {url}"))?
        .error_for_status()?
        .bytes()
        .await?;
    // Write beside the target and move it into place. Written straight to the
    // final name, a write cut short — a full disk, a quit — left a file that
    // exists, which is the only thing `ensure_models` asks: the download was
    // never retried and every load failed from then on.
    let tmp = to.with_extension("part");
    std::fs::write(&tmp, &bytes).context(trf!("无法写入模型文件", "could not write the model file"))?;
    std::fs::rename(&tmp, to).context(trf!("无法保存模型文件", "could not save the model file"))?;
    Ok(())
}

/// Extract Latin text from an image file. Returns empty string if no text found.
pub async fn ocr_image(models_dir: PathBuf, image_path: String) -> Result<String> {
    let (det, rec) = ensure_models(&models_dir).await?;

    // Model loading + inference is CPU-bound and synchronous.
    tokio::task::spawn_blocking(move || -> Result<String> {
        // A model file that will not load is a cache to drop, not a permanent
        // state: the next run downloads it again rather than failing forever.
        let detection_model = Model::load_file(&det).map_err(|e| {
            let _ = std::fs::remove_file(&det);
            anyhow::anyhow!(trf!("加载检测模型失败(已清除缓存,下次会重新下载):{e}", "could not load the text-detection model (its cache was cleared; it downloads again next time): {e}"))
        })?;
        let recognition_model = Model::load_file(&rec).map_err(|e| {
            let _ = std::fs::remove_file(&rec);
            anyhow::anyhow!(trf!("加载识别模型失败(已清除缓存,下次会重新下载):{e}", "could not load the text-recognition model (its cache was cleared; it downloads again next time): {e}"))
        })?;

        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection_model),
            recognition_model: Some(recognition_model),
            ..Default::default()
        })?;

        let img = image::open(&image_path)
            .context(trf!("打开图片失败", "could not open the image"))?
            .into_rgb8();
        let (w, h) = img.dimensions();
        let source = ImageSource::from_bytes(img.as_raw(), (w, h))
            .map_err(|e| anyhow::anyhow!(trf!("图像预处理失败:{e:?}", "image preprocessing failed: {e:?}")))?;
        let input = engine.prepare_input(source)?;
        engine.get_text(&input)
    })
    .await
    .context(trf!("OCR 任务异常", "the OCR task failed"))?
}
