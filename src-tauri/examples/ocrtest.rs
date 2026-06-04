//! Headless check for Latin OCR (downloads models on first run).
//!   cargo run --example ocrtest -- <image.png|jpg>

use anyhow::Context;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).context("usage: ocrtest <image>")?;
    let dir = std::env::temp_dir().join("chaty-ocr-models");
    eprintln!("models dir: {}\nimage: {path}\n(running…)", dir.display());

    let text = chaty_lib::ocr::ocr_image(dir, path)
        .await
        .map_err(|e| anyhow::anyhow!("{e:#}"))?;

    println!("--- OCR text ({} chars) ---\n{text}", text.chars().count());
    Ok(())
}
