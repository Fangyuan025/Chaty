//! Read a user-attached file into plain text so it can be injected as context
//! ("长文档秒问"). Text files auto-decode UTF-8 / GB18030; PDFs are extracted.

use std::path::Path;

use serde::Serialize;
use tauri::Manager;

/// Hard cap on returned characters (the model's context is the real limit).
const MAX_CHARS: usize = 16000;

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Attachment {
    pub name: String,
    pub kind: String,
    pub text: String,
    pub chars: usize,
    pub truncated: bool,
    /// Embedded raster images extracted from the document (cached copies) —
    /// vision models see these alongside the text; empty for plain text files.
    pub images: Vec<String>,
}

#[tauri::command]
pub async fn read_attachment(app: tauri::AppHandle, path: String) -> Result<Attachment, String> {
    let p = Path::new(&path);
    let name = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("file")
        .to_string();
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (kind, mut text) = match ext.as_str() {
        "pdf" => {
            let path = path.clone();
            let extracted =
                tokio::task::spawn_blocking(move || crate::rag::extract_pdf(&path))
                    .await
                    .map_err(|e| e.to_string())??;
            ("pdf".to_string(), extracted)
        }
        "docx" => ("docx".to_string(), crate::rag::extract_docx(&path)?),
        "xlsx" => ("xlsx".to_string(), crate::rag::extract_xlsx(&path)?),
        "pptx" => ("pptx".to_string(), crate::rag::extract_pptx(&path)?),
        "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif" => {
            let dir = app
                .path()
                .app_data_dir()
                .map_err(|e| e.to_string())?
                .join("ocr-models");
            let text = crate::ocr::ocr_image(dir, path.clone())
                .await
                .map_err(|e| format!("OCR 失败：{e:#}"))?;
            ("image".to_string(), text)
        }
        _ => ("text".to_string(), read_text_file(&path)?),
    };

    // Embedded images (docx/xlsx/pptx/pdf) ride along so vision models can SEE
    // charts, photos and screenshots inside the document — not just the text.
    let images = if matches!(ext.as_str(), "pdf" | "docx" | "xlsx" | "pptx") {
        let p2 = path.clone();
        tokio::task::spawn_blocking(move || crate::docimg::extract_embedded_images(&p2, 6))
            .await
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let total = text.chars().count();
    let truncated = total > MAX_CHARS;
    if truncated {
        text = text.chars().take(MAX_CHARS).collect();
    }
    let text = text.trim().to_string();
    if text.is_empty() && images.is_empty() {
        return Err("没有从文件中解析到文本内容".into());
    }

    Ok(Attachment {
        name,
        kind,
        chars: total,
        truncated,
        text,
        images,
    })
}

/// Read a text file as UTF-8, falling back to GB18030 for legacy Chinese files.
fn read_text_file(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    match std::str::from_utf8(&bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => {
            let (cow, _, _) = encoding_rs::GB18030.decode(&bytes);
            Ok(cow.into_owned())
        }
    }
}
