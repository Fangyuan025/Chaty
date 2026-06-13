//! In-app model downloader: list GGUF files in a HuggingFace repo and stream one
//! into the writable `models/` folder with progress events.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::ipc::Channel;
use tauri::Manager;

const UA: &str = "Chaty model downloader";

/// Sentinel error for a user-cancelled download (frontends match on it).
pub const CANCELLED: &str = "DOWNLOAD_CANCELLED";

static CANCELS: Mutex<Option<HashMap<String, Arc<AtomicBool>>>> = Mutex::new(None);

/// Register a fresh cancel flag for `key` (one in-flight download per key).
pub fn register_cancel(key: &str) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    CANCELS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key.to_string(), flag.clone());
    flag
}

pub fn clear_cancel(key: &str) {
    if let Some(map) = CANCELS.lock().unwrap().as_mut() {
        map.remove(key);
    }
}

/// Ask an in-flight download to stop. `key` is the filename passed to
/// `download_model`, or `"rag-embed"` for the knowledge-base model.
#[tauri::command]
pub fn cancel_download(key: String) {
    if let Some(map) = CANCELS.lock().unwrap().as_ref() {
        if let Some(flag) = map.get(&key) {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HfFile {
    pub name: String,
    pub size: u64,
    pub url: String,
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum DownloadProgress {
    Progress { downloaded: u64, total: u64 },
    Done { path: String },
    Error { message: String },
}

/// Normalize a user-pasted repo reference to `owner/name`.
fn normalize_repo(input: &str) -> String {
    input
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("huggingface.co/")
        .trim_matches('/')
        .split("/tree/")
        .next()
        .unwrap_or("")
        .split("/blob/")
        .next()
        .unwrap_or("")
        .trim_matches('/')
        .to_string()
}

/// List the `.gguf` files in a HuggingFace model repo (recursively), newest API.
#[tauri::command]
pub async fn list_hf_ggufs(repo: String) -> Result<Vec<HfFile>, String> {
    let repo = normalize_repo(&repo);
    if repo.is_empty() || !repo.contains('/') {
        return Err("请输入有效的 HuggingFace 仓库（owner/name）".into());
    }
    let api = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let client = reqwest::Client::builder()
        .user_agent(UA)
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(&api).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("未找到仓库或无 GGUF 文件: {repo}"));
    }
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(items) = json.as_array() {
        for it in items {
            let path = it.get("path").and_then(|p| p.as_str()).unwrap_or("");
            if path.to_lowercase().ends_with(".gguf") {
                let size = it.get("size").and_then(|x| x.as_u64()).unwrap_or(0);
                let url = format!("https://huggingface.co/{repo}/resolve/main/{path}?download=true");
                let name = path.rsplit('/').next().unwrap_or(path).to_string();
                out.push(HfFile { name, size, url });
            }
        }
    }
    if out.is_empty() {
        return Err(format!("该仓库没有 .gguf 文件: {repo}"));
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(out)
}

/// Stream `url` into `models/<filename>`, reporting progress on `on_progress`.
/// Writes to a `.part` file first and renames on success.
#[tauri::command]
pub async fn download_model(
    app: tauri::AppHandle,
    url: String,
    filename: String,
    on_progress: Channel<DownloadProgress>,
) -> Result<(), String> {
    let cancel = register_cancel(&filename);
    let result = download_inner(&app, &url, &filename, &on_progress, &cancel).await;
    clear_cancel(&filename);
    if let Err(ref e) = result {
        let _ = on_progress.send(DownloadProgress::Error { message: e.clone() });
    }
    result
}

async fn download_inner(
    app: &tauri::AppHandle,
    url: &str,
    filename: &str,
    on_progress: &Channel<DownloadProgress>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let safe: String = filename
        .chars()
        .map(|c| if "/\\:*?\"<>|".contains(c) { '_' } else { c })
        .collect();
    if !safe.to_lowercase().ends_with(".gguf") {
        return Err("文件名必须以 .gguf 结尾".into());
    }
    let dest = dir.join(&safe);
    let tmp = dir.join(format!("{safe}.part"));

    let client = reqwest::Client::builder()
        .user_agent(UA)
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("下载失败: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);

    let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    let mut downloaded: u64 = 0;
    let mut last = std::time::Instant::now();
    on_progress
        .send(DownloadProgress::Progress { downloaded, total })
        .ok();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if cancel.load(Ordering::SeqCst) {
            drop(file);
            let _ = std::fs::remove_file(&tmp);
            return Err(CANCELLED.into());
        }
        file.write_all(&chunk).map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        if last.elapsed().as_millis() >= 200 {
            on_progress
                .send(DownloadProgress::Progress { downloaded, total })
                .ok();
            last = std::time::Instant::now();
        }
    }
    file.flush().ok();
    drop(file);
    std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
    on_progress
        .send(DownloadProgress::Done {
            path: dest.to_string_lossy().to_string(),
        })
        .ok();
    Ok(())
}
