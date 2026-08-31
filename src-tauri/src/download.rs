//! In-app model downloader: list GGUF files in a HuggingFace repo and stream one
//! into the writable `models/` folder with progress events.
//!
//! Plain `resolve/main` GETs redirect to `cas-bridge.xethub.hf.co`, which some
//! networks/ASNs block outright with CloudFront 403s (HF forum 158626,
//! xet-core #800) — every LFS file on every repo fails there. On 403 we fall
//! back to the xet chunk protocol via `hf-hub`, whose data plane
//! (`transfer.xethub.hf.co` / `cas-server`) is not behind the same block.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
        .trim_start_matches("hf-mirror.com/")
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

/// Official HuggingFace host — the default endpoint and the only one the xet
/// fallback protocol works against.
pub(crate) const HF_OFFICIAL: &str = "https://huggingface.co";

/// Base URL for HF API / resolve calls. `None`/empty → official. A user-set
/// mirror (Settings → Model → HF endpoint, e.g. `https://hf-mirror.com`) is
/// path-compatible with the official API. Trailing slashes are trimmed so
/// `format!("{base}/api/…")` composes cleanly.
pub(crate) fn hf_base(endpoint: Option<&str>) -> String {
    let e = endpoint.unwrap_or("").trim().trim_end_matches('/');
    if e.is_empty() {
        HF_OFFICIAL.to_string()
    } else {
        e.to_string()
    }
}

/// Only the official endpoint speaks the xet chunk protocol (hf-mirror.com
/// does not) — gate the 403 CDN fallback on this.
pub(crate) fn is_official_hf(base: &str) -> bool {
    matches!(base, HF_OFFICIAL | "https://hf.co")
}

/// Bilingual error for a 403/failure on a mirror endpoint, where the xet
/// fallback is not available by design.
pub(crate) fn mirror_403_message(status: reqwest::StatusCode) -> String {
    format!(
        "镜像端点拒绝了该文件（HTTP {status}）。该镜像不支持 xet 回退 — \
         请在设置中切回官方 HuggingFace 端点（可能需要 VPN/代理）后重试 \
         (the mirror endpoint rejected this file and does not support the xet \
         fallback; switch back to the official HuggingFace endpoint in Settings — \
         possibly with a VPN/proxy — and retry)"
    )
}

// ---------------------------------------------------------------------------
// xet fallback (CDN-blocked networks)
// ---------------------------------------------------------------------------

/// `(repo, revision, path)` out of a HuggingFace resolve URL such as
/// `https://huggingface.co/owner/name/resolve/main/sub/file.gguf?download=true`.
pub(crate) fn parse_hf_resolve_url(url: &str) -> Option<(String, String, String)> {
    let rest = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("https://hf.co/"))
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let rest = rest.split('?').next().unwrap_or(rest);
    let (repo, tail) = rest.split_once("/resolve/")?;
    let (revision, path) = tail.split_once('/')?;
    if repo.matches('/').count() != 1 || revision.is_empty() || path.is_empty() {
        return None;
    }
    Some((repo.to_string(), revision.to_string(), path.to_string()))
}

/// Escape glob metacharacters so a repo path can be used as an exact
/// `allow_patterns` entry (globset character classes match literals).
fn glob_escape(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for c in path.chars() {
        match c {
            '*' | '?' | '{' | '}' => {
                out.push('[');
                out.push(c);
                out.push(']');
            }
            '[' => out.push_str("[[]"),
            ']' => out.push_str("[]]"),
            _ => out.push(c),
        }
    }
    out
}

/// The bilingual "your network blocks the HF CDN" error shown when the plain
/// GET got a 403 and the xet fallback failed too. Advice precedes the detail —
/// some surfaces truncate long messages.
pub(crate) fn cdn_blocked_message(detail: &str) -> String {
    format!(
        "当前网络封锁了 HuggingFace 的下载 CDN（HTTP 403），xet 回退下载也失败了 — \
         这不是应用故障，请更换网络或启用 VPN/代理后重试 \
         (this network blocks the HF CDN and the xet fallback failed too; \
         try another network or a VPN/proxy)。详情: {detail}"
    )
}

/// Forwards hf-hub download progress into a caller-supplied `(downloaded, total)`
/// callback. Byte counts are kept monotonic across per-file deltas and xet
/// aggregate reports for the single-file snapshots we run.
struct XetProgress<F: Fn(u64, u64) + Send + Sync> {
    downloaded: AtomicU64,
    total: AtomicU64,
    emit: F,
}

impl<F: Fn(u64, u64) + Send + Sync> hf_hub::progress::ProgressHandler for XetProgress<F> {
    fn on_progress(&self, event: &hf_hub::progress::ProgressEvent) {
        use hf_hub::progress::{DownloadEvent, ProgressEvent};
        let ProgressEvent::Download(ev) = event else { return };
        match ev {
            DownloadEvent::Start { total_bytes, .. } => {
                self.total.store(*total_bytes, Ordering::SeqCst);
            }
            DownloadEvent::Progress { files } => {
                for f in files {
                    self.downloaded.fetch_max(f.bytes_completed, Ordering::SeqCst);
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                self.downloaded.fetch_max(*bytes_completed, Ordering::SeqCst);
                if *total_bytes > 0 {
                    self.total.store(*total_bytes, Ordering::SeqCst);
                }
            }
            DownloadEvent::Complete => return,
        }
        (self.emit)(self.downloaded.load(Ordering::SeqCst), self.total.load(Ordering::SeqCst));
    }
}

/// Download one repo file over the xet protocol into `dest`.
///
/// Uses hf-hub's `snapshot_download` narrowed to the single file: unlike its
/// `download_file`, the snapshot path reads xet metadata off the resolve 302
/// with a no-redirect HEAD, so it never touches the blocked cas-bridge CDN.
/// The file lands in a throwaway dir under `tmp_root` first (xet writes with
/// repo path structure) and is renamed into `dest` on success.
pub(crate) async fn xet_fallback_download(
    repo: &str,
    revision: &str,
    path: &str,
    dest: &std::path::Path,
    tmp_root: &std::path::Path,
    on_progress: impl Fn(u64, u64) + Send + Sync + 'static,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let (owner, name) = repo.split_once('/').ok_or_else(|| format!("非法仓库名 (bad repo id): {repo}"))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_dir = tmp_root.join("xet-tmp").join(format!("dl-{stamp}"));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| e.to_string())?;

    let client = hf_hub::HFClient::builder()
        .user_agent(UA)
        .cache_enabled(false)
        .build()
        .map_err(|e| e.to_string())?;
    let handler = XetProgress {
        downloaded: AtomicU64::new(0),
        total: AtomicU64::new(0),
        emit: on_progress,
    };
    let model = client.model(owner, name);
    let fut = model
        .snapshot_download()
        .revision(revision.to_string())
        .allow_patterns(vec![glob_escape(path)])
        .local_dir(tmp_dir.clone())
        .progress(hf_hub::progress::Progress::new(handler))
        .send();
    tokio::pin!(fut);
    let result = loop {
        tokio::select! {
            r = &mut fut => break r,
            _ = tokio::time::sleep(std::time::Duration::from_millis(150)) => {
                if cancel.load(Ordering::SeqCst) {
                    // Best-effort: xet's own tasks may still flush into the
                    // dir briefly; clear_stale_xet_tmp at startup catches any
                    // survivor.
                    let _ = std::fs::remove_dir_all(&tmp_dir);
                    return Err(CANCELLED.into());
                }
            }
        }
    };
    let finish = result.map_err(|e| e.to_string()).and_then(|_| {
        let src = tmp_dir.join(path);
        if !src.is_file() {
            return Err(format!("仓库中没有该文件 (file not found in repo): {path}"));
        }
        std::fs::rename(&src, dest).or_else(|_| std::fs::copy(&src, dest).map(|_| ())).map_err(|e| e.to_string())
    });
    let _ = std::fs::remove_dir_all(&tmp_dir);
    finish
}

/// Remove leftovers of cancelled/crashed xet fallback downloads. Called once
/// at startup, when no download can be in flight.
pub fn clear_stale_xet_tmp(app: &tauri::AppHandle) {
    if let Ok(dir) = app.path().app_data_dir() {
        let _ = std::fs::remove_dir_all(dir.join("xet-tmp"));
    }
}

/// Full recursive file listing of a HuggingFace repo: `(path, size)` pairs.
async fn repo_tree(repo: &str, base: &str) -> Result<Vec<(String, u64)>, String> {
    let api = format!("{base}/api/models/{repo}/tree/main?recursive=true");
    // Metadata call — a whole-request timeout is right here; without one a
    // hung connection kept the "preparing download" spinner up forever.
    let client = crate::http::client_secs(UA, 30)?;
    let resp = client.get(&api).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("未找到仓库 (repo not found): {repo}"));
    }
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(items) = json.as_array() {
        for it in items {
            let path = it.get("path").and_then(|p| p.as_str()).unwrap_or("");
            if !path.is_empty() {
                let size = it.get("size").and_then(|x| x.as_u64()).unwrap_or(0);
                out.push((path.to_string(), size));
            }
        }
    }
    Ok(out)
}

/// List the `.gguf` files in a HuggingFace model repo (recursively), newest API.
#[tauri::command]
pub async fn list_hf_ggufs(repo: String, endpoint: Option<String>) -> Result<Vec<HfFile>, String> {
    let base = hf_base(endpoint.as_deref());
    let repo = normalize_repo(&repo);
    if repo.is_empty() || !repo.contains('/') {
        return Err("请输入有效的 HuggingFace 仓库（owner/name）".into());
    }
    let mut out = Vec::new();
    for (path, size) in repo_tree(&repo, &base).await? {
        if path.to_lowercase().ends_with(".gguf") {
            let url = format!("{base}/{repo}/resolve/main/{path}?download=true");
            let name = path.rsplit('/').next().unwrap_or(&path).to_string();
            out.push(HfFile { name, size, url });
        }
    }
    if out.is_empty() {
        return Err(format!("该仓库没有 .gguf 文件: {repo}"));
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(out)
}

// ---------------------------------------------------------------------------
// HF model store: search / browse / detail (quant-level, no raw file lists)
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HfModelHit {
    /// Repo id, `owner/name`.
    pub id: String,
    pub name: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub updated_at: String,
    pub vision: bool,
    /// Parameter count guessed from the name ("Qwen3-4B…" → 4.0).
    pub params_b: Option<f64>,
}

/// One downloadable variant of a model — a GGUF quant (possibly multi-part)
/// or the single MLX build of a repo.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QuantOption {
    pub label: String,
    pub size: u64,
    /// Repo-relative file paths, in download order. Empty for MLX (the whole
    /// repo downloads as one unit).
    pub files: Vec<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HfModelDetail {
    pub id: String,
    pub format: String,
    pub vision: bool,
    pub params_b: Option<f64>,
    pub arch: Option<String>,
    /// Sorted smallest → largest.
    pub quants: Vec<QuantOption>,
    /// Best paired vision encoder (GGUF repos), downloaded alongside a quant.
    pub mmproj: Option<String>,
    pub mmproj_size: u64,
    /// Raw README markdown ("" when absent).
    pub readme: String,
    /// Total machine RAM in MiB — lets the UI hint "fits fully in memory".
    pub total_ram_mb: u64,
}

fn tag_vision(tags: &[String]) -> bool {
    tags.iter().any(|t| {
        t == "image-text-to-text" || t == "vision" || t == "multimodal" || t == "image-to-text"
    })
}

/// "…Qwen3-4B-Instruct…" → 4.0; "…0.6B…" → 0.6; MoE "35B-A3B" → 35.0 (first).
fn params_from_name(name: &str) -> Option<f64> {
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            if i < bytes.len() && (bytes[i] == b'b' || bytes[i] == b'B') {
                // must be a token edge, not e.g. "…4Bit"
                let next = bytes.get(i + 1);
                let edge = next.map_or(true, |c| !c.is_ascii_alphanumeric());
                // and not preceded by a letter ("IQ4B"?) — digit start is enough
                let prev_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
                if edge && prev_ok {
                    if let Ok(v) = name[start..i].parse::<f64>() {
                        if (0.05..=2000.0).contains(&v) {
                            return Some(v);
                        }
                    }
                }
            }
        } else {
            i += 1;
            continue;
        }
        i += 1;
    }
    None
}

fn arch_from_tags(tags: &[String]) -> Option<String> {
    const KNOWN: [&str; 10] =
        ["qwen", "llama", "gemma", "phi", "mistral", "smollm", "glm", "deepseek", "granite", "internlm"];
    tags.iter()
        .find(|t| {
            let t = t.to_lowercase();
            KNOWN.iter().any(|k| t.starts_with(k))
        })
        .cloned()
}

/// Strip a llama.cpp multi-part suffix: "…-00001-of-00003" → "…", true.
fn strip_multipart(stem: &str) -> (&str, bool) {
    let b = stem.as_bytes();
    // pattern: -DDDDD-of-DDDDD (15 bytes)
    if b.len() > 15 {
        let tail = &stem[stem.len() - 15..];
        let tb = tail.as_bytes();
        if tb[0] == b'-'
            && tb[1..6].iter().all(|c| c.is_ascii_digit())
            && &tail[6..10] == "-of-"
            && tb[10..15].iter().all(|c| c.is_ascii_digit())
        {
            return (&stem[..stem.len() - 15], true);
        }
    }
    (stem, false)
}

/// The quant label inside a GGUF filename: the token that looks like a
/// quantization ("Q4_K_M", "IQ4_XS", "F16", "BF16", "Q8_0", "MXFP4"…).
fn quant_label_of(stem: &str) -> String {
    let looks_quant = |tok: &str| -> bool {
        let t = tok.to_ascii_uppercase();
        t == "F16"
            || t == "F32"
            || t == "BF16"
            || t == "FP16"
            || t.starts_with("MXFP")
            || ((t.starts_with('Q') || t.starts_with("IQ"))
                && t.chars().nth(if t.starts_with("IQ") { 2 } else { 1 })
                    .map_or(false, |c| c.is_ascii_digit()))
    };
    // scan '-'/'.'/'_'-separated tokens from the END; join trailing qualifier
    // parts split by '_' ("Q4_K_M" survives because we split on '-'/'.' only)
    for tok in stem.rsplit(['-', '.']) {
        if looks_quant(tok) {
            return tok.to_ascii_uppercase();
        }
    }
    "DEFAULT".into()
}

/// Group a GGUF repo's tree into quant options (multi-part shards summed,
/// mmproj excluded), sorted smallest → largest.
fn gguf_quants(tree: &[(String, u64)]) -> Vec<QuantOption> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, QuantOption> = BTreeMap::new();
    for (path, size) in tree {
        let lower = path.to_lowercase();
        if !lower.ends_with(".gguf") || lower.contains("mmproj") {
            continue;
        }
        let file = path.rsplit('/').next().unwrap_or(path);
        let stem = &file[..file.len() - 5];
        let (stem, _multi) = strip_multipart(stem);
        let label = quant_label_of(stem);
        let e = groups.entry(label.clone()).or_insert_with(|| QuantOption {
            label,
            size: 0,
            files: Vec::new(),
        });
        e.size += size;
        e.files.push(path.clone());
    }
    let mut out: Vec<QuantOption> = groups.into_values().collect();
    for q in &mut out {
        q.files.sort(); // multipart shards in order
    }
    out.sort_by_key(|q| q.size);
    out
}

/// Best mmproj in a GGUF repo (F16 > BF16 > Q8 > rest, then smallest) —
/// mirrors the frontend pairing rule.
fn best_mmproj(tree: &[(String, u64)]) -> Option<(String, u64)> {
    let score = |name: &str| -> u32 {
        let n = name.to_lowercase();
        if n.contains("f16") && !n.contains("bf16") {
            0
        } else if n.contains("bf16") {
            1
        } else if n.contains("q8") {
            2
        } else {
            3
        }
    };
    tree.iter()
        .filter(|(p, _)| {
            let l = p.to_lowercase();
            l.ends_with(".gguf") && l.contains("mmproj")
        })
        .min_by_key(|(p, s)| (score(p), *s))
        .cloned()
}

static AVATAR_CACHE: Mutex<Option<HashMap<String, Option<String>>>> = Mutex::new(None);

/// The author's official HF avatar (organizations first, then user accounts).
/// Cached per author for the app's lifetime; `None` when there isn't one.
#[tauri::command]
pub async fn hf_author_avatar(author: String) -> Result<Option<String>, String> {
    if let Some(hit) = AVATAR_CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .get(&author)
    {
        return Ok(hit.clone());
    }
    let client = crate::http::client(UA, std::time::Duration::from_secs(30))?;
    let mut url: Option<String> = None;
    for kind in ["organizations", "users"] {
        let resp = client
            .get(format!("https://huggingface.co/api/{kind}/{author}/avatar"))
            .send()
            .await;
        if let Ok(r) = resp {
            if r.status().is_success() {
                if let Ok(v) = r.json::<serde_json::Value>().await {
                    url = v.get("avatarUrl").and_then(|u| u.as_str()).map(str::to_string);
                    if url.is_some() {
                        break;
                    }
                }
            }
        }
    }
    AVATAR_CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(author, url.clone());
    Ok(url)
}

/// Search / browse HF models by keyword, format and sort order. An empty
/// query browses trending models — the "storefront" view.
#[tauri::command]
pub async fn hf_search(
    query: String,
    format: String,
    sort: String,
    limit: Option<u32>,
    endpoint: Option<String>,
) -> Result<Vec<HfModelHit>, String> {
    let base = hf_base(endpoint.as_deref());
    let filter = if format == "mlx" { "mlx" } else { "gguf" };
    let sort = match sort.as_str() {
        "downloads" => "downloads",
        "likes" => "likes",
        "updated" => "lastModified",
        _ => "trendingScore",
    };
    let mut url = format!(
        "{base}/api/models?filter={filter}&sort={sort}&direction=-1&limit={}",
        limit.unwrap_or(30).min(50)
    );
    let q = query.trim();
    if !q.is_empty() {
        url.push_str(&format!("&search={}", percent_encoding::utf8_percent_encode(q, percent_encoding::NON_ALPHANUMERIC)));
    }
    let client = crate::http::client(UA, std::time::Duration::from_secs(30))?;
    let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("搜索失败 (search failed): HTTP {}", resp.status()));
    }
    let items: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(arr) = items.as_array() {
        for it in arr {
            let id = it.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }
            let tags: Vec<String> = it
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            let (author, name) = id.split_once('/').unwrap_or(("", id.as_str()));
            out.push(HfModelHit {
                name: name.to_string(),
                author: author.to_string(),
                downloads: it.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                likes: it.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                updated_at: it
                    .get("lastModified")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                vision: tag_vision(&tags),
                params_b: params_from_name(&id),
                id,
            });
        }
    }
    Ok(out)
}

/// Everything the store's detail pane needs for one repo, quant-level.
#[tauri::command]
pub async fn hf_model_detail(
    repo: String,
    format: String,
    endpoint: Option<String>,
) -> Result<HfModelDetail, String> {
    let base = hf_base(endpoint.as_deref());
    let repo = normalize_repo(&repo);
    if repo.is_empty() || !repo.contains('/') {
        return Err("请输入有效的 HuggingFace 仓库（owner/name）".into());
    }
    let client = crate::http::client(UA, std::time::Duration::from_secs(30))?;

    // tags (vision/arch) — tolerate failure, the tree is the critical part
    let tags: Vec<String> = match client
        .get(format!("{base}/api/models/{repo}"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r
            .json::<serde_json::Value>()
            .await
            .ok()
            .and_then(|v| {
                v.get("tags").and_then(|t| t.as_array()).map(|a| {
                    a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect()
                })
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };

    let tree = repo_tree(&repo, &base).await?;
    let (format, quants, mmproj) = if format == "mlx" || (!tree.iter().any(|(p, _)| p.to_lowercase().ends_with(".gguf")) && mlx_repo_check(&mlx_files(&tree)).is_ok()) {
        let files = mlx_files(&tree);
        mlx_repo_check(&files)?;
        let size: u64 = files.iter().map(|(_, s)| *s).sum();
        let label = ["8bit", "6bit", "5bit", "4bit", "3bit", "2bit", "bf16"]
            .iter()
            .find(|b| repo.to_lowercase().contains(**b))
            .map(|b| b.to_uppercase())
            .unwrap_or_else(|| "MLX".into());
        ("mlx".to_string(), vec![QuantOption { label, size, files: Vec::new() }], None)
    } else {
        let quants = gguf_quants(&tree);
        if quants.is_empty() {
            return Err(format!("该仓库没有可下载的模型文件 (no model files): {repo}"));
        }
        ("gguf".to_string(), quants, best_mmproj(&tree))
    };

    // README (best-effort, capped)
    let readme = match client
        .get(format!("{base}/{repo}/raw/main/README.md"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            let mut t = r.text().await.unwrap_or_default();
            // strip YAML front-matter
            if t.starts_with("---") {
                if let Some(end) = t[3..].find("\n---") {
                    t = t[3 + end + 4..].trim_start().to_string();
                }
            }
            t.chars().take(60_000).collect()
        }
        _ => String::new(),
    };

    let (mmproj, mmproj_size) = mmproj.map_or((None, 0), |(p, s)| (Some(p), s));
    Ok(HfModelDetail {
        vision: tag_vision(&tags) || mmproj.is_some(),
        params_b: params_from_name(&repo),
        arch: arch_from_tags(&tags),
        quants,
        mmproj,
        mmproj_size,
        readme,
        total_ram_mb: {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            sys.total_memory() / (1024 * 1024)
        },
        id: repo,
        format,
    })
}

// ---------------------------------------------------------------------------
// MLX repos (folder models: config.json + safetensors + tokenizer files)
// ---------------------------------------------------------------------------

/// Non-shard files worth pulling from an MLX repo.
///
/// Every top-level `.json`, rather than a list of the ones seen so far: the
/// list silently dropped `processor_config.json`, and a multimodal model whose
/// processor configuration never arrived loads with its vision half disabled.
/// A model folder's top-level JSON is configuration — small, and all of it is
/// something a loader may ask for — so the safe default is to take it.
fn mlx_aux_file(name: &str) -> bool {
    name.ends_with(".json") || matches!(name, "chat_template.jinja" | "merges.txt")
}

/// The subset of a repo tree that makes up an MLX folder model (all
/// top-level: mlx-community repos are flat).
fn mlx_files(tree: &[(String, u64)]) -> Vec<(String, u64)> {
    tree.iter()
        .filter(|(p, _)| {
            !p.contains('/') && (p.to_lowercase().ends_with(".safetensors") || mlx_aux_file(p))
        })
        .cloned()
        .collect()
}

fn mlx_repo_check(files: &[(String, u64)]) -> Result<(), String> {
    let has_cfg = files.iter().any(|(p, _)| p == "config.json");
    let has_st = files.iter().any(|(p, _)| p.to_lowercase().ends_with(".safetensors"));
    if has_cfg && has_st {
        Ok(())
    } else {
        Err("该仓库不是 MLX 模型（需要 config.json 与 .safetensors） \
             (not an MLX model repo: config.json + .safetensors required)"
            .into())
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct MlxRepoInfo {
    /// Suggested model folder name (the repo's name segment).
    pub name: String,
    pub files: u32,
    pub total_size: u64,
}

/// Probe a HuggingFace repo as an MLX folder model (config.json +
/// safetensors). Used by the downloader UI when a repo has no GGUFs.
#[tauri::command]
pub async fn list_hf_mlx(repo: String, endpoint: Option<String>) -> Result<MlxRepoInfo, String> {
    let base = hf_base(endpoint.as_deref());
    let repo = normalize_repo(&repo);
    if repo.is_empty() || !repo.contains('/') {
        return Err("请输入有效的 HuggingFace 仓库（owner/name）".into());
    }
    let files = mlx_files(&repo_tree(&repo, &base).await?);
    mlx_repo_check(&files)?;
    Ok(MlxRepoInfo {
        name: repo.rsplit('/').next().unwrap_or(&repo).to_string(),
        files: files.len() as u32,
        total_size: files.iter().map(|(_, s)| *s).sum(),
    })
}

/// Download a whole MLX repo into `models/<Name>/` with aggregate progress.
/// Cancel key = the folder name. On cancel/failure a folder we created is
/// removed entirely, so a half-downloaded model never shows up in the picker.
#[tauri::command]
pub async fn download_mlx_repo(
    app: tauri::AppHandle,
    repo: String,
    endpoint: Option<String>,
    on_progress: Channel<DownloadProgress>,
) -> Result<String, String> {
    let root = app.path().app_data_dir().map_err(|e| e.to_string())?;
    download_mlx_repo_inner(&root, repo, endpoint, on_progress).await
}

async fn download_mlx_repo_inner(
    root: &std::path::Path,
    repo: String,
    endpoint: Option<String>,
    on_progress: Channel<DownloadProgress>,
) -> Result<String, String> {
    let base = hf_base(endpoint.as_deref());
    let repo = normalize_repo(&repo);
    if repo.is_empty() || !repo.contains('/') {
        return Err("请输入有效的 HuggingFace 仓库（owner/name）".into());
    }
    let mut files = mlx_files(&repo_tree(&repo, &base).await?);
    mlx_repo_check(&files)?;
    // Small aux files first, shards last — a failure wastes as little
    // bandwidth as possible and the cleanup below covers the rest.
    files.sort_by_key(|(_, s)| *s);

    let name: String = repo
        .rsplit('/')
        .next()
        .unwrap_or(&repo)
        .chars()
        .map(|c| if "/\\:*?\"<>|".contains(c) { '_' } else { c })
        .collect();
    let dir = root.join("models").join(&name);
    let created = !dir.exists();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let cancel = register_cancel(&name);
    let result = async {
        let client = crate::http::download_client(UA)?;
        let total: u64 = files.iter().map(|(_, s)| *s).sum();
        let mut downloaded: u64 = 0;
        let mut last = std::time::Instant::now();
        let _ = on_progress.send(DownloadProgress::Progress { downloaded, total });
        for (path, size) in &files {
            let url = format!("{base}/{repo}/resolve/main/{path}?download=true");
            let dest = dir.join(path);
            let tmp = dir.join(format!("{path}.part"));
            let mut resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
            if resp.status() == reqwest::StatusCode::FORBIDDEN {
                // Mirrors don't speak the xet protocol — no fallback there,
                // just a clear "switch endpoints" error.
                if !is_official_hf(&base) {
                    return Err(mirror_403_message(resp.status()));
                }
                // CDN block — fetch this file over xet instead (small JSON
                // files are served by hf.co directly and never land here).
                let done_base = downloaded;
                let progress = on_progress.clone();
                xet_fallback_download(
                    &repo,
                    "main",
                    path,
                    &dest,
                    root,
                    move |d, _| {
                        let _ = progress.send(DownloadProgress::Progress {
                            downloaded: done_base + d,
                            total,
                        });
                    },
                    &cancel,
                )
                .await
                .map_err(|e| if e == CANCELLED { e } else { cdn_blocked_message(&e) })?;
                downloaded = done_base + size;
                let _ = on_progress.send(DownloadProgress::Progress { downloaded, total });
                continue;
            }
            if !resp.status().is_success() {
                return Err(format!("下载失败 (download failed): HTTP {} ({path})", resp.status()));
            }
            let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
                if cancel.load(Ordering::SeqCst) {
                    drop(file);
                    let _ = std::fs::remove_file(&tmp);
                    return Err(CANCELLED.into());
                }
                file.write_all(&chunk).map_err(|e| e.to_string())?;
                downloaded += chunk.len() as u64;
                if last.elapsed().as_millis() >= 200 {
                    let _ = on_progress.send(DownloadProgress::Progress { downloaded, total });
                    last = std::time::Instant::now();
                }
            }
            file.flush().ok();
            drop(file);
            std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
        }
        Ok::<String, String>(dir.to_string_lossy().to_string())
    }
    .await;
    clear_cancel(&name);

    match &result {
        Ok(path) => {
            let _ = on_progress.send(DownloadProgress::Done { path: path.clone() });
        }
        Err(e) => {
            // Never leave a half-model behind: it would pass is_mlx_dir and
            // list as loadable. Only folders we created are removed.
            if created {
                let _ = std::fs::remove_dir_all(&dir);
            }
            let _ = on_progress.send(DownloadProgress::Error { message: e.clone() });
        }
    }
    result
}

/// Stream `url` into `models/[subdir/]<filename>`, reporting progress on
/// `on_progress`. Writes to a `.part` file first and renames on success.
/// `subdir` (one path segment) is the folder layout used for vision models —
/// the main GGUF and its mmproj land side by side in their own folder.
#[tauri::command]
pub async fn download_model(
    app: tauri::AppHandle,
    url: String,
    filename: String,
    subdir: Option<String>,
    on_progress: Channel<DownloadProgress>,
) -> Result<(), String> {
    let root = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let cancel = register_cancel(&filename);
    let result =
        download_inner(&root, &url, &filename, subdir.as_deref(), &on_progress, &cancel).await;
    clear_cancel(&filename);
    if let Err(ref e) = result {
        let _ = on_progress.send(DownloadProgress::Error { message: e.clone() });
    }
    result
}

async fn download_inner(
    root: &std::path::Path,
    url: &str,
    filename: &str,
    subdir: Option<&str>,
    on_progress: &Channel<DownloadProgress>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let sanitize = |s: &str| -> String {
        s.chars()
            .map(|c| if "/\\:*?\"<>|".contains(c) { '_' } else { c })
            .collect()
    };
    let mut dir = root.join("models");
    if let Some(sub) = subdir {
        let sub = sanitize(sub.trim());
        if !sub.is_empty() && sub != "." && sub != ".." {
            dir = dir.join(sub);
        }
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let safe: String = sanitize(filename);
    if !safe.to_lowercase().ends_with(".gguf") {
        return Err("文件名必须以 .gguf 结尾".into());
    }
    let dest = dir.join(&safe);
    let tmp = dir.join(format!("{safe}.part"));

    let client = crate::http::download_client(UA)?;
    let mut resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if resp.status() == reqwest::StatusCode::FORBIDDEN {
        // Network-level CDN block (cas-bridge 403) — retry over xet. Only
        // official resolve URLs parse here; mirror endpoints (which don't
        // speak xet) get a clear "switch endpoints" error instead.
        if let Some((repo, revision, path)) = parse_hf_resolve_url(url) {
            let progress = on_progress.clone();
            xet_fallback_download(
                &repo,
                &revision,
                &path,
                &dest,
                root,
                move |downloaded, total| {
                    let _ = progress.send(DownloadProgress::Progress { downloaded, total });
                },
                cancel,
            )
            .await
            .map_err(|e| if e == CANCELLED { e } else { cdn_blocked_message(&e) })?;
            on_progress
                .send(DownloadProgress::Done {
                    path: dest.to_string_lossy().to_string(),
                })
                .ok();
            return Ok(());
        }
        return Err(mirror_403_message(resp.status()));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_file_selection_and_check() {
        let tree: Vec<(String, u64)> = [
            ("config.json", 1200),
            ("model-00001-of-00002.safetensors", 1_000_000),
            ("model-00002-of-00002.safetensors", 900_000),
            ("model.safetensors.index.json", 40_000),
            ("tokenizer.json", 5_000_000),
            ("tokenizer_config.json", 9_000),
            // A multimodal repo's processor configuration. The name list this
            // selection used to be dropped it, and the model then loaded with
            // its vision half disabled.
            ("processor_config.json", 400),
            ("preprocessor_config.json", 500),
            ("README.md", 3_000),          // skipped
            (".gitattributes", 100),       // skipped
            ("images/banner.png", 50_000), // skipped (nested)
        ]
        .iter()
        .map(|(p, s)| (p.to_string(), *s))
        .collect();
        let files = mlx_files(&tree);
        let names: Vec<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
        assert!(names.contains(&"processor_config.json"), "{names:?}");
        assert!(names.contains(&"preprocessor_config.json"), "{names:?}");
        assert!(!names.contains(&"README.md"), "{names:?}");
        assert!(!names.contains(&"images/banner.png"), "{names:?}");
        assert_eq!(files.len(), 8);
        assert!(mlx_repo_check(&files).is_ok());
        // GGUF-style repo (no safetensors) must not pass as MLX.
        let gguf: Vec<(String, u64)> =
            vec![("config.json".into(), 10), ("model.Q4_K_M.gguf".into(), 10)];
        assert!(mlx_repo_check(&mlx_files(&gguf)).is_err());
    }

    #[test]
    fn resolve_url_parsing() {
        assert_eq!(
            parse_hf_resolve_url(
                "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=true"
            ),
            Some(("Qwen/Qwen3-4B-GGUF".into(), "main".into(), "Qwen3-4B-Q4_K_M.gguf".into()))
        );
        // Nested path + no query.
        assert_eq!(
            parse_hf_resolve_url("https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/sub/dir/bge-m3-Q8_0.gguf"),
            Some(("gpustack/bge-m3-GGUF".into(), "main".into(), "sub/dir/bge-m3-Q8_0.gguf".into()))
        );
        // Not HF / not a resolve URL → no fallback.
        assert_eq!(parse_hf_resolve_url("https://example.com/a/b/resolve/main/f.gguf"), None);
        assert_eq!(parse_hf_resolve_url("https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/main/f.gguf"), None);
        assert_eq!(parse_hf_resolve_url("https://huggingface.co/api/models/x/tree/main"), None);
    }

    #[test]
    fn glob_escaping() {
        assert_eq!(glob_escape("model.Q4_K_M.gguf"), "model.Q4_K_M.gguf");
        assert_eq!(glob_escape("a[1]*?.gguf"), "a[[]1[]][*][?].gguf");
    }

    #[test]
    fn hf_base_normalizes_endpoints() {
        // default / empty / whitespace → official
        assert_eq!(hf_base(None), HF_OFFICIAL);
        assert_eq!(hf_base(Some("")), HF_OFFICIAL);
        assert_eq!(hf_base(Some("   ")), HF_OFFICIAL);
        // mirror kept, trailing slash trimmed
        assert_eq!(hf_base(Some("https://hf-mirror.com/")), "https://hf-mirror.com");
        assert_eq!(hf_base(Some("https://hf-mirror.com")), "https://hf-mirror.com");
        // xet gate: only the official endpoint qualifies
        assert!(is_official_hf(&hf_base(None)));
        assert!(!is_official_hf("https://hf-mirror.com"));
        // mirror repo pastes normalize too
        assert_eq!(
            normalize_repo("https://hf-mirror.com/Qwen/Qwen3-4B-GGUF/tree/main"),
            "Qwen/Qwen3-4B-GGUF"
        );
    }

    #[test]
    fn quant_labels_and_grouping() {
        // label extraction
        assert_eq!(quant_label_of("Qwen3-4B-Instruct-Q4_K_M"), "Q4_K_M");
        assert_eq!(quant_label_of("model.IQ4_XS"), "IQ4_XS");
        assert_eq!(quant_label_of("gemma-4-12b-it-qat-Q4_0"), "Q4_0");
        assert_eq!(quant_label_of("llama-f16"), "F16");
        assert_eq!(quant_label_of("weird-model-v2"), "DEFAULT");
        // multipart stripping
        assert_eq!(strip_multipart("m-Q8_0-00001-of-00003"), ("m-Q8_0", true));
        assert_eq!(strip_multipart("m-Q8_0"), ("m-Q8_0", false));

        // grouping: shards summed, mmproj excluded, sorted by size
        let tree: Vec<(String, u64)> = [
            ("a-Q4_K_M.gguf", 400u64),
            ("a-Q8_0-00001-of-00002.gguf", 300),
            ("a-Q8_0-00002-of-00002.gguf", 350),
            ("mmproj-F16.gguf", 90),
            ("README.md", 1),
        ]
        .iter()
        .map(|(p, s)| (p.to_string(), *s))
        .collect();
        let q = gguf_quants(&tree);
        assert_eq!(q.len(), 2);
        assert_eq!(q[0].label, "Q4_K_M");
        assert_eq!(q[0].size, 400);
        assert_eq!(q[1].label, "Q8_0");
        assert_eq!(q[1].size, 650);
        assert_eq!(q[1].files.len(), 2);
        assert!(q[1].files[0].ends_with("00001-of-00002.gguf"));
        // best mmproj picked
        assert_eq!(best_mmproj(&tree).unwrap().0, "mmproj-F16.gguf");
    }

    #[test]
    fn params_guess_from_names() {
        assert_eq!(params_from_name("Qwen/Qwen3-4B-Instruct-GGUF"), Some(4.0));
        assert_eq!(params_from_name("mlx-community/Qwen3-0.6B-4bit"), Some(0.6));
        assert_eq!(params_from_name("org/Qwen3.6-35B-A3B"), Some(35.0));
        assert_eq!(params_from_name("org/model-4bit"), None);
        assert_eq!(params_from_name("org/no-size-here"), None);
    }

    #[test]
    fn repo_normalization() {
        assert_eq!(normalize_repo("mlx-community/Qwen3-4B-4bit"), "mlx-community/Qwen3-4B-4bit");
        assert_eq!(
            normalize_repo("https://huggingface.co/mlx-community/Qwen3-4B-4bit/tree/main"),
            "mlx-community/Qwen3-4B-4bit"
        );
    }

    // -----------------------------------------------------------------------
    // download paths (local HTTP server — no network)
    // -----------------------------------------------------------------------

    fn test_channel() -> (Channel<DownloadProgress>, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let ch = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
            if let tauri::ipc::InvokeResponseBody::Json(j) = body {
                sink.lock().unwrap().push(j);
            }
            Ok(())
        });
        (ch, seen)
    }

    fn fresh_root(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("chaty-dl-test-{tag}-{nanos}"));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// One-shot HTTP server: answers the first request with `status` and `body`.
    fn serve_once(status: &'static str, body: Vec<u8>) -> u16 {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            let hdr = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(hdr.as_bytes());
            let _ = sock.write_all(&body);
        });
        port
    }

    #[tokio::test]
    async fn normal_path_streams_plain_http() {
        let payload: Vec<u8> = b"GGUF"
            .iter()
            .copied()
            .chain((0..50_000u32).flat_map(|i| i.to_le_bytes()))
            .collect();
        let port = serve_once("200 OK", payload.clone());
        let root = fresh_root("ok");
        let (ch, events) = test_channel();
        let cancel = AtomicBool::new(false);
        download_inner(&root, &format!("http://127.0.0.1:{port}/f.gguf"), "f.gguf", None, &ch, &cancel)
            .await
            .unwrap();
        let dest = root.join("models").join("f.gguf");
        assert_eq!(std::fs::read(&dest).unwrap(), payload);
        assert!(!dest.with_extension("gguf.part").exists());
        let evs = events.lock().unwrap();
        assert!(evs.iter().any(|e| e.contains("\"done\"")), "no Done event: {evs:?}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn non_hf_403_reports_plain_error() {
        let port = serve_once("403 Forbidden", Vec::new());
        let root = fresh_root("403");
        let (ch, _events) = test_channel();
        let cancel = AtomicBool::new(false);
        let err = download_inner(&root, &format!("http://127.0.0.1:{port}/f.gguf"), "f.gguf", None, &ch, &cancel)
            .await
            .unwrap_err();
        assert!(err.contains("403"), "unexpected error: {err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    // -----------------------------------------------------------------------
    // real-network e2e (`cargo test download_e2e -- --ignored`)
    //
    // On CDN-blocked networks (cas-bridge 403) these exercise the xet
    // fallback end to end; on open networks the first one exercises the
    // plain reqwest path against real HF. Downloads ~25 MB.
    // -----------------------------------------------------------------------

    const E2E_REPO: &str = "second-state/All-MiniLM-L6-v2-Embedding-GGUF";
    const E2E_FILE: &str = "all-MiniLM-L6-v2-Q4_0.gguf";

    fn assert_gguf(path: &std::path::Path) {
        let bytes = std::fs::read(path).unwrap();
        assert!(bytes.len() > 1_000_000, "file too small: {} bytes", bytes.len());
        assert_eq!(&bytes[..4], b"GGUF", "not a GGUF file");
    }

    #[tokio::test]
    #[ignore]
    async fn download_e2e_model_command_path() {
        let root = fresh_root("e2e-cmd");
        let (ch, events) = test_channel();
        let cancel = AtomicBool::new(false);
        let url = format!("https://huggingface.co/{E2E_REPO}/resolve/main/{E2E_FILE}?download=true");
        download_inner(&root, &url, E2E_FILE, Some("MiniLM"), &ch, &cancel).await.unwrap();
        assert_gguf(&root.join("models").join("MiniLM").join(E2E_FILE));
        let evs = events.lock().unwrap();
        assert!(evs.iter().any(|e| e.contains("\"progress\"")), "no progress events");
        assert!(evs.iter().any(|e| e.contains("\"done\"")), "no Done event");
        // xet tmp dir (if the fallback ran) cleaned up
        let leftovers = std::fs::read_dir(root.join("xet-tmp")).map(|d| d.count()).unwrap_or(0);
        assert_eq!(leftovers, 0, "xet-tmp not cleaned");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn download_e2e_xet_fallback_direct() {
        let root = fresh_root("e2e-xet");
        let dest = root.join(E2E_FILE);
        let seen = Arc::new(AtomicBool::new(false));
        let seen2 = seen.clone();
        xet_fallback_download(
            E2E_REPO,
            "main",
            E2E_FILE,
            &dest,
            &root,
            move |downloaded, total| {
                if downloaded > 0 && total > 0 {
                    seen2.store(true, Ordering::SeqCst);
                }
            },
            &AtomicBool::new(false),
        )
        .await
        .unwrap();
        assert_gguf(&dest);
        assert!(seen.load(Ordering::SeqCst), "no progress callbacks fired");
        // throwaway snapshot dir removed after the rename
        let leftovers = std::fs::read_dir(root.join("xet-tmp")).map(|d| d.count()).unwrap_or(0);
        assert_eq!(leftovers, 0, "xet-tmp not cleaned");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Mixed-path check on CDN-blocked networks: the small JSON/merges files
    /// are served by hf.co directly (plain path) while the ~76 MB safetensors
    /// shard 403s and must arrive via the per-file xet fallback.
    #[tokio::test]
    #[ignore]
    async fn download_e2e_mlx_repo_mixed_paths() {
        let root = fresh_root("e2e-mlx");
        let (ch, events) = test_channel();
        let dir = download_mlx_repo_inner(&root, "mlx-community/SmolLM-135M-Instruct-4bit".into(), None, ch)
            .await
            .unwrap();
        let dir = std::path::PathBuf::from(dir);
        assert!(dir.join("config.json").is_file());
        let shard = std::fs::metadata(dir.join("model.safetensors")).unwrap();
        assert!(shard.len() > 70_000_000, "shard too small: {}", shard.len());
        let evs = events.lock().unwrap();
        assert!(evs.iter().any(|e| e.contains("\"done\"")), "no Done event");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn download_e2e_xet_fallback_cancel() {
        let root = fresh_root("e2e-cancel");
        let dest = root.join(E2E_FILE);
        // Pre-set flag: the poll loop must abort the snapshot at its first tick.
        let cancel = AtomicBool::new(true);
        let err = xet_fallback_download(E2E_REPO, "main", E2E_FILE, &dest, &root, |_, _| {}, &cancel)
            .await
            .unwrap_err();
        assert_eq!(err, CANCELLED);
        assert!(!dest.exists(), "cancelled download left a file behind");
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// Live-network store e2e:
///   cargo test --lib store_e2e -- --ignored --nocapture
#[cfg(test)]
mod store_e2e {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn search_and_detail_both_formats() {
        // GGUF search
        let hits = hf_search("qwen".into(), "gguf".into(), "downloads".into(), Some(10), None)
            .await
            .expect("gguf search");
        assert!(!hits.is_empty(), "no gguf hits");
        assert!(hits.iter().all(|h| h.id.contains('/')));
        eprintln!("gguf top: {} (↓{})", hits[0].id, hits[0].downloads);

        // GGUF detail on a known-stable repo
        let d = hf_model_detail("Qwen/Qwen3-0.6B-GGUF".into(), "gguf".into(), None)
            .await
            .expect("gguf detail");
        assert_eq!(d.format, "gguf");
        assert!(!d.quants.is_empty(), "no quants parsed");
        assert!(d.quants.iter().any(|q| q.label.starts_with('Q')), "no Q-quants: {:?}",
            d.quants.iter().map(|q| q.label.clone()).collect::<Vec<_>>());
        assert!(d.quants.iter().all(|q| q.size > 10_000_000), "sizes look wrong");
        assert!(!d.readme.is_empty(), "readme empty");
        eprintln!("quants: {:?}", d.quants.iter().map(|q| (q.label.clone(), q.size / 1_000_000)).collect::<Vec<_>>());

        // MLX search + detail (auto-detected as single-quant repo)
        let hits = hf_search("qwen".into(), "mlx".into(), "trending".into(), Some(10), None)
            .await
            .expect("mlx search");
        assert!(!hits.is_empty(), "no mlx hits");
        let d = hf_model_detail("mlx-community/Qwen3-0.6B-4bit".into(), "mlx".into(), None)
            .await
            .expect("mlx detail");
        assert_eq!(d.format, "mlx");
        assert_eq!(d.quants.len(), 1);
        assert_eq!(d.quants[0].label, "4BIT");
        assert!(d.quants[0].size > 100_000_000);
        eprintln!("mlx quant: {} {}MB", d.quants[0].label, d.quants[0].size / 1_000_000);

        // Vision repo carries an mmproj
        let d = hf_model_detail("Qwen/Qwen3.5-35B-A3B-GGUF".into(), "gguf".into(), None).await;
        if let Ok(d) = d {
            eprintln!("vision repo: vision={} mmproj={:?}", d.vision, d.mmproj);
        }
    }
}
