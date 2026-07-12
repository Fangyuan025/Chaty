//! Agentic coding tools — the "Code" mode's hands: file read/write/edit, dir
//! listing, glob, content grep, and a sandboxed bash. Everything is confined to
//! a single **workspace** directory the user chooses; paths that try to escape
//! (absolute or `..`) are rejected. On macOS, bash additionally runs under an
//! `sandbox-exec` (seatbelt) profile that only permits writes inside the
//! workspace (+ temp) — a real kernel sandbox. Approval/bypass is enforced by
//! the frontend before these commands are ever invoked.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

/// The active workspace root (absolute, canonicalized). `None` until the user
/// opens a folder for the coding session.
static WORKSPACE: Mutex<Option<PathBuf>> = Mutex::new(None);

const MAX_READ_BYTES: usize = 400 * 1024; // per-file read cap
const MAX_OUTPUT_BYTES: usize = 60 * 1024; // bash stdout/stderr cap (each)
const MAX_GREP_MATCHES: usize = 300;
const MAX_GLOB_HITS: usize = 1000;
const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", "dist", "build", ".venv", "__pycache__"];

fn workspace() -> Result<PathBuf, String> {
    WORKSPACE
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "尚未打开工作区 (no workspace opened)".to_string())
}

/// Resolve `..`/`.` textually (without touching the filesystem, so it works for
/// paths that don't exist yet, e.g. a file about to be created).
fn lexical_normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Resolve a user/model-supplied path against the workspace and guarantee it
/// stays inside it. Relative paths join the root; absolute paths must already be
/// within it. Symlinks that resolve outside are rejected too (for paths that
/// exist).
fn resolve(rel: &str) -> Result<PathBuf, String> {
    if rel.trim().is_empty() {
        return Err("路径为空，请提供文件路径 (empty path — provide a file path)".to_string());
    }
    let root = workspace()?;
    let p = Path::new(rel);
    let joined = if p.is_absolute() { p.to_path_buf() } else { root.join(p) };
    let norm = lexical_normalize(&joined);
    if !norm.starts_with(&root) {
        return Err(format!("路径超出工作区，已拒绝 (path escapes the workspace): {rel}"));
    }
    // If it exists, resolve symlinks and re-check (a symlink could point out).
    if let Ok(canon) = norm.canonicalize() {
        if !canon.starts_with(&root) {
            return Err(format!("路径经符号链接逃逸，已拒绝 (symlink escapes the workspace): {rel}"));
        }
        return Ok(canon);
    }
    Ok(norm)
}

/// Render a path relative to the workspace for display (falls back to the raw
/// path if it somehow isn't under the root).
fn rel_display(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn agent_set_workspace(path: String) -> Result<String, String> {
    let p = PathBuf::from(&path);
    if !p.is_dir() {
        return Err("不是有效的文件夹 (not a directory)".to_string());
    }
    let canon = p.canonicalize().map_err(|e| e.to_string())?;
    let shown = canon.to_string_lossy().to_string();
    let changed = WORKSPACE.lock().unwrap().replace(canon.clone()) != Some(canon);
    if changed {
        // Background jobs, checkpoints and the browser belong to the previous
        // workspace.
        bg_kill_all();
        cp_clear();
        crate::browser::shutdown();
    }
    Ok(shown)
}

#[tauri::command]
pub fn agent_get_workspace() -> Option<String> {
    WORKSPACE.lock().unwrap().as_ref().map(|p| p.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// File tools
// ---------------------------------------------------------------------------

/// Read a text file with line-window paging. Optional 1-based `offset` line
/// and `limit` line count; long files get an actionable footer telling the
/// model exactly which offset continues the read (instead of a blind cut that
/// forced it to guess its way through page after page).
#[tauri::command]
pub fn agent_read_file(
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
    max_chars: Option<usize>,
) -> Result<String, String> {
    const MAX_READ_LINES: usize = 12000; // hard per-call line ceiling
    const MAX_LINE_CHARS: usize = 4000; // pathological single lines (minified JS)

    // The caller (frontend) sizes the budget from the model's ACTUAL context
    // window, so a normal source file fits in ONE call; the default only
    // applies to callers that don't say (tests, older paths). The ceiling
    // matches MAX_READ_BYTES so a full-file diff snapshot never paginates.
    let budget = max_chars.unwrap_or(24_000).clamp(4_000, 400_000);

    let abs = resolve(&path)?;
    let meta = std::fs::metadata(&abs).map_err(|e| format!("读取失败 (read failed): {e}"))?;
    if meta.is_dir() {
        return Err("这是一个目录，请用 list_dir (that's a directory)".to_string());
    }
    let bytes = std::fs::read(&abs).map_err(|e| e.to_string())?;
    let slice = &bytes[..bytes.len().min(MAX_READ_BYTES)];
    let text = String::from_utf8_lossy(slice);

    let all: Vec<&str> = text.lines().collect();
    let total = all.len();
    let start = offset.unwrap_or(1).max(1) - 1;
    if start >= total && total > 0 {
        return Ok(format!(
            "(offset 超出范围:文件共 {total} 行 / offset beyond EOF: file has {total} lines)"
        ));
    }
    let want = limit.unwrap_or(MAX_READ_LINES).clamp(1, MAX_READ_LINES);

    let mut out = String::new();
    let mut end = start; // exclusive
    for (i, line) in all.iter().enumerate().skip(start).take(want) {
        let line: &str = if line.len() > MAX_LINE_CHARS {
            let mut cut = MAX_LINE_CHARS;
            while !line.is_char_boundary(cut) {
                cut -= 1;
            }
            &line[..cut]
        } else {
            line
        };
        if !out.is_empty() && out.len() + line.len() + 1 > budget {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
        end = i + 1;
    }

    if end < total {
        out.push_str(&format!(
            "\n\n[文件共 {total} 行,本次显示第 {}-{end} 行;继续阅读请用 offset={} (file has {total} lines, shown {}-{end}; continue with offset={})]",
            start + 1,
            end + 1,
            start + 1,
            end + 1,
        ));
    }
    Ok(out)
}

/// Download a URL into the workspace (images, archives, any file). Sandboxed
/// through the same `resolve` as every other write, and journaled so rewind
/// removes it like any file the agent created.
#[tauri::command]
pub async fn agent_web_download(url: String, path: String) -> Result<String, String> {
    const CAP: usize = 100 * 1024 * 1024;
    let abs = resolve(&path)?;
    if abs.is_dir() {
        return Err(format!("目标是一个目录 (target is a directory): {path}"));
    }
    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15")
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(url.trim()).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("?")
        .split(';')
        .next()
        .unwrap_or("?")
        .to_string();
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    if bytes.len() > CAP {
        return Err(format!("文件过大 ({} MB),上限 100 MB", bytes.len() / 1024 / 1024));
    }
    cp_record(&abs);
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&abs, &bytes).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    let root = workspace()?;
    Ok(format!(
        "已下载 {} ({} 字节, {ctype})",
        rel_display(&root, &abs),
        bytes.len()
    ))
}

#[tauri::command]
pub fn agent_write_file(path: String, content: String) -> Result<String, String> {
    let abs = resolve(&path)?;
    if abs.is_dir() {
        return Err(format!(
            "目标是一个目录，不能写入；请提供文件路径 (target is a directory, give a file path): {path}"
        ));
    }
    cp_record(&abs);
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&abs, content.as_bytes()).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    let root = workspace()?;
    Ok(format!("已写入 {} ({} 字节)", rel_display(&root, &abs), content.len()))
}

/// Exact-string edit (like a str-replace). `old_string` must appear exactly once
/// unless `replace_all` is set.
#[tauri::command]
pub fn agent_edit_file(
    path: String,
    old_string: String,
    new_string: String,
    replace_all: Option<bool>,
) -> Result<String, String> {
    if old_string == new_string {
        return Err("old_string 与 new_string 相同 (no-op edit)".to_string());
    }
    let abs = resolve(&path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| format!("读取失败 (read failed): {e}"))?;
    cp_record(&abs);
    let count = text.matches(&old_string).count();
    if count == 0 {
        return Err(not_found_error(&text, &old_string));
    }
    let all = replace_all.unwrap_or(false);
    if count > 1 && !all {
        return Err(format!(
            "old_string 出现 {count} 次，不唯一；请提供更多上下文或用 replace_all (not unique: {count} matches)"
        ));
    }
    let pos = text.find(&old_string).unwrap_or(0);
    let start_line = text[..pos].matches('\n').count();
    let updated = if all {
        text.replace(&old_string, &new_string)
    } else {
        text.replacen(&old_string, &new_string, 1)
    };
    std::fs::write(&abs, updated.as_bytes()).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    let root = workspace()?;
    // Echo the edited neighborhood back so the model can confirm the result
    // without spending another read_file step.
    let span = new_string.matches('\n').count() + 1;
    Ok(format!(
        "已编辑 {}（替换 {} 处）。修改后该处内容:\n{}",
        rel_display(&root, &abs),
        if all { count } else { 1 },
        numbered_context(&updated, start_line, span)
    ))
}

/// A few numbered lines around [line0, line0+span) — edit confirmations and
/// mismatch hints both use this.
fn numbered_context(text: &str, line0: usize, span: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let s = line0.saturating_sub(3);
    let e = (line0 + span + 3).min(lines.len());
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate().take(e).skip(s) {
        let l: String = line.chars().take(200).collect();
        out.push_str(&format!("{:>5}  {}\n", i + 1, l));
    }
    out
}

/// "Did you mean": when an exact-match edit misses, locate the line most
/// similar to the needle's first meaningful line and show its neighborhood —
/// one glance instead of a full re-read to fix the next attempt.
fn closest_snippet(text: &str, needle: &str) -> Option<String> {
    let target = needle.lines().map(str::trim).find(|l| !l.is_empty())?;
    let t_tokens: std::collections::HashSet<&str> = target.split_whitespace().collect();
    if t_tokens.is_empty() {
        return None;
    }
    let mut best_line = 0usize;
    let mut best_score = 0.0f32;
    for (i, line) in text.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let score = if l == target {
            1.0
        } else if l.contains(target) || target.contains(l) {
            0.9
        } else {
            let l_tokens: std::collections::HashSet<&str> = l.split_whitespace().collect();
            let inter = t_tokens.intersection(&l_tokens).count() as f32;
            let union = t_tokens.union(&l_tokens).count() as f32;
            inter / union.max(1.0)
        };
        if score > best_score {
            best_score = score;
            best_line = i;
        }
    }
    if best_score < 0.34 {
        return None;
    }
    Some(numbered_context(text, best_line, 1))
}

fn not_found_error(text: &str, old_string: &str) -> String {
    let hint = closest_snippet(text, old_string)
        .map(|s| {
            format!("\n文件中最相似的位置 (closest match — copy old_string verbatim from here):\n{s}")
        })
        .unwrap_or_default();
    format!("未找到 old_string（需与文件内容逐字匹配）(old_string not found — must match exactly){hint}")
}

#[derive(serde::Deserialize)]
pub struct EditOp {
    pub old_string: String,
    pub new_string: String,
    #[serde(default)]
    pub replace_all: bool,
}

/// Several exact-match edits to ONE file, applied atomically: every edit is
/// validated against the in-memory result of the previous ones, and the file
/// is only written when all of them land — a failure changes nothing.
#[tauri::command]
pub fn agent_multi_edit(path: String, edits: Vec<EditOp>) -> Result<String, String> {
    if edits.is_empty() {
        return Err("edits 为空 (no edits given)".to_string());
    }
    let abs = resolve(&path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| format!("读取失败 (read failed): {e}"))?;
    let mut cur = text;
    let total = edits.len();
    for (i, e) in edits.iter().enumerate() {
        let n = i + 1;
        if e.old_string.is_empty() {
            return Err(format!("第 {n}/{total} 条 old_string 为空;未应用任何修改 (edit {n} empty — nothing changed)"));
        }
        if e.old_string == e.new_string {
            return Err(format!("第 {n}/{total} 条 old_string 与 new_string 相同;未应用任何修改 (edit {n} is a no-op — nothing changed)"));
        }
        let count = cur.matches(&e.old_string).count();
        if count == 0 {
            return Err(format!(
                "第 {n}/{total} 条编辑失败,整个 multi_edit 原子回退、文件未改动 (edit {n} failed — atomic, nothing changed):\n{}",
                not_found_error(&cur, &e.old_string)
            ));
        }
        if count > 1 && !e.replace_all {
            return Err(format!(
                "第 {n}/{total} 条 old_string 出现 {count} 次,不唯一;文件未改动 (edit {n} not unique: {count} matches — nothing changed)"
            ));
        }
        cur = if e.replace_all {
            cur.replace(&e.old_string, &e.new_string)
        } else {
            cur.replacen(&e.old_string, &e.new_string, 1)
        };
    }
    cp_record(&abs);
    std::fs::write(&abs, cur.as_bytes()).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    let root = workspace()?;
    Ok(format!("已编辑 {}(应用全部 {total} 处修改)", rel_display(&root, &abs)))
}

// ---- Browser automation tools (CDP; see browser.rs) ----

#[tauri::command]
pub async fn browser_navigate(url: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::navigate(&url))
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))?
}

/// Full-page screenshot (auto-scrolls to trigger lazy content). Returns a temp
/// PNG path the agent loop attaches to the model's next turn (like view_image).
#[tauri::command]
pub async fn browser_screenshot() -> Result<String, String> {
    let png = tokio::task::spawn_blocking(crate::browser::screenshot)
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))??;
    write_shot(png)
}

/// Snapshot of just the current viewport (immediate) — for lazy-load pages,
/// after scrolling. Returns a temp PNG path attached to the next turn.
#[tauri::command]
pub async fn browser_snapshot() -> Result<String, String> {
    let png = tokio::task::spawn_blocking(crate::browser::snapshot)
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))??;
    write_shot(png)
}

/// Scroll the page (to "bottom"/"top" or by pixels) to trigger lazy loading.
#[tauri::command]
pub async fn browser_scroll(to: Option<String>, by: Option<f64>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::scroll_page(to, by))
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))?
}

fn write_shot(png: Vec<u8>) -> Result<String, String> {
    let path = std::env::temp_dir().join(format!(
        "chaty-browser-shot-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&path, png).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn browser_eval(expression: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::eval(&expression))
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))?
}

#[tauri::command]
pub async fn browser_click(selector: Option<String>, text: Option<String>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::click(selector, text))
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))?
}

#[tauri::command]
pub async fn browser_type(
    selector: Option<String>,
    label: Option<String>,
    text: String,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::type_text(selector, label, text))
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))?
}

#[tauri::command]
pub async fn browser_console() -> Result<String, String> {
    tokio::task::spawn_blocking(crate::browser::console)
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))?
}

/// Digest of the current page's interactive elements (links/buttons/inputs).
#[tauri::command]
pub async fn browser_read() -> Result<String, String> {
    tokio::task::spawn_blocking(crate::browser::read_page)
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))?
}

/// Close the automation browser the agent has been driving.
#[tauri::command]
pub async fn browser_close() -> Result<String, String> {
    tokio::task::spawn_blocking(|| {
        crate::browser::shutdown();
        "已关闭浏览器 (browser closed)".to_string()
    })
    .await
    .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))
}

/// Settings → Code: run the agent's browser hidden (headless). Applies the
/// next time the browser starts.
#[tauri::command]
pub fn browser_set_headless(on: bool) {
    crate::browser::set_headless(on);
}

/// Result of rendering a Canvas HTML string headlessly: a screenshot path + the
/// page's console output/errors — the model's "visual + console" pair.
#[derive(serde::Serialize)]
pub struct CanvasCapture {
    pub image: String,
    pub console: String,
}

/// Render a self-contained HTML string in a dedicated HEADLESS browser (never
/// pops a window) and return both a screenshot path and the page's console —
/// the Canvas "see the current page + its errors" path. Writes the PNG under
/// the app cache dir, not the workspace.
#[tauri::command]
pub async fn browser_render_html(app: tauri::AppHandle, html: String) -> Result<CanvasCapture, String> {
    use tauri::Manager;
    let dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("canvas-shots");
    std::fs::create_dir_all(&dir).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    let html_path = dir.join("canvas.html");
    std::fs::write(&html_path, html).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    let url = format!("file://{}", html_path.display());
    let (png, console) = tokio::task::spawn_blocking(move || crate::browser::capture_headless(&url))
        .await
        .map_err(|e| format!("浏览器任务异常 (browser task failed): {e}"))??;
    let png_path = dir.join("canvas-shot.png");
    std::fs::write(&png_path, png).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    Ok(CanvasCapture { image: png_path.to_string_lossy().to_string(), console })
}

/// Resolve a workspace-relative image path to a confined absolute path for the
/// `view_image` tool. Enforces the same sandbox as every other file tool, and
/// checks the file exists and is a supported image — so the vision model only
/// ever sees images from inside the workspace.
#[tauri::command]
pub fn agent_resolve_image(path: String) -> Result<String, String> {
    let abs = resolve(&path)?;
    let ext = abs
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif") {
        return Err(format!(
            "不是支持的图片格式 (not a supported image): {path} — 支持 png/jpg/jpeg/webp/bmp/gif"
        ));
    }
    if !abs.is_file() {
        return Err(format!("图片不存在 (image not found): {path}"));
    }
    Ok(abs.to_string_lossy().to_string())
}

/// File outline: the definition lines (functions/classes/structs/…) with line
/// numbers, so the model can navigate a big file without reading it whole.
/// Regex-free keyword heuristics that cover Rust/TS/JS/Python/Go/Swift/etc.
#[tauri::command]
pub fn agent_outline(path: String) -> Result<String, String> {
    let abs = resolve(&path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| format!("读取失败 (read failed): {e}"))?;
    let mut out = String::new();
    let mut n = 0;
    for (i, line) in text.lines().enumerate() {
        if !is_symbol_line(line.trim_start()) {
            continue;
        }
        let sig: String = line.trim_end().chars().take(160).collect();
        out.push_str(&format!("{:>5}  {sig}\n", i + 1));
        n += 1;
        if n >= 300 {
            out.push_str("… (更多定义已省略 / more omitted)\n");
            break;
        }
    }
    if out.is_empty() {
        return Ok("(未识别到符号定义 — 用 read_file 直接查看 / no definitions recognized)".to_string());
    }
    Ok(out)
}

fn is_symbol_line(t: &str) -> bool {
    let t = t
        .trim_start_matches("export ")
        .trim_start_matches("default ")
        .trim_start_matches("declare ")
        .trim_start_matches("pub(crate) ")
        .trim_start_matches("pub ")
        .trim_start_matches("unsafe ")
        .trim_start_matches("async ")
        .trim_start_matches("static ")
        .trim_start_matches("abstract ")
        .trim_start_matches("public ")
        .trim_start_matches("private ")
        .trim_start_matches("protected ");
    const KEYWORDS: [&str; 13] = [
        "fn ", "def ", "class ", "struct ", "enum ", "trait ", "impl ", "interface ", "type ",
        "function ", "func ", "mod ", "macro_rules!",
    ];
    if KEYWORDS.iter().any(|k| t.starts_with(k)) {
        return true;
    }
    // JS/TS arrow-function or function-expression bindings.
    (t.starts_with("const ") || t.starts_with("let ") || t.starts_with("var "))
        && (t.contains("=>") || t.contains("function"))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// One level of directory listing (directories first, then files, sorted).
#[tauri::command]
pub fn agent_list_dir(path: Option<String>) -> Result<Vec<DirEntry>, String> {
    let abs = match path {
        Some(p) if !p.trim().is_empty() && p != "." => resolve(&p)?,
        _ => workspace()?,
    };
    let rd = std::fs::read_dir(&abs).map_err(|e| format!("列目录失败 (list failed): {e}"))?;
    let mut out: Vec<DirEntry> = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
        out.push(DirEntry { name, is_dir, size });
    }
    out.sort_by(|a, b| match (a.is_dir, b.is_dir) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });
    Ok(out)
}

/// Glob files by pattern (relative to the workspace, e.g. `src/**/*.rs`).
#[tauri::command]
pub fn agent_glob(pattern: String) -> Result<Vec<String>, String> {
    let root = workspace()?;
    let full = root.join(&pattern);
    let full = full.to_str().ok_or_else(|| "无效的模式 (invalid pattern)".to_string())?;
    let mut hits: Vec<String> = Vec::new();
    for entry in glob::glob(full).map_err(|e| e.to_string())?.flatten() {
        if entry.is_file() {
            hits.push(rel_display(&root, &entry));
            if hits.len() >= MAX_GLOB_HITS {
                break;
            }
        }
    }
    hits.sort();
    Ok(hits)
}

/// Fast filename listing for the composer's @-mention picker: walks the
/// workspace (skipping VCS/build dirs and hidden files), optionally filtering
/// by a case-insensitive substring of the relative path, capped for UI use.
#[tauri::command]
pub fn agent_list_files(query: Option<String>, limit: Option<usize>) -> Result<Vec<String>, String> {
    let root = workspace()?;
    let q = query.unwrap_or_default().to_lowercase();
    let cap = limit.unwrap_or(200).min(1000);
    let mut hits: Vec<String> = Vec::new();
    for entry in walkdir::WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(name.starts_with('.') && e.depth() > 0)
                && !(e.file_type().is_dir() && SKIP_DIRS.contains(&name.as_ref()))
        })
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = rel_display(&root, entry.path());
        if q.is_empty() || rel.to_lowercase().contains(&q) {
            hits.push(rel);
            if hits.len() >= cap {
                break;
            }
        }
    }
    hits.sort_by_key(|p| (p.len(), p.clone()));
    Ok(hits)
}

// ---------------------------------------------------------------------------
// Checkpoints: a per-turn journal of every file's FIRST-touch original state
// (written by the agent's write/edit tools). Reverting restores the originals
// newest-turn-first, so "rewind to before turn N" is safe. Session-scoped,
// bash side effects are not journaled (same trade-off as Claude Code rewind).
// ---------------------------------------------------------------------------

struct CpEntry {
    path: PathBuf,
    /// The file's bytes before the first agent touch this turn; `None` = the
    /// file did not exist (revert deletes it).
    original: Option<Vec<u8>>,
}
struct Checkpoint {
    id: u64,
    entries: Vec<CpEntry>,
}
static CHECKPOINTS: Mutex<Vec<Checkpoint>> = Mutex::new(Vec::new());
static CP_NEXT_ID: AtomicU64 = AtomicU64::new(1);
const CP_MAX: usize = 40;
const CP_MAX_FILE: u64 = 8 * 1024 * 1024; // don't journal giant files

/// Record `abs`'s current state into the active checkpoint (first touch only).
fn cp_record(abs: &Path) {
    let mut cps = CHECKPOINTS.lock().unwrap();
    let Some(cp) = cps.last_mut() else { return };
    if cp.entries.iter().any(|e| e.path == abs) {
        return;
    }
    if std::fs::metadata(abs).map(|m| m.len() > CP_MAX_FILE).unwrap_or(false) {
        return;
    }
    let original = std::fs::read(abs).ok();
    cp.entries.push(CpEntry { path: abs.to_path_buf(), original });
}

/// Open a new checkpoint for the coming turn; returns its id.
#[tauri::command]
pub fn agent_checkpoint_begin() -> u64 {
    let mut cps = CHECKPOINTS.lock().unwrap();
    let id = CP_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    cps.push(Checkpoint { id, entries: Vec::new() });
    if cps.len() > CP_MAX {
        cps.remove(0);
    }
    id
}

/// Restore the workspace to the state BEFORE checkpoint `id`: every checkpoint
/// with id >= `id` is reverted, newest first.
#[tauri::command]
pub fn agent_checkpoint_revert_to(id: u64) -> Result<String, String> {
    let mut cps = CHECKPOINTS.lock().unwrap();
    let mut restored = 0usize;
    let mut removed = 0usize;
    while cps.last().map(|c| c.id >= id).unwrap_or(false) {
        let cp = cps.pop().unwrap();
        for e in cp.entries.iter().rev() {
            match &e.original {
                Some(bytes) => {
                    if let Some(parent) = e.path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if std::fs::write(&e.path, bytes).is_ok() {
                        restored += 1;
                    }
                }
                None => {
                    if std::fs::remove_file(&e.path).is_ok() {
                        removed += 1;
                    }
                }
            }
        }
    }
    Ok(format!(
        "已回滚：恢复 {restored} 个文件，删除 {removed} 个新建文件 (reverted: {restored} restored, {removed} removed)"
    ))
}

fn cp_clear() {
    CHECKPOINTS.lock().unwrap().clear();
}

// ---------------------------------------------------------------------------
// Ranked code search (BM25 over line-window chunks) — a "which file handles X?"
// tool that beats grep for the model: multi-term, ranked, typo-tolerant-ish.
// ---------------------------------------------------------------------------

const SEARCH_MAX_FILE: u64 = 200 * 1024; // skip huge files
const SEARCH_MAX_TOTAL: usize = 24 * 1024 * 1024; // stop scanning past this much text
const SEARCH_CHUNK_LINES: usize = 30;
const SEARCH_CHUNK_OVERLAP: usize = 8;

/// Lowercased alphanumeric tokens, with camelCase / snake_case split so
/// `getUserName` matches "user name".
fn code_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in s.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        // split camelCase boundaries
        let mut word = String::new();
        let chars: Vec<char> = raw.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            if i > 0 && c.is_uppercase() && chars[i - 1].is_lowercase() {
                if word.len() > 1 {
                    out.push(word.to_lowercase());
                }
                word.clear();
            }
            word.push(c);
        }
        if word.len() > 1 {
            out.push(word.to_lowercase());
        }
    }
    out
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CodeHit {
    pub path: String,
    pub line: usize,
    pub snippet: String,
    pub score: f32,
}

/// BM25-ranked search over the workspace. Chunks are overlapping line windows;
/// results carry the path + start line + snippet.
#[tauri::command]
pub fn agent_search_code(query: String, k: Option<usize>) -> Result<Vec<CodeHit>, String> {
    let root = workspace()?;
    let q_tokens = code_tokens(&query);
    if q_tokens.is_empty() {
        return Err("查询为空 (empty query)".to_string());
    }
    let top_k = k.unwrap_or(8).clamp(1, 30);

    struct Chunk {
        path: String,
        line: usize,
        text: String,
        tf: HashMap<String, u32>,
        len: u32,
    }
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut df: HashMap<String, u32> = HashMap::new();
    let mut scanned = 0usize;

    'walk: for entry in walkdir::WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(name.starts_with('.') && e.depth() > 0)
                && !(e.file_type().is_dir() && SKIP_DIRS.contains(&name.as_ref()))
        })
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.metadata().map(|m| m.len() > SEARCH_MAX_FILE).unwrap_or(true) {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path()) else { continue };
        if bytes.iter().take(512).any(|&b| b == 0) {
            continue; // binary
        }
        let text = String::from_utf8_lossy(&bytes);
        scanned += text.len();
        let rel = rel_display(&root, entry.path());
        let lines: Vec<&str> = text.lines().collect();
        let mut start = 0usize;
        while start < lines.len() {
            let end = (start + SEARCH_CHUNK_LINES).min(lines.len());
            let body = lines[start..end].join("\n");
            let toks = code_tokens(&body);
            if !toks.is_empty() {
                let mut tf: HashMap<String, u32> = HashMap::new();
                for t in &toks {
                    *tf.entry(t.clone()).or_insert(0) += 1;
                }
                for t in tf.keys() {
                    *df.entry(t.clone()).or_insert(0) += 1;
                }
                chunks.push(Chunk {
                    path: rel.clone(),
                    line: start + 1,
                    text: body,
                    len: toks.len() as u32,
                    tf,
                });
            }
            if end == lines.len() {
                break;
            }
            start = end - SEARCH_CHUNK_OVERLAP;
        }
        if scanned > SEARCH_MAX_TOTAL {
            break 'walk;
        }
    }

    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let n = chunks.len() as f32;
    let avg_len: f32 = chunks.iter().map(|c| c.len as f32).sum::<f32>() / n;
    let (k1, b) = (1.4f32, 0.75f32);
    let mut hits: Vec<CodeHit> = chunks
        .iter()
        .filter_map(|c| {
            let mut score = 0f32;
            for t in &q_tokens {
                let Some(&tf) = c.tf.get(t) else { continue };
                let dfi = *df.get(t).unwrap_or(&1) as f32;
                let idf = ((n - dfi + 0.5) / (dfi + 0.5) + 1.0).ln();
                let tf = tf as f32;
                score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * c.len as f32 / avg_len));
            }
            (score > 0.0).then(|| CodeHit {
                path: c.path.clone(),
                line: c.line,
                snippet: c.text.chars().take(700).collect(),
                score,
            })
        })
        .collect();
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    // At most 2 hits per file so one file can't monopolize the results.
    let mut per_file: HashMap<&str, usize> = HashMap::new();
    let mut out = Vec::new();
    for h in &hits {
        let c = per_file.entry(h.path.as_str()).or_insert(0);
        if *c < 2 {
            *c += 1;
            out.push(h.clone());
            if out.len() >= top_k {
                break;
            }
        }
    }
    Ok(out)
}

/// Regex content search over the workspace (skips VCS/build/binary dirs).
#[tauri::command]
pub fn agent_grep(
    pattern: String,
    path: Option<String>,
    glob: Option<String>,
) -> Result<String, String> {
    let root = workspace()?;
    let start = match path {
        Some(p) if !p.trim().is_empty() && p != "." => resolve(&p)?,
        _ => root.clone(),
    };
    let re = regex::Regex::new(&pattern).map_err(|e| format!("正则无效 (bad regex): {e}"))?;
    let glob_matcher = match glob {
        Some(g) if !g.trim().is_empty() => Some(
            glob::Pattern::new(&g).map_err(|e| format!("glob 无效 (bad glob): {e}"))?,
        ),
        _ => None,
    };

    let mut out = String::new();
    let mut n = 0usize;
    'outer: for entry in walkdir::WalkDir::new(&start)
        .into_iter()
        .filter_entry(|e| {
            !(e.file_type().is_dir()
                && e.file_name().to_str().map(|s| SKIP_DIRS.contains(&s)).unwrap_or(false))
        })
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        if let Some(gm) = &glob_matcher {
            let rel = rel_display(&root, p);
            if !gm.matches(&rel) {
                continue;
            }
        }
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(_) => continue,
        };
        // Skip obviously-binary files.
        if bytes.iter().take(8000).any(|&b| b == 0) {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        let rel = rel_display(&root, p);
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) {
                let shown: String = line.chars().take(300).collect();
                out.push_str(&format!("{rel}:{}: {}\n", i + 1, shown.trim_end()));
                n += 1;
                if n >= MAX_GREP_MATCHES {
                    out.push_str("… (更多结果已省略 / more matches omitted)\n");
                    break 'outer;
                }
            }
        }
    }
    if n == 0 {
        out.push_str("(无匹配 / no matches)");
    }
    Ok(out)
}

/// Quick locate: find files whose PATH contains `query`, and (unless
/// `names_only`) lines whose CONTENT contains `query` — literal, case-
/// insensitive, no regex. Fills the gap between `glob` (name PATTERNS) and
/// `grep` (content REGEX): "find anything to do with X" in one call.
#[tauri::command]
pub fn agent_search_files(
    query: String,
    path: Option<String>,
    names_only: Option<bool>,
) -> Result<String, String> {
    const MAX_NAME_HITS: usize = 60;
    let needle = query.trim().to_lowercase();
    if needle.is_empty() {
        return Err("query 为空 (empty query)".to_string());
    }
    let root = workspace()?;
    let start = match path {
        Some(p) if !p.trim().is_empty() && p != "." => resolve(&p)?,
        _ => root.clone(),
    };
    let contents = !names_only.unwrap_or(false);

    let mut name_hits: Vec<String> = Vec::new();
    let mut content_out = String::new();
    let mut content_n = 0usize;
    let mut name_capped = false;

    'walk: for entry in walkdir::WalkDir::new(&start)
        .into_iter()
        .filter_entry(|e| {
            !(e.file_type().is_dir()
                && e.file_name().to_str().map(|s| SKIP_DIRS.contains(&s)).unwrap_or(false))
        })
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let rel = rel_display(&root, p);

        // Name match on the workspace-relative path.
        if name_hits.len() < MAX_NAME_HITS {
            if rel.to_lowercase().contains(&needle) {
                name_hits.push(rel.clone());
            }
        } else {
            name_capped = true;
        }

        if !contents || content_n >= MAX_GREP_MATCHES {
            continue;
        }
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(_) => continue,
        };
        if bytes.iter().take(8000).any(|&b| b == 0) {
            continue; // binary
        }
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            if line.to_lowercase().contains(&needle) {
                let shown: String = line.chars().take(300).collect();
                content_out.push_str(&format!("{rel}:{}: {}\n", i + 1, shown.trim_end()));
                content_n += 1;
                if content_n >= MAX_GREP_MATCHES {
                    content_out.push_str("… (更多结果已省略 / more matches omitted)\n");
                    break 'walk;
                }
            }
        }
    }

    if name_hits.is_empty() && content_n == 0 {
        return Ok("(无匹配 / no matches)".to_string());
    }

    let mut out = String::new();
    if !name_hits.is_empty() {
        out.push_str(&format!("文件名匹配 (file names, {} 个):\n", name_hits.len()));
        for h in &name_hits {
            out.push_str(&format!("  {h}\n"));
        }
        if name_capped {
            out.push_str("  … (更多文件名已省略 / more names omitted)\n");
        }
    }
    if contents {
        out.push_str(&format!(
            "\n内容匹配 (file contents{}):\n{}",
            if content_n == 0 { ", 无 / none" } else { "" },
            content_out
        ));
    }
    Ok(out.trim_end().to_string())
}

// ---------------------------------------------------------------------------
// Sandboxed bash
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BashResult {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
    pub timed_out: bool,
}

/// macOS seatbelt profile: allow everything by default, then deny all writes and
/// re-allow only inside the workspace and the standard temp dirs. Reads, exec
/// and network stay available (agents need `git`, `npm`, compilers).
#[cfg(target_os = "macos")]
fn seatbelt_profile(root: &Path) -> String {
    format!(
        "(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write* \
         (subpath \"{root}\") (subpath \"/private/tmp\") (subpath \"/tmp\") \
         (subpath \"/private/var/folders\") (subpath \"/var/folders\") \
         (literal \"/dev/null\") (literal \"/dev/stdout\") (literal \"/dev/stderr\") \
         (literal \"/dev/dtracehelper\") (literal \"/dev/tty\"))",
        root = root.display()
    )
}

/// A Finder-launched GUI app inherits a minimal PATH (`/usr/bin:/bin:…`) that
/// misses Homebrew, cargo, nvm, … — so `npm`, `node`, `python3` from user
/// installs "don't exist" inside agent shells. Build a PATH that prepends the
/// common tool locations (only those that exist) to whatever we inherited.
#[cfg(unix)]
fn augmented_path() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut dirs: Vec<String> = vec![
        "/opt/homebrew/bin".into(),
        "/opt/homebrew/sbin".into(),
        "/usr/local/bin".into(),
        format!("{home}/.cargo/bin"),
        format!("{home}/.local/bin"),
        format!("{home}/.bun/bin"),
        format!("{home}/.deno/bin"),
        format!("{home}/go/bin"),
    ];
    // nvm keeps node under versioned dirs — add the newest one if present.
    let nvm = PathBuf::from(format!("{home}/.nvm/versions/node"));
    if let Ok(entries) = std::fs::read_dir(&nvm) {
        let mut versions: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        versions.sort();
        if let Some(latest) = versions.last() {
            dirs.push(latest.join("bin").to_string_lossy().to_string());
        }
    }
    let mut path: Vec<String> = dirs.into_iter().filter(|d| Path::new(d).is_dir()).collect();
    if let Ok(cur) = std::env::var("PATH") {
        for p in cur.split(':') {
            if !path.iter().any(|x| x == p) {
                path.push(p.to_string());
            }
        }
    }
    path.join(":")
}

fn read_capped(mut r: impl Read + Send + 'static) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = r.read_to_end(&mut buf);
        buf
    })
}

fn cap_utf8(bytes: Vec<u8>) -> String {
    let truncated = bytes.len() > MAX_OUTPUT_BYTES;
    let s = String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_OUTPUT_BYTES)]).into_owned();
    if truncated {
        format!("{s}\n… (输出已截断 / output truncated)")
    } else {
        s
    }
}

fn run_bash(root: &Path, command: &str, timeout: Duration) -> Result<BashResult, String> {
    let mut cmd = build_command(root, command);
    cmd.current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("启动命令失败 (spawn failed): {e}"))?;
    let out_h = read_capped(child.stdout.take().unwrap());
    let err_h = read_capped(child.stderr.take().unwrap());

    let started = Instant::now();
    let (code, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status.code().unwrap_or(-1), false),
            Ok(None) => {
                if started.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break (-1, true);
                }
                std::thread::sleep(Duration::from_millis(40));
            }
            Err(e) => return Err(format!("等待命令失败 (wait failed): {e}")),
        }
    };
    let stdout = cap_utf8(out_h.join().unwrap_or_default());
    let stderr = cap_utf8(err_h.join().unwrap_or_default());
    Ok(BashResult { stdout, stderr, code, timed_out })
}

#[cfg(target_os = "macos")]
fn build_command(root: &Path, command: &str) -> Command {
    let mut cmd = Command::new("/usr/bin/sandbox-exec");
    cmd.arg("-p").arg(seatbelt_profile(root)).arg("/bin/sh").arg("-c").arg(command);
    cmd.env("PATH", augmented_path());
    cmd
}

#[cfg(all(unix, not(target_os = "macos")))]
fn build_command(_root: &Path, command: &str) -> Command {
    // No seatbelt off macOS — confinement is the working directory + approval.
    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c").arg(command);
    cmd.env("PATH", augmented_path());
    cmd
}

#[cfg(windows)]
fn build_command(_root: &Path, command: &str) -> Command {
    let mut cmd = Command::new("cmd");
    cmd.arg("/C").arg(command);
    cmd
}

/// Run a shell command inside the workspace. On macOS it is sandboxed (writes
/// confined to the workspace); elsewhere it runs in the workspace dir. The
/// frontend gates this behind per-command approval (or bypass mode).
#[tauri::command]
pub async fn agent_bash(command: String, timeout_secs: Option<u64>) -> Result<BashResult, String> {
    let root = workspace()?;
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(120).clamp(1, 600));
    tokio::task::spawn_blocking(move || run_bash(&root, &command, timeout))
        .await
        .map_err(|e| format!("命令任务异常 (task panicked): {e}"))?
}

// ---------------------------------------------------------------------------
// Background commands (dev servers, long builds): start now, get told later.
// ---------------------------------------------------------------------------

struct BgJob {
    command: String,
    started: Instant,
    /// Live, capped, merged stdout+stderr.
    output: Arc<Mutex<Vec<u8>>>,
    /// Exit code once the process ends (`None` while running).
    code: Option<i32>,
    /// The finished result was already handed to the agent loop.
    reported: bool,
    /// For `bg_kill`: the child's pid (the sandbox wrapper's process group).
    pid: u32,
}

static BG_JOBS: Mutex<Option<HashMap<u64, BgJob>>> = Mutex::new(None);
static BG_NEXT_ID: AtomicU64 = AtomicU64::new(1);
const BG_MAX_JOBS: usize = 8;
const BG_TAIL_BYTES: usize = 8 * 1024;

fn bg_tail(buf: &Arc<Mutex<Vec<u8>>>) -> String {
    let b = buf.lock().unwrap();
    let start = b.len().saturating_sub(BG_TAIL_BYTES);
    String::from_utf8_lossy(&b[start..]).into_owned()
}

fn bg_append(buf: &Arc<Mutex<Vec<u8>>>, chunk: &[u8]) {
    let mut b = buf.lock().unwrap();
    b.extend_from_slice(chunk);
    // Keep memory bounded: retain only the most recent window.
    let cap = MAX_OUTPUT_BYTES;
    if b.len() > cap {
        let cut = b.len() - cap;
        b.drain(..cut);
    }
}

fn bg_stream(mut r: impl Read + Send + 'static, buf: Arc<Mutex<Vec<u8>>>) {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        while let Ok(n) = r.read(&mut chunk) {
            if n == 0 {
                break;
            }
            bg_append(&buf, &chunk[..n]);
        }
    });
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BgInfo {
    pub id: u64,
    pub command: String,
    pub running: bool,
    pub code: Option<i32>,
    pub elapsed_secs: u64,
    pub tail: String,
}

/// Start a background command (same sandbox/PATH as `agent_bash`). Returns an
/// id immediately; the frontend loop polls `agent_bg_reap` and tells the model
/// when it finishes.
#[tauri::command]
pub fn agent_bash_bg(command: String) -> Result<u64, String> {
    let root = workspace()?;
    let mut reg = BG_JOBS.lock().unwrap();
    let jobs = reg.get_or_insert_with(HashMap::new);
    let running = jobs.values().filter(|j| j.code.is_none()).count();
    if running >= BG_MAX_JOBS {
        return Err(format!(
            "后台命令过多（{running} 个在跑），请先用 bg_kill 结束一些 (too many background jobs)"
        ));
    }

    let mut cmd = build_command(&root, &command);
    cmd.current_dir(&root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    // Own process group so bg_kill can take down the whole tree (npm → node …).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd.spawn().map_err(|e| format!("启动命令失败 (spawn failed): {e}"))?;
    let pid = child.id();
    let output: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    bg_stream(child.stdout.take().unwrap(), output.clone());
    bg_stream(child.stderr.take().unwrap(), output.clone());

    let id = BG_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    jobs.insert(
        id,
        BgJob { command, started: Instant::now(), output, code: None, reported: false, pid },
    );
    drop(reg);

    // Monitor thread: record the exit code when the process ends.
    std::thread::spawn(move || {
        let status = child.wait();
        let code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
        if let Some(jobs) = BG_JOBS.lock().unwrap().as_mut() {
            if let Some(job) = jobs.get_mut(&id) {
                job.code = Some(code);
            }
        }
    });
    Ok(id)
}

/// Snapshot of one background job (running or finished).
#[tauri::command]
pub fn agent_bg_output(id: u64) -> Result<BgInfo, String> {
    let reg = BG_JOBS.lock().unwrap();
    let job = reg
        .as_ref()
        .and_then(|j| j.get(&id))
        .ok_or_else(|| format!("没有这个后台命令 (no such background job): #{id}"))?;
    Ok(BgInfo {
        id,
        command: job.command.clone(),
        running: job.code.is_none(),
        code: job.code,
        elapsed_secs: job.started.elapsed().as_secs(),
        tail: bg_tail(&job.output),
    })
}

/// Kill a background job (SIGKILL to its process group on unix).
#[tauri::command]
pub fn agent_bg_kill(id: u64) -> Result<String, String> {
    let mut reg = BG_JOBS.lock().unwrap();
    let job = reg
        .as_mut()
        .and_then(|j| j.get_mut(&id))
        .ok_or_else(|| format!("没有这个后台命令 (no such background job): #{id}"))?;
    if job.code.is_none() {
        bg_kill_pid(job.pid);
        job.reported = true; // killed on request → no completion notice needed
        return Ok(format!("已终止后台命令 #{id} (killed)"));
    }
    Ok(format!("后台命令 #{id} 已经结束 (already finished)"))
}

/// Kill a job's whole process group (unix) / tree (windows).
fn bg_kill_pid(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output();
    }
}

/// All currently RUNNING background jobs (for the UI's indicator).
#[tauri::command]
pub fn agent_bg_list() -> Vec<BgInfo> {
    let reg = BG_JOBS.lock().unwrap();
    let Some(jobs) = reg.as_ref() else { return Vec::new() };
    let mut out: Vec<BgInfo> = jobs
        .iter()
        .filter(|(_, j)| j.code.is_none())
        .map(|(id, j)| BgInfo {
            id: *id,
            command: j.command.clone(),
            running: true,
            code: None,
            elapsed_secs: j.started.elapsed().as_secs(),
            tail: String::new(), // the indicator doesn't need output
        })
        .collect();
    out.sort_by_key(|j| j.id);
    out
}

/// Finished-but-unreported jobs → hand them to the agent loop exactly once.
#[tauri::command]
pub fn agent_bg_reap() -> Vec<BgInfo> {
    let mut out = Vec::new();
    if let Some(jobs) = BG_JOBS.lock().unwrap().as_mut() {
        for (id, job) in jobs.iter_mut() {
            if job.code.is_some() && !job.reported {
                job.reported = true;
                out.push(BgInfo {
                    id: *id,
                    command: job.command.clone(),
                    running: false,
                    code: job.code,
                    elapsed_secs: job.started.elapsed().as_secs(),
                    tail: bg_tail(&job.output),
                });
            }
        }
    }
    out
}

/// Kill every background job (workspace switch / app teardown).
pub fn bg_kill_all() {
    if let Some(jobs) = BG_JOBS.lock().unwrap().as_mut() {
        for job in jobs.values() {
            if job.code.is_none() {
                bg_kill_pid(job.pid);
            }
        }
        jobs.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn set_ws(dir: &Path) {
        *WORKSPACE.lock().unwrap() = Some(dir.canonicalize().unwrap());
    }

    /// Tests share the global WORKSPACE, so they must not run concurrently.
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn search_files_matches_names_and_contents() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-sf-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        set_ws(&tmp);
        std::fs::write(tmp.join("src/auth_service.ts"), "export function login() {}\n").unwrap();
        std::fs::write(tmp.join("src/util.ts"), "// handles AUTH token refresh\nconst x = 1;\n").unwrap();
        std::fs::write(tmp.join("readme.md"), "no match here\n").unwrap();

        // Case-insensitive, matches BOTH the filename and the content line.
        let out = agent_search_files("auth".into(), None, None).unwrap();
        assert!(out.contains("src/auth_service.ts"), "name hit missing: {out}");
        assert!(out.contains("src/util.ts:1:"), "content hit missing: {out}");
        assert!(out.contains("AUTH token"), "content line missing: {out}");
        assert!(!out.contains("readme.md"), "non-match leaked: {out}");

        // names_only skips the content scan.
        let names = agent_search_files("auth".into(), Some(".".into()), Some(true)).unwrap();
        assert!(names.contains("src/auth_service.ts"));
        assert!(!names.contains("src/util.ts:1:"), "names_only should not scan content: {names}");

        // No match → clear message.
        assert!(agent_search_files("zzznope".into(), None, None).unwrap().contains("无匹配"));
        // Empty query → error.
        assert!(agent_search_files("  ".into(), None, None).is_err());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn multi_edit_is_atomic() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-me-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        std::fs::write(tmp.join("f.txt"), "alpha\nbeta\ngamma\n").unwrap();

        // Second edit misses → nothing at all changes.
        let err = agent_multi_edit(
            "f.txt".into(),
            vec![
                EditOp { old_string: "alpha".into(), new_string: "ALPHA".into(), replace_all: false },
                EditOp { old_string: "nope".into(), new_string: "x".into(), replace_all: false },
            ],
        )
        .unwrap_err();
        assert!(err.contains("2/2"), "err should name the failing edit: {err}");
        assert_eq!(std::fs::read_to_string(tmp.join("f.txt")).unwrap(), "alpha\nbeta\ngamma\n");

        // All match → all applied, in order, later edits see earlier results.
        let ok = agent_multi_edit(
            "f.txt".into(),
            vec![
                EditOp { old_string: "alpha".into(), new_string: "one".into(), replace_all: false },
                EditOp { old_string: "one\nbeta".into(), new_string: "one\nTWO".into(), replace_all: false },
            ],
        )
        .unwrap();
        assert!(ok.contains("2"), "{ok}");
        assert_eq!(std::fs::read_to_string(tmp.join("f.txt")).unwrap(), "one\nTWO\ngamma\n");
    }

    #[test]
    fn edit_miss_suggests_closest_line() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-cs-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        std::fs::write(
            tmp.join("g.py"),
            "def add(a, b):\n    return a + b\n\ndef total(items, tax_rate):\n    return sum(items) * (1 + tax_rate)\n",
        )
        .unwrap();
        // Model misremembered the signature — the error should point at the
        // real line so the next attempt can copy it verbatim.
        let err = agent_edit_file(
            "g.py".into(),
            "def total(items, tax):".into(),
            "def total(items, tax, discount):".into(),
            None,
        )
        .unwrap_err();
        assert!(err.contains("def total(items, tax_rate):"), "hint missing: {err}");
        assert!(err.contains("4  "), "line number missing: {err}");
    }

    #[test]
    fn edit_success_echoes_context() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-ec-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        std::fs::write(tmp.join("h.txt"), "l1\nl2\nl3\nl4\nl5\nl6\nl7\n").unwrap();
        let ok = agent_edit_file("h.txt".into(), "l4".into(), "L4-new".into(), None).unwrap();
        assert!(ok.contains("L4-new"), "{ok}");
        assert!(ok.contains("l2") && ok.contains("l7"), "context window wrong: {ok}");
    }

    #[test]
    fn outline_finds_definitions() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-ol-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        std::fs::write(
            tmp.join("mix.ts"),
            "import x from 'y';\n\nexport function parse(s: string) {}\nconst helper = (a: number) => a * 2;\nclass Lexer {\n  private pos = 0;\n  advance() {}\n}\nexport const RE = /x/;\n",
        )
        .unwrap();
        let o = agent_outline("mix.ts".into()).unwrap();
        assert!(o.contains("export function parse"), "{o}");
        assert!(o.contains("const helper"), "{o}");
        assert!(o.contains("class Lexer"), "{o}");
        assert!(!o.contains("import x"), "imports are not symbols: {o}");
        assert!(!o.contains("export const RE"), "non-function const is not a symbol: {o}");
        // rust-ish
        std::fs::write(tmp.join("m.rs"), "use std::fmt;\n\npub(crate) async fn run() {}\nstruct Cfg;\nimpl Cfg {\n    fn new() -> Self { Cfg }\n}\n").unwrap();
        let o = agent_outline("m.rs".into()).unwrap();
        assert!(o.contains("pub(crate) async fn run"), "{o}");
        assert!(o.contains("struct Cfg"), "{o}");
        assert!(o.contains("fn new"), "{o}");
    }

    #[test]
    fn resolve_confines_paths() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        // inside is fine
        assert!(resolve("a/b.txt").is_ok());
        // parent traversal escapes
        assert!(resolve("../evil.txt").is_err());
        assert!(resolve("a/../../evil.txt").is_err());
        // absolute outside escapes
        assert!(resolve("/etc/passwd").is_err());
        // empty / blank paths are rejected with a clear error (a missing "path"
        // arg used to resolve to the workspace root itself → EISDIR on write)
        assert!(resolve("").is_err());
        assert!(resolve("  ").is_err());
        // writing to a directory is rejected with a clear error
        assert!(agent_write_file(".".into(), "x".into()).is_err());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn write_read_edit_roundtrip() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-rw-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        agent_write_file("sub/hi.txt".into(), "hello world\nsecond".into()).unwrap();
        let read = agent_read_file("sub/hi.txt".into(), None, None, None).unwrap();
        assert!(read.contains("hello world"));

        // unique edit
        agent_edit_file("sub/hi.txt".into(), "hello".into(), "hi".into(), None).unwrap();
        assert!(agent_read_file("sub/hi.txt".into(), None, None, None).unwrap().starts_with("hi world"));

        // non-existent old_string errors
        assert!(agent_edit_file("sub/hi.txt".into(), "nope".into(), "x".into(), None).is_err());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn read_file_pages_with_actionable_footer() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-page-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        let body: String = (1..=1000).map(|i| format!("line {i}\n")).collect();
        std::fs::write(tmp.join("big.txt"), &body).unwrap();

        // The headline behavior: a 1000-line source file is ONE read — no paging.
        let full = agent_read_file("big.txt".into(), None, None, None).unwrap();
        assert!(full.contains("line 1000"));
        assert!(!full.contains("offset="));

        // Only when the char budget genuinely can't hold the file does it page,
        // and the footer must carry a FOLLOWABLE offset.
        let page1 = agent_read_file("big.txt".into(), None, None, Some(4000)).unwrap();
        assert!(page1.contains("offset="));
        let tail = page1.rsplit("offset=").next().unwrap();
        let next: usize =
            tail.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap();
        let page2 = agent_read_file("big.txt".into(), Some(next), None, Some(60_000)).unwrap();
        assert!(page2.starts_with(&format!("line {next}")));
        assert!(page2.contains("line 1000"));
        assert!(!page2.contains("offset="));

        // Small file: no footer at all.
        std::fs::write(tmp.join("small.txt"), "hello\nworld\n").unwrap();
        let small = agent_read_file("small.txt".into(), None, None, None).unwrap();
        assert!(!small.contains("offset="));

        // A full-file diff snapshot (max_chars = 400_000) reads a large file
        // WHOLE with no footer — so the diff isn't polluted by pagination text
        // and its line counts are correct. ~100 KB / 2500 lines.
        let big: String = (1..=2500).map(|i| format!("content line number {i}\n")).collect();
        std::fs::write(tmp.join("huge.txt"), &big).unwrap();
        let full_snapshot = agent_read_file("huge.txt".into(), None, None, Some(400_000)).unwrap();
        assert!(full_snapshot.contains("content line number 1\n"));
        assert!(full_snapshot.contains("content line number 2500"));
        assert!(!full_snapshot.contains("offset="), "full-read snapshot must not paginate");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn checkpoint_rewind_restores_files() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-cp-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        cp_clear();

        std::fs::write(tmp.join("a.txt"), "v1").unwrap();

        // Turn 1: edit a.txt + create b.txt.
        let cp1 = agent_checkpoint_begin();
        agent_write_file("a.txt".into(), "v2".into()).unwrap();
        agent_write_file("b.txt".into(), "new".into()).unwrap();
        // Turn 2: edit a.txt again via edit_file.
        let _cp2 = agent_checkpoint_begin();
        agent_edit_file("a.txt".into(), "v2".into(), "v3".into(), None).unwrap();
        assert_eq!(std::fs::read_to_string(tmp.join("a.txt")).unwrap(), "v3");

        // Rewind to before turn 1: a.txt back to v1, b.txt gone.
        agent_checkpoint_revert_to(cp1).unwrap();
        assert_eq!(std::fs::read_to_string(tmp.join("a.txt")).unwrap(), "v1");
        assert!(!tmp.join("b.txt").exists());

        cp_clear();
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn search_code_ranks_and_splits_camel_case() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-search-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        std::fs::write(
            tmp.join("auth.ts"),
            "export function getUserName(token: string) {\n  // validate the login session\n  return decode(token).name;\n}\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("db.ts"),
            "export function createPool() {\n  // database connection pool\n  return new Pool();\n}\n",
        )
        .unwrap();

        // Multi-term + camelCase splitting: "user name login" should hit auth.ts.
        let hits = agent_search_code("user name login".into(), Some(5)).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].path, "auth.ts");
        assert!(hits[0].snippet.contains("getUserName"));
        // Off-topic query prefers the other file.
        let hits2 = agent_search_code("database pool".into(), Some(5)).unwrap();
        assert_eq!(hits2[0].path, "db.ts");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn bg_job_roundtrip() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-bg-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        // The agent shell PATH must include the common tool dirs the GUI app
        // doesn't inherit (this was why npm/node "didn't exist").
        #[cfg(unix)]
        assert!(augmented_path().contains("/usr/bin"));

        let id = agent_bash_bg("echo started; sleep 0.2; echo done-marker".into()).unwrap();
        // Running immediately after spawn.
        let info = agent_bg_output(id).unwrap();
        assert!(info.running || info.code == Some(0));

        // Wait for it to finish, then reap exactly once.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let reaped = agent_bg_reap();
            if let Some(job) = reaped.iter().find(|j| j.id == id) {
                assert_eq!(job.code, Some(0));
                assert!(job.tail.contains("done-marker"));
                break;
            }
            assert!(Instant::now() < deadline, "bg job never finished");
            std::thread::sleep(Duration::from_millis(50));
        }
        // Second reap must not report it again.
        assert!(agent_bg_reap().iter().all(|j| j.id != id));

        // Kill path: a long sleeper dies on request and never gets reported.
        let id2 = agent_bash_bg("sleep 30".into()).unwrap();
        agent_bg_kill(id2).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert!(agent_bg_reap().iter().all(|j| j.id != id2));

        bg_kill_all();
        std::fs::remove_dir_all(&tmp).ok();
    }
}
