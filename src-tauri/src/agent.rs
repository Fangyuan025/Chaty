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

/// Keep Windows console children invisible: a GUI-subsystem app spawning a
/// console process (cmd, taskkill, python …) pops a black console window for
/// every call without CREATE_NO_WINDOW. No-op elsewhere.
pub(crate) fn hide_console(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// The active workspace root (absolute, canonicalized). `None` until the user
/// opens a folder for the coding session.
static WORKSPACE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Language for model-visible tool output. Historically every string carried
/// both languages ("已写入 … (wrote …)"), which taxes small local models with
/// tokens on every single step; now the loop sets the session language once
/// and each string renders in ONE language. Default Zh keeps the existing
/// Chinese-asserting tests (and old frontends that never call set_lang) green.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Zh,
    En,
}
static LANG: Mutex<Lang> = Mutex::new(Lang::Zh);

#[tauri::command]
pub fn agent_set_lang(lang: String) {
    *LANG.lock().unwrap() = if lang == "en" { Lang::En } else { Lang::Zh };
}

/// Hashline anchor mode: read_file prefixes every line with `N:hh→` and the
/// edit_lines tool becomes the documented editor (exact-string edit_file stays
/// executable as a fallback). Off by default; flipped per session by the
/// frontend, or by the CHATY_EDIT_ANCHORS env var (bench A/B, headless boot).
static EDIT_ANCHORS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[tauri::command]
pub fn agent_set_edit_anchors(on: bool) {
    EDIT_ANCHORS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn anchors_on() -> bool {
    EDIT_ANCHORS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Pick the session-language variant of a model-visible string.
pub(crate) fn tr(zh: &str, en: &str) -> String {
    match *LANG.lock().unwrap() {
        Lang::Zh => zh.to_string(),
        Lang::En => en.to_string(),
    }
}

/// `tr` for format-heavy call sites: `trf!("已写入 {}", "wrote {}", path)`.
macro_rules! trf {
    ($zh:literal, $en:literal $(, $arg:expr)* $(,)?) => {
        if crate::agent::lang_is_en() {
            format!($en $(, $arg)*)
        } else {
            format!($zh $(, $arg)*)
        }
    };
}

pub(crate) fn lang_is_en() -> bool {
    *LANG.lock().unwrap() == Lang::En
}

/// Session-scoped extra directories the user granted beyond the workspace
/// (absolute, canonicalized). Cleared when the workspace/session changes.
static GRANTED_DIRS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Error-marker protocol for out-of-workspace access: the frontend recognizes
/// this prefix, asks the user, grants the directory on approval, and retries.
/// Format: `NEED_DIR_GRANT\t<absolute dir>\t<human message>`.
pub(crate) const NEED_DIR_GRANT: &str = "NEED_DIR_GRANT";

fn need_grant_err(target: &Path, raw: &str) -> String {
    // Suggest the closest directory: the path itself if it's an existing dir,
    // else its parent (works for files about to be created too).
    let dir = if target.is_dir() {
        target.to_path_buf()
    } else {
        target.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| target.to_path_buf())
    };
    trf!(
        "{NEED_DIR_GRANT}\t{}\t路径在工作区外，需要用户授权: {raw}",
        "{NEED_DIR_GRANT}\t{}\tpath is outside the workspace — needs the user's approval: {raw}",
        dir.display()
    )
}

fn in_granted_dirs(p: &Path) -> bool {
    GRANTED_DIRS.lock().unwrap().iter().any(|g| p.starts_with(g))
}

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
        .ok_or_else(|| tr("尚未打开工作区", "no workspace opened"))
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

/// The canonical form of a path, tolerating paths that don't exist yet: a
/// missing file canonicalizes its parent and re-appends the name (so `/var/…`
/// and `/private/var/…` spellings compare equal for new files too); a fully
/// non-existent chain stays lexical.
fn canonical_or_lexical(p: &Path) -> PathBuf {
    if let Ok(c) = p.canonicalize() {
        return c;
    }
    if let (Some(parent), Some(name)) = (p.parent(), p.file_name()) {
        if let Ok(cp) = parent.canonicalize() {
            return cp.join(name);
        }
    }
    p.to_path_buf()
}

/// Resolve a user/model-supplied path against the workspace and confine it.
/// Relative paths join the root; absolute paths must land inside the workspace
/// or inside a directory the user granted this session. Anything else returns
/// a `NEED_DIR_GRANT` marker error so the frontend can ask the user and retry
/// (instead of the old flat rejection). The check runs on the CANONICAL path,
/// so a symlink can't smuggle a path out of the allowed roots — and alias
/// spellings (macOS `/var` → `/private/var`) compare correctly.
fn resolve(rel: &str) -> Result<PathBuf, String> {
    if rel.trim().is_empty() {
        return Err(tr("路径为空，请提供文件路径", "empty path — provide a file path"));
    }
    let root = workspace()?;
    let p = Path::new(rel);
    let joined = if p.is_absolute() { p.to_path_buf() } else { root.join(p) };
    let norm = lexical_normalize(&joined);
    let checked = canonical_or_lexical(&norm);
    if checked.starts_with(&root) || in_granted_dirs(&checked) {
        return Ok(checked);
    }
    Err(need_grant_err(&checked, rel))
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
        return Err(tr("不是有效的文件夹", "not a directory"));
    }
    let canon = p.canonicalize().map_err(|e| e.to_string())?;
    let shown = canon.to_string_lossy().to_string();
    let changed = WORKSPACE.lock().unwrap().replace(canon.clone()) != Some(canon);
    if changed {
        // Background jobs, checkpoints, dir grants and the browser belong to
        // the previous workspace.
        bg_kill_all();
        cp_clear();
        dl_clear();
        GRANTED_DIRS.lock().unwrap().clear();
        crate::browser::shutdown();
    }
    Ok(shown)
}

#[tauri::command]
pub fn agent_get_workspace() -> Option<String> {
    WORKSPACE.lock().unwrap().as_ref().map(|p| p.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Session directory grants (access beyond the workspace, user-approved)
// ---------------------------------------------------------------------------

/// Grant access to a directory outside the workspace for this session.
/// Returns the canonical path actually granted.
#[tauri::command]
pub fn agent_grant_dir(path: String) -> Result<String, String> {
    let p = PathBuf::from(&path);
    if !p.is_dir() {
        return Err(tr("不是有效的文件夹", "not a directory"));
    }
    let canon = p.canonicalize().map_err(|e| e.to_string())?;
    let mut dirs = GRANTED_DIRS.lock().unwrap();
    if !dirs.iter().any(|d| *d == canon) {
        dirs.push(canon.clone());
    }
    Ok(canon.to_string_lossy().to_string())
}

/// Revoke a previously granted directory (one-click from the UI).
#[tauri::command]
pub fn agent_revoke_dir(path: String) {
    let p = PathBuf::from(&path);
    GRANTED_DIRS.lock().unwrap().retain(|d| *d != p);
}

/// The directories granted this session (for the UI chips).
#[tauri::command]
pub fn agent_list_grants() -> Vec<String> {
    GRANTED_DIRS.lock().unwrap().iter().map(|d| d.to_string_lossy().to_string()).collect()
}

/// Clear every grant — a new session in the same workspace starts clean.
#[tauri::command]
pub fn agent_clear_grants() {
    GRANTED_DIRS.lock().unwrap().clear();
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
    symbol: Option<String>,
) -> Result<String, String> {
    // Symbol mode: return the definition body + its callers instead of the
    // raw file — a big file becomes one focused, ready-to-reason context.
    if let Some(sym) = symbol.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return read_symbol_context(&path, sym);
    }
    const MAX_READ_LINES: usize = 12000; // hard per-call line ceiling
    const MAX_LINE_CHARS: usize = 4000; // pathological single lines (minified JS)

    // The caller (frontend) sizes the budget from the model's ACTUAL context
    // window, so a normal source file fits in ONE call; the default only
    // applies to callers that don't say (tests, older paths). The ceiling
    // matches MAX_READ_BYTES so a full-file diff snapshot never paginates.
    let budget = max_chars.unwrap_or(24_000).clamp(4_000, 400_000);

    let abs = resolve(&path)?;
    let meta = std::fs::metadata(&abs).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
    if meta.is_dir() {
        return Err(tr("这是一个目录，请用 list_dir", "that's a directory — use list_dir"));
    }
    let bytes = std::fs::read(&abs).map_err(|e| e.to_string())?;
    let slice = &bytes[..bytes.len().min(MAX_READ_BYTES)];
    let text = String::from_utf8_lossy(slice);

    let all: Vec<&str> = text.lines().collect();
    let total = all.len();
    let start = offset.unwrap_or(1).max(1) - 1;
    if start >= total && total > 0 {
        return Ok(trf!(
            "(offset 超出范围:文件共 {total} 行)",
            "(offset beyond EOF: file has {total} lines)"
        ));
    }
    let want = limit.unwrap_or(MAX_READ_LINES).clamp(1, MAX_READ_LINES);

    let anchors = anchors_on();
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
        // Hashline mode: every line carries its edit anchor, so the model can
        // address edit_lines ops straight from what it just read.
        let display = if anchors { format!("{}→{}", anchor_of(i + 1, line), line) } else { line.to_string() };
        if !out.is_empty() && out.len() + display.len() + 1 > budget {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&display);
        end = i + 1;
    }

    if end < total {
        out.push_str(&trf!(
            "\n\n[文件共 {total} 行,本次显示第 {}-{end} 行;继续阅读请用 offset={}]",
            "\n\n[file has {total} lines, shown {}-{end}; continue with offset={}]",
            start + 1,
            end + 1,
        ));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Background downloads (progress-tracked; the agent keeps working meanwhile)
// ---------------------------------------------------------------------------

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DlInfo {
    pub id: u64,
    pub url: String,
    /// Workspace-relative destination for display.
    pub path: String,
    pub downloaded: u64,
    /// Content-Length when the server sent one.
    pub total: Option<u64>,
    pub done: bool,
    pub error: Option<String>,
}

struct DlState {
    info: DlInfo,
    reported: bool,
}

static DOWNLOADS: Mutex<Option<std::collections::HashMap<u64, DlState>>> = Mutex::new(None);
static DL_NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn dl_update(id: u64, f: impl FnOnce(&mut DlState)) {
    if let Some(map) = DOWNLOADS.lock().unwrap().as_mut() {
        if let Some(st) = map.get_mut(&id) {
            f(st);
        }
    }
}

/// Start a BACKGROUND download of a URL into the workspace: returns
/// immediately with an id, streams to disk with live progress (UI badge), and
/// is picked up by `agent_dl_reap` when finished so the model gets notified
/// without ever blocking on the transfer. Sandboxed through the same `resolve`
/// as every other write, and journaled so rewind removes the file.
#[tauri::command]
pub async fn agent_web_download(url: String, path: String) -> Result<String, String> {
    const CAP: u64 = 100 * 1024 * 1024;
    let abs = resolve(&path)?;
    if abs.is_dir() {
        return Err(trf!("目标是一个目录: {path}", "target is a directory: {path}"));
    }
    let root = workspace()?;
    let rel = rel_display(&root, &abs);
    let id = DL_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    {
        let mut guard = DOWNLOADS.lock().unwrap();
        guard.get_or_insert_with(Default::default).insert(
            id,
            DlState {
                info: DlInfo {
                    id,
                    url: url.trim().to_string(),
                    path: rel.clone(),
                    downloaded: 0,
                    total: None,
                    done: false,
                    error: None,
                },
                reported: false,
            },
        );
    }

    let url_owned = url.trim().to_string();
    tokio::spawn(async move {
        let finish = |err: Option<String>| {
            dl_update(id, |st| {
                st.info.done = true;
                st.info.error = err;
            });
        };
        let client = match reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15")
            .timeout(std::time::Duration::from_secs(600))
            .build()
        {
            Ok(c) => c,
            Err(e) => return finish(Some(e.to_string())),
        };
        let resp = match client.get(&url_owned).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => return finish(Some(format!("HTTP {}", r.status()))),
            Err(e) => return finish(Some(e.to_string())),
        };
        let total = resp.content_length();
        dl_update(id, |st| st.info.total = total);
        if total.is_some_and(|t| t > CAP) {
            return finish(Some(trf!("文件过大 ({} MB),上限 100 MB", "file too large ({} MB) — the cap is 100 MB", total.unwrap() / 1024 / 1024)));
        }
        cp_record(&abs);
        if let Some(parent) = abs.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return finish(Some(e.to_string()));
            }
        }
        let tmp = abs.with_extension("part");
        let mut file = match std::fs::File::create(&tmp) {
            Ok(f) => f,
            Err(e) => return finish(Some(trf!("写入失败: {e}", "write failed: {e}"))),
        };
        let mut resp = resp;
        let mut written: u64 = 0;
        loop {
            match resp.chunk().await {
                Ok(Some(chunk)) => {
                    written += chunk.len() as u64;
                    if written > CAP {
                        let _ = std::fs::remove_file(&tmp);
                        return finish(Some(tr("文件过大,上限 100 MB", "file too large — the cap is 100 MB")));
                    }
                    if let Err(e) = std::io::Write::write_all(&mut file, &chunk) {
                        let _ = std::fs::remove_file(&tmp);
                        return finish(Some(trf!("写入失败: {e}", "write failed: {e}")));
                    }
                    dl_update(id, |st| st.info.downloaded = written);
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = std::fs::remove_file(&tmp);
                    return finish(Some(e.to_string()));
                }
            }
        }
        drop(file);
        if let Err(e) = std::fs::rename(&tmp, &abs) {
            let _ = std::fs::remove_file(&tmp);
            return finish(Some(trf!("写入失败: {e}", "write failed: {e}")));
        }
        dl_update(id, |st| st.info.downloaded = written);
        finish(None);
    });

    let u = url.trim();
    Ok(trf!(
        "已开始后台下载 #{id}: {u} → {rel}。下载不会阻塞你,继续做别的;完成或失败时系统会自动通知你,也可随时继续当前任务。",
        "download #{id} started in the background: {u} → {rel}. It won't block you — keep working; you'll be notified when it finishes or fails, and you can keep going meanwhile."
    ))
}

/// Live status of all downloads this session (UI progress badge).
#[tauri::command]
pub fn agent_dl_list() -> Vec<DlInfo> {
    DOWNLOADS
        .lock()
        .unwrap()
        .as_ref()
        .map(|m| m.values().map(|s| s.info.clone()).collect())
        .unwrap_or_default()
}

/// Finished-but-unreported downloads: marks them reported and returns them —
/// the agent loop injects these as tool results so the model learns the outcome.
#[tauri::command]
pub fn agent_dl_reap() -> Vec<DlInfo> {
    let mut out = Vec::new();
    if let Some(map) = DOWNLOADS.lock().unwrap().as_mut() {
        for st in map.values_mut() {
            if st.info.done && !st.reported {
                st.reported = true;
                out.push(st.info.clone());
            }
        }
    }
    out
}

/// Forget all download records (workspace switch — files stay where they are).
fn dl_clear() {
    *DOWNLOADS.lock().unwrap() = None;
}

/// One-call repo orientation: README lede, manifest summary, a two-level
/// directory tree, entry points, and a language census — the "walk around the
/// codebase for ten steps" a fresh session used to spend on list_dir chains.
#[tauri::command]
pub fn agent_understand_repo() -> Result<String, String> {
    let root = workspace()?;
    let mut out = String::new();

    // README lede.
    for name in ["README.md", "README", "readme.md", "README.zh.md"] {
        if let Ok(text) = std::fs::read_to_string(root.join(name)) {
            let lede: String = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .take(6)
                .collect::<Vec<_>>()
                .join("\n");
            out.push_str(&format!("[{name}]\n{}\n\n", lede.chars().take(600).collect::<String>()));
            break;
        }
    }

    // Manifests → project identity + how to run/test it.
    if let Ok(pkg) = std::fs::read_to_string(root.join("package.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&pkg) {
            let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("?");
            let scripts = v
                .get("scripts")
                .and_then(|x| x.as_object())
                .map(|m| m.keys().take(10).cloned().collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            out.push_str(&format!("[package.json] name={name} · scripts: {scripts}\n"));
        }
    }
    for (mf, label) in [
        ("Cargo.toml", "rust crate"),
        ("pyproject.toml", "python project"),
        ("requirements.txt", "python requirements"),
        ("go.mod", "go module"),
    ] {
        if root.join(mf).is_file() {
            out.push_str(&format!("[{mf}] {label}\n"));
        }
    }

    // Two-level tree + language census + entry points.
    let mut tree = String::new();
    let mut census: HashMap<String, usize> = HashMap::new();
    let mut entries_found: Vec<String> = Vec::new();
    let mut listed = 0;
    for entry in walkdir::WalkDir::new(&root)
        .max_depth(2)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(name.starts_with('.') && e.depth() > 0)
                && !(e.file_type().is_dir() && SKIP_DIRS.contains(&name.as_ref()))
        })
        .flatten()
    {
        if entry.depth() == 0 {
            continue;
        }
        let rel = rel_display(&root, entry.path());
        if entry.file_type().is_dir() {
            let count = std::fs::read_dir(entry.path()).map(|d| d.count()).unwrap_or(0);
            if listed < 60 {
                tree.push_str(&format!("{}{}/ ({count})\n", "  ".repeat(entry.depth() - 1), entry.file_name().to_string_lossy()));
                listed += 1;
            }
        } else {
            if listed < 60 && entry.depth() == 1 {
                tree.push_str(&format!("{}\n", entry.file_name().to_string_lossy()));
                listed += 1;
            }
            let stem = entry.file_name().to_string_lossy().to_lowercase();
            if matches!(
                stem.as_str(),
                "main.py" | "app.py" | "index.ts" | "index.js" | "main.ts" | "main.js" | "main.rs" | "lib.rs" | "app.tsx" | "main.go"
            ) {
                entries_found.push(rel.clone());
            }
        }
    }
    // Census over the whole workspace (extensions of source files).
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
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
                let ext = ext.to_lowercase();
                if matches!(ext.as_str(), "rs" | "ts" | "tsx" | "js" | "jsx" | "py" | "go" | "swift" | "java" | "c" | "cpp" | "h" | "rb" | "php" | "css" | "html" | "vue" | "kt") {
                    *census.entry(ext).or_insert(0) += 1;
                }
            }
        }
    }
    let mut census: Vec<(String, usize)> = census.into_iter().collect();
    census.sort_by(|a, b| b.1.cmp(&a.1));
    let census_str = census
        .iter()
        .take(6)
        .map(|(e, n)| format!(".{e}×{n}"))
        .collect::<Vec<_>>()
        .join(" · ");

    out.push_str(&trf!("\n[目录 (top 2 levels)]\n{tree}", "\n[directory, top 2 levels]\n{tree}"));
    if !census_str.is_empty() {
        out.push_str(&trf!("\n[语言构成] {census_str}\n", "\n[language mix] {census_str}\n"));
    }
    if !entries_found.is_empty() {
        entries_found.truncate(8);
        let entries = entries_found.join(", ");
        out.push_str(&trf!("[入口候选] {entries}\n", "[entry-point candidates] {entries}\n"));
    }
    if out.trim().is_empty() {
        out = tr("(空工作区)", "(empty workspace)");
    }
    Ok(out)
}

/// Smart minimal validation: figure out which tests relate to the changed
/// files, run JUST those, and summarize failures — the find/filter/interpret
/// work the model used to burn steps on. Targets default to the files touched
/// this turn (checkpoint journal).
#[tauri::command]
pub async fn agent_validate_change(files: Option<Vec<String>>) -> Result<String, String> {
    let root = workspace()?;
    let mut targets: Vec<PathBuf> = Vec::new();
    match files {
        Some(fs) if !fs.is_empty() => {
            for f in fs {
                targets.push(resolve(&f)?);
            }
        }
        _ => {
            let cps = CHECKPOINTS.lock().unwrap();
            if let Some(cp) = cps.last() {
                for e in &cp.entries {
                    targets.push(e.path.clone());
                }
            }
        }
    }
    targets.retain(|p| p.is_file());
    if targets.is_empty() {
        return Ok(tr(
            "本轮还没有记录到文件改动;可传 files 参数明确指定要验证的文件",
            "no tracked changes this turn — pass files explicitly to validate them",
        ));
    }
    let stems: Vec<String> = targets
        .iter()
        .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(|s| s.to_lowercase()))
        .filter(|s| s.len() >= 3)
        .collect();
    let rels: Vec<String> = targets.iter().map(|p| rel_display(&root, p)).collect();

    // Related test files: conventional test names whose CONTENT mentions one
    // of the changed stems (cheap import/usage heuristic).
    let mut py_tests: Vec<String> = Vec::new();
    let mut js_tests: Vec<String> = Vec::new();
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
        if !entry.file_type().is_file() || py_tests.len() + js_tests.len() >= 12 {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_lowercase();
        let is_py = (name.starts_with("test_") || name.ends_with("_test.py")) && name.ends_with(".py");
        let is_js = name.contains(".test.") || name.contains(".spec.") ||
            entry.path().components().any(|c| c.as_os_str() == "__tests__");
        if !is_py && !is_js {
            continue;
        }
        if entry.metadata().map(|m| m.len() > SEARCH_MAX_FILE).unwrap_or(true) {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(entry.path()) else { continue };
        let lower = body.to_lowercase();
        let rel = rel_display(&root, entry.path());
        // A test relates if it mentions a changed stem — or IS a changed file.
        let related = stems.iter().any(|st| lower.contains(st.as_str()))
            || rels.iter().any(|r| *r == rel);
        if !related {
            continue;
        }
        if is_py {
            py_tests.push(rel);
        } else if is_js {
            js_tests.push(rel);
        }
    }

    let mut ran_any = false;
    let mut out = trf!("验证目标: {}\n", "validating: {}\n", rels.join(", "));
    let timeout = Duration::from_secs(180);
    let run_cmd = |title: &str, cmd: String, out: &mut String| {
        out.push_str(&format!("\n$ {cmd}\n"));
        match run_bash(&root, &cmd, timeout, None, true) {
            Ok(r) => {
                let merged = format!("{}\n{}", r.stdout, r.stderr);
                // Failure digest: the lines a human would read first.
                let fails: Vec<&str> = merged
                    .lines()
                    .filter(|l| {
                        let t = l.trim_start();
                        t.starts_with("FAILED") || t.starts_with("ERROR") || t.contains("AssertionError")
                            || t.starts_with("✗") || t.starts_with("×") || t.contains("FAIL ")
                            || (t.contains("failed") && t.contains("passed"))
                            || t.starts_with("test result:")
                    })
                    .take(14)
                    .collect();
                if r.timed_out {
                    out.push_str(&trf!("⏱ 超时({title})\n", "⏱ timed out ({title})\n"));
                } else if r.code == 0 {
                    out.push_str(&tr("✓ 通过\n", "✓ passed\n"));
                } else {
                    out.push_str(&trf!("✗ 失败 (exit {})\n", "✗ failed (exit {})\n", r.code));
                }
                if !fails.is_empty() {
                    out.push_str(&format!("{}\n", fails.join("\n")));
                }
                if r.code != 0 && !r.timed_out {
                    // Tail carries the actual assertion/context.
                    let tail: String = merged
                        .lines()
                        .rev()
                        .take(20)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("\n");
                    out.push_str(&trf!("--- 输出尾部 ---\n{}\n", "--- output tail ---\n{}\n", tail.chars().take(1800).collect::<String>()));
                }
            }
            Err(e) => out.push_str(&trf!("(无法运行: {e})\n", "(could not run: {e})\n")),
        }
    };

    if !py_tests.is_empty() {
        py_tests.truncate(6);
        // Fresh PYTHONPYCACHEPREFIX per run: .pyc validation keys on
        // (mtime-seconds, size), so an agent that edits and validates within
        // the same second would otherwise execute STALE bytecode.
        run_cmd(
            "pytest",
            format!(
                "PYTHONPYCACHEPREFIX=$(mktemp -d) python3 -m pytest -x -q {}",
                py_tests.join(" ")
            ),
            &mut out,
        );
        ran_any = true;
    }
    if !js_tests.is_empty() {
        js_tests.truncate(6);
        let pkg = std::fs::read_to_string(root.join("package.json")).unwrap_or_default();
        let runner = if pkg.contains("\"vitest\"") {
            Some(format!("npx vitest run {}", js_tests.join(" ")))
        } else if pkg.contains("\"jest\"") {
            Some(format!("npx jest {}", js_tests.join(" ")))
        } else {
            None
        };
        match runner {
            Some(cmd) => {
                run_cmd("js tests", cmd, &mut out);
                ran_any = true;
            }
            None => out.push_str(&tr(
                "\n(找到 JS/TS 测试文件但未识别出 vitest/jest——请用 bash 跑项目自己的测试命令)\n",
                "\n(JS/TS test files found but no vitest/jest detected — run the project's own test command via bash)\n",
            )),
        }
    }
    if root.join("Cargo.toml").is_file()
        && targets.iter().any(|p| p.extension().is_some_and(|e| e == "rs"))
    {
        let mut filters: Vec<&str> = stems.iter().map(|s| s.as_str()).take(3).collect();
        filters.dedup();
        run_cmd(
            "cargo test",
            format!("cargo test {}", filters.join(" ")),
            &mut out,
        );
        ran_any = true;
    }

    if !ran_any {
        out.push_str(&tr(
            "\n没有发现与改动相关的测试(按 test_*.py / *.test.* / *.spec.* / cargo 约定查找)。如果项目有自己的测试命令,请直接用 bash 运行。",
            "\nno tests related to the change were found (looked for test_*.py / *.test.* / *.spec.* / cargo conventions) — if the project has its own test command, run it via bash.",
        ));
    }
    Ok(out)
}

/// Cheap post-edit syntax gate. Returns None when no checker exists for the
/// file type; Some(Err(msg)) when the file fails to parse. Checkers are
/// millisecond-cheap: pure-Rust parsing for JSON/TOML, `python3 -m
/// py_compile` / `bash -n` / `node --check` when those binaries exist.
/// (.ts/.rs need whole-project context — deliberately unchecked.)
pub(crate) fn syntax_check(abs: &Path) -> Option<Result<(), String>> {
    let ext = abs.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    let run = |bin: &str, args: &[&str]| -> Option<Result<(), String>> {
        let mut cmd = Command::new(bin);
        cmd.args(args).stdout(Stdio::null()).stderr(Stdio::piped());
        hide_console(&mut cmd); // post-edit checks would flash a console per edit
        // augmented_path is unix-only (GUI launches lack brew/nvm paths);
        // Windows inherits the parent environment as-is.
        #[cfg(unix)]
        cmd.env("PATH", augmented_path());
        let out = cmd.output().ok()?; // binary missing → no checker
        if out.status.success() {
            Some(Ok(()))
        } else {
            let err = String::from_utf8_lossy(&out.stderr);
            let tail: String = err.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
            Some(Err(tail.chars().take(600).collect()))
        }
    };
    let p = abs.to_string_lossy();
    match ext.as_str() {
        "json" => {
            let text = std::fs::read_to_string(abs).ok()?;
            Some(serde_json::from_str::<serde_json::Value>(&text).map(|_| ()).map_err(|e| e.to_string()))
        }
        "toml" => {
            let text = std::fs::read_to_string(abs).ok()?;
            Some(text.parse::<toml::Value>().map(|_| ()).map_err(|e| e.to_string()))
        }
        // compile() instead of py_compile: no __pycache__ dropped into the
        // user's workspace.
        "py" => {
            let args: &[&str] =
                &["-c", "import sys; compile(open(sys.argv[1], 'rb').read(), sys.argv[1], 'exec')", &p];
            // Windows official installers ship `python.exe` (no python3);
            // worse, the Microsoft-Store stub NAMED python3 exists on stock
            // installs and exits non-zero with a store hint — which would
            // flag every .py edit as a syntax error. Prefer `python`, and
            // treat a failure that mentions the store stub as "no checker".
            #[cfg(windows)]
            {
                let looks_like_stub = |r: &Result<(), String>| {
                    r.as_ref()
                        .err()
                        .map_or(false, |e| e.contains("Microsoft Store") || e.contains("app store") || e.contains("AppData\\Local\\Microsoft\\WindowsApps"))
                };
                let first = run("python", args);
                match first {
                    Some(ref r) if looks_like_stub(r) => run("python3", args).filter(|r| !looks_like_stub(r)),
                    Some(r) => Some(r),
                    None => run("python3", args).filter(|r| !looks_like_stub(r)),
                }
            }
            #[cfg(not(windows))]
            run("python3", args)
        }
        "sh" | "bash" => run("bash", &["-n", &p]),
        "js" | "mjs" | "cjs" => run("node", &["--check", &p]),
        _ => None,
    }
}

/// The syntax note appended to a successful write/edit: silent when clean or
/// uncheckable; loud when THIS edit broke a previously-parsable file; softer
/// when the file was already broken before the edit.
fn syntax_note(abs: &Path, was_clean: Option<bool>) -> String {
    match syntax_check(abs) {
        Some(Err(e)) => match was_clean {
            Some(true) => trf!(
                "\n⚠️ 语法检查失败——本次编辑把一个原本可解析的文件改坏了:\n{e}\n请立即修复;如需还原,该文件已有检查点可回退。",
                "\n⚠️ syntax check failed — this edit BROKE a previously-parsable file:\n{e}\nfix now, or rewind the checkpoint."
            ),
            _ => trf!("\n⚠️ 语法检查失败:\n{e}", "\n⚠️ syntax check failed:\n{e}"),
        },
        _ => String::new(),
    }
}


#[tauri::command]
pub fn agent_write_file(path: String, content: String) -> Result<String, String> {
    let abs = resolve(&path)?;
    if abs.is_dir() {
        return Err(trf!(
            "目标是一个目录，不能写入；请提供文件路径: {path}",
            "target is a directory — give a file path: {path}"
        ));
    }
    cp_record(&abs);
    let was_clean = abs.is_file().then(|| syntax_check(&abs)).flatten().map(|r| r.is_ok());
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&abs, content.as_bytes()).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    let root = workspace()?;
    Ok(trf!(
        "已写入 {} ({} 字节){}",
        "wrote {} ({} bytes){}",
        rel_display(&root, &abs),
        content.len(),
        syntax_note(&abs, was_clean)
    ))
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
        return Err(tr("old_string 与 new_string 相同", "old_string and new_string are identical — no-op edit"));
    }
    let abs = resolve(&path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
    cp_record(&abs);
    let was_clean = syntax_check(&abs).map(|r| r.is_ok());
    let count = text.matches(&old_string).count();
    if count == 0 {
        return Err(not_found_error(&text, &old_string));
    }
    let all = replace_all.unwrap_or(false);
    if count > 1 && !all {
        return Err(trf!(
            "old_string 出现 {count} 次，不唯一；请提供更多上下文或用 replace_all",
            "old_string is not unique ({count} matches) — add more context or use replace_all"
        ));
    }
    let pos = text.find(&old_string).unwrap_or(0);
    let start_line = text[..pos].matches('\n').count();
    let updated = if all {
        text.replace(&old_string, &new_string)
    } else {
        text.replacen(&old_string, &new_string, 1)
    };
    std::fs::write(&abs, updated.as_bytes()).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    let root = workspace()?;
    // Echo the edited neighborhood back so the model can confirm the result
    // without spending another read_file step.
    let span = new_string.matches('\n').count() + 1;
    Ok(trf!(
        "已编辑 {}（替换 {} 处）。修改后该处内容:\n{}{}",
        "edited {} ({} replacement(s)). The region now reads:\n{}{}",
        rel_display(&root, &abs),
        if all { count } else { 1 },
        numbered_context(&updated, start_line, span),
        syntax_note(&abs, was_clean)
    ))
}

// ── Hashline anchors (ported from xai-org/grok-build's hashline scheme) ──
// Anchor = "LINE:HASH", HASH = 3 lowercase letters from a whitespace-normalized
// FNV-1a-32 of the line. Line numbers alone go stale after any edit above;
// exact-string patches pay for uniqueness in tokens. The anchor pins identity
// (hash) to position (line) so an edit names its target in ~8 characters.

/// Whitespace-normalized FNV-1a-32: trim + collapse internal whitespace runs.
/// Stable across formatter-only churn; distinguishes real content changes.
fn line_hash(line: &str) -> u32 {
    const FNV_OFFSET: u32 = 2_166_136_261;
    const FNV_PRIME: u32 = 16_777_619;
    let mut h = FNV_OFFSET;
    let mut prev_ws = false;
    for byte in line.trim().bytes() {
        if byte.is_ascii_whitespace() {
            if !prev_ws {
                h ^= b' ' as u32;
                h = h.wrapping_mul(FNV_PRIME);
                prev_ws = true;
            }
        } else {
            h ^= byte as u32;
            h = h.wrapping_mul(FNV_PRIME);
            prev_ws = false;
        }
    }
    h
}

/// Three lowercase letters, one per byte region of the hash.
fn encode_line_hash(hash: u32) -> String {
    (0..3).map(|i| (((hash >> (i * 8)) % 26) as u8 + b'a') as char).collect()
}

/// `"22:abc"` for 1-based line `idx1` with content `line`.
pub(crate) fn anchor_of(idx1: usize, line: &str) -> String {
    format!("{idx1}:{}", encode_line_hash(line_hash(line)))
}

/// A parsed `"LINE:HASH"` anchor from model input.
struct ParsedAnchor {
    line1: usize,
    hash: String,
}

fn parse_anchor(s: &str) -> Option<ParsedAnchor> {
    // Models paste the whole prefixed line as the anchor ("142:qzh→  user.save()")
    // — every anchor-smoke edit failed on exactly this. Take what's left of the
    // arrow, then of any whitespace; the line content is redundant, not wrong.
    let s = s.trim();
    let s = s.split('→').next().unwrap_or(s);
    let s = s.split_whitespace().next().unwrap_or(s);
    let (l, h) = s.split_once(':')?;
    let line1: usize = l.trim().parse().ok()?;
    let hash = h.trim().to_ascii_lowercase();
    if line1 == 0 || hash.is_empty() || hash.len() > 4 || !hash.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    Some(ParsedAnchor { line1, hash })
}

const ANCHOR_SHIFT_RADIUS: usize = 20;

/// Resolve an anchor against current lines: exact line+hash, else a unique
/// hash match within ±ANCHOR_SHIFT_RADIUS (edits above shift everything).
/// Ok(0-based index, shifted?) — Err(reason) when missing or ambiguous.
fn resolve_anchor(a: &ParsedAnchor, lines: &[&str]) -> Result<(usize, bool), String> {
    let idx0 = a.line1 - 1;
    if idx0 < lines.len() && encode_line_hash(line_hash(lines[idx0])) == a.hash {
        return Ok((idx0, false));
    }
    let lo = idx0.saturating_sub(ANCHOR_SHIFT_RADIUS);
    let hi = (idx0 + ANCHOR_SHIFT_RADIUS + 1).min(lines.len());
    let hits: Vec<usize> = (lo..hi)
        .filter(|&i| encode_line_hash(line_hash(lines[i])) == a.hash)
        .collect();
    match hits.len() {
        1 => Ok((hits[0], true)),
        0 => Err(trf!(
            "锚点 {}:{} 不匹配当前文件(该行已被改动或行号越界)",
            "anchor {}:{} does not match the current file (line changed or out of range)",
            a.line1, a.hash
        )),
        n => Err(trf!(
            "锚点 {}:{} 有歧义:附近 {n} 行有相同哈希,请 read_file 后用新锚点",
            "anchor {}:{} is ambiguous — {n} nearby lines share this hash; re-read and use fresh anchors",
            a.line1, a.hash
        )),
    }
}

/// `N:hh→content` lines for [line0, line0+span) with ±3 context — the fresh
/// anchors the model needs for its NEXT edit, no re-read required.
fn anchored_context(text: &str, line0: usize, span: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let s = line0.saturating_sub(3);
    let e = (line0 + span + 3).min(lines.len());
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate().take(e).skip(s) {
        let l: String = line.chars().take(200).collect();
        out.push_str(&format!("{}→{}\n", anchor_of(i + 1, line), l));
    }
    out
}

/// One resolved hashline op: replace [start,end) with new_lines (insert when
/// start == end).
struct ResolvedOp {
    start: usize,
    end: usize,
    new_lines: Vec<String>,
}

/// Batch line edits addressed by hashline anchors. `edits` is the model's
/// JSON: an array of {op:"replace",anchor,end_anchor?,content} /
/// {op:"insert_after",anchor,content} — anchor "0" = BOF, "EOF" = EOF for
/// insert_after. Tolerates a stringified array (models double-encode).
#[tauri::command]
pub fn agent_edit_lines(path: String, edits: serde_json::Value) -> Result<String, String> {
    let abs = resolve(&path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
    let had_trailing_nl = text.ends_with('\n');
    let lines: Vec<&str> = text.lines().collect();

    // Accept array | single object | stringified array.
    let arr: Vec<serde_json::Value> = match edits {
        serde_json::Value::Array(a) => a,
        serde_json::Value::String(s) => serde_json::from_str(&s)
            .map_err(|e| trf!("edits 不是合法的 JSON 数组: {e}", "edits is not a valid JSON array: {e}"))?,
        v @ serde_json::Value::Object(_) => vec![v],
        _ => return Err(tr("edits 必须是编辑操作数组", "edits must be an array of edit operations")),
    };
    if arr.is_empty() {
        return Err(tr("edits 为空", "edits is empty"));
    }

    let as_str = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    let mut ops: Vec<ResolvedOp> = Vec::new();
    let mut shifted_notes: Vec<String> = Vec::new();
    for (i, e) in arr.iter().enumerate() {
        let op = as_str(e, "op").unwrap_or_default();
        let content = as_str(e, "content").unwrap_or_default();
        // Anchor prefixes pasted into content = the model copied read_file
        // output verbatim; applying it would corrupt the file.
        for (ln, l) in content.lines().enumerate() {
            let t = l.trim_start();
            if let Some((before, _)) = t.split_once('→') {
                if before.len() <= 8 && before.contains(':') && !before.contains(' ') {
                    return Err(trf!(
                        "content 第 {} 行仍带着锚点前缀(如 \"22:abc→\")。请去掉每行的前缀和 → 分隔符,只保留文件内容后重试",
                        "content line {} still carries an anchor prefix (like \"22:abc→\"). Strip the prefix and the → separator from every line, keep only the file content, and retry",
                        ln + 1
                    ));
                }
            }
        }
        let new_lines: Vec<String> = if content.is_empty() {
            Vec::new()
        } else {
            content.lines().map(str::to_string).collect()
        };
        let anchor_raw = as_str(e, "anchor").unwrap_or_default();
        match op.as_str() {
            "replace" => {
                let a = parse_anchor(&anchor_raw).ok_or_else(|| trf!(
                    "第 {} 个操作的 anchor 无效:应为 read_file 里看到的 \"行号:哈希\"(如 \"22:abc\")",
                    "op {}: invalid anchor — use the \"LINE:HASH\" shown by read_file (e.g. \"22:abc\")",
                    i + 1
                ))?;
                let (s0, sh) = resolve_anchor(&a, &lines)?;
                if sh { shifted_notes.push(trf!("{}:{} → 第 {} 行", "{}:{} → line {}", a.line1, a.hash, s0 + 1)); }
                let e0 = match as_str(e, "end_anchor") {
                    Some(ea) if !ea.trim().is_empty() => {
                        let b = parse_anchor(&ea).ok_or_else(|| tr("end_anchor 无效", "invalid end_anchor"))?;
                        let (idx, sh2) = resolve_anchor(&b, &lines)?;
                        if sh2 { shifted_notes.push(trf!("{}:{} → 第 {} 行", "{}:{} → line {}", b.line1, b.hash, idx + 1)); }
                        if idx < s0 {
                            return Err(tr("end_anchor 在 anchor 之前", "end_anchor precedes anchor"));
                        }
                        idx + 1
                    }
                    _ => s0 + 1,
                };
                ops.push(ResolvedOp { start: s0, end: e0, new_lines });
            }
            "insert_after" => {
                let at = match anchor_raw.trim() {
                    "0" | "0:" => 0,
                    "EOF" | "eof" => lines.len(),
                    s => {
                        let a = parse_anchor(s).ok_or_else(|| trf!(
                            "第 {} 个操作的 anchor 无效:\"行号:哈希\"、\"0\"(文件头)或 \"EOF\"(文件尾)",
                            "op {}: invalid anchor — \"LINE:HASH\", \"0\" (BOF) or \"EOF\"",
                            i + 1
                        ))?;
                        let (idx, sh) = resolve_anchor(&a, &lines)?;
                        if sh { shifted_notes.push(trf!("{}:{} → 第 {} 行", "{}:{} → line {}", a.line1, a.hash, idx + 1)); }
                        idx + 1
                    }
                };
                if new_lines.is_empty() {
                    return Err(tr("insert_after 的 content 不能为空", "insert_after needs non-empty content"));
                }
                ops.push(ResolvedOp { start: at, end: at, new_lines });
            }
            other => {
                return Err(trf!(
                    "未知 op \"{other}\":支持 replace / insert_after",
                    "unknown op \"{other}\": supported ops are replace / insert_after"
                ));
            }
        }
    }

    // Overlap check, then bottom-up application so indices stay valid.
    let mut order: Vec<usize> = (0..ops.len()).collect();
    order.sort_by_key(|&i| (ops[i].start, ops[i].end));
    for w in order.windows(2) {
        let (a, b) = (&ops[w[0]], &ops[w[1]]);
        if b.start < a.end {
            return Err(tr("编辑区间重叠,请拆成不重叠的操作", "edit ranges overlap — split into non-overlapping ops"));
        }
    }
    cp_record(&abs);
    let was_clean = syntax_check(&abs).map(|r| r.is_ok());
    let mut new_lines_all: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    for &i in order.iter().rev() {
        let op = &ops[i];
        new_lines_all.splice(op.start..op.end, op.new_lines.iter().cloned());
    }
    let mut updated = new_lines_all.join("\n");
    if had_trailing_nl && !updated.ends_with('\n') {
        updated.push('\n');
    }
    std::fs::write(&abs, updated.as_bytes()).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;

    let root = workspace()?;
    let first = order.first().map(|&i| ops[i].start).unwrap_or(0);
    let span: usize = ops.iter().map(|o| o.new_lines.len().max(1)).sum();
    let shifted = if shifted_notes.is_empty() {
        String::new()
    } else {
        trf!(
            "\n(部分锚点行号已漂移,按内容对齐: {})",
            "\n(some anchors had shifted and were matched by content: {})",
            shifted_notes.join(", ")
        )
    };
    Ok(trf!(
        "已编辑 {}({} 个操作)。{}修改后该区域(含新锚点):\n{}{}",
        "edited {} ({} op(s)).{} The region now reads (fresh anchors included):\n{}{}",
        rel_display(&root, &abs),
        ops.len(),
        shifted,
        anchored_context(&updated, first, span),
        syntax_note(&abs, was_clean)
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
            trf!(
                "\n文件中最相似的位置(从这里逐字复制 old_string):\n{s}",
                "\nclosest match in the file (copy old_string verbatim from here):\n{s}"
            )
        })
        .unwrap_or_default();
    trf!(
        "未找到 old_string（需与文件内容逐字匹配）{hint}",
        "old_string not found — it must match the file content exactly{hint}"
    )
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
        return Err(tr("edits 为空", "no edits given"));
    }
    let abs = resolve(&path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
    let mut cur = text;
    let total = edits.len();
    for (i, e) in edits.iter().enumerate() {
        let n = i + 1;
        if e.old_string.is_empty() {
            return Err(trf!("第 {n}/{total} 条 old_string 为空;未应用任何修改", "edit {n}/{total} has an empty old_string — nothing changed"));
        }
        if e.old_string == e.new_string {
            return Err(trf!("第 {n}/{total} 条 old_string 与 new_string 相同;未应用任何修改", "edit {n}/{total} is a no-op — nothing changed"));
        }
        let count = cur.matches(&e.old_string).count();
        if count == 0 {
            return Err(trf!(
                "第 {n}/{total} 条编辑失败,整个 multi_edit 原子回退、文件未改动:\n{}",
                "edit {n}/{total} failed — multi_edit is atomic, nothing changed:\n{}",
                not_found_error(&cur, &e.old_string)
            ));
        }
        if count > 1 && !e.replace_all {
            return Err(trf!(
                "第 {n}/{total} 条 old_string 出现 {count} 次,不唯一;文件未改动",
                "edit {n}/{total} is not unique ({count} matches) — nothing changed"
            ));
        }
        cur = if e.replace_all {
            cur.replace(&e.old_string, &e.new_string)
        } else {
            cur.replacen(&e.old_string, &e.new_string, 1)
        };
    }
    cp_record(&abs);
    let was_clean = syntax_check(&abs).map(|r| r.is_ok());
    std::fs::write(&abs, cur.as_bytes()).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    let root = workspace()?;
    Ok(trf!(
        "已编辑 {}(应用全部 {total} 处修改){}",
        "edited {} (all {total} edits applied){}",
        rel_display(&root, &abs),
        syntax_note(&abs, was_clean)
    ))
}

// ---- Browser automation tools (CDP; see browser.rs) ----

#[tauri::command]
pub async fn browser_navigate(url: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::navigate(&url))
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))?
}

/// Full-page screenshot (auto-scrolls to trigger lazy content). Returns a temp
/// PNG path the agent loop attaches to the model's next turn (like view_image).
#[tauri::command]
pub async fn browser_screenshot() -> Result<String, String> {
    let png = tokio::task::spawn_blocking(crate::browser::screenshot)
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))??;
    write_shot(png)
}

/// Snapshot of just the current viewport (immediate) — for lazy-load pages,
/// after scrolling. Returns a temp PNG path attached to the next turn.
#[tauri::command]
pub async fn browser_snapshot() -> Result<String, String> {
    let png = tokio::task::spawn_blocking(crate::browser::snapshot)
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))??;
    write_shot(png)
}

/// Scroll the page (to "bottom"/"top" or by pixels) to trigger lazy loading.
#[tauri::command]
pub async fn browser_scroll(to: Option<String>, by: Option<f64>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::scroll_page(to, by))
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))?
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
    std::fs::write(&path, png).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn browser_eval(expression: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || crate::browser::eval(&expression))
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))?
}

#[tauri::command]
pub async fn browser_click(
    selector: Option<String>,
    text: Option<String>,
    // One call, several clicks in order (form flows / wizards).
    steps: Option<Vec<ClickStep>>,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || match steps {
        Some(s) if !s.is_empty() => {
            crate::browser::click_seq(s.into_iter().map(|c| (c.selector, c.text)).collect())
        }
        _ => crate::browser::click(selector, text),
    })
    .await
    .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))?
}

#[derive(serde::Deserialize)]
pub struct ClickStep {
    pub selector: Option<String>,
    pub text: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct TypeStep {
    pub selector: Option<String>,
    pub label: Option<String>,
    pub text: String,
}

#[tauri::command]
pub async fn browser_type(
    selector: Option<String>,
    label: Option<String>,
    text: Option<String>,
    // One call, several fields filled in order (a whole form at once).
    steps: Option<Vec<TypeStep>>,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || match steps {
        Some(s) if !s.is_empty() => {
            crate::browser::type_seq(s.into_iter().map(|t| (t.selector, t.label, t.text)).collect())
        }
        _ => crate::browser::type_text(selector, label, text.unwrap_or_default()),
    })
    .await
    .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))?
}

#[tauri::command]
pub async fn browser_console() -> Result<String, String> {
    tokio::task::spawn_blocking(crate::browser::console)
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))?
}

/// Digest of the current page's interactive elements (links/buttons/inputs).
#[tauri::command]
pub async fn browser_read() -> Result<String, String> {
    tokio::task::spawn_blocking(crate::browser::read_page)
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))?
}

/// Close the automation browser the agent has been driving.
#[tauri::command]
pub async fn browser_close() -> Result<String, String> {
    tokio::task::spawn_blocking(|| {
        crate::browser::shutdown();
        tr("已关闭浏览器", "browser closed")
    })
    .await
    .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))
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
    std::fs::create_dir_all(&dir).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    let html_path = dir.join("canvas.html");
    std::fs::write(&html_path, html).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    let url = format!("file://{}", html_path.display());
    let (png, console) = tokio::task::spawn_blocking(move || crate::browser::capture_headless(&url))
        .await
        .map_err(|e| trf!("浏览器任务异常: {e}", "browser task failed: {e}"))??;
    let png_path = dir.join("canvas-shot.png");
    std::fs::write(&png_path, png).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    Ok(CanvasCapture { image: png_path.to_string_lossy().to_string(), console })
}

/// Resolve a workspace-relative image path to a confined absolute path for the
/// `view_image` tool. Enforces the same sandbox as every other file tool, and
/// checks the file exists and is a supported image — so the vision model only
/// ever sees images from inside the workspace.
/// Extract readable text (and cached embedded images) from a document in the
/// workspace — pdf / docx / xlsx / pptx. `read_file` routes here from the
/// frontend so the Code agent reads documents the same way chat attachments
/// do. Scanned / image-only PDFs with almost no text layer get a few embedded
/// images OCR'd automatically so text-only models still see the content;
/// vision models can additionally `view_image` the cached page images.
#[tauri::command]
pub async fn agent_read_doc(app: tauri::AppHandle, path: String) -> Result<String, String> {
    use tauri::Manager as _;
    let abs = resolve(&path)?;
    let models_dir = app.path().app_data_dir().ok().map(|d| d.join("ocr-models"));
    read_doc_core(abs, models_dir).await
}

/// Testable core of `agent_read_doc` — `models_dir` enables the scanned-PDF
/// OCR fallback (None in contexts without an app handle).
pub(crate) async fn read_doc_core(
    abs: PathBuf,
    models_dir: Option<PathBuf>,
) -> Result<String, String> {
    const DOC_MAX_CHARS: usize = 40_000;

    if !abs.is_file() {
        return Err(trf!("文件不存在: {}", "file not found: {}", abs.display()));
    }
    let ext = abs.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    let abs_str = abs.to_string_lossy().to_string();
    let text = match ext.as_str() {
        "pdf" => {
            let p = abs_str.clone();
            tokio::task::spawn_blocking(move || pdf_extract::extract_text(&p))
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| trf!("PDF 解析失败: {e}", "PDF parse failed: {e}"))?
        }
        "docx" => crate::rag::extract_docx(&abs_str)?,
        "xlsx" => crate::rag::extract_xlsx(&abs_str)?,
        "pptx" => crate::rag::extract_pptx(&abs_str)?,
        _ => {
            return Err(trf!(
                "不支持的文档类型: .{ext} — 文本文件请直接用 read_file",
                "unsupported document type: .{ext} — for plain text use read_file"
            ))
        }
    };

    let images = {
        let p = abs_str.clone();
        tokio::task::spawn_blocking(move || crate::docimg::extract_embedded_images(&p, 8))
            .await
            .unwrap_or_default()
    };

    let mut text = text.trim().to_string();
    let total = text.chars().count();
    let truncated = total > DOC_MAX_CHARS;
    if truncated {
        text = text.chars().take(DOC_MAX_CHARS).collect();
    }

    // Scanned document: the text layer is empty/thin but pages exist as
    // images — OCR a few so the content is readable without vision.
    let mut ocr_note = String::new();
    if total < 200 && !images.is_empty() {
        if let Some(models_dir) = models_dir {
            for (i, img) in images.iter().take(4).enumerate() {
                if let Ok(t) = crate::ocr::ocr_image(models_dir.clone(), img.clone()).await {
                    let t = t.trim().to_string();
                    if !t.is_empty() {
                        ocr_note.push_str(&format!("\n--- 图片 {} OCR ---\n{t}\n", i + 1));
                    }
                }
            }
        }
    }

    let mut out = trf!(
        "[.{ext} 文档已提取 {total} 字符{}]\n{text}",
        "[.{ext} document text extracted, {total} chars{}]\n{text}",
        if truncated { tr(",超出预算已截断", ", truncated to budget") } else { String::new() }
    );
    if !ocr_note.is_empty() {
        out.push_str(&tr(
            "\n\n[文本层极少——已自动 OCR 内嵌图片:]",
            "\n\n[scanned document — embedded images were OCR'd:]",
        ));
        out.push_str(&ocr_note);
    }
    if !images.is_empty() {
        out.push_str(&trf!(
            "\n\n[内嵌图片 ×{} 已缓存——需要看图表/照片内容时,用 view_image 打开这些路径:]\n{}",
            "\n\n[{} embedded image(s) cached — open these paths with view_image to see charts/photos:]\n{}",
            images.len(),
            images.join("\n")
        ));
    }
    Ok(out)
}

#[tauri::command]
pub fn agent_resolve_image(path: String) -> Result<String, String> {
    // Images extracted from documents (agent_read_doc / chat attachments)
    // live in the app's own temp cache — no directory grant needed. Guard
    // against `..` escapes by re-checking the CANONICAL path's prefix.
    let doc_cache = std::env::temp_dir().join("chaty-doc-imgs");
    if let (Ok(canon), Ok(cache_canon)) =
        (std::fs::canonicalize(&path), std::fs::canonicalize(&doc_cache))
    {
        if canon.starts_with(&cache_canon) && canon.is_file() {
            return Ok(canon.to_string_lossy().to_string());
        }
    }
    let abs = resolve(&path)?;
    let ext = abs
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif") {
        return Err(trf!(
            "不是支持的图片格式: {path} — 支持 png/jpg/jpeg/webp/bmp/gif",
            "not a supported image: {path} — png/jpg/jpeg/webp/bmp/gif only"
        ));
    }
    if !abs.is_file() {
        return Err(trf!("图片不存在: {path}", "image not found: {path}"));
    }
    Ok(abs.to_string_lossy().to_string())
}

/// File outline: the definition lines (functions/classes/structs/…) with line
/// numbers, so the model can navigate a big file without reading it whole.
/// Regex-free keyword heuristics that cover Rust/TS/JS/Python/Go/Swift/etc.
#[tauri::command]
pub fn agent_outline(path: String) -> Result<String, String> {
    let abs = resolve(&path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
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
            out.push_str(&tr("… (更多定义已省略)\n", "… (more omitted)\n"));
            break;
        }
    }
    if out.is_empty() {
        return Ok(tr(
            "(未识别到符号定义 — 用 read_file 直接查看)",
            "(no definitions recognized — read the file directly with read_file)",
        ));
    }
    Ok(out)
}

/// `read_file` with `symbol`: the enclosing definition (brace-matched for
/// {}-languages, indentation-scoped for Python), plus every call site in the
/// workspace — target + callers in one round trip.
fn read_symbol_context(path: &str, sym: &str) -> Result<String, String> {
    let root = workspace()?;
    let abs = resolve(path)?;
    let text = std::fs::read_to_string(&abs).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
    let lines: Vec<&str> = text.lines().collect();

    // Locate the definition line: a symbol line that names `sym`.
    let word_hit = |line: &str| -> bool {
        line.match_indices(sym).any(|(i, _)| {
            let before = line[..i].chars().next_back();
            let after = line[i + sym.len()..].chars().next();
            !matches!(before, Some(c) if c.is_alphanumeric() || c == '_')
                && !matches!(after, Some(c) if c.is_alphanumeric() || c == '_')
        })
    };
    let def_idx = lines.iter().position(|l| is_symbol_line(l.trim_start()) && word_hit(l));
    let Some(def_idx) = def_idx else {
        // Help the model recover: list the definitions this file DOES have.
        let mut have = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            if is_symbol_line(l.trim_start()) {
                have.push(format!("  L{}: {}", i + 1, l.trim().chars().take(120).collect::<String>()));
                if have.len() >= 20 {
                    break;
                }
            }
        }
        return Err(trf!(
            "文件里没有名为 {sym} 的定义。该文件的定义有:\n{}",
            "no definition named {sym} in {path}. The file's definitions are:\n{}",
            have.join("\n")
        ));
    };

    // Block extent: brace matching from the def line; if the def line opens no
    // brace (Python etc.), take the indentation-scoped suite instead.
    let def_indent = lines[def_idx].len() - lines[def_idx].trim_start().len();
    let mut end_idx = def_idx;
    let mut depth = 0i32;
    let mut saw_brace = false;
    'outer: for (i, line) in lines.iter().enumerate().skip(def_idx) {
        for ch in line.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    saw_brace = true;
                }
                '}' => {
                    depth -= 1;
                    if saw_brace && depth == 0 {
                        end_idx = i;
                        break 'outer;
                    }
                }
                _ => {}
            }
        }
        if i > def_idx + 400 {
            end_idx = i;
            break;
        }
        end_idx = i;
    }
    if !saw_brace {
        // Indentation scope (def foo(): …) — run until a non-blank line at or
        // below the definition's indentation.
        end_idx = def_idx;
        for (i, line) in lines.iter().enumerate().skip(def_idx + 1) {
            if line.trim().is_empty() {
                end_idx = i;
                continue;
            }
            let ind = line.len() - line.trim_start().len();
            if ind <= def_indent {
                break;
            }
            end_idx = i;
            if i > def_idx + 400 {
                break;
            }
        }
        while end_idx > def_idx && lines[end_idx].trim().is_empty() {
            end_idx -= 1;
        }
    }

    let mut out = trf!("[符号 {sym} · {path} L{}-L{}]\n", "[symbol {sym} · {path} L{}-L{}]\n", def_idx + 1, end_idx + 1);
    for (i, line) in lines.iter().enumerate().take(end_idx + 1).skip(def_idx) {
        out.push_str(&format!("{:>5}  {}\n", i + 1, line.trim_end()));
    }

    // Call sites across the workspace (word-boundary, def line excluded).
    let rel_self = rel_display(&root, &abs);
    let mut callers = Vec::new();
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
            continue;
        }
        let ftext = String::from_utf8_lossy(&bytes);
        if !ftext.contains(sym) {
            continue;
        }
        let frel = rel_display(&root, entry.path());
        for (i, line) in ftext.lines().enumerate() {
            if frel == rel_self && i >= def_idx && i <= end_idx {
                continue; // the definition itself
            }
            if word_hit(line) {
                callers.push(format!("{frel}:{}: {}", i + 1, line.trim().chars().take(140).collect::<String>()));
                if callers.len() >= 12 {
                    break 'walk;
                }
            }
        }
    }
    if callers.is_empty() {
        out.push_str(&tr(
            "\n调用者: 工作区内没有其它引用\n",
            "\ncallers: no other references in the workspace\n",
        ));
    } else {
        out.push_str(&trf!("\n调用者 ({} 处):\n{}\n", "\ncallers ({}):\n{}\n", callers.len(), callers.join("\n")));
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
    let rd = std::fs::read_dir(&abs).map_err(|e| trf!("列目录失败: {e}", "list failed: {e}"))?;
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
    let full = full.to_str().ok_or_else(|| tr("无效的模式", "invalid pattern"))?;
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
    Ok(trf!(
        "已回滚：恢复 {restored} 个文件，删除 {removed} 个新建文件",
        "reverted: {restored} file(s) restored, {removed} new file(s) removed"
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


/// Ranked code search: BM25 over line-window chunks, AGGREGATED PER FILE and
/// fused with a filename-match signal and an exact-phrase boost, then dressed
/// with the file's matching definition lines (same heuristic as `outline`).
/// One call answers "which files handle X, and through which functions?" —
/// the decide/filter work small models used to do across many grep rounds.
#[tauri::command]
pub fn agent_search_code(query: String, k: Option<usize>) -> Result<String, String> {
    let root = workspace()?;
    let q_tokens = code_tokens(&query);
    if q_tokens.is_empty() {
        return Err(tr("查询为空", "empty query"));
    }
    let q_lower = query.to_lowercase();
    let top_files = k.unwrap_or(6).clamp(1, 20);

    struct Chunk {
        file: usize,
        line: usize,
        text: String,
        tf: HashMap<String, u32>,
        len: u32,
    }
    let mut files: Vec<(String, String)> = Vec::new(); // (rel, full text)
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
        let text = String::from_utf8_lossy(&bytes).to_string();
        scanned += text.len();
        let rel = rel_display(&root, entry.path());
        let fid = files.len();
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
                chunks.push(Chunk { file: fid, line: start + 1, text: body, len: toks.len() as u32, tf });
            }
            if end == lines.len() {
                break;
            }
            start = end - SEARCH_CHUNK_OVERLAP;
        }
        files.push((rel, text));
        if scanned > SEARCH_MAX_TOTAL {
            break 'walk;
        }
    }

    if chunks.is_empty() {
        return Ok(tr("(没有匹配的代码)", "(no matches)"));
    }
    let n = chunks.len() as f32;
    let avg_len: f32 = chunks.iter().map(|c| c.len as f32).sum::<f32>() / n;
    let (k1, b) = (1.4f32, 0.75f32);
    let score_of = |c: &Chunk| -> f32 {
        let mut score = 0f32;
        for t in &q_tokens {
            let Some(&tf) = c.tf.get(t) else { continue };
            let dfi = *df.get(t).unwrap_or(&1) as f32;
            let idf = ((n - dfi + 0.5) / (dfi + 0.5) + 1.0).ln();
            let tf = tf as f32;
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * c.len as f32 / avg_len));
        }
        score
    };

    // Per-file aggregation: sum of the two best chunks + fusion signals.
    struct FileRank {
        best: Option<usize>, // index of best chunk
        top2: [f32; 2],
        hits: usize,
    }
    let mut ranks: Vec<FileRank> = files.iter().map(|_| FileRank { best: None, top2: [0.0; 2], hits: 0 }).collect();
    for (i, c) in chunks.iter().enumerate() {
        let s = score_of(c);
        if s <= 0.0 {
            continue;
        }
        let r = &mut ranks[c.file];
        r.hits += 1;
        if s > r.top2[0] {
            r.top2[1] = r.top2[0];
            r.top2[0] = s;
            r.best = Some(i);
        } else if s > r.top2[1] {
            r.top2[1] = s;
        }
    }
    let max_chunk = ranks.iter().map(|r| r.top2[0]).fold(0f32, f32::max).max(f32::EPSILON);

    let mut scored: Vec<(usize, f32, bool, bool)> = Vec::new(); // (fid, score, name_hit, exact_hit)
    for (fid, r) in ranks.iter().enumerate() {
        let mut score = r.top2[0] + r.top2[1];
        // Filename signal: query tokens appearing in the path outrank body-only
        // matches ("auth" should surface auth.ts even with sparse text hits).
        let name_toks = code_tokens(&files[fid].0);
        let name_hit = q_tokens.iter().any(|t| name_toks.contains(t));
        if name_hit {
            score += 0.6 * max_chunk;
        }
        // Exact-phrase boost: a literal (case-insensitive) occurrence of the
        // whole query is grep-grade evidence.
        let exact_hit = q_tokens.len() > 1 && files[fid].1.to_lowercase().contains(&q_lower);
        if exact_hit {
            score += 0.8 * max_chunk;
        }
        if score > 0.0 {
            scored.push((fid, score, name_hit, exact_hit));
        }
    }
    if scored.is_empty() {
        return Ok(tr("(没有匹配的代码)", "(no matches)"));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_files);

    let mut out = tr("相关文件 (ranked):\n", "relevant files (ranked):\n");
    for (rank, (fid, score, name_hit, exact_hit)) in scored.iter().enumerate() {
        let (rel, text) = &files[*fid];
        let mut tags = Vec::new();
        if *name_hit {
            tags.push(tr("文件名匹配", "name match"));
        }
        if *exact_hit {
            tags.push(tr("精确短语", "exact phrase"));
        }
        let tag_str = if tags.is_empty() { String::new() } else { format!(" · {}", tags.join(" · ")) };
        out.push_str(&trf!(
            "\n{}. {rel}  (score {:.2} · {} 处命中{tag_str})\n",
            "\n{}. {rel}  (score {:.2} · {} hit(s){tag_str})\n",
            rank + 1,
            score,
            ranks[*fid].hits
        ));
        // Matching definition lines — the file's API surface for this query.
        let mut syms = 0;
        for (i, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if !is_symbol_line(t) {
                continue;
            }
            let lt = t.to_lowercase();
            if q_tokens.iter().any(|q| lt.contains(q.as_str())) {
                out.push_str(&format!("   定义 L{}: {}\n", i + 1, t.chars().take(140).collect::<String>()));
                syms += 1;
                if syms >= 4 {
                    break;
                }
            }
        }
        if let Some(ci) = ranks[*fid].best {
            let c = &chunks[ci];
            out.push_str(&format!(
                "   ── {rel}:{} ──\n{}\n",
                c.line,
                c.text.chars().take(500).collect::<String>()
            ));
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
    let re = regex::Regex::new(&pattern).map_err(|e| trf!("正则无效: {e}", "bad regex: {e}"))?;
    let glob_matcher = match glob {
        Some(g) if !g.trim().is_empty() => Some(
            glob::Pattern::new(&g).map_err(|e| trf!("glob 无效: {e}", "bad glob: {e}"))?,
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
                    out.push_str(&tr("… (更多结果已省略)\n", "… (more matches omitted)\n"));
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
        return Err(tr("query 为空", "empty query"));
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
                    content_out.push_str(&tr("… (更多结果已省略)\n", "… (more matches omitted)\n"));
                    break 'walk;
                }
            }
        }
    }

    if name_hits.is_empty() && content_n == 0 {
        return Ok(tr("(无匹配)", "(no matches)"));
    }

    let mut out = String::new();
    if !name_hits.is_empty() {
        out.push_str(&trf!("文件名匹配 ({} 个):\n", "file-name matches ({}):\n", name_hits.len()));
        for h in &name_hits {
            out.push_str(&format!("  {h}\n"));
        }
        if name_capped {
            out.push_str(&tr("  … (更多文件名已省略)\n", "  … (more names omitted)\n"));
        }
    }
    if contents {
        out.push_str(&trf!(
            "\n内容匹配{}:\n{}",
            "\nfile-content matches{}:\n{}",
            if content_n == 0 { tr(", 无", ", none") } else { String::new() },
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
/// re-allow only inside the workspace, the user's session-granted directories,
/// and the standard temp dirs. Reads, exec and network stay available (agents
/// need `git`, `npm`, compilers). Built per bash call, so a fresh grant applies
/// to the next command immediately.
#[cfg(target_os = "macos")]
fn seatbelt_profile(root: &Path) -> String {
    let grants: String = GRANTED_DIRS
        .lock()
        .unwrap()
        .iter()
        .map(|d| format!(" (subpath \"{}\")", d.display()))
        .collect();
    format!(
        "(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write* \
         (subpath \"{root}\"){grants} (subpath \"/private/tmp\") (subpath \"/tmp\") \
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
    let slice = &bytes[..bytes.len().min(MAX_OUTPUT_BYTES)];
    // Chinese-locale Windows consoles emit the ANSI codepage (GBK), not
    // UTF-8 — `dir`, error messages etc. turned to mojibake under a plain
    // lossy conversion. Valid UTF-8 passes through; otherwise decode as GBK.
    #[cfg(windows)]
    let s = match std::str::from_utf8(slice) {
        Ok(ok) => ok.to_owned(),
        Err(_) => encoding_rs::GBK.decode(slice).0.into_owned(),
    };
    #[cfg(not(windows))]
    let s = String::from_utf8_lossy(slice).into_owned();
    if truncated {
        trf!("{s}\n… (输出已截断)", "{s}\n… (output truncated)")
    } else {
        s
    }
}

/// Does this shell command invoke `sudo` (as a command word, not a substring
/// of an unrelated token like "sudoers")?
pub(crate) fn command_uses_sudo(command: &str) -> bool {
    let b = command.as_bytes();
    let mut i = 0;
    while let Some(pos) = command[i..].find("sudo") {
        let s = i + pos;
        let before_ok = s == 0 || matches!(b[s - 1], b' ' | b'\t' | b'\n' | b';' | b'&' | b'|' | b'(');
        let after = s + 4;
        let after_ok = after >= b.len() || matches!(b[after], b' ' | b'\t' | b'\n');
        if before_ok && after_ok {
            return true;
        }
        i = s + 4;
    }
    false
}

fn run_bash(
    root: &Path,
    command: &str,
    timeout: Duration,
    // Some(password) for an approved sudo run: fed to `sudo -S` on stdin (never
    // in argv / the command string / logs) and the command runs UN-sandboxed
    // (a privileged action the user explicitly approved — it can't work inside
    // the seatbelt write-jail anyway).
    stdin: Option<String>,
    sandboxed: bool,
) -> Result<BashResult, String> {
    let mut cmd = build_command(root, command, sandboxed);
    cmd.current_dir(root)
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| trf!("启动命令失败: {e}", "spawn failed: {e}"))?;
    if let Some(pw) = stdin {
        if let Some(mut sink) = child.stdin.take() {
            use std::io::Write as _;
            let _ = sink.write_all(pw.as_bytes());
            // drop `sink` → EOF, so `sudo -S` stops waiting for more input.
        }
    }
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
            Err(e) => return Err(trf!("等待命令失败: {e}", "wait failed: {e}")),
        }
    };
    let stdout = cap_utf8(out_h.join().unwrap_or_default());
    let stderr = cap_utf8(err_h.join().unwrap_or_default());
    Ok(BashResult { stdout, stderr, code, timed_out })
}

#[cfg(target_os = "macos")]
fn build_command(root: &Path, command: &str, sandboxed: bool) -> Command {
    let mut cmd = if sandboxed {
        let mut c = Command::new("/usr/bin/sandbox-exec");
        c.arg("-p").arg(seatbelt_profile(root)).arg("/bin/sh").arg("-c").arg(command);
        c
    } else {
        // Un-sandboxed (approved sudo): a privileged action can't run in the
        // write-jail. Confinement is the explicit user approval.
        let mut c = Command::new("/bin/sh");
        c.arg("-c").arg(command);
        c
    };
    cmd.env("PATH", augmented_path());
    cmd
}

#[cfg(all(unix, not(target_os = "macos")))]
fn build_command(_root: &Path, command: &str, _sandboxed: bool) -> Command {
    // No seatbelt off macOS — confinement is the working directory + approval.
    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c").arg(command);
    cmd.env("PATH", augmented_path());
    cmd
}

#[cfg(windows)]
fn build_command(_root: &Path, command: &str, _sandboxed: bool) -> Command {
    let mut cmd = Command::new("cmd");
    cmd.arg("/C").arg(command);
    hide_console(&mut cmd); // every agent step would flash a console otherwise
    cmd
}

/// If `command` runs sudo, make it read its password from stdin (`sudo -S`) so
/// an approved password can be piped in non-interactively. Returns the possibly
/// rewritten command (unchanged when there's no sudo or `-S` is already there).
fn ensure_sudo_stdin(command: &str) -> String {
    if !command_uses_sudo(command) || command.contains("sudo -S") || command.contains("sudo --stdin") {
        return command.to_string();
    }
    command.replacen("sudo ", "sudo -S ", 1)
}

/// Run a shell command inside the workspace. On macOS it is sandboxed (writes
/// confined to the workspace); elsewhere it runs in the workspace dir. The
/// frontend gates this behind per-command approval (or bypass mode).
///
/// `sudo_password`: present only when the user approved a `sudo` command in the
/// dedicated dialog and entered a password — it is piped to `sudo -S` on stdin
/// (never placed in argv, the command string, or any log), and such a command
/// runs UN-sandboxed since a privileged action can't work in the write-jail.
#[tauri::command]
pub async fn agent_bash(
    command: String,
    timeout_secs: Option<u64>,
    sudo_password: Option<String>,
) -> Result<BashResult, String> {
    let root = workspace()?;
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(120).clamp(1, 600));
    let is_sudo = command_uses_sudo(&command);
    let (command, stdin, sandboxed) = if is_sudo {
        let cmd = if sudo_password.is_some() { ensure_sudo_stdin(&command) } else { command };
        let stdin = sudo_password.map(|p| if p.ends_with('\n') { p } else { format!("{p}\n") });
        (cmd, stdin, false)
    } else {
        (command, None, true)
    };
    let piped_password = stdin.is_some();
    let mut res = tokio::task::spawn_blocking(move || run_bash(&root, &command, timeout, stdin, sandboxed))
        .await
        .map_err(|e| trf!("命令任务异常: {e}", "command task panicked: {e}"))??;
    // sudo's retry loop hits end-of-input after a rejected password, so its
    // LAST stderr line is "no password was provided" — which reads as if the
    // password never arrived. When we know it did, say what actually happened.
    if piped_password && res.code != 0 && res.stderr.contains("no password was provided") {
        res.stderr.push_str(&tr(
            "\n[Chaty] 密码已通过安全通道送达,但被 sudo 拒绝——上面的 \"no password was provided\" 只是 sudo 重试时读到输入结束的提示。请检查密码是否正确后重试。",
            "\n[Chaty] The password WAS delivered over the secure channel but sudo rejected it — the \"no password was provided\" line above is just sudo hitting end-of-input on retry. Check the password and try again.",
        ));
    }
    Ok(res)
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
    // Background jobs always run sandboxed with no stdin — an approved sudo
    // password could never reach them, so sudo would always fail with a
    // misleading "no password was provided". Refuse up front.
    if command_uses_sudo(&command) {
        return Err(tr(
            "bash_bg 不支持 sudo:后台任务在沙盒中运行且没有交互输入,密码无法送达。请用前台 bash 执行。",
            "sudo is not supported in background jobs (sandboxed, no interactive input — the password can't reach it). Run it with the foreground bash tool.",
        ));
    }
    let root = workspace()?;
    let mut reg = BG_JOBS.lock().unwrap();
    let jobs = reg.get_or_insert_with(HashMap::new);
    let running = jobs.values().filter(|j| j.code.is_none()).count();
    if running >= BG_MAX_JOBS {
        return Err(trf!(
            "后台命令过多（{running} 个在跑），请先用 bg_kill 结束一些",
            "too many background jobs ({running} running) — bg_kill some first"
        ));
    }

    // Background jobs (dev servers, builds) always run sandboxed — sudo isn't
    // meaningful for a long-running process and would need an interactive tty.
    let mut cmd = build_command(&root, &command, true);
    cmd.current_dir(&root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    // Own process group so bg_kill can take down the whole tree (npm → node …).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd.spawn().map_err(|e| trf!("启动命令失败: {e}", "spawn failed: {e}"))?;
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
        .ok_or_else(|| trf!("没有这个后台命令: #{id}", "no such background job: #{id}"))?;
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
        .ok_or_else(|| trf!("没有这个后台命令: #{id}", "no such background job: #{id}"))?;
    if job.code.is_none() {
        bg_kill_pid(job.pid);
        job.reported = true; // killed on request → no completion notice needed
        return Ok(trf!("已终止后台命令 #{id}", "killed background job #{id}"));
    }
    Ok(trf!("后台命令 #{id} 已经结束", "background job #{id} already finished"))
}

/// Kill a job's whole process group (unix) / tree (windows).
fn bg_kill_pid(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let mut cmd = std::process::Command::new("taskkill");
        cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
        let _ = hide_console(&mut cmd).output();
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

    #[test]
    fn hashline_anchor_basics() {
        // Whitespace-normalized: indentation and internal runs don't matter.
        assert_eq!(line_hash("  return x  "), line_hash("return    x"));
        assert_ne!(line_hash("return x"), line_hash("returnx"));
        let a = anchor_of(22, "return x");
        assert!(a.starts_with("22:") && a.len() == 22.to_string().len() + 4);
        let p = parse_anchor(&a).unwrap();
        assert_eq!(p.line1, 22);
        assert!(parse_anchor("0:abc").is_none());
        assert!(parse_anchor("x:abc").is_none());
        assert!(parse_anchor("5:ab1").is_none());
        // Uppercase hashes are tolerated (normalized down).
        assert!(parse_anchor("5:ABC").is_some());
        // A whole pasted read_file line is tolerated — content after the
        // arrow (or whitespace) is redundant, not wrong.
        let p = parse_anchor("142:qzh→            user.save()").unwrap();
        assert_eq!((p.line1, p.hash.as_str()), (142, "qzh"));
        let p = parse_anchor("7:abc  some trailing text").unwrap();
        assert_eq!(p.line1, 7);
    }

    #[test]
    fn hashline_resolve_shift_and_ambiguity() {
        let lines = vec!["alpha", "beta", "gamma", "beta", "delta"];
        // Exact hit.
        let h = encode_line_hash(line_hash("gamma"));
        let (idx, shifted) = resolve_anchor(&ParsedAnchor { line1: 3, hash: h }, &lines).unwrap();
        assert_eq!((idx, shifted), (2, false));
        // Shifted but unique.
        let h = encode_line_hash(line_hash("delta"));
        let (idx, shifted) = resolve_anchor(&ParsedAnchor { line1: 3, hash: h }, &lines).unwrap();
        assert_eq!((idx, shifted), (4, true));
        // Ambiguous: two "beta" lines within radius.
        let h = encode_line_hash(line_hash("beta"));
        let err = resolve_anchor(&ParsedAnchor { line1: 3, hash: h }, &lines).unwrap_err();
        assert!(err.contains("歧义") || err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn hashline_edit_lines_end_to_end() {
        let _g = serial();
        let dir = std::env::temp_dir().join(format!("chaty-agent-hl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        set_ws(&dir);
        let f = dir.join("a.txt");
        std::fs::write(&f, "one\ntwo\nthree\nfour\n").unwrap();
        let a2 = anchor_of(2, "two");
        let a3 = anchor_of(3, "three");
        // replace range [2..3] + insert at EOF, applied bottom-up in one batch.
        let edits = serde_json::json!([
            { "op": "replace", "anchor": a2, "end_anchor": a3, "content": "TWO\nTHREE" },
            { "op": "insert_after", "anchor": "EOF", "content": "five" },
        ]);
        let out = agent_edit_lines("a.txt".into(), edits).unwrap();
        assert!(out.contains("2 个操作") || out.contains("2 op(s)"), "{out}");
        assert!(out.contains('→'), "fresh anchors expected: {out}");
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "one\nTWO\nTHREE\nfour\nfive\n");
        // Stale anchor after the file changed → clear error.
        let stale = serde_json::json!([{ "op": "replace", "anchor": a2, "content": "x" }]);
        let err = agent_edit_lines("a.txt".into(), stale).unwrap_err();
        assert!(err.contains("不匹配") || err.contains("does not match"), "{err}");
        // Content carrying pasted anchor prefixes is rejected.
        let a1 = anchor_of(1, "one");
        let paste = serde_json::json!([{ "op": "replace", "anchor": a1, "content": "5:abc→one" }]);
        let err = agent_edit_lines("a.txt".into(), paste).unwrap_err();
        assert!(err.contains("锚点前缀") || err.contains("anchor prefix"), "{err}");
        // Overlapping ranges are rejected.
        let a1 = anchor_of(1, "one");
        let a4 = anchor_of(4, "four");
        let overlap = serde_json::json!([
            { "op": "replace", "anchor": a1, "end_anchor": a4, "content": "x" },
            { "op": "replace", "anchor": a4, "content": "y" },
        ]);
        let err = agent_edit_lines("a.txt".into(), overlap).unwrap_err();
        assert!(err.contains("重叠") || err.contains("overlap"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hashline_read_prefixes_when_enabled() {
        let _g = serial();
        let dir = std::env::temp_dir().join(format!("chaty-agent-hlr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        set_ws(&dir);
        std::fs::write(dir.join("b.txt"), "hello\nworld\n").unwrap();
        agent_set_edit_anchors(true);
        let out = agent_read_file("b.txt".into(), None, None, None, None).unwrap();
        agent_set_edit_anchors(false);
        assert!(out.lines().next().unwrap().starts_with("1:"), "{out}");
        assert!(out.contains("→hello"), "{out}");
        let plain = agent_read_file("b.txt".into(), None, None, None, None).unwrap();
        assert!(plain.starts_with("hello"), "{plain}");
    }

    /// Tests share the global WORKSPACE, so they must not run concurrently.
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Model-visible strings render in ONE language, driven by the session
    /// switch. Default stays Zh (every other test asserts Chinese output);
    /// this test flips to En, checks a write round-trip, and flips back.
    #[test]
    fn tool_output_language_switch() {
        let _g = serial();
        let dir = std::env::temp_dir().join(format!("chaty-agent-lang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        set_ws(&dir);
        agent_set_lang("en".into());
        let msg = agent_write_file("hello.txt".into(), "hi".into()).unwrap();
        assert!(msg.starts_with("wrote "), "en output expected, got: {msg}");
        assert!(!msg.contains("已写入"));
        let err = agent_edit_file("hello.txt".into(), "nope".into(), "x".into(), None).unwrap_err();
        assert!(err.contains("old_string not found"), "en error expected, got: {err}");
        agent_set_lang("zh".into());
        let msg = agent_write_file("hello2.txt".into(), "hi".into()).unwrap();
        assert!(msg.starts_with("已写入"), "zh output expected after switch back, got: {msg}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Windows consoles emit the ANSI codepage (GBK on Chinese systems) — the
    /// old lossy-UTF-8 decode turned every non-ASCII byte into mojibake.
    #[test]
    fn cap_utf8_decodes_console_output() {
        // Plain UTF-8 passes through unchanged on every platform.
        assert_eq!(cap_utf8("hello 世界".as_bytes().to_vec()), "hello 世界");
        #[cfg(windows)]
        {
            // "找不到文件" (file not found) as GBK bytes — what a Chinese-locale
            // cmd.exe actually writes.
            let (gbk, _, _) = encoding_rs::GBK.encode("找不到文件 test");
            assert!(std::str::from_utf8(&gbk).is_err(), "fixture must not be valid UTF-8");
            assert_eq!(cap_utf8(gbk.into_owned()), "找不到文件 test");
        }
    }

    /// The post-edit syntax gate must never flag a VALID .py file as broken —
    /// on Windows this guards the python-vs-python3 selection (the Microsoft
    /// Store stub named python3 exits non-zero with a store hint). Passes as
    /// Ok on machines with a real Python, and as "no checker" without one;
    /// an Err on valid code is the bug.
    #[test]
    fn syntax_gate_accepts_valid_python() {
        let tmp = std::env::temp_dir().join(format!("chaty-agent-pyok-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let f = tmp.join("ok.py");
        std::fs::write(&f, "def add(a, b):\n    return a + b\n").unwrap();
        match syntax_check(&f) {
            Some(Err(e)) => panic!("valid python flagged as broken: {e}"),
            _ => {} // Ok(()) with a Python installed, None without — both fine
        }
        let _ = std::fs::remove_dir_all(&tmp);
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
    fn out_of_workspace_access_asks_then_grants_then_revokes() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-grant-ws-{}", std::process::id()));
        let outside = std::env::temp_dir().join(format!("chaty-grant-out-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(outside.join("sub")).unwrap();
        std::fs::write(outside.join("sub/data.txt"), "outside-content").unwrap();
        set_ws(&tmp);
        agent_clear_grants();

        // 1. Out-of-workspace access returns the ASK marker (not a flat error),
        //    carrying the directory to grant.
        let target = outside.join("sub/data.txt");
        let err = resolve(&target.to_string_lossy()).unwrap_err();
        assert!(err.starts_with(NEED_DIR_GRANT), "marker missing: {err}");
        let dir_in_err = err.split('\t').nth(1).unwrap();
        assert!(
            PathBuf::from(dir_in_err).ends_with("sub"),
            "suggested dir should be the file's parent: {dir_in_err}"
        );

        // 2. Granting the dir makes the same path resolve (and stay confined
        //    to the grant).
        let granted = agent_grant_dir(outside.to_string_lossy().to_string()).expect("grant");
        assert!(agent_list_grants().contains(&granted));
        let ok = resolve(&target.to_string_lossy()).expect("resolves after grant");
        assert!(ok.ends_with("sub/data.txt"));
        // a real read through the tool works too
        assert!(std::fs::read_to_string(&ok).unwrap().contains("outside-content"));
        // …but a path outside BOTH workspace and grant still asks
        assert!(resolve("/etc/hosts").unwrap_err().starts_with(NEED_DIR_GRANT));

        // 3. One-click revoke → back to asking.
        agent_revoke_dir(granted.clone());
        assert!(agent_list_grants().is_empty());
        assert!(resolve(&target.to_string_lossy()).unwrap_err().starts_with(NEED_DIR_GRANT));

        // 4. Grants are cleared when the workspace changes.
        let _ = agent_grant_dir(outside.to_string_lossy().to_string());
        assert!(!agent_list_grants().is_empty());
        let other = std::env::temp_dir().join(format!("chaty-grant-ws2-{}", std::process::id()));
        std::fs::create_dir_all(&other).unwrap();
        let _ = agent_set_workspace(other.to_string_lossy().to_string());
        assert!(agent_list_grants().is_empty(), "workspace switch must clear grants");

        // 5. Granting a non-directory is rejected.
        assert!(agent_grant_dir(outside.join("sub/data.txt").to_string_lossy().to_string()).is_err());

        agent_clear_grants();
        std::fs::remove_dir_all(&tmp).ok();
        std::fs::remove_dir_all(&outside).ok();
        std::fs::remove_dir_all(&other).ok();
    }

    #[test]
    fn write_read_edit_roundtrip() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-rw-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        agent_write_file("sub/hi.txt".into(), "hello world\nsecond".into()).unwrap();
        let read = agent_read_file("sub/hi.txt".into(), None, None, None, None).unwrap();
        assert!(read.contains("hello world"));

        // unique edit
        agent_edit_file("sub/hi.txt".into(), "hello".into(), "hi".into(), None).unwrap();
        assert!(agent_read_file("sub/hi.txt".into(), None, None, None, None).unwrap().starts_with("hi world"));

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
        let full = agent_read_file("big.txt".into(), None, None, None, None).unwrap();
        assert!(full.contains("line 1000"));
        assert!(!full.contains("offset="));

        // Only when the char budget genuinely can't hold the file does it page,
        // and the footer must carry a FOLLOWABLE offset.
        let page1 = agent_read_file("big.txt".into(), None, None, Some(4000), None).unwrap();
        assert!(page1.contains("offset="));
        let tail = page1.rsplit("offset=").next().unwrap();
        let next: usize =
            tail.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap();
        let page2 = agent_read_file("big.txt".into(), Some(next), None, Some(60_000), None).unwrap();
        assert!(page2.starts_with(&format!("line {next}")));
        assert!(page2.contains("line 1000"));
        assert!(!page2.contains("offset="));

        // Small file: no footer at all.
        std::fs::write(tmp.join("small.txt"), "hello\nworld\n").unwrap();
        let small = agent_read_file("small.txt".into(), None, None, None, None).unwrap();
        assert!(!small.contains("offset="));

        // A full-file diff snapshot (max_chars = 400_000) reads a large file
        // WHOLE with no footer — so the diff isn't polluted by pagination text
        // and its line counts are correct. ~100 KB / 2500 lines.
        let big: String = (1..=2500).map(|i| format!("content line number {i}\n")).collect();
        std::fs::write(tmp.join("huge.txt"), &big).unwrap();
        let full_snapshot = agent_read_file("huge.txt".into(), None, None, Some(400_000), None).unwrap();
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
        let out = agent_search_code("user name login".into(), Some(5)).unwrap();
        assert!(out.contains("1. auth.ts"), "auth.ts must rank first:\n{out}");
        assert!(out.contains("getUserName"), "snippet/definitions missing:\n{out}");
        // Off-topic query prefers the other file.
        let out2 = agent_search_code("database pool".into(), Some(5)).unwrap();
        assert!(out2.contains("1. db.ts"), "db.ts must rank first:\n{out2}");

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

    #[test]
    fn detects_sudo_and_rewrites_for_stdin() {
        assert!(command_uses_sudo("sudo apt-get install x"));
        assert!(command_uses_sudo("echo hi && sudo rm -rf /tmp/x"));
        assert!(command_uses_sudo("(sudo -k)"));
        assert!(command_uses_sudo("sudo"));
        // Not sudo: substrings / unrelated tokens.
        assert!(!command_uses_sudo("ls /etc/sudoers"));
        assert!(!command_uses_sudo("pseudo run"));
        assert!(!command_uses_sudo("cat sudo.txt"));
        assert!(!command_uses_sudo("echo sudoku"));
        // Rewrite adds -S once, and only when needed.
        assert_eq!(ensure_sudo_stdin("sudo apt-get install x"), "sudo -S apt-get install x");
        assert_eq!(ensure_sudo_stdin("sudo -S already"), "sudo -S already");
        assert_eq!(ensure_sudo_stdin("ls -la"), "ls -la");
    }

    /// validate_change on a real mini pytest project: the related test file is
    /// discovered by content, the failing assertion is summarized, and the
    /// fixed version passes. The unrelated test file must NOT be selected.
    #[test]
    fn validate_change_runs_related_pytest() {
        let _g = serial();
        if std::process::Command::new("python3")
            .args(["-m", "pytest", "--version"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("SKIP: pytest not available");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("chaty-agent-vc-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        std::fs::write(tmp.join("calc.py"), "def add(a, b):\n    return a - b\n").unwrap();
        std::fs::write(
            tmp.join("test_calc.py"),
            "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("test_other.py"),
            "def test_unrelated():\n    assert True\n",
        )
        .unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let out = rt
            .block_on(agent_validate_change(Some(vec!["calc.py".into()])))
            .expect("validate");
        eprintln!("{out}");
        assert!(out.contains("test_calc.py"), "related test not selected:\n{out}");
        assert!(!out.contains("test_other.py"), "unrelated test selected:\n{out}");
        assert!(out.contains("✗ 失败"), "failure not reported:\n{out}");

        std::fs::write(tmp.join("calc.py"), "def add(a, b):\n    return a + b\n").unwrap();
        let out = rt
            .block_on(agent_validate_change(Some(vec!["calc.py".into()])))
            .expect("validate 2");
        assert!(out.contains("✓ 通过"), "fixed code must pass:\n{out}");

        // No tracked changes and no files → instructive message.
        cp_clear();
        let out = rt.block_on(agent_validate_change(None)).expect("validate 3");
        assert!(out.contains("没有记录到文件改动"), "empty-state message missing: {out}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// understand_repo must assemble README lede, manifest line, tree,
    /// language census and entry candidates from a synthesized repo.
    #[test]
    fn understand_repo_builds_digest() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-ur-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        set_ws(&tmp);
        std::fs::write(tmp.join("README.md"), "# Demo\n\nA tiny demo service.\n").unwrap();
        std::fs::write(
            tmp.join("package.json"),
            r#"{"name":"demo","scripts":{"dev":"vite","test":"vitest run"}}"#,
        )
        .unwrap();
        std::fs::write(tmp.join("src/index.ts"), "export const x = 1;\n").unwrap();
        std::fs::write(tmp.join("src/util.ts"), "export const y = 2;\n").unwrap();
        std::fs::write(tmp.join("src/main.py"), "print('hi')\n").unwrap();

        let out = agent_understand_repo().expect("digest");
        eprintln!("{out}");
        assert!(out.contains("A tiny demo service"), "README lede missing:\n{out}");
        assert!(out.contains("name=demo"), "manifest missing:\n{out}");
        assert!(out.contains("dev, test") || out.contains("test, dev"), "scripts missing:\n{out}");
        assert!(out.contains("src/ (3)"), "tree missing:\n{out}");
        assert!(out.contains(".ts×2"), "census missing:\n{out}");
        assert!(out.contains("src/index.ts"), "entry candidate missing:\n{out}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Real-vitest probe (fixture project prepared outside the test):
    ///   CHATY_TEST_VC_JS=<fixture dir with package.json+vitest installed> \
    ///   cargo test --lib validate_change_js_probe -- --ignored --nocapture
    /// The fixture's src/math.js is broken (add = a - b); the related
    /// src/math.test.js must be selected (other.test.js not), fail, then pass
    /// after the fix.
    #[test]
    #[ignore]
    fn validate_change_js_probe() {
        let _g = serial();
        let dir = std::env::var("CHATY_TEST_VC_JS").expect("set CHATY_TEST_VC_JS=<fixture dir>");
        set_ws(std::path::Path::new(&dir));
        // Reset the fixture to its broken state.
        std::fs::write(
            std::path::Path::new(&dir).join("src/math.js"),
            "export function add(a, b) { return a - b; }\n",
        )
        .unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let out = rt
            .block_on(agent_validate_change(Some(vec!["src/math.js".into()])))
            .expect("validate js");
        eprintln!("{out}");
        assert!(out.contains("vitest"), "vitest runner not picked: {out}");
        assert!(out.contains("math.test.js"), "related test not selected: {out}");
        assert!(!out.contains("other.test.js"), "unrelated test selected: {out}");
        assert!(out.contains("✗ 失败"), "failure not reported: {out}");

        std::fs::write(
            std::path::Path::new(&dir).join("src/math.js"),
            "export function add(a, b) { return a + b; }\n",
        )
        .unwrap();
        let out = rt
            .block_on(agent_validate_change(Some(vec!["src/math.js".into()])))
            .expect("validate js 2");
        assert!(out.contains("✓ 通过"), "fixed code must pass: {out}");
    }

    /// The syntax gate: breaking a previously-parsable file must produce the
    /// loud note; fixing it back must stay silent; a fresh broken file gets
    /// the generic warning; uncheckable types are untouched.
    #[test]
    fn edits_run_the_syntax_gate() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-syn-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        std::fs::write(tmp.join("cfg.json"), b"{\"a\": 1}\n").unwrap();
        // Break it: the note must say the edit broke a clean file.
        let msg = agent_edit_file("cfg.json".into(), "1}".into(), "1,}".into(), None).unwrap();
        assert!(msg.contains("语法检查失败"), "broken JSON must warn: {msg}");
        assert!(msg.contains("改坏了"), "should call out breaking a clean file: {msg}");
        // Fix it back: silent.
        let msg = agent_edit_file("cfg.json".into(), "1,}".into(), "1}".into(), None).unwrap();
        assert!(!msg.contains("语法检查失败"), "clean edit must be silent: {msg}");

        // Fresh broken python via write_file → generic warning (python3 present
        // on dev machines; skip the assert if not).
        if std::process::Command::new("python3").arg("--version").output().is_ok() {
            let msg = agent_write_file("t.py".into(), "def broken(:\n    pass\n".into()).unwrap();
            assert!(msg.contains("语法检查失败"), "broken py must warn: {msg}");
            assert!(!msg.contains("改坏了"), "fresh file isn't 'broken by this edit': {msg}");
            let msg = agent_write_file("t.py".into(), "def ok():\n    pass\n".into()).unwrap();
            assert!(!msg.contains("语法检查失败"), "valid py must be silent: {msg}");
            assert!(!tmp.join("__pycache__").exists(), "checker must not drop __pycache__");
        }

        // Uncheckable type: no note ever.
        let msg = agent_write_file("notes.txt".into(), "anything at all ((".into()).unwrap();
        assert!(!msg.contains("语法检查"), "txt must not be checked: {msg}");
        cp_clear();
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// read_file's symbol mode: brace-matched block for {}-languages,
    /// indentation suite for Python, callers across the workspace, and a
    /// helpful definition list when the symbol doesn't exist.
    #[test]
    fn read_file_symbol_context() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-sym-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        set_ws(&tmp);
        std::fs::write(
            tmp.join("src/auth.ts"),
            "export function refreshToken(s: Session) {\n  if (expired(s)) {\n    return issue(s.user);\n  }\n  return s.token;\n}\n\nexport function other() {\n  return 1;\n}\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("src/mw.ts"),
            "import { refreshToken } from './auth';\nconst t = refreshToken(sess);\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("src/calc.py"),
            "def add(a, b):\n    total = a + b\n    return total\n\ndef unrelated():\n    pass\n",
        )
        .unwrap();

        // Brace language: exact block, not the whole file; callers listed.
        let out = agent_read_file("src/auth.ts".into(), None, None, None, Some("refreshToken".into()))
            .expect("symbol read");
        eprintln!("{out}");
        assert!(out.contains("L1-L6"), "block extent wrong:\n{out}");
        assert!(!out.contains("function other"), "block leaked past its braces:\n{out}");
        assert!(out.contains("mw.ts:2"), "caller missing:\n{out}");
        assert!(out.contains("调用者"), "callers section missing:\n{out}");

        // Python: indentation-scoped suite.
        let out = agent_read_file("src/calc.py".into(), None, None, None, Some("add".into()))
            .expect("py symbol read");
        assert!(out.contains("L1-L3"), "python suite extent wrong:\n{out}");
        assert!(!out.contains("unrelated"), "python suite leaked:\n{out}");

        // Unknown symbol → error listing the definitions the file has.
        let err = agent_read_file("src/auth.ts".into(), None, None, None, Some("nonexistent".into()))
            .unwrap_err();
        assert!(err.contains("refreshToken"), "recovery list missing: {err}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Fused code search: the file whose NAME matches and which holds the
    /// matching definitions must outrank files with only incidental token
    /// overlap; the digest must surface definition lines and the exact-phrase
    /// tag; a junk query reports no matches.
    #[test]
    fn search_code_ranks_by_fusion() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-scr-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        set_ws(&tmp);
        std::fs::write(
            tmp.join("src/auth.ts"),
            "export function refreshToken(session: Session) {\n  // renew the auth token before expiry\n  return issueToken(session.user);\n}\nexport function validateSession(token: string) {\n  return verify(token);\n}\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("src/middleware.ts"),
            "import { validateSession } from './auth';\nexport function guard(req: Request) {\n  // checks the session token on every request\n  return validateSession(req.token);\n}\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("src/render.ts"),
            "export function draw(canvas: Canvas) {\n  // paints pixels, nothing to do with sessions\n}\n",
        )
        .unwrap();

        let out = agent_search_code("auth token refresh".into(), None).expect("search");
        eprintln!("{out}");
        let auth_pos = out.find("src/auth.ts").expect("auth.ts in results");
        let mid_pos = out.find("src/middleware.ts").unwrap_or(usize::MAX);
        assert!(auth_pos < mid_pos, "auth.ts must rank above middleware.ts:\n{out}");
        assert!(out.contains("文件名匹配"), "filename signal missing:\n{out}");
        assert!(out.contains("refreshToken"), "definition line missing:\n{out}");
        assert!(out.contains("定义 L"), "definition lines section missing:\n{out}");

        // Exact-phrase boost: the literal phrase lives only in middleware.ts.
        let out = agent_search_code("checks the session token".into(), None).expect("search");
        let mid = out.find("src/middleware.ts").expect("middleware in results");
        let auth = out.find("src/auth.ts").unwrap_or(usize::MAX);
        assert!(mid < auth, "exact phrase must put middleware.ts first:\n{out}");
        assert!(out.contains("精确短语"), "exact-phrase tag missing:\n{out}");

        let none = agent_search_code("zebra quantum lighthouse".into(), None).expect("search");
        assert!(none.contains("没有匹配"), "junk query must report no matches: {none}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// read_doc_core must extract document text (docx synthesized in-test —
    /// a zip with word/document.xml) and reject unsupported extensions.
    #[test]
    fn read_doc_extracts_docx_text() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-doc-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);

        let docx = tmp.join("spec.docx");
        {
            use std::io::Write as _;
            let f = std::fs::File::create(&docx).unwrap();
            let mut z = zip::ZipWriter::new(f);
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
            z.start_file("word/document.xml", opts).unwrap();
            z.write_all(
                br#"<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>agent reads documents now</w:t></w:r></w:p></w:body></w:document>"#,
            )
            .unwrap();
            z.finish().unwrap();
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        let out = rt.block_on(read_doc_core(docx, None)).expect("docx extract");
        assert!(out.contains("agent reads documents now"), "text missing: {out}");
        assert!(out.contains(".docx 文档已提取"), "header missing: {out}");

        // Unsupported extension → clear steer back to read_file.
        std::fs::write(tmp.join("notes.rtf"), b"x").unwrap();
        let err = rt.block_on(read_doc_core(tmp.join("notes.rtf"), None)).unwrap_err();
        assert!(err.contains("read_file"), "steer missing: {err}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Real-PDF e2e (fixtures generated outside the test):
    ///   CHATY_TEST_PDF=<text+image.pdf> CHATY_TEST_PDF_SCAN=<image-only.pdf> \
    ///   cargo test --lib read_doc_pdf_e2e -- --ignored --nocapture
    /// The scanned fixture exercises the automatic-OCR fallback (needs the
    /// app's ocr-models dir, downloaded on first OCR use).
    #[test]
    #[ignore]
    fn read_doc_pdf_e2e() {
        let _g = serial();
        let pdf = std::env::var("CHATY_TEST_PDF").expect("set CHATY_TEST_PDF");
        let tmp = std::env::temp_dir().join(format!("chaty-agent-pdf-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::copy(&pdf, tmp.join("doc.pdf")).unwrap();
        set_ws(&tmp);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let out = rt.block_on(read_doc_core(tmp.join("doc.pdf"), None)).expect("pdf extract");
        eprintln!("--- text+image pdf ---\n{}", out.chars().take(900).collect::<String>());
        // NOTE: Chrome's PDF text uses ligatures (ﬁ) — assert ligature-free words.
        assert!(out.contains("text layer and an embedded chart"), "pdf text layer missing: {out}");
        assert!(out.contains("内嵌图片"), "embedded image list missing: {out}");
        // The listed image paths must be viewable without a dir grant.
        let img = out.lines().find(|l| l.contains("chaty-doc-imgs")).expect("image path");
        let resolved = agent_resolve_image(img.trim().to_string()).expect("cache whitelist");
        assert!(std::path::Path::new(&resolved).is_file());

        if let Ok(scan) = std::env::var("CHATY_TEST_PDF_SCAN") {
            std::fs::copy(&scan, tmp.join("scan.pdf")).unwrap();
            let models =
                dirs_home().join("Library/Application Support/com.chaty.desktop/ocr-models");
            let out =
                rt.block_on(read_doc_core(tmp.join("scan.pdf"), Some(models))).expect("scan pdf");
            eprintln!("--- scanned pdf ---\n{}", out.chars().take(900).collect::<String>());
            assert!(out.contains("OCR"), "scanned pdf should carry OCR output: {out}");
            assert!(
                out.to_uppercase().contains("SCANNED FIXTURE TOKEN"),
                "OCR should read the page text: {out}"
            );
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    fn dirs_home() -> PathBuf {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
    }

    /// The piped password must reach REAL `sudo -S` (the `cat` test above only
    /// proves generic stdin plumbing). A deliberately wrong password makes
    /// sudo answer "Sorry, try again" — proof the bytes arrived — without
    /// needing the user's real credentials. If sudo instead reports only
    /// "no password was provided", the bytes never made it.
    #[test]
    #[ignore] // exercises the real sudo binary — run manually
    fn sudo_stdin_reaches_real_sudo() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-realsudo-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        let cmd = ensure_sudo_stdin("sudo -k whoami");
        assert!(cmd.starts_with("sudo -S "), "rewrite: {cmd}");
        let out =
            run_bash(&tmp, &cmd, Duration::from_secs(10), Some("definitely-wrong\n".into()), false)
                .unwrap();
        let all = format!("{}\n{}", out.stdout, out.stderr);
        assert!(
            all.contains("Sorry, try again") || all.contains("incorrect password"),
            "password bytes must reach sudo — got: {all}"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn sudo_password_piped_via_stdin_unsandboxed() {
        let _g = serial();
        let tmp = std::env::temp_dir().join(format!("chaty-agent-sudo-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        set_ws(&tmp);
        // No real sudo: echo stdin back to prove the password is delivered on
        // STDIN (never in argv). Windows runs through `cmd /C`, which has no
        // `cat` — `findstr /r /c:.` is the built-in stdin-echo equivalent.
        // command_uses_sudo(false) so this stays sandboxed — fine, we're only
        // checking the stdin plumbing via run_bash directly.
        #[cfg(windows)]
        let (echo1, echo2) = ("findstr /r /c:.", "findstr /r /c:.");
        #[cfg(not(windows))]
        let (echo1, echo2) = ("cat", "cat -");
        let out = run_bash(&tmp, echo1, Duration::from_secs(5), Some("s3cret\n".into()), true).unwrap();
        assert_eq!(out.stdout.trim(), "s3cret", "stdin must reach the child process");

        // Proof the password is NOT in the command string / argv: a sandboxed
        // run of a sudo-shaped echo would only echo what's in argv.
        let echoed = run_bash(&tmp, echo2, Duration::from_secs(5), Some("hunter2\n".into()), true).unwrap();
        assert_eq!(echoed.stdout.trim(), "hunter2");

        std::fs::remove_dir_all(&tmp).ok();
    }
}

