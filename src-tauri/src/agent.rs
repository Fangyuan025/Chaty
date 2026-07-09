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
        // Background jobs belong to the workspace they were started in.
        bg_kill_all();
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

/// Read a text file. Optional 1-based `offset` line and `limit` line count.
#[tauri::command]
pub fn agent_read_file(
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Result<String, String> {
    let abs = resolve(&path)?;
    let meta = std::fs::metadata(&abs).map_err(|e| format!("读取失败 (read failed): {e}"))?;
    if meta.is_dir() {
        return Err("这是一个目录，请用 list_dir (that's a directory)".to_string());
    }
    let bytes = std::fs::read(&abs).map_err(|e| e.to_string())?;
    let truncated_bytes = bytes.len() > MAX_READ_BYTES;
    let slice = &bytes[..bytes.len().min(MAX_READ_BYTES)];
    let text = String::from_utf8_lossy(slice);

    let out = if offset.is_some() || limit.is_some() {
        let start = offset.unwrap_or(1).max(1) - 1;
        let take = limit.unwrap_or(usize::MAX);
        text.lines().skip(start).take(take).collect::<Vec<_>>().join("\n")
    } else {
        text.into_owned()
    };
    Ok(if truncated_bytes {
        format!("{out}\n\n… (文件已截断 / file truncated at {MAX_READ_BYTES} bytes)")
    } else {
        out
    })
}

#[tauri::command]
pub fn agent_write_file(path: String, content: String) -> Result<String, String> {
    let abs = resolve(&path)?;
    if abs.is_dir() {
        return Err(format!(
            "目标是一个目录，不能写入；请提供文件路径 (target is a directory, give a file path): {path}"
        ));
    }
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
    let count = text.matches(&old_string).count();
    if count == 0 {
        return Err("未找到 old_string（需与文件内容逐字匹配）(old_string not found — must match exactly)".to_string());
    }
    let all = replace_all.unwrap_or(false);
    if count > 1 && !all {
        return Err(format!(
            "old_string 出现 {count} 次，不唯一；请提供更多上下文或用 replace_all (not unique: {count} matches)"
        ));
    }
    let updated = if all {
        text.replace(&old_string, &new_string)
    } else {
        text.replacen(&old_string, &new_string, 1)
    };
    std::fs::write(&abs, updated.as_bytes()).map_err(|e| format!("写入失败 (write failed): {e}"))?;
    let root = workspace()?;
    Ok(format!(
        "已编辑 {}（替换 {} 处）",
        rel_display(&root, &abs),
        if all { count } else { 1 }
    ))
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
        let read = agent_read_file("sub/hi.txt".into(), None, None).unwrap();
        assert!(read.contains("hello world"));

        // unique edit
        agent_edit_file("sub/hi.txt".into(), "hello".into(), "hi".into(), None).unwrap();
        assert!(agent_read_file("sub/hi.txt".into(), None, None).unwrap().starts_with("hi world"));

        // non-existent old_string errors
        assert!(agent_edit_file("sub/hi.txt".into(), "nope".into(), "x".into(), None).is_err());

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
