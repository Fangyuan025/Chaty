//! Append-only application error log (owner spec, issue-#5 follow-up):
//! panics, front-end uncaught errors, and native-crash post-mortems all land
//! in one file the user can attach to a GitHub issue.
//!
//! Location: `<app-data>/logs/chaty-error.log`, size-capped by keeping the
//! newest half when it outgrows ~1 MiB.

use std::io::Write;
use std::path::PathBuf;

/// The app's data dir (same location Tauri uses for the identifier), without
/// needing an AppHandle — engine-level code (llama guard, panic hook) runs
/// before/outside Tauri state.
pub(crate) fn chaty_data_dir() -> PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share"));
    let dir = base.unwrap_or_else(std::env::temp_dir).join("com.chaty.desktop");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

pub fn error_log_path() -> PathBuf {
    let dir = chaty_data_dir().join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("chaty-error.log")
}

const MAX_LOG_BYTES: u64 = 1024 * 1024;

/// Keep the file bounded: over the cap, retain the newest half.
fn rotate(path: &PathBuf) {
    let Ok(meta) = std::fs::metadata(path) else { return };
    if meta.len() <= MAX_LOG_BYTES {
        return;
    }
    if let Ok(data) = std::fs::read(path) {
        let keep = data.len() / 2;
        // Cut at a line boundary so entries stay readable.
        let start = data[data.len() - keep..]
            .iter()
            .position(|b| *b == b'\n')
            .map(|i| data.len() - keep + i + 1)
            .unwrap_or(data.len() - keep);
        let _ = std::fs::write(path, &data[start..]);
    }
}

/// Append one entry to an explicit file (testable without touching the real
/// user log — the first version's tests polluted the owner's actual
/// app-data, which is exactly the sin this module records for others).
fn append_error_to(path: &PathBuf, kind: &str, detail: &str) {
    rotate(path);
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let capped: String = detail.chars().take(8000).collect();
    let entry = format!(
        "[{ts}] chaty v{} {} [{kind}]\n{capped}\n---\n",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(entry.as_bytes());
    }
}

/// Append one entry. Never panics, never blocks meaningfully; detail is
/// length-capped so a runaway error can't balloon the file in one write.
pub fn append_error(kind: &str, detail: &str) {
    // Unit tests must NEVER reach the real user log — three separate test
    // runs have stamped false entries into the owner's actual log through
    // code that called this transitively. Compiled out of existence in
    // `cargo test`; tests that test appending use append_error_to directly.
    #[cfg(test)]
    {
        let _ = (kind, detail);
    }
    #[cfg(not(test))]
    append_error_to(&error_log_path(), kind, detail);
}

/// Names of crash reports for OUR app in `dir` modified after `newer_than`
/// (pure + testable). macOS writes `Chaty-<date>.ips` there when a native
/// crash (Metal, the MLX sidecar, WebKit) kills the process before any
/// in-process hook can run.
fn crash_reports_in(dir: &std::path::Path, newer_than: Option<std::time::SystemTime>) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<String> = rd
        .flatten()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            (name.starts_with("chaty") && (name.ends_with(".ips") || name.ends_with(".crash")))
                && match (newer_than, e.metadata().and_then(|m| m.modified())) {
                    (Some(mark), Ok(mtime)) => mtime > mark,
                    (None, _) => true,
                    (_, Err(_)) => false,
                }
        })
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    out.sort();
    out
}

/// macOS startup post-mortem: surface new crash reports into the error log
/// so an issue report can carry the real stack. Watermarked so each report
/// is mentioned once.
#[cfg(target_os = "macos")]
pub fn sweep_native_crash_reports() {
    let Some(home) = std::env::var_os("HOME") else { return };
    let dir = std::path::PathBuf::from(home).join("Library/Logs/DiagnosticReports");
    let mark = chaty_data_dir().join("logs").join("crash-sweep.mark");
    let last = std::fs::metadata(&mark).and_then(|m| m.modified()).ok();
    for name in crash_reports_in(&dir, last) {
        append_error(
            "native-crash",
            &format!("macOS 崩溃报告(提 issue 请一并附上): ~/Library/Logs/DiagnosticReports/{name}"),
        );
    }
    let _ = std::fs::create_dir_all(mark.parent().unwrap());
    let _ = std::fs::write(&mark, "swept\n");
}

/// Route Rust panics into the log (the default hook still prints to stderr).
pub fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let bt = std::backtrace::Backtrace::force_capture();
        append_error("panic", &format!("{info}\n{bt}"));
        default(info);
    }));
}

/// Front-end uncaught errors / rejections report through this.
#[tauri::command]
pub fn log_app_error(kind: String, detail: String) {
    let kind: String = kind.chars().take(40).collect();
    append_error(&format!("frontend:{kind}"), &detail);
}

/// Open the log in the OS default viewer (creating it so the open succeeds).
#[tauri::command]
pub fn open_error_log() -> Result<(), String> {
    let path = error_log_path();
    if !path.exists() {
        append_error("info", "log created — no errors recorded yet");
    }
    crate::commands::open_default(&path.to_string_lossy())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crash-report scan picks OUR reports, respects the watermark, and
    /// ignores other apps' files.
    #[test]
    fn crash_report_scan_filters_and_watermarks() {
        let dir = std::env::temp_dir().join(format!("chaty-ips-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Chaty-2026-08-05-101010.ips"), "{}").unwrap();
        std::fs::write(dir.join("chaty-2026-08-05-090909.crash"), "x").unwrap();
        std::fs::write(dir.join("Safari-2026-08-05.ips"), "{}").unwrap();
        let all = crash_reports_in(&dir, None);
        assert_eq!(all.len(), 2, "ours only: {all:?}");
        assert!(all.iter().all(|n| n.to_lowercase().starts_with("chaty")));
        // Watermark in the future → nothing new.
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(3600);
        assert!(crash_reports_in(&dir, Some(future)).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Entries append with a HUMAN-readable timestamp + separator, rotation
    /// keeps the file bounded — and the test writes ONLY to its own temp
    /// file, never the user's real log (the first version polluted it).
    #[test]
    fn append_and_rotate() {
        let dir = std::env::temp_dir().join(format!("chaty-errlog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chaty-error.log");
        append_error_to(&path, "test-kind", "hello 错误 world");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[test-kind]"));
        assert!(text.contains("hello 错误 world"));
        // Timestamp is human-readable local time, not raw unix seconds.
        assert!(
            regex::Regex::new(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]").unwrap().is_match(&text),
            "timestamp must be YYYY-MM-DD HH:MM:SS, got:\n{text}"
        );
        assert!(!text.contains("[unix:"), "raw unix timestamps are user-hostile");

        // Force a rotation with a big synthetic tail.
        let big = "x".repeat(600_000);
        append_error_to(&path, "bulk1", &big);
        append_error_to(&path, "bulk2", &big);
        append_error_to(&path, "marker-final", "the newest entry survives");
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len < MAX_LOG_BYTES + 700_000, "rotation must bound the file, got {len}");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("marker-final"), "newest entries must survive rotation");
        std::fs::remove_dir_all(&dir).ok();
    }
}
