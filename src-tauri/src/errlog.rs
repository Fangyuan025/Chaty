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

/// Append one entry. Never panics, never blocks meaningfully; detail is
/// length-capped so a runaway error can't balloon the file in one write.
pub fn append_error(kind: &str, detail: &str) {
    let path = error_log_path();
    rotate(&path);
    // No chrono dep: unix seconds are good enough to correlate with an issue
    // report, and every entry also carries app version + OS.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let capped: String = detail.chars().take(8000).collect();
    let entry = format!(
        "[unix:{ts}] chaty v{} {} [{kind}]\n{capped}\n---\n",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = f.write_all(entry.as_bytes());
    }
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

    /// Entries append with header + separator, and rotation keeps the file
    /// bounded while preserving the newest entries.
    #[test]
    fn append_and_rotate() {
        let path = error_log_path();
        let before = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        append_error("test-kind", "hello 错误 world");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[test-kind]"));
        assert!(text.contains("hello 错误 world"));
        assert!(std::fs::metadata(&path).unwrap().len() > before);

        // Force a rotation with a big synthetic tail.
        let big = "x".repeat(600_000);
        append_error("bulk1", &big);
        append_error("bulk2", &big);
        append_error("marker-final", "the newest entry survives");
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len < MAX_LOG_BYTES + 700_000, "rotation must bound the file, got {len}");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("marker-final"), "newest entries must survive rotation");
    }
}
