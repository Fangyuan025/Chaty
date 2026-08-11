//! Silent upstream-follow for directory-shaped official skills.
//!
//! The tiktok-video skill's SUPPORT FILES (scripts/references/examples) have
//! always been byte-for-byte mirrors of the public upstream repo — every
//! upstream fix used to require a Chaty repack just to move them. This module
//! follows upstream quietly instead: at most once a day (and only when
//! online) it compares the upstream HEAD against the local mirror, downloads
//! changed trees into app data, and `use_skill` materializes the freshest
//! layer. SKILL.md — the model-facing procedure, a Chaty-owned rewrite — is
//! deliberately NOT synced: behavior contracts (script CLIs) stay stable
//! upstream, so newer scripts keep working under the bundled doc, and a doc
//! change still ships through a normal release.
//!
//! Failure of ANY step falls back to the bundled files without a sound —
//! offline machines simply run what shipped. `CHATY_SKILL_SYNC=0` disables
//! the whole mechanism.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Upstream repos per skill. The only entry today; the plumbing is generic.
const UPSTREAMS: &[(&str, &str)] = &[("tiktok-video", "Fangyuan025/tiktok-video-skill")];

/// Only these subtrees are support files; everything else upstream (README,
/// SKILL.md, CI config) is not ours to ship.
const PREFIXES: &[&str] = &["scripts/", "references/", "examples/"];

/// A live tree missing any of these is a truncated or broken fetch — reject
/// it rather than materialize a skill that cannot run.
const CRITICAL: &[&str] =
    &["scripts/pipeline.py", "scripts/common.py", "scripts/setup.sh", "references/writing-guide.md"];

const CHECK_EVERY: Duration = Duration::from_secs(24 * 3600);
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 20 * 1024 * 1024;
const UA: &str = "chaty-skill-sync (+https://github.com/Fangyuan025/Chaty)";

/// App-data root shared with the Tauri identifier's own convention, reachable
/// without an AppHandle (the sync thread and chaty-headless both need it).
fn data_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        dirs_home().join("Library/Application Support/com.chaty.desktop")
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| dirs_home().join("AppData/Roaming"))
            .join("com.chaty.desktop")
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| dirs_home().join(".local/share"))
            .join("com.chaty.desktop")
    }
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Test override: only skillsync reads this, so tests can redirect the live
/// root without mutating HOME (process-global env races other test modules).
fn live_root() -> PathBuf {
    std::env::var_os("CHATY_SKILL_LIVE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("skill-live"))
}

fn skill_dir(name: &str) -> PathBuf {
    live_root().join(name)
}

/// A path from the upstream tree is only acceptable when it stays inside the
/// allowed subtrees and cannot escape or pollute: no traversal, no absolute
/// paths, no VCS/debris files. (The tree API is trusted-ish, but a compromised
/// response must not become a filesystem write outside our directory.)
pub(crate) fn allowed_path(p: &str) -> bool {
    if !PREFIXES.iter().any(|pre| p.starts_with(pre)) {
        return false;
    }
    if p.contains("..") || p.starts_with('/') || p.contains('\\') || p.contains('\0') {
        return false;
    }
    let debris = ["__pycache__", ".DS_Store", ".git"];
    if p.split('/').any(|seg| debris.contains(&seg) || seg.is_empty()) {
        return false;
    }
    !p.ends_with(".pyc")
}

/// The live layer served to materialization: a content rev plus every file.
#[derive(serde::Serialize, Clone)]
pub struct LiveSupport {
    pub rev: String,
    pub files: Vec<LiveFile>,
}

#[derive(serde::Serialize, Clone)]
pub struct LiveFile {
    pub path: String,
    pub text: String,
}

/// Read the current live layer for a skill, if a complete one exists.
/// Text-only by design (bundled support files are TS string literals, so the
/// upstream mirror is text too); any unreadable file rejects the layer.
pub fn live_support(name: &str) -> Option<LiveSupport> {
    if std::env::var("CHATY_SKILL_SYNC").as_deref() == Ok("0") {
        return None;
    }
    let base = skill_dir(name);
    let rev = std::fs::read_to_string(base.join("current")).ok()?.trim().to_string();
    if rev.is_empty() || rev.contains('/') || rev.contains("..") {
        return None;
    }
    let root = base.join(&rev);
    let mut files = Vec::new();
    collect_files(&root, &root, &mut files).ok()?;
    let have: std::collections::HashSet<&str> = files.iter().map(|f| f.path.as_str()).collect();
    if !CRITICAL.iter().all(|c| have.contains(c)) {
        return None;
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Some(LiveSupport { rev, files })
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<LiveFile>) -> std::io::Result<()> {
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        if p.is_dir() {
            collect_files(root, &p, out)?;
        } else {
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
            if !allowed_path(&rel) {
                continue;
            }
            let text = std::fs::read_to_string(&p)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            out.push(LiveFile { path: rel, text });
        }
    }
    Ok(())
}

#[tauri::command]
pub fn skill_live_support(name: String) -> Option<LiveSupport> {
    live_support(&name)
}

/// One quiet sync pass for every registered upstream. Call from a background
/// thread at app start; every failure is a silent fallback to bundled files.
pub fn tick() {
    if std::env::var("CHATY_SKILL_SYNC").as_deref() == Ok("0") {
        return;
    }
    for (name, repo) in UPSTREAMS {
        if let Err(e) = sync_one(name, repo) {
            eprintln!("[skillsync] {name}: {e} (bundled files remain in effect)");
        }
    }
}

fn throttled(base: &Path) -> bool {
    let stamp = base.join("last-check");
    if let Ok(meta) = std::fs::metadata(&stamp) {
        if let Ok(modified) = meta.modified() {
            if SystemTime::now().duration_since(modified).unwrap_or(CHECK_EVERY) < CHECK_EVERY {
                return true;
            }
        }
    }
    false
}

fn sync_one(name: &str, repo: &str) -> Result<(), String> {
    let base = skill_dir(name);
    std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;
    if throttled(&base) {
        return Ok(());
    }
    // Stamp FIRST: a failing network must not retry on every launch.
    let _ = std::fs::write(base.join("last-check"), "");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async {
        let client = crate::http::client_secs(UA, 30)?;
        let head: serde_json::Value = client
            .get(format!("https://api.github.com/repos/{repo}/commits/HEAD"))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        let sha = head["sha"].as_str().unwrap_or_default().to_string();
        if sha.len() < 7 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("upstream HEAD has no usable sha".into());
        }
        let current = std::fs::read_to_string(base.join("current")).unwrap_or_default();
        if current.trim() == sha {
            return Ok(());
        }

        let tree: serde_json::Value = client
            .get(format!("https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        if tree["truncated"].as_bool() == Some(true) {
            return Err("upstream tree listing truncated".into());
        }
        let mut picks: Vec<(String, u64)> = Vec::new();
        for item in tree["tree"].as_array().cloned().unwrap_or_default() {
            if item["type"].as_str() != Some("blob") {
                continue;
            }
            let path = item["path"].as_str().unwrap_or_default().to_string();
            if !allowed_path(&path) {
                continue;
            }
            let size = item["size"].as_u64().unwrap_or(0);
            if size > MAX_FILE_BYTES {
                return Err(format!("{path} exceeds the per-file cap"));
            }
            picks.push((path, size));
        }
        if picks.iter().map(|(_, s)| s).sum::<u64>() > MAX_TOTAL_BYTES {
            return Err("upstream tree exceeds the total cap".into());
        }
        {
            let have: std::collections::HashSet<&str> =
                picks.iter().map(|(p, _)| p.as_str()).collect();
            if !CRITICAL.iter().all(|c| have.contains(c)) {
                return Err("upstream tree is missing critical files".into());
            }
        }

        let staging = base.join(format!(".staging-{sha}"));
        let _ = std::fs::remove_dir_all(&staging);
        for (path, _) in &picks {
            let bytes = client
                .get(format!("https://raw.githubusercontent.com/{repo}/{sha}/{path}"))
                .send()
                .await
                .map_err(|e| e.to_string())?
                .error_for_status()
                .map_err(|e| e.to_string())?
                .bytes()
                .await
                .map_err(|e| e.to_string())?;
            // Text-only mirror: reject a layer we couldn't re-serve.
            let text = std::str::from_utf8(&bytes).map_err(|_| format!("{path} is not UTF-8"))?;
            let dest = staging.join(path);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&dest, text).map_err(|e| e.to_string())?;
        }

        // Atomic-enough flip: rename the staging tree into place, then point
        // `current` at it. A crash between the two leaves the old rev live.
        let dest = base.join(&sha);
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::rename(&staging, &dest).map_err(|e| e.to_string())?;
        std::fs::write(base.join("current"), format!("{sha}\n")).map_err(|e| e.to_string())?;
        // Prune every other rev dir — one live layer is all we serve.
        if let Ok(entries) = std::fs::read_dir(&base) {
            for e in entries.flatten() {
                let p = e.path();
                let fname = e.file_name().to_string_lossy().to_string();
                if p.is_dir() && fname != sha {
                    let _ = std::fs::remove_dir_all(&p);
                }
            }
        }
        eprintln!("[skillsync] {name}: now following upstream {}", &sha[..7]);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_path_keeps_support_and_rejects_escapes() {
        for good in [
            "scripts/pipeline.py",
            "scripts/setup.sh",
            "references/writing-guide.md",
            "examples/history-storyboard.json",
        ] {
            assert!(allowed_path(good), "{good} should be allowed");
        }
        for bad in [
            "SKILL.md",                       // Chaty-owned, never synced
            "README.md",
            ".github/workflows/ci.yml",
            "scripts/../SKILL.md",            // traversal
            "/etc/passwd",
            "scripts/__pycache__/x.pyc",
            "scripts/x.pyc",
            "scripts/.DS_Store",
            "scripts\\windows.py",
            "scripts//double.py",
        ] {
            assert!(!allowed_path(bad), "{bad} should be rejected");
        }
    }

    /// live_support serves a complete tree and rejects an incomplete one —
    /// the gate that keeps a truncated fetch from shadowing working bundled
    /// files.
    #[test]
    fn live_support_requires_the_critical_set() {
        let home = std::env::temp_dir().join(format!("chaty-skillsync-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::env::set_var("CHATY_SKILL_LIVE_DIR", &home);
        let base = skill_dir("tiktok-video");
        let rev = base.join("abc123");
        for f in ["scripts/pipeline.py", "scripts/common.py", "scripts/setup.sh"] {
            let p = rev.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, "print('hi')").unwrap();
        }
        std::fs::write(base.join("current"), "abc123\n").unwrap();
        // writing-guide missing → incomplete → refused
        assert!(live_support("tiktok-video").is_none());
        let wg = rev.join("references/writing-guide.md");
        std::fs::create_dir_all(wg.parent().unwrap()).unwrap();
        std::fs::write(&wg, "# guide").unwrap();
        let live = live_support("tiktok-video").expect("complete tree serves");
        assert_eq!(live.rev, "abc123");
        assert_eq!(live.files.len(), 4);
        assert!(live.files.iter().any(|f| f.path == "references/writing-guide.md"));
        // The kill switch wins.
        std::env::set_var("CHATY_SKILL_SYNC", "0");
        assert!(live_support("tiktok-video").is_none());
        std::env::remove_var("CHATY_SKILL_SYNC");
        std::env::remove_var("CHATY_SKILL_LIVE_DIR");
        let _ = std::fs::remove_dir_all(&home);
    }

    /// Real-network integration: full tick against the live upstream repo.
    /// `CHATY_TEST_SKILLSYNC=1 cargo test --release skillsync_tick -- --ignored`
    #[test]
    #[ignore]
    fn skillsync_tick_follows_upstream() {
        if std::env::var("CHATY_TEST_SKILLSYNC").is_err() {
            eprintln!("set CHATY_TEST_SKILLSYNC=1 to run");
            return;
        }
        let home = std::env::temp_dir().join(format!("chaty-skillsync-net-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::env::set_var("CHATY_SKILL_LIVE_DIR", &home);
        tick();
        let live = live_support("tiktok-video").expect("tick should produce a live layer");
        assert!(live.files.iter().any(|f| f.path == "scripts/pipeline.py"));
        assert!(live.files.iter().any(|f| f.path == "references/writing-guide.md"));
        eprintln!("live rev {} with {} files", live.rev, live.files.len());
        // Second tick inside the throttle window is a no-op that stays Ok.
        tick();
        std::env::remove_var("CHATY_SKILL_LIVE_DIR");
        let _ = std::fs::remove_dir_all(&home);
    }
}
