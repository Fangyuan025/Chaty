//! Lightweight self-update against GitHub Releases.
//!
//! We ship a per-user Inno Setup installer, so instead of Tauri's updater (which
//! expects its own NSIS/MSI bundle + signing) we just check the latest release,
//! and on request download its `*-setup.exe` and launch it — the installer
//! closes the running app and replaces it in place. On macOS the disk image is
//! opened; on Linux the new AppImage takes the running one's place.

use serde::Serialize;
use tauri::ipc::Channel;

const REPO: &str = "Fangyuan025/Chaty";
const UA: &str = "Chaty-Updater";

/// Release asset to look for: the Inno installer on Windows, the disk image on
/// macOS (Tauri's `dmg` bundle target), the AppImage on Linux — which used to
/// be offered the Windows installer, there being no third case.
#[cfg(target_os = "macos")]
const ASSET_EXT: &str = ".dmg";
#[cfg(windows)]
const ASSET_EXT: &str = ".exe";
#[cfg(all(unix, not(target_os = "macos")))]
const ASSET_EXT: &str = ".appimage";

/// How far the installer's download has got, for the update button.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateProgress {
    pub downloaded: u64,
    /// None when the server does not say how big it is.
    pub total: Option<u64>,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub available: bool,
    pub current: String,
    pub latest: String,
    pub url: Option<String>,
    pub notes: Option<String>,
}

/// A release asset this machine can install: the platform's kind of file and,
/// on Linux, a build for this CPU — the AppImage is x86-64 only.
fn asset_fits(name: &str) -> bool {
    let n = name.to_lowercase();
    if !n.ends_with(ASSET_EXT) {
        return false;
    }
    if cfg!(all(unix, not(target_os = "macos"))) {
        return if cfg!(target_arch = "x86_64") {
            n.contains("amd64") || n.contains("x86_64")
        } else {
            n.contains("aarch64") || n.contains("arm64")
        };
    }
    true
}

/// Parse a version like "v0.3.3" / "0.3.3" / "2.0.4-beta.1" into a comparable
/// key. The 4th element ranks release-ness: a prerelease suffix ("-beta",
/// "-rc.1") sorts BELOW the plain release of the same triple, so "2.0.4" is
/// offered as an upgrade to a user on "2.0.4-beta.1" — the old triple-only
/// compare saw them as equal.
fn ver_tuple(s: &str) -> (u64, u64, u64, u8) {
    let s = s.trim().trim_start_matches('v');
    let is_release = !s.contains('-');
    let mut parts = s.split('.').map(|p| {
        p.chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u64>()
            .unwrap_or(0)
    });
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        u8::from(is_release),
    )
}

async fn fetch_latest() -> anyhow::Result<UpdateInfo> {
    let current = env!("CARGO_PKG_VERSION").to_string();
    // Without a timeout an unreachable GitHub held the About panel's check
    // in "checking…" forever.
    let client = crate::http::client(UA, std::time::Duration::from_secs(15)).map_err(anyhow::Error::msg)?;
    let body = client
        .get(format!("https://api.github.com/repos/{REPO}/releases/latest"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;

    let latest_tag = json["tag_name"].as_str().unwrap_or("").to_string();
    let notes = json["body"].as_str().filter(|s| !s.is_empty()).map(str::to_string);
    let url = json["assets"]
        .as_array()
        .and_then(|arr| arr.iter().find(|a| a["name"].as_str().is_some_and(asset_fits)))
        .and_then(|a| a["browser_download_url"].as_str())
        .map(str::to_string);

    let available =
        !latest_tag.is_empty() && url.is_some() && ver_tuple(&latest_tag) > ver_tuple(&current);

    Ok(UpdateInfo {
        available,
        current,
        latest: latest_tag.trim_start_matches('v').to_string(),
        url,
        notes,
    })
}

/// Check GitHub for a newer release (never errors out to the UI).
#[tauri::command]
pub async fn check_update() -> UpdateInfo {
    match fetch_latest().await {
        Ok(info) => info,
        Err(e) => {
            eprintln!("update check failed: {e:#}");
            UpdateInfo {
                current: env!("CARGO_PKG_VERSION").to_string(),
                ..Default::default()
            }
        }
    }
}

/// Download the installer for `url`, reporting progress as it comes, launch
/// it, and quit so it can replace files.
#[tauri::command]
pub async fn run_update(app: tauri::AppHandle, url: String, on_progress: Channel<UpdateProgress>) -> Result<(), String> {
    // One download at a time: two would write the same file at once.
    static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    struct Running;
    impl Drop for Running {
        fn drop(&mut self) {
            RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }
    if RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return Err(trf!("新版本已经在下载了", "the new version is already downloading"));
    }
    let _running = Running;

    // Linux: the running AppImage is replaced where it is, so first make sure
    // there is one — a build run any other way has nothing to swap.
    #[cfg(all(unix, not(target_os = "macos")))]
    let appimage = std::env::var_os("APPIMAGE").map(std::path::PathBuf::from).ok_or_else(|| {
        trf!(
            "只有 AppImage 版的 Chaty 能自动更新,请从发布页下载新版本",
            "only the AppImage build of Chaty updates itself — download the new version from the releases page"
        )
    })?;

    #[cfg(target_os = "macos")]
    let dest = std::env::temp_dir().join("Chaty-update.dmg");
    #[cfg(windows)]
    let dest = std::env::temp_dir().join("Chaty-update-setup.exe");
    // Beside the AppImage it replaces, so the swap is a rename on one disk.
    #[cfg(all(unix, not(target_os = "macos")))]
    let dest = appimage.with_extension("AppImage.part");

    download(&url, &dest, &on_progress).await?;

    // Windows: run the Inno installer — it closes the app and replaces it.
    #[cfg(windows)]
    std::process::Command::new(&dest)
        .spawn()
        .map_err(|e| trf!("无法启动安装程序:{e}", "failed to launch installer: {e}"))?;
    // macOS: mount the .dmg in Finder; the user drags Chaty to Applications.
    #[cfg(target_os = "macos")]
    std::process::Command::new("open")
        .arg(&dest)
        .spawn()
        .map_err(|e| trf!("无法打开磁盘映像:{e}", "failed to open disk image: {e}"))?;
    // Linux: the new image takes the old one's name (the running one keeps
    // its open file until it exits), then starts in its place.
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| trf!("无法设置新版本的执行权限:{e}", "could not make the new version executable: {e}"))?;
        std::fs::rename(&dest, &appimage)
            .map_err(|e| trf!("无法替换当前的 AppImage:{e}", "could not replace the running AppImage: {e}"))?;
        std::process::Command::new(&appimage)
            .spawn()
            .map_err(|e| trf!("无法启动新版本:{e}", "could not start the new version: {e}"))?;
    }
    // Give the installer a beat to start, then exit so it can replace the binary.
    std::thread::sleep(std::time::Duration::from_millis(400));
    app.exit(0);
    Ok(())
}

/// Stream the installer into `dest`, telling the button how far it has got.
/// Written to disk as it arrives — it used to be held whole in memory, with
/// nothing to show until it was all there — and checked for its full size.
async fn download(url: &str, dest: &std::path::Path, on_progress: &Channel<UpdateProgress>) -> Result<(), String> {
    crate::http::download_to(UA, url, dest, |downloaded, total| {
        let _ = on_progress.send(UpdateProgress { downloaded, total });
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::{asset_fits, download, ver_tuple, UpdateProgress};
    use std::sync::{Arc, Mutex};
    use tauri::ipc::Channel;

    /// A server that answers one request with `body`, claiming `claimed` bytes.
    fn serve(body: Vec<u8>, claimed: usize) -> u16 {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            let hdr = format!("HTTP/1.1 200 OK\r\nContent-Length: {claimed}\r\nConnection: close\r\n\r\n");
            let _ = sock.write_all(hdr.as_bytes());
            for part in body.chunks(64 * 1024) {
                let _ = sock.write_all(part);
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
        });
        port
    }

    fn progress_channel() -> (Channel<UpdateProgress>, Arc<Mutex<Vec<serde_json::Value>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let ch = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
            if let tauri::ipc::InvokeResponseBody::Json(j) = body {
                sink.lock().unwrap().push(serde_json::from_str(&j).unwrap());
            }
            Ok(())
        });
        (ch, seen)
    }

    /// The installer reaches the disk whole, and the button hears how far it
    /// has got along the way — from nothing to all of it.
    #[tokio::test]
    async fn the_download_streams_to_disk_and_says_how_far_it_has_got() {
        let body: Vec<u8> = (0..1_200_000u32).map(|i| (i % 251) as u8).collect();
        let port = serve(body.clone(), body.len());
        let dest = std::env::temp_dir().join(format!("chaty-update-test-{}", std::process::id()));
        let (ch, seen) = progress_channel();
        download(&format!("http://127.0.0.1:{port}/Chaty.dmg"), &dest, &ch).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        let seen = seen.lock().unwrap();
        assert!(seen.len() >= 3, "progress along the way: {seen:?}");
        assert_eq!(seen.first().unwrap()["downloaded"], 0);
        assert_eq!(seen.last().unwrap()["downloaded"], body.len() as u64);
        assert!(seen.iter().all(|p| p["total"] == body.len() as u64));
        let _ = std::fs::remove_file(&dest);
    }

    /// A download that stops short of what the server promised is not
    /// launched as an installer.
    #[tokio::test]
    async fn a_short_download_is_refused() {
        let port = serve(vec![7u8; 1000], 5000);
        let dest = std::env::temp_dir().join(format!("chaty-update-short-{}", std::process::id()));
        let (ch, _) = progress_channel();
        assert!(download(&format!("http://127.0.0.1:{port}/x"), &dest, &ch).await.is_err());
        assert!(!dest.exists());
    }

    /// The asset this platform installs, and nothing another one does.
    #[test]
    fn each_platform_finds_its_own_installer() {
        let assets = ["Chaty_2.3.8_aarch64.dmg", "Chaty_2.3.8_amd64.AppImage", "Chaty_2.3.8_x64-setup.exe"];
        let found: Vec<&str> = assets.into_iter().filter(|a| asset_fits(a)).collect();
        #[cfg(target_os = "macos")]
        assert_eq!(found, ["Chaty_2.3.8_aarch64.dmg"]);
        #[cfg(windows)]
        assert_eq!(found, ["Chaty_2.3.8_x64-setup.exe"]);
        #[cfg(all(unix, not(target_os = "macos"), target_arch = "x86_64"))]
        assert_eq!(found, ["Chaty_2.3.8_amd64.AppImage"]);
        #[cfg(all(unix, not(target_os = "macos"), not(target_arch = "x86_64")))]
        assert!(found.is_empty());
    }

    /// Prerelease suffixes sort below the plain release of the same triple —
    /// and plain triples keep their ordinary ordering (incl. the two-digit
    /// components a lexicographic compare would garble).
    #[test]
    fn version_compare_handles_prerelease_and_digits() {
        assert!(ver_tuple("2.0.4") > ver_tuple("2.0.4-beta.1"));
        assert!(ver_tuple("v2.0.4-rc.2") > ver_tuple("2.0.3"));
        assert!(ver_tuple("2.0.10") > ver_tuple("2.0.9"));
        assert!(ver_tuple("2.1.0") > ver_tuple("2.0.99"));
        assert_eq!(ver_tuple("v1.2.3"), ver_tuple("1.2.3"));
        assert!(ver_tuple("2.0.4-beta.1") <= ver_tuple("2.0.4-beta.1"));
    }
}
