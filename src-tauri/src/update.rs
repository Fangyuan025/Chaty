//! Lightweight self-update against GitHub Releases.
//!
//! We ship a per-user Inno Setup installer, so instead of Tauri's updater (which
//! expects its own NSIS/MSI bundle + signing) we just check the latest release,
//! and on request download its `*-setup.exe` and launch it — the installer
//! closes the running app and replaces it in place.

use serde::Serialize;

const REPO: &str = "Fangyuan025/Chaty";
const UA: &str = "Chaty-Updater";

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub available: bool,
    pub current: String,
    pub latest: String,
    pub url: Option<String>,
    pub notes: Option<String>,
}

/// Parse a version like "v0.3.3" / "0.3.3" into a comparable tuple.
fn ver_tuple(s: &str) -> (u64, u64, u64) {
    let s = s.trim().trim_start_matches('v');
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
    )
}

async fn fetch_latest() -> anyhow::Result<UpdateInfo> {
    let current = env!("CARGO_PKG_VERSION").to_string();
    let client = reqwest::Client::builder().user_agent(UA).build()?;
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
        .and_then(|arr| {
            arr.iter().find(|a| {
                a["name"]
                    .as_str()
                    .is_some_and(|n| n.to_lowercase().ends_with(".exe"))
            })
        })
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

/// Download the installer for `url`, launch it, and quit so it can replace files.
#[tauri::command]
pub async fn run_update(app: tauri::AppHandle, url: String) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .user_agent(UA)
        .build()
        .map_err(|e| e.to_string())?;
    let bytes = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .bytes()
        .await
        .map_err(|e| e.to_string())?;

    let path = std::env::temp_dir().join("Chaty-update-setup.exe");
    std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;

    std::process::Command::new(&path)
        .spawn()
        .map_err(|e| format!("无法启动安装程序: {e}"))?;
    // Give the installer a beat to start, then exit so it can replace the binary.
    std::thread::sleep(std::time::Duration::from_millis(400));
    app.exit(0);
    Ok(())
}
