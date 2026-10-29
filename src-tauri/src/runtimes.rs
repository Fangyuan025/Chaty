//! Runtime packs: the engines models run on, updated apart from the app.
//!
//! A new model architecture lands in llama.cpp, stable-diffusion.cpp or
//! audio.cpp weeks before a Chaty release carries it (issue #27: LFM3 would
//! not load until one did). The engines that run as their own processes —
//! chaty-sd, chaty-audio, chaty-mlx — speak a small JSON protocol, so a newer
//! build of one can stand in for the bundled one with nothing else changed.
//!
//! Packs are published by CI as a rolling `runtimes` release on GitHub: a
//! signed index (`runtimes.json` + `runtimes.json.sig`, Ed25519, the public
//! key below) naming one zip per engine and platform with its SHA-256. A pack
//! is used only when its protocol is the one this app speaks; the bundled
//! engine is always kept, a pack that fails to start is set aside and the
//! bundled one takes over, and the user can go back to it at any time.
//! Updates are by hand (a button) or, when the user turns it on, automatic.
//!
//! llama.cpp is in-process and is updated with the app until it, too, runs as
//! its own process.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use tauri::ipc::Channel;
use tauri::Emitter;

const UA: &str = "Chaty-Runtimes";
const INDEX_URL: &str = "https://github.com/Fangyuan025/Chaty/releases/download/runtimes/runtimes.json";
/// The key CI signs the index with (`tools/runtime-sign`), raw Ed25519, base64.
const INDEX_PUBKEY: &str = "d+GUC9VDQCppjsa+H+bE+r7Aq4d78w5nsHNfP4QSJI4=";
/// An automatic check at most this often.
const CHECK_EVERY_SECS: u64 = 24 * 3600;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Llama,
    Sd,
    Audio,
    Mlx,
}

/// What the app was built with, kept equal to the build pins by a test.
pub struct Bundled {
    pub upstream: &'static str,
    pub rev: &'static str,
    /// The upstream commit's date — what "newer" is measured by.
    pub date: &'static str,
}

impl Engine {
    pub const ALL: [Engine; 4] = [Engine::Llama, Engine::Sd, Engine::Audio, Engine::Mlx];

    pub fn id(self) -> &'static str {
        match self {
            Engine::Llama => "llama",
            Engine::Sd => "sd",
            Engine::Audio => "audio",
            Engine::Mlx => "mlx",
        }
    }

    fn from_id(id: &str) -> Option<Engine> {
        Engine::ALL.into_iter().find(|e| e.id() == id)
    }

    /// Runs as its own process, so a pack can stand in for the bundled one.
    pub fn swappable(self) -> bool {
        !matches!(self, Engine::Llama)
    }

    /// The sidecar protocol this app speaks with the engine. A pack built for
    /// another is never used; bump with any change an older binary would
    /// misread.
    pub fn protocol(self) -> &'static str {
        "1"
    }

    fn exe(self) -> &'static str {
        match (self, cfg!(windows)) {
            (Engine::Sd, true) => "chaty-sd.exe",
            (Engine::Sd, false) => "chaty-sd",
            (Engine::Audio, true) => "chaty-audio.exe",
            (Engine::Audio, false) => "chaty-audio",
            (Engine::Mlx, _) => "chaty-mlx",
            (Engine::Llama, _) => "",
        }
    }

    /// Present on this machine at all (MLX is Apple Silicon only).
    fn here(self) -> bool {
        !matches!(self, Engine::Mlx) || cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    pub fn bundled(self) -> Bundled {
        match self {
            Engine::Llama => Bundled { upstream: "ggml-org/llama.cpp", rev: "8642541b255bfb5dc39cd351a178be09013a55d6", date: "2026-10-28T11:30:38Z" },
            Engine::Sd => Bundled { upstream: "leejet/stable-diffusion.cpp", rev: "88411ef1e0688ff2df1010aeeb5d92b2d8cea2be", date: "2026-09-23T17:45:14Z" },
            Engine::Audio => Bundled { upstream: "0xShug0/audio.cpp", rev: "955c8725c611d511774e6be132aff6609163b2d2", date: "2026-09-25T22:44:03Z" },
            Engine::Mlx => Bundled { upstream: "Fangyuan025/mlx-swift-lm", rev: "c11767bc3aa01f33683ef39fa5adb9ad47f214ef", date: "2026-09-05T06:58:53Z" },
        }
    }
}

/// This build's platform, as packs are named for it.
fn platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos-arm64"
    } else if cfg!(windows) {
        "windows-x64"
    } else {
        "linux-x64"
    }
}

/// One pack in the index.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Pack {
    pub engine: Engine,
    pub version: String,
    pub platform: String,
    pub protocol: String,
    /// Oldest app version the pack was built for.
    #[serde(default)]
    pub min_app: String,
    pub upstream: String,
    pub rev: String,
    pub date: String,
    pub url: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    #[serde(default)]
    packs: Vec<Pack>,
}

/// What the user chose, kept beside the packs.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct State {
    #[serde(default)]
    auto_update: bool,
    /// engine id → the pack version in use (absent: the bundled engine).
    #[serde(default)]
    active: BTreeMap<String, String>,
    /// `engine:version` of packs that failed to start, never offered again.
    #[serde(default)]
    broken: BTreeSet<String>,
    #[serde(default)]
    last_check: u64,
}

fn root() -> PathBuf {
    crate::errlog::chaty_data_dir().join("runtimes")
}

fn pack_dir(e: Engine, version: &str) -> PathBuf {
    root().join(e.id()).join(version)
}

fn load_state() -> State {
    std::fs::read_to_string(root().join("state.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_state(s: &State) {
    let _ = std::fs::create_dir_all(root());
    if let Ok(j) = serde_json::to_string_pretty(s) {
        let _ = std::fs::write(root().join("state.json"), j);
    }
}

fn cached_index() -> Index {
    std::fs::read_to_string(root().join("index.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The engine binary to start: the pack in use when there is a sound one,
/// else None — the bundled engine.
pub fn active_binary(e: Engine) -> Option<PathBuf> {
    active_pack(e).map(|(dir, _)| dir.join(e.exe()))
}

/// The pack in use and its manifest, when it can run here: not set aside,
/// speaking this app's protocol, and newer than the engine the app came with
/// (an app update can bring a newer one than the pack chosen before it).
fn active_pack(e: Engine) -> Option<(PathBuf, Pack)> {
    if !e.swappable() {
        return None;
    }
    let state = load_state();
    let version = state.active.get(e.id())?;
    if state.broken.contains(&format!("{}:{version}", e.id())) {
        return None;
    }
    let dir = pack_dir(e, version);
    let manifest: Pack = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).ok()?).ok()?;
    if manifest.protocol != e.protocol() || manifest.date.as_str() <= e.bundled().date {
        return None;
    }
    dir.join(e.exe()).is_file().then_some((dir, manifest))
}

/// A pack that would not start: set aside for good, and the bundled engine
/// used from now on. The caller tries again with it.
pub fn set_aside(e: Engine, why: &str) {
    let mut state = load_state();
    if let Some(version) = state.active.remove(e.id()) {
        crate::errlog::append_error(
            "runtime-pack",
            &trf!(
                "{} 运行时包 {version} 启动失败,已改回自带版本:{why}",
                "the {} runtime pack {version} failed to start; back on the bundled engine: {why}",
                e.id()
            ),
        );
        state.broken.insert(format!("{}:{version}", e.id()));
        save_state(&state);
    }
}

/// The test override a sidecar honours (`CHATY_SD_SIDECAR` …); with it set, no
/// pack is used.
fn override_var(e: Engine) -> &'static str {
    match e {
        Engine::Sd => "CHATY_SD_SIDECAR",
        Engine::Audio => "CHATY_AUDIO_SIDECAR",
        Engine::Mlx => "CHATY_MLX_SIDECAR",
        Engine::Llama => "",
    }
}

/// Start engine `e` with its pack when one is in use, else with `bundled`.
/// When the pack's start fails and the bundled engine's then succeeds, the
/// pack was at fault: it is set aside for good. When both fail, the model
/// was — the pack stays, and the bundled engine's error is the one returned.
pub fn start_with<T>(
    e: Engine,
    bundled: Option<PathBuf>,
    missing: impl Fn() -> anyhow::Error,
    start: impl Fn(&Path) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let pack = if std::env::var_os(override_var(e)).is_some() { None } else { active_binary(e) };
    let Some(pack) = pack else {
        return start(&bundled.ok_or_else(&missing)?);
    };
    match start(&pack) {
        Ok(t) => Ok(t),
        Err(with_pack) => {
            let Some(bundled) = bundled else { return Err(with_pack) };
            let t = start(&bundled)?;
            set_aside(e, &format!("{with_pack:#}"));
            Ok(t)
        }
    }
}

/// The newest pack in `index` this app can use for `e` on this platform, when
/// it is newer than the engine the app came with.
fn best_pack<'a>(index: &'a Index, e: Engine, broken: &BTreeSet<String>) -> Option<&'a Pack> {
    index
        .packs
        .iter()
        .filter(|p| p.engine == e && p.platform == platform() && p.protocol == e.protocol())
        .filter(|p| p.date.as_str() > e.bundled().date)
        .filter(|p| !broken.contains(&format!("{}:{}", e.id(), p.version)))
        .max_by(|a, b| (a.date.as_str(), a.version.as_str()).cmp(&(b.date.as_str(), b.version.as_str())))
}

/// A newer pack exists, but for a protocol this app does not speak: the user
/// needs a newer Chaty to use it.
fn newer_needs_app(index: &Index, e: Engine, current_date: &str) -> Option<String> {
    index
        .packs
        .iter()
        .filter(|p| p.engine == e && p.platform == platform() && p.protocol != e.protocol())
        .filter(|p| p.date.as_str() > current_date)
        .map(|p| p.min_app.clone())
        .max()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatus {
    pub engine: Engine,
    pub swappable: bool,
    /// The engine in use: "bundled", or a pack's version.
    pub active: String,
    pub upstream: String,
    pub rev: String,
    pub date: String,
    /// A newer pack this app can use, when there is one.
    pub update: Option<Pack>,
    /// A newer engine that needs a newer Chaty (its version), when there is one.
    pub needs_app: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimesInfo {
    pub auto_update: bool,
    /// Unix seconds of the last check, 0 when never.
    pub last_check: u64,
    pub runtimes: Vec<RuntimeStatus>,
}

fn info() -> RuntimesInfo {
    let state = load_state();
    let index = cached_index();
    let runtimes = Engine::ALL
        .into_iter()
        .filter(|e| e.here())
        .map(|e| {
            let b = e.bundled();
            let pack = active_pack(e).map(|(_, p)| p);
            let (active, upstream, rev, date) = match &pack {
                Some(p) => (p.version.clone(), p.upstream.clone(), p.rev.clone(), p.date.clone()),
                None => ("bundled".to_string(), b.upstream.to_string(), b.rev.to_string(), b.date.to_string()),
            };
            let update = e
                .swappable()
                .then(|| best_pack(&index, e, &state.broken))
                .flatten()
                .filter(|p| p.date.as_str() > date.as_str())
                .cloned();
            let needs_app = newer_needs_app(&index, e, &date);
            RuntimeStatus { engine: e, swappable: e.swappable(), active, upstream, rev, date, update, needs_app }
        })
        .collect();
    RuntimesInfo { auto_update: state.auto_update, last_check: state.last_check, runtimes }
}

/// Is `sig` (base64) a signature of `body` by `pubkey` (base64)?
fn signed_by(pubkey: &str, body: &[u8], sig: &str) -> bool {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD;
    let (Ok(key), Ok(sig)) = (b64.decode(pubkey.trim()), b64.decode(sig.trim())) else {
        return false;
    };
    ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, key).verify(body, &sig).is_ok()
}

/// Fetch the index and keep it if its signature holds.
async fn fetch_index() -> Result<(), String> {
    let client = crate::http::client_secs(UA, 20)?;
    let get = |url: String| {
        let client = client.clone();
        async move {
            client
                .get(url)
                .send()
                .await
                .and_then(|r| r.error_for_status())
                .map_err(|e| trf!("无法获取运行时列表:{e}", "could not fetch the runtime list: {e}"))?
                .bytes()
                .await
                .map_err(|e| trf!("无法获取运行时列表:{e}", "could not fetch the runtime list: {e}"))
        }
    };
    // CHATY_RUNTIMES_INDEX: another index, signed by the same key (testing a
    // pack before it is published).
    let url = std::env::var("CHATY_RUNTIMES_INDEX").unwrap_or_else(|_| INDEX_URL.to_string());
    let body = get(url.clone()).await?;
    let sig = get(format!("{url}.sig")).await?;
    if !signed_by(INDEX_PUBKEY, &body, &String::from_utf8_lossy(&sig)) {
        return Err(trf!(
            "运行时列表的签名对不上,已忽略这次获取的内容",
            "the runtime list's signature does not match — what was fetched was ignored"
        ));
    }
    let _: Index = serde_json::from_slice(&body).map_err(|e| trf!("运行时列表格式不对:{e}", "the runtime list is malformed: {e}"))?;
    std::fs::create_dir_all(root()).map_err(|e| e.to_string())?;
    std::fs::write(root().join("index.json"), &body).map_err(|e| e.to_string())?;
    let mut state = load_state();
    state.last_check = now();
    save_state(&state);
    Ok(())
}

/// Hex SHA-256 of a file.
fn sha256_of(path: &Path) -> Result<String, String> {
    use sha2::Digest;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = sha2::Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| e.to_string())?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Start `exe` and wait for its `ready` line: the pack runs here, and speaks
/// the protocol it says it does.
fn probe(e: Engine, exe: &Path) -> Result<(), String> {
    use std::io::BufRead;
    let mut cmd = std::process::Command::new(exe);
    cmd.stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    crate::agent::hide_console(&mut cmd);
    let mut child = cmd.spawn().map_err(|err| trf!("新引擎无法启动:{err}", "the new engine would not start: {err}"))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                if v.get("event").is_some() {
                    let _ = tx.send(v);
                    return;
                }
            }
        }
    });
    let first = rx.recv_timeout(std::time::Duration::from_secs(20));
    let _ = child.kill();
    let _ = child.wait();
    let ev = first.map_err(|_| trf!("新引擎启动后没有应答", "the new engine started but did not answer"))?;
    if ev["event"] != "ready" {
        return Err(trf!("新引擎的应答不对:{ev}", "the new engine answered something else: {ev}"));
    }
    let spoke = ev["protocol"].as_str().unwrap_or("1");
    if spoke != e.protocol() {
        return Err(trf!(
            "新引擎的协议版本是 {spoke},这个 Chaty 需要 {}",
            "the new engine speaks protocol {spoke}; this Chaty needs {}",
            e.protocol()
        ));
    }
    Ok(())
}

/// Unpack `zip` into `dir`, every entry kept inside it.
fn unpack(zip: &Path, dir: &Path) -> Result<(), String> {
    let f = std::fs::File::open(zip).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(f).map_err(|e| trf!("运行时包无法解压:{e}", "the runtime pack does not unpack: {e}"))?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let Some(rel) = entry.enclosed_name() else {
            return Err(trf!("运行时包里有越界的路径", "the runtime pack holds a path outside itself"));
        };
        let out = dir.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut w = std::fs::File::create(&out).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut w).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode & 0o777));
        }
    }
    Ok(())
}

/// How far an install has got, for its row.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
    /// "download" | "verify" | "start"
    pub stage: &'static str,
}

/// Download, check and switch to the newest usable pack for `e`. Takes effect
/// the next time the engine starts; a running one is left alone.
async fn install(e: Engine, progress: impl Fn(InstallProgress) + Send + Sync + 'static) -> Result<Pack, String> {
    if !e.swappable() {
        return Err(trf!("这个引擎随应用一起更新", "this engine is updated with the app"));
    }
    let state = load_state();
    let index = cached_index();
    let pack = best_pack(&index, e, &state.broken)
        .cloned()
        .ok_or_else(|| trf!("没有可用的新版本", "there is no newer version to install"))?;
    let dir = pack_dir(e, &pack.version);
    if dir.join("manifest.json").is_file() {
        let mut state = load_state();
        state.active.insert(e.id().to_string(), pack.version.clone());
        save_state(&state);
        return Ok(pack);
    }
    std::fs::create_dir_all(root().join(e.id())).map_err(|err| err.to_string())?;
    let zip = root().join(e.id()).join(format!("{}.zip.part", pack.version));
    let progress = std::sync::Arc::new(progress);
    let p = progress.clone();
    crate::http::download_to(UA, &pack.url, &zip, move |downloaded, total| {
        p(InstallProgress { downloaded, total, stage: "download" })
    })
    .await?;
    progress(InstallProgress { downloaded: pack.size, total: Some(pack.size), stage: "verify" });
    let staged = root().join(e.id()).join(format!("{}.partial", pack.version));
    let result = (|| {
        let got = sha256_of(&zip)?;
        if !got.eq_ignore_ascii_case(&pack.sha256) {
            return Err(trf!(
                "运行时包校验不通过(SHA-256 对不上),已丢弃",
                "the runtime pack failed its check (SHA-256 mismatch) and was discarded"
            ));
        }
        let _ = std::fs::remove_dir_all(&staged);
        std::fs::create_dir_all(&staged).map_err(|err| err.to_string())?;
        unpack(&zip, &staged)?;
        std::fs::write(staged.join("manifest.json"), serde_json::to_string_pretty(&pack).unwrap_or_default())
            .map_err(|err| err.to_string())?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&zip);
    if let Err(err) = result {
        let _ = std::fs::remove_dir_all(&staged);
        return Err(err);
    }
    progress(InstallProgress { downloaded: pack.size, total: Some(pack.size), stage: "start" });
    let exe = staged.join(e.exe());
    let probed = tauri::async_runtime::spawn_blocking(move || probe(e, &exe))
        .await
        .map_err(|err| err.to_string())?;
    if let Err(err) = probed {
        let _ = std::fs::remove_dir_all(&staged);
        let mut state = load_state();
        state.broken.insert(format!("{}:{}", e.id(), pack.version));
        save_state(&state);
        return Err(err);
    }
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::rename(&staged, &dir).map_err(|err| err.to_string())?;
    let mut state = load_state();
    let previous = state.active.insert(e.id().to_string(), pack.version.clone());
    save_state(&state);
    prune(e, &pack.version, previous.as_deref());
    Ok(pack)
}

/// Keep the pack in use and the one before it (a running engine may still be
/// it); remove the rest.
fn prune(e: Engine, keep: &str, previous: Option<&str>) {
    let Ok(rd) = std::fs::read_dir(root().join(e.id())) else { return };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name != keep && Some(name.as_str()) != previous {
            let _ = std::fs::remove_dir_all(entry.path());
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

// ---- commands ------------------------------------------------------------

#[tauri::command]
pub fn runtimes_info() -> RuntimesInfo {
    info()
}

/// Ask for the newest list, then say where every engine stands.
#[tauri::command]
pub async fn runtimes_check() -> Result<RuntimesInfo, String> {
    fetch_index().await?;
    Ok(info())
}

#[tauri::command]
pub async fn runtimes_update(engine: String, on_progress: Channel<InstallProgress>) -> Result<RuntimesInfo, String> {
    let e = Engine::from_id(&engine).ok_or_else(|| format!("unknown engine {engine}"))?;
    install(e, move |p| {
        let _ = on_progress.send(p);
    })
    .await?;
    Ok(info())
}

/// Back to the engine the app came with.
#[tauri::command]
pub fn runtimes_use_bundled(engine: String) -> RuntimesInfo {
    if let Some(e) = Engine::from_id(&engine) {
        let mut state = load_state();
        state.active.remove(e.id());
        save_state(&state);
    }
    info()
}

#[tauri::command]
pub fn runtimes_set_auto(on: bool) -> RuntimesInfo {
    let mut state = load_state();
    state.auto_update = on;
    save_state(&state);
    info()
}

/// Automatic updates, when turned on: a check a minute after launch and
/// daily after that, each newer usable pack installed for the next time its
/// engine starts. The UI hears of each one (`runtime-updated`).
pub fn start_auto_updates(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        loop {
            let state = load_state();
            if state.auto_update && now().saturating_sub(state.last_check) >= CHECK_EVERY_SECS && fetch_index().await.is_ok() {
                for e in Engine::ALL.into_iter().filter(|e| e.swappable() && e.here()) {
                    let has_update = info().runtimes.iter().any(|r| r.engine == e && r.update.is_some());
                    if !has_update {
                        continue;
                    }
                    match install(e, |_| {}).await {
                        Ok(pack) => {
                            let _ = app.emit("runtime-updated", serde_json::json!({ "engine": e, "version": pack.version }));
                        }
                        Err(err) => crate::errlog::append_error("runtime-auto-update", &format!("{}: {err}", e.id())),
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(engine: Engine, version: &str, protocol: &str, date: &str) -> Pack {
        Pack {
            engine,
            version: version.into(),
            platform: platform().into(),
            protocol: protocol.into(),
            min_app: "2.3.8".into(),
            upstream: "x/y".into(),
            rev: "abc".into(),
            date: date.into(),
            url: "https://example.invalid/p.zip".into(),
            size: 1,
            sha256: "00".into(),
        }
    }

    /// The newest pack for this app's protocol and platform, never one set
    /// aside; a newer engine for another protocol says a newer Chaty is needed.
    #[test]
    fn the_newest_usable_pack_is_chosen() {
        let index = Index {
            packs: vec![
                pack(Engine::Sd, "2026.10.01", "1", "2026-10-01T00:00:00Z"),
                pack(Engine::Sd, "2026.10.20", "1", "2026-10-20T00:00:00Z"),
                pack(Engine::Sd, "2026.10.25", "2", "2026-10-25T00:00:00Z"),
                pack(Engine::Audio, "2026.10.22", "1", "2026-10-22T00:00:00Z"),
            ],
        };
        let none = BTreeSet::new();
        assert_eq!(best_pack(&index, Engine::Sd, &none).unwrap().version, "2026.10.20");
        let broken: BTreeSet<String> = ["sd:2026.10.20".to_string()].into();
        assert_eq!(best_pack(&index, Engine::Sd, &broken).unwrap().version, "2026.10.01");
        assert_eq!(newer_needs_app(&index, Engine::Sd, "2026-10-20T00:00:00Z").as_deref(), Some("2.3.8"));
        assert!(best_pack(&index, Engine::Mlx, &none).is_none());
        // Nothing older than the engine the app came with.
        let old = Index { packs: vec![pack(Engine::Sd, "2020.01.01", "1", "2020-01-01T00:00:00Z")] };
        assert!(best_pack(&old, Engine::Sd, &none).is_none());
    }

    /// A signature over the exact bytes, by the right key, and nothing else.
    #[test]
    fn the_index_is_taken_only_with_a_good_signature() {
        use base64::Engine as _;
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let pair = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        use ring::signature::KeyPair;
        let b64 = base64::engine::general_purpose::STANDARD;
        let public = b64.encode(pair.public_key().as_ref());
        let body = br#"{"packs":[]}"#;
        let sig = b64.encode(pair.sign(body).as_ref());
        assert!(signed_by(&public, body, &sig));
        assert!(!signed_by(&public, br#"{"packs":[1]}"#, &sig));
        assert!(!signed_by(&public, body, "AAAA"));
        assert!(!signed_by("not a key", body, &sig));
    }

    /// The bundled engines named here are the ones the build pins.
    #[test]
    fn bundled_versions_match_the_build_pins() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let read = |p: &str| std::fs::read_to_string(root.join(p)).unwrap();
        assert!(read("sd-sidecar/CMakeLists.txt").contains(Engine::Sd.bundled().rev));
        assert!(read("audio-sidecar/CMakeLists.txt").contains(Engine::Audio.bundled().rev));
        assert!(read("mlx-sidecar/Package.swift").contains(Engine::Mlx.bundled().rev));
    }

    /// A pack's archive cannot write outside its folder.
    #[test]
    fn an_archive_path_outside_the_pack_is_refused() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("chaty-rt-zip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let zip_path = dir.join("p.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&zip_path).unwrap());
            z.start_file::<_, ()>("../escape.txt", zip::write::SimpleFileOptions::default()).unwrap();
            z.write_all(b"x").unwrap();
            z.finish().unwrap();
        }
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        assert!(unpack(&zip_path, &out).is_err());
        assert!(!dir.join("escape.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End to end against a real signed index: fetch it, install the newest
    /// pack for the engine, and that pack is the engine started from now on.
    /// Use a scratch HOME — the packs and the choice land in the data folder.
    /// `HOME=/tmp/x CHATY_RUNTIMES_INDEX=http://…/runtimes.json CHATY_TEST_RUNTIME=sd cargo test … -- --ignored`
    #[test]
    #[ignore]
    fn a_pack_installs_from_the_index_and_is_used() {
        let e = Engine::from_id(&std::env::var("CHATY_TEST_RUNTIME").expect("set CHATY_TEST_RUNTIME")).unwrap();
        tauri::async_runtime::block_on(fetch_index()).expect("fetch");
        let offered = info().runtimes.into_iter().find(|r| r.engine == e).and_then(|r| r.update).expect("an update on offer");
        let stages = std::sync::Mutex::new(Vec::new());
        let stages = std::sync::Arc::new(stages);
        let s = stages.clone();
        let pack = tauri::async_runtime::block_on(install(e, move |p| s.lock().unwrap().push(p.stage))).expect("install");
        assert_eq!(pack, offered);
        let mut seen = stages.lock().unwrap().clone();
        seen.dedup();
        assert_eq!(seen, ["download", "verify", "start"]);
        assert_eq!(active_binary(e), Some(pack_dir(e, &pack.version).join(e.exe())));
        let now = info().runtimes.into_iter().find(|r| r.engine == e).unwrap();
        assert_eq!((now.active.as_str(), now.rev.as_str()), (pack.version.as_str(), pack.rev.as_str()));
        assert!(now.update.is_none());
        eprintln!("{} now runs {} ({})", e.id(), pack.version, pack.rev);
    }
}
