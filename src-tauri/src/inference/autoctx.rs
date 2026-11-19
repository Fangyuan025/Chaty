//! The automatic context each model was found to hold on this machine.
//!
//! An automatic context is as long as memory holds, and finding that out is
//! work: a GGUF model on a card that cannot take its fit allocates, fails,
//! halves and reads all of its weights again for every try; an MLX model runs
//! a first pass through the whole model to measure what a token costs. The
//! answer does not change while the model, the machine and the app do not, so
//! the context a load settles on is kept here, and the next automatic load of
//! the same model starts from it. What memory is free that day is still
//! checked by the engine, and a day that has less shortens that load only.
//!
//! Kept per model path, with what the answer depends on: the file (size and
//! modification time — a model downloaded again may be another quantization),
//! the build that measured it (the fits are this app's arithmetic), the
//! engine settings that change what the GPU carries (GPU layers, the
//! speculative head), and the machine's memory and GPU. When any of those
//! differ the entry is not used, and the load's own answer replaces it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const FILE: &str = "auto-context.json";

/// The build that measured an entry: its commit when it has one (a re-cut of a
/// version is another build), otherwise its version.
const BUILD: &str = match option_env!("CHATY_BUILD") {
    Some(b) => b,
    None => env!("CARGO_PKG_VERSION"),
};

/// Which entry a load reads and writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Key {
    path: String,
    id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Entry {
    id: String,
    #[serde(flatten)]
    fit: CtxFit,
}

/// What an automatic load settled on, as an engine reports it and is handed
/// it back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CtxFit {
    pub n_ctx: u32,
    /// What a token of context costs and what the cache holds at any length,
    /// as the engine budgets them — from an engine that has to measure them
    /// (MLX runs the model to), so a load that starts from `n_ctx` can still
    /// weigh it against the memory free that day without measuring again.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_bytes: Option<u64>,
}

impl Key {
    /// The entry for loading `path` (a GGUF file or an MLX folder) with the
    /// automatic context. `gpu_layers` as the load takes it (negative or
    /// absent: automatic); `None` when the model's file cannot be read.
    pub fn new(path: &str, engine: &str, gpu_layers: Option<i32>, speculative: bool) -> Option<Self> {
        let p = Path::new(path);
        let canonical = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        let (size, mtime) = weights_stamp(&canonical)?;
        let gpu = match gpu_layers {
            Some(n) if n >= 0 => n.to_string(),
            _ => "auto".into(),
        };
        let id = format!("{BUILD}|{engine}|{size}|{mtime}|gpu {gpu}|mtp {speculative}|{}", machine());
        Some(Self { path: canonical.to_string_lossy().into_owned(), id })
    }
}

/// What `key`'s model settled on here last time, if it was short of its
/// trained length.
pub fn recall(key: &Key) -> Option<CtxFit> {
    recall_in(&crate::errlog::chaty_data_dir(), key)
}

/// Keep what an automatic load settled on (`fit`, as the engine reported it):
/// a context short of the trained length is kept for the next load, one that
/// is the whole of it forgets the entry. An engine that reports nothing
/// leaves the entry as it was.
pub fn settle(key: &Key, fit: Option<CtxFit>, trained: Option<u32>) {
    settle_in(&crate::errlog::chaty_data_dir(), key, fit, trained)
}

fn recall_in(base: &Path, key: &Key) -> Option<CtxFit> {
    read(base)
        .get(&key.path)
        .filter(|e| e.id == key.id && e.fit.n_ctx > 0)
        .map(|e| e.fit)
}

fn settle_in(base: &Path, key: &Key, fit: Option<CtxFit>, trained: Option<u32>) {
    let Some(fit) = fit.filter(|f| f.n_ctx > 0) else { return };
    let mut all = read(base);
    let short = trained.is_none_or(|t| fit.n_ctx < t);
    let before = all.get(&key.path).map(|e| (e.id.clone(), e.fit));
    if short {
        all.insert(key.path.clone(), Entry { id: key.id.clone(), fit });
    } else {
        all.remove(&key.path);
    }
    if all.get(&key.path).map(|e| (e.id.clone(), e.fit)) == before {
        return;
    }
    // Models deleted since are dropped on the way.
    all.retain(|path, _| Path::new(path).exists());
    eprintln!(
        "[autoctx] {}: {}",
        key.path,
        if short { format!("context {} kept for the next automatic load", fit.n_ctx) } else { "whole context, nothing kept".into() }
    );
    write(base, &all);
}

fn read(base: &Path) -> BTreeMap<String, Entry> {
    std::fs::read(base.join(FILE))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn write(base: &Path, all: &BTreeMap<String, Entry>) {
    let Ok(text) = serde_json::to_vec_pretty(all) else { return };
    let _ = std::fs::create_dir_all(base);
    // Whole or not at all: a load reading a half-written file would find no
    // entries, not wrong ones, but there is no reason to lose them.
    let tmp: PathBuf = base.join(format!("{FILE}.tmp"));
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, base.join(FILE));
    }
}

/// Size and modification time of what a load reads into memory: the GGUF
/// file, or an MLX folder's weight files (its configs may be rewritten by the
/// load itself).
fn weights_stamp(path: &Path) -> Option<(u64, u64)> {
    let stamp = |m: &std::fs::Metadata| {
        m.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs())
    };
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_dir() {
        return Some((meta.len(), stamp(&meta)));
    }
    let (mut size, mut newest, mut any) = (0u64, 0u64, false);
    for entry in std::fs::read_dir(path).ok()?.flatten() {
        if entry.path().extension().is_some_and(|e| e == "safetensors") {
            if let Ok(m) = entry.metadata() {
                size += m.len();
                newest = newest.max(stamp(&m));
                any = true;
            }
        }
    }
    any.then_some((size, newest))
}

/// What the machine has to hold a model in: its RAM and its GPU.
fn machine() -> String {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let ram_gib = (sys.total_memory() as f64 / (1u64 << 30) as f64).round() as u64;
    let gpu = crate::gpu::detect_gpu().map_or_else(
        || "no gpu".to_string(),
        |g| {
            // Linux reports a budget there, which moves with what else is
            // running; the card's own size does not.
            let size = if cfg!(target_os = "linux") { g.dedicated_vram_mb.unwrap_or(0) } else { g.vram_mb };
            format!("{} {size} MiB", g.name)
        },
    );
    format!("ram {ram_gib} GiB|{gpu}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("chaty-autoctx-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn n(n_ctx: u32) -> Option<CtxFit> {
        Some(CtxFit { n_ctx, ..Default::default() })
    }

    fn key(dir: &Path, id: &str) -> Key {
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"x").unwrap();
        Key { path: model.to_string_lossy().into_owned(), id: id.into() }
    }

    #[test]
    fn a_short_context_is_kept_and_a_whole_one_forgotten() {
        let base = fresh("kept");
        let k = key(&base, "a");
        assert_eq!(recall_in(&base, &k), None);
        settle_in(&base, &k, n(40960), Some(262144));
        assert_eq!(recall_in(&base, &k), n(40960));
        // An engine that reports nothing (one from before this) leaves it.
        settle_in(&base, &k, None, Some(262144));
        assert_eq!(recall_in(&base, &k), n(40960));
        // Halved again on a tighter day: the shorter one is kept.
        settle_in(&base, &k, n(20480), Some(262144));
        assert_eq!(recall_in(&base, &k), n(20480));
        // The whole trained length: nothing to remember.
        settle_in(&base, &k, n(262144), Some(262144));
        assert_eq!(recall_in(&base, &k), None);
    }

    #[test]
    fn an_entry_is_only_for_the_same_model_build_settings_and_machine() {
        let base = fresh("id");
        let k = key(&base, "build1|llama|1|2|gpu auto|mtp false|ram 48 GiB");
        settle_in(&base, &k, n(32768), Some(131072));
        let other = Key { id: "build2|llama|1|2|gpu auto|mtp false|ram 48 GiB".into(), ..k.clone() };
        assert_eq!(recall_in(&base, &other), None);
        // Measured again under the new id, it replaces the old entry.
        settle_in(&base, &other, n(16384), Some(131072));
        assert_eq!(recall_in(&base, &other), n(16384));
        assert_eq!(recall_in(&base, &k), None);
    }

    #[test]
    fn models_since_deleted_are_dropped() {
        let base = fresh("gone");
        let gone = base.join("gone.gguf");
        std::fs::write(&gone, b"x").unwrap();
        let g = Key { path: gone.to_string_lossy().into_owned(), id: "a".into() };
        settle_in(&base, &g, n(8192), Some(32768));
        std::fs::remove_file(&gone).unwrap();
        let k = key(&base, "a");
        settle_in(&base, &k, n(8192), Some(32768));
        assert_eq!(read(&base).len(), 1);
    }

    #[test]
    fn the_stamp_reads_a_file_and_an_mlx_folders_weights() {
        let dir = fresh("stamp");
        let f = dir.join("a.gguf");
        std::fs::write(&f, vec![0u8; 10]).unwrap();
        assert_eq!(weights_stamp(&f).map(|s| s.0), Some(10));
        let mlx = dir.join("mlx");
        std::fs::create_dir(&mlx).unwrap();
        assert_eq!(weights_stamp(&mlx), None);
        std::fs::write(mlx.join("model-1.safetensors"), vec![0u8; 3]).unwrap();
        std::fs::write(mlx.join("model-2.safetensors"), vec![0u8; 4]).unwrap();
        std::fs::write(mlx.join("config.json"), b"{}").unwrap();
        assert_eq!(weights_stamp(&mlx).map(|s| s.0), Some(7));
    }
}

/// Real engines, the way `load_model` drives them: an automatic load, kept,
/// then the same model again — which must start from what was kept, give the
/// same context, and say nothing about it. HOME is a scratch folder, so the
/// user's own file is not touched. Run (one engine or both):
/// CHATY_AUTOCTX_GGUF=a.gguf CHATY_AUTOCTX_MLX=dir cargo test --lib second_automatic_load -- --ignored --nocapture --test-threads=1
#[cfg(test)]
mod real {
    use super::*;
    use crate::inference::{llama_sidecar::LlamaSidecar, mlx::MlxEngine, InferenceBackend, ModelInfo};

    fn twice(engine: &str, path: &str, load: impl Fn(Option<CtxFit>) -> (Box<dyn InferenceBackend>, ModelInfo)) {
        let key = Key::new(path, engine, None, false).expect("key");
        assert_eq!(recall(&key), None, "a fresh HOME knows nothing");
        let (first, a) = load(recall(&key));
        first.unload();
        settle(&key, a.ctx_fit, a.n_ctx_train);
        let known = recall(&key);
        eprintln!("{engine}: first {:?} of {:?}, warning {:?}, kept {known:?}", a.n_ctx, a.n_ctx_train, a.warning);
        let (second, b) = load(known);
        second.unload();
        settle(&key, b.ctx_fit, b.n_ctx_train);
        eprintln!("{engine}: second {:?}, warning {:?}, kept {:?}", b.n_ctx, b.warning, recall(&key));
        if a.n_ctx < a.n_ctx_train {
            assert!(known.is_some(), "a short context is kept");
            assert_eq!(b.n_ctx, a.n_ctx, "the second load starts where the first settled");
            assert_ne!(b.warning.as_deref(), Some("ctx-clamped"), "and says nothing of it");
            assert_eq!(recall(&key), known, "nothing changed, nothing rewritten");
        } else {
            assert_eq!(known, None, "a whole context is not kept");
        }
    }

    #[test]
    #[ignore]
    fn second_automatic_load_starts_where_the_first_settled() {
        let home = std::env::temp_dir().join(format!("chaty-autoctx-home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        std::env::set_var("HOME", &home);
        if let Ok(gguf) = std::env::var("CHATY_AUTOCTX_GGUF") {
            twice("llama", &gguf, |known| {
                let (e, info) = LlamaSidecar::load(&gguf, None, None, known.map(|f| f.n_ctx), false, |_| {}).expect("load");
                (Box::new(e), info)
            });
        }
        if let Ok(dir) = std::env::var("CHATY_AUTOCTX_MLX") {
            twice("mlx", &dir, |known| {
                let (e, info) = MlxEngine::load(&dir, None, known, false, |_| {}).expect("load");
                (Box::new(e), info)
            });
        }
        eprintln!("{}", std::fs::read_to_string(home.join("Library/Application Support/com.chaty.desktop/auto-context.json")).unwrap_or_default());
    }
}
