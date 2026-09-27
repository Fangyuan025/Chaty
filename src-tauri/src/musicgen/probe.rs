//! Recognising a music model on disk, and finding the files it needs.
//!
//! audio.cpp writes its own GGUFs: `general.architecture = audiocpp`, and a
//! model file names its family in `audiocpp.model_spec.family`. Most families
//! ship as one self-contained file; YuE2 and MiniMax Music 3 are folders of
//! parts (see [`Layout`]). What the user picks — "the model" in the picker —
//! is the file that tells one quantization from another; the rest belongs to
//! it and is not listed on its own.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use serde::Serialize;

use super::family::{self, Family, Layout};

/// What a GGUF is, as far as music generation cares.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    /// An audio.cpp file naming its family.
    Model { family: String, weight_type: Option<String>, params: u64 },
    /// An audio.cpp file with no family of its own (a VAE, a component).
    Component { weight_type: Option<String>, params: u64 },
    /// Anything else.
    Other,
}

type Stamp = (u64, Option<SystemTime>);
/// Headers are re-read only when the file changes: `list_models` asks about
/// every GGUF in the models folder each time the picker opens.
static CACHE: Mutex<Option<HashMap<PathBuf, (Stamp, Kind)>>> = Mutex::new(None);

fn is_gguf(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

fn lower_name(p: &Path) -> String {
    p.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_lowercase()
}

fn kind(path: &Path) -> Kind {
    if !is_gguf(path) {
        return Kind::Other;
    }
    let meta = std::fs::metadata(path).ok();
    let key = (meta.as_ref().map(|m| m.len()).unwrap_or(0), meta.as_ref().and_then(|m| m.modified().ok()));
    if let Some(hit) = CACHE
        .lock()
        .ok()
        .and_then(|c| c.as_ref().and_then(|m| m.get(path).cloned()))
        .filter(|(stamp, _)| *stamp == key)
    {
        return hit.1;
    }
    let k = read_kind(path);
    if let Ok(mut c) = CACHE.lock() {
        c.get_or_insert_with(HashMap::new).insert(path.to_path_buf(), (key, k.clone()));
    }
    k
}

fn read_kind(path: &Path) -> Kind {
    let Ok(h) = crate::inference::gguf::read_header_file(path, true, true) else {
        return Kind::Other;
    };
    if h.has_tokenizer || h.arch.as_deref() != Some("audiocpp") {
        return Kind::Other;
    }
    // A package's VAE counts toward its size but not its parameters.
    let params = h
        .tensors
        .iter()
        .filter(|t| !t.name.starts_with("vae_weights/"))
        .map(|t| t.dims.iter().product::<u64>())
        .sum();
    match h.audiocpp_family {
        Some(family) if !family.is_empty() => Kind::Model { family, weight_type: h.audiocpp_weight_type, params },
        _ => Kind::Component { weight_type: h.audiocpp_weight_type, params },
    }
}

fn siblings(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    v.sort();
    v
}

/// The MiniMax Music 3 package this folder holds: one of its family-tagged
/// parts (the condition encoder, the vocoder) is here.
fn is_minimax_dir(dir: &Path) -> bool {
    siblings(dir)
        .iter()
        .any(|p| matches!(kind(p), Kind::Model { ref family, .. } if family == family::MINIMAX_MUSIC3.id))
}

/// The family of a file the user can pick, or None when the file is not one
/// (a part of a package, or not a music model at all).
pub fn pick_family(path: &Path) -> Option<String> {
    match kind(path) {
        // MiniMax's family-tagged files are its small parts; the language
        // model quantization is the pick.
        Kind::Model { family, .. } if family != family::MINIMAX_MUSIC3.id => Some(family),
        Kind::Component { .. } if lower_name(path).starts_with("language_model") => path
            .parent()
            .filter(|d| is_minimax_dir(d))
            .map(|_| family::MINIMAX_MUSIC3.id.to_string()),
        _ => None,
    }
}

/// Is this file a music model the user picks — what the music engine loads?
/// Families Chaty's engine is not built with count too: loading one says so,
/// instead of llama.cpp failing on a file it cannot read.
pub fn is_music_model(path: &Path) -> bool {
    pick_family(path).is_some()
}

/// Is this file an audio.cpp GGUF at all (a model or one of its parts)? Such
/// a file is never a chat model.
pub fn is_audiocpp_file(path: &Path) -> bool {
    !matches!(kind(path), Kind::Other)
}

/// The music models in a folder, smallest first — the quickest to try.
pub fn music_models_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = siblings(dir).into_iter().filter(|p| is_music_model(p)).collect();
    found.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX));
    found
}

pub fn music_model_in_dir(dir: &Path) -> Option<PathBuf> {
    music_models_in_dir(dir).into_iter().next()
}

/// A part a music model needs and has.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Component {
    /// "vae", "config", "tokenizer" or "weights".
    pub role: &'static str,
    /// Relative to the model's folder, with `/` separators.
    pub file: String,
    pub size: u64,
}

/// A download that supplies a missing part.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Suggestion {
    pub role: &'static str,
    pub repo: String,
    /// Repo-relative path.
    pub file: String,
    /// Where it goes, relative to the model's folder.
    pub dest: String,
    pub size: u64,
}

/// Everything known about a music model before it is loaded.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MusicProbe {
    /// The file picked.
    pub path: String,
    /// audio.cpp's family name ("yue2").
    pub family: String,
    pub family_name: String,
    /// Chaty's engine runs this family.
    pub supported: bool,
    pub layout: Option<Layout>,
    /// "Q4_0", from the file's own weight type.
    pub quant: Option<String>,
    pub params_b: Option<f64>,
    /// Everything that is loaded, together.
    pub size_mb: u64,
    pub components: Vec<Component>,
    /// Required parts not found (folder-relative paths).
    pub missing: Vec<String>,
    /// Downloads for what is missing.
    pub suggestions: Vec<Suggestion>,
    /// What the engine is given: the file, or the package folder.
    pub model_path: String,
    /// The package's files the engine is told to use (session option → file).
    pub files: BTreeMap<String, String>,
}

fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn rel(dir: &Path, file: &str) -> PathBuf {
    file.split('/').fold(dir.to_path_buf(), |acc, part| acc.join(part))
}

fn name_of(p: &Path) -> String {
    p.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string()
}

/// The repo path of a file of the family's published package.
fn repo_file(fam: &Family, file: &str) -> String {
    if fam.repo_dir.is_empty() {
        file.to_string()
    } else {
        format!("{}/{}", fam.repo_dir, file)
    }
}

/// Files of the published YuE2 package besides the model: (path, size,
/// required).
const YUE2_SIDECARS: &[(&str, u64, bool)] = &[
    ("sidecars/yue2-model-config.json", 959, true),
    ("sidecars/yue2-vae-config.json", 1378, true),
    ("sidecars/yue2-qwen.tiktoken", 2_561_218, true),
    // The engine has defaults for everything in it.
    ("sidecars/yue2-generation-config.json", 466, false),
];
/// YuE2's VAEs, preferred first: F16 is what audio.cpp loads by default and
/// what the published demos pair with the Q8/Q4 models.
const YUE2_VAES: &[(&str, u64)] = &[("yue2-vae-f16.gguf", 265_218_656), ("yue2-vae-f32.gguf", 530_537_760)];

/// The MiniMax Music 3 package's fixed files (path, size).
const MINIMAX_FIXED: &[(&str, u64)] = &[
    ("condition_encoder.gguf", 100_677_184),
    ("vocoder.gguf", 216_704_192),
    ("config.json", 107),
    ("config/condition_encoder.json", 292),
    ("config/language_model.json", 1596),
    ("config/rvq_depth_decoder.json", 274),
    ("config/transformer.json", 294),
    ("config/vocoder.json", 251),
    ("tokenizer/tokenizer.json", 11_423_801),
    ("tokenizer/tokenizer_config.json", 377),
];
/// Its quantized parts: flow transformer and RVQ depth decoder, by
/// quantization (sizes of the published files).
const MINIMAX_TRANSFORMERS: &[(&str, u64)] = &[
    ("q4_0", 1_396_392_768),
    ("q4_k", 1_396_397_600),
    ("q8_0", 2_606_842_688),
    ("bf16", 9_727_686_016),
];
const MINIMAX_DEPTH: &[(&str, u64)] = &[("q4_k", 405_752_480), ("q8_0", 714_028_960), ("bf16", 1_292_061_152)];

/// The quantization in a MiniMax part's name: `language_model_q4_0.gguf` → q4_0.
fn part_quant(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".gguf")?;
    ["language_model_", "rvq_depth_decoder_", "transformer_"]
        .iter()
        .find_map(|prefix| stem.strip_prefix(prefix))
        .map(str::to_string)
}

/// Probe a pickable music model file: its family, and what of its folder is
/// there. `None` when the file is not one.
pub fn probe_music_model(path: &Path) -> Option<MusicProbe> {
    let fam_id = pick_family(path)?;
    let fam = family::by_id(&fam_id);
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let (weight_type, params) = match kind(path) {
        Kind::Model { weight_type, params, .. } | Kind::Component { weight_type, params } => (weight_type, params),
        Kind::Other => (None, 0),
    };
    let mut p = MusicProbe {
        path: path.to_string_lossy().to_string(),
        family_name: fam.map(|f| f.name.to_string()).unwrap_or_else(|| fam_id.clone()),
        supported: fam.is_some(),
        layout: fam.map(|f| f.layout),
        family: fam_id,
        quant: weight_type.filter(|w| w != "orig").map(|w| w.to_uppercase()),
        params_b: None,
        size_mb: 0,
        components: Vec::new(),
        missing: Vec::new(),
        suggestions: Vec::new(),
        model_path: path.to_string_lossy().to_string(),
        files: BTreeMap::new(),
    };
    let mut size = file_len(path);
    let mut params = params;
    match fam.map(|f| f.layout) {
        Some(Layout::Yue2) => {
            let fam = fam.unwrap();
            p.model_path = dir.to_string_lossy().to_string();
            p.files.insert("yue2.model_gguf".into(), name_of(path));
            // The VAE: a family-less audio.cpp GGUF beside it, the published
            // names in their order first.
            let mut vaes: Vec<PathBuf> =
                siblings(&dir).into_iter().filter(|s| matches!(kind(s), Kind::Component { .. })).collect();
            let rank = |s: &PathBuf| {
                let n = lower_name(s);
                YUE2_VAES.iter().position(|(v, _)| *v == n).unwrap_or(if n.contains("vae") { 100 } else { 200 })
            };
            vaes.sort_by_key(|s| (rank(s), s.clone()));
            match vaes.first() {
                Some(v) => {
                    size += file_len(v);
                    p.files.insert("yue2.vae_gguf".into(), name_of(v));
                    p.components.push(Component { role: "vae", file: name_of(v), size: file_len(v) });
                }
                None => {
                    let (file, sz) = YUE2_VAES[0];
                    p.missing.push(file.into());
                    p.suggestions.push(Suggestion { role: "vae", repo: fam.repo.into(), file: repo_file(fam, file), dest: file.into(), size: sz });
                }
            }
            for (file, sz, required) in YUE2_SIDECARS {
                let f = rel(&dir, file);
                if f.is_file() {
                    p.components.push(Component { role: "config", file: file.to_string(), size: file_len(&f) });
                } else {
                    if *required {
                        p.missing.push(file.to_string());
                    }
                    // Optional ones come along when anything is fetched, so
                    // the folder ends up as the published package.
                    p.suggestions.push(Suggestion {
                        role: "config",
                        repo: fam.repo.into(),
                        file: repo_file(fam, file),
                        dest: file.to_string(),
                        size: *sz,
                    });
                }
            }
        }
        Some(Layout::MiniMax) => {
            let fam = fam.unwrap();
            p.model_path = dir.to_string_lossy().to_string();
            let lm_quant = part_quant(&lower_name(path)).unwrap_or_default();
            p.quant = Some(lm_quant.to_uppercase()).filter(|q| !q.is_empty()).or(p.quant);
            p.files.insert("minimax_music3.language_model_gguf".into(), name_of(path));
            // The flow transformer and depth decoder: the same quantization
            // as the language model when there is one, else the smallest
            // there is — the published default mix is a Q4_0 model with a
            // Q8_0 depth decoder.
            let mut pick_part = |prefix: &str, option: &str, published: &[(&str, u64)], prefer: &[&str]| {
                let have: Vec<PathBuf> = siblings(&dir)
                    .into_iter()
                    .filter(|s| lower_name(s).starts_with(prefix) && is_gguf(s))
                    .collect();
                let quant_of = |s: &PathBuf| part_quant(&lower_name(s)).unwrap_or_default();
                let chosen = have
                    .iter()
                    .find(|s| quant_of(s) == lm_quant)
                    .or_else(|| prefer.iter().find_map(|q| have.iter().find(|s| quant_of(s) == *q)))
                    .or_else(|| have.iter().min_by_key(|s| file_len(s)))
                    .cloned();
                match chosen {
                    Some(c) => {
                        size += file_len(&c);
                        params += match kind(&c) {
                            Kind::Component { params, .. } | Kind::Model { params, .. } => params,
                            Kind::Other => 0,
                        };
                        p.files.insert(option.into(), name_of(&c));
                        p.components.push(Component { role: "weights", file: name_of(&c), size: file_len(&c) });
                    }
                    None => {
                        let q = published
                            .iter()
                            .find(|(q, _)| *q == lm_quant)
                            .or_else(|| prefer.iter().find_map(|w| published.iter().find(|(q, _)| q == w)))
                            .copied()
                            .unwrap_or(published[0]);
                        let file = format!("{prefix}_{}.gguf", q.0);
                        p.missing.push(file.clone());
                        p.suggestions.push(Suggestion {
                            role: "weights",
                            repo: fam.repo.into(),
                            file: repo_file(fam, &file),
                            dest: file,
                            size: q.1,
                        });
                    }
                }
            };
            pick_part("transformer", "minimax_music3.flow_transformer_gguf", MINIMAX_TRANSFORMERS, &["q4_0", "q8_0"]);
            pick_part("rvq_depth_decoder", "minimax_music3.rvq_depth_decoder_gguf", MINIMAX_DEPTH, &["q8_0", "q4_k"]);
            for (file, sz) in MINIMAX_FIXED {
                let f = rel(&dir, file);
                let role = if file.starts_with("tokenizer") {
                    "tokenizer"
                } else if file.ends_with(".json") {
                    "config"
                } else {
                    "weights"
                };
                if f.is_file() {
                    size += if file.ends_with(".gguf") { file_len(&f) } else { 0 };
                    p.components.push(Component { role, file: file.to_string(), size: file_len(&f) });
                } else {
                    p.missing.push(file.to_string());
                    p.suggestions.push(Suggestion {
                        role,
                        repo: fam.repo.into(),
                        file: repo_file(fam, file),
                        dest: file.to_string(),
                        size: *sz,
                    });
                }
            }
        }
        // One file is the whole model.
        Some(Layout::Single) | None => {}
    }
    if p.missing.is_empty() {
        p.suggestions.clear();
    }
    p.size_mb = size / (1024 * 1024);
    p.params_b = (params > 0).then(|| (params as f64 / 1e8).round() / 10.0);
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::gguf::tests::gguf_bytes;

    fn fresh_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("chaty-music-probe-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn audiocpp_file(dir: &Path, name: &str, family: Option<&str>, weight_type: &str) -> PathBuf {
        let p = dir.join(name);
        let mut kv = vec![("general.architecture", "audiocpp"), ("audiocpp.weight_type", weight_type)];
        if let Some(f) = family {
            kv.push(("audiocpp.model_spec.family", f));
        }
        std::fs::write(&p, gguf_bytes(&kv, &[("model_weights/x", &[1000, 1000])])).unwrap();
        p
    }

    fn touch(dir: &Path, file: &str) {
        let p = rel(dir, file);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"{}").unwrap();
    }

    #[test]
    fn a_bare_yue2_model_lists_everything_it_needs_from_its_repo() {
        let dir = fresh_dir("yue2-bare");
        let main = audiocpp_file(&dir, "yue2-3b-q4_0.gguf", Some("yue2"), "q4_0");
        assert!(is_music_model(&main));
        let p = probe_music_model(&main).expect("a music model");
        assert_eq!((p.family.as_str(), p.family_name.as_str(), p.supported), ("yue2", "YuE2", true));
        assert_eq!(p.quant.as_deref(), Some("Q4_0"));
        assert_eq!(p.model_path, dir.to_string_lossy());
        assert_eq!(p.files.get("yue2.model_gguf").map(String::as_str), Some("yue2-3b-q4_0.gguf"));
        assert_eq!(
            p.missing,
            vec!["yue2-vae-f16.gguf", "sidecars/yue2-model-config.json", "sidecars/yue2-vae-config.json", "sidecars/yue2-qwen.tiktoken"]
        );
        // The optional generation config comes along with the rest.
        assert_eq!(p.suggestions.len(), 5);
        assert!(p.suggestions.iter().all(|s| s.repo == "audio-cpp/Yue2-3B-GGUF" && s.file == s.dest));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_complete_yue2_folder_needs_nothing_and_picks_the_f16_vae() {
        let dir = fresh_dir("yue2-full");
        let main = audiocpp_file(&dir, "yue2-3b-q4_0.gguf", Some("yue2"), "q4_0");
        audiocpp_file(&dir, "yue2-vae-f32.gguf", None, "f32");
        audiocpp_file(&dir, "yue2-vae-f16.gguf", None, "f16");
        for f in ["sidecars/yue2-model-config.json", "sidecars/yue2-vae-config.json", "sidecars/yue2-qwen.tiktoken"] {
            touch(&dir, f);
        }
        let p = probe_music_model(&main).unwrap();
        assert!(p.missing.is_empty(), "{:?}", p.missing);
        assert!(p.suggestions.is_empty(), "an optional file alone is no reason to download");
        assert_eq!(p.files.get("yue2.vae_gguf").map(String::as_str), Some("yue2-vae-f16.gguf"));
        // The VAE is a part, never a model — of any kind.
        let vae = dir.join("yue2-vae-f16.gguf");
        assert!(!is_music_model(&vae) && is_audiocpp_file(&vae));
        assert_eq!(music_models_in_dir(&dir), vec![main]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_minimax_package_is_picked_by_its_language_model_and_matches_quantizations() {
        let dir = fresh_dir("minimax");
        audiocpp_file(&dir, "condition_encoder.gguf", Some("minimax_music3"), "orig");
        audiocpp_file(&dir, "vocoder.gguf", Some("minimax_music3"), "orig");
        let q4 = audiocpp_file(&dir, "language_model_q4_0.gguf", None, "q4_0");
        let q8 = audiocpp_file(&dir, "language_model_q8_0.gguf", None, "q8_0");
        audiocpp_file(&dir, "transformer_q4_0.gguf", None, "q4_0");
        audiocpp_file(&dir, "transformer_q8_0.gguf", None, "q8_0");
        audiocpp_file(&dir, "rvq_depth_decoder_q8_0.gguf", None, "q8_0");
        for (f, _) in MINIMAX_FIXED.iter().filter(|(f, _)| !f.ends_with(".gguf")) {
            touch(&dir, f);
        }
        // The small family-tagged parts are not picks; the language models are.
        assert!(!is_music_model(&dir.join("vocoder.gguf")));
        assert!(!is_music_model(&dir.join("transformer_q4_0.gguf")));
        let mut picks = music_models_in_dir(&dir);
        picks.sort();
        assert_eq!(picks, vec![q4.clone(), q8.clone()]);

        let p = probe_music_model(&q4).unwrap();
        assert_eq!(p.family, "minimax_music3");
        assert_eq!(p.quant.as_deref(), Some("Q4_0"));
        assert!(p.missing.is_empty(), "{:?}", p.missing);
        assert_eq!(p.files.get("minimax_music3.flow_transformer_gguf").map(String::as_str), Some("transformer_q4_0.gguf"));
        // No Q4_0 depth decoder: the published default, Q8_0.
        assert_eq!(p.files.get("minimax_music3.rvq_depth_decoder_gguf").map(String::as_str), Some("rvq_depth_decoder_q8_0.gguf"));
        let p8 = probe_music_model(&q8).unwrap();
        assert_eq!(p8.files.get("minimax_music3.flow_transformer_gguf").map(String::as_str), Some("transformer_q8_0.gguf"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_minimax_language_model_alone_asks_for_the_rest_of_its_package() {
        let dir = fresh_dir("minimax-bare");
        audiocpp_file(&dir, "vocoder.gguf", Some("minimax_music3"), "orig");
        let lm = audiocpp_file(&dir, "language_model_q8_0.gguf", None, "q8_0");
        let p = probe_music_model(&lm).unwrap();
        assert!(p.missing.contains(&"transformer_q8_0.gguf".to_string()), "{:?}", p.missing);
        assert!(p.missing.contains(&"rvq_depth_decoder_q8_0.gguf".to_string()));
        assert!(p.missing.contains(&"tokenizer/tokenizer.json".to_string()));
        assert!(!p.missing.contains(&"vocoder.gguf".to_string()));
        assert!(p.suggestions.iter().all(|s| s.repo == "audio-cpp/MiniMax-Music3-GGUF"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_single_file_model_is_whole() {
        let dir = fresh_dir("single");
        let f = audiocpp_file(&dir, "stable-audio-3-small-music-q8_0.gguf", Some("stable_audio"), "q8_0");
        let p = probe_music_model(&f).unwrap();
        assert_eq!(p.family_name, "Stable Audio 3");
        assert_eq!(p.model_path, f.to_string_lossy());
        assert!(p.missing.is_empty() && p.files.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn other_ggufs_are_not_music() {
        let dir = fresh_dir("other");
        let chat = dir.join("qwen.gguf");
        std::fs::write(&chat, gguf_bytes(&[("general.architecture", "qwen3"), ("tokenizer.ggml.model", "gpt2")], &[])).unwrap();
        let flux = dir.join("flux.gguf");
        std::fs::write(&flux, gguf_bytes(&[("general.architecture", "flux")], &[("double_blocks.0.x", &[4])])).unwrap();
        assert!(!is_music_model(&chat) && !is_audiocpp_file(&chat));
        assert!(!is_music_model(&flux) && !is_audiocpp_file(&flux));
        assert!(probe_music_model(&chat).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A family Chaty's engine is not built with is still recognised, so the
    /// load can say what it is instead of llama.cpp failing on it.
    #[test]
    fn an_unsupported_family_is_named() {
        let dir = fresh_dir("unsupported");
        let f = audiocpp_file(&dir, "kokoro.gguf", Some("kokoro_tts"), "q8_0");
        let p = probe_music_model(&f).unwrap();
        assert!(!p.supported);
        assert_eq!(p.family_name, "kokoro_tts");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn part_quantizations_are_read_from_names() {
        assert_eq!(part_quant("language_model_q4_0.gguf").as_deref(), Some("q4_0"));
        assert_eq!(part_quant("transformer_q4_k.gguf").as_deref(), Some("q4_k"));
        assert_eq!(part_quant("rvq_depth_decoder_bf16.gguf").as_deref(), Some("bf16"));
    }
}
