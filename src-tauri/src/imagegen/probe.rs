//! Recognising an image model on disk, and finding the files it needs.
//!
//! A GGUF is either a language model — it carries a tokenizer — or it is not,
//! and a diffusion denoiser is recognised from what is left: an architecture
//! the diffusion converters write, or tensor names only a denoiser has. The
//! same test reads `.safetensors` headers, where Comfy-Org and friends publish
//! most denoisers and nearly every VAE.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::family::{self, Family, Role, Tensors};

/// What a weights file is, as far as image generation cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A language model GGUF (has a tokenizer) — a chat model, and possibly a
    /// text encoder for some image model.
    Llm,
    /// A vision projector GGUF (`general.architecture = clip`).
    Mmproj,
    /// A diffusion denoiser, alone or inside an all-in-one checkpoint.
    Denoiser,
    /// A standalone autoencoder.
    Vae,
    /// Anything else (a T5 or CLIP encoder, a LoRA, an unreadable file).
    Other,
}

/// The extensions a weights file can have here.
pub fn is_weights_file(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "gguf" | "safetensors" | "sft"))
}

fn is_gguf(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

fn lower_name(p: &Path) -> String {
    p.file_name().and_then(|s| s.to_str()).unwrap_or_default().to_lowercase()
}

/// Everything read from one file's header that later questions need.
#[derive(Debug, Clone)]
struct Probe {
    kind: Kind,
    family: Option<&'static Family>,
    /// Total tensor elements — the parameter count, for a denoiser.
    params: u64,
}

/// Headers are re-read only when the file changes. `list_models` asks about
/// every file in the models folder each time the picker opens, and a GGUF's
/// header is not free to read.
type Stamp = (u64, Option<SystemTime>);
static CACHE: Mutex<Option<HashMap<PathBuf, (Stamp, Probe)>>> = Mutex::new(None);

fn probe(path: &Path) -> Probe {
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
    let p = read_probe(path);
    if let Ok(mut c) = CACHE.lock() {
        c.get_or_insert_with(HashMap::new).insert(path.to_path_buf(), (key, p.clone()));
    }
    p
}

fn read_probe(path: &Path) -> Probe {
    let other = Probe { kind: Kind::Other, family: None, params: 0 };
    let name = lower_name(path);
    if is_gguf(path) {
        let Ok(h) = crate::inference::gguf::read_header_file(path, true, true) else {
            return other;
        };
        if h.has_tokenizer {
            return Probe { kind: Kind::Llm, ..other };
        }
        if h.arch.as_deref() == Some("clip") || name.contains("mmproj") {
            return Probe { kind: Kind::Mmproj, ..other };
        }
        let names: Vec<(&str, &[u64])> = h.tensors.iter().map(|t| (t.name.as_str(), t.dims.as_slice())).collect();
        let tensors = Tensors::new(names);
        let known_arch = h.arch.as_deref().is_some_and(|a| family::DIFFUSION_ARCHS.contains(&a));
        if known_arch || tensors.is_denoiser() {
            let fam = family::detect(&tensors, h.arch.as_deref(), &name);
            let params = h.tensors.iter().map(|t| t.dims.iter().product::<u64>()).sum();
            return Probe { kind: Kind::Denoiser, family: Some(fam), params };
        }
        if tensors.is_vae() {
            return Probe { kind: Kind::Vae, ..other };
        }
        return other;
    }
    // safetensors: tensor names and shapes live in the JSON header. An MLX
    // model folder is a language model in safetensors and is never an image
    // model's file.
    if path.parent().is_some_and(crate::inference::mlx::is_mlx_dir) {
        return other;
    }
    let Ok((_, header)) = crate::inference::mlx::read_st_header(path) else {
        return other;
    };
    let Some(obj) = header.as_object() else { return other };
    let shapes: Vec<(String, Vec<u64>)> = obj
        .iter()
        .filter(|(k, _)| k.as_str() != "__metadata__")
        .map(|(k, v)| {
            let dims = v
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();
            (k.clone(), dims)
        })
        .collect();
    let names: Vec<(&str, &[u64])> = shapes.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    let tensors = Tensors::new(names);
    if tensors.is_denoiser() {
        let fam = family::detect(&tensors, None, &name);
        let params = shapes.iter().map(|(_, d)| d.iter().product::<u64>()).sum();
        return Probe { kind: Kind::Denoiser, family: Some(fam), params };
    }
    if tensors.is_vae() || name.contains("vae") || name == "ae.safetensors" || name == "ae.sft" {
        return Probe { kind: Kind::Vae, ..other };
    }
    other
}

/// Is this file a diffusion model — the thing the image engine loads?
pub fn is_image_model(path: &Path) -> bool {
    is_weights_file(path) && probe(path).kind == Kind::Denoiser
}

/// The kind of a weights file (cached).
pub fn kind_of(path: &Path) -> Kind {
    if !is_weights_file(path) {
        return Kind::Other;
    }
    probe(path).kind
}

/// A companion the user picked by hand in Settings, per role key. An empty
/// string means "none" — the user took an auto-detected file away.
pub type Overrides = HashMap<String, String>;

/// How a companion was found.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Component {
    pub role: Role,
    pub path: String,
    pub size_mb: u64,
    /// "override" (chosen in Settings) | "folder" (beside the model) |
    /// "shared" (another model's folder — one encoder serves several models).
    pub source: &'static str,
}

/// A download that would supply a missing companion.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Suggestion {
    pub role: Role,
    pub repo: String,
    pub file: String,
    pub size: u64,
}

/// Everything known about an image model before it is loaded.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageProbe {
    pub path: String,
    pub family: String,
    pub family_name: String,
    pub all_in_one: bool,
    pub params_b: Option<f64>,
    pub quant: Option<String>,
    pub size_mb: u64,
    pub components: Vec<Component>,
    /// Required companions that were not found.
    pub missing: Vec<Role>,
    /// Downloads that fill `missing`.
    pub suggestions: Vec<Suggestion>,
    pub requires: Vec<Role>,
    pub optional: Vec<Role>,
    pub defaults: family::Defaults,
    /// Reference-picture editing is possible: the family edits and the
    /// encoder's vision tower is present.
    pub edits: bool,
}

fn size_mb(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) / (1024 * 1024)
}

fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_file() && is_weights_file(p)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// Does `path` look like the file `role` asks for, for this family?
fn fits(role: Role, path: &Path, fam: &Family, need_hint: bool) -> bool {
    let name = lower_name(path);
    let hinted = fam
        .hints
        .iter()
        .find(|(r, _)| *r == role)
        .is_some_and(|(_, hs)| hs.iter().any(|h| name.contains(h)));
    if need_hint && !hinted {
        return false;
    }
    let kind = kind_of(path);
    match role {
        Role::Vae => kind == Kind::Vae || name.contains("vae") || name.starts_with("ae."),
        Role::Llm => {
            (kind == Kind::Llm && !name.contains("mmproj"))
                || (!is_gguf(path) && hinted && kind == Kind::Other)
        }
        Role::LlmVision => kind == Kind::Mmproj || name.contains("mmproj"),
        Role::ClipL => name.contains("clip_l") || name.contains("clip-l"),
        Role::ClipG => name.contains("clip_g") || name.contains("clip-g"),
        Role::T5xxl => name.contains("t5"),
    }
}

/// Find a family's companions for the model at `model`: a hand-picked file
/// first, then the model's own folder, then — by name only, so an unrelated
/// file is never taken for one — the other folders under the models roots.
pub fn find_components(model: &Path, fam: &Family, overrides: &Overrides, roots: &[PathBuf]) -> Vec<Component> {
    let dir = model.parent().unwrap_or(Path::new("."));
    let beside: Vec<PathBuf> = files_in(dir).into_iter().filter(|p| p != model).collect();
    let mut elsewhere: Option<Vec<PathBuf>> = None;
    let mut out: Vec<Component> = Vec::new();

    let wanted: Vec<Role> = fam.requires.iter().chain(fam.optional.iter()).copied().collect();
    for role in wanted {
        if role == Role::LlmVision {
            continue; // paired with the encoder below
        }
        if let Some(o) = overrides.get(role.key()) {
            if o.is_empty() {
                continue; // taken away on purpose
            }
            let p = PathBuf::from(o);
            if p.is_file() {
                out.push(Component { role, size_mb: size_mb(&p), path: o.clone(), source: "override" });
            }
            continue;
        }
        // Beside the model: the family's own names first, then anything of
        // the right kind — the user put it there.
        let here = beside
            .iter()
            .find(|p| fits(role, p, fam, true))
            .or_else(|| beside.iter().find(|p| fits(role, p, fam, false)));
        if let Some(p) = here {
            out.push(Component { role, size_mb: size_mb(p), path: p.to_string_lossy().to_string(), source: "folder" });
            continue;
        }
        if fam.hints.iter().all(|(r, _)| *r != role) {
            continue; // nothing to recognise it by elsewhere
        }
        let pool = elsewhere.get_or_insert_with(|| {
            let mut v = Vec::new();
            for root in roots {
                v.extend(files_in(root));
                if let Ok(rd) = std::fs::read_dir(root) {
                    let mut subs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir() && p != dir).collect();
                    subs.sort();
                    for s in subs {
                        v.extend(files_in(&s));
                    }
                }
            }
            v
        });
        if let Some(p) = pool.iter().find(|p| *p != model && fits(role, p, fam, true)) {
            out.push(Component { role, size_mb: size_mb(p), path: p.to_string_lossy().to_string(), source: "shared" });
        }
    }

    // The vision tower belongs to the encoder: an mmproj of some other model
    // would load and then read the picture wrong.
    if fam.optional.contains(&Role::LlmVision) || fam.requires.contains(&Role::LlmVision) {
        match overrides.get(Role::LlmVision.key()) {
            Some(o) if o.is_empty() => {}
            Some(o) if Path::new(o).is_file() => {
                out.push(Component { role: Role::LlmVision, size_mb: size_mb(Path::new(o)), path: o.clone(), source: "override" });
            }
            _ => {
                if let Some(llm) = out.iter().find(|c| c.role == Role::Llm) {
                    // `find_mmproj` wants a name in common once a folder holds
                    // several GGUFs — and an image model's folder always does
                    // (the denoiser is one). There, the encoder being the only
                    // language model beside a single projector is pairing
                    // enough.
                    let lone = || {
                        let files = files_in(Path::new(&llm.path).parent()?);
                        let llms = files.iter().filter(|p| kind_of(p) == Kind::Llm).count();
                        let projs: Vec<&PathBuf> = files.iter().filter(|p| kind_of(p) == Kind::Mmproj).collect();
                        (llms == 1 && projs.len() == 1).then(|| projs[0].clone())
                    };
                    if let Some(mm) = crate::inference::llama::find_mmproj(&llm.path).or_else(lone) {
                        out.push(Component {
                            role: Role::LlmVision,
                            size_mb: size_mb(&mm),
                            path: mm.to_string_lossy().to_string(),
                            source: llm.source,
                        });
                    }
                }
            }
        }
    }
    out
}

/// Everything about the image model at `path`. `None` when it is not one.
pub fn probe_image_model(path: &Path, overrides: &Overrides, roots: &[PathBuf]) -> Option<ImageProbe> {
    let p = probe(path);
    if p.kind != Kind::Denoiser {
        return None;
    }
    let fam = p.family.unwrap_or(&family::GENERIC);
    let name = lower_name(path);
    let components = find_components(path, fam, overrides, roots);
    let have = |r: Role| components.iter().any(|c| c.role == r);
    let missing: Vec<Role> = fam.requires.iter().copied().filter(|r| !have(*r)).collect();
    let suggestions = fam
        .companions
        .iter()
        .filter(|c| missing.contains(&c.role))
        .map(|c| Suggestion { role: c.role, repo: c.repo.into(), file: c.file.into(), size: c.size })
        .collect();
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    let quant = Some(crate::download::quant_label_of(stem)).filter(|q| q != "DEFAULT");
    let own = size_mb(path)
        + components.iter().filter(|c| c.source == "folder").map(|c| c.size_mb).sum::<u64>();
    Some(ImageProbe {
        path: path.to_string_lossy().to_string(),
        family: fam.id.into(),
        family_name: fam.name.into(),
        all_in_one: fam.all_in_one,
        params_b: (p.params > 0).then(|| (p.params as f64 / 1e8).round() / 10.0),
        quant,
        size_mb: own,
        edits: fam.edits && have(Role::LlmVision),
        requires: fam.requires.to_vec(),
        optional: fam.optional.to_vec(),
        defaults: family::defaults_for(fam, &name),
        components,
        missing,
        suggestions,
    })
}

/// The denoiser in a model folder, when the folder holds one — it is the
/// model, whatever else (a larger text encoder, a VAE) sits beside it.
pub fn image_model_in_dir(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = files_in(dir).into_iter().filter(|p| is_image_model(p)).collect();
    // Several quantizations side by side: the largest is the best one.
    found.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    found.pop()
}

/// What the image engine is loaded with, from Settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct LoadOptions {
    /// Companions picked by hand, per role key.
    pub components: Overrides,
    /// "gpu" (default: everything on the GPU, spilling only what does not
    /// fit) or "cpu".
    pub device: String,
    /// Keep the weights in RAM and stream them to the GPU as needed.
    pub offload_to_cpu: bool,
    pub text_encoder_on_cpu: bool,
    pub vae_on_cpu: bool,
    pub flash_attn: bool,
    /// 0 = the physical core count.
    pub threads: i32,
    /// Per-GPU budget in GiB; 0 = whatever is free.
    pub max_vram_gb: f32,
    pub mmap: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            components: Overrides::new(),
            device: "gpu".into(),
            offload_to_cpu: false,
            text_encoder_on_cpu: false,
            vae_on_cpu: false,
            flash_attn: true,
            threads: 0,
            max_vram_gb: 0.0,
            mmap: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::gguf::tests::gguf_bytes;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("chaty-img-probe-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_st(path: &Path, names: &[&str]) {
        let mut hdr = serde_json::Map::new();
        for n in names {
            hdr.insert((*n).into(), serde_json::json!({"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}));
        }
        let h = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(&h);
        b.extend_from_slice(&[0, 0, 0, 0]);
        std::fs::write(path, b).unwrap();
    }

    /// The layout the store downloads into: the denoiser, its VAE and its
    /// text encoder in one folder. The encoder is the largest file there and
    /// a perfectly good chat model — it must still not be taken for the model.
    #[test]
    fn a_qwen_image_folder_resolves_to_its_denoiser_and_companions() {
        let dir = tmpdir("q21");
        let model = dir.join("qwen-image-2.1-Q4_K_M.gguf");
        std::fs::write(
            &model,
            gguf_bytes(&[], &[("model.diffusion_model.modulation.1.weight", &[4096, 16384]), ("model.diffusion_model.transformer_blocks.0.attn.to_q.weight", &[4096, 4096])]),
        )
        .unwrap();
        let mut llm = gguf_bytes(&[("general.architecture", "qwen3vl"), ("tokenizer.ggml.model", "gpt2")], &[]);
        llm.extend(std::iter::repeat_n(0u8, 4096)); // larger than the denoiser
        std::fs::write(dir.join("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"), llm).unwrap();
        std::fs::write(dir.join("mmproj-F16.gguf"), gguf_bytes(&[("general.architecture", "clip")], &[])).unwrap();
        write_st(&dir.join("qwen_image_2.1_vae_bf16.safetensors"), &["decoder.conv_in.weight", "encoder.conv_in.weight"]);

        assert!(is_image_model(&model));
        assert_eq!(kind_of(&dir.join("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf")), Kind::Llm);
        assert_eq!(image_model_in_dir(&dir).as_deref(), Some(model.as_path()));

        let probe = probe_image_model(&model, &Overrides::new(), &[]).unwrap();
        assert_eq!(probe.family, "qwen-image-2.1");
        assert!(probe.missing.is_empty(), "missing {:?}", probe.missing);
        let role = |r: Role| probe.components.iter().find(|c| c.role == r).map(|c| lower_name(Path::new(&c.path)));
        assert_eq!(role(Role::Vae).as_deref(), Some("qwen_image_2.1_vae_bf16.safetensors"));
        assert_eq!(role(Role::Llm).as_deref(), Some("qwen3-vl-8b-instruct-ud-q4_k_xl.gguf"));
        assert_eq!(role(Role::LlmVision).as_deref(), Some("mmproj-f16.gguf"));
        assert!(probe.edits);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing companions are reported with the downloads that supply them;
    /// an encoder in another model's folder is used when its name says it is
    /// the right one, and a hand-picked "none" is respected.
    #[test]
    fn missing_companions_come_with_downloads_and_shared_ones_are_reused() {
        let root = tmpdir("shared");
        let zdir = root.join("Z-Image-Turbo");
        std::fs::create_dir_all(&zdir).unwrap();
        let model = zdir.join("z_image_turbo-Q4_K.gguf");
        std::fs::write(&model, gguf_bytes(&[], &[("cap_embedder.1.weight", &[2560, 3840]), ("noise_refiner.0.x", &[1])])).unwrap();

        let probe = probe_image_model(&model, &Overrides::new(), std::slice::from_ref(&root)).unwrap();
        assert_eq!(probe.family, "z-image-turbo");
        assert_eq!(probe.missing, vec![Role::Vae, Role::Llm]);
        assert_eq!(probe.suggestions.len(), 2);
        assert_eq!(probe.defaults.steps, 8);

        // A chat model the user already has doubles as the encoder.
        let chat = root.join("Qwen3-4B-Instruct-2507");
        std::fs::create_dir_all(&chat).unwrap();
        std::fs::write(chat.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), gguf_bytes(&[("tokenizer.ggml.model", "gpt2")], &[])).unwrap();
        // An unrelated chat model is never taken for it.
        let other = root.join("Llama");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("Llama-3-8B.gguf"), gguf_bytes(&[("tokenizer.ggml.model", "llama")], &[])).unwrap();
        let probe = probe_image_model(&model, &Overrides::new(), std::slice::from_ref(&root)).unwrap();
        let llm = probe.components.iter().find(|c| c.role == Role::Llm).unwrap();
        assert_eq!(llm.source, "shared");
        assert!(llm.path.contains("Qwen3-4B"));
        assert_eq!(probe.missing, vec![Role::Vae]);

        let mut ov = Overrides::new();
        ov.insert("llm".into(), String::new());
        let probe = probe_image_model(&model, &ov, std::slice::from_ref(&root)).unwrap();
        assert!(probe.missing.contains(&Role::Llm));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_chat_model_is_not_an_image_model() {
        let dir = tmpdir("chat");
        let p = dir.join("Qwen3-4B-Q4_K_M.gguf");
        std::fs::write(&p, gguf_bytes(&[("general.architecture", "qwen3"), ("tokenizer.ggml.model", "gpt2")], &[])).unwrap();
        assert!(!is_image_model(&p));
        assert!(probe_image_model(&p, &Overrides::new(), &[]).is_none());
        assert!(image_model_in_dir(&dir).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
