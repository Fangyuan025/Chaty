//! Diffusion model families: how to recognise one from its tensor names, what
//! it needs beside the denoiser to draw anything, and the settings it was made
//! to be run with.
//!
//! A diffusion GGUF is the denoiser alone — Qwen-Image 2.1's is 7B parameters
//! of transformer that cannot read a prompt or produce a pixel by itself. The
//! text encoder and the VAE are separate files, different for every family and
//! not interchangeable between them, so this table is what turns "a GGUF the
//! chat engine refused" into "Qwen-Image 2.1, which still needs its Qwen3-VL
//! encoder".

use serde::Serialize;

/// A file a diffusion model needs besides its own weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    /// Autoencoder: latents to pixels and back.
    Vae,
    /// A language model used as the text encoder (Qwen-Image, Z-Image).
    Llm,
    /// That language model's vision tower — image editing reads the picture
    /// through it.
    LlmVision,
    ClipL,
    ClipG,
    T5xxl,
}

impl Role {
    pub const ALL: [Role; 6] = [Role::Vae, Role::Llm, Role::LlmVision, Role::ClipL, Role::ClipG, Role::T5xxl];

    pub fn key(self) -> &'static str {
        match self {
            Role::Vae => "vae",
            Role::Llm => "llm",
            Role::LlmVision => "llmVision",
            Role::ClipL => "clipL",
            Role::ClipG => "clipG",
            Role::T5xxl => "t5xxl",
        }
    }

    pub fn from_key(k: &str) -> Option<Role> {
        Role::ALL.into_iter().find(|r| r.key() == k)
    }
}

/// A companion file on Hugging Face that is known to fit a family.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Companion {
    pub role: Role,
    pub repo: &'static str,
    /// Repo-relative path.
    pub file: &'static str,
    /// Bytes, for the download estimate.
    pub size: u64,
}

/// The settings a family was made to be run with.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Defaults {
    pub steps: u32,
    pub cfg_scale: f32,
    /// Distilled guidance (FLUX-dev style); `None` for models without it.
    pub guidance: Option<f32>,
    /// Empty = the engine's default for this model.
    pub sampler: &'static str,
    pub scheduler: &'static str,
    /// 0 = the engine's default.
    pub flow_shift: f32,
    /// The side of a square at the model's native resolution.
    pub base_size: u32,
    /// Width and height must be multiples of this.
    pub align: u32,
    /// Whether a negative prompt does anything (it does not at CFG 1).
    pub negative_prompt: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Family {
    pub id: &'static str,
    pub name: &'static str,
    /// Loaded as one all-in-one checkpoint (`model_path`) rather than as a
    /// separate denoiser (`diffusion_model_path`).
    pub all_in_one: bool,
    /// Companions the model cannot run without.
    pub requires: &'static [Role],
    /// Companions that add a capability when present.
    pub optional: &'static [Role],
    /// What to download for the required ones.
    pub companions: &'static [Companion],
    /// Lower-case fragments a companion's file name carries, per role — how a
    /// matching encoder is recognised in another model's folder.
    #[serde(skip)]
    pub hints: &'static [(Role, &'static [&'static str])],
    pub defaults: Defaults,
    /// Edits a reference picture (needs `LlmVision`).
    pub edits: bool,
}

const MB: u64 = 1_000_000;

const FLUX_VAE: Companion = Companion {
    role: Role::Vae,
    repo: "Comfy-Org/z_image_turbo",
    file: "split_files/vae/ae.safetensors",
    size: 335 * MB,
};

pub static QWEN_IMAGE_21: Family = Family {
    id: "qwen-image-2.1",
    name: "Qwen-Image 2.1",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[Role::LlmVision],
    companions: &[
        Companion {
            role: Role::Vae,
            repo: "Comfy-Org/Qwen-Image-2.1",
            file: "vae/qwen_image_2.1_vae_bf16.safetensors",
            size: 680 * MB,
        },
        // unsloth's measured recommendation for this model: the Dynamic 2.0
        // 4-bit rung, not the uniform Q4_K_M.
        Companion {
            role: Role::Llm,
            repo: "unsloth/Qwen3-VL-8B-Instruct-GGUF",
            file: "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
            size: 5_150 * MB,
        },
    ],
    hints: &[
        (Role::Vae, &["qwen_image_2.1_vae", "qwen-image-2.1-vae", "qwen_image_2_1_vae"]),
        (Role::Llm, &["qwen3-vl-8b", "qwen3vl-8b", "qwen3vl_8b", "qwen3-vl"]),
        (Role::LlmVision, &["mmproj"]),
    ],
    defaults: Defaults {
        steps: 20,
        cfg_scale: 6.0,
        guidance: None,
        sampler: "euler",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 32,
        negative_prompt: true,
    },
    edits: true,
};

pub static QWEN_IMAGE: Family = Family {
    id: "qwen-image",
    name: "Qwen-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[Role::LlmVision],
    companions: &[
        Companion {
            role: Role::Vae,
            repo: "Comfy-Org/Qwen-Image_ComfyUI",
            file: "split_files/vae/qwen_image_vae.safetensors",
            size: 254 * MB,
        },
        Companion {
            role: Role::Llm,
            repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            file: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            size: 4_680 * MB,
        },
    ],
    hints: &[
        (Role::Vae, &["qwen_image_vae", "qwen-image-vae"]),
        (Role::Llm, &["qwen2.5-vl-7b", "qwen2.5-vl", "qwen_2.5_vl", "qwen25-vl"]),
        (Role::LlmVision, &["mmproj"]),
    ],
    defaults: Defaults {
        steps: 20,
        cfg_scale: 2.5,
        guidance: None,
        sampler: "euler",
        scheduler: "",
        flow_shift: 3.0,
        base_size: 1024,
        align: 16,
        negative_prompt: true,
    },
    edits: true,
};

const Z_IMAGE_COMPANIONS: &[Companion] = &[
    FLUX_VAE,
    Companion {
        role: Role::Llm,
        repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        size: 2_500 * MB,
    },
];
const Z_IMAGE_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, &["ae.safetensors", "ae.sft", "flux_vae", "flux-vae", "flux1_vae"]),
    (Role::Llm, &["qwen3-4b", "qwen_3_4b", "qwen3_4b"]),
];

pub static Z_IMAGE_TURBO: Family = Family {
    id: "z-image-turbo",
    name: "Z-Image Turbo",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: Z_IMAGE_COMPANIONS,
    hints: Z_IMAGE_HINTS,
    defaults: Defaults {
        steps: 8,
        cfg_scale: 1.0,
        guidance: None,
        sampler: "",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 16,
        negative_prompt: false,
    },
    edits: false,
};

pub static Z_IMAGE: Family = Family {
    id: "z-image",
    name: "Z-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: Z_IMAGE_COMPANIONS,
    hints: Z_IMAGE_HINTS,
    defaults: Defaults {
        steps: 28,
        cfg_scale: 5.0,
        guidance: None,
        sampler: "",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 16,
        negative_prompt: true,
    },
    edits: false,
};

const FLUX_COMPANIONS: &[Companion] = &[
    FLUX_VAE,
    Companion {
        role: Role::ClipL,
        repo: "comfyanonymous/flux_text_encoders",
        file: "clip_l.safetensors",
        size: 246 * MB,
    },
    Companion {
        role: Role::T5xxl,
        repo: "city96/t5-v1_1-xxl-encoder-gguf",
        file: "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
        size: 2_900 * MB,
    },
];
const FLUX_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, &["ae.safetensors", "ae.sft", "flux_vae", "flux-vae", "flux1_vae"]),
    (Role::ClipL, &["clip_l", "clip-l"]),
    (Role::T5xxl, &["t5xxl", "t5-v1_1-xxl", "t5_xxl", "t5-xxl"]),
];

pub static FLUX_DEV: Family = Family {
    id: "flux-dev",
    name: "FLUX.1 [dev]",
    all_in_one: false,
    requires: &[Role::Vae, Role::ClipL, Role::T5xxl],
    optional: &[],
    companions: FLUX_COMPANIONS,
    hints: FLUX_HINTS,
    defaults: Defaults {
        steps: 20,
        cfg_scale: 1.0,
        guidance: Some(3.5),
        sampler: "euler",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 16,
        negative_prompt: false,
    },
    edits: false,
};

pub static FLUX_SCHNELL: Family = Family {
    id: "flux-schnell",
    name: "FLUX.1 [schnell]",
    all_in_one: false,
    requires: &[Role::Vae, Role::ClipL, Role::T5xxl],
    optional: &[],
    companions: FLUX_COMPANIONS,
    hints: FLUX_HINTS,
    defaults: Defaults {
        steps: 4,
        cfg_scale: 1.0,
        guidance: None,
        sampler: "euler",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 16,
        negative_prompt: false,
    },
    edits: false,
};

pub static CHROMA: Family = Family {
    id: "chroma",
    name: "Chroma",
    all_in_one: false,
    requires: &[Role::Vae, Role::T5xxl],
    optional: &[],
    companions: &[FLUX_VAE, FLUX_COMPANIONS[2]],
    hints: FLUX_HINTS,
    defaults: Defaults {
        steps: 26,
        cfg_scale: 4.0,
        guidance: None,
        sampler: "euler",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 16,
        negative_prompt: true,
    },
    edits: false,
};

/// SD3 / 3.5: the encoders and the VAE live in gated repositories, so nothing
/// is fetched automatically — they are chosen in Settings.
pub static SD3: Family = Family {
    id: "sd3",
    name: "Stable Diffusion 3.x",
    all_in_one: false,
    requires: &[Role::Vae, Role::ClipL, Role::ClipG, Role::T5xxl],
    optional: &[],
    companions: &[],
    hints: &[
        (Role::Vae, &["sd3", "vae"]),
        (Role::ClipL, &["clip_l", "clip-l"]),
        (Role::ClipG, &["clip_g", "clip-g"]),
        (Role::T5xxl, &["t5xxl", "t5-v1_1-xxl", "t5_xxl", "t5-xxl"]),
    ],
    defaults: Defaults {
        steps: 28,
        cfg_scale: 4.5,
        guidance: None,
        sampler: "euler",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 64,
        negative_prompt: true,
    },
    edits: false,
};

pub static SDXL: Family = Family {
    id: "sdxl",
    name: "Stable Diffusion XL",
    all_in_one: true,
    requires: &[],
    optional: &[Role::Vae],
    companions: &[],
    hints: &[(Role::Vae, &["sdxl_vae", "sdxl-vae"])],
    defaults: Defaults {
        steps: 25,
        cfg_scale: 7.0,
        guidance: None,
        sampler: "euler_a",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 64,
        negative_prompt: true,
    },
    edits: false,
};

pub static SD1: Family = Family {
    id: "sd1",
    name: "Stable Diffusion 1.x / 2.x",
    all_in_one: true,
    requires: &[],
    optional: &[Role::Vae],
    companions: &[],
    hints: &[(Role::Vae, &["vae-ft-mse", "sd-vae", "sd_vae"])],
    defaults: Defaults {
        steps: 20,
        cfg_scale: 7.0,
        guidance: None,
        sampler: "euler_a",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 512,
        align: 64,
        negative_prompt: true,
    },
    edits: false,
};

/// A diffusion model none of the above describes. It still runs when the
/// engine knows it and its companions are chosen by hand in Settings.
pub static GENERIC: Family = Family {
    id: "generic",
    name: "Diffusion",
    all_in_one: false,
    requires: &[],
    optional: &[Role::Vae, Role::Llm, Role::ClipL, Role::ClipG, Role::T5xxl],
    companions: &[],
    hints: &[],
    defaults: Defaults {
        steps: 20,
        cfg_scale: 5.0,
        guidance: None,
        sampler: "",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align: 64,
        negative_prompt: true,
    },
    edits: false,
};

pub static ALL: [&Family; 11] = [
    &QWEN_IMAGE_21,
    &QWEN_IMAGE,
    &Z_IMAGE_TURBO,
    &Z_IMAGE,
    &FLUX_DEV,
    &FLUX_SCHNELL,
    &CHROMA,
    &SD3,
    &SDXL,
    &SD1,
    &GENERIC,
];

pub fn by_id(id: &str) -> Option<&'static Family> {
    ALL.iter().copied().find(|f| f.id == id)
}

/// `general.architecture` values the diffusion converters write (city96's
/// ComfyUI-GGUF tools, stable-diffusion.cpp's own). Any of these is an image
/// model whatever its tensors look like; plenty of diffusion GGUFs carry no
/// architecture at all and are recognised by their tensors instead.
pub const DIFFUSION_ARCHS: &[&str] = &[
    "flux", "sd1", "sd2", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan",
    "lumina2", "qwen_image", "chroma", "hunyuan", "mochi",
];

/// Tensor-name prefixes only a diffusion denoiser has (after the
/// `model.diffusion_model.` prefix all-in-one checkpoints add).
const DENOISER_PREFIXES: &[&str] = &[
    "double_blocks.",
    "single_blocks.",
    "joint_blocks.",
    "transformer_blocks.",
    "noise_refiner.",
    "context_refiner.",
    "input_blocks.",
    "time_text_embed.",
    "img_in.",
    "x_embedder.",
    "t_embedder.",
    "time_in.",
];

pub const SD_CKPT_PREFIX: &str = "model.diffusion_model.";

fn strip(name: &str) -> &str {
    name.strip_prefix(SD_CKPT_PREFIX).unwrap_or(name)
}

/// The tensor names of a file, looked up the way the rules below need.
pub struct Tensors<'a> {
    names: Vec<(&'a str, &'a [u64])>,
}

impl<'a> Tensors<'a> {
    pub fn new(names: Vec<(&'a str, &'a [u64])>) -> Self {
        Self { names }
    }
    fn has(&self, exact: &str) -> bool {
        self.names.iter().any(|(n, _)| strip(n) == exact)
    }
    fn any_prefix(&self, prefix: &str) -> bool {
        self.names.iter().any(|(n, _)| strip(n).starts_with(prefix))
    }
    fn raw_prefix(&self, prefix: &str) -> bool {
        self.names.iter().any(|(n, _)| n.starts_with(prefix))
    }
    fn dims(&self, exact: &str) -> Option<&'a [u64]> {
        self.names.iter().find(|(n, _)| strip(n) == exact).map(|(_, d)| *d)
    }
    /// The file is a diffusion denoiser (or a checkpoint containing one).
    pub fn is_denoiser(&self) -> bool {
        self.raw_prefix(SD_CKPT_PREFIX) || DENOISER_PREFIXES.iter().any(|p| self.names.iter().any(|(n, _)| n.starts_with(p)))
    }
    /// Only autoencoder weights: a standalone VAE file.
    pub fn is_vae(&self) -> bool {
        !self.names.is_empty()
            && self.names.iter().all(|(n, _)| {
                n.starts_with("decoder.")
                    || n.starts_with("encoder.")
                    || n.starts_with("first_stage_model.")
                    || n.starts_with("quant_conv.")
                    || n.starts_with("post_quant_conv.")
                    || n.starts_with("vae.")
                    || n.starts_with("conv_in.")
                    || n.starts_with("conv_out.")
                    || n.starts_with("latents_")
            })
    }
}

/// Which family a denoiser belongs to. `file_name` breaks the ties tensors
/// cannot: a turbo distillation has the same weights' shapes as its base.
pub fn detect(t: &Tensors, arch: Option<&str>, file_name: &str) -> &'static Family {
    let name = file_name.to_lowercase();

    // Qwen-Image 2.1: one fused MLP per block and a single top-level
    // modulation — the first version had separate image/text modulations.
    if t.any_prefix("transformer_blocks.")
        && (t.any_prefix("transformer_blocks.0.img_mlp.gate_up") || t.has("modulation.1.weight"))
    {
        return &QWEN_IMAGE_21;
    }
    if arch == Some("qwen_image") || t.any_prefix("transformer_blocks.0.img_mod.") {
        return &QWEN_IMAGE;
    }
    // Z-Image is a Lumina-2 style network (the converters even call it
    // "lumina2"); its caption embedder takes Qwen3-4B's 2560-wide states,
    // Lumina-Image 2.0's takes Gemma-2's 2304.
    if t.any_prefix("noise_refiner.") && t.any_prefix("cap_embedder.") {
        let qwen3_4b = t.dims("cap_embedder.1.weight").is_some_and(|d| d.first() == Some(&2560))
            || t.dims("cap_embedder.0.weight").is_some_and(|d| d.first() == Some(&2560));
        if qwen3_4b {
            return if name.contains("turbo") { &Z_IMAGE_TURBO } else { &Z_IMAGE };
        }
        return &GENERIC;
    }
    if t.any_prefix("double_blocks.") && t.any_prefix("single_blocks.") {
        if t.any_prefix("distilled_guidance_layer.") {
            return &CHROMA;
        }
        // FLUX.2 shares the block names and nothing else — another encoder,
        // another VAE.
        if t.any_prefix("double_stream_modulation_img.") {
            return &GENERIC;
        }
        return if t.any_prefix("guidance_in.") { &FLUX_DEV } else { &FLUX_SCHNELL };
    }
    if t.any_prefix("joint_blocks.") {
        return &SD3;
    }
    // All-in-one Stable Diffusion checkpoints carry their own VAE and CLIP.
    if t.raw_prefix("model.diffusion_model.input_blocks.") && t.raw_prefix("first_stage_model.") {
        if t.raw_prefix("conditioner.embedders.1.") || t.any_prefix("label_emb.") {
            return &SDXL;
        }
        return &SD1;
    }
    &GENERIC
}

/// The family a Hugging Face repo (or file) name points at — for the store,
/// which has to offer the companions before there is a file to read.
pub fn guess_by_name(name: &str) -> Option<&'static Family> {
    let n = name.to_lowercase().replace(['_', ' '], "-");
    if n.contains("qwen-image-2.1") || n.contains("qwen-image-2-1") {
        Some(&QWEN_IMAGE_21)
    } else if n.contains("qwen-image") {
        Some(&QWEN_IMAGE)
    } else if n.contains("z-image-turbo") {
        Some(&Z_IMAGE_TURBO)
    } else if n.contains("z-image") {
        Some(&Z_IMAGE)
    } else if n.contains("chroma") {
        Some(&CHROMA)
    } else if n.contains("flux") && n.contains("schnell") {
        Some(&FLUX_SCHNELL)
    } else if n.contains("flux.1") || n.contains("flux1") || n.contains("flux-1") {
        Some(&FLUX_DEV)
    } else {
        None
    }
}

/// Defaults adjusted for a distilled checkpoint of an otherwise normal family
/// (SDXL-Turbo, SD-Turbo, Lightning, LCM…): a handful of steps at CFG 1.
pub fn defaults_for(f: &Family, file_name: &str) -> Defaults {
    let mut d = f.defaults.clone();
    let name = file_name.to_lowercase();
    if matches!(f.id, "sd1" | "sdxl") && ["turbo", "lightning", "hyper", "lcm"].iter().any(|k| name.contains(k)) {
        d.steps = 4;
        d.cfg_scale = 1.0;
        d.negative_prompt = false;
        if name.contains("lcm") {
            d.sampler = "lcm";
        } else {
            d.sampler = "euler_a";
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(names: &[(&'static str, &'static [u64])]) -> Tensors<'static> {
        Tensors::new(names.to_vec())
    }

    /// The tensor tables of the real files (headers read from Hugging Face).
    #[test]
    fn each_family_is_recognised_from_its_tensors() {
        // unsloth/Qwen-Image-2.1-GGUF: no metadata at all, checkpoint prefix.
        let q21 = t(&[
            ("model.diffusion_model.img_in.weight", &[64, 4096]),
            ("model.diffusion_model.modulation.1.weight", &[4096, 16384]),
            ("model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight", &[4096, 24576]),
        ]);
        assert!(q21.is_denoiser());
        assert_eq!(detect(&q21, None, "qwen-image-2.1-Q4_K_M.gguf").id, "qwen-image-2.1");
        // leejet/Qwen-Image-2.1-GGUF: same, without the prefix.
        let q21b = t(&[("modulation.1.weight", &[4096, 16384]), ("transformer_blocks.0.attn.to_q.weight", &[4096, 4096])]);
        assert_eq!(detect(&q21b, None, "qwen_image_2.1-Q4_K.gguf").id, "qwen-image-2.1");

        let q1 = t(&[("transformer_blocks.0.img_mod.1.weight", &[3072, 18432]), ("txt_norm.weight", &[3584])]);
        assert_eq!(detect(&q1, Some("qwen_image"), "Qwen_Image-Q4_K_M.gguf").id, "qwen-image");

        let z = t(&[
            ("cap_embedder.1.weight", &[2560, 3840]),
            ("noise_refiner.0.attention.qkv.weight", &[3840, 11520]),
            ("x_embedder.weight", &[64, 3840]),
        ]);
        assert_eq!(detect(&z, None, "z_image_turbo-Q4_K.gguf").id, "z-image-turbo");
        assert_eq!(detect(&z, Some("lumina2"), "z-image-Q4_K_M.gguf").id, "z-image");
        let lumina = t(&[("cap_embedder.1.weight", &[2304, 2304]), ("noise_refiner.0.x", &[1])]);
        assert_eq!(detect(&lumina, Some("lumina2"), "lumina.gguf").id, "generic");

        let dev = t(&[("double_blocks.0.x", &[1]), ("single_blocks.0.x", &[1]), ("guidance_in.in_layer.weight", &[1])]);
        assert_eq!(detect(&dev, Some("flux"), "flux1-dev-Q4_K_S.gguf").id, "flux-dev");
        let schnell = t(&[("double_blocks.0.x", &[1]), ("single_blocks.0.x", &[1])]);
        assert_eq!(detect(&schnell, Some("flux"), "flux1-schnell-Q4_K_S.gguf").id, "flux-schnell");
        let chroma = t(&[("double_blocks.0.x", &[1]), ("single_blocks.0.x", &[1]), ("distilled_guidance_layer.in_proj.weight", &[1])]);
        assert_eq!(detect(&chroma, None, "chroma.gguf").id, "chroma");

        let sd3 = t(&[("joint_blocks.0.x", &[1])]);
        assert_eq!(detect(&sd3, Some("sd3"), "sd3.5_medium-Q4_K_M.gguf").id, "sd3");

        let sd15 = t(&[
            ("model.diffusion_model.input_blocks.0.0.weight", &[1]),
            ("first_stage_model.decoder.conv_in.weight", &[1]),
            ("cond_stage_model.transformer.text_model.x", &[1]),
        ]);
        assert_eq!(detect(&sd15, None, "sd-v1-5-Q4_0.gguf").id, "sd1");
        let sdxl = t(&[
            ("model.diffusion_model.input_blocks.0.0.weight", &[1]),
            ("first_stage_model.decoder.conv_in.weight", &[1]),
            ("conditioner.embedders.1.model.x", &[1]),
        ]);
        assert_eq!(detect(&sdxl, None, "sdxl.gguf").id, "sdxl");
    }

    #[test]
    fn a_language_model_or_a_vae_is_not_a_denoiser() {
        let llm = t(&[("token_embd.weight", &[1]), ("blk.0.attn_q.weight", &[1])]);
        assert!(!llm.is_denoiser());
        let vae = t(&[("decoder.conv_in.weight", &[1]), ("encoder.conv_in.weight", &[1])]);
        assert!(!vae.is_denoiser());
        assert!(vae.is_vae());
    }

    #[test]
    fn a_repo_name_points_at_its_family() {
        assert_eq!(guess_by_name("unsloth/Qwen-Image-2.1-GGUF").map(|f| f.id), Some("qwen-image-2.1"));
        assert_eq!(guess_by_name("QuantStack/Qwen-Image-GGUF").map(|f| f.id), Some("qwen-image"));
        assert_eq!(guess_by_name("leejet/Z-Image-Turbo-GGUF").map(|f| f.id), Some("z-image-turbo"));
        assert_eq!(guess_by_name("city96/FLUX.1-schnell-gguf").map(|f| f.id), Some("flux-schnell"));
        assert_eq!(guess_by_name("city96/FLUX.1-dev-gguf").map(|f| f.id), Some("flux-dev"));
        assert_eq!(guess_by_name("Qwen/Qwen3-4B-GGUF").map(|f| f.id), None);
    }

    #[test]
    fn a_distilled_checkpoint_runs_in_a_few_steps() {
        let d = defaults_for(&SDXL, "sdxl_turbo_1.0.gguf");
        assert_eq!((d.steps, d.cfg_scale), (4, 1.0));
        let d = defaults_for(&SDXL, "sd_xl_base_1.0.gguf");
        assert_eq!(d.steps, 25);
    }
}
