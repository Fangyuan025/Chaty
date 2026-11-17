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
    /// The text encoder's vocabulary as a `tokenizer.json`, for an encoder
    /// whose vocabulary the engine does not carry (MOSS-Image's Qwen3.5).
    Tokenizer,
}

impl Role {
    pub const ALL: [Role; 7] = [Role::Vae, Role::Llm, Role::LlmVision, Role::ClipL, Role::ClipG, Role::T5xxl, Role::Tokenizer];

    pub fn key(self) -> &'static str {
        match self {
            Role::Vae => "vae",
            Role::Llm => "llm",
            Role::LlmVision => "llmVision",
            Role::ClipL => "clipL",
            Role::ClipG => "clipG",
            Role::T5xxl => "t5xxl",
            Role::Tokenizer => "tokenizer",
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
    /// The schedule a distilled model was trained to draw in, for exactly
    /// `steps` steps (`steps + 1` values, ending at 0) — given to the engine
    /// while the steps and the scheduler are the model's own. Empty = the
    /// engine's scheduler.
    #[serde(skip)]
    pub sigmas: &'static [f32],
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

/// A family's defaults written out: the shape every flow-matching DiT
/// below shares — Euler, the engine's own scheduler and shift, 1024².
const fn flow(steps: u32, cfg_scale: f32, align: u32) -> Defaults {
    Defaults {
        steps,
        cfg_scale,
        guidance: None,
        sampler: "euler",
        scheduler: "",
        flow_shift: 0.0,
        base_size: 1024,
        align,
        // At CFG 1 the negative branch is never run.
        negative_prompt: cfg_scale > 1.0,
        sigmas: &[],
    }
}

const MB: u64 = 1_000_000;

const FLUX_VAE: Companion = Companion {
    role: Role::Vae,
    repo: "Comfy-Org/z_image_turbo",
    file: "split_files/vae/ae.safetensors",
    size: 335 * MB,
};

// Qwen3-VL 8B reads Qwen-Image 2.1's prompt; the plain "qwen3-vl" would take
// a 4B (FLUX.3's, Krea 2's) for it.
const QWEN3_VL_8B_HINTS: &[&str] = &["qwen3-vl-8b", "qwen3vl-8b", "qwen3vl_8b", "qwen3_vl_8b"];
const QWEN_IMAGE_21_COMPANIONS: &[Companion] = &[
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
];
const QWEN_IMAGE_21_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, &["qwen_image_2.1_vae", "qwen-image-2.1-vae", "qwen_image_2_1_vae"]),
    (Role::Llm, QWEN3_VL_8B_HINTS),
    (Role::LlmVision, &["mmproj"]),
];

pub static QWEN_IMAGE_21: Family = Family {
    id: "qwen-image-2.1",
    name: "Qwen-Image 2.1",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[Role::LlmVision],
    companions: QWEN_IMAGE_21_COMPANIONS,
    hints: QWEN_IMAGE_21_HINTS,
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
        sigmas: &[],
    },
    edits: true,
};

/// The 8-step distillation: its schedule is saved with the checkpoint and
/// not the scheduler's, so it is given (stable-diffusion.cpp's
/// `--sigmas qwen-image-2.1-turbo`).
pub static QWEN_IMAGE_21_TURBO: Family = Family {
    id: "qwen-image-2.1-turbo",
    name: "Qwen-Image 2.1 Turbo",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[Role::LlmVision],
    companions: QWEN_IMAGE_21_COMPANIONS,
    hints: QWEN_IMAGE_21_HINTS,
    defaults: Defaults {
        sigmas: &[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0],
        ..flow(8, 1.0, 32)
    },
    edits: true,
};

const QWEN_IMAGE_VAE: Companion = Companion {
    role: Role::Vae,
    repo: "Comfy-Org/Qwen-Image_ComfyUI",
    file: "split_files/vae/qwen_image_vae.safetensors",
    size: 254 * MB,
};
const QWEN25_VL_7B: Companion = Companion {
    role: Role::Llm,
    repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
    file: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    size: 4_680 * MB,
};
const QWEN25_VL_7B_HINTS: &[&str] = &["qwen2.5-vl-7b", "qwen2.5-vl", "qwen_2.5_vl", "qwen25-vl"];

pub static QWEN_IMAGE: Family = Family {
    id: "qwen-image",
    name: "Qwen-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[Role::LlmVision],
    companions: &[QWEN_IMAGE_VAE, QWEN25_VL_7B],
    hints: QWEN_IMAGE_HINTS,
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
        sigmas: &[],
    },
    edits: true,
};

const QWEN_IMAGE_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, &["qwen_image_vae", "qwen-image-vae"]),
    (Role::Llm, QWEN25_VL_7B_HINTS),
    (Role::LlmVision, &["mmproj"]),
];

/// Qwen-Image Edit (the first, 2509, 2511): Qwen-Image's network, the
/// picture read through the encoder's vision tower — without it there is
/// nothing to edit, so it comes with the encoder.
pub static QWEN_IMAGE_EDIT: Family = Family {
    id: "qwen-image-edit",
    name: "Qwen-Image Edit",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm, Role::LlmVision],
    optional: &[],
    companions: &[
        QWEN_IMAGE_VAE,
        QWEN25_VL_7B,
        Companion {
            role: Role::LlmVision,
            repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            file: "mmproj-F16.gguf",
            size: 1_354 * MB,
        },
    ],
    hints: QWEN_IMAGE_HINTS,
    defaults: Defaults { flow_shift: 3.0, ..flow(20, 4.0, 16) },
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
const FLUX_VAE_HINTS: &[&str] = &["ae.safetensors", "ae.sft", "flux_vae", "flux-vae", "flux1_vae"];
const Z_IMAGE_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, FLUX_VAE_HINTS),
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
        sigmas: &[],
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
        sigmas: &[],
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
    (Role::Vae, FLUX_VAE_HINTS),
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
        sigmas: &[],
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
        sigmas: &[],
    },
    edits: false,
};

// FLUX.2 [klein]: a Qwen3 reads the prompt (4B for the 4B model, 8B for the
// 9B), and the latents are FLUX.2's own — 32 channels, not FLUX.1's 16, so
// FLUX.1's `ae.safetensors` does not fit. The distilled checkpoints draw in
// four steps at CFG 1; the base ones are undistilled and want real CFG.
const FLUX2_VAE: Companion = Companion {
    role: Role::Vae,
    repo: "Comfy-Org/Flux-2-Klein-4B",
    file: "split_files/vae/flux2-vae.safetensors",
    size: 336 * MB,
};
const FLUX2_VAE_HINTS: &[&str] = &["flux2-vae", "flux2_vae", "flux-2-vae", "flux2-ae", "flux2_ae"];
// The plain Qwen3 releases, not the 2507 instruct/thinking ones: the encoder
// is read for its hidden states, and klein was trained on these.
const KLEIN_4B_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, FLUX2_VAE_HINTS),
    (Role::Llm, &["qwen3-4b-q", "qwen3-4b-ud-", "qwen3-4b-bf16", "qwen3-4b-f16", "qwen3-4b.", "qwen_3_4b"]),
];
const KLEIN_9B_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, FLUX2_VAE_HINTS),
    (Role::Llm, &["qwen3-8b-q", "qwen3-8b-ud-", "qwen3-8b-bf16", "qwen3-8b-f16", "qwen3-8b.", "qwen_3_8b"]),
];
const KLEIN_4B_COMPANIONS: &[Companion] = &[
    FLUX2_VAE,
    Companion {
        role: Role::Llm,
        repo: "unsloth/Qwen3-4B-GGUF",
        file: "Qwen3-4B-Q4_K_M.gguf",
        size: 2_497 * MB,
    },
];
const KLEIN_9B_COMPANIONS: &[Companion] = &[
    FLUX2_VAE,
    Companion {
        role: Role::Llm,
        repo: "unsloth/Qwen3-8B-GGUF",
        file: "Qwen3-8B-Q4_K_M.gguf",
        size: 5_027 * MB,
    },
];
const KLEIN_DISTILLED: Defaults = Defaults {
    steps: 4,
    cfg_scale: 1.0,
    guidance: None,
    sampler: "euler",
    scheduler: "",
    flow_shift: 0.0,
    base_size: 1024,
    align: 16,
    negative_prompt: false,
    sigmas: &[],
};
const KLEIN_BASE: Defaults = Defaults {
    steps: 20,
    cfg_scale: 4.0,
    guidance: None,
    sampler: "euler",
    scheduler: "",
    flow_shift: 0.0,
    base_size: 1024,
    align: 16,
    negative_prompt: true,
    sigmas: &[],
};

pub static FLUX2_KLEIN_4B: Family = Family {
    id: "flux2-klein-4b",
    name: "FLUX.2 [klein] 4B",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: KLEIN_4B_COMPANIONS,
    hints: KLEIN_4B_HINTS,
    defaults: KLEIN_DISTILLED,
    edits: false,
};

pub static FLUX2_KLEIN_BASE_4B: Family = Family {
    id: "flux2-klein-base-4b",
    name: "FLUX.2 [klein] Base 4B",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: KLEIN_4B_COMPANIONS,
    hints: KLEIN_4B_HINTS,
    defaults: KLEIN_BASE,
    edits: false,
};

pub static FLUX2_KLEIN_9B: Family = Family {
    id: "flux2-klein-9b",
    name: "FLUX.2 [klein] 9B",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: KLEIN_9B_COMPANIONS,
    hints: KLEIN_9B_HINTS,
    defaults: KLEIN_DISTILLED,
    edits: false,
};

pub static FLUX2_KLEIN_BASE_9B: Family = Family {
    id: "flux2-klein-base-9b",
    name: "FLUX.2 [klein] Base 9B",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: KLEIN_9B_COMPANIONS,
    hints: KLEIN_9B_HINTS,
    defaults: KLEIN_BASE,
    edits: false,
};

/// Which FLUX.2 a denoiser is, by the width of the text it takes in: three
/// of the encoder's hidden layers side by side — Qwen3-4B's 2560 for klein
/// 4B, Qwen3-8B's 4096 for 9B, Mistral Small's 5120 for [dev]. Base and
/// distilled share every shape; the name tells them apart.
fn flux2(t: &Tensors, name: &str) -> &'static Family {
    let base = name.contains("base");
    // In a GGUF the input width comes first, in a safetensors file last.
    let width = t.dims("txt_in.weight").map(|d| (d.first().copied(), d.last().copied()));
    let is = |w: u64| width.is_some_and(|(a, b)| a == Some(w) || b == Some(w));
    if is(7680) {
        if base { &FLUX2_KLEIN_BASE_4B } else { &FLUX2_KLEIN_4B }
    } else if is(12288) {
        if base { &FLUX2_KLEIN_BASE_9B } else { &FLUX2_KLEIN_9B }
    } else if is(15360) {
        &FLUX2_DEV
    } else {
        &GENERIC
    }
}

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
        sigmas: &[],
    },
    edits: false,
};

/// Chroma1-Radiance draws in pixels: T5 reads the prompt, and there is no
/// VAE at all.
pub static CHROMA_RADIANCE: Family = Family {
    id: "chroma-radiance",
    name: "Chroma1-Radiance",
    all_in_one: false,
    requires: &[Role::T5xxl],
    optional: &[],
    companions: &[FLUX_COMPANIONS[2]],
    hints: FLUX_HINTS,
    defaults: flow(20, 3.5, 16),
    edits: false,
};

// ---- Families stable-diffusion.cpp added since: what each is told apart by
// is the engine's own test (`ModelLoader::get_sd_version`), and its files and
// settings are the engine's recipes (examples/server/recipes).

/// Qwen3-VL 4B as a text encoder: FLUX.3 Image Klein, Krea 2, Iris, Mage-Flow.
const QWEN3_VL_4B: Companion = Companion {
    role: Role::Llm,
    repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
    file: "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    size: 2_497 * MB,
};
const QWEN3_VL_4B_HINTS: &[&str] = &["qwen3vl-4b", "qwen3-vl-4b", "qwen3vl_4b", "qwen3_vl_4b"];

const FLUX3_HINTS: &[(Role, &[&str])] = &[(Role::Vae, FLUX2_VAE_HINTS), (Role::Llm, QWEN3_VL_4B_HINTS)];

pub static FLUX3_KLEIN: Family = Family {
    id: "flux3-klein",
    name: "FLUX.3 Image Klein",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[FLUX2_VAE, QWEN3_VL_4B],
    hints: FLUX3_HINTS,
    defaults: flow(16, 1.0, 16),
    edits: false,
};

pub static FLUX3_KLEIN_BASE: Family = Family {
    id: "flux3-klein-base",
    name: "FLUX.3 Image Klein Base",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[FLUX2_VAE, QWEN3_VL_4B],
    hints: FLUX3_HINTS,
    defaults: flow(20, 4.0, 16),
    edits: false,
};

/// FLUX.2 [dev]: Mistral Small 3.2 (24B) reads the prompt.
pub static FLUX2_DEV: Family = Family {
    id: "flux2-dev",
    name: "FLUX.2 [dev]",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[
        FLUX2_VAE,
        Companion {
            role: Role::Llm,
            repo: "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
            file: "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
            size: 14_334 * MB,
        },
    ],
    hints: &[(Role::Vae, FLUX2_VAE_HINTS), (Role::Llm, &["mistral-small-3.2", "mistral-small-3_2", "mistral_small_3.2"])],
    defaults: Defaults { guidance: Some(4.0), ..flow(20, 1.0, 16) },
    edits: false,
};

/// MiniMax-H3-Image draws in six steps on a schedule of its own. Its prompt
/// is read by P-H3-Image's text encoder — Qwen3-VL 4B with the connector to
/// the H3 transformer, about 9 GB — which the engine takes in place of the
/// 32B MiniMax-H3 encoder (18 GB at 4 bits); either is recognised.
const H3_TEXT_ENCODER: Companion = Companion {
    role: Role::Llm,
    repo: "PrunaAI/P-H3-Image-Comfy",
    file: "text_encoders/p-h3-image_text_encoder_bf16.safetensors",
    size: 9_138 * MB,
};
const H3_LLM_HINTS: &[&str] = &["p-h3-image_text_encoder", "p_h3_image_text_encoder", "qwen3vl_32b_minimax_h3", "qwen3vl-32b-minimax-h3"];

pub static MINIMAX_H3_IMAGE: Family = Family {
    id: "minimax-h3-image",
    name: "MiniMax-H3-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[
        Companion {
            role: Role::Vae,
            repo: "Comfy-Org/MiniMax-H3-Image",
            file: "vae/minimax_h3_image_vae_fp16.safetensors",
            size: 1_169 * MB,
        },
        H3_TEXT_ENCODER,
    ],
    hints: &[(Role::Vae, &["minimax_h3_image_vae", "minimax-h3-image-vae"]), (Role::Llm, H3_LLM_HINTS)],
    defaults: Defaults {
        sigmas: &[1.0, 0.967_523_7, 0.887_548_5, 0.75, 0.532_774, 0.232_008_87, 0.0],
        ..flow(6, 1.0, 32)
    },
    edits: false,
};

/// P-H3-Image: the H3 transformer pruned, its own single-frame VAE.
pub static P_H3_IMAGE: Family = Family {
    id: "p-h3-image",
    name: "P-H3-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[
        Companion {
            role: Role::Vae,
            repo: "PrunaAI/P-H3-Image-Comfy",
            file: "vae/p-h3-image_vae_bf16.safetensors",
            size: 5_559 * MB,
        },
        H3_TEXT_ENCODER,
    ],
    hints: &[(Role::Vae, &["p-h3-image_vae", "p_h3_image_vae"]), (Role::Llm, H3_LLM_HINTS)],
    defaults: Defaults {
        sigmas: &[1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0],
        ..flow(8, 1.0, 32)
    },
    edits: false,
};

/// Krea 2: Qwen3-VL 4B, and the Qwen-Image VAE (Wan 2.1's) Krea ships.
const KREA2_HINTS: &[(Role, &[&str])] =
    &[(Role::Vae, &["qwen_image_vae", "qwen-image-vae", "wan_2.1_vae", "wan2.1_vae"]), (Role::Llm, QWEN3_VL_4B_HINTS)];

pub static KREA2_TURBO: Family = Family {
    id: "krea2-turbo",
    name: "Krea 2 Turbo",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[QWEN_IMAGE_VAE, QWEN3_VL_4B],
    hints: KREA2_HINTS,
    defaults: flow(8, 1.0, 16),
    edits: false,
};

pub static KREA2_RAW: Family = Family {
    id: "krea2-raw",
    name: "Krea 2 Raw",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[QWEN_IMAGE_VAE, QWEN3_VL_4B],
    hints: KREA2_HINTS,
    defaults: flow(20, 3.5, 16),
    edits: false,
};

/// MOSS-Image: Qwen3.5 9B reads the prompt, in a vocabulary the engine does
/// not carry — its `tokenizer.json` comes along.
const MOSS_COMPANIONS: &[Companion] = &[
    FLUX2_VAE,
    Companion {
        role: Role::Llm,
        repo: "unsloth/Qwen3.5-9B-GGUF",
        file: "Qwen3.5-9B-Q4_K_M.gguf",
        size: 5_681 * MB,
    },
    Companion {
        role: Role::Tokenizer,
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        file: "tokenizer/tokenizer.json",
        size: 20 * MB,
    },
];
const MOSS_HINTS: &[(Role, &[&str])] = &[
    (Role::Vae, FLUX2_VAE_HINTS),
    (Role::Llm, &["qwen3.5-9b", "qwen3_5-9b", "qwen3.5_9b", "qwen_3.5_9b"]),
    (Role::Tokenizer, &["tokenizer"]),
];
const MOSS_DEFAULTS: Defaults = Defaults { flow_shift: 5.0, ..flow(20, 3.0, 16) };

pub static MOSS_IMAGE: Family = Family {
    id: "moss-image",
    name: "MOSS-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm, Role::Tokenizer],
    optional: &[],
    companions: MOSS_COMPANIONS,
    hints: MOSS_HINTS,
    defaults: MOSS_DEFAULTS,
    edits: false,
};

pub static MOSS_IMAGE_BASE: Family = Family {
    id: "moss-image-base",
    name: "MOSS-Image Base",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm, Role::Tokenizer],
    optional: &[],
    companions: MOSS_COMPANIONS,
    hints: MOSS_HINTS,
    defaults: MOSS_DEFAULTS,
    edits: false,
};

/// ERNIE-Image: Ministral 3 (3B) reads the prompt, FLUX.2's VAE draws.
const ERNIE_COMPANIONS: &[Companion] = &[
    FLUX2_VAE,
    Companion {
        role: Role::Llm,
        repo: "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
        file: "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        size: 2_146 * MB,
    },
];
const ERNIE_HINTS: &[(Role, &[&str])] =
    &[(Role::Vae, FLUX2_VAE_HINTS), (Role::Llm, &["ministral-3-3b", "ministral_3_3b", "ministral-3b"])];

pub static ERNIE_IMAGE_TURBO: Family = Family {
    id: "ernie-image-turbo",
    name: "ERNIE-Image Turbo",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: ERNIE_COMPANIONS,
    hints: ERNIE_HINTS,
    defaults: flow(8, 1.0, 16),
    edits: false,
};

pub static ERNIE_IMAGE: Family = Family {
    id: "ernie-image",
    name: "ERNIE-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: ERNIE_COMPANIONS,
    hints: ERNIE_HINTS,
    defaults: flow(20, 4.0, 16),
    edits: false,
};

/// Anima: Qwen3 0.6B Base reads the prompt, the Qwen-Image VAE draws — both
/// as the model's own repository ships them.
pub static ANIMA: Family = Family {
    id: "anima",
    name: "Anima",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[
        QWEN_IMAGE_VAE,
        Companion {
            role: Role::Llm,
            repo: "circlestone-labs/Anima",
            file: "split_files/text_encoders/qwen_3_06b_base.safetensors",
            size: 1_192 * MB,
        },
    ],
    hints: &[(Role::Vae, &["qwen_image_vae", "qwen-image-vae"]), (Role::Llm, &["qwen_3_06b", "qwen3-0.6b", "qwen3_0.6b"])],
    defaults: flow(20, 4.0, 16),
    edits: false,
};

/// Ovis-Image: the Ovis 2.5 encoder, FLUX.1's VAE.
pub static OVIS_IMAGE: Family = Family {
    id: "ovis-image",
    name: "Ovis-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[
        FLUX_VAE,
        Companion {
            role: Role::Llm,
            repo: "Comfy-Org/Ovis-Image",
            file: "split_files/text_encoders/ovis_2.5.safetensors",
            size: 5_141 * MB,
        },
    ],
    hints: &[(Role::Vae, FLUX_VAE_HINTS), (Role::Llm, &["ovis_2.5", "ovis-2.5", "ovis2.5"])],
    defaults: flow(20, 5.0, 16),
    edits: false,
};

/// LongCat-Image: a FLUX-shaped transformer reading Qwen2.5-VL 7B.
pub static LONGCAT_IMAGE: Family = Family {
    id: "longcat-image",
    name: "LongCat-Image",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: &[FLUX_VAE, QWEN25_VL_7B],
    hints: &[(Role::Vae, FLUX_VAE_HINTS), (Role::Llm, QWEN25_VL_7B_HINTS)],
    defaults: Defaults { flow_shift: 3.0, ..flow(20, 4.0, 16) },
    edits: false,
};

/// Iris-3B draws in pixels: no VAE, Qwen3-VL 4B reads the prompt.
pub static IRIS: Family = Family {
    id: "iris",
    name: "Iris-3B",
    all_in_one: false,
    requires: &[Role::Llm],
    optional: &[],
    companions: &[QWEN3_VL_4B],
    hints: &[(Role::Llm, QWEN3_VL_4B_HINTS)],
    defaults: flow(20, 3.0, 16),
    edits: false,
};

/// Mage-Flow: Qwen-Image's block layout around a 128-channel latent of its
/// own VAE, Qwen3-VL 4B reading the prompt.
const MAGE_FLOW_HINTS: &[(Role, &[&str])] = &[(Role::Vae, &["mage_flow_vae", "mage-flow-vae", "mage_vae"]), (Role::Llm, QWEN3_VL_4B_HINTS)];
const MAGE_FLOW_COMPANIONS: &[Companion] = &[
    Companion {
        role: Role::Vae,
        repo: "Comfy-Org/Mage-Flow",
        file: "vae/mage_flow_vae_bf16.safetensors",
        size: 345 * MB,
    },
    QWEN3_VL_4B,
];

pub static MAGE_FLOW_TURBO: Family = Family {
    id: "mage-flow-turbo",
    name: "Mage-Flow Turbo",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: MAGE_FLOW_COMPANIONS,
    hints: MAGE_FLOW_HINTS,
    defaults: flow(4, 1.0, 16),
    edits: false,
};

pub static MAGE_FLOW: Family = Family {
    id: "mage-flow",
    name: "Mage-Flow",
    all_in_one: false,
    requires: &[Role::Vae, Role::Llm],
    optional: &[],
    companions: MAGE_FLOW_COMPANIONS,
    hints: MAGE_FLOW_HINTS,
    defaults: flow(20, 5.0, 16),
    edits: false,
};

/// SD3 / 3.5: the three encoders come from where the engine's own guide
/// takes them (CLIP-L and T5 are FLUX.1's, one file serving both). The VAE
/// is only in Stability's gated repository: the GGUFs that carry it inside
/// (second-state's, as stable-diffusion.cpp converts the original file) are
/// complete with the encoders; one without it needs it chosen in Settings.
pub static SD3: Family = Family {
    id: "sd3",
    name: "Stable Diffusion 3.x",
    all_in_one: false,
    requires: &[Role::Vae, Role::ClipL, Role::ClipG, Role::T5xxl],
    optional: &[],
    companions: &[
        FLUX_COMPANIONS[1],
        Companion {
            role: Role::ClipG,
            repo: "Comfy-Org/stable-diffusion-3.5-fp8",
            file: "text_encoders/clip_g.safetensors",
            size: 1_389 * MB,
        },
        FLUX_COMPANIONS[2],
    ],
    hints: &[
        (Role::Vae, &["sd3_vae", "sd3.5_vae", "sd3-vae"]),
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
        sigmas: &[],
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
        sigmas: &[],
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
        sigmas: &[],
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
        sigmas: &[],
    },
    edits: false,
};

pub static ALL: [&Family; 35] = [
    &QWEN_IMAGE_21,
    &QWEN_IMAGE_21_TURBO,
    &QWEN_IMAGE,
    &QWEN_IMAGE_EDIT,
    &Z_IMAGE_TURBO,
    &Z_IMAGE,
    &FLUX2_KLEIN_4B,
    &FLUX2_KLEIN_BASE_4B,
    &FLUX2_KLEIN_9B,
    &FLUX2_KLEIN_BASE_9B,
    &FLUX_DEV,
    &FLUX_SCHNELL,
    &CHROMA,
    &CHROMA_RADIANCE,
    &FLUX3_KLEIN,
    &FLUX3_KLEIN_BASE,
    &FLUX2_DEV,
    &MINIMAX_H3_IMAGE,
    &P_H3_IMAGE,
    &KREA2_TURBO,
    &KREA2_RAW,
    &MOSS_IMAGE,
    &MOSS_IMAGE_BASE,
    &ERNIE_IMAGE_TURBO,
    &ERNIE_IMAGE,
    &ANIMA,
    &OVIS_IMAGE,
    &LONGCAT_IMAGE,
    &IRIS,
    &MAGE_FLOW_TURBO,
    &MAGE_FLOW,
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

/// Tensors, by full name, only one family's denoiser has — the newer
/// networks name their blocks in ways a language model might too
/// (`layers.`, `blocks.`), so one tensor each is the tell, as in the engine.
const DENOISER_MARKERS: &[&str] = &[
    "emb_in.image.weight",
    "video_embedder.weight",
    "txtfusion.projector.weight",
    "text_fusion.projector.weight",
    "video_patch_proj.weight",
    "layers.0.adaLN_sa_ln.weight",
    "llm_adapter.blocks.0.cross_attn.q_proj.weight",
    "y_embedder.layer_pool.weight",
    "nerf_final_layer_conv.weight",
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
        self.raw_prefix(SD_CKPT_PREFIX)
            || DENOISER_PREFIXES.iter().any(|p| self.names.iter().any(|(n, _)| n.starts_with(p)))
            || DENOISER_MARKERS.iter().any(|m| self.has(m))
    }
    /// The checkpoint carries its own autoencoder (an all-in-one file).
    pub fn has_vae(&self) -> bool {
        self.raw_prefix("first_stage_model.")
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
    let name = file_name.to_lowercase().replace('_', "-");
    // Distilled and base checkpoints share every shape; the name tells.
    let turbo = name.contains("turbo");
    let base = name.contains("base");

    // The networks whose converters write another family's architecture
    // (Krea 2 as `qwen_image`, MiniMax-H3-Image as `hyvid`, Anima as
    // `cosmos`) or none: told apart by the tensor only each has, before
    // the older families' broader tests can claim them.
    if t.has("emb_in.image.weight") && t.has("txt_mode_blocks.0.q_proj.weight") {
        return if base { &FLUX3_KLEIN_BASE } else { &FLUX3_KLEIN };
    }
    if t.has("video_embedder.weight") && t.has("shared_block_modulation.weight") {
        return if base { &MOSS_IMAGE_BASE } else { &MOSS_IMAGE };
    }
    if t.has("txtfusion.projector.weight") || t.has("text_fusion.projector.weight") {
        return if turbo { &KREA2_TURBO } else { &KREA2_RAW };
    }
    if t.any_prefix("nerf_final_layer_conv.") {
        return &CHROMA_RADIANCE;
    }
    if t.has("y_embedder.layer_pool.weight") && t.has("modulation_cores.adaln_img.weight") {
        return &IRIS;
    }
    if t.has("llm_adapter.blocks.0.cross_attn.q_proj.weight") {
        return &ANIMA;
    }
    if t.has("double_blocks.0.img_mlp.gate_proj.weight") {
        return &OVIS_IMAGE;
    }
    if t.has("layers.0.adaLN_sa_ln.weight") {
        return if turbo { &ERNIE_IMAGE_TURBO } else { &ERNIE_IMAGE };
    }
    // The H3 transformer: with an audio head it is P-H3-Image (pruned from
    // the video model) or the video model itself, which the studio does not
    // draw with — only the name tells them apart before the VAE is read.
    if t.has("video_patch_proj.weight") && t.has("condition_proj.weight") {
        if !t.has("audio_patch_proj.weight") {
            return &MINIMAX_H3_IMAGE;
        }
        return if name.contains("p-h3") || name.contains("ph3") { &P_H3_IMAGE } else { &GENERIC };
    }
    // Mage-Flow keeps Qwen-Image's block names around a 128-channel latent.
    if t.any_prefix("transformer_blocks.0.img_mod.") && t.dims("img_in.weight").is_some_and(|d| d.contains(&128)) {
        return if turbo { &MAGE_FLOW_TURBO } else { &MAGE_FLOW };
    }

    // Qwen-Image 2.1: one fused MLP per block and a single top-level
    // modulation — the first version had separate image/text modulations.
    // (mflux's MLX saves split the MLP into proj and gate_layer and nest the
    // modulation's linear as `modulation.layers.1`.)
    if t.has("txt_in.text_norm.weight")
        || (t.any_prefix("transformer_blocks.")
            && (t.any_prefix("transformer_blocks.0.img_mlp.gate_up")
                || t.any_prefix("transformer_blocks.0.img_mlp.gate_layer")
                || t.has("modulation.1.weight")
                || t.has("modulation.layers.1.weight")))
    {
        return if turbo { &QWEN_IMAGE_21_TURBO } else { &QWEN_IMAGE_21 };
    }
    if arch == Some("qwen_image")
        || t.any_prefix("transformer_blocks.0.img_mod.")
        || t.any_prefix("transformer_blocks.0.img_mod_linear.")
    {
        return if name.contains("edit") { &QWEN_IMAGE_EDIT } else { &QWEN_IMAGE };
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
            return flux2(t, &name);
        }
        // LongCat-Image: FLUX's blocks reading Qwen2.5-VL's 3584-wide states.
        let ctx = t.dims("txt_in.weight").or_else(|| t.dims("context_embedder.weight"));
        if ctx.is_some_and(|d| d.first() == Some(&3584) || d.last() == Some(&3584)) {
            return &LONGCAT_IMAGE;
        }
        return if t.any_prefix("guidance_in.") { &FLUX_DEV } else { &FLUX_SCHNELL };
    }
    // FLUX.1 in diffusers' names, as mflux saves it: double-stream blocks
    // under `transformer_blocks`, single-stream ones after them. FLUX.2 and
    // FIBO share those names; FLUX.1 alone projects CLIP's pooled vector
    // (`time_text_embed.text_embedder`).
    if t.any_prefix("single_transformer_blocks.")
        && t.any_prefix("x_embedder.")
        && t.any_prefix("time_text_embed.text_embedder.")
    {
        return if t.any_prefix("time_text_embed.guidance_embedder.") { &FLUX_DEV } else { &FLUX_SCHNELL };
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
    // The repo's own name, past its owner: "anima" is a word inside others.
    let own = n.rsplit('/').next().unwrap_or(&n);
    let turbo = n.contains("turbo");
    let base = n.contains("base");
    if n.contains("qwen-image-2.1") || n.contains("qwen-image-2-1") {
        Some(if turbo { &QWEN_IMAGE_21_TURBO } else { &QWEN_IMAGE_21 })
    } else if n.contains("qwen-image-edit") {
        Some(&QWEN_IMAGE_EDIT)
    } else if n.contains("qwen-image") {
        Some(&QWEN_IMAGE)
    } else if n.contains("z-image-turbo") {
        Some(&Z_IMAGE_TURBO)
    } else if n.contains("z-image") {
        Some(&Z_IMAGE)
    } else if n.contains("chroma") && n.contains("radiance") {
        Some(&CHROMA_RADIANCE)
    } else if n.contains("chroma") {
        Some(&CHROMA)
    } else if (n.contains("flux.3") || n.contains("flux-3") || n.contains("flux3")) && n.contains("klein") {
        Some(if base { &FLUX3_KLEIN_BASE } else { &FLUX3_KLEIN })
    } else if (n.contains("flux.2") || n.contains("flux-2") || n.contains("flux2")) && n.contains("klein") {
        Some(match (n.contains("9b"), base) {
            (true, true) => &FLUX2_KLEIN_BASE_9B,
            (true, false) => &FLUX2_KLEIN_9B,
            (false, true) => &FLUX2_KLEIN_BASE_4B,
            (false, false) => &FLUX2_KLEIN_4B,
        })
    } else if (n.contains("flux.2") || n.contains("flux-2") || n.contains("flux2")) && n.contains("dev") {
        Some(&FLUX2_DEV)
    } else if n.contains("flux") && n.contains("schnell") {
        Some(&FLUX_SCHNELL)
    } else if n.contains("flux.1") || n.contains("flux1") || n.contains("flux-1") {
        Some(&FLUX_DEV)
    } else if n.contains("minimax-h3-image") {
        Some(&MINIMAX_H3_IMAGE)
    } else if n.contains("p-h3-image") {
        Some(&P_H3_IMAGE)
    } else if n.contains("krea-2") || n.contains("krea2") {
        Some(if turbo { &KREA2_TURBO } else { &KREA2_RAW })
    } else if n.contains("moss-image") {
        Some(if base { &MOSS_IMAGE_BASE } else { &MOSS_IMAGE })
    } else if n.contains("ernie-image") {
        Some(if turbo { &ERNIE_IMAGE_TURBO } else { &ERNIE_IMAGE })
    } else if (own.starts_with("anima") || own.contains("-anima")) && !own.contains("animagine") && !own.contains("animate") {
        Some(&ANIMA)
    } else if n.contains("ovis-image") {
        Some(&OVIS_IMAGE)
    } else if n.contains("longcat-image") {
        Some(&LONGCAT_IMAGE)
    } else if n.contains("iris-3b") {
        Some(&IRIS)
    } else if n.contains("mage-flow") {
        Some(if turbo { &MAGE_FLOW_TURBO } else { &MAGE_FLOW })
    } else if n.contains("stable-diffusion-3") || n.contains("sd3") {
        Some(&SD3)
    } else {
        None
    }
}

/// The family a Hugging Face repo is, by its name and then by the models
/// its card says it was made from (`base_model:` tags) — a fine-tune or a
/// quantization under a name of its own.
pub fn guess_by_repo(id: &str, tags: &[String]) -> Option<&'static Family> {
    guess_by_name(id).or_else(|| {
        let bases = tags.iter().filter_map(|t| t.strip_prefix("base_model:"));
        // `base_model:quantized:X` and the like name X after the relation.
        let mut named: Vec<&str> = bases.map(|b| b.rsplit(':').next().unwrap_or(b)).collect();
        // What it is a quantization of says it most directly.
        named.sort_by_key(|b| !tags.iter().any(|t| t == &format!("base_model:quantized:{b}")));
        named.into_iter().find_map(guess_by_name)
    })
}

/// Defaults adjusted for a distilled checkpoint of an otherwise normal family
/// (SDXL-Turbo, SD-Turbo, Lightning, LCM…): a handful of steps at CFG 1.
/// Sampling acceleration for a family: which of the engine's step caches
/// suits its denoiser, and how eagerly it may reuse a step ("balanced" keeps
/// the engine's own threshold, "fast" skips more). UNet models take UCache,
/// DiT models EasyCache; a model of unknown shape gets Spectrum, which serves
/// both. `None` = every step computed.
pub fn step_cache(family: &str, level: &str) -> Option<(&'static str, f32)> {
    let fast = match level {
        "balanced" => false,
        "fast" => true,
        _ => return None,
    };
    Some(match family {
        "sd1" | "sdxl" => ("ucache", if fast { 1.5 } else { 1.0 }),
        "generic" => ("spectrum", 0.0),
        _ => ("easycache", if fast { 0.35 } else { 0.2 }),
    })
}

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
    // Anima Turbo: its card's "CFG 1 and 8-12 steps".
    if f.id == "anima" && name.contains("turbo") {
        d.steps = 10;
        d.cfg_scale = 1.0;
        d.negative_prompt = false;
    }
    d
}

/// The schedule to give the engine for this round: the family's own, while
/// the round is drawn in its steps with the engine's default scheduler.
pub fn sigmas_for(family: &str, steps: u32, scheduler: &str) -> Vec<f32> {
    match by_id(family) {
        Some(f) if !f.defaults.sigmas.is_empty() && scheduler.is_empty() && f.defaults.sigmas.len() == steps as usize + 1 => {
            f.defaults.sigmas.to_vec()
        }
        _ => Vec::new(),
    }
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
        // mflux's MLX save: the MLP split in two, the modulation nested.
        let q21m = t(&[
            ("modulation.layers.1.weight", &[16384, 512]),
            ("transformer_blocks.0.img_mlp.gate_layer.weight", &[12288, 512]),
            ("transformer_blocks.0.attn.to_q.weight", &[4096, 512]),
        ]);
        assert_eq!(detect(&q21m, None, "qwen-image-2.1-mlx-4bit").id, "qwen-image-2.1");

        let q1 = t(&[("transformer_blocks.0.img_mod.1.weight", &[3072, 18432]), ("txt_norm.weight", &[3584])]);
        assert_eq!(detect(&q1, Some("qwen_image"), "Qwen_Image-Q4_K_M.gguf").id, "qwen-image");
        // mflux's MLX save of 2512 names the modulation `img_mod_linear`.
        let q1m = t(&[("transformer_blocks.0.img_mod_linear.weight", &[18432, 384]), ("txt_norm.weight", &[3584])]);
        assert_eq!(detect(&q1m, None, "Qwen-Image-2512-4bit").id, "qwen-image");

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
        // mflux's MLX saves use diffusers' names.
        let dev_m = t(&[("single_transformer_blocks.0.proj_mlp.weight", &[1]), ("x_embedder.weight", &[1]), ("context_embedder.weight", &[1]), ("time_text_embed.text_embedder.linear_1.weight", &[1]), ("time_text_embed.guidance_embedder.linear_1.weight", &[1])]);
        assert_eq!(detect(&dev_m, None, "FLUX.1-Krea-dev-mflux-4bit").id, "flux-dev");
        let schnell_m = t(&[("single_transformer_blocks.0.proj_mlp.weight", &[1]), ("x_embedder.weight", &[1]), ("context_embedder.weight", &[1]), ("time_text_embed.text_embedder.linear_1.weight", &[1])]);
        assert_eq!(detect(&schnell_m, None, "FLUX1-schnell-mlx-4bit").id, "flux-schnell");
        // FLUX.2 [klein] in the same names is not FLUX.1: no pooled-text
        // projection, a modulation shared by all blocks.
        let klein_m = t(&[("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", &[1]), ("x_embedder.weight", &[1]), ("context_embedder.weight", &[1]), ("time_guidance_embed.linear_1.weight", &[1]), ("double_stream_modulation_img.linear.weight", &[1])]);
        assert_eq!(detect(&klein_m, None, "FLUX.2-klein-4B-mflux-4bit").id, "generic");
        // FLUX.2 in its own names (stable-diffusion.cpp's and unsloth's
        // GGUFs, BFL's single-file safetensors): the width of `txt_in` says
        // which Qwen3 reads the prompt (issue #25).
        let klein = |w: u64| -> Vec<(&'static str, &'static [u64])> {
            let txt: &'static [u64] = Box::leak(vec![w, 3072].into_boxed_slice());
            vec![("double_blocks.0.img_attn.qkv.weight", &[3072, 9216]), ("single_blocks.0.linear1.weight", &[3072, 27648]), ("double_stream_modulation_img.lin.weight", &[3072, 18432]), ("txt_in.weight", txt)]
        };
        assert_eq!(detect(&t(&klein(7680)), Some("flux"), "flux-2-klein-4b-Q4_0.gguf").id, "flux2-klein-4b");
        assert_eq!(detect(&t(&klein(7680)), None, "flux-2-klein-base-4b-Q8_0.gguf").id, "flux2-klein-base-4b");
        assert_eq!(detect(&t(&klein(12288)), None, "flux-2-klein-9b-Q4_0.gguf").id, "flux2-klein-9b");
        assert_eq!(detect(&t(&klein(12288)), None, "flux-2-klein-base-9b.safetensors").id, "flux2-klein-base-9b");
        // FLUX.2 [dev] reads Mistral Small's 3 × 5120.
        assert_eq!(detect(&t(&klein(15360)), Some("flux"), "flux2-dev-Q4_0.gguf").id, "flux2-dev");
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

    /// The families stable-diffusion.cpp added, from the tensor tables of
    /// the GGUFs on the store's shelf (headers read from Hugging Face) — each
    /// written under another family's architecture, or none.
    #[test]
    fn the_newer_families_are_told_apart_as_the_engine_does() {
        let flux3 = t(&[("emb_in.image.weight", &[128, 3072]), ("txt_mode_blocks.0.q_proj.weight", &[3072, 3072]), ("single_blocks.0.attn_out.weight", &[3072, 3072])]);
        assert!(flux3.is_denoiser());
        assert_eq!(detect(&flux3, Some("flux"), "flux-3-image-klein-Q4_K_M.gguf").id, "flux3-klein");
        assert_eq!(detect(&flux3, None, "flux-3-image-klein-base-Q8_0.gguf").id, "flux3-klein-base");

        let moss = t(&[("video_embedder.weight", &[128, 3072]), ("shared_block_modulation.weight", &[3072, 18432]), ("layers.0.k_norm.weight", &[128])]);
        assert!(moss.is_denoiser());
        assert_eq!(detect(&moss, Some("moss_image"), "MOSS-Image-SFT-Q4_K_M.gguf").id, "moss-image");
        assert_eq!(detect(&moss, None, "MOSS-Image-Base-Q4_K_M.gguf").id, "moss-image-base");

        // Krea 2's converter writes `qwen_image`: not Qwen-Image.
        let krea = t(&[("txtfusion.projector.weight", &[12, 1]), ("blocks.0.attn.qknorm.knorm.scale", &[128])]);
        assert!(krea.is_denoiser());
        assert_eq!(detect(&krea, Some("qwen_image"), "krea2_turbo-Q4_K_M.gguf").id, "krea2-turbo");
        assert_eq!(detect(&krea, Some("qwen_image"), "Krea-2-Raw-Q8_0.gguf").id, "krea2-raw");

        // MiniMax-H3-Image as `hyvid`; P-H3 (an audio head) as `wan`.
        let h3 = t(&[("video_patch_proj.weight", &[96, 5376]), ("condition_proj.weight", &[5120, 5376])]);
        assert!(h3.is_denoiser());
        assert_eq!(detect(&h3, Some("hyvid"), "minimax_h3_image_pruned-Q4_K_M.gguf").id, "minimax-h3-image");
        let ph3 = t(&[
            ("model.diffusion_model.video_patch_proj.weight", &[96, 5376]),
            ("model.diffusion_model.condition_proj.weight", &[5120, 5376]),
            ("model.diffusion_model.audio_patch_proj.weight", &[32, 5376]),
        ]);
        assert_eq!(detect(&ph3, Some("wan"), "P-H3-Image-AD-Q4_K.gguf").id, "p-h3-image");
        // The video model itself is not drawn with here.
        assert_eq!(detect(&ph3, Some("wan"), "minimax_h3_fl2va-Q4_K_M.gguf").id, "generic");

        let ernie = t(&[("layers.0.adaLN_sa_ln.weight", &[4096])]);
        assert!(ernie.is_denoiser());
        assert_eq!(detect(&ernie, Some("wan"), "ernie-image-turbo-Q4_K_M.gguf").id, "ernie-image-turbo");
        assert_eq!(detect(&ernie, None, "ernie-image-UD-Q4_K_M.gguf").id, "ernie-image");

        let anima = t(&[("llm_adapter.blocks.0.cross_attn.q_proj.weight", &[1024, 1024])]);
        assert_eq!(detect(&anima, Some("cosmos"), "anima-turbo-v1.1-Q4_K_M.gguf").id, "anima");
        assert_eq!(defaults_for(&ANIMA, "anima-turbo-v1.1-q4_k_m.gguf").steps, 10);

        let ovis = t(&[("double_blocks.0.img_mlp.gate_proj.weight", &[1]), ("single_blocks.0.x", &[1])]);
        assert_eq!(detect(&ovis, Some("flux"), "ovis_image-Q4_0.gguf").id, "ovis-image");
        let longcat = t(&[("double_blocks.0.x", &[1]), ("single_blocks.0.x", &[1]), ("txt_in.weight", &[3584, 3072])]);
        assert_eq!(detect(&longcat, Some("flux"), "LongCat-Image-Q4_K_M.gguf").id, "longcat-image");
        let radiance = t(&[("double_blocks.0.x", &[1]), ("single_blocks.0.x", &[1]), ("distilled_guidance_layer.in_proj.weight", &[1]), ("nerf_final_layer_conv.weight", &[1])]);
        assert_eq!(detect(&radiance, None, "Chroma1-Radiance-v0.4-Q8_0.gguf").id, "chroma-radiance");
        let iris = t(&[("y_embedder.layer_pool.weight", &[1]), ("modulation_cores.adaln_img.weight", &[1])]);
        assert_eq!(detect(&iris, None, "Sakura-Iris-3B-Q4_K_M.gguf").id, "iris");

        // Qwen-Image's network: an edit, a turbo, Mage-Flow's 128-channel latent.
        let q1 = t(&[("transformer_blocks.0.img_mod.1.weight", &[3072, 18432]), ("img_in.weight", &[64, 3072])]);
        assert_eq!(detect(&q1, Some("qwen_image"), "qwen-image-edit-2511-Q4_K_M.gguf").id, "qwen-image-edit");
        assert_eq!(detect(&q1, Some("qwen_image"), "Qwen_Image-Q4_K_M.gguf").id, "qwen-image");
        let mage = t(&[("transformer_blocks.0.img_mod.1.weight", &[1]), ("img_in.weight", &[128, 2560])]);
        assert_eq!(detect(&mage, None, "mage-flow-turbo-Q8_0.gguf").id, "mage-flow-turbo");
        let q21 = t(&[("txt_in.text_norm.weight", &[4096]), ("transformer_blocks.0.attn.to_q.weight", &[1])]);
        assert_eq!(detect(&q21, None, "qwen-image-2.1-turbo-Q4_K_M.gguf").id, "qwen-image-2.1-turbo");
        assert_eq!(detect(&q21, None, "qwen-image-2.1-Q4_K_M.gguf").id, "qwen-image-2.1");

        // FLUX.2 [dev] reads Mistral Small's 3 × 5120, in either layout.
        let dev = |dims: &'static [u64]| t(&[("double_blocks.0.x", &[1]), ("single_blocks.47.linear1.weight", &[1]), ("double_stream_modulation_img.lin.weight", &[1]), ("txt_in.weight", dims)]);
        assert_eq!(detect(&dev(&[15360, 6144]), Some("flux"), "flux2-dev-Q4_K_M.gguf").id, "flux2-dev");
        assert_eq!(detect(&dev(&[6144, 15360]), None, "flux2-dev.safetensors").id, "flux2-dev");
    }

    #[test]
    fn a_checkpoint_that_carries_its_vae_says_so() {
        // second-state's SD 3.5: the denoiser and the VAE in one file.
        let sd35 = t(&[("first_stage_model.decoder.conv_in.bias", &[512]), ("model.diffusion_model.joint_blocks.0.x", &[1])]);
        assert!(sd35.has_vae());
        assert_eq!(detect(&sd35, None, "sd3.5_large-Q4_0.gguf").id, "sd3");
        // city96's: the denoiser alone.
        let city = t(&[("joint_blocks.0.x", &[1]), ("context_embedder.weight", &[4096, 2432])]);
        assert!(!city.has_vae());
    }

    #[test]
    fn a_distilled_model_draws_on_its_own_schedule_only_as_it_was_made() {
        assert_eq!(sigmas_for("minimax-h3-image", 6, "").len(), 7);
        assert_eq!(sigmas_for("qwen-image-2.1-turbo", 8, "").len(), 9);
        assert_eq!(sigmas_for("p-h3-image", 8, "").last(), Some(&0.0));
        // Other steps, or a scheduler chosen: the engine's.
        assert!(sigmas_for("minimax-h3-image", 12, "").is_empty());
        assert!(sigmas_for("minimax-h3-image", 6, "karras").is_empty());
        assert!(sigmas_for("flux3-klein", 16, "").is_empty());
        for f in ALL {
            let s = f.defaults.sigmas;
            if !s.is_empty() {
                assert_eq!(s.len(), f.defaults.steps as usize + 1, "{}", f.id);
                assert!(s.windows(2).all(|w| w[0] > w[1]) && s.last() == Some(&0.0), "{}", f.id);
            }
        }
    }

    /// Every family that needs a file says where it comes from, unless the
    /// file exists only behind a gated repository (SD3's VAE).
    #[test]
    fn every_required_file_has_a_download_but_sd3s_vae() {
        for f in ALL {
            for r in f.requires {
                let has = f.companions.iter().any(|c| c.role == *r);
                assert!(has || (f.id == "sd3" && *r == Role::Vae), "{} lacks a download for {:?}", f.id, r);
            }
            for c in f.companions {
                assert!(f.hints.iter().any(|(r, hs)| *r == c.role && !hs.is_empty()) || c.role == Role::Tokenizer || f.id == "sd3", "{}: {:?} cannot be found again", f.id, c.role);
            }
        }
        // Ids are unique: settings and history are keyed by them.
        let mut ids: Vec<&str> = ALL.iter().map(|f| f.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), ALL.len());
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
        assert_eq!(guess_by_name("leejet/FLUX.2-klein-4B-GGUF").map(|f| f.id), Some("flux2-klein-4b"));
        assert_eq!(guess_by_name("unsloth/FLUX.2-klein-4B-GGUF").map(|f| f.id), Some("flux2-klein-4b"));
        assert_eq!(guess_by_name("leejet/FLUX.2-klein-base-9B-GGUF").map(|f| f.id), Some("flux2-klein-base-9b"));
        assert_eq!(guess_by_name("flux-2-klein-9b-Q4_0.gguf").map(|f| f.id), Some("flux2-klein-9b"));
        assert_eq!(guess_by_name("city96/FLUX.2-dev-gguf").map(|f| f.id), Some("flux2-dev"));
        assert_eq!(guess_by_name("Qwen/Qwen3-4B-GGUF").map(|f| f.id), None);
        assert_eq!(guess_by_name("unsloth/Qwen-Image-2.1-Turbo-GGUF").map(|f| f.id), Some("qwen-image-2.1-turbo"));
        assert_eq!(guess_by_name("unsloth/Qwen-Image-Edit-2511-GGUF").map(|f| f.id), Some("qwen-image-edit"));
        assert_eq!(guess_by_name("unsloth/flux-3-image-klein-GGUF").map(|f| f.id), Some("flux3-klein"));
        assert_eq!(guess_by_name("unsloth/flux-3-image-klein-base-GGUF").map(|f| f.id), Some("flux3-klein-base"));
        assert_eq!(guess_by_name("Abiray/MiniMax-H3-Image-GGUF").map(|f| f.id), Some("minimax-h3-image"));
        assert_eq!(guess_by_name("AtomicChat/P-H3-Image-GGUF").map(|f| f.id), Some("p-h3-image"));
        assert_eq!(guess_by_name("vantagewithai/Krea-2-Turbo-GGUF").map(|f| f.id), Some("krea2-turbo"));
        assert_eq!(guess_by_name("molbal/krea2-gguf").map(|f| f.id), Some("krea2-raw"));
        assert_eq!(guess_by_name("QuantStack/FLUX.1-Krea-dev-GGUF").map(|f| f.id), Some("flux-dev"));
        assert_eq!(guess_by_name("unsloth/MOSS-Image-SFT-GGUF").map(|f| f.id), Some("moss-image"));
        assert_eq!(guess_by_name("unsloth/ERNIE-Image-Turbo-GGUF").map(|f| f.id), Some("ernie-image-turbo"));
        assert_eq!(guess_by_name("vanes430/Anima-Turbo-V1.1-GGUF").map(|f| f.id), Some("anima"));
        assert_eq!(guess_by_name("someone/Animagine-XL-4.0-GGUF").map(|f| f.id), None);
        assert_eq!(guess_by_name("silveroxides/Chroma1-Radiance-GGUF").map(|f| f.id), Some("chroma-radiance"));
        assert_eq!(guess_by_name("second-state/stable-diffusion-3.5-large-GGUF").map(|f| f.id), Some("sd3"));
    }

    #[test]
    fn a_fine_tune_is_known_by_the_model_it_was_made_from() {
        let tags = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // A name of its own, a base model on its card.
        let t1 = tags(&["gguf", "base_model:black-forest-labs/FLUX.2-klein-9B", "base_model:quantized:black-forest-labs/FLUX.2-klein-9B"]);
        assert_eq!(guess_by_repo("rectangleworm/PornMaster_Klein-9b", &t1).map(|f| f.id), Some("flux2-klein-9b"));
        let t2 = tags(&["base_model:Tongyi-MAI/Z-Image-Turbo", "base_model:finetune:Tongyi-MAI/Z-Image-Turbo"]);
        assert_eq!(guess_by_repo("ChenDY/NitroFusion2", &t2).map(|f| f.id), Some("z-image-turbo"));
        // The text encoder named as a base model says nothing.
        let t3 = tags(&["base_model:Qwen/Qwen3-VL-8B-Instruct"]);
        assert_eq!(guess_by_repo("someone/Unrelated-GGUF", &t3).map(|f| f.id), None);
        // The name wins over the card.
        assert_eq!(guess_by_repo("unsloth/Qwen-Image-2.1-Turbo-GGUF", &t3).map(|f| f.id), Some("qwen-image-2.1-turbo"));
    }

    #[test]
    fn a_distilled_checkpoint_runs_in_a_few_steps() {
        let d = defaults_for(&SDXL, "sdxl_turbo_1.0.gguf");
        assert_eq!((d.steps, d.cfg_scale), (4, 1.0));
        let d = defaults_for(&SDXL, "sd_xl_base_1.0.gguf");
        assert_eq!(d.steps, 25);
    }

    #[test]
    fn each_denoiser_gets_the_step_cache_it_supports() {
        assert_eq!(step_cache("qwen-image-2.1", "off"), None);
        assert_eq!(step_cache("qwen-image-2.1", ""), None);
        assert_eq!(step_cache("qwen-image-2.1", "balanced"), Some(("easycache", 0.2)));
        assert_eq!(step_cache("flux-dev", "fast"), Some(("easycache", 0.35)));
        assert_eq!(step_cache("sdxl", "balanced"), Some(("ucache", 1.0)));
        assert_eq!(step_cache("sd1", "fast"), Some(("ucache", 1.5)));
        assert_eq!(step_cache("generic", "fast"), Some(("spectrum", 0.0)));
    }
}
