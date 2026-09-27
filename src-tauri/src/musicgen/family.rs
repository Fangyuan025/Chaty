//! The music model families Chaty's engine runs, and what the studio needs to
//! know about each: how its GGUFs are packaged, where the composer's words go,
//! how long a piece can be, which edits of an earlier piece it can make, and
//! the stages a piece goes through (for the percentage).
//!
//! The parameters themselves — steps, guidance, sampling — are the engine's
//! request options, set by name; their labels and recommended values live with
//! the studio (src/lib/musicGen.ts).

use serde::{Deserialize, Serialize};

/// How a family's GGUFs are packaged (audio.cpp's own packages).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Layout {
    /// One self-contained GGUF: weights, configs and tokenizer embedded. The
    /// file is the model; a folder may hold several quantizations.
    Single,
    /// YuE2: the main GGUF (the quantization picked), a VAE GGUF and a
    /// `sidecars/` folder of configs and the tokenizer, side by side.
    Yue2,
    /// MiniMax Music 3: component GGUFs (language model, RVQ depth decoder,
    /// flow transformer in several quantizations; condition encoder, vocoder),
    /// `config/` and `tokenizer/`. The pick is a language model quantization.
    MiniMax,
}

/// The piece's length, as the family takes it.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Length {
    /// The request option.
    pub option: &'static str,
    /// Option units per second of music (1 = seconds; YuE2 counts tokens).
    pub per_second: f32,
    /// The recommended length, in seconds; 0 = the model decides ("auto").
    pub default_s: f32,
    pub min_s: f32,
    pub max_s: f32,
    /// What asks the model to decide by itself, when it can.
    pub auto_value: Option<&'static str>,
    /// A budget the model may stop short of, rather than an exact length.
    pub is_limit: bool,
}

/// An edit of an earlier piece of the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EditKind {
    /// YuE2: the piece's melody (the score the model wrote for it), new style
    /// or lyrics.
    Rearrange,
    /// YuE2: the first seconds of the piece kept as they are, the rest made
    /// again — or the piece made longer.
    Continue,
    /// ACE-Step: a stretch of the piece made again.
    Repaint,
    /// ACE-Step: the piece in a new style, its structure kept.
    Cover,
    /// Stable Audio: a piece started from this one, at some strength.
    Variation,
    /// Stable Audio: a stretch of the piece filled in again.
    Inpaint,
}

impl EditKind {
    /// The edit replaces a stretch of the piece (start and end).
    pub fn is_range(self) -> bool {
        matches!(self, EditKind::Repaint | EditKind::Inpaint)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Family {
    /// audio.cpp's family name (`audiocpp.model_spec.family`).
    pub id: &'static str,
    pub name: &'static str,
    /// Where its official GGUFs are published: the repo, and the folder in it
    /// (audio-cpp/audio.cpp-gguf keeps one folder per model).
    pub repo: &'static str,
    pub repo_dir: &'static str,
    pub layout: Layout,
    /// The composer's description goes into the request text…
    pub prompt_text: bool,
    /// …and/or into this request option.
    pub prompt_option: Option<&'static str>,
    /// The request option lyrics go into; None = the family sings nothing.
    pub lyrics: Option<&'static str>,
    pub lyrics_required: bool,
    /// Lyrics that ask for an instrumental piece; None = it cannot be asked.
    pub instrumental_lyrics: Option<&'static str>,
    pub length: Option<Length>,
    /// Plans a score before the music (YuE2's `cot`).
    pub planning: bool,
    /// Request options always sent (unless an edit sets them).
    pub fixed: &'static [(&'static str, &'static str)],
    pub edits: &'static [EditKind],
    /// The stages of a piece, in order, with their share of the work — the
    /// percentage is made from these.
    pub stages: &'static [(&'static str, f32)],
    /// Music tokens a second, where the tokens stage runs to an end the model
    /// chooses under a far larger budget (YuE2); 0 = the stage's total is what
    /// it will do.
    pub token_rate: f32,
    /// A structured prompt format (MiDashengLM's tags); plain words are
    /// wrapped in it.
    pub prompt_template: Option<&'static str>,
}

pub const YUE2: Family = Family {
    id: "yue2",
    name: "YuE2",
    repo: "audio-cpp/Yue2-3B-GGUF",
    repo_dir: "",
    layout: Layout::Yue2,
    prompt_text: false,
    prompt_option: Some("style"),
    lyrics: Some("lyrics"),
    lyrics_required: false,
    // An empty lyrics field is YuE2's instrumental.
    instrumental_lyrics: Some(""),
    length: Some(Length {
        option: "semantic_max_tokens",
        per_second: 25.0,
        default_s: 0.0,
        min_s: 10.0,
        max_s: 360.0,
        auto_value: None,
        is_limit: true,
    }),
    planning: true,
    fixed: &[],
    edits: &[EditKind::Rearrange, EditKind::Continue],
    stages: &[("score", 0.10), ("tokens", 0.45), ("render", 0.40), ("decode", 0.05)],
    token_rate: 25.0,
    prompt_template: None,
};

pub const ACE_STEP: Family = Family {
    id: "ace_step",
    name: "ACE-Step 1.5",
    repo: "audio-cpp/audio.cpp-gguf",
    repo_dir: "ACE-Step1.5-GGUF",
    layout: Layout::Single,
    prompt_text: true,
    prompt_option: None,
    lyrics: Some("lyrics"),
    lyrics_required: false,
    instrumental_lyrics: Some("[Instrumental]"),
    length: Some(Length {
        option: "duration_seconds",
        per_second: 1.0,
        default_s: 0.0,
        min_s: 10.0,
        max_s: 600.0,
        auto_value: Some("-1"),
        is_limit: false,
    }),
    planning: false,
    fixed: &[("route", "text2music")],
    edits: &[EditKind::Repaint, EditKind::Cover],
    stages: &[("tokens", 0.30), ("render", 0.65), ("decode", 0.05)],
    token_rate: 0.0,
    prompt_template: None,
};

pub const HEARTMULA: Family = Family {
    id: "heartmula",
    name: "HeartMuLa",
    repo: "audio-cpp/audio.cpp-gguf",
    repo_dir: "HeartMuLa-GGUF",
    layout: Layout::Single,
    // Its prompt is a description and a list of tags; the composer's words
    // are both.
    prompt_text: true,
    prompt_option: Some("tags"),
    lyrics: Some("lyrics"),
    lyrics_required: false,
    instrumental_lyrics: Some(""),
    length: Some(Length {
        option: "duration_sec",
        per_second: 1.0,
        default_s: 120.0,
        min_s: 10.0,
        max_s: 300.0,
        auto_value: None,
        is_limit: true,
    }),
    planning: false,
    fixed: &[],
    edits: &[],
    stages: &[("tokens", 0.80), ("render", 0.20)],
    token_rate: 0.0,
    prompt_template: None,
};

pub const STABLE_AUDIO: Family = Family {
    id: "stable_audio",
    name: "Stable Audio 3",
    repo: "audio-cpp/audio.cpp-gguf",
    repo_dir: "Stable-Audio-3-Small-Music-GGUF",
    layout: Layout::Single,
    prompt_text: true,
    prompt_option: None,
    lyrics: None,
    lyrics_required: false,
    instrumental_lyrics: None,
    length: Some(Length {
        option: "duration_seconds",
        per_second: 1.0,
        default_s: 120.0,
        min_s: 1.0,
        max_s: 380.0,
        auto_value: None,
        is_limit: false,
    }),
    planning: false,
    fixed: &[],
    edits: &[EditKind::Variation, EditKind::Inpaint],
    stages: &[("render", 0.92), ("decode", 0.08)],
    token_rate: 0.0,
    prompt_template: None,
};

pub const MINIMAX_MUSIC3: Family = Family {
    id: "minimax_music3",
    name: "MiniMax Music 3",
    repo: "audio-cpp/MiniMax-Music3-GGUF",
    repo_dir: "",
    layout: Layout::MiniMax,
    prompt_text: true,
    prompt_option: None,
    lyrics: Some("lyrics"),
    lyrics_required: true,
    instrumental_lyrics: None,
    length: Some(Length {
        option: "duration_sec",
        per_second: 1.0,
        default_s: 20.0,
        min_s: 5.0,
        max_s: 300.0,
        auto_value: None,
        is_limit: true,
    }),
    planning: false,
    fixed: &[],
    edits: &[],
    stages: &[("tokens", 0.65), ("render", 0.30), ("decode", 0.05)],
    token_rate: 0.0,
    prompt_template: None,
};

pub const MIDASHENGLM_GEN: Family = Family {
    id: "midashenglm_gen",
    name: "MiDashengLM-Gen",
    repo: "audio-cpp/audio.cpp-gguf",
    repo_dir: "MiDashengLM-Gen-GGUF",
    layout: Layout::Single,
    prompt_text: true,
    prompt_option: None,
    lyrics: None,
    lyrics_required: false,
    instrumental_lyrics: None,
    length: Some(Length {
        option: "duration_sec",
        per_second: 1.0,
        default_s: 20.0,
        min_s: 2.0,
        max_s: 60.0,
        auto_value: None,
        is_limit: true,
    }),
    planning: false,
    fixed: &[],
    edits: &[],
    stages: &[("tokens", 0.95), ("decode", 0.05)],
    token_rate: 0.0,
    // Its prompt is tagged by layer; words without tags describe the music.
    prompt_template: Some("<|caption|> {} <|music|> {}"),
};

/// Every family the engine is built with (audio-sidecar/CMakeLists.txt,
/// CHATY_AUDIO_MODELS).
pub const FAMILIES: &[&Family] = &[&YUE2, &ACE_STEP, &HEARTMULA, &STABLE_AUDIO, &MINIMAX_MUSIC3, &MIDASHENGLM_GEN];

pub fn by_id(id: &str) -> Option<&'static Family> {
    FAMILIES.iter().copied().find(|f| f.id == id)
}

/// The family a Hugging Face repo publishes, by its name — for the store,
/// before anything is downloaded.
pub fn guess_by_repo(repo: &str) -> Option<&'static Family> {
    let r = repo.to_lowercase().replace(['-', '_', '.'], "");
    let has = |k: &str| r.contains(k);
    if has("yue") {
        Some(&YUE2)
    } else if has("acestep") {
        Some(&ACE_STEP)
    } else if has("heartmula") {
        Some(&HEARTMULA)
    } else if has("stableaudio") {
        Some(&STABLE_AUDIO)
    } else if has("minimaxmusic") {
        Some(&MINIMAX_MUSIC3)
    } else if has("midashenglmgen") || has("midashenglm") {
        Some(&MIDASHENGLM_GEN)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_is_found_by_its_id_and_its_repo() {
        for f in FAMILIES {
            assert_eq!(by_id(f.id).map(|x| x.id), Some(f.id));
            let repo = if f.repo_dir.is_empty() { f.repo.to_string() } else { format!("{}/{}", f.repo, f.repo_dir) };
            let path = repo.rsplit('/').next().unwrap();
            assert_eq!(guess_by_repo(path).map(|x| x.id), Some(f.id), "{repo}");
            let total: f32 = f.stages.iter().map(|(_, w)| w).sum();
            assert!((total - 1.0).abs() < 1e-4, "{}: stage shares add to {total}", f.id);
        }
        assert!(guess_by_repo("Qwen/Qwen3-8B-GGUF").is_none());
    }

    #[test]
    fn range_edits_are_the_ones_that_replace_a_stretch() {
        assert!(EditKind::Repaint.is_range() && EditKind::Inpaint.is_range());
        assert!(!EditKind::Continue.is_range() && !EditKind::Cover.is_range());
    }
}
