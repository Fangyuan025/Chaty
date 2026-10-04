// The music studio's arithmetic: which parameters each model family takes and
// what it recommends for them, the request a piece is made with, and how far
// along one is. Kept free of React so every rule here is a plain function with
// a test.

import type {
  EngineOption,
  MusicEdit,
  MusicFamilySpec,
  MusicLoadOptions,
  MusicRecord,
  MusicRequest,
} from "./ipc";

/** A label in the app's languages (English fills in for a missing one). */
export interface L {
  zh: string;
  en: string;
  pt?: string;
}

export function tl(l: L, lang: string): string {
  return lang === "zh" ? l.zh : lang === "pt" ? (l.pt ?? l.en) : l.en;
}

/** One of a family's request options, as the studio offers it. */
export interface ParamDef {
  /** The engine's option name. */
  key: string;
  label: L;
  tip?: L;
  kind: "int" | "float" | "enum" | "bool" | "text";
  /** The recommended value — what the engine uses when nothing is sent. */
  def: string;
  /** How the recommendation reads when it depends on other settings. */
  defLabel?: L;
  min?: number;
  max?: number;
  step?: number;
  choices?: { value: string; label?: L }[];
  /** Where it is shown: the composer's quick panel and Settings, or only
   *  Settings (in a section of its own). */
  group: "main" | "sampling" | "score" | "more";
}

// ---- The families' parameters (audio.cpp's docs; recommended = its own
// defaults, which are what the model's authors publish) ----

const STEPS = (def: number, max = 100): ParamDef => ({
  key: "num_inference_steps",
  label: { zh: "步数", en: "Steps", pt: "Passos" },
  tip: {
    zh: "扩散/流匹配的步数。越多越细致也越慢",
    en: "Diffusion / flow steps. More is finer and slower",
    pt: "Passos de difusão. Mais é mais fino e mais lento",
  },
  kind: "int",
  def: String(def),
  min: 1,
  max,
  step: 1,
  group: "main",
});

const GUIDANCE = (def: number, max = 20, defLabel?: L): ParamDef => ({
  key: "guidance_scale",
  label: { zh: "引导强度", en: "Guidance", pt: "Orientação" },
  tip: {
    zh: "越高越贴近描述,过高会生硬",
    en: "Higher follows the description more closely; too high sounds forced",
    pt: "Mais alto segue a descrição mais de perto",
  },
  kind: "float",
  def: String(def),
  defLabel,
  min: 0,
  max,
  step: 0.05,
  group: "main",
});

/** A sampler's settings, under a prefix ("semantic_", "abc_", "lm_", ""). */
function sampling(prefix: string, group: ParamDef["group"], d: { t: number; p?: number; k?: number; rp?: number; w?: number }, kMax = 500): ParamDef[] {
  const out: ParamDef[] = [
    {
      key: `${prefix}temperature`,
      label: { zh: "温度", en: "Temperature", pt: "Temperatura" },
      tip: { zh: "越高越多变,越低越保守", en: "Higher varies more, lower plays it safe", pt: "Mais alto varia mais" },
      kind: "float",
      def: String(d.t),
      min: 0,
      max: 2,
      step: 0.05,
      group,
    },
  ];
  if (d.p != null)
    out.push({ key: `${prefix}top_p`, label: { zh: "Top-P", en: "Top-P" }, kind: "float", def: String(d.p), min: 0.05, max: 1, step: 0.01, group });
  if (d.k != null)
    out.push({ key: `${prefix}top_k`, label: { zh: "Top-K", en: "Top-K" }, kind: "int", def: String(d.k), min: d.k === 0 ? 0 : 1, max: kMax, step: 1, group });
  if (d.rp != null)
    out.push({
      key: `${prefix}repetition_penalty`,
      label: { zh: "重复惩罚", en: "Repetition penalty", pt: "Penalidade de repetição" },
      kind: "float",
      def: String(d.rp),
      min: 0.8,
      max: 2,
      step: 0.005,
      group,
    });
  if (d.w != null)
    out.push({
      key: `${prefix}penalty_window`,
      label: { zh: "惩罚窗口", en: "Penalty window", pt: "Janela de penalidade" },
      kind: "int",
      def: String(d.w),
      min: 1,
      max: 1000,
      step: 1,
      group,
    });
  return out;
}

const NEGATIVE = (def = ""): ParamDef => ({
  key: "negative_prompt",
  label: { zh: "反向描述", en: "Negative prompt", pt: "Prompt negativo" },
  tip: { zh: "不想要的元素", en: "What to keep out", pt: "O que evitar" },
  kind: "text",
  def,
  group: "more",
});

export const FAMILY_PARAMS: Record<string, ParamDef[]> = {
  yue2: [
    STEPS(8, 64),
    GUIDANCE(1.0, 20, { zh: "1.0(直接生成时 1.01)", en: "1.0 (1.01 without planning)", pt: "1.0 (1.01 sem planejamento)" }),
    ...sampling("semantic_", "sampling", { t: 1.0, p: 0.95, k: 100, rp: 1.2, w: 50 }),
    {
      key: "semantic_min_tokens",
      label: { zh: "最短(音乐令牌)", en: "Shortest (music tokens)", pt: "Mais curto (tokens)" },
      tip: { zh: "每秒 25 个令牌;不到这个长度不会结束", en: "25 tokens a second; the piece does not end before this", pt: "25 tokens por segundo" },
      kind: "int",
      def: "200",
      min: 0,
      max: 9000,
      step: 25,
      group: "sampling",
    },
    ...sampling("abc_", "score", { t: 0.7, p: 0.9, k: 30, rp: 1.005, w: 100 }),
    {
      key: "abc_max_tokens",
      label: { zh: "乐谱最长(令牌)", en: "Longest score (tokens)", pt: "Partitura máxima (tokens)" },
      kind: "int",
      def: "4096",
      min: 64,
      max: 8192,
      step: 64,
      group: "score",
    },
    {
      key: "abc_min_tokens",
      label: { zh: "乐谱最短(令牌)", en: "Shortest score (tokens)", pt: "Partitura mínima (tokens)" },
      kind: "int",
      def: "32",
      min: 0,
      max: 4096,
      step: 8,
      group: "score",
    },
  ],
  ace_step: [
    STEPS(8, 100),
    GUIDANCE(1.0),
    {
      key: "language",
      label: { zh: "演唱语言", en: "Vocal language", pt: "Idioma vocal" },
      kind: "enum",
      def: "en",
      choices: ["en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "yue"].map((v) => ({ value: v })),
      group: "main",
    },
    { key: "bpm", label: { zh: "BPM(节拍速度)", en: "BPM" }, tip: { zh: "空 = 由模型规划", en: "Empty = the planner decides" }, kind: "int", def: "", min: 30, max: 300, step: 1, group: "main" },
    { key: "keyscale", label: { zh: "调性", en: "Key", pt: "Tonalidade" }, tip: { zh: "如 C major;空 = 由模型规划", en: "e.g. C major; empty = the planner decides" }, kind: "text", def: "", group: "main" },
    { key: "timesignature", label: { zh: "拍号", en: "Time signature", pt: "Fórmula de compasso" }, tip: { zh: "如 4;空 = 由模型规划", en: "e.g. 4; empty = the planner decides" }, kind: "text", def: "", group: "main" },
    {
      key: "sampler_mode",
      label: { zh: "采样器", en: "Sampler", pt: "Amostrador" },
      kind: "enum",
      def: "euler",
      choices: [{ value: "euler" }, { value: "heun" }],
      group: "sampling",
    },
    ...sampling("lm_", "sampling", { t: 0.85, p: 0.9, k: 0, rp: 1.0 }),
    {
      key: "lm_cfg_scale",
      label: { zh: "规划器引导", en: "Planner guidance", pt: "Orientação do planejador" },
      kind: "float",
      def: "2.0",
      min: 0,
      max: 10,
      step: 0.1,
      group: "sampling",
    },
    {
      key: "audio_cover_strength",
      label: { zh: "翻唱强度", en: "Cover strength", pt: "Força do cover" },
      tip: { zh: "翻唱时保留原曲结构的程度", en: "How much of the source's structure a cover keeps" },
      kind: "float",
      def: "1.0",
      min: 0,
      max: 1,
      step: 0.05,
      group: "more",
    },
    {
      key: "repaint_mode",
      label: { zh: "重绘方式", en: "Repaint mode", pt: "Modo de repintura" },
      kind: "enum",
      def: "balanced",
      choices: [{ value: "balanced" }, { value: "conservative" }, { value: "aggressive" }],
      group: "more",
    },
    {
      key: "repaint_strength",
      label: { zh: "重绘强度", en: "Repaint strength", pt: "Força da repintura" },
      kind: "float",
      def: "0.5",
      min: 0,
      max: 1,
      step: 0.05,
      group: "more",
    },
    NEGATIVE("NO USER INPUT"),
  ],
  heartmula: [
    STEPS(10, 50),
    GUIDANCE(1.5, 10),
    ...sampling("", "sampling", { t: 1.0, k: 50 }),
    {
      key: "codec_guidance_scale",
      label: { zh: "编解码器引导", en: "Codec guidance", pt: "Orientação do codec" },
      kind: "float",
      def: "1.25",
      min: 0,
      max: 5,
      step: 0.05,
      group: "sampling",
    },
    {
      key: "infinite_mode",
      label: { zh: "超长模式", en: "Long mode", pt: "Modo longo" },
      tip: { zh: "按歌词分段生成更长的歌", en: "Longer songs, made section by section of the lyrics" },
      kind: "bool",
      def: "false",
      group: "more",
    },
  ],
  stable_audio: [
    STEPS(8, 100),
    GUIDANCE(1.0),
    {
      key: "sampler",
      label: { zh: "采样器", en: "Sampler", pt: "Amostrador" },
      kind: "enum",
      def: "pingpong",
      choices: [{ value: "pingpong" }, { value: "euler" }, { value: "dpmpp-2m" }, { value: "dpmpp-3m-sde" }],
      group: "main",
    },
    {
      key: "apg_scale",
      label: { zh: "APG 引导", en: "APG scale" },
      kind: "float",
      def: "1.0",
      min: 0,
      max: 5,
      step: 0.05,
      group: "sampling",
    },
    NEGATIVE(),
  ],
  minimax_music3: [
    STEPS(30, 100),
    GUIDANCE(1.7, 10),
    {
      key: "ar_guidance_scale",
      label: { zh: "作曲引导", en: "Composition guidance", pt: "Orientação da composição" },
      tip: { zh: "作曲(自回归)阶段的引导强度;0 = 关闭", en: "Guidance of the composing (autoregressive) stage; 0 = off" },
      kind: "float",
      def: "1.5",
      min: 0,
      max: 10,
      step: 0.05,
      group: "main",
    },
    ...sampling("", "sampling", { t: 1.0, k: 50 }).filter((p) => p.key !== "temperature"),
  ],
  midashenglm_gen: [
    GUIDANCE(2.0, 10),
    {
      key: "stop_threshold",
      label: { zh: "停止阈值", en: "Stop threshold", pt: "Limiar de parada" },
      kind: "float",
      def: "0.5",
      min: 0,
      max: 1,
      step: 0.05,
      group: "sampling",
    },
    {
      key: "min_stop_step",
      label: { zh: "最少步数", en: "Minimum steps", pt: "Passos mínimos" },
      kind: "int",
      def: "5",
      min: 0,
      max: 100,
      step: 1,
      group: "sampling",
    },
  ],
};

/** Options the composer sets itself, or the studio manages — never offered
 *  as a setting of their own. */
const MANAGED = new Set([
  "style",
  "lyrics",
  "tags",
  "text",
  "seed",
  "cot",
  "route",
  "abc",
  "abc_file",
  "export_semantic",
  "stop_after",
  "semantic_prefix",
  "semantic_prefix_file",
  "semantic_max_tokens",
  "nar_noise_file",
  "duration_seconds",
  "duration_sec",
  "audio_input_kind",
  "init_noise_level",
  "inpaint_mask_start_seconds",
  "inpaint_mask_end_seconds",
  "repainting_start",
  "repainting_end",
  "batch_size",
  "audio_codes",
  "source_audio",
  "track_name",
  "complete_track_classes",
]);

/** An option the engine lists, as a setting: its own name, its own
 *  description, the default it reports. */
export function fromEngine(o: EngineOption, group: ParamDef["group"] = "more"): ParamDef {
  const choices = o.kind.includes("|") ? o.kind.split("|").map((v) => ({ value: v })) : undefined;
  const kind: ParamDef["kind"] = choices
    ? "enum"
    : o.kind === "int"
      ? "int"
      : o.kind === "float"
        ? "float"
        : o.kind === "bool"
          ? "bool"
          : "text";
  const num = (v: string) => (v !== "" && Number.isFinite(Number(v)) ? Number(v) : undefined);
  return {
    key: o.name,
    label: { zh: o.name, en: o.name },
    tip: o.description ? { zh: o.description, en: o.description } : undefined,
    kind,
    def: o.default,
    min: num(o.min),
    max: num(o.max),
    step: kind === "int" ? 1 : kind === "float" ? 0.01 : undefined,
    choices,
    group,
  };
}

/** Every request option the studio offers for a family: its own table, then
 *  whatever else the engine lists. */
export function familyParams(family: string, engine: EngineOption[] = []): ParamDef[] {
  const curated = FAMILY_PARAMS[family] ?? [];
  const known = new Set(curated.map((p) => p.key));
  const extra = engine.filter((o) => !known.has(o.name) && !MANAGED.has(o.name)).map((o) => fromEngine(o));
  return [...curated, ...extra];
}

/** The engine's session options worth a setting: everything but the files a
 *  package pick already names. */
export function sessionParams(engine: EngineOption[] = []): ParamDef[] {
  return engine.filter((o) => !/_gguf$/.test(o.name)).map((o) => fromEngine(o, "more"));
}

// ---- Settings ----

/** What Settings stores for the music studio. Every parameter is per family
 *  and absent = recommended. */
export interface MusicSettings {
  /** Family → request option → value. */
  musParams: Record<string, Record<string, string>>;
  /** Family → length in seconds (0 = recommended / the model decides). */
  musLength: Record<string, number>;
  /** YuE2's planning: "full" (score first, recommended), "melody", "off". */
  musPlanning: "full" | "melody" | "off";
  musSeedLock: boolean;
  musSeed: number;
  musOutputDir: string;
  musDevice: "gpu" | "cpu";
  musThreads: number;
  /** Family → engine session option → value. */
  musSession: Record<string, Record<string, string>>;
  /** Play a piece as soon as it is made. */
  musAutoPlay: boolean;
}

export const MUSIC_SETTINGS_DEFAULTS: MusicSettings = {
  musParams: {},
  musLength: {},
  musPlanning: "full",
  musSeedLock: false,
  musSeed: 42,
  musOutputDir: "",
  // GPU, like every engine here: it falls back to the CPU by itself when the
  // GPU cannot take it.
  musDevice: "gpu",
  musThreads: 0,
  musSession: {},
  musAutoPlay: true,
};

/** The value a parameter runs with: the setting, else the recommendation. */
export function paramValue(s: MusicSettings, family: string, p: ParamDef): string {
  const v = s.musParams[family]?.[p.key];
  return v != null && v !== "" ? v : p.def;
}

/** Whether a parameter is moved off its recommendation. */
export function isCustom(s: MusicSettings, family: string, key: string): boolean {
  const v = s.musParams[family]?.[key];
  return v != null && v !== "";
}

/** Set (or, with null, reset to the recommendation) one parameter. */
export function withParam(s: MusicSettings, family: string, key: string, value: string | null): Pick<MusicSettings, "musParams"> {
  const cur = { ...(s.musParams[family] ?? {}) };
  if (value == null) delete cur[key];
  else cur[key] = value;
  return { musParams: { ...s.musParams, [family]: cur } };
}

/** The piece's length in seconds (0 = the family's recommendation). */
export function lengthOf(s: MusicSettings, spec: MusicFamilySpec): number {
  const v = s.musLength[spec.id];
  return v && v > 0 && spec.length ? Math.min(spec.length.maxS, Math.max(spec.length.minS, v)) : 0;
}

/** The length the composer shows: the setting, the recommendation, or 0 when
 *  the model decides. */
export function shownLength(s: MusicSettings, spec: MusicFamilySpec): number {
  return lengthOf(s, spec) || spec.length?.defaultS || 0;
}

/** "3:05" */
export function fmtClock(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** What the composer holds. */
export interface MusicDraft {
  prompt: string;
  lyrics: string;
  /** "song" (with lyrics) or "instrumental". */
  mode: "song" | "instrumental";
  /** A score (ABC) to follow (YuE2). */
  scorePath: string | null;
  /** An edit of an earlier round. */
  edit: MusicEdit | null;
}

export const EMPTY_DRAFT: MusicDraft = { prompt: "", lyrics: "", mode: "song", scorePath: null, edit: null };

/** The request the studio sends for the composer's contents. */
export function buildMusicRequest(d: MusicDraft, s: MusicSettings, spec: MusicFamilySpec): MusicRequest {
  const options: Record<string, string> = {};
  for (const [k, v] of Object.entries(s.musParams[spec.id] ?? {})) if (v !== "") options[k] = v;
  const instrumental = d.mode === "instrumental" || !spec.lyrics;
  // An instrumental is always planned: its score's voice is what moves to an
  // instrument (planning "off" means a full score for it).
  if (spec.planning && s.musPlanning !== "full" && !(instrumental && s.musPlanning === "off")) options.cot = s.musPlanning;
  return {
    prompt: d.prompt.trim(),
    lyrics: instrumental ? "" : d.lyrics,
    instrumental: instrumental && !!spec.lyrics,
    seconds: lengthOf(s, spec),
    seed: s.musSeedLock ? Math.max(0, Math.floor(s.musSeed)) : -1,
    options,
    scorePath: spec.planning ? d.scorePath : null,
    edit: d.edit && spec.edits.includes(d.edit.kind) ? d.edit : null,
    outDir: s.musOutputDir.trim() || null,
  };
}

/** A recorded round back in the composer and settings ("reuse"). */
export function draftFromRecord(r: MusicRecord): MusicDraft {
  const p = r.params ?? {};
  return {
    prompt: r.prompt,
    lyrics: r.lyrics || p.lyrics || "",
    mode: p.instrumental ? "instrumental" : "song",
    scorePath: p.scorePath ?? null,
    edit: null,
  };
}

export function settingsFromRecord(r: MusicRecord, s: MusicSettings): Partial<MusicSettings> {
  const p = r.params ?? {};
  const opts = { ...(p.options ?? {}) };
  const planning = opts.cot === "off" || opts.cot === "melody" ? opts.cot : "full";
  delete opts.cot;
  return {
    musParams: { ...s.musParams, [r.family]: opts },
    musLength: { ...s.musLength, [r.family]: p.seconds ?? 0 },
    musPlanning: planning as MusicSettings["musPlanning"],
  };
}

/** How the engine is loaded. Every family's engine options go along; the
 *  backend uses those of the model it loads. */
export function musicLoadOptions(s: MusicSettings): MusicLoadOptions {
  const sessionOptions: Record<string, Record<string, string>> = {};
  for (const [family, opts] of Object.entries(s.musSession)) {
    const kept: Record<string, string> = {};
    for (const [k, v] of Object.entries(opts ?? {})) if (v.trim() !== "") kept[k] = v.trim();
    if (Object.keys(kept).length) sessionOptions[family] = kept;
  }
  return { device: s.musDevice, threads: s.musThreads, sessionOptions };
}

// ---- Progress ----

/** Where a piece stands, from the engine's reports. */
export interface MusicProgress {
  stage: string;
  done: number;
  total: number;
  /** Seconds of music composed, once known. */
  seconds: number | null;
}

/** The stages this piece goes through: the family's, minus those the request
 *  skips (YuE2 writes no score of its own when it follows one, or plans
 *  nothing; ACE-Step's repaint and cover start from the piece's audio, and
 *  its language model writes nothing — measured with the engine: prepare,
 *  render, save). */
export function runStages(spec: MusicFamilySpec, req: MusicRequest): [string, number][] {
  let stages = spec.stages;
  if (spec.planning) {
    const cot = req.options?.cot ?? "full";
    const follows = !!req.scorePath || req.edit?.kind === "rearrange" || req.edit?.kind === "continue";
    // A YuE2 instrumental writes its score first even with planning off.
    const plans = req.instrumental ? !follows : cot !== "off" && !follows;
    if (!plans) stages = stages.filter(([s]) => s !== "score");
  }
  if (req.edit?.kind === "repaint" || req.edit?.kind === "cover") stages = stages.filter(([s]) => s !== "tokens");
  const total = stages.reduce((a, [, w]) => a + w, 0) || 1;
  return stages.map(([s, w]) => [s, w / total]);
}

/** Seconds a YuE2 song will likely run: its budget, or — under the model's
 *  own six-minute limit — a guess from its lyrics. */
export function expectedSeconds(req: MusicRequest, spec: MusicFamilySpec, keptSeconds = 0): number {
  const max = spec.length?.maxS ?? 360;
  if (req.seconds > 0) return req.seconds;
  if (req.instrumental || !req.lyrics.trim()) return Math.min(max, Math.max(keptSeconds + 30, 90));
  const lines = req.lyrics.split("\n").map((l) => l.trim()).filter(Boolean);
  const sections = lines.filter((l) => /^\[.*\]$/.test(l)).length;
  const sung = lines.length - sections;
  const guess = 12 + sung * 4.5 + sections * 4;
  return Math.min(max, Math.max(keptSeconds + 20, 30, guess));
}

/** Overall percentage, 0–100, never moving backwards past `prev`. */
export function musicPercent(
  spec: MusicFamilySpec,
  req: MusicRequest,
  p: MusicProgress,
  prev = 0,
  keptSeconds = 0,
): number {
  if (p.stage === "prepare") return Math.max(prev, 1);
  if (p.stage === "save") return Math.max(prev, 99);
  const stages = runStages(spec, req);
  const i = stages.findIndex(([s]) => s === p.stage);
  if (i < 0) return prev;
  const before = stages.slice(0, i).reduce((a, [, w]) => a + w, 0);
  let total = p.total;
  // The token budget is far beyond where the song will end: measure against
  // where it likely ends instead.
  if (p.stage === "tokens" && spec.tokenRate > 0 && total > 0) {
    total = Math.min(total, expectedSeconds(req, spec, keptSeconds) * spec.tokenRate);
  }
  const frac = total > 0 ? Math.min(0.98, p.done / total) : 0;
  const v = (before + stages[i][1] * frac) * 100;
  return Math.max(prev, Math.min(99, v));
}

/** Seconds left, from the pace so far; null until there is a pace. */
export function musicEta(pct: number, elapsedMs: number): number | null {
  if (pct < 3 || pct >= 99 || elapsedMs < 4000) return null;
  return Math.max(0, Math.round(((elapsedMs / 1000) * (100 - pct)) / pct));
}

/** The lyrics' section tags, offered as one-click inserts. */
export const SECTION_TAGS = ["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Outro]"];

/** Ideas for an empty studio. */
export const IDEAS: { zh: { prompt: string; lyrics: string }[]; en: { prompt: string; lyrics: string }[] } = {
  zh: [
    { prompt: "华语流行,温暖女声,木吉他,轻柔鼓点,夏夜", lyrics: "[Verse]\n晚风吹过街角的灯\n你的影子落在我身旁\n[Chorus]\n就这样慢慢走吧\n把夏天唱成一首歌" },
    { prompt: "lo-fi hip hop, 慵懒电钢琴, 黑胶噪声, 雨声", lyrics: "" },
    { prompt: "电影感交响乐,弦乐渐强,史诗,铜管", lyrics: "" },
    { prompt: "摇滚,失真吉他,有力男声,现场感", lyrics: "[Verse]\n引擎在夜里轰鸣\n公路一直通向天明\n[Chorus]\n别回头 向前冲\n让风替我们大声唱" },
  ],
  en: [
    { prompt: "indie pop, bright acoustic guitar, soft drums, warm female vocal", lyrics: "[Verse]\nSoft morning light is touching the window\nI hear the city waking below\n[Chorus]\nStay with the rhythm, let it carry us home\nSing with the sunrise, we are never alone" },
    { prompt: "lo-fi hip hop, mellow rhodes, vinyl crackle, rain", lyrics: "" },
    { prompt: "cinematic orchestral, swelling strings, epic brass", lyrics: "" },
    { prompt: "rock, overdriven guitars, powerful male vocal, live energy", lyrics: "[Verse]\nEngines roaring through the night\nHighway running to the light\n[Chorus]\nDon't look back, just drive\nLet the wind sing we're alive" },
  ],
};
