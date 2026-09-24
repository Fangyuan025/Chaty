// The image studio's arithmetic: picture sizes from an aspect ratio, the
// settings a generation actually runs with, and how far along it is. Kept
// free of React so every rule here is a plain function with a test.

import type { ImageDefaults, ImageLoadOptions, ImageRequest } from "./ipc";

/** Aspect ratios offered in the studio, landscape and portrait paired. */
export const ASPECTS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"] as const;
export type Aspect = (typeof ASPECTS)[number] | "custom";

/** Samplers the engine knows (stable-diffusion.cpp's names). */
export const SAMPLERS = [
  "euler",
  "euler_a",
  "heun",
  "dpm2",
  "dpm++2s_a",
  "dpm++2m",
  "dpm++2mv2",
  "ipndm",
  "ipndm_v",
  "lcm",
  "ddim_trailing",
  "tcd",
  "res_multistep",
  "res_2s",
  "er_sde",
  "dpm++2m_sde",
  "lms",
];

export const SCHEDULERS = [
  "discrete",
  "karras",
  "exponential",
  "ays",
  "gits",
  "sgm_uniform",
  "simple",
  "smoothstep",
  "kl_optimal",
  "lcm",
  "bong_tangent",
  "beta",
];

/** Base resolutions offered (the side of the equivalent square). */
export const BASE_SIZES = [512, 768, 1024, 1280, 1536, 2048];

/** Round to the nearest multiple of `align`, never below one step of it. */
function snap(v: number, align: number): number {
  return Math.max(align, Math.round(v / align) * align);
}

/** Width × height for an aspect ratio at the area of a `base`² square, both
 *  multiples of `align` — what the model was trained at, reshaped. */
export function sizeFor(aspect: string, base: number, align: number): { width: number; height: number } {
  const [a, b] = aspect.split(":").map(Number);
  const ratio = a > 0 && b > 0 ? a / b : 1;
  const area = base * base;
  const w = Math.sqrt(area * ratio);
  return { width: snap(w, align), height: snap(w / ratio, align) };
}

/** The studio's settings, as the settings panel stores them. Zero or empty
 *  means "what the loaded model recommends". */
export interface ImageSettings {
  imgAspect: string;
  imgBase: number;
  imgCustomW: number;
  imgCustomH: number;
  imgSteps: number;
  imgCfg: number;
  imgGuidance: number;
  imgSampler: string;
  imgScheduler: string;
  imgFlowShift: number;
  imgBatch: number;
  imgNegative: string;
  imgClipSkip: number;
  imgVaeTiling: boolean;
  imgPreview: "proj" | "vae" | "none";
  imgPreviewInterval: number;
  imgFormat: "png" | "jpg";
  imgOutputDir: string;
  imgSeedLock: boolean;
  imgSeed: number;
  imgStrength: number;
  imgDevice: "gpu" | "cpu";
  imgOffload: boolean;
  imgTeCpu: boolean;
  imgVaeCpu: boolean;
  imgFlashAttn: boolean;
  imgThreads: number;
  imgMaxVram: number;
  imgMmap: boolean;
  /** Companions picked by hand: model path → role → file ("" = none). */
  imgComponents: Record<string, Record<string, string>>;
}

export const IMAGE_SETTINGS_DEFAULTS: ImageSettings = {
  imgAspect: "1:1",
  imgBase: 0,
  imgCustomW: 1024,
  imgCustomH: 1024,
  imgSteps: 0,
  imgCfg: 0,
  imgGuidance: 0,
  imgSampler: "",
  imgScheduler: "",
  imgFlowShift: 0,
  imgBatch: 1,
  imgNegative: "",
  imgClipSkip: -1,
  imgVaeTiling: false,
  imgPreview: "proj",
  imgPreviewInterval: 1,
  imgFormat: "png",
  imgOutputDir: "",
  imgSeedLock: false,
  imgSeed: 42,
  imgStrength: 0.75,
  // GPU, and all of it: the image engine offloads the way the chat engine
  // does — everything on the GPU, only what does not fit spills to RAM.
  imgDevice: "gpu",
  imgOffload: false,
  imgTeCpu: false,
  imgVaeCpu: false,
  imgFlashAttn: true,
  imgThreads: 0,
  imgMaxVram: 0,
  imgMmap: false,
  imgComponents: {},
};

/** The values a generation runs with: the setting where one is set, the
 *  model's recommendation where it is left on auto. */
export interface Effective {
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  guidance: number | null;
  sampler: string;
  scheduler: string;
  flowShift: number;
  batch: number;
}

export function effective(s: ImageSettings, d: ImageDefaults): Effective {
  const base = s.imgBase > 0 ? s.imgBase : d.baseSize;
  const size =
    s.imgAspect === "custom"
      ? { width: snap(s.imgCustomW || d.baseSize, d.align), height: snap(s.imgCustomH || d.baseSize, d.align) }
      : sizeFor(s.imgAspect || "1:1", base, d.align);
  return {
    ...size,
    steps: s.imgSteps > 0 ? s.imgSteps : d.steps,
    cfgScale: s.imgCfg > 0 ? s.imgCfg : d.cfgScale,
    guidance: d.guidance == null ? null : s.imgGuidance > 0 ? s.imgGuidance : d.guidance,
    sampler: s.imgSampler || d.sampler,
    scheduler: s.imgScheduler || d.scheduler,
    flowShift: s.imgFlowShift > 0 ? s.imgFlowShift : d.flowShift,
    batch: Math.min(8, Math.max(1, s.imgBatch || 1)),
  };
}

/** The request the studio sends for `prompt`. */
export function buildRequest(
  prompt: string,
  negative: string,
  s: ImageSettings,
  d: ImageDefaults,
  ref: { path: string; edit: boolean } | null,
): ImageRequest {
  const e = effective(s, d);
  return {
    prompt,
    negativePrompt: d.negativePrompt ? negative : "",
    width: e.width,
    height: e.height,
    steps: e.steps,
    cfgScale: e.cfgScale,
    guidance: e.guidance,
    sampler: e.sampler,
    scheduler: e.scheduler,
    seed: s.imgSeedLock ? Math.max(0, Math.floor(s.imgSeed)) : -1,
    batchCount: e.batch,
    flowShift: e.flowShift,
    vaeTiling: s.imgVaeTiling,
    clipSkip: s.imgClipSkip,
    preview: s.imgPreview,
    previewInterval: Math.max(1, s.imgPreviewInterval || 1),
    format: s.imgFormat,
    initImage: ref && !ref.edit ? ref.path : null,
    strength: s.imgStrength,
    refImages: ref && ref.edit ? [ref.path] : [],
    outDir: s.imgOutputDir.trim() || null,
  };
}

/** The settings a recorded request ran with, for "reuse prompt & settings":
 *  everything that shapes the picture, back as it was sent. The seed stays as
 *  it is (reusing one has its own action), and so does the output folder. */
export function settingsFromRequest(p: Partial<ImageRequest>, s: ImageSettings): Partial<ImageSettings> {
  return {
    imgAspect: "custom",
    imgCustomW: p.width ?? s.imgCustomW,
    imgCustomH: p.height ?? s.imgCustomH,
    imgSteps: p.steps ?? s.imgSteps,
    imgCfg: p.cfgScale ?? s.imgCfg,
    // null: the model takes no guidance, so the setting has nothing to say.
    imgGuidance: p.guidance ?? s.imgGuidance,
    imgSampler: p.sampler ?? s.imgSampler,
    imgScheduler: p.scheduler ?? s.imgScheduler,
    imgFlowShift: p.flowShift ?? s.imgFlowShift,
    imgBatch: p.batchCount ?? s.imgBatch,
    imgVaeTiling: p.vaeTiling ?? s.imgVaeTiling,
    imgClipSkip: p.clipSkip ?? s.imgClipSkip,
    imgPreview: p.preview ?? s.imgPreview,
    imgPreviewInterval: p.previewInterval ?? s.imgPreviewInterval,
    imgFormat: p.format ?? s.imgFormat,
    imgStrength: p.strength ?? s.imgStrength,
  };
}

/** The reference picture a recorded request used, if any. */
export function referenceOf(p: Partial<ImageRequest>): string | null {
  return p.initImage || p.refImages?.[0] || null;
}

/** How the engine is loaded, for the model at `path`. */
export function loadOptions(s: ImageSettings, path: string): ImageLoadOptions {
  return {
    components: s.imgComponents[path] ?? {},
    device: s.imgDevice,
    offloadToCpu: s.imgOffload,
    textEncoderOnCpu: s.imgTeCpu,
    vaeOnCpu: s.imgVaeCpu,
    flashAttn: s.imgFlashAttn,
    threads: s.imgThreads,
    maxVramGb: s.imgMaxVram,
    mmap: s.imgMmap,
  };
}

/** Where a generation stands, from the engine's reports. */
export interface GenState {
  stage: string;
  index: number;
  count: number;
  step: number;
  steps: number;
  /** Seconds per sampling step, smoothed. 0 = not measured yet. */
  secsPerStep: number;
}

// How the bar is shared out: reading the prompt is quick, sampling is nearly
// all of it, decoding the latents is the tail.
const ENCODE_END = 4;
const SAMPLE_END = 92;

/** Overall percentage, 0–100. A lazy weight load keeps the bar where it was
 *  (`prev`) rather than inventing movement. */
export function percent(g: GenState, prev = 0): number {
  const frac = g.steps > 0 ? Math.min(1, g.step / g.steps) : 0;
  let v: number;
  switch (g.stage) {
    case "encode":
      v = ENCODE_END * 0.5;
      break;
    case "sample":
      v = ENCODE_END + ((SAMPLE_END - ENCODE_END) * (g.index + frac)) / Math.max(1, g.count);
      break;
    case "decode":
      v = SAMPLE_END + (100 - SAMPLE_END) * frac;
      break;
    default:
      v = prev;
  }
  return Math.max(prev, Math.min(100, v));
}

/** Seconds left, from the sampling rate; null until a step has been timed. */
export function etaSeconds(g: GenState): number | null {
  if (g.stage !== "sample" || g.secsPerStep <= 0 || g.steps <= 0) return null;
  const left = (g.count - g.index - 1) * g.steps + (g.steps - g.step);
  return Math.max(0, Math.round(left * g.secsPerStep));
}

/** Smooth a step time into the running average. */
export function smooth(prev: number, sample: number): number {
  if (!(sample > 0)) return prev;
  return prev > 0 ? prev * 0.6 + sample * 0.4 : sample;
}

/** "12.3s", "2m 05s" */
export function fmtDuration(ms: number): string {
  const s = Math.max(0, ms / 1000);
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${String(Math.round(s % 60)).padStart(2, "0")}s`;
}
