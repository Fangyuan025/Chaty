import { describe, expect, test } from "vitest";

import {
  EMPTY_DRAFT,
  MUSIC_SETTINGS_DEFAULTS,
  buildMusicRequest,
  draftFromRecord,
  expectedSeconds,
  familyParams,
  isCustom,
  lengthOf,
  musicEta,
  musicLoadOptions,
  musicPercent,
  paramValue,
  runStages,
  sessionParams,
  settingsFromRecord,
  withParam,
  type MusicSettings,
} from "./musicGen";
import type { EngineOption, MusicFamilySpec, MusicRecord, MusicRequest } from "./ipc";

// The backend's descriptors (src-tauri/src/musicgen/family.rs), as the studio
// receives them.
const YUE2: MusicFamilySpec = {
  id: "yue2",
  name: "YuE2",
  repo: "audio-cpp/Yue2-3B-GGUF",
  repoDir: "",
  layout: "yue2",
  promptText: false,
  promptOption: "style",
  lyrics: "lyrics",
  lyricsRequired: false,
  instrumentalLyrics: "",
  length: { option: "semantic_max_tokens", perSecond: 25, defaultS: 0, minS: 10, maxS: 360, autoValue: null, isLimit: true },
  planning: true,
  fixed: [],
  edits: ["rearrange", "continue"],
  stages: [
    ["score", 0.1],
    ["tokens", 0.45],
    ["render", 0.4],
    ["decode", 0.05],
  ],
  tokenRate: 25,
  promptTemplate: null,
};

const STABLE: MusicFamilySpec = {
  ...YUE2,
  id: "stable_audio",
  name: "Stable Audio 3",
  layout: "single",
  promptText: true,
  promptOption: null,
  lyrics: null,
  instrumentalLyrics: null,
  length: { option: "duration_seconds", perSecond: 1, defaultS: 120, minS: 1, maxS: 380, autoValue: null, isLimit: false },
  planning: false,
  edits: ["variation", "inpaint"],
  stages: [
    ["render", 0.92],
    ["decode", 0.08],
  ],
  tokenRate: 0,
};

const S: MusicSettings = { ...MUSIC_SETTINGS_DEFAULTS };

describe("parameters", () => {
  test("every family's table starts at its recommendation", () => {
    const steps = familyParams("yue2").find((p) => p.key === "num_inference_steps")!;
    expect(paramValue(S, "yue2", steps)).toBe("8");
    expect(isCustom(S, "yue2", "num_inference_steps")).toBe(false);
    const s2 = { ...S, ...withParam(S, "yue2", "num_inference_steps", "16") };
    expect(paramValue(s2, "yue2", steps)).toBe("16");
    expect(isCustom(s2, "yue2", "num_inference_steps")).toBe(true);
    // Per family: another family's steps are untouched.
    expect(isCustom(s2, "ace_step", "num_inference_steps")).toBe(false);
    const s3 = { ...s2, ...withParam(s2, "yue2", "num_inference_steps", null) };
    expect(isCustom(s3, "yue2", "num_inference_steps")).toBe(false);
  });

  test("options the engine lists beyond the table are offered too, the managed ones never", () => {
    const engine: EngineOption[] = [
      { name: "guidance_scale", kind: "float", description: "", default: "1.0", min: "0", max: "20", required: false },
      { name: "style", kind: "string", description: "", default: "", min: "", max: "", required: true },
      { name: "nar_noise_file", kind: "path", description: "", default: "", min: "", max: "", required: false },
      { name: "new_knob", kind: "a|b|c", description: "a new one", default: "b", min: "", max: "", required: false },
    ];
    const keys = familyParams("yue2", engine).map((p) => p.key);
    expect(keys.filter((k) => k === "guidance_scale")).toHaveLength(1);
    expect(keys).not.toContain("style");
    expect(keys).not.toContain("nar_noise_file");
    const knob = familyParams("yue2", engine).find((p) => p.key === "new_knob")!;
    expect(knob.kind).toBe("enum");
    expect(knob.choices?.map((c) => c.value)).toEqual(["a", "b", "c"]);
    expect(knob.def).toBe("b");
    // Package files are what the pick names, not settings.
    const session = sessionParams([
      { name: "yue2.model_gguf", kind: "string", description: "", default: "x", min: "", max: "", required: false },
      { name: "yue2.attention", kind: "auto|flash|eager", description: "", default: "auto", min: "", max: "", required: false },
    ]);
    expect(session.map((p) => p.key)).toEqual(["yue2.attention"]);
  });

  test("the engine's options go along for every family, blanks dropped", () => {
    const s = { ...S, musSession: { yue2: { "yue2.attention": "eager", "yue2.vae_weight_type": " " }, ace_step: {} } };
    expect(musicLoadOptions(s)).toEqual({ device: "gpu", threads: 0, sessionOptions: { yue2: { "yue2.attention": "eager" } } });
  });
});

describe("requests", () => {
  const draft = { ...EMPTY_DRAFT, prompt: " indie pop ", lyrics: "[Verse]\nHi" };

  test("recommended settings send nothing but the composer's words", () => {
    const r = buildMusicRequest(draft, S, YUE2);
    expect(r).toMatchObject({ prompt: "indie pop", lyrics: "[Verse]\nHi", instrumental: false, seconds: 0, seed: -1, options: {} });
  });

  test("only what moved off the recommendation is sent", () => {
    const s: MusicSettings = {
      ...S,
      musParams: { yue2: { num_inference_steps: "16", semantic_top_k: "" } },
      musPlanning: "off",
      musLength: { yue2: 90 },
      musSeedLock: true,
      musSeed: 7,
    };
    const r = buildMusicRequest(draft, s, YUE2);
    expect(r.options).toEqual({ num_inference_steps: "16", cot: "off" });
    expect(r.seconds).toBe(90);
    expect(r.seed).toBe(7);
  });

  test("instrumental sends no lyrics; a family that sings nothing never does", () => {
    expect(buildMusicRequest({ ...draft, mode: "instrumental" }, S, YUE2)).toMatchObject({ lyrics: "", instrumental: true });
    expect(buildMusicRequest(draft, S, STABLE)).toMatchObject({ lyrics: "", instrumental: false });
    // A score to follow is YuE2's alone; an edit the family does not make is dropped.
    const r = buildMusicRequest(
      { ...draft, scorePath: "/a.abc", edit: { kind: "rearrange", parentId: "p", start: 0, end: 0, strength: 0 } },
      S,
      STABLE,
    );
    expect(r.scorePath).toBeNull();
    expect(r.edit).toBeNull();
  });

  test("a length is clamped to the family's range", () => {
    expect(lengthOf({ ...S, musLength: { yue2: 5 } }, YUE2)).toBe(10);
    expect(lengthOf({ ...S, musLength: { yue2: 9999 } }, YUE2)).toBe(360);
    expect(lengthOf(S, YUE2)).toBe(0);
  });

  test("a recorded round comes back into the composer and settings", () => {
    const rec: MusicRecord = {
      id: "r",
      sessionId: "s",
      prompt: "jazz",
      lyrics: "",
      params: { instrumental: true, seconds: 60, options: { cot: "melody", num_inference_steps: "12" } },
      audio: { path: "/x.wav", seconds: 58, sampleRate: 48000, channels: 2, seed: 3 },
      peaks: [],
      model: "m",
      family: "yue2",
      createdAt: 0,
      elapsedMs: 0,
    };
    expect(draftFromRecord(rec)).toMatchObject({ prompt: "jazz", mode: "instrumental", edit: null });
    const back = settingsFromRecord(rec, S);
    expect(back.musPlanning).toBe("melody");
    expect(back.musParams?.yue2).toEqual({ num_inference_steps: "12" });
    expect(back.musLength?.yue2).toBe(60);
  });
});

describe("progress", () => {
  const req = (over: Partial<MusicRequest> = {}): MusicRequest => ({
    prompt: "pop",
    lyrics: "[Verse]\na\nb\nc\nd",
    instrumental: false,
    seconds: 0,
    seed: -1,
    options: {},
    ...over,
  });

  test("the stages a piece goes through follow the request", () => {
    expect(runStages(YUE2, req()).map(([s]) => s)).toEqual(["score", "tokens", "render", "decode"]);
    expect(runStages(YUE2, req({ options: { cot: "off" } })).map(([s]) => s)).toEqual(["tokens", "render", "decode"]);
    // Following a score (its own or a parent's) writes none.
    expect(runStages(YUE2, req({ scorePath: "/a.abc" })).map(([s]) => s)).not.toContain("score");
    expect(runStages(YUE2, req({ edit: { kind: "continue", parentId: "p", start: 0, end: 0, strength: 0 } })).map(([s]) => s)).not.toContain(
      "score",
    );
    const total = runStages(YUE2, req({ options: { cot: "off" } })).reduce((a, [, w]) => a + w, 0);
    expect(total).toBeCloseTo(1);
  });

  test("the percentage runs through the stages and never goes back", () => {
    const r = req({ seconds: 40 });
    let prev = 0;
    const seen: number[] = [];
    for (const p of [
      { stage: "prepare", done: 0, total: 0 },
      { stage: "score", done: 100, total: 4096 },
      { stage: "tokens", done: 500, total: 9000 },
      { stage: "tokens", done: 1000, total: 9000 },
      { stage: "render", done: 2, total: 8 },
      { stage: "decode", done: 1, total: 2 },
      { stage: "save", done: 0, total: 0 },
    ]) {
      prev = musicPercent(YUE2, r, { ...p, seconds: null }, prev);
      seen.push(prev);
    }
    expect(seen).toEqual([...seen].sort((a, b) => a - b));
    // Half the 40 s composed: past the score and halfway through composing.
    expect(seen[3]).toBeCloseTo((0.1 + 0.45 * 0.98) * 100, 0);
    expect(seen[2]).toBeCloseTo((0.1 + 0.45 * 0.5) * 100, 0);
    expect(seen[6]).toBe(99);
    // A report of an earlier stage does not pull it back.
    expect(musicPercent(YUE2, r, { stage: "score", done: 1, total: 4096, seconds: null }, 60)).toBe(60);
  });

  test("a song with no length set is measured against where it will likely end", () => {
    const r = req();
    const exp = expectedSeconds(r, YUE2);
    expect(exp).toBeGreaterThanOrEqual(30);
    expect(exp).toBeLessThanOrEqual(360);
    // 9000 tokens is six minutes of budget: at the likely end, composing is
    // nearly done rather than a sliver of the way.
    const pct = musicPercent(YUE2, r, { stage: "tokens", done: Math.round(exp * 25 * 0.9), total: 9000, seconds: null });
    expect(pct).toBeGreaterThan(45);
    // Instrumental: a guess of its own, and never before what is kept.
    expect(expectedSeconds(req({ instrumental: true, lyrics: "" }), YUE2, 200)).toBeGreaterThanOrEqual(230);
  });

  test("a family whose totals are exact uses them as they are", () => {
    expect(musicPercent(STABLE, req(), { stage: "render", done: 4, total: 8, seconds: null })).toBeCloseTo(46, 0);
  });

  test("an ETA only once there is a pace", () => {
    expect(musicEta(2, 60_000)).toBeNull();
    expect(musicEta(50, 60_000)).toBe(60);
    expect(musicEta(99.5, 60_000)).toBeNull();
  });
});
