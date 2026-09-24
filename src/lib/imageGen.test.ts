import { describe, expect, test } from "vitest";

import {
  IMAGE_SETTINGS_DEFAULTS,
  buildRequest,
  effective,
  etaSeconds,
  loadOptions,
  percent,
  sizeFor,
  smooth,
  type GenState,
} from "./imageGen";
import type { ImageDefaults } from "./ipc";

const QWEN21: ImageDefaults = {
  steps: 20,
  cfgScale: 6,
  guidance: null,
  sampler: "euler",
  scheduler: "",
  flowShift: 0,
  baseSize: 1024,
  align: 32,
  negativePrompt: true,
};

describe("picture sizes", () => {
  test("a square is the base itself", () => {
    expect(sizeFor("1:1", 1024, 32)).toEqual({ width: 1024, height: 1024 });
  });

  test("other ratios keep the area and land on the model's grid", () => {
    const s = sizeFor("16:9", 1024, 32);
    expect(s.width % 32).toBe(0);
    expect(s.height % 32).toBe(0);
    expect(s.width).toBeGreaterThan(s.height);
    // Within one grid step of the square's pixel count.
    expect(Math.abs(s.width * s.height - 1024 * 1024)).toBeLessThan(1024 * 32 * 2);
    const p = sizeFor("9:16", 1024, 32);
    expect([p.width, p.height]).toEqual([s.height, s.width]);
  });
});

describe("settings left on auto follow the model", () => {
  test("zero and empty mean the model's recommendation", () => {
    const e = effective(IMAGE_SETTINGS_DEFAULTS, QWEN21);
    expect(e).toMatchObject({ width: 1024, height: 1024, steps: 20, cfgScale: 6, sampler: "euler", batch: 1 });
    expect(e.guidance).toBeNull();
  });

  test("a set value wins, and the batch stays in range", () => {
    const e = effective({ ...IMAGE_SETTINGS_DEFAULTS, imgSteps: 30, imgCfg: 4.5, imgBase: 512, imgBatch: 99 }, QWEN21);
    expect(e).toMatchObject({ width: 512, height: 512, steps: 30, cfgScale: 4.5, batch: 8 });
  });

  test("a custom size snaps to the model's grid", () => {
    const e = effective({ ...IMAGE_SETTINGS_DEFAULTS, imgAspect: "custom", imgCustomW: 1000, imgCustomH: 700 }, QWEN21);
    expect(e.width).toBe(992);
    expect(e.height).toBe(704);
  });

  test("distilled guidance only for models that have it", () => {
    const flux = { ...QWEN21, guidance: 3.5, negativePrompt: false };
    expect(effective(IMAGE_SETTINGS_DEFAULTS, flux).guidance).toBe(3.5);
    expect(effective({ ...IMAGE_SETTINGS_DEFAULTS, imgGuidance: 2 }, flux).guidance).toBe(2);
  });
});

describe("the request", () => {
  test("random seed unless locked; negative prompt only where it does something", () => {
    const r = buildRequest("a cat", "blurry", IMAGE_SETTINGS_DEFAULTS, QWEN21, null);
    expect(r.seed).toBe(-1);
    expect(r.negativePrompt).toBe("blurry");
    expect(r.initImage).toBeNull();
    expect(r.refImages).toEqual([]);
    const locked = buildRequest("a cat", "blurry", { ...IMAGE_SETTINGS_DEFAULTS, imgSeedLock: true, imgSeed: 7 }, { ...QWEN21, negativePrompt: false }, null);
    expect(locked.seed).toBe(7);
    expect(locked.negativePrompt).toBe("");
  });

  test("a reference picture is edited by an editing model, a starting point otherwise", () => {
    expect(buildRequest("x", "", IMAGE_SETTINGS_DEFAULTS, QWEN21, { path: "/a.png", edit: true }).refImages).toEqual(["/a.png"]);
    expect(buildRequest("x", "", IMAGE_SETTINGS_DEFAULTS, QWEN21, { path: "/a.png", edit: false }).initImage).toBe("/a.png");
  });

  test("the engine loads on the GPU, all of it, by default", () => {
    const o = loadOptions(IMAGE_SETTINGS_DEFAULTS, "/m/q.gguf");
    expect(o).toMatchObject({ device: "gpu", offloadToCpu: false, textEncoderOnCpu: false, vaeOnCpu: false, flashAttn: true });
    const picked = loadOptions({ ...IMAGE_SETTINGS_DEFAULTS, imgComponents: { "/m/q.gguf": { vae: "/v.safetensors" } } }, "/m/q.gguf");
    expect(picked.components).toEqual({ vae: "/v.safetensors" });
  });
});

describe("progress", () => {
  const g = (p: Partial<GenState>): GenState => ({ stage: "sample", index: 0, count: 1, step: 0, steps: 20, secsPerStep: 0, ...p });

  test("moves forward through the phases and ends at 100", () => {
    const enc = percent(g({ stage: "encode" }));
    const half = percent(g({ step: 10 }), enc);
    const sampled = percent(g({ step: 20 }), half);
    const dec = percent(g({ stage: "decode", step: 1, steps: 1 }), sampled);
    expect(enc).toBeGreaterThan(0);
    expect(half).toBeGreaterThan(enc);
    expect(sampled).toBeGreaterThan(half);
    expect(dec).toBe(100);
  });

  test("a batch shares the sampling span between its pictures", () => {
    const first = percent(g({ count: 2, index: 0, step: 20 }));
    const second = percent(g({ count: 2, index: 1, step: 10 }));
    expect(second).toBeGreaterThan(first);
    expect(first).toBeCloseTo(48, 0);
  });

  test("a weight load holds the bar where it was", () => {
    expect(percent(g({ stage: "weights", step: 500, steps: 1000 }), 30)).toBe(30);
  });

  test("time left comes from the step rate", () => {
    expect(etaSeconds(g({ step: 5 }))).toBeNull();
    expect(etaSeconds(g({ step: 5, secsPerStep: 2 }))).toBe(30);
    expect(etaSeconds(g({ count: 2, index: 0, step: 5, secsPerStep: 1 }))).toBe(35);
    expect(smooth(0, 2)).toBe(2);
    expect(smooth(2, 4)).toBeCloseTo(2.8);
  });
});
