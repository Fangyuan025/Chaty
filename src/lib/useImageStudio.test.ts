import { describe, expect, test } from "vitest";

import { IMAGE_SETTINGS_DEFAULTS, buildRequest } from "./imageGen";
import type { ImageDefaults } from "./ipc";
import { parseDraft, restored } from "./useImageStudio";

const D: ImageDefaults = {
  steps: 8,
  cfgScale: 1,
  guidance: null,
  sampler: "euler",
  scheduler: "",
  flowShift: 0,
  baseSize: 1024,
  align: 16,
  negativePrompt: false,
};

describe("a session's unsent draft", () => {
  test("reads back whatever state it was stored in", () => {
    expect(parseDraft("")).toEqual({ prompt: "", negative: null, reference: null });
    expect(parseDraft("not json")).toEqual({ prompt: "", negative: null, reference: null });
    expect(parseDraft('{"prompt":"a cat"}')).toEqual({ prompt: "a cat", negative: null, reference: null });
    expect(parseDraft('{"prompt":"x","negative":"blurry","reference":{"path":"/a.png","parentId":"r1"}}')).toEqual({
      prompt: "x",
      negative: "blurry",
      reference: { path: "/a.png", parentId: "r1" },
    });
    expect(parseDraft('{"reference":{"parentId":"r1"}}').reference).toBeNull();
  });

  test("a round that made nothing gives its prompt and picture back", () => {
    const sent = { ...buildRequest("give it a hat", "", IMAGE_SETTINGS_DEFAULTS, D, { path: "/r1.png", edit: true }), parentId: "r1" };
    const empty = { prompt: "", negative: null, reference: null };
    expect(restored(empty, sent)).toEqual({ prompt: "give it a hat", negative: null, reference: { path: "/r1.png", parentId: "r1" } });
    // Something typed since is not overwritten.
    const typed = { prompt: "new idea", negative: null, reference: null };
    expect(restored(typed, sent)).toBe(typed);
  });
});
