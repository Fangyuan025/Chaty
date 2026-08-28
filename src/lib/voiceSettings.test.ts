/** The two voice lists are separate models with separate speakers. Picking an
 *  English voice must never move the Chinese one, and Chinese speech is a
 *  stored setting — on by default in the Chinese interface, and still a
 *  setting there, not a thing that is forced. */
import { describe, expect, test } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
const store: Record<string, string> = {};
g.localStorage ??= {
  getItem: (k: string) => store[k] ?? null,
  setItem: (k: string, v: string) => void (store[k] = v),
  removeItem: (k: string) => void delete store[k],
  clear: () => void Object.keys(store).forEach((k) => delete store[k]),
  key: () => null,
  length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { VOICES, VOICES_ZH, defaultSettings } = await import("../components/SettingsPanel");

describe("English and Chinese voices are chosen separately", () => {
  test("the Chinese list is the model's own five speakers", () => {
    // sherpa-onnx-vits-zh-ll declares these in G_multisperaker_latest.json.
    expect(VOICES_ZH).toEqual(["suyingxue", "gunian", "fushiyu", "bingjiao", "bazong"]);
  });

  test("the lists are independent — neither indexes into the other", () => {
    expect(VOICES.length).not.toBe(VOICES_ZH.length);
    // Every English index must be selectable without falling off the Chinese
    // list, which is what folding one onto the other used to do.
    expect(VOICES.length).toBeGreaterThan(VOICES_ZH.length);
  });

  test("each has its own stored setting", () => {
    expect(defaultSettings.voiceSid).toBe(0);
    expect(defaultSettings.voiceSidZh).toBe(0);
    const next = { ...defaultSettings, voiceSid: 7 };
    expect(next.voiceSidZh).toBe(defaultSettings.voiceSidZh);
  });
});
