/** The locale contract behind community localization (issue #6): zh/en are
 *  complete by construction, community locales may be partial and fall back
 *  to English, and the agent layer only ever sees zh|en. */
import { describe, expect, test } from "vitest";

import { agentLang, LANGS, lookup, T, type TKey } from "./i18n";

describe("i18n locale contract", () => {
  test("the native pair is complete and non-empty on every key", () => {
    for (const [k, e] of Object.entries(T)) {
      expect(e.zh, `${k}.zh`).toBeTruthy();
      expect(e.en, `${k}.en`).toBeTruthy();
    }
  });

  test("a community locale falls back to English, never a blank or raw key", () => {
    const untranslated = (Object.keys(T) as TKey[]).find(
      (k) => !(T[k] as Record<string, string | undefined>).pt,
    );
    expect(untranslated).toBeTruthy(); // pt starts empty by definition
    expect(lookup(untranslated!, "pt")).toBe(lookup(untranslated!, "en"));
  });

  test("variable substitution works across locales", () => {
    const key = (Object.keys(T) as TKey[]).find((k) => T[k].en.includes("{name}"));
    if (key) {
      expect(lookup(key, "pt", { name: "abc" })).toContain("abc");
    }
  });

  test("LANGS drives the switch: unique ids, native pair first", () => {
    const ids = LANGS.map((l) => l.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain("zh");
    expect(ids).toContain("en");
  });

  test("the agent layer only ever sees zh|en", () => {
    expect(agentLang("zh")).toBe("zh");
    expect(agentLang("en")).toBe("en");
    expect(agentLang("pt")).toBe("en");
  });
});
