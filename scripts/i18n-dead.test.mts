/** Dead-key guard for the i18n table (issue #6 aftermath): a key defined in
 *  `T` but referenced NOWHERE in src/ is translator-facing debt — community
 *  contributors translate it for nothing. The audit that introduced this
 *  guard deleted 21 such keys, each orphaned by a feature REPLACEMENT
 *  (preview→Canvas, downloader→store): the new component shipped new keys,
 *  the old component's file was deleted, and nothing ever red-flagged the
 *  surplus entries — adding a key is compiler-forced, removing one was
 *  invisible optional work. This test makes the asymmetry symmetric.
 *
 *  Method = the audit's pass 2: a key is ALIVE if its quoted name appears
 *  anywhere outside i18n.tsx — literal `t("k")` calls, ternaries
 *  `t(on ? "a" : "b")`, TKey-returning functions, key maps. That covers
 *  every dynamic pattern except template-literal key construction, which the
 *  companion assertion bans outright (none exists in the codebase).
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, test } from "vitest";

import { T } from "../src/lib/i18n";

function walk(dir: string, out: string[]) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) {
      walk(p, out);
    } else if (/\.(ts|tsx)$/.test(name) && !/\.test\.(ts|tsx)$/.test(name) && name !== "i18n.tsx") {
      out.push(p);
    }
  }
  return out;
}

describe("i18n dead-key guard", () => {
  const files = walk("src", []);
  // One big haystack: 566 keys × includes() beats 566 × file loop.
  const haystack = files.map((f) => readFileSync(f, "utf8")).join("\n");

  test("every defined key is referenced somewhere in src/", () => {
    const dead = (Object.keys(T) as (keyof typeof T)[]).filter(
      (k) => !haystack.includes(`"${k}"`) && !haystack.includes(`'${k}'`),
    );
    expect(
      dead,
      `dead i18n keys — delete them from src/lib/i18n.tsx (or reference them); ` +
        `every dead key is a string community translators will translate for nothing`,
    ).toEqual([]);
  });

  test("no template-literal t() keys — they would defeat this guard", () => {
    // t(`...`) constructs keys invisibly to the string search above; the
    // codebase uses explicit ternaries / key maps instead. Keep it that way.
    const offenders = files.filter((f) => /\bt\(\s*`/.test(readFileSync(f, "utf8")));
    expect(offenders, "use explicit key maps or ternaries, not template keys").toEqual([]);
  });
});
