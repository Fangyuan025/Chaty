/** Drift lock: officialSkills.ts is generated — editing resources/skills/
 *  without rerunning scripts/bundle-skills.mts must turn CI red. Lives in
 *  scripts/ (node territory) because src/ is typechecked against browser
 *  libs and cannot import node:fs. */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, test } from "vitest";

import { OFFICIAL_SKILL_FILES, OFFICIAL_SKILL_SUPPORT } from "../src/lib/officialSkills";

const dir = "resources/skills";

function walk(root: string, rel = ""): string[] {
  const out: string[] = [];
  for (const name of readdirSync(path.join(root, rel)).sort()) {
    if (name === ".DS_Store" || name === "__pycache__" || name.endsWith(".pyc")) continue;
    const r = rel ? `${rel}/${name}` : name;
    if (statSync(path.join(root, r)).isDirectory()) out.push(...walk(root, r));
    else out.push(r);
  }
  return out;
}

describe("skill bundle freshness", () => {
  test("bundle is in sync with resources/skills — rerun scripts/bundle-skills.mts after editing", () => {
    const byFile = new Map(OFFICIAL_SKILL_FILES.map((f) => [f.file, f.text]));
    for (const e of readdirSync(dir).sort()) {
      if (e === ".DS_Store") continue;
      const p = path.join(dir, e);
      if (e.endsWith(".md") && statSync(p).isFile()) {
        expect(byFile.get(e), `${e} drifted`).toBe(readFileSync(p, "utf8"));
      } else if (statSync(p).isDirectory()) {
        expect(byFile.get(`${e}/SKILL.md`), `${e}/SKILL.md drifted`).toBe(readFileSync(path.join(p, "SKILL.md"), "utf8"));
        const hash = createHash("sha256");
        for (const f of walk(p).filter((x) => x !== "SKILL.md")) {
          hash.update(f).update("\0").update(readFileSync(path.join(p, f), "utf8")).update("\0");
        }
        expect(OFFICIAL_SKILL_SUPPORT[e]?.rev, `${e} support drifted`).toBe(hash.digest("hex").slice(0, 16));
      }
    }
  });
});
