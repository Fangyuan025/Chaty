/**
 * Regenerate src/lib/officialSkills.ts from resources/skills/.
 * Run after editing any official skill:  npx tsx scripts/bundle-skills.mts
 *
 * Two skill shapes:
 *  - `<name>.md` — knowledge-only skill; the whole file is the prompt text.
 *  - `<name>/SKILL.md` + support files — the SKILL.md is the prompt text and
 *    everything else (scripts, references, examples) is materialized into the
 *    workspace at `.chaty/skills/<name>/` on first use_skill, so the skill can
 *    ship runnable tooling without ever putting code into model context.
 */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const dir = "resources/skills";
const entries = readdirSync(dir).sort();

const skillRows: string[] = [];
const supportRows: string[] = [];

function walkFiles(root: string, rel = ""): string[] {
  const out: string[] = [];
  for (const name of readdirSync(path.join(root, rel)).sort()) {
    // Editor/runtime debris must never ship: a stray .pyc once rode the
    // bundle into every workspace as mangled UTF-8.
    if (name === ".DS_Store" || name === "__pycache__" || name.endsWith(".pyc")) continue;
    const r = rel ? `${rel}/${name}` : name;
    if (statSync(path.join(root, r)).isDirectory()) out.push(...walkFiles(root, r));
    else out.push(r);
  }
  return out;
}

for (const e of entries) {
  const p = path.join(dir, e);
  if (e.endsWith(".md") && statSync(p).isFile()) {
    skillRows.push(`  { file: ${JSON.stringify(e)}, text: ${JSON.stringify(readFileSync(p, "utf8"))} },`);
  } else if (statSync(p).isDirectory()) {
    const skillMd = path.join(p, "SKILL.md");
    skillRows.push(`  { file: ${JSON.stringify(`${e}/SKILL.md`)}, text: ${JSON.stringify(readFileSync(skillMd, "utf8"))} },`);
    const files = walkFiles(p).filter((f) => f !== "SKILL.md");
    const hash = createHash("sha256");
    const fileRows = files.map((f) => {
      const text = readFileSync(path.join(p, f), "utf8");
      hash.update(f).update("\0").update(text).update("\0");
      return `      { path: ${JSON.stringify(f)}, text: ${JSON.stringify(text)} },`;
    });
    supportRows.push(
      `  ${JSON.stringify(e)}: {\n    rev: ${JSON.stringify(hash.digest("hex").slice(0, 16))},\n    files: [\n${fileRows.join("\n")}\n    ],\n  },`,
    );
  }
}

writeFileSync(
  "src/lib/officialSkills.ts",
  `// Official skills, bundled as text (generated from resources/skills/ by
// scripts/bundle-skills.mts — edit the markdown/scripts there, not this file).
//
// They ship enabled; a user skill with the same name shadows one, and any of
// them can be turned off in Settings. Content is procedural knowledge distilled
// from Chaty's own development: the practices that repeatedly separated a fix
// that held from one that had to be redone.

export const OFFICIAL_SKILL_FILES: { file: string; text: string }[] = [
${skillRows.join("\n")}
];

/** Support files for directory-shaped skills (scripts, references) — written
 *  to <workspace>/.chaty/skills/<name>/ when the skill is first used, keyed by
 *  a content hash so unchanged bundles skip the writes. Never model context. */
export const OFFICIAL_SKILL_SUPPORT: Record<string, { rev: string; files: { path: string; text: string }[] }> = {
${supportRows.join("\n")}
};
`,
);
console.log(`bundled ${skillRows.length} skills, ${supportRows.length} with support files`);
