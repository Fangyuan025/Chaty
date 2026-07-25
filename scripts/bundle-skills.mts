/**
 * Regenerate src/lib/officialSkills.ts from resources/skills/*.md.
 * Run after editing any official skill:  npx tsx scripts/bundle-skills.mts
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const dir = "resources/skills";
const files = readdirSync(dir).filter((f) => f.endsWith(".md")).sort();
const rows = files.map((f) => `  { file: ${JSON.stringify(f)}, text: ${JSON.stringify(readFileSync(path.join(dir, f), "utf8"))} },`);
writeFileSync(
  "src/lib/officialSkills.ts",
  `// Official skills, bundled as text (generated from resources/skills/*.md by
// scripts/bundle-skills.mts — edit the markdown, not this file).
//
// They ship enabled; a user skill with the same name shadows one, and any of
// them can be turned off in Settings. Content is procedural knowledge distilled
// from Chaty's own development: the practices that repeatedly separated a fix
// that held from one that had to be redone.

export const OFFICIAL_SKILL_FILES: { file: string; text: string }[] = [
${rows.join("\n")}
];
`,
);
console.log(`bundled ${files.length} skills`);
