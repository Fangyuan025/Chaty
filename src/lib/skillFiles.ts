// File-based skills (2.0 M3): procedural knowledge as SKILL.md files.
//
// The economics that make this the highest-leverage feature for a small model:
// a ~3B-active model can't hold a project's procedures in its weights, but a
// page of markdown costs almost nothing — and only when it's needed. So the
// system prompt carries ONE line per skill (name + when-to-use), and the body
// loads only after the model picks it. Same budget philosophy as the deferred
// tool tier, applied to knowledge instead of tools.
//
// Format (frontmatter + markdown body):
//
//   ---
//   name: release
//   description: Cut a release — version bump, gates, tag, assets
//   when: the user asks to ship, release, or publish a version
//   ---
//   1. Run scripts/bump-version.sh …
//
// Discovery: <workspace>/.chaty/skills/*.md (project) then ~/.chaty/skills/*.md
// (global). A project skill shadows a global one with the same name — the
// project always knows better than the default.

export interface SkillFile {
  name: string;
  description: string;
  /** Trigger phrase for the index line; falls back to description. */
  when?: string;
  body: string;
  /** Where it came from — project skills shadow global ones. */
  scope: "project" | "global";
  path: string;
}

const MAX_BODY = 8000;

/** Parse a SKILL.md. Returns null when it has no usable frontmatter name —
 *  a malformed file is skipped, never half-loaded. */
export function parseSkill(text: string, path: string, scope: SkillFile["scope"]): SkillFile | null {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(text.trim());
  if (!m) return null;
  const meta: Record<string, string> = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = /^([a-zA-Z_][\w-]*)\s*:\s*(.*)$/.exec(line.trim());
    if (kv) meta[kv[1].toLowerCase()] = kv[2].trim().replace(/^["']|["']$/g, "");
  }
  const name = (meta.name ?? "").trim();
  if (!name || !/^[a-zA-Z0-9_-]{1,32}$/.test(name)) return null;
  const body = m[2].trim();
  if (!body) return null;
  return {
    name,
    description: meta.description ?? "",
    when: meta.when || undefined,
    body: body.slice(0, MAX_BODY),
    scope,
    path,
  };
}

/** Project skills shadow global ones by name; each list keeps file order. */
export function mergeSkills(project: SkillFile[], global: SkillFile[]): SkillFile[] {
  const byName = new Map<string, SkillFile>();
  for (const s of global) byName.set(s.name, s);
  for (const s of project) byName.set(s.name, s); // project wins
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

/** The index that rides in every system prompt: one line per skill, no body.
 *  Returns "" when there are no skills — the prompt must stay byte-identical
 *  for users who have none (locked by the golden test). */
export function skillIndex(skills: SkillFile[], lang: "zh" | "en"): string {
  if (!skills.length) return "";
  const head =
    lang === "zh"
      ? `\n\n可用技能(按需调用 use_skill 获取完整步骤,不要凭猜测执行):`
      : `\n\nAvailable skills (call use_skill to load the full steps — don't improvise them):`;
  const lines = skills.map((s) => `\n- ${s.name}: ${s.when || s.description}`);
  return head + lines.join("");
}

/** The loaded body, framed so the model treats it as procedure to follow. */
export function skillBody(skill: SkillFile, lang: "zh" | "en"): string {
  const head =
    lang === "zh"
      ? `技能「${skill.name}」的完整步骤(来自 ${skill.path},请照此执行):`
      : `Skill "${skill.name}" — full procedure (from ${skill.path}; follow it):`;
  return `${head}\n\n${skill.body}`;
}

// ── Discovery ────────────────────────────────────────────────────────────────

import { OFFICIAL_SKILL_FILES } from "./officialSkills";

const DISABLED_KEY = "chaty.skillsDisabled";

export function disabledSkills(): string[] {
  try {
    const raw = localStorage.getItem(DISABLED_KEY);
    const arr = raw ? (JSON.parse(raw) as string[]) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

export function setDisabledSkills(names: string[]): void {
  localStorage.setItem(DISABLED_KEY, JSON.stringify(names));
}

/** Skills that ship with the app. Parsed at call time so a malformed bundle
 *  degrades to "no official skills" instead of breaking the turn. */
export function officialSkills(): SkillFile[] {
  const out: SkillFile[] = [];
  for (const f of OFFICIAL_SKILL_FILES) {
    const s = parseSkill(f.text, `official:${f.file}`, "global");
    if (s) out.push(s);
  }
  return out;
}

/**
 * Load every skill available to this turn: official (unless disabled), then
 * global user skills (~/.chaty/skills/*.md), then project skills
 * (<workspace>/.chaty/skills/*.md) — later sources shadow earlier ones by name.
 *
 * `readFile`/`glob` are injected so this works headless (bench) and in tests.
 */
export async function loadSkills(
  glob: (pattern: string) => Promise<string[]>,
  readFile: (path: string) => Promise<string>,
  homeDir?: string,
): Promise<SkillFile[]> {
  const disabled = new Set(disabledSkills());
  const read = async (paths: string[], scope: SkillFile["scope"]) => {
    const out: SkillFile[] = [];
    for (const p of paths.slice(0, 24)) {
      try {
        // read_file may be in hashline-anchor mode — strip prefixes so skill
        // text reaches the model clean (same guard as the project doc).
        const text = (await readFile(p)).replace(/^\d+:[a-z]{2,4}→/gm, "");
        const parsed = parseSkill(text, p, scope);
        if (parsed) out.push(parsed);
      } catch {
        /* unreadable file — skip */
      }
    }
    return out;
  };

  let globals: SkillFile[] = [];
  if (homeDir) {
    try {
      globals = await read(await glob(`${homeDir}/.chaty/skills/*.md`), "global");
    } catch {
      /* no global skills dir */
    }
  }
  let project: SkillFile[] = [];
  try {
    project = await read(await glob(".chaty/skills/*.md"), "project");
  } catch {
    /* no project skills dir */
  }
  const userSkills = mergeSkills(project, globals);
  return mergeSkills(userSkills, officialSkills()).filter((s) => !disabled.has(s.name));
}
