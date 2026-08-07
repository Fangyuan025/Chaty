import { beforeEach, describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const store = new Map<string, string>();
(globalThis as Record<string, unknown>).localStorage = {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => void store.set(k, v),
  removeItem: (k: string) => void store.delete(k),
};
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));

const {
  loadSkills,
  mergeSkills,
  officialSkills,
  parseSkill,
  setDisabledSkills,
  skillBody,
  skillIndex,
} = await import("./skillFiles");
const { systemPrompt } = await import("./agentLoop");

const SKILL = `---
name: release
description: Cut a release
when: the user asks to ship a version
---
1. Bump the version.
2. Run the gates.`;

beforeEach(() => store.clear());

describe("parsing", () => {
  test("frontmatter + body", () => {
    const s = parseSkill(SKILL, ".chaty/skills/release.md", "project")!;
    expect(s.name).toBe("release");
    expect(s.when).toBe("the user asks to ship a version");
    expect(s.body).toContain("Bump the version");
    expect(s.body).not.toContain("---");
  });

  test("malformed files are skipped, never half-loaded", () => {
    expect(parseSkill("no frontmatter at all", "x.md", "project")).toBeNull();
    expect(parseSkill("---\ndescription: x\n---\nbody", "x.md", "project")).toBeNull(); // no name
    expect(parseSkill("---\nname: ok\n---\n", "x.md", "project")).toBeNull(); // no body
    expect(parseSkill("---\nname: bad name!\n---\nbody", "x.md", "project")).toBeNull();
  });
});

describe("prompt economics", () => {
  test("index carries one line per skill — never the body", () => {
    const s = parseSkill(SKILL, "p.md", "project")!;
    const idx = skillIndex([s], "en");
    expect(idx).toContain("- release: the user asks to ship a version");
    expect(idx).not.toContain("Bump the version");
    expect(idx.split("\n").filter((l) => l.startsWith("- ")).length).toBe(1);
  });

  test("no skills ⇒ empty index (prompt stays byte-identical)", () => {
    expect(skillIndex([], "en")).toBe("");
    expect(systemPrompt("/ws", false, "normal", undefined, false, false, [])).toBe(
      systemPrompt("/ws", false, "normal", undefined, false, false),
    );
  });

  test("skills ⇒ index appended, body still absent", () => {
    const s = parseSkill(SKILL, "p.md", "project")!;
    const withSkills = systemPrompt("/ws", false, "normal", undefined, false, false, [s]);
    expect(withSkills).toContain("Available skills");
    expect(withSkills).toContain("- release:");
    expect(withSkills).not.toContain("Bump the version");
  });

  test("body loads framed as procedure, with provenance", () => {
    const s = parseSkill(SKILL, ".chaty/skills/release.md", "project")!;
    const body = skillBody(s, "en");
    expect(body).toContain(".chaty/skills/release.md");
    expect(body).toContain("Bump the version");
  });
});

describe("precedence & discovery", () => {
  test("project shadows global by name", () => {
    const g = parseSkill(SKILL.replace("Bump the version", "GLOBAL"), "g.md", "global")!;
    const p = parseSkill(SKILL.replace("Bump the version", "PROJECT"), "p.md", "project")!;
    const merged = mergeSkills([p], [g]);
    expect(merged.length).toBe(1);
    expect(merged[0].body).toContain("PROJECT");
  });

  test("official skills ship parseable and non-empty", () => {
    const off = officialSkills();
    expect(off.length).toBeGreaterThanOrEqual(3);
    for (const s of off) {
      expect(s.when || s.description).toBeTruthy();
      expect(s.body.length).toBeGreaterThan(100);
    }
  });

  test("directory-shaped skill: tiktok-video ships prompt + runnable support", async () => {
    const { officialSkillSupport, skillRoot } = await import("./skillFiles");
    const tv = officialSkills().find((s) => s.name === "tiktok-video")!;
    expect(tv).toBeTruthy();
    expect(tv.path).toBe("official:tiktok-video/SKILL.md");
    // The body is the creative procedure — it references scripts via the
    // placeholder and NEVER inlines script code into model context.
    expect(tv.body).toContain("{SKILL_ROOT}");
    expect(tv.body.length).toBeLessThanOrEqual(8000);
    expect(tv.body).not.toContain("def main(");

    const support = officialSkillSupport("tiktok-video")!;
    expect(support.rev).toMatch(/^[0-9a-f]{16}$/);
    const paths = support.files.map((f) => f.path);
    for (const must of ["scripts/pipeline.py", "scripts/setup.sh", "scripts/compose.py", "references/writing-guide.md"]) {
      expect(paths).toContain(must);
    }
    // Knowledge-only skills have no support files.
    expect(officialSkillSupport("mac-app")).toBeNull();
    expect(skillRoot("tiktok-video")).toBe(".chaty/skills/tiktok-video");
  });

  test("user skills shadow official ones; disabled ones drop out", async () => {
    const files: Record<string, string> = {
      ".chaty/skills/verify-before-push.md": SKILL.replace("name: release", "name: verify-before-push").replace("Bump the version", "MY OWN STEPS"),
      ".chaty/skills/broken.md": "not a skill",
    };
    const glob = async (pattern: string) =>
      pattern.startsWith(".chaty") ? Object.keys(files) : [];
    const readFile = async (p: string) => files[p] ?? Promise.reject(new Error("nope"));

    const loaded = await loadSkills(glob, readFile);
    const mine = loaded.find((s) => s.name === "verify-before-push")!;
    expect(mine.body).toContain("MY OWN STEPS"); // user version wins
    expect(loaded.some((s) => s.name === "broken")).toBe(false); // malformed skipped
    expect(loaded.some((s) => s.name === "debug-by-mechanism")).toBe(true); // official still there

    setDisabledSkills(["debug-by-mechanism"]);
    const after = await loadSkills(glob, readFile);
    expect(after.some((s) => s.name === "debug-by-mechanism")).toBe(false);
  });

  test("hashline anchors are stripped from skill text", async () => {
    const anchored = "---\nname: anchored\ndescription: d\n---\n1:abc→step one\n2:def→step two";
    const loaded = await loadSkills(
      async (p) => (p.startsWith(".chaty") ? ["s.md"] : []),
      async () => anchored,
    );
    const s = loaded.find((x) => x.name === "anchored")!;
    expect(s.body).toContain("step one");
    expect(s.body).not.toContain("1:abc→");
  });
});
