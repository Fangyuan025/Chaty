import { describe, expect, test } from "vitest";

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
  loadMemoryIndex,
  MEMORY_DIR,
  MEMORY_INDEX,
  memoryIndexDoc,
  rememberFact,
  slugify,
} = await import("./memoryFiles");
const { systemPrompt } = await import("./agentLoop");
const { setMemoryToolEnabled, toolSpec, needsApproval } = await import("./toolRegistry");

function fakeFs() {
  const files = new Map<string, string>();
  return {
    files,
    readFile: async (p: string) => {
      const v = files.get(p);
      if (v === undefined) throw new Error("missing");
      return v;
    },
    writeFile: async (p: string, c: string) => void files.set(p, c),
  };
}

describe("prompt economics (the M0 golden contract holds)", () => {
  test("no memory ⇒ byte-identical prompt", () => {
    expect(memoryIndexDoc("", "en")).toBe("");
    expect(systemPrompt("/ws", false, "normal", undefined, false, false, [], "")).toBe(
      systemPrompt("/ws", false, "normal", undefined, false, false),
    );
  });

  test("index rides capped and never cut mid-line", () => {
    const index = Array.from({ length: 80 }, (_, i) => `- [fact ${i}](f${i}.md) — hook ${i}`).join("\n");
    const doc = memoryIndexDoc(index, "en");
    expect(doc.length).toBeLessThan(2600);
    expect(doc.endsWith(")") || /hook \d+$/.test(doc)).toBe(true); // whole lines only
    expect(doc).toContain("- [fact 0]"); // newest-first head survives
    expect(doc).not.toContain("- [fact 79]");
  });

  test("memory on ⇒ index + write nudge appear", () => {
    const p = systemPrompt("/ws", false, "normal", undefined, false, false, [], "- [a](a.md) — x");
    expect(p).toContain("Project memory");
    expect(p).toContain("- [a](a.md) — x");
    expect(p).toContain("remember() the NON-OBVIOUS findings");
  });
});

describe("rememberFact", () => {
  test("writes the fact file and upserts the index, newest first", async () => {
    const fs = fakeFs();
    const out1 = await rememberFact(fs, "Build rule", "run gate.sh first", "en");
    expect(out1).toContain("Remembered: Build rule");
    expect(fs.files.get(`${MEMORY_DIR}/build-rule.md`)).toContain("run gate.sh first");
    await rememberFact(fs, "Ports", "dev server owns 1420", "en");
    const idx = fs.files.get(MEMORY_INDEX)!;
    expect(idx.indexOf("[Ports]")).toBeLessThan(idx.indexOf("[Build rule]"));
    // The link must be a path read_file can resolve from the workspace root —
    // the bare filename made the model's read silently miss (M4 crash).
    expect(idx).toContain(`](${MEMORY_DIR}/build-rule.md)`);
    expect(idx).not.toMatch(/\]\(build-rule\.md\)/);
    // Same title again ⇒ update in place, no duplicate line.
    await rememberFact(fs, "Build rule", "gate.sh THEN tag", "en");
    const idx2 = fs.files.get(MEMORY_INDEX)!;
    expect(idx2.match(/build-rule\.md/g)?.length).toBe(1);
    expect(fs.files.get(`${MEMORY_DIR}/build-rule.md`)).toContain("THEN tag");
  });

  test("missing args come back as a correction, not a write", async () => {
    const fs = fakeFs();
    const out = await rememberFact(fs, "", "x", "en");
    expect(out).toContain("ERROR");
    expect(fs.files.size).toBe(0);
  });

  test("slugs stay path-safe", () => {
    expect(slugify("Build rule: gate.sh FIRST!")).toBe("build-rule-gate-sh-first");
    expect(slugify("///")).toBe("note");
  });
});

describe("registry contract", () => {
  test("remember registers only when enabled; approval-free by confinement", () => {
    setMemoryToolEnabled(false);
    expect(toolSpec("remember")).toBeUndefined();
    setMemoryToolEnabled(true);
    try {
      expect(toolSpec("remember")?.source).toBe("memory");
      expect(needsApproval("remember")).toBe(false);
    } finally {
      setMemoryToolEnabled(false);
    }
  });

  test("loadMemoryIndex strips hashline anchors and tolerates absence", async () => {
    const fs = fakeFs();
    expect(await loadMemoryIndex(fs)).toBe("");
    fs.files.set(MEMORY_INDEX, "1:abc→- [a](a.md) — x\n2:def→- [b](b.md) — y\n");
    expect(await loadMemoryIndex(fs)).toBe("- [a](a.md) — x\n- [b](b.md) — y\n");
  });
});
