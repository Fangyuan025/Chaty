import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { looksDegenerate } = await import("./agentLoop");

/** The guard has to catch a stream that has stopped saying anything without
 *  ever cutting one that is merely repetitive — a false positive here kills a
 *  working step, which is worse than the thing being guarded against. */
describe("output that has stopped carrying information", () => {
  test("one character to the token cap", () => {
    expect(looksDegenerate("Here we go: " + "!".repeat(2000))).toBe(true);
    expect(looksDegenerate("x".repeat(500))).toBe(true);
  });

  test("a short cycle counts too, once it has gone on long enough", () => {
    expect(looksDegenerate("ab".repeat(500))).toBe(true);
    expect(looksDegenerate("...".repeat(400))).toBe(true);
    // Under the longer bar a cycle is left alone — it might still be prose.
    expect(looksDegenerate("ab".repeat(300))).toBe(false);
  });

  test("nothing shorter than the window is ever cut", () => {
    expect(looksDegenerate("!".repeat(399))).toBe(false);
    expect(looksDegenerate("")).toBe(false);
  });
});

describe("things a model writes on purpose, which must survive", () => {
  test("a markdown rule and a table separator", () => {
    expect(looksDegenerate("intro\n\n" + "-".repeat(80) + "\n\nmore prose here")).toBe(false);
    expect(looksDegenerate("| a | b |\n|" + "-".repeat(60) + "|\n| 1 | 2 |")).toBe(false);
  });

  test("a banner comment", () => {
    expect(looksDegenerate("// " + "=".repeat(76) + "\n// Section\n// " + "=".repeat(76))).toBe(false);
  });

  test("base64 and a hash — high entropy, long, legitimate", () => {
    const b64 = "iVBORw0KGgoAAAANSUhEUg".repeat(40);
    expect(looksDegenerate(b64)).toBe(false);
    // A real hash is not a repeating cycle; this is what one looks like.
    const hash = [...Array(64)].map((_, i) => "0123456789abcdef"[(i * 7 + 3) % 16]).join("");
    expect(looksDegenerate(hash.repeat(20))).toBe(false);
  });

  test("indented code — a lot of spaces, but not only spaces", () => {
    const code = Array.from({ length: 60 }, (_, i) => "    ".repeat(4) + `const v${i} = ${i};`).join("\n");
    expect(looksDegenerate(code)).toBe(false);
  });

  test("prose that repeats a word is not degenerate", () => {
    expect(looksDegenerate("very ".repeat(200))).toBe(false);
  });

  test("only the TAIL decides — a clean answer after a long rule survives", () => {
    expect(looksDegenerate("=".repeat(500) + "\n\nAnd here is the actual answer to your question.")).toBe(false);
  });
});
