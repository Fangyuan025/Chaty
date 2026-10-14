import { describe, expect, it } from "vitest";
import { webSourceBudget, webSources } from "./webSources";
import type { WebResearch } from "./ipc";

const research = (n: number, pageChars = 900): WebResearch => ({
  results: Array.from({ length: n }, (_, i) => ({
    title: `Result ${i + 1}`,
    url: `https://site${i + 1}.example/page`,
    snippet: `Snippet ${i + 1} about the topic, a sentence or two long.`,
  })),
  // Every other page could be read.
  pages: Array.from({ length: n }, (_, i) => i)
    .filter((i) => i % 2 === 0)
    .map((i) => ({ title: `Result ${i + 1}`, url: `https://site${i + 1}.example/page`, text: "word ".repeat(pageChars / 5) })),
});

describe("webSources", () => {
  it("takes every result when there is room — no cap of six", () => {
    const { blocks, sources } = webSources(research(14), webSourceBudget(32768), 1);
    expect(sources).toHaveLength(14);
    expect(blocks[0].startsWith("【1】 Result 1\n")).toBe(true);
    expect(blocks[13].startsWith("【14】 Result 14\n")).toBe(true);
  });

  it("keeps the engine's order, page text where read and snippet where not", () => {
    const { sources } = webSources(research(4), 100000, 1);
    expect(sources.map((s) => s.title)).toEqual(["Result 1", "Result 2", "Result 3", "Result 4"]);
    expect(sources[0].snippet.startsWith("word")).toBe(true);
    expect(sources[1].snippet.startsWith("Snippet 2")).toBe(true);
  });

  it("stops at the budget, a long page giving way to its snippet", () => {
    const small = webSources(research(14, 4000), 2500, 1);
    const all = small.sources.length;
    expect(all).toBeGreaterThan(0);
    expect(all).toBeLessThanOrEqual(14);
    // A page that did not fit came in as its snippet rather than not at all.
    expect(small.sources.some((s, i) => i % 2 === 0 && s.snippet.startsWith("Snippet"))).toBe(true);
  });

  it("numbers on from the sources already there and skips their URLs", () => {
    const r = research(3);
    const { blocks, sources } = webSources(r, 100000, 3, new Set([r.results[0].url]));
    expect(sources.map((s) => s.title)).toEqual(["Result 2", "Result 3"]);
    expect(blocks[0].startsWith("【3】 Result 2")).toBe(true);
  });

  it("budgets a quarter of the window, at least a few pages", () => {
    expect(webSourceBudget(32768)).toBe(8192);
    expect(webSourceBudget(4096)).toBe(2500);
    expect(webSourceBudget(undefined)).toBe(2500);
  });
});
