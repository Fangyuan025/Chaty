import { describe, expect, test } from "vitest";

(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { evictStaleImages } = await import("./agentLoop");
type ChatMessage = Parameters<typeof evictStaleImages>[0][number];

function shots(n: number): ChatMessage[] {
  const msgs: ChatMessage[] = [{ role: "system", content: "sys", images: [] }];
  for (let i = 0; i < n; i++) {
    msgs.push({ role: "user", content: `screenshot ${i}`, images: [`/tmp/${i}.png`] });
    msgs.push({ role: "assistant", content: `looked at ${i}`, images: [] });
  }
  return msgs;
}
const live = (m: ChatMessage[]) => m.reduce((n, x) => n + (x.images?.length ?? 0), 0);

describe("stale screenshots are dropped only when dropping them pays", () => {
  test("an engine that reuses across a new image keeps them all", () => {
    // Measured: evicting cost Gemma-4 685ms → 1422ms and Qwen3.5 2.9s → 5.7s,
    // because rewriting a cached message kills the prefix for no encode saved.
    const msgs = shots(4);
    evictStaleImages(msgs, false);
    expect(live(msgs)).toBe(4);
    expect(msgs.every((m) => !m.content.includes("stale"))).toBe(true);
  });

  test("an engine that re-encodes everything keeps only the newest one", () => {
    // A second live screenshot is re-encoded on every screenshot round for as
    // long as it stays: 1799/3939/3988ms holding two against 1798/1821/1871ms
    // holding one.
    const msgs = shots(4);
    evictStaleImages(msgs, true);
    expect(live(msgs)).toBe(1);
  });

  test("it is the OLDEST that go", () => {
    const msgs = shots(4);
    evictStaleImages(msgs, true);
    const kept = msgs.filter((m) => m.images?.length).map((m) => m.images![0]);
    expect(kept).toEqual(["/tmp/3.png"]);
  });

  test("a dropped screenshot says so, once", () => {
    const msgs = shots(4);
    evictStaleImages(msgs, true);
    evictStaleImages(msgs, true);
    const dropped = msgs.find((m) => m.content.startsWith("screenshot 0"))!;
    expect(dropped.content).toMatch(/stale|截图已过期/);
    expect(dropped.content.match(/stale|截图已过期/g)).toHaveLength(1);
  });

  test("nothing is rewritten when there is nothing to drop", () => {
    const msgs = shots(1);
    const before = JSON.stringify(msgs);
    evictStaleImages(msgs, true);
    expect(JSON.stringify(msgs)).toBe(before);
  });

  test("a transcript with no screenshots at all is untouched", () => {
    const msgs: ChatMessage[] = [{ role: "user", content: "hi", images: [] }];
    const before = JSON.stringify(msgs);
    evictStaleImages(msgs, true);
    expect(JSON.stringify(msgs)).toBe(before);
  });
});
