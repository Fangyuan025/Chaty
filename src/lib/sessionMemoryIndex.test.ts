import { describe, expect, test } from "vitest";
import { sessionMemoryIndex } from "./memoryFiles";

/** The memory index rides in the system prompt, so a session must keep the one
 *  it started with: re-reading it after every `remember` rewrote the front of
 *  the prompt and cost the whole conversation's prefill on every turn. */
describe("a session keeps the memory index it started with", () => {
  test("later turns get the first turn's index, even after the file changed", async () => {
    const pins = new Map<string, string>();
    let onDisk = "- [a](.chaty/memory/a.md) — first";
    const load = async () => onDisk;
    expect(await sessionMemoryIndex(pins, "s1", load)).toBe(onDisk);
    const first = onDisk;
    onDisk += "\n- [b](.chaty/memory/b.md) — written by the model this turn";
    expect(await sessionMemoryIndex(pins, "s1", load)).toBe(first);
  });

  test("a new session reads the index as it is now", async () => {
    const pins = new Map<string, string>();
    let onDisk = "- [a](.chaty/memory/a.md) — first";
    const load = async () => onDisk;
    await sessionMemoryIndex(pins, "s1", load);
    onDisk += "\n- [b](.chaty/memory/b.md) — later";
    expect(await sessionMemoryIndex(pins, "s2", load)).toBe(onDisk);
  });

  test("an empty index is kept too — a first remember must not change the prompt", async () => {
    const pins = new Map<string, string>();
    let onDisk = "";
    const load = async () => onDisk;
    expect(await sessionMemoryIndex(pins, "s1", load)).toBe("");
    onDisk = "- [a](.chaty/memory/a.md) — the first fact ever";
    expect(await sessionMemoryIndex(pins, "s1", load)).toBe("");
  });
});
