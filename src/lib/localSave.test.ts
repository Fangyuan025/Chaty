/** Remembering something is best-effort: storage can be full, switched off or
 *  unavailable, and an effect that throws over it takes the whole interface
 *  down to the error boundary. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { localSave } from "./localSave";

const store = new Map<string, string>();
const g = globalThis as Record<string, unknown>;
g.localStorage = {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => void store.set(k, v),
  removeItem: (k: string) => void store.delete(k),
  clear: () => store.clear(),
  key: () => null,
  length: 0,
};

afterEach(() => {
  vi.restoreAllMocks();
  store.clear();
});

describe("localSave", () => {
  it("writes and says it did", () => {
    expect(localSave("chaty.k", "v")).toBe(true);
    expect(store.get("chaty.k")).toBe("v");
  });

  it("survives a storage that refuses, and says it could not", () => {
    vi.spyOn(globalThis.localStorage, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(() => localSave("chaty.k", "v")).not.toThrow();
    expect(localSave("chaty.k", "v")).toBe(false);
    expect(warn).toHaveBeenCalled();
  });
});
