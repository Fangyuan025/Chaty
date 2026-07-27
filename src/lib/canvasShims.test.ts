import { describe, expect, it } from "vitest";
import { PREVIEW_SHIMS } from "../components/CanvasPanel";

/** The generated shim code must PARSE. A template-escape slip ('\n' in the
 *  TS source becomes a real newline inside the shim's single-quoted string)
 *  produced a SyntaxError that silently killed the console and error shims —
 *  the "canvas console shows nothing" incident. new Function() is a pure
 *  syntax gate: it compiles without executing. */
describe("canvas preview shims", () => {
  for (const [name, tag] of Object.entries(PREVIEW_SHIMS)) {
    it(`${name} generates syntactically valid JS`, () => {
      const bodies = [...tag.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
      expect(bodies.length).toBeGreaterThan(0);
      for (const body of bodies) {
        expect(() => new Function(body)).not.toThrow();
      }
    });
  }
});
