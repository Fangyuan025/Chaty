/** Generalization sweep: every tool that HAS required args must survive every
 *  known malformed shape with its arguments intact. The individual shapes are
 *  fixture-tested in parseToolCall.test.ts from real dumps; this sweep proves
 *  the repairs are tool-agnostic — no rule anchored on one lucky tool name. */
import { describe, expect, it } from "vitest";
import { parseToolCall } from "./agentLoop";
import { ARG_EXAMPLE, REQUIRED_ARGS } from "./toolRegistry";

/** Malformed emitters, each derived from a REAL dump shape (see the fixture
 *  file), parameterized over tool + args JSON. */
const SHAPES: Record<string, (name: string, args: string) => string> = {
  clean: (n, a) => `<tool_call>{"name":"${n}","arguments":${a}}</tool_call>`,
  nameEquals: (n, a) => `<tool_call>{"name="${n}","arguments":${a}}</tool_call>`,
  argumentsBracket: (n, a) => `<tool_call>{"name":"${n}","arguments>${a}}`,
  droppedArgsKey: (n, a) => `<tool_call>{"name":"${n}", ${a}}`,
  siblingObject: (n, a) => `<tool_call>{"name":"${n}"}\n${a}`,
  secondBlock: (n, a) => `<tool_call>{"name":"${n}"}</tool_call>\n<tool_call>${a}</tool_call>`,
  argsAfterTag: (n, a) => `<tool_call>{"name="${n}">\n${a}\n`,
  missingCloser: (n, a) => `<tool_call>{"name":"${n}","arguments":${a.slice(0, -1)}`,
  extraCloser: (n, a) => `<tool_call>{"name":"${n}","arguments":${a}}}`,
};

describe("malformed-shape sweep over every required-args tool", () => {
  const tools = Object.keys(REQUIRED_ARGS).filter((t) => ARG_EXAMPLE[t]);
  it("covers a meaningful tool set", () => {
    expect(tools.length).toBeGreaterThanOrEqual(15);
  });

  for (const tool of tools) {
    const example = ARG_EXAMPLE[tool];
    const required = REQUIRED_ARGS[tool];
    for (const [shape, emit] of Object.entries(SHAPES)) {
      it(`${tool} × ${shape}`, () => {
        const c = parseToolCall(emit(tool, example));
        expect(c, "call must parse").not.toBeNull();
        expect(c!.name).toBe(tool);
        // Every required entry (any alias) must have survived the repair.
        for (const k of required) {
          const ok = k.split("|").some((alt) => {
            const v = c!.args[alt];
            return v !== undefined && v !== "" && v !== null;
          });
          // Only assert when the example actually carries one of the aliases
          // (browser_type's example uses label+text; text is the required one).
          const inExample = k.split("|").some((alt) => example.includes(`"${alt}"`));
          if (inExample) expect(ok, `${tool}.${k} lost by ${shape}`).toBe(true);
        }
      });
    }
  }
});
