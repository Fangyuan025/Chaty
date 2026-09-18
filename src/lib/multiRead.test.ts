/** Several files in one call, and what happens when one of them cannot be
 *  read: the others still come back, the failure is named, and the step is a
 *  failure. Driven through the REAL loop. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC } = await import("@tauri-apps/api/mocks");
const { runAgentTurn } = await import("./agentLoop");

type Ev = { type: string; [k: string]: unknown };
type Chan = { onmessage?: (ev: Ev) => void };
type Step = { name: string; status: string; result: string; fullText?: boolean; id: string };

const call = (args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name: "multi_read", arguments: args })}</tool_call>`;

const FILES: Record<string, string> = {
  "src/app.ts": "export const app = 1;",
  "src/db.ts": "export const db = 2;",
};

async function run(rounds: string[]) {
  const script = [...rounds];
  const asked: string[] = [];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "generate") {
      const ch = a.onEvent as Chan;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "agent_read_file") {
      const path = String(a.path);
      asked.push(path);
      const body = FILES[path];
      if (body === undefined) throw new Error(`no such file: ${path}`);
      return body;
    }
    return null;
  });
  const steps: Step[] = [];
  const kept = new Map<string, string>();
  await runAgentTurn(
    "read those files",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off", maxSteps: 4,
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s: { id: string; call: { name: string }; status: string; result?: string; fullText?: boolean }) => {
        if (s.status !== "running") {
          steps.push({ id: s.id, name: s.call.name, status: s.status, result: s.result ?? "", fullText: s.fullText });
        }
      },
      onStats: () => {}, onPlan: () => {}, onCompacted: () => {},
      onError: (e: string) => { throw new Error(e); },
      onFinal: () => {},
      onStepText: (id: string, text: string) => kept.set(id, text),
      onLiveStep: () => {}, onLiveStepGone: () => {},
    } as never,
  );
  return { asked, steps, kept };
}

describe("reading several files at once", () => {
  it("reads them all in one call", async () => {
    const { asked, steps } = await run([call({ paths: ["src/app.ts", "src/db.ts"] }), "Both read."]);
    expect(asked).toEqual(["src/app.ts", "src/db.ts"]);
    expect(steps).toHaveLength(1);
    expect(steps[0].status).toBe("done");
    expect(steps[0].result).toContain("===== src/app.ts =====");
    expect(steps[0].result).toContain("export const app = 1;");
    expect(steps[0].result).toContain("===== src/db.ts =====");
    expect(steps[0].result).toContain("export const db = 2;");
  });

  it("keeps what it could read when one file fails, and still fails", async () => {
    const { steps } = await run([call({ paths: ["src/app.ts", "src/gone.ts", "src/db.ts"] }), "Two of three."]);
    expect(steps[0].status).toBe("error");
    // Everything that could be read is there…
    expect(steps[0].result).toContain("export const app = 1;");
    expect(steps[0].result).toContain("export const db = 2;");
    // …and the one that could not is named, both in place and in the summary.
    expect(steps[0].result).toContain("no such file: src/gone.ts");
    expect(steps[0].result).toContain("1 of 3 files could not be read");
  });

  it("takes the list however the model writes it", async () => {
    const byFiles = await run([call({ files: ["src/app.ts"] }), "ok"]);
    expect(byFiles.asked).toEqual(["src/app.ts"]);
    const asOneString = await run([call({ paths: "src/app.ts, src/db.ts" }), "ok"]);
    expect(asOneString.asked).toEqual(["src/app.ts", "src/db.ts"]);
  });

  /// The card keeps a trimmed copy; opened, it must show what the model was
  /// actually given — all of every file, not the first 6,000 characters.
  it("hands the host the model's own copy when the card has to trim it", async () => {
    const big = (name: string) => `// ${name}\n${"x".repeat(5000)}\nexport const end_of_${name} = true;\n`;
    FILES["big/one.ts"] = big("one");
    FILES["big/two.ts"] = big("two");
    try {
      const { steps, kept } = await run([call({ paths: ["big/one.ts", "big/two.ts"] }), "Read both."]);
      const step = steps.find((s) => s.name === "multi_read")!;
      expect(step.result.length).toBeLessThan(8200); // the card's copy is trimmed
      // The host is handed the model's copy for that step (it marks the card
      // openable itself — the loop does not re-send the step).
      expect(kept.has(step.id)).toBe(true);
      const whole = kept.get(step.id) ?? "";
      expect(whole).toContain("export const end_of_one = true;");
      expect(whole).toContain("export const end_of_two = true;");
      expect(whole.length).toBeGreaterThan(10000);
    } finally {
      delete FILES["big/one.ts"];
      delete FILES["big/two.ts"];
    }
  });

  it("is corrected before it runs when given no paths at all", async () => {
    // Like every tool with required arguments: the empty call is sent back to
    // be written properly rather than executed and recorded.
    const { asked, steps } = await run([call({}), "ok"]);
    expect(asked).toEqual([]);
    expect(steps.filter((s) => s.name === "multi_read")).toHaveLength(0);
  });
});
