/** The agent's own past, searchable: what this session said before compaction
 *  dropped it, and the sessions the user pointed at with @. Driven through the
 *  REAL loop, against the calls it actually makes to the store. */
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

const call = (args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name: "search_history", arguments: args })}</tool_call>`;

const read = (args: Record<string, unknown>) =>
  `<tool_call>${JSON.stringify({ name: "read_history", arguments: args })}</tool_call>`;

const hit = (
  sessionId: string,
  title: string,
  turn: number,
  role: string,
  text: string,
  step?: { stepId: string; status: string },
) => ({
  sessionId, title, updatedAt: Date.UTC(2026, 8, 16, 9, 30), turn, role, text,
  stepId: step?.stepId ?? null, status: step?.status ?? null,
});

/** A session as code_session_read returns one. */
const READ = {
  sessionId: "sess-old",
  title: "Tooltip work",
  updatedAt: Date.UTC(2026, 8, 16, 9, 30),
  totalTurns: 4,
  turns: [
    { turn: 3, role: "assistant", text: "改好了。", steps: [
      { stepId: "s7", name: "edit_file", args: "path=src/Tip.tsx", status: "done", resultChars: 120 },
      { stepId: "s8", name: "bash", args: "command=npm test", status: "error", resultChars: 4300 },
    ] },
  ],
};

/** Runs one turn; returns what the store was asked and what came back. */
async function run(rounds: string[], sessionId: string | undefined, hits: (q: Record<string, unknown>) => unknown[]) {
  const script = [...rounds];
  const asked: Record<string, unknown>[] = [];
  const prompts: string[] = [];
  mockIPC(async (cmd, args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "generate") {
      const req = a.request as { messages?: { role: string; content: string }[] };
      prompts.push(req?.messages?.[0]?.content ?? "");
      const ch = a.onEvent as Chan;
      ch.onmessage?.({ type: "token", text: script.shift() ?? "Done." });
      ch.onmessage?.({ type: "done", stats: { completionTokens: 8, tokensPerSecond: 50, promptTokens: 100 } });
      return null;
    }
    if (cmd === "code_session_search") {
      asked.push(a);
      return hits(a);
    }
    if (cmd === "code_session_list") {
      return [
        { id: "sess-now", title: "本会话", workspace: null, updatedAt: 0 },
        { id: "sess-old", title: "Tooltip work", workspace: null, updatedAt: 0 },
      ];
    }
    if (cmd === "code_session_read") {
      asked.push(a);
      return READ;
    }
    if (cmd === "code_step_text_get") {
      asked.push(a);
      return a.stepId === "s8" ? "FAIL tests/tip.test.ts:12 expected 300 got 0" : null;
    }
    return null;
  });
  const results: string[] = [];
  await runAgentTurn(
    "what did we decide about the tooltip?",
    [],
    "/tmp/ws",
    "en",
    {
      thinkMode: "off", maxSteps: 6, sessionId,
      signal: { cancelled: false },
      approve: async () => true,
      approveDir: async () => false,
      approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {},
      onStep: (s: { status: string; result?: string }) => {
        if (s.status !== "running" && s.result) results.push(s.result);
      },
      onStats: () => {}, onPlan: () => {}, onCompacted: () => {},
      onError: (e: string) => { throw new Error(e); },
      onStepText: () => {}, onLiveStep: () => {}, onLiveStepGone: () => {},
      onFinal: () => {},
    } as never,
  );
  return { asked, results, prompt: prompts[0] ?? "" };
}

describe("a session can read its own past", () => {
  it("searches the session it belongs to when the model names none", async () => {
    const { asked, results, prompt } = await run(
      [call({ query: "tooltip delay" }), "300ms, as you asked earlier."],
      "sess-now",
      () => [hit("sess-now", "Tooltip work", 4, "user", "make the tooltip delay 300ms")],
    );
    expect(prompt).toContain("search_history");
    // This session first, then every session — "where was this discussed" is
    // half of what gets asked, and one scope cannot answer it.
    expect(asked[0]).toEqual({ query: "tooltip delay", sessionId: "sess-now", limit: 8 });
    expect(asked[1]).toEqual({ query: "tooltip delay", sessionId: undefined, limit: 12 });
    expect(results[0]).toContain("this session");
    expect(results[0]).toContain("message 4");
    expect(results[0]).toContain("make the tooltip delay 300ms");
  });

  it("looks in the other sessions when this one has nothing, and says so", async () => {
    const { asked, results } = await run(
      [call({ query: "tooltip delay" }), "Found it in the other session."],
      "sess-now",
      (a) => (a.sessionId ? [] : [hit("sess-old", "Tooltip work", 2, "assistant", "settled on 300ms")]),
    );
    expect(asked.map((a) => a.sessionId)).toEqual(["sess-now", undefined]);
    expect(results[0]).toContain("nothing in this session");
    expect(results[0]).toContain('session "Tooltip work"');
    expect(results[0]).toContain("settled on 300ms");
  });

  it("takes all the sessions, or one the user pointed at", async () => {
    const all = await run([call({ query: "x", session: "all" }), "ok"], "sess-now", () => [
      hit("sess-old", "Old", 1, "user", "x marks it"),
    ]);
    expect(all.asked).toEqual([{ query: "x", sessionId: undefined, limit: 8 }]);
    const one = await run([call({ query: "x", session: "sess-old" }), "ok"], "sess-now", () => [
      hit("sess-old", "Old", 1, "user", "x marks it"),
    ]);
    expect(one.asked[0]).toEqual({ query: "x", sessionId: "sess-old", limit: 8 });
  });

  /// A search inside one session that barely caught gets what the session was
  /// about with it — one line answers nothing, and a small model then goes
  /// hunting the filesystem instead.
  it("says what a session was about when the words barely caught", async () => {
    const { asked, results } = await run(
      [call({ query: "x", session: "sess-old" }), "ok"],
      "sess-now",
      (a) => (a.query ? [hit("sess-old", "Old", 1, "user", "x marks it")] : [hit("sess-old", "Old", 9, "assistant", "we settled on two retries")]),
    );
    expect(asked.map((a) => a.query)).toEqual(["x", ""]);
    expect(results[0]).toContain("x marks it");
    expect(results[0]).toContain("we settled on two retries");
  });

  it("asks for words when given neither words nor a session", async () => {
    const { asked, results } = await run([call({}), "ok"], "sess-now", () => []);
    expect(asked).toEqual([]);
    expect(results[0]).toContain("query");
  });

  it("hands back the step a hit came from, and whether it worked", async () => {
    const { results } = await run(
      [call({ query: "npm test" }), "Found it."],
      "sess-now",
      () => [hit("sess-old", "Tooltip work", 3, "bash", "bash command=npm test [失败 error failed]", { stepId: "s8", status: "error" })],
    );
    expect(results[0]).toContain("✗");
    expect(results[0]).toContain("step=s8");
    // …and it says how to read that step, and the turn it sat in.
    expect(results[0]).toContain("read_history");
  });

  it("reads a turn back with every tool call of it, marked", async () => {
    const { asked, results } = await run([read({ session: "sess-old", turn: 3 }), "ok"], "sess-now", () => []);
    expect(asked[0]).toMatchObject({ sessionId: "sess-old", turn: 3 });
    expect(results[0]).toContain("Tooltip work");
    expect(results[0]).toContain("✓ edit_file path=src/Tip.tsx");
    expect(results[0]).toContain("✗ bash command=npm test");
    expect(results[0]).toContain("step=s8");
    // The results themselves are NOT in it — that is the next call.
    expect(results[0]).not.toContain("FAIL tests/tip.test.ts");
  });

  it("reads one step's whole result when the step comes back", async () => {
    const { asked, results } = await run([read({ session: "sess-old", step: "s8" }), "ok"], "sess-now", () => []);
    expect(asked[0]).toMatchObject({ sessionId: "sess-old", stepId: "s8" });
    expect(results[0]).toBe("FAIL tests/tip.test.ts:12 expected 300 got 0");
  });

  it("reads its OWN session when none is named", async () => {
    const { asked } = await run([read({ turn: 2 }), "ok"], "sess-now", () => []);
    expect(asked[0]).toMatchObject({ sessionId: "sess-now", turn: 2 });
  });

  it("says so when a step has no record kept", async () => {
    const { results } = await run([read({ step: "gone" }), "ok"], "sess-now", () => []);
    expect(results[0]).toContain("gone");
    expect(results[0]).toContain("read_history");
  });

  /// A model writes what it has in front of it — often the title from a
  /// search hit rather than the id.
  it("takes a session by its title as well as its id", async () => {
    const { asked } = await run([read({ session: "Tooltip work", turn: 3 }), "ok"], "sess-now", () => []);
    expect(asked[0]).toMatchObject({ sessionId: "sess-old", turn: 3 });
  });

  it("names the sessions there are when the one asked for is not one", async () => {
    const { asked, results } = await run([read({ session: "没这个会话" }), "ok"], "sess-now", () => []);
    expect(asked).toEqual([]); // nothing read
    expect(results[0]).toContain("Tooltip work");
    expect(results[0]).toContain("session=sess-old");
  });

  it("is not there at all for a caller that keeps no sessions", async () => {
    const { prompt } = await run(["Done."], undefined, () => []);
    expect(prompt).not.toContain("search_history");
  });
});
