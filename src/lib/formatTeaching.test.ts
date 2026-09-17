/** A model taught its own tool-call format must see no JSON calls anywhere it
 *  looks: Gemma 4, taught its native format, turned to JSON for every call
 *  after the first because the tool docs, the post-orientation hint and the
 *  corrections all showed JSON — and because each of its own calls came back
 *  in its history followed by a JSON copy. */
import { afterEach, describe, expect, it } from "vitest";
import agentLoopSrc from "./agentLoop.ts?raw";
import jitHintsSrc from "./jitHints.ts?raw";
import mcpSrc from "./mcp.ts?raw";
import memoryFilesSrc from "./memoryFiles.ts?raw";
import skillFilesSrc from "./skillFiles.ts?raw";
import toolDocsSrc from "./toolDocs.ts?raw";
import toolRegistrySrc from "./toolRegistry.ts?raw";
import wrapupGateSrc from "./wrapupGate.ts?raw";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { CALL_CLOSERS, callExample, callTag, formatOf, plainArgs, renderCall, setCallFormat } = await import("./callFormat");
const { systemPrompt, describeInvalidCall, parseToolCall, runAgentTurn, stubWrittenBodies, xmlRunsOn } = await import("./agentLoop");
const { jitHintFor, missingArgLadder } = await import("./jitHints");
const { planEcho } = await import("./wrapupGate");
const { withoutToolCallSpans } = await import("./voiceText");
const { rememberFact } = await import("./memoryFiles");
const { fullDoc } = await import("./mcp");

type Fmt = "json" | "xml" | "gemma" | "lfm";
const NATIVE = ["xml", "gemma", "lfm"] as const;

// A JSON object with a quoted key — the shape these tests keep out.
const JSON_KEY = /\{\s*\\?"[a-z_]+\\?"\s*:/;

describe("a turn that teaches another format", () => {
  afterEach(() => setCallFormat("json"));

  it("writes tool docs as argument lists", () => {
    expect(plainArgs('- read_file: read a file. args: { "path": string, "offset"?: number(1-based), "limit"?: number }')).toBe(
      "- read_file: read a file. args: path: string, offset?: number(1-based), limit?: number",
    );
    expect(plainArgs("- list_dir: list a folder.")).toBe("- list_dir: list a folder.");
    for (const f of NATIVE) {
      for (const zh of [true, false]) {
        const p = systemPrompt("/ws", zh, "deep", undefined, true, false, undefined, undefined, f);
        const bad = p.split("\n").filter((l) => JSON_KEY.test(l) && !l.includes("[{"));
        expect(bad, `${f} ${zh ? "zh" : "en"}`).toEqual([]);
      }
    }
    // JSON keeps its JSON docs.
    expect(systemPrompt("/ws", false, "deep", undefined, true, false)).toMatch(/args: \{ "path": string/);
  });

  it("writes one-line examples in the model's own shape", () => {
    setCallFormat("gemma");
    expect(callExample("read_file", '{"path":"src/app.ts"}')).toBe('call:read_file{path:<|"|>src/app.ts<|"|>}');
    setCallFormat("xml");
    expect(callExample("read_file", '{"path":"src/app.ts"}')).toBe("<function=read_file><parameter=path>src/app.ts</parameter></function>");
    setCallFormat("lfm");
    expect(callExample("bg_output", '{"id":7}')).toBe("bg_output(id=7)");
    setCallFormat("json");
    expect(callExample("read_file", '{"path":"src/app.ts"}')).toBe('read_file {"path":"src/app.ts"}');
  });

  it("shows no JSON in the hints and corrections", () => {
    setCallFormat("gemma");
    const hint = jitHintFor("understand_repo", "[目录]", "zh", new Set());
    expect(hint).toContain("call:search_code{query:");
    expect(hint).not.toMatch(JSON_KEY);
    for (const attempt of [1, 2, 3]) {
      expect(missingArgLadder("read_file", "path", '{"path":"src/app.ts"}', attempt, "en")).not.toMatch(JSON_KEY);
    }
    const note = describeInvalidCall("<tool_call>\n<task_complete>\n</task_complete>", 1, "zh");
    expect(note).toContain("直接给出最终答复");
    expect(note).not.toContain("JSON");
  });

  it("speaks the turn's format in the plan echo, the memory error, MCP references and nudges", async () => {
    const todos = [{ content: "建目录", status: "pending" }] as never;
    const mcp = {
      name: "search",
      description: "Search the docs.",
      input_schema: {
        type: "object",
        properties: { query: { type: "string", description: "what to find" }, page: { type: "number" } },
        required: ["query"],
      },
    };
    const noFs = { readFile: async () => "", writeFile: async () => {} };
    for (const f of NATIVE) {
      setCallFormat(f);
      const echo = planEcho(todos, "zh");
      expect(echo, f).toContain(renderCall("bash", { command: "mkdir -p <项目目录>" }, f));
      expect(echo, f).not.toContain("<tool_call>{");
      const err = await rememberFact(noFs, "", "", "en");
      expect(err, f).toContain("remember");
      expect(err, f).not.toMatch(JSON_KEY);
      const doc = fullDoc("mcp_search", mcp);
      expect(doc, f).toContain("- query: string — what to find");
      expect(doc, f).toContain("- page?: number");
      expect(doc, f).not.toMatch(JSON_KEY);
      expect(callTag(true), f).not.toMatch(JSON_KEY);
      expect(callTag(true), f).not.toContain('{"');
    }
    // A JSON turn reads exactly as it always did.
    setCallFormat("json");
    expect(planEcho(todos, "zh")).toContain('<tool_call>{"name":"bash","arguments":{"command":"mkdir -p <项目目录>"}}</tool_call>');
    expect(await rememberFact(noFs, "", "", "en")).toContain('e.g. {"title":"build rule"');
    expect(fullDoc("mcp_search", mcp)).toContain('args schema: {"type":"object"');
    expect(callTag(true)).toBe("一行 <tool_call>");
    expect(callTag(false)).toBe("a single <tool_call> line");
  });
});

describe("a call Chaty writes back into history", () => {
  afterEach(() => setCallFormat("json"));

  it("is written in the format it came in, and reads back the same", () => {
    const args = { path: "src/app.py", content: 'print("hi")\n  x = 1', edits: [{ old_string: "a", new_string: "b" }] };
    for (const f of ["json", "xml", "gemma", "lfm"] as const) {
      const block = renderCall("write_file", args, f);
      expect(formatOf(block), f).toBe(f);
      const back = parseToolCall(block);
      expect(back?.name, f).toBe("write_file");
      expect(back?.args.path, f).toBe("src/app.py");
      expect(back?.args.content, f).toBe(args.content);
      expect(back?.args.edits, f).toEqual(args.edits);
    }
  });

  it("keeps its format when a written body is trimmed away", () => {
    const big = "x = 1\n".repeat(200);
    for (const f of NATIVE) {
      const turn = `先写文件。\n${renderCall("write_file", { path: "a.py", content: big }, f)}`;
      const slim = stubWrittenBodies(turn, "zh");
      expect(slim.length, f).toBeLessThan(turn.length / 2);
      expect(formatOf(slim), f).toBe(f);
      expect(slim, f).not.toContain("<tool_call>{");
      const c = parseToolCall(slim);
      expect(c?.args.path, f).toBe("a.py");
      expect(String(c?.args.content), f).toContain("已省略");
    }
    const json = `<tool_call>${JSON.stringify({ name: "write_file", arguments: { path: "a.py", content: big } })}</tool_call>`;
    expect(stubWrittenBodies(json, "zh")).toMatch(/^<tool_call>\{"name":"write_file","arguments":\{"path":"a\.py","content":"\(已省略/);
  });
});

describe("Gemma's calls as it actually writes them", () => {
  // Each shape below was a rejected call in a real E4B run.
  it("reads objects written as JSON inside an array", () => {
    const c = parseToolCall(
      '<|tool_call>call:edit_file{path:<|"|>src/fruitstand.py<|"|>,edits:[{\n"old_string": "for i in range(len(names)):",\n"new_string": "for name in names:",\n"replace_all": false\n},\n{\n"old_string": "total += CATALOG[names[i]](order[names[i]])",\n"new_string": "total += CATALOG[name](order[name])\\n",\n"replace_all": false\n}]}<tool_call|>',
    );
    expect(c?.name).toBe("edit_file");
    expect(c?.args.path).toBe("src/fruitstand.py");
    expect(c?.args.edits).toEqual([
      { old_string: "for i in range(len(names)):", new_string: "for name in names:", replace_all: false },
      { old_string: "total += CATALOG[names[i]](order[names[i]])", new_string: "total += CATALOG[name](order[name])\n", replace_all: false },
    ]);
  });

  it("takes the call after the thought over one quoted in it", () => {
    const c = parseToolCall(
      '<|channel>thought\nThe format is `<|tool_call>call:工具名{参数名:<|"|>文字值<|"|>}<tool_call|>`.\nThe previous call was `<|tool_call>call:read_file{path:<|"|>old.py<|"|>}<tool_call|>`.<channel|><|tool_call>call:read_file{path:<|"|>src/fruitstand.py<|"|>}<tool_call|>',
    );
    expect(c).toEqual({ name: "read_file", args: { path: "src/fruitstand.py" } });
  });

  it("reads a file written after content: with no <|\"|> around it", () => {
    const body = '"""Fruit stand pricing."""\n\nTAX_RATE = 0.08\n\n\ndef tax(a):\n    return round(a * (1 + TAX_RATE), 2)\n';
    expect(parseToolCall(`<|tool_call>call:write_file{path:<|"|>src/fruitstand.py<|"|>,content:${body}}<tool_call|>`)?.args).toEqual({
      path: "src/fruitstand.py",
      content: body,
    });
    // The stop sequence took the closer: the call ends at its brace.
    expect(parseToolCall(`<|tool_call>call:write_file{path:<|"|>a.py<|"|>,content:${body}}`)?.args.content).toBe(body);
    // Bare text followed by a quoted argument is not guessed at.
    expect(parseToolCall(`<|tool_call>call:write_file{content:${body},path:<|"|>a.py<|"|>}<tool_call|>`)).toBeNull();
  });

  it("reads text in single quotes", () => {
    // Gemma-26B, eleven rejected calls in one run.
    expect(
      parseToolCall(
        `<|tool_call>call:edit_file{new_string:':root {\n  --paper: #eef3f7;\n}',old_string:':root {\n  --paper: #f7f3ea;\n}',path:<|"|>index.html<|"|>}<tool_call|>`,
      )?.args,
    ).toEqual({ new_string: ":root {\n  --paper: #eef3f7;\n}", old_string: ":root {\n  --paper: #f7f3ea;\n}", path: "index.html" });
    expect(
      parseToolCall(
        `<|tool_call>call:edit_file{edits:[{new_string:'.margin-note {',old_string:'.card-note {',replace_all:true},{new_string:'function f() {\\n  store.setItem("visits", n);\\n}',old_string:'return n;',replace_all:true}],path:<|"|>index.html<|"|>}<tool_call|>`,
      )?.args.edits,
    ).toEqual([
      { new_string: ".margin-note {", old_string: ".card-note {", replace_all: true },
      { new_string: 'function f() {\n  store.setItem("visits", n);\n}', old_string: "return n;", replace_all: true },
    ]);
    // A quote inside the text does not end it.
    expect(parseToolCall(`<|tool_call>call:bash{command:'grep -q 'n + 1' index.html',path:<|"|>.<|"|>}<tool_call|>`)?.args.command).toBe(
      "grep -q 'n + 1' index.html",
    );
  });

  it("still reads the template's own form", () => {
    expect(parseToolCall('<|tool_call>call:bg_output{id:7,tail:true,note:<|"|>a, b: c<|"|>}<tool_call|>')?.args).toEqual({
      id: 7,
      tail: true,
      note: "a, b: c",
    });
  });
});

type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };
type Req = { messages: { role: string; content: string }[]; params: { stop: string[] } };

/** The requests of a turn whose first reply is `reply` (as the engine hands it
 *  over: the closing marker is the stop sequence, so it is not there). */
async function requestsAfter(f: Fmt, reply: string): Promise<Req[]> {
  const reqs: Req[] = [];
  mockIPC(async (cmd, args) => {
    if (cmd !== "generate") return null;
    const a = args as { request: Req; onEvent: Chan };
    reqs.push(JSON.parse(JSON.stringify(a.request)) as Req);
    a.onEvent.onmessage?.({ type: "token", text: reqs.length === 1 ? reply : "计划已列好,开始动手。" });
    a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } });
    return null;
  });
  await runAgentTurn(
    "列个计划", [], "/tmp/ws", "zh",
    {
      thinkMode: "normal", maxSteps: 4, temperature: 0.3, toolFormat: f,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    {
      onThinking: () => {}, onAssistantText: () => {}, onStep: () => {}, onPlan: () => {}, onFinal: () => {},
      onError: (m: string) => { throw new Error(m); },
    } as never,
  );
  clearMocks();
  return reqs;
}

describe("the model's own call, in the next round's prompt", () => {
  afterEach(() => setCallFormat("json"));

  const PLAN: Record<Fmt, [string, string]> = {
    json: ['<tool_call>{"name":"update_plan","arguments":{"todos":[{"content":"写页面","status":"in_progress"}]}}', "</tool_call>"],
    xml: ['<tool_call>\n<function=update_plan>\n<parameter=todos>\n[{"content":"写页面","status":"in_progress"}]\n</parameter>\n</function>\n', "</tool_call>"],
    gemma: ['<|tool_call>call:update_plan{todos:[{content:<|"|>写页面<|"|>,status:<|"|>in_progress<|"|>}]}', "<tool_call|>"],
    lfm: ['<|tool_call_start|>[update_plan(todos=[{"content":"写页面","status":"in_progress"}])]', "<|tool_call_end|>"],
  };

  for (const f of ["json", "xml", "gemma", "lfm"] as const) {
    it(`is there once, closed, with no JSON copy (${f})`, async () => {
      const [reply, closer] = PLAN[f];
      const reqs = await requestsAfter(f, reply);
      expect(reqs.length).toBeGreaterThanOrEqual(2);
      expect(reqs[0].params.stop).toEqual(expect.arrayContaining(CALL_CLOSERS));
      const turn = reqs[1].messages.filter((m) => m.role === "assistant").slice(-1)[0]?.content ?? "";
      expect(turn).toContain(`${reply.trim()}${closer}`.replace(/\n<\/tool_call>$/, "</tool_call>").slice(0, 40));
      expect(turn.trim().endsWith(closer)).toBe(true);
      expect(turn.match(/update_plan/g)?.length).toBe(1);
      if (f !== "json") {
        for (const m of reqs[1].messages) {
          expect(m.content, `${m.role}: ${m.content.slice(0, 80)}`).not.toContain("<tool_call>{");
        }
        if (f !== "xml") expect(turn).not.toContain("</tool_call>");
      }
    });
  }
});

describe("an XML call with closers the model kept writing", () => {
  afterEach(() => setCallFormat("json"));

  it("is cut when the closers run on, and the call still runs", async () => {
    const cmds: string[] = [];
    const reqs: Req[] = [];
    // Like the engine: a cancel stops the stream.
    let cancelled = false;
    mockIPC(async (cmd, args) => {
      cmds.push(cmd);
      if (cmd.includes("cancel")) {
        cancelled = true;
        return null;
      }
      if (cmd !== "generate") return null;
      cancelled = false;
      const a = args as { request: Req; onEvent: Chan };
      reqs.push(JSON.parse(JSON.stringify(a.request)) as Req);
      const chunks =
        reqs.length === 1
          ? ["<tool_call>\n", "<function=update_plan>\n", "<parameter=todos>\n", '[{"content":"写页面","status":"in_progress"}]\n', ...Array<string>(120).fill("</parameter>\n")]
          : ["计划已列好,开始动手。"];
      for (const text of chunks) {
        if (cancelled) break;
        a.onEvent.onmessage?.({ type: "token", text });
      }
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } });
      return null;
    });
    await runAgentTurn(
      "列个计划", [], "/tmp/ws", "zh",
      {
        thinkMode: "normal", maxSteps: 4, temperature: 0.3, toolFormat: "xml",
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onStep: () => {}, onPlan: () => {}, onFinal: () => {},
        onError: (m: string) => { throw new Error(m); },
      } as never,
    );
    clearMocks();
    expect(cmds.some((c) => c.includes("cancel"))).toBe(true);
    expect(reqs.length).toBeGreaterThanOrEqual(2);
    // Kept as generated — rewriting it would cost a hybrid model the whole
    // cache — and bounded by the cut: a handful of closers, not thousands.
    const turn = reqs[1].messages.filter((m) => m.role === "assistant").slice(-1)[0]?.content ?? "";
    expect(turn.startsWith("<tool_call>\n<function=update_plan>")).toBe(true);
    expect(turn.split("</parameter>").length - 1).toBeLessThan(20);
  });
});

describe("an XML call written loosely", () => {
  // A 4B, one run: `<parameter="path">` read as a missing path for a dozen
  // rounds, an unnamed edits array reported as "identical" strings, and a
  // list_dir followed by `</path>` until the token cap.
  afterEach(() => setCallFormat("json"));

  it("reads names in quotes", () => {
    expect(
      parseToolCall('<tool_call>\n<function=edit_file>\n<parameter="path">index.html</parameter>\n<parameter=edits">[{"old_string":"a","new_string":"b"}]</parameter>\n</function>\n'),
    ).toEqual({ name: "edit_file", args: { path: "index.html", edits: [{ old_string: "a", new_string: "b" }] } });
    expect(parseToolCall('<tool_call>\n<function="read_file">\n<parameter=path>\na.py\n</parameter>\n</function>\n')?.name).toBe("read_file");
  });

  it("is cut only once every value it opened is closed", () => {
    expect(xmlRunsOn("<tool_call>\n<function=list_dir>\n</parameter>\n</parameter>\n</path>\n</path>\n")).toBe(true);
    expect(
      xmlRunsOn("<tool_call>\n<function=write_file>\n<parameter=path>\na.html\n</parameter>\n<parameter=content>\n<main>\n<div>\n</div>\n</section>\n</main>\n</body>\n</html>\n"),
    ).toBe(false);
    expect(xmlRunsOn("<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n")).toBe(false);
  });

  it("lists names without quotes in the docs", () => {
    expect(plainArgs('- edit_file: replace text. args: { "path", "old_string", "new_string", "replace_all"?: boolean }')).toBe(
      "- edit_file: replace text. args: path, old_string, new_string, replace_all?: boolean",
    );
  });

  it("asks for old_string when a call brings neither it nor edits", async () => {
    const reqs = await requestsAfter("xml", "<tool_call>\n<function=edit_file>\n<parameter=path>\nindex.html\n</parameter>\n</function>\n");
    const said = reqs[1].messages.map((m) => m.content).join("\n");
    expect(said).toContain('"old_string"');
    expect(said).not.toContain("相同");
  });
});

describe("LFM's tool calls on screen", () => {
  // Owner report: an LFM model's calls showed up whole in the reply — the
  // display dropped their markers and left `[read_file(path='…')]` behind.
  it("are dropped whole, an unfinished one too", () => {
    expect(withoutToolCallSpans("先读文件。<|tool_call_start|>[read_file(path='a.py')]<|tool_call_end|>")).toBe("先读文件。");
    expect(withoutToolCallSpans("<|tool_call_start|>[edit_file(path='a.py', old_string='x")).toBe("");
    expect(withoutToolCallSpans("a plain answer")).toBe("a plain answer");
  });

  it("never reach a turn's live prose", async () => {
    const shown: string[] = [];
    let n = 0;
    mockIPC(async (cmd, args) => {
      if (cmd !== "generate") return null;
      const a = args as { onEvent: Chan };
      n++;
      const chunks =
        n === 1
          ? ["先列个计划。", "<|tool_call_start|>", "[update_plan(todos=", '[{"content":"写页面","status":"in_progress"}])]']
          : ["计划已列好。"];
      for (const text of chunks) a.onEvent.onmessage?.({ type: "token", text });
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } });
      return null;
    });
    await runAgentTurn(
      "列个计划", [], "/tmp/ws", "zh",
      {
        thinkMode: "normal", maxSteps: 4, temperature: 0.3, toolFormat: "lfm",
        signal: { cancelled: false },
        approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: (t: string) => shown.push(t), onStep: () => {}, onPlan: () => {}, onFinal: () => {},
        onError: (m: string) => { throw new Error(m); },
      } as never,
    );
    clearMocks();
    expect(shown.length).toBeGreaterThan(0);
    expect(shown.some((t) => t.includes("update_plan(") || t.includes("[update_plan"))).toBe(false);
  });
});

describe("the prompt sources", () => {
  // Every example a model sees must come from callFormat's helpers; a JSON call
  // written by hand into a string is a JSON example in every format's turn.
  it("hold no hand-written JSON call", () => {
    const sources: Record<string, string> = {
      "agentLoop.ts": agentLoopSrc,
      "jitHints.ts": jitHintsSrc,
      "wrapupGate.ts": wrapupGateSrc,
      "memoryFiles.ts": memoryFilesSrc,
      "toolDocs.ts": toolDocsSrc,
      "toolRegistry.ts": toolRegistrySrc,
      "mcp.ts": mcpSrc,
      "skillFiles.ts": skillFilesSrc,
    };
    const handWritten = [/<tool_call>\{"/, /\b[a-z_]+ \{\\?"[a-z_]+\\?":/];
    for (const [f, src] of Object.entries(sources)) {
      expect(src.length, f).toBeGreaterThan(500);
      const bad = src
        .split("\n")
        .filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l) && handWritten.some((re) => re.test(l)));
      expect(bad, f).toEqual([]);
    }
  });
});
