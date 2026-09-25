/** A write or edit card is shown while the model is still writing the call.
 *  Whatever point the output has reached — mid-key, mid-escape, mid-marker —
 *  what the card shows must be a prefix of what the call will turn out to say,
 *  and once the call is complete it must say exactly what parseToolCall reads. */
import { describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { renderCall } = await import("./callFormat");
const { parseToolCall, runAgentTurn } = await import("./agentLoop");
const { liveFileCall, liveScan } = await import("./liveCall");

type Fmt = "xml" | "json" | "gemma" | "lfm";
const FORMATS: Fmt[] = ["xml", "json", "gemma", "lfm"];

const CONTENT = [
  "<!doctype html>",
  '<html lang="zh">',
  "<body>",
  '  <p class="note">价格: {"a": 1, \'b\': [2, 3]} \\ 结束</p>',
  "  <script>if (a < b && c > d) console.log(`x`);</script>",
  "</body>",
  "</html>",
].join("\n");

/** Every prefix of `full`, as the stream would show it. Characters, not
 *  tokens: a token boundary can fall anywhere a character boundary can. */
function* prefixes(full: string): Generator<string> {
  for (let n = 0; n <= full.length; n++) yield full.slice(0, n);
}

describe("a write the model is still writing", () => {
  for (const f of FORMATS) {
    it(`reads only what has arrived, in ${f}`, () => {
      const args = { path: "site/index.html", content: CONTENT };
      const full = `<think>\nWrite the page.\n</think>\n\n${renderCall("write_file", args, f)}`;
      let sawPartial = false;
      for (const p of prefixes(full)) {
        const v = liveFileCall(p);
        if (!v) continue;
        expect(v.name).toBe("write_file");
        if (v.path !== undefined) expect(v.path).toBe(args.path);
        if (v.content !== undefined) {
          expect(CONTENT.startsWith(v.content), `${f} at ${p.length}: ${JSON.stringify(v.content.slice(-20))}`).toBe(true);
          if (!v.contentDone && v.content.length > 10) sawPartial = true;
        }
      }
      expect(sawPartial).toBe(true);
      const end = liveFileCall(full);
      expect(end?.content).toBe(CONTENT);
      expect(end?.contentDone).toBe(true);
      const parsed = parseToolCall(full);
      expect(parsed?.args.content).toBe(end?.content);
    });
  }

  it("is not a card while the call is only quoted in the reasoning", () => {
    const quoted = `<think>\nI could call ${renderCall("write_file", { path: "a.txt", content: "hi" }, "xml")} but`;
    expect(liveFileCall(quoted)).toBeNull();
    // Gemma's thought channel likewise.
    expect(liveFileCall(`<|channel>thought\nmaybe ${renderCall("write_file", { path: "a", content: "b" }, "gemma")} or not`)).toBeNull();
  });

  it("is a card when the call is written mid-thought, where it runs", () => {
    // Qwen3.5 writes the call inside its reasoning; generation stops at the
    // closer, and the call is run.
    const v = liveFileCall(`<think>\nWrite it.\n<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhel`);
    expect(v).toMatchObject({ name: "write_file", path: "a.txt", content: "hel" });
    expect(liveFileCall(`<|channel>thought\n${renderCall("write_file", { path: "a", content: "b" }, "gemma")}`)?.path).toBe("a");
  });

  it("treats a thought marker inside the file being written as file text", () => {
    const content = "The model writes <think> tags here </think> as text.";
    const v = liveFileCall(renderCall("write_file", { path: "notes.md", content }, "xml"));
    expect(v?.content).toBe(content);
  });

  it("is not a card for any other tool", () => {
    expect(liveFileCall(renderCall("bash", { command: "ls" }, "xml"))).toBeNull();
    expect(liveFileCall(renderCall("read_file", { path: "a.ts" }, "json"))).toBeNull();
  });
});

describe("an edit the model is still writing", () => {
  const OLD = 'const price = 1;\nconsole.log("old");';
  const NEW = 'const price = 2;\nconsole.log("new", price);';
  for (const f of FORMATS) {
    it(`reads old and new text as they arrive, in ${f}`, () => {
      const args = { path: "src/app.ts", old_string: OLD, new_string: NEW };
      const full = renderCall("edit_file", args, f);
      for (const p of prefixes(full)) {
        const e = liveFileCall(p)?.edits?.[0];
        if (!e) continue;
        if (e.old !== undefined) expect(OLD.startsWith(e.old)).toBe(true);
        if (e.new !== undefined) expect(NEW.startsWith(e.new)).toBe(true);
      }
      const end = liveFileCall(full)?.edits?.[0];
      expect([end?.old, end?.oldDone, end?.new, end?.newDone]).toEqual([OLD, true, NEW, true]);
    });

    it(`reads an edits array one replacement after another, in ${f}`, () => {
      const edits = [
        { old_string: 'title = "Old"', new_string: 'title = "New"' },
        { old_string: "def total(xs):\n    return sum(xs)", new_string: "def total(xs):\n    return round(sum(xs), 2)" },
      ];
      const full = renderCall("edit_file", { path: "app.py", edits }, f);
      let sawSecond = false;
      for (const p of prefixes(full)) {
        const got = liveFileCall(p)?.edits;
        if (!got) continue;
        expect(got.length).toBeLessThanOrEqual(2);
        got.forEach((e, k) => {
          if (e.old !== undefined) expect(edits[k].old_string.startsWith(e.old), `${f} @${p.length} old ${k}`).toBe(true);
          if (e.new !== undefined) expect(edits[k].new_string.startsWith(e.new), `${f} @${p.length} new ${k}`).toBe(true);
        });
        if (got.length === 2) sawSecond = true;
      }
      expect(sawSecond).toBe(true);
      const end = liveFileCall(full)?.edits?.map((e) => [e.old, e.new]);
      expect(end).toEqual(edits.map((e) => [e.old_string, e.new_string]));
    });
  }
});

describe("an XML call written the other ways", () => {
  // The same spellings the call parser reads, read the same way while they
  // stream — the edit probe and the live card both depend on it.
  it("reads a path written as an element next to <parameter> values", () => {
    const v = liveFileCall("<tool_call>\n<function=edit_file>\n<path>cart.ts</path>\n<parameter=old_string>\nabc\nde");
    expect(v?.path).toBe("cart.ts");
    expect(v?.edits?.[0]).toMatchObject({ old: "abc\nde", oldDone: false });
  });

  it("reads elements closed as parameters, and the other opener spellings", () => {
    const v = liveFileCall("<tool_call>\n<function=edit_file>\n<path>a.ts</parameter>\n<parameter>old_string>\nx\n</parameter>\n<parameter name=\"new_string\">\ny");
    expect(v?.path).toBe("a.ts");
    expect(v?.edits?.[0]).toMatchObject({ old: "x", oldDone: true, new: "y", newDone: false });
  });

  it("keeps markup inside a value as the value", () => {
    const v = liveFileCall("<tool_call>\n<function=write_file>\n<parameter=path>\ni.html\n</parameter>\n<parameter=content>\n<title>t</title>\n<body>");
    expect(v?.path).toBe("i.html");
    expect(v?.content).toBe("<title>t</title>\n<body>");
  });
});

describe("the live view", () => {
  const FILE = ["import os", "", "def total(xs):", "    return sum(xs)", "", "def main():", "    print(total([1, 2]))", "", "main()"].join("\n");
  const kinds = (rows: { kind: string }[]) => rows.map((r) => r.kind).join(" ");

  it("writes a new file line by line, the head on the line being written", () => {
    const s = liveScan({ name: "write_file", before: "", content: "a\nb\nc", contentDone: false })!;
    expect(kinds(s.rows)).toBe("add add add");
    expect(s.head).toBe(2);
  });

  it("does not call the rest of a rewritten file removed before the model reaches it", () => {
    const before = "one\ntwo\nthree\nfour\nfive";
    const s = liveScan({ name: "write_file", before, content: "one\nTWO\nthr", contentDone: false })!;
    expect(s.rows.filter((r) => r.kind === "del")).toEqual([]);
    expect(s.rows.slice(-4).every((r) => r.kind === "pending")).toBe(true);
    expect(s.rows[s.head!].text).toBe("thr");
    const done = liveScan({ name: "write_file", before, content: "one\nTWO", contentDone: true })!;
    expect(done.removed).toBe(4);
  });

  it("finds the block first, the head on the lines being copied out", () => {
    const s = liveScan({ name: "edit_file", before: FILE, edits: [{ old: "def total(xs):\n    return s", oldDone: false }] })!;
    expect(kinds(s.rows)).toBe(Array(9).fill("ctx").join(" "));
    expect(s.rows.map((r, i) => (r.found ? i : -1)).filter((i) => i >= 0)).toEqual([2, 3]);
    expect(s.head).toBe(3);
  });

  it("then rewrites it in place, old lines not yet reached dimmed", () => {
    const s = liveScan({
      name: "edit_file",
      before: FILE,
      edits: [{ old: "def total(xs):\n    return sum(xs)", oldDone: true, new: "def total(xs):\n    return rou", newDone: false }],
    })!;
    // The file above and below stays as it was; the block reads as written so far.
    expect(s.rows.slice(0, 3).map((r) => r.text)).toEqual(["import os", "", "def total(xs):"]);
    expect(s.rows[s.head!]).toEqual({ kind: "add", text: "    return rou" });
    expect(s.rows.some((r) => r.kind === "pending" && r.text === "    return sum(xs)")).toBe(true);
    expect(s.rows.slice(-4).map((r) => r.text)).toEqual(["def main():", "    print(total([1, 2]))", "", "main()"]);
    const done = liveScan({
      name: "edit_file",
      before: FILE,
      edits: [{ old: "    return sum(xs)", oldDone: true, new: "    return round(sum(xs), 2)", newDone: true }],
    })!;
    expect(kinds(done.rows)).toBe("ctx ctx ctx del add ctx ctx ctx ctx ctx");
  });

  it("takes several edits in the order written, moving on to the next block", () => {
    const s = liveScan({
      name: "multi_edit",
      before: FILE,
      edits: [
        { old: "import os", oldDone: true, new: "import os\nimport sys\nimport json", newDone: true },
        { old: "main()", oldDone: false },
      ],
    })!;
    // The first change stays in place; the head has moved down to the next block.
    expect(s.rows.slice(0, 3).map((r) => [r.kind, r.text])).toEqual([
      ["ctx", "import os"],
      ["add", "import sys"],
      ["add", "import json"],
    ]);
    expect(s.rows[s.head!].text).toBe("def main():");
    expect(s.added).toBe(2);
    // Placed in the file as the first edit left it: `main()` first occurs at
    // the call inside main — the tool would match there too.
    const second = liveScan({
      name: "multi_edit",
      before: FILE,
      edits: [
        { old: "import os", oldDone: true, new: "import os\nimport sys", newDone: true },
        { old: "\nmain()", oldDone: true, new: "\nif __name__ == '__main__':\n    main()", newDone: true },
      ],
    })!;
    expect(second.rows.filter((r) => r.kind === "del").map((r) => r.text)).toEqual(["main()"]);
    expect(second.rows[second.rows.length - 1].text).toBe("    main()");
  });

  it("finds a block whose indentation the model got wrong, as the edit tool does", () => {
    const s = liveScan({ name: "edit_file", before: FILE, edits: [{ old: "return sum(xs)", oldDone: true }] })!;
    expect(s.rows[s.head!].text).toBe("    return sum(xs)");
    const loose = liveScan({ name: "edit_file", before: FILE, edits: [{ old: "def total(xs):\nreturn sum(xs)", oldDone: true }] })!;
    expect(loose.rows.filter((r) => r.found).map((r) => r.text)).toEqual(["def total(xs):", "    return sum(xs)"]);
  });

  it("shows the replacements alone while the file is not known", () => {
    const s = liveScan({ name: "edit_file", edits: [{ old: "x = 1", oldDone: true, new: "x = 2", newDone: true }] })!;
    expect(s.rows).toEqual([
      { kind: "del", text: "x = 1" },
      { kind: "add", text: "x = 2" },
    ]);
  });
});

describe("a live card in a turn", () => {
  type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };
  type Step = { id: string; status: string; live?: { pending?: boolean; content?: string }; diff?: { after: string } };

  async function turn(
    first: string[],
    format: "xml" | "json",
    read: () => string,
    stop?: { afterTokens?: number; atApproval?: boolean },
  ) {
    const signal = { cancelled: false };
    const live: Step[] = [];
    const steps: Step[] = [];
    const gone: string[] = [];
    let n = 0;
    mockIPC(async (cmd, args) => {
      if (cmd === "agent_read_file_raw") return read();
      if (cmd === "agent_write_file") return "wrote site/index.html";
      if (cmd !== "generate") return null;
      const a = args as { onEvent: Chan };
      n++;
      let sent = 0;
      for (const text of n === 1 ? first : ["写好了。"]) {
        a.onEvent.onmessage?.({ type: "token", text });
        if (stop?.afterTokens && ++sent >= stop.afterTokens) {
          signal.cancelled = true;
          break;
        }
      }
      a.onEvent.onmessage?.({ type: "done", stats: { completionTokens: 1, tokensPerSecond: 1, promptTokens: 1 } });
      return null;
    });
    await runAgentTurn(
      "写一个页面", [], "/tmp/ws", "zh",
      {
        thinkMode: "normal", maxSteps: 4, temperature: 0.3, toolFormat: format,
        signal,
        approve: async () => {
          if (!stop?.atApproval) return true;
          signal.cancelled = true;
          return false;
        },
        approveDir: async () => false, approveSudo: async () => ({ ok: false }),
      } as never,
      {
        onThinking: () => {}, onAssistantText: () => {}, onPlan: () => {}, onFinal: () => {},
        onStep: (st: Step) => steps.push(structuredClone(st)),
        onLiveStep: (st: Step) => live.push(structuredClone(st)),
        onLiveStepGone: (id: string) => gone.push(id),
        onError: (m: string) => { throw new Error(m); },
      } as never,
    );
    clearMocks();
    return { live, steps, gone };
  }

  it("becomes the step that runs it, under the same id", async () => {
    const call = renderCall("write_file", { path: "site/index.html", content: CONTENT }, "xml");
    const chunks = ["<think>\n写页面\n</think>\n\n", ...call.match(/[\s\S]{1,7}/g)!];
    const { live, steps, gone } = await turn(chunks, "xml", () => "<!doctype html>\n<html>\n</html>");
    expect(live.length).toBeGreaterThan(0);
    expect(live.every((st) => st.live?.pending)).toBe(true);
    expect(live[live.length - 1]?.live?.content).toBe(CONTENT);
    const id = live[0].id;
    expect(steps.map((st) => [st.id, st.status])).toEqual([
      [id, "running"],
      [id, "done"],
    ]);
    // Running, the card still shows what is being written; done, its diff.
    expect(steps[0].live?.content).toBe(CONTENT);
    expect(steps[0].live?.pending).toBeUndefined();
    expect(steps[1].live).toBeUndefined();
    expect(steps[1].diff?.after).toBe(CONTENT);
    expect(gone).toEqual([]);
  });

  it("is withdrawn when the call it showed never runs", async () => {
    // No path: the call is sent back for its missing argument, not run.
    const call = '<tool_call>{"name": "write_file", "arguments": {"content": "<p>hi</p>"}}</tool_call>';
    const { live, steps, gone } = await turn([call], "json", () => "");
    expect(live.length).toBeGreaterThan(0);
    expect(gone).toEqual([live[0].id]);
    expect(steps.some((st) => st.id === live[0].id)).toBe(false);
  });

  // Owner: a card for a write that was interrupted is not a write — it goes.
  it("goes when the turn is stopped while the call is being written", async () => {
    // The first token carries most of the call, so its card is on screen
    // when the stop comes.
    const call = renderCall("write_file", { path: "site/index.html", content: CONTENT }, "xml");
    const half = call.slice(0, Math.floor(call.length * 0.6));
    const { live, steps, gone } = await turn([half, "more"], "xml", () => "", { afterTokens: 1 });
    expect(live.length).toBeGreaterThan(0);
    expect(gone).toEqual([live[0].id]);
    expect(steps).toEqual([]);
  });

  it("goes when the turn is stopped at its approval", async () => {
    const call = renderCall("write_file", { path: "site/index.html", content: CONTENT }, "xml");
    const { live, steps, gone } = await turn([call], "xml", () => "", { atApproval: true });
    expect(live.length).toBeGreaterThan(0);
    expect(gone).toEqual([live[0].id]);
    expect(steps).toEqual([]);
  });
});
