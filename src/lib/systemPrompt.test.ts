import { describe, expect, test } from "vitest";

// agentLoop pulls in the Tauri IPC layer — give it a window + mock before import
// (same pattern as bench/coder/runner.mts).
(globalThis as Record<string, unknown>).window = globalThis;
const { mockIPC } = await import("@tauri-apps/api/mocks");
mockIPC(() => Promise.resolve(null));
const { systemPrompt, agentSetEditAnchors, nowLine, grantsLine, planLine } = await import("./agentLoop");
const { setHistoryToolEnabled } = await import("./toolRegistry");
// The app always runs a turn inside a session, so search_history is part of
// every prompt it sends: measure what ships, not a prompt nobody gets.
setHistoryToolEnabled(true);

const variants = [
  { zh: true, vision: false, label: "zh plain", maxChars: 4507 },
  { zh: true, vision: true, label: "zh vision", maxChars: 5817 },
  { zh: false, vision: false, label: "en plain", maxChars: 7944 },
  { zh: false, vision: true, label: "en vision", maxChars: 10054 },
] as const;
// Caps anchored to the post-slimming sizes (2026-07 WS1: 3545 / 4432 / 6031 /
// 7516 JS chars at think=normal, no project doc; before slimming they were
// 5292 / 8801 / 6837 / 10346). en chars run higher than zh because Latin
// spells out what CJK packs into single chars — but en is now pure Latin
// (~4 chars/token vs ~1 for CJK), so it's the cheaper prompt in tokens.
// The prompt is re-prefetched on every agent step, so growth here is a
// per-step tax on slow local prefill — any increase must be deliberate.
// Deliberate increases, and what bought them:
//   2026-09-10  vision +50 zh / +100 en — browser_key. The browser suite could
//   click and it could fill a field, and there was no way to press a key: no
//   Enter to submit, no Escape to dismiss, no Tab, no arrows. The line is
//   peer-sized with browser_click's (209 en chars against its 211); the rest
//   of the cost is simply that there is one more tool than there was.
//   2026-09-17  all four +~172 zh / +~296 en — bg_input. A command that stops
//   to ask is moved to the background to be typed into; without the line, a
//   model that already knew a command wanted answers had no way to give them.
//   Qwen3.6 35B ran out its 12 minutes on `npm init` and on a Python REPL and
//   edited a menu program's code to get round it; Gemma-4 26B ran out on the
//   REPL. With it the same runs took 56s / 20s / 21s and 15s, all by typing.
//   2026-09-17  all four +211 zh / +408 en — search_history. A session's
//   transcript outlives its context window; without the line the model could
//   not reach what compaction had dropped, nor read the session a user
//   pointed at with @ (it only ever sees that session's outline otherwise).
//   The line spends ~60 zh / ~155 en of that saying it searches the
//   CONVERSATION and not the code, in the words that conversation used:
//   with the shorter wording Qwen3-8B answered "what did we decide earlier"
//   by running search_code over the workspace five times and timing out on
//   the step limit, then by searching the web in English for a Chinese
//   transcript's answer.

//   2026-09-18  all four +100 zh / +208 en — multi_read. Reading five files
//   cost five rounds, each re-sending the whole prompt and the whole
//   transcript; one call reads them together, and a file it cannot read is
//   named without losing the ones it could.

//   2026-09-18  en +20 — the call rule and the reasoning line no longer name
//   `<think>` (Qwen's markup handed to every family) and say instead that
//   reasoning does not count as the "nothing else in that message" the rule
//   forbids. Gemma 4, whose reasoning is a thought channel, was opening that
//   channel and closing it again empty in code mode at every level.

//   2026-09-19  all four +176 zh / +280 en — read_history. search_history
//   finds WHERE something was said; reading it back — a whole turn with every
//   tool call it made, then one of those calls' whole result — was what the
//   model had no way to ask for. Owner's design: the search names the step,
//   the read lists the calls with ✓/✗, and the step id goes back to the same
//   tool for the result itself.

//   2026-10-12  all four +67 zh / +157 en — no emoji, no headings on a short
//   answer. Code's answers came decorated with emoji, and so did files the
//   agent wrote (owner: they read as cheap). A clause on the summary rule
//   ("no emoji unless asked") did not hold: Gemma-4 26B read "make the README
//   attractive" as asking, and listed the emoji it used in its answer. As a
//   rule of its own that names "attractive" as no reason and wants an
//   explicit request, on the README and pros/cons tasks (3 runs each):
//   answers 0 emoji on Gemma-4 26B and Qwen3.6 35B, README files 0 and 0/0/6
//   against 12/7 and 12/13 before. list_dir also stopped handing the model
//   📁/📄 to imitate.

//   2026-10-24  vision +151 zh / +237 en — browser_tabs, and browser_eval
//   saying how it runs. A link that opens a new tab (target=_blank,
//   window.open) left the agent reading the tab it clicked in, with no way to
//   see or reach the page it had opened; the browser now follows such a tab,
//   and the line names the one tool that lists, opens, switches and closes the
//   session's tabs. browser_eval's line says it runs as the console does
//   (top-level await, the last expression's value as the result): a body with
//   `const` and no `return` used to come back undefined, and `await` in one
//   was a syntax error.

//   2026-10-24  all four +120 zh / +343 en — a word after each small step.
//   The call rule forbade any prose in a message that made a call, so the
//   user watched a run of tool cards with no idea how it was going; it now
//   forbids only what comes after the call, and a rule asks for a sentence
//   or two after each small step — outside the thinking, which the user
//   usually doesn't see (Qwen3.5 9B wrote exactly that sentence every step,
//   inside its thought). With the app's sampling, one run each, before →
//   after: Qwen3.6 35B 3 of 19 calls with words, 721 s, tests failing → 2
//   of 4, 77 s, passing; Qwen3.5 9B 0/14, 519 s → 0/4, 101 s; Gemma-4 26B
//   0/3, 70 s → 0/4, 95 s — the small two keep it in their thought.

//   2026-10-24  all four +119 zh / +138 en — the word to the user rides in
//   the call. Asked for words before its calls, Qwen3.6 35B with thinking on
//   wrote them inside its thought every time (one update in 11 runs of a
//   find-the-bug task); with thinking off, 6 of 8 calls. The one thing a
//   thinking model always writes outside its thought is the call — so the
//   rule now asks for a `note` argument, with an example on a read (an
//   example on a command taught it the note was for commands). A turn's
//   first quiet call, and a quiet one after two, is sent back once with the
//   call shown with its note in place (agentLoop's note guard). 35B: 0/9,
//   0/8 → 5/9, 6/9, 4/8, 5/9 over four runs, never more than two quiet
//   calls running; Qwen3.5 9B 15/16, Gemma-4 26B 10/11.

//   2026-10-25  all four +28 zh / +93 en — the summary needs no call. "The
//   user can't see your thinking, only your calls" taught Qwen3.5 9B that a
//   call was the only way to speak: planning, it handed in its plan as the
//   note of understand_repo, again and again, until the identical-call
//   breaker stopped it. The note is for while it works; the summary (the
//   plan, planning) is the answer, with no call.

describe("systemPrompt size gate", () => {
  for (const v of variants) {
    test(`${v.label} ≤ ${v.maxChars} chars`, () => {
      const p = systemPrompt("/ws", v.zh, "normal", undefined, v.vision);
      const bytes = new TextEncoder().encode(p).length;
      console.log(`${v.label}: ${p.length} JS chars, ${bytes} UTF-8 bytes`);
      expect(p.length).toBeLessThanOrEqual(v.maxChars);
    });
  }
});

describe("systemPrompt behavior contracts", () => {
  const zh = systemPrompt("/ws", true, "normal", undefined, false);
  const en = systemPrompt("/ws", false, "normal", undefined, false);

  test("one-tool-per-message + tool_call protocol", () => {
    expect(zh).toContain("每次只调用一个工具");
    expect(zh).toContain("</tool_call>");
    expect(en).toContain("Call ONE tool at a time");
    expect(en).toContain("</tool_call>");
  });

  test("edit_file atomicity contract", () => {
    expect(zh).toContain("原子提交");
    expect(en).toContain("atomic edits array");
  });

  test("prompt-injection defense block", () => {
    expect(zh).toContain("防提示词注入");
    expect(en).toContain("prompt-injection defense");
  });

  test("no persistent cwd", () => {
    expect(zh).toContain("单独的 cd 不会保留到下一条命令");
    expect(en).toContain("NO persistent working directory");
  });

  test("vision doc only rides along when visionReady", () => {
    const zhVision = systemPrompt("/ws", true, "normal", undefined, true);
    expect(zhVision.length).toBeGreaterThan(zh.length);
    expect(zhVision).toContain("browser_");
    expect(zh).not.toContain("browser_navigate");
  });

  // Text browser mode: the vision suite minus the two screenshot tools, with
  // the guidance tail swapped. Derived from VISION_TOOL_DOCS by line filter —
  // this breaks loudly if those doc lines are reworked and the derivation
  // silently stops matching.
  test("browser text mode documents the suite without screenshot tools", () => {
    for (const isZh of [true, false]) {
      const p = systemPrompt("/ws", isZh, "normal", undefined, false, true);
      expect(p).toContain("- browser_navigate:");
      expect(p).toContain("- browser_read:");
      expect(p).toContain("- browser_click:");
      expect(p).toContain("- browser_type:");
      expect(p).toContain("- browser_eval:");
      expect(p).not.toContain("browser_screenshot");
      expect(p).not.toContain("browser_snapshot");
      expect(p).toContain(isZh ? "就是你的眼睛" : "browser_read is your eyes");
    }
    // vision wins when both flags are set — screenshots ARE available then
    const both = systemPrompt("/ws", false, "normal", undefined, true, true);
    expect(both).toContain("- browser_screenshot:");
  });

  // The swap is a startsWith match on the doc lines — this breaks loudly if
  // someone reworks those lines and the anchor variants silently stop applying.
  test("anchor mode swaps the editor docs in both languages", () => {
    agentSetEditAnchors(true);
    try {
      for (const isZh of [true, false]) {
        const p = systemPrompt("/ws", isZh, "normal", undefined, false);
        expect(p).toContain("- edit_lines:");
        // No mention may survive anywhere (docs, write_file preference,
        // caution line) — a prompt recommending an unlisted tool made the
        // model avoid editing altogether in the first anchor smoke.
        expect(p).not.toContain("edit_file");
        expect(p).toContain(isZh ? "行号:哈希→" : 'LINE:HASH→');
      }
    } finally {
      agentSetEditAnchors(false);
    }
    expect(systemPrompt("/ws", false, "normal", undefined, false)).toContain("- edit_file:");
  });
  /** The prompt prefix is what the KV cache is resumed from. A system prompt
   *  that says what time it is invalidates itself every sixty seconds, and the
   *  turn that follows re-reads the whole conversation to rebuild a cache it
   *  already had — on a hybrid model, which cannot rewind its recurrent state,
   *  it throws the cache away entirely rather than trimming it. The clock lives
   *  on the turn's user message instead; this pins it there. */
  test("the system prompt does not change as the clock does", () => {
    const RealDate = Date;
    const build = () => systemPrompt("/ws", true, "normal", undefined, false, false, [], "");
    const first = build();
    try {
      const later = new RealDate(RealDate.now() + 3 * 3600_000 + 61_000);
      // @ts-expect-error narrow test shim: a Date that is always "later"
      globalThis.Date = class extends RealDate {
        constructor(...a: ConstructorParameters<typeof RealDate>) {
          if (a.length) super(...a);
          else super(later.getTime());
        }
        static now() {
          return later.getTime();
        }
      };
      expect(build()).toBe(first);
    } finally {
      globalThis.Date = RealDate;
    }
  });

  test("the turn's own message carries the clock", () => {
    const at = new Date("2026-09-06T01:23:00");
    expect(nowLine(true, at)).toContain("01:23");
    expect(nowLine(false, at)).toMatch(/Current date & time/);
    // At the tail, so it is never part of what an earlier turn established.
    expect(nowLine(true, at).startsWith("\n\n")).toBe(true);
  });

  test("the turn names the folders granted outside the workspace, and says once when they are gone", () => {
    const line = grantsLine(["/Users/me/data", "/opt/assets"], true, []);
    expect(line).toContain("/Users/me/data、/opt/assets");
    expect(grantsLine(["/x"], false, [])).toMatch(/Folders outside the workspace/);
    // None now: silent unless the model was told of some.
    expect(grantsLine([], true, [])).toBe("");
    const told = [{ role: "user" as const, content: "改一下" + line }];
    const revoked = grantsLine([], true, told);
    expect(revoked).toContain("撤销");
    expect(grantsLine([], true, [...told, { role: "user" as const, content: "继续" + revoked }])).toBe("");
    // Not in the system prompt: a grant would change it, and re-read it all.
    expect(systemPrompt("/w", true, "normal")).not.toContain("工作区外可以访问的目录");
  });

  test("plan mode is said on the turn", () => {
    expect(planLine(true)).toContain("计划模式");
    expect(planLine(false)).toMatch(/^\n\n\(Plan mode/);
  });
});
