import { describe, expect, test, beforeEach } from "vitest";
import {
  contextLimit,
  fitTranscript,
  calibrate,
  calibrationFactor,
  messageTokens,
  rawMessageTokens,
  rawTokens,
  resetCalibration,
  textTokens,
} from "./ctxBudget";

beforeEach(() => resetCalibration());

describe("rawTokens — the cost before any correction", () => {
  test("charges CJK far more per character than Latin", () => {
    // The bug this file exists for: a length/2.5 rule read these as equal.
    expect(rawTokens("缓存是一种技术")).toBeGreaterThan(rawTokens("a cache tech"));
  });
  test("never returns zero for real text", () => {
    for (const s of ["a", "字", "()", "🙂"]) expect(rawTokens(s)).toBeGreaterThan(0);
  });
  test("empty text is free", () => {
    expect(rawTokens("")).toBe(0);
  });
  test("an astral character is not one cheap character", () => {
    expect(rawTokens("🙂")).toBeGreaterThan(rawTokens("a"));
  });
});

describe("calibration — the engine's count corrects the guess", () => {
  test("starts neutral, so the first turn uses the raw estimate", () => {
    expect(calibrationFactor()).toBe(1);
    expect(textTokens("hello")).toBe(rawTokens("hello"));
  });

  test("an under-estimate is pulled up toward what was actually charged", () => {
    const msgs = [{ content: "缓存是一种将频繁访问的数据临时存储的技术。" }];
    const predicted = rawMessageTokens(msgs);
    calibrate(predicted, predicted * 2); // engine charged twice what we guessed
    expect(calibrationFactor()).toBeGreaterThan(1.5);
    expect(messageTokens(msgs)).toBeGreaterThan(predicted);
  });

  test("an over-estimate is pulled down", () => {
    calibrate(1000, 500);
    expect(calibrationFactor()).toBeLessThan(1);
  });

  test("one strange turn cannot make the budget wild", () => {
    calibrate(1, 100_000);
    expect(calibrationFactor()).toBeLessThanOrEqual(4);
    resetCalibration();
    calibrate(100_000, 1);
    expect(calibrationFactor()).toBeGreaterThanOrEqual(0.5);
  });

  test("nonsense readings are ignored rather than poisoning the ratio", () => {
    calibrate(0, 500);
    calibrate(500, 0);
    calibrate(-3, 7);
    expect(calibrationFactor()).toBe(1);
  });

  test("it follows a conversation that changes character", () => {
    for (let i = 0; i < 6; i++) calibrate(100, 300); // long Chinese stretch
    const chinese = calibrationFactor();
    for (let i = 0; i < 6; i++) calibrate(100, 100); // then a code review
    expect(calibrationFactor()).toBeLessThan(chinese);
  });

  test("switching models forgets the old tokenizer", () => {
    calibrate(100, 400);
    expect(calibrationFactor()).toBeGreaterThan(1);
    resetCalibration();
    expect(calibrationFactor()).toBe(1);
  });
});

describe("messageTokens", () => {
  test("charges per-message framing, so many short turns are not free", () => {
    const one = [{ content: "abcdefghij" }];
    const ten = Array.from({ length: 10 }, () => ({ content: "a" }));
    expect(messageTokens(ten)).toBeGreaterThan(messageTokens(one));
  });
});

describe("contextLimit — one rule for both modes", () => {
  test("always leaves room for the reply", () => {
    for (const nCtx of [4096, 8192, 32768, 131072]) {
      for (const gen of [undefined, 512, 4096, 32768, 1_000_000]) {
        const limit = contextLimit(nCtx, gen);
        expect(limit).toBeLessThan(nCtx);
        expect(nCtx - limit).toBeGreaterThanOrEqual(700);
      }
    }
  });

  test("a huge reply cap cannot starve the conversation", () => {
    // The old code-mode rule ignored the cap entirely; the old chat rule let it
    // grow without bound. Neither could survive gen == window.
    expect(contextLimit(32768, 32768)).toBeGreaterThanOrEqual(32768 * 0.5);
  });

  test("turning the reply cap off does not collapse the budget", () => {
    // Chat mode used to fall back to a bare 2048 here.
    expect(contextLimit(131072, undefined)).toBeGreaterThan(131072 * 0.9);
  });

  test("a bigger reply cap leaves less room for history, never more", () => {
    expect(contextLimit(32768, 8192)).toBeLessThanOrEqual(contextLimit(32768, 1024));
  });

  test("both modes are asked the same question and get the same answer", () => {
    expect(contextLimit(32768, 4096)).toBe(contextLimit(32768, 4096));
  });
});

describe("fitTranscript — what the summariser gets to read", () => {
  const lines = Array.from({ length: 60 }, (_, i) => `user: turn number ${i} with some content`);

  test("a short stretch is passed through untouched", () => {
    expect(fitTranscript(lines.slice(0, 3), 10_000, "en")).toBe(lines.slice(0, 3).join("\n"));
  });

  test("the opening survives — it is what nothing later restates", () => {
    // The old flat slice(-7000) kept the end and dropped exactly this.
    const out = fitTranscript(lines, 60, "en");
    expect(out).toContain("turn number 0");
  });

  test("the gap is marked, not papered over", () => {
    const out = fitTranscript(lines, 60, "en");
    expect(out).toContain("omitted here");
    expect(out).not.toContain("turn number 30");
  });

  test("it stays within the budget it was given", () => {
    for (const budget of [40, 100, 300]) {
      expect(textTokens(fitTranscript(lines, budget, "en"))).toBeLessThanOrEqual(budget * 1.1);
    }
  });

  test("it cuts on line boundaries, never mid-sentence", () => {
    for (const line of fitTranscript(lines, 60, "en").split("\n")) {
      expect(line === "…(some turns omitted here)…" || lines.includes(line)).toBe(true);
    }
  });

  test("a budget too small for even one line still yields that line", () => {
    expect(fitTranscript(lines, 1, "en")).toContain("turn number 0");
  });

  test("empty in, empty out", () => {
    expect(fitTranscript([], 100, "en")).toBe("");
  });
});

describe("fitTranscript — no single turn eats the whole budget", () => {
  test("one huge tool result does not crowd out every other turn", () => {
    const lines = [
      `tool: <tool_result name="read_file">${"y".repeat(20000)}</tool_result>`,
      "user: and then what",
      "assistant: I checked the config",
      "user: which key was wrong",
    ];
    const out = fitTranscript(lines, 300, "en");
    // The dump is clipped and marked, and the turns after it still make it in.
    expect(out).toContain("truncated");
    expect(out).toContain("which key was wrong");
    expect(textTokens(out)).toBeLessThanOrEqual(300 * 1.2);
  });

  test("the clipped line keeps its opening, which is the part that identifies it", () => {
    const out = fitTranscript(
      [`tool: <tool_result name="read_file">${"y".repeat(20000)}</tool_result>`, "user: ok"],
      300,
      "en",
    );
    expect(out).toContain('name="read_file"');
  });
});

describe("attached pictures are context too", () => {
  test("an image costs far more than the sentence that carries it", () => {
    // Counting only the text let a conversation with screenshots call itself
    // comfortable while it was already past the window.
    const withImg = messageTokens([{ content: "look at this", images: ["/tmp/a.png"] }]);
    const without = messageTokens([{ content: "look at this" }]);
    expect(withImg).toBeGreaterThan(without + 500);
  });

  test("two pictures cost about twice one", () => {
    const one = messageTokens([{ content: "x", images: ["/a.png"] }]);
    const two = messageTokens([{ content: "x", images: ["/a.png", "/b.png"] }]);
    expect(two - one).toBeGreaterThan(800);
  });

  test("an empty image list changes nothing", () => {
    expect(messageTokens([{ content: "hi", images: [] }])).toBe(messageTokens([{ content: "hi" }]));
  });

  test("the calibration is fed the same picture cost it will be charged for", () => {
    // Otherwise the engine's promptTokens reads as a wild under-estimate and
    // the ratio climbs for every text-only turn after it.
    const msgs = [{ content: "look", images: ["/a.png"] }];
    expect(rawMessageTokens(msgs)).toBeGreaterThan(500);
  });
});
