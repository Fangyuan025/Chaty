// Channel-style reasoning normalization (Gemma 4 / Harmony → <think>).
// The shapes here are real: the Gemma 4 chat template opens thought with
// `<|channel>thought\n` and closes it with `<channel|>`; small quants
// routinely sail past their turn end and re-open the channel, and streaming
// buffers end mid-marker. Each case is a leak we shipped at least once.
import { describe, expect, it } from "vitest";
import { answerOnly, normalizeChannels, stripThink } from "./voiceText";

describe("normalizeChannels", () => {
  it("leaves marker-free text untouched", () => {
    const s = "plain answer with a < b and <div> markup";
    expect(normalizeChannels(s)).toBe(s);
  });

  it("converts the canonical Gemma 4 shape", () => {
    expect(normalizeChannels("<|channel>thought\nreasoning<channel|>answer")).toBe(
      "<think>reasoning</think>answer",
    );
  });

  it("keeps a streaming thought open", () => {
    expect(normalizeChannels("<|channel>thought\nstill going")).toBe("<think>still going");
  });

  // ATEM (Muse Glimmer) names a recipient instead of a channel, and the
  // generation prompt has already emitted `<|start|>assistant`, so what the
  // engine streams begins bare at ` to=self<|message|>`.
  it("converts an ATEM turn addressed to self and then to the user", () => {
    expect(
      normalizeChannels(
        " to=self<|message|>reasoning here<|eom|><|start|>assistant to=user<|message|>42<|eot|>",
      ),
    ).toBe("<think>reasoning here</think>42");
  });

  it("keeps a streaming ATEM thought open", () => {
    expect(normalizeChannels(" to=self<|message|>still thinking")).toBe("<think>still thinking");
  });

  it("reads an ATEM turn that never reasons as plain answer text", () => {
    expect(normalizeChannels(" to=user<|message|>direct answer<|eot|>")).toBe("direct answer");
  });

  it("treats a tool recipient as leaving the reasoning channel", () => {
    expect(
      normalizeChannels(" to=self<|message|>plan<|eom|><|start|>assistant to=fs.read<|message|>call"),
    ).toBe("<think>plan</think>call");
  });

  it("converts EVERY thought block, not just the first", () => {
    expect(
      normalizeChannels(
        "<|channel>thought\nfirst<channel|>part one<turn|>\n<|turn>model\n<|channel>thought\nsecond<channel|>part two",
      ),
    ).toBe("<think>first</think>part one<think>second</think>part two");
  });

  it("a close marker followed by prose starting with 'Final' stays prose", () => {
    expect(normalizeChannels("<|channel>thought\nplan<channel|>Final result: 55")).toBe(
      "<think>plan</think>Final result: 55",
    );
  });

  it("closes a thought at a Harmony final channel", () => {
    expect(
      normalizeChannels("<|channel|>analysis<|message|>reasoning<|channel|>final<|message|>answer"),
    ).toBe("<think>reasoning</think>answer");
  });

  it("holds back a partial marker at the buffer end while streaming", () => {
    expect(normalizeChannels("<|")).toBe("");
    expect(normalizeChannels("<|chan")).toBe("");
    expect(normalizeChannels("<|channel>thou")).toBe("");
    expect(normalizeChannels("answer text <|channel>thou")).toBe("answer text ");
    // Ordinary text tails are not markers.
    expect(normalizeChannels("a < b")).toBe("a < b");
    expect(normalizeChannels("use <div")).toBe("use <div");
  });

  it("strips control tokens (<|think|>, turn markers) everywhere", () => {
    expect(normalizeChannels("<|think|>\n<|turn>model\nanswer<turn|>\n")).toBe("answer");
  });

  it("plain <think> tags (Qwen convention) pass through untouched", () => {
    const s = "<think>reasoning</think>answer";
    expect(normalizeChannels(s)).toBe(s);
    // …even when a piped control token sits in the same message.
    expect(normalizeChannels("<|think|><think>r</think>a")).toBe("<think>r</think>a");
  });

  it("unpiped <end>/<message> in ordinary text are not markers", () => {
    const s = "tags like <end> and <message> are prose";
    expect(normalizeChannels(s)).toBe(s);
  });
});

describe("stripThink / answerOnly on channel input", () => {
  it("stripThink drops all reasoning including an unclosed tail", () => {
    expect(
      stripThink("<|channel>thought\na<channel|>mid<|channel>thought\ncut off by eos"),
    ).toBe("mid");
  });

  it("answerOnly never reads a re-opened thought aloud", () => {
    expect(
      answerOnly("<|channel>thought\na<channel|>speak this<|channel>thought\nnot this"),
    ).toBe("speak this");
  });
});
