import { describe, expect, it } from "vitest";
import { stabilizeTail } from "./streamMarkdown";

describe("the block being written, held back where it has not decided", () => {
  it("a line of only marks waits: no heading flash under a paragraph, no rule flash", () => {
    // A paragraph and a "-": a setext heading for a moment, then a list.
    expect(stabilizeTail("Some text\n-")).toBe("Some text");
    expect(stabilizeTail("Some text\n- ")).toBe("Some text");
    expect(stabilizeTail("Some text\n- item")).toBe("Some text\n- item");
    expect(stabilizeTail("---")).toBe("");
    expect(stabilizeTail("**")).toBe("");
    expect(stabilizeTail("##")).toBe("");
    expect(stabilizeTail("## Title")).toBe("## Title");
    expect(stabilizeTail("Intro\n2.")).toBe("Intro");
    expect(stabilizeTail("Intro\n2. Second")).toBe("Intro\n2. Second");
  });

  it("a table waits for its delimiter row", () => {
    expect(stabilizeTail("| a | b |")).toBe("");
    expect(stabilizeTail("| a | b |\n|---|--")).toBe("");
    expect(stabilizeTail("| a | b |\n|---|---|\n| 1")).toBe("| a | b |\n|---|---|\n| 1");
    // Pipes that are not a table are text.
    expect(stabilizeTail("| not\nreally")).toBe("| not\nreally");
  });

  it("bold, strikethrough and inline code are closed while open", () => {
    expect(stabilizeTail("This is **very imp")).toBe("This is **very imp**");
    expect(stabilizeTail("This is **")).toBe("This is ");
    expect(stabilizeTail("Run `npm te")).toBe("Run `npm te`");
    expect(stabilizeTail("Run `")).toBe("Run ");
    expect(stabilizeTail("~~old")).toBe("~~old~~");
    // Inside code, ** is code.
    expect(stabilizeTail("Use `a ** b` here")).toBe("Use `a ** b` here");
    expect(stabilizeTail("**done** and `x`")).toBe("**done** and `x`");
    // A finished rule is a rule: its asterisks are not an open bold.
    expect(stabilizeTail("***\n")).toBe("***\n");
    expect(stabilizeTail("Text\n\n***\n")).toBe("Text\n\n***\n");
  });

  it("a fenced block is left as it is, closed or not", () => {
    const closed = "```mermaid\ngraph TD\n  A --> B\n```";
    expect(stabilizeTail(closed)).toBe(closed);
    expect(stabilizeTail("```mermaid\ngraph TD\n`")).toBe("```mermaid\ngraph TD\n`");
    expect(stabilizeTail("~~~\ncode")).toBe("~~~\ncode");
  });

  it("display math waits until it is closed", () => {
    expect(stabilizeTail("Area:\n$$\\pi r")).toBe("Area:\n");
    expect(stabilizeTail("$")).toBe("");
    expect(stabilizeTail("Area:\n$$\\pi r^2$$")).toBe("Area:\n$$\\pi r^2$$");
  });
});
