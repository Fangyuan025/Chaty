import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LangProvider } from "./i18n";
import { parseThinking, withErrorNote } from "./reasoning";
import { AssistantMessage } from "../components/AssistantMessage";

// Issue #18: a reply stopped, or ended by an error, mid-thought kept a
// "Thinking" label with its dots running, and the error sat inside the thought.
describe("a reply that ended mid-thought", () => {
  it("puts an error after the reasoning, not inside it", () => {
    const text = withErrorNote("<think>looking at the picture", "multimodal prefill failed");
    const p = parseThinking(text);
    expect(p.thinking).toBe(false);
    expect(p.reasoning).toBe("looking at the picture");
    expect(p.answer).toBe("**multimodal prefill failed**");
  });

  it("closes a channel-style thought the same way", () => {
    const p = parseThinking(withErrorNote("<|channel>thought\nstill going", "boom"));
    expect(p.thinking).toBe(false);
    expect(p.reasoning).toBe("still going");
    expect(p.answer).toBe("**boom**");
  });

  it("leaves a finished thought alone", () => {
    expect(withErrorNote("<think>a</think>answer", "boom")).toBe("<think>a</think>answer\n\n**boom**");
    expect(withErrorNote("plain answer", "boom")).toBe("plain answer\n\n**boom**");
  });

  const render = (content: string, streaming: boolean) =>
    renderToStaticMarkup(
      createElement(LangProvider, null, createElement(AssistantMessage, { content, streaming })),
    );

  it("is shown as thinking only while it streams", () => {
    expect(render("<think>still thinking", true)).toContain("is-thinking");
    const stopped = render("<think>still thinking", false);
    expect(stopped).not.toContain("is-thinking");
    expect(stopped).toContain("think-toggle");
  });
});
