import { normalizeChannels } from "./voiceText";

/** Split a streamed assistant message into its `<think>` reasoning and answer.
 *  A message can carry SEVERAL think blocks (interleaved reasoning, or a
 *  runaway that re-opened its thought channel): all block contents feed the
 *  reasoning panel, everything outside is answer, and only a trailing
 *  unclosed block counts as "still thinking" — while the reply streams. */
export function parseThinking(raw: string): {
  reasoning: string;
  answer: string;
  thinking: boolean;
  hasThink: boolean;
} {
  // Channel-style reasoning markers (Gemma 4 / Harmony) → <think> convention.
  let content = normalizeChannels(raw);
  const close = "</think>";
  const chunks: string[] = [];
  let answer = "";
  let thinking = false;
  let hasThink = false;
  // Orphan close tag: reasoning streamed without an opening <think> (a
  // pre-open-trained model whose prompt lost the tag). Everything before
  // the close is reasoning.
  const oi0 = content.indexOf("<think>");
  const ci0 = content.indexOf(close);
  if (ci0 !== -1 && (oi0 === -1 || ci0 < oi0)) {
    chunks.push(content.slice(0, ci0).trim());
    content = content.slice(ci0 + close.length);
    hasThink = true;
  }
  const re = /<think>([\s\S]*?)(?:<\/think>|$)/g;
  let cursor = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(content))) {
    answer += content.slice(cursor, m.index);
    chunks.push(m[1].replace(/^\n+/, "").trim());
    cursor = m.index + m[0].length;
    hasThink = true;
    thinking = !m[0].endsWith(close);
  }
  answer += content.slice(cursor);
  return {
    reasoning: chunks.filter(Boolean).join("\n\n"),
    answer: answer.replace(/^\s+/, ""),
    thinking,
    hasThink,
  };
}

/** A reply that ends in an error: the error goes after the reasoning, not
 *  inside it. A stream that dies mid-thought leaves its think block open, and
 *  the error appended to it was shown as part of the thinking, under a
 *  "Thinking" that never ended (issue #18). */
export function withErrorNote(text: string, message: string): string {
  const closed = parseThinking(text).thinking ? `${text}\n</think>` : text;
  return `${closed}\n\n**${message}**`;
}
