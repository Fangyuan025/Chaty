// Text helpers shared by the composer's read-aloud and the live voice mode.

const SOURCE_RE = /[【[（(]\s*来源\s*[\d０-９,，、\s]+[】\])）]/g;

/** Strip `<think>` reasoning and inline source markers; trim. */
export function stripThink(s: string): string {
  return s
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<\/?think>/g, "")
    .replace(SOURCE_RE, "")
    .trim();
}

/** The answer portion only — skip `<think>` reasoning so TTS never reads it. */
export function answerOnly(text: string): string {
  const ci = text.lastIndexOf("</think>");
  if (ci !== -1) return text.slice(ci + "</think>".length);
  if (text.includes("<think>")) return ""; // reasoning still open
  return text;
}

/** Split off the prefix ending at the last sentence boundary: [complete, rest]. */
export function cutSentences(buf: string): [string, string] {
  const m = buf.match(/^[\s\S]*[.!?。！？\n]/);
  if (!m) return ["", buf];
  return [buf.slice(0, m[0].length), buf.slice(m[0].length)];
}

/** Reduce assistant text to something natural to read aloud (TTS). */
export function forSpeech(text: string): string {
  return stripThink(text)
    .replace(/```[\s\S]*?```/g, ". ")
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_`#>~|]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 1200);
}
