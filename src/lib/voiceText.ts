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

// Emoji + pictographs + variation selectors / ZWJ / regional indicators.
const EMOJI_RE = /[\p{Extended_Pictographic}\p{Regional_Indicator}\u{FE0F}\u{20E3}\u{200D}]/gu;

/** Remove emoji/pictographs (TTS can't read them; live mode forbids them). */
export function stripEmoji(s: string): string {
  return s.replace(EMOJI_RE, "");
}

// Common LLM "closer" filler that sounds robotic when spoken aloud.
const OFFER = String.raw`(?:and |so |well |okay,? )?(?:please )?(?:just )?(?:let me know\b|feel free\b|don'?t hesitate\b|hope (?:this|that|it) helps\b|is there anything else\b)[^.!?]*[.!?"')\]]*`;
const TRAILING_OFFER_MID = new RegExp(String.raw`([.!?])\s+${OFFER}\s*$`, "i");
const TRAILING_OFFER_ALL = new RegExp(String.raw`^${OFFER}\s*$`, "i");

/** Drop a trailing "let me know if…/feel free to…/hope this helps" closer. */
export function dropTrailingOffer(s: string): string {
  return s.replace(TRAILING_OFFER_MID, "$1").replace(TRAILING_OFFER_ALL, "").trim();
}

/** Reduce assistant text to something natural to read aloud (TTS). Flattens
 *  any lists the model produced into flowing prose and drops robotic closers. */
export function forSpeech(text: string): string {
  const cleaned = stripEmoji(stripThink(text))
    .replace(/```[\s\S]*?```/g, ". ")
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    // Drop line-leading bullet / numbered-list markers so lists read as prose.
    .replace(/^[ \t]*(?:[-*•‣◦·]|\d+[.)])[ \t]+/gm, "")
    .replace(/[*_`#>~|]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return dropTrailingOffer(cleaned).slice(0, 1200);
}
