// Text helpers shared by the composer's read-aloud and the live voice mode.

const SOURCE_RE = /[【[（(]\s*来源\s*[\d０-９,，、\s]+[】\])）]/g;

const THINK = "\x00THINK\x00";
const FINAL = "\x00FINAL\x00";
const END = "\x00END\x00";

/**
 * Translate "channel"-style reasoning markup (Harmony-like:
 * `<|channel|>thought<|message|>…<|channel|>final<|message|>…`, including
 * partially-rendered spellings like `<|channel>` / `<channel|>`) into the
 * `<think>…</think>` convention the rest of the app understands. Returns the
 * input untouched when no channel markers are present.
 */
export function normalizeChannels(s: string): string {
  if (!/<[|｜]?(channel|turn)\b|\bchannel[|｜]>|\bturn[|｜]>/i.test(s)) return s;
  const t = s
    // Gemma 4 control tokens that should never render: <|think|>, turn markers.
    .replace(/<[|｜]?think[|｜]?>\n?/gi, "")
    .replace(/<[|｜]?turn[|｜]?>\s*(model|assistant|user|system|tool)?\s*\n?/gi, "")
    .replace(/<[|｜]?start[|｜]?>\s*(assistant)?/gi, "")
    .replace(/<[|｜]?(end|return)[|｜]?>/gi, END)
    // Reasoning channel opener: <|channel>thought\n (Gemma 4) / Harmony variants.
    .replace(/<[|｜]?channel[|｜]?>\s*(analysis|thought|thinking)\w*[ \t]*\n?(<[|｜]?message[|｜]?>)?/gi, THINK)
    .replace(/<[|｜]?channel[|｜]?>\s*final[ \t]*\n?(<[|｜]?message[|｜]?>)?/gi, FINAL)
    .replace(/<[|｜]?message[|｜]?>/gi, "")
    // Any remaining bare channel marker is a close: Gemma 4 ends thought with <channel|>.
    .replace(/<[|｜]?channel[|｜]?>/gi, END);
  const dropMarks = (x: string) => x.split(END).join("").split(FINAL).join("").split(THINK).join("");

  const ti = t.indexOf(THINK);
  if (ti === -1) return dropMarks(t);
  const afterThink = ti + THINK.length;
  const fi = t.indexOf(FINAL, afterThink);
  const ei = t.indexOf(END, afterThink);
  const closeAt = fi !== -1 ? fi : ei;
  const pre = dropMarks(t.slice(0, ti));
  if (closeAt === -1) {
    // Still reasoning (streaming) — leave the think block open.
    return `${pre}<think>${dropMarks(t.slice(afterThink))}`;
  }
  const answerStart = closeAt + (fi !== -1 ? FINAL.length : END.length);
  const reasoning = dropMarks(t.slice(afterThink, closeAt));
  const answer = dropMarks(t.slice(answerStart));
  return `${pre}<think>${reasoning}</think>${answer}`;
}

/** Strip `<think>` reasoning and inline source markers; trim. */
export function stripThink(s: string): string {
  return normalizeChannels(s)
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<\/?think>/g, "")
    .replace(SOURCE_RE, "")
    .trim();
}

/** The answer portion only — skip `<think>` reasoning so TTS never reads it. */
export function answerOnly(text: string): string {
  const t = normalizeChannels(text);
  const ci = t.lastIndexOf("</think>");
  if (ci !== -1) return t.slice(ci + "</think>".length);
  if (t.includes("<think>")) return ""; // reasoning still open
  return t;
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

/** Strip a model's reasoning/quotes from a generated title and clamp length. */
export function cleanTitle(raw: string): string {
  // answerOnly normalizes channel-style reasoning (Gemma 4) into <think> and
  // returns only the answer — empty if the model never left its reasoning,
  // in which case callers keep their default title rather than show thought text.
  const t = answerOnly(raw);
  const firstLine = t.split("\n").map((s) => s.trim()).find(Boolean) ?? "";
  return firstLine
    .replace(/^["'「『《<[(]+|["'」』》>\])。.!！?？:：]+$/g, "")
    .trim()
    .slice(0, 24);
}
