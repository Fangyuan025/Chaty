// Text helpers shared by the composer's read-aloud and the live voice mode.

const SOURCE_RE = /[【[（(]\s*来源\s*[\d０-９,，、\s]+[】\])）]/g;

const THINK = "\x00THINK\x00";
const FINAL = "\x00FINAL\x00";
const END = "\x00END\x00";

// Streaming buffers can end mid-marker (`<|`, `<|channel>thou`) — until the
// fragment resolves it must not render as answer text. Anything after the
// last `<` that is still a prefix of a known marker (optionally a channel
// opener awaiting its name) is held back; ordinary text tails (`a < b`,
// `<div`) are left alone.
const MARKER_WORDS = ["channel", "turn", "think", "message", "end", "return", "start"];
const CHANNEL_NAMES = ["analysis", "thought", "thinking", "final"];
function trimPartialMarker(s: string): string {
  const lt = s.lastIndexOf("<");
  if (lt === -1) return s;
  const tail = s.slice(lt);
  if (tail.includes(">")) {
    // Tag complete — only a channel opener still typing its name is partial.
    const m = /^<[|｜]?channel[|｜]?>[ \t]*([a-z]*)$/.exec(tail);
    if (m && CHANNEL_NAMES.some((n) => n !== m[1] && n.startsWith(m[1]))) return s.slice(0, lt);
    return s;
  }
  const body = tail.replace(/^<[|｜]?/, "");
  if (MARKER_WORDS.some((w) => w.startsWith(body) || (w + "|").startsWith(body) || (w + "｜").startsWith(body)))
    return s.slice(0, lt);
  return s;
}

/**
 * Translate "channel"-style reasoning markup (Harmony-like:
 * `<|channel|>thought<|message|>…<|channel|>final<|message|>…`, including
 * partially-rendered spellings like `<|channel>` / `<channel|>`) into the
 * `<think>…</think>` convention the rest of the app understands. Returns the
 * input untouched when no channel markers are present.
 *
 * A single generation can carry SEVERAL reasoning spans (a small model that
 * sails past its turn end re-opens a thought channel), so the conversion
 * walks the text sequentially: every thought span becomes its own
 * `<think>…</think>` block, text between spans stays answer text, and an
 * unterminated span is left open for the streaming UI.
 */
export function normalizeChannels(s: string): string {
  const held = trimPartialMarker(s);
  // Control-token words (think/message/end/return/start) require AT LEAST ONE
  // pipe — a plain `<think>` is the Qwen convention this function's OUTPUT
  // uses and must pass through untouched.
  if (
    !/<[|｜]?(channel|turn)\b|\b(channel|turn)[|｜]>|<(?:[|｜](?:think|message|end|return|start)|(?:think|message|end|return|start)[|｜])/i.test(
      held,
    )
  )
    return held;
  const t = held
    // Gemma 4 control tokens that should never render: <|think|>, turn markers.
    .replace(/<(?:[|｜]think[|｜]?|think[|｜])>\n?/gi, "")
    .replace(/<[|｜]?turn[|｜]?>[ \t]*(model|assistant|user|system|tool)?[ \t]*\n?/gi, "")
    .replace(/<(?:[|｜]start[|｜]?|start[|｜])>[ \t]*(assistant)?/gi, "")
    .replace(/<(?:[|｜](?:end|return)[|｜]?|(?:end|return)[|｜])>/gi, END)
    // Reasoning channel opener: <|channel>thought\n (Gemma 4) / Harmony
    // variants. The name must be terminated by a newline, a <|message|> tag,
    // or the buffer end — a bare close marker followed by prose that merely
    // STARTS with one of these words ("<channel|>Final result: …") is a close
    // plus ordinary text, not an opener.
    .replace(
      /<[|｜]?channel[|｜]?>[ \t]*(analysis|thought|thinking)(?:[ \t]*\n|<(?:[|｜]message[|｜]?|message[|｜])>|$)/g,
      THINK,
    )
    .replace(/<[|｜]?channel[|｜]?>[ \t]*final(?:[ \t]*\n|<(?:[|｜]message[|｜]?|message[|｜])>|$)/g, FINAL)
    .replace(/<(?:[|｜]message[|｜]?|message[|｜])>/gi, "")
    // Any remaining bare channel marker is a close: Gemma 4 ends thought with <channel|>.
    .replace(/<[|｜]?channel[|｜]?>/gi, END);
  const dropMarks = (x: string) => x.split(END).join("").split(FINAL).join("").split(THINK).join("");

  let out = "";
  let i = 0;
  for (;;) {
    const ti = t.indexOf(THINK, i);
    if (ti === -1) {
      out += dropMarks(t.slice(i));
      break;
    }
    out += dropMarks(t.slice(i, ti));
    const from = ti + THINK.length;
    const fi = t.indexOf(FINAL, from);
    const ei = t.indexOf(END, from);
    const close = fi === -1 ? ei : ei === -1 ? fi : Math.min(fi, ei);
    if (close === -1) {
      // Still reasoning (streaming) — leave the think block open.
      out += `<think>${dropMarks(t.slice(from))}`;
      break;
    }
    out += `<think>${dropMarks(t.slice(from, close))}</think>`;
    i = close + (close === fi ? FINAL.length : END.length);
  }
  return out;
}

/** Strip `<think>` reasoning and inline source markers; trim. A trailing
 *  unclosed block (generation cut mid-thought) is reasoning too — dropped. */
export function stripThink(s: string): string {
  return normalizeChannels(s)
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<think>[\s\S]*$/, "")
    .replace(/<\/?think>/g, "")
    .replace(SOURCE_RE, "")
    .trim();
}

/** The answer portion only — skip `<think>` reasoning so TTS never reads it. */
export function answerOnly(text: string): string {
  const t = normalizeChannels(text);
  const ci = t.lastIndexOf("</think>");
  if (ci === -1) return t.includes("<think>") ? "" : t; // reasoning still open
  const rest = t.slice(ci + "</think>".length);
  // A RE-OPENED thought after the last close (interleaved reasoning) must not
  // be spoken either.
  const oi = rest.indexOf("<think>");
  return oi === -1 ? rest : rest.slice(0, oi);
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
