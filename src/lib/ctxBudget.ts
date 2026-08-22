/** How much of the context window a conversation occupies.
 *
 *  Chat and Code used to answer this question with two different guesses. One
 *  counted every CJK character as a token and divided the rest by 3.6; the
 *  other divided everything by 2.5 and knew nothing about scripts. Measured
 *  against a real tokenizer both were wrong, in opposite directions and on
 *  different content: the CJK-blind one read dense Chinese at 0.24x of its true
 *  cost, and the CJK-aware one read source code at 0.67x. Under-reading is the
 *  dangerous direction — compaction fires late, or never.
 *
 *  The fix is not a better guess. Every reply carries the engine's own
 *  `promptTokens`, so after the first turn the exact cost of a prompt we can
 *  measure ourselves is known. `calibrate` folds that truth back in, and the
 *  estimate stops being a heuristic and becomes a correction on a heuristic —
 *  per model, per conversation, whatever the language or content.
 */

/** Message framing (role markers, turn delimiters) the text itself doesn't show. */
const PER_MESSAGE_OVERHEAD = 8;

function isWide(code: number): boolean {
  return (
    (code >= 0x3000 && code <= 0x9fff) || // CJK punctuation, kana, ideographs
    (code >= 0xac00 && code <= 0xd7a3) || // Hangul syllables
    (code >= 0xf900 && code <= 0xfaff) || // CJK compatibility ideographs
    (code >= 0xff00 && code <= 0xffef) || // full-width forms
    (code >= 0x20000 && code <= 0x2ebef) // rare ideograph planes: byte-fallback territory
  );
}

/** The uncalibrated cost of a string. Deliberately pessimistic — being early
 *  to compact costs a summary; being late costs the turn. */
export function rawTokens(text: string): number {
  let wide = 0;
  let rest = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0) ?? 0;
    if (c > 0xffff && !isWide(c)) {
      // Emoji and other astral characters are several tokens each.
      rest += 8;
    } else if (isWide(c)) {
      wide++;
    } else {
      rest++;
    }
  }
  // A rare ideograph can cost more than one token, so wide characters are not
  // charged at parity; Latin runs at ~3.3 chars/token once punctuation and
  // indentation are in the mix, which is where code lives.
  return Math.ceil(wide * 1.2 + rest / 3.3);
}

// Running ratio between what the engine charged and what we predicted. Kept as
// two sums rather than a ratio so a single odd turn cannot swing it.
let predicted = 0;
let charged = 0;

/** Forget the calibration — a different model tokenizes differently. */
export function resetCalibration(): void {
  predicted = 0;
  charged = 0;
}

/** Fold in one turn of ground truth: what we predicted a prompt would cost,
 *  and what the engine actually charged for it. */
export function calibrate(predictedTokens: number, actualTokens: number): void {
  if (predictedTokens <= 0 || actualTokens <= 0) return;
  // Decay, so the ratio follows a conversation that changes character
  // (a Chinese chat that turns into a code review) instead of averaging it away.
  predicted = predicted * 0.75 + predictedTokens;
  charged = charged * 0.75 + actualTokens;
}

/** What the running calibration says our raw estimate is off by. 1 until the
 *  first reply lands; clamped so one strange turn cannot make the budget wild. */
export function calibrationFactor(): number {
  if (predicted <= 0 || charged <= 0) return 1;
  return Math.min(4, Math.max(0.5, charged / predicted));
}

/** Calibrated cost of a string. */
export function textTokens(text: string): number {
  return Math.ceil(rawTokens(text) * calibrationFactor());
}

/** Calibrated cost of a conversation, framing included. */
export function messageTokens(messages: { content: string }[]): number {
  const raw = messages.reduce((n, m) => n + rawTokens(m.content) + PER_MESSAGE_OVERHEAD, 0);
  return Math.ceil(raw * calibrationFactor());
}

/** Uncalibrated cost of a conversation — what to hand `calibrate` alongside
 *  the engine's `promptTokens`, so the ratio does not chase itself. */
export function rawMessageTokens(messages: { content: string }[]): number {
  return messages.reduce((n, m) => n + rawTokens(m.content) + PER_MESSAGE_OVERHEAD, 0);
}

/**
 * How much of the window the transcript may occupy before it must be compacted.
 *
 * Both modes used to answer this differently — code mode took a flat 80% of the
 * window and ignored the generation cap entirely, chat mode subtracted
 * `maxTokens + 700` but fell back to a bare 2048 whenever the user switched the
 * cap off. So the same conversation compacted at two different places depending
 * on which screen it was on, and a large generation cap could leave code mode
 * with no room to actually reply. One rule now serves both.
 *
 * The reserve is what the reply needs plus ~700 for the system prompt and the
 * chat template's own framing. A very large cap is not allowed to eat more than
 * a quarter of the window: a model permitted 32k of output in a 32k window would
 * otherwise leave nothing for the conversation that prompted it.
 */
export function contextLimit(nCtx: number, maxGenTokens?: number): number {
  const want = maxGenTokens && maxGenTokens > 0 ? maxGenTokens : 2048;
  const reserve = Math.min(Math.max(want, 512), Math.floor(nCtx * 0.25)) + 700;
  return Math.max(Math.floor(nCtx * 0.5), nCtx - reserve);
}
