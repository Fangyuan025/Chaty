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
/**
 * What an attached image costs the window.
 *
 * A picture is not free context: downscaled to the caps the engines use, one
 * screenshot reaches the model as roughly a thousand visual tokens — measured at
 * 1038 prompt tokens for a single 1 MP image plus 38 of text on MLX Qwen3.5, and
 * about 340 for Gemma-4's 2 MP cap on llama.cpp. Counting only the text left a
 * conversation carrying a few screenshots calling itself comfortable while it
 * was already over the window, and it also poisoned the calibration: the engine
 * reports the visual tokens in `promptTokens`, so a prediction that ignores them
 * reads as a wild under-estimate and pushes the ratio up for every text-only
 * turn after it. The high end is the safe one to charge — over-reserving costs a
 * summary, under-reserving costs the turn — and the calibration settles it
 * against what the engine actually charges.
 */
const IMAGE_TOKENS = 1000;

type Countable = { content: string; images?: string[]; reasoning_content?: string };

function rawMessageCost(m: Countable): number {
  return (
    rawTokens(m.content) +
    // Reasoning carried in its own field is still tokens in the prompt. The
    // templates that read it from there are exactly the ones it gets split out
    // for, and they render it straight back — but the budget counted only
    // `content`, so a reasoning-heavy history read as very nearly free.
    // Measured on Qwen3.8 27B, which does split it: an 8k conversation walked
    // off the end of its own window — six turns answered "context" with
    // nothing generated — while the budget still reported room to spare. Same
    // shape as pictures counting for nothing before 2.1.2.
    rawTokens(m.reasoning_content ?? "") +
    PER_MESSAGE_OVERHEAD +
    (m.images?.length ?? 0) * IMAGE_TOKENS
  );
}

export function messageTokens(messages: Countable[]): number {
  const raw = messages.reduce((n, m) => n + rawMessageCost(m), 0);
  return Math.ceil(raw * calibrationFactor());
}

/** Uncalibrated cost of a conversation — what to hand `calibrate` alongside
 *  the engine's `promptTokens`, so the ratio does not chase itself. */
export function rawMessageTokens(messages: Countable[]): number {
  return messages.reduce((n, m) => n + rawMessageCost(m), 0);
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

/**
 * Fit a run of transcript lines into `maxTokens` for a summarisation pass.
 *
 * Chat mode used to do this with `transcript.slice(-7000)` — a flat character
 * cap that kept the END of the stretch being summarised. That threw away the
 * opening of the conversation, which is where the user states the goal, the
 * constraints and the preferences, and which nothing later restates; and the
 * recent end was already being kept verbatim as the tail, so the summary spent
 * its budget describing what the model could still read. It also cut mid-word,
 * handing the summariser a sentence fragment as its first line.
 *
 * When everything fits, nothing is dropped. When it does not, the opening is
 * kept in full and the MIDDLE is elided, on line boundaries, with a marker so
 * the summariser knows a gap is there rather than inventing continuity across
 * it.
 */
export function fitTranscript(rawLines: string[], maxTokens: number, lang: "zh" | "en"): string {
  const join = (xs: string[]) => xs.join("\n");
  if (!rawLines.length) return "";
  if (textTokens(join(rawLines)) <= maxTokens) return join(rawLines);

  // No single turn may eat the whole budget. One 4000-character tool result is
  // a line like any other here, and left whole it would crowd out every other
  // turn — the summariser would see one file dump instead of the shape of the
  // work. Its opening carries what matters (the path, the first rows, the
  // error); the middle of a dump does not.
  const perLine = Math.max(120, Math.floor(maxTokens * 0.25));
  const clip = lang === "zh" ? "…(此条已截断)" : "…(truncated)";
  const lines = rawLines.map((l) => {
    if (textTokens(l) <= perLine) return l;
    // textTokens is an estimate; walk back from a character budget derived
    // from it rather than trusting a single division.
    let cut = Math.max(80, Math.floor((l.length * perLine) / Math.max(1, textTokens(l))));
    while (cut > 80 && textTokens(l.slice(0, cut)) > perLine) cut = Math.floor(cut * 0.9);
    return l.slice(0, cut) + clip;
  });

  const marker = lang === "zh" ? "……(中间若干轮已省略)……" : "…(some turns omitted here)…";
  // The opening gets the larger share: it is the part that never comes back.
  const headBudget = Math.floor(maxTokens * 0.6);
  const head: string[] = [];
  let used = 0;
  for (const line of lines) {
    const c = textTokens(line);
    if (used + c > headBudget && head.length) break;
    head.push(line);
    used += c;
  }
  const tail: string[] = [];
  let tailUsed = textTokens(marker);
  for (let i = lines.length - 1; i >= head.length; i--) {
    const c = textTokens(lines[i]);
    if (used + tailUsed + c > maxTokens) break;
    tail.unshift(lines[i]);
    tailUsed += c;
  }
  if (!tail.length) return join([...head, marker]);
  return join([...head, marker, ...tail]);
}

/**
 * A summary chat mode has already written for a conversation: the text, how
 * many leading messages it stands in for, and a fingerprint of exactly those
 * messages.
 */
export type Compacted = { summary: string; covered: number; print: string };

/** Fingerprint of the leading `upto` messages — cheap, and enough to notice an
 *  edit or a regenerate rewriting the stretch a summary was written from. */
export function historyPrint(msgs: { content: string }[], upto: number): string {
  return `${upto}:${msgs.slice(0, upto).reduce((n, m) => n + m.content.length, 0)}`;
}

/**
 * Does a summary already written still stand?
 *
 * Chat mode used to re-derive its compaction summary on every single turn. The
 * summary rides in the system message — the opening of the prompt — so a new
 * wording each turn moved every token behind it, and the engine could match
 * nothing it had already computed. Measured on Gemma-4 26B through MLX: 99% of
 * the window reused on the turns before compaction began, then 0% on every
 * turn after it, plus a whole extra generation per turn to write the summary
 * again. Code mode had already learned this (`compactMessages` compacts in
 * place and leaves real headroom, so it is not re-run every round); this is the
 * same lesson on the chat side.
 *
 * Returns the tail to send beside the standing summary, or null when a new
 * summary has to be written: the stretch it covered has changed underneath it,
 * or the conversation has since grown past the room it left.
 */
export function standingTail<T extends { content: string; reasoning_content?: string }>(
  memo: Compacted | null | undefined,
  msgs: T[],
  budget: number,
  /** The summary as it will actually appear in the prompt, prefix and all. */
  summaryText: string,
): T[] | null {
  if (!memo) return null;
  if (memo.covered > msgs.length) return null;
  if (historyPrint(msgs, memo.covered) !== memo.print) return null;
  const tail = msgs.slice(memo.covered);
  const cost =
    messageTokens([{ content: summaryText }]) +
    messageTokens(tail);
  return cost <= budget * 0.85 ? tail : null;
}
