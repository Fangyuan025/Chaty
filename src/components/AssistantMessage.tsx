import { memo, useEffect, useRef, useState } from "react";
import { useHeightOnChange } from "../lib/smoothHeight";
import { Markdown, StreamingContext } from "./Markdown";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { PrefillRing } from "./PrefillRing";
import { normalizeChannels, withoutToolCallSpans } from "../lib/voiceText";
import { parseThinking } from "../lib/reasoning";

const SOURCE_RE = /[【[（(]\s*来源\s*[\d０-９,，、\s]+[】\])）]/g;
/** Verbose citation forms (【来源1、2】 / (source 3)) → bare 【1】【2】 anchors. */
const SOURCE_WORDY_RE = /[【[（(]\s*(?:来源|source)\s*([\d,，、\s]+)[】\])）]/gi;

/** Prepare the answer text for citation rendering: with sources, normalise
 *  wordy markers into 【N】 (picked up by Markdown's cite anchors); without,
 *  strip them entirely as before. */
function prepareCitations(answer: string, hasSources: boolean): string {
  if (!hasSources) return answer.replace(SOURCE_RE, "");
  return answer.replace(SOURCE_WORDY_RE, (_, nums: string) =>
    nums
      .split(/[,，、\s]+/)
      .filter(Boolean)
      .map((n) => `【${n}】`)
      .join(""),
  );
}

export type SearchKind = "" | "web" | "kb" | "mix";

export const AssistantMessage = memo(function AssistantMessage({
  content,
  streaming,
  searching,
  composing,
  prefill,
  hideThinking,
  sources,
  onSkipThink,
}: {
  content: string;
  streaming: boolean;
  searching?: SearchKind;
  composing?: boolean;
  /** How far a long prompt has been read (0..1) before the first token. */
  prefill?: number | null;
  hideThinking?: boolean;
  sources?: { title: string; url: string; snippet: string }[];
  /** The reply being written here: its thought can be cut short and the
   *  answer begun (Skip, shown on hover while it thinks). */
  onSkipThink?: () => void;
}) {
  const { t } = useI18n();
  const busyHint = composing
    ? t("composing")
    : searching
      ? t(searching === "kb" ? "searchingKb" : searching === "mix" ? "searchingMix" : "searching")
      : null;
  const hasSources = (sources?.length ?? 0) > 0;
  // A reply rendered block by block re-renders only the block being written
  // (see Markdown) — the way for one that streams. One already finished when
  // it came on screen renders whole: the same page at under half the cost,
  // which is most of what opening a long conversation spent. A reply that
  // streamed here keeps its blocks once it ends, so finishing does not
  // rebuild it, and the previews in it, from scratch.
  const streamedHere = useRef(false);
  if (streaming) streamedHere.current = true;
  const blocks = streamedHere.current;

  const { reasoning, answer, thinking: open, hasThink } = parseThinking(withoutToolCallSpans(content));
  // An unclosed think block is "thinking" only while the reply streams. One
  // that was stopped, or ended by an error, is finished reasoning — it kept a
  // "Thinking" label and its dots running forever after (issue #18).
  const thinking = open && streaming;
  // Manual override of the panel; until the user clicks, follow the thinking state
  // (expanded while reasoning, auto-collapsed once the answer starts).
  const [override, setOverride] = useState<boolean | null>(null);
  // Focus-follows-generation: while reasoning streams (and the user hasn't
  // manually expanded), show a small window pinned to the latest text with
  // the older lines fading out above.
  const focusMode = thinking && override === null && !hideThinking;
  const thinkBodyRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (focusMode && thinkBodyRef.current) {
      thinkBodyRef.current.scrollTop = thinkBodyRef.current.scrollHeight;
    }
  }, [reasoning, focusMode]);
  // Opened or closed — by a click, or folding itself away as the answer
  // starts — the thought goes from one height to the other: the answer under
  // it used to jump up by the window's height as its first words came.
  const thinkRef = useHeightOnChange<HTMLDivElement>(override ?? thinking);
  // Asked once, Skip steps aside: the thought ends at the model's next token.
  const [skipAsked, setSkipAsked] = useState(false);
  useEffect(() => {
    if (!thinking) setSkipAsked(false);
  }, [thinking]);

  // Thinking off: never use the reasoning panel — strip the whole <think> block
  // and any stray tags so a buggy/empty panel can never appear.
  if (hideThinking) {
    const answer = prepareCitations(
      normalizeChannels(withoutToolCallSpans(content))
        .replace(/<think>[\s\S]*?<\/think>/g, "")
        .replace(/<think>[\s\S]*$/, "")
        .replace(/<\/?think>/g, ""),
      hasSources,
    ).replace(/^\s+/, "");
    return (
      <div className="bubble">
        <StreamingContext.Provider value={streaming}>
        {answer && (
          <div className="answer" data-copy={answer}>
            <Markdown cites={sources} blocks={blocks}>{answer}</Markdown>
          </div>
        )}
        {busyHint && !answer ? (
          <span className="searching-hint">
            <span className="searching-spinner" />
            {busyHint}
          </span>
        ) : streaming && !answer && prefill != null ? (
          <span className="searching-hint">
            <PrefillRing frac={prefill} />
            <span className="prefill-label">{t("cmPrefill")}</span>
          </span>
        ) : streaming && !answer ? (
          <span className="cursor" />
        ) : null}
        </StreamingContext.Provider>
      </div>
    );
  }

  const cleanAnswer = prepareCitations(answer, hasSources);
  const expanded = override ?? thinking;
  // Nothing written yet. With thinking on, the engine sends the opened
  // thought before it reads the prompt, and the panel that opened — "Thinking…"
  // with nothing under it — stood where the reading progress belongs, or
  // flashed up before it: the first progress comes once a chunk is read. The
  // panel waits for the first words of the thought.
  const opening = streaming && !reasoning && !cleanAnswer;
  const showThink = hasThink && (thinking || reasoning.length > 0) && !opening;

  return (
    <div className="bubble">
      <StreamingContext.Provider value={streaming}>
      {showThink && (
        <div className="think" ref={thinkRef}>
          <div className="think-head">
            <button
              className="think-toggle"
              type="button"
              onClick={() => setOverride(!expanded)}
            >
              <span className={`think-caret ${expanded ? "open" : ""}`}><Icon name="chevron-right" size={11} strokeWidth={2} /></span>
              {thinking ? (
                <span className="think-label is-thinking">{t("thinking")}</span>
              ) : (
                <span className="think-label">
                  {expanded ? t("thoughtCollapse") : t("thoughtExpand")}
                </span>
              )}
            </button>
            {onSkipThink && (
              <button
                type="button"
                className={`think-skip${thinking && !skipAsked ? " on" : ""}`}
                title={t("skipThinkTitle")}
                tabIndex={thinking && !skipAsked ? 0 : -1}
                aria-hidden={thinking && !skipAsked ? undefined : true}
                onClick={() => {
                  if (!thinking || skipAsked) return;
                  setSkipAsked(true);
                  onSkipThink();
                }}
              >
                {t("skipThink")}
              </button>
            )}
          </div>
          {expanded && (
            <div ref={thinkBodyRef} className={`think-body ${focusMode ? "focus" : ""}`}>
              <Markdown blocks={blocks}>{reasoning}</Markdown>
            </div>
          )}
        </div>
      )}

      {cleanAnswer && (
        <div className="answer" data-copy={cleanAnswer}>
          <Markdown cites={sources} blocks={blocks}>{cleanAnswer}</Markdown>
        </div>
      )}

      {busyHint && !cleanAnswer && !thinking ? (
        <span className="searching-hint">
          <span className="searching-spinner" />
          {busyHint}
        </span>
      ) : opening && prefill != null ? (
        <span className="searching-hint">
          <PrefillRing frac={prefill} />
          <span className="prefill-label">{t("cmPrefill")}</span>
        </span>
      ) : streaming && !cleanAnswer && (!thinking || opening) ? (
        <span className="cursor" />
      ) : null}
      </StreamingContext.Provider>
    </div>
  );
});
