import { memo, useEffect, useRef, useState } from "react";
import { Markdown, StreamingContext } from "./Markdown";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { normalizeChannels } from "../lib/voiceText";

/** Split a streamed assistant message into its `<think>` reasoning and answer.
 *  A message can carry SEVERAL think blocks (interleaved reasoning, or a
 *  runaway that re-opened its thought channel): all block contents feed the
 *  reasoning panel, everything outside is answer, and only a trailing
 *  unclosed block counts as "still thinking". */
function parseThinking(raw: string): {
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
  hideThinking,
  sources,
}: {
  content: string;
  streaming: boolean;
  searching?: SearchKind;
  composing?: boolean;
  hideThinking?: boolean;
  sources?: { title: string; url: string; snippet: string }[];
}) {
  const { t } = useI18n();
  const busyHint = composing
    ? t("composing")
    : searching
      ? t(searching === "kb" ? "searchingKb" : searching === "mix" ? "searchingMix" : "searching")
      : null;
  const hasSources = (sources?.length ?? 0) > 0;

  // Thinking off: never use the reasoning panel — strip the whole <think> block
  // and any stray tags so a buggy/empty panel can never appear.
  if (hideThinking) {
    const answer = prepareCitations(
      normalizeChannels(content)
        .replace(/<think>[\s\S]*?<\/think>/g, "")
        .replace(/<think>[\s\S]*$/, "")
        .replace(/<\/?think>/g, ""),
      hasSources,
    ).replace(/^\s+/, "");
    return (
      <div className="bubble">
        <StreamingContext.Provider value={streaming}>
        {answer && (
          <div className="answer">
            <Markdown cites={sources}>{answer}</Markdown>
          </div>
        )}
        {busyHint && !answer ? (
          <span className="searching-hint">
            <span className="searching-spinner" />
            {busyHint}
          </span>
        ) : streaming && !answer ? (
          <span className="cursor" />
        ) : null}
        </StreamingContext.Provider>
      </div>
    );
  }

  const { reasoning, answer, thinking, hasThink } = parseThinking(content);
  const cleanAnswer = prepareCitations(answer, hasSources);
  // Manual override of the panel; until the user clicks, follow the thinking state
  // (expanded while reasoning, auto-collapsed once the answer starts).
  const [override, setOverride] = useState<boolean | null>(null);
  const expanded = override ?? thinking;
  const showThink = hasThink && (thinking || reasoning.length > 0);
  // Focus-follows-generation: while reasoning streams (and the user hasn't
  // manually expanded), show a small window pinned to the latest text with
  // the older lines fading out above.
  const focusMode = thinking && override === null;
  const thinkBodyRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (focusMode && thinkBodyRef.current) {
      thinkBodyRef.current.scrollTop = thinkBodyRef.current.scrollHeight;
    }
  }, [reasoning, focusMode]);

  return (
    <div className="bubble">
      <StreamingContext.Provider value={streaming}>
      {showThink && (
        <div className="think">
          <button
            className="think-toggle"
            type="button"
            onClick={() => setOverride(!expanded)}
          >
            <span className={`think-caret ${expanded ? "open" : ""}`}><Icon name="chevron-right" size={11} strokeWidth={2} /></span>
            {thinking ? (
              <span className="think-label is-thinking">
                {t("thinking")}
                <span className="dots">
                  <i />
                  <i />
                  <i />
                </span>
              </span>
            ) : (
              <span className="think-label">
                {expanded ? t("thoughtCollapse") : t("thoughtExpand")}
              </span>
            )}
          </button>
          {expanded && (
            <div ref={thinkBodyRef} className={`think-body ${focusMode ? "focus" : ""}`}>
              <Markdown>{reasoning}</Markdown>
            </div>
          )}
        </div>
      )}

      {cleanAnswer && (
        <div className="answer">
          <Markdown cites={sources}>{cleanAnswer}</Markdown>
        </div>
      )}

      {busyHint && !cleanAnswer && !thinking ? (
        <span className="searching-hint">
          <span className="searching-spinner" />
          {busyHint}
        </span>
      ) : streaming && !cleanAnswer && !thinking ? (
        <span className="cursor" />
      ) : null}
      </StreamingContext.Provider>
    </div>
  );
});
