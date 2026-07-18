import { memo, useEffect, useRef, useState } from "react";
import { Markdown } from "./Markdown";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { normalizeChannels } from "../lib/voiceText";
import { copyToClipboard } from "../lib/clipboard";

/** Hover copy-the-whole-reply button (markdown source, think block excluded). */
function CopyReply({ text }: { text: string }) {
  const [ok, setOk] = useState(false);
  const { t } = useI18n();
  if (!text) return null;
  return (
    <button
      className={`msg-copy ${ok ? "done" : ""}`}
      type="button"
      title={t("copyReply")}
      onClick={() =>
        void copyToClipboard(text).then(() => {
          setOk(true);
          setTimeout(() => setOk(false), 1400);
        })
      }
    >
      {ok ? (
        <svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 13l4 4L19 7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true">
          <rect x="9" y="9" width="11" height="11" rx="2" fill="none" stroke="currentColor" strokeWidth="1.7" />
          <path d="M5 15V5a2 2 0 0 1 2-2h8" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
        </svg>
      )}
    </button>
  );
}

/** Split a streamed assistant message into its `<think>` reasoning and answer. */
function parseThinking(raw: string): {
  reasoning: string;
  answer: string;
  thinking: boolean;
  hasThink: boolean;
} {
  // Channel-style reasoning markers (Gemma 4 / Harmony) → <think> convention.
  const content = normalizeChannels(raw);
  const open = "<think>";
  const close = "</think>";
  const oi = content.indexOf(open);
  if (oi === -1) {
    // Orphan close tag: reasoning streamed without an opening <think> (a
    // pre-open-trained model whose prompt lost the tag). Everything before
    // the close is reasoning.
    const ci0 = content.indexOf(close);
    if (ci0 !== -1) {
      return {
        reasoning: content.slice(0, ci0).trim(),
        answer: content.slice(ci0 + close.length).replace(/^\s+/, ""),
        thinking: false,
        hasThink: true,
      };
    }
    return { reasoning: "", answer: content, thinking: false, hasThink: false };
  }
  const afterOpen = oi + open.length;
  const ci = content.indexOf(close, afterOpen);
  if (ci === -1) {
    // Opening tag seen but not yet closed → still reasoning.
    return {
      reasoning: content.slice(afterOpen).replace(/^\n+/, ""),
      answer: "",
      thinking: true,
      hasThink: true,
    };
  }
  const reasoning = content.slice(afterOpen, ci).trim();
  const answer = (content.slice(0, oi) + content.slice(ci + close.length)).replace(/^\s+/, "");
  return { reasoning, answer, thinking: false, hasThink: true };
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
        .replace(/<\/?think>/g, ""),
      hasSources,
    ).replace(/^\s+/, "");
    return (
      <div className="bubble">
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
        {!streaming && <CopyReply text={answer} />}
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
      {!streaming && <CopyReply text={cleanAnswer} />}
    </div>
  );
});
