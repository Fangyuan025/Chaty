import { useState } from "react";
import { Markdown } from "./Markdown";
import { useI18n } from "../lib/i18n";

/** Split a streamed assistant message into its `<think>` reasoning and answer. */
function parseThinking(content: string): {
  reasoning: string;
  answer: string;
  thinking: boolean;
  hasThink: boolean;
} {
  const open = "<think>";
  const close = "</think>";
  const oi = content.indexOf(open);
  if (oi === -1) {
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

export function AssistantMessage({
  content,
  streaming,
  searching,
}: {
  content: string;
  streaming: boolean;
  searching?: boolean;
}) {
  const { t } = useI18n();
  const { reasoning, answer, thinking, hasThink } = parseThinking(content);
  const cleanAnswer = answer.replace(SOURCE_RE, "");
  // Manual override of the panel; until the user clicks, follow the thinking state
  // (expanded while reasoning, auto-collapsed once the answer starts).
  const [override, setOverride] = useState<boolean | null>(null);
  const expanded = override ?? thinking;
  const showThink = hasThink && (thinking || reasoning.length > 0);

  return (
    <div className="bubble">
      {showThink && (
        <div className="think">
          <button
            className="think-toggle"
            type="button"
            onClick={() => setOverride(!expanded)}
          >
            <span className={`think-caret ${expanded ? "open" : ""}`}>▶</span>
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
            <div className="think-body">
              <Markdown>{reasoning}</Markdown>
            </div>
          )}
        </div>
      )}

      {cleanAnswer && (
        <div className="answer">
          <Markdown>{cleanAnswer}</Markdown>
        </div>
      )}

      {searching && !cleanAnswer && !thinking ? (
        <span className="searching-hint">
          <span className="searching-spinner" />
          {t("searching")}
        </span>
      ) : streaming && !cleanAnswer && !thinking ? (
        <span className="cursor" />
      ) : null}
    </div>
  );
}
