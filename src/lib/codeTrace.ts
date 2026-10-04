/** A Code-mode session as a trace: JSON Lines, one record a line, in the
 *  order things happened — the session, then each turn's request, every tool
 *  call with what the model wrote and what it was given back, and the answer.
 *  A step's card keeps a trimmed copy of its result; the trace carries the
 *  whole of it (and of a capped diff), read back from where the host keeps
 *  them. */

import type { ChatMessage } from "./ipc";
import type { PlanItem, ToolStep } from "./agentLoop";
import type { TurnChange } from "./turnChanges";

/** A session's turns, as Code mode holds them. */
export interface TraceMsg {
  id: string;
  role: "user" | "assistant";
  text: string;
  images?: string[];
  attachments?: { name: string; kind: string; path?: string }[];
  prompt?: ChatMessage[];
  thinking?: string;
  steps: ToolStep[];
  plan?: PlanItem[];
  compacted?: boolean;
  paused?: boolean;
  changes?: TurnChange[];
}

export interface TraceSession {
  id: string;
  title: string;
  workspace: string | null;
  model: string | null;
  app: string;
}

/** What the host kept for a step, by key: the step id (what the model was
 *  given) or `<id>:diff` (a capped diff, as JSON). */
export type StepTextLookup = (key: string) => Promise<string | null>;

export async function buildCodeTrace(
  session: TraceSession,
  msgs: TraceMsg[],
  stepText: StepTextLookup,
  now: Date = new Date(),
): Promise<string> {
  const lines: unknown[] = [{ type: "session", ...session, exportedAt: now.toISOString() }];
  let turn = 0;
  for (const m of msgs) {
    if (m.role === "user") {
      turn += 1;
      lines.push({
        type: "user",
        turn,
        id: m.id,
        text: m.text,
        ...(m.images?.length ? { images: m.images } : {}),
        ...(m.attachments?.length ? { attachments: m.attachments } : {}),
      });
      continue;
    }
    for (const [i, s] of m.steps.entries()) {
      // What the model was given: the whole of it when the card only kept a part.
      const full = s.fullText ? await stepText(s.id).catch(() => null) : null;
      let diff = s.diff;
      if (s.fullDiff) {
        const whole = await stepText(`${s.id}:diff`).catch(() => null);
        if (whole) {
          try {
            diff = JSON.parse(whole);
          } catch {
            /* keep the card's */
          }
        }
      }
      lines.push({
        type: "tool_call",
        turn,
        step: i + 1,
        id: s.id,
        name: s.call.name,
        args: s.call.args,
        status: s.status,
        ...(s.thinking ? { thinking: s.thinking } : {}),
        ...(full != null || s.result != null ? { result: full ?? s.result } : {}),
        ...(diff ? { diff } : {}),
        ...(s.diffCounts ? { diffCounts: s.diffCounts } : {}),
        ...(s.image ? { image: s.image } : {}),
      });
    }
    lines.push({
      type: "assistant",
      turn,
      id: m.id,
      text: m.text,
      ...(m.thinking ? { thinking: m.thinking } : {}),
      ...(m.plan?.length ? { plan: m.plan } : {}),
      ...(m.changes?.length ? { changes: m.changes.map(({ rel, added, removed, created, deleted, binary }) => ({ path: rel, added, removed, created, deleted, binary })) } : {}),
      ...(m.compacted ? { compacted: true } : {}),
      ...(m.paused ? { paused: true } : {}),
    });
    // The newest turn keeps the messages it ended on, as the model saw them.
    if (m.prompt?.length) lines.push({ type: "context", turn, messages: m.prompt });
  }
  return lines.map((l) => JSON.stringify(l)).join("\n") + "\n";
}
