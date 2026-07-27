/** Wrap-up gate: app-layer checks that run at the moment the model tries to
 *  END a turn — where the webapp audit found it cutting corners (todo lists
 *  written once and never honored; page edits shipped without ever opening
 *  the page). Pure decision logic, kept out of the loop so it's testable.
 *
 *  The gate fires AT MOST ONCE per turn (the loop tracks `nudged`): its job
 *  is to catch an oversight, not to argue with a model that has decided. */

interface TodoLike {
  content: string;
  status: string; // "pending" | "in_progress" | "done"
}

export interface WrapupState {
  /** The turn's current plan (empty when the model never made one). */
  plan: TodoLike[];
  /** Step index of the last edit to a web-source file; -1 = none. */
  lastWebEditStep: number;
  /** Step index of the last browser_* action; -1 = none. */
  lastBrowserActionStep: number;
  /** A local dev server is known to be running (bash_bg / auto-converted). */
  serverCtx: boolean;
  /** The dev server's URL when one was seen in command output. */
  devServerUrl?: string;
  /** The gate already fired this turn. */
  nudged: boolean;
}

/** Files whose edits are expected to be verifiable in a browser. Plain
 *  .ts/.js only count when a dev server is around — a CLI project's sources
 *  must not trip a "check it in the browser" nudge. */
export function isWebSourceFile(path: string, serverCtx: boolean): boolean {
  if (/\.(html?|css|scss|sass|less|tsx|jsx|vue|svelte)$/i.test(path)) return true;
  return serverCtx && /\.(ts|js|mjs|cjs)$/i.test(path);
}

/** First local-origin URL in command output (dev server banner). */
export function devServerUrlFrom(text: string): string | undefined {
  const m = text.match(/https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?(?:\/[^\s"'）)\]>]*)?/i);
  return m?.[0];
}

/** Compact, model-facing echo of the plan. The old update_plan result was a
 *  bare "计划已更新" — the plan went to a UI panel and NEVER re-entered the
 *  model's context, which is exactly how todos became decoration. */
export function planEcho(todos: TodoLike[], lang: "zh" | "en"): string {
  const done = todos.filter((t) => t.status === "done").length;
  const cur = todos.find((t) => t.status === "in_progress");
  const pending = todos.filter((t) => t.status === "pending");
  const unfinished = todos.length - done;
  if (lang === "zh") {
    let s = `计划已更新(已记录,无需重发):${done}/${todos.length} 完成`;
    if (cur) s += `;进行中:${cur.content}`;
    if (pending.length) s += `;待办 ${pending.length} 项`;
    s += "。";
    if (unfinished > 0) s += "下一步:直接用具体工具执行未完成事项,不要再调 update_plan,除非状态有变化。";
    return s;
  }
  let s = `Plan updated (recorded — no need to re-send): ${done}/${todos.length} done`;
  if (cur) s += `; in progress: ${cur.content}`;
  if (pending.length) s += `; ${pending.length} pending`;
  s += ".";
  if (unfinished > 0) s += " Next: execute the unfinished items with concrete tools; only call update_plan again when a status changes.";
  return s;
}

const listOf = (items: TodoLike[], zh: boolean): string => {
  const names = items.slice(0, 4).map((t) => `「${t.content}」`);
  const more = items.length > 4;
  return zh
    ? names.join("、") + (more ? " 等" : "")
    : names.join(", ") + (more ? ", …" : "");
};

/** The one-shot wrap-up nudge, or null when the turn may end. */
export function wrapupNudge(st: WrapupState, lang: "zh" | "en"): string | null {
  if (st.nudged) return null;
  const zh = lang === "zh";
  const notes: string[] = [];

  const undone = st.plan.filter((t) => t.status !== "done");
  if (undone.length) {
    notes.push(
      zh
        ? `- 待办清单还有 ${undone.length} 项未完成:${listOf(undone, true)}。逐项完成并用 update_plan 标记 done;哪一项确实不做了,也要更新状态并在答复里说明原因。`
        : `- The todo list still has ${undone.length} unfinished item(s): ${listOf(undone, false)}. Finish them and mark them done with update_plan; if one is genuinely out, update its status and say why in your answer.`,
    );
  }

  // Web edits with no browser look AFTERWARDS — only when there demonstrably
  // is something to look at (a live dev server, or the browser was already in
  // use this turn).
  if (
    st.lastWebEditStep >= 0 &&
    st.lastWebEditStep > st.lastBrowserActionStep &&
    (st.serverCtx || st.lastBrowserActionStep >= 0)
  ) {
    const target = st.devServerUrl;
    notes.push(
      zh
        ? `- 页面代码在最近的改动之后没有经过浏览器走查。用 browser_navigate 打开${target ? ` ${target}` : " dev server 的页面"},把改动涉及的路径实际点一遍,确认没有 [console] 报错再交付。`
        : `- The page code changed after the last browser check. browser_navigate to ${target ?? "the dev server"}, walk the paths your change touches, and confirm there are no [console] errors before delivering.`,
    );
  }

  if (!notes.length) return null;
  return (
    (zh
      ? "[收尾检查] 先别交付,还有没闭环的事项:\n"
      : "[wrap-up check] Don't deliver yet — loose ends:\n") +
    notes.join("\n") +
    (zh
      ? "\n处理完再给最终答复。"
      : "\nHandle these, then give your final answer.")
  );
}
