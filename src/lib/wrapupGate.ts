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
  /** Source-code edits since the last QUALIFYING execution (non-read-only
   *  bash, bash_bg, validate_change): file paths + rough changed-line volume.
   *  `ls`/`cat` after an edit is not verification — only running something is.
   *  The loop clears this on every qualifying SUCCESSFUL execution. */
  codeEditsSinceExec: { files: string[]; lines: number };
  /** The most recent run/build/validation that failed with no green run
   *  after it (command or tool name) — escalates the run-check note into a
   *  hard "don't deliver on a red build". */
  lastFailedRun?: string | null;
  /** The gate already fired this turn (as many times as allowed). */
  nudged: boolean;
  /** Which firing this is (1-based). A repeat of the run-check note must not
   *  be verbatim — identical corrections don't break attractors (A/B-1). */
  attempt?: number;
}

/** Files whose edits deserve a "did you actually run it?" check. Docs and
 *  config are deliberately out — nudging a README edit is pure friction. */
export function isSourceCodeFile(path: string): boolean {
  return /\.(py|rs|go|ts|tsx|js|jsx|mjs|cjs|c|cc|cpp|h|hpp|java|rb|php|sh|bash|zsh|swift|kt|kts|scala|lua|pl|r|m|vue|svelte)$/i.test(
    path,
  );
}

/** The delivery bar for the run-check nudge: a single small edit stays
 *  frictionless; real work (multiple files, or ~a screenful of new code)
 *  earns a "run it before you ship it". */
const RUN_CHECK_MIN_FILES = 2;
const RUN_CHECK_MIN_LINES = 30;

/** Whether the un-verified edit volume clears the nudge bar — exported so the
 *  loop can grant the run-check note a second shot when the model ignored the
 *  first one entirely (CalendarApp repro round 9: it "handled" the nudge by
 *  ticking todos and re-delivering, still with zero runs). */
export function runCheckAboveBar(files: number, lines: number): boolean {
  return files > 0 && (files >= RUN_CHECK_MIN_FILES || lines >= RUN_CHECK_MIN_LINES);
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
  const webNote =
    st.lastWebEditStep >= 0 &&
    st.lastWebEditStep > st.lastBrowserActionStep &&
    (st.serverCtx || st.lastBrowserActionStep >= 0);
  if (webNote) {
    const target = st.devServerUrl;
    notes.push(
      zh
        ? `- 页面代码在最近的改动之后没有经过浏览器走查。用 browser_navigate 打开${target ? ` ${target}` : " dev server 的页面"},把改动涉及的路径实际点一遍,确认没有 [console] 报错再交付。`
        : `- The page code changed after the last browser check. browser_navigate to ${target ?? "the dev server"}, walk the paths your change touches, and confirm there are no [console] errors before delivering.`,
    );
  }

  // Code edits with no RUN afterwards (the non-web sibling of the check
  // above). One nudge, never a gate: below the volume bar it stays silent so
  // trivial tasks keep their flow, and the message itself offers the honest
  // way out — say why a run isn't needed. When the browser note already
  // fired, files it covers don't count twice.
  const codeFiles = webNote
    ? st.codeEditsSinceExec.files.filter((f) => !isWebSourceFile(f, st.serverCtx))
    : st.codeEditsSinceExec.files;
  if (
    codeFiles.length > 0 &&
    (codeFiles.length >= RUN_CHECK_MIN_FILES || st.codeEditsSinceExec.lines >= RUN_CHECK_MIN_LINES)
  ) {
    const shown = codeFiles.slice(0, 3).join(", ") + (codeFiles.length > 3 ? ", …" : "");
    if (st.lastFailedRun) {
      // A red build outranks everything: the model ran verification, saw it
      // fail, and is trying to deliver anyway. Name the failing run and
      // demand green — "syntax passed" / "--version worked" is not green.
      notes.push(
        zh
          ? `- 最近一次运行验证是失败的(${st.lastFailedRun}),之后没有任何一次成功的运行。不允许带着编译/构建错误交付:读失败输出、修复错误,重新运行同样的验证直到通过(exit 0)。语法检查(-parse)和 --version 不算验证。如果失败纯属环境问题,在答复里给出证据。`
          : `- Your most recent verification FAILED (${st.lastFailedRun}) and no successful run followed. Do not deliver with compile/build errors: read the failure output, fix it, and re-run the same verification until it passes (exit 0). Syntax-only checks (-parse) and --version are not verification. If the failure is genuinely environmental, show the evidence in your answer.`,
      );
    } else if ((st.attempt ?? 1) >= 2) {
      // Second firing with the ledger untouched: the model "handled" the
      // first nudge without attempting a single run (repro round 9: ticked
      // todos, re-delivered). Verbatim repeats don't land — name the refusal
      // and give one concrete order.
      notes.push(
        zh
          ? `- 这是第二次提醒:上一次收尾检查之后你仍然没有执行过任何验证,更新计划状态或补写文档都不算。现在就发一条 validate_change(不带参数即可)——它会自动构建/类型检查你写的代码并给出错误清单。除非你能在答复里给出确凿理由说明这些代码无法在本机验证,否则不要直接交付。`
          : `- Second reminder: since the last wrap-up check you still have not executed a single verification — updating plan status or writing docs does not count. Issue one validate_change call right now (no arguments needed) — it will build/type-check what you wrote and hand you the error list. Do not deliver without it unless your answer gives a concrete reason these files cannot be verified on this machine.`,
      );
    } else {
      notes.push(
        zh
          ? `- 代码(${shown})在最近的改动之后没有任何运行验证——只读命令不算。至少做一样:用 validate_change 验证改动文件、直接运行它、或跑相关测试/最小冒烟;确认真的能跑再交付。如果确属无需运行的琐碎改动,在答复里说明原因。`
          : `- Code (${shown}) changed after the last run — read-only commands don't count as verification. Do at least one: validate_change the edited files, execute them directly, or run the relevant tests / a minimal smoke; confirm it actually runs before delivering. If it genuinely needs no run, say why in your answer.`,
      );
    }
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
