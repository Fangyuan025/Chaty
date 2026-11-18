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

import { callExample, renderCall } from "./callFormat";

export interface WrapupState {
  /** The turn's current plan (empty when the model never made one). */
  plan: TodoLike[];
  /** Step index of the last edit to a web-source file; -1 = none. */
  lastWebEditStep: number;
  /** Step index of the last browser_* action; -1 = none. */
  lastBrowserActionStep: number;
  /** A local dev server is known to be running (bash_bg / auto-converted). */
  serverCtx: boolean;
  /** An .html file was delivered this turn — the page is walkable even with
   *  no dev server (serve it or open it), so the browser note applies. */
  htmlEdited?: boolean;
  /** The dev server's URL when one was seen in command output. */
  devServerUrl?: string;
  /** Source-code edits since the last QUALIFYING execution (non-read-only
   *  bash, bash_bg, validate_change): file paths + rough changed-line volume.
   *  `ls`/`cat` after an edit is not verification — only running something is.
   *  The loop clears this on every qualifying SUCCESSFUL execution, and drops
   *  the page files a browser walkthrough of the local page covered — running
   *  a page is loading it; the browser note judges whether that was fresh. */
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
  /** A macOS-app delivery (SwiftUI @main / electron entry written this turn)
   *  with no packaged .app bundle anywhere in the workspace. */
  macAppMissingBundle?: boolean;
  /** The .app was packaged BEFORE the last app-source edit — the delivered
   *  bundle is not the delivered code (minesweeper audit: `swift test`
   *  refreshed debug, the release binary in the bundle stayed old). */
  macAppStaleBundle?: boolean;
  /** App-scale delivery whose functions were never EXECUTED: zero test runs,
   *  zero real invocations, zero browser walkthroughs. Compiling and
   *  launching is the entry ticket, not the bar (owner spec, all stacks). */
  functionalUnverified?: boolean;
  /** Ledger files that are page code without a server: plain scripts beside
   *  a page opened as a file (see `isPageScript`). They belong to the browser
   *  note, like any other page file. */
  pageFiles?: string[];
}

// ── Pages opened as files ──
// A page written this turn and opened from disk (file://) runs its scripts
// as surely as one behind a dev server does. Its plain .js used to count as
// page code only when a server was known, so a walkthrough of a file:// page
// cleared nothing and the run-check note fired after it, then fired again
// as "you still have not executed a single verification" — while
// validate_change, asked for by name, answered that page files are verified
// in the browser (Qwen3.5 4B, a single-page Y2K site: walked twice, held
// back twice, its answer lost).

/** One spelling for one file: workspace-absolute, `/`-separated, `.` and
 *  `..` resolved, lower-cased (macOS and Windows compare paths that way). */
export function normPath(p: string, workspace: string): string {
  let s = p.trim().replace(/\\/g, "/");
  const abs = s.startsWith("/") || /^[a-z]:\//i.test(s);
  if (!abs) s = `${workspace.replace(/\\/g, "/").replace(/\/+$/, "")}/${s}`;
  const drive = /^[a-z]:/i.exec(s)?.[0] ?? "";
  const out: string[] = [];
  for (const seg of s.slice(drive.length).split("/")) {
    if (!seg || seg === ".") continue;
    if (seg === "..") out.pop();
    else out.push(seg);
  }
  return `${drive}/${out.join("/")}`.toLowerCase();
}

/** The file a file:// URL names (query and fragment dropped), or null. */
export function fileUrlPath(url: string): string | null {
  const m = /^file:\/\/(?:localhost)?(\/[^?#]*)/i.exec(url.trim());
  if (!m) return null;
  let p = m[1];
  try {
    p = decodeURIComponent(p);
  } catch {
    /* keep it as written */
  }
  // file:///C:/x → C:/x
  return /^\/[a-z]:\//i.test(p) ? p.slice(1) : p;
}

/** The folder a normalized path sits in. */
export function dirOf(p: string): string {
  const i = p.lastIndexOf("/");
  return i <= 0 ? "/" : p.slice(0, i);
}

/** A plain script a page loads as it is — no build step in between. */
export function isPlainScript(path: string): boolean {
  return /\.(m?js|cjs)$/i.test(path);
}

/** Is this plain script page code: under the folder of a page written this
 *  turn or walked from disk? The same reach the server rule has always had
 *  (behind a server every .js is page code), drawn around the page instead
 *  of the whole workspace. */
export function isPageScript(path: string, workspace: string, pageDirs: Iterable<string>): boolean {
  if (!isPlainScript(path)) return false;
  const p = normPath(path, workspace);
  for (const d of pageDirs) if (p.startsWith(d === "/" ? "/" : `${d}/`)) return true;
  return false;
}

// ── The answer the gate held back ──
// The gate holds back an answer written for the user and asks for loose ends
// to be tied. What a small model writes next is often a reply to the gate,
// not to the user — "运行验证完成 … 所有检查已通过 ✅" — and that used to be
// all the user got: the answer to what they had asked was gone.

/** Below this an answer held back is a placeholder ("Done."), not one worth
 *  keeping beside what follows. */
const HELD_MIN = 24;

function trigrams(s: string): Set<string> {
  const t = s.toLowerCase().replace(/[\s*_`#>|\-—:：,，.。!！?？()（）[\]【】✅✓]+/g, "");
  const out = new Set<string>();
  for (let i = 0; i + 2 < t.length; i++) out.add(t.slice(i, i + 3));
  return out;
}

/** Above this share of shared character trigrams — of the held answer, or of
 *  the latest — the latest is the answer given again. Measured: a rewrite
 *  shares 0.64–0.73 (a condensed Chinese one is the low end, seen from its
 *  own side), a check report 0.07–0.22. Erring low shows both, erring high
 *  loses the answer, so the line sits nearer the rewrites. */
const SAME_ANSWER = 0.55;

/** What the user is shown when a turn ends after the gate held an answer
 *  back: the latest answer when it is that answer again (rewritten,
 *  condensed or extended), else the held answer with the latest after it —
 *  the check's outcome added to the answer rather than put in its place. */
export function answerAfterGate(held: string | null | undefined, latest: string): string {
  const h = (held ?? "").trim();
  const l = latest.trim();
  if (h.length < HELD_MIN || l === h || l.includes(h)) return latest;
  if (!l || h.includes(l)) return h;
  const hg = trigrams(h);
  const lg = trigrams(l);
  if (hg.size && lg.size) {
    let shared = 0;
    for (const g of hg) if (lg.has(g)) shared++;
    if (Math.max(shared / hg.size, shared / lg.size) >= SAME_ANSWER) return latest;
  }
  return `${h}\n\n${l}`;
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

/** Is a browser_navigate target the page being built — a local dev server or
 *  a local file — rather than somewhere on the web? The same hosts as
 *  browser.rs `is_local_page_url`, stricter on the rest: about:/data: pages
 *  and bare relative paths aren't the app, so walking them earns nothing; a
 *  scheme-less `localhost:5173` is, since navigation adds the http://. */
export function isLocalPageUrl(url: string): boolean {
  const u = url.trim().toLowerCase();
  if (u.startsWith("file://")) return true;
  let rest = u;
  if (/^https?:\/\//.test(u)) rest = u.replace(/^https?:\/\//, "");
  // Any other scheme (about:, data:, mailto:…) is not the app; the colon in
  // `localhost:5173` is a port, not a scheme.
  else if (/^[a-z][\w+.-]*:(?!\d)/.test(u)) return false;
  const hostPort = rest.split(/[/?#]/)[0];
  const host = hostPort.startsWith("[") ? hostPort.slice(1).split("]")[0] : hostPort.split(":")[0];
  return host === "localhost" || host.endsWith(".localhost") || host === "127.0.0.1" || host === "0.0.0.0" || host === "::1";
}

/** The URL a browser_navigate / browser_refresh result says it loaded — its
 *  first line, "Loaded: <url>" / "Reloaded (cache ignored): <url>" (or the
 *  zh wording), after redirects. Null when the result isn't one of those. */
export function loadedUrlFrom(result: string): string | null {
  const m = /^(?:Loaded|Reloaded \(cache ignored\)|已打开|已刷新\(忽略缓存\))[:：]\s*(\S+)/.exec(result.trimStart());
  return m ? m[1] : null;
}

/** A plan item that is about looking before doing: the workspace, the code. */
export const LOOKING =
  /了解|查看|看看|看一下|分析|梳理|调研|阅读|熟悉|摸清|检查(现有|工作区|项目|目录|代码)|\b(explore|understand|inspect|investigate|survey|review|look (at|into|around)|read (the|through)|check the (repo|workspace|project|code))\b/i;

/** Compact, model-facing echo of the plan. The old update_plan result was a
 *  bare "计划已更新" — the plan went to a UI panel and NEVER re-entered the
 *  model's context, which is exactly how todos became decoration. */
export function planEcho(todos: TodoLike[], lang: "zh" | "en"): string {
  const done = todos.filter((t) => t.status === "done").length;
  const cur = todos.find((t) => t.status === "in_progress");
  const pending = todos.filter((t) => t.status === "pending");
  const unfinished = todos.length - done;
  // A fresh plan (nothing done yet) gets a literal example call — the
  // strongest known steering for small models (missing-args ladder lesson);
  // "go execute" prose alone left them re-sending the plan (rounds 14/21).
  const first = todos.find((t) => t.status !== "done");
  // The example fits what the first item is: "了解当前工作区状态" answered with
  // "mkdir, or write the first file" told a 4B two things at once, and it
  // sent the plan again nine times while thinking "use understand_repo".
  const looking = !!first && LOOKING.test(first.content);
  // With its note: a model that copies the example exactly (a Qwen3.5 4B did)
  // copied its silence too, and the calls after went on without a word.
  const kick =
    done === 0 && first
      ? looking
        ? renderCall("understand_repo", { note: "…" })
        : renderCall("bash", { note: "…", command: "mkdir -p <项目目录>" })
      : "";
  if (lang === "zh") {
    let s = `计划已更新(已记录,无需重发):${done}/${todos.length} 完成`;
    if (cur) s += `;进行中:${cur.content}`;
    if (pending.length) s += `;待办 ${pending.length} 项`;
    s += "。";
    if (unfinished > 0) s += "下一步:直接用具体工具执行未完成事项,不要再调 update_plan,除非状态有变化。";
    if (kick) s += `现在就发第一个动作调用(${looking ? "先看工作区" : "建目录/写文件"}),把 note 的 … 换成你对用户说的一句话,格式如:${kick}`;
    return s;
  }
  let s = `Plan updated (recorded — no need to re-send): ${done}/${todos.length} done`;
  if (cur) s += `; in progress: ${cur.content}`;
  if (pending.length) s += `; ${pending.length} pending`;
  s += ".";
  if (unfinished > 0) s += " Next: execute the unfinished items with concrete tools; only call update_plan again when a status changes.";
  if (kick) s += ` Issue the first action call now (${looking ? "look at the workspace" : "mkdir / write the first file"}), with the note's … replaced by a sentence to the user, e.g.: ${kick}`;
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
    (st.serverCtx || st.lastBrowserActionStep >= 0 || st.htmlEdited === true);
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
  const pageFiles = new Set(st.pageFiles ?? []);
  const codeFiles = webNote
    ? st.codeEditsSinceExec.files.filter((f) => !isWebSourceFile(f, st.serverCtx) && !pageFiles.has(f))
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
          ? `- 这是第二次提醒:上一次收尾检查之后你仍然没有执行过任何验证,更新计划状态或补写文档都不算。现在就发一条 validate_change(不带参数即可)——它会自动构建/类型检查你写的代码并给出错误清单;如果它回答没有可跑的测试,就直接运行程序或一条最小冒烟命令。除非你能在答复里给出确凿理由说明这些代码无法在本机验证,否则不要直接交付。`
          : `- Second reminder: since the last wrap-up check you still have not executed a single verification — updating plan status or writing docs does not count. Issue one validate_change call right now (no arguments needed) — it will build/type-check what you wrote and hand you the error list; if it reports nothing to run, execute the program or a minimal smoke command instead. Do not deliver without it unless your answer gives a concrete reason these files cannot be verified on this machine.`,
      );
    } else {
      notes.push(
        zh
          ? `- 代码(${shown})在最近的改动之后没有任何运行验证——只读命令不算。至少做一样:用 validate_change 验证改动文件、直接运行它、或跑相关测试/最小冒烟;确认真的能跑再交付。如果确属无需运行的琐碎改动,在答复里说明原因。`
          : `- Code (${shown}) changed after the last run — read-only commands don't count as verification. Do at least one: validate_change the edited files, execute them directly, or run the relevant tests / a minimal smoke; confirm it actually runs before delivering. If it genuinely needs no run, say why in your answer.`,
      );
    }
  }

  // Functional bar, every stack: a build can be green and the app can even
  // launch while every actual FUNCTION is broken. Demand executed proof.
  if (st.functionalUnverified) {
    notes.push(
      zh
        ? `- 编译通过/能启动只是及格线:这次交付还没有任何一条基本功能被真正执行过——没有跑测试,没有用真实输入实跑程序,也没有浏览器走查。逐条执行核心功能并留证:核心逻辑测试(swift test / pytest / cargo test / npm test)、CLI 真实输入实跑、curl 探每个接口、或浏览器点一遍每个功能;方法参考 ${callExample("use_skill", '{"name":"debug-playbook"}')}。全部跑通再交付,答复里写明每条功能各自的验证方式。`
        : `- Compiling and launching is the entry ticket, not the bar: not one basic function of this delivery has actually been EXECUTED — no test run, no real-input invocation, no browser walkthrough. Exercise each core function and keep the proof: core-logic tests (swift test / pytest / cargo test / npm test), real CLI runs, curl on every endpoint, or a browser click-through of every feature; see ${callExample("use_skill", '{"name":"debug-playbook"}')}. Deliver only when they all pass, and name each function's proof in your answer.`,
    );
  }
  if (st.macAppStaleBundle) {
    notes.push(
      zh
        ? `- 打包的 .app 早于你最后的源码改动——工作区里的应用不包含你最新的修改,之前的启动验证验的是旧版本。现在:重新构建(注意 swift test 只刷新 debug 产物,release 包需要 swift build -c release)→ 重新打包 .app → 重新启动确认 LAUNCH OK,然后再交付。`
        : `- The packaged .app predates your last source edits — the bundle in the workspace does not contain your latest changes, and the earlier launch check verified the OLD version. Now: rebuild (note swift test only refreshes DEBUG products; a release bundle needs swift build -c release) → re-package the .app → re-launch for LAUNCH OK → then deliver.`,
    );
  }
  // macOS-app deliverable: the bar is a packaged .app that launches, not
  // sources that compile (owner spec). Independent of the run-check note —
  // a build can be green while nothing was ever packaged.
  if (st.macAppMissingBundle) {
    notes.push(
      zh
        ? `- 这是 macOS 应用任务:交付物是打包好的 .app 且启动验证过,但工作区里没有任何 .app(找不到 */Contents/MacOS)。构建 → 组装 .app → 启动确认存活(打印 LAUNCH OK)→ 再交付。完整配方:${callExample("use_skill", '{"name":"mac-app"}')}。`
        : `- This is a macOS app task: the deliverable is a packaged .app you have launch-verified, but the workspace has no .app bundle (no */Contents/MacOS). Build → assemble the .app → launch it and confirm it stays alive (LAUNCH OK) → then deliver. Full recipe: ${callExample("use_skill", '{"name":"mac-app"}')}.`,
    );
  }

  if (!notes.length) return null;
  return (
    (zh
      ? "[收尾检查] 先别交付,还有没闭环的事项:\n"
      : "[wrap-up check] Don't deliver yet — loose ends:\n") +
    notes.join("\n") +
    // What comes after the check is the answer the user reads: a model told
    // only "then answer" answered the check ("all checks passed ✅").
    (zh
      ? "\n处理完再给最终答复:完整回答用户原本的需求(做了什么、怎么用),验证情况放在后面作补充,不要只汇报检查结果。"
      : "\nHandle these, then give your final answer: a complete answer to what the user asked (what you did, how to use it), with the verification after it — not just a report that the checks passed.")
  );
}
