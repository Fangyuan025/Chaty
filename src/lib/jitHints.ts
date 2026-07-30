/** Just-in-time hints: guidance that used to sit in the system prompt on
 *  every step now rides into the conversation only when the situation it
 *  teaches actually occurs — appended INSIDE the tool_result content, so it
 *  inherits compaction and the untrusted-content wrapper automatically.
 *  Shown-tracking is per turn: a hint can reappear next turn (cheap), which
 *  also self-heals when compaction elides the earlier copy. */

export type HintKey = "browser" | "editFail" | "webFlow" | "afterOrient" | "anchorRead";

const BROWSER_HINT: Record<"zh" | "en", string> = {
  zh: `[浏览器提示] 内容/状态用 browser_read(交互返回已带最新页面文字,通常直接看返回即可);外观/布局/图片/颜色必须 snapshot 或 screenshot 用视觉亲眼看,读文字看不出;但**截图只证明外观**——验收按钮/表单/跳转等交互功能,必须 browser_click/browser_type 实际操作一遍,再看返回文字或 browser_read 确认结果,截完图就收工不算测试。每次交互后先核实结果再继续,绝不凭猜测连续操作。已想好顺序的多次点击或多字段填表,一次用 steps 传完,不要拆成多次调用;点「提交/检查」等不可逆按钮前,先 snapshot 视觉确认所选内容无误。点导航/提交类按钮后,看返回的最新文字判断成败——成功就继续,绝不重复点同一按钮;翻页要点一次、读一次确认。CSS 选择器只支持标准语法(没有 :contains):按文字定位用 text 参数。浏览器是持久配置,之前登录过的站点仍在登录态。文字摘要对不上预期或页面似乎没反应时,不要反复 read 或硬猜,立即截图亲眼确认。任务做完 browser_close。`,
  en: `[Browser hint] For content/state use browser_read (interaction results already carry the latest page text — usually just read the return). For looks/layout/images/colors you MUST snapshot or screenshot and see it with vision — text can't show those; but a screenshot only proves LOOKS — verify buttons/forms/navigation by actually clicking/typing them and reading the result text; a screenshot alone is not a test. Verify the outcome after every interaction before acting again; never chain actions on guesses. When you already know a sequence of clicks or fields, pass them ALL in one steps call; before any irreversible submit/check button, snapshot first to visually confirm your selections. After a navigation/submit click, judge success from the returned text — if it worked, move on and NEVER click the same button again; paginate as click-once-read-once. CSS selectors are standard-syntax only (no :contains) — locate by visible text via the text arg. The profile is persistent — logins survive. If the text doesn't match expectations, stop guessing and screenshot. browser_close when done.`,
};

const EDIT_FAIL_HINT: Record<"zh" | "en", string> = {
  zh: `[编辑提示] old_string 匹配失败的恢复法:先用 read_file 重读目标区域,以文件**当前**内容为准(你记忆里的内容可能已过时);从读到的内容或上面的"最相似位置"里**逐字**复制(含空格与缩进)作为 old_string;同一文件多处修改用 edits 数组一次原子提交。`,
  en: `[Edit hint] Recovering from a failed old_string match: re-read the target region with read_file and trust the file's CURRENT content (your memory of it may be stale); copy old_string VERBATIM (spaces and indentation included) from what you just read or from the closest-match shown above; for several changes in one file use one atomic edits array.`,
};

// understand_repo is legitimately called with {} — and in no-think mode the
// next call tends to copy that empty-arguments shape (observed as a
// deterministic `search_code {}` spiral in the A/B-1 bench autopsy). One
// concrete next-step example right after orientation breaks the momentum.
const AFTER_ORIENT_HINT: Record<"zh" | "en", string> = {
  zh: `[下一步] 接下来的工具调用都要带具体 arguments,例如 search_code {"query":"哪里处理登录鉴权"} 或 read_file {"path":"src/app.ts"}——不要发出空 arguments 的调用。`,
  en: `[Next] Every following tool call needs concrete arguments, e.g. search_code {"query":"where login auth is handled"} or read_file {"path":"src/app.ts"} — never issue a call with empty arguments.`,
};

// Anchor mode: the first anchored read is where the model decides HOW it will
// edit. Two of three anchor-bench runs on the same task explored, read, and
// then quit without ever touching an editor — the docs alone don't bridge
// "these prefixes" to "this is how you edit". Say it at the moment of reading.
const ANCHOR_READ_HINT: Record<"zh" | "en", string> = {
  zh: `[编辑提示] 行首的 "行号:哈希→" 是编辑锚点。要修改这个文件,直接调用 edit_lines,把锚点原样抄进去,例如:{"path":"<该文件>","edits":[{"op":"replace","anchor":"22:abc","content":"新的这一行"}]}——不要用 bash/sed 改文件。`,
  en: `[Edit hint] The "LINE:HASH→" prefixes are edit anchors. To change this file, call edit_lines and copy an anchor verbatim, e.g. {"path":"<this file>","edits":[{"op":"replace","anchor":"22:abc","content":"the new line"}]} — do not edit files via bash/sed.`,
};

const ANCHOR_LINE_RE = /^\d+:[a-z]{2,4}→/m;

/** Escalating correction for a tool call with a missing required argument —
 *  the quick15 sympy-12419 autopsy: repeating ONE identical correction let a
 *  no-think model re-send `search_code {}` six times straight into the pause.
 *  Each attempt must add NEW information: 1 = fill the argument (an example);
 *  2 = break the tool fixation (name concrete ALTERNATIVE actions);
 *  3 = the tool is disabled for the rest of the turn (the loop enforces it).
 *  Empty-args calls are never executed — recording one plants the exemplar
 *  no-think models then imitate (the original A/B-1 spiral). */
export function missingArgLadder(
  name: string,
  arg: string,
  example: string,
  attempt: number,
  lang: "zh" | "en",
): string {
  const zh = lang === "zh";
  if (attempt <= 1) {
    return zh
      ? `ERROR: 缺少 "${arg}" 参数——请带上它重发 ${name},例如 arguments: ${example}`
      : `ERROR: missing "${arg}" — re-issue ${name} WITH it, e.g. arguments: ${example}`;
  }
  if (attempt === 2) {
    return zh
      ? `ERROR: 你已连续两次发出没有 "${arg}" 的 ${name}。先停下这个工具。如果还不知道 ${arg} 该填什么,就换一个具体动作推进:list_dir {"path":"."} 看目录结构,或 read_file 打开一个具体文件,或 grep {"pattern":"关键词"}。想再用 ${name},必须带上 ${arg},例如 ${example}。`
      : `ERROR: that is the second ${name} in a row without "${arg}". Stop using this tool for a moment. If you don't know what ${arg} should be, make a DIFFERENT concrete move instead: list_dir {"path":"."} to see the layout, read_file on a specific file, or grep {"pattern":"a keyword"}. To use ${name} again, you MUST include ${arg}, e.g. ${example}.`;
  }
  return zh
    ? `${name} 已暂时停用(连续 ${attempt} 次空参数)。先用 list_dir / read_file 等带具体参数的工具实际推进几步;之后再用 ${name} 时必须带上 ${arg},例如 ${example}。`
    : `${name} is temporarily disabled (${attempt} empty-argument calls in a row). Make real progress with other tools first (list_dir / read_file with concrete arguments); when you come back to ${name}, you MUST include ${arg}, e.g. ${example}.`;
}

// A local dev server just came up (bash/bash_bg/bg_output printed a local
// origin). This is the moment webapp discipline is decided — say it now, not
// in the every-step system prompt.
const WEB_FLOW_HINT: Record<"zh" | "en", string> = {
  zh: `[Webapp 提示] 本地 server 在跑。开发闭环:每次改完代码 → **browser_refresh 刷新页面**(截图/读取看到的都是旧页面)再亲眼验证,不要凭代码推断;首次打开用 browser_navigate;交互结果里的 [console] 段就是页面报错,出现了必须先修掉再继续。长驻命令(dev server、watch)一律用 bash_bg,前台 bash 会卡住;server 日志用 bg_output 查。交付前把改动涉及的页面路径实际走查一遍——交互功能要真点(browser_click)真输(browser_type),截图不算验收。`,
  en: `[Webapp hint] A local server is running. The loop: after EVERY code change → **browser_refresh** (screenshots/reads without it show the STALE page), then verify with your own eyes — never infer from code; browser_navigate only for the first open; a [console] section in interaction results IS the page erroring — fix it before anything else. Long-running commands (dev server, watch) always go through bash_bg — foreground bash stalls; check server logs with bg_output. Before delivering, actually walk the pages your change touches — interactive features get REAL clicks/typing, screenshots don't count as verification.`,
};

const LOCAL_URL_RE = /https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?/i;

/** Returns a hint to append to this tool result, or "" — and marks it shown. */
export function jitHintFor(
  name: string,
  resultText: string,
  lang: "zh" | "en",
  shown: Set<HintKey>,
): string {
  if (name.startsWith("browser_") && !shown.has("browser")) {
    shown.add("browser");
    return BROWSER_HINT[lang];
  }
  if (name === "understand_repo" && !shown.has("afterOrient")) {
    shown.add("afterOrient");
    return AFTER_ORIENT_HINT[lang];
  }
  if (
    (name === "bash" || name === "bash_bg" || name === "bg_output") &&
    !shown.has("webFlow") &&
    LOCAL_URL_RE.test(resultText)
  ) {
    shown.add("webFlow");
    return WEB_FLOW_HINT[lang];
  }
  if (name === "read_file" && !shown.has("anchorRead") && ANCHOR_LINE_RE.test(resultText.slice(0, 400))) {
    shown.add("anchorRead");
    return ANCHOR_READ_HINT[lang];
  }
  if (
    (name === "edit_file" || name === "multi_edit") &&
    !shown.has("editFail") &&
    /未找到 old_string|不唯一|old_string not found|not unique/.test(resultText)
  ) {
    shown.add("editFail");
    return EDIT_FAIL_HINT[lang];
  }
  return "";
}
