// AUTO-EXTRACTED verbatim from the pre-M0 agentLoop.ts by
// scripts/extract-tool-docs (one-shot). These are MODEL-VISIBLE strings: any
// edit here changes the agent's behavior and must be deliberate and
// bench-gated — the golden test (promptGolden.test.ts) will fail on any drift.

export interface Bi {
  zh: string;
  en: string;
}

/** One doc line per tool, exactly as it appears in the system prompt. */
export const DOC_LINES: Record<string, Bi> = {
  read_file: {
    zh: "- read_file: 读取文件(pdf/docx/xlsx/pptx 也能读:自动提取文本,扫描件 OCR)。只关心某个函数/类时传 symbol,返回该定义完整代码块+调用处清单。args: { \"path\": string, \"offset\"?: number(起始行,从1开始), \"limit\"?: number, \"symbol\"?: string }",
    en: "- read_file: read a file (pdf/docx/xlsx/pptx too — text auto-extracted, scans OCR'd). Pass symbol to get one function/class definition plus its call sites. args: { \"path\": string, \"offset\"?: number(1-based), \"limit\"?: number, \"symbol\"?: string }",
  },
  write_file: {
    zh: "- write_file: 新建文件,或整体重写(覆盖全部内容);修改已有文件优先用 edit_file。args: { \"path\": string, \"content\": string }",
    en: "- write_file: create a file, or rewrite one wholesale (replaces ALL content); to modify an existing file prefer edit_file. args: { \"path\": string, \"content\": string }",
  },
  edit_file: {
    zh: "- edit_file: 精确替换(old_string 须与文件逐字匹配且唯一,除非 replace_all=true)。同一文件改多处给 edits 数组,一次原子提交(任一条失败则整体不改),不要拆成多次调用。args: { \"path\", \"old_string\", \"new_string\", \"replace_all\"? } 或 { \"path\", \"edits\": [{ \"old_string\", \"new_string\", \"replace_all\"? }] }",
    en: "- edit_file: exact replacement (old_string must match the file verbatim and be unique unless replace_all=true). For several changes in one file, pass an atomic edits array in ONE call (any failure = nothing applied) instead of many small calls. args: { \"path\", \"old_string\", \"new_string\", \"replace_all\"? } or { \"path\", \"edits\": [{ \"old_string\", \"new_string\", \"replace_all\"? }] }",
  },
  outline: {
    zh: "- outline: 文件的定义大纲(函数/类+行号),不读全文即掌握结构。args: { \"path\": string }",
    en: "- outline: definition outline (functions/classes + line numbers) without reading the whole file. args: { \"path\": string }",
  },
  list_dir: {
    zh: "- list_dir: 列出目录一层内容(不传 path = 工作区根)。args: { \"path\"?: string }",
    en: "- list_dir: list one directory level (no path = workspace root). args: { \"path\"?: string }",
  },
  glob: {
    zh: "- glob: 按通配符找文件(如 \"src/**/*.ts\")。args: { \"pattern\": string }",
    en: "- glob: find files by pattern (e.g. \"src/**/*.ts\"). args: { \"pattern\": string }",
  },
  grep: {
    zh: "- grep: 用正则搜索文件内容。args: { \"pattern\": string, \"path\"?: string, \"glob\"?: string }",
    en: "- grep: regex search over file contents. args: { \"pattern\": string, \"path\"?: string, \"glob\"?: string }",
  },
  search_files: {
    zh: "- search_files: 按关键词(字面)搜文件名+内容;names_only=true 只搜文件名。args: { \"query\": string, \"path\"?: string, \"names_only\"?: boolean }",
    en: "- search_files: literal keyword search over file names + contents; names_only=true for names only. args: { \"query\": string, \"path\"?: string, \"names_only\"?: boolean }",
  },
  search_code: {
    zh: "- search_code: 按含义提问代码库,返回按相关度排序的文件+关键定义。探索陌生代码优先用它。args: { \"query\": \"哪里处理登录鉴权\", \"k\"?: number }",
    en: "- search_code: ask the codebase by meaning; returns relevance-ranked files with their key definitions. First choice for unfamiliar code. args: { \"query\": \"where login auth is handled\", \"k\"?: number }",
  },
  search_docs: {
    zh: "- search_docs: 检索用户的知识库文档(需求、设计稿、笔记)。args: { \"query\": string }",
    en: "- search_docs: search the user's knowledge-base documents. args: { \"query\": string }",
  },
  bash: {
    zh: "- bash: 在工作区执行 shell 命令(沙箱,写限工作区);会等命令结束,不要用它启动 dev server 等不退出的进程。args: { \"command\": string, \"timeout_secs\"?: number }",
    en: "- bash: run a shell command in the workspace (sandboxed, writes limited to the workspace); waits for exit — don't start dev servers with it. args: { \"command\": string, \"timeout_secs\"?: number }",
  },
  bash_bg: {
    zh: "- bash_bg: 后台启动长时间运行的命令(dev server、慢构建),立即返回 id,结束时系统自动通知你;不支持 sudo(要特权用前台 bash)。args: { \"command\": string }",
    en: "- bash_bg: start a long-running command in the background (dev server, slow build); returns an id, you're notified when it ends; sudo unsupported (use foreground bash). args: { \"command\": string }",
  },
  bg_output: {
    zh: "- bg_output: 查看后台命令的状态与最近输出。args: { \"id\": number }",
    en: "- bg_output: status + recent output of a background job. args: { \"id\": number }",
  },
  bg_kill: {
    zh: "- bg_kill: 终止后台命令(整棵进程树)。args: { \"id\": number }",
    en: "- bg_kill: kill a background job (whole process tree). args: { \"id\": number }",
  },
  understand_repo: {
    zh: "- understand_repo: 一次拿到仓库速览(README/manifest/目录树/语言/入口)。接手陌生工作区的第一个动作。args: {}",
    en: "- understand_repo: one-call repo overview (README, manifest, tree, languages, entry points). First move in an unfamiliar workspace. args: {}",
  },
  validate_change: {
    zh: "- validate_change: 自动找出与改动相关的测试并只跑最小集;不传参数验证本轮全部改动。args: { \"files\"?: string[] }",
    en: "- validate_change: find and run just the tests related to the change; no args = everything changed this turn. args: { \"files\"?: string[] }",
  },
  web_search: {
    zh: "- web_search: 联网搜索;site 参数限站内(github.com 返回结构化仓库/issue/代码;reddit/youtube/bilibili 及任意域名均可)。args: { \"query\": string, \"site\"?: string }",
    en: "- web_search: web search; site scopes to one site (github.com returns structured repos/issues/code; reddit/youtube/bilibili/any domain). args: { \"query\": string, \"site\"?: string }",
  },
  web_fetch: {
    zh: "- web_fetch: 抓取 URL,按内容类型自动处理(文章→Markdown、GitHub 文件→raw 源码、视频→字幕、PDF→文本);要 HTML 源码传 raw=true。args: { \"url\": string, \"raw\"?: boolean }",
    en: "- web_fetch: fetch a URL, handled by content type (article→Markdown, GitHub file→raw source, video→transcript, PDF→text); raw=true for HTML. args: { \"url\": string, \"raw\"?: boolean }",
  },
  web_download: {
    zh: "- web_download: 后台下载文件到工作区路径,完成/失败时自动通知;在那之前不要读取该文件或重复发起。args: { \"url\": string, \"path\": string }",
    en: "- web_download: background-download a file into the workspace; you're notified on completion — don't read it or re-request before that. args: { \"url\": string, \"path\": string }",
  },
  update_plan: {
    zh: "- update_plan: 制定/更新任务待办清单;完成一步标 done、下一步标 in_progress。args: { \"todos\": [{ \"content\": string, \"status\": \"pending\"|\"in_progress\"|\"done\" }] }",
    en: "- update_plan: create/update the todo plan; mark finished steps done, the next one in_progress. args: { \"todos\": [{ \"content\": string, \"status\": \"pending\"|\"in_progress\"|\"done\" }] }",
  },
  ask_user: {
    zh: "- ask_user: 需要用户拍板的决策(方案分歧、需求不明、破坏性操作)向用户提选择题,不要乱猜。args: { \"question\": string, \"options\": string[] }",
    en: "- ask_user: when a decision is the user's (competing approaches, unclear requirements, destructive ops), ask a multiple-choice question — don't guess. args: { \"question\": string, \"options\": string[] }",
  },
  view_image: {
    zh: "- view_image: 查看工作区里的图片(视觉模型直接看;非视觉模型自动 OCR 出文字)。args: { \"path\": string }",
    en: "- view_image: view an image in the workspace (vision models see it; others get OCR'd text). args: { \"path\": string }",
  },
  browser_navigate: {
    zh: "- browser_navigate: 打开网址(或本地文件/dev server)。返回标题+页面文字+可交互元素清单。args: { \"url\": string }",
    en: "- browser_navigate: open a URL (or local file / dev server). Returns title + page text + interactive elements. args: { \"url\": string }",
  },
  browser_read: {
    zh: "- browser_read: 读当前页全部可见文字+元素清单(含输入框当前值)。要\"内容/文字/状态\"时用它。args: {}",
    en: "- browser_read: all visible text of the current page + element list (incl. current input values). Use for content/text/state. args: {}",
  },
  browser_screenshot: {
    zh: "- browser_screenshot: 整页截图并用视觉查看(自动滚动触发懒加载)。要判断\"长什么样\"时用它。args: {}",
    en: "- browser_screenshot: full-page screenshot, seen with vision (auto-scrolls for lazy content). Use to judge how the page LOOKS. args: {}",
  },
  browser_snapshot: {
    zh: "- browser_snapshot: 当前视口截图(即时,不滚动)。args: {}",
    en: "- browser_snapshot: current-viewport screenshot (instant, no scrolling). args: {}",
  },
  browser_scroll: {
    zh: "- browser_scroll: 滚动以加载更多内容。args: { \"to\"?: \"bottom\"|\"top\", \"by\"?: number }",
    en: "- browser_scroll: scroll to load more. args: { \"to\"?: \"bottom\"|\"top\", \"by\"?: number }",
  },
  browser_close: {
    zh: "- browser_close: 关闭浏览器。args: {}",
    en: "- browser_close: close the browser. args: {}",
  },
  browser_console: {
    zh: "- browser_console: 读取页面 JS 控制台输出与异常。args: {}",
    en: "- browser_console: read the page's JS console output and exceptions. args: {}",
  },
  browser_click: {
    zh: "- browser_click: 点击元素,优先用 text 按可见文字;已想好顺序的多个目标必须一次用 steps 传完。args: { \"text\"?, \"selector\"?, \"steps\"?: [{ \"text\"?, \"selector\"? }] }",
    en: "- browser_click: click an element — prefer text (visible label); when you already know a sequence of targets, pass them ALL in one steps call. args: { \"text\"?, \"selector\"?, \"steps\"?: [{ \"text\"?, \"selector\"? }] }",
  },
  browser_type: {
    zh: "- browser_type: 填输入框,也用于下拉框(text=选项可见文字);多个字段一次用 steps 填完。args: { \"text\"?, \"label\"?, \"selector\"?, \"steps\"?: [{ \"text\", \"label\"?, \"selector\"? }] }",
    en: "- browser_type: fill inputs, also selects dropdowns (text = the option's visible label); fill several fields in one steps call. args: { \"text\"?, \"label\"?, \"selector\"?, \"steps\"?: [{ \"text\", \"label\"?, \"selector\"? }] }",
  },
  browser_eval: {
    zh: "- browser_eval: 执行 JavaScript 并返回结果。args: { \"expression\": string }",
    en: "- browser_eval: run JavaScript and return the result. args: { \"expression\": string }",
  },
  edit_lines: {
    zh: "- edit_lines: 按行锚点批量改行(锚点=read_file 行首的 \"22:abc\")。op: replace(anchor 单行,或 anchor+end_anchor 区间;content 为新内容,空串=删除)、insert_after(anchor|\"0\" 文件头|\"EOF\" 文件尾)。content 只写文件内容,不要带锚点前缀。args: { \"path\", \"edits\": [{ \"op\": \"replace\", \"anchor\": \"22:abc\", \"content\": \"新行\" }] }",
    en: "- edit_lines: batch line edits addressed by anchors (the \"22:abc\" shown by read_file). op: replace (single anchor, or anchor+end_anchor range; content = new text, \"\" = delete) and insert_after (anchor | \"0\" BOF | \"EOF\"). content is plain file content — never include anchor prefixes. args: { \"path\", \"edits\": [{ \"op\": \"replace\", \"anchor\": \"22:abc\", \"content\": \"new line\" }] }",
  },
};

/** Guidance tail of the vision browser suite (last line of the block). */
export const BROWSER_TAIL_VISION: Bi = {
  zh: "要内容/文字/状态→browser_read;要外观/渲染是否正确→browser_screenshot/snapshot 用视觉亲眼看。每次交互后先核实结果再继续。查资料优先 web_search/web_fetch,需要真实操作网页或视觉验证时才用浏览器。",
  en: "Content/text/state → browser_read; looks/rendering → browser_screenshot/snapshot and SEE it. Verify the result after every interaction before moving on. Research goes through web_search/web_fetch first; open the browser only for real page interaction or visual verification.",
};

/** Tail for the no-vision browser suite (screenshot tools dropped). */
export const BROWSER_TAIL_TEXT: Bi = {
  zh: "该模型没有视觉,截图工具不可用:browser_read 就是你的眼睛——每次交互后先用它核实结果再继续。查资料优先 web_search/web_fetch,需要真实操作网页时才用浏览器。",
  en: "This model has no vision and screenshot tools are unavailable: browser_read is your eyes — verify with it after every interaction before moving on. Research goes through web_search/web_fetch first; open the browser only for real page interaction.",
};

/** Appended to read_file's doc line in anchor mode. */
export const ANCHOR_READ_NOTE: Bi = {
  zh: "每行行首带编辑锚点 \"行号:哈希→\"(如 \"22:abc→\"),编辑时直接把它填进 edit_lines 的 anchor。",
  en: "Every line is prefixed with its edit anchor \"LINE:HASH→\" (e.g. \"22:abc→\") — pass that straight into edit_lines anchors.",
};
