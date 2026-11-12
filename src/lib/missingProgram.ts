/**
 * A command that failed because the machine does not have the program.
 *
 * "python: command not found" is not a mistake in the command, and nothing
 * the model changes about the command fixes it — but a model reads it as one
 * more failed attempt and tries the next spelling: `python -m http.server`,
 * `python3 …`, a server written to a file and run with python, `npx serve`.
 * On a machine with neither Python nor Node that was a whole turn of failed
 * commands (issue #25: "ran 54 commands, 55 failed") for a page that needed
 * no server to be looked at.
 *
 * So the first such failure says what it is, once, and the program is
 * remembered for the turn: a later command that starts it is answered without
 * being run.
 */

/** The program `output` says is not there; null for any other failure.
 *  `command`: what was run — dash's bare "x: not found" is taken for a
 *  missing program only when x is one the command starts (a tool can say
 *  "config.yaml: not found" about a file). */
export function missingProgram(output: string, command?: string): string | null {
  const patterns: RegExp[] = [
    // zsh, fish
    /command not found: ([\w.+-]+)/,
    /Unknown command:? '?([\w.+-]+)'?/,
    // bash: "bash: python: command not found", with a path or without
    /([\w.+-]+): command not found\b(?!:)/,
    // cmd.exe, in English and Chinese
    /'([^'\n]+)' is not recognized as an internal or external command/,
    /'([^'\n]+)' 不是内部或外部命令/,
    // PowerShell, in English and Chinese
    /The term '([^'\n]+)' is not recognized/,
    /无法将[“"]([^”"\n]+)[”"]项识别为/,
    // A script whose interpreter is missing
    /env: ([\w.+-]+): No such file or directory/,
  ];
  for (const re of patterns) {
    const m = re.exec(output);
    if (m) return programName(m[1]);
  }
  // Windows' stand-in for an uninstalled Python.
  if (/Python was not found; run without arguments to install from the Microsoft Store/i.test(output)) return "python";
  // sh / dash: "sh: 1: python: not found".
  const bare = /(?:^|\n)(?:[\w./-]*sh(?:\.exe)?: )?(?:line \d+: |\d+: )?([\w.+-]+): not found\b/.exec(output);
  if (bare && command) {
    const name = programName(bare[1]);
    if (commandPrograms(command).includes(name)) return name;
  }
  return null;
}

/** A program's name as a command starts it: no folder, no .exe, lower case. */
function programName(word: string): string {
  return (word.trim().split(/[\\/]/).pop() ?? word)
    .replace(/\.(exe|cmd|bat|ps1)$/i, "")
    .toLowerCase();
}

const WRAPPERS = new Set(["sudo", "nohup", "time", "exec", "env", "call", "start", "command"]);

/** The programs a shell command starts: the first word of each part of a
 *  pipeline or list, past assignments and wrappers like sudo and nohup. */
export function commandPrograms(command: string): string[] {
  const out: string[] = [];
  const plain = command
    .replace(/^\[detached\]\s*/, "")
    // Redirections are not commands: `2>&1`, `&>log`.
    .replace(/\d*>&\d*-?/g, " ")
    .replace(/&>/g, " >");
  for (const part of plain.split(/&&|\|\||[;|\n&]/)) {
    const words = part.trim().split(/\s+/).filter(Boolean);
    let i = 0;
    while (i < words.length && (/^[A-Za-z_][A-Za-z0-9_]*=/.test(words[i]) || WRAPPERS.has(words[i].toLowerCase()))) i++;
    const w = words[i];
    if (!w || /^[-(){}<>"'$`]/.test(w)) continue;
    const name = programName(w);
    if (name && !out.includes(name)) out.push(name);
  }
  return out;
}

/** Programs a page is usually served or run with — where saying "the page
 *  opens from its file" is the way out. */
const SERVES = new Set(["python", "python3", "py", "node", "npx", "npm", "pnpm", "yarn", "bun", "deno", "php", "ruby", "http-server", "serve", "live-server"]);

/** What the model is told the first time a program turns out to be missing
 *  (`blocked`: and each time it reaches for it again). */
export function missingProgramNote(program: string, zh: boolean, blocked = false): string {
  const serves = SERVES.has(program);
  if (zh) {
    return (
      (blocked
        ? `ERROR: 调用被拦截,没有执行:这台机器上没有 \`${program}\`(本轮已经确认过)。`
        : `[环境] 这台机器上没有 \`${program}\`——不是命令写错了,换参数、换写法重试都没用,本轮不要再调用它。`) +
      (serves ? "如果只是为了打开或验证页面,不需要它:browser_navigate 直接打开文件路径(例如 index.html)即可。" : "") +
      `如果任务离不开 ${program},就在回答里说明已经完成了什么、需要用户先安装 ${program}。`
    );
  }
  return (
    (blocked
      ? `ERROR: intercepted, not run: this machine has no \`${program}\` (established earlier this turn). `
      : `[environment] This machine has no \`${program}\` — the command was not mistyped, and no other arguments or spelling will work; do not call it again this turn. `) +
    (serves ? "To open or check a page you do not need it: browser_navigate takes a file path (e.g. index.html). " : "") +
    `If the task cannot do without ${program}, say in your answer what is done and that ${program} has to be installed first.`
  );
}
