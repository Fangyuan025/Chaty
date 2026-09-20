/** Conservative "is this bash command definitely read-only?" judge, used to
 *  skip the approval dialog for commands whose prompt could only ever be
 *  answered "yes". Fail-closed by design: anything uncertain returns false and
 *  falls through to the normal allowlist / approval flow. This is a
 *  don't-interrupt gate, NOT a security boundary — the workspace sandbox
 *  remains the hard edge either way. */

/** Commands allowed as a segment's first word. Some get extra argument guards
 *  below; everything else on the list is stdout-only by nature (no flag can
 *  make it write to the filesystem — output redirection is rejected globally). */
const SAFE = new Set([
  "ls", "cat", "pwd", "echo", "head", "tail", "wc", "sort", "uniq", "cut",
  "tr", "grep", "rg", "which", "file", "stat", "du", "df", "ps", "date",
  "whoami", "cd", "true", "basename", "dirname", "realpath", "readlink",
  "env", "printenv", "uname", "id", "nl", "tree", "diff", "comm", "column",
  "strings", "shasum", "sha256sum", "md5", "cksum", "od", "xxd", "hexdump",
  "printf", "test", "[", "find", "git", "cargo",
]);

/** git subcommands that are read-only in every form — subject to the flag
 *  guard below, since several of them take the diff machinery's `--output`. */
const GIT_FULL = new Set([
  "status", "log", "diff", "show", "blame", "grep", "ls-files", "ls-tree",
  "ls-remote", "rev-parse", "describe", "shortlog", "cat-file",
]);

/** Flags that make a reading git command write a file or run a program:
 *  `git diff --output=f` (a diff option, so `log`/`show` take it too),
 *  `git grep -O` (opens the matches in a pager command), `--ext-diff` (runs
 *  the configured external diff). */
const gitFlagWrites = (a: string): boolean =>
  a === "-o" ||
  a === "--output" ||
  a.startsWith("--output=") ||
  a === "-O" ||
  a === "--open-files-in-pager" ||
  a.startsWith("--open-files-in-pager=") ||
  a === "-x" ||
  a === "--ext-diff";

/** Flags that turn `git branch` / `git tag` into a mutation. */
const GIT_MUTATING_FLAGS = new Set([
  "-d", "-D", "-m", "-M", "-c", "-C", "-f", "--force", "--delete", "--move",
  "--copy", "--edit-description", "--set-upstream-to", "--unset-upstream",
  "--create-reflog",
]);

function gitIsReadOnly(args: string[]): boolean {
  const sub = args.find((a) => !a.startsWith("-"));
  if (!sub) return true; // bare `git`, `git --version`, `git --help`
  const rest = args.slice(args.indexOf(sub) + 1);
  if (args.some(gitFlagWrites)) return false;
  if (GIT_FULL.has(sub)) return true;
  const positionals = rest.filter((a) => !a.startsWith("-"));
  switch (sub) {
    case "reflog":
      // Bare / `show` / `exists` read; `expire`, `delete` and `drop` rewrite
      // the reflog, which is how history gets thrown away for good.
      return positionals.length === 0 || positionals[0] === "show" || positionals[0] === "exists";
    case "branch":
    case "tag":
      return positionals.length === 0 && !rest.some((a) => GIT_MUTATING_FLAGS.has(a));
    case "remote":
      // bare / -v listing only; `remote show origin` is read-only but rare
      // enough that the over-strict reject is fine.
      return positionals.length === 0;
    case "stash":
      // bare `git stash` SAVES a stash — only the listing/inspecting forms pass.
      return positionals[0] === "list" || positionals[0] === "show";
    case "config":
      // `--get*` reads need a positional key, so allow those; otherwise
      // listing only.
      if (rest.some((a) => a === "--list" || a === "-l")) return !rest.some((a) => a === "--edit" || a === "-e");
      if (rest.some((a) => a.startsWith("--get"))) return true;
      return false;
    default:
      return false;
  }
}

/** Per-command argument guards for tools that CAN write via flags. */
function argsAreReadOnly(cmd: string, args: string[]): boolean {
  switch (cmd) {
    case "sort":
      return !args.some((a) => a === "-o" || a === "--output" || a.startsWith("--output="));
    case "uniq":
      // `uniq in out` writes the second positional.
      return args.filter((a) => !a.startsWith("-")).length < 2;
    case "rg":
      return !args.some((a) => a.startsWith("--pre"));
    case "find":
      return !args.some(
        (a) => ["-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls"].includes(a) || a.startsWith("-fprint"),
      );
    case "env":
      // `env CMD` executes CMD (and `env -i CMD`, `env -S…`) — only the bare
      // environment listing is auto-approved.
      return args.length === 0;
    case "tree":
      // `tree -o FILE` writes the listing to a file.
      return !args.some((a) => a === "-o" || a === "-O" || a.startsWith("--output"));
    case "xxd": {
      // `xxd in out` / `xxd -r dump file` writes the second operand — with
      // `-r`, the decoded bytes. Counting operands means skipping the values
      // of the flags that take one (`xxd -l 64 a.bin` is ONE operand).
      const takesValue = new Set(["-l", "-s", "-c", "-g", "-o"]);
      let operands = 0;
      for (let i = 0; i < args.length; i++) {
        const a = args[i];
        if (a.startsWith("-")) {
          if (takesValue.has(a)) i++;
          continue;
        }
        operands++;
      }
      return operands < 2;
    }
    case "date":
      // `date -s` / `date --set` sets the system clock.
      return !args.some((a) => a === "-s" || a.startsWith("--set"));
    case "git":
      return gitIsReadOnly(args);
    case "cargo":
      // `cargo check` may execute build.rs — this gate is about not
      // interrupting, not about security; the sandbox is the boundary.
      return args[0] === "check";
    default:
      return true;
  }
}

/** Executions that prove nothing about whether the code WORKS: version/help
 *  probes and syntax-only parses. They exit 0 on a project that doesn't even
 *  compile — the CalendarApp audit's `swift --version` + `swiftc -parse`
 *  combo — so they must not count as run-verification. Read-only segments
 *  (`cd`, `echo`…) are neutral; every remaining segment must be symbolic for
 *  the whole command to be. */
export function isSymbolicCheck(cmd: string): boolean {
  if (/[`>]|\$\(|<\(/.test(cmd)) return false; // unparsed shapes: fail closed
  const segments = cmd.trim().split(/\|\||&&|;|\||\n|&/);
  let sawSymbolic = false;
  for (const seg of segments) {
    let s = seg.trim();
    if (!s) continue;
    if (isReadOnlyCommand(s)) continue; // neutral scaffolding, e.g. `cd proj`
    s = s.replace(/^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+/, "");
    let words = s.split(/\s+/);
    // `xcrun [--sdk X] tool …` runs the tool — judge the tool itself.
    if ((words[0] ?? "").split("/").pop() === "xcrun") {
      words = words.slice(1);
      while (words[0]?.startsWith("-")) words = words.slice(words[0].includes("=") ? 1 : 2);
    }
    const bin = (words[0] ?? "").split("/").pop() ?? "";
    const args = words.slice(1);
    const versionProbe =
      args.length > 0 && args.every((a) => /^--?(version|help|V|v)$/.test(a));
    // Syntax-only probes across ecosystems: they pass on code that cannot
    // run (missing imports, type errors, absent deps) — same class as
    // `swiftc -parse`, and models reach for whichever their stack offers.
    const parseOnly =
      (/^swiftc?$/.test(bin) &&
        args.some((a) => a === "-parse" || a === "-dump-parse" || a === "-dump-ast")) ||
      (bin === "node" && args.includes("--check")) ||
      (/^python3?$/.test(bin) && args.join(" ").includes("-m py_compile")) ||
      (bin === "ruby" && args.includes("-c")) ||
      (bin === "php" && args.includes("-l"));
    if (!versionProbe && !parseOnly) return false;
    sawSymbolic = true;
  }
  return sawSymbolic;
}

export function isReadOnlyCommand(cmd: string, opts?: { windows?: boolean }): boolean {
  // v1 doesn't model cmd.exe semantics — never auto-approve on Windows.
  if (opts?.windows) return false;
  const line = cmd.trim();
  if (!line) return false;
  // Substitution and redirection are rejected globally, quotes unparsed:
  // a `>` inside a quoted string only causes a false reject, never a false pass.
  if (/[`>]|\$\(|<\(/.test(line)) return false;
  // Split into segments at every operator; each must independently pass.
  const segments = line.split(/\|\||&&|;|\||\n|&/);
  for (const seg of segments) {
    let s = seg.trim();
    if (!s) return false; // empty segment: stray operator, e.g. leading `;`
    // strip FOO=bar env-assignment prefixes
    s = s.replace(/^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+/, "");
    const words = s.split(/\s+/);
    const first = words[0];
    if (!first || first.includes("/")) return false; // ./script, /bin/x
    if (!SAFE.has(first)) return false;
    if (!argsAreReadOnly(first, words.slice(1))) return false;
  }
  return true;
}
