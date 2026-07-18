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

/** git subcommands that are read-only in every form. */
const GIT_FULL = new Set([
  "status", "log", "diff", "show", "blame", "grep", "ls-files", "ls-tree",
  "ls-remote", "rev-parse", "describe", "shortlog", "reflog", "cat-file",
]);

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
  if (GIT_FULL.has(sub)) return true;
  const positionals = rest.filter((a) => !a.startsWith("-"));
  switch (sub) {
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
