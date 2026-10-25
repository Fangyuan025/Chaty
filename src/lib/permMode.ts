/** Code mode's permission modes — the five Claude Code has, picked under the
 *  input (or cycled with Shift+Tab) — and what each makes of a call that
 *  needs a say: run it, ask, or (planning) send it back unrun.
 *
 *  Whatever the mode, two things always ask, outside this module: a folder
 *  outside the workspace, and sudo. And whatever the mode short of planning,
 *  the user's own allowances hold — the command allowlist, read-only
 *  commands (Settings → Code), and "always allow" from the approval dialog. */

import type { ToolCall } from "./agentLoop";
import { isReadOnlyCommand } from "./readOnlyCmd";

export type PermMode = "auto" | "manual" | "acceptEdits" | "plan" | "bypass";

/** Menu order — also the numbers 1–5 and the Shift+Tab cycle. */
export const PERM_MODES: PermMode[] = ["auto", "manual", "acceptEdits", "plan", "bypass"];

/** The modes a choice is remembered in; planning and bypassing last until
 *  the app quits. */
export const LASTING_MODES: PermMode[] = ["auto", "manual", "acceptEdits"];

export const isPermMode = (v: unknown): v is PermMode => PERM_MODES.includes(v as PermMode);

const EDIT_TOOLS = new Set(["write_file", "edit_file", "multi_edit", "edit_lines"]);

/** A call that changes the workspace's files through the file tools — what
 *  checkpoints cover, and what planning holds back. */
export const isFileEdit = (name: string): boolean => EDIT_TOOLS.has(name);

export type Verdict = "allow" | "ask" | "plan";

export interface PermContext {
  /** Settings → Code: obviously read-only commands run without asking. */
  readOnlyAuto: boolean;
  /** Settings → Code: command prefixes that never ask. */
  allowlist: string[];
  /** "Always allow" from the approval dialog, this session. */
  sessionAllowed: boolean;
  windows: boolean;
}

export function permVerdict(mode: PermMode, call: ToolCall, ctx: PermContext): Verdict {
  if (mode === "bypass") return "allow";
  const name = call.name as string;
  if (mode === "plan" && (isFileEdit(name) || name === "web_download")) return "plan";
  if (ctx.sessionAllowed) return "allow";
  if (isFileEdit(name)) return mode === "acceptEdits" || mode === "auto" ? "allow" : "ask";
  if (name === "bash" || name === "bash_bg") {
    const cmd = String(call.args.command ?? "").trim();
    if (ctx.readOnlyAuto && isReadOnlyCommand(cmd, { windows: ctx.windows })) return "allow";
    if (ctx.allowlist.some((p) => cmd === p || cmd.startsWith(p + " "))) return "allow";
    if (mode === "acceptEdits" && isFileOpCommand(cmd)) return "allow";
    if (mode === "auto" && !isRiskyCommand(cmd)) return "allow";
    return "ask";
  }
  // Auto: what stays in the workspace or the run itself goes ahead; a tool
  // from elsewhere (an MCP server's) can act outside it, and asks.
  if (mode === "auto" && AUTO_TOOLS.has(name)) return "allow";
  return "ask";
}

const AUTO_TOOLS = new Set(["web_download", "validate_change", "bg_input"]);

// ── Splitting a command line into the commands it runs ─────────────────────

/** Strings a shell runs as commands of their own: `$(…)`, backticks, and
 *  the script of `sh -c` / `bash -c` / `eval`. */
function innerScripts(line: string): string[] {
  const out: string[] = [];
  for (const m of line.matchAll(/\$\(([^()]*)\)/g)) out.push(m[1]);
  for (const m of line.matchAll(/`([^`]*)`/g)) out.push(m[1]);
  for (const m of line.matchAll(/\b(?:ba|z|da|k)?sh\s+(?:-\w+\s+)*-c\s+(['"])([\s\S]*?)\1/g)) out.push(m[2]);
  for (const m of line.matchAll(/\beval\s+(['"])([\s\S]*?)\1/g)) out.push(m[2]);
  return out;
}

/** Words that run what follows them, and the flags of theirs that take a
 *  value. */
const WRAPPERS: Record<string, Set<string>> = {
  env: new Set(["-u", "-C", "-S"]),
  time: new Set(),
  nice: new Set(["-n"]),
  nohup: new Set(),
  command: new Set(),
  exec: new Set(["-a"]),
  builtin: new Set(),
  xargs: new Set(["-I", "-J", "-L", "-n", "-P", "-s", "-E", "-d", "-a"]),
  timeout: new Set(["-s", "-k", "--signal", "--kill-after"]),
  caffeinate: new Set(["-t", "-w"]),
  stdbuf: new Set(["-i", "-o", "-e"]),
  doas: new Set(["-u"]),
};

const base = (w: string) => (w.split(/[\\/]/).pop() ?? w).toLowerCase();

/** Each command the line runs, as its words (quotes dropped): one per
 *  segment between operators, past env assignments and wrappers, plus the
 *  command after `find … -exec`. */
function commandsOf(line: string, depth = 0): string[][] {
  const out: string[][] = [];
  if (depth < 3) for (const s of innerScripts(line)) out.push(...commandsOf(s, depth + 1));
  for (const seg of line.split(/\|\||&&|;|\||\n|&|[(){}]/)) {
    let words = seg
      .replace(/['"]/g, "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    while (words.length && /^[A-Za-z_][A-Za-z0-9_]*=/.test(words[0])) words = words.slice(1);
    for (let guard = 0; guard < 4 && words.length; guard++) {
      const w = base(words[0]);
      const takes = WRAPPERS[w];
      if (!takes) break;
      let i = 1;
      while (i < words.length && (words[i].startsWith("-") || /^[A-Za-z_][A-Za-z0-9_]*=/.test(words[i]))) {
        i += takes.has(words[i]) ? 2 : 1;
      }
      // `timeout 30 cmd`, `nice 10 cmd`: the duration before the command.
      if ((w === "timeout" || w === "nice") && /^\d/.test(words[i] ?? "")) i++;
      words = words.slice(i);
    }
    if (!words.length) continue;
    out.push(words);
    const ex = words.findIndex((x) => ["-exec", "-execdir", "-ok", "-okdir"].includes(x));
    if (base(words[0]) === "find" && ex >= 0) out.push(words.slice(ex + 1));
  }
  return out;
}

// ── Accept edits: the file operations that go with editing ─────────────────

const FILE_OPS = new Set(["mkdir", "touch", "cp", "mv"]);

/** `mkdir`, `touch`, `cp`, `mv` — alone or with read-only commands — which
 *  Claude Code's accept-edits mode lets through with the edits themselves.
 *  The sandbox keeps them in the workspace. */
export function isFileOpCommand(cmd: string): boolean {
  if (/[`>]|\$\(|<\(/.test(cmd)) return false;
  const cmds = commandsOf(cmd);
  if (!cmds.length) return false;
  let sawOp = false;
  for (const words of cmds) {
    if (FILE_OPS.has(base(words[0]))) {
      sawOp = true;
      continue;
    }
    if (!isReadOnlyCommand(words.join(" "))) return false;
  }
  return sawOp;
}

// ── Auto: what is hard to take back ────────────────────────────────────────

/** Always worth a question: deleting, ending processes, the machine's own
 *  settings and disks, other machines. */
const RISKY = new Set([
  "rm", "rmdir", "unlink", "shred", "srm", "dd", "diskutil", "fdisk", "mount", "umount",
  "kill", "pkill", "killall", "launchctl", "systemctl", "service", "shutdown", "reboot", "halt",
  "crontab", "osascript", "chown", "su", "ssh", "scp", "sftp", "ftp",
  "networksetup", "scutil", "pmset", "nvram", "csrutil", "spctl", "tccutil", "systemsetup", "security",
  "brew", "apt", "apt-get", "yum", "dnf", "pacman", "port", "zypper", "apk", "snap",
  "choco", "winget", "scoop",
  // Windows
  "del", "erase", "rd", "remove-item", "ri", "taskkill", "stop-process", "format", "reg", "set-executionpolicy",
]);

/** Words that, as an argument of a deploy / cloud / container tool, change
 *  something that is not on this machine — or remove what is. */
const REMOTE_CHANGE = new Set([
  "deploy", "publish", "release", "delete", "destroy", "remove", "rm", "rmi", "prune", "create",
  "put", "update", "apply", "push", "scale", "rollback", "promote", "sync", "cp", "mv", "mb", "rb",
  "invoke", "kill", "stop", "restart", "drain", "cordon", "patch", "replace", "edit", "taint",
  "import", "up", "install", "uninstall", "upgrade", "login", "logout", "secrets",
]);
const REMOTE_TOOLS = new Set([
  "aws", "gcloud", "gsutil", "az", "doctl", "fly", "flyctl", "vercel", "netlify", "heroku",
  "firebase", "wrangler", "railway", "surge", "supabase", "terraform", "tofu", "pulumi",
  "kubectl", "helm", "docker", "podman",
]);

/** Package managers: what leaves the machine or installs for all of it. */
const PUBLISH = new Set(["publish", "unpublish", "deprecate", "owner", "access", "dist-tag", "login", "adduser", "logout", "token", "yank", "push", "upload"]);

function gitRisky(args: string[]): boolean {
  const pos = args.filter((a) => !a.startsWith("-"));
  const sub = pos[0];
  const has = (...f: string[]) => args.some((a) => f.includes(a));
  switch (sub) {
    case "push":
    case "filter-branch":
    case "filter-repo":
    case "rm":
      return true;
    case "reset":
      return has("--hard", "--merge", "--keep");
    case "clean":
      return !has("-n", "--dry-run");
    case "checkout":
      // `checkout -- path` / `checkout .` throw away changes; `-f` too.
      return has("--", ".", "-f", "--force");
    case "restore":
      // Without --staged it overwrites the working copy.
      return !has("--staged", "-S") || has("--worktree", "-W");
    case "stash":
      return pos[1] === "drop" || pos[1] === "clear";
    case "branch":
    case "tag":
      return has("-D", "-d", "--delete", "-f", "--force");
    case "update-ref":
      return has("-d");
    case "reflog":
      return pos[1] === "expire" || pos[1] === "delete";
    case "gc":
      return args.some((a) => a.startsWith("--prune"));
    case "worktree":
      return pos[1] === "remove" || pos[1] === "prune";
    case "submodule":
      return pos[1] === "deinit";
    default:
      return false;
  }
}

/** `gh`: reading is fine; anything that writes to GitHub asks. */
function ghRisky(args: string[]): boolean {
  const pos = args.filter((a) => !a.startsWith("-"));
  if (!pos.length) return false;
  if (pos[0] === "api") {
    const i = args.findIndex((a) => a === "-X" || a === "--method");
    const method = i >= 0 ? (args[i + 1] ?? "").toUpperCase() : args.find((a) => /^--method=/.test(a))?.split("=")[1]?.toUpperCase();
    if (method && method !== "GET") return true;
    return args.some((a) => ["-f", "-F", "--field", "--raw-field", "--input"].includes(a) || /^--(raw-)?field=/.test(a));
  }
  const READ = new Set(["view", "list", "status", "diff", "checks", "search", "browse", "help", "version"]);
  return !(pos.length >= 2 ? READ.has(pos[1]) : READ.has(pos[0]));
}

/** `curl` / `wget` that send rather than fetch. */
function sends(bin: string, args: string[]): boolean {
  if (bin === "curl") {
    const i = args.findIndex((a) => a === "-X" || a === "--request");
    const method = (i >= 0 ? args[i + 1] : args.find((a) => /^-X[A-Za-z]/.test(a))?.slice(2)) ?? "";
    if (method && method.toUpperCase() !== "GET" && method.toUpperCase() !== "HEAD") return true;
    return args.some(
      (a) =>
        ["-d", "-F", "-T", "--form", "--upload-file", "--json"].includes(a) || a.startsWith("--data") || a.startsWith("--form"),
    );
  }
  if (bin === "wget") {
    return args.some((a) => a.startsWith("--post-") || a.startsWith("--body-") || (a.startsWith("--method") && !/=(GET|HEAD)$/i.test(a)));
  }
  return false;
}

/** Whether auto mode asks before this command: something that deletes,
 *  overwrites history, publishes or deploys, sends data out, ends
 *  processes, or reaches past the project — the hard to take back. Routine
 *  work (build, test, install into the project, run, git commit) goes
 *  ahead. A rule list, not a judge: it errs toward asking on what it names,
 *  and the sandbox still bounds what it does not. */
export function isRiskyCommand(cmd: string): boolean {
  const line = cmd.trim();
  if (!line) return false;
  // Something fetched, then run as it arrives.
  if (/\b(curl|wget)\b[^;&]*\|\s*(sudo\s+)?(ba|z|da|k|fi)?sh\b|\b(curl|wget)\b[^;&]*\|\s*(sudo\s+)?(python3?|node|ruby|perl)\b/.test(line)) {
    return true;
  }
  for (const words of commandsOf(line)) {
    const bin = base(words[0]);
    const args = words.slice(1);
    const pos = args.filter((a) => !a.startsWith("-"));
    if (RISKY.has(bin) || /^mkfs/.test(bin)) {
      // `brew` and the like: looking things up is fine.
      if (["brew", "apt", "apt-get", "port", "choco", "winget", "scoop", "snap"].includes(bin)) {
        if (["list", "info", "search", "show", "config", "doctor", "outdated", "deps", "uses", "desc", "home", "policy"].includes(pos[0] ?? "")) continue;
        if (args.length > 0 && args.every((a) => /^--?(version|help|v|h)$/.test(a))) continue;
      }
      return true;
    }
    if (bin === "git" && gitRisky(args)) return true;
    if (bin === "gh" && ghRisky(args)) return true;
    if (sends(bin, args)) return true;
    if (bin === "rsync" && (args.includes("--delete") || pos.some((a) => /^[^/]*:/.test(a)))) return true;
    if (bin === "find" && args.includes("-delete")) return true;
    if (bin === "defaults" && ["write", "delete", "import"].includes(pos[0] ?? "")) return true;
    if (["npm", "pnpm", "yarn", "bun"].includes(bin)) {
      if (pos.some((a) => PUBLISH.has(a))) return true;
      if (args.some((a) => a === "-g" || a === "--global" || a === "--location=global") || pos[0] === "global") return true;
    }
    if (["cargo", "gem", "poetry", "uv", "twine", "dotnet", "mvn", "gradle", "pip", "pip3"].includes(bin)) {
      if (pos.some((a) => PUBLISH.has(a) || a === "deploy")) return true;
      if ((bin === "pip" || bin === "pip3") && args.some((a) => a === "--break-system-packages")) return true;
    }
    if (REMOTE_TOOLS.has(bin) && pos.slice(0, 3).some((a) => REMOTE_CHANGE.has(a))) {
      // Containers built and run here are the project's own business.
      if ((bin === "docker" || bin === "podman") && !pos.slice(0, 3).some((a) => ["push", "rm", "rmi", "prune", "kill", "stop", "login", "logout"].includes(a))) {
        continue;
      }
      return true;
    }
  }
  return false;
}
