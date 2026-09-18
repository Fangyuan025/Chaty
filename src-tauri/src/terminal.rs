//! Commands the model can talk to.
//!
//! A command that asks something — `[Y/n]`, `Password:`, a REPL's `>>>`, a
//! scaffolder's arrow-key menu — used to get end-of-file on stdin and fail, or
//! sit until the timeout killed it. Code mode now keeps such a command running
//! in the background and lets the model type into it (`bg_input`).
//!
//! Two ways a command gets a keyboard:
//!
//! * a whole terminal (pseudo-terminal; ConPTY on Windows), for programs that
//!   draw on the screen or need one to start at all — REPLs, editors, TUIs,
//!   project scaffolders. What they show is read back as a terminal screen.
//! * for every other foreground `bash` command, only its input: stdin is a
//!   terminal on unix (so `read -p` and `input()` prompt as they do for a
//!   person) and an open pipe on Windows. Output stays on pipes exactly as
//!   before, so colours, pagers and progress bars do not change. A command that
//!   stops and asks is moved to the background, the way a dev server is.
//!
//! This file holds the parts that do not depend on the agent's registry: the
//! terminal itself, reading its screen, naming keys, and recognising a prompt.

use std::io::{Read, Write};
use std::path::Path;

/// The size of the terminal a command is given — and of the screen the model
/// reads back.
pub const ROWS: u16 = 40;
pub const COLS: u16 = 160;

/// A command running in a terminal of its own.
pub struct PtyProcess {
    pub reader: Box<dyn Read + Send>,
    pub writer: Box<dyn Write + Send>,
    pub child: Box<dyn portable_pty::Child + Send + Sync>,
    /// Kept for the life of the command: dropping it closes the terminal (on
    /// Windows that is what lets the reader see the end).
    pub master: Box<dyn portable_pty::MasterPty + Send>,
    pub pid: u32,
}

/// Start `program args…` in a fresh terminal, with the environment changes
/// `envs` (`None` removes a variable) on top of this process's own.
pub fn spawn_pty(
    program: &std::ffi::OsStr,
    args: &[std::ffi::OsString],
    envs: &[(std::ffi::OsString, Option<std::ffi::OsString>)],
    cwd: &Path,
) -> Result<PtyProcess, String> {
    let pty = portable_pty::native_pty_system();
    let pair = pty
        .openpty(portable_pty::PtySize { rows: ROWS, cols: COLS, pixel_width: 0, pixel_height: 0 })
        .map_err(|e| format!("pty: {e:#}"))?;
    let mut cmd = portable_pty::CommandBuilder::new(program);
    cmd.args(args);
    for (k, v) in envs {
        match v {
            Some(v) => cmd.env(k, v),
            None => cmd.env_remove(k),
        }
    }
    // A terminal program's defaults for what it may assume of the terminal.
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLUMNS", COLS.to_string());
    cmd.env("LINES", ROWS.to_string());
    cmd.cwd(cwd);
    let child = pair.slave.spawn_command(cmd).map_err(|e| format!("spawn failed: {e:#}"))?;
    drop(pair.slave);
    let reader = pair.master.try_clone_reader().map_err(|e| format!("pty: {e:#}"))?;
    let writer = pair.master.take_writer().map_err(|e| format!("pty: {e:#}"))?;
    let pid = child.process_id().unwrap_or(0);
    Ok(PtyProcess { reader, writer, child, master: pair.master, pid })
}

/// A terminal for a command's stdin alone (unix). Not made the command's
/// controlling terminal: a shell that exits would then hang up everything it
/// left running, and `npm run dev &` must outlive the shell that started it.
#[cfg(unix)]
pub fn stdin_tty() -> std::io::Result<(std::fs::File, std::fs::File)> {
    use std::os::unix::io::FromRawFd;
    let mut master: libc::c_int = -1;
    let mut slave: libc::c_int = -1;
    let mut size = libc::winsize { ws_row: ROWS, ws_col: COLS, ws_xpixel: 0, ws_ypixel: 0 };
    let rc = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut size,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    for fd in [master, slave] {
        unsafe {
            libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
        }
    }
    Ok(unsafe { (std::fs::File::from_raw_fd(master), std::fs::File::from_raw_fd(slave)) })
}

/// Whether the program on this terminal has switched it out of line mode —
/// what a menu, an editor or a REPL with line editing does before it waits
/// for a key.
#[cfg(unix)]
pub fn tty_raw(master: &std::fs::File) -> bool {
    use std::os::unix::io::AsRawFd;
    fd_raw(master.as_raw_fd())
}

/// `tty_raw` for a terminal known by its descriptor.
#[cfg(unix)]
pub fn fd_raw(fd: std::os::unix::io::RawFd) -> bool {
    let mut t: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut t) } != 0 {
        return false;
    }
    t.c_lflag & libc::ICANON == 0
}

/// The screen a terminal shows after `bytes`, as text: trailing blanks trimmed
/// from each line and blank lines from the end. `lf_only` is output that came
/// through a pipe, where a line ends in `\n` alone and a terminal would stair-
/// step it.
pub fn render_screen(bytes: &[u8], lf_only: bool) -> String {
    let mut parser = vt100::Parser::new(ROWS, COLS, 0);
    if lf_only {
        parser.process(&crlf(bytes));
    } else {
        parser.process(bytes);
    }
    let text = parser.screen().contents();
    let mut lines: Vec<&str> = text.lines().map(|l| l.trim_end()).collect();
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    while lines.first().is_some_and(|l| l.is_empty()) {
        lines.remove(0);
    }
    lines.join("\n")
}

/// `\n` → `\r\n`, leaving existing `\r\n` alone.
pub fn crlf(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + bytes.len() / 16);
    let mut prev = 0u8;
    for &b in bytes {
        if b == b'\n' && prev != b'\r' {
            out.push(b'\r');
        }
        out.push(b);
        prev = b;
    }
    out
}

/// Terminal output as a plain log: escape sequences dropped, and a line that
/// was redrawn with `\r` (a progress bar) kept as it last read.
pub fn plain_log(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            // CSI: parameters, then one final byte in @..~
            Some('[') => {
                chars.next();
                for d in chars.by_ref() {
                    if ('@'..='~').contains(&d) {
                        break;
                    }
                }
            }
            // OSC: up to BEL or ESC \
            Some(']') => {
                chars.next();
                while let Some(d) = chars.next() {
                    if d == '\u{7}' {
                        break;
                    }
                    if d == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            Some(_) => {
                chars.next();
            }
            None => {}
        }
    }
    out.split('\n')
        .map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            line.rsplit('\r').next().unwrap_or(line)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The bytes a named key sends. `terminal` is whether the command reads a
/// terminal (Enter is `\r` there, and a line ends in `\n` on a pipe).
pub fn key_bytes(name: &str, terminal: bool) -> Option<Vec<u8>> {
    let n: String = name.trim().to_ascii_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    let n = n.replace('+', "-");
    let fixed: &[u8] = match n.as_str() {
        "enter" | "return" => {
            return Some(if terminal { b"\r".to_vec() } else if cfg!(windows) { b"\r\n".to_vec() } else { b"\n".to_vec() })
        }
        "tab" => b"\t",
        "shift-tab" | "backtab" => b"\x1b[Z",
        "esc" | "escape" => b"\x1b",
        "backspace" => b"\x7f",
        "delete" | "del" => b"\x1b[3~",
        "space" => b" ",
        "up" | "arrowup" => b"\x1b[A",
        "down" | "arrowdown" => b"\x1b[B",
        "right" | "arrowright" => b"\x1b[C",
        "left" | "arrowleft" => b"\x1b[D",
        "home" => b"\x1b[H",
        "end" => b"\x1b[F",
        "pageup" | "pgup" => b"\x1b[5~",
        "pagedown" | "pgdn" => b"\x1b[6~",
        _ => {
            // ctrl-x / ^x
            let letter = n.strip_prefix("ctrl-").or_else(|| n.strip_prefix('^'))?;
            let mut it = letter.chars();
            let (Some(c), None) = (it.next(), it.next()) else { return None };
            if !c.is_ascii_lowercase() {
                return None;
            }
            return Some(vec![c as u8 - b'a' + 1]);
        }
    };
    Some(fixed.to_vec())
}

/// Names `key_bytes` knows, for a message that has to list them.
pub const KEY_NAMES: &str = "enter, tab, shift-tab, esc, backspace, delete, space, up, down, left, right, home, end, pageup, pagedown, ctrl-a … ctrl-z";

/// The line a command stopped on, when it reads like a question put to a
/// person — the last non-empty line of its output. Heuristic by nature: a
/// command is only asked about once it has stopped printing.
pub fn prompt_line(output: &str) -> Option<String> {
    prompt(output).map(|p| p.line)
}

/// A question a command stopped on.
pub struct Prompt {
    pub line: String,
    /// Unmistakably a question — a choice offered, a key asked for, a REPL's
    /// prompt, a password — rather than a line that merely ends in `:` or
    /// `?`, which a command may also print before it gets on with its work
    /// (`printf "Building: "`). A weak one is only believed after a longer
    /// silence.
    pub strong: bool,
}

pub fn prompt(output: &str) -> Option<Prompt> {
    let lines: Vec<&str> = output.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let last = *lines.last()?;
    if last.chars().count() > 160 {
        return None;
    }
    let strong = |line: &str| Some(Prompt { line: line.to_string(), strong: true });
    let lower = last.to_lowercase();
    // A choice asked for, anywhere on the line: [Y/n], (yes/no), (y), （是/否）
    let choice = regex::Regex::new(
        r"(?i)[\[(（【]\s*(?:y(?:es)?\s*/\s*n(?:o)?|n(?:o)?\s*/\s*y(?:es)?|y|yes|是\s*/\s*否|否\s*/\s*是)\s*[\])）】]",
    )
    .unwrap();
    if choice.is_match(last) {
        return strong(last);
    }
    // Not "to quit": a server's "Press CTRL+C to quit" asks nothing.
    const WORDS: [&str; 13] = [
        "press any key",
        "press enter",
        "press return",
        "hit enter",
        "to continue",
        "password",
        "passphrase",
        "按任意键",
        "按回车",
        "请输入",
        "请选择",
        "密码",
        "口令",
    ];
    if WORDS.iter().any(|w| lower.contains(w)) {
        return strong(last);
    }
    // The prompt of an arrow-key menu: the active question's marker is on one
    // of the last few lines, not necessarily the last.
    let tail = &lines[lines.len().saturating_sub(6)..];
    if tail.iter().any(|l| l.starts_with('◆') || l.starts_with("? ") || l.contains('❯') || l.contains("(Use arrow keys)")) {
        return strong(last);
    }
    let end = last.chars().last()?;
    match end {
        '›' | '»' | '❯' => strong(last),
        // `>>>`, `sqlite>`, `irb(main):001>` — but not the end of an HTML tag.
        '>' if !last.contains('<') => strong(last),
        // A shell's own prompt: `bash-3.2$`, `root@host:/#`
        '$' | '#' if last.chars().count() <= 60 => strong(last),
        // Full-width too: a Chinese program asks with "？" and "：".
        '?' | ':' | '？' | '：' => Some(Prompt { line: last.to_string(), strong: false }),
        _ => None,
    }
}

/// The part of a shell command that runs last — after the last `&&`, `||`, `;`
/// or `|` outside quotes — and whether its input is piped in.
fn last_segment(command: &str) -> (&str, bool) {
    let b = command.as_bytes();
    let mut quote: Option<u8> = None;
    let mut start = 0;
    let mut piped = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match quote {
            Some(q) => {
                if c == b'\\' && q == b'"' {
                    i += 1;
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'\'' | b'"' => quote = Some(c),
                b'\\' => i += 1,
                b';' | b'|' | b'&' => {
                    // A lone `|` feeds the next command; `||` does not.
                    piped = c == b'|' && b.get(i + 1) != Some(&b'|') && (i == 0 || b[i - 1] != b'|');
                    start = i + 1;
                }
                _ => {}
            },
        }
        i += 1;
    }
    (command[start.min(command.len())..].trim(), piped)
}

/// The words of a simple command, quotes removed and leading `VAR=value`
/// assignments skipped.
fn words(segment: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for c in segment.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None if c == '\'' || c == '"' => {
                quote = Some(c);
                any = true;
            }
            None if c.is_whitespace() => {
                if any || !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            None => cur.push(c),
        }
    }
    if any || !cur.is_empty() {
        out.push(cur);
    }
    while out.first().is_some_and(|w| w.contains('=') && !w.starts_with('-')) {
        out.remove(0);
    }
    out
}

/// Commands that are a conversation or a screen rather than a job: they only
/// work in a terminal and do nothing until they are typed into. Run as one
/// from the start.
pub fn interactive_command(command: &str) -> bool {
    let (seg, piped) = last_segment(command);
    // Input piped in, or a heredoc: the answers are already given.
    if piped || command.contains("<<") || seg.contains(" < ") {
        return false;
    }
    let w = words(seg);
    let Some(first) = w.first() else { return false };
    let prog = first.rsplit(['/', '\\']).next().unwrap_or(first).to_ascii_lowercase();
    let prog = prog.strip_suffix(".exe").unwrap_or(&prog).to_string();
    let rest: Vec<&str> = w[1..].iter().map(String::as_str).collect();
    let operands = rest.iter().filter(|a| !a.starts_with('-')).count();
    let has = |f: &str| rest.contains(&f);
    // `bash -c 'npm init'`: the command is the script it is given.
    if matches!(prog.as_str(), "bash" | "sh" | "zsh" | "dash") {
        return rest
            .iter()
            .position(|a| *a == "-c" || *a == "-lc")
            .and_then(|i| rest.get(i + 1))
            .is_some_and(|script| interactive_command(script));
    }
    match prog.as_str() {
        // Screens.
        "vim" | "vi" | "nvim" | "nano" | "emacs" | "less" | "more" | "top" | "htop" | "btop" | "tmux"
        | "screen" | "man" | "watch" => true,
        // REPLs, when started as one: no script to run, or asked for.
        "python" | "python3" | "ipython" | "bpython" | "irb" | "pry" | "ghci" | "lua" | "julia" | "iex"
        | "erl" | "clj" | "sbcl" | "node" | "deno" | "bun" => {
            if has("-i") || has("--interactive") {
                return true;
            }
            match prog.as_str() {
                "deno" | "bun" if rest.first() == Some(&"create") => true,
                "node" | "deno" | "bun" => rest.is_empty() || rest == ["repl"],
                "ipython" | "bpython" | "irb" | "pry" | "ghci" => operands == 0,
                _ => rest.is_empty() || rest.iter().all(|a| matches!(*a, "-q" | "-u" | "-B")),
            }
        }
        // A Windows shell with nothing to run is a shell to talk to. With a
        // command to run (-Command, /C, -File) it is an ordinary command.
        "cmd" | "powershell" | "pwsh" => !rest.iter().any(|a| {
            let a = a.to_ascii_lowercase();
            matches!(a.as_str(), "/c" | "/k" | "-c" | "-command" | "-f" | "-file" | "-encodedcommand")
        }),
        "sqlite3" => operands <= 1,
        "psql" => !has("-c") && !has("-f") && !has("--command") && !has("--file"),
        "mysql" | "mariadb" => !has("-e") && !has("--execute"),
        "redis-cli" | "mongosh" | "mongo" => operands == 0,
        "ssh" => operands == 1,
        "sftp" | "ftp" | "telnet" => true,
        // Project scaffolders: a questionnaire unless told to skip it.
        "npm" | "pnpm" | "yarn" => {
            let sub = rest.first().copied().unwrap_or("");
            let skip = has("-y") || has("--yes");
            sub == "create"
                || (sub == "init" && operands >= 2)
                || (sub == "init" && !skip && (prog == "npm" || prog == "yarn"))
        }
        "npx" | "pnpx" | "bunx" => rest
            .iter()
            .find(|a| !a.starts_with('-'))
            .is_some_and(|p| {
                let p = p.rsplit('/').next().unwrap_or(p);
                p.starts_with("create-") || p.starts_with("@angular/cli") && rest.contains(&"new")
            }),
        "ng" => rest.first() == Some(&"new"),
        "vue" => rest.first() == Some(&"create"),
        _ => false,
    }
}

/// Test runners, which change into a watch mode that never exits when their
/// stdin is a terminal. They keep the old stdin: nothing.
pub fn test_command(command: &str) -> bool {
    let c = command.to_ascii_lowercase();
    ["vitest", "jest", "react-scripts test", "ng test", "karma", "mocha"]
        .iter()
        .any(|t| c.contains(t))
        || regex::Regex::new(r"\b(npm|pnpm|yarn|bun)\s+(run\s+)?test\b").unwrap().is_match(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Windows shell with nothing to run is a conversation; one that was
    /// given a command to run is not (`powershell -Command …` is how half the
    /// Windows commands a model writes look).
    #[test]
    fn a_windows_shell_with_nothing_to_run_is_a_conversation() {
        for c in ["cmd", "cmd.exe", "powershell", "pwsh", "powershell -NoProfile", "powershell -NoLogo"] {
            assert!(interactive_command(c), "{c}");
        }
        for c in [
            "cmd /C echo hi",
            "powershell -NoProfile -Command \"Write-Output hi\"",
            "pwsh -File build.ps1",
            "echo hi | cmd",
        ] {
            assert!(!interactive_command(c), "{c}");
        }
    }

    #[test]
    fn a_question_put_to_a_person_is_a_prompt() {
        for p in [
            "Do you want to continue? [Y/n]",
            "Need to install the following packages:\n  create-vite@6\nOk to proceed? (y)",
            "Overwrite existing file? (yes/no)",
            "Enter your name:",
            "Password:",
            ">>>",
            "sqlite>",
            "irb(main):001>",
            "? Project name › vite-project",
            "◆  Select a framework:\n│  ● Vanilla\n│  ○ Vue\n└",
            "Press any key to continue . . .",
            "bash-3.2$",
            "第 1 次，你猜是几？",
            "姓名：",
            "确定要删除吗？（是/否）",
            "请输入验证码",
        ] {
            assert!(prompt_line(p).is_some(), "{p}");
        }
        assert!(prompt("确定要删除吗？（是/否）").unwrap().strong);
        assert!(!prompt("第 1 次，你猜是几？").unwrap().strong);
        assert!(prompt("Do you want to continue? [Y/n]").unwrap().strong);
        assert!(prompt("Password:").unwrap().strong);
        assert!(!prompt("Enter your name:").unwrap().strong, "a colon alone is a weak question");
        assert!(!prompt("Building:").unwrap().strong);
        for p in [
            "Compiling chaty v2.2.1",
            "</html>",
            "added 212 packages in 4s",
            "Cloning into 'repo'...",
            "",
            "Step 3/10 : RUN npm install",
        ] {
            assert!(prompt_line(p).is_none(), "{p}");
        }
    }

    #[test]
    fn commands_that_are_a_conversation_start_in_a_terminal() {
        for c in [
            "python3",
            "python3 -i",
            "node",
            "cd app && npm create vite@latest",
            "npx create-next-app@latest",
            "npm init",
            "bun create vite",
            "sqlite3 app.db",
            "psql -d shop",
            "ssh devbox",
            "vim notes.md",
            "FOO=1 top",
            "/bin/bash -c 'npm init'",
        ] {
            assert!(interactive_command(c), "{c}");
        }
        for c in [
            "python3 app.py",
            "python3 -c 'print(1)'",
            "node server.js",
            "npm init -y",
            "npm install",
            "sqlite3 app.db 'select 1'",
            "psql -d shop -c 'select 1'",
            "ssh devbox uptime",
            "echo y | npm init",
            "python3 <<EOF\nprint(1)\nEOF",
            "git status",
            "bash -c 'npm run build'",
            "bash deploy.sh",
        ] {
            assert!(!interactive_command(c), "{c}");
        }
    }

    #[test]
    fn test_runners_keep_a_closed_stdin() {
        assert!(test_command("npx vitest"));
        assert!(test_command("npm test"));
        assert!(test_command("cd web && npm run test -- --reporter dot"));
        assert!(!test_command("npm run build"));
        assert!(!test_command("python3 -m pytest"));
    }

    #[test]
    fn keys_are_the_bytes_a_terminal_sends() {
        assert_eq!(key_bytes("Enter", true).unwrap(), b"\r");
        assert_eq!(key_bytes("down", true).unwrap(), b"\x1b[B");
        assert_eq!(key_bytes("Ctrl+C", true).unwrap(), vec![3]);
        assert_eq!(key_bytes("^d", true).unwrap(), vec![4]);
        assert!(key_bytes("hyper", true).is_none());
    }

    #[test]
    fn a_screen_reads_as_it_is_drawn() {
        // A menu redrawn in place: the last drawing is what shows.
        let frames = b"? Pick one\r\n\x1b[32m> a\x1b[0m\r\n  b\r\n\x1b[2A\r\x1b[2K  a\r\n\x1b[2K> b\r\n";
        assert_eq!(render_screen(frames, false), "? Pick one\n  a\n> b");
        // Pipe output with bare \n lines does not stair-step.
        assert_eq!(render_screen(b"one\ntwo\n", true), "one\ntwo");
        assert_eq!(plain_log(b"\x1b[1mbold\x1b[0m\r\n10%\r50%\r100%\n"), "bold\n100%\n");
    }
}
