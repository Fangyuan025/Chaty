//! Finding the text an edit means when it is not written exactly as the file
//! has it — and saying precisely where it went wrong when it cannot be found.
//!
//! A model does not copy old_string out of the file; it retypes it from
//! memory, token by token, and the retyping drifts in ways that say nothing
//! about which lines it means: four spaces become two, a trailing space goes,
//! an emoji loses its variation selector (tokenizers rarely keep U+FE0F), a
//! curly quote comes back straight, the line numbers of an echoed region are
//! copied along with its text. Every such miss cost a full round — the whole
//! call written out, rejected, read again, written again — and on a long
//! block a small model often misses again the same way.
//!
//! So the search runs in tiers, strictest first, and each takes only a UNIQUE
//! place:
//!   1. the text as written (done by the caller: `str::matches`);
//!   2. line by line once each line is normalised — surrounding and repeated
//!      whitespace, invisible characters, quote and dash variants, CR — with
//!      blank lines set aside, and with copied line-number prefixes removed;
//!   3. the same with JSON escapes (`\n`, `\"`) undone;
//!   4. the one window of the file every line of which closely resembles the
//!      corresponding line of old_string, clearly ahead of any other.
//!
//! Applying a loose match keeps the lines the model did not mean to change as
//! the FILE has them: a line that reads the same in old_string and new_string
//! is context, and its retyped copy — with the emoji or the escape it could
//! not reproduce — must not overwrite the original.

use std::collections::HashSet;

/// How a match was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Whitespace, invisible characters, quote/dash variants or copied line
    /// numbers aside, the lines are the file's.
    Normalized,
    /// …after undoing JSON escapes written into the value.
    Unescaped,
    /// Closely similar, and unambiguously the closest.
    Similar,
}

#[derive(Debug)]
pub struct Located {
    pub tier: Tier,
    /// The file with the edit applied.
    pub text: String,
    /// 0-based line where the replaced span starts.
    pub start_line: usize,
    /// Lines the replacement occupies.
    pub span: usize,
    /// 1-based line range the match covered in the original file.
    pub lines: (usize, usize),
    /// Mean line similarity (1.0 below the Similar tier).
    pub similarity: f32,
}

#[derive(Debug)]
pub enum Miss {
    /// Nowhere; the report says where the closest place is and how it differs.
    NotFound(String),
    /// Several places once normalised (1-based first lines).
    Ambiguous(Vec<usize>),
}

/// Line similarity a window needs on EVERY line to count as a candidate — and
/// what a partly written old_string is checked against while it streams.
const LINE_FLOOR: f32 = 0.5;
/// Mean similarity a window needs to be taken without an exact match.
const ACCEPT_MULTI: f32 = 0.88;
const ACCEPT_SINGLE: f32 = 0.92;
/// How far ahead of the runner-up the best window has to be.
const MARGIN: f32 = 0.05;
/// Characters of a line compared for similarity (minified lines run to
/// thousands; their head says enough).
const SIM_CHARS: usize = 400;

// ───────────────────────────── normalisation ─────────────────────────────

fn invisible(c: char) -> bool {
    matches!(
        c,
        '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{00AD}' | '\u{FE0E}' | '\u{FE0F}'
    )
}

/// A line as compared: trimmed, whitespace runs collapsed, invisible
/// characters dropped, quote/dash/ellipsis variants folded.
pub fn norm_line(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut space = false;
    for c in line.chars() {
        if invisible(c) {
            continue;
        }
        let c = match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '\u{2212}' => '-',
            c if c.is_whitespace() => ' ',
            c => c,
        };
        if c == '\u{2026}' {
            out.push_str("...");
            space = false;
            continue;
        }
        if c == ' ' {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(c);
    }
    out
}

/// Line-number prefixes copied along with the text: this app's own echoes
/// (`   12  code`, `12:abc→code`) and the usual others (`cat -n`'s tab, `12→`,
/// `12 | `, `12│`). Taken off only when EVERY non-blank line carries the same
/// kind and the numbers climb with the lines — that is a listing, not content.
pub fn strip_line_numbers(s: &str) -> Option<String> {
    #[derive(PartialEq, Clone, Copy)]
    enum Sep {
        TwoSpaces,
        Tab,
        Anchor,
        Arrow,
        Bar,
        BoxBar,
    }
    fn split(line: &str) -> Option<(usize, Sep, &str)> {
        let lead = line.len() - line.trim_start_matches(' ').len();
        let rest = &line[lead..];
        let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits == 0 || digits > 7 {
            return None;
        }
        let n: usize = rest[..digits].parse().ok()?;
        let after = &rest[digits..];
        if after.trim().is_empty() {
            // A numbered blank line: `   12` or `   12  `.
            return Some((n, Sep::TwoSpaces, ""));
        }
        if let Some(r) = after.strip_prefix("  ") {
            return Some((n, Sep::TwoSpaces, r));
        }
        if let Some(r) = after.strip_prefix('\t') {
            return Some((n, Sep::Tab, r));
        }
        if let Some(r) = after.strip_prefix('→') {
            return Some((n, Sep::Arrow, r));
        }
        if let Some(r) = after.strip_prefix(" | ").or_else(|| after.strip_prefix(" |")) {
            return Some((n, Sep::Bar, r));
        }
        if let Some(r) = after.strip_prefix('│').or_else(|| after.strip_prefix(" │ ")) {
            return Some((n, Sep::BoxBar, r));
        }
        let r = after.strip_prefix(':')?;
        let hash: String = r.chars().take(3).collect();
        if hash.len() == 3 && hash.chars().all(|c| c.is_ascii_lowercase()) {
            return r[3..].strip_prefix('→').map(|body| (n, Sep::Anchor, body));
        }
        None
    }
    let lines: Vec<&str> = s.split('\n').collect();
    let mut sep: Option<Sep> = None;
    let mut last: Option<(usize, usize)> = None; // (number, line index)
    let mut out = Vec::with_capacity(lines.len());
    let mut numbered = 0;
    for (i, line) in lines.iter().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            out.push(String::new());
            continue;
        }
        let (n, kind, body) = split(line)?;
        // A numbered blank line says nothing about the separator.
        if !body.is_empty() {
            match sep {
                None => sep = Some(kind),
                Some(k) if k != kind => return None,
                _ => {}
            }
        }
        if let Some((pn, pi)) = last {
            if n != pn + (i - pi) {
                return None;
            }
        }
        last = Some((n, i));
        numbered += 1;
        out.push(body.to_string());
    }
    (numbered >= 1 && sep.is_some()).then(|| out.join("\n"))
}

/// `s` with its JSON escapes undone (`\n`, `\t`, `\"`, `\\`) — None when it
/// holds none to undo.
pub fn unescape_json_like(s: &str) -> Option<String> {
    if !(s.contains("\\n") || s.contains("\\\"") || s.contains("\\t")) {
        return None;
    }
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.peek().copied() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            _ => {
                out.push(c);
                continue;
            }
        }
        it.next();
    }
    Some(out)
}

/// Lines the model indented with spaces, in a file indented with tabs only,
/// re-indented with tabs. A Makefile recipe indented with spaces is a syntax
/// error (`make` says "missing separator"), and a model cannot see the
/// difference in what it read back: a Qwen3.5 4B, sampling with a repeat
/// penalty that made the tab token dearer, added `go vet` under `go test`
/// with four spaces. Only when every indented line of the file opens with a
/// tab (a single space — a block comment's ` *` — is alignment, not indent),
/// and only for lines the model wrote rather than copied from `old`.
pub fn tabbed_like_file(new: &str, old: &str, file: &str) -> Option<String> {
    let spaces = |l: &str| l.len() - l.trim_start_matches(' ').len();
    let mut tabbed = 0;
    for l in file.lines() {
        if l.starts_with('\t') {
            tabbed += 1;
        } else if spaces(l) >= 2 && !l.trim().is_empty() {
            return None;
        }
    }
    if tabbed < 2 {
        return None;
    }
    let copied: HashSet<&str> = old.lines().collect();
    let written = |l: &str| !copied.contains(l) && spaces(l) >= 2 && !l.trim().is_empty();
    // A tab is four spaces as models write it, unless they indent by two.
    let min = new.lines().filter(|l| written(l)).map(spaces).min()?;
    let width = if min == 2 { 2 } else { 4 };
    let mut out = String::with_capacity(new.len());
    for piece in new.split_inclusive('\n') {
        let line = piece.trim_end_matches(['\n', '\r']);
        if written(line) {
            let n = spaces(line);
            out.push_str(&"\t".repeat(n / width));
            out.push_str(&" ".repeat(n % width));
            out.push_str(&piece[n..]);
        } else {
            out.push_str(piece);
        }
    }
    (out != new).then_some(out)
}

/// Recipe lines of a Makefile the model wrote with no indentation at all.
/// A recipe line has to start with a tab — `make` stops with "missing
/// separator" otherwise — and models add `go vet ./...` under `go test ./...`
/// flush left (Gemma 4 E4B and a Qwen3.5 4B both did). Only lines that are
/// plainly recipe lines are touched: inside a rule (after its target line or
/// another recipe line, no blank line between), and not a target, an
/// assignment, a directive or a comment themselves.
pub fn makefile_recipe_tabs(new: &str, old: &str, path: &str) -> Option<String> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let is_make = name == "Makefile" || name == "makefile" || name == "GNUmakefile" || name.ends_with(".mk");
    if !is_make {
        return None;
    }
    let target = |l: &str| {
        let t = l.trim_end();
        match t.find(':') {
            Some(i) => !t[..i].contains('=') && !t[i..].starts_with(":=") && !t.starts_with('#'),
            None => false,
        }
    };
    let assignment = |l: &str| {
        let head = l.split('#').next().unwrap_or("");
        ["=", ":=", "?=", "+=", "!="].iter().any(|op| head.contains(op)) && !target(l)
    };
    const DIRECTIVES: &[&str] = &[
        "include", "-include", "sinclude", "ifeq", "ifneq", "ifdef", "ifndef", "else", "endif",
        "define", "endef", "export", "unexport", "override", "vpath", "private",
    ];
    let directive = |l: &str| DIRECTIVES.iter().any(|d| l == *d || l.starts_with(&format!("{d} ")) || l.starts_with(&format!("{d}(")));
    let mut in_rule = old.lines().next().is_some_and(|l| l.starts_with('\t') || target(l));
    let mut out = String::with_capacity(new.len() + 8);
    let mut changed = false;
    for piece in new.split_inclusive('\n') {
        let line = piece.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            in_rule = false;
        } else if line.starts_with('\t') {
            in_rule = true;
        } else if line.starts_with(' ') || line.starts_with('#') || directive(line) || assignment(line) {
            // left as written
        } else if target(line) {
            in_rule = true;
        } else if in_rule {
            out.push('\t');
            changed = true;
        }
        out.push_str(piece);
    }
    changed.then_some(out)
}

/// A value written as one line with its line breaks spelled out: `\n` and
/// `\t` as two characters each, where real ones were meant. A model does this
/// in formats that take text as written (Gemma 4 wrote a Makefile rule as
/// `test:\n\tgo test\n\tgo vet ./...`, and the file got those characters).
/// Taken as line breaks only when every sign agrees: the value has no real
/// line break, an escape stands outside any quoted string (inside one, `\n`
/// is what code says), and the text it replaces — `was` — spells none itself.
pub fn spelled_out_breaks(value: &str, was: &str) -> Option<String> {
    if value.contains('\n') || !value.contains("\\n") || was.contains("\\n") {
        return None;
    }
    let mut quote: Option<char> = None;
    let mut outside = false;
    let mut it = value.chars().peekable();
    while let Some(c) = it.next() {
        match (quote, c) {
            (Some(q), '\\') => {
                it.next();
                let _ = q;
            }
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'' | '`') => quote = Some(c),
            (None, '\\') => {
                if it.peek() == Some(&'n') {
                    outside = true;
                }
                it.next();
            }
            (None, _) => {}
        }
    }
    if !outside {
        return None;
    }
    unescape_json_like(value)
}

// ───────────────────────────── similarity ─────────────────────────────

/// 1 − edit distance / longer length, over the lines' first SIM_CHARS chars.
pub fn similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    let a: Vec<char> = a.chars().take(SIM_CHARS).collect();
    let b: Vec<char> = b.chars().take(SIM_CHARS).collect();
    let (la, lb) = (a.len(), b.len());
    let longer = la.max(lb);
    if longer == 0 {
        return 1.0;
    }
    // Lengths alone can rule a pair out below the floor.
    if (la.min(lb) as f32) < longer as f32 * LINE_FLOOR {
        return la.min(lb) as f32 / longer as f32;
    }
    let mut prev: Vec<usize> = (0..=lb).collect();
    let mut cur = vec![0usize; lb + 1];
    for i in 1..=la {
        cur[0] = i;
        for j in 1..=lb {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    1.0 - prev[lb] as f32 / longer as f32
}

/// Character-bigram Dice coefficient: cheap, for ranking whole windows when
/// only a report is wanted.
fn bigrams(s: &str) -> HashSet<(char, char)> {
    let v: Vec<char> = s.chars().take(SIM_CHARS).collect();
    v.windows(2).map(|w| (w[0], w[1])).collect()
}

fn dice(a: &HashSet<(char, char)>, b: &HashSet<(char, char)>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f32;
    2.0 * inter / (a.len() + b.len()).max(1) as f32
}

// ───────────────────────────── the file, line by line ─────────────────────────────

struct FileLines<'a> {
    text: &'a str,
    /// Byte offset where each line starts.
    starts: Vec<usize>,
    /// Each line, without its '\n' (a CRLF file's lines keep their '\r').
    raw: Vec<&'a str>,
    /// Indices of the non-blank lines, and their normalised text.
    nb: Vec<usize>,
    norm: Vec<String>,
}

impl<'a> FileLines<'a> {
    fn new(text: &'a str) -> Self {
        let mut starts = Vec::new();
        let mut raw = Vec::new();
        let mut at = 0;
        for line in text.split('\n') {
            starts.push(at);
            raw.push(line);
            at += line.len() + 1;
        }
        let mut nb = Vec::new();
        let mut norm = Vec::new();
        for (i, l) in raw.iter().enumerate() {
            let n = norm_line(l);
            if !n.is_empty() {
                nb.push(i);
                norm.push(n);
            }
        }
        FileLines { text, starts, raw, nb, norm }
    }

    /// Byte range of whole lines `a..=b` (without the '\n' after `b`).
    fn span(&self, a: usize, b: usize) -> (usize, usize) {
        (self.starts[a], self.starts[b] + self.raw[b].len())
    }
}

/// old_string (or new_string) as lines, with the non-blank ones normalised.
struct Needle {
    lines: Vec<String>,
    /// (index into `lines`, normalised text) of each non-blank line.
    nb: Vec<(usize, String)>,
}

impl Needle {
    fn new(s: &str) -> Self {
        let lines: Vec<String> =
            s.trim_matches('\n').split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect();
        let nb = lines
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                let n = norm_line(l);
                (!n.is_empty()).then_some((i, n))
            })
            .collect();
        Needle { lines, nb }
    }
}

/// Every window start (into `file.nb`) where the needle's non-blank lines
/// equal the file's, normalised.
fn normalized_hits(file: &FileLines, needle: &Needle) -> Vec<usize> {
    let n = needle.nb.len();
    if n == 0 || file.norm.len() < n {
        return Vec::new();
    }
    (0..=file.norm.len() - n)
        .filter(|&i| (0..n).all(|k| file.norm[i + k] == needle.nb[k].1))
        .collect()
}

/// The mean similarity of the window at `i`, or None when a line falls
/// below the floor.
fn window_score(file: &FileLines, needle: &Needle, i: usize) -> Option<f32> {
    let n = needle.nb.len();
    let mut sum = 0.0;
    for k in 0..n {
        let s = similarity(&file.norm[i + k], &needle.nb[k].1);
        if s < LINE_FLOOR {
            return None;
        }
        sum += s;
    }
    Some(sum / n as f32)
}

/// The best and runner-up windows by exact similarity, among the candidates
/// every line of which clears the floor.
fn similar_windows(file: &FileLines, needle: &Needle) -> Option<(usize, f32, f32)> {
    let n = needle.nb.len();
    if n == 0 || file.norm.len() < n {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    let mut second = 0.0f32;
    for i in 0..=file.norm.len() - n {
        if let Some(s) = window_score(file, needle, i) {
            match best {
                Some((_, b)) if s <= b => second = second.max(s),
                _ => {
                    if let Some((_, b)) = best {
                        second = second.max(b);
                    }
                    best = Some((i, s));
                }
            }
        }
    }
    best.map(|(i, s)| (i, s, second))
}

/// A changed line with the file's own spelling kept wherever the model's copy
/// of the old line and its new line agree. The model retyped `was` from memory
/// — straight quotes for curly ones, an emoji without its selector — and wrote
/// `now` the same way; the part it did not mean to change is taken from the
/// FILE, found by aligning `was` to it character by character. Lines too long
/// to align, or that share nothing, come back as written.
fn keep_file_spelling(was: &str, now: &str, file: &str) -> String {
    let w: Vec<char> = was.chars().collect();
    let n: Vec<char> = now.chars().collect();
    let f: Vec<char> = file.chars().collect();
    if w.len() > SIM_CHARS || f.len() > SIM_CHARS || w == f {
        return now.to_string();
    }
    let mut p = w.iter().zip(&n).take_while(|(a, b)| a == b).count();
    let mut s = w.iter().rev().zip(n.iter().rev()).take_while(|(a, b)| a == b).count();
    p = p.min(w.len()).min(n.len());
    s = s.min(w.len() - p).min(n.len() - p);
    if p == 0 && s == 0 {
        return now.to_string();
    }
    // LCS alignment of `was` onto the file line: for each position in `was`,
    // the file position it corresponds to.
    let (a, b) = (w.len(), f.len());
    let mut dp = vec![vec![0u16; b + 1]; a + 1];
    for x in (0..a).rev() {
        for y in (0..b).rev() {
            dp[x][y] = if w[x] == f[y] || norm_line(&w[x].to_string()) == norm_line(&f[y].to_string()) {
                dp[x + 1][y + 1] + 1
            } else {
                dp[x + 1][y].max(dp[x][y + 1])
            };
        }
    }
    let mut map = vec![0usize; a + 1];
    let (mut x, mut y) = (0, 0);
    while x < a {
        map[x] = y;
        if y < b && (w[x] == f[y] || norm_line(&w[x].to_string()) == norm_line(&f[y].to_string())) {
            x += 1;
            y += 1;
        } else if y < b && dp[x][y + 1] > dp[x + 1][y] {
            y += 1;
        } else {
            x += 1;
        }
    }
    map[a] = b;
    // Where `was`'s kept prefix ends and its kept suffix starts, in the file.
    let head_end = map[p];
    let tail_start = map[a - s].max(head_end);
    let mut out: String = f[..head_end].iter().collect();
    out.extend(&n[p..n.len() - s]);
    out.extend(&f[tail_start..]);
    out
}

/// The edit applied at window `i` of `file.nb`.
///
/// Lines of new_string that read the same as a line of old_string (paired in
/// order, by longest common subsequence) are context: they come out as the
/// FILE has them, with the blank lines the file keeps between them. The rest
/// are the change, re-indented from old_string's indentation to the file's.
fn apply_at(file: &FileLines, needle: &Needle, new: &str, i: usize) -> (String, usize, usize, (usize, usize)) {
    let n = needle.nb.len();
    let aligned: Vec<usize> = (0..n).map(|k| file.nb[i + k]).collect();
    let (first, last) = (aligned[0], aligned[n - 1]);
    let crlf = file.raw[first].ends_with('\r');

    let new_lines: Vec<&str> = new.trim_matches('\n').split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let new_norm: Vec<String> = new_lines.iter().map(|l| norm_line(l)).collect();

    // LCS between old's non-blank lines and new's lines, on normalised text.
    let (a, b) = (n, new_lines.len());
    let mut dp = vec![vec![0u32; b + 1]; a + 1];
    for x in (0..a).rev() {
        for y in (0..b).rev() {
            dp[x][y] = if !new_norm[y].is_empty() && needle.nb[x].1 == new_norm[y] {
                dp[x + 1][y + 1] + 1
            } else {
                dp[x + 1][y].max(dp[x][y + 1])
            };
        }
    }
    let mut pair: Vec<Option<usize>> = vec![None; b]; // new line → old non-blank index
    let (mut x, mut y) = (0, 0);
    while x < a && y < b {
        if !new_norm[y].is_empty() && needle.nb[x].1 == new_norm[y] {
            pair[y] = Some(x);
            x += 1;
            y += 1;
        } else if dp[x + 1][y] >= dp[x][y + 1] {
            x += 1;
        } else {
            y += 1;
        }
    }

    let indent = |l: &str| l[..l.len() - l.trim_start().len()].to_string();
    let from = indent(&needle.lines[needle.nb[0].0]);
    let to = indent(file.raw[first].trim_end_matches('\r'));
    let reindent = |l: &str| -> String {
        if from == to || l.is_empty() {
            return l.to_string();
        }
        match l.strip_prefix(from.as_str()) {
            Some(rest) => format!("{to}{rest}"),
            None => l.to_string(),
        }
    };

    // A changed run the same length as the lines it replaces is a line-for-line
    // rewrite: each new line is the old one changed, and keeps the file's
    // spelling where the model left it alone (keep_file_spelling).
    let mut rewrite: Vec<Option<usize>> = vec![None; b];
    {
        let (mut y, mut prev_k): (usize, Option<usize>) = (0, None);
        while y < b {
            if pair[y].is_some() {
                prev_k = pair[y];
                y += 1;
                continue;
            }
            let start = y;
            while y < b && pair[y].is_none() {
                y += 1;
            }
            let from = prev_k.map_or(0, |k| k + 1);
            let to = if y < b { pair[y].unwrap_or(a) } else { a };
            let olds: Vec<usize> = (from..to).collect();
            let news: Vec<usize> = (start..y).filter(|&j| !new_norm[j].is_empty()).collect();
            if olds.len() == news.len() {
                for (k, j) in olds.into_iter().zip(news) {
                    rewrite[j] = Some(k);
                }
            }
        }
    }

    let mut out: Vec<String> = Vec::with_capacity(b);
    let mut prev: Option<usize> = None;
    for (y, line) in new_lines.iter().enumerate() {
        let line: &str = &match rewrite[y] {
            Some(k) => {
                let was = needle.lines[needle.nb[k].0].trim();
                let file_line = file.raw[aligned[k]].trim_end_matches('\r');
                let kept = keep_file_spelling(was, line.trim(), file_line.trim());
                let lead = &line[..line.len() - line.trim_start().len()];
                format!("{lead}{kept}")
            }
            None => line.to_string(),
        };
        match pair[y] {
            Some(k) => {
                // The file's blank lines between two kept lines stay.
                if let Some(p) = prev {
                    if k == p + 1 {
                        for f in aligned[p] + 1..aligned[k] {
                            out.push(file.raw[f].to_string());
                        }
                    }
                }
                out.push(file.raw[aligned[k]].to_string());
                prev = Some(k);
            }
            None => {
                let l = reindent(line);
                out.push(if crlf { format!("{l}\r") } else { l });
                prev = None;
            }
        }
    }

    let (mut start, mut end) = file.span(first, last);
    let replacement = if new.trim_matches('\n').is_empty() {
        // Deleting lines takes their line break too, or a blank line is left
        // where they were.
        if end < file.text.len() {
            end += 1;
        } else {
            start = start.saturating_sub(1);
        }
        String::new()
    } else {
        out.join("\n")
    };
    let mut text = String::with_capacity(file.text.len() + replacement.len());
    text.push_str(&file.text[..start]);
    text.push_str(&replacement);
    text.push_str(&file.text[end..]);
    let span = replacement.matches('\n').count() + 1;
    (text, first, span, (first + 1, last + 1))
}

// ───────────────────────────── the search ─────────────────────────────

/// The variants of (old, new) worth trying, and the tier each earns.
fn variants(old: &str, new: &str) -> Vec<(String, String, Tier)> {
    let mut v = vec![(old.to_string(), new.to_string(), Tier::Normalized)];
    if let Some(o) = strip_line_numbers(old) {
        let nw = strip_line_numbers(new).unwrap_or_else(|| new.to_string());
        v.push((o, nw, Tier::Normalized));
    }
    let base = v.clone();
    for (o, nw, _) in base {
        if let Some(uo) = unescape_json_like(&o) {
            let un = unescape_json_like(&nw).unwrap_or(nw);
            v.push((uo, un, Tier::Unescaped));
        }
    }
    v
}

/// Where an edit whose old_string is not in the file verbatim should go.
/// The caller has already tried the exact text.
pub fn locate(text: &str, old: &str, new: &str) -> Result<Located, Miss> {
    let file = FileLines::new(text);
    let vs = variants(old, new);

    // Tiers 2 and 3: exact after a transform, then normalised lines.
    let mut ambiguous: Option<Vec<usize>> = None;
    for (o, nw, tier) in &vs {
        if o != old {
            let count = text.matches(o.as_str()).count();
            if count == 1 {
                let pos = text.find(o.as_str()).unwrap_or(0);
                let start_line = text[..pos].matches('\n').count();
                let span = nw.matches('\n').count() + 1;
                let lines = (start_line + 1, start_line + o.matches('\n').count() + 1);
                return Ok(Located {
                    tier: *tier,
                    text: text.replacen(o.as_str(), nw, 1),
                    start_line,
                    span,
                    lines,
                    similarity: 1.0,
                });
            }
        }
        let needle = Needle::new(o);
        let hits = normalized_hits(&file, &needle);
        match hits.len() {
            0 => {}
            1 => {
                let (text, start_line, span, lines) = apply_at(&file, &needle, nw, hits[0]);
                return Ok(Located { tier: *tier, text, start_line, span, lines, similarity: 1.0 });
            }
            _ => {
                ambiguous.get_or_insert_with(|| hits.iter().map(|&h| file.nb[h] + 1).collect());
            }
        }
    }
    if let Some(at) = ambiguous {
        return Err(Miss::Ambiguous(at));
    }

    // Tier 4: one clearly closest window.
    let mut best: Option<(f32, f32, usize, usize)> = None; // (score, runner-up, variant, window)
    for (vi, (o, _, _)) in vs.iter().enumerate() {
        let needle = Needle::new(o);
        if let Some((i, s, second)) = similar_windows(&file, &needle) {
            if best.is_none_or(|(bs, ..)| s > bs) {
                best = Some((s, second, vi, i));
            }
        }
    }
    if let Some((s, second, vi, i)) = best {
        let (o, nw, _) = &vs[vi];
        let needle = Needle::new(o);
        let need = if needle.nb.len() >= 2 { ACCEPT_MULTI } else { ACCEPT_SINGLE };
        if s >= need && s - second >= MARGIN {
            let (text, start_line, span, lines) = apply_at(&file, &needle, nw, i);
            return Ok(Located { tier: Tier::Similar, text, start_line, span, lines, similarity: s });
        }
    }
    Err(Miss::NotFound(report(&file, &vs)))
}

/// Could the old_string still being written match — given its complete
/// lines so far? False only when no place in the file has every one of those
/// lines above the similarity floor, which is also the least any match
/// `locate` takes: the answer never cuts off an edit that would have landed.
/// Fewer than two lines are not judged.
pub fn prefix_viable(text: &str, partial_old: &str) -> bool {
    let Some(cut) = partial_old.rfind('\n') else { return true };
    let complete = &partial_old[..cut];
    if complete.trim().is_empty() || text.contains(complete.trim_matches('\n')) {
        return true;
    }
    let file = FileLines::new(text);
    for (o, _, _) in variants(complete, "") {
        let needle = Needle::new(&o);
        if needle.nb.len() < 2 {
            return true;
        }
        let n = needle.nb.len();
        if file.norm.len() >= n && (0..=file.norm.len() - n).any(|i| window_score(&file, &needle, i).is_some()) {
            return true;
        }
    }
    false
}

/// What to tell the model when nothing matched: the closest place, line by
/// line, with the lines that differ marked — enough to write the next attempt
/// without another read.
fn report(file: &FileLines, vs: &[(String, String, Tier)]) -> String {
    // The needle to report against: numbers stripped if they were copied.
    let o = vs.iter().find(|(_, _, t)| *t == Tier::Normalized).map(|v| v.0.as_str()).unwrap_or("");
    let o = vs.get(1).filter(|v| v.2 == Tier::Normalized).map(|v| v.0.as_str()).unwrap_or(o);
    let needle = Needle::new(o);
    let n = needle.nb.len();
    let zh = !crate::agent::lang_is_en();
    let copy_hint = if zh {
        "按上面文件里的原样逐字复制要改的行(不要带行号),old_string 只需包含要改的行和一两行上下文;如果文件在你上次读取后改过,先 read_file 再改。"
    } else {
        "Copy the lines exactly as the file has them above (without the line numbers); old_string only needs the lines you change plus a line or two of context. If the file changed since you last read it, read_file first."
    };
    if n == 0 || file.norm.is_empty() {
        return if zh {
            "未找到 old_string(需与文件内容逐字匹配)".to_string()
        } else {
            "old_string not found — it must match the file content exactly".to_string()
        };
    }
    // Rank windows cheaply, then measure the leader exactly.
    let file_bi: Vec<HashSet<(char, char)>> = file.norm.iter().map(|l| bigrams(l)).collect();
    let needle_bi: Vec<HashSet<(char, char)>> = needle.nb.iter().map(|(_, l)| bigrams(l)).collect();
    let m = n.min(file.norm.len());
    let mut best = (0usize, -1.0f32);
    for i in 0..=file.norm.len() - m {
        let s: f32 = (0..m).map(|k| dice(&file_bi[i + k], &needle_bi[k])).sum::<f32>() / m as f32;
        if s > best.1 {
            best = (i, s);
        }
    }
    let (i, rough) = best;
    if rough < 0.3 {
        return if zh {
            "未找到 old_string:文件里没有和它相像的内容。文件可能在你上次读取后改过(比如被之前的编辑改动),或者你想的是另一个文件——先用 read_file 看它现在的内容再改。".to_string()
        } else {
            "old_string not found, and nothing in the file resembles it. The file may have changed since you last read it (an earlier edit?), or you have another file in mind — read_file it as it is now before editing.".to_string()
        };
    }
    let first = file.nb[i];
    let last = file.nb[i + m - 1];
    let pct = {
        let exact: f32 = (0..m).map(|k| similarity(&file.norm[i + k], &needle.nb[k].1)).sum::<f32>() / m as f32;
        (exact * 100.0).round() as u32
    };
    let differs: HashSet<usize> =
        (0..m).filter(|&k| file.norm[i + k] != needle.nb[k].1).map(|k| file.nb[i + k]).collect();
    let yours: std::collections::HashMap<usize, &str> = (0..m)
        .filter(|&k| file.norm[i + k] != needle.nb[k].1)
        .map(|k| (file.nb[i + k], needle.lines[needle.nb[k].0].trim()))
        .collect();
    let from = first.saturating_sub(2);
    let to = (last + 2).min(file.raw.len() - 1);
    let mut rows: Vec<String> = Vec::new();
    let mut yours_shown = 0;
    for f in from..=to {
        let mark = if differs.contains(&f) { "✗" } else { " " };
        let body: String = file.raw[f].trim_end_matches('\r').chars().take(200).collect();
        let mut row = format!("{mark}{:>5}  {body}", f + 1);
        if let Some(y) = yours.get(&f) {
            if yours_shown < 3 {
                let y: String = y.chars().take(120).collect();
                row.push_str(&if zh { format!("    ← 你写的是: {y}") } else { format!("    ← you wrote: {y}") });
                yours_shown += 1;
            }
        }
        rows.push(row);
    }
    // Long windows: the head and the tail say where; the middle is the file.
    if rows.len() > 36 {
        let tail = rows.split_off(rows.len() - 8);
        rows.truncate(24);
        rows.push("      …".to_string());
        rows.extend(tail);
    }
    let marked = differs.len();
    let head = if zh {
        format!(
            "未找到 old_string(需与文件内容逐字匹配)。最接近的是第 {}-{} 行(相似度 {pct}%),标 ✗ 的 {marked} 行和你写的不一样:",
            first + 1,
            last + 1
        )
    } else {
        format!(
            "old_string not found — it must match the file exactly. The closest place is lines {}-{} ({pct}% alike); the {marked} line(s) marked ✗ differ from what you wrote:",
            first + 1,
            last + 1
        )
    };
    format!("{head}\n{}\n{copy_hint}", rows.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(text: &str, old: &str, new: &str) -> Located {
        locate(text, old, new).unwrap_or_else(|m| panic!("expected a match, got {m:?}"))
    }

    #[test]
    fn a_makefile_recipe_line_written_flush_left_gets_its_tab() {
        // Gemma 4 E4B, verbatim in shape.
        assert_eq!(
            makefile_recipe_tabs("test:\n\tgo test ./...\ngo vet ./...", "test:\n\tgo test ./...", "Makefile").as_deref(),
            Some("test:\n\tgo test ./...\n\tgo vet ./...")
        );
        // Targets, assignments, directives, comments and what follows a blank
        // line are not recipe lines; other files are not Makefiles.
        let untouched = "test:\n\tgo test\n\nlint: fmt\nGOFLAGS := -v\ninclude common.mk\n# note\nall: test";
        assert_eq!(makefile_recipe_tabs(untouched, "x", "build/rules.mk"), None);
        assert_eq!(makefile_recipe_tabs("test:\n\tgo test\ngo vet", "x", "notes.md"), None);
    }

    #[test]
    fn lines_indented_with_spaces_follow_a_tab_indented_file() {
        let makefile = "build:\n\tgo build ./...\n\ntest:\n\tgo test ./...\n";
        assert_eq!(
            tabbed_like_file("\tgo test ./...\n    go vet ./...", "\tgo test ./...", makefile).as_deref(),
            Some("\tgo test ./...\n\tgo vet ./...")
        );
        // Two levels, and alignment past a whole indent kept as spaces.
        let go = "func a() {\n\tif x {\n\t\treturn\n\t}\n}\n";
        assert_eq!(
            tabbed_like_file("\tif x {\n        y()\n          // z\n\t}", "\tif x {\n\t}", go).as_deref(),
            Some("\tif x {\n\t\ty()\n\t\t  // z\n\t}")
        );
        // A file that indents with spaces anywhere is left to its own ways,
        // and so is one with nothing indented to go by.
        assert_eq!(tabbed_like_file("    x", "y", "a:\n\tb\n\tc\n    d\n"), None);
        assert_eq!(tabbed_like_file("    x", "y", "a\nb\n"), None);
        // A block comment's " *" is alignment, not an indent.
        assert!(tabbed_like_file("    x", "y", "/*\n * doc\n */\nfunc a() {\n\tb()\n\tc()\n}\n").is_some());
    }

    #[test]
    fn normalises_what_retyping_changes() {
        assert_eq!(norm_line("  let  x =\t1;  "), "let x = 1;");
        assert_eq!(norm_line("icon = \"☀\u{FE0F}\""), "icon = \"☀\"");
        assert_eq!(norm_line("say \u{201C}hi\u{201D} \u{2014} ok\u{2026}"), "say \"hi\" - ok...");
        assert_eq!(norm_line("a\u{200D}b\u{00A0}c"), "ab c");
    }

    #[test]
    fn line_numbers_copied_from_an_echo_come_off() {
        let echoed = "   12  fn a() {\n   13      b();\n   14  }";
        assert_eq!(strip_line_numbers(echoed).unwrap(), "fn a() {\n    b();\n}");
        let anchors = "4:abc→let x = 1;\n5:qwe→let y = 2;";
        assert_eq!(strip_line_numbers(anchors).unwrap(), "let x = 1;\nlet y = 2;");
        // Numbers that do not climb with the lines are content.
        assert!(strip_line_numbers("1  apple\n5  pear").is_none());
        assert!(strip_line_numbers("let x = 1;").is_none());
        // A blank line in between is still the same listing.
        assert_eq!(strip_line_numbers("   7  a\n\n   9  b").unwrap(), "a\n\nb");
    }

    #[test]
    fn a_retyped_block_lands_and_keeps_the_files_own_lines() {
        let text = "fn main() {\n    let icon = \"☀\u{FE0F}\";\n\n    println!(\"{icon}\");\n}\n";
        // The emoji lost its selector, the indentation went to two, and the
        // blank line was dropped.
        let old = "  let icon = \"☀\";\n  println!(\"{icon}\");";
        let new = "  let icon = \"☀\";\n  let n = 2;\n  println!(\"{icon}\");";
        let m = ok(text, old, new);
        assert_eq!(m.tier, Tier::Normalized);
        // Kept lines are the file's (selector and all), and the new one is
        // indented as the file is. The blank line went where the model's
        // new_string put its new line — it is the model's layout there.
        assert_eq!(
            m.text,
            "fn main() {\n    let icon = \"☀\u{FE0F}\";\n    let n = 2;\n    println!(\"{icon}\");\n}\n"
        );
    }

    #[test]
    fn blank_lines_between_kept_lines_stay() {
        let text = "a\n\nb\nc\n";
        let m = ok(text, "a\nb", "a\nb\nB2");
        assert_eq!(m.text, "a\n\nb\nB2\nc\n");
    }

    #[test]
    fn deleting_lines_takes_their_line_break() {
        let text = "keep\n  drop me\n  and me\nkeep too\n";
        let m = ok(text, "drop me\nand me", "");
        assert_eq!(m.text, "keep\nkeep too\n");
    }

    #[test]
    fn crlf_files_stay_crlf() {
        let text = "one\r\n  two\r\nthree\r\n";
        let m = ok(text, "two\nthree", "two\n2.5\nthree");
        assert_eq!(m.text, "one\r\n  two\r\n  2.5\r\nthree\r\n");
    }

    #[test]
    fn copied_line_numbers_still_find_the_lines() {
        let text = "x = 1\ny = 2\nz = 3\n";
        let m = ok(text, "    2  y = 2\n    3  z = 3", "    2  y = 20\n    3  z = 3");
        assert_eq!(m.text, "x = 1\ny = 20\nz = 3\n");
    }

    #[test]
    fn escapes_written_out_are_undone() {
        let text = "a = \"q\"\nb = 1\n";
        let m = ok(text, "a = \\\"q\\\"\\nb = 1", "a = \\\"r\\\"\\nb = 1");
        assert_eq!(m.tier, Tier::Unescaped);
        assert_eq!(m.text, "a = \"r\"\nb = 1\n");
    }

    #[test]
    fn a_close_block_is_taken_only_when_it_is_clearly_the_one() {
        let text = "function buildWave(w){\n  const q=[];\n  for (let i = 0; i < w; i++) q.push('grunt');\n  return q;\n}\n";
        // One retyped line drifted (spacing inside the call) — still clearly here.
        let old = "  const q=[];\n  for (let i=0; i<w; i++) q.push('grunt');\n  return q;";
        let new = "  let q=[];\n  for (let i=0; i<w; i++) q.push('grunt');\n  return q;";
        let m = ok(text, old, new);
        assert_eq!(m.tier, Tier::Similar);
        // The kept line is the file's spelling, not the drifted one.
        assert!(m.text.contains("  let q=[];\n  for (let i = 0; i < w; i++) q.push('grunt');"), "{}", m.text);

        // Two equally close blocks: not guessed.
        let twice = "if (a) {\n  run(1);\n}\nif (b) {\n  run(1);\n}\n";
        assert!(locate(twice, "if (c) {\n  run(1);", "x").is_err());
    }

    #[test]
    fn a_changed_line_keeps_the_files_spelling_where_it_was_not_changed() {
        // Verbatim shape from a Qwen2 7B run: the model's copy of the line has
        // a straight quote where the file has a curly one, in old and new alike.
        let text = "def greet(name):\n    message = f\"\u{201C}Welcome back\u{201D}, {name}!\"\n    return message\n";
        let old = "    message = f\"\u{201C}Welcome back\", {name}!\"";
        let new = "    message = f\"\u{201C}Welcome home\", {name}!\"";
        let m = ok(text, old, new);
        assert!(m.text.contains("f\"\u{201C}Welcome home\u{201D}, {name}!\""), "{}", m.text);
        // An emoji whose selector the model dropped keeps it.
        let emoji = "    case .mild: return \"\u{2600}\u{FE0F}\" // sunny\n";
        let m = ok(emoji, "case .mild: return \"\u{2600}\" // sunny", "case .mild: return \"\u{2600}\" // clear");
        assert_eq!(m.text, "    case .mild: return \"\u{2600}\u{FE0F}\" // clear\n");
    }

    #[test]
    fn line_breaks_spelled_out_are_read_as_line_breaks_only_when_meant() {
        assert_eq!(
            spelled_out_breaks("test:\\n\\tgo test ./...\\n\\tgo vet ./...", "test:").as_deref(),
            Some("test:\n\tgo test ./...\n\tgo vet ./...")
        );
        // Inside a string it is what the code says.
        assert!(spelled_out_breaks("printf(\"done\\n\");", "printf(\"ok\");").is_none());
        // Already has real line breaks, or the replaced text spells them too.
        assert!(spelled_out_breaks("a\\nb\nc", "a").is_none());
        assert!(spelled_out_breaks("x\\ny", "w\\nz").is_none());
    }

    #[test]
    fn a_normalised_match_in_two_places_is_named_not_guessed() {
        let text = "  x = 1\n  y = 2\n\n  x = 1\n  y = 2\n";
        match locate(text, "x = 1\ny = 2", "x = 3\ny = 2") {
            Err(Miss::Ambiguous(at)) => assert_eq!(at, vec![1, 4]),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_miss_shows_the_closest_lines_and_what_differs() {
        let text = "def add(a, b):\n    return a + b\n\ndef total(items, tax_rate):\n    s = sum(items)\n    return s * (1 + tax_rate)\n";
        let old = "def total(items, tax):\n    s = sum(items)\n    return s * (1 + tax)";
        let Err(Miss::NotFound(r)) = locate(text, old, "x") else { panic!("should miss") };
        assert!(r.contains("✗    4  def total(items, tax_rate):"), "{r}");
        assert!(r.contains("def total(items, tax):"), "the model's own line is quoted: {r}");
        assert!(r.contains("     5      s = sum(items)"), "unchanged lines unmarked: {r}");
    }

    #[test]
    fn nothing_alike_says_so() {
        let Err(Miss::NotFound(r)) = locate("alpha\nbeta\n", "zzzz qqqq\nwwww", "x") else { panic!() };
        assert!(r.contains("resembles") || r.contains("相像"), "{r}");
    }

    #[test]
    fn a_prefix_is_cut_only_when_nothing_could_match() {
        let text = "fn a() {\n    let x = 1;\n    let y = 2;\n}\n";
        assert!(prefix_viable(text, "fn a() {\n    let x = 1;\n    let"));
        // Drifted but plausible lines are not cut.
        assert!(prefix_viable(text, "fn a() {\n  let x=1;\n"));
        // Lines that are nowhere in the file are.
        assert!(!prefix_viable(text, "class Widget extends Base {\n  render(props) {\n"));
        // One line is not judged.
        assert!(prefix_viable(text, "class Widget extends Base {\n"));
    }
}
