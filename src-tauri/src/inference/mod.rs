//! Inference abstraction layer.
//!
//! Every engine (mock, llama.cpp, candle/mistral.rs, remote …) implements
//! [`InferenceBackend`]. The rest of the app only ever talks to this trait, so
//! swapping or adding engines never touches the command/UI layer.

pub mod audio;
pub mod gguf;
pub mod jinja;
pub mod llama;
pub mod llama_embed;
pub mod llama_log;
pub mod llama_sidecar;
pub use llama::llama_backend_pub;
pub mod mlx;
pub mod mock;
pub mod sd;

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

/// Live sidecar PIDs — the llama.cpp and MLX engines, the image engine and the
/// music engine all run as child processes. The app's quit path exits via `libc::_exit` (dodging a ggml
/// teardown SIGABRT), which skips every destructor — without an explicit reap,
/// quitting while a model is loaded orphans a sidecar that keeps the entire
/// model resident. lib.rs calls `kill_sidecars_now` on exit. (On Windows the
/// sidecars see their stdin close when the app goes and exit by themselves.)
pub(crate) static SIDECAR_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// A model load the user cancelled. Set by `cancel_load`, cleared as the next
/// load begins: while it is set, a sidecar that dies inside the load is not a
/// crash — no smaller GPU offload is tried, no runtime pack is set aside as
/// broken, and the bundled engine is not tried in its place.
pub(crate) static LOAD_CANCELLED: AtomicBool = AtomicBool::new(false);

pub fn load_cancelled() -> bool {
    LOAD_CANCELLED.load(std::sync::atomic::Ordering::SeqCst)
}

/// The chat's Skip on a reply that is still thinking: end the thought where it
/// stands and answer. Set by `skip_thinking` and taken once by the engine
/// writing the reply, which closes the thought in the reply's own markup and
/// goes on sampling from there — what was thought stays thought, the cache
/// holds the reply exactly as it reads. Each engine process has its own; the
/// app passes a Skip on to the sidecar doing the work. Cleared as a reply
/// starts, so one asked for too late never ends the next reply's thought.
pub static THINK_SKIP: AtomicBool = AtomicBool::new(false);

/// What ends the thought `text` — a reply as streamed so far, the opener the
/// engine sent first included — is in the middle of, written the way its model
/// writes it; `None` when it is not in one. The markups are the ones the
/// chat's reasoning panel reads: `<think>`, a namespaced `<ifm|think…>` (K2
/// Horizon), Gemma 4's thought channel, Harmony's analysis channel and ATEM's
/// `to=self` span. The innermost — latest — open one is the one closed.
pub fn think_closer(text: &str) -> Option<String> {
    open_thought(text).map(|(_, close)| close)
}

/// The thought `text` is in, as [`think_closer`] finds it: where its opener
/// begins, and what closes it.
pub fn open_thought(text: &str) -> Option<(usize, String)> {
    let nl = if text.ends_with('\n') { "" } else { "\n" };
    let mut best: Option<(usize, String)> = None;
    let mut offer = |at: usize, closer: String| {
        if best.as_ref().is_none_or(|(b, _)| at > *b) {
            best = Some((at, closer));
        }
    };
    let open_at = |opener: &str, closes: &[&str]| -> Option<usize> {
        let at = text.rfind(opener)?;
        let after = &text[at + opener.len()..];
        (!closes.iter().any(|c| after.contains(c))).then_some(at)
    };
    if let Some(at) = open_at("<think>", &["</think>"]) {
        offer(at, format!("{nl}</think>\n\n"));
    }
    // `<ns|think>`, `<ns|think_fast>`: the namespace before the pipe keeps it
    // apart from Gemma 4's `<|think|>` control token, which opens nothing.
    let mut from = text.len();
    while let Some(bar) = text[..from].rfind("|think") {
        from = bar;
        let Some(lt) = text[..bar].rfind('<') else { break };
        let ns = &text[lt + 1..bar];
        let rest = &text[bar + 1..];
        let Some(gt) = rest.find('>') else { continue };
        let name = &rest[..gt];
        let named = name == "think"
            || name.strip_prefix("think_").is_some_and(|r| !r.is_empty() && r.chars().all(|c| c.is_ascii_lowercase()));
        if ns.is_empty() || !ns.chars().all(|c| c.is_ascii_alphanumeric()) || !named {
            continue;
        }
        let close = format!("</{ns}|{name}>");
        if !rest[gt..].contains(&close) {
            offer(lt, format!("{nl}{close}\n"));
        }
        break;
    }
    for name in ["thought", "thinking"] {
        if let Some(at) = open_at(&format!("<|channel>{name}"), &["<channel|>"]) {
            offer(at, "<channel|>".into());
        }
    }
    if let Some(at) = open_at("<|channel|>analysis", &["<|end|>", "<|channel|>final", "<|return|>"]) {
        offer(at, "<|end|><|start|>assistant<|channel|>final<|message|>".into());
    }
    if let Some(at) = open_at("to=self<|message|>", &["<|eom|>", "<|eot|>", "<|end|>"]) {
        offer(at, "<|eom|><|start|>assistant to=user<|message|>".into());
    }
    best
}

#[cfg(test)]
mod think_closer_tests {
    use super::think_closer;

    #[test]
    fn each_markup_is_closed_its_own_way() {
        assert_eq!(think_closer("<think>\nweighing it").as_deref(), Some("\n</think>\n\n"));
        assert_eq!(think_closer("<think>\nweighing it\n").as_deref(), Some("</think>\n\n"));
        assert_eq!(think_closer("<ifm|think>\nplan").as_deref(), Some("\n</ifm|think>\n"));
        assert_eq!(think_closer("<ifm|think_fast>\nplan").as_deref(), Some("\n</ifm|think_fast>\n"));
        assert_eq!(think_closer("<|channel>thought\nplan").as_deref(), Some("<channel|>"));
        assert_eq!(
            think_closer("<|channel|>analysis<|message|>plan").as_deref(),
            Some("<|end|><|start|>assistant<|channel|>final<|message|>")
        );
        assert_eq!(
            think_closer(" to=self<|message|>plan").as_deref(),
            Some("<|eom|><|start|>assistant to=user<|message|>")
        );
    }

    #[test]
    fn a_reply_out_of_its_thought_has_nothing_to_close() {
        assert_eq!(think_closer(""), None);
        assert_eq!(think_closer("just an answer"), None);
        assert_eq!(think_closer("<think>\nplan\n</think>\n\nanswer"), None);
        assert_eq!(think_closer("<ifm|think>\nplan\n</ifm|think>\nanswer"), None);
        assert_eq!(think_closer("<|channel>thought\nplan<channel|>answer"), None);
        assert_eq!(think_closer("<|turn>model\n<|think|>"), None);
        assert_eq!(
            think_closer("<|channel|>analysis<|message|>plan<|end|><|start|>assistant<|channel|>final<|message|>4"),
            None
        );
        assert_eq!(think_closer(" to=self<|message|>plan<|eom|><|start|>assistant to=user<|message|>4"), None);
    }

    #[test]
    fn the_latest_open_thought_is_the_one_closed() {
        // A second thought after an answer: that one is open.
        assert_eq!(
            think_closer("<|channel>thought\na<channel|>part one<|channel>thought\nmore").as_deref(),
            Some("<channel|>")
        );
        assert_eq!(think_closer("<think>a</think>b<think>\nmore").as_deref(), Some("\n</think>\n\n"));
    }
}

/// A thought going round: what it writes now it has written twice before,
/// word for word or nearly. A small model that falls into it does not climb
/// out — it writes the same paragraphs until the reply's length or the
/// context runs out, minutes of it, and the answer never comes. Caught, the
/// thought is closed the way a Skip closes it (in the reply's own markup,
/// given to the model as its next tokens), so the reply answers from what it
/// had thought; both modes, both engines.
///
/// The test is on the text, so it reads the same however an engine counts its
/// tokens (whitespace runs read as one space). Either of two:
/// - Going round: of the last `WINDOW` characters, nine in ten sit in a
///   `GRAM`-character stretch this thought has written three times. A draft
///   and its clean copy are two; a loop gets there on its third pass however
///   long its turn is, and a number or a word changed here and there does not
///   hide it.
/// - Counting: with every run of digits read as one, the last `COUNTED`
///   characters are one line over and over — "In base 331, 2 is 2." then
///   332, 333… (a 4B wrote three hundred of them and answered nothing). Only
///   the very same line: a check that comes out one way or another ("37 is
///   prime", "38 = 2 × 19") is reasoning, and is left alone.
pub struct ThoughtLoop {
    /// Where the open thought's opener begins in the text.
    start: Option<usize>,
    /// This thought has been called a loop already.
    fired: bool,
    /// Bytes of the text read so far.
    read: usize,
    norm: Vec<char>,
    space: bool,
    /// `norm` with each run of digits as one `#`.
    counted: Vec<char>,
    digits: bool,
    /// Each `GRAM`-character stretch of `norm`, hashed, by where it ends.
    grams: Vec<u64>,
    counts: std::collections::HashMap<u64, u32>,
    /// The text's length at which to look again.
    next: usize,
}

/// The openers of a tool call, in the formats models write them.
const ACTS: [&str; 4] = ["<tool_call>", "<|tool_call>", "<|tool_call_start|>", "<function="];

impl Default for ThoughtLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl ThoughtLoop {
    const GRAM: usize = 64;
    const WINDOW: usize = 1200;
    const SEEN: u32 = 3;
    const SHARE: f64 = 0.9;
    const COUNTED: usize = 2400;
    /// The longest line a count is looked for in.
    const LINE: usize = 400;

    pub fn new() -> Self {
        Self {
            start: None,
            fired: false,
            read: 0,
            norm: Vec::new(),
            space: false,
            counted: Vec::new(),
            digits: false,
            grams: Vec::new(),
            counts: std::collections::HashMap::new(),
            next: 0,
        }
    }

    /// Is the reply `head` + `out` (the opener the prompt left open, then
    /// what was generated) in a thought that has started going round? Yes
    /// once per thought. Cheap between looks: it reads the text again only
    /// every sixty-four bytes or so (further apart as the reply grows).
    pub fn due(&mut self, head: &str, out: &str) -> bool {
        let len = head.len() + out.len();
        if len < self.next {
            return false;
        }
        let next = len + (len / 128).clamp(64, 1024);
        self.next = next;
        let text = [head, out].concat();
        let Some((at, _)) = open_thought(&text) else {
            self.start = None;
            return false;
        };
        if self.start != Some(at) {
            *self = Self { start: Some(at), read: at, next, ..Self::new() };
        }
        // A call begun inside the thought (Gemma 4 writes one in its thought
        // channel) is the model acting: what repeats there is the call's
        // content, and a close given in the middle of it would break it.
        if self.fired || ACTS.iter().any(|m| text[at..].contains(m)) {
            return false;
        }
        for c in text[self.read..].chars() {
            let c = if c.is_whitespace() {
                if self.space {
                    continue;
                }
                self.space = true;
                ' '
            } else {
                self.space = false;
                c
            };
            self.norm.push(c);
            if !c.is_ascii_digit() {
                self.counted.push(c);
                self.digits = false;
            } else if !self.digits {
                self.counted.push('#');
                self.digits = true;
            }
            if self.norm.len() >= Self::GRAM {
                let h = self.norm[self.norm.len() - Self::GRAM..]
                    .iter()
                    .fold(0xcbf2_9ce4_8422_2325u64, |h, &c| (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3));
                self.grams.push(h);
                *self.counts.entry(h).or_default() += 1;
            }
        }
        self.read = text.len();
        if self.counting() {
            self.fired = true;
            return true;
        }
        if self.norm.len() < Self::WINDOW {
            return false;
        }
        // A character counts as said before when a stretch around it was
        // written three times: a number changed costs that number, not the
        // sixty-four stretches it sits in. Records written one after another
        // (CSS rules, JSON rows) share runs shorter than a stretch — a field
        // changes every few dozen characters — so a long table is not a loop.
        let n = self.norm.len();
        let from = n - Self::WINDOW;
        let seen = |g: usize| self.counts.get(&self.grams[g]).copied().unwrap_or(0) >= Self::SEEN;
        // `grams[g]` covers characters g .. g + GRAM.
        let mut reach = (from.saturating_sub(Self::GRAM - 1)..from).filter(|&g| seen(g)).map(|g| g + Self::GRAM).max().unwrap_or(0);
        let mut again = 0usize;
        for c in from..n {
            if c < self.grams.len() && seen(c) {
                reach = c + Self::GRAM;
            }
            again += usize::from(c < reach);
        }
        if again as f64 >= Self::WINDOW as f64 * Self::SHARE {
            self.fired = true;
            return true;
        }
        false
    }

    /// The last `COUNTED` characters, digits aside, one line repeated: some
    /// period of at most `LINE` characters that the whole stretch keeps to.
    fn counting(&self) -> bool {
        const KEY: usize = 32;
        let m = &self.counted;
        if m.len() < Self::COUNTED {
            return false;
        }
        let tail = &m[m.len() - Self::COUNTED..];
        let w = tail.len();
        let key = &tail[w - KEY..];
        (1..=Self::LINE).any(|p| tail[w - KEY - p..w - p] == *key && (p..w).all(|i| tail[i] == tail[i - p]))
    }
}

#[cfg(test)]
mod thought_loop_tests {
    use super::ThoughtLoop;

    /// Words that read as prose and never repeat a stretch of thirty-two
    /// characters: a small generator over a fixed vocabulary.
    fn prose(seed: u64, words: usize) -> String {
        const VOCAB: &[&str] = &[
            "the", "file", "returns", "a", "value", "when", "we", "check", "whether", "index", "is", "past", "end",
            "so", "maybe", "loop", "should", "stop", "earlier", "but", "then", "test", "expects", "three", "items",
            "and", "parser", "reads", "header", "first", "which", "means", "offset", "shifts", "by", "one", "line",
        ];
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut out = String::new();
        for i in 0..words {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            out.push_str(VOCAB[(x >> 33) as usize % VOCAB.len()]);
            out.push(if i % 13 == 12 { '.' } else { ' ' });
            if i % 40 == 39 {
                out.push_str("\n\n");
            }
        }
        out
    }

    /// Where in `thought` (fed a few characters at a time, as tokens come)
    /// the watch calls it a loop.
    fn caught(head: &str, thought: &str) -> Option<usize> {
        let mut w = ThoughtLoop::new();
        let chars: Vec<char> = thought.chars().collect();
        let mut out = String::new();
        for piece in chars.chunks(4) {
            out.extend(piece);
            if w.due(head, &out) {
                return Some(out.chars().count());
            }
        }
        None
    }

    #[test]
    fn a_paragraph_written_over_and_over_is_caught_on_its_third_pass() {
        let intro = prose(1, 200);
        let para = "Wait, let me re-check the boundary. If the index equals the length, the loop has already gone one past the end, so the guard must use < and not <=. That means the fix is in the condition, not in the body. Let me verify this again.\n\n";
        let p = para.chars().count();
        let at = caught("<think>\n", &format!("{intro}{}", para.repeat(20))).expect("a loop");
        let into = at - intro.chars().count();
        assert!(into > 2 * p, "caught before the third pass: {into} of a {p}-character turn");
        assert!(into < 2 * p + 1300, "caught late: {into} of a {p}-character turn");
    }

    #[test]
    fn a_long_turn_is_caught_as_surely_as_a_short_one() {
        let turn = format!("{}\n\n", prose(7, 450));
        let p = turn.chars().count();
        assert!(p > 2000);
        let at = caught("", &format!("<think>\n{}{}", prose(2, 60), turn.repeat(5))).expect("a loop");
        assert!(at > 2 * p && at < 3 * p + 600, "{at} for a {p}-character turn");
        // A line said again and again.
        assert!(caught("<think>\n", &format!("{}{}", prose(3, 80), "I'll write the file now.\n".repeat(200))).is_some());
        // Chinese goes round the same way.
        let zh = "等一下，我再检查一遍边界条件：如果下标等于长度，循环已经越过末尾一位，所以判断应该用小于而不是小于等于。也就是说问题出在条件上，不在循环体里。我再确认一次。\n";
        assert!(caught("<think>\n", &format!("{}{}", prose(4, 80), zh.repeat(30))).is_some());
    }

    #[test]
    fn a_number_changed_each_time_round_does_not_hide_it() {
        let mut thought = format!("<think>\n{}", prose(5, 120));
        for i in 0..20 {
            thought.push_str(&format!(
                "Attempt {i}: the test still fails because the parser reads the header first, which shifts every offset by one line. So the fix must go in the reader, before the offsets are computed. Let me look at the reader once more and try it.\n\n"
            ));
        }
        assert!(caught("", &thought).is_some());
    }

    #[test]
    fn a_thought_counting_up_the_same_line_is_caught() {
        // As a 4B wrote it, three hundred lines on.
        let lines: String = (3..400).map(|b| format!("    *   In base {b}, 2 is 2.\n")).collect();
        let at = caught("<think>\n", &format!("{}{lines}", prose(21, 150))).expect("a count");
        // 2400 characters with the numbers read as one: about 115 lines.
        assert!(at < prose(21, 150).chars().count() + 4000, "caught late: {at}");
        // Plain counting, and a sum carried on.
        let count: String = (1..2000).map(|i| format!("{i}, ")).collect();
        assert!(caught("<think>\n", &count).is_some());
        let sums: String = (1..400).map(|i| format!("{} + 7 = {}. ", i * 7, i * 7 + 7)).collect();
        assert!(caught("<think>\n", &sums).is_some());
    }

    #[test]
    fn a_check_run_down_a_list_is_reasoning() {
        // Each number checked comes out one way or the other.
        let primes: String = (2..400u32)
            .map(|n| match (2..n).find(|d| d * d <= n && n % d == 0) {
                Some(d) => format!("Is {n} prime? {n} = {d} × {}, so it is not prime.\n", n / d),
                None => format!("Is {n} prime? No divisor up to its square root, so {n} is prime.\n"),
            })
            .collect();
        assert_eq!(caught("<think>\n", &primes), None);
        // Trying sums, each one different.
        let ops: Vec<usize> = prose(22, 300).bytes().map(|b| usize::from(b) % 4).collect();
        let tries: String = (0..300).map(|i| format!("Try {} {} {} = {}? ", i % 7, ["+", "-", "×", "÷"][ops[i]], i % 5, i % 11)).collect();
        assert_eq!(caught("<think>\n", &tries), None);
    }

    #[test]
    fn thinking_that_moves_on_is_left_alone() {
        // Long reasoning that never says the same thing.
        assert_eq!(caught("<think>\n", &prose(11, 6000)), None);
        // A draft of a function and its clean copy: written twice, not three times.
        let code: String = (0..60).map(|i| format!("    if value == {i} {{\n        return table[{}];\n    }}\n", i * 7)).collect();
        let thought = format!("{}\n```rust\n{code}```\nThat works. The final version:\n```rust\n{code}```\n{}", prose(12, 100), prose(13, 300));
        assert_eq!(caught("<think>\n", &thought), None);
        // Code made of nearly the same lines, indentation and all — a few
        // dozen of them. (Pages of one line with its numbers changed are a
        // count: see above.)
        let css: String = (0..50).map(|i| format!(".item-{i} {{\n    margin: {}px;\n    padding: 0;\n}}\n", i % 9)).collect();
        assert_eq!(caught("<think>\n", &format!("{}{css}", prose(14, 50))), None);
        let rows: String = (0..30)
            .map(|i| format!("  {{\"id\": {i}, \"name\": \"user{}\", \"active\": true, \"role\": \"member\"}},\n", i * 37 % 101))
            .collect();
        assert_eq!(caught("<think>\n", &format!("{}[\n{rows}]", prose(15, 50))), None);
        let cards: String = (0..40)
            .map(|i| format!("<div class=\"card\">\n  <h3>{}</h3>\n  <p>{}</p>\n</div>\n", prose(100 + i, 3), prose(200 + i, 9)))
            .collect();
        assert_eq!(caught("<think>\n", &format!("{}{cards}", prose(16, 50))), None);
    }

    #[test]
    fn only_a_thought_is_watched() {
        let para = "The answer is the same as before and I will say it once more for you. ".repeat(60);
        // An answer that goes round is not this watch's business.
        assert_eq!(caught("", &format!("<think>\nshort</think>\n\n{para}")), None);
        assert_eq!(caught("", &para), None);
        // A thought opened in the prompt is watched from there.
        assert!(caught("<|turn>model\n<|channel>thought\n", &para).is_some());
        // A call written inside the thought is the model acting, whatever
        // it repeats.
        let edits = "{old_string:<|\"|>    if qty >= 10:<|\"|>,new_string:<|\"|>    if qty >= 12:<|\"|>,replace_all:false},".repeat(40);
        assert_eq!(caught("<|channel>thought\n", &format!("{}<|tool_call>call:edit_file{{edits:[{edits}", prose(17, 40))), None);
    }

    /// Real replies, one JSON object per line (`{"src", "t"}`), fed as they
    /// streamed: which the watch would have closed, and where.
    ///   CHATY_THOUGHTS=/path/raws.jsonl cargo test --lib real_thoughts -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_thoughts() {
        let path = std::env::var("CHATY_THOUGHTS").expect("CHATY_THOUGHTS");
        let (mut n, mut hit) = (0, 0);
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let t = v["t"].as_str().unwrap_or("");
            n += 1;
            if let Some(at) = caught("", t) {
                hit += 1;
                let cs: Vec<char> = t.chars().collect();
                let tail: String = cs[at.saturating_sub(600)..at].iter().collect();
                println!("== {} at {at}/{}\n{tail}\n", v["src"], cs.len());
            }
        }
        println!("{hit} of {n} caught");
    }

    #[test]
    fn each_thought_is_called_a_loop_once() {
        let para = "Let me re-check: two plus two is four, so the total is four, which matches. ".repeat(60);
        let mut w = ThoughtLoop::new();
        let mut out = String::from("<think>\n");
        let mut fired = 0;
        for c in para.chars() {
            out.push(c);
            fired += w.due("", &out) as u32;
        }
        assert_eq!(fired, 1);
        // Closed, then a second thought that goes round as well.
        out.push_str("\n</think>\n\nFour.<think>\n");
        for c in para.chars() {
            out.push(c);
            fired += w.due("", &out) as u32;
        }
        assert_eq!(fired, 2);
    }
}

/// The engine processes a load started — every sidecar not in `before`,
/// taken as the load began — ended now. Names checked as in
/// `kill_sidecars_now`; sysinfo's kill ends a process on every platform.
pub fn kill_sidecars_except(before: &[u32]) {
    let pids: Vec<u32> = SIDECAR_PIDS.lock().unwrap().iter().copied().filter(|p| !before.contains(p)).collect();
    if pids.is_empty() {
        return;
    }
    let mut sys = sysinfo::System::new();
    let want: Vec<sysinfo::Pid> = pids.iter().map(|p| sysinfo::Pid::from_u32(*p)).collect();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&want), true);
    for pid in pids {
        if let Some(p) = sys.process(sysinfo::Pid::from_u32(pid)) {
            let name = p.name().to_string_lossy();
            if name.contains("chaty-mlx") || name.contains("chaty-sd") || name.contains("chaty-audio") || name.contains("chaty-llama") {
                p.kill();
            }
        }
    }
}

pub fn kill_sidecars_now() {
    let pids: Vec<u32> = std::mem::take(&mut *SIDECAR_PIDS.lock().unwrap());
    if pids.is_empty() {
        return;
    }
    // A pid only means something while the process behind it is still the one
    // that was recorded. A sidecar that died on its own (or was reaped on a
    // path that failed to deregister it) leaves a number the system is free
    // to hand to anybody, so the name is checked before the signal.
    let mut sys = sysinfo::System::new();
    let want: Vec<sysinfo::Pid> = pids.iter().map(|p| sysinfo::Pid::from_u32(*p)).collect();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&want), true);
    for pid in pids {
        let ours = sys
            .process(sysinfo::Pid::from_u32(pid))
            .map(|p| {
                let name = p.name().to_string_lossy();
                name.contains("chaty-mlx") || name.contains("chaty-sd") || name.contains("chaty-audio") || name.contains("chaty-llama")
            })
            .unwrap_or(false);
        if !ours {
            continue;
        }
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
        #[cfg(not(unix))]
        let _ = pid;
    }
}

/// A single chat turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool's result, delivered under its own role rather than folded into a
    /// user turn. Chat templates key "is this turn still part of the current
    /// request" off the last *user* message, so a tool result posing as one
    /// makes the template discard the assistant reasoning that preceded it —
    /// costing the model the thread of its own work and voiding the KV prefix
    /// every step. Used only where the model's template actually renders it
    /// (probed at load, never assumed).
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// The turn's thinking, kept out of `content`. Some templates read
    /// reasoning only from a structured field and never split it back out of
    /// the content — a turn stored inline reaches them as an empty thought
    /// followed by its own markup. Sent only where the template is probed to
    /// use it; `None` everywhere else keeps the wire shape unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Image attachments (absolute file paths) for vision models. Ignored —
    /// and expected empty — when the loaded model has no mmproj.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

/// Sampling / decoding parameters. Sensible defaults so the UI can omit them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub seed: Option<u64>,
    /// Top-k cutoff (0 = disabled).
    pub top_k: u32,
    /// Min-p cutoff (0 = disabled).
    pub min_p: f32,
    /// Repetition penalty (1.0 = off).
    pub repeat_penalty: f32,
    /// Stop sequences — generation halts (and the trailing match is trimmed) when
    /// any of these appears in the output.
    pub stop: Vec<String>,
    /// Reasoning control for models without the `/no_think` soft switch
    /// (Qwen3.5+): `Some(false)` force-disables thinking by pre-filling an empty
    /// `<think></think>` block; `Some(true)`/`None` leave the model default.
    pub think: Option<bool>,
    /// Native reasoning-effort level for models whose chat template takes a
    /// `reasoning_effort` kwarg (Qwen3.8: `low` | `medium` | `xhigh`). `None`
    /// leaves the model's own default; ignored by models without the ladder.
    pub effort: Option<String>,
    /// A side generation — a chat title, a search-query rewrite — that must
    /// leave the conversation's cache alone. Run on the conversation's own
    /// cache it replaced everything the conversation had built, so the turn
    /// after it re-read the whole of it: 0% reused on every turn of a chat with
    /// web search on (measured on Qwen3.6 35B).
    pub scratch: bool,
    /// The think ceiling (Settings → Chat for a reply, Settings → Code for
    /// each step of a turn), in tokens of the reply: a thought still open
    /// when the reply reaches it is closed there, as a Skip closes it — what
    /// was thought kept, the answer or the step's action begun. 0 = none.
    pub think_budget: u32,
    /// Chat's think floor (experimental): until the reply has run this many
    /// tokens, a thought it is in may not close, nor the reply end. 0 = none.
    pub think_floor: u32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 512,
            seed: None,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.1,
            stop: Vec::new(),
            think: None,
            effort: None,
            scratch: false,
            think_budget: 0,
            think_floor: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub params: GenParams,
}

/// The conversation and message a streaming reply belongs to.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveTarget {
    pub conversation_id: String,
    pub message_id: String,
    /// The model writing the reply, recorded with it.
    #[serde(default)]
    pub model: Option<String>,
}

/// Streaming protocol pushed to the frontend over a Tauri `Channel`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum StreamEvent {
    /// Generation accepted; prompt is being processed.
    Started,
    /// Prompt-processing progress: `processed` of `total` prompt tokens are in
    /// the KV cache. Only emitted for prefills long enough to be worth a
    /// progress ring (more than one decode batch of new tokens).
    Prefill { processed: u32, total: u32 },
    /// One decoded piece of text (not necessarily a whole token).
    Token { text: String },
    /// Generation finished cleanly.
    Done { stats: GenStats },
    /// Generation aborted with an error.
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub tokens_per_second: f32,
    /// Why generation ended: "eos" (model finished), "length" (max_tokens hit),
    /// "context" (context window full), "stop" (stop sequence), "cancelled".
    pub stop_reason: String,
    /// Prompt tokens resumed from the previous turn's KV cache (0 = evaluated
    /// from scratch). Cross-conversation contamination shows up here first.
    pub reused: u32,
}

/// Metadata about the currently loaded model, surfaced to the UI. The
/// llama.cpp sidecar sends it back over its pipe, so it reads as leniently as
/// it is written: a field one side lacks takes its default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub backend: String,
    pub loaded: bool,
    /// Model architecture from GGUF metadata (e.g. "llama", "qwen2").
    pub arch: Option<String>,
    /// On-disk size of the tensors, in MB.
    pub size_mb: Option<u64>,
    /// Parameter count in billions.
    pub params_b: Option<f64>,
    /// Context length the model was trained with.
    pub n_ctx_train: Option<u32>,
    /// Context length we actually loaded it with.
    pub n_ctx: Option<u32>,
    /// Total transformer layers in the model.
    pub n_layer: Option<u32>,
    /// Layers actually offloaded to the GPU (0 = pure CPU).
    pub gpu_layers: i32,
    /// Name of the GPU used for offload, if any.
    pub gpu_name: Option<String>,
    /// Pretty model name from `general.name`, if present.
    pub model_name: Option<String>,
    /// Quantization (e.g. "Q5_K_M"), derived from `general.file_type`.
    pub quant: Option<String>,
    /// Embedding dimension.
    pub n_embd: Option<u32>,
    /// Whether the GGUF ships a chat template.
    pub has_chat_template: bool,
    /// Best-effort: the model appears to support `<think>` reasoning.
    pub supports_thinking: bool,
    /// The chat template honours the `/no_think` soft switch (Qwen3, not 3.5+).
    pub think_switch: bool,
    /// Native reasoning-effort ladder the chat template accepts, weakest
    /// first (Qwen3.8: `["low", "medium", "xhigh"]`). Empty ⇒ the model has
    /// no effort control and the UI keeps its plain thinking toggle.
    #[serde(default)]
    pub effort_levels: Vec<String>,
    /// The chat template renders a tool result under its own role AND keeps
    /// the assistant reasoning that preceded it — which a result posing as a
    /// user turn discards. Probed at load; false ⇒ results stay user turns and
    /// the rendered prompt is byte-identical to what earlier builds produced.
    #[serde(default)]
    pub tool_role: bool,
    /// The template reads a turn's thinking from a structured
    /// `reasoning_content` field rather than splitting it out of the content.
    #[serde(default)]
    pub reasoning_field: bool,
    /// The tool-call format the chat template teaches its model, read from the
    /// template at load (see `native_tool_format`): "xml", "json", "gemma" or
    /// "lfm". None ⇒ the template names none; the agent uses the fallback the
    /// user picked.
    #[serde(default)]
    pub tool_format: Option<String>,
    /// Best-effort: the chat template supports tool / function calling.
    pub supports_tools: bool,
    /// Best-effort: the model appears to be multimodal (vision).
    pub multimodal: bool,
    /// The vision encoder (mmproj) is loaded — the model can actually see
    /// images this session. `multimodal && !vision_ready` means "the model
    /// could do vision, but its mmproj GGUF is missing next to the weights".
    pub vision_ready: bool,
    /// Whether ONE prompt may carry several pictures. Gemma-4 through MLX
    /// cannot: it encodes the first and then rejects the token-count mismatch,
    /// failing the whole round — which is what a tall page's tiled screenshot
    /// hands it. Everything else here says true.
    pub multi_image: bool,
    /// Path of the paired mmproj GGUF, when one was found.
    pub mmproj: Option<String>,
    /// The model file carries a multi-token-prediction head — the extra block
    /// speculative decoding guesses with. A capability of the FILE, so it stays
    /// true when the user has the feature switched off; the settings toggle is
    /// enabled on this and greyed out on everything else.
    #[serde(default)]
    pub speculative: bool,
    /// Speculative decoding is actually running for this load: the model has a
    /// head, the setting allows it, and the head loaded. Never true when
    /// `speculative` is false.
    #[serde(default)]
    pub speculative_on: bool,
    /// Non-fatal load warning code for the UI (e.g. "gpu-oom" when the GPU
    /// offload had to be reduced to fit memory). `None` on a clean load.
    pub warning: Option<String>,
    /// "chat" for a language model, "image" for a diffusion model, "music"
    /// for a text-to-music model. The whole interface follows this: an image
    /// model turns the app into an image studio and a music model into a music
    /// studio, each with only the settings and tools that apply to it.
    pub kind: String,
    /// What the image engine loaded, when `kind` is "image".
    #[serde(default, skip_serializing_if = "Option::is_none", skip_deserializing)]
    pub image: Option<crate::imagegen::ImageModelInfo>,
    /// What the music engine loaded, when `kind` is "music".
    #[serde(default, skip_serializing_if = "Option::is_none", skip_deserializing)]
    pub music: Option<crate::musicgen::MusicModelInfo>,
}


/// Which tool-call format a chat template teaches its model. The template is
/// how the model was trained to write a call; a guess from its name is not.
/// Most specific first — a Qwen3.5 template mentions `<tool_call>` too.
///  - "gemma": Gemma 4, `<|tool_call>call:name{key:<|"|>text<|"|>}<tool_call|>`
///  - "lfm":   LFM2, `<|tool_call_start|>[name(key='v')]<|tool_call_end|>`
///  - "xml":   Qwen3.5/3.6/3.8, `<tool_call>\n<function=name>\n<parameter=key>\n…`
///  - "json":  Hermes, `<tool_call>{"name": …, "arguments": {…}}</tool_call>`
///    (Qwen2.5/Qwen3, QwQ)
pub fn native_tool_format(template: &str) -> Option<&'static str> {
    // K2 Horizon: `<ifm|tool_call>name` + `<ifm|arg_key>`/`<ifm|arg_value>`
    // pairs, all special tokens of its own.
    if template.contains("<ifm|tool_call>") {
        Some("ifm")
    } else if template.contains("<arg_key>") {
        // GLM-4.5/4.6/4.7: the same pairs without a namespace, after a bare
        // tool name in `<tool_call>` — which JSON would otherwise claim.
        Some("glm")
    } else if template.contains("<param name=") {
        // MiniCPM5: `<function name="…"><param name="…">…</param></function>`.
        Some("minicpm")
    } else if template.contains("<|tool_call>") {
        Some("gemma")
    } else if template.contains("<|tool_call_start|>") {
        Some("lfm")
    } else if template.contains("<function=") && template.contains("<parameter=") {
        Some("xml")
    } else if template.contains("<tool_call>") {
        Some("json")
    } else {
        None
    }
}

#[cfg(test)]
mod tool_format_tests {
    use super::native_tool_format;

    /// Fragments of the real templates each family ships.
    #[test]
    fn each_family_template_names_its_own_format() {
        let qwen35 = "{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{{- '<parameter=' + args_name + '>\\n' }}";
        let qwen3 = "{{- '<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}";
        let gemma4 = "{{- '<|tool_call>call:' + function['name'] + '{' -}}";
        let lfm = "<|tool_call_start|>[{{ tool_call.name }}(...)]<|tool_call_end|>";
        assert_eq!(native_tool_format(qwen35), Some("xml"));
        assert_eq!(native_tool_format(qwen3), Some("json"));
        assert_eq!(native_tool_format(gemma4), Some("gemma"));
        assert_eq!(native_tool_format(lfm), Some("lfm"));
        let k2 = "{{- \"\\n<ifm|tool_call>\" + tool_call.name + \"\\n\" }}{{- \"<ifm|arg_key>\" + key + \"</ifm|arg_key>\\n\" }}";
        assert_eq!(native_tool_format(k2), Some("ifm"));
        // GLM-4.7-Flash, verbatim.
        let glm = "{{- '<tool_call>' + tc.name -}}\n{% set _args = tc.arguments %}{% for k, v in _args.items() %}<arg_key>{{ k }}</arg_key><arg_value>{{ v | tojson(ensure_ascii=False) if v is not string else v }}</arg_value>{% endfor %}</tool_call>";
        assert_eq!(native_tool_format(glm), Some("glm"));
        let minicpm5 = "{{- '<function name=\"' ~ tool_call.name ~ '\">' }}{{- '<param name=\"' ~ param_name ~ '\">' }}";
        assert_eq!(native_tool_format(minicpm5), Some("minicpm"));
        assert_eq!(native_tool_format("{% for m in messages %}{{ m.content }}{% endfor %}"), None);
    }

    /// An MLX folder's template: chat_template.jinja, or the tokenizer config's
    /// chat_template — a string, or a list of named ones (read together).
    #[test]
    fn an_mlx_folder_template_is_found_in_either_place() {
        use super::mlx::mlx_chat_template;
        let dir = std::env::temp_dir().join(format!("chaty-mlx-template-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(mlx_chat_template(&dir), None);
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":[{"name":"default","template":"plain"},{"name":"tool_use","template":"<tool_call>{{ x | tojson }}</tool_call>"}]}"#,
        )
        .unwrap();
        assert_eq!(mlx_chat_template(&dir).as_deref().and_then(native_tool_format), Some("json"));
        std::fs::write(dir.join("chat_template.jinja"), "<|tool_call>call:{{ name }}").unwrap();
        assert_eq!(mlx_chat_template(&dir).as_deref().and_then(native_tool_format), Some("gemma"));
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Synchronously release the model's memory (block until freed). Called
    /// before loading a replacement so two models never coexist in RAM.
    fn unload(&self) {}
    /// Short identifier for telemetry / UI ("mock", "llama.cpp", …).
    fn name(&self) -> &str;

    /// The image engine behind this backend, when it is one. Image generation
    /// reaches the loaded engine through this; every chat backend says `None`.
    fn as_image(&self) -> Option<&sd::SdEngine> {
        None
    }

    /// The music engine behind this backend, when it is one.
    fn as_music(&self) -> Option<&audio::AudioEngine> {
        None
    }

    /// Stream a completion for `req`, emitting [`StreamEvent`]s on `sink`.
    /// Implementations should send `Started` first and exactly one terminal
    /// `Done` or `Error` last, and stop early when `cancel` becomes `true`.
    async fn generate(
        &self,
        req: GenRequest,
        sink: Channel<StreamEvent>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()>;

    /// One-shot, non-streaming generation returning the full text. Powers
    /// vision analysis (Code mode / KB / Canvas). Default: unsupported.
    async fn generate_collect(
        &self,
        _req: GenRequest,
        _cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<String> {
        anyhow::bail!("this backend does not support one-shot generation")
    }
}
