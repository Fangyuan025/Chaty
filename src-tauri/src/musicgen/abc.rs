//! YuE2's scores, made instrumental.
//!
//! YuE2 has no instrumental switch: empty lyrics still get sung. Its makers'
//! way (the `yue2-music` skill's instrumental workflow) is to let the model
//! write its score, move every note of the Vocal part into the Ins part, and
//! render that score with section tags for lyrics. This is that conversion,
//! ported from the skill's `abc_tools.py`, `compile_score.py` and
//! `instrumentalize.py` (MIT License, Copyright (c) 2026 yue2-music
//! contributors, github.com/multimodal-art-projection/YuE).
//!
//! It reads only the native two-voice dialect YuE2 writes and fails on
//! anything else rather than guessing at timing. Times are ticks: 1024 to a
//! quarter note, fine enough for every length and meter the dialect allows.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use regex::Regex;

/// Ticks to a quarter note.
const Q: i64 = 1024;
/// Ticks to a whole note.
const WHOLE: i64 = 4 * Q;
const DURATIONS: [i64; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];
const QUALITIES: [&str; 15] = ["", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5", "sus4", "sus2", "6", "m6", "7sus4", "m(maj7)"];
const LETTERS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
const NATURAL: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
const SECTIONS: [&str; 8] = ["intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro", "instrumental"];
const VOCAL_DEF: &str = r#"V: Vocal clef=treble name="Vocal Melody" snm="Vocal""#;
const INS_DEF: &str = r#"V: Ins clef=treble name="Ins Melody" snm="Inst.""#;

type Res<T> = Result<T, String>;

fn fail<T>(msg: impl Into<String>) -> Res<T> {
    Err(msg.into())
}

fn letter_index(c: char) -> usize {
    LETTERS.iter().position(|&l| l == c).expect("a note letter")
}

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"^(?:"(?P<chord>[^"\n]*)"|\[K:(?P<key>[^\]\n]+)\]|(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-gz])(?P<oct>[,']*)(?P<dur>[0-9]*)(?P<tie>-?))"#)
            .expect("token regex")
    })
}

fn chord_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let pitch = "[A-G](?:bb|##|b|#)?";
        let qualities: Vec<String> = QUALITIES.iter().map(|q| regex::escape(q)).collect();
        Regex::new(&format!("^{pitch}(?:{})(?:/{pitch})?$", qualities.join("|"))).expect("chord regex")
    })
}

/// Sharps (+) or flats (−) in a key: the standard major and minor keys only.
fn key_fifths(key: &str) -> Res<i32> {
    const MAJOR: [&str; 15] = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"];
    const MINOR: [&str; 15] = ["Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"];
    MAJOR
        .iter()
        .position(|k| *k == key)
        .or_else(|| MINOR.iter().position(|k| *k == key))
        .map(|i| i as i32 - 7)
        .ok_or_else(|| format!("Unsupported key {key:?}; use a standard major or minor K: field"))
}

/// Each letter's alteration in a key, C to B.
fn key_accidentals(key: &str) -> Res<[i32; 7]> {
    let count = key_fifths(key)?;
    let mut out = [0; 7];
    let order = if count > 0 { "FCGDAEB" } else { "BEADGCF" };
    for c in order.chars().take(count.unsigned_abs() as usize) {
        out[letter_index(c)] = count.signum();
    }
    Ok(out)
}

fn meter_value(text: &str) -> Res<(i64, i64)> {
    let parts: Vec<&str> = text.split('/').collect();
    let parse = |s: &str| s.parse::<i64>().ok().filter(|v| *v > 0 && !s.starts_with('0') && s.bytes().all(|b| b.is_ascii_digit()));
    match parts.as_slice() {
        [n, d] => match (parse(n), parse(d)) {
            (Some(n), Some(d)) if d <= 1024 && d & (d - 1) == 0 => Ok((n, d)),
            (Some(_), Some(d)) => fail(format!("Unsupported meter denominator {d}")),
            _ => fail(format!("Unsupported meter {text:?}; write an explicit fraction")),
        },
        _ => fail(format!("Unsupported meter {text:?}; write an explicit fraction")),
    }
}

fn bar_ticks((n, d): (i64, i64)) -> i64 {
    WHOLE * n / d
}

#[derive(Debug, Clone, PartialEq)]
struct Voice {
    meter: (i64, i64),
    key: String,
    time: i64,
    /// (onset, MIDI pitch, duration), ties merged.
    notes: Vec<(i64, i32, i64)>,
    /// (start, length, meter)
    bars: Vec<(i64, i64, (i64, i64))>,
    chords: Vec<(i64, String)>,
    keys: Vec<(i64, String)>,
    /// A tie into the next note: (pitch, written pitch).
    pending: Option<(i32, i32)>,
}

#[derive(Debug)]
struct Score {
    bpm: i64,
    vocal: Voice,
    ins: Voice,
    /// Line index → is it the Vocal part's.
    music_lines: BTreeMap<usize, bool>,
}

fn parse_bar(body: &str, voice: &mut Voice, unit: i64, context: &str) -> Res<()> {
    let length = bar_ticks(voice.meter);
    let start = voice.time;
    let mut offset = 0;
    // Native exporters carry an accidental by letter, across octaves.
    let mut local: HashMap<char, i32> = HashMap::new();
    if body == "Z" {
        if voice.pending.is_some() {
            return fail(format!("{context}: tie enters a full-measure rest"));
        }
        offset = length;
    } else {
        let mut cursor = 0;
        while cursor < body.len() {
            let rest = &body[cursor..];
            if rest.starts_with(char::is_whitespace) {
                cursor += rest.chars().next().map_or(1, char::len_utf8);
                continue;
            }
            let Some(m) = token_re().captures(rest) else {
                let shown: String = rest.chars().take(24).collect();
                return fail(format!("{context}: unsupported token at {shown:?}"));
            };
            cursor += m.get(0).map_or(0, |g| g.end());
            if offset >= length {
                return fail(format!("{context}: event after the measure end"));
            }
            if let Some(chord) = m.name("chord") {
                if !chord_re().is_match(chord.as_str()) {
                    return fail(format!("{context}: unsupported chord {:?}", chord.as_str()));
                }
                voice.chords.push((start + offset, chord.as_str().to_string()));
                continue;
            }
            if let Some(key) = m.name("key") {
                key_accidentals(key.as_str())?;
                voice.key = key.as_str().to_string();
                voice.keys.push((start + offset, voice.key.clone()));
                local.clear();
                continue;
            }
            let note = m.name("note").map(|g| g.as_str()).unwrap_or_default();
            let acc = m.name("acc").map(|g| g.as_str()).unwrap_or_default();
            let octave = m.name("oct").map(|g| g.as_str()).unwrap_or_default();
            let tie = m.name("tie").is_some_and(|g| !g.as_str().is_empty());
            let units = match m.name("dur").map(|g| g.as_str()).unwrap_or_default() {
                "" => 1,
                d => d.parse::<i64>().unwrap_or(0),
            };
            if !DURATIONS.contains(&units) {
                return fail(format!("{context}: unsupported duration {units}; split it into tied supported lengths"));
            }
            let duration = units * unit;
            if offset + duration > length {
                return fail(format!("{context}: note/rest exceeds meter duration"));
            }
            if octave.contains(',') && octave.contains('\'') {
                return fail(format!("{context}: mixed octave marks"));
            }
            if note == "z" {
                if !acc.is_empty() || !octave.is_empty() || tie {
                    return fail(format!("{context}: a rest cannot have accidentals, octave marks or ties"));
                }
                if voice.pending.is_some() {
                    return fail(format!("{context}: tie enters a rest"));
                }
            } else {
                let c = note.chars().next().expect("one letter");
                let letter = c.to_ascii_uppercase();
                let li = letter_index(letter);
                let mut written = 60 + NATURAL[li] + if c.is_ascii_lowercase() { 12 } else { 0 };
                written += 12 * (octave.matches('\'').count() as i32 - octave.matches(',').count() as i32);
                let mut alteration = match local.get(&letter) {
                    Some(a) => *a,
                    None => key_accidentals(&voice.key)?[li],
                };
                if !acc.is_empty() {
                    alteration = match acc {
                        "=" => 0,
                        "_" => -1,
                        "__" => -2,
                        "^" => 1,
                        _ => 2,
                    };
                    local.insert(letter, alteration);
                }
                let mut pitch = written + alteration;
                if let Some((old_pitch, old_written)) = voice.pending {
                    // An unmarked continuation keeps its tied accidental across
                    // a barline; later untied notes in the bar are not altered.
                    if acc.is_empty() && written == old_written {
                        pitch = old_pitch;
                    }
                    if pitch != old_pitch {
                        return fail(format!("{context}: tie changes pitch from {old_pitch} to {pitch}"));
                    }
                    if let Some(last) = voice.notes.last_mut() {
                        last.2 += duration;
                    }
                } else {
                    if !(0..=127).contains(&pitch) {
                        return fail(format!("{context}: pitch {pitch} is outside MIDI range"));
                    }
                    voice.notes.push((start + offset, pitch, duration));
                }
                voice.pending = if tie { Some((pitch, written)) } else { None };
            }
            offset += duration;
        }
    }
    if offset != length {
        return fail(format!("{context}: duration {offset} ticks != meter duration {length}"));
    }
    voice.bars.push((start, length, voice.meter));
    voice.time += length;
    Ok(())
}

/// The bars of one music line, its full-bar rests (`Z2`…) counted out.
fn line_bars(line: &str) -> Vec<String> {
    let body = &line[..line.len() - 1];
    let mut out = Vec::new();
    for bar in body.split('|') {
        let bar = bar.trim();
        match full_rest(bar) {
            Some(n) => out.extend(std::iter::repeat_n("Z".to_string(), n)),
            None => out.push(bar.to_string()),
        }
    }
    out
}

/// `Z` / `Z2` / `Z3` / `Z4`: how many resting measures.
fn full_rest(bar: &str) -> Option<usize> {
    match bar {
        "Z" => Some(1),
        "Z2" => Some(2),
        "Z3" => Some(3),
        "Z4" => Some(4),
        _ => None,
    }
}

fn parse(text: &str) -> Res<Score> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 12 {
        return fail("Incomplete native two-voice ABC");
    }
    if lines[0] != "X:1" || lines[1] != "T:" {
        return fail("Expected native X:1 and blank T: header");
    }
    let meter = match lines[2].strip_prefix("M:") {
        Some(m) => meter_value(m)?,
        None => return fail("Missing header M:"),
    };
    let unit_den = lines[3]
        .strip_prefix("L:1/")
        .and_then(|d| d.parse::<i64>().ok())
        .filter(|d| *d > 0 && *d <= 1024 && d & (d - 1) == 0)
        .ok_or("Expected L:1/<power of two>, usually L:1/32")?;
    let unit = WHOLE / unit_den;
    let bpm = lines[4]
        .strip_prefix("Q:1/4=")
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .ok_or("Expected integer quarter-note tempo Q:1/4=<BPM>")?;
    if lines[5] != VOCAL_DEF || lines[6] != INS_DEF {
        return fail("Preserve native Vocal and Ins voice definitions");
    }
    let key = lines[7].strip_prefix("K:").ok_or("Missing header K:")?.to_string();
    key_accidentals(&key)?;
    let fresh = || Voice {
        meter,
        key: key.clone(),
        time: 0,
        notes: Vec::new(),
        bars: Vec::new(),
        chords: Vec::new(),
        keys: vec![(0, key.clone())],
        pending: None,
    };
    let (mut vocal, mut ins) = (fresh(), fresh());
    let mut music_lines = BTreeMap::new();
    let mut cursor = 8;
    let mut group = 0;
    while cursor < lines.len() {
        while cursor < lines.len() && lines[cursor].starts_with("% ") {
            cursor += 1;
        }
        if cursor == lines.len() {
            return fail("Dangling section comment without music");
        }
        group += 1;
        let mut counts = [0usize; 2];
        for (vi, name) in ["Vocal", "Ins"].iter().enumerate() {
            let context = format!("group {group}, {name}");
            if cursor >= lines.len() || lines[cursor] != format!("V: {name}") {
                return fail(format!("{context}: expected V: {name}"));
            }
            cursor += 1;
            let voice = if vi == 0 { &mut vocal } else { &mut ins };
            let mut fields = BTreeSet::new();
            while cursor < lines.len() && (lines[cursor].starts_with("M:") || lines[cursor].starts_with("K:")) {
                let (field, value) = lines[cursor].split_at(1);
                let value = &value[1..];
                if !fields.insert(field.to_string()) {
                    return fail(format!("{context}: duplicate {field}: field"));
                }
                if field == "M" {
                    voice.meter = meter_value(value)?;
                } else {
                    key_accidentals(value)?;
                    voice.key = value.to_string();
                    voice.keys.push((voice.time, value.to_string()));
                }
                cursor += 1;
            }
            if cursor >= lines.len() {
                return fail(format!("{context}: missing music line"));
            }
            let line = lines[cursor];
            if !line.ends_with('|') {
                return fail(format!("{context}: music line must end with a plain barline"));
            }
            music_lines.insert(cursor, vi == 0);
            cursor += 1;
            let bars = line_bars(line);
            if bars.iter().any(String::is_empty) {
                return fail(format!("{context}: empty measure or unsupported double/repeat barline"));
            }
            if !(1..=4).contains(&bars.len()) {
                return fail(format!("{context}: expected 1–4 measures after expanding Z rests"));
            }
            counts[vi] = bars.len();
            for bar in &bars {
                let ctx = format!("{context}, bar {}", voice.bars.len() + 1);
                parse_bar(bar, voice, unit, &ctx)?;
            }
        }
        if counts[0] != counts[1] {
            return fail(format!("group {group}: voices have different measure counts"));
        }
    }
    if vocal.pending.is_some() || ins.pending.is_some() {
        return fail("unresolved tie at end of score");
    }
    if !ins.chords.is_empty() {
        return fail("Native chord symbols belong in Vocal, not Ins");
    }
    if vocal.bars != ins.bars {
        return fail("Voice meter/time grids differ");
    }
    if vocal.keys != ins.keys {
        return fail("Voice key-change timelines differ");
    }
    Ok(Score { bpm, vocal, ins, music_lines })
}

/// Durations in ABC units, longest first (ties join them).
fn lengths(units: i64) -> Res<Vec<i64>> {
    if units <= 0 {
        return fail(format!("Unrepresentable duration in ABC units: {units}"));
    }
    let mut left = units;
    let mut out = Vec::new();
    for part in DURATIONS.iter().rev() {
        while left >= *part {
            out.push(*part);
            left -= part;
        }
    }
    Ok(out)
}

/// A pitch written in a key, given the accidentals already in force in the
/// bar (which it may change).
fn spelling(pitch: i32, key: &str, active: &mut HashMap<char, i32>) -> Res<String> {
    let signature = key_accidentals(key)?;
    let prefer_sharp = signature.iter().sum::<i32>() >= 0;
    // (score, letter, alteration, octave), the smallest wins.
    type Candidate = ((bool, i32, bool), char, i32, i32);
    let mut best: Option<Candidate> = None;
    for (li, &letter) in LETTERS.iter().enumerate() {
        for alteration in [-1, 0, 1] {
            let base = pitch - alteration - 60 - NATURAL[li];
            if base.rem_euclid(12) != 0 {
                continue;
            }
            let octave = base.div_euclid(12);
            let score = (alteration != signature[li], alteration.abs(), if prefer_sharp { alteration < 0 } else { alteration > 0 });
            let cand = (score, letter, alteration, octave);
            if best.as_ref().is_none_or(|b| cand < *b) {
                best = Some(cand);
            }
        }
    }
    let (_, letter, alteration, octave) = best.ok_or("no spelling")?;
    let li = letter_index(letter);
    let mut out = String::new();
    if alteration != *active.get(&letter).unwrap_or(&signature[li]) {
        out.push(match alteration {
            -1 => '_',
            0 => '=',
            _ => '^',
        });
        active.insert(letter, alteration);
    }
    if octave < 0 {
        out.push(letter);
        out.push_str(&",".repeat((-octave) as usize));
    } else if octave == 0 {
        out.push(letter);
    } else {
        out.push(letter.to_ascii_lowercase());
        out.push_str(&"'".repeat((octave - 1) as usize));
    }
    Ok(out)
}

/// Runs of resting measures written `Z`, `Z2`…
fn compress(bars: &[String]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < bars.len() {
        if bars[i] != "Z" {
            out.push_str(&bars[i]);
            out.push('|');
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < bars.len() && bars[end] == "Z" {
            end += 1;
        }
        let n = end - i;
        out.push('Z');
        if n > 1 {
            out.push_str(&n.to_string());
        }
        out.push('|');
        i = end;
    }
    out
}

/// The smallest power-of-two ABC unit (as 1/L of a whole note) a time falls on.
fn grid(ticks: i64) -> i64 {
    if ticks == 0 {
        1
    } else {
        WHOLE / gcd(ticks.abs(), WHOLE)
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

struct Bar {
    start: i64,
    end: i64,
    meter: (i64, i64),
    key: String,
    section: Option<String>,
}

fn rests(units: i64) -> Res<String> {
    Ok(lengths(units)?.iter().map(|n| if *n == 1 { "z".to_string() } else { format!("z{n}") }).collect())
}

/// One instrumental melody, its bars and harmony, written in the dialect.
fn compile(bpm: i64, bars: Vec<Bar>, notes: &[(i64, i32, i64)], chords: &[(i64, String)]) -> Res<String> {
    let total = bars.last().map_or(0, |b| b.end);
    let mut unit_den = 32;
    for b in &bars {
        unit_den = unit_den.max(4 * b.meter.1);
    }
    for (start, _, dur) in notes {
        if *start < 0 || *dur <= 0 || start + dur > total {
            return fail("a note lies outside the score");
        }
        unit_den = unit_den.max(grid(*start)).max(grid(*dur));
    }
    if notes.is_empty() {
        return fail("Instrumental score has no sounding notes");
    }
    for w in notes.windows(2) {
        if w[0].0 + w[0].2 > w[1].0 {
            return fail("Overlapping melody notes; choose one lead line before compilation");
        }
    }
    for (when, symbol) in chords {
        if *when < 0 || *when >= total || !chord_re().is_match(symbol) {
            return fail("unsupported chord symbol or onset");
        }
        unit_den = unit_den.max(grid(*when));
    }
    if chords.windows(2).any(|w| w[0].0 == w[1].0) {
        return fail("Conflicting chord events at the same onset");
    }
    if unit_den > 1024 {
        return fail("Rhythm is outside the power-of-two grid");
    }
    let unit = WHOLE / unit_den;
    let mut ins = Vec::new();
    let mut vocal = Vec::new();
    for bar in &bars {
        let (start, end) = (bar.start, bar.end);
        let events: Vec<&(i64, i32, i64)> = notes.iter().filter(|n| n.0 < end && n.0 + n.2 > start).collect();
        let mut pieces = String::new();
        let mut cursor = start;
        let mut active = HashMap::new();
        for (onset, pitch, duration) in &events {
            let a = start.max(*onset);
            let b = end.min(onset + duration);
            if a > cursor {
                pieces.push_str(&rests((a - cursor) / unit)?);
            }
            let parts = lengths((b - a) / unit)?;
            for (i, count) in parts.iter().enumerate() {
                pieces.push_str(&spelling(*pitch, &bar.key, &mut active)?);
                if *count != 1 {
                    pieces.push_str(&count.to_string());
                }
                if i < parts.len() - 1 || b < onset + duration {
                    pieces.push('-');
                }
            }
            cursor = b;
        }
        if cursor < end {
            pieces.push_str(&rests((end - cursor) / unit)?);
        }
        ins.push(if events.is_empty() { "Z".to_string() } else { pieces });
        let current = chords.iter().rev().find(|(w, _)| *w <= start).map(|(_, s)| s.clone());
        let mut changes = vec![(start, current)];
        changes.extend(chords.iter().filter(|(w, _)| start < *w && *w < end).map(|(w, s)| (*w, Some(s.clone()))));
        let mut chunks = String::new();
        for (i, (when, symbol)) in changes.iter().enumerate() {
            let stop = changes.get(i + 1).map_or(end, |c| c.0);
            if let Some(s) = symbol {
                chunks.push('"');
                chunks.push_str(s);
                chunks.push('"');
            }
            chunks.push_str(&rests((stop - when) / unit)?);
        }
        vocal.push(if changes.iter().any(|(_, s)| s.is_some()) { chunks } else { "Z".to_string() });
    }
    let first = &bars[0];
    let mut lines = vec![
        "X:1".to_string(),
        "T:".to_string(),
        format!("M:{}/{}", first.meter.0, first.meter.1),
        format!("L:1/{unit_den}"),
        format!("Q:1/4={bpm}"),
        VOCAL_DEF.to_string(),
        INS_DEF.to_string(),
        format!("K:{}", first.key),
    ];
    let mut index = 0;
    while index < bars.len() {
        let bar = &bars[index];
        let previous = if index > 0 { &bars[index - 1] } else { bar };
        let mut end = index + 1;
        while end < bars.len()
            && end - index < 4
            && bars[end].meter == bar.meter
            && bars[end].key == bar.key
            && bars[end].section == bar.section
        {
            end += 1;
        }
        if let Some(section) = &bar.section {
            if index == 0 || Some(section) != previous.section.as_ref() {
                lines.push(format!("% {section}"));
            }
        }
        for (name, rendered) in [("Vocal", &vocal), ("Ins", &ins)] {
            lines.push(format!("V: {name}"));
            if bar.meter != previous.meter {
                lines.push(format!("M:{}/{}", bar.meter.0, bar.meter.1));
            }
            if bar.key != previous.key {
                lines.push(format!("K:{}", bar.key));
            }
            lines.push(compress(&rendered[index..end]));
        }
        index = end;
    }
    Ok(lines.join("\n") + "\n")
}

/// Where each section begins, from the score's `% section` comments.
fn section_starts(text: &str, score: &Score) -> Res<HashMap<i64, String>> {
    let mut cursor = 0;
    let mut pending: Option<String> = None;
    let mut out = HashMap::new();
    for (index, line) in text.lines().enumerate() {
        if let Some(label) = line.strip_prefix("% ") {
            if !SECTIONS.contains(&label) {
                return fail(format!("Unknown section label: {label}"));
            }
            pending = Some(label.to_string());
        }
        if score.music_lines.get(&index) != Some(&true) {
            continue;
        }
        if let Some(label) = pending.take() {
            if let Some(bar) = score.vocal.bars.get(cursor) {
                out.insert(bar.0, label);
            }
        }
        cursor += line_bars(line).len();
    }
    Ok(out)
}

/// Harmony as it sounds: a chord repeated where it is already in force is
/// the same harmony.
fn harmony(chords: &[(i64, String)]) -> Vec<(i64, String)> {
    let mut out: Vec<(i64, String)> = Vec::new();
    for c in chords {
        if out.last().is_none_or(|l| l.1 != c.1) {
            out.push(c.clone());
        }
    }
    out
}

/// What a conversion did, for the record and the tests.
#[derive(Debug, Clone, PartialEq)]
pub struct Converted {
    pub abc: String,
    /// Notes the Vocal part had (all moved to Ins).
    pub vocal_notes: usize,
    /// Notes Ins ends up with.
    pub ins_notes: usize,
    /// Whether the score has chords: they ask for `cot=full`, else `melody`.
    pub has_chords: bool,
    /// The score's sections as lyrics (`[Verse]\n\n[Chorus]\n`).
    pub section_lyrics: String,
}

/// Move every note of a YuE2 score's Vocal part into its Ins part. Where an
/// Ins note sounds under a Vocal one, the Vocal one wins and the Ins note
/// keeps only what is not covered. Harmony, meter, keys, tempo and sections
/// are kept, and the result is read back to prove it.
pub fn instrumental_score(text: &str) -> Res<Converted> {
    if text.trim().is_empty() {
        return fail("Need a nonempty native score");
    }
    let source = parse(text)?;
    let (vocal, ins) = (&source.vocal, &source.ins);
    if vocal.notes.is_empty() && ins.notes.is_empty() {
        return fail("The score contains no sounding notes");
    }
    let occupied: Vec<(i64, i64)> = vocal.notes.iter().map(|(t, _, d)| (*t, t + d)).collect();
    let mut merged: Vec<(i64, i32, i64)> = vocal.notes.clone();
    for (start, pitch, duration) in &ins.notes {
        let mut pieces = vec![(*start, start + duration)];
        for (left, right) in &occupied {
            let mut remaining = Vec::new();
            for (a, b) in pieces {
                if *right <= a || *left >= b {
                    remaining.push((a, b));
                } else {
                    if a < *left {
                        remaining.push((a, *left));
                    }
                    if *right < b {
                        remaining.push((*right, b));
                    }
                }
            }
            pieces = remaining;
        }
        merged.extend(pieces.into_iter().map(|(a, b)| (a, *pitch, b - a)));
    }
    merged.sort();
    let sections = section_starts(text, &source)?;
    let starts: BTreeSet<i64> = vocal.bars.iter().map(|b| b.0).collect();
    if vocal.keys.iter().any(|(t, _)| !starts.contains(t)) {
        return fail("An inline key change occurs inside a bar");
    }
    // Sections carry on until the next one begins.
    let mut section: Option<String> = None;
    let mut bars = Vec::new();
    for (start, length, meter) in &vocal.bars {
        let key = vocal.keys.iter().rev().find(|(t, _)| t <= start).map(|(_, k)| k.clone()).unwrap_or_default();
        if let Some(s) = sections.get(start) {
            section = Some(s.clone());
        }
        bars.push(Bar { start: *start, end: start + length, meter: *meter, key, section: section.clone() });
    }
    let mut chords: Vec<(i64, String)> = Vec::new();
    for (when, symbol) in &vocal.chords {
        if let Some(last) = chords.last() {
            if last.0 == *when {
                if last.1 != *symbol {
                    return fail("Conflicting chord labels at the same time");
                }
                continue;
            }
        }
        chords.push((*when, symbol.clone()));
    }
    let converted = compile(source.bpm, bars, &merged, &chords)?;
    let target = parse(&converted)?;
    if target.ins.notes != merged || !target.vocal.notes.is_empty() {
        return fail("Voice transfer changed the intended note events");
    }
    if target.ins.bars != ins.bars || target.bpm != source.bpm {
        return fail("Voice transfer changed meter, timing or tempo");
    }
    if harmony(&target.vocal.chords) != harmony(&vocal.chords) {
        return fail("Voice transfer changed harmony");
    }
    Ok(Converted {
        has_chords: !target.vocal.chords.is_empty(),
        section_lyrics: section_lyrics(&converted),
        vocal_notes: vocal.notes.len(),
        ins_notes: merged.len(),
        abc: converted,
    })
}

/// A score cut short at the planner's token limit, made instrumental from
/// its whole groups: the longest beginning that reads as a score (four bars
/// at least) and converts to no more text than the planner wrote — its
/// makers render nothing past the planner's own budget.
pub fn salvage(text: &str) -> Option<Converted> {
    prefixes(text).into_iter().find_map(|p| {
        let c = instrumental_score(&p).ok()?;
        (c.abc.len() <= text.len()).then_some(c)
    })
}

/// The longest whole-group beginning of a score cut short (see `salvage`).
pub fn complete_prefix(text: &str) -> Option<String> {
    prefixes(text).into_iter().next()
}

/// Every beginning of a score that ends with a whole group and reads, at
/// least four bars long, longest first.
fn prefixes(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    // Where each group begins: its section comments, then `V: Vocal`.
    let mut starts = Vec::new();
    for i in 9..lines.len() {
        if lines[i] == "V: Vocal" {
            let mut j = i;
            while j > 8 && lines[j - 1].starts_with("% ") {
                j -= 1;
            }
            starts.push(j);
        }
    }
    starts.push(lines.len());
    let mut out = Vec::new();
    for &cut in starts.iter().rev() {
        if cut <= 8 {
            break;
        }
        let candidate = lines[..cut].join("\n") + "\n";
        match parse(&candidate) {
            Ok(score) if score.vocal.bars.len() >= 4 => out.push(candidate),
            Ok(_) => break,
            Err(_) => {}
        }
    }
    out
}

/// Whether a score carries harmony: rendered with `cot=full`, else `melody`.
pub fn has_chords(text: &str) -> bool {
    match parse(text) {
        Ok(s) => !s.vocal.chords.is_empty(),
        // Not the dialect read here: whether it quotes a chord at all.
        Err(_) => text.lines().skip(8).any(|l| !l.starts_with("V:") && l.contains('"')),
    }
}

/// A score's `% section` comments as lyrics of section tags only: what an
/// instrumental is rendered with (no sung words).
pub fn section_lyrics(text: &str) -> String {
    let labels: Vec<String> = text.lines().filter_map(|l| l.strip_prefix("% ")).map(title_case).collect();
    if labels.is_empty() {
        return String::new();
    }
    labels.iter().map(|l| format!("[{l}]")).collect::<Vec<_>>().join("\n\n") + "\n"
}

/// Python's `str.title()`: each run of letters capitalised.
fn title_case(s: &str) -> String {
    let mut out = String::new();
    let mut prev_letter = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_letter {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_letter = true;
        } else {
            out.push(c);
            prev_letter = false;
        }
    }
    out
}

/// What YuE2 plans an instrumental from: the sections only, no words.
pub const PLANNING_LYRICS: &str = "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n";

/// A style that asks for no voice at all, the way the instrumental workflow
/// writes it: "Instrumental, …, no vocals, no singing, no choir, no spoken
/// words."
pub fn instrumental_style(style: &str) -> String {
    let mut s = style.trim().trim_end_matches(['.', ',']).to_string();
    if s.is_empty() {
        s = "Expressive instrumental music".into();
    }
    let lower = s.to_lowercase();
    let starts = lower.strip_prefix("instrumental").is_some_and(|rest| !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_'));
    if !starts {
        s = format!("Instrumental, {s}");
    }
    for condition in ["no vocals", "no singing", "no choir", "no spoken words"] {
        if !s.to_lowercase().contains(condition) {
            s.push_str(", ");
            s.push_str(condition);
        }
    }
    s.push('.');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=96\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n";

    /// A small native score: an intro on Ins, a sung verse with harmony, an
    /// Ins fill that overlaps the voice, an accidental and a tie over a bar.
    fn song() -> String {
        format!(
            "{HEAD}K:G\n% intro\nV: Vocal\n\"G\"z32|\nV: Ins\nd16B16|\n% verse\nV: Vocal\n\"G\"B8d8\"Am7\"c8A8|\"D7\"^c16d16-|d16z16|\nV: Ins\nZ|z16g16|z24B8|\n"
        )
    }

    #[test]
    fn the_voice_moves_to_the_instrument() {
        let c = instrumental_score(&song()).expect("converts");
        let out = parse(&c.abc).unwrap();
        assert!(out.vocal.notes.is_empty(), "nothing is left to sing");
        // B D C A, C# D (tied over the bar) from the voice; D B intro; the
        // fill G is cut by the held D, and the last B stands alone.
        let pitches: Vec<i32> = out.ins.notes.iter().map(|n| n.1).collect();
        assert_eq!(pitches, vec![74, 71, 71, 74, 72, 69, 73, 74, 71]);
        assert_eq!(c.vocal_notes, 7 - 1, "six sung notes once the tie is joined");
        // Harmony stays where it was.
        assert_eq!(
            harmony(&out.vocal.chords).iter().map(|c| c.1.as_str()).collect::<Vec<_>>(),
            vec!["G", "Am7", "D7"]
        );
        assert!(c.has_chords);
        assert_eq!(c.section_lyrics, "[Intro]\n\n[Verse]\n");
        assert_eq!(out.bpm, 96);
        assert_eq!(out.vocal.bars.len(), 4);
    }

    #[test]
    fn the_written_score_reads_back_as_written() {
        let c = instrumental_score(&song()).unwrap();
        assert_eq!(
            c.abc,
            format!(
                "{HEAD}K:G\n% intro\nV: Vocal\n\"G\"z32|\nV: Ins\nd16B16|\n% verse\nV: Vocal\n\"G\"z16\"Am7\"z16|\"D7\"z32|\"D7\"z32|\nV: Ins\nB8d8c8A8|^c16d16-|d16z8B8|\n"
            )
        );
        // Converting again changes nothing: there is no voice left to move.
        assert_eq!(instrumental_score(&c.abc).unwrap().abc, c.abc);
    }

    #[test]
    fn accidentals_hold_through_the_bar_by_letter() {
        // In C, ^F then f: both sharp (the native writer's rule).
        let text = format!("{HEAD}K:C\nV: Vocal\n^F8f8z16|\nV: Ins\nZ|\nV: Vocal\nZ|\nV: Ins\nZ|\nV: Vocal\nZ|\nV: Ins\nZ|\n");
        let s = parse(&text).unwrap();
        assert_eq!(s.vocal.notes.iter().map(|n| n.1).collect::<Vec<_>>(), vec![66, 78]);
    }

    #[test]
    fn what_is_not_the_dialect_is_refused() {
        assert!(instrumental_score("").is_err());
        assert!(instrumental_score("X:1\nT:Song\n").is_err());
        // A tuplet.
        let bad = format!("{HEAD}K:C\nV: Vocal\n(3CDE z29|\nV: Ins\nZ|\nV: Vocal\nZ|\nV: Ins\nZ|\nV: Vocal\nZ|\nV: Ins\nZ|\n");
        assert!(instrumental_score(&bad).unwrap_err().contains("unsupported token"));
        // A bar one unit short.
        let short = format!("{HEAD}K:C\nV: Vocal\nC31|\nV: Ins\nZ|\nV: Vocal\nZ|\nV: Ins\nZ|\nV: Vocal\nZ|\nV: Ins\nZ|\n");
        assert!(instrumental_score(&short).is_err());
    }

    /// Every `.abc` in `CHATY_ABC_DIR` converted beside itself as
    /// `<name>.instrumental.abc` — to compare with the skill's own
    /// `instrumentalize.py` on scores the model really wrote.
    #[test]
    #[ignore = "reads scores from CHATY_ABC_DIR"]
    fn real_scores_convert_like_the_skill_does() {
        let dir = std::path::PathBuf::from(std::env::var("CHATY_ABC_DIR").expect("set CHATY_ABC_DIR"));
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if !name.ends_with(".abc") || name.ends_with(".instrumental.abc") {
                continue;
            }
            match instrumental_score(&std::fs::read_to_string(&p).unwrap()) {
                Ok(c) => {
                    std::fs::write(dir.join(name.replace(".abc", ".instrumental.abc")), &c.abc).unwrap();
                    eprintln!("{name}: {} sung notes moved, {} in Ins", c.vocal_notes, c.ins_notes);
                }
                Err(e) => eprintln!("{name}: {e}"),
            }
        }
    }

    #[test]
    fn styles_ask_for_no_voice() {
        assert_eq!(
            instrumental_style("warm solo piano, gentle."),
            "Instrumental, warm solo piano, gentle, no vocals, no singing, no choir, no spoken words."
        );
        assert_eq!(
            instrumental_style("Instrumental lo-fi beat, no vocals"),
            "Instrumental lo-fi beat, no vocals, no singing, no choir, no spoken words."
        );
        assert_eq!(instrumental_style("instrumentals of the 80s").split(", ").next(), Some("Instrumental"));
        assert!(instrumental_style("").starts_with("Instrumental, Expressive instrumental music"));
    }

    #[test]
    fn a_score_cut_short_keeps_its_whole_groups() {
        let whole = song();
        // Cut in the middle of the verse's Ins line, as a token limit does.
        let cut = &whole[..whole.len() - 6];
        assert!(parse(cut).is_err());
        // What is left whole is the one-bar intro: under four bars there is
        // no score to use.
        assert_eq!(complete_prefix(cut), None);
        // A longer score cut in its last group keeps the groups before it.
        let longer = whole.clone() + "% outro\nV: Vocal\n\"G\"z32|\nV: Ins\nB32|\nV: Vocal\nZ|\nV: Ins\nd16";
        let kept = complete_prefix(&longer).expect("four whole bars");
        assert_eq!(kept, whole + "% outro\nV: Vocal\n\"G\"z32|\nV: Ins\nB32|\n");
        // Whole already: kept as it is.
        assert_eq!(complete_prefix(&song()).as_deref(), Some(song().as_str()));
        // Made instrumental, it may not outgrow what the planner wrote.
        let saved = salvage(&longer).expect("converts");
        assert!(saved.abc.len() <= longer.len());
        assert_eq!(saved.ins_notes, instrumental_score(&kept).unwrap().ins_notes);
    }

    #[test]
    fn sections_become_tags() {
        assert_eq!(section_lyrics("% pre-chorus\nx\n% outro\n"), "[Pre-Chorus]\n\n[Outro]\n");
        assert_eq!(section_lyrics("X:1\n"), "");
    }

    #[test]
    fn spelling_follows_the_key() {
        let mut a = HashMap::new();
        assert_eq!(spelling(66, "D", &mut a).unwrap(), "F", "F# is in D major");
        let mut a = HashMap::new();
        assert_eq!(spelling(66, "C", &mut a).unwrap(), "^F");
        assert_eq!(spelling(66, "C", &mut a).unwrap(), "F", "already sharp in this bar");
        let mut a = HashMap::new();
        assert_eq!(spelling(70, "F", &mut a).unwrap(), "B", "Bb is in F major");
        let mut a = HashMap::new();
        assert_eq!(spelling(48, "C", &mut a).unwrap(), "C,");
        assert_eq!(spelling(84, "C", &mut a).unwrap(), "c'");
    }
}
