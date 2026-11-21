//! The model's own chat template, rendered as Jinja.
//!
//! llama.cpp's `llama_chat_apply_template` does not run Jinja. It recognises a
//! template by substrings — `<|im_start|>` means ChatML — and prints its
//! built-in rendition of that family. Whatever a template does beyond the
//! canonical format is lost without a word: a BOS the template writes itself,
//! a default system prompt, a date header. MiniCPM5's official GGUF declares
//! no automatic BOS and writes `{{- bos_token }}` into its template instead;
//! rendered the llama.cpp way it had no BOS at all and answered in fragments
//! (issue #20).
//!
//! The built-in path is not wrong everywhere, and where it is right it has to
//! stay: it is what the cache-reuse work on the Qwen line was measured
//! against, and Qwen3's real template drops past reasoning from history —
//! switching those models wholesale would turn 99% reuse into none. So the
//! choice is made per model, on evidence: at first use, a short conversation
//! is rendered both ways and tokenized, and only a template whose tokens come
//! out different — the built-in rendition is NOT what the model was trained
//! on — is rendered here from then on.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use minijinja::{Environment, Error, ErrorKind, Value};

/// A template compiled once and the special-token text it is given.
pub struct Compiled {
    env: Environment<'static>,
    pub bos: String,
    pub eos: String,
    /// The template offers `preserve_thinking` (LFM3's does): keep the
    /// reasoning of earlier turns instead of re-rendering them with an empty
    /// thinking block.
    keeps_thinking_on_request: bool,
    /// Inkling's: a turn is blocks of thought, text and calls, and the
    /// template builds them from structured fields (see `inkling_turn`).
    inkling: bool,
    /// Olala's: a turn is named channels, which the template builds from
    /// structured fields too (see `olala_raw`).
    olala: bool,
}

/// One message as a template sees it.
pub struct Turn<'a> {
    pub role: &'a str,
    pub content: &'a str,
    pub reasoning: Option<&'a str>,
}

impl Compiled {
    /// Compile `source` in an environment that behaves like the one Hugging
    /// Face renders chat templates in: blocks trimmed, Python string and dict
    /// methods, `raise_exception`, `strftime_now`.
    pub fn new(source: &str, bos: &str, eos: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> String {
            chrono::Local::now().format(&fmt).to_string()
        });
        // transformers replaces Jinja's `tojson` with its own: `json.dumps`
        // with `ensure_ascii`, `indent`, `separators` and `sort_keys`, and no
        // HTML escaping. Templates call it with those keywords (EXAONE's wraps
        // every tool result in `tojson(ensure_ascii=False)`), and minijinja's
        // own filter refuses them — the render failed for any conversation
        // with a tool result in it.
        env.add_filter("tojson", py_tojson);
        env.add_template_owned("chat", without_generation_tags(source))
            .map_err(|e| anyhow!(trf!("模型的对话模板无法按 Jinja 解析:{e}", "the chat template does not parse as Jinja: {e}")))?;
        Ok(Self {
            env,
            bos: bos.to_string(),
            eos: eos.to_string(),
            keeps_thinking_on_request: source.contains("preserve_thinking"),
            inkling: source.contains("<|message_model|>") && source.contains("<|content_invoke_tool_json|>"),
            olala: super::is_olala_template(source),
        })
    }

    pub fn render(&self, turns: &[Turn], add_generation_prompt: bool) -> Result<String> {
        self.render_with(turns, add_generation_prompt, Kwargs::default())
    }

    /// Render with the caller's template kwargs. Each is left undefined when
    /// the caller has no preference, so the template's own default applies.
    pub fn render_with(&self, turns: &[Turn], add_generation_prompt: bool, kw: Kwargs) -> Result<String> {
        // Field order is what Python's dict would give: a template that dumps
        // a whole message with `tojson` prints its keys in this order.
        #[derive(serde::Serialize)]
        struct Msg<'a> {
            role: &'a str,
            content: std::borrow::Cow<'a, str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            reasoning_content: Option<std::borrow::Cow<'a, str>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool_calls: Option<Vec<serde_json::Value>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool_call_id: Option<String>,
        }
        let mut last_call: Option<String> = None;
        // Olala's calls of the last assistant turn, `(id, name)`, which the
        // tool results after it answer in order.
        let mut olala_calls: std::collections::VecDeque<(String, String)> = Default::default();
        // Inkling's turns go back as the model wrote them, spliced in where
        // the template puts a stand-in (see `inkling_turn`).
        let mut spliced: Vec<(String, String)> = Vec::new();
        let messages: Vec<Value> = turns
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut m = Msg {
                    role: t.role,
                    content: t.content.into(),
                    reasoning_content: t.reasoning.map(Into::into),
                    tool_calls: None,
                    name: None,
                    tool_call_id: None,
                };
                if self.inkling && t.role == "assistant" {
                    let blocks = inkling_turn(t.content);
                    last_call = blocks.calls.last().map(|(name, _)| name.clone()).or(last_call.take());
                    if let Some(raw) = inkling_raw(t.content) {
                        let stand_in = format!("\u{F8FF}chaty-turn-{i}\u{F8FF}");
                        spliced.push((format!("<|message_model|><|content_text|>{stand_in}<|end_message|>"), raw.to_string()));
                        m.content = stand_in.into();
                        m.reasoning_content = None;
                    } else {
                        m.content = blocks.text.into();
                        if m.reasoning_content.is_none() {
                            m.reasoning_content = blocks.thought.map(Into::into);
                        }
                        if !blocks.calls.is_empty() {
                            m.tool_calls = Some(
                                blocks
                                    .calls
                                    .into_iter()
                                    .map(|(name, args)| serde_json::json!({ "type": "function", "function": { "name": name, "arguments": args } }))
                                    .collect(),
                            );
                        }
                    }
                } else if self.inkling && t.role == "tool" {
                    m.name = last_call.clone();
                } else if self.olala && t.role == "assistant" {
                    if let Some(raw) = olala_raw(t.content) {
                        olala_calls = olala_call_ids(raw).into();
                        let stand_in = format!("\u{F8FF}chaty-turn-{i}\u{F8FF}");
                        spliced.push((
                            format!("{}<|channel_end|><|channel_start|>final<|content|>{stand_in}<|channel_end|>", super::OLALA_ANALYSIS),
                            raw.to_string(),
                        ));
                        m.content = stand_in.into();
                        m.reasoning_content = None;
                    }
                } else if self.olala && t.role == "tool" {
                    // The template heads a result with the call it answers,
                    // `{"name": …, "id": …}`; without them, `{}`.
                    if let Some((id, name)) = olala_calls.pop_front() {
                        m.name = Some(name);
                        m.tool_call_id = Some(id);
                    }
                    // Its own tool_output channel holds the result; Chaty's
                    // envelope around it is one more thing to imitate, and
                    // Olala did — answering with `<tool_result>`s it wrote.
                    m.content = olala_result_body(t.content).into();
                }
                Value::from_serialize(m)
            })
            .collect();
        let tmpl = self.env.get_template("chat").map_err(|e| anyhow!("{e}"))?;
        let mut ctx: std::collections::BTreeMap<&str, Value> = std::collections::BTreeMap::new();
        ctx.insert("messages", Value::from(messages));
        ctx.insert("add_generation_prompt", Value::from(add_generation_prompt));
        ctx.insert("bos_token", Value::from(self.bos.as_str()));
        ctx.insert("eos_token", Value::from(self.eos.as_str()));
        if let Some(on) = kw.enable_thinking {
            ctx.insert("enable_thinking", Value::from(on));
        }
        if let Some(rung) = kw.reasoning_effort {
            ctx.insert("reasoning_effort", Value::from(rung));
        }
        // A template that drops earlier turns' reasoning unless asked not to
        // is asked not to. Re-rendered without it, the turn the model just
        // wrote no longer matches the cache — and on a hybrid model, whose
        // state cannot rewind, every turn then re-reads the whole conversation
        // (LFM3: 0% reuse turn after turn). The same trade the owner took for
        // Gemma 4: reasoning stays in history, compaction trims it when the
        // window tightens.
        if self.keeps_thinking_on_request {
            ctx.insert("preserve_thinking", Value::from(true));
        }
        let mut out = tmpl
            .render(Value::from_serialize(&ctx))
            .map_err(|e| anyhow!(trf!("模型的对话模板渲染失败:{e}", "the chat template failed to render: {e}")))?;
        for (stand_in, raw) in &spliced {
            if out.contains(stand_in.as_str()) {
                out = out.replacen(stand_in.as_str(), raw, 1);
            }
        }
        Ok(out)
    }
}

/// An Inkling turn as the model wrote it, taken apart into what its template
/// builds one from.
///
/// The model writes a turn as blocks, each opened by `<|message_model|>` and
/// closed by `<|end_message|>`: a thought (`<|content_thinking|>`), text
/// (`<|content_text|>`), or a call (`NAME<|content_invoke_tool_json|>` and
/// `{"name":…,"args":{…}}`). The template renders a turn from its
/// `reasoning_content`, `content` and `tool_calls`, wrapping the content in a
/// text block of its own — handed the blocks as they were written, it put
/// them all inside one text block, calls included, and the model read back
/// a turn it had never written. Text outside any block (a turn stored before
/// the model's blocks were known, or cut short) stays text; a call whose JSON
/// does not parse stays where it was, as text.
#[derive(Debug, Default, PartialEq)]
pub struct InklingTurn {
    pub thought: Option<String>,
    pub text: String,
    pub calls: Vec<(String, serde_json::Value)>,
}

pub fn inkling_turn(content: &str) -> InklingTurn {
    const HEADER: &str = "<|message_model|>";
    const END: &str = "<|end_message|>";
    let mut out = InklingTurn::default();
    let mut thought = String::new();
    let content = content.trim_end().strip_suffix("<|content_model_end_sampling|>").unwrap_or(content);
    let mut rest = content;
    while !rest.is_empty() {
        let Some(at) = rest.find(HEADER) else {
            out.text.push_str(rest);
            break;
        };
        out.text.push_str(&rest[..at]);
        let block = &rest[at + HEADER.len()..];
        let (body, next) = match block.find(END) {
            Some(e) => (&block[..e], &block[e + END.len()..]),
            None => (block, ""),
        };
        rest = next;
        if let Some(t) = body.strip_prefix("<|content_thinking|>") {
            thought.push_str(t);
        } else if let Some(t) = body.strip_prefix("<|content_text|>") {
            out.text.push_str(t);
        } else if let Some((name, json)) = body.split_once("<|content_invoke_tool_json|>") {
            let parsed = serde_json::from_str::<serde_json::Value>(json).ok();
            let args = parsed.as_ref().and_then(|v| v.get("args").cloned()).filter(|a| a.is_object());
            // The name before the marker, or — when the model left it out
            // there — the one in its JSON.
            let name = Some(name.trim())
                .filter(|n| !n.is_empty())
                .map(str::to_string)
                .or_else(|| parsed.as_ref().and_then(|v| v.get("name")?.as_str().map(str::to_string)))
                .unwrap_or_default();
            match args {
                Some(args) if !name.is_empty() => out.calls.push((name, args)),
                _ => out.text.push_str(&rest_of(HEADER, body, END, next.is_empty() && !block.contains(END))),
            }
        } else {
            out.text.push_str(&rest_of(HEADER, body, END, next.is_empty() && !block.contains(END)));
        }
    }
    if !thought.is_empty() {
        out.thought = Some(thought);
    }
    out
}

/// An Inkling turn as it goes back to the model: the blocks exactly as it
/// wrote them, less the turn's end, which the template writes after the turn.
/// Taken apart and built again by the template, a turn came back as the
/// template writes one — a call's tool name before its marker, its JSON keys
/// sorted and spaces dropped — where the model had written its call without
/// the name, or its text and call in another order; and the model's state is
/// partly recurrent, so a turn that is not the one in its cache to the token
/// is the whole cache read again. `None` for a turn with no blocks in it (one
/// another model wrote), which the template renders as text.
pub fn inkling_raw(content: &str) -> Option<&str> {
    let raw = content.trim_end();
    let raw = raw.strip_suffix("<|content_model_end_sampling|>").unwrap_or(raw);
    (raw.starts_with("<|message_model|>") && raw.contains("<|end_message|>")).then_some(raw)
}

/// An Olala turn as it goes back to the model: its channels exactly as it
/// wrote them, less the turn's end, which the template writes after the turn.
/// Rebuilt by the template a call came back with the template's own id
/// (`call_0`) and `tojson`'s spacing, and its state is partly recurrent (Mamba
/// layers), so a turn that is not the one in its cache to the token is the
/// whole cache read again. `None` for a turn with no channels in it (one
/// another model wrote), which the template renders as an answer.
pub fn olala_raw(content: &str) -> Option<&str> {
    let raw = content.trim_end();
    let raw = raw.strip_suffix("<|im_end|>").unwrap_or(raw);
    raw.starts_with("<|channel_start|>").then_some(raw)
}

/// A tool result as Olala's tool_output channel takes it: what Chaty's
/// `<tool_result name="…">` envelope holds, the envelope itself left out.
fn olala_result_body(content: &str) -> &str {
    let Some(rest) = content.strip_prefix("<tool_result") else { return content };
    let Some(gt) = rest.find(">\n") else { return content };
    let body = &rest[gt + 2..];
    let trimmed = body.trim_end();
    trimmed.strip_suffix("\n</tool_result>").or_else(|| trimmed.strip_suffix("</tool_result>")).unwrap_or(body)
}

/// The calls an Olala turn makes, `(id, name)` in the order written.
fn olala_call_ids(raw: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = raw;
    while let Some(at) = rest.find("<call") {
        let call = &rest[at..];
        let end = call.find("</call>").map_or(call.len(), |e| e + "</call>".len());
        let body = &call[..end];
        let id = body
            .split_once("id=\"")
            .and_then(|(_, r)| r.split_once('"'))
            .map(|(id, _)| id.to_string())
            .unwrap_or_default();
        let name = body
            .split_once("<name>")
            .and_then(|(_, r)| r.split_once("</name>"))
            .map(|(n, _)| n.trim().to_string())
            .unwrap_or_default();
        if !name.is_empty() {
            out.push((id, name));
        }
        rest = &call[end..];
    }
    out
}

/// A block put back as it was written.
fn rest_of(header: &str, body: &str, end: &str, unclosed: bool) -> String {
    if unclosed {
        format!("{header}{body}")
    } else {
        format!("{header}{body}{end}")
    }
}

/// What the caller asks of a template beyond the conversation.
///
/// `enable_thinking`: the convention most templates that reason read (Qwen3,
/// EXAONE 4, SmolLM3, MiniCPM5, K2 Horizon…). Set only when the caller has a
/// preference — a template that defaults to thinking OFF (EXAONE 4) could
/// never be made to think otherwise.
///
/// `reasoning_effort`: the rung, for templates that offer a ladder under that
/// name (Qwen3.8: low/medium/xhigh; K2 Horizon: low/medium/high, which also
/// picks the reasoning tag the turn opens with).
#[derive(Clone, Copy, Default, Debug)]
pub struct Kwargs<'a> {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<&'a str>,
}

/// `tojson` as transformers defines it — Python's `json.dumps`:
/// separators `", "` and `": "` (`","` between items once indented), no HTML
/// escaping, `ensure_ascii` off unless asked for, keys in the order given
/// unless `sort_keys`. Also accepts the indent positionally, as Jinja does.
fn py_tojson(
    value: Value,
    indent: Option<Value>,
    kwargs: minijinja::value::Kwargs,
) -> std::result::Result<Value, Error> {
    let ensure_ascii: bool = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let sort_keys: bool = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let indent: Option<Value> = match kwargs.get::<Option<Value>>("indent")? {
        Some(v) => Some(v),
        None => indent,
    };
    let indent: Option<String> = match indent {
        None => None,
        Some(v) if v.is_none() || v.is_undefined() => None,
        Some(v) => match v.as_i64() {
            Some(n) => Some(" ".repeat(n.max(0) as usize)),
            None => Some(v.as_str().unwrap_or("").to_string()),
        },
    };
    let separators: Option<(String, String)> = kwargs
        .get::<Option<Value>>("separators")?
        .and_then(|v| {
            let mut it = v.try_iter().ok()?;
            let a = it.next()?.as_str()?.to_string();
            let b = it.next()?.as_str()?.to_string();
            Some((a, b))
        });
    let (item_sep, key_sep) = separators.unwrap_or_else(|| {
        (if indent.is_some() { ",".to_string() } else { ", ".to_string() }, ": ".to_string())
    });
    let opts = JsonOpts { ensure_ascii, sort_keys, indent, item_sep, key_sep };
    let mut out = String::new();
    write_json(&value, &opts, 0, &mut out)?;
    Ok(Value::from_safe_string(out))
}

struct JsonOpts {
    ensure_ascii: bool,
    sort_keys: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
}

fn write_json_str(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_json(v: &Value, o: &JsonOpts, level: usize, out: &mut String) -> std::result::Result<(), Error> {
    use minijinja::value::ValueKind;
    let newline = |out: &mut String, level: usize| {
        if let Some(ind) = &o.indent {
            out.push('\n');
            for _ in 0..level {
                out.push_str(ind);
            }
        }
    };
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => out.push_str(&v.to_string()),
        ValueKind::String => write_json_str(v.as_str().unwrap_or(""), o.ensure_ascii, out),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&o.item_sep);
                }
                newline(out, level + 1);
                write_json(item, o, level + 1, out)?;
            }
            newline(out, level);
            out.push(']');
        }
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if o.sort_keys {
                keys.sort_by_key(|k| k.to_string());
            }
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(&o.item_sep);
                }
                newline(out, level + 1);
                write_json_str(&k.to_string(), o.ensure_ascii, out);
                out.push_str(&o.key_sep);
                write_json(&v.get_item(k)?, o, level + 1, out)?;
            }
            newline(out, level);
            out.push('}');
        }
        _ => write_json_str(&v.to_string(), o.ensure_ascii, out),
    }
    Ok(())
}

/// `{% generation %}…{% endgeneration %}` is a transformers extension that
/// marks the assistant's tokens for training masks; rendering prints what is
/// inside and nothing else. minijinja does not know the tag (LFM2's and K2
/// Horizon's templates use it), so it becomes `{% if true %}…{% endif %}` —
/// a block tag that does nothing, which is exactly what it is to the renderer.
///
/// It has to stay a TAG. Deleting it merged the text on either side into one
/// run, and trim_blocks / lstrip_blocks no longer applied where the tag had
/// stood: the next `{%-` then ate the newline before it along with the
/// indentation, and K2 Horizon's stored turns came out `assistant<ifm|think>`
/// where transformers writes `assistant\n<ifm|think>` — every earlier turn
/// in a different shape from the one the model generated.
fn without_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let Some(close) = rest[open..].find("%}") else { break };
        let tag = &rest[open + 2..open + close];
        let word = tag.trim().trim_matches('-').trim();
        let stand_in = match word {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };
        out.push_str(&rest[..open]);
        match stand_in {
            // Whitespace control on the tag keeps applying to its surroundings.
            Some(w) => {
                out.push_str("{%");
                if tag.starts_with('-') {
                    out.push('-');
                }
                out.push(' ');
                out.push_str(w);
                out.push(' ');
                if tag.ends_with('-') {
                    out.push('-');
                }
                out.push_str("%}");
            }
            None => out.push_str(&rest[open..open + close + 2]),
        }
        rest = &rest[open + close + 2..];
    }
    out.push_str(rest);
    out
}

/// Per template: the compiled template when it should be used, `None` when the
/// built-in rendering is faithful (or the template cannot be compiled). Keyed
/// by the template text and its special tokens — the same file loaded again
/// is decided once.
type Decisions = HashMap<u64, Option<Arc<Compiled>>>;
static DECIDED: OnceLock<Mutex<Decisions>> = OnceLock::new();

pub fn key_of(source: &str, bos: &str, eos: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut h);
    bos.hash(&mut h);
    eos.hash(&mut h);
    h.finish()
}

pub fn decided(key: u64) -> Option<Option<Arc<Compiled>>> {
    DECIDED.get_or_init(Default::default).lock().ok()?.get(&key).cloned()
}

pub fn decide(key: u64, choice: Option<Arc<Compiled>>) {
    if let Ok(mut d) = DECIDED.get_or_init(Default::default).lock() {
        d.insert(key, choice);
    }
}

/// The conversations the two renderings are compared on. Plain turns only —
/// no reasoning, no tools — so what differs is the wire format itself, and
/// no generation prompt, whose thinking preamble the engine manages on its own.
/// One with a system message and one without: some templates add a default
/// system prompt only when none is given.
pub fn probes() -> [Vec<(&'static str, &'static str)>; 2] {
    [
        vec![
            ("system", "You are a helpful assistant."),
            ("user", "Hello."),
            ("assistant", "Hi! How can I help?"),
            ("user", "What is 2 + 2?"),
        ],
        vec![
            ("user", "Hello."),
            ("assistant", "Hi! How can I help?"),
            ("user", "What is 2 + 2?"),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns<'a>(v: &'a [(&'a str, &'a str)]) -> Vec<Turn<'a>> {
        v.iter().map(|(r, c)| Turn { role: r, content: c, reasoning: None }).collect()
    }

    /// The template shape at the heart of issue #20: the BOS is the
    /// template's to write.
    #[test]
    fn a_template_that_writes_its_own_bos_gets_it() {
        let src = "{{- bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let c = Compiled::new(src, "<s>", "</s>").unwrap();
        let out = c.render(&turns(&[("user", "hi")]), true).unwrap();
        assert_eq!(out, "<s><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    /// Hugging Face's environment: trimmed blocks, Python string methods,
    /// raise_exception, namespaces and loop controls — the constructs real
    /// templates are made of.
    #[test]
    fn templates_written_for_transformers_render_the_same_way() {
        let src = r#"{%- set ns = namespace(sys='', told=false) -%}
{%- for m in messages -%}
  {%- if m.role == 'system' -%}{%- set ns.sys = m.content.strip() -%}{%- continue -%}{%- endif -%}
  {%- if m.role not in ['user', 'assistant'] -%}{{ raise_exception('bad role') }}{%- endif %}
[{{ m.role.upper() }}] {{ (ns.sys + ' ' if not ns.told and ns.sys else '') ~ m.content.split('|')[0] }}
{% set ns.told = true %}
{%- endfor -%}"#;
        let c = Compiled::new(src, "", "").unwrap();
        let out = c
            .render(&turns(&[("system", "  be brief  "), ("user", "a|b"), ("assistant", "ok")]), false)
            .unwrap();
        assert_eq!(out, "[USER] be brief a\n[ASSISTANT] ok\n");
        let err = c.render(&turns(&[("tool", "x")]), false).unwrap_err().to_string();
        assert!(err.contains("bad role"), "{err}");
    }

    #[test]
    fn reasoning_reaches_templates_that_read_it() {
        let src = "{% for m in messages %}{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}{% endfor %}";
        let c = Compiled::new(src, "", "").unwrap();
        let out = c
            .render(&[Turn { role: "assistant", content: "42", reasoning: Some("add") }], false)
            .unwrap();
        assert_eq!(out, "<think>add</think>42");
    }

    /// LFM2's template wraps assistant turns in transformers' `generation`
    /// tag; rendering prints the body and nothing else.
    #[test]
    fn generation_tags_render_their_body() {
        let src = "{% for m in messages %}{% if m.role == 'assistant' %}{%- generation -%} A:{{ m.content }} {%- endgeneration -%}{% else %}U:{{ m.content }}{% endif %}|{% endfor %}";
        let c = Compiled::new(src, "", "").unwrap();
        let out = c.render(&turns(&[("user", "q"), ("assistant", "a")]), false).unwrap();
        assert_eq!(out, "U:q|A:a|");
    }

    /// The tag's own whitespace rules still apply where it stood. K2 Horizon's
    /// template prints the role, then opens `{% generation %}` on the next
    /// line; transformers keeps the newline between them (expected value from
    /// jinja2 with transformers' settings), and dropping the tag used to let
    /// the following `{%-` eat it.
    #[test]
    fn generation_tags_keep_the_whitespace_around_them() {
        let src = "{%- for message in messages -%}\n    {%- if message.role == \"assistant\" -%}\n        {{- '<s>' + message.role }}\n        {% generation %}\n        {%- if true -%}\n            {{- '<t>\\n' + message.content }}\n        {%- endif -%}\n        {{- '<e>' -}}\n        {%- endgeneration -%}\n    {%- else -%}\n        {{- '<s>' + message.role + '\\n' + message.content + '<e>' }}\n    {%- endif -%}\n{%- endfor -%}\n";
        let c = Compiled::new(src, "", "").unwrap();
        let out = c.render(&turns(&[("user", "q"), ("assistant", "a")]), false).unwrap();
        assert_eq!(out, "<s>user\nq<e><s>assistant\n<t>\na<e>");
    }

    /// `tojson` renders as Python's `json.dumps` does, keywords and all.
    #[test]
    fn tojson_matches_transformers() {
        let one = |src: &str, content: &str| {
            Compiled::new(src, "", "").unwrap().render(&turns(&[("tool", content)]), false).unwrap()
        };
        let content = "结果 \"ok\"\n<b>";
        // transformers' default: no ASCII escaping, no HTML escaping.
        assert_eq!(one("{{ messages[0].content | tojson }}", content), "\"结果 \\\"ok\\\"\\n<b>\"");
        assert_eq!(one("{{ messages[0].content | tojson(ensure_ascii=False) }}", content), "\"结果 \\\"ok\\\"\\n<b>\"");
        assert_eq!(
            one("{{ messages[0].content | tojson(ensure_ascii=True) }}", content),
            "\"\\u7ed3\\u679c \\\"ok\\\"\\n<b>\""
        );
        // Python's separators, and a message's keys in the order given.
        assert_eq!(
            one("{{ messages | tojson }}", "x"),
            "[{\"role\": \"tool\", \"content\": \"x\"}]"
        );
        assert_eq!(
            one("{{ messages | tojson(sort_keys=True) }}", "x"),
            "[{\"content\": \"x\", \"role\": \"tool\"}]"
        );
        let ind = Compiled::new("{{ {'a': 1, 'b': [true, none]} | tojson(indent=2) }}", "", "").unwrap();
        assert_eq!(ind.render(&[], false).unwrap(), "{\n  \"a\": 1,\n  \"b\": [\n    true,\n    null\n  ]\n}");
    }

    /// A template that keeps earlier reasoning only when asked
    /// (`preserve_thinking`, LFM3's) is asked; one that never mentions it is
    /// rendered as before.
    #[test]
    fn earlier_reasoning_is_kept_where_the_template_offers_it() {
        let src = "{%- set keep = preserve_thinking | default(false) -%}{% for m in messages %}{% if m.role == 'assistant' and not keep %}<think></think>{{ m.content.split('</think>')[-1] }}{% else %}{{ m.content }}{% endif %}|{% endfor %}";
        let c = Compiled::new(src, "", "").unwrap();
        let turns = [
            Turn { role: "user", content: "q", reasoning: None },
            Turn { role: "assistant", content: "<think>R</think>A", reasoning: None },
        ];
        assert_eq!(c.render(&turns, false).unwrap(), "q|<think>R</think>A|");
        let plain = Compiled::new("{% for m in messages %}{{ m.content }}|{% endfor %}", "", "").unwrap();
        assert_eq!(plain.render(&turns, false).unwrap(), "q|<think>R</think>A|");
    }

    /// A template that reads `enable_thinking` hears the caller's choice, and
    /// keeps its own default when there is none.
    #[test]
    fn thinking_reaches_the_template() {
        let src = "{% if enable_thinking is defined and enable_thinking is true %}ON{% elif enable_thinking is defined %}OFF{% else %}DEFAULT{% endif %}";
        let c = Compiled::new(src, "", "").unwrap();
        let think = |on| Kwargs { enable_thinking: on, ..Kwargs::default() };
        assert_eq!(c.render_with(&[], true, think(Some(true))).unwrap(), "ON");
        assert_eq!(c.render_with(&[], true, think(Some(false))).unwrap(), "OFF");
        assert_eq!(c.render_with(&[], true, think(None)).unwrap(), "DEFAULT");
    }

    /// The rung reaches a template that offers a ladder (K2 Horizon's shape:
    /// the rung picks the tag the turn opens with), and the default stands
    /// when none is asked for.
    #[test]
    fn effort_reaches_the_template() {
        let src = "{%- set effort = reasoning_effort | default('high') -%}{%- if enable_thinking is defined and enable_thinking is false -%}{{- '<t>\\n</t>\\n' }}{%- elif effort == 'high' -%}{{- '<t>\\n' }}{%- elif effort == 'low' -%}{{- '<t_faster>\\n' }}{%- endif -%}";
        let c = Compiled::new(src, "", "").unwrap();
        let kw = |on, rung| Kwargs { enable_thinking: on, reasoning_effort: rung };
        assert_eq!(c.render_with(&[], true, kw(None, None)).unwrap(), "<t>\n");
        assert_eq!(c.render_with(&[], true, kw(Some(true), Some("low"))).unwrap(), "<t_faster>\n");
        assert_eq!(c.render_with(&[], true, kw(Some(false), Some("low"))).unwrap(), "<t>\n</t>\n");
    }

    #[test]
    fn a_template_that_is_not_jinja_is_refused_not_guessed() {
        assert!(Compiled::new("{% for m in messages %}", "", "").is_err());
    }

    /// Inkling's own template, from the Inkling 2 Mini GGUF.
    const INKLING2: &str = include_str!("testdata/inkling2.jinja");

    /// A turn read back is the turn the model wrote: its blocks in their
    /// order, then the turn's end — what the cache already holds.
    #[test]
    fn an_inkling_turn_reads_back_as_it_was_written() {
        let t = Compiled::new(INKLING2, "", "").unwrap();
        let said = "<|message_model|><|content_thinking|>17*23 is 391.<|end_message|><|message_model|><|content_text|>391<|end_message|>";
        let out = t
            .render(
                &[
                    Turn { role: "user", content: "17*23?", reasoning: None },
                    Turn { role: "assistant", content: said, reasoning: None },
                    Turn { role: "user", content: "thanks", reasoning: None },
                ],
                true,
            )
            .unwrap();
        assert!(out.contains(&format!("<|content_text|>17*23?<|end_message|>{said}<|content_model_end_sampling|><|message_user|>")), "{out}");
        assert!(out.starts_with("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"), "{out}");
    }

    #[test]
    fn an_inkling_call_and_its_result_read_back_natively() {
        let t = Compiled::new(INKLING2, "", "").unwrap();
        let call = "<|message_model|>read_file<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{\"path\":\"a.txt\",\"limit\":5}}<|end_message|>";
        let said = format!("<|message_model|><|content_thinking|>Read it.<|end_message|>{call}");
        let out = t
            .render(
                &[
                    Turn { role: "user", content: "what is in a.txt?", reasoning: None },
                    Turn { role: "assistant", content: &said, reasoning: None },
                    Turn { role: "tool", content: "hello", reasoning: None },
                ],
                true,
            )
            .unwrap();
        // As written — not the template's canonical call, keys sorted — and
        // the result under the call's name.
        assert!(
            out.contains(&format!("{said}<|content_model_end_sampling|><|message_tool|>read_file<|content_text|>hello<|end_message|>")),
            "{out}"
        );
    }

    /// A call written without the tool's name before its marker, after a
    /// text block: back to the token as written, the result named from the
    /// call's JSON. Rebuilt by the template, it gained the name and lost the
    /// text block's close, and the whole cache went with it.
    #[test]
    fn an_inkling_turn_goes_back_exactly_as_written() {
        let t = Compiled::new(INKLING2, "", "").unwrap();
        let said = "<|message_model|><|content_thinking|>Run it.<|end_message|><|message_model|><|content_text|>跑测试。<|end_message|><|message_model|><|content_invoke_tool_json|>{\"name\": \"bash\", \"args\": {\"command\": \"pytest\"}}<|end_message|>";
        let out = t
            .render(
                &[
                    Turn { role: "user", content: "test it", reasoning: None },
                    Turn { role: "assistant", content: said, reasoning: None },
                    Turn { role: "tool", content: "ok", reasoning: None },
                ],
                true,
            )
            .unwrap();
        assert!(out.contains(&format!("{said}<|content_model_end_sampling|><|message_tool|>bash<|content_text|>ok<|end_message|>")), "{out}");
        // A plain turn (another model's) is still text.
        let out = t.render(&[Turn { role: "user", content: "hi", reasoning: None }, Turn { role: "assistant", content: "hello", reasoning: None }], false).unwrap();
        assert!(out.contains("<|message_model|><|content_text|>hello<|end_message|><|content_model_end_sampling|>"), "{out}");
    }

    const OLALA: &str = include_str!("testdata/olala.jinja");

    /// Olala's turns go back as it wrote them: its channels, the call with its
    /// own id and spacing, and the reasoning of turns before the last question
    /// kept (`preserve_thinking`), so each prompt is the last one and more.
    #[test]
    fn an_olala_turn_goes_back_exactly_as_written() {
        let t = Compiled::new(OLALA, "", "").unwrap();
        let said = "<|channel_start|>analysis<|content|>Read it first.<|channel_end|><|channel_start|>final<|content|><|channel_end|><|channel_start|>tools<|content|><toolcalls><call id=\"call_1\"><name>read_file</name><arguments>{\"path\":\"a.md\"}</arguments></call></toolcalls><|channel_end|>";
        let answered = "<|channel_start|>analysis<|content|>It says hi.<|channel_end|><|channel_start|>final<|content|>It says hi.<|channel_end|>";
        let turns = [
            Turn { role: "user", content: "what is in a.md?", reasoning: None },
            Turn { role: "assistant", content: said, reasoning: None },
            Turn { role: "tool", content: "hi", reasoning: None },
            Turn { role: "assistant", content: answered, reasoning: None },
            Turn { role: "user", content: "thanks", reasoning: None },
        ];
        let out = t.render(&turns, true).unwrap();
        assert!(
            out.contains(&format!("<|im_start|>assistant{said}<|im_end|><|im_start|>tool<|channel_start|>metadata<|content|>{{\"name\": \"read_file\", \"id\": \"call_1\"}}<|channel_end|><|channel_start|>tool_output<|content|>hi<|channel_end|><|im_end|><|im_start|>assistant{answered}<|im_end|>")),
            "{out}"
        );
        assert!(out.ends_with("<|im_start|>user<|channel_start|>text<|content|>thanks<|channel_end|><|im_end|><|im_start|>assistant<|channel_start|>analysis<|content|>"), "{out}");
        // The turn before is the start of the one after.
        let before = t.render(&turns[..4], false).unwrap();
        assert!(out.starts_with(&before), "{before}\n---\n{out}");
        // Thinking off is the `disabled` rung, the analysis channel closed.
        let off = t.render_with(&turns[..1], true, Kwargs { reasoning_effort: Some("disabled"), ..Default::default() }).unwrap();
        assert!(off.contains("Reasoning effort: disabled"), "{off}");
        assert!(off.ends_with("<|im_start|>assistant<|channel_start|>analysis<|content|><|channel_end|>"), "{off}");
        // Another model's plain turn is an answer.
        let plain = t.render(&[Turn { role: "user", content: "hi", reasoning: None }, Turn { role: "assistant", content: "hello", reasoning: None }], false).unwrap();
        assert!(plain.contains("<|im_start|>assistant<|channel_start|>analysis<|content|><|channel_end|><|channel_start|>final<|content|>hello<|channel_end|><|im_end|>"), "{plain}");
    }

    #[test]
    fn an_inkling_turn_is_taken_apart_into_its_blocks() {
        let turn = inkling_turn("<|message_model|><|content_thinking|>a<|end_message|><|message_model|><|content_text|>b<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"x\":1}}<|end_message|>");
        assert_eq!(turn.thought.as_deref(), Some("a"));
        assert_eq!(turn.text, "b");
        assert_eq!(turn.calls, vec![("f".to_string(), serde_json::json!({ "x": 1 }))]);
        // Plain text is text; a call that does not parse stays as written.
        assert_eq!(inkling_turn("just text"), InklingTurn { text: "just text".into(), ..Default::default() });
        let broken = "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{<|end_message|>";
        assert_eq!(inkling_turn(broken).text, broken);
        assert!(inkling_turn(broken).calls.is_empty());
        // Cut short mid-thought: the thought so far.
        assert_eq!(inkling_turn("<|message_model|><|content_thinking|>half").thought.as_deref(), Some("half"));
    }
}
