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
            .map_err(|e| anyhow!("chat template does not parse as Jinja: {e}"))?;
        Ok(Self { env, bos: bos.to_string(), eos: eos.to_string() })
    }

    pub fn render(&self, turns: &[Turn], add_generation_prompt: bool) -> Result<String> {
        self.render_with(turns, add_generation_prompt, None)
    }

    /// `enable_thinking`: the convention most templates that reason read
    /// (Qwen3, EXAONE 4, SmolLM3, MiniCPM5…). Left undefined when the caller
    /// has no preference, so the template's own default applies — and set
    /// when it does, or a template that defaults to thinking OFF (EXAONE 4)
    /// could never be made to think.
    pub fn render_with(
        &self,
        turns: &[Turn],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        // Field order is what Python's dict would give: a template that dumps
        // a whole message with `tojson` prints its keys in this order.
        #[derive(serde::Serialize)]
        struct Msg<'a> {
            role: &'a str,
            content: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            reasoning_content: Option<&'a str>,
        }
        let messages: Vec<Value> = turns
            .iter()
            .map(|t| Value::from_serialize(Msg { role: t.role, content: t.content, reasoning_content: t.reasoning }))
            .collect();
        let tmpl = self.env.get_template("chat").map_err(|e| anyhow!("{e}"))?;
        let base = minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos.as_str(),
            eos_token => self.eos.as_str(),
        };
        let ctx = match enable_thinking {
            Some(on) => minijinja::context! { enable_thinking => on, ..base },
            None => base,
        };
        tmpl.render(ctx).map_err(|e| anyhow!("chat template failed to render: {e}"))
    }
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
/// inside and nothing else. minijinja does not know the tag (LFM2's template
/// uses it), so it is taken out before compiling — the body stays.
fn without_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let Some(close) = rest[open..].find("%}") else { break };
        let tag = &rest[open + 2..open + close];
        let word = tag.trim().trim_matches('-').trim();
        if word == "generation" || word == "endgeneration" {
            // Whitespace control on the tag still applies to its surroundings.
            let head = &rest[..open];
            out.push_str(if tag.starts_with('-') { head.trim_end() } else { head });
            rest = &rest[open + close + 2..];
            if tag.ends_with('-') {
                rest = rest.trim_start();
            }
        } else {
            out.push_str(&rest[..open + close + 2]);
            rest = &rest[open + close + 2..];
        }
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

    /// A template that reads `enable_thinking` hears the caller's choice, and
    /// keeps its own default when there is none.
    #[test]
    fn thinking_reaches_the_template() {
        let src = "{% if enable_thinking is defined and enable_thinking is true %}ON{% elif enable_thinking is defined %}OFF{% else %}DEFAULT{% endif %}";
        let c = Compiled::new(src, "", "").unwrap();
        assert_eq!(c.render_with(&[], true, Some(true)).unwrap(), "ON");
        assert_eq!(c.render_with(&[], true, Some(false)).unwrap(), "OFF");
        assert_eq!(c.render_with(&[], true, None).unwrap(), "DEFAULT");
    }

    #[test]
    fn a_template_that_is_not_jinja_is_refused_not_guessed() {
        assert!(Compiled::new("{% for m in messages %}", "", "").is_err());
    }
}
