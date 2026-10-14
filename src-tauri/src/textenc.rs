//! Text in the character set it was written in — web pages and files alike —
//! and the text of an HTML document as a reader sees it.
//!
//! A page or a file that is not UTF-8 used to be read as UTF-8 anyway, or
//! as GBK whatever it was: a GBK page with its charset only in a `<meta>`, a
//! Windows "Unicode" (UTF-16) text file, a Big5 or Shift_JIS document came
//! through as rows of U+FFFD or as the wrong characters — in the sources
//! under an answer, in a knowledge base, and in what the model was given.

use encoding_rs::Encoding;

/// Bytes whose encoding nothing declared: a byte-order mark if there is one,
/// UTF-8 if they are UTF-8, else what chardetng (Firefox's detector) makes
/// of them — GBK, Big5, Shift_JIS, EUC-KR, windows-125x and the rest.
pub fn decode_unlabeled(bytes: &[u8]) -> String {
    if let Some((enc, _)) = Encoding::for_bom(bytes) {
        // `decode` drops the mark itself.
        return enc.decode(bytes).0.into_owned();
    }
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    let mut detector = chardetng::EncodingDetector::new(chardetng::Iso2022JpDetection::Deny);
    // A megabyte says all there is to say about the encoding.
    detector.feed(&bytes[..bytes.len().min(1 << 20)], true);
    let enc = detector.guess(None, chardetng::Utf8Detection::Allow);
    enc.decode(bytes).0.into_owned()
}

/// A web page or HTML file: by the Content-Type charset (`header`), else the
/// page's own `<meta>`, else as `decode_unlabeled`.
pub fn decode_page(bytes: &[u8], header: Option<&str>) -> String {
    let declared = header
        .and_then(|l| Encoding::for_label(l.trim().as_bytes()))
        .or_else(|| meta_charset(&bytes[..bytes.len().min(4096)]).and_then(|l| Encoding::for_label(l.as_bytes())));
    match declared {
        // `decode` lets a byte-order mark overrule the label, as browsers do.
        Some(enc) => enc.decode(bytes).0.into_owned(),
        None => decode_unlabeled(bytes),
    }
}

/// The `charset=` of a Content-Type value.
pub fn charset_param(ct: &str) -> Option<&str> {
    let lower = ct.to_ascii_lowercase();
    let at = lower.find("charset=")? + "charset=".len();
    let v = ct[at..].split(';').next()?.trim().trim_matches(['"', '\'']);
    (!v.is_empty()).then_some(v)
}

/// The charset a page declares in its head: `<meta charset="gbk">`, or
/// `<meta http-equiv="Content-Type" content="text/html; charset=gb2312">`.
fn meta_charset(head: &[u8]) -> Option<String> {
    let lower: Vec<u8> = head.iter().map(u8::to_ascii_lowercase).collect();
    let mut from = 0;
    while let Some(at) = find(&lower[from..], b"<meta") {
        let tag_start = from + at;
        let tag_end = find(&lower[tag_start..], b">").map_or(lower.len(), |e| tag_start + e);
        let tag = &lower[tag_start..tag_end];
        if let Some(c) = find(tag, b"charset") {
            let mut rest = tag[c + "charset".len()..].iter().skip_while(|b| b.is_ascii_whitespace());
            if rest.next() == Some(&b'=') {
                let label: String = rest
                    .skip_while(|b| b.is_ascii_whitespace() || **b == b'"' || **b == b'\'')
                    .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b':' | b'.'))
                    .map(|&b| b as char)
                    .collect();
                if !label.is_empty() {
                    return Some(label);
                }
            }
        }
        from = tag_end;
    }
    None
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Elements whose text a reader never sees.
const HIDDEN: &[&str] = &["script", "style", "noscript", "template", "head", "title", "svg", "iframe", "object"];
/// Elements that start a line of their own.
const BLOCKS: &[&str] = &[
    "p", "div", "br", "li", "dt", "dd", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
    "header", "footer", "aside", "nav", "main", "blockquote", "pre", "table", "ul", "ol", "dl", "hr",
    "figure", "figcaption", "form", "fieldset", "details", "summary", "address",
];

/// The text of an HTML document as a reader sees it: no tags, no scripts or
/// styles, a line per block (paragraph, heading, list item, table row), and
/// the white space inside a line folded the way a browser folds it.
pub fn html_text(html: &str) -> String {
    let doc = scraper::Html::parse_document(html);
    let mut raw = String::new();
    for node in doc.root_element().descendants() {
        if let Some(el) = node.value().as_element() {
            if BLOCKS.contains(&el.name()) {
                raw.push('\n');
            } else if matches!(el.name(), "td" | "th") {
                raw.push(' ');
            }
            continue;
        }
        let Some(text) = node.value().as_text() else { continue };
        let mut hidden = false;
        let mut pre = false;
        for a in node.ancestors() {
            if let Some(e) = a.value().as_element() {
                hidden |= HIDDEN.contains(&e.name());
                pre |= matches!(e.name(), "pre" | "textarea");
            }
        }
        if hidden {
            continue;
        }
        if pre {
            raw.push_str(text);
        } else {
            // A line break in the source is a space on the page.
            raw.extend(text.chars().map(|c| if c.is_whitespace() { ' ' } else { c }));
        }
    }
    raw.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZH: &str = "今天是2026年10月14日，农历九月初五";

    fn enc(e: &'static Encoding, s: &str) -> Vec<u8> {
        e.encode(s).0.into_owned()
    }

    #[test]
    fn a_page_is_read_in_the_charset_it_declares() {
        let html = format!("<html><head><meta charset=\"gb2312\"><title>t</title></head><body><p>{ZH}</p></body></html>");
        assert!(decode_page(&enc(encoding_rs::GBK, &html), None).contains(ZH));
        let html = format!("<head><META HTTP-EQUIV=\"Content-Type\" CONTENT=\"text/html; charset=GBK\"></head><p>{ZH}</p>");
        assert!(decode_page(&enc(encoding_rs::GBK, &html), None).contains(ZH));
        let html = format!("<p>{ZH}</p>");
        assert!(decode_page(&enc(encoding_rs::GBK, &html), Some("gbk")).contains(ZH));
    }

    #[test]
    fn undeclared_text_is_read_in_the_encoding_it_is_in() {
        // A long enough sample for the detector, as a real file would be.
        let zh = ZH.repeat(20);
        assert_eq!(decode_unlabeled(zh.as_bytes()), zh);
        assert_eq!(decode_unlabeled(&enc(encoding_rs::GBK, &zh)), zh, "GBK");
        let tw = "今天是二零二六年十月十四日，農曆九月初五，距離下一個節氣霜降還有九天。".repeat(10);
        assert_eq!(decode_unlabeled(&enc(encoding_rs::BIG5, &tw)), tw, "Big5");
        let ja = "今日は二〇二六年十月十四日です。日本語のテキストファイルを読み込みます。".repeat(10);
        assert_eq!(decode_unlabeled(&enc(encoding_rs::SHIFT_JIS, &ja)), ja, "Shift_JIS");
    }

    #[test]
    fn byte_order_marks_are_read_and_dropped() {
        let mut utf16 = vec![0xFF, 0xFE];
        for u in ZH.encode_utf16() {
            utf16.extend_from_slice(&u.to_le_bytes());
        }
        assert_eq!(decode_unlabeled(&utf16), ZH, "UTF-16LE, as Notepad saves \"Unicode\"");
        let mut utf8 = vec![0xEF, 0xBB, 0xBF];
        utf8.extend_from_slice(ZH.as_bytes());
        assert_eq!(decode_unlabeled(&utf8), ZH);
    }

    #[test]
    fn charset_params_and_meta_tags_are_found() {
        assert_eq!(charset_param("text/html; charset=GB2312"), Some("GB2312"));
        assert_eq!(charset_param("text/html;charset=\"utf-8\""), Some("utf-8"));
        assert_eq!(charset_param("text/html"), None);
        assert_eq!(meta_charset(b"<meta name=x><meta charset='big5'>").as_deref(), Some("big5"));
        assert_eq!(meta_charset(b"<meta name=\"viewport\" content=\"width=device-width\">"), None);
    }

    #[test]
    fn html_text_is_what_a_reader_sees() {
        let html = "<html><head><title>T</title><style>p{color:red}</style>\
            <script>var _hmt = [];</script></head><body>\
            <h1>Lifetimes</h1><p>A reference   is\n valid <b>while</b> its data lives.</p>\
            <ul><li>one</li><li>two</li></ul>\
            <p>x<script>document.createElement(\"script\")</script>y</p>\
            <table><tr><td>a</td><td>b</td></tr></table></body></html>";
        let text = html_text(html);
        assert_eq!(text, "Lifetimes\nA reference is valid while its data lives.\none\ntwo\nxy\na b");
    }
}
