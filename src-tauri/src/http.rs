//! The crate's HTTP identities and client factory.
//!
//! Before this module the same Safari user-agent string was pasted in three
//! files, four `const UA` declarations disagreed on what "UA" meant, and
//! `reqwest::Client::builder()` appeared at 29 call sites — several of them
//! rebuilding a client (and its connection pool) on every single request.
//!
//! Two identities exist ON PURPOSE and must not be merged:
//!
//! * [`BROWSER_UA`] — for fetching public web pages that gate or degrade
//!   content for non-browser agents (search results, article extraction).
//!   Sites serve a different document to `curl/1.0`; presenting a normal
//!   browser is what makes the extractor work at all.
//! * A branded UA (`"Chaty model downloader"`, `"Chaty-Updater"`, …) — for
//!   talking to APIs that WANT to know who is calling: Hugging Face, GitHub
//!   releases, our own endpoints. Rate limits and abuse handling there are
//!   per-client, so identifying honestly is the correct behavior.
//!
//! Picking the wrong one is a real bug in both directions, so each caller
//! still names its identity explicitly — the sharing is of the string and the
//! builder, never of the decision.

use std::time::Duration;

/// Safari on macOS. For public pages that serve non-browsers a worse document.
pub const BROWSER_UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15";

/// Build a client with an explicit identity and timeout.
pub fn client(ua: &str, timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(ua)
        .timeout(timeout)
        .build()
        .map_err(|e| e.to_string())
}

/// Seconds-flavored shorthand for the common case.
pub fn client_secs(ua: &str, secs: u64) -> Result<reqwest::Client, String> {
    client(ua, Duration::from_secs(secs))
}

/// Client for STREAMING downloads (model files, installers, embedders):
/// bounded connect + a between-chunks read timeout, and deliberately NO
/// whole-request timeout — a 20 GB model on a modest line is not an error,
/// but a CDN that goes silent for a minute is. Before this factory the
/// download clients had no timeout at all, and a stalled connection looked
/// like "download stuck at 43%" forever with no way out but a restart.
pub fn download_client(ua: &str) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(ua)
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())
}

/// A web page's text, in its own character set.
///
/// reqwest is built here without its `charset` feature, so
/// `Response::text()` reads every page as UTF-8 — and a page in GBK, which
/// many Chinese sites declare only in a `<meta>` tag, came out as rows of
/// U+FFFD, in the sources under an answer and in what the model was given.
/// The charset comes from the Content-Type header, else the page's own
/// `<meta>`, else a byte-order mark; with none, UTF-8 — unless the bytes are
/// not UTF-8 and read more cleanly as GB18030.
pub async fn page_text(resp: reqwest::Response) -> Result<String, String> {
    let header = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(charset_param)
        .map(str::to_string);
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    Ok(decode_page(&bytes, header.as_deref()))
}

/// `page_text` for bytes already read; `header` is the Content-Type charset.
pub fn decode_page(bytes: &[u8], header: Option<&str>) -> String {
    let declared = header
        .and_then(|l| encoding_rs::Encoding::for_label(l.trim().as_bytes()))
        .or_else(|| meta_charset(&bytes[..bytes.len().min(4096)]).and_then(|l| encoding_rs::Encoding::for_label(l.as_bytes())));
    if let Some(enc) = declared {
        // `decode` lets a byte-order mark overrule the label, as browsers do.
        return enc.decode(bytes).0.into_owned();
    }
    if let Some((enc, _)) = encoding_rs::Encoding::for_bom(bytes) {
        return enc.decode(bytes).0.into_owned();
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            let utf8 = encoding_rs::UTF_8.decode(bytes).0;
            let gb = encoding_rs::GB18030.decode(bytes).0;
            let bad = |s: &str| s.chars().filter(|&c| c == '\u{FFFD}').count();
            if bad(&gb) < bad(&utf8) { gb.into_owned() } else { utf8.into_owned() }
        }
    }
}

/// The `charset=` of a Content-Type value.
fn charset_param(ct: &str) -> Option<&str> {
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

#[cfg(test)]
mod tests {
    use super::*;

    const GBK_PAGE_TEXT: &str = "今天是2026年10月14日，农历九月初五";

    fn gbk(s: &str) -> Vec<u8> {
        encoding_rs::GBK.encode(s).0.into_owned()
    }

    #[test]
    fn a_page_is_read_in_the_charset_it_declares() {
        let html = format!("<html><head><meta charset=\"gb2312\"><title>t</title></head><body><p>{GBK_PAGE_TEXT}</p></body></html>");
        assert!(decode_page(&gbk(&html), None).contains(GBK_PAGE_TEXT));
        let html = format!("<head><META HTTP-EQUIV=\"Content-Type\" CONTENT=\"text/html; charset=GBK\"></head><p>{GBK_PAGE_TEXT}</p>");
        assert!(decode_page(&gbk(&html), None).contains(GBK_PAGE_TEXT));
        // The header wins over a page that says nothing.
        let html = format!("<p>{GBK_PAGE_TEXT}</p>");
        assert!(decode_page(&gbk(&html), Some("gbk")).contains(GBK_PAGE_TEXT));
    }

    #[test]
    fn an_undeclared_page_is_utf8_unless_it_is_not() {
        let html = format!("<p>{GBK_PAGE_TEXT}</p>");
        assert!(decode_page(html.as_bytes(), None).contains(GBK_PAGE_TEXT));
        assert!(decode_page(&gbk(&html), None).contains(GBK_PAGE_TEXT), "GBK with no declaration");
    }

    #[test]
    fn charset_params_and_meta_tags_are_found() {
        assert_eq!(charset_param("text/html; charset=GB2312"), Some("GB2312"));
        assert_eq!(charset_param("text/html;charset=\"utf-8\""), Some("utf-8"));
        assert_eq!(charset_param("text/html"), None);
        assert_eq!(meta_charset(b"<meta name=x><meta charset='big5'>").as_deref(), Some("big5"));
        assert_eq!(meta_charset(b"<meta name=\"viewport\" content=\"width=device-width\">"), None);
    }
}
