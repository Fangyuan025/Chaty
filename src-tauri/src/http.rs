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

/// A web page's text, in its own character set (see `textenc`). reqwest is
/// built here without its `charset` feature, so `Response::text()` reads
/// every page as UTF-8.
pub async fn page_text(resp: reqwest::Response) -> Result<String, String> {
    let header = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(crate::textenc::charset_param)
        .map(str::to_string);
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    Ok(crate::textenc::decode_page(&bytes, header.as_deref()))
}

/// Stream `url` into `dest`, telling `progress` (bytes so far, total when the
/// server says) a few times a second, and refusing a body that stops short of
/// the length it was announced with. The partial file is removed on failure.
pub async fn download_to(
    ua: &str,
    url: &str,
    dest: &std::path::Path,
    progress: impl Fn(u64, Option<u64>),
) -> Result<(), String> {
    use std::io::Write;
    let client = download_client(ua)?;
    let mut resp = client
        .get(url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| trf!("下载失败:{e}", "the download failed: {e}"))?;
    let total = resp.content_length();
    let mut file = std::fs::File::create(dest).map_err(|e| trf!("无法写入下载的文件:{e}", "could not write the download: {e}"))?;
    let mut done = 0u64;
    let mut last = std::time::Instant::now();
    progress(done, total);
    loop {
        let chunk = match resp.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(dest);
                return Err(trf!("下载中断:{e}", "the download broke off: {e}"));
            }
        };
        if let Err(e) = file.write_all(&chunk) {
            drop(file);
            let _ = std::fs::remove_file(dest);
            return Err(trf!("无法写入下载的文件:{e}", "could not write the download: {e}"));
        }
        done += chunk.len() as u64;
        if last.elapsed().as_millis() >= 150 {
            progress(done, total);
            last = std::time::Instant::now();
        }
    }
    let flushed = file.flush();
    drop(file);
    if let Err(e) = flushed {
        let _ = std::fs::remove_file(dest);
        return Err(trf!("无法写入下载的文件:{e}", "could not write the download: {e}"));
    }
    if total.is_some_and(|t| t != done) {
        let _ = std::fs::remove_file(dest);
        return Err(trf!(
            "下载不完整(收到 {} / {} 字节),请重试",
            "the download came down incomplete ({} of {} bytes) — try again",
            done,
            total.unwrap_or(0)
        ));
    }
    progress(done, total);
    Ok(())
}

