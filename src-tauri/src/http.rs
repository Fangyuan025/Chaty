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
