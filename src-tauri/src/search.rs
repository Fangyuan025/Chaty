//! Web search + page-content fetching via DuckDuckGo's HTML endpoint (no key).
//! Grounds answers when the user enables "联网搜索" — like ChatGPT, we don't
//! just hand the model link snippets, we fetch the top pages' actual text.

use std::time::Duration;

use base64::Engine;
use percent_encoding::{percent_decode_str, utf8_percent_encode, NON_ALPHANUMERIC};
use scraper::{Html, Selector};
use serde::Serialize;
use serde_json::Value;
use tokio::task::JoinSet;

const PER_PAGE_CHARS: usize = 900;

// A current browser UA. DuckDuckGo started returning an HTTP-202 anomaly /
// verification page to the old Chrome/124 GET requests, which broke web search
// for every Chaty user. A fresh UA + POST gets real results again.
const UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15";

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PageContent {
    pub title: String,
    pub url: String,
    pub text: String,
}

#[derive(Serialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct WebResearch {
    pub results: Vec<SearchResult>,
    pub pages: Vec<PageContent>,
}

fn build_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(UA)
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| e.to_string())
}

/// Just the result list (titles/urls/snippets).
#[tauri::command]
pub async fn web_search(query: String) -> Result<Vec<SearchResult>, String> {
    let client = build_client()?;
    ddg_search(&client, &query).await
}

/// Result list **plus** the fetched main text of the top pages, for grounding.
#[tauri::command]
pub async fn web_research(query: String) -> Result<WebResearch, String> {
    let query = query.trim().to_string();
    if query.is_empty() {
        return Ok(WebResearch::default());
    }
    let client = build_client()?;
    let results = ddg_search(&client, &query).await?;

    // Fetch the top few pages concurrently; skip any that fail or time out.
    let mut set = JoinSet::new();
    for r in results.iter().take(6) {
        let client = client.clone();
        let url = r.url.clone();
        let title = r.title.clone();
        set.spawn(async move { fetch_page(&client, &url, &title).await });
    }
    let mut pages = Vec::new();
    while let Some(joined) = set.join_next().await {
        if let Ok(Some(page)) = joined {
            pages.push(page);
        }
    }

    Ok(WebResearch { results, pages })
}

/// Free, no-key web search with a multi-provider fallback chain — search
/// engines block scrapers aggressively (DDG now returns an HTTP-202 anomaly
/// page), so we try several independent sources and return the first that
/// yields results. Bing's HTML is the most scrape-tolerant major engine;
/// Wikipedia and DuckDuckGo's Instant-Answer JSON API are never blocked, so
/// the feature degrades gracefully instead of dying when SERPs are throttled.
async fn ddg_search(client: &reqwest::Client, query: &str) -> Result<Vec<SearchResult>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let mut last_err = String::new();
    macro_rules! try_provider {
        ($call:expr) => {
            match $call.await {
                Ok(r) if !r.is_empty() => return Ok(r),
                Ok(_) => {}
                Err(e) => last_err = e,
            }
        };
    }
    // Brave is an independent index with genuine relevance for both Chinese and
    // English and returns direct URLs — the best free, no-key SERP available.
    // The rest are fallbacks; Wikipedia is full-text (noisy for common Chinese
    // characters), so it sits near the end, and the DuckDuckGo Instant-Answer
    // API is the never-blocked last resort.
    try_provider!(brave_search(client, query));
    try_provider!(bing_search(client, query));
    try_provider!(ddg_endpoint(client, "https://html.duckduckgo.com/html/", query, parse_ddg));
    try_provider!(ddg_endpoint(client, "https://lite.duckduckgo.com/lite/", query, parse_lite));
    try_provider!(wikipedia_search(client, query));
    try_provider!(ddg_instant_answer(client, query));

    if last_err.is_empty() {
        Ok(Vec::new())
    } else {
        Err(format!("搜索请求失败 (all search providers failed): {last_err}"))
    }
}

/// Brave Search HTML scrape (no key). Independent index, strong relevance for
/// CJK + English, and result links are already direct URLs. Class names are
/// build-hashed (svelte-*), so we key off the stable `data-type="web"`
/// container, the `.title` element, and the first external link.
async fn brave_search(client: &reqwest::Client, query: &str) -> Result<Vec<SearchResult>, String> {
    let enc = utf8_percent_encode(query, NON_ALPHANUMERIC);
    let url = format!("https://search.brave.com/search?q={enc}&source=web");
    let resp = client
        .get(&url)
        .header(reqwest::header::REFERER, "https://search.brave.com/")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("Brave HTTP {}", resp.status()));
    }
    let html = resp.text().await.map_err(|e| e.to_string())?;
    Ok(parse_brave(&html))
}

fn parse_brave(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let (Ok(item_sel), Ok(link_sel), Ok(title_sel), Ok(desc_sel)) = (
        Selector::parse(r#"div[data-type="web"]"#),
        Selector::parse("a[href]"),
        Selector::parse(".title"),
        Selector::parse(".snippet-description, .snippet-content"),
    ) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for el in doc.select(&item_sel) {
        let url = el
            .select(&link_sel)
            .filter_map(|a| a.value().attr("href"))
            .find(|h| h.starts_with("http") && !h.contains("brave.com"))
            .unwrap_or_default()
            .to_string();
        if url.is_empty() || is_ad_url(&url) {
            continue;
        }
        // The `.title` element holds the full title in its `title=` attribute
        // (the visible text is line-clamped).
        let title = el
            .select(&title_sel)
            .next()
            .map(|t| {
                t.value()
                    .attr("title")
                    .map(str::to_string)
                    .unwrap_or_else(|| t.text().collect::<String>())
            })
            .unwrap_or_default()
            .trim()
            .to_string();
        let snippet = el
            .select(&desc_sel)
            .next()
            .map(|s| s.text().collect::<String>().split_whitespace().collect::<Vec<_>>().join(" "))
            .unwrap_or_default();
        out.push(SearchResult {
            title: if title.is_empty() { url.clone() } else { title },
            url,
            snippet,
        });
        if out.len() >= 8 {
            break;
        }
    }
    out
}

/// Bing HTML scrape (no key). Real result URLs are base64-wrapped in Bing's
/// `/ck/a?…&u=a1<base64url>` click-tracker, so we decode them back.
async fn bing_search(client: &reqwest::Client, query: &str) -> Result<Vec<SearchResult>, String> {
    let enc = utf8_percent_encode(query, NON_ALPHANUMERIC);
    // The market MUST match the query's script. Forcing an English locale on a
    // Chinese query makes Bing return total garbage (e.g. 刘华强 → baseball
    // scores), so route CJK queries to the zh-CN market.
    let cjk = query
        .chars()
        .any(|c| matches!(c as u32, 0x3400..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x2a6df));
    let mkt = if cjk { "zh-CN" } else { "en-US" };
    let url = format!("https://www.bing.com/search?q={enc}&mkt={mkt}&count=14");
    let resp = client
        .get(&url)
        .header(reqwest::header::REFERER, "https://www.bing.com/")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("Bing HTTP {}", resp.status()));
    }
    let html = resp.text().await.map_err(|e| e.to_string())?;
    Ok(parse_bing(&html))
}

fn parse_bing(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let (Ok(item_sel), Ok(link_sel), Ok(cap_sel)) = (
        Selector::parse("li.b_algo"),
        Selector::parse("h2 a"),
        Selector::parse(".b_caption p, p.b_lineclamp2, p.b_lineclamp3"),
    ) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for el in doc.select(&item_sel) {
        let Some(a) = el.select(&link_sel).next() else { continue };
        let title = a.text().collect::<String>().trim().to_string();
        if title.is_empty() {
            continue;
        }
        let url = decode_bing_url(a.value().attr("href").unwrap_or_default());
        if url.is_empty() || !url.starts_with("http") || is_ad_url(&url) {
            continue;
        }
        let snippet = el
            .select(&cap_sel)
            .next()
            .map(|s| s.text().collect::<String>().split_whitespace().collect::<Vec<_>>().join(" "))
            .unwrap_or_default();
        out.push(SearchResult { title, url, snippet });
        if out.len() >= 8 {
            break;
        }
    }
    out
}

/// Unwrap a Bing `/ck/a?…&u=a1<base64url>` redirect into the real URL.
fn decode_bing_url(href: &str) -> String {
    if !href.contains("/ck/a") {
        return if href.starts_with("http") { href.to_string() } else { String::new() };
    }
    let Some(idx) = href.find("u=a1") else { return String::new() };
    let enc = href[idx + 4..].split('&').next().unwrap_or_default();
    for engine in [
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
        &base64::engine::general_purpose::STANDARD_NO_PAD,
    ] {
        if let Ok(bytes) = engine.decode(enc) {
            if let Ok(s) = String::from_utf8(bytes) {
                if s.starts_with("http") {
                    return s;
                }
            }
        }
    }
    String::new()
}

/// Wikipedia full-text search (official API, no key, never blocked). Used as a
/// grounding fallback; picks the language edition by script (CJK → zh).
async fn wikipedia_search(client: &reqwest::Client, query: &str) -> Result<Vec<SearchResult>, String> {
    let cjk = query.chars().any(|c| matches!(c as u32, 0x3400..=0x9fff | 0xf900..=0xfaff));
    let lang = if cjk { "zh" } else { "en" };
    let enc = utf8_percent_encode(query, NON_ALPHANUMERIC);
    let url = format!(
        "https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={enc}&format=json&srlimit=6&srprop=snippet"
    );
    let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let json: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(arr) = json["query"]["search"].as_array() {
        for it in arr {
            let title = it["title"].as_str().unwrap_or("").to_string();
            if title.is_empty() {
                continue;
            }
            let snippet = strip_tags(it["snippet"].as_str().unwrap_or(""));
            let slug = title.replace(' ', "_");
            let page = utf8_percent_encode(&slug, NON_ALPHANUMERIC).to_string();
            out.push(SearchResult {
                title,
                url: format!("https://{lang}.wikipedia.org/wiki/{page}"),
                snippet,
            });
        }
    }
    Ok(out)
}

/// DuckDuckGo Instant-Answer JSON API (official, no key, never blocked). Limited
/// to an abstract + related topics — the last-resort grounding source.
async fn ddg_instant_answer(client: &reqwest::Client, query: &str) -> Result<Vec<SearchResult>, String> {
    let enc = utf8_percent_encode(query, NON_ALPHANUMERIC);
    let url = format!("https://api.duckduckgo.com/?q={enc}&format=json&no_html=1&t=chaty");
    let body = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let json: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let abs = json["AbstractText"].as_str().unwrap_or("");
    let abs_url = json["AbstractURL"].as_str().unwrap_or("");
    if !abs.is_empty() && abs_url.starts_with("http") {
        out.push(SearchResult {
            title: json["Heading"].as_str().unwrap_or(query).to_string(),
            url: abs_url.to_string(),
            snippet: abs.to_string(),
        });
    }
    fn walk(v: &Value, out: &mut Vec<SearchResult>) {
        if out.len() >= 8 {
            return;
        }
        if let Some(arr) = v.as_array() {
            for it in arr {
                walk(it, out);
            }
        } else if let Some(topics) = v["Topics"].as_array() {
            for it in topics {
                walk(it, out);
            }
        } else if let (Some(url), Some(text)) = (v["FirstURL"].as_str(), v["Text"].as_str()) {
            if url.starts_with("http") && !text.is_empty() {
                out.push(SearchResult {
                    title: text.chars().take(80).collect(),
                    url: url.to_string(),
                    snippet: text.to_string(),
                });
            }
        }
    }
    walk(&json["RelatedTopics"], &mut out);
    out.truncate(8);
    Ok(out)
}

/// Strip HTML tags (Wikipedia snippets wrap matches in `<span>`).
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// POST `q=<query>` to a DDG endpoint and parse its HTML with `parser`.
async fn ddg_endpoint(
    client: &reqwest::Client,
    endpoint: &str,
    query: &str,
    parser: fn(&str) -> Vec<SearchResult>,
) -> Result<Vec<SearchResult>, String> {
    let body = format!("q={}&kl=wt-wt", utf8_percent_encode(query, NON_ALPHANUMERIC));
    let resp = client
        .post(endpoint)
        .header(reqwest::header::REFERER, endpoint)
        .header(reqwest::header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let html = resp.text().await.map_err(|e| e.to_string())?;
    Ok(parser(&html))
}

/// DDG sponsored/ad links route through `duckduckgo.com/y.js` (or /ad/) — drop them.
fn is_ad_url(url: &str) -> bool {
    url.contains("duckduckgo.com/y.js") || url.contains("/y.js?") || url.contains(".ad_")
}

fn parse_ddg(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let (Ok(result_sel), Ok(title_sel), Ok(snippet_sel)) = (
        Selector::parse("div.result"),
        Selector::parse("a.result__a"),
        Selector::parse(".result__snippet"),
    ) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for el in doc.select(&result_sel) {
        let Some(title_el) = el.select(&title_sel).next() else {
            continue;
        };
        let title = title_el.text().collect::<String>().trim().to_string();
        if title.is_empty() {
            continue;
        }
        let url = title_el
            .value()
            .attr("href")
            .map(decode_ddg_url)
            .unwrap_or_default();
        if url.is_empty() || is_ad_url(&url) || !url.starts_with("http") {
            continue;
        }
        let snippet = el
            .select(&snippet_sel)
            .next()
            .map(|s| s.text().collect::<String>().trim().to_string())
            .unwrap_or_default();
        out.push(SearchResult { title, url, snippet });
        if out.len() >= 8 {
            break;
        }
    }
    out
}

/// Parse the `lite.duckduckgo.com` results page (a table of `a.result-link`
/// rows, with snippets in adjacent `.result-snippet` cells; ads use
/// `result-sponsored`, which we ignore).
fn parse_lite(html: &str) -> Vec<SearchResult> {
    let doc = Html::parse_document(html);
    let (Ok(link_sel), Ok(snip_sel)) = (
        Selector::parse("a.result-link"),
        Selector::parse(".result-snippet"),
    ) else {
        return Vec::new();
    };
    let snippets: Vec<String> = doc
        .select(&snip_sel)
        .map(|s| s.text().collect::<String>().split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let mut out = Vec::new();
    for (i, a) in doc.select(&link_sel).enumerate() {
        let title = a.text().collect::<String>().trim().to_string();
        let url = a.value().attr("href").map(decode_ddg_url).unwrap_or_default();
        if title.is_empty() || url.is_empty() || is_ad_url(&url) || !url.starts_with("http") {
            continue;
        }
        let snippet = snippets.get(i).cloned().unwrap_or_default();
        out.push(SearchResult { title, url, snippet });
        if out.len() >= 8 {
            break;
        }
    }
    out
}

async fn fetch_page(client: &reqwest::Client, url: &str, title: &str) -> Option<PageContent> {
    let resp = client.get(url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    // Skip obviously non-HTML responses (pdf, images, etc.).
    let is_html = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("html"))
        .unwrap_or(true);
    if !is_html {
        return None;
    }
    let html = resp.text().await.ok()?;
    let text = extract_main_text(&html, PER_PAGE_CHARS);
    if text.chars().count() < 80 {
        return None; // too little usable text (likely JS-rendered)
    }
    Some(PageContent {
        title: title.to_string(),
        url: url.to_string(),
        text,
    })
}

/// Crude readability: collect reasonably long paragraph/heading text.
fn extract_main_text(html: &str, cap: usize) -> String {
    let doc = Html::parse_document(html);
    let Ok(sel) = Selector::parse("article p, main p, p") else {
        return String::new();
    };
    let mut buf = String::new();
    for el in doc.select(&sel) {
        let raw = el.text().collect::<String>();
        let t = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if t.chars().count() < 24 {
            continue;
        }
        buf.push_str(&t);
        buf.push('\n');
        if buf.chars().count() >= cap {
            break;
        }
    }
    buf.trim().chars().take(cap).collect()
}

fn extract_title(html: &str) -> Option<String> {
    let doc = Html::parse_document(html);
    let sel = Selector::parse("title").ok()?;
    let t = doc.select(&sel).next()?.text().collect::<String>();
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

/// Fetch a single user-provided URL and return its title + main text.
#[tauri::command]
pub async fn fetch_url(url: String) -> Result<PageContent, String> {
    let url = url.trim().to_string();
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("无效的链接".into());
    }
    let client = build_client()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("请求失败: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let html = resp.text().await.map_err(|e| e.to_string())?;
    let text = extract_main_text(&html, 6000);
    if text.chars().count() < 40 {
        return Err("该网页正文为空（可能是动态渲染页面）".into());
    }
    let title = extract_title(&html).unwrap_or_else(|| url.clone());
    Ok(PageContent { title, url, text })
}

/// DDG wraps links as `//duckduckgo.com/l/?uddg=<encoded>&...`.
fn decode_ddg_url(href: &str) -> String {
    if let Some(idx) = href.find("uddg=") {
        let enc = href[idx + 5..].split('&').next().unwrap_or_default();
        return percent_decode_str(enc).decode_utf8_lossy().into_owned();
    }
    if let Some(stripped) = href.strip_prefix("//") {
        return format!("https://{stripped}");
    }
    href.to_string()
}
