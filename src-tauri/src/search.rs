//! Web search + page-content fetching via DuckDuckGo's HTML endpoint (no key).
//! Grounds answers when the user enables "联网搜索" — like ChatGPT, we don't
//! just hand the model link snippets, we fetch the top pages' actual text.

use std::time::Duration;

use percent_encoding::{percent_decode_str, utf8_percent_encode, NON_ALPHANUMERIC};
use scraper::{Html, Selector};
use serde::Serialize;
use tokio::task::JoinSet;

const PER_PAGE_CHARS: usize = 900;

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
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
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

async fn ddg_search(client: &reqwest::Client, query: &str) -> Result<Vec<SearchResult>, String> {
    let encoded = utf8_percent_encode(query.trim(), NON_ALPHANUMERIC).to_string();
    let url = format!("https://html.duckduckgo.com/html/?q={encoded}");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("搜索请求失败: {e}"))?;
    let html = resp.text().await.map_err(|e| e.to_string())?;
    Ok(parse_ddg(&html))
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
        if url.is_empty() {
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
