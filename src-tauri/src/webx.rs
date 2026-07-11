//! Extended web tools for the coding agent — the "no-blind-spots" layer on
//! top of search.rs:
//!  * `site_search` — structured in-site search: GitHub (repos + issues via
//!    the key-less REST API, code via Sourcegraph's public stream API),
//!    Reddit (via its still-open RSS endpoints), and a `site:` engine-chain
//!    fallback for every other domain (including x.com, whose only key-less
//!    surface is the engines' snapshot index).
//!  * `fetch_page_ex` — content-type-aware fetching: readable Markdown for
//!    HTML (dom_smoothie Readability + htmd), raw source on demand, plain
//!    text passthrough for code/JSON/CSS/XML, PDF text extraction, and
//!    metadata for binaries — plus harvested links/images so the agent can
//!    walk into sub-pages.
//! Everything is key-less and talks to the sites directly (no proxy service).

use std::time::Duration;

use scraper::{Html, Selector};
use serde::Serialize;
use serde_json::Value;
use url::Url;

use crate::search::{engine_search, SearchResult};

const UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15";
/// Character budget for returned text — the agent has its own read budget,
/// anything bigger just burns context.
const TEXT_CAP: usize = 60_000;
const PDF_CAP_BYTES: usize = 15 * 1024 * 1024;

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SiteResult {
    /// repo | issue | code | post | web
    pub kind: String,
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LinkRef {
    pub url: String,
    pub text: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PageEx {
    /// Final URL after redirects / rewrites (github blob → raw, reddit → rss).
    pub url: String,
    /// markdown | source | text | pdf | binary
    pub kind: String,
    pub content_type: String,
    pub title: String,
    pub text: String,
    pub truncated: bool,
    /// Links found on the page (same-host first) — fetch them to go deeper.
    pub links: Vec<LinkRef>,
    /// Image URLs found on the page — save with web_download.
    pub images: Vec<String>,
    /// Body size for binaries (when known).
    pub bytes: Option<u64>,
}

fn client(secs: u64) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(UA)
        .timeout(Duration::from_secs(secs))
        .build()
        .map_err(|e| e.to_string())
}

fn cap(s: &str, max: usize) -> (String, bool) {
    if s.chars().count() <= max {
        return (s.to_string(), false);
    }
    (s.chars().take(max).collect(), true)
}

// ---------------------------------------------------------------- site search

/// Structured in-site search. `site` is a domain like "github.com" or
/// "reddit.com"; anything unrecognized degrades to an engine `site:` query,
/// so *every* site is searchable.
#[tauri::command]
pub async fn site_search(site: String, query: String) -> Result<Vec<SiteResult>, String> {
    let s = site
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("www.")
        .trim_end_matches('/')
        .to_ascii_lowercase();
    let query = query.trim().to_string();
    if query.is_empty() {
        return Ok(Vec::new());
    }
    match s.as_str() {
        "github.com" | "github" => github_search(&query).await,
        "reddit.com" | "reddit" => reddit_search(&query, None).await,
        "youtube.com" | "youtube" | "youtu.be" => youtube_search(&query).await,
        "bilibili.com" | "bilibili" | "b23.tv" => bilibili_search(&query).await,
        // r/rust style scoping: site "reddit.com/r/rust"
        _ if s.starts_with("reddit.com/r/") => {
            let sub = s.trim_start_matches("reddit.com/r/").to_string();
            reddit_search(&query, Some(&sub)).await
        }
        _ => engine_site_search(&s, &query).await,
    }
}

/// `site:` filter over the existing key-less engine chain (Brave → Bing →
/// DDG → …). For x.com this searches the engines' snapshot index — the only
/// key-less view of X that exists.
async fn engine_site_search(site: &str, query: &str) -> Result<Vec<SiteResult>, String> {
    let results: Vec<SearchResult> = engine_search(&format!("site:{site} {query}")).await?;
    Ok(results
        .into_iter()
        .filter(|r| {
            Url::parse(&r.url)
                .ok()
                .and_then(|u| u.host_str().map(|h| h.contains(site.split('/').next().unwrap_or(site))))
                .unwrap_or(false)
        })
        .map(|r| SiteResult { kind: "web".into(), title: r.title, url: r.url, snippet: r.snippet })
        .collect())
}

/// GitHub: repositories + issues/PRs via the key-less REST search API
/// (10 req/min unauthenticated — plenty for an agent), code matches via
/// Sourcegraph's public stream API (GitHub's own code search requires auth).
/// If everything comes back empty (rate-limited), fall back to `site:`.
async fn github_search(query: &str) -> Result<Vec<SiteResult>, String> {
    let c = client(12)?;
    let (repos, issues, code) = tokio::join!(
        github_repos(&c, query),
        github_issues(&c, query),
        sourcegraph_code(&c, query),
    );
    let mut out = Vec::new();
    out.extend(repos.unwrap_or_default());
    out.extend(issues.unwrap_or_default());
    out.extend(code.unwrap_or_default());
    if out.is_empty() {
        return engine_site_search("github.com", query).await;
    }
    Ok(out)
}

async fn github_repos(c: &reqwest::Client, query: &str) -> Result<Vec<SiteResult>, String> {
    let url = format!(
        "https://api.github.com/search/repositories?q={}&per_page=6",
        urlencoding(query)
    );
    let v = c
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&v).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for item in v["items"].as_array().unwrap_or(&Vec::new()) {
        let name = item["full_name"].as_str().unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let stars = item["stargazers_count"].as_u64().unwrap_or(0);
        let desc = item["description"].as_str().unwrap_or_default();
        let lang = item["language"].as_str().unwrap_or_default();
        out.push(SiteResult {
            kind: "repo".into(),
            title: format!("{name} (★{stars}{})", if lang.is_empty() { String::new() } else { format!(", {lang}") }),
            url: item["html_url"].as_str().unwrap_or_default().to_string(),
            snippet: desc.chars().take(200).collect(),
        });
    }
    Ok(out)
}

async fn github_issues(c: &reqwest::Client, query: &str) -> Result<Vec<SiteResult>, String> {
    let url = format!(
        "https://api.github.com/search/issues?q={}&per_page=6",
        urlencoding(query)
    );
    let v = c
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&v).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for item in v["items"].as_array().unwrap_or(&Vec::new()) {
        let title = item["title"].as_str().unwrap_or_default();
        if title.is_empty() {
            continue;
        }
        let state = item["state"].as_str().unwrap_or("?");
        let comments = item["comments"].as_u64().unwrap_or(0);
        let body = item["body"].as_str().unwrap_or_default();
        out.push(SiteResult {
            kind: "issue".into(),
            title: format!("{title} [{state}, {comments} comments]"),
            url: item["html_url"].as_str().unwrap_or_default().to_string(),
            snippet: body.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(220).collect(),
        });
    }
    Ok(out)
}

/// Public code search over Sourcegraph's index of open-source repos.
/// The stream endpoint needs no key; we read the whole SSE body and pull the
/// `event: matches` frames.
async fn sourcegraph_code(c: &reqwest::Client, query: &str) -> Result<Vec<SiteResult>, String> {
    let url = format!(
        "https://sourcegraph.com/.api/search/stream?q={}&display=8",
        urlencoding(&format!("context:global count:8 {query}"))
    );
    let body = c
        .get(&url)
        .header("Accept", "text/event-stream")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let mut in_matches = false;
    for line in body.lines() {
        if let Some(ev) = line.strip_prefix("event: ") {
            in_matches = ev.trim() == "matches";
            continue;
        }
        if !in_matches {
            continue;
        }
        let Some(data) = line.strip_prefix("data: ") else { continue };
        let Ok(arr) = serde_json::from_str::<Value>(data) else { continue };
        for m in arr.as_array().unwrap_or(&Vec::new()) {
            if m["type"].as_str() != Some("content") {
                continue;
            }
            let repo = m["repository"].as_str().unwrap_or_default();
            let path = m["path"].as_str().unwrap_or_default();
            if repo.is_empty() || path.is_empty() {
                continue;
            }
            let mut snippet = String::new();
            let mut first_line = 0u64;
            // Older frames use lineMatches; newer ones chunkMatches.
            if let Some(lms) = m["lineMatches"].as_array() {
                for lm in lms.iter().take(2) {
                    if first_line == 0 {
                        first_line = lm["lineNumber"].as_u64().unwrap_or(0) + 1;
                    }
                    snippet.push_str(lm["line"].as_str().unwrap_or_default().trim());
                    snippet.push('\n');
                }
            } else if let Some(cms) = m["chunkMatches"].as_array() {
                for cm in cms.iter().take(2) {
                    if first_line == 0 {
                        first_line = cm["contentStart"]["line"].as_u64().unwrap_or(0) + 1;
                    }
                    snippet.push_str(cm["content"].as_str().unwrap_or_default().trim());
                    snippet.push('\n');
                }
            }
            // repo is e.g. "github.com/tauri-apps/tauri" — link straight to
            // the file on the host itself.
            let url = if repo.starts_with("github.com/") {
                format!("https://{repo}/blob/HEAD/{path}#L{}", first_line.max(1))
            } else {
                format!("https://sourcegraph.com/{repo}/-/blob/{path}")
            };
            out.push(SiteResult {
                kind: "code".into(),
                title: format!("{repo}/{path}"),
                url,
                snippet: snippet.chars().take(300).collect(),
            });
            if out.len() >= 8 {
                return Ok(out);
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- youtube

/// The Android-app UA — YouTube's innertube player answers it without the
/// proof-of-origin token the web client now demands for caption URLs.
const YT_UA: &str = "com.google.android.youtube/20.10.38 (Linux; U; Android 14) gzip";

/// YouTube search via the results page's embedded ytInitialData JSON —
/// structured video results (title/length/channel/views), no key.
async fn youtube_search(query: &str) -> Result<Vec<SiteResult>, String> {
    let c = client(15)?;
    let url = format!("https://www.youtube.com/results?search_query={}", urlencoding(query));
    let html = c.get(&url).send().await.map_err(|e| e.to_string())?.text().await.map_err(|e| e.to_string())?;
    let data = match extract_json_var(&html, "var ytInitialData = ") {
        Some(v) => v,
        None => return engine_site_search("youtube.com", query).await,
    };
    let mut out = Vec::new();
    collect_video_renderers(&data, &mut out);
    if out.is_empty() {
        return engine_site_search("youtube.com", query).await;
    }
    out.truncate(10);
    Ok(out)
}

/// Parse the JSON blob assigned to a JS variable in a page (`var x = {...};`).
fn extract_json_var(html: &str, marker: &str) -> Option<Value> {
    let start = html.find(marker)? + marker.len();
    let rest = &html[start..];
    let end = rest.find(";</script>")?;
    serde_json::from_str(&rest[..end]).ok()
}

fn collect_video_renderers(v: &Value, out: &mut Vec<SiteResult>) {
    if out.len() >= 12 {
        return;
    }
    match v {
        Value::Object(map) => {
            if let Some(vr) = map.get("videoRenderer") {
                if let Some(id) = vr["videoId"].as_str() {
                    let title = vr["title"]["runs"][0]["text"].as_str().unwrap_or("").to_string();
                    let length = vr["lengthText"]["simpleText"].as_str().unwrap_or("");
                    let views = vr["viewCountText"]["simpleText"].as_str().unwrap_or("");
                    let channel = vr["ownerText"]["runs"][0]["text"].as_str().unwrap_or("");
                    let desc: String = vr["detailedMetadataSnippets"][0]["snippetText"]["runs"]
                        .as_array()
                        .map(|runs| runs.iter().filter_map(|r| r["text"].as_str()).collect())
                        .unwrap_or_default();
                    if !title.is_empty() {
                        let mut meta = Vec::new();
                        for m in [length, channel, views] {
                            if !m.is_empty() {
                                meta.push(m);
                            }
                        }
                        out.push(SiteResult {
                            kind: "video".into(),
                            title: if meta.is_empty() { title } else { format!("{title} ({})", meta.join(", ")) },
                            url: format!("https://www.youtube.com/watch?v={id}"),
                            snippet: desc.chars().take(200).collect(),
                        });
                    }
                }
            }
            for val in map.values() {
                collect_video_renderers(val, out);
            }
        }
        Value::Array(arr) => {
            for val in arr {
                collect_video_renderers(val, out);
            }
        }
        _ => {}
    }
}

/// The 11-char video id, when `url` is a YouTube video link in any shape
/// (watch?v=, youtu.be/, /shorts/, /live/, /embed/).
fn youtube_video_id(url: &str) -> Option<String> {
    let u = Url::parse(url).ok()?;
    let host = u.host_str()?.trim_start_matches("www.").trim_start_matches("m.");
    let ok_id = |s: &str| {
        (s.len() == 11 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
            .then(|| s.to_string())
    };
    match host {
        "youtu.be" => ok_id(u.path().trim_matches('/')),
        "youtube.com" | "music.youtube.com" => {
            if u.path() == "/watch" {
                return u.query_pairs().find(|(k, _)| k == "v").and_then(|(_, v)| ok_id(&v));
            }
            let segs: Vec<&str> = u.path().trim_matches('/').split('/').collect();
            if segs.len() == 2 && matches!(segs[0], "shorts" | "live" | "embed" | "v") {
                return ok_id(segs[1]);
            }
            None
        }
        _ => None,
    }
}

/// Video → understanding pipeline: metadata + the full caption transcript
/// (with periodic [mm:ss] markers), via the innertube player API and the
/// caption track it hands out. Works for auto-generated captions too.
async fn fetch_youtube(video_id: &str) -> Result<PageEx, String> {
    let c = reqwest::Client::builder()
        .user_agent(YT_UA)
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let body = serde_json::json!({
        "context": {"client": {
            "clientName": "ANDROID", "clientVersion": "20.10.38",
            "androidSdkVersion": 34, "hl": "en", "gl": "US"
        }},
        "videoId": video_id,
    });
    let v: Value = {
        let t = c
            .post("https://www.youtube.com/youtubei/v1/player?prettyPrint=false")
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| e.to_string())?
            .text()
            .await
            .map_err(|e| e.to_string())?;
        serde_json::from_str(&t).map_err(|e| e.to_string())?
    };
    let status = v["playabilityStatus"]["status"].as_str().unwrap_or("?");
    if status != "OK" {
        let reason = v["playabilityStatus"]["reason"].as_str().unwrap_or(status);
        return Err(format!("视频不可用 (video unavailable): {reason}"));
    }
    let d = &v["videoDetails"];
    let title = d["title"].as_str().unwrap_or("").to_string();
    let author = d["author"].as_str().unwrap_or("");
    let secs: u64 = d["lengthSeconds"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0);
    let views = d["viewCount"].as_str().unwrap_or("?");
    let desc: String = d["shortDescription"].as_str().unwrap_or("").chars().take(600).collect();

    let mut text = format!(
        "视频 (video): {title}\n频道 (channel): {author} · 时长 (length): {}:{:02} · 播放 (views): {views}\n\n简介 (description):\n{desc}\n",
        secs / 60,
        secs % 60
    );

    let tracks = v["captions"]["playerCaptionsTracklistRenderer"]["captionTracks"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    match pick_caption_track(&tracks) {
        Some(track) => {
            let base = track["baseUrl"].as_str().unwrap_or_default();
            let lang = track["languageCode"].as_str().unwrap_or("?");
            let kind = if track["kind"].as_str() == Some("asr") { " 自动生成/auto" } else { "" };
            let xml = c.get(base).send().await.map_err(|e| e.to_string())?.text().await.map_err(|e| e.to_string())?;
            let transcript = timedtext_to_transcript(&xml);
            if transcript.is_empty() {
                text.push_str("\n(字幕轨为空 / caption track came back empty)\n");
            } else {
                text.push_str(&format!("\n—— 字幕转写 (transcript, {lang}{kind}) ——\n{transcript}\n"));
            }
        }
        None => {
            text.push_str("\n(此视频没有字幕轨,无法转写 / no caption track on this video)\n");
        }
    }

    let (text, truncated) = cap(text.trim(), TEXT_CAP);
    Ok(PageEx {
        url: format!("https://www.youtube.com/watch?v={video_id}"),
        kind: "video".into(),
        content_type: "video/youtube".into(),
        title,
        text,
        truncated,
        links: Vec::new(),
        images: Vec::new(),
        bytes: None,
    })
}

// ---------------------------------------------------------------- bilibili

const BILI_UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15";

/// Bilibili in-site video search via the public web-interface API (no key,
/// no cookie — a Referer header is all it wants). Returns structured videos
/// with title / author / duration / view count.
async fn bilibili_search(query: &str) -> Result<Vec<SiteResult>, String> {
    let c = reqwest::Client::builder()
        .user_agent(BILI_UA)
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!(
        "https://api.bilibili.com/x/web-interface/wbi/search/type?search_type=video&keyword={}&page=1",
        urlencoding(query)
    );
    let body = c
        .get(&url)
        .header(reqwest::header::REFERER, "https://www.bilibili.com/")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    if v["code"].as_i64() != Some(0) {
        // Anti-crawl / rate limit → engine snapshot fallback.
        return engine_site_search("bilibili.com", query).await;
    }
    let mut out = Vec::new();
    for r in v["data"]["result"].as_array().unwrap_or(&Vec::new()) {
        let bvid = r["bvid"].as_str().unwrap_or("");
        if bvid.is_empty() {
            continue;
        }
        // Titles come with <em class="keyword"> highlight tags — strip them.
        let title = strip_tags(r["title"].as_str().unwrap_or(""));
        let author = r["author"].as_str().unwrap_or("");
        let dur = r["duration"].as_str().unwrap_or("");
        let plays = r["play"].as_i64().unwrap_or(0);
        let desc = strip_tags(r["description"].as_str().unwrap_or(""));
        let mut meta = Vec::new();
        if !dur.is_empty() {
            meta.push(dur.to_string());
        }
        if !author.is_empty() {
            meta.push(author.to_string());
        }
        if plays > 0 {
            meta.push(format!("{plays} 播放"));
        }
        out.push(SiteResult {
            kind: "video".into(),
            title: if meta.is_empty() { title } else { format!("{title} ({})", meta.join(", ")) },
            url: format!("https://www.bilibili.com/video/{bvid}"),
            snippet: desc.chars().take(200).collect(),
        });
        if out.len() >= 12 {
            break;
        }
    }
    if out.is_empty() {
        return engine_site_search("bilibili.com", query).await;
    }
    Ok(out)
}

/// The BVxxxx id from any Bilibili video URL (bilibili.com/video/BV…, or a
/// b23.tv short link path). None for non-video Bilibili links.
fn bilibili_bvid(url: &str) -> Option<String> {
    let u = Url::parse(url).ok()?;
    let host = u.host_str()?.trim_start_matches("www.").trim_start_matches("m.");
    if host != "bilibili.com" {
        return None;
    }
    for seg in u.path().trim_matches('/').split('/') {
        if (seg.starts_with("BV") || seg.starts_with("bv")) && seg.len() >= 10 {
            return Some(seg.to_string());
        }
    }
    None
}

/// Bilibili video metadata (title / UP / views / likes / description) via the
/// public web-interface/view API.
async fn fetch_bilibili(bvid: &str) -> Result<PageEx, String> {
    let c = reqwest::Client::builder()
        .user_agent(BILI_UA)
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("https://api.bilibili.com/x/web-interface/view?bvid={bvid}");
    let body = c
        .get(&url)
        .header(reqwest::header::REFERER, "https://www.bilibili.com/")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    if v["code"].as_i64() != Some(0) {
        return Err(format!(
            "B站视频不可用 (unavailable): {}",
            v["message"].as_str().unwrap_or("?")
        ));
    }
    let d = &v["data"];
    let title = d["title"].as_str().unwrap_or("").to_string();
    let author = d["owner"]["name"].as_str().unwrap_or("");
    let secs = d["duration"].as_i64().unwrap_or(0);
    let stat = &d["stat"];
    let views = stat["view"].as_i64().unwrap_or(0);
    let likes = stat["like"].as_i64().unwrap_or(0);
    let danmaku = stat["danmaku"].as_i64().unwrap_or(0);
    let desc: String = d["desc"].as_str().unwrap_or("").chars().take(800).collect();
    let text = format!(
        "视频 (video): {title}\nUP 主 (uploader): {author} · 时长 (length): {}:{:02} · 播放 (views): {views} · 点赞 (likes): {likes} · 弹幕 (danmaku): {danmaku}\n\n简介 (description):\n{desc}\n\n（B站视频正文以弹幕/字幕形式存在,需登录才能取字幕;此处提供公开元信息与简介 / public metadata only — captions require login）",
        secs / 60,
        secs % 60
    );
    let (text, truncated) = cap(text.trim(), TEXT_CAP);
    Ok(PageEx {
        url: format!("https://www.bilibili.com/video/{bvid}"),
        kind: "video".into(),
        content_type: "video/bilibili".into(),
        title,
        text,
        truncated,
        links: Vec::new(),
        images: Vec::new(),
        bytes: None,
    })
}

/// Manual captions beat auto-generated; within each class prefer Chinese,
/// then English, then whatever is first.
fn pick_caption_track(tracks: &[Value]) -> Option<&Value> {
    for want_asr in [false, true] {
        let best = tracks
            .iter()
            .filter(|t| (t["kind"].as_str() == Some("asr")) == want_asr)
            .min_by_key(|t| {
                let lang = t["languageCode"].as_str().unwrap_or("");
                if lang.starts_with("zh") {
                    0
                } else if lang.starts_with("en") {
                    1
                } else {
                    2
                }
            });
        if best.is_some() {
            return best;
        }
    }
    None
}

/// timedtext XML (`<p t="ms" d="ms">text</p>`) → readable transcript with a
/// [mm:ss] marker roughly every 45 seconds.
fn timedtext_to_transcript(xml: &str) -> String {
    let mut out = String::new();
    let mut last_mark: i64 = -100_000;
    for chunk in xml.split("<p ").skip(1) {
        let Some(gt) = chunk.find('>') else { continue };
        let attrs = &chunk[..gt];
        if attrs.trim_end().ends_with('/') {
            continue; // self-closing, no text
        }
        let t_ms: i64 = attrs
            .split("t=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let Some(end) = chunk[gt..].find("</p>") else { continue };
        let raw = &chunk[gt + 1..gt + end];
        let mut line = unescape_xml(raw).replace('\n', " ");
        if line.contains('<') {
            line = strip_tags(&line); // asr tracks may carry <s> word tags
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if t_ms - last_mark >= 45_000 {
            let secs = t_ms / 1000;
            out.push_str(&format!("\n[{}:{:02}] ", secs / 60, secs % 60));
            last_mark = t_ms;
        }
        out.push_str(line);
        out.push(' ');
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------- reddit rss

/// Reddit's JSON API is locked down (403 for non-browser clients), but the
/// RSS/Atom endpoints remain open — search, subreddits, and full post
/// comment threads all have one.
async fn reddit_search(query: &str, subreddit: Option<&str>) -> Result<Vec<SiteResult>, String> {
    let c = client(12)?;
    let url = match subreddit {
        Some(sub) => format!(
            "https://www.reddit.com/r/{sub}/search.rss?q={}&restrict_sr=1&limit=10",
            urlencoding(query)
        ),
        None => format!("https://www.reddit.com/search.rss?q={}&limit=10", urlencoding(query)),
    };
    let body = c.get(&url).send().await.map_err(|e| e.to_string())?.text().await.map_err(|e| e.to_string())?;
    let entries = parse_atom(&body);
    if entries.is_empty() {
        return engine_site_search("reddit.com", query).await;
    }
    Ok(entries
        .into_iter()
        .map(|e| SiteResult {
            kind: "post".into(),
            title: e.title,
            url: e.link,
            snippet: e.text.chars().take(240).collect(),
        })
        .collect())
}

struct AtomEntry {
    title: String,
    link: String,
    text: String,
}

/// Minimal Atom parser — enough for Reddit's feeds without an XML dependency.
/// Splits on <entry> blocks and pulls title / link href / content (which is
/// escaped HTML; we unescape it and strip the tags).
fn parse_atom(xml: &str) -> Vec<AtomEntry> {
    let mut out = Vec::new();
    for block in xml.split("<entry>").skip(1) {
        let block = block.split("</entry>").next().unwrap_or("");
        let title = tag_text(block, "title").map(|t| unescape_xml(&t)).unwrap_or_default();
        let link = attr_of(block, "<link", "href").unwrap_or_default();
        let content = tag_text(block, "content")
            .map(|c| strip_tags(&unescape_xml(&c)))
            .unwrap_or_default();
        if title.is_empty() && content.is_empty() {
            continue;
        }
        out.push(AtomEntry { title, link, text: content });
    }
    out
}

fn tag_text(block: &str, tag: &str) -> Option<String> {
    let open = block.find(&format!("<{tag}"))?;
    let rest = &block[open..];
    let start = rest.find('>')? + 1;
    let end = rest.find(&format!("</{tag}>"))?;
    if end <= start {
        return None;
    }
    Some(rest[start..end].to_string())
}

fn attr_of(block: &str, tag_open: &str, attr: &str) -> Option<String> {
    let at = block.find(tag_open)?;
    let rest = &block[at..block[at..].find('>')? + at + 1];
    let key = format!("{attr}=\"");
    let s = rest.find(&key)? + key.len();
    let e = rest[s..].find('"')? + s;
    Some(unescape_xml(&rest[s..e]))
}

fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&#32;", " ")
        .replace("&amp;", "&")
}

/// Drop tags from an HTML fragment, keeping the text with sane spacing.
fn strip_tags(html: &str) -> String {
    let frag = Html::parse_fragment(html);
    let text = frag.root_element().text().collect::<Vec<_>>().join(" ");
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------- fetch

/// Content-type-aware page fetch. HTML becomes readable Markdown (or raw
/// source with `raw=true`), code/JSON/etc. pass through as text, PDFs are
/// text-extracted, binaries return metadata. Links and images found on HTML
/// pages are harvested so the agent can navigate into sub-pages.
#[tauri::command]
pub async fn fetch_page_ex(url: String, raw: Option<bool>) -> Result<PageEx, String> {
    let raw = raw.unwrap_or(false);
    let url = rewrite_url(url.trim());

    // YouTube links go through the video-understanding pipeline:
    // metadata + full caption transcript.
    if let Some(vid) = youtube_video_id(&url) {
        return fetch_youtube(&vid).await;
    }

    // Reddit URLs answer 403 to plain HTTP clients but happily serve RSS.
    if let Some(rss_url) = reddit_rss_url(&url) {
        return fetch_reddit(&rss_url).await;
    }

    // Bilibili video pages hide the content in a JS payload; the public
    // web-interface API returns clean metadata (title / UP / views / desc).
    if let Some(bvid) = bilibili_bvid(&url) {
        return fetch_bilibili(&bvid).await;
    }

    let c = client(20)?;
    let resp = c.get(&url).send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    let final_url = resp.url().to_string();
    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if !status.is_success() {
        return Err(format!("HTTP {status} for {final_url}"));
    }

    let is_html = ctype.contains("html") || ctype.contains("xhtml");
    let is_text = ctype.starts_with("text/")
        || ctype.contains("json")
        || ctype.contains("javascript")
        || ctype.contains("xml")
        || ctype.contains("x-sh")
        || ctype.contains("toml")
        || ctype.contains("yaml")
        || ctype.is_empty();

    if ctype.contains("pdf") {
        let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
        if bytes.len() > PDF_CAP_BYTES {
            return Err(format!("PDF too large ({} MB)", bytes.len() / 1024 / 1024));
        }
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| format!("PDF 解析失败: {e}"))?;
        let (text, truncated) = cap(text.trim(), TEXT_CAP);
        return Ok(PageEx {
            url: final_url,
            kind: "pdf".into(),
            content_type: ctype,
            title: String::new(),
            text,
            truncated,
            links: Vec::new(),
            images: Vec::new(),
            bytes: Some(bytes.len() as u64),
        });
    }

    if !is_html && !is_text {
        // Binary: report metadata; the agent downloads it with web_download.
        let len = resp.content_length();
        return Ok(PageEx {
            url: final_url.clone(),
            kind: "binary".into(),
            content_type: ctype.clone(),
            title: String::new(),
            text: format!(
                "[binary {} — {}] 用 web_download 保存到工作区 (save it with web_download)",
                ctype,
                len.map(|b| format!("{:.1} KB", b as f64 / 1024.0)).unwrap_or_else(|| "size unknown".into())
            ),
            truncated: false,
            links: Vec::new(),
            images: Vec::new(),
            bytes: len,
        });
    }

    let body = resp.text().await.map_err(|e| e.to_string())?;

    if !is_html {
        // Source / data files pass through untouched.
        let (text, truncated) = cap(&body, TEXT_CAP);
        return Ok(PageEx {
            url: final_url,
            kind: "text".into(),
            content_type: ctype,
            title: String::new(),
            text,
            truncated,
            links: Vec::new(),
            images: Vec::new(),
            bytes: None,
        });
    }

    // HTML — harvest links/images from the full DOM first.
    let (links, images, doc_title) = harvest(&body, &final_url);

    if raw {
        let (text, truncated) = cap(&body, TEXT_CAP);
        return Ok(PageEx {
            url: final_url,
            kind: "source".into(),
            content_type: ctype,
            title: doc_title,
            text,
            truncated,
            links,
            images,
            bytes: None,
        });
    }

    // Readability first (clean article), whole-page conversion as fallback.
    let (title, md) = readable_markdown(&body, &final_url).unwrap_or_else(|| {
        let md = htmd::convert(&body).unwrap_or_default();
        (doc_title.clone(), md)
    });
    let md = squeeze_blank_lines(&md);
    let (text, truncated) = cap(md.trim(), TEXT_CAP);
    Ok(PageEx {
        url: final_url,
        kind: "markdown".into(),
        content_type: ctype,
        title: if title.is_empty() { doc_title } else { title },
        text,
        truncated,
        links,
        images,
        bytes: None,
    })
}

/// GitHub blob pages carry the file inside a JSON payload the HTML extractor
/// can't reach — rewrite to raw.githubusercontent.com, which serves the file
/// itself.
fn rewrite_url(url: &str) -> String {
    if let Ok(u) = Url::parse(url) {
        if u.host_str() == Some("github.com") {
            let segs: Vec<&str> = u.path().trim_matches('/').split('/').collect();
            if segs.len() >= 5 && (segs[2] == "blob" || segs[2] == "raw") {
                return format!(
                    "https://raw.githubusercontent.com/{}/{}/{}",
                    segs[0],
                    segs[1],
                    segs[3..].join("/")
                );
            }
        }
    }
    url.to_string()
}

/// reddit.com content URL → its RSS twin (None for non-reddit URLs).
fn reddit_rss_url(url: &str) -> Option<String> {
    let u = Url::parse(url).ok()?;
    let host = u.host_str()?;
    if !(host == "reddit.com" || host.ends_with(".reddit.com")) {
        return None;
    }
    let path = u.path().trim_end_matches('/');
    if path.ends_with(".rss") {
        return Some(url.to_string());
    }
    let q = u.query().map(|q| format!("?{q}")).unwrap_or_default();
    Some(format!("https://www.reddit.com{path}.rss{q}"))
}

async fn fetch_reddit(rss_url: &str) -> Result<PageEx, String> {
    let c = client(12)?;
    let resp = c.get(rss_url).send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!(
            "Reddit HTTP {status} — 被限流了,过一会再试 (rate limited, retry in a minute)"
        ));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let entries = parse_atom(&body);
    if entries.is_empty() {
        return Err("Reddit 返回了空 feed (empty feed)".to_string());
    }
    let title = entries[0].title.clone();
    let mut md = String::new();
    let mut links = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        if i == 0 {
            md.push_str(&format!("# {}\n\n{}\n", e.title, e.text));
        } else {
            md.push_str(&format!("\n---\n**{}**\n{}\n", e.title, e.text));
        }
        if !e.link.is_empty() {
            links.push(LinkRef { url: e.link.clone(), text: e.title.chars().take(60).collect() });
        }
    }
    let (text, truncated) = cap(md.trim(), TEXT_CAP);
    Ok(PageEx {
        url: rss_url.to_string(),
        kind: "markdown".into(),
        content_type: "application/atom+xml".into(),
        title,
        text,
        truncated,
        links,
        images: Vec::new(),
        bytes: None,
    })
}

/// Readability extraction → Markdown. None when the page has no articleish
/// body (dashboards, index pages) — caller falls back to whole-page htmd.
fn readable_markdown(html: &str, url: &str) -> Option<(String, String)> {
    let mut r = dom_smoothie::Readability::new(html, Some(url), None).ok()?;
    let article = r.parse().ok()?;
    let md = htmd::convert(&article.content).ok()?;
    if md.trim().len() < 200 {
        return None; // too thin to trust — use the whole page instead
    }
    Some((article.title, md))
}

/// Collect links (same-host first) + image URLs + <title> from a page.
fn harvest(html: &str, base: &str) -> (Vec<LinkRef>, Vec<String>, String) {
    let doc = Html::parse_document(html);
    let base_url = Url::parse(base).ok();
    let host = base_url.as_ref().and_then(|u| u.host_str().map(str::to_string)).unwrap_or_default();

    let title = Selector::parse("title")
        .ok()
        .and_then(|s| doc.select(&s).next().map(|t| t.text().collect::<String>().trim().to_string()))
        .unwrap_or_default();

    let mut seen = std::collections::HashSet::new();
    let mut internal = Vec::new();
    let mut external = Vec::new();
    if let Ok(a_sel) = Selector::parse("a[href]") {
        for a in doc.select(&a_sel) {
            let Some(href) = a.value().attr("href") else { continue };
            if href.starts_with('#') || href.starts_with("javascript:") || href.starts_with("mailto:") {
                continue;
            }
            let abs = match &base_url {
                Some(b) => match b.join(href) {
                    Ok(u) if u.scheme().starts_with("http") => u.to_string(),
                    _ => continue,
                },
                None => continue,
            };
            let clean = abs.split('#').next().unwrap_or(&abs).to_string();
            if !seen.insert(clean.clone()) {
                continue;
            }
            let text: String = a.text().collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
            let link = LinkRef { text: text.chars().take(70).collect(), url: clean.clone() };
            if clean.contains(&host) && !host.is_empty() {
                internal.push(link);
            } else {
                external.push(link);
            }
        }
    }
    internal.extend(external);
    internal.truncate(40);

    let mut images = Vec::new();
    if let Ok(img_sel) = Selector::parse("img[src]") {
        let mut seen_img = std::collections::HashSet::new();
        for img in doc.select(&img_sel) {
            let Some(src) = img.value().attr("src") else { continue };
            if src.starts_with("data:") {
                continue;
            }
            if let Some(b) = &base_url {
                if let Ok(u) = b.join(src) {
                    let s = u.to_string();
                    if seen_img.insert(s.clone()) {
                        images.push(s);
                        if images.len() >= 15 {
                            break;
                        }
                    }
                }
            }
        }
    }

    (internal, images, title)
}

fn squeeze_blank_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blanks = 0;
    for line in s.lines() {
        if line.trim().is_empty() {
            blanks += 1;
            if blanks > 1 {
                continue;
            }
        } else {
            blanks = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn urlencoding(s: &str) -> String {
    use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
    utf8_percent_encode(s, NON_ALPHANUMERIC).to_string()
}

// -------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_github_blob_urls() {
        assert_eq!(
            rewrite_url("https://github.com/tauri-apps/tauri/blob/dev/Cargo.toml"),
            "https://raw.githubusercontent.com/tauri-apps/tauri/dev/Cargo.toml"
        );
        assert_eq!(rewrite_url("https://github.com/tauri-apps/tauri"), "https://github.com/tauri-apps/tauri");
        assert_eq!(rewrite_url("https://example.com/blob/x"), "https://example.com/blob/x");
    }

    #[test]
    fn reddit_urls_get_rss_twins() {
        assert_eq!(
            reddit_rss_url("https://www.reddit.com/r/rust/comments/abc/some_post/").as_deref(),
            Some("https://www.reddit.com/r/rust/comments/abc/some_post.rss")
        );
        assert_eq!(
            reddit_rss_url("https://reddit.com/search?q=tauri").as_deref(),
            Some("https://www.reddit.com/search.rss?q=tauri")
        );
        assert_eq!(reddit_rss_url("https://example.com/r/rust"), None);
    }

    #[test]
    fn atom_parses_entries() {
        let xml = r#"<feed><entry><title>Hello &amp; hi</title><link href="https://r.it/x"/><content type="html">&lt;p&gt;Body &lt;b&gt;text&lt;/b&gt;&lt;/p&gt;</content></entry></feed>"#;
        let es = parse_atom(xml);
        assert_eq!(es.len(), 1);
        assert_eq!(es[0].title, "Hello & hi");
        assert_eq!(es[0].link, "https://r.it/x");
        assert_eq!(es[0].text, "Body text");
    }

    #[test]
    fn youtube_ids_parse_from_every_url_shape() {
        for (u, want) in [
            ("https://www.youtube.com/watch?v=jNQXAC9IVRw", Some("jNQXAC9IVRw")),
            ("https://youtu.be/jNQXAC9IVRw?t=10", Some("jNQXAC9IVRw")),
            ("https://m.youtube.com/watch?v=jNQXAC9IVRw&list=x", Some("jNQXAC9IVRw")),
            ("https://www.youtube.com/shorts/jNQXAC9IVRw", Some("jNQXAC9IVRw")),
            ("https://www.youtube.com/embed/jNQXAC9IVRw", Some("jNQXAC9IVRw")),
            ("https://www.youtube.com/@SomeChannel", None),
            ("https://example.com/watch?v=jNQXAC9IVRw", None),
        ] {
            assert_eq!(youtube_video_id(u).as_deref(), want, "url: {u}");
        }
    }

    #[test]
    fn bilibili_bvid_parses() {
        assert_eq!(
            bilibili_bvid("https://www.bilibili.com/video/BV1xx411c7mD").as_deref(),
            Some("BV1xx411c7mD")
        );
        assert_eq!(
            bilibili_bvid("https://www.bilibili.com/video/BV1xx411c7mD?p=2&t=30").as_deref(),
            Some("BV1xx411c7mD")
        );
        assert_eq!(bilibili_bvid("https://www.bilibili.com/"), None);
        assert_eq!(bilibili_bvid("https://example.com/video/BV1xx411c7mD"), None);
    }

    #[test]
    fn timedtext_xml_becomes_marked_transcript() {
        let xml = r#"<?xml version="1.0"?><timedtext format="3"><body>
<p t="1200" d="2000">hello
world</p>
<p t="3300" d="1000">it&#39;s me</p>
<p t="50000" d="1000"><s>tagged</s> words</p>
<p t="60000" d="0" />
</body></timedtext>"#;
        let t = timedtext_to_transcript(xml);
        assert!(t.starts_with("[0:01] hello world it's me"), "{t}");
        assert!(t.contains("[0:50] tagged words"), "{t}");
        assert!(!t.contains("<s>"), "{t}");
    }

    // ---- real-network tests: cargo test -p chaty webx -- --ignored ----

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    #[ignore]
    fn real_github_site_search() {
        let r = rt().block_on(site_search("github.com".into(), "tauri set_zoom".into())).unwrap();
        assert!(!r.is_empty(), "github search returned nothing");
        let kinds: std::collections::HashSet<_> = r.iter().map(|x| x.kind.as_str()).collect();
        println!("github kinds: {kinds:?} ({} results)", r.len());
        for x in r.iter().take(6) {
            println!("  [{}] {} — {}", x.kind, x.title, x.url);
        }
        assert!(kinds.contains("repo") || kinds.contains("issue") || kinds.contains("code"));
    }

    #[test]
    #[ignore]
    fn real_reddit_site_search() {
        let r = rt().block_on(site_search("reddit.com".into(), "tauri".into())).unwrap();
        assert!(!r.is_empty(), "reddit search returned nothing");
        println!("reddit: {} results, first: {} — {}", r.len(), r[0].title, r[0].url);
        assert!(r[0].url.contains("reddit.com"));
    }

    #[test]
    #[ignore]
    fn real_x_site_search() {
        let r = rt().block_on(site_search("x.com".into(), "tauri".into())).unwrap();
        println!("x.com: {} results", r.len());
        for x in r.iter().take(3) {
            println!("  {} — {}", x.title, x.url);
        }
        assert!(!r.is_empty(), "x.com snapshot search returned nothing");
    }

    #[test]
    #[ignore]
    fn real_bilibili_site_search() {
        let r = rt().block_on(site_search("bilibili.com".into(), "rust 教程".into())).unwrap();
        assert!(!r.is_empty(), "bilibili search returned nothing");
        println!("bilibili: {} results", r.len());
        for x in r.iter().take(4) {
            println!("  [{}] {} — {}", x.kind, x.title, x.url);
        }
        assert!(r.iter().any(|x| x.url.contains("/video/BV")), "no BV video results");
    }

    #[test]
    #[ignore]
    fn real_fetch_bilibili_video() {
        // "字幕君交流场所" — a stable, high-view public video.
        let p = rt()
            .block_on(fetch_page_ex("https://www.bilibili.com/video/BV1xx411c7mD".into(), None))
            .unwrap();
        println!("bili kind={} title={} len={}", p.kind, p.title, p.text.len());
        println!("{}", p.text.chars().take(400).collect::<String>());
        assert_eq!(p.kind, "video");
        assert!(!p.title.is_empty());
        assert!(p.text.contains("播放") && p.text.contains("UP 主"), "metadata missing: {}", p.text);
    }

    #[test]
    #[ignore]
    fn real_fetch_article_markdown() {
        let p = rt()
            .block_on(fetch_page_ex(
                "https://en.wikipedia.org/wiki/Rust_(programming_language)".into(),
                None,
            ))
            .unwrap();
        println!("article kind={} title={} len={} links={}", p.kind, p.title, p.text.len(), p.links.len());
        assert_eq!(p.kind, "markdown");
        assert!(p.text.len() > 2000);
        assert!(p.text.contains("Rust"));
        assert!(!p.links.is_empty(), "no links harvested");
    }

    #[test]
    #[ignore]
    fn real_fetch_github_blob_as_source() {
        let p = rt()
            .block_on(fetch_page_ex(
                "https://github.com/tauri-apps/tauri/blob/dev/Cargo.toml".into(),
                None,
            ))
            .unwrap();
        println!("blob kind={} ct={} len={}", p.kind, p.content_type, p.text.len());
        assert_eq!(p.kind, "text");
        assert!(p.text.contains("[workspace]") || p.text.contains("[package]"));
    }

    #[test]
    #[ignore]
    fn real_fetch_raw_html_source() {
        let p = rt()
            .block_on(fetch_page_ex("https://example.com".into(), Some(true)))
            .unwrap();
        assert_eq!(p.kind, "source");
        assert!(p.text.contains("<html") || p.text.contains("<!doctype"));
    }

    #[test]
    #[ignore]
    fn real_fetch_reddit_post() {
        let r = rt().block_on(site_search("reddit.com".into(), "tauri".into())).unwrap();
        let post = r.iter().find(|x| x.url.contains("/comments/")).expect("no post in results");
        match rt().block_on(fetch_page_ex(post.url.clone(), None)) {
            Ok(p) => {
                println!("reddit post title={} len={} links={}", p.title, p.text.len(), p.links.len());
                assert_eq!(p.kind, "markdown");
                assert!(!p.text.is_empty());
            }
            // Back-to-back test runs trip Reddit's rate limit; that's the
            // environment, not a logic failure — the error must say so though.
            Err(e) if e.contains("429") => println!("SKIP: rate limited ({e})"),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    #[ignore]
    fn real_youtube_site_search() {
        let r = rt().block_on(site_search("youtube.com".into(), "rust borrow checker explained".into())).unwrap();
        assert!(!r.is_empty(), "youtube search returned nothing");
        println!("youtube: {} results", r.len());
        for x in r.iter().take(4) {
            println!("  [{}] {} — {}", x.kind, x.title, x.url);
        }
        assert!(r.iter().any(|x| x.kind == "video" && x.url.contains("watch?v=")), "no structured video results");
    }

    #[test]
    #[ignore]
    fn real_fetch_youtube_transcript() {
        // "Me at the zoo" — the first YouTube video; manual English captions.
        let p = rt()
            .block_on(fetch_page_ex("https://www.youtube.com/watch?v=jNQXAC9IVRw".into(), None))
            .unwrap();
        println!("video title={} len={}", p.title, p.text.len());
        println!("{}", p.text.chars().take(600).collect::<String>());
        assert_eq!(p.kind, "video");
        assert!(p.title.to_lowercase().contains("zoo"), "{}", p.title);
        assert!(p.text.contains("字幕转写"), "no transcript section: {}", p.text);
        assert!(p.text.to_lowercase().contains("elephants"), "transcript content missing");
        assert!(p.text.contains("[0:"), "no time markers");
    }

    #[test]
    #[ignore]
    fn real_youtube_search_then_transcribe() {
        // The full pipeline the agent will run: in-site search → pick a video
        // → fetch → get a usable transcript.
        let r = rt().block_on(site_search("youtube.com".into(), "rust lifetimes tutorial".into())).unwrap();
        let vid = r.iter().find(|x| x.kind == "video").expect("no video results");
        println!("picked: {} — {}", vid.title, vid.url);
        let p = rt().block_on(fetch_page_ex(vid.url.clone(), None)).unwrap();
        println!("kind={} title={} text[..300]={}", p.kind, p.title, p.text.chars().take(300).collect::<String>());
        assert_eq!(p.kind, "video");
        // Most tutorials have captions (auto-generated at minimum); accept the
        // honest no-caption note as a pass only if the metadata came through.
        assert!(!p.title.is_empty());
        assert!(p.text.contains("字幕转写") || p.text.contains("没有字幕轨"), "{}", p.text);
    }

    #[test]
    #[ignore]
    fn real_fetch_binary_metadata() {
        // GitHub org avatar: a stable, always-there PNG.
        let p = rt()
            .block_on(fetch_page_ex("https://avatars.githubusercontent.com/u/54212428".into(), None))
            .unwrap();
        println!("binary ct={} bytes={:?}", p.content_type, p.bytes);
        assert_eq!(p.kind, "binary");
        assert!(p.content_type.contains("image") || p.content_type.contains("octet"));
    }
}
