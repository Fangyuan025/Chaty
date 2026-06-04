//! Headless check for the web search + page-fetch backend.
//!   cargo run --example searchtest -- "your query"

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let q = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "勾股定理 历史".to_string());

    if q.starts_with("http") {
        let page = chaty_lib::search::fetch_url(q)
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        eprintln!("title: {}\nurl: {}\n({} chars)\n", page.title, page.url, page.text.chars().count());
        println!("{}", page.text.chars().take(400).collect::<String>());
        return Ok(());
    }

    eprintln!("query: {q}\n");
    let res = chaty_lib::search::web_research(q)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    eprintln!("--- {} results ---", res.results.len());
    for (i, r) in res.results.iter().enumerate() {
        eprintln!("[{}] {} | {}", i + 1, r.title, r.url);
    }

    eprintln!("\n--- {} pages fetched ---", res.pages.len());
    for (i, p) in res.pages.iter().enumerate() {
        println!(
            "\n[{}] {} ({} chars)\n{}\n",
            i + 1,
            p.title,
            p.text.chars().count(),
            p.text.chars().take(300).collect::<String>()
        );
    }
    Ok(())
}
