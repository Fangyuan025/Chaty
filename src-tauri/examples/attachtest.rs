//! Headless check for attachment text extraction.
//!   cargo run --example attachtest -- <file>

use anyhow::Context;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).context("usage: attachtest <file>")?;
    let a = chaty_lib::attach::read_attachment(path)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;
    eprintln!(
        "name={} kind={} chars={} truncated={}\n--- preview ---",
        a.name, a.kind, a.chars, a.truncated
    );
    println!("{}", a.text.chars().take(500).collect::<String>());
    Ok(())
}
