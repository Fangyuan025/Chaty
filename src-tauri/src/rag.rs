//! Local RAG (retrieval-augmented generation) — hushdoc-style, fully offline.
//!
//! Pipeline (mirrors hushdoc's recipe, adapted to the Rust/llama.cpp stack):
//!   ingest:  extract text (pdf/txt/md/code) → paragraph-aware chunking
//!            (~800 chars, 120 overlap) → bge-m3 embeddings (llama.cpp,
//!            GPU-accelerated, L2-normalized) → SQLite (vectors as f32 blobs)
//!   search:  dense cosine top-N  +  BM25 top-N (ASCII words + CJK bigrams)
//!            → reciprocal-rank fusion → MMR diversification → neighbor-chunk
//!            expansion → top-k passages with provenance.
//!
//! bge-m3 is multilingual (zh+en), 1024-d, ~730 MB at Q8_0 — downloaded once
//! into app-data/rag/. The embedder runs on its own worker thread with a
//! persistent embeddings context, independent of the chat model.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::Mutex;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use rusqlite::{params, Connection};
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::Manager;

const EMBED_FILE: &str = "bge-m3-Q8_0.gguf";
const EMBED_URLS: &[&str] = &[
    "https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf",
    "https://huggingface.co/lm-kit/bge-m3-gguf/resolve/main/bge-m3-Q8_0.gguf",
];
/// Refuse to index more text than this rather than dying halfway through it.
/// Roughly ten thousand chunks — an hour of embedding on its own, and the point
/// at which the memory the ingest needs stops being predictable. A clear message
/// naming the size beats a process that disappears.
const MAX_INDEX_CHARS: usize = 8_000_000;
const CHUNK_CHARS: usize = 800;
const CHUNK_OVERLAP: usize = 120;
/// Candidates pulled per retriever before fusion.
const RETRIEVE_N: usize = 24;
/// Survivors after RRF, before MMR.
const FUSED_N: usize = 12;

// ---------------------------------------------------------------------------
// Embedder: persistent llama.cpp embeddings worker
// ---------------------------------------------------------------------------

enum EmbedJob {
    Embed {
        texts: Vec<String>,
        reply: Sender<Result<Vec<Vec<f32>>, String>>,
    },
}

struct Embedder {
    tx: Sender<EmbedJob>,
    worker: Option<std::thread::JoinHandle<()>>,
}

static EMBEDDER: Mutex<Option<Embedder>> = Mutex::new(None);

/// Unload the cached embedding model (bge-m3, ~730 MB), freeing its memory. The
/// worker drops the model + context when its channel closes; we join so the
/// memory is actually back by the time this returns. Re-loads lazily on next use.
pub fn embed_unload() {
    if let Some(e) = EMBEDDER.lock().unwrap().take() {
        let Embedder { tx, worker } = e;
        drop(tx); // closes the channel → the worker exits and drops the model
        if let Some(w) = worker {
            let _ = w.join();
        }
    }
}

fn embed_model_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("rag")
        .join(EMBED_FILE))
}

/// How many of the embedder's layers may go on the GPU.
///
/// This load happens with a chat model ALREADY resident, which is the one case
/// where "offload everything" is certain to be wrong: a card with room for a 9B
/// is not a card with room for a 9B and 730 MB more. And an over-ask is not an
/// allocation error to back off from — the Vulkan driver takes the process down
/// with it (issue #9), which is how an import could end with the window
/// vanishing and nothing at all in the log (issue #13). Sizing is therefore the
/// chat model's: against what is FREE, under any cap a previous crash left.
///
/// The same rule on every platform, Apple Silicon included: `fit_layers`
/// against what is free right now. On a Mac "free" has to satisfy both Metal's
/// working set and the machine's memory, because the GPU and CPU share one pool
/// and an MLX chat model lives in a different process from this one.
fn embedder_gpu_layers(backend: &llama_cpp_2::llama_backend::LlamaBackend, path: &str) -> i32 {
    let want = crate::gpu::detect_gpu().and_then(|g| {
        let file = std::fs::metadata(path).map(|m| m.len()).ok()?;
        let n_layer = crate::inference::llama::probe_n_layer(backend, path)?;
        Some(crate::gpu::fit_layers(file, n_layer, crate::gpu::embed_free_bytes(g.vram_mb)))
    });
    embedder_layers(want, crate::inference::llama::gpu_layer_cap())
}

/// The most tokens one chunk may hand the embedder.
///
/// bge-m3 is an ENCODER, and llama.cpp cannot split one sequence across
/// micro-batches for an encoder — so a chunk longer than the context's
/// micro-batch is not an error it returns. It is `GGML_ASSERT(n_ubatch >=
/// n_tokens)` and `abort()`: the whole app gone, nothing raised, nothing in the
/// log. That was issue #13. The micro-batch defaulted to 512 while chunks were
/// cut at 1020, and 800 characters of Chinese run far past 512 tokens — on the
/// reported book 283 of its 377 chunks did (median 652, largest 882), so the
/// first batch killed the app every time, while English, at roughly 200 tokens
/// a chunk, never came near the line. Hence whichever is smaller: the window
/// (less room for special tokens) or the micro-batch the context really got.
fn embed_token_limit(n_ctx: u32, n_ubatch: u32) -> usize {
    (n_ctx as usize).saturating_sub(4).min(n_ubatch as usize).max(1)
}

/// The decision itself, separated from the machinery needed to ask the
/// questions: what VRAM allows, and what a previous crash allows.
///
/// `None` means the question could not be answered — no GPU, or a file whose
/// layer count would not read. Neither is a reason to guess upward with a chat
/// model already on the card; the CPU is slower and survives.
fn embedder_layers(want: Option<i32>, cap: Option<i32>) -> i32 {
    let want = want.unwrap_or(0);
    cap.map_or(want, |c| want.min(c))
}

/// Start the embedding worker. `layers` forces an offload (tests compare CPU
/// against GPU); `None` lets `embedder_gpu_layers` decide, as the app does.
fn embedder_start(model_path: &PathBuf, layers: Option<i32>) -> Result<Embedder, String> {
    let path = model_path.to_string_lossy().to_string();
    let (tx, rx) = std::sync::mpsc::channel::<EmbedJob>();
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    let worker = std::thread::Builder::new()
        .name("chaty-embed".into())
        .spawn(move || {
            let backend = match crate::inference::llama_backend_pub() {
                Ok(b) => b,
                Err(e) => {
                    let _ = init_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            // On the GPU everywhere, sized by the rule the chat model uses rather
            // than by asking for all of it — it loads with a chat model already
            // resident, the one case where "everything" is certainly wrong.
            let layers = layers.unwrap_or_else(|| embedder_gpu_layers(backend, &path));
            let params = LlamaModelParams::default().with_n_gpu_layers(layers.max(0) as u32);
            // macOS loads through malloc, exactly as the chat model does. What
            // kept this embedder on the CPU was an mmap'd load on Metal leaving
            // its pages wired after unload, never returned to the kernel; malloc
            // is freed synchronously, and residency sets are off app-wide
            // (GGML_METAL_NO_RESIDENCY). Measured before this moved: see
            // `rag_embed_gpu_probe`.
            #[cfg(target_os = "macos")]
            let params = params.with_use_mmap(false);
            // A death in here is remembered: the same guard the chat model arms
            // (Windows, where drivers take the process down), so the next
            // attempt asks for less instead of repeating this one.
            let _crash_guard = crate::inference::llama::LoadGuard::arm(layers);
            let model = match LlamaModel::load_from_file(backend, &path, &params) {
                Ok(m) => m,
                Err(e) => {
                    let _ = init_tx.send(Err(format!("加载嵌入模型失败 (failed to load embedding model): {e:#}")));
                    return;
                }
            };
            let n_ctx = 1024u32;
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                // A whole chunk in one micro-batch — see `embed_token_limit`.
                .with_n_batch(n_ctx)
                .with_n_ubatch(n_ctx)
                .with_embeddings(true)
                .with_pooling_type(LlamaPoolingType::Mean)
                .with_n_threads(crate::gpu::cpu_worker_threads() as i32);
            let mut ctx = match model.new_context(backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = init_tx.send(Err(format!("创建嵌入上下文失败 (failed to create embedding context): {e:#}")));
                    return;
                }
            };
            let _ = init_tx.send(Ok(()));
            // Read back rather than assumed: whatever the backend actually
            // granted is the number the assertion checks against.
            let limit = embed_token_limit(n_ctx, ctx.n_ubatch());

            while let Ok(job) = rx.recv() {
                match job {
                    EmbedJob::Embed { texts, reply } => {
                        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
                        let mut failed: Option<String> = None;
                        for text in &texts {
                            let tokens = match model.str_to_token(text, AddBos::Always) {
                                Ok(t) => t,
                                Err(e) => {
                                    failed = Some(format!("tokenize failed: {e}"));
                                    break;
                                }
                            };
                            let take = tokens.len().min(limit);
                            let mut batch = LlamaBatch::new(take.max(1), 1);
                            let mut add_err = None;
                            for (i, tok) in tokens[..take].iter().enumerate() {
                                if let Err(e) = batch.add(*tok, i as i32, &[0], true) {
                                    add_err = Some(e.to_string());
                                    break;
                                }
                            }
                            if let Some(e) = add_err {
                                failed = Some(e);
                                break;
                            }
                            ctx.clear_kv_cache();
                            if let Err(e) = ctx.decode(&mut batch) {
                                failed = Some(format!("embed decode failed: {e}"));
                                break;
                            }
                            match ctx.embeddings_seq_ith(0) {
                                Ok(v) => {
                                    let mut v = v.to_vec();
                                    l2_normalize(&mut v);
                                    out.push(v);
                                }
                                Err(e) => {
                                    failed = Some(format!("embeddings unavailable: {e}"));
                                    break;
                                }
                            }
                        }
                        let _ = reply.send(match failed {
                            Some(e) => Err(e),
                            None => Ok(out),
                        });
                    }
                }
            }
        })
        .map_err(|e| e.to_string())?;

    match init_rx.recv() {
        Ok(Ok(())) => Ok(Embedder { tx, worker: Some(worker) }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("嵌入线程启动失败 (embedder thread failed to start)".into()),
    }
}

/// Embed a batch of texts, lazily starting the worker on first use.
fn embed(app: &tauri::AppHandle, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
    let model_path = embed_model_path(app)?;
    if !model_path.exists() {
        return Err("RAG_MODEL_MISSING".into());
    }
    let mut guard = EMBEDDER.lock().unwrap();
    if guard.is_none() {
        *guard = Some(embedder_start(&model_path, None)?);
    }
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    guard
        .as_ref()
        .unwrap()
        .tx
        .send(EmbedJob::Embed { texts, reply: reply_tx })
        .map_err(|_| "嵌入线程已退出 (embedder exited)".to_string())?;
    drop(guard); // don't hold the lock while embedding
    reply_rx
        .recv()
        .map_err(|_| "嵌入线程已退出 (embedder exited)".to_string())?
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Storage (SQLite): documents + chunks with embedded vectors
// ---------------------------------------------------------------------------

static DB: Mutex<Option<Connection>> = Mutex::new(None);

fn with_db<T>(
    app: &tauri::AppHandle,
    f: impl FnOnce(&Connection) -> Result<T, String>,
) -> Result<T, String> {
    let mut guard = DB.lock().unwrap();
    if guard.is_none() {
        let dir = app
            .path()
            .app_data_dir()
            .map_err(|e| e.to_string())?
            .join("rag");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let conn = Connection::open(dir.join("rag.db")).map_err(|e| e.to_string())?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS docs(
               id INTEGER PRIMARY KEY,
               name TEXT NOT NULL,
               path TEXT,
               chunks INTEGER NOT NULL DEFAULT 0,
               created_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS chunks(
               id INTEGER PRIMARY KEY,
               doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
               seq INTEGER NOT NULL,
               text TEXT NOT NULL,
               embedding BLOB NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, seq);",
        )
        .map_err(|e| e.to_string())?;
        // Migration: per-doc search scope (errors = column already exists).
        let _ = conn.execute(
            "ALTER TABLE docs ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1",
            [],
        );
        let dropped = repair_incomplete_docs(&conn);
        if dropped > 0 {
            crate::errlog::append_error(
                "kb-repair",
                &format!(
                    "移除了 {dropped} 个索引未完成的文档(需要重新导入) \
                     (removed {dropped} document(s) whose indexing never \
                     finished — re-import them)"
                ),
            );
        }
        *guard = Some(conn);
    }
    f(guard.as_ref().unwrap())
}

/// Take out documents whose indexing never finished, and report how many.
///
/// A document is complete only when its recorded count matches the chunks
/// actually stored for it; anything else is wreckage — the process died
/// mid-embed, or an error left the row behind. Such a document is worse than
/// a missing one: it sits in the list looking ready, gets ticked, and then
/// contributes nothing to the answer, which is how issue #13 read from the
/// user's side ("已经添加进去,但…模型不用知识库回答"). Run on open, so a
/// machine that already carries one is repaired by updating rather than by
/// being told to go and delete it by hand.
fn repair_incomplete_docs(conn: &Connection) -> usize {
    let dropped = conn
        .execute(
            // `chunks = 0` is the ingest's own "still indexing" state, so a
            // row still wearing it belongs to a run that is no longer here —
            // a document with nothing in it is never a legitimate result
            // (an empty extraction is refused long before any row is written).
            "DELETE FROM docs WHERE chunks < 1 OR chunks <> (
               SELECT COUNT(*) FROM chunks WHERE chunks.doc_id = docs.id
             )",
            [],
        )
        .unwrap_or(0);
    // Foreign keys are not enforced on this connection, so the cascade is done
    // by hand — the same sweep the ingest path already does before an insert.
    let _ = conn.execute("DELETE FROM chunks WHERE doc_id NOT IN (SELECT id FROM docs)", []);
    dropped
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Text extraction + chunking
// ---------------------------------------------------------------------------

fn extract_text(path: &str) -> Result<String, String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "pdf" => extract_pdf(path),
        "docx" => extract_docx(path),
        "xlsx" => extract_xlsx(path),
        "pptx" => extract_pptx(path),
        _ => {
            // Text-ish files (code, markup, config, …): decode as UTF-8 with a
            // GBK fallback. Any text-based extension just works here.
            let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
            Ok(match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(e) => {
                    let (s, _, _) = encoding_rs::GBK.decode(e.as_bytes());
                    s.into_owned()
                }
            })
        }
    }
}

/// Extract visible text from a .pptx: every `ppt/slides/slideN.xml` part,
/// tags stripped, one block per slide.
pub(crate) fn extract_pptx(path: &str) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("PPTX 解析失败 (not a valid .pptx): {e}"))?;
    let mut slides: Vec<(usize, String)> = Vec::new();
    for i in 0..zip.len() {
        let Ok(mut entry) = zip.by_index(i) else { continue };
        let name = entry.name().to_string();
        let Some(num) = name
            .strip_prefix("ppt/slides/slide")
            .and_then(|r| r.strip_suffix(".xml"))
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let mut xml = String::new();
        use std::io::Read as _;
        if entry.read_to_string(&mut xml).is_err() {
            continue;
        }
        // <a:t>text runs</a:t> hold the visible text; join runs with spaces.
        let mut text = String::new();
        let mut rest = xml.as_str();
        while let Some(open) = rest.find("<a:t>") {
            rest = &rest[open + 5..];
            if let Some(close) = rest.find("</a:t>") {
                text.push_str(&rest[..close]);
                text.push(' ');
                rest = &rest[close + 6..];
            } else {
                break;
            }
        }
        if !text.trim().is_empty() {
            slides.push((num, text.trim().to_string()));
        }
    }
    slides.sort_by_key(|(n, _)| *n);
    let out = slides
        .into_iter()
        .map(|(n, t)| format!("[幻灯片 {n}] {t}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    if out.is_empty() {
        return Err("没有从演示文稿中解析到文本 (no text found in the deck)".into());
    }
    Ok(out)
}

/// Run a parser that may assert instead of erroring. `None` on panic.
fn caught<T>(f: impl FnOnce() -> T) -> Option<T> {
    crate::errlog::handled_panic(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok()
    })
}

/// Extract text from a PDF, surviving a parser that panics on it.
///
/// `pdf-extract` asserts rather than errors on encodings it has not
/// implemented — `assert!(name == "Identity-H")` fires on the CMap most
/// CJK-authored PDFs use, and it took down two of the owner's own textbooks.
/// A panic there is not a fault to report; it is a file we cannot fully read.
///
/// So it is caught, twice over. First the whole document, which is the fast
/// path, byte-for-byte what this used to do, and the one that knows how to
/// decrypt. If that asserts, the pages are read one at a time and every page
/// that parses is kept: a textbook whose front matter uses an exotic font
/// still gives up its lessons, and the reader is told how much was skipped
/// rather than left to wonder.
pub(crate) fn extract_pdf(path: &str) -> Result<String, String> {
    let p = path.to_string();
    if let Some(whole) = caught(move || pdf_extract::extract_text(&p)) {
        return whole
            .map_err(|e| format!("PDF 解析失败 (PDF extraction failed): {e}"))
            .and_then(has_text);
    }
    let p = path.to_string();
    let doc = caught(move || pdf_extract::Document::load(&p))
        .and_then(|r| r.ok())
        .ok_or_else(pdf_unreadable)?;
    pdf_by_page(&doc)
}

/// The same, for a PDF already in memory (a download).
pub(crate) fn extract_pdf_bytes(bytes: &[u8]) -> Result<String, String> {
    if let Some(whole) = caught(|| pdf_extract::extract_text_from_mem(bytes)) {
        return whole
            .map_err(|e| format!("PDF 解析失败 (PDF extraction failed): {e}"))
            .and_then(has_text);
    }
    let doc = caught(|| pdf_extract::Document::load_mem(bytes))
        .and_then(|r| r.ok())
        .ok_or_else(pdf_unreadable)?;
    pdf_by_page(&doc)
}

/// A PDF that parses fine and says nothing is a scan: pages of pictures with
/// no text layer under them. Silently handing back an empty string left the
/// model staring at a blank document with no idea why — 199 of the 200 pages
/// in the owner's grammar manual are exactly this.
fn has_text(t: String) -> Result<String, String> {
    if t.chars().filter(|c| !c.is_whitespace()).count() >= 16 {
        return Ok(t);
    }
    Err(no_text_layer())
}

/// What a PDF with no text layer is reported as — also how the knowledge base
/// recognises one, to read its pages instead.
fn no_text_layer() -> String {
    "PDF 里没有可提取的文字,多半是扫描件(整页都是图,没有文本层) \
     (no extractable text in this PDF — it is almost certainly a scan: \
     pages of images with no text layer)"
        .to_string()
}

fn pdf_unreadable() -> String {
    "PDF 解析失败:这个文件用了解析器不支持的字体编码      (PDF extraction failed: this file uses a font encoding the extractor does not support)"
        .to_string()
}

/// Page by page, keeping what parses.
fn pdf_by_page(doc: &pdf_extract::Document) -> Result<String, String> {
    let pages: Vec<u32> = doc.get_pages().keys().copied().collect();
    let total = pages.len();
    let mut out = String::new();
    let mut skipped = 0usize;
    for n in pages {
        let page = caught(|| {
            let mut s = String::new();
            {
                let mut dev = pdf_extract::PlainTextOutput::new(&mut s);
                pdf_extract::output_doc_page(doc, &mut dev, n).ok()?;
            }
            Some(s)
        })
        .flatten();
        match page {
            Some(s) => out.push_str(&s),
            None => skipped += 1,
        }
    }
    if total == 0 || skipped == total {
        return Err(pdf_unreadable());
    }
    // Judged on what the pages actually yielded, before the note about the
    // ones that failed is added to it.
    out = has_text(out)?;
    if skipped > 0 {
        let note = format!(
            "[已跳过 {skipped}/{total} 页:字体编码不受支持 \
             — skipped {skipped} of {total} pages: unsupported font encoding]"
        );
        // In front when most of the book did not come through, so a reader
        // cannot mistake a scrap for the whole thing.
        out = if skipped * 2 > total {
            format!("{note}\n\n{out}")
        } else {
            format!("{out}\n\n{note}\n")
        };
    }
    Ok(out)
}

/// Extract visible text from a .docx (OOXML: a zip whose `word/document.xml`
/// holds the body). Paragraph/line/tab tags become whitespace; all other tags
/// are stripped and the basic XML entities decoded.
pub(crate) fn extract_docx(path: &str) -> Result<String, String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("DOCX 解析失败 (not a valid .docx): {e}"))?;
    let mut xml = String::new();
    zip.by_name("word/document.xml")
        .map_err(|_| "DOCX 缺少 word/document.xml (corrupt .docx)".to_string())?
        .read_to_string(&mut xml)
        .map_err(|e| e.to_string())?;

    // Turn structural tags into whitespace before stripping the rest.
    let xml = xml
        .replace("</w:p>", "\n\n")
        .replace("<w:br/>", "\n")
        .replace("<w:br />", "\n")
        .replace("<w:tab/>", "\t")
        .replace("<w:tab />", "\t");
    let mut out = String::with_capacity(xml.len());
    let mut in_tag = false;
    for c in xml.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    Ok(out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'"))
}

/// Extract a .xlsx workbook as tab-separated rows (one block per sheet). calamine
/// resolves shared strings, numbers and dates, so the result reads like a CSV.
pub(crate) fn extract_xlsx(path: &str) -> Result<String, String> {
    use calamine::{open_workbook, Reader, Xlsx};
    let mut wb: Xlsx<_> =
        open_workbook(path).map_err(|e| format!("XLSX 解析失败 (failed to read .xlsx): {e}"))?;
    let mut out = String::new();
    for name in wb.sheet_names() {
        let range = match wb.worksheet_range(&name) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if range.is_empty() {
            continue;
        }
        out.push_str(&format!("# {name}\n"));
        for row in range.rows() {
            let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
            if cells.iter().all(|s| s.trim().is_empty()) {
                continue; // skip blank rows
            }
            out.push_str(&cells.join("\t"));
            out.push('\n');
        }
        out.push('\n');
    }
    Ok(out)
}

/// Paragraph-aware sliding-window chunking: split on blank lines, pack
/// paragraphs into ~CHUNK_CHARS windows, carry CHUNK_OVERLAP tail context.
fn chunk_text(text: &str) -> Vec<String> {
    let mut paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    if paragraphs.is_empty() {
        paragraphs = vec![text.trim()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in paragraphs {
        // Oversized single paragraph: hard-split by chars.
        if p.chars().count() > CHUNK_CHARS {
            if !cur.trim().is_empty() {
                chunks.push(cur.trim().to_string());
                cur.clear();
            }
            // Walk the paragraph by character boundaries rather than
            // collecting it. `Vec<char>` is FOUR bytes per character, so a file
            // with no blank line in it — minified JS, one-line JSON, a log —
            // arrives as a single paragraph and quadruples in memory before a
            // single chunk exists. That is what took the app down on a large
            // import; the slicing below allocates one chunk at a time.
            let idx: Vec<usize> = p.char_indices().map(|(i, _)| i).collect();
            let n = idx.len();
            let mut i = 0usize;
            while i < n {
                let end = (i + CHUNK_CHARS).min(n);
                let from = idx[i];
                let to = if end < n { idx[end] } else { p.len() };
                chunks.push(p[from..to].trim().to_string());
                if end == n {
                    break;
                }
                i = end.saturating_sub(CHUNK_OVERLAP);
            }
            continue;
        }
        if cur.chars().count() + p.chars().count() + 1 > CHUNK_CHARS && !cur.trim().is_empty() {
            chunks.push(cur.trim().to_string());
            // Overlap: carry the tail of the previous chunk forward.
            let tail: String = cur
                .chars()
                .rev()
                .take(CHUNK_OVERLAP)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            cur = tail;
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(p);
    }
    if !cur.trim().is_empty() {
        chunks.push(cur.trim().to_string());
    }
    chunks.retain(|c| c.chars().count() >= 20);
    chunks
}

// ---------------------------------------------------------------------------
// BM25 (ASCII words + CJK bigrams)
// ---------------------------------------------------------------------------

/// Scripts written without spaces between words. They are indexed as single
/// characters and pairs of them, so a short query still meets a phrase.
fn unspaced(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF        // Hiragana, Katakana
        | 0x31F0..=0x31FF      // Katakana extensions
        | 0x3400..=0x4DBF      // CJK extension A
        | 0x4E00..=0x9FFF      // CJK unified ideographs
        | 0xF900..=0xFAFF      // CJK compatibility ideographs
        | 0x20000..=0x3134F    // CJK extensions B–G
        | 0xAC00..=0xD7AF      // Hangul syllables
        | 0x0E00..=0x0EFF      // Thai, Lao
        | 0x1000..=0x109F      // Myanmar
        | 0x1780..=0x17FF      // Khmer
    )
}

/// Part of a word in a script that spaces its words: a letter or digit of
/// ANY script, or a combining mark that lives inside words (an Indic virama
/// is neither letter nor digit, but the word breaks without it).
fn word_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c as u32, 0x0300..=0x036F | 0x0900..=0x0DFF | 0x200C | 0x200D)
}

/// Keyword tokens for BM25.
///
/// This kept ASCII words and treated everything from U+3400 up as Chinese.
/// Japanese kana sits below that and was never indexed; Cyrillic, Greek,
/// Arabic and the rest were dropped outright; an accented letter split its
/// word (Portuguese `manutenção` became `manuten`); and full-width
/// punctuation — `，` `！` — came out as tokens of its own.
fn bm25_tokens(s: &str) -> Vec<String> {
    // Full-width letters and digits are the ASCII ones in another width:
    // `ＧＰＵ` should find `gpu`.
    let fold = |c: char| -> char {
        let u = c as u32;
        if (0xFF10..=0xFF19).contains(&u) || (0xFF21..=0xFF3A).contains(&u) || (0xFF41..=0xFF5A).contains(&u) {
            char::from_u32(u - 0xFEE0).unwrap_or(c)
        } else {
            c
        }
    };
    let chars: Vec<char> = s.chars().map(fold).collect();
    let mut out = Vec::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut Vec<String>| {
        if word.chars().count() >= 2 {
            out.push(std::mem::take(word));
        } else {
            word.clear();
        }
    };
    for (i, &c) in chars.iter().enumerate() {
        if unspaced(c) {
            flush(&mut word, &mut out);
            out.push(c.to_string());
            if let Some(&next) = chars.get(i + 1) {
                if unspaced(next) {
                    out.push([c, next].iter().collect());
                }
            }
        } else if word_char(c) {
            word.extend(c.to_lowercase());
        } else {
            flush(&mut word, &mut out);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// BM25 over a candidate corpus held in memory (rebuilt per search from the
/// DB — corpora are small enough that this stays in the low milliseconds).
fn bm25_scores(corpus_tokens: &[Vec<String>], query: &str) -> Vec<f32> {
    let n = corpus_tokens.len();
    if n == 0 {
        return Vec::new();
    }
    let mut df: HashMap<&str, u32> = HashMap::new();
    for toks in corpus_tokens {
        let uniq: HashSet<&str> = toks.iter().map(|s| s.as_str()).collect();
        for t in uniq {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    let avgdl =
        corpus_tokens.iter().map(|t| t.len() as f32).sum::<f32>() / n as f32;
    let (k1, b) = (1.5f32, 0.75f32);
    let q_tokens = bm25_tokens(query);

    corpus_tokens
        .iter()
        .map(|toks| {
            if toks.is_empty() {
                return 0.0;
            }
            let mut tf: HashMap<&str, u32> = HashMap::new();
            for t in toks {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            let dl = toks.len() as f32;
            q_tokens
                .iter()
                .map(|q| {
                    let f = *tf.get(q.as_str()).unwrap_or(&0) as f32;
                    if f == 0.0 {
                        return 0.0;
                    }
                    let dfq = *df.get(q.as_str()).unwrap_or(&0) as f32;
                    let idf = ((n as f32 - dfq + 0.5) / (dfq + 0.5) + 1.0).ln();
                    idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl.max(1.0)))
                })
                .sum::<f32>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Search: dense + BM25 → RRF → MMR → neighbor expansion
// ---------------------------------------------------------------------------

struct ChunkRow {
    id: i64,
    doc_id: i64,
    doc_name: String,
    seq: i64,
    text: String,
    emb: Vec<f32>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RagHit {
    pub doc_name: String,
    pub seq: i64,
    pub text: String,
    pub score: f32,
}

#[tauri::command(async)]
pub fn rag_search(app: tauri::AppHandle, query: String, k: Option<usize>) -> Result<Vec<RagHit>, String> {
    // How many chunks a question may cite. The old ceiling was 12 with a
    // default of 6, both written here — so a knowledge base of two hundred
    // documents answered out of six chunks no matter how much of it was
    // relevant, and nothing in the app could raise that. The caller decides
    // now; the clamp only keeps a runaway value from reading the whole table
    // into one prompt.
    let k = k.unwrap_or(6).clamp(1, 64);
    let rows: Vec<ChunkRow> = with_db(&app, |conn| {
        let mut stmt = conn
            .prepare(
                "SELECT c.id, c.doc_id, d.name, c.seq, c.text, c.embedding
                 FROM chunks c JOIN docs d ON d.id = c.doc_id
                 WHERE d.enabled = 1",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| {
                Ok(ChunkRow {
                    id: r.get(0)?,
                    doc_id: r.get(1)?,
                    doc_name: r.get(2)?,
                    seq: r.get(3)?,
                    text: r.get(4)?,
                    emb: blob_to_vec(&r.get::<_, Vec<u8>>(5)?),
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok(rows)
    })?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    // Dense retrieval (vectors are L2-normalized → dot = cosine).
    let qv = embed(&app, vec![query.clone()])?
        .pop()
        .ok_or("empty embedding")?;
    let mut dense: Vec<(usize, f32)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (i, dot(&qv, &r.emb)))
        .collect();
    dense.sort_by(|a, b| b.1.total_cmp(&a.1));
    dense.truncate(RETRIEVE_N);

    // Sparse retrieval.
    let corpus_tokens: Vec<Vec<String>> = rows.iter().map(|r| bm25_tokens(&r.text)).collect();
    let bm = bm25_scores(&corpus_tokens, &query);
    let mut sparse: Vec<(usize, f32)> = bm.iter().copied().enumerate().collect();
    sparse.sort_by(|a, b| b.1.total_cmp(&a.1));
    sparse.truncate(RETRIEVE_N);

    // Reciprocal-rank fusion (k=60).
    let mut fused: HashMap<usize, f32> = HashMap::new();
    for (rank, (i, _)) in dense.iter().enumerate() {
        *fused.entry(*i).or_insert(0.0) += 1.0 / (60.0 + rank as f32 + 1.0);
    }
    for (rank, (i, s)) in sparse.iter().enumerate() {
        if *s > 0.0 {
            *fused.entry(*i).or_insert(0.0) += 1.0 / (60.0 + rank as f32 + 1.0);
        }
    }
    let mut fused: Vec<(usize, f32)> = fused.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused.truncate(FUSED_N);

    // MMR diversification (λ = 0.72) down to k.
    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<(usize, f32)> = fused.clone();
    while selected.len() < k && !remaining.is_empty() {
        let mut best = 0usize;
        let mut best_score = f32::MIN;
        for (pos, (i, rel)) in remaining.iter().enumerate() {
            let max_sim = selected
                .iter()
                .map(|s| dot(&rows[*i].emb, &rows[*s].emb))
                .fold(0.0f32, f32::max);
            let score = 0.72 * rel - 0.28 * max_sim;
            if score > best_score {
                best_score = score;
                best = pos;
            }
        }
        selected.push(remaining.remove(best).0);
    }

    // Neighbor expansion: pull seq±1 of the same doc for coherent context.
    let by_key: HashMap<(i64, i64), &ChunkRow> =
        rows.iter().map(|r| ((r.doc_id, r.seq), r)).collect();
    let mut seen: HashSet<i64> = HashSet::new();
    let mut hits = Vec::new();
    for i in selected {
        let r = &rows[i];
        let mut text = String::new();
        for s in [r.seq - 1, r.seq, r.seq + 1] {
            if let Some(n) = by_key.get(&(r.doc_id, s)) {
                if seen.insert(n.id) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&n.text);
                }
            }
        }
        if text.is_empty() {
            continue; // fully covered by an earlier hit's expansion
        }
        hits.push(RagHit {
            doc_name: r.doc_name.clone(),
            seq: r.seq,
            text,
            score: dot(&qv, &r.emb),
        });
    }
    Ok(hits)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Ingestion + management commands
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RagProgress {
    /// "extract" | "embed" | "done"
    pub phase: &'static str,
    pub frac: f32,
}

const CAPTION_PROMPT: &str = "Describe this image thoroughly for search and retrieval. Cover: the main subject and scene, any people/objects, colors and layout, and — if it's a chart, diagram, screenshot or document — what it conveys and any labels. Be factual and specific. Do not add commentary.";

/// A scanned page is read, not described: what it says is what questions
/// will be about.
const TRANSCRIBE_PROMPT: &str = "This is one page of a scanned document. Transcribe all of the text on it exactly as written, in reading order and in its original language (Chinese stays Chinese) — no translation, no summary, no commentary. If the page has no text, briefly describe what it shows.";

/// Pages of a scan read into the knowledge base, at most.
const MAX_SCAN_PAGES: usize = 500;

async fn vision_ready(app: &tauri::AppHandle) -> bool {
    use tauri::Manager;
    let state = app.state::<crate::state::AppState>();
    let ready = state.model.read().await.as_ref().map(|m| m.vision_ready).unwrap_or(false);
    ready
}

/// Ask the loaded vision model about an image — `prompt` says what for.
/// Returns `None` (not an error) when no vision model is active; the caller
/// falls back to OCR-only indexing.
async fn vision_caption(
    app: &tauri::AppHandle,
    path: &str,
    on_progress: &Channel<RagProgress>,
    prompt: &str,
    max_tokens: u32,
    frac: f32,
) -> Option<String> {
    use tauri::Manager;
    if !vision_ready(app).await {
        return None;
    }
    let state = app.state::<crate::state::AppState>();
    let backend = state.backend().await?;
    let _ = on_progress.send(RagProgress { phase: "vision", frac });
    let req = crate::inference::GenRequest {
        messages: vec![crate::inference::ChatMessage {
            role: crate::inference::Role::User,
            content: prompt.to_string(),
            reasoning_content: None,
            images: vec![path.to_string()],
        }],
        params: crate::inference::GenParams {
            temperature: 0.3,
            max_tokens,
            think: Some(false),
            ..Default::default()
        },
    };
    state.cancel.store(false, std::sync::atomic::Ordering::SeqCst);
    match backend.generate_collect(req, state.cancel.clone()).await {
        Ok(t) => {
            let t = crate::commands::strip_think_blocks(&t).trim().to_string();
            (!t.is_empty()).then_some(t)
        }
        Err(e) => {
            eprintln!("vision caption failed (indexing OCR only): {e:#}");
            None
        }
    }
}

/// Read a PDF that has no text layer from its page images, one page at a
/// time, with the loaded vision model. The docs said a scan "gets in, read by
/// the vision model"; in fact it described at most six of the file's images,
/// and without a vision model refused it with "almost certainly a scan" and
/// no way forward (issue #15). OCR is no substitute here: it reads Latin
/// script only, and would index a Chinese page as noise.
async fn read_scanned_pdf(
    app: &tauri::AppHandle,
    path: &str,
    on_progress: &Channel<RagProgress>,
) -> Result<String, String> {
    if !vision_ready(app).await {
        return Err("这个 PDF 没有文字层,是扫描件(每一页都是图片)。要导入它,需要先加载一个能看图的模型——\
                    模型选择器里带视觉(Vision)标记的——Chaty 会用它逐页读出上面的文字。\
                    (This PDF has no text layer — it is a scan, every page a picture. To import it, load a \
                    model that can see images — marked Vision in the model picker — and Chaty will read \
                    each page with it.)"
            .into());
    }
    let p = path.to_string();
    let pages = tokio::task::spawn_blocking(move || crate::docimg::pdf_page_images(&p, MAX_SCAN_PAGES))
        .await
        .unwrap_or_default();
    if pages.is_empty() {
        return Err("这个 PDF 没有文字层,而它的页面图片用了暂不支持的编码(如 JBIG2、CCITT),读不出来。\
                    可以先用别的工具把它转成带文字层的 PDF,或把页面导出成图片再导入。\
                    (This PDF has no text layer, and its page images use an encoding Chaty cannot read yet, \
                    such as JBIG2 or CCITT. Convert it to a PDF with a text layer, or export the pages as \
                    images and import those.)"
            .into());
    }
    let mut texts = Vec::with_capacity(pages.len());
    for (i, img) in pages.iter().enumerate() {
        let frac = i as f32 / pages.len() as f32;
        texts.push(vision_caption(app, img, on_progress, TRANSCRIBE_PROMPT, 1500, frac).await);
    }
    scan_document(&texts).ok_or_else(|| {
        "这个 PDF 没有文字层,视觉模型也没能从页面图片里读出内容。\
         (This PDF has no text layer, and the vision model could not read anything from its pages.)"
            .into()
    })
}

/// A scan's page readings as one document: each under its page number, pages
/// that yielded nothing left out. `None` when no page did.
fn scan_document(pages: &[Option<String>]) -> Option<String> {
    let read: Vec<String> = pages
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let t = t.as_deref().map(str::trim).filter(|t| !t.is_empty())?;
            Some(format!("[第 {n} 页 / page {n}]\n{t}", n = i + 1))
        })
        .collect();
    (!read.is_empty()).then(|| {
        format!(
            "[本文档没有文字层,以下内容来自对页面图像的识别 \
             (no text layer in this document — what follows is read from its page images)]\n\n{}",
            read.join("\n\n")
        )
    })
}

#[tauri::command]
pub async fn rag_add_document(
    app: tauri::AppHandle,
    path: String,
    // When ingesting from a folder, the selected folder's path. The document is
    // then named by its path relative to that folder's parent (e.g.
    // `myproject/src/lib/ipc.ts`) so the knowledge base preserves — and the model
    // can see — the project's file structure, not just bare file names.
    root: Option<String>,
    // Whether a document's embedded images may be sent to the vision model.
    // Defaults to on when the caller says nothing, which is what every caller
    // did before there was a switch.
    caption_images: Option<bool>,
    on_progress: Channel<RagProgress>,
) -> Result<(), String> {
    // Images are indexed two ways, both async so they run before the blocking
    // section: (1) a vision-model description of what the image SHOWS (objects,
    // scene, any chart/diagram meaning) when a vision model is loaded, and
    // (2) OCR of any embedded text (ocrs, Latin). Either alone is enough to
    // index; together they make an image findable by content, not just its
    // literal text.
    let ext = std::path::Path::new(&path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let is_image = matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif");
    // Say what is being indexed BEFORE indexing it. Everything below can end
    // the process outright rather than returning an error — the vision model
    // captioning a PDF's embedded figures and the embedder both allocate on
    // top of a chat model that is already resident — and on Windows such a
    // death leaves no crash report to sweep. The marker is what turns "it just
    // closes, the log is empty" (issue #13) into a file name and a phase.
    //
    // The NAME only, never the full path: this log exists to be pasted into a
    // public issue, and a directory tree is somebody's home folder.
    let crumb = crate::errlog::Inflight::begin(
        "kb-index",
        &format!(
            "{} ({ext}, {})",
            std::path::Path::new(&path)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "document".into()),
            std::fs::metadata(&path)
                .map(|m| format!("{:.1} MB", m.len() as f64 / 1_048_576.0))
                .unwrap_or_else(|_| "size unknown".into()),
        ),
    );
    let (ocr_text, vision_text) = if is_image {
        // (1) Vision caption — best-effort; skipped when no vision model is loaded.
        let vision_text = vision_caption(&app, &path, &on_progress, CAPTION_PROMPT, 480, 0.0).await;
        // (2) OCR text.
        let _ = on_progress.send(RagProgress { phase: "extract", frac: 0.0 });
        let dir = app
            .path()
            .app_data_dir()
            .map_err(|e| e.to_string())?
            .join("ocr-models");
        let ocr = crate::ocr::ocr_image(dir, path.clone())
            .await
            .map(|t| t.trim().to_string())
            .unwrap_or_default();
        // An image with neither a caption nor OCR text has nothing to index.
        if vision_text.is_none() && ocr.is_empty() {
            return Err(
                "无法从图片中提取内容：未加载视觉模型且未识别到文字。(Nothing to index from the image — no vision model loaded and no OCR text found.)"
                    .into(),
            );
        }
        (Some(ocr), vision_text)
    } else {
        (None, None)
    };

    // Documents with EMBEDDED images (docx/xlsx/pptx/pdf): caption each one
    // with the vision model so charts/photos inside the document are findable
    // by what they show — appended to the text before chunking. Best-effort:
    // skipped without a vision model.
    // Off, and the document is indexed from its text alone. This step runs the
    // vision model several times ON TOP of whatever chat model is already
    // resident, which is the least predictable memory cost in the whole
    // pipeline and the first thing to try turning off when indexing takes the
    // app down with it (issue #13).
    let caption_images = caption_images.unwrap_or(true);

    // A PDF's text is read first: whether it has any decides what its
    // pictures are — figures beside the text, or the pages themselves.
    let pdf_text = if ext == "pdf" {
        let p = path.clone();
        Some(tokio::task::spawn_blocking(move || extract_text(&p)).await.map_err(|e| e.to_string())?)
    } else {
        None
    };
    let scanned = matches!(&pdf_text, Some(Err(e)) if *e == no_text_layer());
    let scan_text = if scanned {
        Some(read_scanned_pdf(&app, &path, &on_progress).await?)
    } else {
        None
    };

    let embedded_captions: Vec<String> = if caption_images
        && !scanned
        && matches!(ext.as_str(), "pdf" | "docx" | "xlsx" | "pptx")
    {
        let p2 = path.clone();
        let imgs = tokio::task::spawn_blocking(move || crate::docimg::extract_embedded_images(&p2, 6))
            .await
            .unwrap_or_default();
        let mut caps = Vec::new();
        for (i, img) in imgs.iter().enumerate() {
            if let Some(c) = vision_caption(&app, img, &on_progress, CAPTION_PROMPT, 480, 0.0).await {
                caps.push(format!("[文档内嵌图片 {} (embedded image)] {c}", i + 1));
            }
        }
        caps
    } else {
        Vec::new()
    };

    tokio::task::spawn_blocking(move || {
        let basename = || {
            std::path::Path::new(&path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("document")
                .to_string()
        };
        // Folder import → name by path relative to the root folder's parent, so
        // the project folder name and subdirectory layout are kept (and shown to
        // the model). Single-file add (no root) keeps the bare file name.
        let name = root
            .as_deref()
            .map(std::path::Path::new)
            .map(|r| r.parent().unwrap_or(r))
            .and_then(|base| std::path::Path::new(&path).strip_prefix(base).ok())
            .and_then(|rel| rel.to_str())
            .map(|s| s.replace('\\', "/"))
            .unwrap_or_else(basename);
        let _ = on_progress.send(RagProgress { phase: "extract", frac: 0.0 });
        crumb.update(&format!("{name}: extracting text"));
        let text = match ocr_text {
            // Image: combine the vision description with any OCR'd text so the
            // chunk is retrievable by visual content AND literal text.
            Some(ocr) => {
                let mut parts: Vec<String> = Vec::new();
                if let Some(v) = &vision_text {
                    if !v.trim().is_empty() {
                        parts.push(format!("[图像内容 / Image description]\n{}", v.trim()));
                    }
                }
                if !ocr.trim().is_empty() {
                    parts.push(format!("[图中文字 / Text in image]\n{}", ocr.trim()));
                }
                parts.join("\n\n")
            }
            None => {
                // Vision captions of the document's embedded images are indexed
                // alongside the text so figures are findable by what they show —
                // and when there is no text to stand beside, they ARE the
                // document. A scan is pages of pictures; having just described
                // those pictures and then reporting "no extractable text" threw
                // away the one reading of the file we had.
                let extracted = match scan_text {
                    Some(pages) => Ok(pages),
                    None => pdf_text.unwrap_or_else(|| extract_text(&path)),
                };
                match (extracted, embedded_captions.is_empty()) {
                    (Ok(mut t), false) => {
                        t.push_str("\n\n");
                        t.push_str(&embedded_captions.join("\n\n"));
                        t
                    }
                    (Ok(t), true) => t,
                    // Nothing described and nothing extracted: the extraction
                    // error is the honest thing to report.
                    (Err(e), true) => return Err(e),
                    (Err(_), false) => format!(
                        "[本文档没有文字层,以下内容来自对页面图像的识别 \
                         (no text layer in this document — what follows is read \
                         from its images)]\n\n{}",
                        embedded_captions.join("\n\n")
                    ),
                }
            }
        };
        let chars = text.chars().count();
        if chars > MAX_INDEX_CHARS {
            return Err(format!(
                "文档过大,无法索引:{chars} 字符,上限 {MAX_INDEX_CHARS}。请拆分后再导入。\n\
                 (Document too large to index: {chars} characters, limit {MAX_INDEX_CHARS}. Split it and import the parts.)"
            ));
        }
        let chunks = chunk_text(&text);
        if chunks.is_empty() {
            return Err("文档中没有可索引的文本 (no indexable text in document)".into());
        }

        // Replace an existing doc of the same name.
        with_db(&app, |conn| {
            conn.execute("DELETE FROM docs WHERE name = ?1", params![name])
                .map_err(|e| e.to_string())?;
            conn.execute(
                "DELETE FROM chunks WHERE doc_id NOT IN (SELECT id FROM docs)",
                [],
            )
            .map_err(|e| e.to_string())?;
            // chunks = 0 means "indexing, not usable yet". The real count is
            // written once every chunk is in — see the update below. Recording
            // the final count HERE was issue #13's second symptom: a run that
            // died mid-embed left a document that claimed 1500 chunks, held a
            // handful, and answered nothing, while looking perfectly indexed.
            conn.execute(
                "INSERT INTO docs(name, path, chunks, created_at) VALUES(?1, ?2, ?3, ?4)",
                params![
                    name,
                    path,
                    0i64,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0)
                ],
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        let doc_id: i64 = with_db(&app, |conn| {
            conn.query_row("SELECT id FROM docs WHERE name = ?1", params![name], |r| {
                r.get(0)
            })
            .map_err(|e| e.to_string())
        })?;

        // Embed in small batches, streaming progress. Anything that goes
        // wrong from here on has a half-written document to answer for, so it
        // leaves through `fail`, which takes it back out.
        let total = chunks.len();
        let fail = |e: String| -> String {
            let _ = with_db(&app, |conn| {
                conn.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])
                    .map_err(|x| x.to_string())?;
                conn.execute("DELETE FROM docs WHERE id = ?1", params![doc_id])
                    .map_err(|x| x.to_string())?;
                Ok(())
            });
            e
        };
        for (i, batch) in chunks.chunks(8).enumerate() {
            crumb.update(&format!(
                "{name}: embedding chunk {}/{total}",
                (i * 8 + batch.len()).min(total)
            ));
            let embs = embed(&app, batch.to_vec()).map_err(&fail)?;
            with_db(&app, |conn| {
                for (j, (text, emb)) in batch.iter().zip(&embs).enumerate() {
                    conn.execute(
                        "INSERT INTO chunks(doc_id, seq, text, embedding) VALUES(?1, ?2, ?3, ?4)",
                        params![doc_id, (i * 8 + j) as i64, text, vec_to_blob(emb)],
                    )
                    .map_err(|e| e.to_string())?;
                }
                Ok(())
            })
            .map_err(&fail)?;
            let done = ((i * 8 + batch.len()) as f32 / total as f32).min(1.0);
            let _ = on_progress.send(RagProgress { phase: "embed", frac: done });
        }
        // Every chunk is in: the document is usable, and now says so.
        with_db(&app, |conn| {
            conn.execute(
                "UPDATE docs SET chunks = ?1 WHERE id = ?2",
                params![total as i64, doc_id],
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        })
        .map_err(&fail)?;
        let _ = on_progress.send(RagProgress { phase: "done", frac: 1.0 });
        Ok(())
    })
    .await
    .map_err(|e| format!("索引任务异常 (indexing task panicked): {e}"))?
}

/// File extensions the knowledge base can ingest: documents (PDF/DOCX),
/// OCR-able images, and a broad set of text/code/markup/config files (all read
/// as UTF-8). Kept in sync with the file-picker filter in KnowledgePanel.tsx.
const SUPPORTED_EXTS: &[&str] = &[
    // documents + images
    "pdf", "docx", "xlsx", "png", "jpg", "jpeg", "webp", "bmp", "gif",
    // plain docs / data
    "txt", "md", "markdown", "mdx", "rst", "org", "tex", "log", "csv", "tsv", "json", "jsonl",
    "ndjson", "yaml", "yml", "toml", "ini", "cfg", "conf", "properties", "env",
    // markup / web
    "html", "htm", "xml", "css", "scss", "sass", "less", "vue", "svelte", "astro",
    // code
    "js", "mjs", "cjs", "jsx", "ts", "tsx", "py", "pyi", "rs", "go", "java", "kt", "kts", "c", "h",
    "cpp", "cc", "cxx", "hpp", "hh", "cs", "rb", "php", "swift", "scala", "sh", "bash", "zsh",
    "fish", "ps1", "bat", "sql", "lua", "r", "jl", "pl", "pm", "dart", "ex", "exs", "erl", "hs",
    "clj", "cljs", "elm", "ml", "fs", "vb", "gradle", "groovy", "m", "mm",
];

/// Recursively collect ingestable files under `dir` (walks subdirectories).
/// Skips hidden entries (dotfiles/dot-dirs) and symlinks (avoids cycles), and
/// caps the result so an accidental huge folder can't run away.
#[tauri::command(async)]
pub fn rag_list_supported_files(dir: String) -> Result<Vec<String>, String> {
    const MAX_FILES: usize = 5000;
    let root = std::path::PathBuf::from(&dir);
    if !root.is_dir() {
        return Err("不是有效的文件夹 (not a directory)".into());
    }
    let mut out: Vec<String> = Vec::new();
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        let rd = match std::fs::read_dir(&p) {
            Ok(rd) => rd,
            Err(_) => continue, // unreadable dir → skip silently
        };
        for entry in rd.flatten() {
            if entry
                .file_name()
                .to_str()
                .map(|s| s.starts_with('.'))
                .unwrap_or(false)
            {
                continue; // hidden entry
            }
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_symlink() {
                continue;
            }
            let path = entry.path();
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file()
                && path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|e| SUPPORTED_EXTS.contains(&e.to_lowercase().as_str()))
                    .unwrap_or(false)
            {
                if let Some(s) = path.to_str() {
                    out.push(s.to_string());
                    if out.len() >= MAX_FILES {
                        out.sort();
                        return Ok(out);
                    }
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RagDoc {
    pub id: i64,
    pub name: String,
    pub chunks: i64,
    pub enabled: bool,
}

#[tauri::command(async)]
pub fn rag_list_documents(app: tauri::AppHandle) -> Result<Vec<RagDoc>, String> {
    with_db(&app, |conn| {
        let mut stmt = conn
            .prepare("SELECT id, name, chunks, enabled FROM docs ORDER BY created_at DESC")
            .map_err(|e| e.to_string())?;
        let docs = stmt
            .query_map([], |r| {
                Ok(RagDoc {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    chunks: r.get(2)?,
                    enabled: r.get::<_, i64>(3)? != 0,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok(docs)
    })
}

/// Concatenated text from the enabled documents, capped at `max_chars`.
/// Used to feed the deep-dive podcast transcript generator. Chunks are pulled
/// in document/sequence order and de-duplicated by their (doc, seq) overlap.
#[tauri::command(async)]
pub fn rag_corpus(app: tauri::AppHandle, max_chars: Option<usize>) -> Result<String, String> {
    let cap = max_chars.unwrap_or(12000).clamp(1000, 40000);
    with_db(&app, |conn| {
        let mut stmt = conn
            .prepare(
                "SELECT d.name, c.text
                 FROM chunks c JOIN docs d ON d.id = c.doc_id
                 WHERE d.enabled = 1
                 ORDER BY c.doc_id, c.seq",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        let mut out = String::new();
        let mut last_doc = String::new();
        for (name, text) in rows {
            if out.chars().count() >= cap {
                break;
            }
            if name != last_doc {
                out.push_str(&format!("\n\n# {name}\n\n"));
                last_doc = name;
            }
            out.push_str(&text);
            out.push_str("\n\n");
        }
        let trimmed: String = out.trim().chars().take(cap).collect();
        if trimmed.is_empty() {
            return Err("知识库为空或全部文档已禁用 (knowledge base is empty or all documents are disabled)".into());
        }
        Ok(trimmed)
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RagDocText {
    pub name: String,
    pub text: String,
}

/// Per-document text from the enabled knowledge base, for grounding an overview
/// report with one citation per file. Each document's text is capped to a fair
/// share of `max_chars` (so a single big file can't crowd the others out), and
/// the overall total is capped too.
#[tauri::command(async)]
pub fn rag_corpus_docs(
    app: tauri::AppHandle,
    max_chars: Option<usize>,
) -> Result<Vec<RagDocText>, String> {
    let cap = max_chars.unwrap_or(16000).clamp(2000, 60000);
    with_db(&app, |conn| {
        let n_docs: i64 = conn
            .query_row("SELECT COUNT(*) FROM docs WHERE enabled = 1", [], |r| r.get(0))
            .unwrap_or(0);
        if n_docs == 0 {
            return Err("知识库为空或全部文档已禁用 (knowledge base is empty or all documents are disabled)".into());
        }
        let per_doc = (cap / n_docs as usize).clamp(800, 8000);

        let mut stmt = conn
            .prepare(
                "SELECT d.name, c.text
                 FROM chunks c JOIN docs d ON d.id = c.doc_id
                 WHERE d.enabled = 1
                 ORDER BY c.doc_id, c.seq",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        let mut out: Vec<RagDocText> = Vec::new();
        let mut total = 0usize;
        for (name, text) in rows {
            if total >= cap {
                break;
            }
            match out.last_mut() {
                Some(last) if last.name == name => {
                    if last.text.chars().count() < per_doc {
                        last.text.push('\n');
                        last.text.push_str(&text);
                        total += text.chars().count();
                    }
                }
                _ => {
                    let t: String = text.chars().take(per_doc).collect();
                    total += t.chars().count();
                    out.push(RagDocText { name, text: t });
                }
            }
        }
        // Trim each doc to its fair share for a clean, predictable budget.
        for d in &mut out {
            if d.text.chars().count() > per_doc {
                d.text = d.text.chars().take(per_doc).collect();
            }
            d.text = d.text.trim().to_string();
        }
        Ok(out)
    })
}

/// Toggle whether a document participates in retrieval (custom query scope).
#[tauri::command]
pub fn rag_set_doc_enabled(app: tauri::AppHandle, id: i64, enabled: bool) -> Result<(), String> {
    with_db(&app, |conn| {
        conn.execute(
            "UPDATE docs SET enabled = ?2 WHERE id = ?1",
            params![id, enabled as i64],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    })
}

#[tauri::command(async)]
pub fn rag_remove_document(app: tauri::AppHandle, id: i64) -> Result<(), String> {
    with_db(&app, |conn| {
        conn.execute("DELETE FROM chunks WHERE doc_id = ?1", params![id])
            .map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM docs WHERE id = ?1", params![id])
            .map_err(|e| e.to_string())?;
        Ok(())
    })
}

/// Empty the whole knowledge base — drop every document and its chunks. The
/// embedding model is left as-is (downloaded once); only indexed content goes.
#[tauri::command(async)]
pub fn rag_clear_all(app: tauri::AppHandle) -> Result<(), String> {
    with_db(&app, |conn| {
        conn.execute("DELETE FROM chunks", [])
            .map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM docs", [])
            .map_err(|e| e.to_string())?;
        Ok(())
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RagStatus {
    pub model_ready: bool,
    pub docs: i64,
    pub chunks: i64,
}

#[tauri::command]
pub fn rag_status(app: tauri::AppHandle) -> Result<RagStatus, String> {
    let model_ready = embed_model_path(&app).map(|p| p.exists()).unwrap_or(false);
    let (docs, chunks) = with_db(&app, |conn| {
        let docs: i64 = conn
            .query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let chunks: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        Ok((docs, chunks))
    })?;
    Ok(RagStatus { model_ready, docs, chunks })
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum RagDlProgress {
    Progress { downloaded: u64, total: u64 },
    Done,
    Error { message: String },
}

/// Download the bge-m3 embedding model (~730 MB, one-time).
#[tauri::command]
pub async fn rag_download_model(
    app: tauri::AppHandle,
    endpoint: Option<String>,
    on_progress: Channel<RagDlProgress>,
) -> Result<(), String> {
    let dest = embed_model_path(&app)?;
    if dest.exists() {
        let _ = on_progress.send(RagDlProgress::Done);
        return Ok(());
    }
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
    let tmp = dest.with_extension("part");
    let cancel = crate::download::register_cancel("rag-embed");

    // Honour the HF endpoint setting: rewrite the official host to the chosen
    // base. On a mirror, `parse_hf_resolve_url` won't match the rewritten URL,
    // so the xet fallback (official-only protocol) is skipped naturally.
    let base = crate::download::hf_base(endpoint.as_deref());
    let urls: Vec<String> = EMBED_URLS
        .iter()
        .map(|u| u.replace(crate::download::HF_OFFICIAL, &base))
        .collect();

    let client = crate::http::download_client("Chaty-RAG")?;
    let mut last_err = String::new();
    for url in &urls {
        let resp = match client.get(url).send().await.and_then(|r| r.error_for_status()) {
            Ok(r) => r,
            Err(e) => {
                // CDN-blocked network (cas-bridge 403): retry this URL over xet.
                if e.status() == Some(reqwest::StatusCode::FORBIDDEN) {
                    if let Some((repo, revision, path)) = crate::download::parse_hf_resolve_url(url) {
                        let tmp_root = app.path().app_data_dir().map_err(|e| e.to_string())?;
                        let progress = on_progress.clone();
                        let result = crate::download::xet_fallback_download(
                            &repo,
                            &revision,
                            &path,
                            &dest,
                            &tmp_root,
                            move |downloaded, total| {
                                let _ = progress.send(RagDlProgress::Progress { downloaded, total });
                            },
                            &cancel,
                        )
                        .await;
                        match result {
                            Ok(()) => {
                                crate::download::clear_cancel("rag-embed");
                                let _ = on_progress.send(RagDlProgress::Done);
                                return Ok(());
                            }
                            Err(msg) if msg == crate::download::CANCELLED => {
                                crate::download::clear_cancel("rag-embed");
                                return Err(msg);
                            }
                            Err(msg) => {
                                last_err = crate::download::cdn_blocked_message(&msg);
                                continue;
                            }
                        }
                    }
                }
                last_err = e.to_string();
                continue;
            }
        };
        let total = resp.content_length().unwrap_or(0);
        let mut resp = resp;
        let mut file = match std::fs::File::create(&tmp) {
            Ok(f) => f,
            Err(e) => return Err(e.to_string()),
        };
        use std::io::Write;
        let mut downloaded: u64 = 0;
        let mut ok = true;
        loop {
            if cancel.load(std::sync::atomic::Ordering::SeqCst) {
                drop(file);
                let _ = std::fs::remove_file(&tmp);
                crate::download::clear_cancel("rag-embed");
                return Err(crate::download::CANCELLED.into());
            }
            match resp.chunk().await {
                Ok(Some(bytes)) => {
                    if file.write_all(&bytes).is_err() {
                        ok = false;
                        break;
                    }
                    downloaded += bytes.len() as u64;
                    let _ = on_progress.send(RagDlProgress::Progress { downloaded, total });
                }
                Ok(None) => break,
                Err(e) => {
                    last_err = e.to_string();
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            drop(file);
            std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
            crate::download::clear_cancel("rag-embed");
            let _ = on_progress.send(RagDlProgress::Done);
            return Ok(());
        }
    }
    crate::download::clear_cancel("rag-embed");
    let _ = std::fs::remove_file(&tmp);
    let msg = format!("嵌入模型下载失败 (embedding model download failed): {last_err}");
    let _ = on_progress.send(RagDlProgress::Error { message: msg.clone() });
    Err(msg)
}

#[cfg(test)]
mod tests {
    /// A chunk may never exceed the micro-batch the encoder context was given:
    /// past it llama.cpp aborts the process instead of returning an error.
    #[test]
    fn a_chunk_never_exceeds_the_encoders_micro_batch() {
        use super::embed_token_limit;
        assert_eq!(embed_token_limit(1024, 1024), 1020, "window minus special tokens");
        assert_eq!(
            embed_token_limit(1024, 512),
            512,
            "a smaller micro-batch binds — 512 was the default, and the crash"
        );
        assert!(embed_token_limit(1024, 512) < 606, "the reported book's first batch would now fit");
        assert_eq!(embed_token_limit(0, 512), 1, "never zero tokens");
    }

    /// The embedder shares a card with a chat model that is already on it, so
    /// "offload everything" is the one answer that is certainly wrong — and an
    /// over-ask is not an error to retry, it is a Vulkan driver taking the
    /// process down (issue #13 arrived as an import that closed the window and
    /// logged nothing). What VRAM allows and what a previous crash allows both
    /// bind; neither is optional.
    #[test]
    fn the_embedder_never_asks_for_more_gpu_than_it_is_allowed() {
        use super::embedder_layers;
        assert_eq!(embedder_layers(Some(20), None), 20, "no crash, no cap: what fits");
        assert_eq!(embedder_layers(Some(20), Some(4)), 4, "a cap from a crash binds");
        assert_eq!(
            embedder_layers(Some(3), Some(9)),
            3,
            "a cap wider than the ask leaves it alone"
        );
        assert_eq!(embedder_layers(Some(20), Some(0)), 0, "CPU-only after a crash");
        assert_eq!(embedder_layers(None, None), 0, "no GPU, or an unreadable file: CPU");
        assert_eq!(
            embedder_layers(None, Some(40)),
            0,
            "an unanswered question is never widened by a generous cap"
        );
    }

    /// A half-indexed document is taken back out on open, and a whole one is
    /// left alone. This is issue #13's second symptom: the crash left a doc
    /// row claiming every chunk while holding a few, so the document looked
    /// ready, could be ticked, and answered nothing.
    #[test]
    fn a_document_that_never_finished_indexing_is_removed_on_open() {
        use rusqlite::params;
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE docs(id INTEGER PRIMARY KEY, name TEXT NOT NULL, path TEXT,
               chunks INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
             CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
               seq INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL);",
        )
        .unwrap();
        let add = |id: i64, name: &str, claimed: i64, actual: i64| {
            conn.execute(
                "INSERT INTO docs(id, name, chunks, created_at) VALUES(?1, ?2, ?3, 0)",
                params![id, name, claimed],
            )
            .unwrap();
            for seq in 0..actual {
                conn.execute(
                    "INSERT INTO chunks(doc_id, seq, text, embedding) VALUES(?1, ?2, 'x', X'00')",
                    params![id, seq],
                )
                .unwrap();
            }
        };
        add(1, "whole.pdf", 3, 3); // finished
        add(2, "crashed.pdf", 1500, 4); // died mid-embed — the reported shape
        add(3, "started.pdf", 0, 0); // died before the first batch landed
        // A doc claiming 0 and holding 0 is only "complete" if nothing was
        // ever meant to be there, which the ingest refuses to create — so it
        // goes too, and the counts below say which survived.
        assert_eq!(super::repair_incomplete_docs(&conn), 2);

        let names: Vec<String> = conn
            .prepare("SELECT name FROM docs ORDER BY id")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(names, vec!["whole.pdf".to_string()], "only the finished one stays");
        let orphans: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks WHERE doc_id <> 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(orphans, 0, "their chunks go with them");
        // Idempotent: a second open finds nothing left to repair.
        assert_eq!(super::repair_incomplete_docs(&conn), 0);
    }

    /// A parser that asserts must reach the caller as an error, never as a
    /// panic — `pdf-extract` asserts on font encodings it has not implemented
    /// (`assert!(name == "Identity-H")`), which is most CJK-authored PDFs.
    #[test]
    fn a_pdf_that_makes_the_parser_assert_comes_back_as_an_error() {
        // Not a PDF at all: the parser fails on it one way or another, and
        // either way this must return rather than unwind past us.
        let junk = vec![b'%'; 4096];
        assert!(super::extract_pdf_bytes(&junk).is_err());

        let dir = std::env::temp_dir().join("chaty-pdf-panic-test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("broken.pdf");
        std::fs::write(&f, b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n").unwrap();
        assert!(super::extract_pdf(&f.to_string_lossy()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The flag that keeps a caught panic out of the error log must come back
    /// down when the closure unwinds — otherwise the NEXT real panic in this
    /// thread goes unreported.
    #[test]
    fn a_handled_panic_does_not_silence_the_one_after_it() {
        let out = std::panic::catch_unwind(|| {
            crate::errlog::handled_panic(|| panic!("expected"));
        });
        assert!(out.is_err());
        // Back to normal: a panic here would be logged again. Observable only
        // through the flag, which `handled_panic` owns, so run it once more
        // and check it still returns the closure's value.
        assert_eq!(crate::errlog::handled_panic(|| 7), 7);
    }

    // Issue #15: a scan becomes a document of its pages' readings, each under
    // its page number; a page that read as nothing is left out, not numbered
    // as empty — and a scan nothing could be read from is no document at all.
    #[test]
    fn a_scan_is_indexed_page_by_page() {
        let doc = super::scan_document(&[
            Some("2007年3月20日 官方站上线".into()),
            None,
            Some("  \n".into()),
            Some("第四页的文字".into()),
        ])
        .unwrap();
        assert!(doc.contains("no text layer"));
        assert!(doc.contains("[第 1 页 / page 1]\n2007年3月20日 官方站上线"));
        assert!(doc.contains("[第 4 页 / page 4]\n第四页的文字"));
        assert!(!doc.contains("page 2") && !doc.contains("page 3"));
        assert!(super::scan_document(&[None, Some(" ".into())]).is_none());
    }

    /// A PDF whose pages parse but carry no text is a scan, and saying so is
    /// the difference between "empty document" and "I cannot read this".
    #[test]
    fn a_pdf_with_no_text_layer_says_it_is_a_scan() {
        let err = super::has_text("  \n \t ".to_string()).unwrap_err();
        assert!(err.contains("扫描件") && err.contains("scan"));
        // …and it is how the knowledge base knows to read the pages instead.
        assert_eq!(err, super::no_text_layer());
        // A page number and a stray mark are not a text layer either.
        assert!(super::has_text("1\n\n2".to_string()).is_err());
        // Anything with real content through, untouched.
        let real = "Leçon 7 — le conditionnel présent et ses emplois".to_string();
        assert_eq!(super::has_text(real.clone()).unwrap(), real);
    }

    /// Point this at a real file to check extraction end to end. Skipped
    /// unless CHATY_PDF_PROBE names one — the interesting PDFs are the
    /// owner's, not the repo's.
    #[test]
    fn pdf_probe() {
        let Ok(path) = std::env::var("CHATY_PDF_PROBE") else { return };
        match super::extract_pdf(&path) {
            Ok(t) => println!("PROBE ok: {} chars\n{}", t.len(), &t[..t.len().min(300)]),
            Err(e) => println!("PROBE err: {e}"),
        }
    }

    use super::*;

    /// Real-embedder semantic probe (the knowledge base's core signal path,
    /// which nothing else on a CI runner can exercise):
    ///   CHATY_TEST_EMBED_GGUF=<bge-m3 .gguf> \
    ///   cargo test --lib rag_embedder_semantic_probe -- --ignored
    #[test]
    #[ignore]
    fn rag_embedder_semantic_probe() {
        let model = std::env::var("CHATY_TEST_EMBED_GGUF").expect("set CHATY_TEST_EMBED_GGUF");
        let emb = embedder_start(&PathBuf::from(model), None).expect("embedder start");
        let (rtx, rrx) = std::sync::mpsc::channel();
        emb.tx
            .send(EmbedJob::Embed {
                texts: vec![
                    "a small kitten playing".into(),
                    "a young cat".into(),
                    "a carburetor engine part".into(),
                ],
                reply: rtx,
            })
            .unwrap();
        let vs = rrx.recv().unwrap().expect("embed batch");
        assert_eq!(vs.len(), 3, "one vector per text");
        assert!(vs[0].len() >= 256, "real embedding dims, got {}", vs[0].len());
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let kitten_cat = dot(&vs[0], &vs[1]);
        let kitten_carb = dot(&vs[0], &vs[2]);
        assert!(
            kitten_cat > kitten_carb + 0.05,
            "semantic order broken: kitten~cat {kitten_cat} vs kitten~carburetor {kitten_carb}"
        );
        drop(emb); // exercises the worker-shutdown path
    }

    /// How many tokens each chunk of a real document becomes under bge-m3's own
    /// tokenizer. Characters are a poor proxy: 800 characters of English is
    /// about 200 tokens, 800 of Chinese several times that, and the embedder's
    /// limits are in tokens. Vocabulary only — nothing is embedded.
    ///   CHATY_TEST_EMBED_GGUF=<bge-m3 .gguf> CHATY_TEST_EMBED_PDF=<a pdf> \
    ///   cargo test --release --lib rag_chunk_token_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn rag_chunk_token_probe() {
        let model_path = std::env::var("CHATY_TEST_EMBED_GGUF").expect("set CHATY_TEST_EMBED_GGUF");
        let pdf = std::env::var("CHATY_TEST_EMBED_PDF").expect("set CHATY_TEST_EMBED_PDF");
        let text = extract_pdf(&pdf).expect("extract");
        let chunks = chunk_text(&text);
        let backend = crate::inference::llama_backend_pub().expect("backend");
        let params = LlamaModelParams::default().with_vocab_only(true);
        let model = LlamaModel::load_from_file(backend, &model_path, &params).expect("vocab");
        let mut lens: Vec<usize> = chunks
            .iter()
            .map(|c| model.str_to_token(c, AddBos::Always).map(|t| t.len()).unwrap_or(0))
            .collect();
        let first_batch_max = lens.iter().take(8).cloned().max().unwrap_or(0);
        lens.sort_unstable();
        let pct = |p: f64| lens[((lens.len() - 1) as f64 * p) as usize];
        println!(
            "\nPROBE {} chunks: tokens min {} median {} p95 {} max {}",
            lens.len(), lens[0], pct(0.5), pct(0.95), lens[lens.len() - 1]
        );
        println!(
            "PROBE over 512 (the default micro-batch): {}   over 1020 (our truncation): {}   largest in the first batch of 8: {}",
            lens.iter().filter(|&&n| n > 512).count(),
            lens.iter().filter(|&&n| n > 1020).count(),
            first_batch_max
        );
    }

    /// CPU against GPU on a real document — speed, whether the vectors agree,
    /// and whether the GPU gives its memory back. The embedder was held on the
    /// CPU on macOS because a Metal load was seen to leave its pages wired after
    /// unload; moving it is only safe if that no longer happens, so this
    /// measures it instead of assuming. Run alone, nothing else on the GPU:
    ///   CHATY_TEST_EMBED_GGUF=<bge-m3 .gguf> CHATY_TEST_EMBED_PDF=<a pdf> \
    ///   cargo test --release --lib rag_embed_gpu_probe -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore]
    fn rag_embed_gpu_probe() {
        // As the app does, before the first Metal initialisation.
        #[cfg(target_os = "macos")]
        std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");
        let model = PathBuf::from(std::env::var("CHATY_TEST_EMBED_GGUF").expect("set CHATY_TEST_EMBED_GGUF"));
        let pdf = std::env::var("CHATY_TEST_EMBED_PDF").expect("set CHATY_TEST_EMBED_PDF");
        let text = extract_pdf(&pdf).expect("extract");
        let chunks = chunk_text(&text);
        println!("PROBE {} chars -> {} chunks", text.chars().count(), chunks.len());

        let wired_mb = || -> f64 {
            let out = std::process::Command::new("vm_stat").output().map(|o| o.stdout).unwrap_or_default();
            let out = String::from_utf8_lossy(&out);
            let page = out
                .lines()
                .next()
                .and_then(|l| l.split("page size of ").nth(1))
                .and_then(|r| r.split_whitespace().next())
                .and_then(|n| n.parse::<f64>().ok())
                .unwrap_or(16384.0);
            let pages = out
                .lines()
                .find(|l| l.starts_with("Pages wired down"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|n| n.trim().trim_end_matches('.').parse::<f64>().ok())
                .unwrap_or(0.0);
            pages * page / 1_048_576.0
        };
        let backend = crate::inference::llama_backend_pub().expect("backend");
        let auto = embedder_gpu_layers(backend, &model.to_string_lossy());
        println!("PROBE the app would offload {auto} layer(s) on this machine right now");

        let run = |label: &str, layers: i32| -> Vec<Vec<f32>> {
            let w0 = wired_mb();
            let t0 = std::time::Instant::now();
            let emb = embedder_start(&model, Some(layers)).expect("embedder start");
            let load = t0.elapsed().as_secs_f64();
            let w1 = wired_mb();
            let t1 = std::time::Instant::now();
            let mut out = Vec::with_capacity(chunks.len());
            for batch in chunks.chunks(8) {
                let (rtx, rrx) = std::sync::mpsc::channel();
                emb.tx.send(EmbedJob::Embed { texts: batch.to_vec(), reply: rtx }).unwrap();
                out.extend(rrx.recv().unwrap().expect("embed batch"));
            }
            let secs = t1.elapsed().as_secs_f64();
            drop(emb); // closes the channel and joins: the model is dropped here
            std::thread::sleep(std::time::Duration::from_millis(1500));
            let w2 = wired_mb();
            println!(
                "PROBE {label:<4} layers={layers:<3} load {load:.1}s  embed {n} chunks {secs:.1}s ({rate:.1}/s)  wired {w0:.0} MB -> loaded {d1:+.0} -> after unload {d2:+.0}",
                n = out.len(),
                rate = out.len() as f64 / secs,
                d1 = w1 - w0,
                d2 = w2 - w0,
            );
            out
        };
        let cpu = run("cpu", 0);
        let gpu = run("gpu", auto.max(1));
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let cos: Vec<f32> = cpu.iter().zip(&gpu).map(|(a, b)| dot(a, b)).collect();
        let min = cos.iter().cloned().fold(1.0f32, f32::min);
        let mean = cos.iter().sum::<f32>() / cos.len() as f32;
        println!("PROBE cpu vs gpu vectors: cosine min {min:.5}, mean {mean:.5}");
        assert!(min > 0.99, "the GPU must embed the same text to the same place (min cosine {min})");
    }

    /// A file with no blank line in it is ONE paragraph. The old splitter
    /// collected it into a `Vec<char>` first — four bytes a character — and a
    /// large import took the app with it. Chunks must come out the same, and
    /// multi-byte text must not be sliced through a character.
    #[test]
    fn splits_a_single_huge_paragraph_without_collecting_it() {
        let para = "字符".repeat(5_000); // 10k chars, 30k bytes, no blank lines
        let chunks = chunk_text(&para);
        assert!(chunks.len() > 10, "one long paragraph must still split");
        assert!(chunks.iter().all(|c| c.chars().count() <= CHUNK_CHARS + 1));
        // Every chunk is valid text that came out of the original.
        assert!(chunks.iter().all(|c| para.contains(c.as_str())));
        // And the whole paragraph is covered, overlaps aside.
        let joined: String = chunks.concat();
        assert!(joined.chars().count() >= para.chars().count());
    }

    #[test]
    fn chunking_overlaps_and_respects_min_len() {
        let text = "第一段。\n\n第二段内容比较短。\n\n".to_string()
            + &"很长的段落".repeat(400); // forces hard splits
        let chunks = chunk_text(&text);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.chars().count() >= 20));
        assert!(chunks.iter().all(|c| c.chars().count() <= CHUNK_CHARS + 1));
    }

    #[test]
    fn tokenizer_handles_mixed_cjk_ascii() {
        let toks = bm25_tokens("Metal 后端 GPU 加速");
        assert!(toks.contains(&"metal".to_string()));
        assert!(toks.contains(&"后端".to_string())); // CJK bigram
        assert!(toks.contains(&"gpu".to_string()));
    }

    /// Every script a knowledge base might be written in, not just ASCII and
    /// Chinese.
    #[test]
    fn tokenizer_indexes_every_script() {
        let has = |text: &str, tok: &str| bm25_tokens(text).contains(&tok.to_string());
        // Japanese kana was below the old cut-off and never indexed.
        assert!(has("キャッシュの再利用", "キャ"));
        assert!(has("キャッシュの再利用", "再利"));
        // Cyrillic and Greek were dropped entirely.
        assert!(has("Привет, мир", "привет"));
        assert!(has("Καλημέρα κόσμε", "καλημέρα"));
        // An accented letter no longer splits its word.
        assert!(has("A manutenção preventiva", "manutenção"));
        // Combining marks stay inside the word.
        assert!(has("हिन्दी भाषा", "हिन्दी"));
        // Full-width letters meet their ASCII spelling.
        assert!(has("ＧＰＵ加速", "gpu"));
        // Full-width punctuation is punctuation.
        assert!(!bm25_tokens("你好，世界！").iter().any(|t| t.contains('，') || t.contains('！')));
    }

    #[test]
    fn bm25_ranks_relevant_doc_higher() {
        let corpus = vec![
            bm25_tokens("苹果公司发布了新的统一内存架构芯片"),
            bm25_tokens("今天天气很好，适合户外散步"),
            bm25_tokens("统一内存让 GPU 与 CPU 共享同一块物理内存"),
        ];
        let scores = bm25_scores(&corpus, "统一内存 GPU");
        assert!(scores[2] > scores[1]);
        assert!(scores[0] > scores[1]);
    }
}
