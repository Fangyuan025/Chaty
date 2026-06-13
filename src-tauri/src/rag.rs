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
}

static EMBEDDER: Mutex<Option<Embedder>> = Mutex::new(None);

fn embed_model_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("rag")
        .join(EMBED_FILE))
}

fn embedder_start(model_path: &PathBuf) -> Result<Embedder, String> {
    let path = model_path.to_string_lossy().to_string();
    let (tx, rx) = std::sync::mpsc::channel::<EmbedJob>();
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("chaty-embed".into())
        .spawn(move || {
            let backend = match crate::inference::llama_backend_pub() {
                Ok(b) => b,
                Err(e) => {
                    let _ = init_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            // Small model: offload everything; mmap is fine (kept resident).
            let params = LlamaModelParams::default().with_n_gpu_layers(999);
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
                            let take = tokens.len().min(n_ctx as usize - 4);
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
        Ok(Ok(())) => Ok(Embedder { tx }),
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
        *guard = Some(embedder_start(&model_path)?);
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
        *guard = Some(conn);
    }
    f(guard.as_ref().unwrap())
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
        "pdf" => pdf_extract::extract_text(path)
            .map_err(|e| format!("PDF 解析失败 (PDF extraction failed): {e}")),
        _ => {
            // Text-ish files: decode as UTF-8 with GBK fallback.
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
            let cs: Vec<char> = p.chars().collect();
            let mut i = 0;
            while i < cs.len() {
                let end = (i + CHUNK_CHARS).min(cs.len());
                chunks.push(cs[i..end].iter().collect::<String>().trim().to_string());
                if end == cs.len() {
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

fn bm25_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_alphanumeric() {
            word.push(c.to_ascii_lowercase());
        } else {
            if word.len() >= 2 {
                out.push(std::mem::take(&mut word));
            } else {
                word.clear();
            }
            // CJK: index unigrams+bigrams so short Chinese queries match.
            if (c as u32) >= 0x3400 {
                out.push(c.to_string());
                if i + 1 < chars.len() && (chars[i + 1] as u32) >= 0x3400 {
                    let mut bg = String::new();
                    bg.push(c);
                    bg.push(chars[i + 1]);
                    out.push(bg);
                }
            }
        }
        i += 1;
    }
    if word.len() >= 2 {
        out.push(word);
    }
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

#[tauri::command]
pub fn rag_search(app: tauri::AppHandle, query: String, k: Option<usize>) -> Result<Vec<RagHit>, String> {
    let k = k.unwrap_or(6).clamp(1, 12);
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

#[tauri::command]
pub async fn rag_add_document(
    app: tauri::AppHandle,
    path: String,
    on_progress: Channel<RagProgress>,
) -> Result<(), String> {
    // Images go through the same OCR engine as attachments (ocrs, Latin).
    // It's async, so run it before entering the blocking section.
    let ext = std::path::Path::new(&path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let ocr_text = if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif") {
        let _ = on_progress.send(RagProgress { phase: "extract", frac: 0.0 });
        let dir = app
            .path()
            .app_data_dir()
            .map_err(|e| e.to_string())?
            .join("ocr-models");
        Some(
            crate::ocr::ocr_image(dir, path.clone())
                .await
                .map_err(|e| format!("OCR 失败 (OCR failed): {e:#}"))?,
        )
    } else {
        None
    };

    tokio::task::spawn_blocking(move || {
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();
        let _ = on_progress.send(RagProgress { phase: "extract", frac: 0.0 });
        let text = match ocr_text {
            Some(t) => t,
            None => extract_text(&path)?,
        };
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
            conn.execute(
                "INSERT INTO docs(name, path, chunks, created_at) VALUES(?1, ?2, ?3, ?4)",
                params![
                    name,
                    path,
                    chunks.len() as i64,
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

        // Embed in small batches, streaming progress.
        let total = chunks.len();
        for (i, batch) in chunks.chunks(8).enumerate() {
            let embs = embed(&app, batch.to_vec())?;
            with_db(&app, |conn| {
                for (j, (text, emb)) in batch.iter().zip(&embs).enumerate() {
                    conn.execute(
                        "INSERT INTO chunks(doc_id, seq, text, embedding) VALUES(?1, ?2, ?3, ?4)",
                        params![doc_id, (i * 8 + j) as i64, text, vec_to_blob(emb)],
                    )
                    .map_err(|e| e.to_string())?;
                }
                Ok(())
            })?;
            let done = ((i * 8 + batch.len()) as f32 / total as f32).min(1.0);
            let _ = on_progress.send(RagProgress { phase: "embed", frac: done });
        }
        let _ = on_progress.send(RagProgress { phase: "done", frac: 1.0 });
        Ok(())
    })
    .await
    .map_err(|e| format!("索引任务异常 (indexing task panicked): {e}"))?
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RagDoc {
    pub id: i64,
    pub name: String,
    pub chunks: i64,
    pub enabled: bool,
}

#[tauri::command]
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
#[tauri::command]
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

#[tauri::command]
pub fn rag_remove_document(app: tauri::AppHandle, id: i64) -> Result<(), String> {
    with_db(&app, |conn| {
        conn.execute("DELETE FROM chunks WHERE doc_id = ?1", params![id])
            .map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM docs WHERE id = ?1", params![id])
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

    let client = reqwest::Client::builder()
        .user_agent("Chaty-RAG")
        .build()
        .map_err(|e| e.to_string())?;
    let mut last_err = String::new();
    for url in EMBED_URLS {
        let resp = match client.get(*url).send().await.and_then(|r| r.error_for_status()) {
            Ok(r) => r,
            Err(e) => {
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
    use super::*;

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
