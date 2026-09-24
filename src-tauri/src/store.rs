//! Conversation persistence (SQLite via rusqlite, bundled).

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tauri::State;

/// Managed handle to the on-disk database.
pub struct Db(pub Mutex<Connection>);

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub model_path: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub pinned: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct StoredMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: i64,
    /// Image attachment paths (vision models), stored as a JSON array.
    #[serde(default)]
    pub images: Vec<String>,
}

/// JSON-decode the `images` column ('[]' default; tolerate legacy NULL/garbage).
fn images_from_json(s: Option<String>) -> Vec<String> {
    s.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    model_path  TEXT,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    pinned      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    images          TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS code_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    workspace   TEXT,
    data        TEXT NOT NULL DEFAULT '[]',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS code_step_texts (
    session_id  TEXT NOT NULL,
    step_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    PRIMARY KEY (session_id, step_id)
);
CREATE TABLE IF NOT EXISTS image_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    pinned      INTEGER NOT NULL DEFAULT 0,
    draft       TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS image_generations (
    id              TEXT PRIMARY KEY,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT NOT NULL DEFAULT '',
    params          TEXT NOT NULL DEFAULT '{}',
    images          TEXT NOT NULL DEFAULT '[]',
    model           TEXT NOT NULL DEFAULT '',
    family          TEXT NOT NULL DEFAULT '',
    created_at      INTEGER NOT NULL,
    elapsed_ms      INTEGER NOT NULL DEFAULT 0,
    session_id      TEXT NOT NULL DEFAULT '',
    parent_id       TEXT
);
CREATE INDEX IF NOT EXISTS idx_image_generations_created ON image_generations(created_at);
";

pub fn init_db(path: &Path) -> rusqlite::Result<Db> {
    let conn = Connection::open(path)?;
    conn.execute_batch(SCHEMA)?;
    // Migration for DBs created before v0.9.2: add the `pinned` column. Fresh
    // DBs already have it from SCHEMA, so the duplicate-column error is ignored.
    let _ = conn.execute(
        "ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
        [],
    );
    // Migration for DBs created before v1.7: image attachments (vision models).
    let _ = conn.execute(
        "ALTER TABLE messages ADD COLUMN images TEXT NOT NULL DEFAULT '[]'",
        [],
    );
    migrate_image_sessions(&conn);
    // Sweep messages left behind by a conversation that is already gone. No
    // screen can reach one — every read goes through a conversation id — so
    // they were pure weight in the file and a wrong number in Settings → Data.
    // `save_message` no longer creates them; this clears what earlier builds
    // did, and costs nothing once the table is clean.
    if let Ok(n) = conn.execute(
        "DELETE FROM messages
         WHERE conversation_id NOT IN (SELECT id FROM conversations)",
        [],
    ) {
        if n > 0 {
            eprintln!("store: swept {n} message(s) whose conversation no longer exists");
        }
    }
    Ok(Db(Mutex::new(conn)))
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Take the connection back after a panic poisoned its lock, the way the
/// search state already does.
///
/// Treating poisoning as an error made one panic permanent: every later
/// command failed on the poisoning alone, so the app stopped reading and
/// writing conversations for the rest of the session and only a restart
/// brought it back. SQLite finalizes a statement when it drops, so what is
/// behind the lock is a usable connection rather than a half-written
/// structure — and the recovery is logged, because a connection that really
/// is broken should leave a trace instead of a silence.
fn lock_connection(db: &Mutex<Connection>) -> std::sync::MutexGuard<'_, Connection> {
    db.lock().unwrap_or_else(|poisoned| {
        crate::errlog::append_error(
            "store-lock-recovered",
            "the conversation database lock was poisoned by an earlier panic; recovered",
        );
        poisoned.into_inner()
    })
}

fn lock<'a>(db: &'a State<'_, Db>) -> Result<std::sync::MutexGuard<'a, Connection>, String> {
    Ok(lock_connection(&db.0))
}

/// Create or update a conversation (id supplied by the caller).
#[tauri::command]
pub fn save_conversation(
    db: State<'_, Db>,
    id: String,
    title: String,
    model_path: Option<String>,
) -> Result<(), String> {
    let conn = lock(&db)?;
    let now = now_ms();
    conn.execute(
        "INSERT INTO conversations (id, title, model_path, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?4)
         ON CONFLICT(id) DO UPDATE SET title = ?2, model_path = ?3, updated_at = ?4",
        params![id, title, model_path, now],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Append (or replace) a message and bump the conversation's `updated_at`.
#[tauri::command]
pub fn save_message(
    db: State<'_, Db>,
    id: String,
    conversation_id: String,
    role: String,
    content: String,
    images: Option<Vec<String>>,
) -> Result<(), String> {
    let conn = lock(&db)?;
    let now = now_ms();
    let images_json =
        serde_json::to_string(&images.unwrap_or_default()).unwrap_or_else(|_| "[]".into());
    // Only for a conversation that still exists. Cancelling a stream and
    // deleting its conversation in the same breath used to land here *after*
    // the delete — the reply was written back under an id nothing pointed to
    // any more, leaving a message no screen could ever show and an inflated
    // count in Settings → Data. The guard makes that unreachable from any
    // caller, whatever the ordering.
    conn.execute(
        "INSERT OR REPLACE INTO messages (id, conversation_id, role, content, created_at, images)
         SELECT ?1, ?2, ?3, ?4, ?5, ?6
         WHERE EXISTS (SELECT 1 FROM conversations WHERE id = ?2)",
        params![id, conversation_id, role, content, now, images_json],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
        params![now, conversation_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn list_conversations(db: State<'_, Db>) -> Result<Vec<Conversation>, String> {
    let conn = lock(&db)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, title, model_path, created_at, updated_at, pinned
             FROM conversations ORDER BY pinned DESC, updated_at DESC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok(Conversation {
                id: r.get(0)?,
                title: r.get(1)?,
                model_path: r.get(2)?,
                created_at: r.get(3)?,
                updated_at: r.get(4)?,
                pinned: r.get(5)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_messages(db: State<'_, Db>, conversation_id: String) -> Result<Vec<StoredMessage>, String> {
    let conn = lock(&db)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, role, content, created_at, images FROM messages
             WHERE conversation_id = ?1 ORDER BY created_at ASC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![conversation_id], |r| {
            Ok(StoredMessage {
                id: r.get(0)?,
                role: r.get(1)?,
                content: r.get(2)?,
                created_at: r.get(3)?,
                images: images_from_json(r.get(4)?),
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

#[derive(Deserialize)]
pub struct MsgIn {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Vec<String>,
}

/// Replace ALL messages of a conversation with `messages` (in order). Used by
/// edit / regenerate, which truncate the conversation. Runs in one transaction.
#[tauri::command]
pub fn replace_messages(
    db: State<'_, Db>,
    conversation_id: String,
    messages: Vec<MsgIn>,
) -> Result<(), String> {
    let mut conn = lock(&db)?;
    let base = now_ms();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    // Same rule as `save_message`: a conversation deleted while its turn was
    // being regenerated must not get its messages written back underneath it.
    let alive: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM conversations WHERE id = ?1",
            params![conversation_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    if alive == 0 {
        return Ok(());
    }
    tx.execute(
        "DELETE FROM messages WHERE conversation_id = ?1",
        params![conversation_id],
    )
    .map_err(|e| e.to_string())?;
    for (i, m) in messages.iter().enumerate() {
        let images_json = serde_json::to_string(&m.images).unwrap_or_else(|_| "[]".into());
        tx.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at, images)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![m.id, conversation_id, m.role, m.content, base + i as i64, images_json],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.execute(
        "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
        params![base, conversation_id],
    )
    .map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn delete_conversation(db: State<'_, Db>, id: String) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute("DELETE FROM messages WHERE conversation_id = ?1", params![id])
        .map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM conversations WHERE id = ?1", params![id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Delete EVERY conversation and message. Powers Settings → "clear all chats".
#[tauri::command]
pub fn clear_all_conversations(db: State<'_, Db>) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute("DELETE FROM messages", [])
        .map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM conversations", [])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Conversation ids whose message bodies contain `query` (case-insensitive),
/// ordered by most-recently-updated. Powers the sidebar full-text search.
#[tauri::command]
pub fn search_conversations(db: State<'_, Db>, query: String) -> Result<Vec<String>, String> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let pattern = format!("%{}%", q.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
    let conn = lock(&db)?;
    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT m.conversation_id
             FROM messages m
             JOIN conversations c ON c.id = m.conversation_id
             WHERE m.content LIKE ?1 ESCAPE '\\'
             ORDER BY c.updated_at DESC",
        )
        .map_err(|e| e.to_string())?;
    let ids = stmt
        .query_map(params![pattern], |row| row.get::<_, String>(0))
        .map_err(|e| e.to_string())?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(ids)
}

/// Pin or unpin a conversation (pinned ones sort to the top of the sidebar).
#[tauri::command]
pub fn set_conversation_pinned(db: State<'_, Db>, id: String, pinned: bool) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute(
        "UPDATE conversations SET pinned = ?1 WHERE id = ?2",
        params![pinned as i64, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn rename_conversation(db: State<'_, Db>, id: String, title: String) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute(
        "UPDATE conversations SET title = ?1 WHERE id = ?2",
        params![title, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Code-mode sessions (agentic coding). Stored as one JSON blob per session —
// the frontend owns the shape (messages + tool steps); we just persist it.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeSessionMeta {
    pub id: String,
    pub title: String,
    pub workspace: Option<String>,
    pub updated_at: i64,
}

#[tauri::command]
pub fn code_session_save(
    db: State<'_, Db>,
    id: String,
    title: String,
    workspace: Option<String>,
    data: String,
) -> Result<(), String> {
    let conn = lock(&db)?;
    let now = now_ms();
    conn.execute(
        "INSERT INTO code_sessions (id, title, workspace, data, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5)
         ON CONFLICT(id) DO UPDATE SET title = ?2, workspace = ?3, data = ?4, updated_at = ?5",
        params![id, title, workspace, data, now],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn code_session_list(db: State<'_, Db>) -> Result<Vec<CodeSessionMeta>, String> {
    let conn = lock(&db)?;
    let mut stmt = conn
        .prepare("SELECT id, title, workspace, updated_at FROM code_sessions ORDER BY updated_at DESC")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok(CodeSessionMeta {
                id: r.get(0)?,
                title: r.get(1)?,
                workspace: r.get(2)?,
                updated_at: r.get(3)?,
            })
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}

#[tauri::command]
pub fn code_session_load(db: State<'_, Db>, id: String) -> Result<Option<String>, String> {
    let conn = lock(&db)?;
    conn.query_row("SELECT data FROM code_sessions WHERE id = ?1", params![id], |r| {
        r.get::<_, String>(0)
    })
    .map(Some)
    .or_else(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other.to_string()),
    })
}

/// One place in a session's transcript where the words were found. The agent
/// reads its own past this way: what it wrote before a compaction dropped it,
/// and what was said in another session the user pointed at.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SessionHit {
    pub session_id: String,
    pub title: String,
    pub updated_at: i64,
    /// 1-based position of the message inside the session.
    pub turn: usize,
    /// "user", "assistant", or the tool the step called.
    pub role: String,
    pub text: String,
    /// When the hit is inside a tool step: the step's id, so its whole result
    /// can be read back, and whether that step succeeded.
    pub step_id: Option<String>,
    pub status: Option<String>,
}

/// One tool step of a past turn, as `read_history` lists it: what was called,
/// whether it worked, and the handle for reading its whole result.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HistoryStep {
    pub step_id: String,
    pub name: String,
    pub args: String,
    /// "done" | "error" | "denied" — as the card recorded it.
    pub status: String,
    /// Size of the result kept for it, so the reader knows what it is asking
    /// for before it asks.
    pub result_chars: usize,
}

/// One turn of a past session, with its tool steps listed but not spelled out.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HistoryTurn {
    pub turn: usize,
    pub role: String,
    pub text: String,
    pub steps: Vec<HistoryStep>,
}

/// A session read back: the turns asked for, and how many there are in all.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HistoryRead {
    pub session_id: String,
    pub title: String,
    pub updated_at: i64,
    pub total_turns: usize,
    pub turns: Vec<HistoryTurn>,
}

/// What to look for. A model does not write search-box queries: it writes a
/// sentence ("tooltip 悬停延迟 delay 毫秒"), so every word being present is
/// the wrong test — the words are scored instead. Chinese is written without
/// spaces, so a CJK word also contributes its two-character windows, which is
/// how it can match text it does not equal.
fn search_terms(query: &str) -> Vec<Vec<char>> {
    let mut out: Vec<Vec<char>> = Vec::new();
    let mut push = |w: Vec<char>| {
        if w.len() >= 2 && !out.contains(&w) && out.len() < 32 {
            out.push(w);
        }
    };
    // Punctuation comes off the ENDS of a word, never out of its middle: a
    // filter that dropped every dot turned `Login.tsx` into `logintsx`, which
    // is in no transcript anywhere — and file names are what a search of a
    // coding session is mostly made of.
    let edge = |c: char| ",.?!:;，。？！：；、“”'\"()[]<>".contains(c);
    for word in query.split_whitespace() {
        let w: Vec<char> = word
            .trim_matches(edge)
            .chars()
            .flat_map(|c| c.to_lowercase())
            .collect();
        if w.is_empty() {
            continue;
        }
        let cjk = w.iter().any(|c| ('\u{3400}'..='\u{9fff}').contains(c));
        if cjk && w.len() > 2 {
            for win in w.windows(2) {
                push(win.to_vec());
            }
        }
        push(w);
    }
    out
}

fn find_from(hay: &[char], needle: &[char], from: usize) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|&i| hay[i..i + needle.len()] == *needle)
}

/// How well a piece of transcript answers the query, and where its best word
/// was found. Longer words count for more, so a whole phrase outranks the
/// fragments it is made of. None ⇒ not a hit at all.
fn score(text: &[char], terms: &[Vec<char>]) -> Option<(usize, usize)> {
    let mut total = 0;
    let mut best_len = 0;
    let mut at = 0;
    for t in terms {
        if let Some(i) = find_from(text, t, 0) {
            total += t.len();
            if t.len() > best_len {
                best_len = t.len();
                at = i;
            }
        }
    }
    (total > 0).then_some((total, at))
}

/// A readable window around the words that were found.
fn excerpt(chars: &[char], at: usize, want: usize) -> String {
    let lead = want / 4;
    let start = at.saturating_sub(lead);
    let end = (start + want).min(chars.len());
    let body: String = chars[start..end].iter().collect();
    let body = body.split_whitespace().collect::<Vec<_>>().join(" ");
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        body,
        if end < chars.len() { "…" } else { "" }
    )
}

/// The text of a piece, folded for matching. to_lowercase can widen a char
/// (ß → ss); when it does, match the text as written so positions stay true.
fn folded(text: &str) -> (Vec<char>, Vec<char>) {
    let chars: Vec<char> = text.chars().collect();
    let lowered: Vec<char> = chars.iter().flat_map(|c| c.to_lowercase()).collect();
    let hay = if lowered.len() == chars.len() { lowered } else { chars.clone() };
    (chars, hay)
}

/// What a step shows a searcher: the call and what came back.
fn step_args(step: &serde_json::Value) -> String {
    match step["call"]["args"].as_object() {
        Some(map) => map
            .iter()
            .map(|(k, v)| match v.as_str() {
                Some(s) => format!("{k}={s}"),
                None => format!("{k}={v}"),
            })
            .collect::<Vec<_>>()
            .join(" "),
        None => String::new(),
    }
}

/// What a step shows a searcher: the call, whether it worked, and what came
/// back. The step's own id rides along so a hit can be read in full later.
fn step_text(step: &serde_json::Value) -> (String, String, Option<String>, Option<String>) {
    let name = step["call"]["name"].as_str().unwrap_or("tool").to_string();
    let args = step_args(step);
    let result = step["result"].as_str().unwrap_or("");
    let status = step["status"].as_str().unwrap_or("done").to_string();
    let id = step["id"].as_str().map(str::to_string);
    // The CALL is part of the record, not just its output: the tool's name
    // leads the searchable text (so "edit_file Login" finds the edit), and a
    // step that failed says so in both languages (so "bash 失败" finds it).
    let flag = match status.as_str() {
        "error" => " [失败 error failed]",
        "denied" => " [已拒绝 denied]",
        _ => "",
    };
    (name.clone(), format!("{name} {args}{flag}\n{result}"), id, Some(status))
}

/// Everything in one session's transcript worth searching, in the order it
/// was said: the turn it belongs to, who said it, and the words.
type Piece = (usize, String, String, Option<String>, Option<String>);

fn session_pieces(data: &str) -> Vec<Piece> {
    let parsed: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let msgs = match parsed.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for (i, m) in msgs.iter().enumerate() {
        let turn = i + 1;
        let role = m["role"].as_str().unwrap_or("assistant").to_string();
        if let Some(text) = m["text"].as_str().filter(|t| !t.trim().is_empty()) {
            out.push((turn, role.clone(), text.to_string(), None, None));
        }
        if let Some(steps) = m["steps"].as_array() {
            for step in steps {
                let (name, body, id, status) = step_text(step);
                if !body.trim().is_empty() {
                    out.push((turn, name, body, id, status));
                }
            }
        }
    }
    out
}

/// What one session has to say about the query, best first. The score rides
/// along so the whole result can be ordered by it.
fn session_hits(
    id: &str,
    title: &str,
    updated_at: i64,
    data: &str,
    terms: &[Vec<char>],
    cap: usize,
) -> Vec<(usize, SessionHit)> {
    let pieces = session_pieces(data);
    let hit = |turn: usize, role: &str, text: String, step: Option<String>, status: Option<String>| SessionHit {
        session_id: id.to_string(),
        title: title.to_string(),
        updated_at,
        turn,
        role: role.to_string(),
        text,
        step_id: step,
        status,
    };
    // No words: an overview of the session — how it opened and where it got to.
    if terms.is_empty() {
        let brief = |t: &str| {
            let s = t.split_whitespace().collect::<Vec<_>>().join(" ");
            let cut: String = s.chars().take(400).collect();
            if s.chars().count() > 400 { format!("{cut}…") } else { cut }
        };
        let first = pieces.iter().find(|(_, role, _, _, _)| role == "user");
        let last = pieces.iter().rev().find(|(_, role, _, _, _)| role == "assistant");
        return first
            .into_iter()
            .chain(last)
            .map(|(turn, role, text, step, status)| {
                (0, hit(*turn, role, brief(text), step.clone(), status.clone()))
            })
            .collect();
    }
    let mut out: Vec<(usize, SessionHit)> = Vec::new();
    for (turn, role, text, step, status) in pieces {
        let (chars, hay) = folded(&text);
        if let Some((points, at)) = score(&hay, terms) {
            out.push((points, hit(turn, &role, excerpt(&chars, at, 400), step, status)));
        }
    }
    // Best answers first; between equals, the earlier one (a decision is
    // usually made before it is repeated).
    out.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.turn.cmp(&b.1.turn)));
    out.truncate(cap);
    out
}

/// One session read back turn by turn: what was said, and what each turn's
/// tool steps were — the call, whether it worked, and the handle for its whole
/// result. The results themselves are NOT here on purpose: a session's steps
/// hold megabytes between them, and a reader that wants one of them asks for
/// that one (see `code_step_text_get`).
fn session_turns(data: &str, want: Option<usize>, text_cap: usize) -> (usize, Vec<HistoryTurn>) {
    let parsed: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return (0, Vec::new()),
    };
    let msgs = match parsed.as_array() {
        Some(a) => a,
        None => return (0, Vec::new()),
    };
    let mut turns = Vec::new();
    for (i, m) in msgs.iter().enumerate() {
        let turn = i + 1;
        if want.is_some_and(|w| w != turn) {
            continue;
        }
        // One turn asked for is read whole; a whole session is read as far as
        // each message's cap, since the point of that read is the shape of the
        // conversation rather than every word of it.
        let cap = if want.is_some() { text_cap.max(4000) } else { text_cap };
        let raw = m["text"].as_str().unwrap_or("");
        let text = if raw.chars().count() > cap {
            let cut: String = raw.chars().take(cap).collect();
            format!("{cut}…")
        } else {
            raw.to_string()
        };
        let steps = m["steps"]
            .as_array()
            .map(|list| {
                list.iter()
                    .map(|step| HistoryStep {
                        step_id: step["id"].as_str().unwrap_or("").to_string(),
                        name: step["call"]["name"].as_str().unwrap_or("tool").to_string(),
                        args: step_args(step),
                        status: step["status"].as_str().unwrap_or("done").to_string(),
                        result_chars: step["result"].as_str().map(|r| r.chars().count()).unwrap_or(0),
                    })
                    .collect()
            })
            .unwrap_or_default();
        turns.push(HistoryTurn {
            turn,
            role: m["role"].as_str().unwrap_or("assistant").to_string(),
            text,
            steps,
        });
    }
    (msgs.len(), turns)
}

/// Read a past Code session: the whole thing, or one turn of it.
#[tauri::command]
pub fn code_session_read(
    db: State<'_, Db>,
    session_id: String,
    turn: Option<usize>,
    text_cap: Option<usize>,
) -> Result<Option<HistoryRead>, String> {
    let conn = lock(&db)?;
    let row = conn
        .query_row(
            "SELECT id, title, updated_at, data FROM code_sessions WHERE id = ?1",
            params![session_id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, String>(3)?,
                ))
            },
        )
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(other.to_string()),
        })?;
    let Some((id, title, updated_at, data)) = row else { return Ok(None) };
    let (total, turns) = session_turns(&data, turn, text_cap.unwrap_or(1200));
    Ok(Some(HistoryRead { session_id: id, title, updated_at, total_turns: total, turns }))
}

/// Search past Code sessions: this one before a compaction dropped it, one the
/// user pointed at, or all of them. An empty query with a session asks what
/// that session was about.
#[tauri::command]
pub fn code_session_search(
    db: State<'_, Db>,
    query: String,
    session_id: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<SessionHit>, String> {
    let terms = search_terms(&query);
    let limit = limit.unwrap_or(8).clamp(1, 30) as usize;
    let conn = lock(&db)?;
    let mut rows: Vec<(String, String, i64, String)> = Vec::new();
    if let Some(id) = session_id.as_deref() {
        let row = conn
            .query_row(
                "SELECT id, title, updated_at, data FROM code_sessions WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.to_string()),
            })?;
        rows.extend(row);
    } else {
        // Every session is read: a query is scored, not matched word for word,
        // so no single word can be required of the blob up front.
        let mut stmt = conn
            .prepare("SELECT id, title, updated_at, data FROM code_sessions ORDER BY updated_at DESC")
            .map_err(|e| e.to_string())?;
        rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .and_then(|it| it.collect())
            .map_err(|e| e.to_string())?;
    }
    // One session may not fill the answer on its own when several were asked.
    let per_session = if session_id.is_some() { limit } else { 3.min(limit) };
    let mut scored: Vec<(usize, i64, SessionHit)> = Vec::new();
    for (id, title, updated_at, data) in rows {
        // A session none of the words appear in at all is skipped before its
        // JSON is parsed — the cost of searching everything is then reading
        // the blobs, not decoding them.
        if !terms.is_empty() {
            let low = data.to_lowercase();
            if !terms.iter().any(|t| low.contains(&t.iter().collect::<String>())) {
                continue;
            }
        }
        for (points, hit) in session_hits(&id, &title, updated_at, &data, &terms, per_session) {
            scored.push((points, updated_at, hit));
        }
    }
    // The best answers first, and among equals the most recent session.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.turn.cmp(&b.2.turn)));
    let mut out: Vec<SessionHit> = scored.into_iter().map(|(_, _, h)| h).collect();
    out.truncate(limit);
    Ok(out)
}

#[tauri::command]
pub fn code_session_delete(db: State<'_, Db>, id: String) -> Result<(), String> {
    // Its background jobs go with it: the running ones stopped, the history dropped.
    crate::agent::bg_forget_session(&id);
    let conn = lock(&db)?;
    delete_code_session_rows(&conn, &id).map_err(|e| e.to_string())
}

/// A session's row and the step texts kept for it.
fn delete_code_session_rows(conn: &Connection, id: &str) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM code_sessions WHERE id = ?1", params![id])?;
    conn.execute("DELETE FROM code_step_texts WHERE session_id = ?1", params![id])?;
    Ok(())
}

/// The exact text the model was given for one tool step. The step card in
/// the session keeps a copy trimmed for the renderer; this one is what the
/// card shows when opened. It lives here, apart from the session, because a
/// session holding every step's whole result — 384 KB for each large file
/// read — grew until the webview's renderer was killed.
#[tauri::command]
pub fn code_step_text_put(
    db: State<'_, Db>,
    session_id: String,
    step_id: String,
    text: String,
) -> Result<(), String> {
    let conn = lock(&db)?;
    put_step_text(&conn, &session_id, &step_id, &text).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn code_step_text_get(
    db: State<'_, Db>,
    session_id: String,
    step_id: String,
) -> Result<Option<String>, String> {
    let conn = lock(&db)?;
    get_step_text(&conn, &session_id, &step_id).map_err(|e| e.to_string())
}

fn put_step_text(conn: &Connection, session_id: &str, step_id: &str, text: &str) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO code_step_texts (session_id, step_id, text) VALUES (?1, ?2, ?3)",
        params![session_id, step_id, text],
    )?;
    Ok(())
}

fn get_step_text(conn: &Connection, session_id: &str, step_id: &str) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT text FROM code_step_texts WHERE session_id = ?1 AND step_id = ?2",
        params![session_id, step_id],
        |r| r.get::<_, String>(0),
    )
    .map(Some)
    .or_else(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    })
}

// ---- Image sessions (the image studio's conversations) ----
//
// The chat's shape, for pictures: a session is a conversation, and each round
// in it — a prompt and the pictures it made — is a generation. A session is
// not tied to a model: switching models carries on in the same one, each
// round remembering which model drew it. Only the image studio shows them.

/// A generation table from before sessions (an early build of the studio)
/// gains the columns, and each of its rounds becomes a session of its own so
/// nothing already drawn goes out of reach.
fn migrate_image_sessions(conn: &Connection) {
    let _ = conn.execute("ALTER TABLE image_generations ADD COLUMN session_id TEXT NOT NULL DEFAULT ''", []);
    let _ = conn.execute("ALTER TABLE image_generations ADD COLUMN parent_id TEXT", []);
    let _ = conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_generations_session ON image_generations(session_id, created_at)",
        [],
    );
    let _ = conn.execute(
        "INSERT OR IGNORE INTO image_sessions (id, title, created_at, updated_at)
         SELECT id, substr(prompt, 1, 40), created_at, created_at
         FROM image_generations WHERE session_id = ''",
        [],
    );
    let _ = conn.execute("UPDATE image_generations SET session_id = id WHERE session_id = ''", []);
}

/// One picture of a generation, on disk.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ImageItem {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub seed: i64,
}

/// One round of a session: the prompt, the settings it ran with, and the
/// pictures it made.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ImageRecord {
    pub id: String,
    pub prompt: String,
    pub negative_prompt: String,
    /// The request as it was sent (size, steps, CFG, sampler, seed, reference
    /// picture …), so a picture can be made again or varied.
    pub params: serde_json::Value,
    pub images: Vec<ImageItem>,
    pub model: String,
    pub family: String,
    pub created_at: i64,
    pub elapsed_ms: i64,
    #[serde(default)]
    pub session_id: String,
    /// The round whose picture this one started from (multi-turn editing).
    #[serde(default)]
    pub parent_id: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ImageSession {
    pub id: String,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub pinned: bool,
}

/// A session opened: its rounds in order, and the unsent prompt it was left
/// with (the frontend owns that shape, as it does a code session's).
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ImageSessionData {
    pub session: ImageSession,
    pub draft: String,
    pub records: Vec<ImageRecord>,
}

const IMAGE_COLS: &str =
    "id, prompt, negative_prompt, params, images, model, family, created_at, elapsed_ms, session_id, parent_id";

fn image_row(r: &rusqlite::Row) -> rusqlite::Result<ImageRecord> {
    Ok(ImageRecord {
        id: r.get(0)?,
        prompt: r.get(1)?,
        negative_prompt: r.get(2)?,
        params: serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or(serde_json::Value::Null),
        images: serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or_default(),
        model: r.get(5)?,
        family: r.get(6)?,
        created_at: r.get(7)?,
        elapsed_ms: r.get(8)?,
        session_id: r.get(9)?,
        parent_id: r.get(10)?,
    })
}

/// Rounds matching `filter` (a WHERE clause over one parameter, or none),
/// oldest first — the order a session reads in.
fn image_records(conn: &Connection, filter: Option<(&str, &str)>) -> rusqlite::Result<Vec<ImageRecord>> {
    let sql = format!(
        "SELECT {IMAGE_COLS} FROM image_generations {} ORDER BY created_at ASC",
        filter.map(|(w, _)| format!("WHERE {w}")).unwrap_or_default()
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = match filter {
        Some((_, v)) => stmt.query_map(params![v], image_row)?.collect(),
        None => stmt.query_map([], image_row)?.collect(),
    };
    rows
}

/// Keep a finished round — only in a session that still exists, like a chat
/// reply (a session deleted while its picture was being drawn must not come
/// back). Returns whether it was kept.
fn image_record_insert_conn(conn: &Connection, r: &ImageRecord) -> rusqlite::Result<bool> {
    let n = conn.execute(
        &format!(
            "INSERT OR REPLACE INTO image_generations ({IMAGE_COLS})
             SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
             WHERE EXISTS (SELECT 1 FROM image_sessions WHERE id = ?10)"
        ),
        params![
            r.id,
            r.prompt,
            r.negative_prompt,
            r.params.to_string(),
            serde_json::to_string(&r.images).unwrap_or_else(|_| "[]".into()),
            r.model,
            r.family,
            r.created_at,
            r.elapsed_ms,
            r.session_id,
            r.parent_id,
        ],
    )?;
    if n > 0 {
        conn.execute(
            "UPDATE image_sessions SET updated_at = ?1 WHERE id = ?2",
            params![now_ms(), r.session_id],
        )?;
    }
    Ok(n > 0)
}

pub(crate) fn image_record_insert(db: &Db, r: &ImageRecord) -> Result<bool, String> {
    image_record_insert_conn(&lock_connection(&db.0), r).map_err(|e| e.to_string())
}

fn image_session_upsert(conn: &Connection, id: &str, title: &str) -> rusqlite::Result<()> {
    let now = now_ms();
    conn.execute(
        "INSERT INTO image_sessions (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?3)
         ON CONFLICT(id) DO UPDATE SET title = ?2, updated_at = ?3",
        params![id, title, now],
    )?;
    Ok(())
}

/// A session for a round that came without one (a caller other than the
/// studio): named after its prompt, so the round is never out of reach.
pub(crate) fn image_session_ensure(db: &Db, id: &str, title: &str) -> Result<(), String> {
    let conn = lock_connection(&db.0);
    conn.execute(
        "INSERT OR IGNORE INTO image_sessions (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?3)",
        params![id, title, now_ms()],
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

fn image_sessions(conn: &Connection, id: Option<&str>) -> rusqlite::Result<Vec<ImageSession>> {
    let sql = format!(
        "SELECT id, title, created_at, updated_at, pinned FROM image_sessions {}
         ORDER BY pinned DESC, updated_at DESC",
        if id.is_some() { "WHERE id = ?1" } else { "" }
    );
    let mut stmt = conn.prepare(&sql)?;
    let map = |r: &rusqlite::Row| {
        Ok(ImageSession {
            id: r.get(0)?,
            title: r.get(1)?,
            created_at: r.get(2)?,
            updated_at: r.get(3)?,
            pinned: r.get(4)?,
        })
    };
    let rows = match id {
        Some(id) => stmt.query_map(params![id], map)?.collect(),
        None => stmt.query_map([], map)?.collect(),
    };
    rows
}

fn image_session_delete_conn(conn: &Connection, id: &str) -> rusqlite::Result<Vec<ImageRecord>> {
    let recs = image_records(conn, Some(("session_id = ?1", id)))?;
    conn.execute("DELETE FROM image_generations WHERE session_id = ?1", params![id])?;
    conn.execute("DELETE FROM image_sessions WHERE id = ?1", params![id])?;
    Ok(recs)
}

/// Sessions a search finds by their prompts (titles are matched in the
/// sidebar), most recent first.
fn image_session_search_conn(conn: &Connection, query: &str) -> rusqlite::Result<Vec<String>> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let pattern = format!("%{}%", q.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
    let mut stmt = conn.prepare(
        "SELECT DISTINCT g.session_id FROM image_generations g
         JOIN image_sessions s ON s.id = g.session_id
         WHERE g.prompt LIKE ?1 ESCAPE '\\' OR g.negative_prompt LIKE ?1 ESCAPE '\\'
         ORDER BY s.updated_at DESC",
    )?;
    let ids = stmt.query_map(params![pattern], |r| r.get::<_, String>(0))?.collect();
    ids
}

fn remove_image_files(records: &[ImageRecord]) {
    for r in records {
        for im in &r.images {
            let _ = std::fs::remove_file(&im.path);
        }
    }
}

/// Create or rename a session (id supplied by the caller), like
/// `save_conversation`.
#[tauri::command]
pub fn image_session_save(db: State<'_, Db>, id: String, title: String) -> Result<(), String> {
    let conn = lock(&db)?;
    image_session_upsert(&conn, &id, &title).map_err(|e| e.to_string())
}

/// Every session, pinned first, then most recently used.
#[tauri::command]
pub fn image_session_list(db: State<'_, Db>) -> Result<Vec<ImageSession>, String> {
    let conn = lock(&db)?;
    image_sessions(&conn, None).map_err(|e| e.to_string())
}

/// One session with its rounds, oldest first, and its unsent draft.
#[tauri::command]
pub fn image_session_get(db: State<'_, Db>, id: String) -> Result<Option<ImageSessionData>, String> {
    let conn = lock(&db)?;
    let Some(session) = image_sessions(&conn, Some(&id)).map_err(|e| e.to_string())?.into_iter().next() else {
        return Ok(None);
    };
    let draft: String = conn
        .query_row("SELECT draft FROM image_sessions WHERE id = ?1", params![id], |r| r.get(0))
        .map_err(|e| e.to_string())?;
    let records = image_records(&conn, Some(("session_id = ?1", &id))).map_err(|e| e.to_string())?;
    Ok(Some(ImageSessionData { session, draft, records }))
}

/// Keep the prompt a session was left with. Typing does not move the session
/// up the list — only a new round does.
#[tauri::command]
pub fn image_session_draft(db: State<'_, Db>, id: String, draft: String) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute("UPDATE image_sessions SET draft = ?1 WHERE id = ?2", params![draft, id])
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn image_session_rename(db: State<'_, Db>, id: String, title: String) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute("UPDATE image_sessions SET title = ?1 WHERE id = ?2", params![title, id])
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn image_session_set_pinned(db: State<'_, Db>, id: String, pinned: bool) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute("UPDATE image_sessions SET pinned = ?1 WHERE id = ?2", params![pinned as i64, id])
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Delete a session and its rounds; `delete_files` also removes the pictures
/// it made (never a reference picture brought in from elsewhere).
#[tauri::command]
pub fn image_session_delete(db: State<'_, Db>, id: String, delete_files: bool) -> Result<(), String> {
    let conn = lock(&db)?;
    let recs = image_session_delete_conn(&conn, &id).map_err(|e| e.to_string())?;
    drop(conn);
    if delete_files {
        remove_image_files(&recs);
    }
    Ok(())
}

#[tauri::command]
pub fn image_session_search(db: State<'_, Db>, query: String) -> Result<Vec<String>, String> {
    let conn = lock(&db)?;
    image_session_search_conn(&conn, &query).map_err(|e| e.to_string())
}

/// Delete one round of a session; `delete_files` also removes its pictures.
#[tauri::command]
pub fn image_generation_delete(db: State<'_, Db>, id: String, delete_files: bool) -> Result<(), String> {
    let conn = lock(&db)?;
    let recs = image_records(&conn, Some(("id = ?1", &id))).map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM image_generations WHERE id = ?1", params![id])
        .map_err(|e| e.to_string())?;
    drop(conn);
    if delete_files {
        remove_image_files(&recs);
    }
    Ok(())
}

/// Delete every session and round; `delete_files` also removes the pictures.
#[tauri::command]
pub fn image_history_clear(db: State<'_, Db>, delete_files: bool) -> Result<(), String> {
    let conn = lock(&db)?;
    let recs = if delete_files { image_records(&conn, None).map_err(|e| e.to_string())? } else { Vec::new() };
    conn.execute("DELETE FROM image_generations", []).map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM image_sessions", []).map_err(|e| e.to_string())?;
    drop(conn);
    remove_image_files(&recs);
    Ok(())
}

/// Aggregate counters for the Settings → Data statistics panel.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DataStats {
    pub conversations: i64,
    pub messages: i64,
    pub code_sessions: i64,
    /// Pictures made in the image studio.
    pub images: i64,
    pub db_bytes: u64,
}

#[tauri::command]
pub fn data_stats(app: tauri::AppHandle, db: State<'_, Db>) -> Result<DataStats, String> {
    let conn = lock(&db)?;
    let count = |sql: &str| -> Result<i64, String> {
        conn.query_row(sql, [], |r| r.get(0)).map_err(|e| e.to_string())
    };
    let conversations = count("SELECT COUNT(*) FROM conversations")?;
    // Reachable messages only — a stray row whose conversation is gone is not
    // something the user has, so counting it just makes the number wrong.
    let messages = count(
        "SELECT COUNT(*) FROM messages m
         JOIN conversations c ON c.id = m.conversation_id",
    )?;
    let code_sessions = count("SELECT COUNT(*) FROM code_sessions").unwrap_or(0);
    let images = conn
        .prepare("SELECT images FROM image_generations")
        .and_then(|mut st| {
            st.query_map([], |r| r.get::<_, String>(0))
                .map(|rows| rows.flatten().map(|j| serde_json::from_str::<Vec<ImageItem>>(&j).map(|v| v.len() as i64).unwrap_or(0)).sum())
        })
        .unwrap_or(0);
    drop(conn);
    // The database is three files in WAL mode, and the log routinely outgrows
    // the main one — reporting only `chaty.db` understated what it occupies.
    let db_bytes = tauri::Manager::path(&app)
        .app_data_dir()
        .ok()
        .map(|d| {
            ["chaty.db", "chaty.db-wal", "chaty.db-shm"]
                .iter()
                .filter_map(|f| std::fs::metadata(d.join(f)).ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    Ok(DataStats { conversations, messages, code_sessions, images, db_bytes })
}

#[cfg(test)]
mod tests {
    use rusqlite::{params, Connection};
    use super::{search_terms, session_hits};

    /// A session as the frontend stores one: messages, and the tool steps
    /// under them.
    fn transcript() -> String {
        serde_json::json!([
            { "role": "user", "text": "登录页面的密码框要支持粘贴" },
            {
                "role": "assistant",
                "text": "改好了,粘贴事件不再被拦截。",
                "steps": [
                    {
                        "id": "s1",
                        "status": "done",
                        "call": { "name": "edit_file", "args": { "path": "src/Login.tsx" } },
                        "result": "edited src/Login.tsx (+3 −1)"
                    },
                    {
                        "id": "s2",
                        "status": "error",
                        "call": { "name": "bash", "args": { "command": "npm test" } },
                        "result": "12 passed"
                    }
                ]
            },
            { "role": "user", "text": "顺便把 tooltip 的悬停延迟定成 300 毫秒" },
            { "role": "user", "text": "动画时长保持 150 毫秒不变" }
        ])
        .to_string()
    }

    fn hits(query: &str, cap: usize) -> Vec<(usize, super::SessionHit)> {
        session_hits("s1", "登录页", 10, &transcript(), &search_terms(query), cap)
    }

    #[test]
    fn a_session_is_searched_by_its_words() {
        let found = hits("粘贴", 8);
        assert_eq!(found.len(), 2, "{found:#?}");
        assert_eq!(found[0].1.turn, 1);
        assert_eq!(found[0].1.role, "user");
        assert!(found[0].1.text.contains("密码框要支持粘贴"), "{}", found[0].1.text);
        assert_eq!(found[1].1.role, "assistant");
    }

    /// The failure a real model found: it searches with a sentence, not with
    /// the words the transcript happens to use. Every word being present is
    /// the wrong test — the best-matching lines come back, and the one that
    /// matches most comes first.
    #[test]
    fn a_sentence_finds_the_line_it_is_about() {
        let found = hits("tooltip 悬停延迟 delay 毫秒 ms", 8);
        assert!(!found.is_empty(), "a natural query found nothing");
        assert!(found[0].1.text.contains("300 毫秒"), "{}", found[0].1.text);
        // The line the query is about outranks one that merely shares a word.
        assert!(found.len() >= 2, "{found:#?}");
        assert!(found[0].0 > found[1].0, "{found:#?}");
        assert!(found[1].1.text.contains("150 毫秒"), "{}", found[1].1.text);
    }

    /// A tool step is part of the transcript: what the command was and what it
    /// printed.
    #[test]
    fn a_step_is_part_of_the_transcript() {
        let found = hits("npm test", 8);
        assert_eq!(found[0].1.role, "bash", "{found:#?}");
        assert!(found[0].1.text.contains("12 passed"), "{}", found[0].1.text);
        assert!(hits("cargo clippy", 8).is_empty());
    }

    /// Case folds, the excerpt is a window around the words rather than the
    /// whole file, and a cap is a cap.
    #[test]
    fn an_excerpt_is_a_window_around_the_words() {
        let long = format!("{}needle{}", "a ".repeat(400), " b".repeat(400));
        let data = serde_json::json!([{ "role": "user", "text": long }]).to_string();
        let found = session_hits("s1", "t", 10, &data, &search_terms("NEEDLE"), 8);
        assert_eq!(found.len(), 1);
        assert!(found[0].1.text.contains("needle"), "{}", found[0].1.text);
        assert!(found[0].1.text.starts_with('…') && found[0].1.text.ends_with('…'));
        assert!(found[0].1.text.chars().count() < 500, "{}", found[0].1.text.chars().count());
        assert_eq!(hits("粘贴", 1).len(), 1);
    }

    /// No words asks what a session was about: how it opened and where it got to.
    #[test]
    fn an_empty_query_asks_what_a_session_was_about() {
        let found = hits("  ", 8);
        assert_eq!(found.len(), 2, "{found:#?}");
        assert_eq!(found[0].1.role, "user");
        assert!(found[0].1.text.contains("密码框"));
        assert_eq!(found[1].1.role, "assistant");
        assert!(found[1].1.text.contains("粘贴事件"));
    }

    /// A hit inside a tool step says which step it was and whether that step
    /// worked, so the reader can go and read the whole of it.
    #[test]
    fn a_hit_in_a_step_carries_the_step() {
        let found = hits("npm test", 8);
        assert_eq!(found[0].1.role, "bash");
        assert_eq!(found[0].1.step_id.as_deref(), Some("s2"));
        assert_eq!(found[0].1.status.as_deref(), Some("error"));
        // A hit in what someone SAID has no step to read.
        let said = hits("粘贴", 8);
        assert!(said.iter().all(|(_, h)| h.step_id.is_none()));
    }

    /// The call itself is searchable, not only what it printed: by tool name,
    /// by its arguments, and by whether it failed.
    #[test]
    fn a_tool_call_is_part_of_the_record() {
        let byName = hits("edit_file", 8);
        assert_eq!(byName[0].1.role, "edit_file", "{byName:#?}");
        assert_eq!(byName[0].1.step_id.as_deref(), Some("s1"));
        let byArgs = hits("Login.tsx", 8);
        assert!(byArgs.iter().any(|(_, h)| h.role == "edit_file"), "{byArgs:#?}");
        // The failed step is findable as a failure, in either language.
        for q in ["失败", "failed"] {
            let failures = hits(q, 8);
            assert!(
                failures.iter().any(|(_, h)| h.status.as_deref() == Some("error")),
                "{q}: {failures:#?}"
            );
        }
    }

    /// A session read back: every turn, every step named with its outcome —
    /// and not one step's result, which is what keeps this readable.
    #[test]
    fn a_session_reads_back_turn_by_turn() {
        let (total, turns) = super::session_turns(&transcript(), None, 1200);
        assert_eq!(total, 4);
        assert_eq!(turns.len(), 4);
        assert_eq!(turns[1].role, "assistant");
        assert_eq!(turns[1].steps.len(), 2);
        assert_eq!(turns[1].steps[0].name, "edit_file");
        assert_eq!(turns[1].steps[0].args, "path=src/Login.tsx");
        assert_eq!(turns[1].steps[0].status, "done");
        assert_eq!(turns[1].steps[1].status, "error");
        assert!(turns[1].steps[1].result_chars > 0);
        // One turn, read on its own.
        let (total, one) = super::session_turns(&transcript(), Some(3), 1200);
        assert_eq!(total, 4);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].turn, 3);
        assert!(one[0].text.contains("tooltip"));
    }

    /// A long message is cut for a whole-session read and kept for a single
    /// turn — the first answers "what happened", the second "what exactly".
    #[test]
    fn a_whole_session_is_read_shorter_than_one_turn() {
        let long = "x".repeat(6000);
        let data = serde_json::json!([{ "role": "user", "text": long }]).to_string();
        let (_, all) = super::session_turns(&data, None, 1200);
        assert_eq!(all[0].text.chars().count(), 1201); // 1200 + the ellipsis
        let (_, one) = super::session_turns(&data, Some(1), 1200);
        assert_eq!(one[0].text.chars().count(), 4001);
    }

    /// A session saved by an older version — or half-written — is skipped, not
    /// a failed search.
    #[test]
    fn a_session_that_cannot_be_read_is_skipped() {
        for data in ["", "{}", "[{\"role\":\"user\"}]", "not json at all"] {
            assert!(session_hits("s1", "t", 10, data, &search_terms("anything"), 8).is_empty());
        }
    }

    /// One panic while a query held the lock used to end persistence for the
    /// session: every later command failed on the poisoning rather than on
    /// anything wrong with the database underneath it.
    #[test]
    fn a_poisoned_database_lock_is_taken_back() {
        use std::sync::Mutex;
        let db = Mutex::new(Connection::open_in_memory().expect("in-memory db"));
        db.lock()
            .unwrap()
            .execute("CREATE TABLE t (v INTEGER)", [])
            .expect("create");

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = db.lock().unwrap();
            panic!("a query gave up while holding the connection");
        }));
        assert!(panicked.is_err(), "the fixture must actually panic");
        assert!(db.lock().is_err(), "and must actually poison the lock");

        let conn = super::lock_connection(&db);
        conn.execute("INSERT INTO t VALUES (1)", []).expect("still writable");
        let n: i64 = conn
            .query_row("SELECT count(*) FROM t", [], |r| r.get(0))
            .expect("still readable");
        assert_eq!(n, 1);
    }

    /// The schema plus the migrations `init_db` applies, on an in-memory DB.
    fn db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(super::SCHEMA).unwrap();
        conn
    }

    /// The text a step's model was given is kept per session, and goes when
    /// the session does.
    #[test]
    fn step_texts_are_kept_per_session_and_deleted_with_it() {
        let conn = db();
        super::put_step_text(&conn, "s1", "a", "first").unwrap();
        super::put_step_text(&conn, "s1", "a", "second").unwrap(); // a re-sent step replaces
        super::put_step_text(&conn, "s2", "a", "other").unwrap();
        assert_eq!(super::get_step_text(&conn, "s1", "a").unwrap().as_deref(), Some("second"));
        assert_eq!(super::get_step_text(&conn, "s1", "b").unwrap(), None);

        super::delete_code_session_rows(&conn, "s1").unwrap();
        assert_eq!(super::get_step_text(&conn, "s1", "a").unwrap(), None);
        assert_eq!(super::get_step_text(&conn, "s2", "a").unwrap().as_deref(), Some("other"));
    }

    fn conv(conn: &Connection, id: &str) {
        conn.execute(
            "INSERT INTO conversations (id, title, model_path, created_at, updated_at)
             VALUES (?1, 't', NULL, 0, 0)",
            params![id],
        )
        .unwrap();
    }

    /// The write `save_message` performs: a message only lands when its
    /// conversation is still there. Cancelling a stream and deleting the
    /// conversation in the same breath used to save the reply afterwards,
    /// leaving a row no screen could reach.
    fn save(conn: &Connection, id: &str, conv_id: &str) -> usize {
        conn.execute(
            "INSERT OR REPLACE INTO messages (id, conversation_id, role, content, created_at, images)
             SELECT ?1, ?2, 'assistant', 'hi', 0, '[]'
             WHERE EXISTS (SELECT 1 FROM conversations WHERE id = ?2)",
            params![id, conv_id],
        )
        .unwrap()
    }

    #[test]
    fn message_never_outlives_its_conversation() {
        let conn = db();
        conv(&conn, "c1");
        assert_eq!(save(&conn, "m1", "c1"), 1, "live conversation accepts the message");

        conn.execute("DELETE FROM messages WHERE conversation_id = 'c1'", []).unwrap();
        conn.execute("DELETE FROM conversations WHERE id = 'c1'", []).unwrap();
        assert_eq!(save(&conn, "m2", "c1"), 0, "a deleted conversation takes nothing");

        let left: i64 =
            conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0)).unwrap();
        assert_eq!(left, 0, "no orphan may be created");
    }

    fn image_round(id: &str, session: &str, prompt: &str, at: i64) -> super::ImageRecord {
        super::ImageRecord {
            id: id.into(),
            prompt: prompt.into(),
            negative_prompt: String::new(),
            params: serde_json::json!({ "steps": 8 }),
            images: vec![super::ImageItem { path: format!("/tmp/{id}.png"), width: 512, height: 512, seed: 1 }],
            model: "z-image".into(),
            family: "z-image".into(),
            created_at: at,
            elapsed_ms: 10,
            session_id: session.into(),
            parent_id: None,
        }
    }

    /// Rounds drawn before sessions existed each become a session of their
    /// own, and stay readable.
    #[test]
    fn early_image_rounds_become_sessions() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE image_generations (
                id TEXT PRIMARY KEY, prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL DEFAULT '',
                params TEXT NOT NULL DEFAULT '{}', images TEXT NOT NULL DEFAULT '[]',
                model TEXT NOT NULL DEFAULT '', family TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL, elapsed_ms INTEGER NOT NULL DEFAULT 0);
             INSERT INTO image_generations (id, prompt, created_at) VALUES ('g1', 'a lighthouse at dusk', 5);",
        )
        .unwrap();
        conn.execute_batch(super::SCHEMA).unwrap();
        super::migrate_image_sessions(&conn);
        super::migrate_image_sessions(&conn); // idempotent

        let sessions = super::image_sessions(&conn, None).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "g1");
        assert_eq!(sessions[0].title, "a lighthouse at dusk");
        let rounds = super::image_records(&conn, Some(("session_id = ?1", "g1"))).unwrap();
        assert_eq!(rounds.len(), 1);
        assert_eq!(rounds[0].prompt, "a lighthouse at dusk");
    }

    /// A round is kept only in a session that still exists, moves the session
    /// up, and goes with it; its prompt finds it.
    #[test]
    fn an_image_round_lives_and_dies_with_its_session() {
        let conn = db();
        super::migrate_image_sessions(&conn);
        assert!(!super::image_record_insert_conn(&conn, &image_round("r0", "gone", "x", 1)).unwrap());

        super::image_session_upsert(&conn, "s1", "cats").unwrap();
        super::image_session_upsert(&conn, "s2", "dogs").unwrap();
        conn.execute("UPDATE image_sessions SET updated_at = 1", []).unwrap();
        assert!(super::image_record_insert_conn(&conn, &image_round("r1", "s1", "a ginger cat", 10)).unwrap());
        let mut second = image_round("r2", "s1", "make it wear a hat", 20);
        second.parent_id = Some("r1".into());
        assert!(super::image_record_insert_conn(&conn, &second).unwrap());

        let list = super::image_sessions(&conn, None).unwrap();
        assert_eq!(list[0].id, "s1", "a new round moves its session to the top");
        let rounds = super::image_records(&conn, Some(("session_id = ?1", "s1"))).unwrap();
        assert_eq!(rounds.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["r1", "r2"], "oldest first");
        assert_eq!(rounds[1].parent_id.as_deref(), Some("r1"));

        assert_eq!(super::image_session_search_conn(&conn, "HAT").unwrap(), ["s1"]);
        assert!(super::image_session_search_conn(&conn, "dog").unwrap().is_empty(), "titles are matched by the sidebar");
        assert!(super::image_session_search_conn(&conn, "100%").unwrap().is_empty());

        let gone = super::image_session_delete_conn(&conn, "s1").unwrap();
        assert_eq!(gone.len(), 2, "the deleted rounds come back for their files");
        let left: i64 = conn.query_row("SELECT COUNT(*) FROM image_generations", [], |r| r.get(0)).unwrap();
        assert_eq!(left, 0);
        assert_eq!(super::image_sessions(&conn, None).unwrap().len(), 1);
    }

    /// The statistics panel counts what the user can actually open, and the
    /// startup sweep clears rows earlier builds stranded.
    #[test]
    fn stats_count_reachable_messages_and_sweep_clears_the_rest() {
        let conn = db();
        conv(&conn, "c1");
        save(&conn, "m1", "c1");
        // An orphan as an older build would have left it.
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at, images)
             VALUES ('m2', 'gone', 'assistant', 'x', 0, '[]')",
            [],
        )
        .unwrap();

        let raw: i64 = conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0)).unwrap();
        let reachable: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(raw, 2, "the table still holds the stranded row");
        assert_eq!(reachable, 1, "but only one message is reachable — that is the count to show");

        let swept = conn
            .execute(
                "DELETE FROM messages WHERE conversation_id NOT IN (SELECT id FROM conversations)",
                [],
            )
            .unwrap();
        assert_eq!(swept, 1, "the sweep takes the stranded row");
        let after: i64 = conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0)).unwrap();
        assert_eq!(after, reachable, "raw and reachable agree once the table is clean");
    }
}
