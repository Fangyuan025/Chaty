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
    Ok(Db(Mutex::new(conn)))
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn lock<'a>(db: &'a State<'_, Db>) -> Result<std::sync::MutexGuard<'a, Connection>, String> {
    db.0.lock().map_err(|e| e.to_string())
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
    conn.execute(
        "INSERT OR REPLACE INTO messages (id, conversation_id, role, content, created_at, images)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
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

#[tauri::command]
pub fn code_session_delete(db: State<'_, Db>, id: String) -> Result<(), String> {
    let conn = lock(&db)?;
    conn.execute("DELETE FROM code_sessions WHERE id = ?1", params![id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Aggregate counters for the Settings → Data statistics panel.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DataStats {
    pub conversations: i64,
    pub messages: i64,
    pub code_sessions: i64,
    pub db_bytes: u64,
}

#[tauri::command]
pub fn data_stats(app: tauri::AppHandle, db: State<'_, Db>) -> Result<DataStats, String> {
    let conn = lock(&db)?;
    let count = |sql: &str| -> Result<i64, String> {
        conn.query_row(sql, [], |r| r.get(0)).map_err(|e| e.to_string())
    };
    let conversations = count("SELECT COUNT(*) FROM conversations")?;
    let messages = count("SELECT COUNT(*) FROM messages")?;
    let code_sessions = count("SELECT COUNT(*) FROM code_sessions").unwrap_or(0);
    drop(conn);
    let db_bytes = tauri::Manager::path(&app)
        .app_data_dir()
        .ok()
        .map(|d| d.join("chaty.db"))
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    Ok(DataStats { conversations, messages, code_sessions, db_bytes })
}
