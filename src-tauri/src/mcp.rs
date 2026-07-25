//! Minimal MCP client (2.0 M1): initialize / tools/list / tools/call over
//! stdio or Streamable HTTP. Hand-rolled on purpose — the app needs three
//! methods, not a framework, and every line here is auditable. The frontend
//! owns server CONFIG (localStorage, like every other setting); this module
//! owns server PROCESSES and wire protocol.
//!
//! Safety posture:
//! - stdio servers are child processes tracked in MCP_PIDS; the app's quit
//!   path exits via `libc::_exit` (skips destructors), so lib.rs calls
//!   `kill_all_now()` on exit — same contract as the MLX sidecars.
//! - one in-flight request per server (the agent loop is serial anyway);
//!   every call has a hard timeout so a wedged server can't wedge the agent.
//! - tool RESULTS are returned verbatim to the TS side, which wraps them in
//!   the untrusted-content defense (registry marks every MCP tool untrusted).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const INIT_TIMEOUT: Duration = Duration::from_secs(15);
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
const PROTOCOL_VERSION: &str = "2025-06-18";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "lowercase")]
pub enum McpTransport {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    Http {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    /// JSON schema of the arguments, passed through for lean-doc synthesis
    /// and required-args validation on the TS side.
    pub input_schema: Value,
}

// ── Connection plumbing ──────────────────────────────────────────────────────

struct StdioConn {
    child: Child,
    stdin: std::process::ChildStdin,
    /// Replies (any JSON with an "id") from the reader thread.
    replies: Receiver<Value>,
}

enum Conn {
    Stdio(StdioConn),
    Http {
        url: String,
        headers: HashMap<String, String>,
        session: Option<String>,
    },
}

struct Server {
    name: String,
    conn: Conn,
    next_id: u64,
}

static SERVERS: Mutex<Option<HashMap<String, Server>>> = Mutex::new(None);
static MCP_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());
/// name → child pid, maintained across a call's lifetime. mcp_call checks a
/// Server OUT of SERVERS while a request is in flight, so a disconnect that
/// only consulted SERVERS couldn't kill the child — this map can, and killing
/// the child collapses the pipe, which wakes the blocked reader immediately.
static NAME_PIDS: Mutex<Option<HashMap<String, u32>>> = Mutex::new(None);

/// Kill every stdio MCP server. Called from the app's exit path, which skips
/// destructors (`libc::_exit`) — without this, quitting orphans the servers.
pub fn kill_all_now() {
    let pids: Vec<u32> = std::mem::take(&mut *MCP_PIDS.lock().unwrap());
    for pid in pids {
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
        #[cfg(not(unix))]
        let _ = pid;
    }
}

fn untrack_pid(pid: u32) {
    MCP_PIDS.lock().unwrap().retain(|p| *p != pid);
}

// ── JSON-RPC over the two transports ─────────────────────────────────────────

fn stdio_request(conn: &mut StdioConn, id: u64, body: &Value, timeout: Duration) -> Result<Value, String> {
    let line = serde_json::to_string(body).map_err(|e| e.to_string())?;
    conn.stdin
        .write_all(line.as_bytes())
        .and_then(|_| conn.stdin.write_all(b"\n"))
        .and_then(|_| conn.stdin.flush())
        .map_err(|e| format!("MCP server pipe closed: {e}"))?;
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err("MCP request timed out".into());
        }
        let msg = conn
            .replies
            .recv_timeout(left)
            .map_err(|_| "MCP request timed out (server silent or exited)".to_string())?;
        if msg.get("id").and_then(Value::as_u64) == Some(id) {
            return Ok(msg);
        }
        // A reply to an earlier timed-out request — drop and keep waiting.
    }
}

fn stdio_notify(conn: &mut StdioConn, body: &Value) -> Result<(), String> {
    let line = serde_json::to_string(body).map_err(|e| e.to_string())?;
    conn.stdin
        .write_all(line.as_bytes())
        .and_then(|_| conn.stdin.write_all(b"\n"))
        .and_then(|_| conn.stdin.flush())
        .map_err(|e| format!("MCP server pipe closed: {e}"))
}

/// Parse a Streamable-HTTP response body: plain JSON, or an SSE stream where
/// each `data:` line carries one JSON-RPC message — the reply is the one
/// whose id matches.
fn parse_http_body(body: &str, content_type: &str, id: u64) -> Result<Value, String> {
    if content_type.contains("text/event-stream") {
        for line in body.lines() {
            if let Some(data) = line.strip_prefix("data:") {
                if let Ok(v) = serde_json::from_str::<Value>(data.trim()) {
                    if v.get("id").and_then(Value::as_u64) == Some(id) {
                        return Ok(v);
                    }
                }
            }
        }
        return Err("MCP HTTP: no matching reply in event stream".into());
    }
    serde_json::from_str(body).map_err(|e| format!("MCP HTTP: bad JSON reply: {e}"))
}

async fn http_request(
    url: &str,
    headers: &HashMap<String, String>,
    session: &mut Option<String>,
    id: u64,
    body: &Value,
    timeout: Duration,
) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| e.to_string())?;
    let mut req = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    for (k, v) in headers {
        req = req.header(k.as_str(), v.as_str());
    }
    if let Some(s) = session.as_deref() {
        req = req.header("mcp-session-id", s);
    }
    let resp = req.json(body).send().await.map_err(|e| format!("MCP HTTP: {e}"))?;
    if let Some(s) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
        *session = Some(s.to_string());
    }
    let status = resp.status();
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("MCP HTTP {status}: {}", text.chars().take(300).collect::<String>()));
    }
    parse_http_body(&text, &ctype, id)
}

/// One JSON-RPC request. Takes the server BY VALUE and hands it back: the
/// stdio path blocks on the reply channel, and blocking inside an async fn
/// would pin a tokio worker (on a current-thread runtime it freezes the whole
/// runtime — caught by the wedged-server test). spawn_blocking needs owned
/// data, so ownership threads through instead of a &mut.
async fn request(
    mut server: Server,
    method: &str,
    params: Value,
    timeout: Duration,
) -> (Server, Result<Value, String>) {
    let id = server.next_id;
    server.next_id += 1;
    let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
    let (server, reply) = match server.conn {
        Conn::Stdio(_) => tokio::task::spawn_blocking(move || {
            let Conn::Stdio(ref mut c) = server.conn else { unreachable!() };
            let r = stdio_request(c, id, &body, timeout);
            (server, r)
        })
        .await
        .expect("mcp blocking task panicked"),
        Conn::Http { ref url, ref headers, ref mut session } => {
            // Sidestep the borrow of three fields at once.
            let url = url.clone();
            let headers = headers.clone();
            let mut sess = session.take();
            let r = http_request(&url, &headers, &mut sess, id, &body, timeout).await;
            if let Conn::Http { ref mut session, .. } = server.conn {
                *session = sess;
            }
            (server, r)
        }
    };
    let reply = match reply {
        Ok(v) => v,
        Err(e) => return (server, Err(e)),
    };
    if let Some(err) = reply.get("error") {
        let msg = err.get("message").and_then(Value::as_str).unwrap_or("unknown error");
        return (server, Err(format!("MCP error: {msg}")));
    }
    let result = reply.get("result").cloned().unwrap_or(Value::Null);
    (server, Ok(result))
}

async fn notify(server: &mut Server, method: &str) -> Result<(), String> {
    let body = json!({ "jsonrpc": "2.0", "method": method });
    match &mut server.conn {
        Conn::Stdio(c) => stdio_notify(c, &body),
        Conn::Http { url, headers, session } => {
            // Fire-and-forget; Streamable HTTP replies 202 to notifications.
            let _ = http_request(url, headers, session, 0, &body, Duration::from_secs(5)).await;
            Ok(())
        }
    }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

fn spawn_stdio(command: &str, args: &[String], env: &HashMap<String, String>) -> Result<StdioConn, String> {
    let mut cmd = Command::new(command);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // stderr passes through to the app log — server diagnostics stay visible.
        .stderr(Stdio::inherit());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to start MCP server `{command}`: {e}"))?;
    MCP_PIDS.lock().unwrap().push(child.id());
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let (tx, rx): (Sender<Value>, Receiver<Value>) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            // Replies only; notifications from the server are ignored for now.
            if v.get("id").is_some() && tx.send(v).is_err() {
                break;
            }
        }
    });
    Ok(StdioConn { child, stdin, replies: rx })
}

fn parse_tools(result: &Value) -> Vec<McpToolInfo> {
    result
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    Some(McpToolInfo {
                        name: t.get("name")?.as_str()?.to_string(),
                        description: t
                            .get("description")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        input_schema: t.get("inputSchema").cloned().unwrap_or(json!({})),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

async fn connect_inner(name: &str, transport: McpTransport) -> Result<Vec<McpToolInfo>, String> {
    disconnect_inner(name); // reconnect = clean slate
    let conn = match &transport {
        McpTransport::Stdio { command, args, env } => Conn::Stdio(spawn_stdio(command, args, env)?),
        McpTransport::Http { url, headers } => Conn::Http {
            url: url.clone(),
            headers: headers.clone(),
            session: None,
        },
    };
    if let Conn::Stdio(ref c) = conn {
        NAME_PIDS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .insert(name.to_string(), c.child.id());
    }
    let server = Server { name: name.to_string(), conn, next_id: 1 };
    let (server, init) = request(
        server,
        "initialize",
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "chaty", "version": env!("CARGO_PKG_VERSION") },
        }),
        INIT_TIMEOUT,
    )
    .await;
    let mut server = server;
    if let Err(e) = init {
        kill_server(server);
        return Err(format!("initialize failed: {e}"));
    }
    if let Err(e) = notify(&mut server, "notifications/initialized").await {
        kill_server(server);
        return Err(e);
    }
    let (server, listed) = request(server, "tools/list", json!({}), INIT_TIMEOUT).await;
    let listed = match listed {
        Ok(v) => v,
        Err(e) => {
            kill_server(server);
            return Err(e);
        }
    };
    let tools = parse_tools(&listed);
    SERVERS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(name.to_string(), server);
    Ok(tools)
}

fn kill_server(server: Server) {
    NAME_PIDS.lock().unwrap().get_or_insert_with(HashMap::new).remove(&server.name);
    if let Server { conn: Conn::Stdio(mut c), .. } = server {
        let pid = c.child.id();
        let _ = c.child.kill();
        let _ = c.child.wait();
        untrack_pid(pid);
    }
}

fn disconnect_inner(name: &str) {
    let server = SERVERS
        .lock()
        .unwrap()
        .as_mut()
        .and_then(|m| m.remove(name));
    if let Some(s) = server {
        kill_server(s);
        return;
    }
    // Not in the map — either unknown, or checked out by an in-flight call.
    // Kill by pid so the call unblocks instead of waiting out its timeout.
    let pid = NAME_PIDS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .remove(name);
    if let Some(pid) = pid {
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
        untrack_pid(pid);
    }
}

/// Flatten a tools/call result into model-facing text. Non-text content is
/// noted, not dropped silently.
fn content_text(result: &Value) -> String {
    let mut out = String::new();
    if let Some(items) = result.get("content").and_then(Value::as_array) {
        for item in items {
            match item.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(item.get("text").and_then(Value::as_str).unwrap_or(""));
                }
                Some(other) => {
                    out.push_str(&format!("\n[{} content omitted]", other));
                }
                None => {}
            }
        }
    }
    if out.is_empty() {
        out = serde_json::to_string(result).unwrap_or_default();
    }
    out
}

// ── Tauri commands ───────────────────────────────────────────────────────────

/// Connect (or reconnect) a server and return its tool list.
#[tauri::command]
pub async fn mcp_connect(name: String, transport: McpTransport) -> Result<Vec<McpToolInfo>, String> {
    connect_inner(&name, transport).await
}

#[tauri::command]
pub fn mcp_disconnect(name: String) {
    disconnect_inner(&name);
}

/// Call one tool. `is_error` results come back as ERROR text so the agent
/// loop renders a red step and the model sees a correctable failure.
#[tauri::command]
pub async fn mcp_call(server: String, tool: String, args: Value) -> Result<String, String> {
    // Take the server out of the map for the duration of the call so the
    // global lock is NOT held across await/IO (one in-flight call per server;
    // a second call to the same server errors instead of deadlocking).
    let s = SERVERS
        .lock()
        .unwrap()
        .as_mut()
        .and_then(|m| m.remove(&server))
        .ok_or_else(|| format!("MCP server `{server}` is not connected"))?;
    let (s, res) = request(
        s,
        "tools/call",
        json!({ "name": tool, "arguments": args }),
        CALL_TIMEOUT,
    )
    .await;
    let still_alive = match &s.conn {
        Conn::Stdio(_) => NAME_PIDS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .contains_key(&server),
        Conn::Http { .. } => true,
    };
    if still_alive {
        SERVERS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .insert(server, s);
    } else {
        kill_server(s); // reap the Child handle; the process is already dead
    }
    let result = res?;
    let text = content_text(&result);
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return Ok(format!("ERROR: {text}"));
    }
    Ok(text)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A five-line stdio MCP server in python3: replies to initialize and
    /// tools/list, and echoes tool arguments back from tools/call.
    #[cfg(unix)]
    const FAKE_SERVER: &str = r#"
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m: continue
    i, meth = m["id"], m["method"]
    if meth == "initialize":
        r = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "fake"}}
    elif meth == "tools/list":
        r = {"tools": [{"name": "echo", "description": "Echo the input back.", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
    elif meth == "tools/call":
        a = m["params"]["arguments"]
        if a.get("boom"):
            r = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
        else:
            r = {"content": [{"type": "text", "text": "echo: " + a.get("text", "")}]}
    else:
        r = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": i, "result": r}) + "\n")
    sys.stdout.flush()
"#;

    #[cfg(unix)]
    fn fake_transport() -> McpTransport {
        McpTransport::Stdio {
            command: "python3".into(),
            args: vec!["-c".into(), FAKE_SERVER.into()],
            env: HashMap::new(),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdio_handshake_list_call_and_error_path() {
        let tools = connect_inner("t1", fake_transport()).await.expect("connect");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].input_schema["required"][0], "text");

        let out = mcp_call("t1".into(), "echo".into(), json!({"text": "hi"}))
            .await
            .expect("call");
        assert_eq!(out, "echo: hi");

        // isError results surface as ERROR text (red step), not a hard failure.
        let err = mcp_call("t1".into(), "echo".into(), json!({"boom": true}))
            .await
            .expect("isError call still returns Ok");
        assert!(err.starts_with("ERROR:"), "{err}");

        // Unknown server errors cleanly.
        let missing = mcp_call("nope".into(), "echo".into(), json!({})).await;
        assert!(missing.is_err());

        // Tests share the global PID list — assert THIS server's pid is gone,
        // not that the list is empty (a concurrent test may have its own).
        let pid = match &SERVERS.lock().unwrap().as_ref().unwrap().get("t1").unwrap().conn {
            Conn::Stdio(c) => c.child.id(),
            _ => unreachable!(),
        };
        disconnect_inner("t1");
        assert!(!MCP_PIDS.lock().unwrap().contains(&pid), "pid untracked on disconnect");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wedged_server_times_out_instead_of_hanging() {
        // A server that never replies to tools/call.
        let silent = r#"
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m: continue
    if m["method"] == "initialize":
        r = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "s"}}
    elif m["method"] == "tools/list":
        r = {"tools": []}
    else:
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}) + "\n")
    sys.stdout.flush()
"#;
        let t = McpTransport::Stdio {
            command: "python3".into(),
            args: vec!["-c".into(), silent.into()],
            env: HashMap::new(),
        };
        connect_inner("t2", t).await.expect("connect");
        // Shrink the wait by racing the 60s call timeout against a short one:
        // the call must not hang forever; here we just verify the plumbing
        // returns once the child is killed out from under it.
        let call = tokio::time::timeout(
            Duration::from_secs(5),
            mcp_call("t2".into(), "anything".into(), json!({})),
        );
        // Kill the server while the call waits — the reader thread ends and
        // recv_timeout surfaces an error well before the outer timeout.
        let killer = async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            disconnect_inner("t2");
        };
        let (res, _) = tokio::join!(call, killer);
        match res {
            Ok(inner) => assert!(inner.is_err(), "call must error, got {inner:?}"),
            Err(_) => panic!("call hung past 5s after server death"),
        }
    }

    // Against the REAL official reference server (downloads via npx on first
    // run). Run: cargo test -p chaty real_everything -- --ignored --nocapture
    #[cfg(unix)]
    #[tokio::test]
    #[ignore]
    async fn real_everything_server_e2e() {
        let t = McpTransport::Stdio {
            command: "npx".into(),
            args: vec!["-y".into(), "@modelcontextprotocol/server-everything".into()],
            env: HashMap::new(),
        };
        let tools = connect_inner("everything", t).await.expect("connect real server");
        eprintln!("tools: {:?}", tools.iter().map(|t| &t.name).collect::<Vec<_>>());
        assert!(tools.iter().any(|t| t.name == "echo"), "reference server exposes echo");
        let out = mcp_call("everything".into(), "echo".into(), json!({"message": "chaty-e2e"}))
            .await
            .expect("echo call");
        eprintln!("echo → {out}");
        assert!(out.contains("chaty-e2e"), "{out}");
        let sum = mcp_call("everything".into(), "get-sum".into(), json!({"a": 20, "b": 22}))
            .await
            .expect("add call");
        eprintln!("add → {sum}");
        assert!(sum.contains("42"), "{sum}");
        disconnect_inner("everything");
    }

    // Store certification: every runnable catalog entry must connect and
    // answer its probe — the evidence behind the store's "certified" badge.
    // Run: cargo test -p chaty store_cert -- --ignored --nocapture
    #[cfg(unix)]
    #[tokio::test]
    #[ignore]
    async fn store_certification() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../src/lib/mcpStore.catalog.json"
        ))
        .expect("catalog");
        let cat: Value = serde_json::from_str(&raw).expect("catalog json");
        let tmp = std::env::temp_dir().join("chaty-mcp-cert");
        std::fs::create_dir_all(&tmp).unwrap();
        let dir = tmp.to_string_lossy().to_string();
        let mut certified = 0;
        for e in cat["entries"].as_array().unwrap() {
            if e["transport"] != "stdio" || e["probe"].is_null() {
                continue; // http/tokened entries need credentials — hand-tested instead
            }
            let id = e["id"].as_str().unwrap();
            let args: Vec<String> = e["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_str().unwrap().replace("{dir}", &dir))
                .collect();
            let t = McpTransport::Stdio {
                command: e["command"].as_str().unwrap().into(),
                args,
                env: HashMap::new(),
            };
            let tools = connect_inner(id, t).await.unwrap_or_else(|e| panic!("{id}: {e}"));
            let probe = &e["probe"];
            let expect = probe["expect"].as_str().unwrap().replace("{dir}", &dir);
            let out = mcp_call(
                id.into(),
                probe["tool"].as_str().unwrap().into(),
                probe["args"].clone(),
            )
            .await
            .unwrap_or_else(|e| panic!("{id} probe: {e}"));
            assert!(out.contains(&expect), "{id} probe mismatch: {out}");
            eprintln!("✓ {id}: {} tools, probe ok", tools.len());
            disconnect_inner(id);
            certified += 1;
        }
        assert!(certified >= 3, "only {certified} entries certified");
    }

    #[test]
    fn http_body_parsing_covers_json_and_sse() {
        let plain = r#"{"jsonrpc":"2.0","id":3,"result":{"ok":true}}"#;
        let v = parse_http_body(plain, "application/json", 3).unwrap();
        assert_eq!(v["result"]["ok"], true);

        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"n\":1}}\n\n";
        let v = parse_http_body(sse, "text/event-stream", 7).unwrap();
        assert_eq!(v["result"]["n"], 1);

        assert!(parse_http_body(sse, "text/event-stream", 8).is_err());
    }

    #[test]
    fn tool_parsing_tolerates_minimal_entries() {
        let r = json!({"tools": [{"name": "a"}, {"name": "b", "description": "d", "inputSchema": {"type": "object"}}, {"notatool": 1}]});
        let tools = parse_tools(&r);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].description, "");
    }

    #[test]
    fn content_flattening() {
        let r = json!({"content": [{"type": "text", "text": "a"}, {"type": "image", "data": "…"}, {"type": "text", "text": "b"}]});
        assert_eq!(content_text(&r), "a\n[image content omitted]\nb");
    }
}
