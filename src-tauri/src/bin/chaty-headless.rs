//! chaty-headless — stdio JSON-lines server exposing the REAL Coder tool layer
//! (agent.rs) plus the inference engines, so ChatyCoder-Bench can drive the
//! production agent loop (src/lib/agentLoop.ts) outside the Tauri app.
//!
//! Protocol (one JSON object per line, both directions):
//!   → {"id":1,"cmd":"load_model","args":{"path":"…","nCtx":16384}}
//!   ← {"id":1,"type":"result","result":{…ModelInfo…}}
//!   → {"id":2,"cmd":"generate","args":{"request":{…GenRequest…}}}
//!   ← {"id":2,"type":"event","event":{"type":"token","text":"…"}}   (× many)
//!   ← {"id":2,"type":"result","result":null}
//!   ← {"id":N,"type":"error","message":"…"}   (command failed — the message
//!      is the same string a Tauri command rejection would carry, so markers
//!      like NEED_DIR_GRANT survive verbatim)
//!
//! Arg keys accept both snake_case and camelCase (the runner's mockIPC bridge
//! forwards exactly what ipc.ts sends, which is camelCase).

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{json, Value};

use chaty_lib::agent as ag;
use chaty_lib::inference::{llama::LlamaEngine, mlx::MlxEngine, mock::MockBackend};
use chaty_lib::inference::{GenRequest, InferenceBackend, ModelInfo};
use chaty_lib::search;

type Engine = Arc<dyn InferenceBackend>;

static ENGINE: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();
static CANCEL: OnceLock<Mutex<Arc<AtomicBool>>> = OnceLock::new();

fn engine_slot() -> &'static Mutex<Option<Engine>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

fn cancel_slot() -> &'static Mutex<Arc<AtomicBool>> {
    CANCEL.get_or_init(|| Mutex::new(Arc::new(AtomicBool::new(false))))
}

/// Line-atomic stdout emit (stdout to a pipe is block-buffered — flush).
fn emit(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", v);
    let _ = out.flush();
}

fn reply(id: u64, r: Result<Value, String>) {
    match r {
        Ok(result) => emit(json!({"id": id, "type": "result", "result": result})),
        Err(message) => emit(json!({"id": id, "type": "error", "message": message})),
    }
}

/// Fetch an arg accepting both snake_case and camelCase keys.
fn arg<'a>(args: &'a Value, snake: &str) -> Option<&'a Value> {
    if let Some(v) = args.get(snake) {
        if !v.is_null() {
            return Some(v);
        }
    }
    let mut camel = String::new();
    for (i, part) in snake.split('_').enumerate() {
        if i == 0 {
            camel.push_str(part);
        } else if let Some(c) = part.chars().next() {
            camel.push(c.to_ascii_uppercase());
            camel.push_str(&part[c.len_utf8()..]);
        }
    }
    args.get(&camel).filter(|v| !v.is_null())
}

fn s_arg(args: &Value, k: &str) -> Option<String> {
    arg(args, k).and_then(|v| v.as_str()).map(String::from)
}
fn req_s(args: &Value, k: &str) -> Result<String, String> {
    s_arg(args, k).ok_or_else(|| format!("missing arg: {k}"))
}
fn u_arg(args: &Value, k: &str) -> Option<u64> {
    arg(args, k).and_then(|v| v.as_u64())
}
fn b_arg(args: &Value, k: &str) -> Option<bool> {
    arg(args, k).and_then(|v| v.as_bool())
}

fn ok<T: serde::Serialize>(v: T) -> Result<Value, String> {
    serde_json::to_value(v).map_err(|e| e.to_string())
}
fn res<T: serde::Serialize>(r: Result<T, String>) -> Result<Value, String> {
    r.and_then(ok)
}

fn load_engine(path: &str, n_ctx: Option<u32>) -> Result<ModelInfo, String> {
    let (engine, info): (Engine, ModelInfo) = if path == "mock" {
        let e = MockBackend::new("mock");
        let info = ModelInfo {
            name: "mock".into(),
            path: "mock".into(),
            backend: "mock".into(),
            loaded: true,
            arch: None,
            size_mb: None,
            params_b: None,
            n_ctx_train: Some(16384),
            n_ctx,
            n_layer: None,
            gpu_layers: 0,
            gpu_name: None,
            model_name: Some("mock".into()),
            quant: None,
            n_embd: None,
            has_chat_template: true,
            supports_thinking: false,
            think_switch: false,
            effort_levels: Vec::new(),
            tool_role: false,
            reasoning_field: false,
            supports_tools: true,
            multimodal: false,
            vision_ready: false,
            multi_image: true,
            mmproj: None,
            warning: None,
        };
        (Arc::new(e), info)
    } else if Path::new(path).join("config.json").is_file() {
        let (e, info) = MlxEngine::load(path, n_ctx, |_| {}).map_err(|e| e.to_string())?;
        (Arc::new(e), info)
    } else {
        let (e, info) = LlamaEngine::load(path, None, n_ctx).map_err(|e| e.to_string())?;
        (Arc::new(e), info)
    };
    if let Some(old) = engine_slot().lock().unwrap().replace(engine) {
        old.unload();
    }
    Ok(info)
}

async fn dispatch(cmd: &str, args: Value, id: u64) {
    let r: Result<Value, String> = match cmd {
        "load_model" => {
            let path = req_s(&args, "path");
            let n_ctx = u_arg(&args, "n_ctx").map(|v| v as u32);
            match path {
                Ok(p) => res(load_engine(&p, n_ctx)),
                Err(e) => Err(e),
            }
        }
        "generate" => {
            let engine = engine_slot().lock().unwrap().clone();
            let Some(engine) = engine else {
                reply(id, Err("no model loaded".into()));
                return;
            };
            let req: GenRequest = match arg(&args, "request")
                .cloned()
                .ok_or_else(|| "missing arg: request".to_string())
                .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
            {
                Ok(r) => r,
                Err(e) => {
                    reply(id, Err(e));
                    return;
                }
            };
            let cancel = Arc::new(AtomicBool::new(false));
            *cancel_slot().lock().unwrap() = cancel.clone();
            let sink = tauri::ipc::Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
                if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                    if let Ok(ev) = serde_json::from_str::<Value>(&s) {
                        emit(json!({"id": id, "type": "event", "event": ev}));
                    }
                }
                Ok(())
            });
            match engine.generate(req, sink, cancel).await {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e.to_string()),
            }
        }
        "cancel_generation" => {
            cancel_slot().lock().unwrap().store(true, Ordering::SeqCst);
            Ok(Value::Null)
        }

        // ---- workspace & grants ----
        "agent_set_workspace" => req_s(&args, "path").and_then(|p| res(ag::agent_set_workspace(p))),
        "agent_set_lang" => req_s(&args, "lang").map(|l| {
            ag::agent_set_lang(l);
            Value::Null
        }),
        "agent_set_edit_anchors" => {
            ag::agent_set_edit_anchors(b_arg(&args, "on").unwrap_or(false));
            Ok(Value::Null)
        }
        "agent_get_workspace" => ok(ag::agent_get_workspace()),
        "skill_live_support" => match req_s(&args, "name") {
            Ok(n) => ok(chaty_lib::skillsync::live_support(&n)),
            Err(e) => Err(e),
        },
        "agent_grant_dir" => req_s(&args, "path").and_then(|p| res(ag::agent_grant_dir(p))),
        "agent_revoke_dir" => req_s(&args, "path").map(|p| {
            ag::agent_revoke_dir(p);
            Value::Null
        }),
        "agent_list_grants" => ok(ag::agent_list_grants()),
        "agent_clear_grants" => {
            ag::agent_clear_grants();
            Ok(Value::Null)
        }

        // ---- files & search ----
        "agent_read_file" => req_s(&args, "path").and_then(|p| {
            res(ag::agent_read_file(
                p,
                u_arg(&args, "offset").map(|v| v as usize),
                u_arg(&args, "limit").map(|v| v as usize),
                u_arg(&args, "max_chars").map(|v| v as usize),
                s_arg(&args, "symbol"),
            ))
        }),
        "agent_write_file" => match (req_s(&args, "path"), req_s(&args, "content")) {
            (Ok(p), Ok(c)) => res(ag::agent_write_file(p, c)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        },
        "agent_edit_file" => match (
            req_s(&args, "path"),
            req_s(&args, "old_string"),
            req_s(&args, "new_string"),
        ) {
            (Ok(p), Ok(o), Ok(n)) => res(ag::agent_edit_file(p, o, n, b_arg(&args, "replace_all"))),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => Err(e),
        },
        "agent_edit_lines" => req_s(&args, "path").and_then(|p| {
            let edits = arg(&args, "edits").cloned().unwrap_or(Value::Null);
            res(ag::agent_edit_lines(p, edits))
        }),
        "agent_multi_edit" => {
            let edits = arg(&args, "edits")
                .cloned()
                .ok_or_else(|| "missing arg: edits".to_string())
                .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()));
            match (req_s(&args, "path"), edits) {
                (Ok(p), Ok(e)) => res(ag::agent_multi_edit(p, e)),
                (Err(e), _) | (_, Err(e)) => Err(e),
            }
        }
        "agent_outline" => req_s(&args, "path").and_then(|p| res(ag::agent_outline(p))),
        "agent_list_dir" => res(ag::agent_list_dir(s_arg(&args, "path"))),
        "agent_glob" => req_s(&args, "pattern").and_then(|p| res(ag::agent_glob(p))),
        "agent_list_files" => res(ag::agent_list_files(
            s_arg(&args, "query"),
            u_arg(&args, "limit").map(|v| v as usize),
        )),
        "agent_search_code" => req_s(&args, "query").and_then(|q| {
            res(ag::agent_search_code(q, u_arg(&args, "k").map(|v| v as usize)))
        }),
        "agent_grep" => req_s(&args, "pattern").and_then(|p| {
            res(ag::agent_grep(p, s_arg(&args, "path"), s_arg(&args, "glob")))
        }),
        "agent_search_files" => req_s(&args, "query").and_then(|q| {
            res(ag::agent_search_files(
                q,
                s_arg(&args, "path"),
                b_arg(&args, "names_only"),
            ))
        }),
        "agent_understand_repo" => res(ag::agent_understand_repo()),
        "agent_validate_change" => {
            let files = arg(&args, "files")
                .cloned()
                .map(|v| serde_json::from_value::<Vec<String>>(v).map_err(|e| e.to_string()))
                .transpose();
            match files {
                Ok(f) => res(ag::agent_validate_change(f).await),
                Err(e) => Err(e),
            }
        }

        // ---- checkpoints ----
        "agent_checkpoint_begin" => ok(ag::agent_checkpoint_begin()),
        "agent_checkpoint_revert_to" => u_arg(&args, "id")
            .ok_or_else(|| "missing arg: id".to_string())
            .and_then(|i| res(ag::agent_checkpoint_revert_to(i))),

        // ---- bash & background jobs ----
        "agent_bash" => match req_s(&args, "command") {
            Ok(c) => res(
                ag::agent_bash(c, u_arg(&args, "timeout_secs"), s_arg(&args, "sudo_password"))
                    .await,
            ),
            Err(e) => Err(e),
        },
        "agent_bash_bg" => req_s(&args, "command").and_then(|c| res(ag::agent_bash_bg(c))),
        "agent_bg_output" => u_arg(&args, "id")
            .ok_or_else(|| "missing arg: id".to_string())
            .and_then(|i| res(ag::agent_bg_output(i))),
        "agent_bg_kill" => u_arg(&args, "id")
            .ok_or_else(|| "missing arg: id".to_string())
            .and_then(|i| res(ag::agent_bg_kill(i))),
        "agent_bg_list" => ok(ag::agent_bg_list()),
        "agent_bg_reap" => ok(ag::agent_bg_reap()),

        // ---- web tools ----
        "agent_web_download" => match (req_s(&args, "url"), req_s(&args, "path")) {
            (Ok(u), Ok(p)) => res(ag::agent_web_download(u, p).await),
            (Err(e), _) | (_, Err(e)) => Err(e),
        },
        "agent_dl_list" => ok(ag::agent_dl_list()),
        "agent_dl_reap" => ok(ag::agent_dl_reap()),
        "web_search" => match req_s(&args, "query") {
            Ok(q) => res(search::web_search(q).await),
            Err(e) => Err(e),
        },
        "fetch_url" => match req_s(&args, "url") {
            Ok(u) => res(search::fetch_url(u).await),
            Err(e) => Err(e),
        },

        // ---- browser tools (automation Chrome; set CHATY_BROWSER_HEADLESS=1
        // so bench sessions never open a visible window) ----
        "browser_refresh" => res(ag::browser_refresh().await),
        "browser_navigate" => match req_s(&args, "url") {
            Ok(u) => res(ag::browser_navigate(u).await),
            Err(e) => Err(e),
        },
        "browser_read" => res(ag::browser_read().await),
        "browser_screenshot" => res(ag::browser_screenshot().await),
        "browser_snapshot" => res(ag::browser_snapshot().await),
        "browser_scroll" => res(
            ag::browser_scroll(s_arg(&args, "to"), args.get("by").and_then(|v| v.as_f64())).await,
        ),
        "browser_eval" => match req_s(&args, "expression") {
            Ok(x) => res(ag::browser_eval(x).await),
            Err(e) => Err(e),
        },
        "browser_click" => {
            let steps = args.get("steps").cloned().and_then(|v| serde_json::from_value(v).ok());
            res(ag::browser_click(s_arg(&args, "selector"), s_arg(&args, "text"), steps).await)
        }
        "browser_type" => {
            let steps = args.get("steps").cloned().and_then(|v| serde_json::from_value(v).ok());
            res(ag::browser_type(
                s_arg(&args, "selector"),
                s_arg(&args, "label"),
                s_arg(&args, "text"),
                steps,
            )
            .await)
        }
        "browser_console" => res(ag::browser_console().await),
        "browser_close" => res(ag::browser_close().await),
        // MCP (2.0): the bench drives real servers through the real client.
        "mcp_connect" => {
            let name = match req_s(&args, "name") { Ok(x) => x, Err(e) => return reply(id, Err(e)) };
            match serde_json::from_value(args.get("transport").cloned().unwrap_or_default()) {
                Ok(t) => res(chaty_lib::mcp::mcp_connect(name, t).await),
                Err(e) => Err(format!("bad transport: {e}")),
            }
        }
        "mcp_disconnect" => match req_s(&args, "name") {
            Ok(name) => { chaty_lib::mcp::mcp_disconnect(name); Ok(serde_json::Value::Null) }
            Err(e) => Err(e),
        },
        "mcp_call" => {
            let server = match req_s(&args, "server") { Ok(x) => x, Err(e) => return reply(id, Err(e)) };
            let tool = match req_s(&args, "tool") { Ok(x) => x, Err(e) => return reply(id, Err(e)) };
            res(chaty_lib::mcp::mcp_call(server, tool, args.get("args").cloned().unwrap_or_default()).await)
        }
        "agent_resolve_image" => match req_s(&args, "path") {
            Ok(p) => res(ag::agent_resolve_image(p)),
            Err(e) => Err(e),
        },
        _ => Err(format!("unsupported in headless bench: {cmd}")),
    };
    reply(id, r);
}

fn main() {
    // Reap headless Chromes left by SIGKILLed bench runs before this one
    // launches its own (a live sibling's browser is skipped by pid check).
    chaty_lib::browser::sweep_orphan_browsers();
    // Bench A/B hook: flip hashline anchors on from the environment so the
    // whole session (read_file prefixes + edit_lines) runs in anchor mode.
    if std::env::var("CHATY_EDIT_ANCHORS").map(|v| v == "1").unwrap_or(false) {
        ag::agent_set_edit_anchors(true);
    }
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let id = msg.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
        let cmd = msg
            .get("cmd")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let args = msg.get("args").cloned().unwrap_or_else(|| json!({}));
        // Every command runs on the async runtime so cancel_generation can be
        // handled while a generate is streaming.
        tauri::async_runtime::spawn(async move {
            dispatch(&cmd, args, id).await;
        });
    }
}
