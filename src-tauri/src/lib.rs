// #[macro_use]: agent.rs declares `trf!` (bilingual format), which browser.rs
// and any later module use — declaring it once here beats each module keeping
// its own copy under a different name (browser.rs had `btr!`).
#[macro_use]
pub mod agent;
pub mod attach;
pub mod browser;
pub mod docimg;
pub mod mic;
pub mod rag;
mod commands;
pub mod download;
pub mod gpu;
pub mod http;
pub mod inference;
pub mod mcp;
pub mod ocr;
pub mod search;
mod state;
mod store;
pub mod update;
pub mod voice;
pub mod webx;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

use state::AppState;

/// Generation counter for pending close-from-fullscreen hides: showing the
/// window (tray / Dock / shortcut) bumps it, which cancels any hide still
/// waiting for macOS's permission — otherwise a reopen during the wait would
/// get yanked back down the moment the OS relented.
#[cfg(target_os = "macos")]
static HIDE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bring the main window to the foreground.
fn show_main_window(app: &tauri::AppHandle) {
    #[cfg(target_os = "macos")]
    {
        HIDE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Undo the app-level hide (⌘H-style) the close path uses; unhiding
        // slides back into the fullscreen Space, window still fullscreen.
        let _ = app.show();
    }
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

/// Spotlight-style toggle: hide if it's already up front, otherwise summon it.
fn toggle_main_window(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let up = w.is_visible().unwrap_or(false) && w.is_focused().unwrap_or(false);
        if up {
            let _ = w.hide();
        } else {
            let _ = w.show();
            let _ = w.unminimize();
            let _ = w.set_focus();
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // ggml's Metal backend keeps every weight buffer in an MTLResidencySet
    // when built against the macOS 15+ SDK, which shows up as a wired-memory
    // balloon the size of the model (and froze machines on big models with
    // locally-built binaries). ggml gates this on a runtime env var — set it
    // before the first Metal init so ALL builds behave like the shipped CI
    // ones, regardless of the SDK they were compiled with.
    // (`CHATY_METAL_RESIDENCY=1` opts back in for benchmarking.)
    #[cfg(target_os = "macos")]
    if std::env::var_os("CHATY_METAL_RESIDENCY").is_none() {
        std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");
    }

    // Auto-grant microphone/camera (no WebView2 permission prompt) and allow
    // audio autoplay without a user gesture (for streaming TTS). Must be set
    // before the webview is created. Windows-only: macOS uses WKWebView, which
    // honours the system mic permission (see NSMicrophoneUsageDescription in the
    // bundle Info.plist) and prompts once.
    #[cfg(windows)]
    std::env::set_var(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
        "--use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required",
    );

    tauri::Builder::default()
        // Single instance must be registered first; focus the window on relaunch.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            // A `chaty://` deep link launched a second instance — the deep-link
            // plugin forwards the URL to `on_open_url`; here we just surface the
            // window.
            show_main_window(app);
        }))
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_os::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state == ShortcutState::Pressed {
                        toggle_main_window(app);
                    }
                })
                .build(),
        )
        .manage(AppState::default())
        .setup(|app| {
            // ---- conversation database ----
            let dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&dir)?;
            let db = store::init_db(&dir.join("chaty.db"))?;
            app.manage(db);

            // ---- browser automation: persistent profile (logins survive) ----
            if let Ok(data) = app.path().app_data_dir() {
                browser::set_profile_dir(data.join("browser-profile"));
            }

            // ---- models folder (drop-in GGUF hot-swap) ----
            commands::ensure_models_dir(app.handle());
            // Folder layout: loose GGUFs migrate into one folder per model
            // (vision models keep their mmproj beside the weights). On the first
            // launch after updating from an old (loose-layout) version this pops
            // a one-time native dialog to organize; otherwise it's silent.
            commands::migrate_or_prompt_models(app.handle());
            // Leftovers of cancelled xet fallback downloads (CDN-blocked networks).
            download::clear_stale_xet_tmp(app.handle());

            // ---- chaty:// deep link ----
            // macOS registers the scheme via Info.plist (CFBundleURLTypes from
            // the plugin config) and Windows via the installer; Linux + Windows
            // dev need a runtime registration.
            #[cfg(any(target_os = "linux", all(debug_assertions, windows)))]
            {
                use tauri_plugin_deep_link::DeepLinkExt;
                let _ = app.deep_link().register_all();
            }

            // ---- macOS: unlock getUserMedia in WKWebView ----
            // WebKit ships `navigator.mediaDevices` behind a preferences flag
            // that's off for embedded webviews (Safari enables it for itself).
            // Without this, getUserMedia is simply absent no matter what TCC /
            // entitlements say. KVC onto WKPreferences flips it; the reload
            // makes the already-created page re-evaluate its bindings.
            #[cfg(target_os = "macos")]
            if let Some(w) = app.get_webview_window("main") {
                use objc2::msg_send;
                use objc2::runtime::AnyObject;
                use objc2_foundation::{NSNumber, NSString};
                let _ = w.with_webview(|webview| unsafe {
                    let wk: *mut AnyObject = webview.inner().cast();
                    let config: *mut AnyObject = msg_send![wk, configuration];
                    let prefs: *mut AnyObject = msg_send![config, preferences];
                    let yes = NSNumber::new_bool(true);
                    let key = NSString::from_str("mediaDevicesEnabled");
                    let _: () = msg_send![
                        prefs,
                        setValue: &*yes as *const NSNumber as *const AnyObject,
                        forKey: &*key
                    ];
                });
                let _ = w.eval("location.reload()");
            }

            // ---- system tray (labels default to English; the UI syncs the
            // language via `set_tray_language` on startup) ----
            let show_i = MenuItem::with_id(app, "show", "Show Chaty", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &quit_i])?;
            if let Some(icon) = app.default_window_icon().cloned() {
                TrayIconBuilder::with_id("main-tray")
                    .icon(icon)
                    .tooltip("Chaty")
                    .menu(&menu)
                    .show_menu_on_left_click(false)
                    .on_menu_event(|app, event| match event.id.as_ref() {
                        "show" => show_main_window(app),
                        "quit" => app.exit(0),
                        _ => {}
                    })
                    .on_tray_icon_event(|tray, event| {
                        if let TrayIconEvent::Click {
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        } = event
                        {
                            show_main_window(tray.app_handle());
                        }
                    })
                    .build(app)?;
            }

            // ---- global hotkey: summons/hides the window ----
            // macOS: Cmd+Shift+Space (Cmd+Space is Spotlight). Else: Ctrl+Shift+Space.
            // Non-fatal: a conflict with another app must not block startup.
            #[cfg(target_os = "macos")]
            let mods = Modifiers::SUPER | Modifiers::SHIFT;
            #[cfg(not(target_os = "macos"))]
            let mods = Modifiers::CONTROL | Modifiers::SHIFT;
            let _ = app
                .global_shortcut()
                .register(Shortcut::new(Some(mods), Code::Space));

            Ok(())
        })
        // Closing the window hides it to the tray instead of quitting.
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                // macOS: a fullscreen window lives in its own Space, and hiding
                // just the window there strands that Space on screen with
                // nothing in it (a black screen until the user swipes away).
                // Dropping out of fullscreen first only trades that for a
                // windowed frame flashing on the way out — and `orderOut:` is
                // dropped while the Space animates, so the window could end up
                // neither hidden nor fullscreen.
                //
                // Hiding the whole app is what macOS itself does for Cmd+H:
                // it slides out of the fullscreen Space in one motion, and
                // unhiding slides back in with the window STILL fullscreen.
                // That is the behavior a close-to-tray should have here.
                #[cfg(target_os = "macos")]
                if window.is_fullscreen().unwrap_or(false) {
                    // The ⌘H slide is the ONE exit the owner accepted — but
                    // macOS refuses NSApp.hide the whole time the fullscreen
                    // menu bar is revealed, and clicking the red X requires
                    // the mouse to be up there revealing it (measured:
                    // NSApp.isHidden stays false while the cursor rests on
                    // top, flips the moment it leaves). Exiting fullscreen
                    // instead trades that for system animations the owner
                    // rejected twice (windowed flash; ghost-titlebar shrink).
                    //
                    // So: be patient. Keep asking to hide — with the honest
                    // NSApp.isHidden signal, on the main thread — until the
                    // menu bar retracts (people move the mouse within moments
                    // of clicking) or a minute passes. Reopening meanwhile
                    // bumps HIDE_EPOCH, which cancels the pending hide so a
                    // fresh window can't be yanked back down.
                    let my_epoch = HIDE_EPOCH
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                        + 1;
                    let _ = window.app_handle().hide();
                    let w = window.clone();
                    std::thread::spawn(move || {
                        use std::sync::atomic::Ordering;
                        use std::time::Duration;
                        for _ in 0..240 {
                            std::thread::sleep(Duration::from_millis(250));
                            if HIDE_EPOCH.load(Ordering::SeqCst) != my_epoch {
                                return; // reopened — stand down
                            }
                            let (tx, rx) = std::sync::mpsc::channel::<bool>();
                            let wh = w.clone();
                            let epoch = my_epoch;
                            if w
                                .run_on_main_thread(move || {
                                    let hidden = objc2::MainThreadMarker::new()
                                        .map(|mtm| {
                                            objc2_app_kit::NSApplication::sharedApplication(mtm)
                                                .isHidden()
                                        })
                                        .unwrap_or(false);
                                    if !hidden && HIDE_EPOCH.load(Ordering::SeqCst) == epoch {
                                        let _ = wh.app_handle().hide();
                                    }
                                    let _ = tx.send(hidden);
                                })
                                .is_err()
                            {
                                return;
                            }
                            if rx.recv_timeout(Duration::from_secs(1)).unwrap_or(false) {
                                return; // hide landed with the proper slide
                            }
                        }
                        // A minute with the cursor parked on the menu bar —
                        // give up quietly; the window stays exactly as it is.
                    });
                    return;
                }
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::load_model,
            commands::eject_model,
            commands::get_model,
            commands::get_hardware_info,
            commands::get_gpu_usage,
            commands::write_text_file,
            commands::write_wav_file,
            download::list_hf_ggufs,
            download::list_hf_mlx,
            download::hf_search,
            download::hf_model_detail,
            download::hf_author_avatar,
            download::download_model,
            download::download_mlx_repo,
            download::cancel_download,
            update::check_update,
            update::run_update,
            commands::list_models,
            commands::delete_model_file,
            commands::open_models_dir,
            commands::open_data_dir,
            commands::open_html_report,
            commands::canvas_session_save,
            commands::canvas_session_load,
            commands::open_external,
            commands::set_ui_zoom,
            commands::set_tray_language,
            commands::generate,
            commands::cancel_generation,
            commands::vision_query,
            commands::image_thumb,
            commands::save_file,
            commands::transcribe,
            commands::synthesize,
            voice::request_mic_permission,
            mic::mic_start,
            mic::mic_level,
            mic::mic_stop,
            mic::mic_cancel,
            rag::rag_status,
            rag::rag_download_model,
            rag::rag_add_document,
            rag::rag_list_supported_files,
            rag::rag_list_documents,
            rag::rag_remove_document,
            rag::rag_search,
            rag::rag_set_doc_enabled,
            rag::rag_corpus,
            rag::rag_corpus_docs,
            rag::rag_clear_all,
            agent::agent_set_workspace,
            agent::agent_set_lang,
            agent::agent_set_edit_anchors,
            agent::agent_edit_lines,
            agent::agent_get_workspace,
            agent::agent_grant_dir,
            agent::agent_revoke_dir,
            agent::agent_list_grants,
            agent::agent_clear_grants,
            agent::agent_read_file,
            agent::agent_read_doc,
            agent::agent_validate_change,
            agent::agent_understand_repo,
            agent::agent_write_file,
            agent::agent_edit_file,
            agent::agent_multi_edit,
            agent::agent_outline,
            agent::agent_resolve_image,
            agent::browser_navigate,
            agent::browser_refresh,
            agent::browser_screenshot,
            agent::browser_snapshot,
            agent::browser_scroll,
            agent::browser_eval,
            agent::browser_click,
            agent::browser_type,
            agent::browser_console,
            agent::browser_read,
            agent::browser_close,
            agent::browser_set_headless,
            agent::browser_render_html,
            commands::image_data_url,
            agent::agent_list_dir,
            agent::agent_glob,
            agent::agent_grep,
            agent::agent_search_files,
            agent::agent_list_files,
            agent::agent_search_code,
            agent::agent_bash,
            agent::agent_bash_bg,
            agent::agent_bg_output,
            agent::agent_bg_kill,
            agent::agent_bg_reap,
            agent::agent_bg_list,
            agent::agent_checkpoint_begin,
            agent::agent_checkpoint_revert_to,
            store::save_conversation,
            store::save_message,
            store::replace_messages,
            store::list_conversations,
            store::get_messages,
            store::delete_conversation,
            store::clear_all_conversations,
            store::set_conversation_pinned,
            store::rename_conversation,
            store::code_session_save,
            store::code_session_list,
            store::code_session_load,
            store::code_session_delete,
            store::search_conversations,
            store::data_stats,
            search::web_search,
            search::web_research,
            search::fetch_url,
            webx::site_search,
            webx::fetch_page_ex,
            agent::agent_web_download,
            agent::agent_dl_list,
            agent::agent_dl_reap,
            attach::read_attachment,
            mcp::mcp_connect,
            mcp::mcp_disconnect,
            mcp::mcp_call,
        ])
        .build(tauri::generate_context!())
        .expect("error while running tauri application")
        .run(|app, event| {
            match event {
                // Exit the process before Tauri's teardown runs. Dropping the
                // live inference engine from the main thread races its worker
                // thread (llama.cpp context / Metal buffers) and segfaults on
                // quit, which macOS reports as "Chaty quit unexpectedly".
                // `_exit` (not `exit`) — ONNX Runtime / ggml also register
                // atexit handlers whose teardown crashes the same way; `_exit`
                // skips those too. Safe: SQLite is WAL-journaled, models are
                // read-only, and settings are persisted on change.
                // ExitRequested covers app.exit() (tray Quit); Exit covers the
                // Cocoa `terminate:` path (app-menu Quit / Cmd+Q / logout),
                // which skips ExitRequested and was still reaching ggml's
                // teardown (ggml_metal_rsets_free → ggml_abort → SIGABRT).
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
                    // Same fullscreen trap as hide-to-tray: dying inside a
                    // fullscreen Space leaves that Space up with nothing in it.
                    // We can't wait out the animation here (the `_exit` below is
                    // what keeps ggml's teardown from crashing, and blocking the
                    // main thread would stall the transition anyway), but asking
                    // to leave fullscreen before we go lets the window server
                    // collapse the Space on its own.
                    #[cfg(target_os = "macos")]
                    if let Some(w) = app.get_webview_window("main") {
                        if w.is_fullscreen().unwrap_or(false) {
                            let _ = w.set_fullscreen(false);
                            std::thread::sleep(std::time::Duration::from_millis(250));
                        }
                    }
                    // Kill the automation browser first — the _exit below skips
                    // destructors, which would otherwise orphan Chrome.
                    browser::kill_now();
                    // Same for the MLX sidecar: skipping Drop would orphan a
                    // process holding the whole model in unified memory.
                    inference::mlx::kill_sidecars_now();
                    // And MCP stdio servers — same orphan risk, same reap.
                    mcp::kill_all_now();
                    #[cfg(unix)]
                    unsafe {
                        libc::_exit(0)
                    }
                    #[cfg(not(unix))]
                    std::process::exit(0)
                }
                // macOS: clicking the Dock icon while the window is hidden to
                // the tray must bring it back (no window is ever re-created).
                #[cfg(target_os = "macos")]
                tauri::RunEvent::Reopen { .. } => show_main_window(app),
                _ => {}
            }
        });
}
