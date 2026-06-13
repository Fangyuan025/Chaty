pub mod attach;
pub mod mic;
pub mod rag;
mod commands;
pub mod download;
pub mod gpu;
mod inference;
pub mod ocr;
pub mod search;
mod state;
mod store;
pub mod update;
pub mod voice;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

use state::AppState;

/// Bring the main window to the foreground.
fn show_main_window(app: &tauri::AppHandle) {
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
            show_main_window(app);
        }))
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

            // ---- models folder (drop-in GGUF hot-swap) ----
            commands::ensure_models_dir(app.handle());

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
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::load_model,
            commands::get_model,
            commands::get_hardware_info,
            commands::get_gpu_usage,
            commands::write_text_file,
            commands::write_wav_file,
            download::list_hf_ggufs,
            download::download_model,
            download::cancel_download,
            update::check_update,
            update::run_update,
            commands::list_models,
            commands::open_models_dir,
            commands::set_tray_language,
            commands::generate,
            commands::cancel_generation,
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
            rag::rag_list_documents,
            rag::rag_remove_document,
            rag::rag_search,
            rag::rag_set_doc_enabled,
            rag::rag_corpus,
            store::save_conversation,
            store::save_message,
            store::replace_messages,
            store::list_conversations,
            store::get_messages,
            store::delete_conversation,
            store::rename_conversation,
            store::search_conversations,
            search::web_search,
            search::web_research,
            search::fetch_url,
            attach::read_attachment,
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
