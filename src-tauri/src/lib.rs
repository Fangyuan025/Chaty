pub mod attach;
mod commands;
mod inference;
pub mod ocr;
pub mod search;
mod state;
mod store;
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
    // before the webview is created.
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

            // ---- global hotkey: Ctrl+Shift+Space summons/hides the window ----
            // Non-fatal: a conflict with another app must not block startup.
            let _ = app.global_shortcut().register(Shortcut::new(
                Some(Modifiers::CONTROL | Modifiers::SHIFT),
                Code::Space,
            ));

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
            commands::list_models,
            commands::set_tray_language,
            commands::generate,
            commands::cancel_generation,
            commands::transcribe,
            commands::synthesize,
            store::save_conversation,
            store::save_message,
            store::list_conversations,
            store::get_messages,
            store::delete_conversation,
            store::rename_conversation,
            search::web_search,
            search::web_research,
            search::fetch_url,
            attach::read_attachment,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
