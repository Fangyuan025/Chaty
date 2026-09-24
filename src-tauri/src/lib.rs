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
pub mod imagegen;
pub mod inference;
pub mod errlog;
pub mod mcp;
pub mod ocr;
pub mod search;
pub mod skillsync;
mod state;
mod store;
mod user_skills;
mod terminal;
pub mod update;
pub mod voice;
pub mod webx;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

use state::AppState;

/// Generation counter for pending close-from-fullscreen hides: showing the
/// window (tray / Dock / shortcut) bumps it, which cancels a close still
/// waiting for fullscreen to finish exiting — otherwise a reopen during the
/// exit animation would be hidden again the moment it ended.
#[cfg(target_os = "macos")]
static HIDE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The close waiting for fullscreen to finish exiting before it hides the
/// window, by the HIDE_EPOCH it was issued under; 0 when none is.
#[cfg(target_os = "macos")]
static PENDING_HIDE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bring the main window to the foreground.
fn show_main_window(app: &tauri::AppHandle) {
    #[cfg(target_os = "macos")]
    {
        // A new epoch cancels a close still waiting for fullscreen to finish
        // exiting, so the window being summoned is not hidden under the user.
        HIDE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // And undo an app-level hide (⌘H), should there be one.
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
            hide_main_window(app);
        } else {
            let _ = w.show();
            let _ = w.unminimize();
            let _ = w.set_focus();
        }
    }
}

/// Put the main window away to the tray.
///
/// On macOS a fullscreen window lives in a Space of its own. Hiding it there
/// strands that Space on screen with nothing in it, and hiding the whole app
/// instead (the ⌘H slide) is refused for as long as the fullscreen menu bar is
/// showing — which it always is, because that is where the red X is. That made
/// closing from fullscreen a matter of luck: it waited for the cursor to leave
/// the menu bar, and after a minute gave up with the window still up. So a
/// fullscreen window now leaves fullscreen first, with the system's own
/// animation, and is hidden the moment that animation has finished (see
/// `watch_fullscreen_exit`) — `orderOut:` sent while the Space is still
/// animating is dropped. It then comes back from the tray as a normal window.
fn hide_main_window<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    let Some(w) = app.get_webview_window("main") else {
        return;
    };
    #[cfg(target_os = "macos")]
    if w.is_fullscreen().unwrap_or(false) {
        use std::sync::atomic::Ordering;
        let epoch = HIDE_EPOCH.fetch_add(1, Ordering::SeqCst) + 1;
        PENDING_HIDE.store(epoch, Ordering::SeqCst);
        let _ = w.set_fullscreen(false);
        // Should the exit never report back, hide anyway rather than leave
        // the click unanswered.
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(3));
            if HIDE_EPOCH.load(Ordering::SeqCst) == epoch
                && PENDING_HIDE
                    .compare_exchange(epoch, 0, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                let _ = w.hide();
            }
        });
        return;
    }
    let _ = w.hide();
}

/// Finish a close that had to leave fullscreen first: AppKit posts
/// NSWindowDidExitFullScreenNotification once the exit animation is over, and
/// that is when the window can be ordered out. Registered once, for the life
/// of the app; a reopen in the meantime moves HIDE_EPOCH on and the pending
/// close no longer matches.
#[cfg(target_os = "macos")]
fn watch_fullscreen_exit<R: tauri::Runtime>(w: &tauri::WebviewWindow<R>) {
    use block2::RcBlock;
    use objc2::runtime::AnyObject;
    use objc2_foundation::{NSNotification, NSNotificationCenter, NSString};
    use std::ptr::NonNull;
    use std::sync::atomic::Ordering;
    let Ok(ns_window) = w.ns_window() else {
        return;
    };
    let win = w.clone();
    let block = RcBlock::new(move |_: NonNull<NSNotification>| {
        let pending = PENDING_HIDE.load(Ordering::SeqCst);
        if pending != 0
            && pending == HIDE_EPOCH.load(Ordering::SeqCst)
            && PENDING_HIDE
                .compare_exchange(pending, 0, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            let _ = win.hide();
        }
    });
    let name = NSString::from_str("NSWindowDidExitFullScreenNotification");
    // SAFETY: `ns_window` is the NSWindow behind the main window, which lives
    // as long as the app; the block touches only atomics and a Send handle.
    let token = unsafe {
        NSNotificationCenter::defaultCenter().addObserverForName_object_queue_usingBlock(
            Some(&name),
            Some(&*ns_window.cast::<AnyObject>()),
            None,
            &block,
        )
    };
    // The observer is meant to last as long as the app does.
    std::mem::forget(token);
}

/// The window's opening size, in points, for a screen whose usable area is
/// `work_w` × `work_h` points — scaled to the screen rather than a fixed
/// 1040×720, which is half of a 1080p screen and a sliver of anything bigger.
/// The proportions are the ones Claude's desktop app opens at: side by side on
/// a 1920-point screen it measured about 1380×866.
/// Held above the minimum the layout was built for, and below the point where
/// a window stops being one and becomes a wall.
fn default_window_size(work_w: f64, work_h: f64) -> (f64, f64) {
    let w = (work_w * 0.719).clamp(1040.0, 1680.0).min(work_w);
    // Height follows from the width at that app's proportion, so a screen with
    // the Dock along the bottom does not get a letterbox — capped by what the
    // screen actually has to give.
    let h = (w / 1.595).min(work_h * 0.95).clamp(720.0, 1050.0).min(work_h);
    (w.round(), h.round())
}

/// Size the main window for the screen it opens on, centre it, and show it.
/// It is created hidden (tauri.conf.json) so it never appears at one size and
/// then jumps to another — and whatever happens here, it ends up shown.
fn open_main_window<R: tauri::Runtime>(app: &tauri::App<R>) {
    let Some(w) = app.get_webview_window("main") else {
        return;
    };
    let monitor = w
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| w.primary_monitor().ok().flatten());
    if let Some(m) = monitor {
        let scale = m.scale_factor();
        let area = m.work_area();
        let (lw, lh) = default_window_size(
            f64::from(area.size.width) / scale,
            f64::from(area.size.height) / scale,
        );
        let _ = w.set_size(tauri::LogicalSize::new(lw, lh));
        // Centred on the usable area by hand. `center()` measures the window
        // as it is when it runs, and the resize queued just before it has not
        // landed yet: it centred the old 1040×720, and the new size then grew
        // past the right edge of the screen.
        let left = f64::from(area.position.x);
        let top = f64::from(area.position.y);
        let x = left + ((f64::from(area.size.width) - lw * scale) / 2.0).max(0.0);
        let y = top + ((f64::from(area.size.height) - lh * scale) / 2.0).max(0.0);
        let _ = w.set_position(tauri::PhysicalPosition::new(x.round() as i32, y.round() as i32));
    }
    let _ = w.show();
}

#[cfg(test)]
mod window_size_tests {
    use super::default_window_size;

    #[test]
    fn the_window_opens_at_most_of_the_screen_within_bounds() {
        // A 1080p-class Mac, menu bar and a bottom Dock taken: the size
        // Claude's desktop app opens at on the same screen.
        assert_eq!(default_window_size(1920.0, 963.0), (1380.0, 866.0));
        // The same screen with the Dock hidden: the proportion holds.
        assert_eq!(default_window_size(1920.0, 1055.0), (1380.0, 866.0));
        // A 1440p display reaches the ceiling.
        assert_eq!(default_window_size(2560.0, 1415.0), (1680.0, 1050.0));
        // A small laptop: never under the size the layout was built for.
        assert_eq!(default_window_size(1280.0, 775.0), (1040.0, 720.0));
        // A large display: a window, not a wall.
        assert_eq!(default_window_size(3008.0, 1667.0), (1680.0, 1050.0));
        // A screen smaller than that minimum: the window still fits on it.
        assert_eq!(default_window_size(1000.0, 700.0), (1000.0, 700.0));
    }

    /// The Windows/Linux tray icon fills its slot: no transparent margin left
    /// on any side (issue #18 — the app icon's margin made it the smallest
    /// icon in the tray).
    #[test]
    fn the_tray_icon_is_cropped_to_its_square() {
        let png = include_bytes!("../icons/icon.png");
        let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
            .unwrap()
            .to_rgba8();
        let mid = img.height() / 2;
        assert!(img.get_pixel(0, mid)[3] < 16, "the app icon has a margin to crop");
        let icon = super::crop_to_content(&img, 32);
        assert_eq!(icon.dimensions(), (32, 32));
        for (x, y) in [(0, 16), (31, 16), (16, 0), (16, 31)] {
            assert!(icon.get_pixel(x, y)[3] > 200, "edge ({x},{y}) is still transparent");
        }
    }
}

/// `img` cropped to the square around its visible pixels, then scaled to
/// `size`×`size`.
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn crop_to_content(img: &image::RgbaImage, size: u32) -> image::RgbaImage {
    use image::imageops::{self, FilterType};
    let (w, h) = img.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (x, y, p) in img.enumerate_pixels() {
        if p[3] > 16 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x0 > x1 || y0 > y1 {
        return imageops::resize(img, size, size, FilterType::Lanczos3);
    }
    let side = (x1 - x0 + 1).max(y1 - y0 + 1);
    let square = imageops::crop_imm(img, x0, y0, side.min(w - x0), side.min(h - y0)).to_image();
    imageops::resize(&square, size, size, FilterType::Lanczos3)
}

/// The tray icon on Windows and Linux: the app icon cropped to its rounded
/// square. The app icon keeps a macOS-style transparent margin — an eighth of
/// its width on every side — and a tray draws the whole canvas into its slot,
/// so the square came out at three quarters of the size of the icons beside
/// it (issue #18). 32 px: a 16-px slot at 100% and 200% scaling both divide it
/// evenly.
#[cfg(not(target_os = "macos"))]
fn tray_colour_icon() -> Option<tauri::image::Image<'static>> {
    let png = include_bytes!("../icons/icon.png");
    let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .ok()?
        .to_rgba8();
    let icon = crop_to_content(&img, 32);
    let (w, h) = icon.dimensions();
    Some(tauri::image::Image::new_owned(icon.into_raw(), w, h))
}

/// The menu-bar glyph: the app icon's own bubble and graph, drawn black on
/// transparency (icons/tray-template.svg, rendered by
/// scripts/render-tray-icon.swift). 36 px because tray-icon sets the status
/// item 18 pt tall — exactly @2x on a Retina bar. Decoded with the `image`
/// crate the app already carries, rather than widening tauri's features for
/// one PNG.
#[cfg(target_os = "macos")]
fn tray_template_icon() -> Option<tauri::image::Image<'static>> {
    let png = include_bytes!("../icons/tray-template.png");
    let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .ok()?
        .to_rgba8();
    let (w, h) = img.dimensions();
    Some(tauri::image::Image::new_owned(img.into_raw(), w, h))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Panics land in the user-attachable error log from the first instant.
    crate::errlog::install_panic_hook();
    // Native crashes kill the process before any hook — on macOS, surface
    // the OS crash reports left behind by the previous run.
    #[cfg(target_os = "macos")]
    crate::errlog::sweep_native_crash_reports();
    // ...and everywhere, report work the previous run started and never
    // finished. Windows has no crash report to sweep, so without this a death
    // inside native code leaves the log empty (issue #13).
    crate::errlog::sweep_inflight();
    // Browsers whose Chaty died without destructors (the exit handler
    // `_exit()`s, crashes, killed bench bridges) keep running headless —
    // reap them before this run launches its own.
    crate::browser::sweep_orphan_browsers();
    // Official-skill support files follow their public upstream quietly (24h
    // throttle, offline ⇒ bundled files) — never on the startup path.
    std::thread::spawn(crate::skillsync::tick);
    #[cfg(unix)]
    agent::install_termination_cleanup();
    // GPU crash guard (issue #5): if the previous model load took the whole
    // process down (broken Vulkan driver aborts mid-load), block GPU offload
    // for this run BEFORE any llama/ggml init touches the driver.
    let _gpu_cap = crate::inference::llama::apply_gpu_crash_guard();
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
            // ---- main window: sized for the screen, then shown ----
            open_main_window(app);
            #[cfg(target_os = "macos")]
            if let Some(w) = app.get_webview_window("main") {
                watch_fullscreen_exit(&w);
            }

            // ---- conversation database ----
            let dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&dir)?;
            let db = store::init_db(&dir.join("chaty.db"))?;
            app.manage(db);
            // Code sessions' background task history, in the same database.
            agent::init_bg_history(&dir.join("chaty.db"));

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
            // macOS draws menu-bar items as TEMPLATE images — a black glyph on
            // transparency that the system tints to match the bar, light or
            // dark, like every other item up there. The app icon is a
            // full-colour navy square, and in a row of glyphs it was the one
            // thing that was not one. Windows and Linux trays expect colour and
            // keep the app icon.
            #[cfg(target_os = "macos")]
            let (tray_icon, is_template) = match tray_template_icon() {
                Some(glyph) => (Some(glyph), true),
                // Never the colour icon AS a template: macOS would keep only
                // its alpha, and a rounded square with no holes in it is a
                // solid blob.
                None => (app.default_window_icon().cloned(), false),
            };
            #[cfg(not(target_os = "macos"))]
            let (tray_icon, is_template) =
                (tray_colour_icon().or_else(|| app.default_window_icon().cloned()), false);
            if let Some(icon) = tray_icon {
                TrayIconBuilder::with_id("main-tray")
                    .icon(icon)
                    .icon_as_template(is_template)
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
                // Out of fullscreen first when it is in it — see hide_main_window.
                if window.label() == "main" {
                    hide_main_window(window.app_handle());
                } else {
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::load_model,
            errlog::log_app_error,
            errlog::open_error_log,
            commands::eject_model,
            commands::get_model,
            commands::get_hardware_info,
            commands::get_gpu_usage,
            commands::note_frontend_ready,
            commands::attach_generation,
            errlog::clear_error_log,
            commands::get_gpu_layer_cap,
            commands::reset_gpu_layer_cap,
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
            commands::get_models_root,
            commands::set_models_root,
            commands::open_data_dir,
            commands::open_voice_models_dir,
            commands::open_html_report,
            commands::canvas_session_save,
            commands::canvas_session_load,
            commands::open_external,
            commands::set_ui_zoom,
            commands::set_tray_language,
            commands::generate,
            commands::cancel_generation,
            imagegen::image_model_probe,
            imagegen::image_generate,
            imagegen::image_cancel,
            imagegen::image_attach,
            imagegen::image_output_dir,
            imagegen::image_copy,
            store::image_session_save,
            store::image_session_list,
            store::image_session_get,
            store::image_session_draft,
            store::image_session_rename,
            store::image_session_set_pinned,
            store::image_session_delete,
            store::image_session_search,
            store::image_generation_delete,
            store::image_history_clear,
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
            agent::agent_shells,
            agent::agent_set_shell,
            agent::agent_set_edit_anchors,
            agent::agent_edit_lines,
            agent::agent_get_workspace,
            skillsync::skill_live_support,
            agent::agent_grant_dir,
            agent::agent_revoke_dir,
            agent::agent_list_grants,
            agent::agent_clear_grants,
            agent::agent_read_file,
            agent::agent_read_file_raw,
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
            agent::browser_key,
            agent::browser_console,
            agent::browser_read,
            agent::browser_close,
            agent::browser_set_headless,
            agent::browser_render_html,
            commands::image_data_url,
            agent::agent_list_dir,
            agent::agent_glob,
            user_skills::skills_list_user,
            user_skills::skills_import,
            user_skills::skills_delete_user,
            agent::agent_grep,
            agent::agent_search_files,
            agent::agent_list_files,
            agent::agent_search_code,
            agent::agent_bash,
            agent::agent_bash_bg,
            agent::agent_bg_output,
            agent::agent_bg_kill,
            agent::agent_bg_input,
            agent::agent_bg_reap,
            agent::agent_bg_list,
            agent::agent_bg_all,
            agent::agent_bg_log,
            agent::agent_bg_clear_finished,
            agent::agent_set_session,
            agent::agent_checkpoint_changes,
            agent::agent_checkpoint_revert_file,
            agent::agent_restore_file,
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
            store::code_session_search,
            store::code_session_read,
            store::code_step_text_put,
            store::code_step_text_get,
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
                    // Background jobs run in process groups of their own, so
                    // they would outlive the app: stop them, and write down
                    // what became of each in its session's history.
                    agent::bg_kill_all();
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
                    // Same for the MLX and image-engine sidecars: skipping
                    // Drop would orphan a process holding the whole model.
                    inference::kill_sidecars_now();
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


#[cfg(test)]
mod tray_icon_tests {
    /// A template image is read for its alpha alone: anything coloured in it
    /// is thrown away, and anything opaque that should have been a hole fills
    /// in. So the asset must be exactly a black glyph on transparency, at the
    /// size tray-icon draws it.
    #[test]
    fn the_menu_bar_glyph_is_a_black_template_at_retina_size() {
        let png = include_bytes!("../icons/tray-template.png");
        let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
            .expect("decodes")
            .to_rgba8();
        assert_eq!(img.dimensions(), (36, 36), "18 pt at @2x");
        let (mut clear, mut ink) = (0, 0);
        for p in img.pixels() {
            let [r, g, b, a] = p.0;
            if a == 0 {
                clear += 1;
            } else {
                ink += 1;
                assert!(r < 16 && g < 16 && b < 16, "opaque pixels must be black, got {:?}", p.0);
            }
        }
        assert!(clear > ink, "mostly transparency — it is a glyph, not a tile");
        assert!(ink > 100, "and there is a glyph");
        // The corners are where the old icon's navy square was.
        for (x, y) in [(0, 0), (35, 0), (0, 35), (35, 35)] {
            assert_eq!(img.get_pixel(x, y).0[3], 0, "corner ({x},{y}) must be clear");
        }
    }
}
