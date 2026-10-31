//! `chaty://` links on Linux, which the app registers itself as it starts.
//!
//! The deep-link plugin's registration writes a handler `.desktop` file and
//! runs `update-desktop-database` and `xdg-mime` with this process's
//! environment — in the AppImage, the image's library paths (`hostenv`), so
//! the host's xdg-mime ran its tools on the image's older libraries
//! (issue #28). These are the same steps, with the user's environment.
//! macOS takes the scheme from Info.plist, Windows from the installer.

#[cfg(target_os = "linux")]
pub fn register(app: &tauri::AppHandle) {
    use crate::hostenv::HostEnv;
    use tauri::Manager;

    // tauri.conf.json → plugins.deep-link.desktop.schemes
    const SCHEME: &str = "chaty";
    let Ok(bin) = tauri::utils::platform::current_exe() else { return };
    let Some(exe_name) = bin.file_name().map(|n| n.to_string_lossy().into_owned()) else { return };
    let Ok(dir) = app.path().data_dir().map(|d| d.join("applications")) else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    // The AppImage itself, not the binary inside its mount, which is gone
    // once the app quits.
    let exec = app.env().appimage.unwrap_or_else(|| bin.into_os_string());
    let file_name = format!("{exe_name}-handler.desktop");
    let mime = format!("x-scheme-handler/{SCHEME}");
    let entry = handler_entry(&exec.to_string_lossy(), &mime);
    let path = dir.join(&file_name);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(entry.as_str()) {
        if let Err(e) = std::fs::write(&path, &entry) {
            eprintln!("[scheme] could not write {}: {e}", path.display());
            return;
        }
    }
    for (program, args) in [
        ("update-desktop-database", vec![dir.to_string_lossy().into_owned()]),
        ("xdg-mime", vec!["default".into(), file_name.clone(), mime.clone()]),
    ] {
        if let Err(e) = std::process::Command::new(program).args(&args).host_env().status() {
            eprintln!("[scheme] {program}: {e}");
        }
    }
}

/// The handler: hidden from menus, opening a link with the app.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn handler_entry(exec: &str, mime: &str) -> String {
    format!(
        "[Desktop Entry]\nType=Application\nName=Chaty\nExec=\"{exec}\" %u\nTerminal=false\nMimeType={mime};\nNoDisplay=true\n"
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_handler_opens_links_with_the_appimage() {
        let e = super::handler_entry("/home/me/Apps/Chaty_2.3.9_amd64.AppImage", "x-scheme-handler/chaty");
        assert!(e.contains("Exec=\"/home/me/Apps/Chaty_2.3.9_amd64.AppImage\" %u\n"));
        assert!(e.contains("MimeType=x-scheme-handler/chaty;\n"));
        assert!(e.contains("NoDisplay=true"));
    }
}
