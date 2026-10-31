//! The desktop's own file chooser on Linux, through the XDG Desktop Portal
//! (issue #29): Nautilus's on GNOME, Dolphin's on KDE. The dialog plugin's
//! GTK chooser is kept for where no portal answers — the frontend shows it
//! when this says `Unavailable` (src/lib/dialog.ts).
//!
//! Not the plugin's own `xdg-portal` feature: with it, its message boxes need
//! zenity, and a desktop with neither a portal nor zenity could not pick a
//! folder at all.

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct PickRequest {
    save: bool,
    title: Option<String>,
    directory: bool,
    multiple: bool,
    default_path: Option<String>,
    filters: Vec<PickFilter>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct PickFilter {
    name: String,
    extensions: Vec<String>,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub enum Picked {
    /// No portal answered: the frontend shows the GTK chooser instead.
    Unavailable,
    Cancelled,
    Paths { paths: Vec<String> },
}

#[tauri::command]
pub async fn portal_pick(window: tauri::Window, request: PickRequest) -> Picked {
    #[cfg(target_os = "linux")]
    {
        let parent = linux::parent(&window);
        linux::pick(parent, request).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (window, request);
        Picked::Unavailable
    }
}

/// The same with no window to keep the chooser on top of (chaty-headless).
pub async fn pick_unparented(request: PickRequest) -> Picked {
    #[cfg(target_os = "linux")]
    {
        linux::pick(None, request).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = request;
        Picked::Unavailable
    }
}

/// Where a dialog starts, from the plugin's `defaultPath`: a folder opens on
/// that folder; anything else names the file, in its folder.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn start_at(default: Option<&str>) -> (Option<std::path::PathBuf>, Option<String>) {
    let Some(p) = default.filter(|p| !p.is_empty()).map(std::path::Path::new) else {
        return (None, None);
    };
    if p.is_dir() {
        return (p.is_absolute().then(|| p.to_path_buf()), None);
    }
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned());
    let dir = p.parent().filter(|d| d.is_absolute() && d.is_dir()).map(|d| d.to_path_buf());
    (dir, name)
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{start_at, PickRequest, Picked};
    use ashpd::desktop::file_chooser::{FileChooserProxy, FileFilter, SelectedFiles};
    use ashpd::desktop::ResponseError;
    use ashpd::WindowIdentifier;
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use std::time::Duration;

    /// The chooser is kept on top of the app's window. The AppImage runs the
    /// window under X11; a Wayland one would need exporting, and the chooser
    /// then just opens on its own.
    pub fn parent(window: &tauri::Window) -> Option<WindowIdentifier> {
        match window.window_handle().ok()?.as_raw() {
            RawWindowHandle::Xlib(h) => Some(WindowIdentifier::from_xid(h.window)),
            RawWindowHandle::Xcb(h) => Some(WindowIdentifier::from_xid(h.window.get().into())),
            _ => None,
        }
    }

    fn filters(req: &PickRequest) -> Vec<FileFilter> {
        req.filters
            .iter()
            .map(|f| {
                f.extensions.iter().fold(FileFilter::new(&f.name), |filter, ext| {
                    if ext.is_empty() || ext == "*" {
                        filter.glob("*")
                    } else {
                        filter.glob(&format!("*.{ext}"))
                    }
                })
            })
            .collect()
    }

    pub async fn pick(parent: Option<WindowIdentifier>, req: PickRequest) -> Picked {
        // Choosing a folder came with version 3 of the interface; an older
        // portal would let a file be picked where a folder is asked for.
        let probe = tokio::time::timeout(Duration::from_secs(5), FileChooserProxy::new()).await;
        match probe {
            Ok(Ok(proxy)) if !req.directory || proxy.version() >= 3 => {}
            Ok(Ok(proxy)) => {
                eprintln!("[portal] file chooser v{} cannot pick folders", proxy.version());
                return Picked::Unavailable;
            }
            Ok(Err(e)) => {
                eprintln!("[portal] no file chooser: {e}");
                return Picked::Unavailable;
            }
            Err(_) => {
                eprintln!("[portal] no file chooser: no answer in 5 s");
                return Picked::Unavailable;
            }
        }
        let (dir, name) = start_at(req.default_path.as_deref());
        let sent = if req.save {
            match SelectedFiles::save_file()
                .identifier(parent)
                .title(req.title.as_deref())
                .modal(true)
                .current_name(name.as_deref())
                .filters(filters(&req))
                .current_folder::<&std::path::Path>(dir.as_deref())
            {
                Ok(r) => r.send().await,
                Err(e) => Err(e),
            }
        } else {
            match SelectedFiles::open_file()
                .identifier(parent)
                .title(req.title.as_deref())
                .modal(true)
                .multiple(req.multiple)
                .directory(req.directory)
                .filters(filters(&req))
                .current_folder::<&std::path::Path>(dir.as_deref())
            {
                Ok(r) => r.send().await,
                Err(e) => Err(e),
            }
        };
        match sent.and_then(|r| r.response()) {
            Ok(files) => Picked::Paths {
                paths: files
                    .uris()
                    .iter()
                    .filter_map(|u| url::Url::parse(u.as_str()).ok()?.to_file_path().ok())
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect(),
            },
            Err(ashpd::Error::Response(ResponseError::Cancelled)) => Picked::Cancelled,
            Err(e) => {
                eprintln!("[portal] file chooser failed: {e}");
                Picked::Unavailable
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dialog_starts_where_the_default_path_says() {
        let dir = std::env::temp_dir().canonicalize().unwrap();
        let d = dir.to_string_lossy().into_owned();
        assert_eq!(start_at(Some(d.as_str())), (Some(dir.clone()), None));
        let file = dir.join("chat.md").to_string_lossy().into_owned();
        assert_eq!(start_at(Some(file.as_str())), (Some(dir), Some("chat.md".into())));
        assert_eq!(start_at(Some("chat.md")), (None, Some("chat.md".into())));
        assert_eq!(start_at(Some("")), (None, None));
        assert_eq!(start_at(None), (None, None));
    }

    #[test]
    fn the_request_reads_what_the_frontend_sends() {
        let r: PickRequest = serde_json::from_value(serde_json::json!({
            "directory": true,
            "defaultPath": "/tmp",
            "filters": [{ "name": "Markdown", "extensions": ["md"] }]
        }))
        .unwrap();
        assert!(r.directory && !r.multiple && !r.save);
        assert_eq!(r.default_path.as_deref(), Some("/tmp"));
        assert_eq!(r.filters[0].extensions, ["md"]);
        assert_eq!(serde_json::to_value(Picked::Unavailable).unwrap(), serde_json::json!({ "kind": "unavailable" }));
    }
}
