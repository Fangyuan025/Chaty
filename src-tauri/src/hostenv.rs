//! The environment a program of the user's own system starts with.
//!
//! The Linux AppImage's start scripts point PATH, LD_LIBRARY_PATH,
//! XDG_DATA_DIRS and the GTK, GIO, GStreamer, Python and Qt variables into
//! the mounted image, so that the app finds the libraries it ships. Every
//! program the app started inherited them: "Open models folder" ran the
//! image's copy of `xdg-open`, and the file manager, Firefox or git it led to
//! loaded the image's older libnss3, libssl or libgnutls and did not start
//! (issue #28). Programs of the user's system — a folder or link opened,
//! agent commands, terminals, the browser, MCP servers — start without those
//! entries. The app's own engines, which use the image's libraries, keep them.
//!
//! Outside an AppImage nothing is changed.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::sync::OnceLock;

/// Variables the image's start scripts set to a value of their own that is
/// not a list of paths inside it.
const SET_BY_IMAGE: &[&str] = &[
    "APPDIR",
    "APPIMAGE",
    "ARGV0",
    "OWD",
    "GDK_BACKEND",
    "GTK_THEME",
    "PYTHONDONTWRITEBYTECODE",
    "GST_REGISTRY_REUSE_PLUGIN_SCANNER",
];

/// A command that can be given the environment of the user's system.
pub trait HostEnv {
    /// Take the AppImage's entries out of the environment it starts with.
    fn host_env(&mut self) -> &mut Self;
}

impl HostEnv for std::process::Command {
    fn host_env(&mut self) -> &mut Self {
        for (k, v) in changes() {
            match v {
                Some(v) => self.env(k, v),
                None => self.env_remove(k),
            };
        }
        self
    }
}

impl HostEnv for portable_pty::CommandBuilder {
    fn host_env(&mut self) -> &mut Self {
        for (k, v) in changes() {
            match v {
                Some(v) => self.env(k, v),
                None => self.env_remove(k),
            }
        }
        self
    }
}

/// `name` as a program of the user's system sees it.
pub fn var(name: &str) -> Option<String> {
    match changes().iter().find(|(k, _)| k == name) {
        Some((_, v)) => v.as_ref().map(|v| v.to_string_lossy().into_owned()),
        None => std::env::var(name).ok(),
    }
}

/// What to change in this process's environment for a program of the user's
/// system: a new value, or `None` to remove the variable.
fn changes() -> &'static [(OsString, Option<OsString>)] {
    static CHANGES: OnceLock<Vec<(OsString, Option<OsString>)>> = OnceLock::new();
    CHANGES.get_or_init(|| match image_dir() {
        Some(dir) => scrubbed(std::env::vars_os(), &dir),
        None => Vec::new(),
    })
}

/// The mounted (or extracted) AppImage this process runs from.
fn image_dir() -> Option<OsString> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let dir = std::env::var_os("APPDIR").filter(|d| !d.is_empty())?;
    // An APPDIR this process does not run from was inherited from something
    // else (a dev build started from inside another AppImage's terminal).
    let exe = std::env::current_exe().ok()?;
    let real = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone().into());
    (exe.starts_with(&dir) || exe.starts_with(&real)).then_some(dir)
}

fn scrubbed(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
    dir: &OsStr,
) -> Vec<(OsString, Option<OsString>)> {
    let dir = Path::new(dir);
    let inside = |entry: &str| !entry.is_empty() && Path::new(entry).starts_with(dir);
    let mut out = Vec::new();
    for (k, v) in vars {
        if SET_BY_IMAGE.iter().any(|s| OsStr::new(s) == k) {
            out.push((k, None));
            continue;
        }
        let v = v.to_string_lossy();
        let entries: Vec<&str> = v.split(':').collect();
        if !entries.iter().any(|e| inside(e)) {
            continue;
        }
        // The scripts append the old value after a ':', which leaves an
        // empty entry (the current directory, for PATH) when there was none.
        let kept: Vec<&str> = entries.into_iter().filter(|e| !e.is_empty() && !inside(e)).collect();
        out.push((k, (!kept.is_empty()).then(|| kept.join(":").into())));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(vars: &[(&str, &str)]) -> Vec<(String, Option<String>)> {
        let vars = vars.iter().map(|(k, v)| (OsString::from(k), OsString::from(v)));
        scrubbed(vars, OsStr::new("/tmp/.mount_chatyAb12"))
            .into_iter()
            .map(|(k, v)| (k.to_string_lossy().into_owned(), v.map(|v| v.to_string_lossy().into_owned())))
            .collect()
    }

    #[test]
    fn the_images_entries_are_taken_out() {
        let got = run(&[
            ("PATH", "/tmp/.mount_chatyAb12/usr/bin/:/tmp/.mount_chatyAb12/usr/sbin/:/usr/local/bin:/usr/bin"),
            ("LD_LIBRARY_PATH", "/tmp/.mount_chatyAb12/usr/lib/:/tmp/.mount_chatyAb12/lib64/:"),
            ("XDG_DATA_DIRS", "/tmp/.mount_chatyAb12/usr/share:/usr/share:/usr/local/share:/usr/share"),
            ("GTK_PATH", "/tmp/.mount_chatyAb12//usr/lib/gtk-3.0"),
            ("GTK_THEME", "Adwaita:dark"),
            ("APPIMAGE", "/home/me/Chaty.AppImage"),
            ("HOME", "/home/me"),
            ("LANG", "en_US.UTF-8"),
        ]);
        assert_eq!(
            got,
            vec![
                ("PATH".into(), Some("/usr/local/bin:/usr/bin".into())),
                ("LD_LIBRARY_PATH".into(), None),
                ("XDG_DATA_DIRS".into(), Some("/usr/share:/usr/local/share:/usr/share".into())),
                ("GTK_PATH".into(), None),
                ("GTK_THEME".into(), None),
                ("APPIMAGE".into(), None),
            ]
        );
    }

    #[test]
    fn a_path_that_only_starts_like_the_image_is_kept() {
        let got = run(&[("PATH", "/tmp/.mount_chatyAb12/usr/bin:/tmp/.mount_chatyAb123/bin:/usr/bin")]);
        assert_eq!(got, vec![("PATH".into(), Some("/tmp/.mount_chatyAb123/bin:/usr/bin".into()))]);
    }

    #[test]
    fn nothing_changes_without_the_images_entries() {
        assert!(run(&[("PATH", "/usr/bin:/bin"), ("LD_LIBRARY_PATH", "/opt/cuda/lib64")]).is_empty());
    }
}
