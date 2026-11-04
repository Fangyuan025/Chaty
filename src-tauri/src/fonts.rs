//! Color emoji fonts WebKitGTK cannot draw, kept away from it (issue #30).
//!
//! The Linux AppImage's WebKitGTK draws text with Skia, and Skia aborts on
//! Fedora's Noto Color Emoji, a COLRv1 font: an out-of-bounds assertion in
//! `colrv1_configure_skpaint` (skia issue 542888014) takes the web process
//! down and leaves a gray window. Before GTK starts, emoji fonts whose COLR
//! table is version 1 are found and left out through a fontconfig file of the
//! app's own that includes the system's; the AppImage's CBDT Noto Color Emoji,
//! a bitmap font Skia draws, takes their place. A `FONTCONFIG_FILE` the user
//! set is theirs, and a system without such a font is not touched.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The fontconfig file this process pointed `FONTCONFIG_FILE` at, if it did:
/// the programs it starts get the user's own (`hostenv`).
static OURS: OnceLock<PathBuf> = OnceLock::new();

pub fn ours() -> Option<&'static Path> {
    OURS.get().map(PathBuf::as_path)
}

/// Leave COLRv1 emoji fonts out of this process's fontconfig, if the system
/// has any. Called before GTK, and before any other thread, is started.
#[cfg(target_os = "linux")]
pub fn guard_colrv1_emoji() {
    if std::env::var_os("FONTCONFIG_FILE").is_some() {
        return;
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(".local/share")));
    let mut dirs = vec![PathBuf::from("/usr/share/fonts"), PathBuf::from("/usr/local/share/fonts")];
    dirs.extend(data_home.map(|d| d.join("fonts")));
    dirs.extend(home.as_ref().map(|h| h.join(".fonts")));
    let rejected: Vec<PathBuf> = dirs.iter().flat_map(|d| colrv1_emoji_fonts(d)).collect();
    if rejected.is_empty() {
        return;
    }
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.parent()?.join("share/chaty/fonts")))
        .filter(|d| d.is_dir());
    let Some(cache) = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| home.map(|h| h.join(".cache")))
    else {
        return;
    };
    let file = cache.join("com.chaty.desktop").join("fontconfig.conf");
    let written = std::fs::create_dir_all(file.parent().unwrap_or(&cache))
        .and_then(|_| std::fs::write(&file, fontconfig_file(&rejected, bundled.as_deref())));
    if let Err(e) = written {
        eprintln!("[fonts] could not write {}: {e}", file.display());
        return;
    }
    eprintln!(
        "[fonts] left out {} — COLRv1, which WebKitGTK's Skia cannot draw",
        rejected.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    );
    std::env::set_var("FONTCONFIG_FILE", &file);
    let _ = OURS.set(file);
}

/// Emoji fonts under `dir` whose COLR table is version 1 or later.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn colrv1_emoji_fonts(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .max_depth(4)
        .into_iter()
        .flatten()
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_lowercase();
            (n.contains("emoji") || n.contains("colr"))
                && [".ttf", ".otf", ".ttc"].iter().any(|x| n.ends_with(x))
        })
        .filter(|e| std::fs::File::open(e.path()).map(|mut f| colr_version(&mut f) >= Some(1)).unwrap_or(false))
        .map(|e| e.into_path())
        .collect()
}

/// The version of a font's COLR table — its first face's, in a collection —
/// or `None` when it has none. Reads the table directory and two bytes.
fn colr_version(f: &mut (impl Read + Seek)) -> Option<u16> {
    let mut head = [0u8; 16];
    f.read_exact(&mut head).ok()?;
    let face = if &head[..4] == b"ttcf" {
        u32::from_be_bytes(head[12..16].try_into().ok()?) as u64
    } else {
        0
    };
    f.seek(SeekFrom::Start(face + 4)).ok()?;
    let mut n = [0u8; 2];
    f.read_exact(&mut n).ok()?;
    let tables = u16::from_be_bytes(n) as usize;
    f.seek(SeekFrom::Start(face + 12)).ok()?;
    let mut dir = vec![0u8; tables.min(256) * 16];
    f.read_exact(&mut dir).ok()?;
    let rec = dir.chunks_exact(16).find(|r| &r[..4] == b"COLR")?;
    let at = u32::from_be_bytes(rec[8..12].try_into().ok()?) as u64;
    f.seek(SeekFrom::Start(at)).ok()?;
    f.read_exact(&mut n).ok()?;
    Some(u16::from_be_bytes(n))
}

/// The system's fontconfig, with `rejected` left out and `bundled` added.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn fontconfig_file(rejected: &[PathBuf], bundled: Option<&Path>) -> String {
    let esc = |p: &Path| {
        p.display().to_string().replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
    };
    let mut out = String::from(
        "<?xml version=\"1.0\"?>\n<!DOCTYPE fontconfig SYSTEM \"urn:fontconfig:fonts.dtd\">\n\
         <!-- Written by Chaty: COLRv1 emoji fonts crash WebKitGTK's Skia (issue #30). -->\n\
         <fontconfig>\n  <include ignore_missing=\"yes\">/etc/fonts/fonts.conf</include>\n",
    );
    if let Some(d) = bundled {
        out += &format!("  <dir>{}</dir>\n", esc(d));
    }
    out += "  <selectfont>\n    <rejectfont>\n";
    for p in rejected {
        out += &format!("      <glob>{}</glob>\n", esc(p));
    }
    out += "    </rejectfont>\n  </selectfont>\n</fontconfig>\n";
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A font's bytes: an sfnt table directory with the given tables, each
    /// table's data a version number.
    fn font(tables: &[(&[u8; 4], u16)]) -> Vec<u8> {
        let mut out = vec![0, 1, 0, 0];
        out.extend((tables.len() as u16).to_be_bytes());
        out.extend([0u8; 6]);
        let data_at = 12 + tables.len() * 16;
        for (i, (tag, _)) in tables.iter().enumerate() {
            out.extend(tag.iter());
            out.extend([0u8; 4]);
            out.extend(((data_at + i * 4) as u32).to_be_bytes());
            out.extend(4u32.to_be_bytes());
        }
        for (_, version) in tables {
            out.extend(version.to_be_bytes());
            out.extend([0u8; 2]);
        }
        out
    }

    #[test]
    fn the_colr_version_is_read_from_the_table_directory() {
        assert_eq!(colr_version(&mut Cursor::new(font(&[(b"cmap", 0), (b"COLR", 1)]))), Some(1));
        assert_eq!(colr_version(&mut Cursor::new(font(&[(b"COLR", 0), (b"CPAL", 0)]))), Some(0));
        // CBDT, a bitmap emoji font Skia draws: no COLR at all.
        assert_eq!(colr_version(&mut Cursor::new(font(&[(b"CBDT", 0), (b"CBLC", 0)]))), None);
        assert_eq!(colr_version(&mut Cursor::new(b"not a font".to_vec())), None);
        // A collection: its first face is read.
        let face = font(&[(b"COLR", 1)]);
        let mut ttc = b"ttcf".to_vec();
        ttc.extend([0, 1, 0, 0, 0, 0, 0, 1]);
        ttc.extend(16u32.to_be_bytes());
        let mut shifted = face.clone();
        // Table offsets are from the file's start: move them past the header
        // (the first record's offset field is 8 bytes into it).
        let at = 12 + 8;
        let off = u32::from_be_bytes(shifted[at..at + 4].try_into().unwrap()) + 16;
        shifted[at..at + 4].copy_from_slice(&off.to_be_bytes());
        ttc.extend(shifted);
        assert_eq!(colr_version(&mut Cursor::new(ttc)), Some(1));
    }

    #[test]
    fn only_colrv1_emoji_fonts_are_found() {
        let dir = std::env::temp_dir().join(format!("chaty-fonts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("google-noto-color-emoji-fonts")).unwrap();
        std::fs::write(dir.join("google-noto-color-emoji-fonts/Noto-COLRv1.ttf"), font(&[(b"COLR", 1)])).unwrap();
        std::fs::write(dir.join("NotoColorEmoji.ttf"), font(&[(b"CBDT", 0)])).unwrap();
        std::fs::write(dir.join("Twemoji.ttf"), font(&[(b"COLR", 0)])).unwrap();
        // A COLRv1 font that is no emoji font is left to WebKit's choice.
        std::fs::write(dir.join("Nabla.ttf"), font(&[(b"COLR", 1)])).unwrap();
        let found = colrv1_emoji_fonts(&dir);
        assert_eq!(found, vec![dir.join("google-noto-color-emoji-fonts/Noto-COLRv1.ttf")]);
        let conf = fontconfig_file(&found, Some(Path::new("/tmp/.mount_x/usr/share/chaty/fonts")));
        assert!(conf.contains("<include ignore_missing=\"yes\">/etc/fonts/fonts.conf</include>"));
        assert!(conf.contains("<dir>/tmp/.mount_x/usr/share/chaty/fonts</dir>"));
        assert!(conf.contains(&format!("<glob>{}</glob>", found[0].display())));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
