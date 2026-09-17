//! Skills the user brings in from outside Chaty (issue #18). A markdown file
//! picked in Settings → Code is copied into `~/.chaty/skills/`, the folder
//! every Code turn already reads global skills from — so an imported skill is
//! an ordinary global skill from then on: listed in the prompt by one line,
//! loaded by `use_skill`, shadowed by a project skill of the same name.
//!
//! Skills written for other tools often arrive without Chaty's frontmatter,
//! or with a name the loader rejects (spaces, capitals, punctuation). Import
//! gives every file a usable one rather than refusing it: the name falls back
//! to the file name (or its folder, for a `SKILL.md`), the description to the
//! body's first line.

use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UserSkill {
    pub name: String,
    pub description: String,
    pub path: String,
}

/// A skill is procedure, not a document store; the loader keeps 8000 chars.
const MAX_BYTES: u64 = 256 * 1024;

fn skills_dir() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| PathBuf::from(h).join(".chaty").join("skills"))
        .ok_or_else(|| trf!("找不到用户主目录", "no home directory"))
}

/// Frontmatter fields (keys lower-cased) and the body, or None when the file
/// opens without a `---` block.
fn split_frontmatter(text: &str) -> Option<(Vec<(String, String)>, String)> {
    let t = text.trim_start_matches('\u{feff}');
    let rest = t.strip_prefix("---\n").or_else(|| t.strip_prefix("---\r\n"))?;
    let end = rest.find("\n---")?;
    let head = &rest[..end];
    let after = &rest[end + 4..];
    let body = after.split_once('\n').map(|(_, b)| b).unwrap_or("");
    let fields = head
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(':')?;
            Some((k.trim().to_lowercase(), v.trim().trim_matches(|c| c == '"' || c == '\'').to_string()))
        })
        .collect();
    Some((fields, body.to_string()))
}

/// A name the skill loader accepts: `[a-z0-9_-]`, at most 32 characters.
fn clean_name(raw: &str) -> String {
    let mapped: String = raw
        .trim()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let mut s = mapped;
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    let s: String = s.trim_matches('-').chars().take(32).collect();
    s.trim_matches('-').to_string()
}

fn one_line(s: &str, max: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    flat.chars().take(max).collect()
}

/// The body's first line of prose — a heading's text counts — for a file that
/// brings no description of its own.
fn first_line(body: &str) -> String {
    body.lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .find(|l| !l.is_empty() && !l.starts_with("```"))
        .map(|l| one_line(l, 160))
        .unwrap_or_default()
}

fn import_into(dir: &Path, src: &Path) -> Result<UserSkill, String> {
    let meta = std::fs::metadata(src).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
    if meta.len() > MAX_BYTES {
        return Err(trf!("技能文件太大(超过 256 KB)", "skill file too large (over 256 KB)"));
    }
    let text = std::fs::read_to_string(src).map_err(|e| trf!("读取失败: {e}", "read failed: {e}"))?;
    let (fields, body) = split_frontmatter(&text).unwrap_or_else(|| (Vec::new(), text.clone()));
    let body = body.trim();
    if body.is_empty() {
        return Err(trf!("文件里没有技能正文", "the file has no skill body"));
    }
    let field = |k: &str| fields.iter().find(|(f, _)| f == k).map(|(_, v)| v.clone()).filter(|v| !v.trim().is_empty());
    // `pdf-tools/SKILL.md` is named by its folder; any other file by its stem.
    let from_path = || {
        let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem.eq_ignore_ascii_case("skill") || stem.eq_ignore_ascii_case("readme") {
            src.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("").to_string()
        } else {
            stem.to_string()
        }
    };
    let mut name = field("name").map(|n| clean_name(&n)).unwrap_or_default();
    if name.is_empty() {
        name = clean_name(&from_path());
    }
    if name.is_empty() {
        name = "skill".into();
    }
    let description = field("description").map(|d| one_line(&d, 200)).unwrap_or_else(|| first_line(body));
    let mut out = format!("---\nname: {name}\ndescription: {description}\n");
    if let Some(when) = field("when") {
        out.push_str(&format!("when: {}\n", one_line(&when, 200)));
    }
    out.push_str("---\n\n");
    out.push_str(body);
    out.push('\n');
    std::fs::create_dir_all(dir).map_err(|e| trf!("创建技能目录失败: {e}", "could not create the skills folder: {e}"))?;
    let dest = dir.join(format!("{name}.md"));
    std::fs::write(&dest, out).map_err(|e| trf!("写入失败: {e}", "write failed: {e}"))?;
    Ok(UserSkill { name, description, path: dest.display().to_string() })
}

fn list_in(dir: &Path) -> Vec<UserSkill> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<UserSkill> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .filter_map(|p| {
            let text = std::fs::read_to_string(&p).ok()?;
            let (fields, body) = split_frontmatter(&text)?;
            if body.trim().is_empty() {
                return None;
            }
            let get = |k: &str| fields.iter().find(|(f, _)| f == k).map(|(_, v)| v.clone());
            let name = get("name").filter(|n| !n.is_empty())?;
            Some(UserSkill { name, description: get("description").unwrap_or_default(), path: p.display().to_string() })
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn delete_in(dir: &Path, path: &str) -> Result<(), String> {
    let target = PathBuf::from(path);
    let inside = match (target.canonicalize(), dir.canonicalize()) {
        (Ok(t), Ok(d)) => t.parent() == Some(d.as_path()),
        _ => false,
    };
    if !inside {
        return Err(trf!("只能删除技能目录里的文件", "only files in the skills folder can be removed"));
    }
    std::fs::remove_file(&target).map_err(|e| trf!("删除失败: {e}", "remove failed: {e}"))
}

/// Global skills in `~/.chaty/skills/`, the ones imported here among them.
#[tauri::command]
pub fn skills_list_user() -> Vec<UserSkill> {
    skills_dir().map(|d| list_in(&d)).unwrap_or_default()
}

/// Copy a picked markdown file into the global skills folder.
#[tauri::command]
pub fn skills_import(path: String) -> Result<UserSkill, String> {
    import_into(&skills_dir()?, Path::new(&path))
}

/// Remove a global skill file.
#[tauri::command]
pub fn skills_delete_user(path: String) -> Result<(), String> {
    delete_in(&skills_dir()?, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("chaty-user-skills-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_skill_with_frontmatter_keeps_its_name_and_description() {
        let src_dir = temp("fm-src");
        let dest = temp("fm-dest");
        let src = src_dir.join("release.md");
        std::fs::write(&src, "---\nname: release\ndescription: Cut a release\nwhen: the user asks to ship\nlicense: MIT\n---\n1. Bump the version\n").unwrap();
        let s = import_into(&dest, &src).unwrap();
        assert_eq!((s.name.as_str(), s.description.as_str()), ("release", "Cut a release"));
        let written = std::fs::read_to_string(dest.join("release.md")).unwrap();
        assert!(written.starts_with("---\nname: release\ndescription: Cut a release\nwhen: the user asks to ship\n---\n\n1. Bump the version"), "{written}");
        assert_eq!(list_in(&dest), vec![s]);
    }

    #[test]
    fn a_skill_from_another_tool_gets_a_usable_name() {
        let src_dir = temp("other-src").join("PDF Tools");
        std::fs::create_dir_all(&src_dir).unwrap();
        let dest = temp("other-dest");
        // No frontmatter at all, named by its folder.
        let src = src_dir.join("SKILL.md");
        std::fs::write(&src, "# Work with PDF files\n\nUse pdftotext first.\n").unwrap();
        let s = import_into(&dest, &src).unwrap();
        assert_eq!(s.name, "pdf-tools");
        assert_eq!(s.description, "Work with PDF files");
        // A name the loader would reject is cleaned, not refused.
        let odd = src_dir.join("odd.md");
        std::fs::write(&odd, "---\nname: My Great Skill!\n---\nsteps\n").unwrap();
        assert_eq!(import_into(&dest, &odd).unwrap().name, "my-great-skill");
        assert_eq!(list_in(&dest).len(), 2);
    }

    #[test]
    fn an_empty_or_huge_file_is_refused_and_removal_stays_in_the_folder() {
        let dest = temp("refuse-dest");
        let src_dir = temp("refuse-src");
        let empty = src_dir.join("empty.md");
        std::fs::write(&empty, "---\nname: empty\n---\n\n").unwrap();
        assert!(import_into(&dest, &empty).is_err());
        let huge = src_dir.join("huge.md");
        std::fs::write(&huge, "x".repeat(300 * 1024)).unwrap();
        assert!(import_into(&dest, &huge).is_err());
        let ok = src_dir.join("ok.md");
        std::fs::write(&ok, "steps").unwrap();
        let s = import_into(&dest, &ok).unwrap();
        assert!(delete_in(&dest, ok.to_str().unwrap()).is_err(), "a file outside the folder is not removed");
        assert!(ok.exists());
        delete_in(&dest, &s.path).unwrap();
        assert!(list_in(&dest).is_empty());
    }
}
