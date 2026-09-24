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

/// A name the skill loader accepts, at most 32 characters: from ASCII, letters,
/// digits, `_` and `-`; beyond ASCII, anything but whitespace and control
/// characters. A whitelist of "letters of every script" is never complete —
/// Devanagari's virama is neither letter nor digit, and `हिन्दी` came out
/// `हिन-दी` — so outside ASCII the rule excludes rather than includes.
///
/// This used to keep ASCII only, which turned a file called `地理学家.md` into
/// nothing, then into the fallback `skill` — and six Chinese-named files
/// selected together were all written to `skill.md`, each over the last, so
/// one survived (issue #18).
fn clean_name(raw: &str) -> String {
    // Lower-cased in every script that has case, not just ASCII.
    let mapped: String = raw
        .trim()
        .chars()
        .flat_map(|c| {
            let keep = c.is_ascii_alphanumeric()
                || c == '_'
                || c == '-'
                || (!c.is_ascii() && !c.is_whitespace() && !c.is_control());
            let lower: Vec<char> = if keep { c.to_lowercase().collect() } else { vec!['-'] };
            lower
        })
        .collect();
    let mut s = mapped;
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    let s: String = s.trim_matches('-').chars().take(32).collect();
    let s = s.trim_matches('-').to_string();
    // A file Windows will not create under any extension.
    const RESERVED: &[&str] = &[
        "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
        "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
    ];
    if RESERVED.contains(&s.as_str()) {
        format!("{s}-skill")
    } else {
        s
    }
}

/// Where a skill called `name` can be written without destroying another one.
/// The same file imported twice lands on itself; a DIFFERENT skill that
/// happens to share the name gets `name-2`, `name-3`… — importing used to
/// replace whatever was there without a word.
fn free_name(dir: &Path, name: &str, render: impl Fn(&str) -> String) -> String {
    // "Is this name free, or already holding exactly what we would write?"
    // Compared with the file as it would be written UNDER THAT NAME — the name
    // is part of the content, so a re-import of a file that was once given
    // `-2` has to be recognised at `-2`.
    let taken = |n: &str| -> Option<bool> {
        let p = dir.join(format!("{n}.md"));
        if !p.exists() {
            return None;
        }
        Some(std::fs::read_to_string(&p).map(|t| t == render(n)).unwrap_or(false))
    };
    match taken(name) {
        None | Some(true) => return name.to_string(),
        Some(false) => {}
    }
    for i in 2..1000 {
        let candidate = format!("{name}-{i}");
        match taken(&candidate) {
            None | Some(true) => return candidate,
            Some(false) => {}
        }
    }
    format!("{name}-{}", std::process::id())
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
    let when = field("when").map(|w| one_line(&w, 200));
    let render = |name: &str| {
        let mut out = format!("---\nname: {name}\ndescription: {description}\n");
        if let Some(when) = &when {
            out.push_str(&format!("when: {when}\n"));
        }
        out.push_str("---\n\n");
        out.push_str(body);
        out.push('\n');
        out
    };
    std::fs::create_dir_all(dir).map_err(|e| trf!("创建技能目录失败: {e}", "could not create the skills folder: {e}"))?;
    // The name decides the file, and the name is written inside the file too:
    // a suffix chosen for the path must be the one in the frontmatter.
    name = free_name(dir, &name, &render);
    let out = render(&name);
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

    /// Issue #18: six skills with Chinese file names, selected together, came
    /// out as ONE skill called `skill` — every non-ASCII character was dropped,
    /// each name fell back to the same word, and each file overwrote the last.
    #[test]
    fn skills_named_in_any_script_keep_their_names_and_do_not_overwrite_each_other() {
        let src_dir = temp("cjk-src");
        let dest = temp("cjk-dest");
        let names = ["地理学家", "历史学家", "人类学家", "提示词工程师", "心理学家", "叙事学家"];
        for n in names {
            let f = src_dir.join(format!("{n}.md"));
            std::fs::write(&f, format!("# {n}智能体人格\n\n你是一位{n}。\n")).unwrap();
            let s = import_into(&dest, &f).unwrap();
            assert_eq!(s.name, n, "a Chinese file name is a name");
        }
        let listed: Vec<String> = list_in(&dest).into_iter().map(|s| s.name).collect();
        assert_eq!(listed.len(), 6, "all six are there: {listed:?}");

        // Accented Latin, Cyrillic: letters, not separators.
        let f = src_dir.join("Résumé Писатель.md");
        std::fs::write(&f, "steps").unwrap();
        assert_eq!(import_into(&dest, &f).unwrap().name, "résumé-писатель");
        // Combining marks belong to the word they sit in.
        let f = src_dir.join("हिन्दी.md");
        std::fs::write(&f, "steps").unwrap();
        assert_eq!(import_into(&dest, &f).unwrap().name, "हिन्दी");
    }

    #[test]
    fn a_name_already_taken_by_another_skill_gets_its_own() {
        let a_dir = temp("dup-a");
        let b_dir = temp("dup-b");
        let dest = temp("dup-dest");
        let a = a_dir.join("persona.md");
        let b = b_dir.join("persona.md");
        std::fs::write(&a, "first persona").unwrap();
        std::fs::write(&b, "second persona").unwrap();
        assert_eq!(import_into(&dest, &a).unwrap().name, "persona");
        assert_eq!(import_into(&dest, &b).unwrap().name, "persona-2", "a different skill is not overwritten");
        // The same file again is the same skill, wherever it landed.
        assert_eq!(import_into(&dest, &a).unwrap().name, "persona");
        assert_eq!(import_into(&dest, &b).unwrap().name, "persona-2");
        assert_eq!(list_in(&dest).len(), 2);
        assert!(std::fs::read_to_string(dest.join("persona.md")).unwrap().contains("first persona"));
    }

    #[test]
    fn a_name_windows_cannot_create_is_changed() {
        let src_dir = temp("res-src");
        let dest = temp("res-dest");
        let f = src_dir.join("con.md");
        std::fs::write(&f, "steps").unwrap();
        assert_eq!(import_into(&dest, &f).unwrap().name, "con-skill");
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
