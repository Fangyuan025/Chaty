//! A zipped document (docx, pptx, xlsx) unpacks to whatever its parts say,
//! and a part can be built to unpack to gigabytes from a few kilobytes — a zip
//! bomb, attached to a chat or dropped into the knowledge base, took the app's
//! memory with it. Every part the app reads itself goes through a cap here;
//! a workbook, which calamine unpacks for itself, is measured before it is
//! opened.

use std::io::Read;

/// The most one part may unpack to. The body of a thousand-page Word file is
/// tens of megabytes.
pub const PART_MAX: u64 = 256 << 20;
/// The most a whole document may unpack to.
pub const TOTAL_MAX: u64 = 1 << 30;

/// Whether an error from here is the size cap — in either language the
/// message was written in.
pub fn is_too_big(e: &str) -> bool {
    e.contains("unpacks to more than") || e.contains("文档解压后超过")
}

fn too_big(max: u64) -> String {
    trf!(
        "文档解压后超过 {} MB，可能已损坏或是特意构造的文件",
        "the document unpacks to more than {} MB — damaged, or built to do that",
        max >> 20
    )
}

/// One part's bytes, refused past `max` without reading further than that.
pub fn read_part(part: impl Read, max: u64) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    part.take(max + 1).read_to_end(&mut buf).map_err(|e| e.to_string())?;
    if buf.len() as u64 > max {
        return Err(too_big(max));
    }
    Ok(buf)
}

/// One part as text, as `read_to_string` reads it: UTF-8 or an error.
pub fn read_part_string(part: impl Read, max: u64) -> Result<String, String> {
    String::from_utf8(read_part(part, max)?).map_err(|e| e.to_string())
}

/// Running total of what a document's parts unpacked to, refused past
/// `TOTAL_MAX`.
#[derive(Default)]
pub struct Budget(u64);

impl Budget {
    pub fn spend(&mut self, n: usize) -> Result<(), String> {
        self.0 += n as u64;
        if self.0 > TOTAL_MAX {
            return Err(too_big(TOTAL_MAX));
        }
        Ok(())
    }
}

/// Unpack every part once, counting and keeping nothing: a document past the
/// caps is refused before a reader that unpacks for itself opens it. One that
/// is not a zip at all is left to that reader to say so.
pub fn check_unpacked(path: &str) -> Result<(), String> {
    let Ok(file) = std::fs::File::open(path) else { return Ok(()) };
    let Ok(mut zip) = zip::ZipArchive::new(file) else { return Ok(()) };
    let mut total = 0u64;
    for i in 0..zip.len() {
        let Ok(part) = zip.by_index(i) else { continue };
        let n = std::io::copy(&mut part.take(PART_MAX + 1), &mut std::io::sink())
            .map_err(|e| e.to_string())?;
        if n > PART_MAX {
            return Err(too_big(PART_MAX));
        }
        total += n;
        if total > TOTAL_MAX {
            return Err(too_big(TOTAL_MAX));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A zip whose one part is `n` zero bytes — a few KB on disk.
    fn bomb(dir: &std::path::Path, name: &str, part: &str, n: usize) -> String {
        let path = dir.join(name);
        let mut z = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .large_file(n as u64 > u32::MAX as u64);
        z.start_file(part, opts).unwrap();
        let block = vec![0u8; 1 << 20];
        let mut left = n;
        while left > 0 {
            let k = left.min(block.len());
            z.write_all(&block[..k]).unwrap();
            left -= k;
        }
        z.finish().unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn a_part_is_read_up_to_its_cap_and_refused_past_it() {
        let ok = read_part(&[7u8; 100][..], 100).unwrap();
        assert_eq!(ok.len(), 100);
        let err = read_part(&[7u8; 101][..], 100).unwrap_err();
        assert!(is_too_big(&err), "{err}");
        assert!(read_part_string(&b"\xff\xfe"[..], 100).is_err());
    }

    #[test]
    fn a_budget_runs_out_past_the_total() {
        let mut b = Budget::default();
        b.spend((TOTAL_MAX - 1) as usize).unwrap();
        b.spend(1).unwrap();
        assert!(b.spend(1).is_err());
    }

    #[test]
    fn documents_that_unpack_past_the_caps_are_refused() {
        let dir = std::env::temp_dir().join(format!("chaty-zipguard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 300 MB of zeros, under a megabyte on disk: a Word body past the cap.
        let docx = bomb(&dir, "bomb.docx", "word/document.xml", 300 << 20);
        assert!(std::fs::metadata(&docx).unwrap().len() < 1 << 20);
        let err = crate::rag::extract_docx(&docx).unwrap_err();
        assert!(is_too_big(&err) && err.contains("256"), "{err}");
        let deck = bomb(&dir, "bomb.pptx", "ppt/slides/slide1.xml", 300 << 20);
        assert!(crate::rag::extract_pptx(&deck).unwrap_err().contains("256 MB"));
        let book = bomb(&dir, "bomb.xlsx", "xl/worksheets/sheet1.xml", 300 << 20);
        assert!(crate::rag::extract_xlsx(&book).unwrap_err().contains("256 MB"));
        // An image the size of a part is skipped, not kept.
        let media = bomb(&dir, "media.docx", "word/media/image1.png", 300 << 20);
        assert!(crate::docimg::extract_embedded_images(&media, 6).is_empty());
        // Not a zip: left to the reader that opens it.
        let txt = dir.join("plain.xlsx");
        std::fs::write(&txt, b"not a zip").unwrap();
        assert!(check_unpacked(&txt.to_string_lossy()).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real documents still read under the caps — prints only what kind each
    /// is and how much came out, never the text:
    ///   CHATY_REAL_DOCS="/a.docx,/b.xlsx" cargo test --lib real_documents -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_documents_still_read_under_the_caps() {
        let paths = std::env::var("CHATY_REAL_DOCS").expect("CHATY_REAL_DOCS");
        for path in paths.split(',').filter(|p| !p.is_empty()) {
            let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
            let t0 = std::time::Instant::now();
            let text = match ext.as_str() {
                "docx" => crate::rag::extract_docx(path),
                "pptx" => crate::rag::extract_pptx(path),
                "xlsx" => crate::rag::extract_xlsx(path),
                _ => continue,
            };
            let images = crate::docimg::extract_embedded_images(path, 6).len();
            let tail: String = path.chars().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
            match text {
                Ok(t) => println!("ok   {ext} chars={} images={images} {:?}  …{tail}", t.chars().count(), t0.elapsed()),
                Err(e) => panic!("{ext} …{tail}: {e}"),
            }
        }
    }
}
