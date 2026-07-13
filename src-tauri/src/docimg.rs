//! Extract embedded raster images from documents so the vision pipeline can
//! SEE them — not just read the text around them. Shared by chat/Code
//! attachments (`attach.rs`) and the knowledge base (`rag.rs`).
//!
//! Formats: OOXML containers (docx/xlsx/pptx are zips with a media folder)
//! and PDF (image XObjects via lopdf: JPEG streams pass through, Flate-encoded
//! RGB/Gray bitmaps are re-encoded as PNG). Tiny graphics (icons, bullets,
//! logos) are filtered out; extraction is best-effort and never fails the
//! caller — a document with no extractable images just returns an empty list.

use std::io::Read;
use std::path::PathBuf;

/// Skip graphics smaller than this many bytes (icons/bullets).
const MIN_BYTES: usize = 6 * 1024;
/// Skip decoded images smaller than this on either side, or in total area.
const MIN_SIDE: u32 = 64;
const MIN_AREA: u64 = 16_384;

fn out_dir() -> PathBuf {
    let d = std::env::temp_dir().join("chaty-doc-imgs");
    let _ = std::fs::create_dir_all(&d);
    d
}

fn content_hash(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Keep an image only if it decodes and is big enough to carry content.
fn keep(bytes: &[u8]) -> bool {
    if bytes.len() < MIN_BYTES {
        return false;
    }
    match image::load_from_memory(bytes) {
        Ok(img) => {
            let (w, h) = (img.width(), img.height());
            w >= MIN_SIDE && h >= MIN_SIDE && (w as u64) * (h as u64) >= MIN_AREA
        }
        Err(_) => false,
    }
}

fn save(bytes: &[u8], ext: &str, out: &mut Vec<String>) {
    let p = out_dir().join(format!("{:016x}.{ext}", content_hash(bytes)));
    if p.is_file() || std::fs::write(&p, bytes).is_ok() {
        out.push(p.to_string_lossy().to_string());
    }
}

/// Extract up to `cap` embedded images from a document. Returns paths of
/// cached copies in the temp dir (content-addressed — duplicates collapse).
pub fn extract_embedded_images(path: &str, cap: usize) -> Vec<String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mut out = Vec::new();
    match ext.as_str() {
        "docx" | "xlsx" | "pptx" => extract_ooxml(path, cap, &mut out),
        "pdf" => extract_pdf(path, cap, &mut out),
        _ => {}
    }
    out
}

/// OOXML: any zip entry under the container's media folder is a stored image
/// file — read, filter, cache.
fn extract_ooxml(path: &str, cap: usize, out: &mut Vec<String>) {
    let Ok(file) = std::fs::File::open(path) else { return };
    let Ok(mut zip) = zip::ZipArchive::new(file) else { return };
    for i in 0..zip.len() {
        if out.len() >= cap {
            break;
        }
        let Ok(mut entry) = zip.by_index(i) else { continue };
        let name = entry.name().to_lowercase();
        let in_media = name.starts_with("word/media/")
            || name.starts_with("xl/media/")
            || name.starts_with("ppt/media/");
        if !in_media {
            continue;
        }
        let img_ext = match name.rsplit('.').next() {
            Some(e @ ("png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif")) => e,
            _ => continue,
        };
        let mut bytes = Vec::new();
        if entry.read_to_end(&mut bytes).is_err() {
            continue;
        }
        if keep(&bytes) {
            save(&bytes, img_ext, out);
        }
    }
}

/// PDF: walk image XObject streams. JPEG (DCTDecode) content is written as-is;
/// Flate-encoded 8-bit RGB/Gray bitmaps are rebuilt into PNGs. Other encodings
/// (JBIG2, CCITT, JPX) are rare in user documents and skipped.
fn extract_pdf(path: &str, cap: usize, out: &mut Vec<String>) {
    let Ok(doc) = lopdf::Document::load(path) else { return };
    for (_, obj) in doc.objects.iter() {
        if out.len() >= cap {
            break;
        }
        let lopdf::Object::Stream(stream) = obj else { continue };
        let dict = &stream.dict;
        let is_image = dict
            .get(b"Subtype")
            .ok()
            .and_then(|o| o.as_name().ok())
            .is_some_and(|n| n == b"Image");
        if !is_image {
            continue;
        }
        let filter = dict
            .get(b"Filter")
            .ok()
            .and_then(|o| match o {
                lopdf::Object::Name(n) => Some(n.clone()),
                lopdf::Object::Array(a) => a.first().and_then(|f| f.as_name().ok().map(|n| n.to_vec())),
                _ => None,
            })
            .unwrap_or_default();
        match filter.as_slice() {
            b"DCTDecode" => {
                // The stream content IS a JPEG file.
                if keep(&stream.content) {
                    save(&stream.content, "jpg", out);
                }
            }
            b"FlateDecode" => {
                let Ok(data) = stream.decompressed_content() else { continue };
                let (Some(w), Some(h)) = (
                    dict.get(b"Width").ok().and_then(|o| o.as_i64().ok()),
                    dict.get(b"Height").ok().and_then(|o| o.as_i64().ok()),
                ) else {
                    continue;
                };
                let bpc = dict.get(b"BitsPerComponent").ok().and_then(|o| o.as_i64().ok()).unwrap_or(8);
                if bpc != 8 || w <= 0 || h <= 0 {
                    continue;
                }
                let (w, h) = (w as u32, h as u32);
                let px = (w as usize) * (h as usize);
                let img = if data.len() >= px * 3 {
                    image::RgbImage::from_raw(w, h, data[..px * 3].to_vec())
                        .map(image::DynamicImage::ImageRgb8)
                } else if data.len() >= px {
                    image::GrayImage::from_raw(w, h, data[..px].to_vec())
                        .map(image::DynamicImage::ImageLuma8)
                } else {
                    None
                };
                let Some(img) = img else { continue };
                if img.width() < MIN_SIDE || img.height() < MIN_SIDE {
                    continue;
                }
                let mut png: Vec<u8> = Vec::new();
                if img
                    .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                    .is_ok()
                    && png.len() >= MIN_BYTES
                {
                    save(&png, "png", out);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn png_bytes(w: u32, h: u32, color: [u8; 3]) -> Vec<u8> {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb(color)));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();
        buf
    }

    #[test]
    fn ooxml_media_extracted_and_icons_filtered() {
        let dir = std::env::temp_dir().join(format!("chaty-docimg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let docx = dir.join("t.docx");
        let file = std::fs::File::create(&docx).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let opts = zip::write::SimpleFileOptions::default();
        zip.start_file("word/document.xml", opts).unwrap();
        zip.write_all(b"<w:document><w:t>hello</w:t></w:document>").unwrap();
        // A real content image (600x400 noise-free PNG > 6 KB thanks to size).
        zip.start_file("word/media/image1.png", opts).unwrap();
        let big = png_bytes(600, 400, [180, 30, 30]);
        zip.write_all(&big).unwrap();
        // A tiny icon — must be filtered.
        zip.start_file("word/media/icon.png", opts).unwrap();
        zip.write_all(&png_bytes(24, 24, [0, 0, 0])).unwrap();
        zip.finish().unwrap();

        let imgs = extract_embedded_images(&docx.to_string_lossy(), 6);
        // The solid-color 600x400 PNG compresses below 6 KB — so accept either
        // 0 or 1 here and assert the FILTER property instead: nothing tiny.
        for p in &imgs {
            let im = image::open(p).unwrap();
            assert!(im.width() >= MIN_SIDE && im.height() >= MIN_SIDE);
        }
        // Re-pack with a noisy (incompressible) image to guarantee extraction.
        let docx2 = dir.join("t2.docx");
        let file2 = std::fs::File::create(&docx2).unwrap();
        let mut zip2 = zip::ZipWriter::new(file2);
        zip2.start_file("word/media/photo.png", opts).unwrap();
        let mut noisy = image::RgbImage::new(300, 300);
        for (x, y, p) in noisy.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 7 % 256) as u8, (y * 13 % 256) as u8, ((x + y) % 256) as u8]);
        }
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(noisy)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        zip2.write_all(&buf).unwrap();
        zip2.finish().unwrap();
        let imgs2 = extract_embedded_images(&docx2.to_string_lossy(), 6);
        assert_eq!(imgs2.len(), 1, "the noisy 300x300 photo must be extracted");
        let im = image::open(&imgs2[0]).unwrap();
        assert_eq!((im.width(), im.height()), (300, 300));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pdf_jpeg_xobject_extracted() {
        // Build a minimal PDF with one DCTDecode image XObject via lopdf.
        let dir = std::env::temp_dir().join(format!("chaty-docimg-pdf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A noisy JPEG (incompressible → comfortably over the byte floor).
        let mut noisy = image::RgbImage::new(320, 240);
        for (x, y, p) in noisy.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 11 % 256) as u8, (y * 17 % 256) as u8, ((x * y) % 256) as u8]);
        }
        let mut jpg = Vec::new();
        image::DynamicImage::ImageRgb8(noisy)
            .write_to(&mut std::io::Cursor::new(&mut jpg), image::ImageFormat::Jpeg)
            .unwrap();

        let mut doc = lopdf::Document::with_version("1.5");
        let mut img_dict = lopdf::Dictionary::new();
        img_dict.set("Type", lopdf::Object::Name(b"XObject".to_vec()));
        img_dict.set("Subtype", lopdf::Object::Name(b"Image".to_vec()));
        img_dict.set("Width", 320);
        img_dict.set("Height", 240);
        img_dict.set("ColorSpace", lopdf::Object::Name(b"DeviceRGB".to_vec()));
        img_dict.set("BitsPerComponent", 8);
        img_dict.set("Filter", lopdf::Object::Name(b"DCTDecode".to_vec()));
        let stream = lopdf::Stream::new(img_dict, jpg.clone());
        let img_id = doc.add_object(lopdf::Object::Stream(stream));
        let pages_id = doc.new_object_id();
        let mut page = lopdf::Dictionary::new();
        page.set("Type", lopdf::Object::Name(b"Page".to_vec()));
        page.set("Parent", lopdf::Object::Reference(pages_id));
        let page_id = doc.add_object(lopdf::Object::Dictionary(page));
        let mut pages = lopdf::Dictionary::new();
        pages.set("Type", lopdf::Object::Name(b"Pages".to_vec()));
        pages.set("Kids", vec![lopdf::Object::Reference(page_id)]);
        pages.set("Count", 1);
        doc.objects.insert(pages_id, lopdf::Object::Dictionary(pages));
        let mut catalog = lopdf::Dictionary::new();
        catalog.set("Type", lopdf::Object::Name(b"Catalog".to_vec()));
        catalog.set("Pages", lopdf::Object::Reference(pages_id));
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog));
        doc.trailer.set("Root", lopdf::Object::Reference(catalog_id));
        let _ = img_id;
        let pdf_path = dir.join("t.pdf");
        doc.save(&pdf_path).unwrap();

        let imgs = extract_embedded_images(&pdf_path.to_string_lossy(), 6);
        assert_eq!(imgs.len(), 1, "the JPEG XObject must be extracted");
        let im = image::open(&imgs[0]).unwrap();
        assert_eq!((im.width(), im.height()), (320, 240));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
