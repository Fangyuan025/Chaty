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
        let is_image = stream
            .dict
            .get(b"Subtype")
            .ok()
            .and_then(|o| o.as_name().ok())
            .is_some_and(|n| n == b"Image");
        if !is_image {
            continue;
        }
        if let Some((bytes, ext)) = pdf_image_file(&stream.dict, &stream.content) {
            if keep(&bytes) {
                save(&bytes, ext, out);
            }
        }
    }
}

/// Longest side a scanned page is handed to the vision model at. Scans come
/// at 150–300 dpi, 2500+ pixels tall; text stays legible well below that, and
/// a model that tiles images by resolution would spend thousands of tokens of
/// its context on a single full-size page.
const PAGE_MAX_SIDE: u32 = 1600;

/// The picture on each page of a PDF, in page order: the largest image each
/// page draws — on a scan, the page itself. Up to `cap` pages; pages with no
/// readable image are skipped, and oversized ones are scaled down.
pub fn pdf_page_images(path: &str, cap: usize) -> Vec<String> {
    let Ok(doc) = lopdf::Document::load(path) else { return Vec::new() };
    let mut out = Vec::new();
    for (_, page_id) in doc.get_pages().into_iter().take(cap) {
        let Ok(images) = doc.get_page_images(page_id) else { continue };
        let Some(page) = images.iter().max_by_key(|i| i.width * i.height) else { continue };
        let Some((bytes, _)) = pdf_image_file(page.origin_dict, page.content) else { continue };
        let Ok(img) = image::load_from_memory(&bytes) else { continue };
        if img.width() < MIN_SIDE || img.height() < MIN_SIDE {
            continue;
        }
        let img = if img.width().max(img.height()) > PAGE_MAX_SIDE {
            img.resize(PAGE_MAX_SIDE, PAGE_MAX_SIDE, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let mut jpg = Vec::new();
        if image::DynamicImage::ImageRgb8(img.to_rgb8())
            .write_to(&mut std::io::Cursor::new(&mut jpg), image::ImageFormat::Jpeg)
            .is_ok()
        {
            save(&jpg, "jpg", &mut out);
        }
    }
    out
}

/// One PDF image stream as an image file. JPEG (DCTDecode) content passes
/// through; Flate-encoded 8-bit RGB/Gray and 1-bit bitmaps (black-and-white
/// scans) are rebuilt as PNG. Other encodings (JBIG2, CCITT, JPX) → `None`.
fn pdf_image_file(dict: &lopdf::Dictionary, content: &[u8]) -> Option<(Vec<u8>, &'static str)> {
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
        // The stream content IS a JPEG file.
        b"DCTDecode" => Some((content.to_vec(), "jpg")),
        b"FlateDecode" => {
            // lopdf's decompressed_content() rejects perfectly ordinary
            // image streams (every Chrome/Chromium print-to-PDF lands
            // here with Error::Type) — inflate the raw bytes ourselves.
            // Predictor'd streams (PNG row filters) stay skipped:
            // reversing those is a different job than inflating.
            let has_predictor = dict
                .get(b"DecodeParms")
                .ok()
                .and_then(|o| o.as_dict().ok())
                .and_then(|d| d.get(b"Predictor").ok())
                .and_then(|p| p.as_i64().ok())
                .is_some_and(|p| p > 1);
            if has_predictor {
                return None;
            }
            let mut data = Vec::new();
            flate2::read::ZlibDecoder::new(content).read_to_end(&mut data).ok()?;
            let w = dict.get(b"Width").ok().and_then(|o| o.as_i64().ok())?;
            let h = dict.get(b"Height").ok().and_then(|o| o.as_i64().ok())?;
            let bpc = dict.get(b"BitsPerComponent").ok().and_then(|o| o.as_i64().ok()).unwrap_or(8);
            if w <= 0 || h <= 0 {
                return None;
            }
            let (w, h) = (w as u32, h as u32);
            let px = (w as usize) * (h as usize);
            let img = match bpc {
                8 if data.len() >= px * 3 => image::RgbImage::from_raw(w, h, data[..px * 3].to_vec())
                    .map(image::DynamicImage::ImageRgb8),
                8 if data.len() >= px => image::GrayImage::from_raw(w, h, data[..px].to_vec())
                    .map(image::DynamicImage::ImageLuma8),
                // One bit a pixel, rows padded to whole bytes, 0 = black.
                1 => {
                    let stride = (w as usize).div_ceil(8);
                    (data.len() >= stride * h as usize).then(|| {
                        image::DynamicImage::ImageLuma8(image::GrayImage::from_fn(w, h, |x, y| {
                            let byte = data[y as usize * stride + x as usize / 8];
                            image::Luma([if byte >> (7 - x % 8) & 1 == 1 { 255 } else { 0 }])
                        }))
                    })
                }
                _ => None,
            }?;
            if img.width() < MIN_SIDE || img.height() < MIN_SIDE {
                return None;
            }
            let mut png: Vec<u8> = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).ok()?;
            Some((png, "png"))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Point this at a real file to see what the image walk actually pulls out
    /// of it. Skipped unless CHATY_DOCIMG_PROBE names one — a scanned PDF is
    /// the interesting case (its pages ARE the images) and those belong to
    /// whoever is holding one, not to the repo.
    #[test]
    fn docimg_probe() {
        let Ok(path) = std::env::var("CHATY_DOCIMG_PROBE") else { return };
        let imgs = extract_embedded_images(&path, 6);
        println!("PROBE {} image(s) from {path}", imgs.len());
        for i in &imgs {
            let size = std::fs::metadata(i).map(|m| m.len()).unwrap_or(0);
            println!("PROBE   {i} ({size} bytes)");
        }
        let pages = pdf_page_images(&path, 500);
        println!("PROBE {} page image(s)", pages.len());
        for p in &pages {
            let dims = image::open(p).map(|i| format!("{}×{}", i.width(), i.height())).unwrap_or_default();
            println!("PROBE   {p} ({dims})");
        }
    }

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

    fn noisy_jpeg(w: u32, h: u32, seed: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([((x * 11 + seed) % 256) as u8, ((y * 17 + seed) % 256) as u8, ((x * y + seed) % 256) as u8])
        });
        let mut jpg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut jpg), image::ImageFormat::Jpeg)
            .unwrap();
        jpg
    }

    /// A PDF whose pages each draw one JPEG — the shape of a scan.
    fn scan_pdf(path: &std::path::Path, pages: &[(u32, u32)]) {
        use lopdf::{Dictionary, Object, Stream};
        let mut doc = lopdf::Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let mut kids = Vec::new();
        for (n, &(w, h)) in pages.iter().enumerate() {
            let mut img = Dictionary::new();
            img.set("Type", Object::Name(b"XObject".to_vec()));
            img.set("Subtype", Object::Name(b"Image".to_vec()));
            img.set("Width", w as i64);
            img.set("Height", h as i64);
            img.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
            img.set("BitsPerComponent", 8);
            img.set("Filter", Object::Name(b"DCTDecode".to_vec()));
            let img_id = doc.add_object(Object::Stream(Stream::new(img, noisy_jpeg(w, h, n as u32 * 40))));
            let mut xobj = Dictionary::new();
            xobj.set("Im0", Object::Reference(img_id));
            let mut res = Dictionary::new();
            res.set("XObject", Object::Dictionary(xobj));
            let mut page = Dictionary::new();
            page.set("Type", Object::Name(b"Page".to_vec()));
            page.set("Parent", Object::Reference(pages_id));
            page.set("Resources", Object::Dictionary(res));
            kids.push(Object::Reference(doc.add_object(Object::Dictionary(page))));
        }
        let mut root_pages = Dictionary::new();
        root_pages.set("Type", Object::Name(b"Pages".to_vec()));
        root_pages.set("Count", pages.len() as i64);
        root_pages.set("Kids", kids);
        doc.objects.insert(pages_id, Object::Dictionary(root_pages));
        let mut catalog = Dictionary::new();
        catalog.set("Type", Object::Name(b"Catalog".to_vec()));
        catalog.set("Pages", Object::Reference(pages_id));
        let catalog_id = doc.add_object(Object::Dictionary(catalog));
        doc.trailer.set("Root", Object::Reference(catalog_id));
        doc.save(path).unwrap();
    }

    // Issue #15: a scanned PDF is read page by page, in page order, and a
    // full-resolution scan is scaled down to what a vision model can take.
    #[test]
    fn scanned_pdf_pages_come_out_in_order_and_scaled() {
        let dir = std::env::temp_dir().join(format!("chaty-docimg-scan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pdf = dir.join("scan.pdf");
        scan_pdf(&pdf, &[(400, 300), (300, 400), (2480, 3508)]);
        let pages = pdf_page_images(&pdf.to_string_lossy(), 500);
        let dims: Vec<(u32, u32)> = pages
            .iter()
            .map(|p| image::open(p).map(|i| (i.width(), i.height())).unwrap())
            .collect();
        assert_eq!(dims[..2], [(400, 300), (300, 400)]);
        assert_eq!(dims[2].1, PAGE_MAX_SIDE, "a 3508-pixel page is scaled to fit");
        assert_eq!(pdf_page_images(&pdf.to_string_lossy(), 2).len(), 2, "the cap holds");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_bit_scans_decode() {
        // 72×72 at one bit a pixel: rows of alternating black and white bytes.
        let (w, h) = (72u32, 72u32);
        let stride = (w as usize).div_ceil(8);
        let raw: Vec<u8> = (0..stride * h as usize).map(|i| if i % 2 == 0 { 0x00 } else { 0xff }).collect();
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut z, &raw).unwrap();
        let mut dict = lopdf::Dictionary::new();
        dict.set("Width", w as i64);
        dict.set("Height", h as i64);
        dict.set("BitsPerComponent", 1);
        dict.set("Filter", lopdf::Object::Name(b"FlateDecode".to_vec()));
        let (png, ext) = pdf_image_file(&dict, &z.finish().unwrap()).expect("1-bit page");
        assert_eq!(ext, "png");
        let img = image::load_from_memory(&png).unwrap().to_luma8();
        assert_eq!((img.get_pixel(0, 0)[0], img.get_pixel(8, 0)[0]), (0, 255));
    }
}

