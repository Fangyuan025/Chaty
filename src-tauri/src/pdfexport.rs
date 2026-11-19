//! A chat saved as a PDF that looks as it does in the app.
//!
//! The page the export builds (App.tsx, `conversationPage`) is laid out and
//! cut into pages by the system's own web view — WebKit on macOS and Linux,
//! WebView2 on Windows — in a hidden window of the app's origin, so the
//! fonts, colours, code, formulas and diagrams are the app's own, and printed
//! to a file with the system's page margins.
//!
//! A printed page's margins are never painted, whatever the page's
//! background: a dark theme came out as a dark sheet on white paper. So the
//! page is printed with no background at all — its own left transparent, the
//! web view's base colour off — and the theme's colour is laid under each
//! whole page afterwards: one fill, edge to edge, with no seam where the
//! printed area meets the margin.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use tauri::webview::PageLoadEvent;
use tauri::{AppHandle, WebviewUrl, WebviewWindow, WebviewWindowBuilder};
use tokio::sync::oneshot;

/// The page's margins, all four sides.
const MARGIN_MM: f64 = 15.0;
const MM_PER_IN: f64 = 25.4;
const PT_PER_IN: f64 = 72.0;
/// What the page's script sets the title to once it is ready to print.
const READY: &str = "chaty-pdf-ready";

static BUSY: AtomicBool = AtomicBool::new(false);
static NEXT: AtomicU32 = AtomicU32::new(0);

/// A sheet of paper, in points.
#[derive(Clone, Copy)]
struct Paper {
    width: f64,
    height: f64,
}

impl Paper {
    fn new(letter: bool) -> Self {
        if letter {
            Paper { width: 612.0, height: 792.0 }
        } else {
            Paper { width: 595.28, height: 841.89 }
        }
    }
}

fn margin_pt() -> f64 {
    MARGIN_MM / MM_PER_IN * PT_PER_IN
}

/// Clears the busy mark however the export ends.
struct Busy;
impl Drop for Busy {
    fn drop(&mut self) {
        BUSY.store(false, Ordering::SeqCst);
    }
}

/// The hidden window goes however the export ends. `destroy`, not `close`:
/// a close is turned into a hide for every window (lib.rs).
struct Hidden(WebviewWindow);
impl Drop for Hidden {
    fn drop(&mut self) {
        let _ = self.0.destroy();
    }
}

/// A one-shot sender that a callback the webview may call more than once can
/// fire the first time.
type Once<T> = std::sync::Arc<Mutex<Option<oneshot::Sender<T>>>>;
fn once<T>() -> (Once<T>, oneshot::Receiver<T>) {
    let (tx, rx) = oneshot::channel();
    (std::sync::Arc::new(Mutex::new(Some(tx))), rx)
}
fn fire<T>(o: &Once<T>, v: T) {
    if let Some(tx) = o.lock().unwrap().take() {
        let _ = tx.send(v);
    }
}

fn failed(detail: impl std::fmt::Display) -> String {
    trf!("导出 PDF 失败:{detail}", "could not export the PDF: {detail}")
}

/// Print `html` (a page built for it) to a PDF at `path`, with `background`
/// (sRGB) under each whole page and `title` as the document's.
#[tauri::command]
pub async fn export_pdf(
    app: AppHandle,
    path: String,
    html: String,
    title: String,
    background: [u8; 3],
    letter: bool,
) -> Result<(), String> {
    if BUSY.swap(true, Ordering::SeqCst) {
        return Err(trf!("上一个 PDF 还在导出", "a PDF is being exported already"));
    }
    let _busy = Busy;
    let paper = Paper::new(letter);
    let label = format!("pdf-export-{}", NEXT.fetch_add(1, Ordering::SeqCst));
    let (loaded, loaded_rx) = once::<()>();
    let (ready, ready_rx) = once::<()>();
    // The page's CSS width at 96 px an inch, as the print lays it out.
    let css_width = (paper.width - 2.0 * margin_pt()) / PT_PER_IN * 96.0;
    let win = WebviewWindowBuilder::new(&app, &label, WebviewUrl::App("print.html".into()))
        .title("Chaty")
        .visible(false)
        .focused(false)
        .skip_taskbar(true)
        .decorations(false)
        .inner_size(css_width, 1100.0)
        .on_page_load(move |_, p| {
            if p.event() == PageLoadEvent::Finished {
                fire(&loaded, ());
            }
        })
        .on_document_title_changed(move |_, t| {
            if t == READY {
                fire(&ready, ());
            }
        })
        .build()
        .map_err(failed)?;
    let win = Hidden(win);
    tokio::time::timeout(Duration::from_secs(20), loaded_rx)
        .await
        .map_err(|_| failed(trf!("打印页没有打开", "the print page did not open")))?
        .map_err(failed)?;
    unpainted(&win.0)?;
    let script = format!(
        "document.open();document.write({});document.close();",
        serde_json::to_string(&html).map_err(failed)?
    );
    win.0.eval(&script).map_err(failed)?;
    // Fonts and pictures load, or the page gives up on them after 10 s.
    tokio::time::timeout(Duration::from_secs(40), ready_rx)
        .await
        .map_err(|_| failed(trf!("页面没有排好版", "the page was not laid out")))?
        .map_err(failed)?;

    let tmp = std::env::temp_dir().join(format!("{label}-{}.pdf", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let printed = print(&win.0, &tmp, paper).await;
    platform::release(&win.0);
    drop(win);
    printed?;
    let dest = PathBuf::from(&path);
    let tmp2 = tmp.clone();
    let done = tauri::async_runtime::spawn_blocking(move || paint(&tmp2, &dest, background, &title))
        .await
        .map_err(failed)?;
    let _ = std::fs::remove_file(&tmp);
    done
}

/// No background of the web view's own under the page: what is not painted
/// stays unpainted in the PDF, for the theme's colour to show through.
fn unpainted(win: &WebviewWindow) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    win.with_webview(|w| unsafe {
        use objc2::msg_send;
        use objc2::runtime::{AnyObject, Bool};
        use objc2_foundation::{NSNumber, NSString};
        let wk: *mut AnyObject = w.inner().cast();
        let no = NSNumber::new_bool(false);
        let key = NSString::from_str("drawsBackground");
        let _: () = msg_send![wk, setValue: &*no as *const NSNumber as *const AnyObject, forKey: &*key];
        // Elements' own backgrounds print (the page asks for them too).
        let config: *mut AnyObject = msg_send![wk, configuration];
        let prefs: *mut AnyObject = msg_send![config, preferences];
        let can: Bool = msg_send![prefs, respondsToSelector: objc2::sel!(setShouldPrintBackgrounds:)];
        if can.as_bool() {
            let _: () = msg_send![prefs, setShouldPrintBackgrounds: Bool::YES];
        }
    })
    .map_err(failed)?;
    #[cfg(windows)]
    win.with_webview(|w| unsafe {
        use webview2_com::Microsoft::Web::WebView2::Win32::{ICoreWebView2Controller2, COREWEBVIEW2_COLOR};
        use windows::core::Interface;
        if let Ok(c2) = w.controller().cast::<ICoreWebView2Controller2>() {
            let _ = c2.SetDefaultBackgroundColor(COREWEBVIEW2_COLOR { A: 0, R: 255, G: 255, B: 255 });
        }
    })
    .map_err(failed)?;
    #[cfg(target_os = "linux")]
    win.with_webview(|w| {
        use webkit2gtk::{SettingsExt, WebViewExt};
        let wv = w.inner();
        wv.set_background_color(&gtk::gdk::RGBA::new(1.0, 1.0, 1.0, 0.0));
        if let Some(s) = WebViewExt::settings(&wv) {
            s.set_print_backgrounds(true);
        }
    })
    .map_err(failed)?;
    Ok(())
}

/// Print the hidden window's page to `out`, on `paper` with the margins.
async fn print(win: &WebviewWindow, out: &Path, paper: Paper) -> Result<(), String> {
    let (done, done_rx) = once::<Result<(), String>>();
    platform::print(win, out.to_path_buf(), paper, done)?;
    tokio::time::timeout(Duration::from_secs(120), done_rx)
        .await
        .map_err(|_| failed(trf!("打印超时", "printing timed out")))?
        .map_err(failed)??;
    match std::fs::metadata(out) {
        Ok(m) if m.len() > 0 => Ok(()),
        _ => Err(failed(trf!("没有生成文件", "no file was written"))),
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{failed, fire, margin_pt, Once, Paper};
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, Bool};
    use objc2::{class, define_class, msg_send, sel, ClassType};
    use objc2_foundation::{NSObject, NSRect, NSSize, NSString};
    use std::path::PathBuf;
    use std::sync::Mutex;
    use tauri::WebviewWindow;

    /// Where to say the print in flight ended. AppKit calls the delegate on
    /// a thread of its own, not the main one.
    static DONE: Mutex<Option<Once<Result<(), String>>>> = Mutex::new(None);
    /// The delegate and the operation of the print, alive until it ends and
    /// let go on the main thread, where they were made.
    struct Kept(#[allow(dead_code)] Retained<Printed>, #[allow(dead_code)] Retained<AnyObject>);
    // SAFETY: only ever set and taken (so dropped) on the main thread; the
    // callback thread does not touch it.
    unsafe impl Send for Kept {}
    static KEPT: Mutex<Option<Kept>> = Mutex::new(None);

    /// Let go of the print, and close the page: WebKit's print machinery
    /// keeps the web view, and with it a web content process of some 70 MB,
    /// alive after the window is gone — one more with every export.
    pub fn release(win: &WebviewWindow) {
        let _ = win.with_webview(|w| unsafe {
            drop(KEPT.lock().unwrap().take());
            let wk: *mut AnyObject = w.inner().cast();
            let can: Bool = msg_send![wk, respondsToSelector: sel!(_close)];
            if can.as_bool() {
                let _: () = msg_send![wk, _close];
            }
        });
    }

    define_class!(
        // SAFETY: NSObject has no subclassing requirements; no Drop.
        #[unsafe(super(NSObject))]
        #[name = "ChatyPdfPrinted"]
        struct Printed;

        impl Printed {
            #[unsafe(method(printOperationDidRun:success:contextInfo:))]
            fn did_run(&self, _op: *mut AnyObject, success: Bool, _ctx: *mut std::ffi::c_void) {
                if let Some(done) = DONE.lock().unwrap().take() {
                    fire(&done, if success.as_bool() { Ok(()) } else { Err(failed(trf!("系统打印没有完成", "the system's print did not finish"))) });
                }
            }
        }
    );

    /// On the system's paper (System Settings → Printers & Scanners → default
    /// paper size), the frontend's guess where there is none.
    pub fn print(win: &WebviewWindow, out: PathBuf, fallback: Paper, done: Once<Result<(), String>>) -> Result<(), String> {
        win.with_webview(move |w| unsafe {
            let wk: *mut AnyObject = w.inner().cast();
            let window: *mut AnyObject = w.ns_window().cast();
            let shared: *mut AnyObject = msg_send![class!(NSPrintInfo), sharedPrintInfo];
            let Some(info): Option<Retained<AnyObject>> = msg_send![shared, copy] else {
                fire(&done, Err(failed(trf!("系统打印不可用", "the system's print is not available"))));
                return;
            };
            let size: NSSize = msg_send![&*info, paperSize];
            if !(size.width > 100.0 && size.height > 100.0) {
                let _: () = msg_send![&*info, setPaperSize: NSSize::new(fallback.width, fallback.height)];
            }
            let m = margin_pt();
            let _: () = msg_send![&*info, setTopMargin: m];
            let _: () = msg_send![&*info, setBottomMargin: m];
            let _: () = msg_send![&*info, setLeftMargin: m];
            let _: () = msg_send![&*info, setRightMargin: m];
            // NSPrintingPaginationMode: automatic down the page, fit across.
            let _: () = msg_send![&*info, setHorizontalPagination: 1isize];
            let _: () = msg_send![&*info, setVerticalPagination: 0isize];
            let _: () = msg_send![&*info, setHorizontallyCentered: Bool::NO];
            let _: () = msg_send![&*info, setVerticallyCentered: Bool::NO];
            let disposition = NSString::from_str("NSPrintSaveJob");
            let _: () = msg_send![&*info, setJobDisposition: &*disposition];
            let path = NSString::from_str(&out.to_string_lossy());
            let url: *mut AnyObject = msg_send![class!(NSURL), fileURLWithPath: &*path];
            let dict: *mut AnyObject = msg_send![&*info, dictionary];
            // NSPrintJobSavingURL — whose value is not its name.
            let key = NSString::from_str("NSJobSavingURL");
            let _: () = msg_send![dict, setObject: url, forKey: &*key];
            let op: *mut AnyObject = msg_send![wk, printOperationWithPrintInfo: &*info];
            let Some(op) = Retained::retain(op) else {
                fire(&done, Err(failed(trf!("系统打印不可用", "the system's print is not available"))));
                return;
            };
            let _: () = msg_send![&*op, setShowsPrintPanel: Bool::NO];
            let _: () = msg_send![&*op, setShowsProgressPanel: Bool::NO];
            // A web view's print view has no size of its own until given one,
            // and prints blank pages without.
            let view: *mut AnyObject = msg_send![&*op, view];
            if !view.is_null() {
                let bounds: NSRect = msg_send![wk, bounds];
                let _: () = msg_send![view, setFrame: bounds];
            }
            let delegate: Retained<Printed> = msg_send![Printed::class(), new];
            let target: *const Printed = &*delegate;
            let op_ptr: *const AnyObject = &*op;
            *DONE.lock().unwrap() = Some(done);
            *KEPT.lock().unwrap() = Some(Kept(delegate, op));
            // Modal for the hidden window: WebKit draws a page for print in
            // its web process, which a plain `runOperation` does not wait
            // for — it saves blank pages.
            let _: () = msg_send![
                op_ptr,
                runOperationModalForWindow: window,
                delegate: target,
                didRunSelector: sel!(printOperationDidRun:success:contextInfo:),
                contextInfo: std::ptr::null_mut::<std::ffi::c_void>()
            ];
        })
        .map_err(failed)
    }
}

#[cfg(windows)]
mod platform {
    use super::{failed, fire, margin_pt, Once, Paper, PT_PER_IN};
    use std::path::PathBuf;
    use tauri::WebviewWindow;
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2Environment6, ICoreWebView2_2, ICoreWebView2_7, COREWEBVIEW2_PRINT_ORIENTATION_PORTRAIT,
    };
    use webview2_com::PrintToPdfCompletedHandler;
    use windows::core::{Interface, HSTRING};

    pub fn release(_: &WebviewWindow) {}

    pub fn print(win: &WebviewWindow, out: PathBuf, paper: Paper, done: Once<Result<(), String>>) -> Result<(), String> {
        win.with_webview(move |w| unsafe {
            let run = || -> windows::core::Result<()> {
                let controller = w.controller();
                // A hidden window's web view is not drawn; printing lays the
                // page out again on its own, but a view the system thinks
                // nobody sees can be held back.
                let _ = controller.SetIsVisible(true);
                let core = controller.CoreWebView2()?;
                let env = core.cast::<ICoreWebView2_2>()?.Environment()?;
                let settings = env.cast::<ICoreWebView2Environment6>()?.CreatePrintSettings()?;
                let inch = |pt: f64| pt / PT_PER_IN;
                settings.SetOrientation(COREWEBVIEW2_PRINT_ORIENTATION_PORTRAIT)?;
                settings.SetPageWidth(inch(paper.width))?;
                settings.SetPageHeight(inch(paper.height))?;
                let m = inch(margin_pt());
                settings.SetMarginTop(m)?;
                settings.SetMarginBottom(m)?;
                settings.SetMarginLeft(m)?;
                settings.SetMarginRight(m)?;
                settings.SetScaleFactor(1.0)?;
                settings.SetShouldPrintBackgrounds(true)?;
                settings.SetShouldPrintHeaderAndFooter(false)?;
                settings.SetShouldPrintSelectionOnly(false)?;
                let done2 = done.clone();
                let handler = PrintToPdfCompletedHandler::create(Box::new(move |hr, ok| {
                    fire(
                        &done2,
                        match hr {
                            Err(e) => Err(failed(e)),
                            Ok(()) if !ok => Err(failed(trf!("系统打印没有完成", "the system's print did not finish"))),
                            Ok(()) => Ok(()),
                        },
                    );
                    Ok(())
                }));
                core.cast::<ICoreWebView2_7>()?.PrintToPdf(&HSTRING::from(out.to_string_lossy().as_ref()), &settings, &handler)
            };
            if let Err(e) = run() {
                fire(&done, Err(failed(e)));
            }
        })
        .map_err(failed)
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{failed, fire, Once, Paper, MARGIN_MM, MM_PER_IN, PT_PER_IN};
    use std::path::PathBuf;
    use tauri::WebviewWindow;
    use webkit2gtk::{PrintOperation, PrintOperationExt};

    pub fn release(_: &WebviewWindow) {}

    pub fn print(win: &WebviewWindow, out: PathBuf, paper: Paper, done: Once<Result<(), String>>) -> Result<(), String> {
        win.with_webview(move |w| {
            let op = PrintOperation::new(&w.inner());
            let settings = gtk::PrintSettings::new();
            // GTK's own "print to file" printer, writing a PDF where it is told.
            settings.set_printer("Print to File");
            settings.set(gtk::PRINT_SETTINGS_OUTPUT_FILE_FORMAT.as_str(), Some("pdf"));
            let uri = gtk::glib::filename_to_uri(&out, None).map(|u| u.to_string()).unwrap_or_default();
            settings.set(gtk::PRINT_SETTINGS_OUTPUT_URI.as_str(), Some(&uri));
            let mm = |pt: f64| pt / PT_PER_IN * MM_PER_IN;
            let size = gtk::PaperSize::new_custom("chaty", "Chaty", mm(paper.width), mm(paper.height), gtk::Unit::Mm);
            let setup = gtk::PageSetup::new();
            setup.set_orientation(gtk::PageOrientation::Portrait);
            setup.set_paper_size(&size);
            setup.set_top_margin(MARGIN_MM, gtk::Unit::Mm);
            setup.set_bottom_margin(MARGIN_MM, gtk::Unit::Mm);
            setup.set_left_margin(MARGIN_MM, gtk::Unit::Mm);
            setup.set_right_margin(MARGIN_MM, gtk::Unit::Mm);
            settings.set_paper_size(&size);
            op.set_print_settings(&settings);
            op.set_page_setup(&setup);
            let ok = done.clone();
            op.connect_finished(move |_| fire(&ok, Ok(())));
            let bad = done.clone();
            op.connect_failed(move |_, e| fire(&bad, Err(failed(e))));
            op.print();
        })
        .map_err(failed)
    }
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
mod platform {
    use super::{Once, Paper};
    use std::path::PathBuf;
    use tauri::WebviewWindow;
    pub fn release(_: &WebviewWindow) {}
    pub fn print(_: &WebviewWindow, _: PathBuf, _: Paper, _: Once<Result<(), String>>) -> Result<(), String> {
        Err(trf!("这个系统不支持导出 PDF", "PDF export is not supported on this system"))
    }
}

/// Lay `background` under each whole page of the PDF at `src`, title it, and
/// write it to `dest`.
fn paint(src: &Path, dest: &Path, background: [u8; 3], title: &str) -> Result<(), String> {
    use lopdf::{Dictionary, Object, Stream};
    let mut doc = lopdf::Document::load(src).map_err(failed)?;
    let [r, g, b] = background.map(|c| f64::from(c) / 255.0);
    // Wider than any page: the fill does not depend on where a page's boxes
    // sit. Saved and restored, so the page's own drawing starts as it did.
    let fill = format!("q {r:.4} {g:.4} {b:.4} rg -10000 -10000 30000 30000 re f Q\n").into_bytes();
    let under = doc.add_object(Stream::new(Dictionary::new(), fill));
    let pages: Vec<_> = doc.get_pages().into_values().collect();
    if pages.is_empty() {
        return Err(failed(trf!("没有页面", "no pages")));
    }
    for page in pages {
        let mut contents: Vec<Object> = vec![Object::Reference(under)];
        contents.extend(doc.get_page_contents(page).into_iter().map(Object::Reference));
        doc.get_dictionary_mut(page).map_err(failed)?.set("Contents", Object::Array(contents));
    }
    // The chat's title, not the print job's.
    let mut utf16 = vec![0xFE, 0xFF];
    for u in title.encode_utf16() {
        utf16.extend_from_slice(&u.to_be_bytes());
    }
    let mut info = Dictionary::new();
    info.set("Title", Object::String(utf16, lopdf::StringFormat::Hexadecimal));
    info.set("Creator", Object::string_literal("Chaty"));
    let info_id = doc.add_object(info);
    doc.trailer.set("Info", Object::Reference(info_id));
    let tmp = dest.with_extension("pdf.partial");
    doc.save(&tmp).map_err(failed)?;
    std::fs::rename(&tmp, dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        failed(e)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-page PDF with a red square on an unpainted page.
    fn sample() -> Vec<u8> {
        use lopdf::{dictionary, Object, Stream};
        let mut doc = lopdf::Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let content = doc.add_object(Stream::new(dictionary! {}, b"1 0 0 rg 100 100 50 50 re f\n".to_vec()));
        let page = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        });
        doc.objects.insert(pages_id, Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page.into()],
            "Count" => 1,
        }));
        let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog);
        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    #[test]
    fn the_background_goes_under_the_page_and_the_title_on_it() {
        let dir = std::env::temp_dir().join(format!("chaty-pdf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.pdf");
        let dest = dir.join("out.pdf");
        std::fs::write(&src, sample()).unwrap();
        paint(&src, &dest, [31, 30, 28], "对话 · chat").unwrap();
        let doc = lopdf::Document::load(&dest).unwrap();
        let page = *doc.get_pages().values().next().unwrap();
        let content = String::from_utf8(doc.get_page_content(page).unwrap()).unwrap();
        // The fill first, then what the page drew, untouched.
        let fill = content.find("0.1216 0.1176 0.1098 rg").expect("background fill");
        let square = content.find("1 0 0 rg").expect("page content kept");
        assert!(fill < square, "{content}");
        let info = doc.trailer.get(b"Info").unwrap().as_reference().unwrap();
        let title = doc.get_dictionary(info).unwrap().get(b"Title").unwrap().as_str().unwrap().to_vec();
        let units: Vec<u16> = title[2..].chunks(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        assert_eq!(String::from_utf16(&units).unwrap(), "对话 · chat");
        assert!(!dir.join("out.pdf.partial").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
