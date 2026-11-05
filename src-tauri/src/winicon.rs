//! Windows: the window's icons — title bar, Alt-Tab, taskbar — from the icon
//! the executable carries, at the sizes the window's DPI asks for.
//!
//! Tauri gives the window one bitmap, the first frame of icon.ico, and Windows
//! stretches it to every place it is shown: a 16 px frame first in the file
//! made the taskbar's icon a blur. Loaded from the executable's resource
//! instead, each icon is the frame drawn for its size, and the shell can go
//! back to the resource for the size it draws at (24 px on a 100 % taskbar).

/// The id tauri-build gives icon.ico among the executable's resources.
#[cfg(windows)]
const ICON_RESOURCE: u16 = 32512;

/// Set the window's big and small icons from the executable's icon resource,
/// sized for the DPI the window is on now. Again after a move to a screen of
/// another scale. Leaves the window as it was if anything is missing.
#[cfg(windows)]
pub fn apply(hwnd: windows::Win32::Foundation::HWND) {
    use std::sync::Mutex;
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{HINSTANCE, LPARAM, WPARAM};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::HiDpi::{GetDpiForWindow, GetSystemMetricsForDpi};
    use windows::Win32::UI::WindowsAndMessaging::{
        DestroyIcon, LoadImageW, SendMessageW, HICON, ICON_BIG, ICON_SMALL, IMAGE_ICON, LR_DEFAULTCOLOR,
        SM_CXICON, SM_CXSMICON, WM_SETICON,
    };

    // The icons this set last; destroyed once the window holds the new ones.
    static OURS: Mutex<Vec<isize>> = Mutex::new(Vec::new());

    unsafe {
        let Ok(module) = GetModuleHandleW(PCWSTR::null()) else { return };
        let dpi = match GetDpiForWindow(hwnd) {
            0 => 96,
            d => d,
        };
        let mut made = Vec::new();
        for (which, metric) in [(ICON_BIG, SM_CXICON), (ICON_SMALL, SM_CXSMICON)] {
            let size = GetSystemMetricsForDpi(metric, dpi);
            if size <= 0 {
                continue;
            }
            let Ok(icon) = LoadImageW(
                Some(HINSTANCE(module.0)),
                PCWSTR(ICON_RESOURCE as usize as *const u16),
                IMAGE_ICON,
                size,
                size,
                LR_DEFAULTCOLOR,
            ) else {
                continue;
            };
            SendMessageW(hwnd, WM_SETICON, Some(WPARAM(which as usize)), Some(LPARAM(icon.0 as isize)));
            made.push(icon.0 as isize);
        }
        if made.is_empty() {
            return;
        }
        if let Ok(mut ours) = OURS.lock() {
            for old in std::mem::replace(&mut *ours, made) {
                let _ = DestroyIcon(HICON(old as *mut core::ffi::c_void));
            }
        }
    }
}
