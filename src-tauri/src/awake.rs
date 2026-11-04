//! Keep the machine from idling into sleep while long work runs — a model
//! downloading overnight, pictures, music, an agent's run (`src/lib/busy.ts`
//! decides when). Only idle sleep: closing the lid or choosing Sleep still
//! sleeps. Each platform's own way: an IOKit power assertion on macOS, the
//! thread execution state on Windows, the desktop portal's Inhibit on Linux
//! (which GNOME and KDE show, and let the user override).

/// Hold off idle sleep (`on`) or let it come again. Idempotent.
#[tauri::command]
pub async fn keep_awake(on: bool) {
    imp::set(on).await;
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::{c_char, c_void, CString};
    use std::sync::Mutex;

    type CFStringRef = *const c_void;
    const UTF8: u32 = 0x0800_0100;
    const LEVEL_ON: u32 = 255;

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(alloc: *const c_void, s: *const c_char, encoding: u32) -> CFStringRef;
        fn CFRelease(cf: *const c_void);
    }
    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOPMAssertionCreateWithName(kind: CFStringRef, level: u32, name: CFStringRef, id: *mut u32) -> i32;
        fn IOPMAssertionRelease(id: u32) -> i32;
    }

    static HELD: Mutex<Option<u32>> = Mutex::new(None);

    fn cfstr(s: &str) -> CFStringRef {
        let c = CString::new(s).unwrap_or_default();
        unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
    }

    pub async fn set(on: bool) {
        let Ok(mut held) = HELD.lock() else { return };
        match (on, *held) {
            (true, None) => unsafe {
                let kind = cfstr("PreventUserIdleSystemSleep");
                let name = cfstr("Chaty is downloading or generating");
                let mut id = 0u32;
                if IOPMAssertionCreateWithName(kind, LEVEL_ON, name, &mut id) == 0 {
                    *held = Some(id);
                }
                CFRelease(kind);
                CFRelease(name);
            },
            (false, Some(id)) => unsafe {
                IOPMAssertionRelease(id);
                *held = None;
            },
            _ => {}
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::sync::mpsc::{channel, Sender};
    use std::sync::{Mutex, OnceLock};
    use windows::Win32::System::Power::{SetThreadExecutionState, ES_CONTINUOUS, ES_SYSTEM_REQUIRED};

    /// The state belongs to the thread that set it, so one thread of its own
    /// sets and clears it.
    fn worker() -> &'static Mutex<Sender<bool>> {
        static TX: OnceLock<Mutex<Sender<bool>>> = OnceLock::new();
        TX.get_or_init(|| {
            let (tx, rx) = channel::<bool>();
            std::thread::spawn(move || {
                for on in rx {
                    let flags = if on { ES_CONTINUOUS | ES_SYSTEM_REQUIRED } else { ES_CONTINUOUS };
                    unsafe { SetThreadExecutionState(flags) };
                }
            });
            Mutex::new(tx)
        })
    }

    pub async fn set(on: bool) {
        if let Ok(tx) = worker().lock() {
            let _ = tx.send(on);
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use ashpd::desktop::inhibit::{InhibitFlags, InhibitOptions, InhibitProxy};
    use ashpd::desktop::Request;
    use tokio::sync::Mutex;

    static HELD: Mutex<Option<Request<()>>> = Mutex::const_new(None);

    pub async fn set(on: bool) {
        let mut held = HELD.lock().await;
        if on && held.is_none() {
            let request = async {
                let proxy = InhibitProxy::new().await.ok()?;
                let options = InhibitOptions::default().set_reason("Chaty is downloading or generating");
                proxy.inhibit(None, InhibitFlags::Suspend.into(), options).await.ok()
            };
            *held = request.await;
        } else if !on {
            if let Some(request) = held.take() {
                let _ = request.close().await;
            }
        }
    }
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
mod imp {
    pub async fn set(_on: bool) {}
}
