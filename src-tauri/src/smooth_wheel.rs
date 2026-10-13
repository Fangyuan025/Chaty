//! A mouse wheel that glides, on macOS.
//!
//! WebKit moves a page the whole length of a wheel notch in one frame; the
//! Chromium apps beside Chaty (Chrome, Claude) ease it over a few, and next to
//! them the wheel here read as stepping. WebKit has no switch for it (its
//! scroll animator is on and leaves the wheel alone), and the page cannot tell
//! a wheel from a trackpad: both arrive as the same pixel deltas. AppKit can.
//!
//! So a local event monitor takes each wheel event that has no precise deltas
//! — a mouse wheel, not a trackpad or a Magic Mouse — before the web view sees
//! it, and hands the same distance back as precise scroll events, one per
//! screen refresh (a display link: a timer beside the refresh moved the page
//! one step in some frames and two in others). The curve is Chromium's for a
//! wheel (cc's ScrollOffsetAnimationCurve): ease-in-out, 200 ms for a notch or
//! two and down to 100 ms for a long throw, and a notch that comes while one
//! is still moving extends it from where the page is, at the speed it is
//! going, so a turning wheel reads as one movement instead of a pulse per
//! notch. The page stops when the curve does — no glide of its own (owner: no
//! inertia). WebKit scrolls the events as a trackpad's, on its scrolling
//! thread: a busy page (a reply streaming in) does not hold them up.
//! Trackpads, Magic Mice and a wheel with a modifier held go through untouched.

#![cfg(target_os = "macos")]

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{class, define_class, msg_send, sel, ClassType};
use objc2_foundation::NSObject;

type CGEventRef = *mut c_void;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventCreateCopy(event: CGEventRef) -> CGEventRef;
    fn CGEventSetIntegerValueField(event: CGEventRef, field: u32, value: i64);
    fn CGEventSetDoubleValueField(event: CGEventRef, field: u32, value: f64);
}
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *const c_void);
    static kCFRunLoopCommonModes: *const c_void;
}
#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    fn CACurrentMediaTime() -> f64;
}

// CGEventField numbers for scroll wheel events.
const DELTA_AXIS_1: u32 = 11;
const DELTA_AXIS_2: u32 = 12;
const IS_CONTINUOUS: u32 = 88;
const FIXED_PT_DELTA_AXIS_1: u32 = 93;
const FIXED_PT_DELTA_AXIS_2: u32 = 94;
const POINT_DELTA_AXIS_1: u32 = 96;
const POINT_DELTA_AXIS_2: u32 = 97;
const SCROLL_PHASE: u32 = 99;
const MOMENTUM_PHASE: u32 = 123;

/// What WebKit moves for one line of a wheel (`Scrollbar::pixelsPerLineStep`):
/// the glide covers exactly the distance the notch would have jumped.
const PIXELS_PER_LINE: f64 = 40.0;
/// A display link that has not ticked for this long while a glide waits on it
/// is not going to (the window went away): the glide is dropped, not left
/// swallowing the wheel.
const STALL: f64 = 0.25;

const NS_EVENT_MASK_SCROLL_WHEEL: u64 = 1 << 22;
// Shift turns the wheel horizontal; command and control zoom. Those stay native.
const MODIFIERS: usize = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20);

define_class!(
    // SAFETY: NSObject has no subclassing requirements; no Drop.
    #[unsafe(super(NSObject))]
    #[name = "ChatyWheelGlide"]
    struct Ticker;

    impl Ticker {
        #[unsafe(method(tick:))]
        fn tick(&self, driver: *mut AnyObject) {
            // SAFETY: called by the display link or timer on the main run loop.
            unsafe { step(driver) };
        }
    }
);

/// How long Chromium gives a wheel movement of `delta` pixels (kInverseDelta:
/// 12 frames at 60 fps up to 120 px, falling to 6 frames from 480 px on).
fn duration(delta: f64) -> f64 {
    let frames = (14.0 - delta.abs() / 60.0).clamp(6.0, 12.0);
    frames / 60.0
}

/// Chromium's bound on a retargeted leg: no longer than the page would take
/// to cover it at its present speed, with room for the easing out — a long
/// leg at a high speed would otherwise overshoot and come back.
fn bound(velocity: f64, delta: f64) -> f64 {
    if velocity.abs() < 0.01 {
        return f64::INFINITY;
    }
    let b = delta / velocity * 2.5;
    if b < 0.0 {
        f64::INFINITY
    } else {
        b
    }
}

/// cubic-bezier(0.42, slope·0.42, 0.58, 1) — ease-in-out, starting at the
/// given slope — at time fraction `x`: (progress, d progress / d x).
fn ease(slope: f64, x: f64) -> (f64, f64) {
    let (x1, y1, x2, y2) = (0.42, slope.clamp(-1000.0, 1000.0) * 0.42, 0.58, 1.0);
    let cx = 3.0 * x1;
    let bx = 3.0 * (x2 - x1) - cx;
    let ax = 1.0 - cx - bx;
    let cy = 3.0 * y1;
    let by = 3.0 * (y2 - y1) - cy;
    let ay = 1.0 - cy - by;
    let fx = |u: f64| ((ax * u + bx) * u + cx) * u;
    let dfx = |u: f64| (3.0 * ax * u + 2.0 * bx) * u + cx;
    // x(u) rises monotonically from 0 to 1; Newton, then bisection if it stalls.
    let mut u = x;
    let mut solved = false;
    for _ in 0..8 {
        let err = fx(u) - x;
        if err.abs() < 1e-7 {
            solved = true;
            break;
        }
        let d = dfx(u);
        if d.abs() < 1e-6 {
            break;
        }
        u -= err / d;
    }
    if !solved || !(0.0..=1.0).contains(&u) {
        let (mut lo, mut hi) = (0.0, 1.0);
        u = x;
        for _ in 0..40 {
            if fx(u) < x {
                lo = u;
            } else {
                hi = u;
            }
            u = (lo + hi) / 2.0;
        }
    }
    let y = ((ay * u + by) * u + cy) * u;
    let dy = (3.0 * ay * u + 2.0 * by) * u + cy;
    let dx = dfx(u);
    (y, if dx.abs() < 1e-9 { 0.0 } else { dy / dx })
}

struct Glide {
    /// Where this leg starts and how far it goes, in pixels from where the
    /// glide began (sign as the wheel's).
    from: f64,
    delta: f64,
    /// Its length in seconds, and the slope the easing starts at (the speed
    /// carried over from the leg before).
    dur: f64,
    slope: f64,
    /// When the leg starts; none until the first frame.
    start: Option<f64>,
    /// The time of the last frame sent.
    last_at: Option<f64>,
    /// Whole pixels sent so far.
    sent: i64,
    /// A copy of the last wheel event: where it happened and in which window.
    template: CGEventRef,
    /// A display link, or an NSTimer where there is none (before macOS 14).
    driver: *mut AnyObject,
    link: bool,
    /// When the driver last ticked (or the glide began), and how often it has.
    ticked: f64,
    ticks: u32,
}

impl Glide {
    /// Position (px) and speed (px/s) at time `at`.
    fn at(&self, at: f64) -> (f64, f64) {
        let Some(start) = self.start else { return (self.from, 0.0) };
        if self.dur <= 0.0 || at - start >= self.dur {
            return (self.from + self.delta, 0.0);
        }
        let x = ((at - start) / self.dur).max(0.0);
        let (y, dydx) = ease(self.slope, x);
        (self.from + self.delta * y.min(1.0), self.delta * dydx / self.dur)
    }

    /// Another notch while this one moves.
    fn add(&mut self, px: f64) {
        match self.last_at {
            // No frame yet: the first leg just gets longer.
            None => {
                self.delta = if self.delta * px > 0.0 { self.delta + px } else { px };
                self.dur = duration(self.delta);
            }
            // From where the page is, at the speed it is going — unless the
            // wheel turned back, which starts over from a standstill.
            Some(at) => {
                let (pos, speed) = self.at(at);
                let left = self.from + self.delta - pos;
                let (delta, speed) = if left * px >= 0.0 { (left + px, speed) } else { (px, 0.0) };
                let dur = duration(delta).min(bound(speed, delta));
                self.from = pos;
                self.delta = delta;
                self.dur = dur;
                self.slope = if dur > 0.0 && dur.is_finite() { speed * dur / delta } else { 0.0 };
                self.start = Some(at);
            }
        }
    }

    /// Stop the driver and let go of what the glide holds.
    unsafe fn end(self) {
        let _: () = msg_send![self.driver, invalidate];
        let _: () = msg_send![self.driver, release];
        CFRelease(self.template);
    }
}

thread_local! {
    static GLIDE: RefCell<Option<Glide>> = const { RefCell::new(None) };
    static TICKER: RefCell<Option<Retained<Ticker>>> = const { RefCell::new(None) };
    /// Display links in a row that never ticked; at two, timers from here on
    /// (one could be a main thread busy for the whole wait).
    static LINK_MISSES: Cell<u8> = const { Cell::new(0) };
}

/// Install the monitor. Main thread, once, after the window exists.
pub fn install() {
    if std::env::var_os("CHATY_NATIVE_WHEEL").is_some() {
        return;
    }
    // SAFETY: plain object creation on the main thread.
    let ticker: Retained<Ticker> = unsafe { msg_send![Ticker::class(), new] };
    TICKER.with(|t| *t.borrow_mut() = Some(ticker));
    let handler = RcBlock::new(|event: NonNull<AnyObject>| -> *mut AnyObject {
        let ev = event.as_ptr();
        // SAFETY: AppKit hands the monitor a live NSEvent on the main thread.
        unsafe {
            if take(ev) {
                std::ptr::null_mut()
            } else {
                ev
            }
        }
    });
    // SAFETY: a class method of NSEvent; the monitor is kept for the app's life.
    unsafe {
        let monitor: *mut AnyObject = msg_send![
            class!(NSEvent),
            addLocalMonitorForEventsMatchingMask: NS_EVENT_MASK_SCROLL_WHEEL,
            handler: &*handler
        ];
        if !monitor.is_null() {
            let _: *mut AnyObject = msg_send![monitor, retain];
        }
    }
    std::mem::forget(handler);
}

/// Whether this event is a wheel notch to glide (and swallow).
unsafe fn take(ev: *mut AnyObject) -> bool {
    let precise: bool = msg_send![ev, hasPreciseScrollingDeltas];
    if precise {
        return false;
    }
    let flags: usize = msg_send![ev, modifierFlags];
    if flags & MODIFIERS != 0 {
        return false;
    }
    let dy: f64 = msg_send![ev, scrollingDeltaY];
    let dx: f64 = msg_send![ev, scrollingDeltaX];
    if dy == 0.0 || dx.abs() > dy.abs() {
        return false;
    }
    let window: *mut AnyObject = msg_send![ev, window];
    let cg: CGEventRef = msg_send![ev, CGEvent];
    if cg.is_null() || window.is_null() {
        return false;
    }
    let template = CGEventCreateCopy(cg);
    if template.is_null() {
        return false;
    }
    let px = dy * PIXELS_PER_LINE;
    let now = CACurrentMediaTime();
    GLIDE.with(|g| {
        let mut g = g.borrow_mut();
        let mut owed = 0.0;
        if g.as_ref().is_some_and(|glide| now - glide.ticked > STALL) {
            let stalled = g.take().expect("a glide");
            if stalled.ticks == 0 {
                // Nothing of it reached the page: it comes along.
                owed = stalled.delta;
                if stalled.link {
                    LINK_MISSES.with(|n| n.set(n.get().saturating_add(1)));
                }
            }
            stalled.end();
        }
        if let Some(glide) = g.as_mut() {
            CFRelease(glide.template);
            glide.template = template;
            glide.add(px);
            return true;
        }
        let delta = owed + px;
        match start_driver(window) {
            Some((driver, link)) => {
                *g = Some(Glide {
                    from: 0.0,
                    delta,
                    dur: duration(delta),
                    slope: 0.0,
                    start: None,
                    last_at: None,
                    sent: 0,
                    template,
                    driver,
                    link,
                    ticked: now,
                    ticks: 0,
                });
                true
            }
            None => {
                CFRelease(template);
                false
            }
        }
    })
}

/// A display link on the wheel's window — or a 120 Hz timer, before macOS 14
/// or where a display link would not tick.
unsafe fn start_driver(window: *mut AnyObject) -> Option<(*mut AnyObject, bool)> {
    let ticker = TICKER.with(|t| t.borrow().clone())?;
    let run_loop: *mut AnyObject = msg_send![class!(NSRunLoop), mainRunLoop];
    // The constant itself: CoreFoundation knows the common modes by address,
    // and a string with the same text names a mode of its own that never runs.
    let common = kCFRunLoopCommonModes as *const AnyObject;
    // NSView's, not NSWindow's: the window answers to the selector too, and
    // its link never ticks.
    let view: *mut AnyObject = msg_send![window, contentView];
    let has_link = LINK_MISSES.with(Cell::get) < 2
        && !view.is_null()
        && msg_send![view, respondsToSelector: sel!(displayLinkWithTarget:selector:)];
    if has_link {
        let link: *mut AnyObject = msg_send![view, displayLinkWithTarget: &*ticker, selector: sel!(tick:)];
        if !link.is_null() {
            let _: *mut AnyObject = msg_send![link, retain];
            let _: () = msg_send![link, addToRunLoop: run_loop, forMode: common];
            return Some((link, true));
        }
    }
    let timer: *mut AnyObject = msg_send![
        class!(NSTimer),
        timerWithTimeInterval: 1.0f64 / 120.0,
        target: &*ticker,
        selector: sel!(tick:),
        userInfo: std::ptr::null_mut::<AnyObject>(),
        repeats: true
    ];
    if timer.is_null() {
        return None;
    }
    let _: *mut AnyObject = msg_send![timer, retain];
    let _: () = msg_send![run_loop, addTimer: timer, forMode: common];
    Some((timer, false))
}

/// One frame of the glide: where the curve is at the moment this frame shows.
unsafe fn step(driver: *mut AnyObject) {
    let send = GLIDE.with(|g| {
        let mut g = g.borrow_mut();
        let glide = g.as_mut()?;
        if glide.driver != driver {
            return None;
        }
        let now = CACurrentMediaTime();
        let at: f64 = if glide.link { msg_send![driver, targetTimestamp] } else { now };
        glide.ticked = now;
        glide.ticks += 1;
        if glide.link {
            LINK_MISSES.with(|n| n.set(0));
        }
        // The first frame is one frame into the movement, not its start.
        let start = *glide.start.get_or_insert_with(|| {
            let shown: f64 = if glide.link { msg_send![driver, timestamp] } else { at - 1.0 / 120.0 };
            if shown < at {
                shown
            } else {
                at - 1.0 / 60.0
            }
        });
        let (pos, _) = glide.at(at);
        glide.last_at = Some(at);
        let want = pos.round() as i64;
        let px = want - glide.sent;
        glide.sent = want;
        if at - start >= glide.dur && px == 0 {
            g.take().expect("a glide").end();
            return None;
        }
        Some((glide.template, px))
    });
    if let Some((template, px)) = send {
        if px != 0 {
            post(template, px);
        }
    }
}

/// A precise scroll event of `px` pixels, where and as the wheel event was.
unsafe fn post(template: CGEventRef, px: i64) {
    let e = CGEventCreateCopy(template);
    if e.is_null() {
        return;
    }
    CGEventSetIntegerValueField(e, IS_CONTINUOUS, 1);
    CGEventSetIntegerValueField(e, SCROLL_PHASE, 0);
    CGEventSetIntegerValueField(e, MOMENTUM_PHASE, 0);
    CGEventSetIntegerValueField(e, DELTA_AXIS_1, px);
    CGEventSetDoubleValueField(e, FIXED_PT_DELTA_AXIS_1, px as f64);
    CGEventSetIntegerValueField(e, POINT_DELTA_AXIS_1, px);
    CGEventSetIntegerValueField(e, DELTA_AXIS_2, 0);
    CGEventSetDoubleValueField(e, FIXED_PT_DELTA_AXIS_2, 0.0);
    CGEventSetIntegerValueField(e, POINT_DELTA_AXIS_2, 0);
    let ns: *mut AnyObject = msg_send![class!(NSEvent), eventWithCGEvent: e];
    if !ns.is_null() {
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let _: () = msg_send![app, sendEvent: ns];
    }
    CFRelease(e);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ease_in_out_runs_from_zero_to_one() {
        assert!(ease(0.0, 0.0).0.abs() < 1e-6);
        assert!((ease(0.0, 1.0).0 - 1.0).abs() < 1e-6);
        assert!((ease(0.0, 0.5).0 - 0.5).abs() < 1e-3);
        let mut last = 0.0;
        for i in 1..=100 {
            let (y, _) = ease(0.0, i as f64 / 100.0);
            assert!(y >= last, "monotonic at {i}");
            last = y;
        }
    }

    #[test]
    fn a_carried_slope_is_the_starting_speed() {
        let (_, d) = ease(1.5, 0.0);
        assert!((d - 1.5).abs() < 1e-3, "{d}");
        // Chromium's bound keeps a carried slope at or under 2.5: no overshoot to speak of.
        let peak = (0..=1000).map(|i| ease(2.5, i as f64 / 1000.0).0).fold(0.0, f64::max);
        assert!(peak < 1.001, "{peak}");
    }

    #[test]
    fn chromium_durations() {
        assert!((duration(40.0) - 0.2).abs() < 1e-9);
        assert!((duration(-120.0) - 0.2).abs() < 1e-9);
        assert!((duration(300.0) - 0.15).abs() < 1e-9);
        assert!((duration(900.0) - 0.1).abs() < 1e-9);
    }

    fn glide(delta: f64) -> Glide {
        Glide {
            from: 0.0,
            delta,
            dur: duration(delta),
            slope: 0.0,
            start: Some(0.0),
            last_at: Some(0.05),
            sent: 0,
            template: std::ptr::null_mut(),
            driver: std::ptr::null_mut(),
            link: true,
            ticked: 0.0,
            ticks: 1,
        }
    }

    #[test]
    fn a_second_notch_keeps_position_and_speed() {
        let mut g = glide(40.0);
        let (pos, speed) = g.at(0.05);
        g.add(40.0);
        let (pos2, speed2) = g.at(0.05);
        assert!((pos2 - pos).abs() < 1e-9);
        assert!((speed2 - speed).abs() / speed < 1e-3, "{speed} vs {speed2}");
        assert!((g.at(0.05 + g.dur).0 - 80.0).abs() < 1e-9);
        // Every frame after it moves at least as far as the one before it did,
        // until the easing out: no dip where the notch came in.
        let frame = 1.0 / 60.0;
        let before = pos - glide(40.0).at(0.05 - frame).0;
        let after = g.at(0.05 + frame).0 - pos;
        assert!(after >= before, "{before} then {after}");
    }

    #[test]
    fn turning_back_drops_what_was_left() {
        let mut g = glide(120.0);
        let (pos, _) = g.at(0.05);
        g.add(-40.0);
        assert!((g.at(0.05).0 - pos).abs() < 1e-9);
        assert!((g.at(1.0).0 - (pos - 40.0)).abs() < 1e-9);
        assert!(g.at(0.05 + 1.0 / 60.0).0 <= pos, "moves back from the first frame on");
    }

    #[test]
    fn notches_before_the_first_frame_add_up() {
        let mut g = glide(40.0);
        g.start = None;
        g.last_at = None;
        g.add(40.0);
        assert_eq!(g.delta, 80.0);
        g.add(-40.0);
        assert_eq!(g.delta, -40.0);
    }
}
