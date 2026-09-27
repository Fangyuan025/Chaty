import { getCurrentWindow } from "@tauri-apps/api/window";

/**
 * The window moves from its top edge whatever is covering it. The title row
 * carries Tauri's `data-tauri-drag-region`, but a full-window layer over it —
 * Live mode, a dialog's backdrop, a menu's click-catcher — takes the press
 * instead, and the window could not be moved while one was open (issue #20
 * found it under the download dialog; Live mode had it too). Rather than
 * give every such layer its own strip, a press in the top band that lands on
 * nothing interactive moves the window once the pointer travels a few
 * pixels. A plain click still reaches the layer (a backdrop still closes
 * its dialog); presses Tauri's own drag regions take are left to Tauri.
 */

/** The title row's height, in CSS pixels. */
const BAND = 44;
/** Travel before a press becomes a window drag. */
const SLOP = 3;

// What blocks a drag, as Tauri's drag script decides it.
const CLICKABLE_TAGS = new Set(["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA", "LABEL", "SUMMARY", "IFRAME", "VIDEO", "AUDIO"]);
const INTERACTIVE_ROLES = new Set(["button", "link", "menuitem", "tab", "checkbox", "radio", "switch", "option", "slider", "separator"]);

function clickable(el: HTMLElement): boolean {
  if (CLICKABLE_TAGS.has(el.tagName)) return true;
  const ce = el.getAttribute("contenteditable");
  if (ce !== null && ce !== "false") return true;
  const ti = el.getAttribute("tabindex");
  if (ti !== null && ti !== "-1") return true;
  return INTERACTIVE_ROLES.has(el.getAttribute("role") ?? "");
}

/** "tauri" when Tauri's drag script takes this press, "block" when something
 *  interactive (or an explicit `data-tauri-drag-region="false"`) is under
 *  it, "free" when it landed on plain surface. */
export function pressKind(path: EventTarget[]): "tauri" | "block" | "free" {
  const target = path[0];
  let tauriDone = false;
  for (const node of path) {
    if (!(node instanceof HTMLElement)) continue;
    const attr = node.getAttribute("data-tauri-drag-region");
    if (!tauriDone) {
      // Tauri's own walk: the first element with an opinion decides.
      if (clickable(node) && attr === null) return "block";
      if (attr === "false") return "block";
      if (attr === "deep") return "tauri";
      if (attr === "" || attr === "true") {
        if (node === target) return "tauri";
        tauriDone = true; // Tauri declines; keep looking for controls above
        continue;
      }
    } else if (clickable(node) && attr === null) {
      return "block";
    }
  }
  return "free";
}

/** Whether the press is on the target's own scrollbar. */
function onScrollbar(el: Element, x: number, y: number): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const r = el.getBoundingClientRect();
  const right = r.left + el.clientLeft + el.clientWidth;
  const bottom = r.top + el.clientTop + el.clientHeight;
  return (el.scrollHeight > el.clientHeight && x > right) || (el.scrollWidth > el.clientWidth && y > bottom);
}

export function installTopBandDrag(): void {
  if (!("__TAURI_INTERNALS__" in window)) return;
  let win: ReturnType<typeof getCurrentWindow>;
  try {
    win = getCurrentWindow();
  } catch {
    return;
  }
  let pending: { x: number; y: number } | null = null;

  document.addEventListener(
    "mousedown",
    (e) => {
      pending = null;
      if (e.button !== 0 || e.detail > 1 || e.clientY >= BAND) return;
      const path = e.composedPath();
      if (pressKind(path) !== "free") return;
      if (path[0] instanceof Element && onScrollbar(path[0], e.clientX, e.clientY)) return;
      pending = { x: e.clientX, y: e.clientY };
    },
    true,
  );
  document.addEventListener(
    "mousemove",
    (e) => {
      if (!pending) return;
      if (!(e.buttons & 1)) {
        pending = null;
        return;
      }
      if (Math.abs(e.clientX - pending.x) + Math.abs(e.clientY - pending.y) < SLOP) return;
      pending = null;
      e.preventDefault();
      void win.startDragging().catch(() => {});
    },
    true,
  );
  const clear = () => {
    pending = null;
  };
  document.addEventListener("mouseup", clear, true);
  window.addEventListener("blur", clear);
}
