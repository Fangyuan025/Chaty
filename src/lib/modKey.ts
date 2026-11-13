import { platform } from "@tauri-apps/plugin-os";

/** How the shortcut modifier is written: "⌘" on macOS, "Ctrl+" elsewhere —
 *  "⌘/Ctrl+Enter" made everyone read past the key that was not theirs. */
export const MOD = (() => {
  try {
    return platform() === "macos" ? "⌘" : "Ctrl+";
  } catch {
    return /mac/i.test(navigator.userAgent) ? "⌘" : "Ctrl+";
  }
})();
