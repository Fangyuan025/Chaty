// Native clipboard via Tauri (no WebView2 permission prompt). Falls back to the
// web Clipboard API if the plugin isn't available (e.g. a plain browser build).
import { readText, writeText } from "@tauri-apps/plugin-clipboard-manager";

export async function copyToClipboard(text: string): Promise<void> {
  try {
    await writeText(text);
  } catch {
    try {
      await navigator.clipboard?.writeText(text);
    } catch {
      /* ignore */
    }
  }
}

export async function readFromClipboard(): Promise<string> {
  try {
    return await readText();
  } catch {
    try {
      return (await navigator.clipboard?.readText()) ?? "";
    } catch {
      return "";
    }
  }
}
