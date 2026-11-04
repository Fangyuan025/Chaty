/**
 * Files pasted into an input — a screenshot, a picture copied from a page, a
 * file copied in Finder or Explorer — become attachments, as a dropped file
 * does. A paste of text stays text.
 */
import type { ClipboardEvent as ReactClipboardEvent } from "react";
import { invoke } from "@tauri-apps/api/core";

/** The first file a paste carries, saved where an attachment can point at it
 *  (the path), or null when it carries none. Takes the paste over when it
 *  does: a copied file also brings its name as text, which is not what was
 *  meant. */
export async function pastedFile(e: ReactClipboardEvent | ClipboardEvent): Promise<string | null> {
  const file = e.clipboardData?.files?.[0];
  if (!file) return null;
  e.preventDefault();
  const data = await new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).replace(/^data:[^,]*,/, ""));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
  return invoke<string>("save_pasted_file", { name: file.name, data });
}
