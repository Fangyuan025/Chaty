import { detectLang, lookup, type Lang } from "./i18n";

/** What the backend reports when every source of a voice model failed
 *  (voice.rs `download_failed`). */
export interface VoiceDownloadFailure {
  /** Folder the model belongs in. */
  dir: string;
  /** Folder holding every voice model — where an archive unpacks. */
  modelsDir: string;
  /** Hugging Face page of the pinned snapshot, on the user's endpoint. */
  hfUrl: string | null;
  /** Files to fetch from that page, paths relative to `dir`. */
  files: string[];
  /** The GitHub release archive; it unpacks to a folder named like `dir`. */
  archive: string;
  /** One short line per source that failed. */
  reasons: string[];
}

const MARK = "VOICE_MODEL_DOWNLOAD ";

export function parseVoiceDownloadFailure(e: unknown): VoiceDownloadFailure | null {
  const s = typeof e === "string" ? e : e instanceof Error ? e.message : "";
  const at = s.indexOf(MARK);
  if (at < 0) return null;
  try {
    return JSON.parse(s.slice(at + MARK.length)) as VoiceDownloadFailure;
  } catch {
    return null;
  }
}

/** The failure as something a person can act on: why, the setting that
 *  usually fixes it, and where to put the files by hand. The old message was
 *  one raw request error naming huggingface.co — no folder, no way forward. */
export function voiceDownloadMessage(f: VoiceDownloadFailure, lang: Lang): string {
  const lines = [lookup("voiceDlFailed", lang, { reason: f.reasons.join("; ") })];
  if (f.hfUrl) {
    const files =
      f.files.length <= 3 ? f.files.join(lang === "zh" ? "、" : ", ") : lookup("voiceDlAllFiles", lang);
    lines.push(lookup("voiceDlMirror", lang));
    lines.push(lookup("voiceDlManual", lang, { url: f.hfUrl, files, dir: f.dir }));
  } else {
    lines.push(lookup("voiceDlArchive", lang, { url: f.archive, dir: f.modelsDir }));
  }
  return lines.join("\n");
}

/** The download was stopped from its progress bar — nothing to report. */
export function isVoiceDownloadCancelled(e: unknown): boolean {
  const s = typeof e === "string" ? e : e instanceof Error ? e.message : "";
  return s.includes("VOICE_DOWNLOAD_CANCELLED");
}

/** A voice IPC error, with a download failure rewritten for the reader. */
export function voiceError(e: unknown): unknown {
  const f = parseVoiceDownloadFailure(e);
  return f ? new Error(voiceDownloadMessage(f, detectLang())) : e;
}
