// Shared time/ETA helpers for download progress bars.

/** Format seconds as "m:ss". */
export function fmtTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export type EtaSample = { t: number; b: number };

/**
 * Push a download sample (call from the progress callback) and return a
 * smoothed estimate of the seconds remaining, or `null` when there isn't
 * enough signal yet (or the transfer is idle/finished). `store` is a caller-held
 * array (e.g. a ref) used as a moving ~6s window so the estimate stays stable.
 */
export function etaSeconds(store: EtaSample[], downloaded: number, total: number): number | null {
  if (!total || downloaded <= 0 || downloaded >= total) {
    if (downloaded <= 0) store.length = 0; // reset between downloads
    return null;
  }
  const now = Date.now();
  // A backwards jump means a fresh download reused this store — restart.
  if (store.length && downloaded < store[store.length - 1].b) store.length = 0;
  if (!store.length || store[store.length - 1].b !== downloaded) {
    store.push({ t: now, b: downloaded });
  }
  while (store.length > 2 && now - store[0].t > 6000) store.shift();
  if (store.length < 2) return null;
  const dt = (now - store[0].t) / 1000;
  const db = downloaded - store[0].b;
  if (dt < 0.4 || db <= 0) return null;
  return (total - downloaded) / (db / dt);
}
