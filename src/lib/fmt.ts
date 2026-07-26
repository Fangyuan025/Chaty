// The app's ONE set of number formatters. Before this module, byte sizes had
// two competing implementations — Settings divided by 1e9, the model store by
// 1<<30 — so the same 35 GB model read ~2.4 GB smaller in one screen than the
// other. Every size in the UI now goes through fmtBytes.

/** Bytes → human string, DECIMAL units (1 GB = 1e9 B) — matching Finder,
 *  Hugging Face, and every download page a model size gets compared against. */
export function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${Math.round(n)} B`;
}

/** Large counts → compact string (download counters, token tallies). */
export function fmtCount(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}
