// Fetching what a music model still lacks — a VAE, configs, the other parts
// of its package — into its own folder, where the engine looks for them.
// Shared by the store (after the model itself) and the companion dialog.

import { downloadModel, hfResolveUrl, type MusicSuggestion } from "./ipc";

function dirname(p: string): string {
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return i > 0 ? p.slice(0, i) : p;
}

/** The folder inside the model's folder a part goes to ("sidecars"), or
 *  undefined for the folder itself. */
export function destSubdir(dest: string): string | undefined {
  const i = dest.lastIndexOf("/");
  return i > 0 ? dest.slice(0, i) : undefined;
}

/** Download `todo` into the folder of the model at `modelPath`. `onBytes`
 *  hears the bytes done so far across all of them; `onFile` the name of the
 *  one under way (the key a cancel names). Throws what the download threw. */
export async function downloadMusicParts(
  modelPath: string,
  todo: MusicSuggestion[],
  onBytes: (done: number) => void,
  onFile: (name: string) => void,
  onError: (message: string) => void,
): Promise<number> {
  const dir = dirname(modelPath);
  let base = 0;
  for (const s of todo) {
    const name = s.dest.split("/").pop() ?? s.dest;
    onFile(name);
    let got = 0;
    await downloadModel(
      hfResolveUrl(s.repo, s.file),
      name,
      (p) => {
        if (p.type === "progress") {
          got = p.downloaded;
          onBytes(base + got);
        } else if (p.type === "error") {
          onError(p.message);
        }
      },
      destSubdir(s.dest),
      dir,
    );
    base += Math.max(got, s.size);
  }
  return base;
}
