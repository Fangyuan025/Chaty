/** A store download, run by the app (lib/downloads.ts) rather than by the
 *  store's window: closing the store leaves it going. What it fetches is what
 *  the store's "Get" always fetched — the quant's files, a vision model's
 *  encoder beside them, an image model's VAE and text encoder, a music
 *  model's parts — and an MLX build as its whole folder. */
import {
  DOWNLOAD_CANCELLED,
  downloadMlxRepo,
  downloadModel,
  hfResolveUrl,
  imageModelProbe,
  modelFolderFor,
  musicModelProbe,
  type HfModelDetail,
} from "./ipc";
import { downloadMusicParts } from "./musicDownload";
import { dlBegin, dlCancelled, dlDone, dlFailed, dlFile, dlProgress, dlStopping } from "./downloads";

/** One download per repo and build. */
export function storeTaskId(d: HfModelDetail, qi: number): string {
  return `store:${d.id}:${d.format === "mlx" ? "mlx" : (d.quants[qi]?.label ?? String(qi))}`;
}

/** Everything the store's "Get" brings, in bytes. */
export function storeNeedBytes(d: HfModelDetail, qi: number): number {
  const quant = d.quants[qi];
  if (!quant) return 0;
  if (d.format === "mlx") return quant.size;
  const companions = d.image
    ? (d.companions ?? []).reduce((a, c) => a + c.size, 0)
    : d.music
      ? (quant.extra ?? 0)
      : 0;
  return quant.size + (d.mmproj ? d.mmprojSize : 0) + companions;
}

export async function runStoreDownload(d: HfModelDetail, qi: number): Promise<void> {
  const quant = d.quants[qi];
  if (!quant) return;
  const id = storeTaskId(d, qi);
  let grandTotal = storeNeedBytes(d, qi);
  const name = d.id.split("/").pop() ?? d.id;
  if (!dlBegin({ id, name, variant: d.format === "mlx" ? "MLX" : quant.label, repo: d.id, total: grandTotal })) return;
  const tick = (done: number) => dlProgress(id, done, grandTotal);
  // The first failure a file reports — what the download is said to have
  // failed with, rather than the bare rejection after it.
  let failed = "";
  const onError = (msg: string) => {
    if (msg !== DOWNLOAD_CANCELLED && !failed) failed = msg;
  };
  const stopIfAsked = () => {
    if (dlStopping(id)) throw DOWNLOAD_CANCELLED;
  };
  try {
    if (d.format === "mlx") {
      dlFile(id, name);
      const folder = await downloadMlxRepo(d.id, (p) => {
        if (p.type === "progress") tick(p.downloaded);
        else if (p.type === "error") onError(p.message);
      });
      dlDone(id, folder);
      return;
    }
    const subdir = modelFolderFor(d.id);
    // A package kept in a folder of a larger repo names the repo apart.
    const repo = d.repo || d.id;
    const files = [...quant.files, ...(d.mmproj ? [d.mmproj] : [])];
    let doneBase = 0;
    let mainPath = "";
    for (const path of files) {
      stopIfAsked();
      const file = path.split("/").pop() ?? path;
      dlFile(id, file);
      let thisFile = 0;
      await downloadModel(
        hfResolveUrl(repo, path),
        file,
        (p) => {
          if (p.type === "progress") {
            thisFile = p.downloaded;
            tick(doneBase + thisFile);
          } else if (p.type === "done") {
            if (!mainPath) mainPath = p.path;
          } else if (p.type === "error") onError(p.message);
        },
        subdir,
      );
      doneBase += thisFile;
    }
    // A music model: its VAE, configs and package parts, into its own folder
    // — asking the model itself, so parts already there (from another
    // quantization) are not fetched again.
    if (d.music && mainPath) {
      const probe = await musicModelProbe(mainPath).catch(() => null);
      const todo = probe?.suggestions ?? [];
      grandTotal = doneBase + todo.reduce((a, c) => a + c.size, 0);
      if (probe && todo.length) {
        stopIfAsked();
        doneBase += await downloadMusicParts(
          probe.path,
          todo,
          (done) => tick(doneBase + done),
          (file) => dlFile(id, file),
          onError,
        );
      }
    }
    // An image model: what it still lacks, into its own folder — asking the
    // model itself, so an encoder another model already brought is reused.
    if (d.image && mainPath) {
      const probe = await imageModelProbe(mainPath).catch(() => null);
      const todo = probe?.suggestions ?? [];
      grandTotal = doneBase + todo.reduce((a, c) => a + c.size, 0);
      const dir = mainPath.replace(/[/\\][^/\\]*$/, "");
      for (const c of todo) {
        stopIfAsked();
        const file = c.file.split("/").pop() ?? c.file;
        dlFile(id, file);
        let thisFile = 0;
        await downloadModel(
          hfResolveUrl(c.repo, c.file),
          file,
          (p) => {
            if (p.type === "progress") {
              thisFile = p.downloaded;
              tick(doneBase + thisFile);
            } else if (p.type === "error") onError(p.message);
          },
          undefined,
          dir,
        );
        doneBase += thisFile;
      }
    }
    dlDone(id, mainPath);
  } catch (e) {
    const msg = typeof e === "string" ? e : e instanceof Error ? e.message : "";
    if (msg === DOWNLOAD_CANCELLED || dlStopping(id)) dlCancelled(id);
    else dlFailed(id, failed || msg);
  }
}
