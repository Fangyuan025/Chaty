/**
 * File and folder pickers. On Linux, the desktop's own chooser through the
 * XDG Desktop Portal — Nautilus's on GNOME, Dolphin's on KDE (issue #29) —
 * and the dialog plugin's GTK one only where no portal answers
 * (src-tauri/src/portal.rs). Elsewhere, the plugin's system dialog as before.
 *
 * Same calls and results as `@tauri-apps/plugin-dialog`'s `open` and `save`.
 */
import { invoke } from "@tauri-apps/api/core";
import {
  open as pluginOpen,
  save as pluginSave,
  type OpenDialogOptions,
  type OpenDialogReturn,
  type SaveDialogOptions,
} from "@tauri-apps/plugin-dialog";
import { platform } from "@tauri-apps/plugin-os";

const IS_LINUX = (() => {
  try {
    return platform() === "linux";
  } catch {
    return false;
  }
})();

type Picked = { kind: "unavailable" } | { kind: "cancelled" } | { kind: "paths"; paths: string[] };

/** The portal's answer, or `undefined` when there is no portal to ask. */
async function portal(request: OpenDialogOptions & { save?: boolean }): Promise<string[] | null | undefined> {
  if (!IS_LINUX) return undefined;
  let picked: Picked;
  try {
    picked = await invoke<Picked>("portal_pick", {
      request: {
        save: request.save ?? false,
        title: request.title,
        directory: request.directory ?? false,
        multiple: request.multiple ?? false,
        defaultPath: request.defaultPath,
        filters: request.filters ?? [],
      },
    });
  } catch {
    return undefined;
  }
  if (picked.kind === "unavailable") return undefined;
  if (picked.kind === "cancelled" || picked.paths.length === 0) return null;
  return picked.paths;
}

export async function open<T extends OpenDialogOptions>(options: T = {} as T): Promise<OpenDialogReturn<T>> {
  const paths = await portal(options);
  if (paths === undefined) return pluginOpen(options);
  if (paths === null) return null as OpenDialogReturn<T>;
  return (options.multiple ? paths : paths[0]) as OpenDialogReturn<T>;
}

export async function save(options: SaveDialogOptions = {}): Promise<string | null> {
  const paths = await portal({ ...options, save: true });
  if (paths === undefined) return pluginSave(options);
  return paths?.[0] ?? null;
}
