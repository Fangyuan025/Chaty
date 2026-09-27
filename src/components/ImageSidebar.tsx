import { useI18n } from "../lib/i18n";
import { imageSessionSearch } from "../lib/ipc";
import type { ImageStudioState } from "../lib/useImageStudio";
import { StudioSidebar } from "./StudioSidebar";

/** The sidebar in the image studio: its sessions, the way the chat lists its
 *  conversations — pinned first, then most recent, searchable, renameable. */
export function ImageSidebar({
  studio,
  busy,
  notify,
}: {
  studio: ImageStudioState;
  busy: boolean;
  notify: (kind: "warn" | "error", text: string) => void;
}) {
  const { t } = useI18n();
  return (
    <StudioSidebar
      studio={studio}
      busy={busy}
      notify={notify}
      search={imageSessionSearch}
      labels={{
        newSession: t("imgNew"),
        search: t("imgSearch"),
        empty: t("imgNoHistory"),
        deleteTitle: t("imgDeleteSession"),
        deleteConfirm: t("imgDeleteSessionConfirm"),

        deleteManyConfirm: (n) => t("imgDeleteSessionsConfirm", { n }),
      }}
    />
  );
}
