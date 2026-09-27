import { useI18n } from "../lib/i18n";
import { musicSessionSearch } from "../lib/ipc";
import type { MusicStudioState } from "../lib/useMusicStudio";
import { StudioSidebar } from "./StudioSidebar";

/** The sidebar in the music studio: its sessions, as the image studio lists
 *  its own. */
export function MusicSidebar({
  studio,
  busy,
  notify,
}: {
  studio: MusicStudioState;
  busy: boolean;
  notify: (kind: "warn" | "error", text: string) => void;
}) {
  const { t } = useI18n();
  return (
    <StudioSidebar
      studio={studio}
      busy={busy}
      notify={notify}
      search={musicSessionSearch}
      labels={{
        newSession: t("musNew"),
        search: t("musSearch"),
        empty: t("musNoHistory"),
        deleteTitle: t("musDeleteSession"),
        deleteConfirm: t("musDeleteSessionConfirm"),
      }}
    />
  );
}
