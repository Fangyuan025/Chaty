import type { TKey } from "./i18n";

/** Hugging Face's errors come from the backend as a marker —
 *  `HF_ERROR\t<kind>\t<status>` (download.rs) — and are said here, in the
 *  UI's language. Anything else passes through as it is. */
export function hfErrorText(msg: string, t: (key: TKey, vars?: Record<string, string | number>) => string): string {
  const [tag, kind, code] = msg.split("\t");
  if (tag !== "HF_ERROR") return msg;
  switch (kind) {
    case "server":
      return t("hfErrServer", { code });
    case "limited":
      return t("hfErrLimited");
    case "timeout":
      return t("hfErrTimeout");
    case "unreachable":
      return t("hfErrUnreachable");
    default:
      return t("hfErrStatus", { code });
  }
}
