import { describe, expect, it } from "vitest";
import { hfErrorText } from "./hfError";
import { lookup, type TKey } from "./i18n";

const zh = (k: TKey, v?: Record<string, string | number>) => lookup(k, "zh", v);
const pt = (k: TKey, v?: Record<string, string | number>) => lookup(k, "pt", v);

describe("hfErrorText", () => {
  it("says a Hugging Face failure in the UI's language", () => {
    expect(hfErrorText("HF_ERROR\tserver\t504", zh)).toContain("HTTP 504");
    expect(hfErrorText("HF_ERROR\tserver\t504", zh)).toContain("服务器");
    expect(hfErrorText("HF_ERROR\tserver\t500", pt)).toContain("HTTP 500");
    expect(hfErrorText("HF_ERROR\tstatus\t400", pt)).toContain("HTTP 400");
    expect(hfErrorText("HF_ERROR\tlimited\t429", zh)).toContain("429");
    expect(hfErrorText("HF_ERROR\tunreachable\t", zh)).toContain("HuggingFace 端点");
    expect(hfErrorText("HF_ERROR\ttimeout\t", pt)).toContain("Endpoint HuggingFace");
  });

  it("leaves other errors as they are", () => {
    expect(hfErrorText("未找到仓库 (repo not found): a/b", zh)).toBe("未找到仓库 (repo not found): a/b");
    expect(hfErrorText("", zh)).toBe("");
  });
});
