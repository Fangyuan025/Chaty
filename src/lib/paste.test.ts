/** A pasted file shows under its own name, not the time it was saved after. */
import { describe, expect, it } from "vitest";
import { attachmentName } from "./paste";

describe("attachmentName", () => {
  it("drops the time a pasted file was saved after", () => {
    expect(attachmentName("/data/attachments/20261104-221420990-image.png")).toBe("image.png");
    expect(attachmentName("C:\\data\\attachments\\20261104-221420990-report_Q3.pdf")).toBe("report_Q3.pdf");
  });
  it("leaves any other name alone", () => {
    expect(attachmentName("/Users/me/2026-11-04 notes.md")).toBe("2026-11-04 notes.md");
    expect(attachmentName("photo.jpg")).toBe("photo.jpg");
  });
});
