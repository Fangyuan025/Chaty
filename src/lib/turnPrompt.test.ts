import { describe, expect, it } from "vitest";
import { turnMessage } from "./turnPrompt";

describe("turnMessage", () => {
  it("is just the question when the turn brings no context", () => {
    expect(turnMessage([], "官方站", "用户的问题：")).toBe("官方站");
  });

  // Issue #15: the question must not read as the last line of the passages.
  it("sets the question apart from retrieved passages, by name", () => {
    const msg = turnMessage(["指令\n\n【1】 书\n……官方站上线了。"], "官方站", "用户的问题：");
    expect(msg.endsWith("\n\n---\n\n用户的问题：官方站")).toBe(true);
    expect(msg.startsWith("指令")).toBe(true);
  });

  it("keeps a date-only turn as it was", () => {
    expect(turnMessage(["当前日期是 2026-09-12。"], "今天几号")).toBe("当前日期是 2026-09-12。\n\n今天几号");
  });
});
