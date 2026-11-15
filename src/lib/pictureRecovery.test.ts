/** Pictures the engine cannot take must not end the session.
 *
 *  Issue #25: a Code session on Qwen3.5 9B took a screenshot, the next prompt
 *  failed with "图文预填充失败:Eval failed with code: 1" — its KV was full,
 *  though the count said otherwise (an M-RoPE image is a few dozen positions
 *  and two thousand cells) — and with the screenshot still in the transcript,
 *  every message after it failed the same way. */
import { afterEach, describe, expect, it } from "vitest";

const g = globalThis as Record<string, unknown>;
g.window = globalThis;
g.localStorage ??= {
  getItem: () => null, setItem: () => {}, removeItem: () => {}, clear: () => {}, key: () => null, length: 0,
};
g.navigator ??= { userAgent: "chaty-test" };

const { mockIPC, clearMocks } = await import("@tauri-apps/api/mocks");
const { runAgentTurn, refusedPromptTokens, failedOnPictures } = await import("./agentLoop");

type Msg = { role: string; content: string; images?: string[] };
type Chan = { onmessage?: (ev: { type: string; [k: string]: unknown }) => void };

async function turn(fail: (msgs: Msg[]) => string | null, history: Msg[], images?: string[]) {
  const sent: Msg[][] = [];
  const errors: string[] = [];
  let finals = 0;
  mockIPC(async (cmd, args) => {
    if (cmd !== "generate") return null;
    const a = args as { request: { messages: Msg[] }; onEvent: Chan };
    sent.push(a.request.messages.map((m) => ({ ...m })));
    const why = fail(a.request.messages);
    if (why) throw new Error(why);
    a.onEvent.onmessage?.({ type: "token", text: "Done." });
    a.onEvent.onmessage?.({ type: "done", stats: { promptTokens: 100, completionTokens: 1, tokensPerSecond: 1, stopReason: "eos" } });
    return null;
  });
  await runAgentTurn(
    "go on", history as never, "/tmp/ws", "zh",
    {
      thinkMode: "off", maxSteps: 6, nCtx: 32768, visionReady: true, mediaPrefixReuse: true, multiImage: true, images,
      signal: { cancelled: false },
      approve: async () => true, approveDir: async () => false, approveSudo: async () => ({ ok: false }),
    } as never,
    { onThinking: () => {}, onAssistantText: () => {}, onStep: () => {}, onFinal: () => { finals++; }, onError: (m) => { errors.push(m); } },
  );
  return { sent, errors, finals };
}

const pictures = (msgs: Msg[]) => msgs.reduce((n, m) => n + (m.images?.length ?? 0), 0);

describe("pictures the engine cannot take", () => {
  afterEach(() => clearMocks());

  it("reads a refusal in either language", () => {
    expect(refusedPromptTokens(new Error("提示词 9001 tokens 超出上下文窗口 8192，请新建对话"))).toBe(9001);
    expect(refusedPromptTokens(new Error("The prompt (9001 tokens) exceeds the 8192-token context window — start a new chat"))).toBe(9001);
    expect(failedOnPictures(new Error("图文预填充失败:Eval failed with code: 1"))).toBe(true);
    expect(failedOnPictures(new Error("multimodal prefill failed: Eval failed with code: 1"))).toBe(true);
  });

  it("a vision prefill that fails goes on without the pictures, the older first", async () => {
    const history: Msg[] = [
      { role: "user", content: "看看页面", images: ["/tmp/a.png"] },
      { role: "assistant", content: "看到了。" },
      { role: "user", content: "再截一张", images: ["/tmp/b.png"] },
      { role: "assistant", content: "好。" },
    ];
    const { sent, errors, finals } = await turn((m) => (pictures(m) > 0 ? "图文预填充失败:Eval failed with code: 1" : null), history);
    expect(errors).toEqual([]);
    expect(finals).toBe(1);
    // Two pictures, then the newest alone, then none.
    expect(sent.map(pictures)).toEqual([2, 1, 0]);
    // The model is told the picture left, so it can take another look.
    expect(sent[2].some((m) => m.content.includes("截图已过期"))).toBe(true);
  });

  it("a prompt refused for its pictures, with nothing to summarise, lets them go", async () => {
    const history: Msg[] = [{ role: "user", content: "看看页面", images: ["/tmp/a.png", "/tmp/b.png"] }, { role: "assistant", content: "好。" }];
    const { sent, errors, finals } = await turn(
      (m) => (pictures(m) > 0 ? "提示词 9100 tokens 超出上下文窗口 8192，请新建对话、缩短输入或减少图片。" : null),
      history,
    );
    expect(errors).toEqual([]);
    expect(finals).toBe(1);
    expect(pictures(sent[sent.length - 1])).toBe(0);
  });

  it("a failure that is not about pictures still ends the turn", async () => {
    const { errors } = await turn(() => "模型未加载", [{ role: "user", content: "hi", images: ["/tmp/a.png"] }]);
    expect(errors.length).toBe(1);
  });
});
