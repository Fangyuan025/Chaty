import { describe, expect, it } from "vitest";
import { chatyPrompt, earlierModels, type ChatyFacts } from "./chatyPrompt";

const facts = (over: Partial<ChatyFacts> = {}, model: Partial<ChatyFacts["model"]> = {}): ChatyFacts => ({
  lang: "en",
  version: "2.4.2",
  os: "macos",
  model: {
    name: "Qwen3-8B-Q4_K_M.gguf",
    backend: "llama.cpp",
    modelName: "Qwen3 8B",
    paramsB: 8.19,
    quant: "Q4_K_M",
    nCtx: 32768,
    visionReady: false,
    ...model,
  },
  hardware: { cpu: "Apple M4 Pro", ramMb: 49152, gpu: "Apple M4 Pro" },
  ...over,
});

describe("Chaty's prompt", () => {
  it("names the model as the chip does, the engine, the window and the computer", () => {
    const p = chatyPrompt(facts());
    expect(p).toContain('"Qwen3-8B-Q4_K_M"');
    expect(p).not.toContain(".gguf");
    expect(p).toContain("8.2B parameters, Q4_K_M");
    expect(p).toContain("llama.cpp (GGUF)");
    expect(p).toContain("32768 tokens");
    expect(p).toContain("macOS, Apple M4 Pro, 48 GB memory. Chaty 2.4.2.");
    // The metadata's name says nothing the file's does not.
    expect(p).not.toContain("metadata calls it");
    // A GPU that is the chip itself is not named twice.
    expect(p).not.toContain("GPU Apple");
  });

  it("gives the metadata's name when it says something else", () => {
    const p = chatyPrompt(facts({}, { name: "model-q4.gguf", modelName: "Llama 3.1 8B Instruct" }));
    expect(p).toContain("its metadata calls it Llama 3.1 8B Instruct");
  });

  it("reads the same for the same load — it opens the prompt the engine reuses", () => {
    expect(chatyPrompt(facts())).toBe(chatyPrompt(facts()));
    expect(chatyPrompt(facts({ lang: "zh" }))).toBe(chatyPrompt(facts({ lang: "zh" })));
  });

  it("says what the model can see", () => {
    expect(chatyPrompt(facts({}, { visionReady: true }))).toContain("You can see images");
    expect(chatyPrompt(facts())).toContain("cannot see images");
  });

  it("says which math is drawn", () => {
    expect(chatyPrompt(facts())).toContain("$…$ inline, $$…$$ on its own line");
  });

  it("is written in Chinese for the Chinese interface", () => {
    const p = chatyPrompt(facts({ lang: "zh", os: "windows", hardware: { cpu: "AMD Ryzen 7", ramMb: 32768, gpu: "NVIDIA RTX 4070" } }, { backend: "mlx", name: "Qwen3.5-4B-mlx-8bit" }));
    expect(p).toContain("你是模型「Qwen3.5-4B-mlx-8bit」");
    expect(p).toContain("由 MLX 加载运行");
    expect(p).toContain("这台电脑：Windows，AMD Ryzen 7，32 GB 内存，GPU NVIDIA RTX 4070。");
    expect(p).not.toMatch(/[一-鿿][,:;]/);
  });

  it("leaves out what is not known", () => {
    const p = chatyPrompt(facts({ hardware: null }, { paramsB: null, quant: null, nCtx: null }));
    expect(p).toContain('You are the model "Qwen3-8B-Q4_K_M", run by');
    expect(p).not.toContain("Context window");
    expect(p).toContain("This computer: macOS. Chaty");
  });

  it("tells a model that took over a conversation whose replies above were another's", () => {
    const history = [
      { role: "user" },
      { role: "assistant", model: "Gemma-4-12B-Q4_K_M" },
      { role: "user" },
      { role: "assistant" }, // written before names were kept
      { role: "assistant", model: "Qwen3-8B-Q4_K_M" },
      { role: "assistant", model: "LFM2-8B" },
      { role: "assistant", model: "Gemma-4-12B-Q4_K_M" },
    ];
    const earlier = earlierModels(history, "Qwen3-8B-Q4_K_M");
    expect(earlier).toEqual(["Gemma-4-12B-Q4_K_M", "LFM2-8B"]);
    const en = chatyPrompt(facts({ earlier }));
    expect(en).toContain('some earlier replies were written by "Gemma-4-12B-Q4_K_M", "LFM2-8B", not by you');
    expect(en).toContain('You are the model "Qwen3-8B-Q4_K_M"');
    const zh = chatyPrompt(facts({ lang: "zh", earlier }));
    expect(zh).toContain("前面有些回复是「Gemma-4-12B-Q4_K_M」、「LFM2-8B」写的，不是你写的");
    // One model all along: the prompt is the one it always was.
    expect(earlierModels(history.slice(4, 5), "Qwen3-8B-Q4_K_M")).toEqual([]);
    expect(chatyPrompt(facts({ earlier: [] }))).toBe(chatyPrompt(facts()));
    expect(chatyPrompt(facts({ earlier: ["Qwen3-8B-Q4_K_M"] }))).toBe(chatyPrompt(facts()));
  });
});
