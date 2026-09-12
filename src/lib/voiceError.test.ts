import { describe, expect, it } from "vitest";
import { parseVoiceDownloadFailure, voiceDownloadMessage, type VoiceDownloadFailure } from "./voiceError";

const WIN_DIR = "C:\\Users\\me\\AppData\\Roaming\\com.chaty.desktop\\voice-models";

const whisper: VoiceDownloadFailure = {
  dir: `${WIN_DIR}\\sherpa-onnx-whisper-base`,
  modelsDir: WIN_DIR,
  hfUrl: "https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-base/tree/bb53ee204431c90d314c1cc08d28d23e5b7927cc",
  files: ["base-encoder.int8.onnx", "base-decoder.int8.onnx", "base-tokens.txt"],
  archive: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2",
  reasons: ["hf-mirror.com: timed out", "github.com: could not connect"],
};

describe("voice download failures", () => {
  it("are recognised in the raw IPC error, and nothing else is", () => {
    const raw = `VOICE_MODEL_DOWNLOAD ${JSON.stringify(whisper)}`;
    expect(parseVoiceDownloadFailure(raw)).toEqual(whisper);
    expect(parseVoiceDownloadFailure(new Error(raw))).toEqual(whisper);
    expect(parseVoiceDownloadFailure("创建 Whisper 失败: bad model")).toBeNull();
    expect(parseVoiceDownloadFailure("VOICE_MODEL_DOWNLOAD {broken")).toBeNull();
  });

  // Issue #14: the reporter could not tell where the files go.
  it("say where the files go and how to get them", () => {
    for (const lang of ["zh", "en"] as const) {
      const msg = voiceDownloadMessage(whisper, lang);
      expect(msg).toContain(whisper.dir);
      expect(msg).toContain(whisper.hfUrl);
      for (const f of whisper.files) expect(msg).toContain(f);
      expect(msg).toContain("hf-mirror.com: timed out");
      expect(msg).toContain("hf-mirror.com");
    }
  });

  it("summarise a long file list instead of printing it", () => {
    const many = { ...whisper, files: ["model.onnx", "lexicon.txt", "tokens.txt", "dict/jieba.dict.utf8"] };
    const msg = voiceDownloadMessage(many, "zh");
    expect(msg).not.toContain("dict/jieba.dict.utf8");
    expect(msg).toContain("dict");
  });

  it("point an archive-only model at the archive and the models folder", () => {
    const kokoro = { ...whisper, hfUrl: null, files: [], dir: `${WIN_DIR}\\kokoro-en-v0_19` };
    const msg = voiceDownloadMessage(kokoro, "en");
    expect(msg).toContain(kokoro.archive);
    expect(msg).toContain(WIN_DIR);
  });
});
