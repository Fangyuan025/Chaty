// chaty-mlx — MLX inference sidecar for Chaty.
//
// Protocol: JSON-lines. Commands arrive on stdin, events leave on stdout,
// human-readable logs go to stderr. One model per process; the Rust side
// ejects a model by killing the process, which is what guarantees the
// memory actually comes back (nothing survives process exit).
//
//   → {"cmd":"load","path":"/abs/model/dir","nCtx":8192}
//   → {"cmd":"generate","messages":[{"role":"user","content":"hi"}],
//      "params":{"temperature":0.7,"topP":0.95,"topK":40,"minP":0.05,
//                "repeatPenalty":1.1,"maxTokens":512,"seed":1,"think":false}}
//   → {"cmd":"cancel"}   → {"cmd":"quit"}
//
//   ← {"event":"ready"}
//   ← {"event":"loadProgress","frac":0.42}
//   ← {"event":"loaded","info":{…}}
//   ← {"event":"prefill","processed":512,"total":8000}
//   ← {"event":"token","text":"…"}
//   ← {"event":"done","promptTokens":8000,"completionTokens":42,
//      "tokensPerSecond":31.5,"stopReason":"eos|length|context|cancelled"}
//   ← {"event":"error","message":"…"}
//
// Stop *sequences* are deliberately not handled here: the Rust engine does
// boundary-safe holdback on the token stream (same code path as llama.cpp)
// and sends {"cmd":"cancel"} when a stop matches.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

// MARK: - wire types

struct WireMessage: Decodable {
    let role: String
    let content: String
    /// Image attachments as absolute file paths (VLM-loaded models only).
    let images: [String]?
}

struct WireParams: Decodable {
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repeatPenalty: Float?
    var maxTokens: Int?
    var seed: UInt64?
    /// Reasoning control: `false` forces no-think (template arg when the chat
    /// template supports `enable_thinking`, else an empty-<think> prefill).
    var think: Bool?
    /// Native reasoning-effort rung for templates taking a `reasoning_effort`
    /// kwarg (Qwen3.8: low | medium | xhigh). Passed straight through.
    var effort: String?
}

struct WireCmd: Decodable {
    let cmd: String
    let path: String?
    let nCtx: Int?
    let messages: [WireMessage]?
    let params: WireParams?
}

// MARK: - serialized stdout writer

final class Out: @unchecked Sendable {
    private let lock = NSLock()
    private let handle = FileHandle.standardOutput

    func emit(_ obj: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(obj),
            let data = try? JSONSerialization.data(withJSONObject: obj)
        else { return }
        lock.lock()
        defer { lock.unlock() }
        handle.write(data)
        handle.write(Data([0x0A]))
    }

    func error(_ message: String) { emit(["event": "error", "message": message]) }
}

let out = Out()

func log(_ s: String) {
    FileHandle.standardError.write(Data(("chaty-mlx: " + s + "\n").utf8))
}

// MARK: - cancel flag

final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var v = false
    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return v
    }
    func set() {
        lock.lock()
        v = true
        lock.unlock()
    }
    func reset() {
        lock.lock()
        v = false
        lock.unlock()
    }
}

// MARK: - model metadata from config.json / tokenizer_config.json

struct ModelMeta {
    var arch: String?
    var nLayer: Int?
    var nEmbd: Int?
    var nCtxTrain: Int?
    var quantBits: Int?
    var quantGroup: Int = 64
    var sizeBytes: UInt64 = 0
    var multimodal = false
    /// Image/video placeholder token ids from config.json — needed to find
    /// where the media embeddings sit inside the expanded token sequence.
    var imageTokenId: Int?
    var videoTokenId: Int?
    var hasChatTemplate = false
    /// Chat template honours an `enable_thinking` kwarg (Qwen3 family).
    var thinkArg = false
    /// Native reasoning-effort ladder the template accepts, weakest first
    /// (Qwen3.8: low/medium/xhigh). Empty ⇒ no effort control.
    var effortLevels: [String] = []
    /// Best effort: the model emits <think> reasoning.
    var supportsThinking = false
    /// Best effort: the chat template supports tool / function calling.
    var supportsTools = false

    var quantLabel: String {
        if let b = quantBits { return "\(b)-bit" }
        return "bf16"
    }

    /// Effective bits per weight incl. group scale/bias overhead, for a rough
    /// parameter-count estimate from the on-disk tensor bytes.
    var bitsPerWeight: Double {
        guard let b = quantBits else { return 16 }
        return Double(b) + 32.0 / Double(max(quantGroup, 1))
    }

    var paramsB: Double {
        guard sizeBytes > 0 else { return 0 }
        return Double(sizeBytes) * 8.0 / bitsPerWeight / 1e9
    }
}

/// Some community quants ship a VLM checkpoint without its processor
/// configuration (neither preprocessor_config.json nor processor_config.json)
/// — the vision tower is right there in the weights, but the VLM factory
/// throws a configurationFileError and the whole model refuses to load. For
/// families whose preprocessing values are architecture constants (mirrored
/// in config.json's vision_config, or baked into the library's decoder
/// defaults) the folder can be healed by writing a minimal
/// preprocessor_config.json. Never overwrites an existing file.
func healProcessorConfig(dir: URL) {
    let pre = dir.appendingPathComponent("preprocessor_config.json")
    let proc = dir.appendingPathComponent("processor_config.json")
    let fm = FileManager.default
    guard !fm.fileExists(atPath: pre.path), !fm.fileExists(atPath: proc.path),
        let cfg = readJSON(dir.appendingPathComponent("config.json")),
        let arch = cfg["model_type"] as? String
    else { return }
    var synthesized: [String: Any]
    switch arch {
    case "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe":
        // Qwen3-VL processor family: mean/std are the 0.5 constants, the
        // geometry fields are mirrored in config.json's vision_config.
        guard let vision = cfg["vision_config"] as? [String: Any] else { return }
        synthesized = [
            "processor_class": "Qwen3VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "patch_size": vision["patch_size"] as? Int ?? 16,
            "merge_size": vision["spatial_merge_size"] as? Int ?? 2,
            "temporal_patch_size": vision["temporal_patch_size"] as? Int ?? 2,
            // Smart-resize band (total pixels), mirroring the official configs.
            // Without it the processor never resizes and an oversized image
            // blows past Metal's limits inside mlx_eval, killing the process.
            "size": ["longest_edge": 16_777_216, "shortest_edge": 65_536],
        ]
    case "gemma4", "gemma4_unified":
        // Gemma4: every pixel constant has an architecture default baked
        // into the library's config decoder, so an almost-empty file
        // suffices — but the processor class must be spelled out (absent,
        // the decoder falls back to the Unified variant, wrong for plain
        // gemma4). Special-token ids are mirrored from config.json where
        // present so quants with re-numbered specials still line up.
        synthesized = [
            "processor_class": arch == "gemma4" ? "Gemma4Processor" : "Gemma4UnifiedProcessor"
        ]
        for key in [
            "image_token_id", "boi_token_id", "eoi_token_id", "audio_token_id", "video_token_id",
        ] {
            if let v = cfg[key] as? Int { synthesized[key] = v }
        }
    default:
        return
    }
    guard
        let data = try? JSONSerialization.data(
            withJSONObject: synthesized, options: [.prettyPrinted, .sortedKeys])
    else { return }
    if (try? data.write(to: pre)) != nil {
        log("healed missing processor config for \(arch): \(pre.path)")
    }
}

/// Cheap identity for an image file (path + size + mtime) — mirrors the GGUF
/// engine's `image_cache_key`.
func imageKey(_ path: String) -> String {
    let attrs = try? FileManager.default.attributesOfItem(atPath: path)
    let size = (attrs?[.size] as? UInt64) ?? 0
    let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
    return "\(path)|\(size)|\(mtime)"
}

func readJSON(_ url: URL) -> [String: Any]? {
    guard let data = try? Data(contentsOf: url) else { return nil }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

func inspectModelDir(_ dir: URL) -> ModelMeta {
    var meta = ModelMeta()
    let cfg = readJSON(dir.appendingPathComponent("config.json")) ?? [:]
    // Some VLM configs nest the text model under text_config.
    let text = (cfg["text_config"] as? [String: Any]) ?? cfg
    meta.arch = cfg["model_type"] as? String
    meta.nLayer = text["num_hidden_layers"] as? Int
    meta.nEmbd = text["hidden_size"] as? Int
    meta.nCtxTrain = text["max_position_embeddings"] as? Int
    if let q = cfg["quantization"] as? [String: Any] {
        meta.quantBits = q["bits"] as? Int
        meta.quantGroup = q["group_size"] as? Int ?? 64
    }
    meta.multimodal = cfg["vision_config"] != nil
    meta.imageTokenId = (cfg["image_token_id"] ?? cfg["image_token_index"]) as? Int
    meta.videoTokenId = (cfg["video_token_id"] ?? cfg["video_token_index"]) as? Int

    if let files = try? FileManager.default.contentsOfDirectory(
        at: dir, includingPropertiesForKeys: [.fileSizeKey])
    {
        for f in files where f.pathExtension == "safetensors" {
            let size = (try? f.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            meta.sizeBytes += UInt64(size)
        }
    }

    // Chat template lives either in tokenizer_config.json or a sibling
    // chat_template.jinja. Only scanned for capability flags — the actual
    // templating is swift-transformers' job.
    var template = ""
    if let tc = readJSON(dir.appendingPathComponent("tokenizer_config.json")),
        let t = tc["chat_template"] as? String
    {
        template = t
    } else if let t = try? String(
        contentsOf: dir.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
    {
        template = t
    }
    meta.hasChatTemplate = !template.isEmpty
    meta.thinkArg = template.contains("enable_thinking")
    if template.contains("reasoning_effort") {
        meta.effortLevels = ["low", "medium", "xhigh"].filter {
            template.contains("'\($0)'") || template.contains("\"\($0)\"")
        }
    }
    let archLower = (meta.arch ?? "").lowercased()
    meta.supportsThinking =
        template.contains("<think>") || meta.thinkArg || archLower.contains("qwen3")
    meta.supportsTools = template.contains("tool_call") || template.contains("tools")
    return meta
}

// MARK: - engine

final class Engine: @unchecked Sendable {
    var container: ModelContainer?
    var meta = ModelMeta()
    var modelDir: URL?
    var nCtxCap = 0
    let cancelFlag = Flag()

    /// Prompt tokens currently materialized in `kvCache` with positions this
    /// engine computed itself (the cache also holds the sampled tokens past
    /// this list, but those were written by the library's decode loop, whose
    /// M-RoPE positions drift after images — the next turn trims them away
    /// and re-evaluates that text instead of trusting them).
    var kvCache: [KVCache]?
    var kvTokens: [Int] = []
    /// Exact number of tokens materialized in `kvCache` (prompt + decoded).
    /// Tracked here because the caches themselves can't be trusted for this:
    /// hybrid models put a MambaCache at layer 0 whose `offset` never
    /// advances, so `cache.first.offset` reads 0 forever — the old code
    /// concluded "nothing to trim" and appended a NEW conversation on top of
    /// the previous one's KV (cross-conversation contamination).
    var kvEvaluated = 0
    /// Identity keys (path|size|mtime) of the images baked into `kvCache`,
    /// in prompt order — the analogue of the GGUF engine's media cache.
    /// Image placeholder tokens are identical for different pictures, so
    /// token equality alone proves nothing about the pixels behind them.
    var kvImageKeys: [String] = []
    /// M-RoPE continuation state (ropeDeltas) from the last prefill chunk.
    /// VLM-factory models compute image-aware positions once per sequence
    /// and continue linearly from this; resuming a trimmed cache without it
    /// would restart positions at zero.
    var kvState: LMOutput.State?

    static let prefillChunk = 512
    /// Only bother the UI with prefill events for prompts big enough to have
    /// a visible gap (mirrors the llama.cpp engine's behaviour).
    static let prefillEventThreshold = 256

    func load(path: String, nCtx: Int?) async {
        let dir = URL(fileURLWithPath: path)
        var meta = inspectModelDir(dir)
        var loadWarning: String?
        if meta.multimodal {
            healProcessorConfig(dir: dir)
            let hasProcessorConfig = ["preprocessor_config.json", "processor_config.json"]
                .contains { fname in
                    FileManager.default.fileExists(
                        atPath: dir.appendingPathComponent(fname).path)
                }
            // A multimodal config with no processor configuration at all —
            // and no healing recipe for its family — can't drive its vision
            // tower, but the language model underneath is fully loadable:
            // the text factory registers these architectures too and its
            // sanitize drops the vision_tower / multi_modal_projector
            // weights. Degrade to text-only instead of refusing to load.
            if !hasProcessorConfig {
                log(
                    "no processor config for \(meta.arch ?? "?") and no healing recipe; "
                        + "degrading to text-only load")
                meta.multimodal = false
                loadWarning = "vision-config-missing"
            }
        }
        self.meta = meta
        self.modelDir = dir
        let useVLM = meta.multimodal
        do {
            // Local-directory load: no downloader involved; the tokenizer
            // comes from swift-transformers via the MLXHuggingFace macro.
            // Natively-multimodal architectures (config carries a
            // vision_config, e.g. the whole Qwen3.5+ family) load through
            // the VLM factory; text-only models — and vision models
            // degraded above — through the LLM factory.
            // withError: C-level MLX failures during weight load (Metal OOM
            // on a too-big model) must surface as a load error, not kill the
            // sidecar — same boxing as the generate path.
            let container: ModelContainer = try await withError {
                if useVLM {
                    return try await VLMModelFactory.shared.loadContainer(
                        from: dir, using: #huggingFaceTokenizerLoader())
                } else {
                    return try await LLMModelFactory.shared.loadContainer(
                        from: dir, using: #huggingFaceTokenizerLoader())
                }
            }
            self.container = container
            let trained = meta.nCtxTrain ?? 4096
            self.nCtxCap = min(nCtx ?? trained, trained)
            var info: [String: Any] = [
                "quant": meta.quantLabel,
                "sizeMb": Int(meta.sizeBytes / (1024 * 1024)),
                "paramsB": (meta.paramsB * 100).rounded() / 100,
                "nCtx": self.nCtxCap,
                "hasChatTemplate": meta.hasChatTemplate,
                "supportsThinking": meta.supportsThinking,
                "thinkArg": meta.thinkArg,
                "supportsTools": meta.supportsTools,
                "effortLevels": meta.effortLevels,
                // VLM-factory models have their vision tower loaded and
                // ready — no separate encoder file like GGUF's mmproj.
                "multimodal": meta.multimodal,
            ]
            if let v = meta.arch { info["arch"] = v }
            if let v = meta.nLayer { info["nLayer"] = v }
            if let v = meta.nEmbd { info["nEmbd"] = v }
            if let v = meta.nCtxTrain { info["nCtxTrain"] = v }
            if let v = loadWarning { info["warning"] = v }
            out.emit(["event": "loaded", "info": info])
        } catch {
            out.error("模型加载失败 (failed to load MLX model): \(error)")
        }
    }

    func generate(messages: [WireMessage], params: WireParams) async {
        guard let container else {
            out.error("尚未加载模型 (no model loaded)")
            return
        }
        cancelFlag.reset()
        do {
            // withError: box MLX's C-level runtime errors (Metal allocation
            // failures, shape errors) into thrown Swift errors. Without it the
            // GLOBAL handler fires — and its default is assertionFailure,
            // which killed the whole sidecar mid-task ("MLX 引擎意外退出",
            // owner crash report 2026-08-01: _mlx_error during eval on the
            // vision round after a screenshot). A generation must be able to
            // fail without taking the engine down with it.
            try await withError {
                try await container.perform { context in
                    try await self.run(context: context, messages: messages, params: params)
                }
            }
        } catch {
            // A failure can leave half-evaluated KV behind — drop the cache
            // so the next turn starts from a clean slate. After a Metal-level
            // error, also return scratch buffers to the OS so a post-OOM
            // retry starts with headroom.
            kvCache = nil
            kvTokens = []
            kvImageKeys = []
            kvState = nil
            kvEvaluated = 0
            Memory.clearCache()
            out.error("生成失败 (generation failed): \(error)")
        }
    }

    private func run(context: ModelContext, messages: [WireMessage], params p: WireParams)
        async throws
    {
        // 1. Chat template → token ids (images ride along inside the chat
        // messages; the VLM processor turns them into embeddings).
        let chat: [Chat.Message] = messages.map { m in
            let imgs: [UserInput.Image] = (m.images ?? []).map {
                .url(URL(fileURLWithPath: $0))
            }
            switch m.role {
            case "system": return .init(role: .system, content: m.content, images: imgs)
            case "assistant": return .init(role: .assistant, content: m.content, images: imgs)
            default: return .init(role: .user, content: m.content, images: imgs)
            }
        }
        let hasImages = messages.contains { !($0.images ?? []).isEmpty }
        var extra: [String: any Sendable] = [:]
        if let think = p.think, meta.thinkArg {
            extra["enable_thinking"] = think
        }
        // Native effort rung — only when the template declares the ladder and
        // thinking isn't off (the template rejects unknown values outright).
        if let effort = p.effort, meta.effortLevels.contains(effort), p.think != false {
            extra["reasoning_effort"] = effort
        }
        let userInput = UserInput(chat: chat, additionalContext: extra.isEmpty ? nil : extra)
        let lmInput = try await context.processor.prepare(input: userInput)
        var tokens = lmInput.text.tokens.asArray(Int32.self).map(Int.init)

        // Ordered image identities — must match the order the processor lays
        // the placeholder runs out (messages render in order, images within a
        // message in array order).
        let imageKeys: [String] = messages.flatMap { ($0.images ?? []).map(imageKey) }

        // End (exclusive) of the last image/video placeholder in the expanded
        // sequence. Everything up to here must be evaluated in ONE prepare()
        // call: getRopeIndex computes image-grid positions from the start of
        // whatever sequence it is handed, and any call carrying pixels resets
        // the rope state — so a media-bearing span is only position-correct
        // at cache offset 0.
        var lastMediaEnd = 0
        var segmented = !hasImages
        if hasImages, let imgId = meta.imageTokenId {
            let vidId = meta.videoTokenId ?? Int.min
            for (i, t) in tokens.enumerated() where t == imgId || t == vidId {
                lastMediaEnd = i + 1
            }
            segmented = lastMediaEnd > 0
        }

        // Models whose template lacks the enable_thinking kwarg (e.g.
        // Qwen3.5+) get the same treatment as the llama.cpp engine: pre-fill
        // an empty think block so the model skips reasoning. Appended tokens
        // land in the chunked text tail, after every image placeholder, so
        // the processor's image positions are untouched — only the legacy
        // whole-input path (exotic VLMs) must not grow the token list.
        if p.think == false, !meta.thinkArg, meta.supportsThinking, segmented {
            tokens += context.tokenizer.encode(
                text: "<think>\n\n</think>\n\n", addSpecialTokens: false)
        }

        // Qwen3.5-style templates open a `<think>` tag in the prompt when
        // thinking is enabled, so the model streams reasoning without the
        // opening tag. Emit it synthetically so the UI's think panel sees a
        // complete block (mirrors the llama.cpp engine's behaviour).
        if p.think != false {
            let tail = context.tokenizer.decode(
                tokenIds: Array(tokens.suffix(6)), skipSpecialTokens: false)
            if tail.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("<think>") {
                out.emit(["event": "token", "text": "<think>\n"])
            }
        }

        let total = tokens.count
        if nCtxCap > 0 && total >= nCtxCap {
            out.emit([
                "event": "done", "promptTokens": total, "completionTokens": 0,
                "tokensPerSecond": 0, "stopReason": "context",
            ])
            return
        }

        var gp = GenerateParameters(temperature: p.temperature ?? 0.7)
        gp.topP = p.topP ?? 1.0
        gp.topK = p.topK ?? 0
        gp.minP = p.minP ?? 0.0
        gp.seed = p.seed
        if let rp = p.repeatPenalty, rp > 1.0 { gp.repetitionPenalty = rp }
        if let mt = p.maxTokens, mt > 0 { gp.maxTokens = mt }

        if !segmented {
            // Exotic vision model without a known placeholder id: the
            // library's own loop drives the (image-aware) prefill in one
            // call — no progress events, no reuse.
            kvCache = nil
            try await self.libraryStream(
                context: context, input: lmInput,
                cache: context.model.newCache(parameters: nil), gp: gp, total: total)
            return
        }

        do {
            // 2. KV prefix reuse (text AND vision turns): trim the previous
            // cache back to the shared prefix, or start fresh when trimming
            // isn't supported (hybrid / recurrent attention can't rewind).
            var start = 0
            var state: LMOutput.State? = nil
            if let cached = kvCache, !kvTokens.isEmpty {
                var common = 0
                while common < min(kvTokens.count, tokens.count - 1),
                    kvTokens[common] == tokens[common]
                {
                    common += 1
                }
                // Media constraint: reuse only when every image of THIS
                // prompt sits inside the shared prefix and is the same file
                // the cache was built from. Anything else (new/changed image,
                // edited history around one) re-evaluates from scratch — a
                // pixels-bearing call restarts M-RoPE at zero, so a partial
                // resume below `lastMediaEnd` can never be position-correct.
                if imageKeys != kvImageKeys || common < lastMediaEnd {
                    common = 0
                }
                // How much sits in the cache comes from OUR ledger, never
                // from the caches themselves: a hybrid model's MambaCache
                // reports offset 0 forever, which the old arithmetic read as
                // "nothing to trim" and then appended the new conversation on
                // top of the previous one's KV — the model could see (and got
                // derailed by) another conversation's prompt.
                let excess = kvEvaluated - common
                if common == 0 || kvEvaluated < kvTokens.count {
                    kvCache = nil
                } else if excess > 0 {
                    if cached.allSatisfy({ $0.isTrimmable }) {
                        for c in cached { _ = c.trim(excess) }
                        kvEvaluated = common
                        start = common
                    } else {
                        kvCache = nil
                    }
                } else {
                    start = common
                }
                if start > 0 { state = kvState }
            }
            if kvCache == nil {
                kvCache = context.model.newCache(parameters: nil)
            }
            let warm = kvCache!

            // 3. Prefill everything but the last token, with progress events.
            // A media-bearing span is atomic (one prepare() call — the only
            // position-correct entry point, and it runs the vision tower);
            // all remaining text chunks through the public text entry point
            // (the same call shape as TokenIterator.step) with the M-RoPE
            // state threaded across calls so positions continue linearly.
            let prefillEnd = max(total - 1, start)
            let emitProgress =
                (prefillEnd - start) > Self.prefillEventThreshold || start < lastMediaEnd
            var pos = start
            var legacyWhole = false
            if pos < lastMediaEnd {
                // A resume point is never inside the media span, so this is a
                // fresh cache at offset 0 — exactly what prepare() needs.
                if cancelFlag.isSet {
                    self.finish(prompt: total, done: 0, tps: 0, reason: "cancelled", generated: nil)
                    return
                }
                if emitProgress {
                    out.emit(["event": "prefill", "processed": 0, "total": total])
                }
                let head = LMInput(
                    text: .init(
                        tokens: MLXArray(tokens[0..<lastMediaEnd].map(Int32.init))[.newAxis]),
                    image: lmInput.image, video: lmInput.video)
                switch try context.model.prepare(head, cache: warm, windowSize: nil) {
                case .logits(let headOut):
                    eval(headOut.logits)
                    state = headOut.state
                    pos = lastMediaEnd
                    if emitProgress {
                        out.emit(["event": "prefill", "processed": pos, "total": total])
                    }
                case .tokens:
                    // The model declined to consume the media in prepare —
                    // nothing was evaluated. Hand the whole prompt to the
                    // library loop instead (images would otherwise be lost).
                    legacyWhole = true
                }
            }
            if legacyWhole {
                kvCache = nil
                try await self.libraryStream(
                    context: context, input: lmInput, cache: warm, gp: gp, total: total)
                return
            }
            while pos < prefillEnd {
                if cancelFlag.isSet {
                    self.finish(
                        prompt: total, done: 0, tps: 0, reason: "cancelled", generated: nil)
                    return
                }
                let end = min(pos + Self.prefillChunk, prefillEnd)
                let chunkText = LMInput.Text(
                    tokens: MLXArray(tokens[pos..<end].map(Int32.init)))
                let result = withPreparedCache(warm, lengths: chunkText.sequenceLengths) {
                    context.model(chunkText[text: .newAxis], cache: warm, state: state)
                }
                eval(result.logits)
                state = result.state
                pos = end
                if emitProgress {
                    out.emit(["event": "prefill", "processed": pos, "total": total])
                }
            }

            // 4. Decode with the M-RoPE state threaded through every step.
            // The library's TokenIterator evaluates the final prompt token
            // and the first sampled one with a fresh rope state — positions
            // restart at zero, which full-attention M-RoPE models (Qwen3-VL)
            // answer with an instant EOS. Rolling our own loop keeps every
            // token at its true position.
            self.decode(
                context: context, cache: warm, tokens: tokens, total: total,
                state: state, gp: gp,
                recorded: Array(tokens.prefix(prefillEnd)), imageKeys: imageKeys,
                reused: start)
        }
    }

    /// The library's own generate loop — used only when the prompt cannot be
    /// segmented (exotic VLM without a known media placeholder id). No prefix
    /// reuse, no progress events (pre-v1.8.0 behaviour).
    private func libraryStream(
        context: ModelContext, input: LMInput, cache: [KVCache], gp: GenerateParameters,
        total: Int
    ) async throws {
        let stream = try MLXLMCommon.generate(
            input: input, cache: cache, parameters: gp, context: context)

        var reason = "eos"
        var info: GenerateCompletionInfo? = nil
        for await gen in stream {
            if cancelFlag.isSet {
                reason = "cancelled"
                break
            }
            switch gen {
            case .chunk(let text):
                out.emit(["event": "token", "text": text])
            case .info(let i):
                info = i
            default:
                break
            }
            // MambaCache never advances its offset — take the largest across
            // layers (the attention caches do track it).
            if nCtxCap > 0, (cache.map(\.offset).max() ?? 0) >= nCtxCap {
                reason = "context"
                break
            }
        }
        if reason == "eos", let i = info {
            switch i.stopReason {
            case .length: reason = "length"
            case .cancelled: reason = "cancelled"
            default: reason = "eos"
            }
        }
        self.finish(
            prompt: total,
            done: info?.generationTokenCount ?? 0,
            tps: info?.tokensPerSecond ?? 0,
            reason: reason,
            generated: nil)
    }

    /// Sampling + decode with per-step state threading. Mirrors the library's
    /// sampler/processor/detokenizer wiring; stop *sequences* stay Rust-side.
    private func decode(
        context: ModelContext, cache: [KVCache], tokens: [Int], total: Int,
        state initialState: LMOutput.State?, gp: GenerateParameters,
        recorded: [Int], imageKeys: [String], reused: Int
    ) {
        var state = initialState
        // The rope state saved for the next turn's resume: ropeDeltas is
        // constant for a given prompt layout, so the pre-decode value is the
        // right one to continue the recorded prefix from.
        let recordedState = initialState

        func stepEval(_ tok: Int) -> MLXArray {
            let t = LMInput.Text(tokens: MLXArray([Int32(tok)]))
            let r = withPreparedCache(cache, lengths: t.sequenceLengths) {
                context.model(t[text: .newAxis], cache: cache, state: state)
            }
            state = r.state
            return r.logits
        }

        let sampler = gp.sampler()
        var processor = gp.processor()
        processor?.prompt(MLXArray(tokens.map(Int32.init)))

        var stopIds = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { stopIds.insert(eos) }
        for t in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(t) { stopIds.insert(id) }
        }
        let unknownId = context.tokenizer.unknownTokenId

        func sample(_ logits: MLXArray) -> Int {
            var l = logits[0..., -1, 0...]
            if let processor { l = processor.process(logits: l) }
            let y = sampler.sample(logits: l)
            processor?.didSample(token: y)
            return y.item(Int.self)
        }

        var detok = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
        let started = Date()
        var done = 0
        var reason = "eos"
        let maxTokens = gp.maxTokens ?? Int.max

        // The final prompt token, evaluated with the threaded state so its
        // logits (which pick the first generated token) are position-true.
        var tok = sample(stepEval(tokens[total - 1]))
        while true {
            if cancelFlag.isSet {
                reason = "cancelled"
                break
            }
            if tok == unknownId || stopIds.contains(tok) { break }
            done += 1
            // Kick the next forward off, then detokenize/emit the current
            // token while the GPU runs it.
            let logits = stepEval(tok)
            asyncEval(logits)
            detok.append(token: tok)
            if let piece = detok.next() {
                out.emit(["event": "token", "text": piece])
            }
            if done >= maxTokens {
                reason = "length"
                break
            }
            // Context budget from our own ledger — hybrid caches misreport
            // offset (MambaCache never advances it), so `total + done` is the
            // only trustworthy population count.
            if nCtxCap > 0, total + done >= nCtxCap {
                reason = "context"
                break
            }
            tok = sample(logits)
        }
        let dt = max(Date().timeIntervalSince(started), 0.001)
        self.finish(
            prompt: total, done: done, tps: Double(done) / dt, reason: reason,
            generated: recorded, images: imageKeys, state: recordedState,
            evaluated: total + done, reused: reused)
    }

    private func finish(
        prompt: Int, done: Int, tps: Double, reason: String, generated: [Int]?,
        images: [String] = [], state: LMOutput.State? = nil, evaluated: Int = 0,
        reused: Int = 0
    ) {
        // Remember the prompt prefix this engine evaluated itself (the cache
        // also holds the decoded tokens past this point; next turn trims them
        // away and re-evaluates that text with clean positions).
        if let generated {
            kvTokens = generated
            kvImageKeys = images
            kvState = state
            kvEvaluated = evaluated
        } else {
            kvCache = nil
            kvTokens = []
            kvImageKeys = []
            kvState = nil
            kvEvaluated = 0
        }
        out.emit([
            "event": "done", "promptTokens": prompt, "completionTokens": done,
            "tokensPerSecond": tps, "stopReason": reason,
            // How many prompt tokens were resumed from the previous turn's
            // cache — observability for the cross-conversation-contamination
            // regression tests (0 = evaluated from scratch).
            "reused": reused,
        ])
    }
}

// MARK: - main loop

@main
struct ChatyMLX {
    static func main() async {
        // Keep MLX's buffer cache modest so an idle sidecar doesn't sit on
        // gigabytes of recycled GPU buffers.
        Memory.cacheLimit = 256 * 1024 * 1024

        let engine = Engine()
        out.emit(["event": "ready"])
        log("ready")

        var generation: Task<Void, Never>? = nil
        do {
            for try await line in FileHandle.standardInput.bytes.lines {
                guard !line.isEmpty, let data = line.data(using: .utf8) else { continue }
                let cmd: WireCmd
                do {
                    cmd = try JSONDecoder().decode(WireCmd.self, from: data)
                } catch {
                    out.error("无法解析命令 (bad command): \(error)")
                    continue
                }
                switch cmd.cmd {
                case "load":
                    guard let path = cmd.path else {
                        out.error("load 缺少 path (load requires path)")
                        continue
                    }
                    await engine.load(path: path, nCtx: cmd.nCtx)
                case "generate":
                    guard let messages = cmd.messages else {
                        out.error("generate 缺少 messages (generate requires messages)")
                        continue
                    }
                    let params = cmd.params ?? WireParams()
                    await generation?.value  // one at a time
                    generation = Task {
                        await engine.generate(messages: messages, params: params)
                    }
                case "cancel":
                    engine.cancelFlag.set()
                case "quit":
                    engine.cancelFlag.set()
                    await generation?.value
                    // _exit skips atexit/static destructors: MLX's Metal
                    // teardown segfaults when run from exit handlers (same
                    // class of crash as the app's old tray-quit issue), and
                    // the OS reclaims everything anyway.
                    _exit(0)
                default:
                    out.error("未知命令 (unknown command): \(cmd.cmd)")
                }
            }
        } catch {
            log("stdin error: \(error)")
        }
        // stdin closed — parent is gone; don't outlive it. (_exit: see above.)
        engine.cancelFlag.set()
        await generation?.value
        _exit(0)
    }
}
