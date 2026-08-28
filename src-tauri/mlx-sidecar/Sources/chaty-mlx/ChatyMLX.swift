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
    var content: String
    /// The turn's thinking, kept out of `content`. Some templates (Qwen3.8)
    /// read reasoning ONLY from this field and never split it back out of the
    /// content, so a turn stored inline reaches them as an empty thought
    /// followed by its own markup — and the prompt stops reproducing what the
    /// model generated. Sent only where the template is probed to use it.
    let reasoningContent: String?
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
    /// A layout learned from the template and proven against its own token ids,
    /// used in place of it when the template cannot reproduce its own live
    /// prompt from a stored turn. Nil means the template is already append-safe.
    /// One per thinking mode, because the switch does not live in one place:
    /// Qwen changes only what follows the last message, Gemma also changes the
    /// system turn (its template opens the thought channel there). A layout
    /// learned in one mode and reused in the other silently disables the
    /// switch — which is exactly what shipping a single one did.
    var layoutOff: LearnedLayout?
    var layoutOn: LearnedLayout?
    /// The same layouts, kept even when the template is good enough to render
    /// with. Rendering and anchoring are different jobs: a well-behaved
    /// template is rendered by itself, but replaying a turn's recorded ids
    /// still needs to know where in the token stream that turn begins and
    /// ends, and only the layout knows the role markers.
    var anchorOff: LearnedLayout?
    var anchorOn: LearnedLayout?

    func layout(thinking: Bool) -> LearnedLayout? { thinking ? layoutOn : layoutOff }
    func anchor(thinking: Bool) -> LearnedLayout? { thinking ? anchorOn : anchorOff }

    /// The block a stored assistant turn must carry when thinking is off —
    /// probed against the real template at load, never named. See
    /// `probeTurnPrefix`.
    var turnPrefix = ""
    /// Native reasoning-effort ladder the template accepts, weakest first
    /// (Qwen3.8: low/medium/xhigh). Empty ⇒ no effort control.
    var effortLevels: [String] = []
    var toolRole = false
    var reasoningField = false
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

/// Jinja comments carry no output, and their `-` markers are supposed to eat
/// the whitespace on that side. The Swift Jinja engine strips the comment but
/// NOT the marked whitespace, so a template written as
///
///     …<turn|>\n{%- endif %}
///     {#- pre-scan -#}
///
/// renders one newline MORE than the reference engine does — and that single
/// stray `\n\n` token (Gemma 4's system/user boundary) is enough to shift the
/// model's behaviour measurably: same weights, same task, the reference
/// runtime writes `.card { }` while we wrote `card { }` on every CSS rule.
///
/// Applying the comment semantics ourselves — delete each comment together
/// with the whitespace its markers claim — makes the rendered prompt
/// byte-identical to the reference. Comments produce no output either way, so
/// a correct engine is unaffected by the rewrite. The original is kept beside
/// it once, so the edit is auditable and reversible.
/// Strip jinja comments from a template that lives in its own file.
/// swift-transformers prefers `chat_template.jinja`, then `chat_template.json`;
/// returns true once one of them supplied the template, so the caller knows
/// the copy inside tokenizer_config.json is not the one being rendered.
func healJinjaTemplateFile(dir: URL) -> Bool {
    let jinja = dir.appendingPathComponent("chat_template.jinja")
    if FileManager.default.fileExists(atPath: jinja.path) {
        if let raw = try? String(contentsOf: jinja, encoding: .utf8) {
            if let healed = strippedJinjaComments(raw) {
                backUpOriginal(jinja, Data(raw.utf8))
                if (try? healed.write(to: jinja, atomically: true, encoding: .utf8)) != nil {
                    log(
                        "healed chat template: applied jinja comment whitespace semantics (\(raw.count) → \(healed.count) chars)"
                    )
                }
            }
            return true
        }
    }
    let json = dir.appendingPathComponent("chat_template.json")
    guard let raw = try? Data(contentsOf: json),
        var obj = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any],
        let template = obj["chat_template"] as? String
    else { return false }
    guard let healed = strippedJinjaComments(template) else { return true }
    obj["chat_template"] = healed
    if let out = try? JSONSerialization.data(
        withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    {
        backUpOriginal(json, raw)
        if (try? out.write(to: json, options: .atomic)) != nil {
            log(
                "healed chat template in chat_template.json: applied jinja comment whitespace semantics (\(template.count) → \(healed.count) chars)"
            )
        }
    }
    return true
}

/// Delete every jinja comment together with the whitespace its `-` markers
/// claim. Returns nil when the template has nothing to strip.
func strippedJinjaComments(_ raw: String) -> String? {
    guard raw.contains("{#") else { return nil }
    // Most specific first: both markers, then each single marker, then plain.
    let rules = [
        "[ \\t]*\\r?\\n?[ \\t]*\\{#-[\\s\\S]*?-#\\}[ \\t]*\\r?\\n?[ \\t]*",
        "[ \\t]*\\r?\\n?[ \\t]*\\{#-[\\s\\S]*?#\\}",
        "\\{#[\\s\\S]*?-#\\}[ \\t]*\\r?\\n?[ \\t]*",
        "\\{#[\\s\\S]*?#\\}",
    ]
    var healed = raw
    for pattern in rules {
        guard let re = try? NSRegularExpression(pattern: pattern) else { continue }
        healed = re.stringByReplacingMatches(
            in: healed, range: NSRange(healed.startIndex..., in: healed), withTemplate: "")
    }
    return healed == raw ? nil : healed
}

/// Snapshot the file about to be rewritten. A heal only ever fires on a file
/// it has not already healed — its own output never triggers it again — so
/// whatever sits there now *is* the pristine original, including after the
/// model is re-downloaded over an earlier heal. Overwrite unconditionally:
/// keeping the first snapshot forever would leave a `.chaty-orig` describing a
/// copy of the model that is no longer on disk. Exactly one heal writes any
/// given file, so this never captures another heal's intermediate state.
func backUpOriginal(_ url: URL, _ data: Data) {
    try? data.write(to: URL(fileURLWithPath: url.path + ".chaty-orig"), options: .atomic)
}

/// One read-modify-write over tokenizer_config.json, covering both repairs it
/// can need, so the file is snapshotted once and the backup is never a
/// half-healed intermediate.
///
/// swift-transformers still defaults `clean_up_tokenization_spaces` to **true**
/// when a tokenizer config omits it, while transformers/mlx-lm default it to
/// false. That legacy BERT-era heuristic rewrites `" ."` → `"."` (and the same
/// for `,` `!` `?`) inside the *decoder*, which mangles code: a CSS rule
/// indented with eight spaces decodes with seven, and the streaming
/// detokenizer that measured new text by character count swallowed the
/// selector's leading `.` outright. Write the modern default in when the
/// config is silent; an explicit setting is left alone.
///
/// `healTemplate` is false when a sibling file already supplied the chat
/// template — then the copy in here is not the one being rendered, and
/// rewriting it would only churn the file.
func healTokenizerConfig(dir: URL, healTemplate: Bool) {
    let cfg = dir.appendingPathComponent("tokenizer_config.json")
    guard let raw = try? Data(contentsOf: cfg),
        var obj = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any]
    else { return }

    var notes: [String] = []
    if obj["clean_up_tokenization_spaces"] == nil {
        obj["clean_up_tokenization_spaces"] = false
        notes.append("clean_up_tokenization_spaces=false (was absent)")
    }
    if healTemplate, let template = obj["chat_template"] as? String,
        let healed = strippedJinjaComments(template)
    {
        obj["chat_template"] = healed
        notes.append(
            "jinja comment whitespace semantics (\(template.count) → \(healed.count) chars)")
    }
    guard !notes.isEmpty,
        let out = try? JSONSerialization.data(
            withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    else { return }
    backUpOriginal(cfg, raw)
    guard (try? out.write(to: cfg, options: .atomic)) != nil else { return }
    log("healed tokenizer config: " + notes.joined(separator: "; "))
}

/// Streaming detokenizer that only ever emits *verified* new text.
///
/// `NaiveStreamingDetokenizer` derives each piece as
/// `newSegment.suffix(newSegment.count - segment.count)` — a character-count
/// delta that assumes re-decoding a longer run of tokens can only append. A
/// decoder that rewrites earlier characters breaks that assumption silently:
/// the piece then starts at the wrong offset and real characters vanish from
/// the model's output (see `healTokenizerConfig` — one eaten space cost every
/// CSS selector its leading `.`). Taking the delta from the verified common
/// prefix instead makes a rewrite cost at most a repeated tail, never a lost
/// character.
struct SafeStreamingDetokenizer {
    private let tokenizer: any MLXLMCommon.Tokenizer
    private var segmentTokens: [Int] = []
    private var segment = ""

    init(tokenizer: any MLXLMCommon.Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    /// Restart the window at the token just emitted, exactly like the library
    /// detokenizer: a bounded segment keeps re-decoding cheap.
    private mutating func startNewSegment() {
        guard let last = segmentTokens.last else {
            segmentTokens.removeAll()
            segment = ""
            return
        }
        segmentTokens = [last]
        segment = tokenizer.decode(tokenIds: segmentTokens)
    }

    mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokenIds: segmentTokens)
        // A byte-fallback character split across tokens decodes to U+FFFD
        // until its last byte arrives — hold the piece back rather than
        // emitting a replacement character.
        if newSegment.last == "\u{fffd}" { return nil }
        let new: String
        if newSegment.hasPrefix(segment) {
            new = String(newSegment.dropFirst(segment.count))
        } else {
            // The decoder rewrote text already streamed out. Nothing can be
            // retracted, so re-anchor on the longest common prefix and emit
            // the rest: worst case a character repeats, none is dropped.
            let common = zip(newSegment, segment).prefix { $0 == $1 }.count
            new = String(newSegment.dropFirst(common))
        }
        if new.isEmpty { return nil }
        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            segment = newSegment
        }
        return new
    }
}

/// How a turn must be written down for the next prompt to be an APPEND onto
/// the last one — the property the KV cache actually needs. Round two has to
/// BEGIN with round one's prompt followed by exactly the text the model
/// generated on top of it; anything less and the prefix dies at the first
/// assistant turn, which a model whose memory cannot rewind answers by
/// re-reading the whole conversation, every step.
///
/// Two dials, because templates disagree on both:
///
/// - **Who speaks a tool result.** A template decides "does this turn still
///   belong to the request being answered" from the index of the last *user*
///   message, so a result wearing that role can push every assistant turn into
///   "already answered" and get its reasoning dropped.
/// - **Where a turn's thinking lives.** Some templates split it back out of
///   `content`; others (Qwen3.8) read it only from a `reasoning_content` field
///   and render an empty thought for anything stored inline.
///
/// Probed rather than assumed, and ordered so the least invasive shape that
/// works is the one adopted.
struct HistoryShape {
    var toolRole = false
    var reasoningField = false
}

/// Does the template still render earlier turns the same way once a NEW user
/// question arrives? Qwen's does not — history loses its reasoning block the
/// moment it stops being the current query — and that is what costs a full
/// prefill on the first step of every follow-up.
/// `block` is whatever the template writes after the assistant header for the
/// turn being produced — `<think></think>` for Qwen, a thought channel for
/// Gemma. Probing with it is what makes this work for any markup: a template
/// that keeps history intact renders a stored turn carrying that block exactly
/// as it rendered the live one, and a template that strips reasoning out of
/// history does not.
func templateSurvivesFollowUp(
    _ tokenizer: any MLXLMCommon.Tokenizer, extra: [String: any Sendable]?, block: String
) -> Bool {
    func render(_ msgs: [[String: any Sendable]]) -> String? {
        guard
            let ids = try? tokenizer.applyChatTemplate(
                messages: msgs, tools: nil, additionalContext: extra)
        else { return nil }
        return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
    }
    let turn: [[String: any Sendable]] = [
        ["role": "user", "content": "q1"],
        ["role": "assistant", "content": block + "PROBE_ANS"],
    ]
    guard let first = render(turn),
        let second = render(turn + [["role": "user", "content": "q2"]])
    else { return true }
    // The probe's own answer missing from the render means the template threw
    // the turn away rather than rendering it — Gemma does exactly that to a
    // reasoning block it considers unterminated. That is the opposite of
    // surviving, and reading it as "nothing to see" is how this came back clean
    // for a model whose cache died on every follow-up.
    guard let cut = first.range(of: "PROBE_ANS", options: .backwards) else { return false }
    return second.hasPrefix(String(first[first.startIndex ..< cut.upperBound]))
}

/// What a thinking-off prompt writes after the assistant header — the block the
/// model continues from, and therefore the block a STORED assistant turn has to
/// carry for the next prompt to be an append of this one. Without it the common
/// prefix ends at the header and the whole conversation is re-read every turn.
///
/// Discovered from the template rather than named, because the block is not one
/// string: Qwen writes `<think>\n\n</think>\n\n`, Gemma writes
/// `<|channel>thought\n<channel|>`, and a template that writes nothing must get
/// nothing added. The answer is then VERIFIED by rendering a turn that carries
/// it — templates that refuse to re-emit the block for historical turns (Qwen's
/// `last_query_index` gate) strip whatever the content holds, and for those the
/// honest answer is the empty string.
/// A conversation rendered one message at a time, learned from the template.
///
/// Qwen's template decides how to render an assistant turn from WHERE it sits:
/// everything before the newest user question is history and loses its reasoning
/// block. So the moment a follow-up question arrives, every earlier turn renders
/// differently and the prompt stops being an append of the last one — a 7856
/// character transcript diverging 161 characters in, which costs a full prefill
/// on every new question. llama.cpp never had this: it renders each message
/// verbatim through its own ChatML fallback, which is what ships today for
/// these same models and what makes its cross-turn reuse 99%.
///
/// This learns the per-role scaffolding from the template's own output and then
/// lays messages out with it, so position stops mattering. It is only trusted
/// when it reproduces the template's own token ids exactly (see `learn`), and
/// only used when the template has already been shown to break the invariant —
/// the alternative there is a guaranteed full re-prefill.
struct LearnedLayout {
    var head = ""
    var open: [String: String] = [:]
    var close: [String: String] = [:]
    var genTail = ""

    /// The part of the generation tail that sits AFTER the assistant header —
    /// the reasoning block the model continues from. A stored turn has to carry
    /// it, or the next prompt stops being an append; `turnPrefix` is what puts
    /// it there.
    func turnPrefix() -> String {
        guard let head = open["assistant"], genTail.hasPrefix(head) else { return "" }
        return String(genTail.dropFirst(head.count))
    }

    func render(_ msgs: [(role: String, content: String)]) -> String {
        // A template may open the prompt differently depending on whether a
        // system message starts it — Gemma emits a system block for thinking
        // even when there is nothing to put in it — so the system opening
        // carries everything before its own content and the plain head is used
        // only when no system message leads.
        var out = msgs.first?.role == "system" ? "" : head
        for m in msgs {
            out += (open[m.role] ?? open["user"] ?? "") + m.content
            out += (close[m.role] ?? close["user"] ?? "")
        }
        return out + genTail
    }

    /// Learn the layout, then prove it: the learned rendering of a probe
    /// conversation must tokenize to exactly what the template produces for the
    /// same conversation. Anything less and this returns nil, because a prompt
    /// that merely looks right is a prompt the model was not trained on.
    static func learn(
        _ tokenizer: any MLXLMCommon.Tokenizer, extra: [String: any Sendable]?
    ) -> LearnedLayout? {
        func render(_ msgs: [[String: any Sendable]]) -> String? {
            guard
                let ids = try? tokenizer.applyChatTemplate(
                    messages: msgs, tools: nil, additionalContext: extra)
            else { return nil }
            return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        }
        func msg(_ role: String, _ c: String) -> [String: any Sendable] {
            ["role": role, "content": c]
        }
        let u1 = "PROBE_U1"
        // Two same-role messages isolate one message's scaffolding: what the
        // second render has beyond the first, minus the generation tail, is
        // exactly one rendered message.
        guard let r1 = render([msg("user", u1)]),
            let r2 = render([msg("user", u1), msg("user", "PROBE_U2")])
        else { return nil }
        let shared = String(r1.prefix(zip(r1, r2).prefix { $0 == $1 }.count))
        let genTail = String(r1.dropFirst(shared.count))
        var seg = String(r2.dropFirst(shared.count))
        seg.removeLast(genTail.count)
        guard let hit = seg.range(of: "PROBE_U2") else { return nil }

        var out = LearnedLayout()
        out.genTail = genTail
        out.open["user"] = String(seg[seg.startIndex ..< hit.lowerBound])
        out.close["user"] = String(seg[hit.upperBound...])
        // `shared` is the head plus the first user message laid out the same way.
        let firstUser = out.open["user"]! + u1 + out.close["user"]!
        out.head = String(shared.dropLast(firstUser.count))

        // The other roles. An assistant turn is read from a HISTORY position —
        // one with a user message after it — because that is where the template
        // shows the bare scaffolding. Read from the live position it would come
        // back wrapped in a reasoning block, and a stored turn that carries its
        // own would then be rendered inside a second one. The block belongs to
        // the content, added by `turnPrefix`, exactly as it is on llama.cpp.
        for role in ["assistant", "tool"] {
            let mark = "PROBE_\(role.uppercased())"
            guard
                let r = render([msg("user", u1), msg(role, mark), msg("user", "PROBE_TAILQ")])
            else { continue }
            guard let h = r.range(of: mark), let nextQ = r.range(of: "PROBE_TAILQ") else { continue }
            // Everything from the end of the first user message to the marker.
            let afterFirst = shared.count
            guard r.count > afterFirst, r.index(r.startIndex, offsetBy: afterFirst) <= h.lowerBound
            else { continue }
            out.open[role] = String(r[r.index(r.startIndex, offsetBy: afterFirst) ..< h.lowerBound])
            // …and from the marker to wherever the next message begins.
            let afterMark = String(r[h.upperBound ..< nextQ.lowerBound])
            guard let userOpen = out.open["user"], afterMark.hasSuffix(userOpen) else { continue }
            out.close[role] = String(afterMark.dropLast(userOpen.count))
        }
        if let r = render([msg("system", "PROBE_SYS"), msg("user", u1)]),
            let h = r.range(of: "PROBE_SYS")
        {
            out.open["system"] = String(r[r.startIndex ..< h.lowerBound])
            if let uOpen = out.open["user"],
                let uPos = r.range(of: u1),
                case let between = String(r[h.upperBound ..< uPos.lowerBound]),
                between.hasSuffix(uOpen)
            {
                out.close["system"] = String(between.dropLast(uOpen.count))
            }
        }

        // Proof. Not "reproduce the template everywhere" — the whole point is to
        // stop it rewriting history — but "reproduce it wherever it is not
        // rewriting anything". Both shapes below are ones the template renders
        // one way only, and the ids must match, since markup that merely looks
        // right can still tokenize differently and miss the cache.
        func matches(_ conv: [(role: String, content: String)]) -> Bool {
            guard
                let expected = try? tokenizer.applyChatTemplate(
                    messages: conv.map { msg($0.role, $0.content) }, tools: nil,
                    additionalContext: extra)
            else { return false }
            return tokenizer.encode(text: out.render(conv), addSpecialTokens: false) == expected
        }
        let m1 = matches([("system", "PROBE_S"), ("user", "PROBE_A")])
        let m2 = matches([("user", "PROBE_A"), ("assistant", "PROBE_B"), ("user", "PROBE_C")])
        if !m1 || !m2 {
            if let e = try? tokenizer.applyChatTemplate(
                messages: [msg("user", "PROBE_A"), msg("assistant", "PROBE_B"), msg("user", "PROBE_C")],
                tools: nil, additionalContext: extra)
            {
            }
            if let e = try? tokenizer.applyChatTemplate(
                messages: [msg("system", "PROBE_S"), msg("user", "PROBE_A")],
                tools: nil, additionalContext: extra)
            {
            }
            return nil
        }

        return out
    }
}

/// Does a stored assistant turn keep the reasoning it was written with?
///
/// This is the question that decides whether a template can reproduce its own
/// live prompt, and asking it any other way misses cases. Reading only the
/// generation tail does: with thinking ON Gemma's tail is a bare assistant
/// header — the model opens its own channel — so there is no block to probe
/// with, the check waved it through, and every follow-up re-read the
/// conversation anyway because the template drops the channel from history.
///
/// `emptyBlock` is the same span the template writes when thinking is OFF, which
/// is where the markup is legible: a complete, empty reasoning span. Splitting
/// it at its closing tag gives the opener and closer this model reasons in —
/// `<think>…</think>` for Qwen, a thought channel for Gemma — and a turn built
/// from them is the shape a real one has.
func templateKeepsStoredReasoning(
    _ tokenizer: any MLXLMCommon.Tokenizer, extra: [String: any Sendable]?, emptyBlock: String
) -> Bool {
    guard let close = emptyBlock.range(of: "<", options: .backwards),
        close.lowerBound > emptyBlock.startIndex
    else { return true }
    let opener = String(emptyBlock[emptyBlock.startIndex ..< close.lowerBound])
    let closer = String(emptyBlock[close.lowerBound...])
    let stored = opener + "PROBE_REASON" + closer + "PROBE_ANS"
    guard
        let ids = try? tokenizer.applyChatTemplate(
            messages: [
                ["role": "user", "content": "q1"],
                ["role": "assistant", "content": stored],
                ["role": "user", "content": "q2"],
            ], tools: nil, additionalContext: extra)
    else { return true }
    return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false).contains("PROBE_REASON")
}

func probeTurnPrefix(
    _ tokenizer: any MLXLMCommon.Tokenizer, extra: [String: any Sendable]?,
    appendsOwnBlock: Bool
) -> String {
    func render(_ msgs: [[String: any Sendable]]) -> String? {
        guard
            let ids = try? tokenizer.applyChatTemplate(
                messages: msgs, tools: nil, additionalContext: extra)
        else { return nil }
        let text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        // A template without the kwarg never writes a block; the sidecar adds
        // one after rendering, so the probe has to model that same step.
        return appendsOwnBlock ? text + Engine.thinkOffPrefix : text
    }
    let opening: [[String: any Sendable]] = [["role": "user", "content": "q"]]
    let follow: [[String: any Sendable]] = [["role": "user", "content": "q2"]]
    let marker = "PROBE_ANSWER"
    guard let first = render(opening),
        let stored = render(opening + [["role": "assistant", "content": marker]] + follow)
    else { return "" }
    // Where the live prompt and the stored rendering part company is the header;
    // everything the live prompt has past that point is the block.
    let common = zip(first, stored).prefix { $0 == $1 }.count
    let prefix = String(first.dropFirst(common))
    guard !prefix.isEmpty else { return "" }
    guard
        let verify = render(
            opening + [["role": "assistant", "content": prefix + marker]] + follow),
        verify.hasPrefix(first + marker)
    else { return "" }
    return prefix
}

func probeHistoryShape(_ tokenizer: any MLXLMCommon.Tokenizer) -> HistoryShape {
    let reasoning = "PROBE_REASONING"
    let answer = "PROBE_ANSWER"
    let reasoned = "\(reasoning)\n</think>\n\n\(answer)"
    // Render the way generation actually will — a template asked without
    // `enable_thinking` takes its no-reasoning branch, and the probe would then
    // be measuring a prompt shape that never occurs.
    func render(_ messages: [[String: any Sendable]]) -> String? {
        guard
            let ids = try? tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: ["enable_thinking": true])
        else { return nil }
        return tokenizer.decode(tokenIds: ids)
    }
    let opening: [[String: any Sendable]] = [
        ["role": "system", "content": "s"],
        ["role": "user", "content": "q"],
    ]
    guard let first = render(opening) else { return HistoryShape() }
    // The stored turn always carries the opening tag; what the model *generated*
    // does not when the template pre-opened it. Both conventions exist (Qwen3.5
    // pre-opens, Qwen3 emits the tag itself).
    let inlineContent = "<think>\n" + reasoned
    let generated =
        first.hasSuffix("<think>\n") || first.hasSuffix("<think>") ? reasoned : inlineContent

    func appends(toolRole: Bool, reasoningField: Bool) -> Bool {
        var turn: [String: any Sendable] = ["role": "assistant"]
        if reasoningField {
            turn["content"] = answer
            turn["reasoning_content"] = reasoning
        } else {
            turn["content"] = inlineContent
        }
        let second =
            opening + [turn, ["role": toolRole ? "tool" : "user", "content": "result"]]
        guard let p = render(second) else { return false }
        return p.hasPrefix(first + generated)
    }
    // Least invasive first: today's shape, then one dial, then both.
    for shape in [
        HistoryShape(toolRole: false, reasoningField: false),
        HistoryShape(toolRole: true, reasoningField: false),
        HistoryShape(toolRole: false, reasoningField: true),
        HistoryShape(toolRole: true, reasoningField: true),
    ] where appends(toolRole: shape.toolRole, reasoningField: shape.reasoningField) {
        return shape
    }
    return HistoryShape()
}

/// Cheap identity for an image file (path + size + mtime) — mirrors the GGUF
/// engine's `image_cache_key`.
/// First index at which `needle` occurs in `hay` at or after `from`.
func firstIndex(of needle: [Int], in hay: [Int], from: Int) -> Int? {
    guard !needle.isEmpty, hay.count >= needle.count else { return nil }
    var i = max(0, from)
    while i + needle.count <= hay.count {
        if Array(hay[i ..< i + needle.count]) == needle { return i }
        i += 1
    }
    return nil
}

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
    /// The exact tokens a turn this engine generated occupies in the KV — the
    /// reasoning block the prompt ended with, then the ids the model produced.
    ///
    /// Keyed by the text the app will store for that turn. Re-encoding that
    /// text is NOT the inverse of generating it: the model emits tokens one at
    /// a time, and encoding the finished string is greedy, so the two disagree
    /// wherever a word could be split more than one way. Measured on
    /// Qwen3.5 2B, the word "Chaty" came back as [1106, 48289] from the model
    /// and [15213, 88] from the encoder — one token pair into the previous
    /// turn, the prefix was gone, and a model whose cache cannot be rewound
    /// discarded all of it. Replaying what the model actually produced removes
    /// the whole class.
    var turnIds: [String: [Int]] = [:]
    /// Bounded: a long session must not grow this without limit.
    var turnOrder: [String] = []
    /// The block the CURRENT prompt ends with, so the turn it produces can be
    /// recorded whole.
    var pendingBlock: [Int] = []

    static let prefillChunk = 512
    /// Only bother the UI with prefill events for prompts big enough to have
    /// a visible gap (mirrors the llama.cpp engine's behaviour).
    static let prefillEventThreshold = 256

    func load(path: String, nCtx: Int?) async {
        let dir = URL(fileURLWithPath: path)
        var meta = inspectModelDir(dir)
        var loadWarning: String?
        healTokenizerConfig(dir: dir, healTemplate: !healJinjaTemplateFile(dir: dir))
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
            // Ask the loaded template itself, rather than guessing from a name.
            let shape = await container.perform { ctx in probeHistoryShape(ctx.tokenizer) }
            meta.toolRole = shape.toolRole
            meta.reasoningField = shape.reasoningField
            if meta.supportsThinking {
                let thinkArg = meta.thinkArg
                let appendsOwn = !thinkArg
                meta.turnPrefix = await container.perform { ctx in
                    probeTurnPrefix(
                        ctx.tokenizer,
                        extra: thinkArg ? ["enable_thinking": false] : nil,
                        appendsOwnBlock: appendsOwn)
                }
            }
            // Only worth replacing a template that has been shown to rewrite
            // history as the conversation grows, and only with a layout that
            // reproduces its own token ids.
            let thinkArgForLayout = meta.thinkArg
            // Each mode is learned and judged on its own. If one cannot be
            // proven, that mode keeps using the template — a slower prompt is
            // always better than a prompt that quietly ignores the switch.
            // The thinking-OFF span, where this model's reasoning markup is
            // legible even when the thinking-ON tail says nothing about it.
            let offBlock: String = await container.perform { ctx in
                let e: [String: any Sendable]? =
                    thinkArgForLayout ? ["enable_thinking": false] : nil
                return LearnedLayout.learn(ctx.tokenizer, extra: e)?.turnPrefix() ?? ""
            }
            let learnFor: @Sendable (ModelContext, Bool) -> (LearnedLayout?, LearnedLayout?) = {
                ctx, thinking in
                let extra: [String: any Sendable]? =
                    thinkArgForLayout ? ["enable_thinking": thinking] : nil
                guard let learned = LearnedLayout.learn(ctx.tokenizer, extra: extra) else {
                    return (nil, nil)
                }
                let block = learned.turnPrefix()
                // Two ways a template stops the next prompt being an append, and
                // both end here. Qwen rewrites history once a follow-up question
                // arrives. Gemma keeps history stable but strips the thought
                // channel out of a stored turn, so the block the live prompt
                // ended with can never be put back — which is what
                // `probeTurnPrefix` comes back empty-handed about. The layout
                // answers both: it lays every message out the same way wherever
                // it sits.
                let stable = templateSurvivesFollowUp(
                    ctx.tokenizer, extra: extra, block: block)
                let reproducible =
                    (block.isEmpty
                        || !probeTurnPrefix(
                            ctx.tokenizer, extra: extra,
                            appendsOwnBlock: !thinkArgForLayout
                        ).isEmpty)
                    && templateKeepsStoredReasoning(
                        ctx.tokenizer, extra: extra, emptyBlock: offBlock)
                return (stable && reproducible ? nil : learned, learned)
            }
            ((meta.layoutOff, meta.anchorOff), (meta.layoutOn, meta.anchorOn)) =
                await container.perform { ctx in
                    (learnFor(ctx, false), learnFor(ctx, true))
                }
            log(
                "layout: thinking-off "
                    + (meta.layoutOff == nil ? "template" : "learned")
                    + ", thinking-on "
                    + (meta.layoutOn == nil ? "template" : "learned"))
            self.meta = meta
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
                "toolRole": meta.toolRole,
                "reasoningField": meta.reasoningField,
                // VLM-factory models have their vision tower loaded and
                // ready — no separate encoder file like GGUF's mmproj.
                "multimodal": meta.multimodal,
                // Can one prompt carry SEVERAL pictures? Gemma-4's MLX
                // implementation cannot: with three images it encodes one and
                // then refuses the mismatch —
                // `imageTokenCountMismatch(expectedVisionTokens: 280,
                // actualPromptTokens: 840)` — and the whole round fails. A tall
                // page screenshot arrives as several tiles, so every one of them
                // failed, and the retries behind each failure are what made a
                // browsing session look like it re-read everything from scratch.
                // Qwen3.5/3.6 take three without complaint.
                "multiImage": (meta.arch ?? "").lowercased() != "gemma4",
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


    /// The empty reasoning block prefilled when thinking is off. Prompt-side and

    /// stored-turn-side must stay the same string or the KV prefix breaks.

    static let thinkOffPrefix = "<think>\n\n</think>\n\n"


    private func run(context: ModelContext, messages: [WireMessage], params p: WireParams)
        async throws
    {
        // The empty reasoning block prefilled when thinking is off (below) has
        // to sit in front of every STORED assistant turn too, or the next
        // prompt is not an append of the last one: round one ends with
        // `…assistant\n<think>\n\n</think>\n\n` and the model continues from
        // there, while round two would render that same turn without it. The
        // common prefix then ends at the assistant header and every turn after
        // the first re-prefills from scratch — measured at 0% KV reuse on the
        // llama.cpp side before the matching fix, with thinking off, which is
        // the default in code mode. A turn that already opens with a reasoning
        // block is left alone; doubling it breaks the prefix just as badly.
        let thinking = p.think != false
        let messages: [WireMessage] = {
            // With thinking on the block is the OPENING of a reasoning trace,
            // and a stored turn needs it just as much: the prompt it was
            // generated from ended with it, so a turn recorded without it is
            // not what the model continued from. Only a turn that already
            // carries one is left alone — doubling it breaks the prefix just as
            // badly as omitting it.
            let prefix =
                meta.layout(thinking: thinking).map { $0.turnPrefix() }
                ?? (thinking ? "" : meta.turnPrefix)
            guard !prefix.isEmpty else { return messages }
            return messages.map { m in
                guard m.role == "assistant",
                    !m.content.hasPrefix(prefix),
                    !m.content.trimmingCharacters(in: .whitespaces).hasPrefix("<think>"),
                    (m.reasoningContent ?? "").isEmpty
                else { return m }
                var copy = m
                copy.content = prefix + m.content
                return copy
            }
        }()

        // 1. Chat template → token ids (images ride along inside the chat
        // messages; the VLM processor turns them into embeddings).
        let chat: [Chat.Message] = messages.map { m in
            let imgs: [UserInput.Image] = (m.images ?? []).map {
                .url(URL(fileURLWithPath: $0))
            }
            switch m.role {
            case "system": return .init(role: .system, content: m.content, images: imgs)
            case "assistant": return .init(role: .assistant, content: m.content, images: imgs)
            // A tool result speaks for itself. Folding it into the user role
            // (as the catch-all did) is what makes a template treat every
            // preceding assistant turn as belonging to an already-answered
            // request and drop its reasoning.
            case "tool": return .init(role: .tool, content: m.content, images: imgs)
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
        // The library's message generator emits role/content only, so a
        // structured `reasoning_content` never reaches the template through it.
        // Templates that read thinking ONLY from that field (Qwen3.8) would see
        // an empty thought and the turn's own markup as content — the prompt
        // then reproduces nothing and the prefix dies. Render the template
        // ourselves in that case; media still goes through the processor, which
        // is the only position-correct entry point for pixels.
        let carriesReasoning = messages.contains { !($0.reasoningContent ?? "").isEmpty }
        let lmInput: LMInput
        // Both thinking modes. Laying every message out the same way wherever it
        // sits means a turn's reasoning travels into history, which the official
        // templates strip.
        //
        // For Gemma that is a DELIBERATE departure — do not quietly restore it.
        // Google asks for two things: thoughts kept between the tool calls of
        // one model turn, and dropped from previous turns. Its template does
        // neither cleanly, stripping every model message unconditionally, which
        // breaks the tool-call half and made an agent step re-read its whole
        // transcript. Keeping the channel fixes that half and costs the other:
        // 5% cache reuse across four chat turns became 100%, and the owner ran
        // it and found no drop in answer quality. Compaction reclaims stale
        // reasoning when the window tightens — carrying the substance rather
        // than the raw trace, which is what Google suggests for long sessions.
        //
        // https://ai.google.dev/gemma/docs/capabilities/thinking
        if let layout = meta.layout(thinking: thinking), !hasImages {
            // Laid out one message at a time, so a follow-up question cannot
            // change how the turns before it render. Pixels still go through the
            // processor — it is the only position-correct entry point for them.
            let text = layout.render(messages.map { (role: $0.role, content: $0.content) })
            let ids = context.tokenizer.encode(text: text, addSpecialTokens: false)
            lmInput = LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init))))
        } else if carriesReasoning && !hasImages {
            let dicts: [[String: any Sendable]] = messages.map { m in
                var d: [String: any Sendable] = ["role": m.role, "content": m.content]
                if let r = m.reasoningContent, !r.isEmpty { d["reasoning_content"] = r }
                return d
            }
            let ids = try context.tokenizer.applyChatTemplate(
                messages: dicts, tools: nil, additionalContext: extra.isEmpty ? nil : extra)
            lmInput = LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init))))
        } else {
            lmInput = try await context.processor.prepare(input: userInput)
        }
        var tokens = lmInput.text.tokens.asArray(Int32.self).map(Int.init)

        // Every stored turn this engine generated, replayed as the tokens the
        // KV actually holds for it. Only `tokens` is rewritten — the library's
        // whole-input fallback still hands the model `lmInput` as the processor
        // built it, so an exotic VLM's path is untouched.
        //
        // Two things break the prompt's ability to reproduce the one before it,
        // and this removes both. The chat template rewrites a stored turn's
        // reasoning — Qwen's cuts everything up to `</think>` with thinking off
        // and writes an empty span in front of the real trace with it on — and
        // that is what the learned layout routes around for a text-only prompt.
        // Pixels cannot use the layout: they must go through the processor,
        // which renders with the template. And re-encoding the turn's text is
        // not the inverse of generating it, so even a faithful rendering can
        // land on different token boundaries. Replaying the recorded ids answers
        // both at once: no rendering to get wrong, nothing to re-encode.
        //
        // Every prompt, not only the ones carrying pictures. Where the cache
        // cannot rewind — a hybrid model's, which is not trimmable — a prompt
        // that stops matching one token early is a prompt that reuses nothing
        // at all, and a plain text turn drifts exactly the same way.
        //
        // Anchored on the role boundary — the layout keeps `<|im_end|>\n<|im_start|>`
        // in the previous role's close and only `assistant\n` in the assistant's
        // open, and those two occur inside ordinary prose. Joined, they cannot.
        //
        // Token-level and BEFORE lastMediaEnd is measured: this moves the image
        // placeholders, and everything downstream reads the final list.
        if !turnIds.isEmpty,
            let layout = meta.layout(thinking: thinking) ?? meta.anchor(thinking: thinking)
        {
            // The role header alone. A thinking template can fold a whole
            // empty thought block into what it writes after `assistant\n`,
            // and that block is not part of the header — it is the first
            // thing the turn's body has to replace, because the prompt this
            // turn was generated from wrote a DIFFERENT one there (an opened
            // thought, not a closed empty one).
            let fullOpener = layout.open["assistant"] ?? ""
            let opener =
                fullOpener.firstIndex(of: "\n").map { String(fullOpener[...$0]) } ?? fullOpener
            var markers: [[Int]] = []
            for c in Set(layout.close.values) {
                let ids = context.tokenizer.encode(text: c + opener, addSpecialTokens: false)
                if !ids.isEmpty { markers.append(ids) }
            }
            let closeIds = context.tokenizer.encode(
                text: layout.close["assistant"] ?? "", addSpecialTokens: false)
            let stored = messages.filter { $0.role == "assistant" }
            if !markers.isEmpty, !stored.isEmpty, !closeIds.isEmpty {
                // Where each turn's body starts, and where the next boundary is.
                var starts: [Int] = []
                var i = 0
                while i < tokens.count {
                    if markers.contains(where: { m in
                        i + 1 >= m.count && Array(tokens[(i + 1 - m.count) ... i]) == m
                    }) {
                        starts.append(i + 1)
                    }
                    i += 1
                }
                // Rebuild once, back to front, so earlier offsets stay valid.
                var out = tokens
                for (k, m) in stored.enumerated().reversed() {
                    guard k < starts.count else { continue }
                    // The body the app stored, minus any block it carries —
                    // the recorded ids already begin with one.
                    var body = m.content
                    let block = layout.turnPrefix()
                    if !block.isEmpty, body.hasPrefix(block) {
                        body = String(body.dropFirst(block.count))
                    }
                    let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
                    let split = (m.reasoningContent ?? "").trimmingCharacters(
                        in: .whitespacesAndNewlines)
                    guard
                        let recorded = turnIds[split.isEmpty ? body : split + "\u{0}" + trimmed]
                            ?? turnIds[body] ?? turnIds[trimmed],
                        !recorded.isEmpty
                    else {
                        if ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_REUSE"] != nil {
                            FileHandle.standardError.write(
                                ("mlx-replay MISS body=<<<\(body.prefix(40))>>>"
                                    + " splitTail=<<<\((m.reasoningContent ?? "nil").suffix(50))>>>"
                                    + " keyTails=" + turnIds.keys.map { "<<<\($0.suffix(50))>>>" }
                                    .joined(separator: " ") + "\n").data(using: .utf8)!)
                        }
                        continue
                    }
                    let from = starts[k]
                    // A turn ends where its OWN close begins — not at the next
                    // assistant boundary, which has the intervening user and
                    // tool messages between it and here, images included.
                    guard let to = firstIndex(of: closeIds, in: out, from: from), to >= from
                    else { continue }
                    if Array(out[from ..< to]) == recorded { continue }
                    // A generated turn holds no pictures. One that appears to
                    // is a mis-measured record, and splicing it in would add
                    // placeholders no pixels answer for.
                    if let img = meta.imageTokenId, recorded.contains(img) { continue }
                    if let vid = meta.videoTokenId, recorded.contains(vid) { continue }
                    out.replaceSubrange(from ..< to, with: recorded)
                }
                tokens = out
            }
            if ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_REUSE"] != nil {
                FileHandle.standardError.write(
                    ("mlx-replay: ran markers=\(markers.count) stored=\(stored.count)"
                        + " closeIds=\(closeIds.count) records=\(turnIds.count)\n")
                        .data(using: .utf8)!)
            }
        } else if ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_REUSE"] != nil {
            FileHandle.standardError.write(
                ("mlx-replay: SKIPPED records=\(turnIds.count)"
                    + " layout=\(meta.layout(thinking: thinking) != nil)"
                    + " anchor=\(meta.anchor(thinking: thinking) != nil)\n").data(using: .utf8)!)
        }

        // Diagnostic: what the model actually sees. Off unless asked for —
        // prompts can carry user content, so this never logs by default.
        if ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_PROMPT"] == "1" {
            log("PROMPT[\(tokens.count) tok]>>>" + context.tokenizer.decode(tokenIds: tokens) + "<<<END")
        }

        // Ordered image identities — must match the order the processor lays
        // the placeholder runs out (messages render in order, images within a
        // message in array order).
        let imagePaths: [String] = messages.flatMap { $0.images ?? [] }
        let imageKeys: [String] = imagePaths.map(imageKey)

        // The placeholder runs, in prompt order — one contiguous block per
        // image, matching `imageKeys` position for position. Knowing where
        // each image sits (not just where the last one ends) is what lets a
        // prompt that only ADDS an image resume from the cache below it.
        var mediaRuns: [(start: Int, end: Int)] = []
        /// One entry per picture from `perImageFrom` on, so each can be given
        /// to `prepare` by itself.
        var perImage: [LMInput.ProcessedImage]? = nil
        var perImageFrom = 0
        var lastMediaEnd = 0
        var segmented = !hasImages
        if hasImages, let imgId = meta.imageTokenId {
            let vidId = meta.videoTokenId ?? Int.min
            for (i, t) in tokens.enumerated() where t == imgId || t == vidId {
                if let last = mediaRuns.last, last.end == i {
                    mediaRuns[mediaRuns.count - 1].end = i + 1
                } else {
                    mediaRuns.append((start: i, end: i + 1))
                }
                lastMediaEnd = i + 1
            }
            segmented = lastMediaEnd > 0
        }

        // Models whose template lacks the enable_thinking kwarg (e.g.
        // Qwen3.5+) get the same treatment as the llama.cpp engine: pre-fill
        // an empty think block so the model skips reasoning. Appended tokens
        // land in the chunked text tail, after every image placeholder, so
        // the processor's image positions are untouched — only the legacy
        if p.think == false, !meta.thinkArg, meta.supportsThinking, segmented {
            tokens += context.tokenizer.encode(
                text: Self.thinkOffPrefix, addSpecialTokens: false)
        }

        // What this prompt ends with after the assistant header. Recorded with
        // the turn it is about to produce, so the NEXT prompt can replay the
        // pair instead of rendering and re-encoding it — see `turnIds`.
        pendingBlock = {
            // Measured off this very prompt rather than derived: whatever the
            // template wrote between the assistant header and here — an opened
            // thought, an empty one, nothing at all — is what the model
            // continued from, so it is what the KV holds and what the next
            // prompt has to put back.
            guard
                let layout = meta.layout(thinking: thinking) ?? meta.anchor(thinking: thinking)
            else { return [] }
            let fullOpener = layout.open["assistant"] ?? ""
            let opener =
                fullOpener.firstIndex(of: "\n").map { String(fullOpener[...$0]) } ?? fullOpener
            // The LAST header in the prompt, across every marker — not the
            // first one that happens to yield something. An empty block is a
            // real answer (thinking off leaves nothing between the header and
            // the model's first token), and treating it as "not found" sent
            // the search back to an EARLIER boundary, whose span reaches over
            // the messages in between — pictures included. Recorded, that
            // block replays those placeholders into the next prompt a second
            // time: featureTokenMismatch(expected: 2443, actual: 1463), which
            // is exactly two of three pictures counted twice.
            var bodyStart = -1
            for c in Set(layout.close.values) {
                let ids = context.tokenizer.encode(text: c + opener, addSpecialTokens: false)
                guard !ids.isEmpty, tokens.count >= ids.count else { continue }
                var i = tokens.count - ids.count
                while i >= 0 {
                    if Array(tokens[i ..< i + ids.count]) == ids {
                        bodyStart = max(bodyStart, i + ids.count)
                        break
                    }
                    i -= 1
                }
            }
            guard bodyStart >= 0 else { return [] }
            let head = Array(tokens[bodyStart...])
            // A block is what the template wrote before the model spoke. It
            // cannot contain a picture, and a recorded turn that did would
            // duplicate one wherever it is replayed.
            if let img = meta.imageTokenId, head.contains(img) { return [] }
            if let vid = meta.videoTokenId, head.contains(vid) { return [] }
            return head
        }()

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
                // A prefix that ends INSIDE a placeholder run leaves half a
                // picture in the cache — two images of different sizes at the
                // same offset diverge partway through their pads. Pull back to
                // where that run begins so the picture is wholly in or wholly
                // out.
                if let straddled = mediaRuns.first(where: {
                    $0.start < common && common < $0.end
                }) {
                    common = straddled.start
                }
                // Media constraint. The cache is about to be trimmed back to
                // the shared prefix, so the only images that survive into this
                // run are the ones whose placeholders sit inside it — and each
                // of those must be the same file the KV was built from. Images
                // above the prefix are trimmed away with everything else, and
                // images this prompt ADDS simply come after: the model reads
                // its positions from the cache's offset, so a span prepared on
                // a warm cache lands exactly where a full pass would put it.
                //
                // This is what llama.cpp's media cache has always done, and
                // why a second screenshot there resumes from the whole
                // conversation instead of re-reading it. It also covers the
                // FIRST screenshot of a long session, where the cache holds
                // only text: the transcript stays, the picture is all that
                // gets evaluated.
                let inPrefix = mediaRuns.prefix { $0.end <= common }.count
                let sameImages =
                    mediaRuns.count == imageKeys.count && inPrefix <= kvImageKeys.count
                    && Array(imageKeys.prefix(inPrefix)) == Array(kvImageKeys.prefix(inPrefix))
                if ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_REUSE"] != nil {
                    let lo = max(0, common - 12)
                    let cached = context.tokenizer.decode(
                        tokenIds: Array(kvTokens[lo ..< min(kvTokens.count, common + 24)]))
                    let fresh = context.tokenizer.decode(
                        tokenIds: Array(tokens[lo ..< min(tokens.count, common + 24)]))
                    FileHandle.standardError.write(
                        ("mlx-diverge @\(common)\n  cached: \(cached.debugDescription)"
                            + "\n  this:   \(fresh.debugDescription)\n")
                            .data(using: .utf8)!)
                    FileHandle.standardError.write(
                        ("mlx-reuse: common=\(common) inPrefix=\(inPrefix) same=\(sameImages)"
                            + " runLens=\(mediaRuns.map { $0.end - $0.start })"
                            + " runs=\(mediaRuns.count) keys=\(imageKeys.count)"
                            + " kvKeys=\(kvImageKeys.count) lastMediaEnd=\(lastMediaEnd)\n")
                            .data(using: .utf8)!)
                }
                if !sameImages {
                    common = 0
                }
                // Every picture above the prefix has to be embedded by this
                // run, and ONLY those — the ones already in the KV must not be
                // fed twice. One at a time, each from the processor run over
                // that single image: a picture's patches don't depend on the
                // words around it, and taking them one at a time is what keeps
                // a conversation holding several screenshots from becoming one
                // enormous forward pass when the cache is cold. Each is checked
                // against the placeholders the full prompt reserved for it; if
                // a template disagrees, don't second-guess it — fall back to
                // the whole span in one call, which is always correct.
                if lmInput.video == nil, mediaRuns.count == imageKeys.count, !mediaRuns.isEmpty {
                    var built: [LMInput.ProcessedImage] = []
                    for k in inPrefix ..< mediaRuns.count {
                        let reduced = try await context.processor.prepare(
                            input: UserInput(chat: [
                                .init(
                                    role: .user, content: "",
                                    images: [.url(URL(fileURLWithPath: imagePaths[k]))])
                            ]))
                        let got = reduced.text.tokens.asArray(Int32.self)
                            .filter { Int($0) == meta.imageTokenId }.count
                        guard got == mediaRuns[k].end - mediaRuns[k].start,
                            let px = reduced.image
                        else { break }
                        built.append(px)
                    }
                    if built.count == mediaRuns.count - inPrefix {
                        perImage = built
                        perImageFrom = inPrefix
                    } else if inPrefix > 0 {
                        // The new ones cannot be fed alone, so nothing below
                        // them can be kept either.
                        common = 0
                    }
                } else if inPrefix > 0 {
                    // Video, or a layout we cannot map one-to-one: the whole
                    // span goes in one call with every frame in it, which only
                    // works from the beginning.
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
                if ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_REUSE"] != nil {
                    FileHandle.standardError.write(
                        ("mlx-reuse: start=\(start) kvEvaluated=\(kvEvaluated) dropped=\(kvCache == nil)\n")
                            .data(using: .utf8)!)
                }
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
            /// Ordinary text, fed in chunks, positions threaded. Returns false
            /// if the caller should stop (cancelled).
            func runText(to stopAt: Int) -> Bool {
                while pos < stopAt {
                    if cancelFlag.isSet {
                        self.finish(
                            prompt: total, done: 0, tps: 0, reason: "cancelled", generated: nil)
                        return false
                    }
                    let end = min(pos + Self.prefillChunk, stopAt)
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
                return true
            }
            if pos < lastMediaEnd {
                // Only the span from the first PICTURE onward has to go through
                // in one pass — the model reads an image's position off the
                // text before it, and that text can just as well be in the
                // cache already. Everything below the first placeholder is
                // ordinary text and chunks like any other, which is what keeps
                // a screenshot in a long conversation from becoming one
                // enormous allocation: the owner's 50k-token round produced
                // nothing in ninety minutes, and it is this that made it one
                // pass rather than the picture itself.
                let first = mediaRuns.firstIndex(where: { $0.end > pos }) ?? mediaRuns.count
                if cancelFlag.isSet {
                    self.finish(prompt: total, done: 0, tps: 0, reason: "cancelled", generated: nil)
                    return
                }
                if emitProgress {
                    out.emit(["event": "prefill", "processed": pos, "total": total])
                }
                // A resume point never lands inside a placeholder run, so each
                // call below starts on a token boundary; the model reads the
                // cache's offset for its positions, so a warm cache continues
                // where it left off instead of restarting at zero.
                if let perImage, first == perImageFrom {
                    // One picture per call, the text before each one chunked
                    // like any other text. The single pass is then the size of
                    // a picture — never the size of the conversation under it.
                    for k in first ..< mediaRuns.count {
                        if !runText(to: max(pos, mediaRuns[k].start)) { return }
                        let head = LMInput(
                            text: .init(
                                tokens: MLXArray(
                                    tokens[pos..<mediaRuns[k].end].map(Int32.init))[.newAxis]),
                            image: perImage[k - perImageFrom], video: nil)
                        switch try context.model.prepare(head, cache: warm, windowSize: nil) {
                        case .logits(let headOut):
                            eval(headOut.logits)
                            state = headOut.state
                            pos = mediaRuns[k].end
                            if emitProgress {
                                out.emit(["event": "prefill", "processed": pos, "total": total])
                            }
                        case .tokens:
                            legacyWhole = true
                        }
                        if legacyWhole { break }
                    }
                } else {
                    // Pixels that could not be split, or a cache that had to be
                    // dropped after they were: the whole span in one call, with
                    // every picture in it — always correct, and the only shape
                    // an exotic processor is known to accept.
                    if !runText(to: max(pos, mediaRuns.first?.start ?? pos)) { return }
                    let head = LMInput(
                        text: .init(
                            tokens: MLXArray(tokens[pos..<lastMediaEnd].map(Int32.init))[.newAxis]),
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
            }
            if legacyWhole {
                kvCache = nil
                try await self.libraryStream(
                    context: context, input: lmInput, cache: warm, gp: gp, total: total)
                return
            }
            if !runText(to: prefillEnd) { return }

            // 4. Decode with the M-RoPE state threaded through every step.
            // The library's TokenIterator evaluates the final prompt token
            // and the first sampled one with a fresh rope state — positions
            // restart at zero, which full-attention M-RoPE models (Qwen3-VL)
            // answer with an instant EOS. Rolling our own loop keeps every
            // token at its true position.
            self.decode(
                context: context, cache: warm, tokens: tokens, total: total,
                state: state, gp: gp,
                recorded: tokens, imageKeys: imageKeys,
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
        // Token ids, not the library's text stream: that stream measures each
        // new piece by character count, which silently drops a character
        // whenever the decoder rewrites earlier text (the bug
        // `SafeStreamingDetokenizer` exists to make impossible). The iterator
        // underneath is the same media-aware one — only the detokenizing moves
        // to our side. Tool calls and stop *strings* were never taken from the
        // library here either; both stay Rust-side.
        let stream = try MLXLMCommon.generateTokens(
            input: input, cache: cache, parameters: gp, context: context)

        var detok = SafeStreamingDetokenizer(tokenizer: context.tokenizer)
        var dumpIds: [Int] = []
        var dumpStreamed = ""
        let dumpTokens = ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_TOKENS"] == "1"
        var reason = "eos"
        var info: GenerateCompletionInfo? = nil
        for await gen in stream {
            if cancelFlag.isSet {
                reason = "cancelled"
                break
            }
            if let token = gen.token {
                detok.append(token: token)
                if dumpTokens { dumpIds.append(token) }
                if let piece = detok.next() {
                    if dumpTokens { dumpStreamed += piece }
                    out.emit(["event": "token", "text": piece])
                }
            } else if let i = gen.info {
                info = i
            }
            // MambaCache never advances its offset — take the largest across
            // layers (the attention caches do track it).
            if nCtxCap > 0, (cache.map(\.offset).max() ?? 0) >= nCtxCap {
                reason = "context"
                break
            }
        }
        if dumpTokens {
            let whole = context.tokenizer.decode(tokenIds: dumpIds)
            log("STREAMED[\(dumpStreamed.count)]>>>" + dumpStreamed + "<<<END")
            log("WHOLE[\(whole.count)]>>>" + whole + "<<<END")
            log("MATCH=\(dumpStreamed == whole)")
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

        var detok = SafeStreamingDetokenizer(tokenizer: context.tokenizer)
        // Every token this turn puts into the cache, so the next turn can
        // match against them instead of trimming them away unseen.
        var generatedIds: [Int] = []
        var dumpIds: [Int] = []
        var dumpStreamed = ""
        let dumpTokens = ProcessInfo.processInfo.environment["CHATY_MLX_DUMP_TOKENS"] == "1"
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
            generatedIds.append(tok)
            if dumpTokens { dumpIds.append(tok) }
            if let piece = detok.next() {
                if dumpTokens { dumpStreamed += piece }
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
        if dumpTokens {
            let whole = context.tokenizer.decode(tokenIds: dumpIds)
            log("STREAMED[\(dumpStreamed.count)]>>>" + dumpStreamed + "<<<END")
            log("WHOLE[\(whole.count)]>>>" + whole + "<<<END")
            log("MATCH=\(dumpStreamed == whole)")
        }
        let dt = max(Date().timeIntervalSince(started), 0.001)
        self.finish(
            prompt: total, done: done, tps: Double(done) / dt, reason: reason,
            generated: recorded + generatedIds, images: imageKeys, state: recordedState,
            evaluated: total + done, reused: reused,
            lastGeneratedIds: generatedIds, lastTokenizer: context.tokenizer)
    }

    private func finish(
        prompt: Int, done: Int, tps: Double, reason: String, generated: [Int]?,
        images: [String] = [], state: LMOutput.State? = nil, evaluated: Int = 0,
        reused: Int = 0, lastGeneratedIds: [Int]? = nil,
        lastTokenizer: (any MLXLMCommon.Tokenizer)? = nil
    ) {
        // Remember EVERY token the cache now holds — the prompt and the reply
        // this turn generated. Recording only the prompt made the generated
        // tail look like excess that the next turn had to trim away, and a
        // model whose memory cannot rewind answered that by clearing the cache
        // and re-reading the whole conversation. With the reply in the ledger,
        // a turn that only appends (an agent posting a tool result) matches
        // right to the end: nothing to trim, nothing to re-read.
        if let generated {
            kvTokens = generated
            kvImageKeys = images
            kvState = state
            kvEvaluated = evaluated
            // What this turn occupies, against the text the app will store for
            // it, so the next prompt can replay rather than re-encode.
            if let ids = lastGeneratedIds, !ids.isEmpty, let tok = lastTokenizer {
                let whole = tok.decode(tokenIds: ids, skipSpecialTokens: false)
                // Under two keys, because the app stores a turn one of two
                // ways: verbatim, or — where the template reads thinking from
                // its own field (Qwen3.8) — with the reasoning split out and
                // only the answer left in `content`. Either is a legitimate
                // way to ask for the same turn back.
                var keys = [whole]
                // The same cut the app makes, both marker spellings normalised
                // first: `<think>` for Qwen, a thought channel for Gemma. A
                // turn that ran out of tokens mid-thought has no answer at all,
                // which is why the reasoning alone has to be part of the key —
                // `content` is empty and would match every such turn.
                let norm = whole.replacingOccurrences(of: "<|channel>", with: "<think>")
                    .replacingOccurrences(of: "<channel|>", with: "</think>")
                var reasoning = "", answer = norm
                if let c = norm.range(of: "</think>") {
                    let o = norm.range(of: "<think>")
                    let head = (o != nil && o!.lowerBound < c.lowerBound) ? o!.upperBound : norm.startIndex
                    reasoning = String(norm[head ..< c.lowerBound])
                    answer = String(norm[c.upperBound...])
                } else if let o = norm.range(of: "<think>") {
                    reasoning = String(norm[o.upperBound...])
                    answer = ""
                }
                let ans = answer.trimmingCharacters(in: .whitespacesAndNewlines)
                let rsn = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
                if !ans.isEmpty, ans != whole { keys.append(ans) }
                if !rsn.isEmpty {
                    keys.append(rsn + "\u{0}" + ans)
                } else {
                    // No markers at all. That is either an ordinary answer —
                    // already keyed by `whole` — or a turn whose thinking block
                    // the TEMPLATE opened, so the model wrote reasoning without
                    // ever announcing it (Qwen3.8). The app stores that one as
                    // reasoning with empty content, and it has to be findable
                    // under that shape too.
                    keys.append(
                        whole.trimmingCharacters(in: .whitespacesAndNewlines) + "\u{0}")
                }
                for key in keys where !key.isEmpty {
                    if turnIds[key] == nil { turnOrder.append(key) }
                    turnIds[key] = pendingBlock + ids
                }
                while turnOrder.count > 128 {
                    turnIds.removeValue(forKey: turnOrder.removeFirst())
                }
            }
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
