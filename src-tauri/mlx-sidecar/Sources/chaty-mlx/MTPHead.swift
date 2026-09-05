// Multi-token prediction for the Qwen3.5 family.
//
// These checkpoints ship a small head in a file of its own (`mtp.safetensors`,
// named by `mlx_lm_extra_tensors.mtp_file` in the config). It is one decoder
// layer, two norms and a fusion projection, and its job is to guess the tokens
// the trunk is about to produce. The trunk then checks a whole run of guesses
// in ONE forward pass instead of one pass per token — the arithmetic per token
// barely changes, but the passes are what cost time on this hardware, so the
// same answer arrives about three times sooner.
//
// It is a speedup, not a change of answer: every token that survives is one the
// trunk itself chose, and the first guess it disagrees with ends the run. What
// comes out is what plain decoding would have produced.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The contract the checkpoint states in `mtplx_mtp_contract` / `mtp_contract`.
/// Nothing here is inferred: an implementation built from tensor shapes alone
/// has several plausible readings and only one of them is this model's.
struct MTPContract: Decodable {
    /// Which trunk hidden the head is fed for its FIRST guess.
    var baseHiddenVariant: String = "post_norm"
    /// Which hidden the head feeds ITSELF for the guesses after that.
    var hiddenVariant: String = "post_norm"
    /// Whether the embedding or the hidden comes first in the fused input.
    var concatOrder: String = "embedding_hidden"
    /// Positions the head's own attention counts from.
    var mtpPositionMode: String = "local"

    enum CodingKeys: String, CodingKey {
        case baseHiddenVariant = "base_hidden_variant"
        case hiddenVariant = "hidden_variant"
        case concatOrder = "concat_order"
        case mtpPositionMode = "mtp_position_mode"
    }

    /// Everything this implementation was written against. A checkpoint that
    /// states something else is not one it can serve, and guessing would show
    /// up as a quietly worse acceptance rate rather than an error.
    var isSupported: Bool {
        baseHiddenVariant == "post_norm" && hiddenVariant == "post_norm"
            && concatOrder == "embedding_hidden" && mtpPositionMode == "local"
    }

    var unsupportedReason: String {
        var parts: [String] = []
        if baseHiddenVariant != "post_norm" { parts.append("base_hidden_variant=\(baseHiddenVariant)") }
        if hiddenVariant != "post_norm" { parts.append("hidden_variant=\(hiddenVariant)") }
        if concatOrder != "embedding_hidden" { parts.append("concat_order=\(concatOrder)") }
        if mtpPositionMode != "local" { parts.append("mtp_position_mode=\(mtpPositionMode)") }
        return parts.joined(separator: ", ")
    }
}

/// What the runtime file says about how to drive the head.
struct MTPRuntime: Decodable {
    var mtpDepthDefault: Int = 3
    var mtpDepthMax: Int = 3
    var mtpContract: MTPContract = .init()
    var recommendedDraftSampler: MTPSampler = .init()

    enum CodingKeys: String, CodingKey {
        case mtpDepthDefault = "mtp_depth_default"
        case mtpDepthMax = "mtp_depth_max"
        case mtpContract = "mtp_contract"
        case recommendedDraftSampler = "recommended_draft_sampler"
    }
}

struct MTPSampler: Decodable {
    var temperature: Float = 1.0
    var topK: Int = 20
    var topP: Float = 0.95
}

/// One attention block, shaped exactly like the trunk's own.
///
/// The head is a single decoder layer of the same design as the layers around
/// it, so it has to match them numerically or its guesses are wrong in ways
/// that only show up as a poor acceptance rate. Two details are easy to miss and
/// both are load-bearing: `q_proj` emits queries AND an output gate side by side
/// (which is why it is twice the width the head count suggests), and rotary
/// position embedding covers only a quarter of each head's dimensions.
private class MTPAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ c: MTPHeadConfiguration) {
        self.heads = c.attentionHeads
        self.kvHeads = c.kvHeads
        self.headDim = c.headDim
        self.scale = pow(Float(c.headDim), -0.5)

        // Twice the query width: the second half is the output gate.
        _qProj.wrappedValue = Linear(c.hiddenSize, heads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)

        self.rope = RoPE(
            dimensions: Int(Float(c.headDim) * c.partialRotaryFactor),
            traditional: false, base: c.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let split = qProj(x).reshaped(B, L, heads, -1).split(parts: 2, axis: -1)
        var queries = split[0]
        let gate = split[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        // `mtp_position_mode: local` — the head counts positions from its own
        // cache, not from where the trunk happens to be in the conversation.
        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let out = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: .causal
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(out * sigmoid(gate))
    }
}

private class MTPMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: MTPHeadConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class MTPLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MTPAttention
    @ModuleInfo var mlp: MTPMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ c: MTPHeadConfiguration) {
        _attention.wrappedValue = MTPAttention(c)
        _mlp.wrappedValue = MTPMLP(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        var h = x + attention(inputNorm(x), cache: cache)
        h = h + mlp(postAttentionNorm(h))
        return h
    }
}

/// Shape of the head, read from the checkpoint's own text configuration.
struct MTPHeadConfiguration {
    var hiddenSize: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var partialRotaryFactor: Float
}

/// The head itself: fuse, transform, read out.
///
/// One step takes the token just settled on and the hidden state the trunk (or
/// the previous step) left behind, and produces the hidden state for the token
/// after it. Both inputs are normed before they are fused — separately, with
/// their own weights — and the fusion is a plain projection of the two laid end
/// to end. `concat_order: embedding_hidden` fixes which end is which; getting it
/// backwards is not an error anywhere, it just quietly guesses badly.
final class MTPHead: Module {
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo var fc: Linear
    @ModuleInfo fileprivate var layers: [MTPLayer]
    @ModuleInfo var norm: RMSNorm

    let contract: MTPContract

    init(_ c: MTPHeadConfiguration, contract: MTPContract) {
        self.contract = contract
        _preFcNormEmbedding.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _preFcNormHidden.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _fc.wrappedValue = Linear(c.hiddenSize * 2, c.hiddenSize, bias: false)
        _layers.wrappedValue = [MTPLayer(c)]
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    /// One draft step. `embedding` is the embedding of the token just settled
    /// on; `hidden` is what the trunk produced for it (first step) or what this
    /// returned last time (the steps after). The result is the hidden state the
    /// next token would be read out of.
    func callAsFunction(embedding: MLXArray, hidden: MLXArray, cache: KVCache?) -> MLXArray {
        let e = preFcNormEmbedding(embedding)
        let h = preFcNormHidden(hidden)
        // `concat_order: embedding_hidden`, which the contract check above has
        // already insisted on: the embedding occupies the first half of the
        // fusion input. Feeding the two halves the other way round loads and
        // runs perfectly well and simply guesses badly, so this is not a
        // detail to infer.
        var x = fc(concatenated([e, h], axis: -1))
        for layer in layers {
            x = layer(x, cache: cache)
        }
        return norm(x)
    }

    func newCache() -> [KVCache] { layers.map { _ in KVCacheSimple() } }
}

extension MTPHead {
    /// Build the head from the sidecar file the checkpoint names.
    ///
    /// The trunk's own loader drops these tensors — they belong to no module in
    /// its graph — so the head is loaded here, separately, and quantized the
    /// same way the file already is: a module whose weights arrived with
    /// `scales` beside them is a quantized one, whatever the config says about
    /// the trunk.
    static func load(
        directory: URL, file: String, config: MTPHeadConfiguration, contract: MTPContract,
        bits: Int, groupSize: Int
    ) throws -> MTPHead {
        let head = MTPHead(config, contract: contract)

        let raw = try loadArrays(url: directory.appending(component: file))
        // Keys arrive as `mtp.fc.weight`; the head IS the `mtp` subtree.
        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(raw.count)
        for (key, value) in raw {
            weights[key.hasPrefix("mtp.") ? String(key.dropFirst(4)) : key] = value
        }

        quantize(model: head, groupSize: groupSize, bits: bits) { path, _ in
            weights["\(path).scales"] != nil
        }

        try head.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(head)
        return head
    }
}

/// What the sidecar needs to know to drive a checkpoint's MTP head, gathered
/// from the files the checkpoint ships. `nil` when the model has no head, which
/// is every model but this family and is not an error.
struct MTPSetup {
    var head: MTPHead
    /// The depth to draft at when nothing is measuring — a pinned override, or
    /// the checkpoint's own default until `tuner` has an opinion.
    var depth: Int
    var sampler: MTPSampler
    /// Measures the depth instead of trusting the checkpoint. `nil` when the
    /// depth was pinned by hand.
    var tuner: MTPDepthTuner?

    /// The depth to draft at this round.
    func nextDepth() -> Int { tuner?.begin() ?? depth }

    /// Read `config.json` + `mtplx_runtime.json` and load the head if both the
    /// file and the contract are ones this implementation can serve.
    static func discover(directory: URL) -> (setup: MTPSetup?, note: String?) {
        // An off switch, for comparing against plain decoding and for backing
        // out of the head without rebuilding.
        if ProcessInfo.processInfo.environment["CHATY_MLX_MTP"] == "0" {
            return (nil, "MTP head disabled (CHATY_MLX_MTP=0)")
        }
        let configURL = directory.appending(component: "config.json")
        guard let configData = try? Data(contentsOf: configURL),
            let root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else { return (nil, nil) }

        guard let extra = root["mlx_lm_extra_tensors"] as? [String: Any],
            let file = extra["mtp_file"] as? String
        else { return (nil, nil) }  // no head: an ordinary checkpoint

        // The runtime file carries depth and the draft sampler; the contract
        // appears in both files and must agree with what this code implements.
        var runtime = MTPRuntime()
        if let data = try? Data(contentsOf: directory.appending(component: "mtplx_runtime.json")),
            let decoded = try? JSONDecoder().decode(MTPRuntime.self, from: data)
        {
            runtime = decoded
        }
        if let contractData = try? JSONSerialization.data(
            withJSONObject: (root["mtplx_mtp_contract"] as? [String: Any]) ?? [:]),
            let decoded = try? JSONDecoder().decode(MTPContract.self, from: contractData)
        {
            runtime.mtpContract = decoded
        }
        guard runtime.mtpContract.isSupported else {
            return (nil, "MTP head ignored — unsupported contract: \(runtime.mtpContract.unsupportedReason)")
        }

        guard let text = root["text_config"] as? [String: Any],
            let hidden = text["hidden_size"] as? Int,
            let intermediate = text["intermediate_size"] as? Int,
            let heads = text["num_attention_heads"] as? Int,
            let kvHeads = text["num_key_value_heads"] as? Int
        else { return (nil, "MTP head ignored — text_config is missing shape fields") }

        let cfg = MTPHeadConfiguration(
            hiddenSize: hidden,
            intermediateSize: intermediate,
            attentionHeads: heads,
            kvHeads: kvHeads,
            headDim: (text["head_dim"] as? Int) ?? (hidden / heads),
            rmsNormEps: Float((text["rms_norm_eps"] as? Double) ?? 1e-6),
            ropeTheta: Float((text["rope_theta"] as? Double) ?? 1_000_000),
            partialRotaryFactor: Float((text["partial_rotary_factor"] as? Double) ?? 1.0))

        let quant = root["quantization"] as? [String: Any]
        do {
            let head = try MTPHead.load(
                directory: directory, file: file, config: cfg, contract: runtime.mtpContract,
                bits: (quant?["bits"] as? Int) ?? 8,
                groupSize: (quant?["group_size"] as? Int) ?? 64)
            let depth = max(1, min(runtime.mtpDepthDefault, runtime.mtpDepthMax))
            // Pinning a depth turns the tuner off: an explicit number is an
            // instruction, and a measurement that overrides it is a bug.
            if let override = ProcessInfo.processInfo.environment["CHATY_MLX_MTP_DEPTH"],
                let d = Int(override), d >= 0
            {
                return (
                    MTPSetup(
                        head: head, depth: max(0, min(d, runtime.mtpDepthMax)),
                        sampler: runtime.recommendedDraftSampler, tuner: nil), nil
                )
            }
            return (
                MTPSetup(
                    head: head, depth: depth, sampler: runtime.recommendedDraftSampler,
                    tuner: MTPDepthTuner(maxDepth: runtime.mtpDepthMax)), nil
            )
        } catch {
            // A head that will not load is a lost speedup, not a lost model.
            return (nil, "MTP head ignored — \(error)")
        }
    }
}

/// Picks the draft depth from what the machine actually delivers.
///
/// How many guesses are worth making is not a property of the checkpoint. A
/// deeper run wins more tokens per round, but every extra guess costs a head
/// step (with a full-vocabulary projection) and widens the batch the trunk has
/// to verify — and when a guess is rejected the whole round pays for a redo.
/// Where those curves cross depends on the GPU, on how loaded it is, and on
/// what is being written: the same head that nearly halves the time on a list
/// of numbers loses to plain decoding on ordinary prose. `mtp_depth_default` in
/// the checkpoint knows none of that; on this machine following it costs about
/// a third of the speedup.
///
/// So the depth is measured instead of assumed, in BLOCKS of consecutive
/// rounds rather than one round at a time. Changing the depth changes the width
/// of the batch the trunk verifies, and the first round at a new width costs
/// noticeably more than the ones after it. Timed one round at a time, every
/// candidate measures the cost of switching to it rather than the cost of
/// running it — which flatters whichever depth happens to be running already
/// and makes the choice self-confirming. A block pays that cost once, throws
/// away the rounds it lands on, and times the steady state.
///
/// Depth 0 — "do not guess" — is one of the candidates, so a head that is not
/// earning its keep on this text is simply switched off until it is.
final class MTPDepthTuner {
    /// Consecutive rounds spent at one depth before the choice is reconsidered.
    private let blockLength = 12
    /// Rounds at the start of a block that are not counted: the switch itself.
    private let settling = 3
    /// Rounds the winner runs before a rival is re-checked. Doubles up to
    /// `probeCeiling` while the winner holds, so a settled turn spends almost
    /// nothing on measurement; back to the floor the moment it changes.
    private let probeFloor = 120
    private let probeCeiling = 960
    /// Old blocks fade at this rate, per counted round.
    private let decay = 0.97

    private struct Arm {
        var tokens = 0.0
        var seconds = 0.0
        var rounds = 0
        var rate: Double { seconds > 0 ? tokens / seconds : 0 }
    }

    private let depths: [Int]
    private var arms: [Int: Arm]

    // The block in progress.
    private var blockDepth = 0
    private var blockLeft = 0
    private var blockRound = 0

    // Scheduling.
    private var warmup = 0
    private var sinceProbe = 0
    private var probeAfter: Int
    private var probeCursor = 0
    private var lastBest: Int?

    // The round being timed.
    private var openDepth: Int?
    private var openCounted = false
    private var openAt = Date.distantPast
    private var openTokens = 0

    init(maxDepth: Int) {
        depths = Array(0 ... max(1, maxDepth))
        arms = Dictionary(uniqueKeysWithValues: depths.map { ($0, Arm()) })
        probeAfter = probeFloor
    }

    /// The depth to draft at now. Closes the previous round first: the window
    /// that ends here is everything that round cost, including emitting the
    /// tokens it won.
    func begin() -> Int {
        flush()
        if blockLeft == 0 { openBlock() }
        blockLeft -= 1
        blockRound += 1
        openDepth = blockDepth
        openCounted = blockRound > settling
        openAt = Date()
        openTokens = 0
        return blockDepth
    }

    /// How many tokens the round in progress settled.
    func credit(_ tokens: Int) { openTokens += tokens }

    /// Close the last round (the turn is over and nothing more will be timed).
    func end() { flush() }

    /// The depth currently believed best, for reporting.
    var settled: Int { best }

    /// What the tuner has measured, one entry per depth, for reporting.
    var summary: String {
        depths.map { d in
            let a = arms[d] ?? Arm()
            return a.rounds > 0
                ? "\(d):\(String(format: "%.1f", a.rate))"
                : "\(d):-"
        }.joined(separator: " ")
    }

    /// Ties go to the shallower depth — guessing less is cheaper to be wrong
    /// about — so the search runs from the deepest down and keeps the last of
    /// equal rates.
    private var best: Int {
        depths.reversed().max(by: { (arms[$0]?.rate ?? 0) < (arms[$1]?.rate ?? 0) }) ?? 0
    }

    private func openBlock() {
        blockRound = 0
        blockLeft = blockLength
        // Every candidate gets one block before any of them is trusted.
        if warmup < depths.count {
            blockDepth = depths[warmup]
            warmup += 1
            return
        }
        let winner = best
        // A winner that holds is re-checked less and less often; one that
        // changes means the text changed, and the schedule starts over.
        if let lastBest {
            probeAfter = winner == lastBest ? min(probeAfter * 2, probeCeiling) : probeFloor
        }
        lastBest = winner
        let rivals = depths.filter { $0 != winner }
        if sinceProbe >= probeAfter, !rivals.isEmpty {
            sinceProbe = 0
            blockDepth = rivals[probeCursor % rivals.count]
            probeCursor += 1
        } else {
            sinceProbe += blockLength
            blockDepth = winner
        }
    }

    private func flush() {
        guard let depth = openDepth, openCounted, openTokens > 0 else {
            openDepth = nil
            return
        }
        let seconds = Date().timeIntervalSince(openAt)
        openDepth = nil
        // A round interrupted by something outside decoding (the turn being
        // cancelled, a long pause in the consumer) would otherwise be recorded
        // as that depth being catastrophically slow.
        guard seconds > 0, seconds < 5 else { return }
        var arm = arms[depth] ?? Arm()
        arm.tokens = arm.tokens * decay + Double(openTokens)
        arm.seconds = arm.seconds * decay + seconds
        arm.rounds += 1
        arms[depth] = arm
    }
}
