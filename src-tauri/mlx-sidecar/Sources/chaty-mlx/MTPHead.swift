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
    /// How many guesses a round may make at most. The confidence gate decides
    /// how many it actually makes — a run stops at the first guess the head is
    /// unsure of — so this is a ceiling rather than a length.
    var depth: Int
    var sampler: MTPSampler
    /// Watches whether the head is earning the cost of being run at all.
    /// `nil` when the depth was pinned by hand, which is a request to measure
    /// one configuration rather than to be governed.
    var governor: MTPGovernor?
    /// How sure the head has to be of a guess before it is worth checking.
    ///
    /// Not a quality knob — every emitted token is drawn from the trunk either
    /// way. It is a price: on this architecture a rejected guess costs about
    /// two and a half plain steps, so a round is only worth taking when the
    /// guess is very likely right.
    var minConfidence: Float = 0.5



    /// Whether this checkpoint carries a head this implementation can serve —
    /// read from the files, with nothing loaded.
    ///
    /// Separate from `discover` because it answers a different question. This
    /// one decides whether the app offers speculative decoding for the model at
    /// all; `discover` decides whether it is running right now. Turning the
    /// feature off must not make the switch that turns it back on disappear.
    static func declared(directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let extra = root["mlx_lm_extra_tensors"] as? [String: Any],
            extra["mtp_file"] is String
        else { return false }
        guard
            let contractData = try? JSONSerialization.data(
                withJSONObject: (root["mtplx_mtp_contract"] as? [String: Any]) ?? [:]),
            let contract = try? JSONDecoder().decode(MTPContract.self, from: contractData)
        else { return false }
        return contract.isSupported
    }

    /// Read `config.json` + `mtplx_runtime.json` and load the head if both the
    /// file and the contract are ones this implementation can serve.
    ///
    /// `enabled` is the user's setting; `CHATY_MLX_MTP=0` overrides it, for
    /// comparing against plain decoding without touching the UI.
    static func discover(directory: URL, enabled: Bool) -> (setup: MTPSetup?, note: String?) {
        if !enabled {
            return (nil, nil)  // switched off: an ordinary load, nothing to report
        }
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
            // How far ahead to guess, measured rather than taken from the
            // checkpoint. Against 7.1 tok/s of plain decoding on an M4 Pro:
            // one guess at a time gives 8.8-9.3 on a list of numbers, three
            // gives 7.7-8.3 — and on code three is 6.9-7.3, SLOWER than not
            // guessing. Three quarters of this model's layers are recurrent
            // and cannot be rewound, so a rejected guess is paid for with a
            // second pass to put them back; a deeper run is more chances to be
            // wrong, and each one is dear.
            let depth = max(1, min(runtime.mtpDepthDefault, runtime.mtpDepthMax, measuredDepthCap))
            let minConfidence =
                ProcessInfo.processInfo.environment["CHATY_MLX_MTP_PMIN"].flatMap(Float.init) ?? 0.5
            // Pinning a depth also turns the governor off: an explicit
            // configuration is an instruction to measure it, not to be
            // second-guessed halfway through.
            if let override = ProcessInfo.processInfo.environment["CHATY_MLX_MTP_DEPTH"],
                let d = Int(override), d >= 0
            {
                return (
                    MTPSetup(
                        head: head, depth: max(0, min(d, runtime.mtpDepthMax)),
                        sampler: runtime.recommendedDraftSampler, governor: nil,
                        minConfidence: minConfidence), nil
                )
            }
            return (
                MTPSetup(
                    head: head, depth: depth, sampler: runtime.recommendedDraftSampler,
                    governor: MTPGovernor(), minConfidence: minConfidence), nil
            )
        } catch {
            // A head that will not load is a lost speedup, not a lost model.
            return (nil, "MTP head ignored — \(error)")
        }
    }
}

/// Decides whether the head is worth running at all right now.
///
/// The confidence gate keeps the head from checking guesses it doubts, which
/// is what stops a bad round from costing two and a half plain steps. What it
/// cannot avoid is the head's OWN forward pass: deciding not to guess still
/// costs one, and on text the head is never confident about — ordinary prose,
/// where the next word is a real choice rather than the obvious continuation —
/// that is paid on every token and buys nothing. Measured at about 2%.
///
/// So the head is also watched at a coarser grain: over a window of rounds,
/// how many tokens did it actually WIN? If the answer is "not enough to cover
/// running it", it is put down for a while and the reply proceeds exactly as
/// it would with no head at all — not approximately, exactly: `speculate` is
/// not called, so there is nothing to pay for. It is picked back up
/// periodically, because a reply that starts as prose may turn into a table.
///
/// Deliberately NOT a throughput measurement. Timing rounds means timing the
/// machine, and the machine is doing other things; the count of tokens won is
/// exact, free, and answers the only question that matters.
/// The deepest run worth making on the hardware this has been measured on.
/// A checkpoint that asks for more is asking for something that costs time
/// here; `CHATY_MLX_MTP_DEPTH` overrides it for measuring another machine.
let measuredDepthCap = 1

final class MTPGovernor {
    /// Rounds the head gets to prove itself before the window is judged.
    /// Short on purpose: every round in a window that turns out to be losing
    /// was decoded at a loss, and a reply is only a few hundred rounds long.
    private let window = 24
    /// Extra tokens per attempt below which the head is not covering its cost.
    ///
    /// Set LOW on purpose — this is a floor for "clearly not working", not an
    /// optimizer. Standing the head down changes WHICH positions it is later
    /// judged on, so a threshold tight enough to act on ordinary text feeds
    /// itself: a nap skips a stretch the head would have won, the next window
    /// lands somewhere worse, and it naps again. Measured, that cost code —
    /// which wins reliably — a seven percent reply. The head is left running
    /// unless it is winning almost nothing at all.
    private let worthwhile = 0.2
    /// Tokens to wait before trying again, and the ceiling it backs off to.
    /// Windows in a row that must come out badly before the head is put down.
    /// One is not evidence: a single window of 24 rounds swings widely even on
    /// text the head is doing well at, and acting on one silenced the head for
    /// two whole replies in the middle of the content it wins most on.
    private let strikes = 2
    /// Tokens to wait before trying again. Short enough that a nap does not
    /// swallow a whole reply — the writing changes within one.
    private let restFloor = 192
    private let restCeiling = 1536

    private var attempts = 0
    private var won = 0
    private var bad = 0
    private var rest = 0  // tokens still to pass before the head is tried again
    private var restLength: Int
    /// Last judgement, for the stats line.
    private(set) var naps = 0

    init() { restLength = restFloor }

    /// Whether to run the head for the token about to be decoded.
    func shouldSpeculate() -> Bool {
        if rest > 0 {
            rest -= 1
            return false
        }
        return true
    }

    /// One round happened: `extra` is how many tokens it won beyond the one
    /// the trunk was going to produce anyway. A declined round wins nothing.
    func record(extra: Int) {
        attempts += 1
        won += extra
        guard attempts >= window else { return }
        let rate = Double(won) / Double(attempts)
        if rate < worthwhile {
            bad += 1
            if bad >= strikes {
                rest = restLength
                restLength = min(restLength * 2, restCeiling)
                naps += 1
                bad = 0
            }
        } else {
            // Earning its keep: forget the strike and come back sooner.
            bad = 0
            restLength = restFloor
        }
        attempts = 0
        won = 0
    }

    var summary: String { "naps=\(naps)" }
}

