// Muse Glimmer — a `muse_glimmer` implementation for the MLX sidecar.
//
// Neither mlx-swift-lm nor upstream mlx-lm/mlx-vlm implements this
// architecture, so the sidecar answered it with
// `unsupportedModelType("muse_glimmer")`. It is a Llama-shaped block with four
// departures, all of which the reference implementation in transformers
// (`models/muse_glimmer/modeling_muse_glimmer.py`) spells out:
//
//   • sandwich norms — a second RMSNorm after attention and after the MLP,
//     with the residual added AFTER that norm rather than before it;
//   • hybrid attention — `layer_types` alternates three sliding-window layers
//     (2048) to one full-attention layer;
//   • NoPE — the full-attention layers take no rotary embedding at all
//     (`no_rope_layers` is 0 exactly where `layer_types` is full_attention);
//   • scaling — queries are RMS-normalised and multiplied by a constant
//     `qk_scale_factor`, and the logits leave through a tanh soft cap.
//
// The checkpoint carries no `q_norm`/`k_norm` weights, which is what makes the
// constant necessary: the normalisation is parameter-free, so the scale it
// removes has to be put back by hand.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct MuseGlimmerConfiguration: Decodable, Sendable {
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var ropeTheta: Float
    var slidingWindow: Int
    var rmsNormEps: Float
    var postNormEps: Float
    var qkScaleFactor: Float
    var outputMultiplier: Float
    var outputSoftCapTemp: Float
    var tieWordEmbeddings: Bool
    var layerTypes: [String]
    var noRopeLayers: [Int]
    var normalizeTokEmbeddings: Bool

    /// True where the layer attends over the sliding window rather than the
    /// whole sequence.
    func isSliding(_ i: Int) -> Bool {
        i < layerTypes.count && layerTypes[i] == "sliding_attention"
    }
    /// The reference gates RoPE on a per-layer flag; 0 means the layer runs
    /// with no positional embedding at all.
    func usesRope(_ i: Int) -> Bool {
        i >= noRopeLayers.count || noRopeLayers[i] == 1
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case outputSoftCapTemp = "output_soft_cap_temp"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case noRopeLayers = "no_rope_layers"
        case normalizeTokEmbeddings = "normalize_tok_embeddings"
        case textConfig = "text_config"
        case packagedFormat = "muse_glimmer_mlx_format"
        case layerRopeTheta = "layer_rope_theta"
        case finalLogitSoftcapping = "final_logit_softcapping"
    }

    /// The packaged artifact states the scale already divided by
    /// `sqrt(head_dim)`; the HF/RC config states the reference's own constant.
    /// Same effective scale, two different numbers under one key name.
    var packagedScale = true

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: CodingKeys.self)
        // Multimodal exports nest the language model's own config; the text
        // keys below read the same either way.
        let c =
            (try? outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)) ?? outer
        packagedScale = try outer.decodeIfPresent(Int.self, forKey: .packagedFormat) != nil
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / attentionHeads)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        postNormEps = try c.decodeIfPresent(Float.self, forKey: .postNormEps) ?? 1e-8
        qkScaleFactor = try c.decodeIfPresent(Float.self, forKey: .qkScaleFactor) ?? 1
        outputMultiplier = try c.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1
        outputSoftCapTemp =
            try c.decodeIfPresent(Float.self, forKey: .outputSoftCapTemp)
            ?? c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        // Two spellings of the same schedule: a flag per layer, or the layer's
        // own rope base with zero standing for "no rotary embedding here".
        if let flags = try c.decodeIfPresent([Int].self, forKey: .noRopeLayers) {
            noRopeLayers = flags
        } else if let thetas = try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta) {
            noRopeLayers = thetas.map { $0 > 0 ? 1 : 0 }
        } else {
            noRopeLayers = []
        }
        normalizeTokEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .normalizeTokEmbeddings) ?? true
    }
}

/// RMS with no learnable weight — the checkpoint has none for q/k, and the
/// scale it removes is restored by `qk_scale_factor`.
private func rmsNormPlain(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps)
}

private class MuseGlimmerAttention: Module {
    let args: MuseGlimmerConfiguration
    let scale: Float
    let useRope: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPELayer?

    init(_ args: MuseGlimmerConfiguration, layer: Int) {
        self.args = args
        self.useRope = args.usesRope(layer)
        // The reference multiplies the QK-normalised queries by
        // `qk_scale_factor / sqrt(head_dim)` and then lets SDPA apply its own
        // `1 / sqrt(head_dim)`. Softmax scaling is linear in q and RoPE is
        // orthogonal, so both fold into one constant here.
        self.scale =
            args.packagedScale
            ? args.qkScaleFactor / Float(args.headDim)
            : args.qkScaleFactor * pow(Float(args.headDim), -0.5)

        let dim = args.hiddenSize
        // The reference keeps `output_gate_proj` beside `q_proj`, both reading
        // the hidden state and both `heads * head_dim` wide. Packaged MLX
        // artifacts fuse the two per head — `[q_head; gate_head]` — into one
        // projection of twice the width, which is how it is split below.
        _qProj.wrappedValue = Linear(dim, 2 * args.attentionHeads * args.headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _oProj.wrappedValue = Linear(args.attentionHeads * args.headDim, dim, bias: false)

        self.rope =
            useRope
            ? initializeRope(
                dims: args.headDim, base: args.ropeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
            : nil
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let width = args.attentionHeads * args.headDim

        let qg = qProj(x).reshaped(B, L, args.attentionHeads, 2 * args.headDim)
        let heads = qg[0..., 0..., 0..., ..<args.headDim]
        let gate = qg[0..., 0..., 0..., args.headDim...].reshaped(B, L, width)
        var queries = heads.transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Parameter-free QK-norm over head_dim, applied before RoPE. The
        // scale it strips is folded into `self.scale`, not reapplied here.
        queries = rmsNormPlain(queries, eps: args.rmsNormEps)
        keys = rmsNormPlain(keys, eps: args.rmsNormEps)

        if useRope, let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let out = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        // Reference: `out = sigmoid(output_gate_proj(x)) * out`.
        return oProj(sigmoid(gate) * out)
    }
}

private class MuseGlimmerMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ dim: Int, _ hidden: Int) {
        _gate.wrappedValue = Linear(dim, hidden, bias: false)
        _up.wrappedValue = Linear(dim, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

/// One block. The checkpoint names the two sandwich norms `post_attn_norm` and
/// `post_ffn_norm`, and keeps Llama's `post_attention_layernorm` for the norm
/// that runs BEFORE the MLP — which is what that name means in Llama.
private class MuseGlimmerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MuseGlimmerAttention
    let mlp: MuseGlimmerMLP

    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attn_norm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var preFfnNorm: RMSNorm
    @ModuleInfo(key: "post_ffn_norm") var postFfnNorm: RMSNorm

    let useSliding: Bool

    init(_ args: MuseGlimmerConfiguration, layer: Int) {
        self.useSliding = args.isSliding(layer)
        _attention.wrappedValue = MuseGlimmerAttention(args, layer: layer)
        self.mlp = MuseGlimmerMLP(args.hiddenSize, args.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.postNormEps)
        _preFfnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postFfnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.postNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + postAttnNorm(attention(inputNorm(x), mask: mask, cache: cache))
        return h + postFfnNorm(mlp(preFfnNorm(h)))
    }
}

private class MuseGlimmerModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [MuseGlimmerBlock]
    /// Parameter-free, so it carries no weight of its own and cannot be
    /// spotted in the checkpoint — the config flag is the only signal.
    let embedNorm: Bool
    let norm: RMSNorm
    let args: MuseGlimmerConfiguration

    init(_ args: MuseGlimmerConfiguration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.embedNorm = args.normalizeTokEmbeddings
        self.layers = (0 ..< args.hiddenLayers).map { MuseGlimmerBlock(args, layer: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        if embedNorm { h = rmsNormPlain(h, eps: args.rmsNormEps) }

        // Two masks, because the two kinds of layer see different spans: the
        // full layers attend over everything, the sliding ones over a 2048
        // band that includes the query position.
        let fullIdx = (0 ..< layers.count).first { !layers[$0].useSliding }
        let slideIdx = (0 ..< layers.count).first { layers[$0].useSliding }
        let fullMask = createAttentionMask(h: h, cache: fullIdx.flatMap { cache?[$0] })
        var slideMask: MLXFast.ScaledDotProductAttentionMaskMode = fullMask
        if let slideIdx {
            slideMask = createAttentionMask(
                h: h, cache: cache?[slideIdx], windowSize: args.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.useSliding ? slideMask : fullMask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class MuseGlimmerModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: MuseGlimmerModelInner
    let config: MuseGlimmerConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MuseGlimmerConfiguration) {
        self.config = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = MuseGlimmerModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        var out = lmHead?(h) ?? model.embedTokens.asLinear(h)
        // Reference: softcap * tanh(logits * output_multiplier / softcap).
        out = out * config.outputMultiplier
        if config.outputSoftCapTemp > 0 {
            let cap = config.outputSoftCapTemp
            out = tanh(out / cap) * cap
        }
        return out
    }

    /// Normalise a multimodal HF export into the shape this model declares.
    ///
    /// Two conventions ship under the same architecture. A packaged MLX
    /// artifact is already in this shape and passes straight through. An HF
    /// export nests the language model under its own prefix, names the four
    /// norms positionally rather than by role, stores those norms as offsets
    /// from 1.0, and keeps the attention gate as a projection of its own.
    ///
    /// The renames are POSITIONAL and must not cascade: the export's
    /// `post_attention_layernorm` is the sandwich norm AFTER attention, while
    /// the name this model uses for it belongs to the norm before the MLP. One
    /// match per key, first match wins.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard weights.keys.contains(where: { $0.hasPrefix("language_model.") }) else {
            return weights
        }
        let renames = [
            (".post_attention_layernorm.", ".post_attn_norm."),
            (".pre_feedforward_layernorm.", ".post_attention_layernorm."),
            (".post_feedforward_layernorm.", ".post_ffn_norm."),
        ]
        let offsetNorms = [
            "input_layernorm.weight", "post_attn_norm.weight",
            "post_attention_layernorm.weight", "post_ffn_norm.weight",
        ]

        var out: [String: MLXArray] = [:]
        var gates: [String: MLXArray] = [:]
        for (rawKey, value) in weights {
            // The vision tower is not built here; its weights are dropped
            // rather than left to fail against a model that has no slot.
            if rawKey.hasPrefix("vision_") { continue }
            var key = rawKey
            if key.hasPrefix("language_model.") {
                key = String(key.dropFirst("language_model.".count))
            }
            for (from, to) in renames where key.contains(from) {
                key = key.replacingOccurrences(of: from, with: to)
                break
            }
            if key.contains(".self_attn.gate_proj.") {
                gates[key.replacingOccurrences(of: ".gate_proj.", with: ".q_proj.")] = value
                continue
            }
            var w = value
            if offsetNorms.contains(where: { key.hasSuffix($0) }) {
                w = w + 1.0
            }
            out[key] = w
        }

        // `[q_head; gate_head]` per head, which is a reordering of output rows
        // and so applies to a quantised tensor's packed weight, scales and
        // biases the same way.
        let H = config.attentionHeads
        let D = config.headDim
        for (key, gate) in gates {
            guard let q = out[key] else { continue }
            let cols = q.dim(1)
            out[key] = concatenated(
                [q.reshaped(H, D, cols), gate.reshaped(H, D, cols)], axis: 1
            ).reshaped(2 * H * D, cols)
        }
        return out
    }

    /// LoRA attaches to the transformer blocks, as it does for every other
    /// model here.
    public var loraLayers: [Module] { model.layers }

    /// Full history on every layer, sliding ones included: the window is
    /// imposed by the banded mask, not by dropping keys. A rotating cache
    /// would make greedy output depend on how the prompt was chunked, and
    /// would leave the sliding layers untrimmable for prefix reuse.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { _ in StandardKVCache() }
    }
}

/// Teach the sidecar's factory about `muse_glimmer`. Idempotent, and safe to
/// call before every load.
public enum MuseGlimmerRegistration {
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType(
            "muse_glimmer",
            creator: { data in
                let config = try JSONDecoder().decode(MuseGlimmerConfiguration.self, from: data)
                return MuseGlimmerModel(config)
            })
    }
}
