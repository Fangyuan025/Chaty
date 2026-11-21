// flint (RecursiveSI), `flint`: a 40B-total, 2.5B-active MoE decoder — two
// dense layers, then 256 sigmoid-routed experts (six a token, renormalised)
// beside a shared one; sliding-window attention with RoPE and full attention
// without it; RMS-normed queries and keys; a learnt softmax offset per head
// (an attention sink). mlx-swift-lm does not carry it; this follows mlx-lm's
// (mlx_lm/models/flint, ml-explore/mlx-lm #1998).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct FlintConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let layerCount: Int
    let intermediateSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let expertCount: Int
    let expertsPerToken: Int
    let moeIntermediateSize: Int
    let sharedIntermediateSize: Int
    let mlpOnlyLayers: [Int]
    let layerTypes: [String]
    let slidingWindow: Int
    let normEpsilon: Float
    let vocabularySize: Int
    let ropeTheta: Float
    let tiedEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case layerCount = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case headCount = "num_attention_heads"
        case kvHeadCount = "num_key_value_heads"
        case headDim = "head_dim"
        case expertCount = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case normEpsilon = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tiedEmbeddings = "tie_word_embeddings"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        layerCount = try c.decode(Int.self, forKey: .layerCount)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        headCount = try c.decode(Int.self, forKey: .headCount)
        kvHeadCount = try c.decode(Int.self, forKey: .kvHeadCount)
        headDim = try c.decode(Int.self, forKey: .headDim)
        expertCount = try c.decode(Int.self, forKey: .expertCount)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedIntermediateSize = try c.decode(Int.self, forKey: .sharedIntermediateSize)
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: layerCount)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 0
        normEpsilon = try c.decodeIfPresent(Float.self, forKey: .normEpsilon) ?? 1e-6
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        let theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeTheta =
            (try? c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters))??.ropeTheta
            ?? theta
        tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false
        guard layerTypes.count == layerCount, headCount > 0, kvHeadCount > 0,
            headCount % kvHeadCount == 0, expertsPerToken > 0, expertsPerToken <= expertCount,
            layerTypes.allSatisfy({ $0 == "full_attention" || ($0 == "sliding_attention" && slidingWindow > 0) })
        else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Invalid flint configuration"))
        }
    }
}

private final class FlintAttention: Module {
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE?
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    /// The learnt softmax offset of each head, passed to attention as a sink.
    @ParameterInfo(key: "softmax_offset") var sinks: MLXArray

    init(_ c: FlintConfiguration, layer: Int) {
        headCount = c.headCount
        kvHeadCount = c.kvHeadCount
        headDim = c.headDim
        scale = pow(Float(c.headDim), -0.5)
        _qProj.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.kvHeadCount * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.kvHeadCount * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.headCount * c.headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.normEpsilon)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.normEpsilon)
        _sinks.wrappedValue = zeros([c.headCount])
        // Only the sliding-window layers are rotated; full attention has no
        // position encoding at all.
        rope =
            c.layerTypes[layer] == "sliding_attention"
            ? RoPE(dimensions: c.headDim, traditional: false, base: c.ropeTheta) : nil
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, headCount, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, l, kvHeadCount, headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(b, l, kvHeadCount, headDim).transposed(0, 2, 1, 3)
        if let rope {
            q = applyRotaryPosition(rope, to: q, offset: cache?.ropeOffset)
            k = applyRotaryPosition(rope, to: k, offset: cache?.ropeOffset)
        }
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask, sinks: sinks)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

private final class FlintMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ dim: Int, _ hidden: Int) {
        _gateProj.wrappedValue = Linear(dim, hidden, bias: false)
        _upProj.wrappedValue = Linear(dim, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

private final class FlintMoE: Module {
    let topK: Int
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP

    init(_ c: FlintConfiguration) {
        topK = c.expertsPerToken
        _gate.wrappedValue = Linear(c.hiddenSize, c.expertCount, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: c.hiddenSize, hiddenDims: c.moeIntermediateSize, numExperts: c.expertCount)
        _sharedExpert.wrappedValue = FlintMLP(c.hiddenSize, c.sharedIntermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Routed in float32: sigmoid scores, the top six, renormalised.
        let scores = sigmoid(gate(x.asType(.float32)))
        let inds = stopGradient(argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        var weights = takeAlong(scores, inds, axis: -1)
        weights = weights / weights.sum(axis: -1, keepDims: true)
        let y = switchMLP(x, inds)
        let routed = (y * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        return routed + sharedExpert(x)
    }
}

private final class FlintLayer: Module {
    let sliding: Bool
    @ModuleInfo(key: "self_attn") var attention: FlintAttention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ c: FlintConfiguration, layer: Int) {
        sliding = c.layerTypes[layer] == "sliding_attention"
        _attention.wrappedValue = FlintAttention(c, layer: layer)
        _mlp.wrappedValue =
            c.mlpOnlyLayers.contains(layer)
            ? FlintMLP(c.hiddenSize, c.intermediateSize) : FlintMoE(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        let n = postNorm(h)
        let m: MLXArray
        if let moe = mlp as? FlintMoE {
            m = moe(n)
        } else {
            m = (mlp as! FlintMLP)(n)
        }
        return h + m
    }
}

private final class FlintInner: Module {
    let config: FlintConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [FlintLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ c: FlintConfiguration) {
        config = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0 ..< c.layerCount).map { FlintLayer(c, layer: $0) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(tokens)
        let n = h.dim(1)
        let full = config.layerTypes.firstIndex(of: "full_attention")
        let swa = config.layerTypes.firstIndex(of: "sliding_attention")
        let fullMask = full.map { makeAttentionMask(n: n, cache: cache?[$0], windowSize: nil) }
        let swaMask = swa.map {
            makeAttentionMask(n: n, cache: cache?[$0], windowSize: config.slidingWindow)
        }
        for (i, layer) in layers.enumerated() {
            let mask = (layer.sliding ? swaMask : fullMask) ?? .none
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

final class FlintModel: Module, LLMModel {
    let config: FlintConfiguration
    @ModuleInfo(key: "model") private var model: FlintInner
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?
    var vocabularySize: Int { config.vocabularySize }
    var loraLayers: [Module] { model.layers }

    init(_ c: FlintConfiguration) {
        config = c
        _model.wrappedValue = FlintInner(c)
        if !c.tiedEmbeddings {
            _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        return lmHead.map { $0(h) } ?? model.embedTokens.asLinear(h)
    }

    func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        var weights = checkpoint.weights
        for key in Array(weights.keys) where key.contains("rotary_emb.inv_freq") {
            weights[key] = nil
        }
        if config.tiedEmbeddings {
            for key in Array(weights.keys) where key.hasPrefix("lm_head.") { weights[key] = nil }
        }
        // RecursiveSI/flint-int4 packs its quantized weights as uint8; MLX
        // reads them as uint32 words of the same bytes.
        for key in Array(weights.keys) where key.hasSuffix(".weight") {
            guard let w = weights[key], w.dtype == .uint8,
                weights[String(key.dropLast(".weight".count)) + ".scales"] != nil
            else { continue }
            weights[key] = w.view(dtype: .uint32)
        }
        // Per-expert weights (an unquantized export) stacked into the switch.
        for l in 0 ..< config.layerCount where !config.mlpOnlyLayers.contains(l) {
            let prefix = "model.layers.\(l).mlp"
            for n in ["gate_proj", "up_proj", "down_proj"] {
                guard weights["\(prefix).experts.0.\(n).weight"] != nil else { continue }
                let parts = (0 ..< config.expertCount).compactMap {
                    weights.removeValue(forKey: "\(prefix).experts.\($0).\(n).weight")
                }
                weights["\(prefix).switch_mlp.\(n).weight"] = stacked(parts)
            }
        }
        checkpoint.weights = weights
        return checkpoint
    }

    func newCache(parameters: GenerateParameters? = nil) throws -> [KVCache] {
        try config.layerTypes.map {
            try makeHybridAttentionKVCache(
                parameters: parameters, slidingWindow: config.slidingWindow,
                usesSlidingWindow: $0 == "sliding_attention")
        }
    }
}

/// Teach the sidecar's factory about flint. Idempotent.
public enum FlintRegistration {
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType(
            "flint",
            creator: { data in
                FlintModel(try JSONDecoder().decode(FlintConfiguration.self, from: data))
            })
    }
}
