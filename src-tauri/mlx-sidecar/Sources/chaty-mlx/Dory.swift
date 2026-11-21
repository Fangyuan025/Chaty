// NVIDIA Nemotron-Labs Dory 1.0, `dory`: a looped nGPT — every hidden state
// on the unit sphere, each block's output slerped into it by a learnt alpha —
// with input layers, a recurrent group run `n_recurrent_loops` times (the
// input re-injected before each pass) and output layers; sliding-window and
// full attention with gated, normalised outputs, and a cache for each pass of
// the recurrent group. mlx-swift-lm does not carry it; this follows mlx-lm's
// open pull request (ml-explore/mlx-lm #1988, pmichalas, a5abfc2), whose
// checkpoints keep q/k/v/g and gate/up already fused.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct DoryConfiguration: Decodable, Sendable {
    let vocabularySize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let layerCount: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let slidingWindow: Int
    let inputLayers: Int
    let recurrentLayers: Int
    let outputLayers: Int
    let loops: Int
    let alphaInit: Float
    let initializerRange: Float
    let gateLogitScaleInit: Float?
    let thetaSliding: Float
    let thetaFull: Float
    let layerTypes: [String]

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layerCount = "num_hidden_layers"
        case headCount = "num_attention_heads"
        case kvHeadCount = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingWindow = "sliding_window"
        case inputLayers = "n_input_layers"
        case recurrentLayers = "n_recurrent_layers"
        case outputLayers = "n_output_layers"
        case loops = "n_recurrent_loops"
        case cacheMode = "recurrent_kv_cache_mode"
        case alphaInit = "ngpt_alpha_init"
        case initializerRange = "initializer_range"
        case gateLogitScaleInit = "attention_output_gate_logit_scale_init"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
    }

    private struct Rope: Decodable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 131072
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 7168
        layerCount = try c.decodeIfPresent(Int.self, forKey: .layerCount) ?? 72
        headCount = try c.decodeIfPresent(Int.self, forKey: .headCount) ?? 16
        kvHeadCount = try c.decodeIfPresent(Int.self, forKey: .kvHeadCount) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        inputLayers = try c.decodeIfPresent(Int.self, forKey: .inputLayers) ?? 16
        recurrentLayers = try c.decodeIfPresent(Int.self, forKey: .recurrentLayers) ?? 40
        outputLayers = try c.decodeIfPresent(Int.self, forKey: .outputLayers) ?? 16
        loops = try c.decodeIfPresent(Int.self, forKey: .loops) ?? 2
        alphaInit = try c.decodeIfPresent(Float.self, forKey: .alphaInit) ?? 0.05
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.0177
        gateLogitScaleInit = try c.decodeIfPresent(Float.self, forKey: .gateLogitScaleInit)
        let rope = (try? c.decodeIfPresent([String: Rope].self, forKey: .ropeParameters)) ?? nil
        thetaSliding = rope?["sliding_attention"]?.ropeTheta ?? 10000
        thetaFull = rope?["full_attention"]?.ropeTheta ?? 10_000_000
        let n = layerCount
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< n).map { $0 % 2 == 1 ? "mlp" : ($0 % 8 == 6 ? "full_attention" : "sliding_attention") }
        // Only the per-pass cache layout is built: every published checkpoint
        // uses it, and `last_loop` rewinds cache offsets mid-forward.
        let mode = try c.decodeIfPresent(String.self, forKey: .cacheMode) ?? "per_loop"
        guard mode == "per_loop", layerTypes.count == n,
            inputLayers + recurrentLayers + outputLayers == n, loops > 0,
            headCount % kvHeadCount == 0
        else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported Dory configuration (\(mode))"))
        }
    }
}

/// Unit norm along the last axis, in float32.
private func unitNorm32(_ x: MLXArray) -> MLXArray {
    let f = x.asType(.float32)
    return f * rsqrt((f * f).sum(axis: -1, keepDims: true) + 1e-6)
}

private func unitNorm(_ x: MLXArray) -> MLXArray { unitNorm32(x).asType(x.dtype) }

private final class DoryAttention: Module {
    let c: DoryConfiguration
    let sliding: Bool
    let rope: RoPE
    let sqkFactor: Float
    let gateFactor: Float
    @ModuleInfo(key: "qkvg_proj") var qkvg: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "sqk") var sqk: MLXArray
    @ParameterInfo(key: "s_gate") var sGate: MLXArray

    init(_ c: DoryConfiguration, sliding: Bool) {
        self.c = c
        self.sliding = sliding
        let (h, k, d) = (c.headCount, c.kvHeadCount, c.headDim)
        rope = RoPE(dimensions: d, traditional: false, base: sliding ? c.thetaSliding : c.thetaFull)
        sqkFactor = sqrt(Float(d)) / c.initializerRange
        gateFactor = sqrt(c.gateLogitScaleInit ?? sqrt(Float(c.hiddenSize))) / c.initializerRange
        _qkvg.wrappedValue = Linear(c.hiddenSize, (2 * h + 2 * k) * d, bias: false)
        _oProj.wrappedValue = Linear(h * d, c.hiddenSize, bias: false)
        _sqk.wrappedValue = ones([h * d])
        _sGate.wrappedValue = ones([h * d])
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let (h, k, d) = (c.headCount, c.kvHeadCount, c.headDim)
        let y = qkvg(x)
        let n = (h + k) * d
        var qk = y[.ellipsis, ..<n].reshaped(b, t, h + k, d).transposed(0, 2, 1, 3)
        var v = y[.ellipsis, n ..< (n + k * d)].reshaped(b, t, k, d).transposed(0, 2, 1, 3)
        let g = y[.ellipsis, (n + k * d)...]
        // RoPE first, then the nGPT norm of every head: queries scaled by
        // sqk², keys by one.
        qk = applyRotaryPosition(rope, to: qk, offset: cache?.ropeOffset)
        let s = sqk.asType(.float32) * sqkFactor
        let qkScale = concatenated([(s * s).reshaped(1, h, 1, d), MLXArray.ones([1, k, 1, d], dtype: .float32)], axis: 1)
        qk = (unitNorm32(qk) * qkScale).asType(qk.dtype)
        let q = qk[0..., ..<h]
        var kk = qk[0..., h...]
        let mask = makeAttentionMask(n: t, cache: cache, windowSize: sliding ? c.slidingWindow : nil)
        if let cache {
            (kk, v) = cache.update(keys: kk, values: v)
        }
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: kk, values: v, scale: 1 / sqrt(Float(d)), mask: mask
        ).transposed(0, 2, 1, 3).reshaped(b, t, h * d)
        let sg = sGate.asType(.float32) * gateFactor
        let gated = unitNorm32(o.asType(.float32) * sigmoid(g.asType(.float32) * (sg * sg))).asType(o.dtype)
        return oProj(gated)
    }
}

private final class DoryMLP: Module {
    let intermediate: Int
    let scale: Float
    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ParameterInfo(key: "suv_gate") var suvGate: MLXArray
    @ParameterInfo(key: "suv_up") var suvUp: MLXArray

    init(_ c: DoryConfiguration) {
        intermediate = c.intermediateSize
        scale = sqrt(Float(c.hiddenSize))
        _gateUp.wrappedValue = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        _suvGate.wrappedValue = ones([c.intermediateSize])
        _suvUp.wrappedValue = ones([c.intermediateSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gu = gateUp(x)
        let g = gu[.ellipsis, ..<intermediate].asType(.float32) * (suvGate.asType(.float32) * scale)
        let u = gu[.ellipsis, intermediate...].asType(.float32) * (suvUp.asType(.float32) * scale)
        return down(unitNorm32(silu(g) * u).asType(gu.dtype))
    }
}

/// The slerp of nGPT: the block's output pulled into the residual by alpha,
/// both on the sphere, the result back on it.
private func slerpNorm(_ res: MLXArray, _ out: MLXArray, _ alpha: MLXArray) -> MLXArray {
    let a = unitNorm32(res)
    let b = unitNorm32(out)
    return unitNorm32(a + alpha * (b - a)).asType(res.dtype)
}

private final class DoryAttentionLayer: Module {
    let alphaScale: Float
    @ParameterInfo(key: "attn_alpha") var alpha: MLXArray
    @ModuleInfo(key: "mixer") var mixer: DoryAttention

    init(_ c: DoryConfiguration, sliding: Bool) {
        alphaScale = c.alphaInit / c.initializerRange
        _alpha.wrappedValue = ones([c.hiddenSize])
        _mixer.wrappedValue = DoryAttention(c, sliding: sliding)
    }

    func callAsFunction(_ h: MLXArray, cache: KVCache?) -> MLXArray {
        slerpNorm(h, mixer(h, cache: cache), abs(alpha.asType(.float32) * alphaScale))
    }
}

private final class DoryMLPLayer: Module {
    let alphaScale: Float
    @ParameterInfo(key: "mlp_alpha") var alpha: MLXArray
    @ModuleInfo(key: "mixer") var mixer: DoryMLP

    init(_ c: DoryConfiguration) {
        alphaScale = c.alphaInit / c.initializerRange
        _alpha.wrappedValue = ones([c.hiddenSize])
        _mixer.wrappedValue = DoryMLP(c)
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        slerpNorm(h, mixer(h), abs(alpha.asType(.float32) * alphaScale))
    }
}

private final class DoryBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo var layers: [Module]

    init(_ c: DoryConfiguration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = c.layerTypes.map {
            $0 == "mlp" ? DoryMLPLayer(c) : DoryAttentionLayer(c, sliding: $0 == "sliding_attention")
        }
    }
}

final class DoryModel: Module, LLMModel {
    let config: DoryConfiguration
    /// Each attention layer's caches, one per pass for the recurrent group:
    /// `[(layer, pass)] → index` into the model's cache list.
    private let plan: [Int: [Int]]
    private let cacheCount: Int
    @ModuleInfo(key: "backbone") private var backbone: DoryBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear
    @ParameterInfo(key: "logit_scale") private var logitScale: MLXArray
    var vocabularySize: Int { config.vocabularySize }
    var loraLayers: [Module] { backbone.layers }

    init(_ c: DoryConfiguration) {
        config = c
        var plan: [Int: [Int]] = [:]
        var n = 0
        let recurrent = c.inputLayers ..< (c.inputLayers + c.recurrentLayers)
        for (i, t) in c.layerTypes.enumerated() where t != "mlp" {
            let passes = recurrent.contains(i) ? c.loops : 1
            plan[i] = (0 ..< passes).map { _ in defer { n += 1 }; return n }
        }
        self.plan = plan
        cacheCount = n
        _backbone.wrappedValue = DoryBackbone(c)
        _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        _logitScale.wrappedValue = ones([c.vocabularySize])
    }

    private func run(_ i: Int, _ h: MLXArray, pass: Int, cache: [KVCache]?) -> MLXArray {
        let layer = backbone.layers[i]
        if let a = layer as? DoryAttentionLayer {
            return a(h, cache: cache?[plan[i]![pass]])
        }
        return (layer as! DoryMLPLayer)(h)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let c = config
        let lo = c.inputLayers
        let hi = c.inputLayers + c.recurrentLayers
        var h = backbone.embeddings(inputs)
        for i in 0 ..< lo { h = run(i, h, pass: 0, cache: cache) }
        let input = h
        var state = input
        for pass in 0 ..< c.loops {
            h = unitNorm(state + input)
            for i in lo ..< hi { h = run(i, h, pass: pass, cache: cache) }
            state = unitNorm(h)
        }
        h = state
        for i in hi ..< c.layerCount { h = run(i, h, pass: 0, cache: cache) }
        let scale = logitScale.asType(.float32) / c.initializerRange
        return (lmHead(h).asType(.float32) * scale).asType(h.dtype)
    }

    func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        var w = checkpoint.weights
        if let s = w.removeValue(forKey: "lm_head.logit_scale") { w["logit_scale"] = s }
        // An unfused export: q/k/v/g and gate/up into the fused projections.
        for key in Array(w.keys) where key.hasSuffix(".mixer.q_proj.weight") {
            let p = String(key.dropLast("q_proj.weight".count))
            let parts = ["q_proj", "k_proj", "v_proj", "g_proj"].compactMap { w.removeValue(forKey: p + $0 + ".weight") }
            if parts.count == 4 { w[p + "qkvg_proj.weight"] = concatenated(parts, axis: 0) }
        }
        for key in Array(w.keys) where key.hasSuffix(".mixer.gate_proj.weight") {
            let p = String(key.dropLast("gate_proj.weight".count))
            if let g = w.removeValue(forKey: p + "gate_proj.weight"), let u = w.removeValue(forKey: p + "up_proj.weight") {
                w[p + "gate_up_proj.weight"] = concatenated([g, u], axis: 0)
            }
        }
        checkpoint.weights = w
        return checkpoint
    }

    func newCache(parameters: GenerateParameters? = nil) throws -> [KVCache] {
        var caches = [KVCache?](repeating: nil, count: cacheCount)
        for (i, slots) in plan {
            for slot in slots {
                caches[slot] = try makeHybridAttentionKVCache(
                    parameters: parameters, slidingWindow: config.slidingWindow,
                    usesSlidingWindow: config.layerTypes[i] == "sliding_attention")
            }
        }
        return caches.map { $0! }
    }
}

/// Teach the sidecar's factory about Dory. Idempotent.
public enum DoryRegistration {
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType(
            "dory",
            creator: { data in
                DoryModel(try JSONDecoder().decode(DoryConfiguration.self, from: data))
            })
    }
}
