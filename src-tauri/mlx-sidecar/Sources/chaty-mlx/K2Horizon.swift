// K2 Horizon — a `k2_horizon` implementation for the MLX sidecar.
//
// Neither mlx-swift-lm nor upstream mlx-lm implements this architecture
// (mlx-lm issue #1876), so the sidecar answered every K2 checkpoint with
// `unsupportedModelType("k2_horizon")`. The reference is IFM's own
// `modeling_k2_horizon.py`; every departure from a Llama block below is taken
// from it, not from a port:
//
//   • grouped RMSNorm — each hidden vector is split into `layernorm_num_groups`
//     contiguous groups, each normalised on its own (in fp32), and one
//     full-width weight applied after;
//   • an optional attention output gate — `gate_proj(x)` through silu or
//     softplus(β = ln 2), multiplied into the attention output per element
//     before `o_proj`;
//   • optional per-head q/k norms — the same grouped norm, one group per head;
//   • sparse layers — a sigmoid (or softmax) router whose bias only CHOOSES
//     experts, never weights them, with shared experts added on top;
//   • MoVA — on sparse layers of the MoVA checkpoints the value projection is
//     itself a routed mixture: silu of each chosen value expert, weighted and
//     summed, renormalised only when more than one is chosen.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct K2HorizonConfiguration: Decodable, Sendable {
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var ropeHeadDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var normGroups: Int
    var numExperts: Int
    var expertsPerToken: Int
    var sharedExperts: Int
    var movaExperts: Int
    var movaExpertsPerToken: Int
    var sparseStep: Int
    var mlpOnlyLayers: [Int]
    var normTopkProb: Bool
    var routerScaling: Float
    var routerScoreFunc: String
    var moeGateBias: Bool
    var queryKeyNorm: Bool
    var attentionGateFunc: String?
    var attentionBias: Bool
    var tieWordEmbeddings: Bool
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int

    /// The reference's own test: a layer is sparse when it is not listed as
    /// MLP-only, experts exist, and it falls on the sparse step.
    func isSparse(_ i: Int) -> Bool {
        !mlpOnlyLayers.contains(i) && numExperts > 0 && (i + 1) % max(sparseStep, 1) == 0
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case ropeHeadDim = "rope_head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case normGroups = "layernorm_num_groups"
        case numExperts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case sharedExperts = "num_shared_experts"
        case movaExperts = "mova_num_experts"
        case movaExpertsPerToken = "mova_num_experts_per_tok"
        case sparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case normTopkProb = "norm_topk_prob"
        case routerScaling = "router_scaling_factor"
        case routerScoreFunc = "router_score_func"
        case moeGateBias = "moe_gate_bias"
        case queryKeyNorm = "query_key_norm"
        case attentionGateFunc = "attention_gate_func"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / attentionHeads)
        ropeHeadDim = try c.decodeIfPresent(Int.self, forKey: .ropeHeadDim) ?? headDim
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        normGroups = try c.decodeIfPresent(Int.self, forKey: .normGroups) ?? 1
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 0
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
        movaExperts = try c.decodeIfPresent(Int.self, forKey: .movaExperts) ?? 0
        movaExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .movaExpertsPerToken) ?? 0
        sparseStep = try c.decodeIfPresent(Int.self, forKey: .sparseStep) ?? 1
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routerScaling = try c.decodeIfPresent(Float.self, forKey: .routerScaling) ?? 1
        routerScoreFunc = try c.decodeIfPresent(String.self, forKey: .routerScoreFunc) ?? "sigmoid"
        moeGateBias = try c.decodeIfPresent(Bool.self, forKey: .moeGateBias) ?? false
        queryKeyNorm = try c.decodeIfPresent(Bool.self, forKey: .queryKeyNorm) ?? false
        attentionGateFunc = try c.decodeIfPresent(String.self, forKey: .attentionGateFunc)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        // Newer configs state the rotary settings in `rope_parameters`
        // (theta and, for YaRN and the like, the scaling); older ones in
        // `rope_theta` + `rope_scaling`.
        let params: [String: StringOrNumber]? =
            try? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        let theta: Float? = params?["rope_theta"]?.asFloat()
        ropeTheta = try theta ?? c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        var scaling: [String: StringOrNumber]? =
            try? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if scaling == nil, let p = params, case .string(let t)? = p["rope_type"], t != "default" {
            scaling = p
        }
        ropeScaling = scaling
    }
}

/// K2's grouped RMSNorm: `groups` contiguous slices of the last axis, each
/// normalised in fp32, then one full-width weight — the reference multiplies
/// in fp32 and casts back, and so does this.
final class K2GroupedRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let groups: Int
    let eps: Float

    init(dimensions: Int, groups: Int, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.groups = max(groups, 1)
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let last = shape[shape.count - 1]
        var g = x.asType(.float32).reshaped(Array(shape.dropLast()) + [groups, last / groups])
        g = g * rsqrt(mean(g * g, axis: -1, keepDims: true) + eps)
        return (weight.asType(.float32) * g.reshaped(shape)).asType(x.dtype)
    }
}

/// Top-k routing as the reference does it: scores in fp32, the bias added only
/// for choosing, the unbiased scores of the chosen experts as their weights,
/// optionally renormalised, then scaled.
func k2Route(
    _ logits: MLXArray, bias: MLXArray?, scoreFunc: String, topK: Int, scaling: Float,
    renormalize: Bool
) -> (weights: MLXArray, indices: MLXArray) {
    let l32 = logits.asType(.float32)
    let scores = scoreFunc == "softmax" ? softmax(l32, axis: -1, precise: true) : sigmoid(l32)
    var choice = scores
    if let bias { choice = choice + bias.asType(.float32) }
    let inds = stopGradient(argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
    var w = takeAlong(scores, inds, axis: -1)
    if renormalize { w = w / w.sum(axis: -1, keepDims: true) }
    return ((w * scaling).asType(logits.dtype), inds)
}

final class K2MLP: Module, UnaryLayer {
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

final class K2SparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Linear
    /// The router's correction bias: it takes part in choosing experts only.
    @ParameterInfo(key: "gate_bias") var gateBias: MLXArray?
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: K2MLP?

    let args: K2HorizonConfiguration

    init(_ args: K2HorizonConfiguration) {
        self.args = args
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _gateBias.wrappedValue = args.moeGateBias ? MLXArray.zeros([args.numExperts]) : nil
        _experts.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedExperts.wrappedValue =
            args.sharedExperts > 0
            ? K2MLP(args.hiddenSize, args.moeIntermediateSize * args.sharedExperts) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (w, inds) = k2Route(
            gate(x), bias: gateBias, scoreFunc: args.routerScoreFunc,
            topK: args.expertsPerToken, scaling: args.routerScaling,
            renormalize: args.normTopkProb)
        var y = (experts(x, inds) * w[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)
        if let sharedExperts { y = y + sharedExperts(x) }
        return y
    }
}

final class K2Attention: Module {
    let args: K2HorizonConfiguration
    let scale: Float
    let mova: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: K2GroupedRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: K2GroupedRMSNorm?
    @ModuleInfo(key: "v_router") var vRouter: Linear?
    @ParameterInfo(key: "v_router_bias") var vRouterBias: MLXArray?
    @ModuleInfo(key: "v_experts") var vExperts: SwitchLinear?

    let rope: RoPELayer

    init(_ args: K2HorizonConfiguration, mova: Bool) {
        self.args = args
        self.mova = mova
        self.scale = pow(Float(args.headDim), -0.5)
        let dim = args.hiddenSize
        let qWidth = args.attentionHeads * args.headDim
        let kvWidth = args.kvHeads * args.headDim
        _qProj.wrappedValue = Linear(dim, qWidth, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(dim, kvWidth, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(qWidth, dim, bias: args.attentionBias)
        if mova {
            _vRouter.wrappedValue = Linear(dim, args.movaExperts, bias: false)
            _vRouterBias.wrappedValue = args.moeGateBias ? MLXArray.zeros([args.movaExperts]) : nil
            _vExperts.wrappedValue = SwitchLinear(
                inputDims: dim, outputDims: kvWidth, numExperts: args.movaExperts, bias: false)
        } else {
            _vProj.wrappedValue = Linear(dim, kvWidth, bias: args.attentionBias)
        }
        if args.attentionGateFunc != nil {
            _gateProj.wrappedValue = Linear(dim, qWidth, bias: false)
        }
        if args.queryKeyNorm {
            _qNorm.wrappedValue = K2GroupedRMSNorm(
                dimensions: qWidth, groups: args.attentionHeads, eps: args.rmsNormEps)
            _kNorm.wrappedValue = K2GroupedRMSNorm(
                dimensions: kvWidth, groups: args.kvHeads, eps: args.rmsNormEps)
        }
        self.rope = initializeRope(
            dims: args.ropeHeadDim, base: args.ropeTheta, traditional: false,
            scalingConfig: args.ropeScaling, maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    /// MoVA: the value is a routed mixture of value experts, each through silu.
    private func movaValues(_ x: MLXArray) -> MLXArray {
        guard let vRouter, let vExperts else { return x }
        let k = args.movaExpertsPerToken
        let (w, inds) = k2Route(
            vRouter(x), bias: vRouterBias, scoreFunc: args.routerScoreFunc, topK: k,
            scaling: args.routerScaling, renormalize: k > 1)
        var xe = expandedDimensions(x, axes: [-2, -3])
        let doSort = inds.size >= 64
        var idx = inds
        var inverse = MLXArray()
        if doSort { (xe, idx, inverse) = gatherSort(x: xe, indices: inds) }
        var v = vExperts(xe, idx, sortedIndices: doSort)
        if doSort { v = scatterUnsort(x: v, invOrder: inverse, shape: inds.shape) }
        v = v.squeezed(axis: -2)
        return (silu(v) * w[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x)
        var k = kProj(x)
        let v = mova ? movaValues(x) : vProj!(x)
        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        var queries = q.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var keys = k.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = v.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var out = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if let gateProj, let fn = args.attentionGateFunc {
            let g = gateProj(x)
            if fn == "silu" {
                out = out * silu(g)
            } else {
                // softplus with β = ln 2: log(1 + e^{βx}) / β, written stably.
                let beta = Float(log(2.0))
                let z = g.asType(.float32) * beta
                let sp = (maximum(z, 0) + log1p(exp(-abs(z)))) / beta
                out = out * sp.asType(out.dtype)
            }
        }
        return oProj(out)
    }
}

final class K2Block: Module {
    @ModuleInfo(key: "self_attn") var attention: K2Attention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputNorm: K2GroupedRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: K2GroupedRMSNorm

    init(_ args: K2HorizonConfiguration, layer: Int) {
        let sparse = args.isSparse(layer)
        _attention.wrappedValue = K2Attention(args, mova: sparse && args.movaExperts > 0)
        _mlp.wrappedValue =
            sparse ? K2SparseMoE(args) : K2MLP(args.hiddenSize, args.intermediateSize)
        _inputNorm.wrappedValue = K2GroupedRMSNorm(
            dimensions: args.hiddenSize, groups: args.normGroups, eps: args.rmsNormEps)
        _postNorm.wrappedValue = K2GroupedRMSNorm(
            dimensions: args.hiddenSize, groups: args.normGroups, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
}

final class K2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [K2Block]
    let norm: K2GroupedRMSNorm

    init(_ args: K2HorizonConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { K2Block(args, layer: $0) }
        self.norm = K2GroupedRMSNorm(
            dimensions: args.hiddenSize, groups: args.normGroups, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class K2HorizonModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: K2ModelInner
    let config: K2HorizonConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: K2HorizonConfiguration) {
        self.config = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = K2ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        return lmHead?(h) ?? model.embedTokens.asLinear(h)
    }

    /// Both layouts a K2 folder arrives in. An mlx-lm conversion is already in
    /// this shape. A checkpoint straight from IFM keeps one tensor per expert
    /// and states the routers' correction biases as the routers' own `bias` —
    /// which is not a bias of the logits here, only of the choice.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if config.tieWordEmbeddings { w["lm_head.weight"] = nil }
        for layer in 0 ..< config.hiddenLayers where config.isSparse(layer) {
            let base = "model.layers.\(layer)"
            if let b = w.removeValue(forKey: "\(base).mlp.gate.bias") {
                w["\(base).mlp.gate_bias"] = b
            }
            if let b = w.removeValue(forKey: "\(base).self_attn.v_router.bias") {
                w["\(base).self_attn.v_router_bias"] = b
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for part in ["weight", "scales", "biases"] {
                    let first = "\(base).mlp.experts.0.\(proj).\(part)"
                    guard w[first] != nil else { continue }
                    let parts = (0 ..< config.numExperts).compactMap {
                        w.removeValue(forKey: "\(base).mlp.experts.\($0).\(proj).\(part)")
                    }
                    w["\(base).mlp.experts.\(proj).\(part)"] = stacked(parts)
                }
            }
            for part in ["weight", "scales", "biases"] {
                let first = "\(base).self_attn.v_experts.0.\(part)"
                guard w[first] != nil else { continue }
                let parts = (0 ..< config.movaExperts).compactMap {
                    w.removeValue(forKey: "\(base).self_attn.v_experts.\($0).\(part)")
                }
                w["\(base).self_attn.v_experts.\(part)"] = stacked(parts)
            }
        }
        return w
    }
}

extension K2HorizonModel {
    public var loraLayers: [Module] { model.layers }
}

/// A partial rotary embedding (`rope_head_dim` < `head_dim`) is not
/// implemented — no released checkpoint uses one — so such a config is refused
/// by name rather than run wrongly.
public struct K2HorizonUnsupported: LocalizedError {
    let detail: String
    public var errorDescription: String? { detail }
}

/// Teach the sidecar's factory about `k2_horizon`. Idempotent, and safe to
/// call before every load.
public enum K2HorizonRegistration {
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType(
            "k2_horizon",
            creator: { data in
                let config = try JSONDecoder().decode(K2HorizonConfiguration.self, from: data)
                if config.ropeHeadDim != config.headDim {
                    throw K2HorizonUnsupported(
                        detail:
                            "K2 Horizon with a partial rotary embedding (rope_head_dim \(config.ropeHeadDim) < head_dim \(config.headDim)) is not supported yet"
                    )
                }
                return K2HorizonModel(config)
            })
    }
}
