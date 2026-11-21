// Inkling 2 — `inkling2` and `inkling2_loop` for the MLX sidecar.
//
// mlx-swift-lm does not carry Thinking Machines' Inkling 2, and mlx-lm has it
// only in an open pull request (ml-explore/mlx-lm #2061, #2086). That PR's
// `mlx_lm/models/inkling2` is the reference here, followed function for
// function; the departures are named where they are made.
//
//   • attention with a learned relative bias over the last `rel_extent`
//     positions, a per-token sink logit for each head, L2-clipped queries and
//     keys, an unweighted RMS norm on the values, a sigmoid output gate and,
//     on full-attention layers, a log-length query scale;
//   • short causal convolutions — on the keys and values (static on sliding
//     layers, with per-token weights on full ones) and after attention and the
//     MLP — each keeping its last `kernel − 1` inputs as state;
//   • attention residuals: every 6 layers the stream is cut into a block, and
//     each layer's attention and MLP read a learned softmax mix of the blocks
//     so far rather than a plain residual;
//   • a sparse MoE of 128 experts, 8 per token, sigmoid routing whose bias
//     only picks, softcapped SwiGLU, its experts in NVFP4 as published;
//   • Inkling 2 Mini runs layers 6–17 three times; from the second pass each
//     layer's q, k, v and MLP inputs are gated with a message read from the
//     blocks the previous pass left. Each pass keeps a cache of its own.
//
// Pictures go through the vision tower in Inkling2Vision.swift; their rows
// replace the embeddings of the image tokens the prompt reserved for them.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Inkling2LoopConfig: Decodable, Sendable {
    var coreStart: Int
    var coreEnd: Int
    var numPasses: Int
    var gateRank: Int

    enum CodingKeys: String, CodingKey {
        case coreStart = "core_start"
        case coreEnd = "core_end"
        case numPasses = "num_passes"
        case gateRank = "gate_rank"
    }
}

public struct Inkling2TextConfiguration: Decodable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var vocabularySize: Int
    var unpaddedVocabularySize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var swaAttentionHeads: Int
    var swaKvHeads: Int
    var swaHeadDim: Int
    var slidingWindow: Int
    var localLayerIds: [Int]
    var dRel: Int
    var relExtent: Int
    var logScalingNFloor: Float
    var logScalingAlpha: Float
    var rmsNormEps: Float
    var qkL2Clip: Float
    var relL2Clip: Float
    var sconvKernel: Int
    var layersPerBlock: Int
    var routedExperts: Int
    var expertsPerToken: Int
    var intermediateSize: Int
    var mlpGateSoftcap: Float
    var mlpUpSoftcap: Float
    var routeScale: Float
    var finalLogitSoftcapping: Float
    var loop: Inkling2LoopConfig?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKvHeads = "swa_num_key_value_heads"
        case swaHeadDim = "swa_head_dim"
        case slidingWindow = "sliding_window_size"
        case localLayerIds = "local_layer_ids"
        case dRel = "d_rel"
        case relExtent = "rel_extent"
        case logScalingNFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps"
        case qkL2Clip = "qk_l2_clip_threshold"
        case relL2Clip = "rel_l2_clip_threshold"
        case sconvKernel = "sconv_kernel_size"
        case layersPerBlock = "n_layers_per_attn_res_block"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case intermediateSize = "intermediate_size"
        case mlpGateSoftcap = "mlp_gate_softcap"
        case mlpUpSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case loop = "loop_config"
    }

    var loopCore: Range<Int> { (loop?.coreStart ?? 0) ..< (loop?.coreEnd ?? 0) }

    /// Every layer the stream passes through, in order: the loop's layers
    /// once for each pass. One cache slot each.
    var layerVisits: [Int] {
        guard let loop else { return Array(0 ..< hiddenLayers) }
        let core = Array(loop.coreStart ..< loop.coreEnd)
        return Array(0 ..< loop.coreStart)
            + Array(repeating: core, count: loop.numPasses).flatMap { $0 }
            + Array(loop.coreEnd ..< hiddenLayers)
    }
}

public struct Inkling2Configuration: Decodable, Sendable {
    var modelType: String
    var text: Inkling2TextConfiguration
    var vision: Inkling2VisionConfiguration?
    var imageTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case imageTokenId = "image_token_id"
    }
}

// MARK: - Helpers

/// RMS norm with no weight, in float32 (mlx.fast.rms_norm with `None`).
private func unweightedRMSNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let x = x.asType(.float32)
    return x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps)
}

/// Scale, then clip each vector's L2 norm to `threshold`; in float32.
private func l2Clip(_ states: MLXArray, scale: MLXArray?, threshold: Float) -> MLXArray {
    var s = states.asType(.float32)
    if let scale { s = s * scale.asType(.float32) }
    let norm = sqrt((s * s).sum(axis: -1, keepDims: true))
    return (s * minimum(threshold / (norm + 1e-6), 1)).asType(states.dtype)
}

// MARK: - Short convolutions

/// The convolution's input: its kept state, then the new tokens, in float32.
private func convWindow(_ x: MLXArray, state: MLXArray?, kernel: Int) -> MLXArray {
    let x = x.asType(.float32)
    let state = state ?? MLXArray.zeros([x.dim(0), kernel - 1, x.dim(-1)], type: Float.self)
    return concatenated([state, x], axis: 1)
}

/// What the convolution keeps of `window`: its last `n` positions, copied.
private func convState(_ window: MLXArray, n: Int) -> MLXArray {
    let L = window.dim(1)
    return window[0..., (L - n) ..< L, 0...] + 0
}

final class Inkling2ShortConv: Module {
    let kernel: Int
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(dims: Int, kernel: Int) {
        self.kernel = kernel
        _conv1d.wrappedValue = Conv1d(
            inputChannels: dims, outputChannels: dims, kernelSize: kernel, groups: dims, bias: false)
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let residual = x.asType(.float32)
        let window = convWindow(x, state: state, kernel: kernel)
        // As the reference, the convolution runs in the weight's dtype.
        let out = conv1d(window.asType(conv1d.weight.dtype)).asType(.float32)
        return (((out + residual) / Float(kernel + 1).squareRoot()).asType(x.dtype), window)
    }
}

final class Inkling2DynamicShortConv: Module {
    let kernel: Int
    let heads: Int
    @ModuleInfo(key: "static_weight") var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") var weightProj: Linear

    init(dims: Int, inputDims: Int, heads: Int, kernel: Int) {
        self.kernel = kernel
        self.heads = heads
        _staticWeight.wrappedValue = MLXArray.zeros([kernel, dims])
        _weightProj.wrappedValue = Linear(inputDims, heads * kernel, bias: false)
    }

    func callAsFunction(_ x: MLXArray, dynamicInput: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let (B, L, C) = (x.dim(0), x.dim(1), x.dim(2))
        let K = kernel
        let dynamic = weightProj(dynamicInput).asType(.float32).reshaped(B, L, heads, 1, K)
        let stat = staticWeight.asType(.float32).transposed().reshaped(heads, -1, K)
        let weights = stat + dynamic
        let window = convWindow(x, state: state, kernel: K)
        let windows = stacked(
            (0 ..< K).map { lag in window[0..., (K - 1 - lag) ..< (K - 1 - lag + L), 0...] }, axis: -1
        ).reshaped(B, L, heads, -1, K)
        var out = (windows * weights).sum(axis: -1).reshaped(B, L, C)
        out = (out + x.asType(.float32)) / Float(2 * K + 1).squareRoot()
        return (out.asType(x.dtype), window)
    }
}

// MARK: - Attention

final class Inkling2ScaledL2Clip: Module {
    let threshold: Float
    @ModuleInfo(key: "scale") var scale: MLXArray

    init(heads: Int, dim: Int, threshold: Float) {
        self.threshold = threshold
        _scale.wrappedValue = MLXArray.ones([heads, dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { l2Clip(x, scale: scale, threshold: threshold) }
}

final class Inkling2RelativeLogits: Module {
    @ModuleInfo(key: "proj") var proj: MLXArray

    init(dRel: Int, extent: Int) {
        _proj.wrappedValue = MLXArray.zeros([dRel, extent])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { matmul(x, proj.asType(x.dtype)) }
}

final class Inkling2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int
    let scale: Float
    let eps: Float
    let kClip: Float
    let full: Bool
    let logFloor: Float
    let logAlpha: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "sink_proj") var sinkProj: Linear
    @ModuleInfo(key: "rel_logits_proj") var relLogits: Inkling2RelativeLogits
    @ModuleInfo(key: "q_l2_clip") var qClip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "rel_l2_clip") var relClip: Inkling2ScaledL2Clip
    // Sliding layers: static convolutions; full layers: per-token weights.
    @ModuleInfo(key: "k_sconv") var kConv: Module
    @ModuleInfo(key: "v_sconv") var vConv: Module

    init(_ a: Inkling2TextConfiguration, sliding: Bool) {
        full = !sliding
        heads = sliding ? a.swaAttentionHeads : a.attentionHeads
        kvHeads = sliding ? a.swaKvHeads : a.kvHeads
        headDim = sliding ? a.swaHeadDim : a.headDim
        window = sliding ? a.slidingWindow : 0
        let extent = sliding ? a.slidingWindow : a.relExtent
        scale = 1 / Float(headDim).squareRoot()
        eps = a.rmsNormEps
        kClip = a.qkL2Clip
        logFloor = a.logScalingNFloor
        logAlpha = a.logScalingAlpha
        let dim = a.hiddenSize
        let qDim = heads * headDim
        let kvDim = kvHeads * headDim
        _qProj.wrappedValue = Linear(dim, qDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvDim, bias: false)
        _rProj.wrappedValue = Linear(dim, heads * a.dRel, bias: false)
        _oProj.wrappedValue = Linear(qDim, dim, bias: false)
        _gateProj.wrappedValue = Linear(dim, qDim, bias: false)
        _sinkProj.wrappedValue = Linear(dim, heads, bias: false)
        _relLogits.wrappedValue = Inkling2RelativeLogits(dRel: a.dRel, extent: extent)
        _qClip.wrappedValue = Inkling2ScaledL2Clip(heads: heads, dim: headDim, threshold: a.qkL2Clip)
        _relClip.wrappedValue = Inkling2ScaledL2Clip(heads: heads, dim: a.dRel, threshold: a.relL2Clip)
        if sliding {
            _kConv.wrappedValue = Inkling2ShortConv(dims: kvDim, kernel: a.sconvKernel)
            _vConv.wrappedValue = Inkling2ShortConv(dims: kvDim, kernel: a.sconvKernel)
        } else {
            _kConv.wrappedValue = Inkling2DynamicShortConv(
                dims: kvDim, inputDims: dim, heads: kvHeads, kernel: a.sconvKernel)
            _vConv.wrappedValue = Inkling2DynamicShortConv(
                dims: kvDim, inputDims: dim, heads: kvHeads, kernel: a.sconvKernel)
        }
    }

    private func conv(_ m: Module, _ x: MLXArray, input: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        if let c = m as? Inkling2ShortConv { return c(x, state: state) }
        return (m as! Inkling2DynamicShortConv)(x, dynamicInput: input, state: state)
    }

    func callAsFunction(
        _ hidden: MLXArray, keyHidden: MLXArray, valueHidden: MLXArray,
        cache: KVCache?, convStates: inout [MLXArray?]
    ) -> MLXArray {
        let (B, L) = (hidden.dim(0), hidden.dim(1))
        let (k0, kw) = conv(kConv, kProj(keyHidden), input: keyHidden, state: convStates[0])
        let (v0, vw) = conv(vConv, vProj(valueHidden), input: valueHidden, state: convStates[1])
        convStates[0] = kw
        convStates[1] = vw

        let gate = gateProj(hidden)
        let sinks = sinkProj(hidden)
        var q = l2Clip(qProj(hidden).reshaped(B, L, heads, -1), scale: qClip.scale, threshold: qClip.threshold)
        let k = l2Clip(k0.reshaped(B, L, kvHeads, -1), scale: nil, threshold: kClip)
        let v = unweightedRMSNorm(v0.reshaped(B, L, kvHeads, -1), eps: eps).asType(v0.dtype)
        var rel = l2Clip(rProj(hidden).reshaped(B, L, heads, -1), scale: relClip.scale, threshold: relClip.threshold)
        let offset = cache?.offset ?? 0
        if full {
            let pos = MLXArray(Int32(offset) ..< Int32(offset + L)).asType(.float32)
            let tau = 1 + logAlpha * log(maximum((pos + 1) / logFloor, 1))
            let t = tau.reshaped(L, 1, 1)
            q = (q.asType(.float32) * t).asType(q.dtype)
            rel = (rel.asType(.float32) * t).asType(rel.dtype)
        }
        let bias = relLogits(rel)  // [B, L, H, R]

        let qh = q.transposed(0, 2, 1, 3)
        var kh = k.transposed(0, 2, 1, 3)
        var vh = v.transposed(0, 2, 1, 3)
        if let cache { (kh, vh) = cache.update(keys: kh, values: vh) }
        let out = attend(qh, kh, vh, bias: bias, sinks: sinks, end: offset + L)

        let o = out.reshaped(B, L, -1).asType(.float32) * sigmoid(gate.asType(.float32))
        return oProj(o.asType(gate.dtype))
    }

    /// Keys sit at positions `end − S ..< end`, queries at `end − L ..< end`
    /// (every slot here keeps its whole history; the window is a mask).
    private func attend(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, sinks: MLXArray, end: Int
    ) -> MLXArray {
        let (B, L, D) = (q.dim(0), q.dim(2), q.dim(3))
        let S = k.dim(2)
        let R = bias.dim(-1)
        let qPos = MLXArray(Int32(end - L) ..< Int32(end))
        let kPos = MLXArray(Int32(end - S) ..< Int32(end))
        let distance = qPos.reshaped(L, 1) - kPos.reshaped(1, S)  // [L, S]
        var visible = (distance .>= 0) .&& (kPos.reshaped(1, S) .>= 0)
        if window > 0 { visible = visible .&& (distance .< Int32(window)) }
        let index = clip(distance, min: 0, max: R - 1)

        if L == 1 && B == 1 {
            // The fused attention takes the sinks of one row.
            let b = bias[0, 0]  // [H, R]
            var m = take(b, index.reshaped(S), axis: -1)  // [H, S]
            m = which(distance.reshaped(1, S) .< Int32(R), m, MLXArray.zeros(like: m))
            let mask = which(visible.reshaped(1, S), m, MLXArray(-Float.infinity)).asType(q.dtype)
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale,
                mask: mask.reshaped(1, heads, 1, S), sinks: sinks[0, 0].asType(q.dtype))
            return out.transposed(0, 2, 1, 3)
        }

        let groups = heads / kvHeads
        // The relative bias of each query and key, by distance, 0 beyond the
        // extent; -inf where the key is out of sight.
        let bh = bias.transposed(0, 2, 1, 3)  // [B, H, L, R]
        var gathered = takeAlong(bh, broadcast(index.reshaped(1, 1, L, S), to: [B, heads, L, S]), axis: -1)
        gathered = which(distance.reshaped(1, 1, L, S) .< Int32(R), gathered.asType(.float32), MLXArray(Float(0)))
        let mask = which(visible.reshaped(1, 1, L, S), gathered, MLXArray(-Float.infinity))
            .reshaped(B, kvHeads, groups, L, S)

        let qf = q.asType(.float32).reshaped(B, kvHeads, groups, L, D)
        let kf = k.asType(.float32).expandedDimensions(axis: 2)
        let logits = matmul(qf, kf.swappedAxes(-1, -2)) * scale + mask
        let sk = sinks.asType(.float32).transposed(0, 2, 1).reshaped(B, kvHeads, groups, L, 1)
        let weights = softmax(concatenated([logits, sk], axis: -1), axis: -1)[.ellipsis, 0 ..< S]
        let out = matmul(weights.asType(v.dtype), v.expandedDimensions(axis: 2))
        return out.reshaped(B, heads, L, D).transposed(0, 2, 1, 3)
    }
}

// MARK: - MoE

/// How a layer's experts are stored, read from the checkpoint's tensors:
/// NVFP4 (4 bits, a UE4M3 scale per 16, and per expert a float scale) on most
/// layers of the published Inkling 2 Mini, MXFP8 (8 bits, an E8M0 scale per
/// 32) on its last four; mlx-lm reads the mode from the same shapes.
public struct Inkling2ExpertQuant: Sendable, Equatable {
    var bits: Int
    var groupSize: Int
    var globalScale: Bool

    var mode: QuantizationMode? {
        switch (bits, groupSize) {
        case (4, 16): return .nvfp4
        case (4, 32): return .mxfp4
        case (8, 32): return .mxfp8
        default: return nil
        }
    }
}

/// A stack of experts in one of MLX's floating-point quantizations. The
/// gathered product is MLX's; NVFP4's per-expert scale — which `gather_qmm`
/// here does not take — is applied after it, as MLX's dequantization applies
/// it (scale / 6 / 448). Not `Quantizable`: the loader's affine pass must leave
/// it as it is.
final class Inkling2FPExperts: Module {
    let quant: Inkling2ExpertQuant
    let mode: QuantizationMode
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "global_scale") var globalScale: MLXArray?

    init(inputDims: Int, outputDims: Int, experts: Int, quant: Inkling2ExpertQuant, mode: QuantizationMode) {
        self.quant = quant
        self.mode = mode
        _weight.wrappedValue = MLXArray.zeros(
            [experts, outputDims, inputDims * quant.bits / 32], type: UInt32.self)
        _scales.wrappedValue = MLXArray.zeros([experts, outputDims, inputDims / quant.groupSize], type: UInt8.self)
        _globalScale.wrappedValue = quant.globalScale ? MLXArray.ones([experts], type: Float.self) : nil
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        var y = gatherQuantizedMM(
            x, weight, scales: scales, biases: nil, rhsIndices: indices, transpose: true,
            groupSize: quant.groupSize, bits: quant.bits, mode: mode, sortedIndices: sorted)
        if let globalScale {
            let g = (globalScale[indices] / Float(6 * 448)).asType(y.dtype)
            y = y * g.reshaped(g.shape + [1, 1])
        }
        return y
    }
}

final class Inkling2Experts: Module {
    let gateCap: Float
    let upCap: Float
    @ModuleInfo(key: "gate_up_proj") var gateUp: Module
    @ModuleInfo(key: "down_proj") var down: Module

    init(_ a: Inkling2TextConfiguration, quant: Inkling2ExpertQuant?) {
        gateCap = a.mlpGateSoftcap
        upCap = a.mlpUpSoftcap
        if let quant, let mode = quant.mode {
            _gateUp.wrappedValue = Inkling2FPExperts(
                inputDims: a.hiddenSize, outputDims: 2 * a.intermediateSize, experts: a.routedExperts,
                quant: quant, mode: mode)
            _down.wrappedValue = Inkling2FPExperts(
                inputDims: a.intermediateSize, outputDims: a.hiddenSize, experts: a.routedExperts,
                quant: quant, mode: mode)
        } else {
            _gateUp.wrappedValue = SwitchLinear(
                inputDims: a.hiddenSize, outputDims: 2 * a.intermediateSize, numExperts: a.routedExperts,
                bias: false)
            _down.wrappedValue = SwitchLinear(
                inputDims: a.intermediateSize, outputDims: a.hiddenSize, numExperts: a.routedExperts,
                bias: false)
        }
    }

    private func apply(_ m: Module, _ x: MLXArray, _ idx: MLXArray, sorted: Bool) -> MLXArray {
        if let e = m as? Inkling2FPExperts { return e(x, idx, sorted: sorted) }
        return (m as! SwitchLinear)(x, idx, sortedIndices: sorted)
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = x.expandedDimensions(axes: [-2, -3])
        let sort = indices.size >= 64
        var idx = indices
        var inverse: MLXArray? = nil
        if sort { (x, idx, inverse) = gatherSort(x: x, indices: indices) }
        let gu = apply(gateUp, x, idx, sorted: sort)
        // Rows alternate gate, up — as in the checkpoint.
        let g = gu[.ellipsis, .stride(from: 0, by: 2)].asType(.float32)
        let u = gu[.ellipsis, .stride(from: 1, by: 2)].asType(.float32)
        let act = (tanh(g / gateCap) * gateCap * sigmoid(g) * (tanh(u / upCap) * upCap)).asType(gu.dtype)
        var y = apply(down, act, idx, sorted: sort)
        if let inverse { y = scatterUnsort(x: y, invOrder: inverse, shape: indices.shape) }
        return y.squeezed(axis: -2)
    }
}

final class Inkling2Router: Module {
    let topK: Int
    let routeScale: Float
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") var bias: MLXArray

    init(_ a: Inkling2TextConfiguration) {
        topK = a.expertsPerToken
        routeScale = a.routeScale
        _weight.wrappedValue = MLXArray.zeros([a.routedExperts, a.hiddenSize], type: Float.self)
        _bias.wrappedValue = MLXArray.zeros([a.routedExperts], type: Float.self)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).transposed()))
        let biased = scores + bias.asType(.float32)
        let k = topK
        let indices = argPartition(-biased, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k]
        var w = takeAlong(scores, indices, axis: -1)
        w = w / (w.sum(axis: -1, keepDims: true) + 1e-20)
        return (indices, w * routeScale)
    }
}

final class Inkling2MoE: Module {
    @ModuleInfo(key: "gate") var gate: Inkling2Router
    @ModuleInfo(key: "experts") var experts: Inkling2Experts

    init(_ a: Inkling2TextConfiguration, quant: Inkling2ExpertQuant?) {
        _gate.wrappedValue = Inkling2Router(a)
        _experts.wrappedValue = Inkling2Experts(a, quant: quant)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        let y = experts(x, indices).asType(.float32) * weights.expandedDimensions(axis: -1)
        return y.sum(axis: -2).asType(x.dtype)
    }
}

// MARK: - Attention residuals and loop messages

/// A softmax over depth: the blocks so far and the current partial block,
/// weighted per query. `blocks` [N, B, L, D], `partial` [B, L, D],
/// `query` [Q, D] → [B, L, Q, D].
private func depthAttention(_ blocks: MLXArray?, _ partial: MLXArray, query: MLXArray) -> MLXArray {
    let q = query.asType(.float32)
    let scale = 1 / Float(partial.dim(-1)).squareRoot()
    let p = partial.asType(.float32).expandedDimensions(axis: 0)
    let stacked = (blocks.map { concatenated([$0.asType(.float32), p]) } ?? p).movedAxis(source: 0, destination: -2)
    let weights = softmax(matmul(stacked, q.transposed()) * scale, axis: -2, precise: true)
    return matmul(weights.swappedAxes(-1, -2), stacked)
}

final class Inkling2BlockResidual: Module {
    let dims: Int
    @ModuleInfo(key: "query") var query: MLXArray

    init(dims: Int, queries: Int = 1) {
        self.dims = dims
        _query.wrappedValue = queries == 1 ? MLXArray.zeros([dims], type: Float.self) : MLXArray.zeros([queries, dims], type: Float.self)
    }

    func callAsFunction(_ blocks: MLXArray?, _ partial: MLXArray) -> MLXArray {
        depthAttention(blocks, partial, query: query.reshaped(-1, dims))
    }
}

final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "forget_gate_proj") var forget: Linear
    @ModuleInfo(key: "message_gate_proj") var message: Linear

    init(dims: Int, rank: Int) {
        _down.wrappedValue = Linear(2 * dims, rank, bias: false)
        _forget.wrappedValue = Linear(rank, dims, bias: false)
        _message.wrappedValue = Linear(rank, dims, bias: false)
    }

    func callAsFunction(_ h: MLXArray, _ msg: MLXArray) -> MLXArray {
        let states = concatenated([unweightedRMSNorm(h), unweightedRMSNorm(msg)], axis: -1)
        let s = down(states.asType(down.weight.dtype)).asType(.float32)
        let f = tanh(forget(s))
        let g = tanh(message(s))
        return (1 - f) * h + g * msg
    }
}

final class Inkling2MessageInput: Module {
    @ModuleInfo(key: "query") var query: MLXArray

    init(dims: Int, looped: Int) {
        _query.wrappedValue = MLXArray.zeros([looped, 4, dims], type: Float.self)
    }

    func callAsFunction(_ bank: (MLXArray, MLXArray), _ i: Int) -> MLXArray {
        depthAttention(bank.0, bank.1, query: query[i])
    }
}

// MARK: - Layers

final class Inkling2Layer: Module {
    let index: Int
    let sliding: Bool
    let boundary: Bool
    let eps: Float

    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2BlockResidual?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2BlockResidual
    @ModuleInfo(key: "mlp") var mlp: Inkling2MoE
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttentionConv: Inkling2ShortConv
    @ModuleInfo(key: "post_mlp_convolution") var postMLPConv: Inkling2ShortConv
    @ModuleInfo(key: "self_attn") var attention: Inkling2Attention
    // Sliding layers: one norm for q, k and v; full layers: one each.
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var kNorm: RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var vNorm: RMSNorm?
    @ModuleInfo(key: "q_message_gate") var qGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") var kGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") var vGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") var mlpGate: Inkling2MessageGate?

    init(_ a: Inkling2TextConfiguration, index: Int, quant: Inkling2ExpertQuant?) {
        self.index = index
        sliding = a.localLayerIds.contains(index)
        boundary = index % a.layersPerBlock == 0
        eps = a.rmsNormEps
        let dims = a.hiddenSize
        _attentionInput.wrappedValue = index > 0 ? Inkling2BlockResidual(dims: dims, queries: sliding ? 1 : 3) : nil
        _mlpInput.wrappedValue = Inkling2BlockResidual(dims: dims)
        _mlp.wrappedValue = Inkling2MoE(a, quant: quant)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: dims, eps: a.rmsNormEps)
        _postAttentionConv.wrappedValue = Inkling2ShortConv(dims: dims, kernel: a.sconvKernel)
        _postMLPConv.wrappedValue = Inkling2ShortConv(dims: dims, kernel: a.sconvKernel)
        _attention.wrappedValue = Inkling2Attention(a, sliding: sliding)
        if sliding {
            _inputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: a.rmsNormEps)
        } else {
            _qNorm.wrappedValue = RMSNorm(dimensions: dims, eps: a.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: dims, eps: a.rmsNormEps)
            _vNorm.wrappedValue = RMSNorm(dimensions: dims, eps: a.rmsNormEps)
        }
        if let loop = a.loop, a.loopCore.contains(index) {
            _qGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
            _kGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
            _vGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
            _mlpGate.wrappedValue = Inkling2MessageGate(dims: dims, rank: loop.gateRank)
        }
    }

    func callAsFunction(
        _ hidden: MLXArray, blocks: MLXArray?, kv: KVCache?, conv: ArraysCache?, messages: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let dtype = hidden.dtype
        var states: [MLXArray?] = conv.map { c in (0 ..< 4).map { c[$0] } } ?? [nil, nil, nil, nil]
        let blockResidual = hidden
        let normedResidual = unweightedRMSNorm(blockResidual)
        let inputs = attentionInput?(blocks, normedResidual)

        // q, k, v inputs: the depth mix (or the stream itself on layer 0),
        // gated by the loop's messages, normed.
        var q: MLXArray, k: MLXArray, v: MLXArray
        if sliding {
            let h = inputs.map { $0[.ellipsis, 0, 0...] } ?? hidden
            if let messages, let qGate, let kGate, let vGate {
                q = qGate(h, messages[.ellipsis, 0, 0...])
                k = kGate(h, messages[.ellipsis, 1, 0...])
                v = vGate(h, messages[.ellipsis, 2, 0...])
            } else {
                (q, k, v) = (h, h, h)
            }
            q = inputNorm!(q).asType(dtype)
            k = inputNorm!(k).asType(dtype)
            v = inputNorm!(v).asType(dtype)
        } else {
            if let inputs {
                (q, k, v) = (inputs[.ellipsis, 0, 0...], inputs[.ellipsis, 1, 0...], inputs[.ellipsis, 2, 0...])
            } else {
                (q, k, v) = (hidden, hidden, hidden)
            }
            if let messages, let qGate, let kGate, let vGate {
                q = qGate(q, messages[.ellipsis, 0, 0...])
                k = kGate(k, messages[.ellipsis, 1, 0...])
                v = vGate(v, messages[.ellipsis, 2, 0...])
            }
            q = qNorm!(q).asType(dtype)
            k = kNorm!(k).asType(dtype)
            v = vNorm!(v).asType(dtype)
        }
        var attnStates: [MLXArray?] = [states[0], states[1]]
        var h = attention(q, keyHidden: k, valueHidden: v, cache: kv, convStates: &attnStates)
        states[0] = attnStates[0]
        states[1] = attnStates[1]
        let (pa, paWindow) = postAttentionConv(h, state: states[2])
        h = pa
        states[2] = paWindow

        var blocks = blocks
        var residual = h.asType(.float32)
        var hf = residual
        if boundary {
            let block = normedResidual.expandedDimensions(axis: 0)
            blocks = blocks.map { concatenated([$0, block]) } ?? block
        } else {
            hf = blockResidual.asType(.float32) + hf
            residual = hf
        }

        var m = mlpInput(blocks, unweightedRMSNorm(hf))[.ellipsis, 0, 0...]
        if let messages, let mlpGate { m = mlpGate(m, messages[.ellipsis, 3, 0...]) }
        var out = mlp(postAttentionNorm(m).asType(dtype))
        let (pm, pmWindow) = postMLPConv(out, state: states[3])
        out = pm
        states[3] = pmWindow

        if let conv {
            let n = postMLPConv.kernel - 1
            for i in 0 ..< 4 { conv[i] = states[i].map { convState($0, n: n) } }
        }
        return ((residual + out.asType(.float32)).asType(dtype), blocks)
    }
}

// MARK: - Model

final class Inkling2TextModel: Module {
    let args: Inkling2TextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2Layer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2BlockResidual
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    init(_ a: Inkling2TextConfiguration, experts: [Int: Inkling2ExpertQuant]) {
        args = a
        _embedTokens.wrappedValue = Embedding(embeddingCount: a.vocabularySize, dimensions: a.hiddenSize)
        _embedNorm.wrappedValue = RMSNorm(dimensions: a.hiddenSize, eps: a.rmsNormEps)
        _layers.wrappedValue = (0 ..< a.hiddenLayers).map {
            Inkling2Layer(a, index: $0, quant: experts[$0])
        }
        _finalInput.wrappedValue = Inkling2BlockResidual(dims: a.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: a.hiddenSize, eps: a.rmsNormEps)
        if let loop = a.loop {
            _messageInput.wrappedValue = Inkling2MessageInput(
                dims: a.hiddenSize, looped: loop.coreEnd - loop.coreStart)
        }
    }

    func embed(_ inputs: MLXArray) -> MLXArray { embedNorm(embedTokens(inputs)) }

    /// `cache`: two per layer visit — the attention's KV, then its four
    /// convolution states. `embeddings`, when given, stand in for `inputs`'.
    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]?, embeddings: MLXArray? = nil) -> MLXArray {
        var h = embeddings ?? embed(inputs!)
        let dtype = h.dtype
        var visit = 0
        func run(_ i: Int, _ h: MLXArray, _ blocks: MLXArray?, _ messages: MLXArray?) -> (MLXArray, MLXArray?) {
            let kv = cache?[2 * visit]
            let conv = cache?[2 * visit + 1] as? ArraysCache
            visit += 1
            return layers[i](h, blocks: blocks, kv: kv, conv: conv, messages: messages)
        }
        var blocks: MLXArray? = nil
        if let loop = args.loop {
            for i in 0 ..< loop.coreStart { (h, blocks) = run(i, h, blocks, nil) }
            let prelude = (h, blocks)
            let preludeBlocks = blocks?.dim(0) ?? 0
            for pass in 0 ..< loop.numPasses {
                var bank: (MLXArray, MLXArray)? = nil
                if pass > 0, let b = blocks {
                    // The previous pass's blocks after its first, and its output.
                    bank = (b[(preludeBlocks + 1)...], unweightedRMSNorm(h))
                    (h, blocks) = prelude
                }
                for (j, i) in (loop.coreStart ..< loop.coreEnd).enumerated() {
                    let messages = bank.map { messageInput!($0, j) }
                    (h, blocks) = run(i, h, blocks, messages)
                }
            }
            for i in loop.coreEnd ..< args.hiddenLayers { (h, blocks) = run(i, h, blocks, nil) }
        } else {
            for i in 0 ..< args.hiddenLayers { (h, blocks) = run(i, h, blocks, nil) }
        }
        let out = finalInput(blocks, unweightedRMSNorm(h))[.ellipsis, 0, 0...]
        return norm(out).asType(dtype)
    }
}

final class Inkling2Outer: Module {
    @ModuleInfo(key: "language_model") var languageModel: Inkling2TextModel
    @ModuleInfo(key: "vision_tower") var visionTower: Inkling2VisionModel?

    init(_ config: Inkling2Configuration, experts: [Int: Inkling2ExpertQuant]) {
        _languageModel.wrappedValue = Inkling2TextModel(config.text, experts: experts)
        if let v = config.vision {
            _visionTower.wrappedValue = Inkling2VisionModel(v, textHidden: config.text.hiddenSize)
        }
    }
}

public class Inkling2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let args: Inkling2TextConfiguration
    /// The token a picture's rows go in for, one per 40×40 pixels.
    public let imageTokenId: Int

    @ModuleInfo(key: "model") fileprivate var model: Inkling2Outer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: Inkling2Configuration, experts: [Int: Inkling2ExpertQuant]) {
        let a = config.text
        args = a
        vocabularySize = a.vocabularySize
        kvHeads = a.layerVisits.map { a.localLayerIds.contains($0) ? a.swaKvHeads : a.kvHeads }
        imageTokenId = config.imageTokenId ?? 200_054
        _model.wrappedValue = Inkling2Outer(config, experts: experts)
        _lmHead.wrappedValue = Linear(a.hiddenSize, a.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model.languageModel(inputs, cache: cache))
    }

    private func logits(_ h: MLXArray) -> MLXArray {
        let cap = args.finalLogitSoftcapping
        let logits = tanh(lmHead(h).asType(.float32) / cap) * cap
        return logits[.ellipsis, 0 ..< args.unpaddedVocabularySize]
    }

    /// Whether this checkpoint carries its vision tower.
    public var seesPictures: Bool { model.visionTower != nil }

    /// Every picture in `image` through the tower, end to end: one row per
    /// image token, in prompt order. Each picture is its own (1, H, W) frame
    /// of the flattened pixels.
    func encode(_ image: LMInput.ProcessedImage) -> MLXArray? {
        guard let tower = model.visionTower, let frames = image.frames, !frames.isEmpty else { return nil }
        var offset = 0
        var rows: [MLXArray] = []
        for f in frames {
            let n = f.h * f.w * 3
            rows.append(tower(image.pixels[offset ..< offset + n].reshaped(f.h, f.w, 3)))
            offset += n
        }
        return concatenated(rows)
    }

    /// A span with pictures in it: the tokens embedded, each picture's rows
    /// put where its image tokens are (in order), then the span run in
    /// chunks. Logits for its last token only — what follows a picture is
    /// always more prompt. Without pictures, the text path of every LLM.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let image = input.image else { return try prepareText(input, cache: cache, state: state, prefill: prefill) }
        guard let features = encode(image) else {
            throw NSError(
                domain: "chaty-mlx", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "this Inkling checkpoint has no vision tower"])
        }
        let tokens = input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis] : input.text.tokens
        let ids = tokens.reshaped(-1).asArray(Int32.self)
        let slot = Int32(imageTokenId)
        var flat = model.languageModel.embed(tokens).reshaped(ids.count, -1)
        var row = 0
        var i = 0
        while i < ids.count {
            guard ids[i] == slot else { i += 1; continue }
            var j = i
            while j < ids.count, ids[j] == slot { j += 1 }
            guard row + (j - i) <= features.dim(0) else { break }
            flat[i ..< j] = features[row ..< row + (j - i)].asType(flat.dtype)
            row += j - i
            i = j
        }
        guard row == features.dim(0) else {
            throw NSError(
                domain: "chaty-mlx", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "image tokens (\(row)) do not match the pictures (\(features.dim(0)))"])
        }
        let embeddings = flat.reshaped(1, ids.count, -1)
        let step = prefill.stepSize ?? 512
        let total = ids.count
        var done = 0
        let last: MLXArray = withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            while total - done > 1 {
                let take = min(step, total - done - 1)
                _ = model.languageModel(nil, cache: cache, embeddings: embeddings[0..., done ..< done + take, 0...])
                asyncEval(cache)
                done += take
            }
            let h = model.languageModel(nil, cache: cache, embeddings: embeddings[0..., done..., 0...])
            return logits(h[0..., (h.dim(1) - 1)..., 0...])
        }
        return .logits(LMOutput(logits: last))
    }

    /// What LLMModel's own `prepare` does: chunks up to the last step, the
    /// rest for the iterator.
    private func prepareText(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let stepSize = prefill.resolvedStepSize()
        let y = input.text
        let total = y.tokens.size
        guard total > stepSize else { return .tokens(y) }
        var processed = 0
        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            var state: LMOutput.State? = state
            processed = try prefill.forEachChunk(
                total: total, reserving: prefill.chunking == .remainder ? stepSize : 1
            ) { range in
                let output = self(y[.newAxis, range], cache: cache.isEmpty ? nil : cache, state: state)
                state = output.state
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
        }
        return .tokens(y[processed...])
    }

    /// Two caches for each layer the stream passes through: its keys and
    /// values (all of them — a sliding layer's window is its mask), and its
    /// four convolution states, which cannot be rewound.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        args.layerVisits.flatMap { _ -> [KVCache] in [KVCacheSimple(), ArraysCache(size: 4)] }
    }

    /// The text model's and the vision tower's weights (the tower's only
    /// when the config declares one); anything else a checkpoint carries —
    /// its multi-token-prediction head — stays out.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let vision = seesPictures
        return weights.filter {
            $0.key.hasPrefix("model.language_model.") || $0.key.hasPrefix("lm_head.")
                || (vision && $0.key.hasPrefix("model.vision_tower."))
        }
    }
}

extension Inkling2Model {
    public var loraLayers: [Module] { model.languageModel.layers }
}

/// Teach the sidecar's factory about Inkling 2. Idempotent.
public enum Inkling2Registration {
    public static func register() async {
        for type in ["inkling2", "inkling2_loop"] {
            await LLMTypeRegistry.shared.registerModelType(
                type,
                creator: { data in
                    let config = try JSONDecoder().decode(Inkling2Configuration.self, from: data)
                    return Inkling2Model(config, experts: Inkling2Registration.expertLayout(config.text))
                })
        }
    }

    /// How each layer's experts are stored — read from the safetensors
    /// headers of the folder being loaded, as mlx-lm reads it from the tensors:
    /// the config does not say. Bits from the packed weight's width, the group
    /// from the scales' (`uint8` scales: a floating-point mode), a per-expert
    /// scale where the checkpoint carries one.
    static func expertLayout(_ a: Inkling2TextConfiguration) -> [Int: Inkling2ExpertQuant] {
        guard let dir = Inkling2Registration.loadingDirectory else { return [:] }
        var weight: [Int: [Int]] = [:]
        var scales: [Int: (String, [Int])] = [:]
        var global = Set<Int>()
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        for url in files where url.pathExtension == "safetensors" {
            guard let header = safetensorsHeader(url) else { continue }
            for (key, value) in header {
                guard key.hasPrefix("model.language_model.layers."), let info = value as? [String: Any]
                else { continue }
                let rest = key.dropFirst("model.language_model.layers.".count)
                guard let dot = rest.firstIndex(of: "."), let layer = Int(rest[..<dot]) else { continue }
                let shape = (info["shape"] as? [Int]) ?? []
                if key.hasSuffix(".mlp.experts.gate_up_proj.weight") { weight[layer] = shape }
                if key.hasSuffix(".mlp.experts.gate_up_proj.scales") {
                    scales[layer] = ((info["dtype"] as? String) ?? "", shape)
                }
                if key.hasSuffix(".mlp.experts.gate_up_proj.global_scale") { global.insert(layer) }
            }
        }
        var out: [Int: Inkling2ExpertQuant] = [:]
        for (layer, (dtype, s)) in scales where dtype == "U8" {
            guard let w = weight[layer], let wl = w.last, let sl = s.last, sl > 0 else { continue }
            let quant = Inkling2ExpertQuant(
                bits: wl * 32 / a.hiddenSize, groupSize: a.hiddenSize / sl, globalScale: global.contains(layer))
            if quant.mode != nil { out[layer] = quant }
        }
        return out
    }

    /// The folder the factory is loading, set by the sidecar before it asks.
    nonisolated(unsafe) static var loadingDirectory: URL?

    private static func safetensorsHeader(_ url: URL) -> [String: Any]? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        guard let n = try? h.read(upToCount: 8), n.count == 8 else { return nil }
        let len = n.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard len < 1 << 28, let body = try? h.read(upToCount: Int(len)) else { return nil }
        return (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
    }
}
