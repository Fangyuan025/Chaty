// Muse Glimmer's vision half — the tower, the adapter, and the processor that
// turns an image into the patches they expect.
//
// The tower is a 50-layer ViT that attends inside 32×32-patch windows except
// on every fourth layer, which sees the whole image. Position information
// arrives twice: a learned 32×32 table bilinearly resampled to whatever grid
// the image actually has, and a 2-D rotary embedding over (width, height).
// The last layer's output is merged 2×2 before it leaves, which is why the
// adapter's input is four times the tower's width.
//
// Ported from the reference MLX implementation, which is the only one that
// exists for this architecture — the implementation the checkpoint itself
// ships is text-only and drops these weights.
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - configuration

public struct MuseGlimmerVisionConfiguration: Decodable, Sendable {
    var hiddenSize: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var patchSize: Int
    var patchTemporal: Int
    var mergeSize: Int
    var posEmbHeight: Int
    var posEmbWidth: Int
    var layerNormEps: Float
    var ropeTheta: Float
    var layerTypes: [String]

    var headDim: Int { hiddenSize / attentionHeads }
    /// Each patch carries every channel of every temporal slice.
    var patchDim: Int { patchTemporal * 3 * patchSize * patchSize }
    func isFull(_ i: Int) -> Bool {
        i < layerTypes.count && layerTypes[i] == "full_attention"
    }

    enum RopeKeys: String, CodingKey {
        case theta = "rope_theta"
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case patchSize = "patch_size"
        case patchTemporal = "patch_temporal"
        case mergeSize = "merge_size"
        case posEmbHeight = "pos_emb_height"
        case posEmbWidth = "pos_emb_width"
        case layerNormEps = "layer_norm_eps"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8960
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 50
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        patchTemporal = try c.decodeIfPresent(Int.self, forKey: .patchTemporal) ?? 2
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        posEmbHeight = try c.decodeIfPresent(Int.self, forKey: .posEmbHeight) ?? 32
        posEmbWidth = try c.decodeIfPresent(Int.self, forKey: .posEmbWidth) ?? 32
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        // `rope_parameters` also carries a `rope_type` string, so it cannot be
        // read as a flat number map.
        let rope = try? c.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters)
        ropeTheta = (try? rope?.decodeIfPresent(Float.self, forKey: .theta)) ?? 10_000 ?? 10_000
        // Every fourth layer, and the last one, look at the whole image.
        let layerCount = hiddenLayers
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< layerCount).map {
                ($0 + 1) % 4 == 0 || $0 == layerCount - 1 ? "full_attention" : "window_attention"
            }
    }
}

public struct MuseGlimmerVLMConfiguration: Decodable, Sendable {
    var text: MuseGlimmerConfiguration
    var vision: MuseGlimmerVisionConfiguration
    var imageTokenId: Int
    var videoTokenId: Int
    var projectorHiddenSize: Int
    var outHiddenSize: Int

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case projectorHiddenSize = "projector_hidden_size"
        case outHiddenSize = "out_hidden_size"
    }

    public init(from decoder: Swift.Decoder) throws {
        // The text configuration reads the nested `text_config` itself.
        text = try MuseGlimmerConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vision = try c.decode(MuseGlimmerVisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200_092
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 200_091
        projectorHiddenSize = try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        outHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .outHiddenSize)
            ?? vision.hiddenSize * vision.mergeSize * vision.mergeSize
    }
}

// MARK: - patch bookkeeping

/// One grid entry per image: temporal depth, height and width IN PATCHES.
struct VisionGrid {
    var t: Int
    var h: Int
    var w: Int
    var count: Int { t * h * w }
}

/// Where each image ends, for a layer that attends over whole images.
func visionFullSplits(_ grid: [VisionGrid]) -> [Int] {
    var points: [Int] = []
    var running = 0
    for g in grid {
        for _ in 0 ..< g.t {
            running += g.h * g.w
            points.append(running)
        }
    }
    // The split points are interior boundaries; the last one is the end.
    return points.isEmpty ? [] : Array(points.dropLast())
}

/// The permutation that gathers patches window by window, plus where each
/// window ends. Returns nil when the image is one window wide and the
/// permutation would be the identity.
func visionWindowIndex(_ grid: [VisionGrid], windowSize: Int) -> (order: [Int32]?, splits: [Int]) {
    var indices: [Int32] = []
    var cumulative: [Int] = [0]
    var offset = 0
    for g in grid {
        let padH = (windowSize - g.h % windowSize) % windowSize
        let padW = (windowSize - g.w % windowSize) % windowSize
        let nH = (g.h + padH) / windowSize
        let nW = (g.w + padW) / windowSize
        for t in 0 ..< g.t {
            let plane = t * g.h * g.w
            for wh in 0 ..< nH {
                for ww in 0 ..< nW {
                    var length = 0
                    for r in 0 ..< windowSize {
                        let row = wh * windowSize + r
                        if row >= g.h { continue }
                        for c in 0 ..< windowSize {
                            let col = ww * windowSize + c
                            if col >= g.w { continue }
                            indices.append(Int32(offset + plane + row * g.w + col))
                            length += 1
                        }
                    }
                    if length > 0 { cumulative.append(cumulative.last! + length) }
                }
            }
        }
        offset += g.count
    }
    let identity = indices.enumerated().allSatisfy { $1 == Int32($0) }
    // Interior boundaries only, matching the full-attention splits.
    let splits = cumulative.count > 2 ? Array(cumulative.dropFirst().dropLast()) : []
    return (identity ? nil : indices, splits)
}

/// One (width, height) pair per patch, one-indexed, in row-major order.
func visionPositionIds(_ grid: [VisionGrid]) -> MLXArray {
    var rows: [Int32] = []
    for g in grid {
        var plane: [Int32] = []
        plane.reserveCapacity(g.h * g.w * 2)
        for h in 0 ..< g.h {
            for w in 0 ..< g.w {
                plane.append(Int32(w + 1))
                plane.append(Int32(h + 1))
            }
        }
        for _ in 0 ..< g.t { rows.append(contentsOf: plane) }
    }
    return MLXArray(rows, [rows.count / 2, 2])
}

// MARK: - tower

private func visionRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

private class MuseGlimmerPatchEmbedder: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding_table") var positionTable: Embedding
    let side: Int

    init(_ args: MuseGlimmerVisionConfiguration) {
        self.side = args.posEmbHeight
        _patchEmbedding.wrappedValue = Linear(args.patchDim, args.hiddenSize, bias: false)
        _positionTable.wrappedValue = Embedding(
            embeddingCount: args.posEmbHeight * args.posEmbWidth, dimensions: args.hiddenSize)
    }

    /// The learned table is a fixed 32×32 grid; an image of any other shape
    /// reads it bilinearly, which is four table lookups per patch weighted by
    /// how far the sample fell between them.
    private func positions(_ grid: [VisionGrid]) -> MLXArray {
        var idx: [[Int32]] = [[], [], [], []]
        var wts: [[Float]] = [[], [], [], []]
        let side = self.side
        for g in grid {
            var h0s: [Int] = [], h1s: [Int] = [], dhs: [Float] = []
            var vh0s: [Bool] = [], vh1s: [Bool] = []
            for i in 0 ..< g.h {
                let sample = (Float(i) + 0.5) * (Float(side) / Float(g.h)) - 0.5
                let f = Int(floor(sample))
                h0s.append(min(max(f, 0), side - 1))
                h1s.append(min(max(f + 1, 0), side - 1))
                dhs.append(sample - Float(f))
                vh0s.append(f >= 0 && f < side)
                vh1s.append(f + 1 >= 0 && f + 1 < side)
            }
            var w0s: [Int] = [], w1s: [Int] = [], dws: [Float] = []
            var vw0s: [Bool] = [], vw1s: [Bool] = []
            for i in 0 ..< g.w {
                let sample = (Float(i) + 0.5) * (Float(side) / Float(g.w)) - 0.5
                let f = Int(floor(sample))
                w0s.append(min(max(f, 0), side - 1))
                w1s.append(min(max(f + 1, 0), side - 1))
                dws.append(sample - Float(f))
                vw0s.append(f >= 0 && f < side)
                vw1s.append(f + 1 >= 0 && f + 1 < side)
            }

            var plane: [[Int32]] = [[], [], [], []]
            var planeW: [[Float]] = [[], [], [], []]
            for i in 0 ..< g.h {
                for j in 0 ..< g.w {
                    let dh = dhs[i], dw = dws[j]
                    plane[0].append(Int32(h0s[i] * side + w0s[j]))
                    plane[1].append(Int32(h0s[i] * side + w1s[j]))
                    plane[2].append(Int32(h1s[i] * side + w0s[j]))
                    plane[3].append(Int32(h1s[i] * side + w1s[j]))
                    planeW[0].append((1 - dh) * (1 - dw) * (vh0s[i] && vw0s[j] ? 1 : 0))
                    planeW[1].append((1 - dh) * dw * (vh0s[i] && vw1s[j] ? 1 : 0))
                    planeW[2].append(dh * (1 - dw) * (vh1s[i] && vw0s[j] ? 1 : 0))
                    planeW[3].append(dh * dw * (vh1s[i] && vw1s[j] ? 1 : 0))
                }
            }
            for _ in 0 ..< g.t {
                for k in 0 ..< 4 {
                    idx[k].append(contentsOf: plane[k])
                    wts[k].append(contentsOf: planeW[k])
                }
            }
        }
        let n = idx[0].count
        let indices = MLXArray(idx.flatMap { $0 }, [4, n])
        let weights = MLXArray(wts.flatMap { $0 }, [4, n])
        return sum(positionTable(indices) * weights.expandedDimensions(axis: -1), axis: 0)
    }

    func callAsFunction(_ pixels: MLXArray, grid: [VisionGrid]) -> MLXArray {
        let embedded = patchEmbedding(pixels)
        return embedded + positions(grid).asType(embedded.dtype)
    }
}

private class MuseGlimmerVisionAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ args: MuseGlimmerVisionConfiguration) {
        self.heads = args.attentionHeads
        self.headDim = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)
        let d = args.hiddenSize
        _qProj.wrappedValue = Linear(d, d, bias: true)
        _kProj.wrappedValue = Linear(d, d, bias: true)
        _vProj.wrappedValue = Linear(d, d, bias: true)
        _proj.wrappedValue = Linear(d, d, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray, splits: [Int], cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let length = x.dim(0)
        var q = qProj(x).reshaped(1, length, heads, -1)
        var k = kProj(x).reshaped(1, length, heads, -1)
        let v = vProj(x).reshaped(1, length, heads, -1).transposed(0, 2, 1, 3)

        let c = cos.expandedDimensions(axes: [0, 2])
        let s = sin.expandedDimensions(axes: [0, 2])
        q = q * c + visionRotateHalf(q) * s
        k = k * c + visionRotateHalf(k) * s
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)

        var out: MLXArray
        if splits.isEmpty {
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none)
        } else {
            // A window is a span of the sequence, so the window mask is just
            // attention run separately over each span.
            let qs = split(q, indices: splits, axis: 2)
            let ks = split(k, indices: splits, axis: 2)
            let vs = split(v, indices: splits, axis: 2)
            out = concatenated(
                (0 ..< qs.count).map {
                    MLXFast.scaledDotProductAttention(
                        queries: qs[$0], keys: ks[$0], values: vs[$0], scale: scale, mask: .none)
                }, axis: 2)
        }
        return proj(out.transposed(0, 2, 1, 3).reshaped(length, -1))
    }
}

private class MuseGlimmerVisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ args: MuseGlimmerVisionConfiguration) {
        _fc1.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

private class MuseGlimmerVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: MuseGlimmerVisionAttention
    @ModuleInfo(key: "mlp") var mlp: MuseGlimmerVisionMLP

    init(_ args: MuseGlimmerVisionConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        _norm2.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        _attn.wrappedValue = MuseGlimmerVisionAttention(args)
        _mlp.wrappedValue = MuseGlimmerVisionMLP(args)
    }

    func callAsFunction(
        _ x: MLXArray, splits: [Int], cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let h = x + attn(norm1(x), splits: splits, cos: cos, sin: sin)
        return h + mlp(norm2(h))
    }
}

class MuseGlimmerVisionTower: Module {
    let args: MuseGlimmerVisionConfiguration

    @ModuleInfo(key: "patch_embedder") fileprivate var patchEmbedder: MuseGlimmerPatchEmbedder
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    fileprivate let layers: [MuseGlimmerVisionBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    /// The tower's own precision, read off the projection every patch goes
    /// through — pixels have to arrive in it.
    var patchWeightDType: DType { patchEmbedder.patchEmbedding.weight.dtype }

    init(_ args: MuseGlimmerVisionConfiguration) {
        self.args = args
        _patchEmbedder.wrappedValue = MuseGlimmerPatchEmbedder(args)
        _lnPre.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        self.layers = (0 ..< args.hiddenLayers).map { _ in MuseGlimmerVisionBlock(args) }
        _lnPost.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
    }

    /// Half the head is rotated by the patch's column, half by its row.
    private func rotary(_ positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let spatial = args.headDim / 2
        let steps = MLXArray(stride(from: 0, to: spatial, by: 2).map { Float($0) })
        let invFreq = 1.0 / pow(args.ropeTheta, steps / Float(spatial))
        let width = positionIds[0..., 0].asType(.float32)
        let height = positionIds[0..., 1].asType(.float32)
        let fw = width.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
        let fh = height.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
        let freqs = concatenated([fw, fh, fw, fh], axis: -1)
        return (cos(freqs), sin(freqs))
    }

    /// 2×2 neighbourhoods leave as one token, four times as wide.
    private func pixelShuffle(_ h: MLXArray, grid: [VisionGrid]) -> MLXArray {
        let merge = args.mergeSize
        let dim = h.dim(-1)
        var out: [MLXArray] = []
        var offset = 0
        for g in grid {
            let chunk = h[offset ..< (offset + g.count)]
                .reshaped(g.t, g.h / merge, merge, g.w / merge, merge, dim)
                .transposed(0, 1, 3, 5, 2, 4)
            out.append(chunk.reshaped(-1, dim * merge * merge))
            offset += g.count
        }
        return out.count == 1 ? out[0] : concatenated(out, axis: 0)
    }

    func callAsFunction(_ pixels: MLXArray, grid: [VisionGrid]) -> MLXArray {
        let fullSplits = visionFullSplits(grid)
        let (order, windowSplits) = visionWindowIndex(grid, windowSize: args.posEmbHeight)

        var h = lnPre(patchEmbedder(pixels, grid: grid))
        var positions = visionPositionIds(grid)
        var gather: MLXArray?
        if let order {
            gather = MLXArray(order)
            h = take(h, gather!, axis: 0)
            positions = take(positions, gather!, axis: 0)
        }
        let (cosine, sine) = rotary(positions)

        for (i, layer) in layers.enumerated() {
            h = layer(
                h, splits: args.isFull(i) ? fullSplits : windowSplits, cos: cosine, sin: sine)
        }
        if let gather {
            h = take(h, argSort(gather, axis: 0), axis: 0)
        }
        return pixelShuffle(lnPost(h), grid: grid)
    }
}

private class MuseGlimmerVisionAdapter: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ args: MuseGlimmerVLMConfiguration) {
        _fc1.wrappedValue = Linear(args.outHiddenSize, args.projectorHiddenSize, bias: false)
        _fc2.wrappedValue = Linear(args.projectorHiddenSize, args.projectorHiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(fc2(gelu(fc1(x)))) }
}

// MARK: - model

public class MuseGlimmerVLM: Module, VLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let config: MuseGlimmerVLMConfiguration

    @ModuleInfo(key: "language_model") var languageModel: MuseGlimmerModel
    @ModuleInfo(key: "vision_tower") var visionTower: MuseGlimmerVisionTower
    @ModuleInfo(key: "vision_adapter") fileprivate var visionAdapter: MuseGlimmerVisionAdapter
    @ModuleInfo(key: "vision_projection") var visionProjection: Linear

    public init(_ args: MuseGlimmerVLMConfiguration) {
        self.config = args
        self.vocabularySize = args.text.vocabularySize
        self.kvHeads = Array(repeating: args.text.kvHeads, count: args.text.hiddenLayers)
        _languageModel.wrappedValue = MuseGlimmerModel(args.text)
        _visionTower.wrappedValue = MuseGlimmerVisionTower(args.vision)
        _visionAdapter.wrappedValue = MuseGlimmerVisionAdapter(args)
        _visionProjection.wrappedValue = Linear(
            args.projectorHiddenSize, args.text.hiddenSize, bias: false)
    }

    /// Image features arrive already normalised — they get a norm of their own
    /// rather than the token embeddings'.
    private func encode(_ pixels: MLXArray, grid: [VisionGrid]) -> MLXArray {
        let dtype = visionTower.patchWeightDType
        let features = visionTower(pixels.asType(dtype), grid: grid)
        return rmsNormPlain(
            visionProjection(visionAdapter(features)), eps: config.text.rmsNormEps)
    }

    private func inputEmbeddings(_ inputIds: MLXArray, image: LMInput.ProcessedImage?)
        -> MLXArray
    {
        var embeddings = languageModel.embed(inputIds)
        guard let image, let frames = image.frames, !frames.isEmpty else { return embeddings }
        let grid = frames.map { VisionGrid(t: $0.t, h: $0.h, w: $0.w) }
        let features = encode(image.pixels, grid: grid).asType(embeddings.dtype)

        // The processor writes one unbroken run of image tokens per image, in
        // the order the images came in, so the features drop straight onto
        // those spans — no gather needed.
        let ids = inputIds.reshaped(-1).asArray(Int32.self)
        let image32 = Int32(config.imageTokenId)
        let video32 = Int32(config.videoTokenId)
        let isSlot = { (t: Int32) in t == image32 || t == video32 }
        var flat = embeddings.reshaped(-1, embeddings.dim(-1))
        var row = 0
        var i = 0
        while i < ids.count {
            guard isSlot(ids[i]) else {
                i += 1
                continue
            }
            var j = i
            while j < ids.count, isSlot(ids[j]) { j += 1 }
            let span = j - i
            guard row + span <= features.dim(0) else { break }
            flat[i ..< j] = features[row ..< (row + span)]
            row += span
            i = j
        }
        embeddings = flat.reshaped(embeddings.shape)
        return embeddings
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(input.text.tokens, image: input.image)
        let result = withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let step = windowSize ?? 512
            let total = embeddings.dim(1)
            var done = 0
            while total - done > 1 {
                let take = min(step, total - done - 1)
                _ = languageModel(
                    nil, cache: cache, inputEmbeddings: embeddings[0..., done ..< (done + take), 0...]
                )
                asyncEval(cache)
                done += take
            }
            eval(cache)
            return languageModel(
                nil, cache: cache, inputEmbeddings: embeddings[0..., done..., 0...])
        }
        return .logits(LMOutput(logits: result))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache, inputEmbeddings: nil)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public var loraLayers: [Module] { languageModel.loraLayers }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // The language half needs the same rewrite as it does on its own; the
        // prefix stays because this model nests it under exactly that key.
        museGlimmerNormalize(
            weights, heads: config.text.attentionHeads, headDim: config.text.headDim,
            stripLanguagePrefix: false)
    }
}

// MARK: - processor

public struct MuseGlimmerProcessorConfiguration: Codable, Sendable {
    public struct ImageProcessor: Codable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let maxImageTokens: Int
        public let mergeSize: Int
        public let patchSize: Int
        public let temporalPatchSize: Int

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case maxImageTokens = "max_image_tokens"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
        }
    }

    public let imageProcessor: ImageProcessor

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }

    var meanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageProcessor.imageMean[0], imageProcessor.imageMean[1], imageProcessor.imageMean[2])
    }
    var stdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageProcessor.imageStd[0], imageProcessor.imageStd[1], imageProcessor.imageStd[2])
    }
}

public struct MuseGlimmerProcessor: UserInputProcessor {
    private let config: MuseGlimmerProcessorConfiguration
    private let tokenizer: any MLXLMCommon.Tokenizer

    public init(_ config: MuseGlimmerProcessorConfiguration, tokenizer: any MLXLMCommon.Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Choose the patch grid whose aspect ratio is closest to the image's,
    /// under the token budget. The unit is a MERGED patch, so the pixel size
    /// this returns is already a multiple of `patch_size * merge_size`.
    static func grid(height: Int, width: Int, unit: Int, maxTokens: Int) -> (Int, Int) {
        var idealH = Double(height) / Double(unit)
        var idealW = Double(width) / Double(unit)
        if idealH * idealW > Double(maxTokens) {
            let shrink = (Double(maxTokens) / (idealH * idealW)).squareRoot()
            idealH *= shrink
            idealW *= shrink
        }
        let target = Double(height) / Double(width)
        var best: (Int, Int)?
        var bestDistance = Double.infinity
        for h in [Int(idealH.rounded(.down)), Int(idealH.rounded(.up))] {
            for w in [Int(idealW.rounded(.down)), Int(idealW.rounded(.up))] {
                guard h >= 1, w >= 1, h * w <= maxTokens else { continue }
                let distance = abs(Double(h) / Double(w) - target)
                if distance < bestDistance {
                    bestDistance = distance
                    best = (h, w)
                }
            }
        }
        return best ?? (1, 1)
    }

    /// Resize, normalise, and cut into patches. Each patch holds every
    /// temporal slice of every channel; a still image is the same frame twice.
    private func patches(_ image: CIImage) throws -> (MLXArray, VisionGrid) {
        let ip = config.imageProcessor
        let unit = ip.patchSize * ip.mergeSize
        let extent = image.extent.size
        let (mergedH, mergedW) = Self.grid(
            height: Int(extent.height), width: Int(extent.width),
            unit: unit, maxTokens: ip.maxImageTokens)
        let pixelHeight = mergedH * unit
        let pixelWidth = mergedW * unit

        var processed = MediaProcessing.inSRGBToneCurveSpace(image)
        processed = MediaProcessing.resampleBicubic(
            processed, to: CGSize(width: pixelWidth, height: pixelHeight))
        processed = MediaProcessing.normalize(
            processed, mean: config.meanTuple, std: config.stdTuple)
        // [1, C, H, W]
        let array = MediaProcessing.asMLXArray(processed)

        let gridH = pixelHeight / ip.patchSize
        let gridW = pixelWidth / ip.patchSize
        let p = ip.patchSize
        // (C, gh, p, gw, p) → (gh, gw, C, p, p), then the temporal repeat.
        var flat = array.reshaped(3, gridH, p, gridW, p)
            .transposed(1, 3, 0, 2, 4)
            .reshaped(gridH * gridW, 1, 3 * p * p)
        flat = tiled(flat, repetitions: [1, ip.temporalPatchSize, 1])
            .reshaped(gridH * gridW, ip.temporalPatchSize * 3 * p * p)
        return (flat, VisionGrid(t: 1, h: gridH, w: gridW))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        if input.images.isEmpty {
            return LMInput(tokens: MLXArray(tokens))
        }

        var pixels: [MLXArray] = []
        var frames: [THW] = []
        for image in input.images {
            let (flat, grid) = try patches(try image.asCIImage())
            pixels.append(flat)
            frames.append(THW(grid.t, grid.h, grid.w))
        }

        // The template writes one placeholder per image. The model expects that
        // to open out into `<|image_start|>`, one token per MERGED patch, then
        // `<|image_end|>` — the markers are not decoration: without them the
        // images run together as one span, which is both the wrong prompt and
        // a prefix that cannot be reused per picture.
        let merged = config.imageProcessor.mergeSize * config.imageProcessor.mergeSize
        let imageToken = tokenizer.convertTokenToId("<|patch|>") ?? 200_092
        let startToken = tokenizer.convertTokenToId("<|image_start|>") ?? 200_080
        let endToken = tokenizer.convertTokenToId("<|image_end|>") ?? 200_081
        var expanded: [Int] = []
        var next = 0
        for token in tokens {
            guard token == imageToken, next < frames.count else {
                expanded.append(token)
                continue
            }
            let frame = frames[next]
            expanded.append(startToken)
            expanded.append(
                contentsOf: Array(repeating: imageToken, count: frame.h * frame.w / merged))
            expanded.append(endToken)
            next += 1
        }
        tokens = expanded

        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: LMInput.ProcessedImage(
                pixels: concatenated(pixels, axis: 0), frames: frames))
    }
}

/// Teach the VLM factory about `muse_glimmer`, model and processor both.
/// Idempotent, and safe to call before every load.
public enum MuseGlimmerVisionRegistration {
    public static func register() async {
        await VLMTypeRegistry.shared.registerModelType(
            "muse_glimmer",
            creator: { data in
                let config = try JSONDecoder().decode(
                    MuseGlimmerVLMConfiguration.self, from: data)
                return MuseGlimmerVLM(config)
            })
        await VLMProcessorTypeRegistry.shared.registerProcessorType(
            "MuseGlimmerProcessor",
            creator: { data, tokenizer in
                let config = try JSONDecoder().decode(
                    MuseGlimmerProcessorConfiguration.self, from: data)
                return MuseGlimmerProcessor(config, tokenizer: tokenizer)
            })
    }
}
