// Inkling 2's eyes — the vision tower, and the preprocessing it expects.
//
// After the reference in ml-explore/mlx-lm #2061 (models/inkling2/vision.py
// and processor.py), function for function, with llama.cpp's inkling2v
// (tools/mtmd) for the one thing the reference leaves open: how large a
// picture may be.
//
//   • a 23-layer ViT over 20-pixel patches, cut and ordered by 2×2 merge
//     block, with a class token in front;
//   • position twice: a learned 32×32 table resampled bilinearly to the
//     picture's own grid, and a 2-D rotary embedding over (column, row);
//   • LayerScale on both residuals of every layer;
//   • each 2×2 block's four patches concatenated and projected into the
//     language model's width: one image token per 40×40 pixels.
//
// A picture is not resized: it is padded up to a whole number of blocks with
// a value below black, as the reference does. Only one too large for 4096
// tokens is scaled down, keeping its shape (llama.cpp's bound).
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Inkling2VisionConfiguration: Decodable, Sendable {
    var hiddenSize = 1024
    var hiddenLayers = 23
    var attentionHeads = 16
    var channels = 3
    var patchSize = 20
    var mergeSize = 2
    var posembGrid = 32
    var projectionHidden = 4096
    var projectorBias = false
    var mlpRatio: Float = 4
    var layerNormEps: Float = 1e-5
    var ropeTheta: Float = 10000

    var headDim: Int { hiddenSize / attentionHeads }
    var intermediateSize: Int { Int(Float(hiddenSize) * mlpRatio) }
    /// One image token's side, in pixels.
    var block: Int { patchSize * mergeSize }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case channels = "num_channels"
        case patchSize = "native_patch_size"
        case mergeSize = "adapter_merge_size"
        case posembGrid = "posemb_grid_size"
        case projectionHidden = "projection_hidden_dim"
        case projectorBias = "projector_bias"
        case mlpRatio = "mlp_ratio"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? mergeSize
        posembGrid = try c.decodeIfPresent(Int.self, forKey: .posembGrid) ?? posembGrid
        projectionHidden = try c.decodeIfPresent(Int.self, forKey: .projectionHidden) ?? projectionHidden
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? projectorBias
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? mlpRatio
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? layerNormEps
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
    }
}

// MARK: - Positions

/// Each patch's row and column on the patch grid. Patches go by merge block
/// in raster order, then the block's own patches row by row.
func inklingGridPositions(gridH: Int, gridW: Int, merge: Int) -> (row: [Int32], col: [Int32]) {
    let blocksW = gridW / merge
    var row = [Int32](repeating: 0, count: gridH * gridW)
    var col = row
    for i in 0 ..< gridH * gridW {
        row[i] = Int32(i / (merge * merge * blocksW) * merge + i / merge % merge)
        col[i] = Int32(i / (merge * merge) % blocksW * merge + i % merge)
    }
    return (row, col)
}

/// The two table entries a position falls between along one axis, and their
/// weights — half-pixel centres, edges clamped (`_bilinear_taps`).
private func bilinearTaps(_ index: Int32, size: Int, side: Int) -> ([Int32], [Float]) {
    let src = (Float(index) + 0.5) * Float(side) / Float(size) - 0.5
    let base = src.rounded(.down)
    var taps: [Int32] = []
    var weights: [Float] = []
    for d in 0 ..< 2 {
        let tap = base + Float(d)
        weights.append(max(1 - abs(src - tap), 0))
        taps.append(Int32(min(max(tap, 0), Float(side - 1))))
    }
    return (taps, weights)
}

// MARK: - Tower

final class Inkling2VisionEmbeddings: Module {
    let merge: Int
    let grid: Int
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "cls_position_embedding") var clsPositionEmbedding: MLXArray

    init(_ a: Inkling2VisionConfiguration) {
        merge = a.mergeSize
        grid = a.posembGrid
        _patchEmbedding.wrappedValue = Linear(
            a.channels * a.patchSize * a.patchSize, a.hiddenSize, bias: false)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: a.posembGrid * a.posembGrid, dimensions: a.hiddenSize)
        _classEmbedding.wrappedValue = MLXArray.zeros([a.hiddenSize])
        _clsPositionEmbedding.wrappedValue = MLXArray.zeros([a.hiddenSize])
    }

    func callAsFunction(_ patches: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        var x = patchEmbedding(patches)
        let (row, col) = inklingGridPositions(gridH: gridH, gridW: gridW, merge: merge)
        var indices: [Int32] = []
        var weights: [Float] = []
        indices.reserveCapacity(row.count * 4)
        weights.reserveCapacity(row.count * 4)
        for n in 0 ..< row.count {
            let (rt, rw) = bilinearTaps(row[n], size: gridH, side: grid)
            let (ct, cw) = bilinearTaps(col[n], size: gridW, side: grid)
            for a in 0 ..< 2 {
                for b in 0 ..< 2 {
                    indices.append(rt[a] * Int32(grid) + ct[b])
                    weights.append(rw[a] * cw[b])
                }
            }
        }
        let idx = MLXArray(indices, [row.count, 4])
        let w = MLXArray(weights, [row.count, 4, 1])
        let pos = (positionEmbedding(idx).asType(.float32) * w).sum(axis: 1)
        x = x + pos.asType(x.dtype)
        let cls = (classEmbedding + clsPositionEmbedding).asType(x.dtype)
        return concatenated([cls.expandedDimensions(axis: 0), x])
    }
}

/// Rotate each (even, odd) pair of the last axis, in float32.
private func inklingVisionRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let s = x.asType(.float32)
    let pairs = s.reshaped(Array(s.shape.dropLast()) + [-1, 2])
    let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1).reshaped(s.shape)
    return (s * cos + rotated * sin).asType(x.dtype)
}

final class Inkling2VisionAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear

    init(_ a: Inkling2VisionConfiguration) {
        heads = a.attentionHeads
        scale = 1 / Float(a.headDim).squareRoot()
        _q.wrappedValue = Linear(a.hiddenSize, a.hiddenSize)
        _k.wrappedValue = Linear(a.hiddenSize, a.hiddenSize)
        _v.wrappedValue = Linear(a.hiddenSize, a.hiddenSize)
        _out.wrappedValue = Linear(a.hiddenSize, a.hiddenSize)
    }

    /// `x` [N, D]; `cos`, `sin` [N, 1, head_dim].
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let qs = inklingVisionRope(q(x).reshaped(n, heads, -1), cos: cos, sin: sin)
        let ks = inklingVisionRope(k(x).reshaped(n, heads, -1), cos: cos, sin: sin)
        let vs = v(x).reshaped(n, heads, -1)
        let o = MLXFast.scaledDotProductAttention(
            queries: qs.transposed(1, 0, 2)[.newAxis], keys: ks.transposed(1, 0, 2)[.newAxis],
            values: vs.transposed(1, 0, 2)[.newAxis], scale: scale, mask: .none)
        return out(o[0].transposed(1, 0, 2).reshaped(n, -1))
    }
}

/// GELU (erf) in float32, as the reference's `_precise_gelu`.
private func preciseGelu(_ x: MLXArray) -> MLXArray { gelu(x.asType(.float32)).asType(x.dtype) }

final class Inkling2VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ a: Inkling2VisionConfiguration) {
        _fc1.wrappedValue = Linear(a.hiddenSize, a.intermediateSize)
        _fc2.wrappedValue = Linear(a.intermediateSize, a.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(preciseGelu(fc1(x))) }
}

final class Inkling2VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2VisionAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2VisionMLP
    @ModuleInfo(key: "layernorm_before") var before: LayerNorm
    @ModuleInfo(key: "layernorm_after") var after: LayerNorm
    @ParameterInfo(key: "lambda_1") var lambda1: MLXArray
    @ParameterInfo(key: "lambda_2") var lambda2: MLXArray

    init(_ a: Inkling2VisionConfiguration) {
        _attention.wrappedValue = Inkling2VisionAttention(a)
        _mlp.wrappedValue = Inkling2VisionMLP(a)
        _before.wrappedValue = LayerNorm(dimensions: a.hiddenSize, eps: a.layerNormEps)
        _after.wrappedValue = LayerNorm(dimensions: a.hiddenSize, eps: a.layerNormEps)
        _lambda1.wrappedValue = MLXArray.ones([a.hiddenSize])
        _lambda2.wrappedValue = MLXArray.ones([a.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + lambda1 * attention(before(x), cos: cos, sin: sin)
        return h + lambda2 * mlp(after(h))
    }
}

final class Inkling2VisionMerger: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ a: Inkling2VisionConfiguration, textHidden: Int) {
        _linear1.wrappedValue = Linear(
            a.mergeSize * a.mergeSize * a.hiddenSize, a.projectionHidden, bias: a.projectorBias)
        _linear2.wrappedValue = Linear(a.projectionHidden, textHidden, bias: a.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(preciseGelu(linear1(x))) }
}

final class Inkling2VisionModel: Module {
    let args: Inkling2VisionConfiguration
    /// The rotary frequencies, per axis (kept off the module's parameters).
    let frequencies: [Float]
    @ModuleInfo(key: "embeddings") var embeddings: Inkling2VisionEmbeddings
    @ModuleInfo(key: "pre_layernorm") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2VisionLayer]
    @ModuleInfo(key: "merger") var merger: Inkling2VisionMerger

    init(_ a: Inkling2VisionConfiguration, textHidden: Int) {
        args = a
        let dims = a.headDim / 2
        frequencies = stride(from: 0, to: dims, by: 2).map { 1 / pow(a.ropeTheta, Float($0) / Float(dims)) }
        _embeddings.wrappedValue = Inkling2VisionEmbeddings(a)
        _preLayerNorm.wrappedValue = LayerNorm(dimensions: a.hiddenSize, eps: a.layerNormEps)
        _layers.wrappedValue = (0 ..< a.hiddenLayers).map { _ in Inkling2VisionLayer(a) }
        _merger.wrappedValue = Inkling2VisionMerger(a, textHidden: textHidden)
    }

    /// One picture, (H, W, C) already normalized, both sides whole blocks →
    /// one row per image token, raster order, in the language model's width.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let (H, W, C) = (pixels.dim(0), pixels.dim(1), pixels.dim(2))
        let p = args.patchSize
        let m = args.mergeSize
        let rows = H / (p * m)
        let cols = W / (p * m)
        // The merge blocks in raster order, then the patches of each block.
        let patches = pixels.reshaped(rows, m, p, cols, m, p, C)
            .transposed(0, 3, 1, 4, 2, 5, 6).reshaped(-1, p * p * C)
        let dtype = embeddings.patchEmbedding.weight.dtype
        var h = embeddings(patches.asType(dtype), gridH: rows * m, gridW: cols * m)
        h = preLayerNorm(h)

        // The class token sits at (0, 0); the patches start from (1, 1).
        let (row, col) = inklingGridPositions(gridH: rows * m, gridW: cols * m, merge: m)
        let invFreq = MLXArray(frequencies)
        let r = MLXArray([Int32(0)] + row.map { $0 + 1 }).asType(.float32)
        let c = MLXArray([Int32(0)] + col.map { $0 + 1 }).asType(.float32)
        let freqs = repeated(
            concatenated(
                [c.expandedDimensions(axis: 1) * invFreq, r.expandedDimensions(axis: 1) * invFreq],
                axis: -1),
            count: 2, axis: -1)
        let cos = MLX.cos(freqs).expandedDimensions(axis: 1)
        let sin = MLX.sin(freqs).expandedDimensions(axis: 1)
        for layer in layers { h = layer(h, cos: cos, sin: sin) }
        return merger(h[1...].reshaped(rows * cols, -1))
    }
}

// MARK: - Preprocessing

/// A picture as the tower reads it, before it is an array.
public struct Inkling2Picture {
    /// (H, W, 3), normalized; H and W whole blocks.
    public let pixels: MLXArray
    public let rows: Int
    public let cols: Int
    public var tokens: Int { rows * cols }
}

public enum Inkling2Pixels {
    /// llama.cpp's bound for this tower (`set_limit_image_tokens(1, 4096)`).
    static let maxTokens = 4096

    /// Read `url`: transparent pixels laid on the reference's gray
    /// checkerboard, padded up to whole 40-pixel blocks with −1 on the 0–255
    /// scale, then `(x / 255 − 0.5) / 0.5`. Only a picture past 4096 tokens is
    /// scaled, to the largest whole-block size under the bound with its shape.
    public static func load(_ url: URL, block: Int = 40) throws -> Inkling2Picture {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw NSError(
                domain: "chaty-mlx", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "cannot read image \(url.path)"])
        }
        var width = image.width
        var height = image.height
        func up(_ v: Int) -> Int { (v + block - 1) / block * block }
        let maxPixels = maxTokens * block * block
        if up(width) * up(height) > maxPixels {
            // calc_size_preserved_ratio: down to whole blocks under the bound.
            let beta = (Double(height) * Double(width) / Double(maxPixels)).squareRoot()
            height = max(block, Int((Double(height) / beta / Double(block)).rounded(.down)) * block)
            width = max(block, Int((Double(width) / beta / Double(block)).rounded(.down)) * block)
        }

        // RGBA8, premultiplied — CoreGraphics composites the picture over
        // what is already in the buffer, the checkerboard (row 0 on top).
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let g: UInt8 = (x / 8 + y / 8) % 2 == 0 ? 204 : 153
                let o = (y * width + x) * 4
                rgba[o] = g
                rgba[o + 1] = g
                rgba[o + 2] = g
            }
        }
        // The file's own values, as the reference reads them (PIL, and stb in
        // llama.cpp, leave an embedded profile alone): drawn in the picture's
        // own RGB space nothing is converted. A Display P3 screenshot read as
        // sRGB moved most pixels of a test picture by up to 20 levels.
        let space: CGColorSpace = {
            if let own = image.colorSpace, own.model == .rgb, own.supportsOutput { return own }
            return CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        }()
        let drawn: Bool = rgba.withUnsafeMutableBytes { buf in
            guard
                let ctx = CGContext(
                    data: buf.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else {
            throw NSError(
                domain: "chaty-mlx", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "cannot decode image \(url.path)"])
        }

        let W = up(width)
        let H = up(height)
        let pad = (-1 / Float(255) - 0.5) / 0.5
        var out = [Float](repeating: pad, count: H * W * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let i = (y * width + x) * 4
                let o = (y * W + x) * 3
                for ch in 0 ..< 3 { out[o + ch] = (Float(rgba[i + ch]) * (1 / Float(255)) - 0.5) / 0.5 }
            }
        }
        return Inkling2Picture(pixels: MLXArray(out, [H, W, 3]), rows: H / block, cols: W / block)
    }

    /// Pictures as the model's `prepare` takes them: their pixels flattened
    /// end to end, each one's (1, H, W) in `frames`.
    public static func processed(_ pictures: [Inkling2Picture]) -> LMInput.ProcessedImage {
        LMInput.ProcessedImage(
            pixels: concatenated(pictures.map { $0.pixels.reshaped(-1) }),
            frames: pictures.map { THW(1, $0.pixels.dim(0), $0.pixels.dim(1)) })
    }
}
