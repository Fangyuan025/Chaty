// The building blocks of the image models, read straight out of an MLX
// diffusion folder.
//
// The folders people publish for MLX image generation are mflux's saves:
// `transformer/`, `text_encoder/`, `vae/` (and `tokenizer/`), each a set of
// numbered safetensors behind a `model.safetensors.index.json` whose weight
// names are mflux's module paths ("noise_refiner.0.attention.to_q.weight").
// A layer that was quantized carries `.scales` and `.biases` beside its
// packed `.weight`; one that was not carries the plain matrix. Rather than
// mirror mflux's module tree in MLXNN and trust reflection to line the names
// up, each model below asks for its tensors by name — a missing or misnamed
// weight is an error that says which one, at load time.
import Foundation
import MLX
import MLXNN

enum DiffusionError: LocalizedError {
    case missingWeight(String)
    case badFolder(String)
    case unsupported(String)

    var errorDescription: String? {
        switch self {
        case .missingWeight(let k): return "缺少权重 \(k) (missing weight \(k))"
        case .badFolder(let m): return m
        case .unsupported(let m): return m
        }
    }
}

/// Every tensor of one component folder (`<model>/transformer` …), loaded
/// lazily — MLX maps the files and pages a tensor in on first use.
final class WeightStore {
    let dir: URL
    private(set) var tensors: [String: MLXArray] = [:]
    /// The quantization the save declares (`quantization_level`), nil when
    /// the weights were saved unquantized.
    let declaredBits: Int?

    init(dir: URL, progress: ((Int, Int) -> Void)? = nil) throws {
        self.dir = dir
        let fm = FileManager.default
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        var files: [String] = []
        var bits: Int? = nil
        if let data = try? Data(contentsOf: index),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let map = obj["weight_map"] as? [String: String] {
                files = Array(Set(map.values)).sorted()
            }
            if let meta = obj["metadata"] as? [String: Any] {
                if let q = meta["quantization_level"] as? String { bits = Int(q) }
                else if let q = meta["quantization_level"] as? Int { bits = q }
            }
        }
        if files.isEmpty {
            files = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? [])
                .filter { $0.hasSuffix(".safetensors") }.sorted()
        }
        if files.isEmpty {
            throw DiffusionError.badFolder("\(dir.lastPathComponent) 里没有权重文件 (no weights in \(dir.lastPathComponent))")
        }
        declaredBits = bits
        for (i, f) in files.enumerated() {
            let arrays = try loadArrays(url: dir.appendingPathComponent(f))
            // Read it in now: the load is where the wait belongs, and where
            // its progress is shown — not the first picture.
            eval(Array(arrays.values))
            tensors.merge(arrays) { a, _ in a }
            progress?(i + 1, files.count)
        }
    }

    /// How many weight files a component folder holds.
    static func shardCount(_ dir: URL) -> Int {
        ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }.count
    }

    func has(_ key: String) -> Bool { tensors[key] != nil }

    func take(_ key: String) throws -> MLXArray {
        guard let t = tensors[key] else { throw DiffusionError.missingWeight("\(dir.lastPathComponent)/\(key)") }
        return t
    }

    func maybe(_ key: String) -> MLXArray? { tensors[key] }

    /// All weight names, for recognising a family by what it holds.
    var keys: Dictionary<String, MLXArray>.Keys { tensors.keys }
}

/// A linear layer as it was saved: a plain matrix, or MLX's affine
/// quantization (bits and group size read back from the shapes, the way
/// mflux itself reads them — a save can mix precisions layer by layer).
struct DLinear {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let bias: MLXArray?
    let groupSize: Int
    let bits: Int

    init(_ w: WeightStore, _ prefix: String, inDim: Int) throws {
        weight = try w.take(prefix + ".weight")
        bias = w.maybe(prefix + ".bias")
        if let s = w.maybe(prefix + ".scales") {
            scales = s
            biases = w.maybe(prefix + ".biases")
            let packed = weight.shape.last ?? 0
            let groups = s.shape.last ?? 1
            bits = max(1, packed * 32 / max(inDim, 1))
            groupSize = max(1, inDim / max(groups, 1))
        } else {
            scales = nil
            biases = nil
            bits = 0
            groupSize = 0
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let scales {
            y = quantizedMM(
                x, weight, scales: scales, biases: biases, transpose: true,
                groupSize: groupSize, bits: bits)
        } else {
            y = matmul(x, weight.T)
        }
        if let bias { y = y + bias }
        return y
    }
}

/// A token embedding, plain or quantized (the text encoders' tables are).
struct DEmbedding {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int

    init(_ w: WeightStore, _ prefix: String, dim: Int) throws {
        weight = try w.take(prefix + ".weight")
        if let s = w.maybe(prefix + ".scales") {
            scales = s
            biases = w.maybe(prefix + ".biases")
            let packed = weight.shape.last ?? 0
            bits = max(1, packed * 32 / max(dim, 1))
            groupSize = max(1, dim / max(s.shape.last ?? 1, 1))
        } else {
            scales = nil
            biases = nil
            bits = 0
            groupSize = 0
        }
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        guard let scales else { return weight[ids] }
        return dequantized(
            weight[ids], scales: scales[ids], biases: biases?[ids], groupSize: groupSize, bits: bits)
    }
}

struct DRMSNorm {
    let weight: MLXArray
    let eps: Float
    init(_ w: WeightStore, _ prefix: String, eps: Float) throws {
        weight = try w.take(prefix + ".weight")
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// LayerNorm, optionally without learned parameters.
struct DLayerNorm {
    let weight: MLXArray?
    let bias: MLXArray?
    let eps: Float
    init(eps: Float) {
        weight = nil
        bias = nil
        self.eps = eps
    }
    init(_ w: WeightStore, _ prefix: String, eps: Float) throws {
        weight = try w.take(prefix + ".weight")
        bias = w.maybe(prefix + ".bias")
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

/// GroupNorm over channels-last input, computed the PyTorch way (statistics
/// per group across all positions and the group's channels) in fp32.
struct DGroupNorm {
    let weight: MLXArray
    let bias: MLXArray
    let groups: Int
    let eps: Float
    init(_ w: WeightStore, _ prefix: String, groups: Int = 32, eps: Float = 1e-6) throws {
        weight = try w.take(prefix + ".weight")
        bias = try w.take(prefix + ".bias")
        self.groups = groups
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let c = shape.last!
        let dtype = x.dtype
        let b = shape[0]
        var h = x.asType(.float32).reshaped([b, -1, groups, c / groups])
        let mu = mean(h, axes: [1, 3], keepDims: true)
        let v = variance(h, axes: [1, 3], keepDims: true)
        h = (h - mu) * rsqrt(v + eps)
        h = h.reshaped(shape)
        return (h * weight.asType(.float32) + bias.asType(.float32)).asType(dtype)
    }
}

/// A 2-D convolution on channels-last input; weights are stored the MLX way,
/// [out, kh, kw, in].
struct DConv2d {
    let weight: MLXArray
    let bias: MLXArray?
    let stride: Int
    let padding: Int
    init(_ w: WeightStore, _ prefix: String, stride: Int = 1, padding: Int = 1) throws {
        weight = try w.take(prefix + ".weight")
        bias = w.maybe(prefix + ".bias")
        self.stride = stride
        self.padding = padding
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: .init(stride), padding: .init(padding))
        if let bias { y = y + bias }
        return y
    }
}

/// Upsample channels-last input by repeating each pixel `scale`×`scale`.
func upsampleNearest(_ x: MLXArray, scale: Int = 2) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let y = broadcast(
        x.reshaped([b, h, 1, w, 1, c]), to: [b, h, scale, w, scale, c])
    return y.reshaped([b, h * scale, w * scale, c])
}

/// Sinusoidal timestep features, cos first — the diffusers/DiT layout.
func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000, flipSinToCos: Bool = true) -> MLXArray {
    let half = dim / 2
    let freqs = exp(-log(maxPeriod) * MLXArray(0 ..< half).asType(.float32) / Float(half))
    let args = t.asType(.float32).reshaped([-1, 1]) * freqs.reshaped([1, -1])
    var emb = flipSinToCos
        ? concatenated([cos(args), sin(args)], axis: -1)
        : concatenated([sin(args), cos(args)], axis: -1)
    if dim % 2 == 1 {
        emb = concatenated([emb, zeros([emb.dim(0), 1])], axis: -1)
    }
    return emb
}
