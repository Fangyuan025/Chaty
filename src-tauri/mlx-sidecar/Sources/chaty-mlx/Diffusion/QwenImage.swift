// Qwen-Image (1.0, 2512) on MLX, from mflux's saves (mflux models/qwen).
//
//   • text: Qwen2.5-VL-7B's language stack over the prompt in Qwen-Image's
//     template, read from the normed last layer, the template's first 34
//     tokens dropped;
//   • transformer: a 20B dual-stream MMDiT — image and text each with their
//     own modulation, norms and MLP, meeting in one joint attention; 2×2
//     patches of a 16-channel latent;
//   • VAE: Wan 2.1's, 16 channels, 8× down. Its causal 3-D convolutions pad
//     time with zeros in front, so for a still picture only the last
//     temporal slice of each kernel sees the frame: they run as 2-D
//     convolutions with that slice.
import Foundation
import MLX
import MLXNN
import Tokenizers

// sd.cpp's wan_21_latent_rgb_proj / _bias: Qwen-Image's (Wan 2.1) latent space.
let wan21LatentRGB: [[Float]] = [
    [0.015123, -0.148418, 0.479828],
    [0.003652, -0.01068, -0.037142],
    [0.212264, 0.063033, 0.016779],
    [0.232999, 0.406476, 0.220125],
    [-0.051864, -0.082384, -0.069396],
    [0.085005, -0.161492, 0.010689],
    [-0.245369, -0.506846, -0.11701],
    [-0.151145, 0.017721, 0.007207],
    [-0.293239, -0.207936, -0.421135],
    [-0.187721, 0.050783, 0.177649],
    [-0.013067, 0.265964, 0.166578],
    [0.028327, 0.109329, 0.108642],
    [-0.205343, 0.043991, 0.148914],
    [0.014307, -0.048647, -0.007219],
    [0.21715, 0.053074, 0.319923],
    [0.155357, 0.083156, 0.06478],
]
let wan21LatentRGBBias: [Float] = [-0.27027, -0.234976, -0.456853]

/// A causal 3-D convolution as it acts on one frame: the kernel's last
/// temporal slice, as a 2-D convolution.
struct DConv3as2 {
    let weight: MLXArray
    let bias: MLXArray?
    let padding: Int
    init(_ w: WeightStore, _ prefix: String, padding: Int = 1) throws {
        let full = try w.take(prefix + ".weight")  // [O, T, kh, kw, I]
        weight = full[0..., full.dim(1) - 1]
        bias = w.maybe(prefix + ".bias")
        self.padding = padding
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: .init(1), padding: .init(padding))
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - Transformer

final class QwenImageTransformer {
    struct Stream {
        let mod: DLinear
        let q: DLinear, k: DLinear, v: DLinear, out: DLinear
        let normQ: DRMSNorm, normK: DRMSNorm
        let mlpIn: DLinear, mlpOut: DLinear
    }
    struct Block { let img: Stream, txt: Stream }

    static let dim = 3072, heads = 24, headDim = 128
    let imgIn: DLinear, txtNorm: DRMSNorm, txtIn: DLinear
    let t1: DLinear, t2: DLinear
    let normOut: DLinear, projOut: DLinear
    let blocks: [Block]

    init(_ w: WeightStore) throws {
        let dim = Self.dim
        imgIn = try DLinear(w, "img_in", inDim: 64)
        txtNorm = try DRMSNorm(w, "txt_norm", eps: 1e-6)
        txtIn = try DLinear(w, "txt_in", inDim: 3584)
        t1 = try DLinear(w, "time_text_embed.timestep_embedder.linear_1", inDim: 256)
        t2 = try DLinear(w, "time_text_embed.timestep_embedder.linear_2", inDim: dim)
        normOut = try DLinear(w, "norm_out.linear", inDim: dim)
        projOut = try DLinear(w, "proj_out", inDim: dim)
        var bs: [Block] = []
        var i = 0
        while w.has("transformer_blocks.\(i).img_mod_linear.weight") {
            let p = "transformer_blocks.\(i)"
            let img = Stream(
                mod: try DLinear(w, "\(p).img_mod_linear", inDim: dim),
                q: try DLinear(w, "\(p).attn.to_q", inDim: dim), k: try DLinear(w, "\(p).attn.to_k", inDim: dim),
                v: try DLinear(w, "\(p).attn.to_v", inDim: dim), out: try DLinear(w, "\(p).attn.attn_to_out.0", inDim: dim),
                normQ: try DRMSNorm(w, "\(p).attn.norm_q", eps: 1e-6), normK: try DRMSNorm(w, "\(p).attn.norm_k", eps: 1e-6),
                mlpIn: try DLinear(w, "\(p).img_ff.mlp_in", inDim: dim), mlpOut: try DLinear(w, "\(p).img_ff.mlp_out", inDim: dim * 4))
            let txt = Stream(
                mod: try DLinear(w, "\(p).txt_mod_linear", inDim: dim),
                q: try DLinear(w, "\(p).attn.add_q_proj", inDim: dim), k: try DLinear(w, "\(p).attn.add_k_proj", inDim: dim),
                v: try DLinear(w, "\(p).attn.add_v_proj", inDim: dim), out: try DLinear(w, "\(p).attn.to_add_out", inDim: dim),
                normQ: try DRMSNorm(w, "\(p).attn.norm_added_q", eps: 1e-6),
                normK: try DRMSNorm(w, "\(p).attn.norm_added_k", eps: 1e-6),
                mlpIn: try DLinear(w, "\(p).txt_ff.mlp_in", inDim: dim), mlpOut: try DLinear(w, "\(p).txt_ff.mlp_out", inDim: dim * 4))
            bs.append(Block(img: img, txt: txt))
            i += 1
        }
        blocks = bs
    }

    private static func angles(_ pos: [Float], _ d: Int) -> MLXArray {
        let omega = 1.0 / pow(MLXArray(Float(10000)), MLXArray(stride(from: 0, to: d, by: 2)).asType(.float32) / Float(d))
        return outer(MLXArray(pos), omega)
    }

    /// Image rope (frame 0, height and width centred on zero) and text rope
    /// (every axis at max(h/2, w/2) + i).
    private func rope(textLen L: Int, h: Int, w: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        var fr = [Float](), hi = [Float](), wi = [Float]()
        for y in -(h - h / 2) ..< (h / 2) {
            for x in -(w - w / 2) ..< (w / 2) { fr.append(0); hi.append(Float(y)); wi.append(Float(x)) }
        }
        let img = concatenated([Self.angles(fr, 16), Self.angles(hi, 56), Self.angles(wi, 56)], axis: -1)
        let start = max(h / 2, w / 2)
        let tp = (0 ..< L).map { Float(start + $0) }
        let txt = concatenated([Self.angles(tp, 16), Self.angles(tp, 56), Self.angles(tp, 56)], axis: -1)
        return (cos(img), sin(img), cos(txt), sin(txt))
    }

    private func applyRope(_ x: MLXArray, _ c: MLXArray, _ s: MLXArray) -> MLXArray {
        let (b, l, h, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let xf = x.asType(.float32).reshaped([b, l, h, d / 2, 2])
        let re = xf[.ellipsis, 0], im = xf[.ellipsis, 1]
        let cc = c.reshaped([1, l, 1, d / 2]), ss = s.reshaped([1, l, 1, d / 2])
        return stacked([re * cc - im * ss, re * ss + im * cc], axis: -1).reshaped([b, l, h, d]).asType(x.dtype)
    }

    private static func ln(_ x: MLXArray) -> MLXArray { MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6) }

    func callAsFunction(_ x: MLXArray, sigma: Float, cond: MLXArray, h: Int, w: Int) -> MLXArray {
        let dim = Self.dim, heads = Self.heads, hd = Self.headDim
        let L = cond.dim(0), N = x.dim(1)
        var img = imgIn(x)
        var txt = txtIn(txtNorm(cond.expandedDimensions(axis: 0)))
        // mflux casts the timestep and its frequencies to the stream's dtype
        // before the sinusoid; the rounding is part of what the model sees.
        let t = MLXArray([sigma]).asType(.bfloat16).asType(.float32)
        let freqs = exp(-log(Float(10000)) * MLXArray(0 ..< 128).asType(.float32) / 128).asType(.bfloat16).asType(.float32)
        let args = t.reshaped([1, 1]) * freqs.reshaped([1, -1]) * 1000
        let temb = t2(silu(t1(concatenated([cos(args), sin(args)], axis: -1).asType(.bfloat16))))  // [1, dim]
        let st = silu(temb)
        let (ic, isn, tc, tsn) = rope(textLen: L, h: h, w: w)
        let scale = 1 / Float(hd).squareRoot()
        func modulate(_ x: MLXArray, _ m: [MLXArray]) -> MLXArray {
            x * (1 + m[1].expandedDimensions(axis: 1)) + m[0].expandedDimensions(axis: 1)
        }
        for b in blocks {
            let im = split(b.img.mod(st), parts: 2, axis: -1), tm = split(b.txt.mod(st), parts: 2, axis: -1)
            let im1 = split(im[0], parts: 3, axis: -1), im2 = split(im[1], parts: 3, axis: -1)
            let tm1 = split(tm[0], parts: 3, axis: -1), tm2 = split(tm[1], parts: 3, axis: -1)
            let ia = modulate(Self.ln(img), im1), ta = modulate(Self.ln(txt), tm1)
            func qkv(_ s: Stream, _ a: MLXArray, _ n: Int, _ c: MLXArray, _ sn: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
                let q = applyRope(s.normQ(s.q(a).reshaped([1, n, heads, hd])), c, sn)
                let k = applyRope(s.normK(s.k(a).reshaped([1, n, heads, hd])), c, sn)
                return (q, k, s.v(a).reshaped([1, n, heads, hd]))
            }
            let (iq, ik, iv) = qkv(b.img, ia, N, ic, isn)
            let (tq, tk, tv) = qkv(b.txt, ta, L, tc, tsn)
            let o = MLXFast.scaledDotProductAttention(
                queries: concatenated([tq, iq], axis: 1).transposed(0, 2, 1, 3),
                keys: concatenated([tk, ik], axis: 1).transposed(0, 2, 1, 3),
                values: concatenated([tv, iv], axis: 1).transposed(0, 2, 1, 3),
                scale: scale, mask: .none
            ).transposed(0, 2, 1, 3).reshaped([1, L + N, dim])
            img = img + im1[2].expandedDimensions(axis: 1) * b.img.out(o[0..., L...])
            txt = txt + tm1[2].expandedDimensions(axis: 1) * b.txt.out(o[0..., ..<L])
            img = img + im2[2].expandedDimensions(axis: 1) * b.img.mlpOut(geluApproximate(b.img.mlpIn(modulate(Self.ln(img), im2))))
            txt = txt + tm2[2].expandedDimensions(axis: 1) * b.txt.mlpOut(geluApproximate(b.txt.mlpIn(modulate(Self.ln(txt), tm2))))
        }
        let ss = split(normOut(st.asType(.bfloat16)), parts: 2, axis: -1)
        img = Self.ln(img) * (1 + ss[0].expandedDimensions(axis: 1)) + ss[1].expandedDimensions(axis: 1)
        return projOut(img)
    }
}

// MARK: - VAE (Wan 2.1)

final class QwenImageVAE {
    struct Norm {
        let w: MLXArray
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let n = sqrt((x * x).sum(axis: -1, keepDims: true))
            let scale = Float(x.dim(-1)).squareRoot()
            return x / maximum(n, MLXArray(Float(1e-12))) * scale * w.reshaped([-1])
        }
    }
    struct Res {
        let n1: Norm, c1: DConv3as2, n2: Norm, c2: DConv3as2, skip: DConv3as2?
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = c1(silu(n1(x)))
            h = c2(silu(n2(h)))
            return h + (skip?(x) ?? x)
        }
    }
    struct Attn {
        let n: Norm, qkv: DConv2d, proj: DConv2d
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let t = split(qkv(n(x)).reshaped([b, h * w, 3 * c]), parts: 3, axis: -1)
            let scores = matmul(t[0], t[1].transposed(0, 2, 1)) * (1 / Float(c).squareRoot())
            return proj(matmul(softmax(scores, axis: -1), t[2]).reshaped([b, h, w, c])) + x
        }
    }

    static let latentMean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    static let latentStd: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ]

    let postQuant: DConv3as2, quant: DConv3as2?
    let dIn: DConv3as2, dMid: (Res, Attn, Res)
    let dUp: [(res: [Res], up: DConv2d?)]
    let dNorm: Norm, dOut: DConv3as2
    let enc: (convIn: DConv3as2, down: [(res: [Res], down: DConv2d?)], mid: (Res, Attn, Res), norm: Norm, convOut: DConv3as2)?

    private static func res(_ w: WeightStore, _ p: String) throws -> Res {
        Res(
            n1: Norm(w: try w.take("\(p).norm1.weight")), c1: try DConv3as2(w, "\(p).conv1.conv3d"),
            n2: Norm(w: try w.take("\(p).norm2.weight")), c2: try DConv3as2(w, "\(p).conv2.conv3d"),
            skip: w.has("\(p).skip_conv.conv3d.weight") ? try DConv3as2(w, "\(p).skip_conv.conv3d", padding: 0) : nil)
    }
    private static func mid(_ w: WeightStore, _ p: String) throws -> (Res, Attn, Res) {
        (
            try res(w, "\(p).resnets.0"),
            Attn(
                n: Norm(w: try w.take("\(p).attentions.0.norm.weight")),
                qkv: try DConv2d(w, "\(p).attentions.0.to_qkv", padding: 0),
                proj: try DConv2d(w, "\(p).attentions.0.proj", padding: 0)),
            try res(w, "\(p).resnets.1"))
    }

    init(_ w: WeightStore) throws {
        postQuant = try DConv3as2(w, "post_quant_conv.conv3d", padding: 0)
        quant = w.has("quant_conv.conv3d.weight") ? try DConv3as2(w, "quant_conv.conv3d", padding: 0) : nil
        dIn = try DConv3as2(w, "decoder.conv_in.conv3d")
        dMid = try Self.mid(w, "decoder.mid_block")
        dUp = try (0 ..< 4).map { i in
            let p = "decoder.up_block\(i)"
            let rs = try (0 ..< 3).map { try Self.res(w, "\(p).resnets.\($0)") }
            let up = w.has("\(p).upsamplers.0.resample_conv.weight") ? try DConv2d(w, "\(p).upsamplers.0.resample_conv") : nil
            return (rs, up)
        }
        dNorm = Norm(w: try w.take("decoder.norm_out.weight"))
        dOut = try DConv3as2(w, "decoder.conv_out.conv3d")
        if w.has("encoder.conv_in.conv3d.weight") {
            let down = try (0 ..< 4).map { i -> (res: [Res], down: DConv2d?) in
                let p = "encoder.down_blocks.\(i)"
                let rs = try (0 ..< 2).map { try Self.res(w, "\(p).resnets.\($0)") }
                let d = w.has("\(p).downsamplers.0.resample_conv.weight")
                    ? try DConv2d(w, "\(p).downsamplers.0.resample_conv", stride: 2, padding: 0) : nil
                return (rs, d)
            }
            enc = (
                try DConv3as2(w, "encoder.conv_in.conv3d"), down, try Self.mid(w, "encoder.mid_block"),
                Norm(w: try w.take("encoder.norm_out.weight")), try DConv3as2(w, "encoder.conv_out.conv3d"))
        } else {
            enc = nil
        }
    }

    /// Latent [h, w, 16] (diffusion space) → picture [H, W, 3] in [-1, 1].
    func decode(_ latent: MLXArray) -> MLXArray {
        var x = (latent.asType(.float32) * MLXArray(Self.latentStd) + MLXArray(Self.latentMean)).expandedDimensions(axis: 0)
        x = dIn(postQuant(x))
        x = dMid.2(dMid.1(dMid.0(x)))
        for u in dUp {
            for r in u.res { x = r(x) }
            if let up = u.up { x = up(repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)) }
        }
        x = dOut(silu(dNorm(x)))
        return x[0]
    }

    /// Picture [H, W, 3] in [-1, 1] → latent [h, w, 16].
    func encode(_ image: MLXArray) -> MLXArray? {
        guard let enc, let quant else { return nil }
        var x = enc.convIn(image.asType(.float32).expandedDimensions(axis: 0))
        for d in enc.down {
            for r in d.res { x = r(x) }
            if let dn = d.down { x = dn(padded(x, widths: [0, [0, 1], [0, 1], 0])) }
        }
        x = enc.mid.2(enc.mid.1(enc.mid.0(x)))
        x = quant(enc.convOut(silu(enc.norm(x))))
        return ((x[0, 0..., 0..., ..<16] - MLXArray(Self.latentMean)) / MLXArray(Self.latentStd))
    }
}

// MARK: - Family

final class QwenImageFamily: ImageFamily {
    let version = "Qwen Image"
    let latentChannels = 16
    let vaeScale = 8
    let align = 16
    let flowKind = FlowKind.discrete
    let defaultShift: Float = 3
    let defaultScheduler = "discrete"
    var previewProj: [[Float]] { wan21LatentRGB }
    var previewBias: [Float] { wan21LatentRGBBias }

    static let template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    let tokenizer: Tokenizer
    let textEncoder: Qwen3TextEncoder
    let transformer: QwenImageTransformer
    let vae: QwenImageVAE
    private var grid = (h: 64, w: 64)

    init(dir: URL, load: LoadReporter) async throws {
        tokenizer = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer"))
        textEncoder = try Qwen3TextEncoder(load.store(dir, "text_encoder"), .qwenImage)
        transformer = try QwenImageTransformer(load.store(dir, "transformer"))
        vae = try QwenImageVAE(load.store(dir, "vae"))
    }

    func encode(_ prompt: String) throws -> MLXArray {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? " " : prompt
        var ids = tokenizer.encode(text: Self.template.replacingOccurrences(of: "{}", with: p), addSpecialTokens: true)
        if ids.count > 1058 { ids = Array(ids[0 ..< 1058]) }
        return textEncoder(MLXArray(ids.map { Int32($0) }).reshaped([1, -1]))[34...]
    }

    func velocity(_ x: MLXArray, sigma: Float, cond: MLXArray) -> MLXArray {
        transformer(x.asType(.bfloat16), sigma: sigma, cond: cond, h: grid.h, w: grid.w)
    }

    func latentShape(width: Int, height: Int) -> [Int] {
        grid = (height / 16, width / 16)
        return [1, grid.h * grid.w, 64]
    }

    /// Tokens (2×2 patches, channel-major) → [2h, 2w, 16].
    func hwc(_ latent: MLXArray) -> MLXArray {
        latent.reshaped([grid.h, grid.w, 16, 2, 2]).transposed(0, 3, 1, 4, 2).reshaped([grid.h * 2, grid.w * 2, 16])
    }

    func decodeHWC(_ latent: MLXArray) -> MLXArray { vae.decode(latent) }

    func encodeImage(_ image: MLXArray) -> MLXArray? {
        guard let lat = vae.encode(image) else { return nil }
        let (h, w) = (lat.dim(0) / 2, lat.dim(1) / 2)
        return lat.reshaped([h, 2, w, 2, 16]).transposed(0, 2, 4, 1, 3).reshaped([1, h * w, 64])
    }
}
