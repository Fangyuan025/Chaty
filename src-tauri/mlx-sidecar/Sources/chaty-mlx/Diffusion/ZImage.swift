// Z-Image and Z-Image Turbo (Tongyi-MAI) on MLX, from mflux's saves.
//
// The reference is mflux's own implementation (models/z_image); every shape
// and cast below follows it, so a Swift run and an mflux run of the same
// weights agree to rounding. Three parts:
//   • the text encoder — Qwen3-4B read up to its second-to-last layer, over
//     the prompt in Qwen3's chat template with thinking on;
//   • the transformer — a single-stream DiT: two noise-refiner blocks over the
//     image tokens, two context-refiner blocks over the caption, then thirty
//     blocks over both, each modulated by the timestep (scale and tanh gate);
//   • FLUX.1's VAE (16 latent channels, 8× down, scale 0.3611, shift 0.1159).
import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - Text encoder (Qwen3 / Qwen3-VL's language stack)

/// How a family reads its text encoder: the shapes, and the few places the
/// references differ (which hidden state, which precision).
struct QwenTextConfig {
    var hidden = 2560, layers = 36, heads = 32, kvHeads = 8, headDim = 128, intermediate = 9728
    var ropeTheta: Float = 1_000_000
    var eps: Float = 1e-6
    var qkEps: Float = 1e-6
    /// Layers run; the hidden state after the last of them is handed back.
    var run = 36
    var finalNorm = false
    /// Keep the residual stream in fp32 (mflux's Z-Image does).
    var f32Stream = false
    /// Attention in fp32 (mflux's Qwen3-VL does).
    var f32Attention = false
    /// Per-head q/k RMSNorm (Qwen3 has it, Qwen2.5 does not).
    var qkNorm = true
    /// Where the weights sit in the save ("encoder." for Qwen-Image's).
    var keyPrefix = ""

    /// Z-Image: Qwen3-4B's second-to-last hidden state, fp32 stream.
    static let zImage = QwenTextConfig(qkEps: 1e-5, run: 35, f32Stream: true)
    /// Qwen-Image 2.1: Qwen3-VL-8B's last layer, normed; attention in fp32.
    static let qwenImage21 = QwenTextConfig(
        hidden: 4096, intermediate: 12288, ropeTheta: 5_000_000, run: 36, finalNorm: true, f32Attention: true)
    /// Qwen-Image (1.0, 2512): Qwen2.5-VL-7B's last layer, normed.
    static let qwenImage = QwenTextConfig(
        hidden: 3584, layers: 28, heads: 28, kvHeads: 4, intermediate: 18944, run: 28, finalNorm: true,
        qkNorm: false, keyPrefix: "encoder.")
}

final class Qwen3TextEncoder {
    struct Layer {
        let inNorm: DRMSNorm, postNorm: DRMSNorm
        let q: DLinear, k: DLinear, v: DLinear, o: DLinear
        let qNorm: DRMSNorm?, kNorm: DRMSNorm?
        let gate: DLinear, up: DLinear, down: DLinear
    }

    let cfg: QwenTextConfig
    let embed: DEmbedding
    let layers: [Layer]
    let norm: DRMSNorm?

    init(_ w: WeightStore, _ cfg: QwenTextConfig) throws {
        self.cfg = cfg
        let kp = cfg.keyPrefix
        embed = try DEmbedding(w, "\(kp)embed_tokens", dim: cfg.hidden)
        var ls: [Layer] = []
        for i in 0 ..< cfg.run {
            let p = "\(kp)layers.\(i)"
            ls.append(
                Layer(
                    inNorm: try DRMSNorm(w, "\(p).input_layernorm", eps: cfg.eps),
                    postNorm: try DRMSNorm(w, "\(p).post_attention_layernorm", eps: cfg.eps),
                    q: try DLinear(w, "\(p).self_attn.q_proj", inDim: cfg.hidden),
                    k: try DLinear(w, "\(p).self_attn.k_proj", inDim: cfg.hidden),
                    v: try DLinear(w, "\(p).self_attn.v_proj", inDim: cfg.hidden),
                    o: try DLinear(w, "\(p).self_attn.o_proj", inDim: cfg.heads * cfg.headDim),
                    qNorm: cfg.qkNorm ? try DRMSNorm(w, "\(p).self_attn.q_norm", eps: cfg.qkEps) : nil,
                    kNorm: cfg.qkNorm ? try DRMSNorm(w, "\(p).self_attn.k_norm", eps: cfg.qkEps) : nil,
                    gate: try DLinear(w, "\(p).mlp.gate_proj", inDim: cfg.hidden),
                    up: try DLinear(w, "\(p).mlp.up_proj", inDim: cfg.hidden),
                    down: try DLinear(w, "\(p).mlp.down_proj", inDim: cfg.intermediate)))
        }
        layers = ls
        norm = cfg.finalNorm ? try DRMSNorm(w, "\(kp)norm", eps: cfg.eps) : nil
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    /// Hidden states for `ids` ([1, L]), [L, hidden] in bf16.
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let L = ids.dim(1), hd = cfg.headDim
        var h = embed(ids)
        if cfg.f32Stream { h = h.asType(.float32) }
        let invFreq = 1.0 / pow(MLXArray(cfg.ropeTheta), MLXArray(stride(from: 0, to: hd, by: 2)).asType(.float32) / Float(hd))
        let freqs = outer(MLXArray(0 ..< L).asType(.float32), invFreq)
        let emb = concatenated([freqs, freqs], axis: -1)
        let cosT = cos(emb).reshaped([1, L, 1, hd]).asType(h.dtype)
        let sinT = sin(emb).reshaped([1, L, 1, hd]).asType(h.dtype)
        let idx = MLXArray(0 ..< L)
        let maskF = MLX.where(
            expandedDimensions(idx, axis: 1) .>= expandedDimensions(idx, axis: 0),
            MLXArray(Float(0)), MLXArray(-Float.infinity)
        ).reshaped([1, 1, L, L])
        let groups = cfg.heads / cfg.kvHeads
        let scale = 1 / Float(hd).squareRoot()
        for l in layers {
            let x = l.inNorm(h)
            var q = l.q(x).reshaped([1, L, cfg.heads, hd])
            var k = l.k(x).reshaped([1, L, cfg.kvHeads, hd])
            var v = l.v(x).reshaped([1, L, cfg.kvHeads, hd])
            if let n = l.qNorm { q = n(q) }
            if let n = l.kNorm { k = n(k) }
            q = q * cosT + rotateHalf(q) * sinT
            k = k * cosT + rotateHalf(k) * sinT
            if groups > 1 {
                k = repeated(k, count: groups, axis: 2)
                v = repeated(v, count: groups, axis: 2)
            }
            var qt = q.transposed(0, 2, 1, 3), kt = k.transposed(0, 2, 1, 3), vt = v.transposed(0, 2, 1, 3)
            if cfg.f32Attention {
                qt = qt.asType(.float32)
                kt = kt.asType(.float32)
                vt = vt.asType(.float32)
            }
            let a = MLXFast.scaledDotProductAttention(
                queries: qt, keys: kt, values: vt, scale: scale, mask: maskF.asType(qt.dtype)
            ).asType(h.dtype).transposed(0, 2, 1, 3).reshaped([1, L, -1])
            h = h + l.o(a)
            let y = l.postNorm(h)
            h = h + l.down(silu(l.gate(y)) * l.up(y))
        }
        if let norm { h = norm(h) }
        return h[0].asType(.bfloat16)
    }
}

// MARK: - Transformer

final class ZImageTransformer {
    struct Block {
        let q: DLinear, k: DLinear, v: DLinear, o: DLinear
        let normQ: DRMSNorm, normK: DRMSNorm
        let w1: DLinear, w2: DLinear, w3: DLinear
        let attnNorm1: DRMSNorm, attnNorm2: DRMSNorm, ffnNorm1: DRMSNorm, ffnNorm2: DRMSNorm
        let ada: DLinear?
    }

    static let dim = 3840, heads = 30, headDim = 128, patch = 2, channels = 16
    var dim: Int { Self.dim }
    var heads: Int { Self.heads }
    var headDim: Int { Self.headDim }
    var patch: Int { Self.patch }
    var channels: Int { Self.channels }
    let xEmbed: DLinear
    let finalLinear: DLinear, finalAda: DLinear
    let t1: DLinear, t2: DLinear
    let capNorm: DRMSNorm, capLinear: DLinear
    let xPad: MLXArray, capPad: MLXArray
    let noiseRefiner: [Block], contextRefiner: [Block], layers: [Block]
    /// cos/sin tables per rope axis: [len, dim/2, 2].
    let ropeTables: [MLXArray]

    init(_ w: WeightStore) throws {
        let key = "2-1"
        let dim = Self.dim, patch = Self.patch, channels = Self.channels
        let embedDim = patch * patch * channels
        xEmbed = try DLinear(w, "all_x_embedder.\(key)", inDim: embedDim)
        finalLinear = try DLinear(w, "all_final_layer.\(key).linear", inDim: dim)
        finalAda = try DLinear(w, "all_final_layer.\(key).adaLN_modulation.0", inDim: 256)
        t1 = try DLinear(w, "t_embedder.linear1", inDim: 256)
        t2 = try DLinear(w, "t_embedder.linear2", inDim: 1024)
        capNorm = try DRMSNorm(w, "cap_embedder.0", eps: 1e-5)
        capLinear = try DLinear(w, "cap_embedder.1", inDim: 2560)
        xPad = try w.take("x_pad_token")
        capPad = try w.take("cap_pad_token")
        func block(_ p: String, modulated: Bool) throws -> Block {
            let hidden = Int(Double(dim) / 3 * 8)
            return Block(
                q: try DLinear(w, "\(p).attention.to_q", inDim: dim),
                k: try DLinear(w, "\(p).attention.to_k", inDim: dim),
                v: try DLinear(w, "\(p).attention.to_v", inDim: dim),
                o: try DLinear(w, "\(p).attention.to_out.0", inDim: dim),
                normQ: try DRMSNorm(w, "\(p).attention.norm_q", eps: 1e-5),
                normK: try DRMSNorm(w, "\(p).attention.norm_k", eps: 1e-5),
                w1: try DLinear(w, "\(p).feed_forward.w1", inDim: dim),
                w2: try DLinear(w, "\(p).feed_forward.w2", inDim: hidden),
                w3: try DLinear(w, "\(p).feed_forward.w3", inDim: dim),
                attnNorm1: try DRMSNorm(w, "\(p).attention_norm1", eps: 1e-5),
                attnNorm2: try DRMSNorm(w, "\(p).attention_norm2", eps: 1e-5),
                ffnNorm1: try DRMSNorm(w, "\(p).ffn_norm1", eps: 1e-5),
                ffnNorm2: try DRMSNorm(w, "\(p).ffn_norm2", eps: 1e-5),
                ada: modulated ? try DLinear(w, "\(p).adaLN_modulation.0", inDim: 256) : nil)
        }
        noiseRefiner = try (0 ..< 2).map { try block("noise_refiner.\($0)", modulated: true) }
        contextRefiner = try (0 ..< 2).map { try block("context_refiner.\($0)", modulated: false) }
        var ls: [Block] = []
        var i = 0
        while w.has("layers.\(i).attention.to_q.weight") {
            ls.append(try block("layers.\(i)", modulated: true))
            i += 1
        }
        layers = ls
        // RopeEmbedder(theta 256, axes_dims [32, 48, 48], axes_lens [1024, 512, 512])
        var tables: [MLXArray] = []
        for (d, len) in [(32, 1024), (48, 512), (48, 512)] {
            let freqs = 1.0 / pow(MLXArray(Float(256)), MLXArray(stride(from: 0, to: d, by: 2)).asType(.float32) / Float(d))
            let f = outer(MLXArray(0 ..< len).asType(.float32), freqs)
            tables.append(stacked([cos(f), sin(f)], axis: -1))
        }
        eval(tables)
        ropeTables = tables
    }

    private func rope(_ ids: MLXArray) -> MLXArray {
        concatenated((0 ..< 3).map { ropeTables[$0][ids[0..., $0]] }, axis: 1)
    }

    private func applyRope(_ x: MLXArray, _ f: MLXArray) -> MLXArray {
        let (b, l, h, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let xr = x.reshaped([b, l, h, d / 2, 2])
        let ff = f.reshaped([1, l, 1, d / 2, 2])
        let re = xr[.ellipsis, 0], im = xr[.ellipsis, 1]
        let c = ff[.ellipsis, 0], s = ff[.ellipsis, 1]
        return stacked([re * c - im * s, re * s + im * c], axis: -1).reshaped([b, l, h, d])
    }

    private func attention(_ b: Block, _ x: MLXArray, _ freqs: MLXArray) -> MLXArray {
        let (n, l) = (x.dim(0), x.dim(1))
        var q = b.q(x).reshaped([n, l, heads, headDim])
        var k = b.k(x).reshaped([n, l, heads, headDim])
        let v = b.v(x).reshaped([n, l, heads, headDim])
        q = applyRope(b.normQ(q), freqs)
        k = applyRope(b.normK(k), freqs)
        let o = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
            scale: 1 / Float(headDim).squareRoot(), mask: nil
        ).transposed(0, 2, 1, 3).reshaped([n, l, dim])
        return b.o(o)
    }

    private func run(_ b: Block, _ x: MLXArray, _ freqs: MLXArray, _ tEmb: MLXArray?) -> MLXArray {
        if let ada = b.ada, let tEmb {
            let m = split(expandedDimensions(ada(tEmb), axis: 1), parts: 4, axis: 2)
            let scaleMsa = 1 + m[0], gateMsa = tanh(m[1]), scaleMlp = 1 + m[2], gateMlp = tanh(m[3])
            var h = x + gateMsa * b.attnNorm2(attention(b, b.attnNorm1(x) * scaleMsa, freqs))
            let y = b.ffnNorm1(h) * scaleMlp
            h = h + gateMlp * b.ffnNorm2(b.w2(silu(b.w1(y)) * b.w3(y)))
            return h
        }
        var h = x + b.attnNorm2(attention(b, b.attnNorm1(x), freqs))
        let y = b.ffnNorm1(h)
        h = h + b.ffnNorm2(b.w2(silu(b.w1(y)) * b.w3(y)))
        return h
    }

    private static func grid(_ f: Int, _ h: Int, _ w: Int, start: Int) -> MLXArray {
        var ids = [Int32]()
        ids.reserveCapacity(f * h * w * 3)
        for a in 0 ..< f { for b in 0 ..< h { for c in 0 ..< w { ids += [Int32(start + a), Int32(b), Int32(c)] } } }
        return MLXArray(ids, [f * h * w, 3])
    }

    /// The model's prediction for latent `x` ([16, 1, h, w]) at `timestep`
    /// (1 − σ), with the caption features `cap` ([L, 2560]).
    func callAsFunction(_ x: MLXArray, timestep: Float, cap: MLXArray) -> MLXArray {
        let tEmb = t2(silu(t1(timestepEmbedding(MLXArray([timestep * 1000]), dim: 256))))

        // Caption: padded to a multiple of 32 by repeating its last row.
        let capLen = cap.dim(0)
        let capPadLen = (32 - capLen % 32) % 32
        let capTotal = capLen + capPadLen
        var capFeats = cap
        if capPadLen > 0 { capFeats = concatenated([cap, repeated(cap[(capLen - 1)...], count: capPadLen, axis: 0)], axis: 0) }
        var capIds = [Int32]()
        for i in 0 ..< capTotal { capIds += [Int32(1 + i), 0, 0] }
        let capPos = MLXArray(capIds, [capTotal, 3])

        // Image: 2×2 patches.
        let (c, f, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let (ht, wt) = (h / patch, w / patch)
        var img = x.reshaped([c, f, 1, ht, patch, wt, patch]).transposed(1, 3, 5, 2, 4, 6, 0)
            .reshaped([f * ht * wt, patch * patch * c])
        let imgLen = img.dim(0)
        let imgPadLen = (32 - imgLen % 32) % 32
        var imgPos = Self.grid(f, ht, wt, start: capTotal + 1)
        if imgPadLen > 0 {
            imgPos = concatenated([imgPos, zeros([imgPadLen, 3], type: Int32.self)], axis: 0)
            img = concatenated([img, repeated(img[(imgLen - 1)...], count: imgPadLen, axis: 0)], axis: 0)
        }

        var xEmb = xEmbed(img)
        if imgPadLen > 0 {
            let padMask = concatenated([zeros([imgLen], type: Bool.self), ones([imgPadLen], type: Bool.self)])
            xEmb = MLX.where(expandedDimensions(padMask, axis: 1), xPad, xEmb)
        }
        let xFreqs = rope(imgPos)
        var xs = expandedDimensions(xEmb, axis: 0)
        for b in noiseRefiner { xs = run(b, xs, xFreqs, tEmb) }

        var capEmb = capLinear(capNorm(capFeats))
        if capPadLen > 0 {
            let padMask = concatenated([zeros([capLen], type: Bool.self), ones([capPadLen], type: Bool.self)])
            capEmb = MLX.where(expandedDimensions(padMask, axis: 1), capPad, capEmb)
        }
        let capFreqs = rope(capPos)
        var cs = expandedDimensions(capEmb, axis: 0)
        for b in contextRefiner { cs = run(b, cs, capFreqs, nil) }

        let xLen = xs.dim(1)
        var u = concatenated([xs, cs], axis: 1)
        let uFreqs = concatenated([xFreqs, capFreqs], axis: 0)
        for b in layers { u = run(b, u, uFreqs, tEmb) }

        let scale = expandedDimensions(1 + finalAda(silu(tEmb)), axis: 1)
        u = finalLinear(MLXFast.layerNorm(u, weight: nil, bias: nil, eps: 1e-6) * scale)
        let out = u[0, ..<imgLen].reshaped([f, ht, wt, 1, patch, patch, c])
            .transposed(6, 0, 3, 1, 4, 2, 5).reshaped([c, f, h, w])
        _ = xLen
        return -out
    }
}

// MARK: - VAE (FLUX.1's AutoencoderKL)

final class FluxVAE {
    struct Resnet {
        let norm1: DGroupNorm, conv1: DConv2d, norm2: DGroupNorm, conv2: DConv2d
        let shortcut: DConv2d?
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = conv1(silu(norm1(x)))
            h = conv2(silu(norm2(h)))
            return (shortcut?(x) ?? x) + h
        }
    }

    struct Attn {
        let norm: DGroupNorm
        let q: DLinear, k: DLinear, v: DLinear, o: DLinear
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let n = norm(x)
            let q = self.q(n).reshaped([b, 1, h * w, c])
            let k = self.k(n).reshaped([b, 1, h * w, c])
            let v = self.v(n).reshaped([b, 1, h * w, c])
            let a = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 1 / Float(c).squareRoot(), mask: nil)
            return x + o(a.reshaped([b, h, w, c]))
        }
    }

    let scaling: Float = 0.3611
    let shift: Float = 0.1159

    // decoder
    let dConvIn: DConv2d
    let dMid: (Resnet, Attn, Resnet)
    let dUp: [([Resnet], DConv2d?)]
    let dNormOut: DGroupNorm
    let dConvOut: DConv2d
    // encoder (for starting from a picture); absent in decoder-only saves
    let enc: Encoder?

    struct Encoder {
        let convIn: DConv2d
        let down: [([Resnet], DConv2d?)]
        let mid: (Resnet, Attn, Resnet)
        let normOut: DGroupNorm
        let convOut: DConv2d
    }

    private static func resnet(_ w: WeightStore, _ p: String, inC: Int, outC: Int) throws -> Resnet {
        Resnet(
            norm1: try DGroupNorm(w, "\(p).norm1"),
            conv1: try DConv2d(w, "\(p).conv1"),
            norm2: try DGroupNorm(w, "\(p).norm2"),
            conv2: try DConv2d(w, "\(p).conv2"),
            shortcut: w.has("\(p).conv_shortcut.weight") ? try DConv2d(w, "\(p).conv_shortcut", padding: 0) : nil)
    }

    private static func attn(_ w: WeightStore, _ p: String, c: Int) throws -> Attn {
        Attn(
            norm: try DGroupNorm(w, "\(p).group_norm"),
            q: try DLinear(w, "\(p).to_q", inDim: c), k: try DLinear(w, "\(p).to_k", inDim: c),
            v: try DLinear(w, "\(p).to_v", inDim: c), o: try DLinear(w, "\(p).to_out.0", inDim: c))
    }

    init(_ w: WeightStore) throws {
        // Z-Image's save names these `conv`, FLUX's `conv2d`.
        let io = w.has("decoder.conv_in.conv.weight") ? "conv" : "conv2d"
        dConvIn = try DConv2d(w, "decoder.conv_in.\(io)")
        dMid = (
            try Self.resnet(w, "decoder.mid_block.resnets.0", inC: 512, outC: 512),
            try Self.attn(w, "decoder.mid_block.attentions.0", c: 512),
            try Self.resnet(w, "decoder.mid_block.resnets.1", inC: 512, outC: 512))
        let chans = [(512, 512), (512, 512), (512, 256), (256, 128)]
        var up: [([Resnet], DConv2d?)] = []
        for (i, (inC, outC)) in chans.enumerated() {
            let p = "decoder.up_blocks.\(i)"
            let rs = try (0 ..< 3).map { try Self.resnet(w, "\(p).resnets.\($0)", inC: $0 == 0 ? inC : outC, outC: outC) }
            let ups = w.has("\(p).upsamplers.0.conv.weight") ? try DConv2d(w, "\(p).upsamplers.0.conv") : nil
            up.append((rs, ups))
        }
        dUp = up
        dNormOut = try DGroupNorm(w, "decoder.conv_norm_out.norm")
        dConvOut = try DConv2d(w, "decoder.conv_out.\(io)")

        if w.has("encoder.conv_in.conv2d.weight") {
            let echans = [(128, 128), (128, 256), (256, 512), (512, 512)]
            var down: [([Resnet], DConv2d?)] = []
            for (i, (inC, outC)) in echans.enumerated() {
                let p = "encoder.down_blocks.\(i)"
                let rs = try (0 ..< 2).map { try Self.resnet(w, "\(p).resnets.\($0)", inC: $0 == 0 ? inC : outC, outC: outC) }
                let ds = w.has("\(p).downsamplers.0.conv.weight")
                    ? try DConv2d(w, "\(p).downsamplers.0.conv", stride: 2, padding: 0) : nil
                down.append((rs, ds))
            }
            enc = Encoder(
                convIn: try DConv2d(w, "encoder.conv_in.conv2d"),
                down: down,
                mid: (
                    try Self.resnet(w, "encoder.mid_block.resnets.0", inC: 512, outC: 512),
                    try Self.attn(w, "encoder.mid_block.attentions.0", c: 512),
                    try Self.resnet(w, "encoder.mid_block.resnets.1", inC: 512, outC: 512)),
                normOut: try DGroupNorm(w, "encoder.conv_norm_out.norm"),
                convOut: try DConv2d(w, "encoder.conv_out.conv2d"))
        } else {
            enc = nil
        }
    }

    /// Latent [h, w, 16] (diffusion space) → picture [H, W, 3] in [-1, 1].
    func decode(_ latent: MLXArray) -> MLXArray {
        var h = (latent / scaling + shift).expandedDimensions(axis: 0)
        h = dConvIn(h)
        h = dMid.2(dMid.1(dMid.0(h)))
        for (rs, up) in dUp {
            for r in rs { h = r(h) }
            if let up { h = up(upsampleNearest(h)) }
        }
        h = dConvOut(silu(dNormOut(h)))
        return h[0]
    }

    /// Picture [H, W, 3] in [-1, 1] → latent [16, 1, h, w] (the mean).
    func encode(_ image: MLXArray) -> MLXArray? {
        guard let enc else { return nil }
        var h = enc.convIn(image.expandedDimensions(axis: 0))
        for (rs, ds) in enc.down {
            for r in rs { h = r(h) }
            if let ds { h = ds(padded(h, widths: [0, [0, 1], [0, 1], 0])) }
        }
        h = enc.mid.2(enc.mid.1(enc.mid.0(h)))
        h = enc.convOut(silu(enc.normOut(h)))
        let mean = h[0, 0..., 0..., ..<16]
        return ((mean - shift) * scaling).transposed(2, 0, 1).expandedDimensions(axis: 1)
    }
}
