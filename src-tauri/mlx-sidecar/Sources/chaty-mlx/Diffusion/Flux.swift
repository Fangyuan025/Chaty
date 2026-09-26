// FLUX.1 (schnell, dev, Krea) on MLX, from mflux's saves (mflux
// models/flux is the reference).
//
//   • text: CLIP-L's pooled vector (the EOS token after the final norm) and
//     T5-XXL's hidden states over the prompt padded to 256 tokens (schnell)
//     or 512 (dev) — T5 runs with no attention mask, so the padding is part
//     of the conditioning;
//   • transformer: 19 double-stream blocks, then 38 single-stream blocks over
//     [text | image]; dev adds a distilled-guidance embedding;
//   • VAE: FLUX's AutoencoderKL — the same network as Z-Image's.
import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - CLIP-L

final class CLIPTextEncoder {
    struct Layer {
        let ln1: DLayerNorm, ln2: DLayerNorm
        let q: DLinear, k: DLinear, v: DLinear, o: DLinear
        let fc1: DLinear, fc2: DLinear
    }
    let tok: DEmbedding, pos: DEmbedding
    let layers: [Layer]
    let finalNorm: DLayerNorm

    init(_ w: WeightStore) throws {
        let p = "text_model"
        tok = try DEmbedding(w, "\(p).embeddings.token_embedding", dim: 768)
        pos = try DEmbedding(w, "\(p).embeddings.position_embedding", dim: 768)
        layers = try (0 ..< 12).map { i in
            let l = "\(p).encoder.layers.\(i)"
            return Layer(
                ln1: try DLayerNorm(w, "\(l).layer_norm1", eps: 1e-5), ln2: try DLayerNorm(w, "\(l).layer_norm2", eps: 1e-5),
                q: try DLinear(w, "\(l).self_attn.q_proj", inDim: 768), k: try DLinear(w, "\(l).self_attn.k_proj", inDim: 768),
                v: try DLinear(w, "\(l).self_attn.v_proj", inDim: 768), o: try DLinear(w, "\(l).self_attn.out_proj", inDim: 768),
                fc1: try DLinear(w, "\(l).mlp.fc1", inDim: 768), fc2: try DLinear(w, "\(l).mlp.fc2", inDim: 3072))
        }
        finalNorm = try DLayerNorm(w, "\(p).final_layer_norm", eps: 1e-5)
    }

    /// The pooled vector ([1, 768]) for 77 token ids.
    func callAsFunction(_ ids: [Int]) -> MLXArray {
        let L = ids.count
        let t = MLXArray(ids.map { Int32($0) }).reshaped([1, L])
        var h = tok(t) + pos(MLXArray(0 ..< L).reshaped([1, L]))
        let idx = MLXArray(0 ..< L)
        let mask = MLX.where(
            expandedDimensions(idx, axis: 1) .>= expandedDimensions(idx, axis: 0), MLXArray(Float(0)), MLXArray(Float(-3.4e38))
        ).reshaped([1, 1, L, L]).asType(.bfloat16)
        for l in layers {
            let x = l.ln1(h)
            let q = l.q(x).reshaped([1, L, 12, 64]).transposed(0, 2, 1, 3)
            let k = l.k(x).reshaped([1, L, 12, 64]).transposed(0, 2, 1, 3)
            let v = l.v(x).reshaped([1, L, 12, 64]).transposed(0, 2, 1, 3)
            let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / 8, mask: mask.asType(q.dtype))
                .transposed(0, 2, 1, 3).reshaped([1, L, 768])
            h = h + l.o(a)
            let y = l.fc1(l.ln2(h))
            h = h + l.fc2(y * sigmoid(1.702 * y))
        }
        h = finalNorm(h)
        let eos = ids.enumerated().max { $0.element < $1.element }!.offset  // first occurrence of the largest id
        return h[0..., eos]
    }
}

// MARK: - T5-XXL

final class T5Encoder {
    struct Block {
        let ln1: MLXArray, q: DLinear, k: DLinear, v: DLinear, o: DLinear, bias: DEmbedding
        let ln2: MLXArray, wi0: DLinear, wi1: DLinear, wo: DLinear
    }
    let shared: DEmbedding
    let blocks: [Block]
    let finalNorm: MLXArray

    init(_ w: WeightStore) throws {
        shared = try DEmbedding(w, "shared", dim: 4096)
        blocks = try (0 ..< 24).map { i in
            let p = "t5_blocks.\(i)"
            let a = "\(p).attention.SelfAttention"
            return Block(
                ln1: try w.take("\(p).attention.layer_norm.weight"),
                q: try DLinear(w, "\(a).q", inDim: 4096), k: try DLinear(w, "\(a).k", inDim: 4096),
                v: try DLinear(w, "\(a).v", inDim: 4096), o: try DLinear(w, "\(a).o", inDim: 4096),
                bias: try DEmbedding(w, "\(a).relative_attention_bias", dim: 64),
                ln2: try w.take("\(p).ff.layer_norm.weight"),
                wi0: try DLinear(w, "\(p).ff.DenseReluDense.wi_0", inDim: 4096),
                wi1: try DLinear(w, "\(p).ff.DenseReluDense.wi_1", inDim: 4096),
                wo: try DLinear(w, "\(p).ff.DenseReluDense.wo", inDim: 10240))
        }
        finalNorm = try w.take("final_layer_norm.weight")
    }

    private static func norm(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        let v = mean(x.asType(.float32) * x.asType(.float32), axis: -1, keepDims: true)
        return w * (x * rsqrt(v + 1e-6))
    }

    /// T5's bidirectional relative-position buckets (32 buckets, max distance 128).
    private static func buckets(_ L: Int) -> MLXArray {
        var out = [Int32]()
        let nb = 16, maxExact = 8
        for i in 0 ..< L {
            for j in 0 ..< L {
                let rel = j - i
                var b = rel > 0 ? nb : 0
                let r = abs(rel)
                if r < maxExact {
                    b += r
                } else {
                    let large = maxExact + Int(floor(log(Float(r) / Float(maxExact)) / log(Float(128) / Float(maxExact)) * Float(nb - maxExact)))
                    b += min(large, nb - 1)
                }
                out.append(Int32(b))
            }
        }
        return MLXArray(out, [L, L])
    }

    func callAsFunction(_ ids: [Int]) -> MLXArray {
        let L = ids.count
        var h = shared(MLXArray(ids.map { Int32($0) }).reshaped([1, L]))
        let bk = Self.buckets(L)
        for b in blocks {
            let x = Self.norm(h, b.ln1)
            let q = b.q(x).reshaped([1, L, 64, 64]).transposed(0, 2, 1, 3)
            let k = b.k(x).reshaped([1, L, 64, 64]).transposed(0, 2, 1, 3)
            let v = b.v(x).reshaped([1, L, 64, 64]).transposed(0, 2, 1, 3)
            let pb = b.bias(bk).transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, 64, L, L]
            let scores = matmul(q, k.transposed(0, 1, 3, 2)) + pb
            let a = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3).reshaped([1, L, 4096])
            h = h + b.o(a)
            let y = Self.norm(h, b.ln2)
            let g = b.wi0(y)
            let gelu = 0.5 * g * (1 + tanh(Float((2 / Double.pi).squareRoot()) * (g + 0.044715 * g * g * g)))
            h = h + b.wo(gelu * b.wi1(y))
        }
        return Self.norm(h, finalNorm)
    }
}

// MARK: - Transformer

final class FluxTransformer {
    struct DoubleBlock {
        let mod: DLinear, modC: DLinear
        let q: DLinear, k: DLinear, v: DLinear, o: DLinear
        let aq: DLinear, ak: DLinear, av: DLinear, ao: DLinear
        let nq: MLXArray, nk: MLXArray, naq: MLXArray, nak: MLXArray
        let ff1: DLinear, ff2: DLinear, cf1: DLinear, cf2: DLinear
    }
    struct Single {
        let mod: DLinear, q: DLinear, k: DLinear, v: DLinear, nq: MLXArray, nk: MLXArray
        let mlp: DLinear, out: DLinear
    }

    let xEmbed: DLinear, ctxEmbed: DLinear
    let tl1: DLinear, tl2: DLinear, xl1: DLinear, xl2: DLinear
    let gl1: DLinear?, gl2: DLinear?
    let doubles: [DoubleBlock], singles: [Single]
    let normOut: DLinear, projOut: DLinear
    var hasGuidance: Bool { gl1 != nil }

    init(_ w: WeightStore) throws {
        let d = 3072
        xEmbed = try DLinear(w, "x_embedder", inDim: 64)
        ctxEmbed = try DLinear(w, "context_embedder", inDim: 4096)
        tl1 = try DLinear(w, "time_text_embed.timestep_embedder.linear_1", inDim: 256)
        tl2 = try DLinear(w, "time_text_embed.timestep_embedder.linear_2", inDim: d)
        xl1 = try DLinear(w, "time_text_embed.text_embedder.linear_1", inDim: 768)
        xl2 = try DLinear(w, "time_text_embed.text_embedder.linear_2", inDim: d)
        if w.has("time_text_embed.guidance_embedder.linear_1.weight") {
            gl1 = try DLinear(w, "time_text_embed.guidance_embedder.linear_1", inDim: 256)
            gl2 = try DLinear(w, "time_text_embed.guidance_embedder.linear_2", inDim: d)
        } else {
            gl1 = nil
            gl2 = nil
        }
        var ds: [DoubleBlock] = []
        var i = 0
        while w.has("transformer_blocks.\(i).attn.to_q.weight") {
            let p = "transformer_blocks.\(i)"
            ds.append(
                DoubleBlock(
                    mod: try DLinear(w, "\(p).norm1.linear", inDim: d), modC: try DLinear(w, "\(p).norm1_context.linear", inDim: d),
                    q: try DLinear(w, "\(p).attn.to_q", inDim: d), k: try DLinear(w, "\(p).attn.to_k", inDim: d),
                    v: try DLinear(w, "\(p).attn.to_v", inDim: d), o: try DLinear(w, "\(p).attn.to_out.0", inDim: d),
                    aq: try DLinear(w, "\(p).attn.add_q_proj", inDim: d), ak: try DLinear(w, "\(p).attn.add_k_proj", inDim: d),
                    av: try DLinear(w, "\(p).attn.add_v_proj", inDim: d), ao: try DLinear(w, "\(p).attn.to_add_out", inDim: d),
                    nq: try w.take("\(p).attn.norm_q.weight"), nk: try w.take("\(p).attn.norm_k.weight"),
                    naq: try w.take("\(p).attn.norm_added_q.weight"), nak: try w.take("\(p).attn.norm_added_k.weight"),
                    ff1: try DLinear(w, "\(p).ff.linear1", inDim: d), ff2: try DLinear(w, "\(p).ff.linear2", inDim: d * 4),
                    cf1: try DLinear(w, "\(p).ff_context.linear1", inDim: d), cf2: try DLinear(w, "\(p).ff_context.linear2", inDim: d * 4)))
            i += 1
        }
        doubles = ds
        var ss: [Single] = []
        i = 0
        while w.has("single_transformer_blocks.\(i).attn.to_q.weight") {
            let p = "single_transformer_blocks.\(i)"
            ss.append(
                Single(
                    mod: try DLinear(w, "\(p).norm.linear", inDim: d),
                    q: try DLinear(w, "\(p).attn.to_q", inDim: d), k: try DLinear(w, "\(p).attn.to_k", inDim: d),
                    v: try DLinear(w, "\(p).attn.to_v", inDim: d),
                    nq: try w.take("\(p).attn.norm_q.weight"), nk: try w.take("\(p).attn.norm_k.weight"),
                    mlp: try DLinear(w, "\(p).proj_mlp", inDim: d), out: try DLinear(w, "\(p).proj_out", inDim: d * 5)))
            i += 1
        }
        singles = ss
        normOut = try DLinear(w, "norm_out.linear", inDim: d)
        projOut = try DLinear(w, "proj_out", inDim: d)
    }

    private static func timeProj(_ t: MLXArray) -> MLXArray {
        let freqs = exp(-log(Float(10000)) * MLXArray(0 ..< 128).asType(.float32) / 128)
        let a = t.asType(.float32).reshaped([-1, 1]) * freqs.reshaped([1, -1])
        return concatenated([cos(a), sin(a)], axis: -1)
    }

    private static func ln(_ x: MLXArray) -> MLXArray { MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6) }

    /// Per-head RMSNorm in fp32 (eps 1e-5, nn.RMSNorm's default).
    private static func headNorm(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: w.asType(.float32), eps: 1e-5).asType(x.dtype)
    }

    private static func rope(_ x: MLXArray, _ c: MLXArray, _ s: MLXArray) -> MLXArray {
        let (b, h, l, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let xf = x.asType(.float32).reshaped([b, h, l, d / 2, 2])
        let re = xf[.ellipsis, 0], im = xf[.ellipsis, 1]
        let cc = c.reshaped([1, 1, l, d / 2]), ss = s.reshaped([1, 1, l, d / 2])
        return stacked([cc * re - ss * im, ss * re + cc * im], axis: -1).reshaped([b, h, l, d])
    }

    func callAsFunction(
        _ x: MLXArray, sigma: Float, t5: MLXArray, pooled: MLXArray, guidance: Float, h: Int, w: Int
    ) -> MLXArray {
        let L = t5.dim(1), N = x.dim(1)
        var img = xEmbed(x)
        var txt = ctxEmbed(t5)
        let ts = MLXArray([sigma * 1000]).asType(.bfloat16)
        var temb = tl2(silu(tl1(Self.timeProj(ts))))
        if let gl1, let gl2 {
            temb = temb + gl2(silu(gl1(Self.timeProj(MLXArray([guidance * 1000]).asType(.bfloat16)))))
        }
        temb = (temb + xl2(silu(xl1(pooled)))).asType(.bfloat16)
        let st = silu(temb)

        // rope: text at (0, 0, 0), image at (0, y, x)
        var fr = [Float](repeating: 0, count: L), hi = [Float](repeating: 0, count: L), wi = [Float](repeating: 0, count: L)
        for y in 0 ..< h { for xx in 0 ..< w { fr.append(0); hi.append(Float(y)); wi.append(Float(xx)) } }
        func ang(_ p: [Float], _ d: Int) -> MLXArray {
            let omega = 1.0 / pow(MLXArray(Float(10000)), MLXArray(stride(from: 0, to: d, by: 2)).asType(.float32) / Float(d))
            return outer(MLXArray(p), omega)
        }
        let a = concatenated([ang(fr, 16), ang(hi, 56), ang(wi, 56)], axis: -1)
        let (rc, rs) = (cos(a), sin(a))
        let scale: Float = 1 / Float(128).squareRoot()

        func heads(_ t: MLXArray) -> MLXArray { t.reshaped([1, t.dim(1), 24, 128]).transposed(0, 2, 1, 3) }
        for b in doubles {
            let m = split(b.mod(st), parts: 6, axis: -1), mc = split(b.modC(st), parts: 6, axis: -1)
            let ni = Self.ln(img) * (1 + m[1].expandedDimensions(axis: 1)) + m[0].expandedDimensions(axis: 1)
            let nt = Self.ln(txt) * (1 + mc[1].expandedDimensions(axis: 1)) + mc[0].expandedDimensions(axis: 1)
            let q = concatenated([Self.headNorm(heads(b.aq(nt)), b.naq), Self.headNorm(heads(b.q(ni)), b.nq)], axis: 2)
            let k = concatenated([Self.headNorm(heads(b.ak(nt)), b.nak), Self.headNorm(heads(b.k(ni)), b.nk)], axis: 2)
            let v = concatenated([heads(b.av(nt)), heads(b.v(ni))], axis: 2)
            let o = MLXFast.scaledDotProductAttention(
                queries: Self.rope(q, rc, rs), keys: Self.rope(k, rc, rs), values: v.asType(.float32), scale: scale, mask: .none
            ).transposed(0, 2, 1, 3).reshaped([1, L + N, 3072])
            img = img + m[2].expandedDimensions(axis: 1) * b.o(o[0..., L...])
            txt = txt + mc[2].expandedDimensions(axis: 1) * b.ao(o[0..., ..<L])
            let ni2 = Self.ln(img) * (1 + m[4].expandedDimensions(axis: 1)) + m[3].expandedDimensions(axis: 1)
            img = img + m[5].expandedDimensions(axis: 1) * b.ff2(gelu(b.ff1(ni2)))
            let nt2 = Self.ln(txt) * (1 + mc[4].expandedDimensions(axis: 1)) + mc[3].expandedDimensions(axis: 1)
            txt = txt + mc[5].expandedDimensions(axis: 1) * b.cf2(geluApproximate(b.cf1(nt2)))
        }
        var hs = concatenated([txt, img], axis: 1)
        for b in singles {
            let m = split(b.mod(st), parts: 3, axis: -1)
            let n = Self.ln(hs) * (1 + m[1].expandedDimensions(axis: 1)) + m[0].expandedDimensions(axis: 1)
            let q = Self.rope(Self.headNorm(heads(b.q(n)), b.nq), rc, rs)
            let k = Self.rope(Self.headNorm(heads(b.k(n)), b.nk), rc, rs)
            let o = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: heads(b.v(n)).asType(.float32), scale: scale, mask: .none
            ).transposed(0, 2, 1, 3).reshaped([1, L + N, 3072])
            // mflux keeps the attention output in fp32 here; the MLP half is promoted to it.
            let cat = concatenated([o, geluApproximate(b.mlp(n)).asType(o.dtype)], axis: -1)
            hs = hs + m[2].expandedDimensions(axis: 1) * b.out(cat)
        }
        let so = split(normOut(st), parts: 2, axis: -1)
        let out = Self.ln(hs[0..., L...]) * (1 + so[0].expandedDimensions(axis: 1)) + so[1].expandedDimensions(axis: 1)
        return projOut(out)
    }
}

// MARK: - Family

final class FluxFamily: ImageFamily {
    let version: String
    let latentChannels = 16
    let vaeScale = 8
    let align = 16
    let flowKind = FlowKind.flux
    let defaultShift: Float
    /// sd.cpp runs FLUX on FLUX's own resolution-dependent schedule.
    let defaultScheduler = "flux"
    var previewProj: [[Float]] { fluxLatentRGB }
    var previewBias: [Float] { fluxLatentRGBBias }

    let clipTok: CLIPTokenizer
    /// T5's ids for a prompt, </s> included: tokenizer.json through
    /// swift-transformers, or SentencePiece's own file in early mflux saves.
    let t5Ids: (String) -> [Int]
    let clip: CLIPTextEncoder, t5: T5Encoder
    let transformer: FluxTransformer
    let vae: FluxVAE
    let t5Length: Int
    /// Distilled guidance for dev, set per generation.
    var guidance: Float = 3.5
    private var grid = (h: 64, w: 64)

    func configure(_ cmd: [String: Any]) {
        guidance = (cmd["guidance"] as? NSNumber)?.floatValue ?? 3.5
    }

    init(dir: URL, load: LoadReporter) async throws {
        (clipTok, t5Ids) = try await Self.tokenizers(dir)
        clip = try CLIPTextEncoder(load.store(dir, "text_encoder"))
        t5 = try T5Encoder(load.store(dir, "text_encoder_2"))
        transformer = try FluxTransformer(load.store(dir, "transformer"))
        vae = try FluxVAE(load.store(dir, "vae"))
        let dev = transformer.hasGuidance
        t5Length = dev ? 512 : 256
        // sd.cpp: FLUX takes a 1.0 shift, dev (guidance-distilled) 1.15.
        defaultShift = dev ? 1.15 : 1.0
        let name = dir.lastPathComponent.lowercased()
        version = dev ? (name.contains("krea") ? "FLUX.1 Krea [dev]" : "FLUX.1 [dev]") : "FLUX.1 [schnell]"
    }

    /// The two tokenizers of a FLUX.1 folder.
    static func tokenizers(_ dir: URL) async throws -> (CLIPTokenizer, (String) -> [Int]) {
        let clip = try CLIPTokenizer(folder: dir.appendingPathComponent("tokenizer"))
        let t5dir = dir.appendingPathComponent("tokenizer_2")
        if FileManager.default.fileExists(atPath: t5dir.appendingPathComponent("tokenizer.json").path) {
            let t = try await AutoTokenizer.from(modelFolder: t5dir)
            return (clip, { t.encode(text: $0) })
        }
        let sp = try SentencePieceTokenizer(file: t5dir.appendingPathComponent("spiece.model"))
        return (clip, { sp.encode($0) })
    }

    private static func fit(_ ids: [Int], _ n: Int, eos: Int, pad: Int) -> [Int] {
        var v = ids
        if v.count > n { v = Array(v[0 ..< (n - 1)]) + [eos] }
        while v.count < n { v.append(pad) }
        return v
    }

    /// The prompt's encoding: T5's states and CLIP's pooled vector, packed
    /// into one array [1 + L, 4096] (the pooled vector in the first row's
    /// first 768 columns) so the server can cache and pass it as one.
    func tokens(_ prompt: String) -> (clip: [Int], t5: [Int]) {
        (Self.fit(clipTok.encode(prompt), 77, eos: clipTok.eos, pad: clipTok.eos),
         Self.fit(t5Ids(prompt), t5Length, eos: 1, pad: 0))
    }

    func encode(_ prompt: String) throws -> MLXArray {
        let (clipIds, t5Ids) = tokens(prompt)
        let pooled = clip(clipIds).asType(.bfloat16)  // [1, 768]
        let states = t5(t5Ids)[0].asType(.bfloat16)  // [L, 4096]
        let head = concatenated([pooled, zeros([1, 4096 - 768], dtype: .bfloat16)], axis: -1)
        return concatenated([head, states], axis: 0)
    }

    func velocity(_ x: MLXArray, sigma: Float, cond: MLXArray) -> MLXArray {
        let pooled = cond[0 ..< 1, ..<768]
        let states = cond[1...].expandedDimensions(axis: 0)
        return transformer(x.asType(.bfloat16), sigma: sigma, t5: states, pooled: pooled, guidance: guidance, h: grid.h, w: grid.w)
    }

    func latentShape(width: Int, height: Int) -> [Int] {
        grid = (height / 16, width / 16)
        return [1, grid.h * grid.w, 64]
    }

    func hwc(_ latent: MLXArray) -> MLXArray {
        latent.reshaped([grid.h, grid.w, 16, 2, 2]).transposed(0, 3, 1, 4, 2).reshaped([grid.h * 2, grid.w * 2, 16])
    }

    func decodeHWC(_ latent: MLXArray) -> MLXArray { vae.decode(latent.asType(.bfloat16)) }

    func encodeImage(_ image: MLXArray) -> MLXArray? {
        guard let lat = vae.encode(image.asType(.bfloat16)) else { return nil }  // [16, 1, h, w]
        let hw = lat[0..., 0].transposed(1, 2, 0)
        let (h, w) = (hw.dim(0) / 2, hw.dim(1) / 2)
        return hw.reshaped([h, 2, w, 2, 16]).transposed(0, 2, 4, 1, 3).reshaped([1, h * w, 64])
    }
}
