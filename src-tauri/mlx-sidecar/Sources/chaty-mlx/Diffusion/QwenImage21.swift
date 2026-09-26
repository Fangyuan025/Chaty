// Qwen-Image 2.1 on MLX, from mflux's saves (mflux models/qwen21 is the
// reference; shapes and casts follow it).
//
//   • text: Qwen3-VL-8B's language stack, the prompt in 2.1's own template
//     (a fixed system line), read from the normed last layer with the system
//     line's tokens dropped;
//   • transformer: a 7B single-stream DiT over [text | image] tokens. One
//     modulation serves every block, and the text rows read it at t = 0
//     ("causal condition") while the image rows read the sampled t; text
//     attends causally to text, the image to everything;
//   • VAE: 64 latent channels, 16× down, a Wan-style network whose "3-D"
//     convolutions reduce to 2-D for a still picture; four output channels
//     (RGBA), the fourth dropped.
import Foundation
import MLX
import MLXNN
import Tokenizers

// sd.cpp's qwen21_latent_rgb_proj / _bias (runtime/latent-preview.h).
let qwen21LatentRGB: [[Float]] = [
    [0.00860495522, 0.01219501462, -0.00321337196],
    [0.0188923309, 0.01246581216, 0.01074959482],
    [0.1255941446, 0.1176879344, -0.0332212352],
    [0.0418238528, 0.1043427754, 0.0121666316],
    [0.02025338, 0.01453670296, -0.000224336044],
    [-0.01896720702, -0.020609903, -0.0322728584],
    [0.00438984796, -0.01374969766, 0.02849196],
    [-0.0374495856, -0.0286777126, -0.069319281],
    [0.01511914734, 0.0242979386, 0.055387887],
    [-0.1138629518, -0.020391466, 0.001550520522],
    [-0.0233650696, -0.0417292018, -0.0362361182],
    [-0.0351603342, -0.0243595924, -0.00216261038],
    [0.01093355288, -0.0373466924, 0.0024131535],
    [0.01778704744, -0.00401984678, -0.0343259192],
    [0.0486059334, 0.0253144, 0.0672564966],
    [0.0309463558, 0.0277963166, 0.0520869622],
    [0.0374485008, 0.0551753676, 0.0225853902],
    [-0.0090809962, -0.004756176, 0.00636443612],
    [-0.0270455652, -0.0384966954, -0.00905908082],
    [-0.00553493756, 0.01484553684, -0.0211502468],
    [0.01319502562, 0.00948005666, 0.0483789212],
    [-0.00931847104, -0.00276452734, -0.01011985302],
    [0.0180478258, 0.01614954356, -0.020942469],
    [-0.0214530434, -0.00272961176, 0.0217887476],
    [-0.0636772304, -0.0208893548, 0.0479167742],
    [-0.0250321236, -0.0286715676, 0.0530110146],
    [-0.01853078078, 0.01647272818, -0.00207747588],
    [0.0023101082, 0.01228800748, 0.01303505006],
    [-0.01243671408, -0.0258638728, -0.0379116264],
    [0.0059893471, 0.0064256355, -0.01234514304],
    [-0.0296733996, -0.023469805, 0.00060018212],
    [-0.0322019498, -0.0529200462, -0.00344987414],
    [-0.00205026458, -0.00846599446, 0.00455971038],
    [-0.01082227064, 0.0315661948, -0.0677753362],
    [0.0645553474, 0.1109666998, 0.0674744864],
    [0.01036801108, -0.0048484121, -0.001529168474],
    [0.01264353566, 0.01548126338, -0.00966374324],
    [-0.0223892408, -0.00871751526, -0.00030642167],
    [0.0271322742, 0.03496524, -0.0089692858],
    [0.0512178672, 0.0173080034, 0.00804227746],
    [0.01210987192, 0.00758025926, -0.00281712586],
    [0.189727839, 0.1210261828, 0.062603892],
    [0.0208058822, 0.00547548182, 0.01262955638],
    [0.00813332858, 0.01015930914, 0.0130177129],
    [-0.000927236014, -0.00152540594, -0.00599213302],
    [0.01663314616, -0.00582789626, 0.0163958132],
    [-0.0252546342, -0.0604193732, -0.1606919922],
    [-0.091722686, -0.0409201224, -0.0959576198],
    [0.0282963112, -0.01387223872, -0.01648814464],
    [0.0552316818, 0.0967547788, 0.0413586632],
    [0.00922849292, 0.00451467542, -0.0529172378],
    [0.0558600768, 0.0122988308, -0.01445942422],
    [0.000210660902, -0.01295958782, -0.01804761764],
    [0.035813625, -0.047250597, -0.1156405142],
    [-0.0506390696, -0.0471914842, 0.0349791468],
    [-0.0480143168, 0.00628389868, -0.0545163826],
    [0.0315499582, 0.0564846606, -0.0430850488],
    [-0.0362330316, -0.01267788554, 0.0061024772],
    [0.0038627542, 0.00911055916, -0.00758526008],
    [-0.0447103298, -0.00835411408, 0.01545872328],
    [-0.015006738, 0.00270612302, -0.00784361356],
    [-0.0221755048, -0.0513344748, -0.0475317424],
    [-0.01036656294, -0.00422146068, -0.0213499052],
    [0.01788952706, 0.0119194419, 0.0397205238],
]
let qwen21LatentRGBBias: [Float] = [-0.043293118, -0.02695978, -0.11986706]

// MARK: - Transformer

final class QwenImage21Transformer {
    struct Block {
        let q: DLinear, k: DLinear, v: DLinear, o: DLinear
        let normQ: DRMSNorm, normK: DRMSNorm
        let proj: DLinear, out: DLinear, gate: DLinear
    }

    static let dim = 4096, heads = 32, headDim = 128, channels = 64
    let imgIn: DLinear
    let mod: DLinear
    let t1: DLinear, t2: DLinear
    let txtNorm: MLXArray, txtIn: DLinear, txtOut: DLinear
    let normOut: DLinear, projOut: DLinear
    let blocks: [Block]
    let timeFreqs: MLXArray

    init(_ w: WeightStore) throws {
        let dim = Self.dim
        imgIn = try DLinear(w, "img_in", inDim: Self.channels)
        mod = try DLinear(w, "modulation.layers.1", inDim: dim)
        t1 = try DLinear(w, "time_text_embed.timestep_embedder.linear_1", inDim: 256)
        t2 = try DLinear(w, "time_text_embed.timestep_embedder.linear_2", inDim: dim)
        txtNorm = try w.take("txt_in.text_norm.weight")
        txtIn = try DLinear(w, "txt_in.in_layer", inDim: 4096)
        txtOut = try DLinear(w, "txt_in.out_layer", inDim: dim)
        normOut = try DLinear(w, "norm_out.linear", inDim: dim)
        projOut = try DLinear(w, "proj_out", inDim: dim)
        var bs: [Block] = []
        var i = 0
        while w.has("transformer_blocks.\(i).attn.to_q.weight") {
            let p = "transformer_blocks.\(i)"
            bs.append(
                Block(
                    q: try DLinear(w, "\(p).attn.to_q", inDim: dim), k: try DLinear(w, "\(p).attn.to_k", inDim: dim),
                    v: try DLinear(w, "\(p).attn.to_v", inDim: dim), o: try DLinear(w, "\(p).attn.to_out.0", inDim: dim),
                    normQ: try DRMSNorm(w, "\(p).attn.norm_q", eps: 1e-6),
                    normK: try DRMSNorm(w, "\(p).attn.norm_k", eps: 1e-6),
                    proj: try DLinear(w, "\(p).img_mlp.proj", inDim: dim),
                    out: try DLinear(w, "\(p).img_mlp.out", inDim: dim * 3),
                    gate: try DLinear(w, "\(p).img_mlp.gate_layer", inDim: dim)))
            i += 1
        }
        blocks = bs
        let half = 128
        timeFreqs = exp(-log(Float(10000)) * MLXArray(0 ..< half).asType(.float32) / Float(half))
    }

    /// cos/sin for the joint sequence: text positions advance on all three
    /// axes; the image sits at the text's end on the frame axis and on a grid
    /// centred on zero for height and width. Pairs (2k, 2k+1) share angle k.
    private func rope(textLen L: Int, h: Int, w: Int) -> (MLXArray, MLXArray) {
        var frame = [Float](), hi = [Float](), wi = [Float]()
        for i in 0 ..< L { frame.append(Float(i)); hi.append(Float(i)); wi.append(Float(i)) }
        for y in -(h - h / 2) ..< (h / 2) {
            for x in -(w - w / 2) ..< (w / 2) {
                frame.append(Float(L)); hi.append(Float(y)); wi.append(Float(x))
            }
        }
        func angles(_ pos: [Float], _ d: Int) -> MLXArray {
            let omega = 1.0 / pow(MLXArray(Float(10000)), MLXArray(stride(from: 0, to: d, by: 2)).asType(.float32) / Float(d))
            return outer(MLXArray(pos), omega)
        }
        let a = concatenated([angles(frame, 16), angles(hi, 56), angles(wi, 56)], axis: -1)  // [S, 64]
        return (cos(a), sin(a))
    }

    private func applyRope(_ x: MLXArray, _ c: MLXArray, _ s: MLXArray) -> MLXArray {
        let (b, l, h, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let xf = x.asType(.float32).reshaped([b, l, h, d / 2, 2])
        let re = xf[.ellipsis, 0], im = xf[.ellipsis, 1]
        let cc = c.reshaped([1, l, 1, d / 2]), ss = s.reshaped([1, l, 1, d / 2])
        return stacked([re * cc - im * ss, re * ss + im * cc], axis: -1).reshaped([b, l, h, d]).asType(x.dtype)
    }

    private static func rows(_ p: MLXArray, text L: Int, image N: Int) -> MLXArray {
        let d = p.dim(-1)
        return concatenated(
            [broadcast(p[1].reshaped([1, 1, d]), to: [1, L, d]), broadcast(p[0].reshaped([1, 1, d]), to: [1, N, d])],
            axis: 1)
    }

    private static func layerNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
    }

    /// The velocity for `x` ([1, N, 64] tokens of an h×w grid) at `sigma`,
    /// given the prompt's hidden states `cond` ([L, 4096]).
    func callAsFunction(_ x: MLXArray, sigma: Float, cond: MLXArray, h: Int, w: Int) -> MLXArray {
        let dim = Self.dim, heads = Self.heads, hd = Self.headDim
        let L = cond.dim(0), N = x.dim(1)
        let tRows = MLXArray([sigma, 0] as [Float])
        let args = tRows.reshaped([2, 1]) * 1000 * timeFreqs.reshaped([1, -1])
        let tProj = concatenated([cos(args), sin(args)], axis: -1).asType(.bfloat16)
        let temb = t2(silu(t1(tProj)))  // [2, dim]
        let modulation = mod(silu(temb))
        let m = split(modulation, parts: 2, axis: -1)
        let mod1 = Self.rows(m[0], text: L, image: N), mod2 = Self.rows(m[1], text: L, image: N)
        let m1 = split(mod1, parts: 2, axis: -1), m2 = split(mod2, parts: 2, axis: -1)
        let scale1 = 1 + m1[0], gate1 = tanh(m1[1]), scale2 = 1 + m2[0], gate2 = tanh(m2[1])

        // text: zero-centred RMSNorm → in → gelu(tanh) → out
        let cf = cond.asType(.float32).expandedDimensions(axis: 0)
        let rrms = rsqrt(mean(cf * cf, axis: -1, keepDims: true) + 1e-6)
        let tn = (cf * rrms * (txtNorm.asType(.float32) + 1)).asType(cond.dtype)
        let txt = txtOut(geluApproximate(txtIn(tn)))
        var hs = concatenated([txt, imgIn(x)], axis: 1)

        let (c, s) = rope(textLen: L, h: h, w: w)
        let scale = 1 / Float(hd).squareRoot()
        for b in blocks {
            let a = Self.layerNorm(hs) * scale1
            var q = b.q(a).reshaped([1, L + N, heads, hd])
            var k = b.k(a).reshaped([1, L + N, heads, hd])
            let v = b.v(a).reshaped([1, L + N, heads, hd])
            q = applyRope(b.normQ(q), c, s).transposed(0, 2, 1, 3)
            k = applyRope(b.normK(k), c, s).transposed(0, 2, 1, 3)
            let vt = v.transposed(0, 2, 1, 3)
            let textOut = MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., ..<L], keys: k[0..., 0..., ..<L], values: vt[0..., 0..., ..<L], scale: scale,
                mask: .causal)
            let imageOut = MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., L...], keys: k, values: vt, scale: scale, mask: .none)
            let att = concatenated([textOut, imageOut], axis: 2).transposed(0, 2, 1, 3).reshaped([1, L + N, dim])
            hs = hs + gate1 * b.o(att)
            let y = Self.layerNorm(hs) * scale2
            hs = hs + gate2 * b.out(silu(b.gate(y)) * b.proj(y))
        }
        let sc = Self.rows(normOut(silu(temb)), text: L, image: N)
        hs = projOut(Self.layerNorm(hs) * (1 + sc))
        return hs[0..., L...]
    }
}

// MARK: - VAE

final class QwenImage21VAE {
    /// Wan's norm: L2 over channels, times √C and a learned gain.
    struct Norm {
        let w: MLXArray
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let xf = x.asType(.float32)
            let n = sqrt((xf * xf).sum(axis: -1, keepDims: true))
            let c = Float(x.dim(-1))
            let y: MLXArray = xf / maximum(n, MLXArray(Float(1e-12))) * c.squareRoot()
            return (y * w.asType(.float32)).asType(x.dtype)
        }
    }

    struct Res {
        let n1: Norm, c1: DConv2d, n2: Norm, c2: DConv2d, sc: DConv2d?
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let r = sc?(x) ?? x
            var h = c1(silu(n1(x)))
            h = c2(silu(n2(h)))
            return h + r
        }
    }

    struct Attn {
        let n: Norm, qkv: DConv2d, proj: DConv2d
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let t = qkv(n(x)).reshaped([b, h * w, 3, c])
            let q = t[0..., 0..., 0], k = t[0..., 0..., 1], v = t[0..., 0..., 2]
            let scores = matmul(q, k.transposed(0, 2, 1)) * (1 / Float(c).squareRoot())
            let o = matmul(softmax(scores, axis: -1), v).reshaped([b, h, w, c])
            return proj(o) + x
        }
    }

    static let latentMean: [Float] = [
        0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
        1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
        2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
        -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.107, 1.7376,
        -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
        -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
        0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1566, 0.0304,
    ]
    static let latentStd: [Float] = [
        3.2001, 3.2936, 3.4321, 3.0091, 3.106, 4.0379, 4.0705, 3.791,
        3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
        3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
        3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
        3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
        3.532, 4.7833, 3.1284, 4.181, 3.8527, 3.8317, 3.5603, 4.3867,
        3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.44, 3.4957,
        3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
    ]

    let postQuant: DConv2d, quant: DConv2d?
    let dIn: DConv2d, dMid: (Res, Attn, Res)
    /// Up blocks: resnets, the upsampling conv, the input/output widths and
    /// the temporal factor of the parameter-free shortcut.
    let dUp: [(res: [Res], up: DConv2d?, inC: Int, outC: Int, ft: Int)]
    let dNorm: Norm, dOut: DConv2d
    let enc: (convIn: DConv2d, down: [(res: [Res], down: DConv2d?, inC: Int, outC: Int, ft: Int, fs: Int)],
              mid: (Res, Attn, Res), norm: Norm, convOut: DConv2d)?

    private static func res(_ w: WeightStore, _ p: String) throws -> Res {
        Res(
            n1: Norm(w: try w.take("\(p).norm1.weight")), c1: try DConv2d(w, "\(p).conv1.conv"),
            n2: Norm(w: try w.take("\(p).norm2.weight")), c2: try DConv2d(w, "\(p).conv2.conv"),
            sc: w.has("\(p).conv_shortcut.conv.weight") ? try DConv2d(w, "\(p).conv_shortcut.conv", padding: 0) : nil)
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
        postQuant = try DConv2d(w, "post_quant_conv.conv", padding: 0)
        quant = w.has("quant_conv.conv.weight") ? try DConv2d(w, "quant_conv.conv", padding: 0) : nil
        dIn = try DConv2d(w, "decoder.conv_in.conv")
        dMid = try Self.mid(w, "decoder.mid_block")
        let dims = [1152, 1152, 1152, 576, 288, 144]
        let tUp = [true, true, true, false]
        var up: [(res: [Res], up: DConv2d?, inC: Int, outC: Int, ft: Int)] = []
        for i in 0 ..< 5 {
            let p = "decoder.up_blocks.\(i)"
            let rs = try (0 ..< 3).map { try Self.res(w, "\(p).resnets.\($0)") }
            let u = i < 4 ? try DConv2d(w, "\(p).upsampler.conv") : nil
            up.append((rs, u, dims[i], dims[i + 1], i < 4 && tUp[i] ? 2 : 1))
        }
        dUp = up
        dNorm = Norm(w: try w.take("decoder.norm_out.weight"))
        dOut = try DConv2d(w, "decoder.conv_out.conv")

        if w.has("encoder.conv_in.conv.weight") {
            let ed = [96, 96, 192, 384, 768, 768]
            let tDown = [false, true, true, true]
            var down: [(res: [Res], down: DConv2d?, inC: Int, outC: Int, ft: Int, fs: Int)] = []
            for i in 0 ..< 5 {
                let p = "encoder.down_blocks.\(i)"
                let rs = try (0 ..< 2).map { try Self.res(w, "\(p).resnets.\($0)") }
                let d = i < 4 ? try DConv2d(w, "\(p).downsampler.conv", stride: 2, padding: 0) : nil
                down.append((rs, d, ed[i], ed[i + 1], i < 4 && tDown[i] ? 2 : 1, i < 4 ? 2 : 1))
            }
            enc = (
                try DConv2d(w, "encoder.conv_in.conv"), down, try Self.mid(w, "encoder.mid_block"),
                Norm(w: try w.take("encoder.norm_out.weight")), try DConv2d(w, "encoder.conv_out.conv"))
        } else {
            enc = nil
        }
    }

    /// The up blocks' shortcut: channels repeated, then laid out as a 2×
    /// nearest upsampling (the last temporal slot of the reference's layout).
    private static func dupUp(_ x: MLXArray, inC: Int, outC: Int, ft: Int) -> MLXArray {
        let (b, h, w) = (x.dim(0), x.dim(1), x.dim(2))
        let r = outC * ft * 4 / inC
        var y = repeated(x, count: r, axis: -1).reshaped([b, h, w, outC, ft, 2, 2])
        y = y[.ellipsis, ft - 1, 0..., 0...]  // [b, h, w, out, 2, 2]
        return y.transposed(0, 1, 4, 2, 5, 3).reshaped([b, h * 2, w * 2, outC])
    }

    /// The down blocks' shortcut: average over the (temporal, spatial) factor,
    /// then a mean over channel groups.
    private static func avgDown(_ x: MLXArray, inC: Int, outC: Int, ft: Int, fs: Int) -> MLXArray {
        let (b, h, w) = (x.dim(0), x.dim(1), x.dim(2))
        var t = expandedDimensions(x, axis: -1)  // [b, h, w, c, 1]
        if ft == 2 { t = concatenated([zeros(like: t), t], axis: -1) }
        t = t.reshaped([b, h / fs, fs, w / fs, fs, inC, ft]).transposed(0, 1, 3, 5, 6, 2, 4)
        let g = inC * ft * fs * fs / outC
        return mean(t.reshaped([b, h / fs, w / fs, outC, g]), axis: -1)
    }

    /// Latent [h, w, 64] (diffusion space) → picture [H, W, 3] in [-1, 1].
    func decode(_ latent: MLXArray) -> MLXArray {
        let m = MLXArray(Self.latentMean), s = MLXArray(Self.latentStd)
        var x = (latent.asType(.float32) * s + m).expandedDimensions(axis: 0)
        x = dIn(postQuant(x))
        x = dMid.2(dMid.1(dMid.0(x)))
        for u in dUp {
            let xin = x
            for r in u.res { x = r(x) }
            if let up = u.up {
                x = up(repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2))
                x = x + Self.dupUp(xin, inC: u.inC, outC: u.outC, ft: u.ft)
            }
        }
        x = dOut(silu(dNorm(x)))
        return x[0, 0..., 0..., ..<3]
    }

    /// Picture [H, W, 3] in [-1, 1] → tokens [1, h·w, 64].
    func encode(_ image: MLXArray) -> MLXArray? {
        guard let enc, let quant else { return nil }
        var x = concatenated([image.asType(.float32), ones([image.dim(0), image.dim(1), 1])], axis: -1)
            .expandedDimensions(axis: 0)
        x = enc.convIn(x)
        for d in enc.down {
            let xin = x
            for r in d.res { x = r(x) }
            if let dn = d.down { x = dn(padded(x, widths: [0, [0, 1], [0, 1], 0])) }
            x = x + Self.avgDown(xin, inC: d.inC, outC: d.outC, ft: d.ft, fs: d.fs)
        }
        x = enc.mid.2(enc.mid.1(enc.mid.0(x)))
        x = quant(enc.convOut(silu(enc.norm(x))))
        let lat = (x[0..., 0..., 0..., ..<64] - MLXArray(Self.latentMean)) / MLXArray(Self.latentStd)
        return lat.reshaped([1, -1, 64])
    }
}

// MARK: - Family

final class QwenImage21Family: ImageFamily {
    let version = "Qwen Image 2.1"
    let latentChannels = 64
    let vaeScale = 16
    let align = 16
    let flowKind = FlowKind.discrete
    let defaultShift: Float = 3
    /// sd.cpp runs 2.1 on FLUX's resolution-dependent schedule.
    let defaultScheduler = "flux"
    var previewProj: [[Float]] { qwen21LatentRGB }
    var previewBias: [Float] { qwen21LatentRGBBias }

    static let systemPrefix = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    let tokenizer: Tokenizer
    let textEncoder: Qwen3TextEncoder
    let transformer: QwenImage21Transformer
    let vae: QwenImage21VAE
    let prefixLen: Int
    private var grid = (h: 64, w: 64)

    init(dir: URL, load: LoadReporter) async throws {
        let tokDir = FileManager.default.fileExists(atPath: dir.appendingPathComponent("processor/tokenizer.json").path)
            ? "processor" : "tokenizer"
        tokenizer = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent(tokDir))
        prefixLen = tokenizer.encode(text: Self.systemPrefix, addSpecialTokens: false).count
        textEncoder = try Qwen3TextEncoder(load.store(dir, "text_encoder"), .qwenImage21)
        transformer = try QwenImage21Transformer(load.store(dir, "transformer"))
        vae = try QwenImage21VAE(load.store(dir, "vae"))
    }

    func encode(_ prompt: String) throws -> MLXArray {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? " " : prompt
        let text = Self.systemPrefix + "<|im_start|>user\n\(p)<|im_end|>\n<|im_start|>assistant\n"
        var ids = tokenizer.encode(text: text, addSpecialTokens: true)
        if ids.count > 2048 { ids = Array(ids[0 ..< 2048]) }
        let h = textEncoder(MLXArray(ids.map { Int32($0) }).reshaped([1, -1]))
        return h[prefixLen...]
    }

    func velocity(_ x: MLXArray, sigma: Float, cond: MLXArray) -> MLXArray {
        transformer(x.asType(.bfloat16), sigma: sigma, cond: cond, h: grid.h, w: grid.w)
    }

    func latentShape(width: Int, height: Int) -> [Int] {
        grid = (height / 16, width / 16)
        return [1, (height / 16) * (width / 16), 64]
    }

    func hwc(_ latent: MLXArray) -> MLXArray { latent.reshaped([grid.h, grid.w, 64]) }
    func decodeHWC(_ latent: MLXArray) -> MLXArray { vae.decode(latent) }
    func encodeImage(_ image: MLXArray) -> MLXArray? { vae.encode(image) }
}
