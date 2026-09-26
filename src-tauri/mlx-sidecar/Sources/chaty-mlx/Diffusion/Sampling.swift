// Schedules and samplers, as stable-diffusion.cpp runs them.
//
// The image studio's settings — sampler, scheduler, steps, flow shift — mean
// what they mean for a GGUF on stable-diffusion.cpp, so the same settings on
// an MLX model do the same thing. This is a port of the parts of sd.cpp's
// `runtime/denoiser.hpp` that apply to the flow-matching models Chaty runs on
// MLX: the two flow denoisers (sigma ↔ timestep and the shift), the sigma
// schedulers, and the k-diffusion samplers, each kept as close to the C++ as
// the array library allows. Latents stay in fp32 through the sampler, as they
// do in sd.cpp.
import Foundation
import MLX

let TIMESTEPS: Float = 1000

enum FlowKind {
    /// Z-Image, Qwen-Image, SD3: the model reads t = σ·1000, σ(t) is a
    /// time-SNR shift of (t+1)/1000.
    case discrete
    /// FLUX: the model reads σ itself, σ(t) is FLUX's exponential shift.
    case flux
}

struct FlowDenoiser {
    let kind: FlowKind
    var shift: Float

    func tToSigma(_ t: Float) -> Float {
        let tt = (t + 1) / TIMESTEPS
        switch kind {
        case .discrete:
            if shift == 1 { return tt }
            return shift * tt / (1 + (shift - 1) * tt)
        case .flux:
            return fluxTimeShift(mu: shift, sigma: 1, t: tt)
        }
    }

    var sigmaMin: Float { tToSigma(0) }
    var sigmaMax: Float { tToSigma(TIMESTEPS - 1) }

    /// What the model is given as its timestep.
    func sigmaToT(_ sigma: Float) -> Float {
        kind == .discrete ? sigma * 1000 : sigma
    }
}

func fluxTimeShift(mu: Float, sigma: Float, t: Float) -> Float {
    exp(mu) / (exp(mu) + pow(1 / t - 1, sigma))
}

// MARK: - Schedulers

enum Schedule {
    /// The names the studio offers for a flow model on this engine.
    static let names = [
        "discrete", "karras", "exponential", "sgm_uniform", "simple", "smoothstep", "kl_optimal", "lcm",
        "bong_tangent", "beta",
    ]

    static func sigmas(_ name: String, steps n: Int, denoiser d: FlowDenoiser, imageSeqLen: Int) -> [Float] {
        let tMax = Int(TIMESTEPS) - 1
        let sMin = d.sigmaMin, sMax = d.sigmaMax
        switch name {
        case "karras":
            let rho: Float = 7
            let lo = pow(max(sMin, 1e-6), 1 / rho), hi = pow(sMax, 1 / rho)
            var r = (0 ..< n).map { i -> Float in
                pow(hi + Float(i) / (Float(n) - 1) * (lo - hi), rho)
            }
            r.append(0)
            return r
        case "exponential":
            let lmin = log(sMin), lmax = log(sMax)
            let step = (lmax - lmin) / Float(n - 1)
            var r = (0 ..< n).map { exp(lmax - step * Float($0)) }
            r.append(0)
            return r
        case "sgm_uniform":
            if n == 0 { return [0] }
            let ts = linearSpace(Float(tMax), 0, n + 1)
            var r = (0 ..< n).map { d.tToSigma(ts[$0]) }
            r.append(0)
            return r
        case "simple":
            let factor = TIMESTEPS / Float(n)
            var r = (0 ..< n).map { i -> Float in
                let idx = max(0, Int(TIMESTEPS) - 1 - Int(Float(i) * factor))
                return d.tToSigma(Float(idx))
            }
            r.append(0)
            return r
        case "smoothstep":
            if n == 1 { return [d.tToSigma(Float(tMax)), 0] }
            var r = (0 ..< n).map { i -> Float in
                let u = 1 - Float(i) / Float(n)
                return d.tToSigma((u * u * (3 - 2 * u) * Float(tMax)).rounded())
            }
            r.append(0)
            return r
        case "kl_optimal":
            if n == 1 { return [sMax, 0] }
            let amin = atan(max(sMin, 1e-6)), amax = atan(sMax)
            var r = (0 ..< n).map { i -> Float in
                let t = Float(i) / Float(n - 1)
                return tan(t * amin + (1 - t) * amax)
            }
            r.append(0)
            return r
        case "lcm":
            let original = 50, k = Int(TIMESTEPS) / original
            var r = (0 ..< n).map { i -> Float in
                let index = (i * original) / n
                return d.tToSigma(Float((original - index) * k - 1))
            }
            r.append(0)
            return r
        case "bong_tangent":
            return bongTangent(n, sMin, sMax)
        case "beta":
            return betaSchedule(n, d)
        case "flux":
            // FLUX's own: a shift that grows with the picture's size.
            let m = (1.15 - 0.5) / Float(4096 - 256)
            let b = Float(0.5) - m * 256
            let mu = Float(imageSeqLen) * m + b
            var r = (0 ... n).map { i -> Float in
                let t = 1 - Float(i) / Float(n)
                return t <= 0 ? 0 : fluxTimeShift(mu: mu, sigma: 1, t: t)
            }
            r[n] = 0
            return r
        case "flux2":
            let mu = flux2Mu(imageSeqLen, n)
            var r = (0 ... n).map { i -> Float in
                let t = 1 - Float(i) / Float(n)
                if t <= 0 { return 0 }
                if t >= 1 { return 1 }
                return fluxTimeShift(mu: mu, sigma: 1, t: t)
            }
            r[n] = 0
            return r
        default:  // "discrete"
            if n == 1 { return [d.tToSigma(Float(tMax)), 0] }
            let step = Float(tMax) / Float(n - 1)
            var r = (0 ..< n).map { d.tToSigma(Float(tMax) - step * Float($0)) }
            r.append(0)
            return r
        }
    }

    static func flux2Mu(_ seq: Int, _ steps: Int) -> Float {
        let a1: Float = 8.73809524e-05, b1: Float = 1.89833333
        let a2: Float = 0.00016927, b2: Float = 0.45666666
        if seq > 4300 { return a2 * Float(seq) + b2 }
        let m200 = a2 * Float(seq) + b2, m10 = a1 * Float(seq) + b1
        let a = (m200 - m10) / 190, b = m200 - 200 * a
        return a * Float(steps) + b
    }

    private static func linearSpace(_ start: Float, _ end: Float, _ n: Int) -> [Float] {
        guard n > 1 else { return [start] }
        let inc = (end - start) / Float(n - 1)
        var r = [start]
        for _ in 1 ..< n { r.append(r.last! + inc) }
        return r
    }

    private static func bongTangentPart(_ steps: Int, _ slope: Float, _ pivot: Float, _ start: Float, _ end: Float) -> [Float] {
        guard steps > 0 else { return [] }
        let pi = Float.pi
        let smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) * 0.5
        let smin = ((2 / pi) * atan(-slope * (Float(steps - 1) - pivot)) + 1) * 0.5
        let srange = smax - smin, sscale = start - end
        if abs(srange) < 1e-8 {
            if steps == 1 { return [start] }
            return (0 ..< steps).map { start + (end - start) * Float($0) / Float(steps - 1) }
        }
        return (0 ..< steps).map { x in
            let v = ((2 / pi) * atan(-slope * (Float(x) - pivot)) + 1) * 0.5
            return ((v - smin) / srange) * sscale + end
        }
    }

    private static func bongTangent(_ n: Int, _ sMin: Float, _ sMax: Float) -> [Float] {
        guard n > 0 else { return [] }
        let start = sMax, end = sMin, middle = sMin + (sMax - sMin) * 0.5
        let p1: Float = 0.6, p2: Float = 0.6
        var s1: Float = 0.2, s2: Float = 0.2
        let steps = n + 2
        let midpoint = Int((Float(steps) * p1 + Float(steps) * p2) * 0.5)
        let p1i = Int(Float(steps) * p1), p2i = Int(Float(steps) * p2)
        let slopeScale = Float(steps) / 40
        s1 /= slopeScale
        s2 /= slopeScale
        let len2 = steps - midpoint, len1 = steps - len2
        var a = bongTangentPart(len1, s1, Float(p1i), start, middle)
        let b = bongTangentPart(len2, s2, Float(p2i - len1), middle, end)
        if !a.isEmpty { a.removeLast() }
        var r = a + b
        while r.count < n + 1 { r.append(end) }
        if r.count > n + 1 { r = Array(r[0 ..< (n + 1)]) }
        r[n] = 0
        return r
    }

    private static func betaSchedule(_ n: Int, _ d: FlowDenoiser) -> [Float] {
        let tMax = Int(TIMESTEPS) - 1
        if n == 0 { return [] }
        if n == 1 { return [d.tToSigma(Float(tMax)), 0] }
        var r: [Float] = []
        var last = -1
        for i in 0 ..< n {
            let u = 1 - Double(i) / Double(n)
            let t = Int((betaPPF(u, 0.6, 0.6) * Double(tMax)).rounded())
            if t != last {
                r.append(d.tToSigma(Float(t)))
                last = t
            }
        }
        r.append(0)
        return r
    }

    private static func logBeta(_ a: Double, _ b: Double) -> Double { lgamma(a) + lgamma(b) - lgamma(a + b) }

    private static func incBeta(_ x: Double, _ a: Double, _ b: Double) -> Double {
        if x <= 0 { return 0 }
        if x >= 1 { return 1 }
        let tiny = 1e-30
        let qab = a + b, qap = a + 1, qam = a - 1
        var c = 1.0, dd = 1 - qab * x / qap
        if abs(dd) < tiny { dd = tiny }
        dd = 1 / dd
        var h = dd
        for m in 1 ... 200 {
            let m2 = Double(2 * m), md = Double(m)
            var aa = md * (b - md) * x / ((qam + m2) * (a + m2))
            dd = 1 + aa * dd
            if abs(dd) < tiny { dd = tiny }
            c = 1 + aa / c
            if abs(c) < tiny { c = tiny }
            dd = 1 / dd
            h *= dd * c
            aa = -(a + md) * (qab + md) * x / ((a + m2) * (qap + m2))
            dd = 1 + aa * dd
            if abs(dd) < tiny { dd = tiny }
            c = 1 + aa / c
            if abs(c) < tiny { c = tiny }
            dd = 1 / dd
            let del = dd * c
            h *= del
            if abs(del - 1) < 3e-7 { break }
        }
        return exp(a * log(x) + b * log(1 - x) - logBeta(a, b)) / a * h
    }

    private static func betaCDF(_ x: Double, _ a: Double, _ b: Double) -> Double {
        if x == 0 { return 0 }
        if x == 1 { return 1 }
        if x < (a + 1) / (a + b + 2) { return incBeta(x, a, b) }
        return 1 - incBeta(1 - x, b, a)
    }

    private static func betaPPF(_ u: Double, _ a: Double, _ b: Double) -> Double {
        var x = 0.5
        for _ in 0 ..< 30 {
            let f = betaCDF(x, a, b) - u
            if abs(f) < 1e-10 { break }
            let df = exp((a - 1) * log(x) + (b - 1) * log(1 - x) - logBeta(a, b))
            x -= f / df
            if x <= 0 { x = 1e-10 }
            if x >= 1 { x = 1 - 1e-10 }
        }
        return x
    }
}

// MARK: - Samplers

/// The model as a sampler sees it: the denoised estimate for `x` at `sigma`.
/// `step` is sd.cpp's step number — negative for the first of two model
/// calls within one step. nil = cancelled.
typealias DenoiseFn = (_ x: MLXArray, _ sigma: Float, _ step: Int) -> MLXArray?

/// Fresh Gaussian noise shaped like the latent, one draw per call.
final class NoiseSource {
    private var key: MLXArray
    let shape: [Int]
    init(seed: UInt64, shape: [Int]) {
        key = MLXRandom.key(seed)
        self.shape = shape
    }
    func next() -> MLXArray {
        let (a, b) = MLXRandom.split(key: key)
        key = a
        return MLXRandom.normal(shape, key: b)
    }
}

enum Sampler {
    static let names = [
        "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm",
        "ddim_trailing", "tcd", "res_multistep", "res_2s", "er_sde", "dpm++2m_sde", "lms",
    ]

    /// sd.cpp's default eta for each sampler.
    static func defaultEta(_ name: String) -> Float {
        switch name {
        case "euler_a", "dpm++2s_a", "er_sde", "dpm++2m_sde": return 1
        default: return 0
        }
    }

    static func ancestralStep(_ from: Float, _ to: Float, _ eta: Float) -> (Float, Float) {
        var up: Float = 0
        if eta <= 0 { return (to, 0) }
        let f2 = from * from, t2 = to * to
        if f2 > 0 { up = min(to, eta * max(t2 * (f2 - t2) / f2, 0).squareRoot()) }
        let d2 = t2 - up * up
        return (d2 > 0 ? d2.squareRoot() : 0, up)
    }

    static func ancestralStepFlow(_ from: Float, _ to: Float, _ etaIn: Float) -> (Float, Float, Float) {
        var down = to, up: Float = 0, alpha: Float = 1
        if etaIn <= 0 || from <= 0 || to <= 0 { return (down, up, alpha) }
        let eta = min(etaIn, 1)
        let ratio = to / from
        down = max(0, min(to, to * (1 + (ratio - 1) * eta)))
        let denom = 1 - down
        if denom <= 0 { return (to, up, alpha) }
        alpha = (1 - to) / denom
        let term = max(-1, min(1, (down / to) * alpha))
        up = to * max(1 - term * term, 0).squareRoot()
        return (down, up, alpha)
    }

    /// Run `name` over `sigmas` from `x`. nil = cancelled.
    static func run(
        _ name: String, model: DenoiseFn, x x0: MLXArray, sigmas: [Float], noise: NoiseSource, eta etaIn: Float?
    ) -> MLXArray? {
        let eta = etaIn ?? defaultEta(name)
        var x = x0
        let steps = sigmas.count - 1
        switch name {
        case "euler_a", "ddim_trailing":
            for i in 0 ..< steps {
                let s = sigmas[i], to = sigmas[i + 1]
                guard let den = model(x, s, i + 1) else { return nil }
                if to == 0 {
                    x = den
                } else if eta == 0 {
                    let r = to / s
                    x = r * x + (1 - r) * den
                } else {
                    let (down, up, alpha) = ancestralStepFlow(s, to, eta)
                    let r = down / s
                    x = r * x + (1 - r) * den
                    if up > 0 {
                        x = x * alpha + noise.next() * up
                    }
                }
            }
        case "heun":
            for i in 0 ..< steps {
                guard let den = model(x, sigmas[i], -(i + 1)) else { return nil }
                var d = (x - den) / sigmas[i]
                let dt = sigmas[i + 1] - sigmas[i]
                if sigmas[i + 1] == 0 {
                    x = x + d * dt
                } else {
                    let x2 = x + d * dt
                    guard let den2 = model(x2, sigmas[i + 1], i + 1) else { return nil }
                    d = (d + (x2 - den2) / sigmas[i + 1]) / 2
                    x = x + d * dt
                }
            }
        case "dpm2":
            for i in 0 ..< steps {
                guard let den = model(x, sigmas[i], -(i + 1)) else { return nil }
                let d = (x - den) / sigmas[i]
                if sigmas[i + 1] == 0 {
                    x = x + d * (sigmas[i + 1] - sigmas[i])
                } else {
                    let mid = exp(0.5 * (log(sigmas[i]) + log(sigmas[i + 1])))
                    let x2 = x + d * (mid - sigmas[i])
                    guard let den2 = model(x2, mid, i + 1) else { return nil }
                    x = x + ((x2 - den2) / mid) * (sigmas[i + 1] - sigmas[i])
                }
            }
        case "dpm++2s_a":
            for i in 0 ..< steps {
                let s = sigmas[i], to = sigmas[i + 1]
                let first = 1 - s < 1e-6
                guard let den = model(x, s, (first ? 1 : -1) * (i + 1)) else { return nil }
                if to == 0 {
                    x = den
                } else {
                    let (down, up, alpha) = ancestralStepFlow(s, to, eta)
                    var di = den
                    if !first {
                        let expS = (((1 - s) / s) * ((1 - down) / down)).squareRoot()
                        let sigmaS = 1 / (expS + 1)
                        let r = sigmaS / s
                        let u = x * r + den * (1 - r)
                        guard let d2 = model(u, sigmaS, i + 1) else { return nil }
                        di = d2
                    }
                    let r = down / s
                    x = x * r + di * (1 - r)
                    if to > 0 && eta > 0 { x = alpha * x + noise.next() * up }
                }
            }
        case "dpm++2m", "dpm++2mv2":
            var old = x
            let tFn = { (s: Float) -> Float in -log(s) }
            for i in 0 ..< steps {
                guard let den = model(x, sigmas[i], i + 1) else { return nil }
                let t = tFn(sigmas[i]), tn = tFn(sigmas[i + 1])
                let h = tn - t
                let a = sigmas[i + 1] / sigmas[i]
                if i == 0 || sigmas[i + 1] == 0 {
                    x = a * x - (exp(-h) - 1) * den
                } else if name == "dpm++2m" {
                    let r = (t - tFn(sigmas[i - 1])) / h
                    let dd = (1 + 1 / (2 * r)) * den - (1 / (2 * r)) * old
                    x = a * x - (exp(-h) - 1) * dd
                } else {
                    let hLast = t - tFn(sigmas[i - 1])
                    let hMin = min(hLast, h), hMax = max(hLast, h)
                    let r = hMax / hMin
                    let hd = (hMax + hMin) / 2
                    let dd = (1 + 1 / (2 * r)) * den - (1 / (2 * r)) * old
                    x = a * x - (exp(-hd) - 1) * dd
                }
                old = den
            }
        case "dpm++2m_sde":
            var old: MLXArray? = nil
            var hLast: Float = 0
            for i in 0 ..< steps {
                guard let den = model(x, sigmas[i], i + 1) else { return nil }
                if sigmas[i + 1] == 0 {
                    x = den
                } else {
                    let t = -log(sigmas[i]), s = -log(sigmas[i + 1])
                    let h = s - t, etaH = eta * h
                    let a = sigmas[i + 1] / sigmas[i] * exp(-etaH)
                    let b = -expm1(-h - etaH)
                    x = a * x + b * den
                    if let old {
                        let r = hLast / h
                        x = x + (0.5 * b / r) * (den - old)
                    }
                    if eta > 0 {
                        x = x + noise.next() * (sigmas[i + 1] * (-expm1(-2 * etaH)).squareRoot())
                    }
                    hLast = h
                }
                old = den
            }
        case "lcm":
            for i in 0 ..< steps {
                guard let den = model(x, sigmas[i], i + 1) else { return nil }
                x = den
                if sigmas[i + 1] > 0 {
                    x = x * (1 - sigmas[i + 1])
                    x = x + noise.next() * sigmas[i + 1]
                }
            }
        case "ipndm", "ipndm_v":
            var hist: [MLXArray] = []
            for i in 0 ..< steps {
                let s = sigmas[i], sn = sigmas[i + 1]
                guard let den = model(x, s, i + 1) else { return nil }
                let d = (x - den) / s
                let order = min(4, i + 1)
                let hn = sn - s
                switch order {
                case 1:
                    x = x + d * hn
                case 2:
                    let h1: MLXArray = hist[hist.count - 1]
                    if name == "ipndm" {
                        let acc: MLXArray = d * Float(3) - h1
                        x = x + acc * (hn / 2)
                    } else {
                        let hn1: Float = i > 0 ? s - sigmas[i - 1] : hn
                        let r: Float = hn / hn1
                        let acc: MLXArray = d * (2 + r) - h1 * r
                        x = x + acc * (hn / 2)
                    }
                case 3:
                    // Spelled out term by term: older compilers give up on
                    // the one-line form.
                    let h1 = hist[hist.count - 1], h2 = hist[hist.count - 2]
                    var acc: MLXArray = d * Float(23)
                    acc = acc - h1 * Float(16)
                    acc = acc + h2 * Float(5)
                    x = x + acc * (hn / 12)
                default:
                    let h1 = hist[hist.count - 1], h2 = hist[hist.count - 2], h3 = hist[hist.count - 3]
                    var acc: MLXArray = d * Float(55)
                    acc = acc - h1 * Float(59)
                    acc = acc + h2 * Float(37)
                    acc = acc - h3 * Float(9)
                    x = x + acc * (hn / 24)
                }
                if hist.count == 3 { hist.removeFirst() }
                hist.append(d)
            }
        case "res_multistep":
            var old = x
            var haveOld = false
            var oldDown: Float = 0
            func phi1(_ t: Float) -> Float { abs(t) < 1e-6 ? 1 + t * 0.5 + t * t / 6 : (exp(t) - 1) / t }
            func phi2(_ t: Float) -> Float { abs(t) < 1e-6 ? 0.5 + t / 6 + t * t / 24 : (phi1(t) - 1) / t }
            for i in 0 ..< steps {
                guard let den = model(x, sigmas[i], i + 1) else { return nil }
                let from = sigmas[i], to = sigmas[i + 1]
                let (down, up, alpha) = ancestralStepFlow(from, to, eta)
                if down == 0 || !haveOld {
                    x = x + ((x - den) / from) * (down - from)
                } else {
                    let t = -log(from), tOld = -log(oldDown), tNext = -log(down), tPrev = -log(sigmas[i - 1])
                    let h = tNext - t
                    let c2 = (tPrev - tOld) / h
                    var b1 = phi1(-h) - phi2(-h) / c2
                    var b2 = phi2(-h) / c2
                    if !b1.isFinite { b1 = 0 }
                    if !b2.isFinite { b2 = 0 }
                    x = exp(-h) * x + h * (b1 * den + b2 * old)
                }
                if to > 0 && up > 0 {
                    x = x * alpha + noise.next() * up
                }
                old = den
                oldDown = down
                haveOld = true
            }
        case "res_2s":
            let c2: Float = 0.5
            func phi1(_ t: Float) -> Float { abs(t) < 1e-6 ? 1 + t * 0.5 + t * t / 6 : (exp(t) - 1) / t }
            func phi2(_ t: Float) -> Float { abs(t) < 1e-6 ? 0.5 + t / 6 + t * t / 24 : (phi1(t) - 1) / t }
            for i in 0 ..< steps {
                let from = sigmas[i], to = sigmas[i + 1]
                guard let den = model(x, from, -(i + 1)) else { return nil }
                let (down, up, alpha) = ancestralStepFlow(from, to, eta)
                let xs = x
                if down == 0 || from == 0 {
                    x = den
                } else {
                    let t = -log(from), tn = -log(down), h = tn - t
                    let a21 = c2 * phi1(-h * c2)
                    let b2 = phi2(-h) / c2, b1 = phi1(-h) - b2
                    let sigmaC2 = exp(-(t + h * c2))
                    let eps1 = den - xs
                    let x2 = xs + eps1 * (h * a21)
                    guard let den2 = model(x2, sigmaC2, i + 1) else { return nil }
                    let eps2 = den2 - xs
                    x = xs + h * (b1 * eps1 + b2 * eps2)
                }
                if to > 0 && up > 0 {
                    x = x * alpha + noise.next() * up
                }
            }
        case "er_sde":
            return erSDE(model: model, x: x, sigmasIn: sigmas, noise: noise, sNoise: eta)
        case "tcd":
            return tcd(model: model, x: x, sigmas: sigmas, noise: noise, eta: eta)
        case "lms":
            return lms(model: model, x: x, sigmas: sigmas)
        default:  // euler
            for i in 0 ..< steps {
                let s = sigmas[i]
                guard let den = model(x, s, i + 1) else { return nil }
                let d = (x - den) / s
                x = x + d * (sigmas[i + 1] - s)
            }
        }
        return x
    }

    private static func erSDE(model: DenoiseFn, x x0: MLXArray, sigmasIn: [Float], noise: NoiseSource, sNoise: Float) -> MLXArray? {
        func flowSigma(_ s: Float) -> Float { min(max(s, 1e-6), 1 - 1e-4) }
        func lambda(_ s: Float) -> Float { let f = flowSigma(s); return f / max(1 - f, 1e-6) }
        func alpha(_ s: Float) -> Float { 1 - flowSigma(s) }
        func scaler(_ x: Float) -> Float { let y = max(x, 0); return y * (exp(pow(y, 0.3)) + 10) }
        var sigmas = sigmasIn
        for i in 0 ..< max(0, sigmas.count - 1) where sigmas[i] > 1 { sigmas[i] = flowSigma(sigmas[i]) }
        let lambdas = sigmas.map(lambda)
        var x = x0
        var old = x, oldD = x
        var haveOld = false, haveOldD = false
        let steps = sigmas.count - 1
        for i in 0 ..< steps {
            guard let den = model(x, sigmas[i], i + 1) else { return nil }
            let stage = min(3, i + 1)
            if sigmas[i + 1] == 0 {
                x = den
            } else {
                let ls = lambdas[i], lt = lambdas[i + 1]
                let aS = alpha(sigmas[i]), aT = alpha(sigmas[i + 1])
                let scS = scaler(ls), scT = scaler(lt)
                let rAlpha = aS > 0 ? aT / aS : 0
                let r = scS > 0 ? scT / scS : 0
                x = rAlpha * r * x + aT * (1 - r) * den
                if stage >= 2 && haveOld {
                    let dt = lt - ls
                    let stepSize = -dt / 200
                    var s: Float = 0, su: Float = 0
                    for p in 0 ..< 200 {
                        let pos = lt + Float(p) * stepSize
                        let sp = scaler(pos)
                        if sp <= 0 { continue }
                        s += 1 / sp
                        if stage >= 3 && haveOldD { su += (pos - ls) / sp }
                    }
                    s *= stepSize
                    let denomD = ls - lambdas[i - 1]
                    if abs(denomD) > 1e-12 {
                        let dd = (den - old) / denomD
                        x = x + aT * (dt + s * scT) * dd
                        if stage >= 3 && haveOldD {
                            let denomU = (ls - lambdas[i - 2]) * 0.5
                            if abs(denomU) > 1e-12 {
                                su *= stepSize
                                let du = (dd - oldD) / denomU
                                x = x + aT * (0.5 * dt * dt + su * scT) * du
                            }
                        }
                        oldD = dd
                        haveOldD = true
                    }
                }
                let sq = lt * lt - ls * ls * r * r
                if sNoise > 0 && sq > 0 {
                    x = x + noise.next() * (aT * max(sq, 0).squareRoot())
                }
            }
            old = den
            haveOld = true
        }
        return x
    }

    private static func tcd(model: DenoiseFn, x x0: MLXArray, sigmas: [Float], noise: NoiseSource, eta: Float) -> MLXArray? {
        let bStart: Float = 0.00085, bEnd: Float = 0.0120
        var ac = [Double](repeating: 0, count: Int(TIMESTEPS))
        var cs = [Double](repeating: 0, count: Int(TIMESTEPS))
        for i in 0 ..< Int(TIMESTEPS) {
            let b = pow(bStart.squareRoot() + (bEnd.squareRoot() - bStart.squareRoot()) * Float(i) / (TIMESTEPS - 1), 2)
            ac[i] = (i == 0 ? 1 : ac[i - 1]) * Double(1 - b)
            cs[i] = ((1 - ac[i]) / ac[i]).squareRoot()
        }
        func tFromSigma(_ s: Float) -> Int {
            guard let hi = cs.firstIndex(where: { $0 >= Double(s) }) else { return Int(TIMESTEPS) - 1 }
            if hi == 0 { return 0 }
            return abs(cs[hi] - Double(s)) < abs(cs[hi - 1] - Double(s)) ? hi : hi - 1
        }
        var x = x0
        for i in 0 ..< sigmas.count - 1 {
            let to = sigmas[i + 1]
            let prevT = tFromSigma(to)
            let ts = Int(floor((1 - eta) * Float(prevT)))
            let s = sigmas[i]
            guard let den = model(x, s, i + 1) else { return nil }
            let d = (x - den) / s
            let aT = 1 / (s * s + 1)
            let aPrev = 1 / (to * to + 1)
            let aS = Float(ac[ts]), bS = 1 - aS
            _ = aT
            x = (aS / aPrev).squareRoot() * den + (bS / aPrev).squareRoot() * d
            if eta > 0 && to > 0 {
                x = (aPrev / aS).squareRoot() * x + (1 / aPrev - 1 / aS).squareRoot() * noise.next()
            }
        }
        return x
    }

    private static func lms(model: DenoiseFn, x x0: MLXArray, sigmas: [Float]) -> MLXArray? {
        let divisions = 1000, shift = 1
        let steps = sigmas.count - 1
        let maxOrder = min(4, steps)
        func coeff(_ order: Int, _ m: Int, _ j: Int) -> Float {
            let a = sigmas[m], dx = (sigmas[m + 1] - a) / Float(divisions), s = sigmas[m - j]
            let b0 = a + 0.5 * dx
            var sum: Float = 0
            for h in 0 ..< divisions {
                let b = Float(h) * dx + b0
                var prod: Float = 1
                for k in 0 ..< j { let t = sigmas[m - k]; prod *= (b - t) / (s - t) }
                for k in (j + 1) ..< max(order, j + 1) { let t = sigmas[m - k]; prod *= (b - t) / (s - t) }
                sum += prod
            }
            return sum * dx
        }
        var x = x0
        var hist: [MLXArray] = []
        for i in 0 ..< steps {
            let s = sigmas[i]
            guard let den = model(x, s, i + 1) else { return nil }
            let order = min(maxOrder, i + 1)
            let c = (0 ..< order).map { coeff(order, i, $0) }
            let d = (x - den) / s
            x = x + d * c[0]
            if maxOrder > 1 {
                let p1 = hist.count + 1
                if i > 0 {
                    let hmax = hist.count - 1
                    if order >= 2 {
                        for k in 2 ... order { x = x + hist[min(hmax, p1 - k + shift)] * c[k - 1] }
                    }
                }
                if p1 == maxOrder { hist.removeFirst() }
                hist.append(d)
            }
        }
        return x
    }
}

// MARK: - EasyCache

/// Step reuse for DiT models — sd.cpp's EasyCache (runtime/easycache.hpp),
/// the "sampling acceleration" setting. Between 15 % and 95 % of the
/// schedule it tracks how much the model's output moves per unit of input
/// change; while the accumulated estimate stays under the threshold, a step
/// reuses each condition's last output-minus-input instead of running the
/// model. Conditions are keyed by index (0 = the prompt, the anchor).
final class EasyCache {
    let threshold: Float
    let startSigma: Float, endSigma: Float
    private var step = -1
    private var active = false
    private var skip = false
    private var diffs: [Int: MLXArray] = [:]
    private var prevIn: MLXArray? = nil, prevOut: MLXArray? = nil
    private var outNorm: Float = 0
    private var rate: Float? = nil
    private var cumulative: Float = 0
    private var lastInChange: Float? = nil
    private(set) var skipped = 0

    init(threshold: Float, denoiser d: FlowDenoiser) {
        self.threshold = threshold
        startSigma = d.tToSigma((1 - 0.15) * (TIMESTEPS - 1))
        endSigma = d.tToSigma((1 - 0.95) * (TIMESTEPS - 1))
    }

    private func begin(_ index: Int, _ sigma: Float) {
        if index == step { return }
        step = index
        skip = false
        lastInChange = nil
        active = sigma <= startSigma && sigma > endSigma
    }

    private static func meanAbs(_ a: MLXArray) -> Float { abs(a).mean().item(Float.self) }

    /// The model's output for condition `cond` at `x`, computed or reused.
    func run(cond: Int, x: MLXArray, sigma: Float, index: Int, compute: () -> MLXArray) -> MLXArray {
        begin(index, sigma)
        if active {
            if skip, let d = diffs[cond] { return x + d }
            if !skip && cond == 0, let pi = prevIn, prevOut != nil, let d = diffs[cond] {
                let change = Self.meanAbs(x - pi)
                lastInChange = change
                if let rate, change > 0, outNorm > 0 {
                    cumulative += rate * change / outNorm
                    if cumulative < threshold {
                        skip = true
                        skipped += 1
                        return x + d
                    }
                    cumulative = 0
                }
            }
        }
        let out = compute()
        if active {
            diffs[cond] = out - x
            if cond == 0 {
                var outChange: Float = 0
                if let po = prevOut { outChange = Self.meanAbs(out - po) }
                prevIn = x
                prevOut = out
                outNorm = Self.meanAbs(out)
                if let lic = lastInChange, lic > 0, outChange > 0 {
                    let r = outChange / lic
                    if r.isFinite { rate = r }
                }
                cumulative = 0
                lastInChange = nil
            }
        }
        return out
    }
}
