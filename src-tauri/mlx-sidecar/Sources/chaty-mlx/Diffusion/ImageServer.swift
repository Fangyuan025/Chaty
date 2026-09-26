// Image generation on MLX: the sidecar's second mode.
//
// Started with `--image`, chaty-mlx speaks the protocol chaty-sd speaks — the
// same commands, the same events in the same order — so the app drives an
// MLX image model exactly as it drives a GGUF one: the studio, its progress
// and previews, stopping now or after the current picture, the files and the
// parameters inside them all behave the same. What differs is only what runs
// the model.
//
//   {"cmd":"load","model":"<folder>", "backend":"cpu"?}   → load_stage* load_progress* (loaded | error)
//   {"cmd":"generate","id":"…", …}                        → stage/progress/preview/cache* image* (done | error)
//   {"cmd":"cancel","mode":"all"|"after_current"}
//   {"cmd":"quit"}
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import Tokenizers
import UniformTypeIdentifiers

// MARK: - Families

/// One image-model family on this engine.
protocol ImageFamily: AnyObject {
    /// What the model is called in the studio ("Z-Image Turbo").
    var version: String { get }
    var latentChannels: Int { get }
    /// Pixels per latent cell.
    var vaeScale: Int { get }
    /// Width and height must be multiples of this.
    var align: Int { get }
    var flowKind: FlowKind { get }
    var defaultShift: Float { get }
    var defaultScheduler: String { get }
    /// Latent → RGB for the quick preview: [channels][3] and a bias.
    var previewProj: [[Float]] { get }
    var previewBias: [Float] { get }
    /// The prompt's encoding, as the transformer takes it.
    func encode(_ prompt: String) throws -> MLXArray
    /// The model's output for latent `x` at `sigma` — the velocity, in
    /// sd.cpp's convention (denoised = x − σ·v).
    func velocity(_ x: MLXArray, sigma: Float, cond: MLXArray) -> MLXArray
    /// The latent as a picture of channels, [h, w, C] — what previews and
    /// the tiled decode work on.
    func hwc(_ latent: MLXArray) -> MLXArray
    /// [h, w, C] (diffusion space) → picture [H, W, 3] in [-1, 1].
    func decodeHWC(_ latent: MLXArray) -> MLXArray
    /// Picture [H, W, 3] in [-1, 1] → latent; nil when the save has no encoder.
    func encodeImage(_ image: MLXArray) -> MLXArray?
    /// The latent's shape for a picture of this size (also where a family
    /// learns the grid it is about to run on).
    func latentShape(width: Int, height: Int) -> [Int]
    /// Per-generation settings a family reads beyond the shared ones.
    func configure(_ cmd: [String: Any])
}

extension ImageFamily {
    func decode(_ latent: MLXArray) -> MLXArray { decodeHWC(hwc(latent)) }
    func configure(_ cmd: [String: Any]) {}
}

/// FLUX.1's latent space, which Z-Image shares (sd.cpp's flux_latent_rgb_proj).
let fluxLatentRGB: [[Float]] = [
    [-0.041168, 0.019917, 0.097253], [0.028096, 0.026730, 0.129576], [0.065618, -0.067950, -0.014651],
    [-0.012998, -0.014762, 0.081251], [0.078567, 0.059296, -0.024687], [-0.015987, -0.003697, 0.005012],
    [0.033605, 0.138999, 0.068517], [-0.024450, -0.063567, -0.030101], [-0.040194, -0.016710, 0.127185],
    [0.112681, 0.088764, -0.041940], [-0.023498, 0.093664, 0.025543], [0.082899, 0.048320, 0.007491],
    [0.075712, 0.074139, 0.081965], [-0.143501, 0.018263, -0.136138], [-0.025767, -0.082035, -0.040023],
    [-0.111849, -0.055589, -0.032361],
]
let fluxLatentRGBBias: [Float] = [0.024600, -0.006937, -0.008089]

final class ZImageFamily: ImageFamily {
    let version: String
    let latentChannels = 16
    let vaeScale = 8
    let align = 16
    let flowKind = FlowKind.discrete
    let defaultShift: Float = 3
    let defaultScheduler = "discrete"
    var previewProj: [[Float]] { fluxLatentRGB }
    var previewBias: [Float] { fluxLatentRGBBias }

    let tokenizer: Tokenizer
    let textEncoder: Qwen3TextEncoder
    let transformer: ZImageTransformer
    let vae: FluxVAE

    init(dir: URL, turbo: Bool, load: LoadReporter) async throws {
        version = turbo ? "Z-Image Turbo" : "Z-Image"
        tokenizer = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer"))
        textEncoder = try Qwen3TextEncoder(load.store(dir, "text_encoder"), .zImage)
        transformer = try ZImageTransformer(load.store(dir, "transformer"))
        vae = try FluxVAE(load.store(dir, "vae"))
    }

    func encode(_ prompt: String) throws -> MLXArray {
        let ids = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]], chatTemplate: nil, addGenerationPrompt: true,
            truncation: true, maxLength: 512, tools: nil, additionalContext: ["enable_thinking": true])
        return textEncoder(MLXArray(ids.map { Int32($0) }).reshaped([1, -1]))
    }

    func velocity(_ x: MLXArray, sigma: Float, cond: MLXArray) -> MLXArray {
        transformer(x.asType(.bfloat16), timestep: 1 - sigma, cap: cond)
    }

    func hwc(_ latent: MLXArray) -> MLXArray { latent[0..., 0].transposed(1, 2, 0) }
    func decodeHWC(_ latent: MLXArray) -> MLXArray { vae.decode(latent.asType(.bfloat16)) }
    func encodeImage(_ image: MLXArray) -> MLXArray? { vae.encode(image.asType(.bfloat16)) }
    func latentShape(width: Int, height: Int) -> [Int] { [16, 1, height / 8, width / 8] }
}

/// Loading reported the way chaty-sd reports it: a stage per component,
/// progress across every weight file of the model.
final class LoadReporter {
    let total: Int
    private var done = 0
    private let emit: (String, Int, Int) -> Void

    init(dir: URL, components: [String], emit: @escaping (String, Int, Int) -> Void) {
        total = max(1, components.map { WeightStore.shardCount(dir.appendingPathComponent($0)) }.reduce(0, +))
        self.emit = emit
    }

    func store(_ dir: URL, _ component: String) throws -> WeightStore {
        emit(component == "transformer" ? "diffusion" : component, done, total)
        return try WeightStore(dir: dir.appendingPathComponent(component)) { _, _ in
            self.done += 1
            self.emit("", self.done, self.total)
        }
    }
}

/// Which family an mflux save is, from what its transformer holds.
enum FamilyProbe {
    /// The transformer's weight names: from its index, or, for a save
    /// without one, from the shards' headers.
    static func transformerKeys(_ dir: URL) -> [String] {
        let tdir = dir.appendingPathComponent("transformer")
        if let data = try? Data(contentsOf: tdir.appendingPathComponent("model.safetensors.index.json")),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let map = obj["weight_map"] as? [String: Any]
        {
            return Array(map.keys)
        }
        var keys: [String] = []
        let shards = ((try? FileManager.default.contentsOfDirectory(atPath: tdir.path)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }
        for f in shards {
            guard let h = FileHandle(forReadingAtPath: tdir.appendingPathComponent(f).path) else { continue }
            defer { try? h.close() }
            guard let lenData = try? h.read(upToCount: 8), lenData.count == 8 else { continue }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len < 100_000_000, let hdr = try? h.read(upToCount: Int(len)),
                let obj = try? JSONSerialization.jsonObject(with: hdr) as? [String: Any]
            else { continue }
            keys += obj.keys.filter { $0 != "__metadata__" }
        }
        return keys
    }

    static func load(dir: URL, progress: @escaping (String, Int, Int) -> Void) async throws -> ImageFamily {
        let keys = transformerKeys(dir)
        if keys.contains(where: { $0.hasPrefix("all_x_embedder.") }) && keys.contains(where: { $0.hasPrefix("context_refiner.") }) {
            // Turbo and base share the architecture; the name is the only tell.
            let turbo = dir.lastPathComponent.lowercased().contains("turbo")
            let load = LoadReporter(dir: dir, components: ["text_encoder", "transformer", "vae"], emit: progress)
            return try await ZImageFamily(dir: dir, turbo: turbo, load: load)
        }
        // FLUX.1: double-stream blocks, then single-stream ones. FLUX.2 and
        // FIBO share those names; FLUX.1 alone projects CLIP's pooled vector.
        if keys.contains(where: { $0.hasPrefix("single_transformer_blocks.") })
            && keys.contains(where: { $0.hasPrefix("x_embedder.") })
            && keys.contains(where: { $0.hasPrefix("time_text_embed.text_embedder.") })
        {
            let load = LoadReporter(
                dir: dir, components: ["text_encoder", "text_encoder_2", "transformer", "vae"], emit: progress)
            return try await FluxFamily(dir: dir, load: load)
        }
        // Qwen-Image (1.0, 2512): a modulation per stream in every block.
        if keys.contains(where: { $0.contains("img_mod_linear") }) && keys.contains(where: { $0.contains("add_q_proj") }) {
            let load = LoadReporter(dir: dir, components: ["text_encoder", "transformer", "vae"], emit: progress)
            return try await QwenImageFamily(dir: dir, load: load)
        }
        // Qwen-Image 2.1: one shared modulation and a gated MLP per block.
        if keys.contains(where: { $0.hasPrefix("modulation.layers.") })
            && keys.contains(where: { $0.contains("img_mlp.gate_layer") })
        {
            let load = LoadReporter(dir: dir, components: ["text_encoder", "transformer", "vae"], emit: progress)
            return try await QwenImage21Family(dir: dir, load: load)
        }
        throw DiffusionError.unsupported(
            "这个 MLX 生图模型的结构 Chaty 还不支持 (this MLX image model's architecture is not supported yet)")
    }
}

// MARK: - Pictures

enum Picture {
    /// [H, W, 3] in [-1, 1] → RGB bytes.
    static func bytes(_ img: MLXArray) -> (Int, Int, [UInt8]) {
        let u8 = (clip(img.asType(.float32) / 2 + 0.5, min: 0, max: 1) * 255).round().asType(.uint8)
        eval(u8)
        return (u8.dim(1), u8.dim(0), u8.asArray(UInt8.self))
    }

    static func cgImage(width: Int, height: Int, rgb: [UInt8]) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(rgb) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    static func encode(_ image: CGImage, jpeg: Bool, quality: Double) -> Data? {
        let data = NSMutableData()
        let type = (jpeg ? UTType.jpeg : UTType.png).identifier as CFString
        guard let dest = CGImageDestinationCreateWithData(data, type, 1, nil) else { return nil }
        let props: [CFString: Any] = jpeg ? [kCGImageDestinationLossyCompressionQuality: quality] : [:]
        CGImageDestinationAddImage(dest, image, props as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    private static let crcTable: [UInt32] = (0 ..< 256).map { n -> UInt32 in
        var c = UInt32(n)
        for _ in 0 ..< 8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    private static func crc32(_ bytes: [UInt8], _ start: UInt32 = 0xFFFF_FFFF) -> UInt32 {
        var c = start
        for b in bytes { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c
    }

    /// An iTXt chunk right after IHDR, as chaty-sd writes it: the settings,
    /// under the key every Stable Diffusion tool reads them from.
    static func addText(_ png: Data, key: String, text: String) -> Data {
        guard png.count >= 33, !text.isEmpty else { return png }
        var body = Array(key.utf8) + [0, 0, 0, 0, 0]
        body += Array(text.utf8)
        let type: [UInt8] = Array("iTXt".utf8)
        let len = UInt32(body.count)
        var chunk: [UInt8] = [UInt8(len >> 24 & 0xFF), UInt8(len >> 16 & 0xFF), UInt8(len >> 8 & 0xFF), UInt8(len & 0xFF)]
        chunk += type + body
        let crc = crc32(body, crc32(type)) ^ 0xFFFF_FFFF
        chunk += [UInt8(crc >> 24 & 0xFF), UInt8(crc >> 16 & 0xFF), UInt8(crc >> 8 & 0xFF), UInt8(crc & 0xFF)]
        var out = png
        out.insert(contentsOf: chunk, at: 33)
        return out
    }

    /// A picture file, scaled to `width`×`height`, as [H, W, 3] in [-1, 1].
    static func load(_ path: String, width: Int, height: Int) -> MLXArray? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
            let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return nil }
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        let ok = buf.withUnsafeMutableBytes { p -> Bool in
            guard let ctx = CGContext(
                data: p.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else { return nil }
        let rgba = MLXArray(buf, [height, width, 4])
        return rgba[0..., 0..., ..<3].asType(.float32) / 127.5 - 1
    }

    static func uniquePath(_ dir: URL, _ stem: String, _ ext: String) -> URL {
        var p = dir.appendingPathComponent(stem + ext)
        var i = 2
        while FileManager.default.fileExists(atPath: p.path) && i < 10000 {
            p = dir.appendingPathComponent("\(stem)-\(i)\(ext)")
            i += 1
        }
        return p
    }
}

// MARK: - Server

final class ImageServer: @unchecked Sendable {
    private let lock = NSLock()
    private var family: ImageFamily?
    private var cancelMode: String? = nil
    private var busy = false
    private var condCache: [String: MLXArray] = [:]

    private func setCancel(_ m: String?) {
        lock.lock()
        cancelMode = m
        lock.unlock()
    }

    private var cancel: String? {
        lock.lock()
        defer { lock.unlock() }
        return cancelMode
    }

    static func deviceName() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(cString: buf)
    }

    static func main() async {
        Memory.cacheLimit = 1024 * 1024 * 1024
        let server = ImageServer()
        out.emit(["event": "ready", "protocol": "1", "devices": [["name": "MTL0", "description": deviceName()]]])
        log("image mode ready")
        var job: Task<Void, Never>? = nil
        do {
            for try await line in FileHandle.standardInput.bytes.lines {
                guard let data = line.data(using: .utf8),
                    let cmd = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let name = cmd["cmd"] as? String
                else { continue }
                switch name {
                case "load":
                    await job?.value
                    job = Task.detached { await server.load(cmd) }
                case "generate":
                    if server.isBusy {
                        out.emit(["event": "error", "id": cmd["id"] as? String ?? "", "scope": "generate",
                                  "message": "an image is already being generated"])
                        continue
                    }
                    await job?.value
                    job = Task.detached { server.generate(cmd) }
                case "cancel":
                    server.setCancel((cmd["mode"] as? String) == "after_current" ? "after_current" : "all")
                case "ping":
                    out.emit(["event": "pong"])
                case "quit":
                    server.setCancel("all")
                    await job?.value
                    _exit(0)
                default:
                    out.emit(["event": "error", "message": "unknown command: \(name)"])
                }
            }
        } catch {
            log("stdin error: \(error)")
        }
        server.setCancel("all")
        await job?.value
        _exit(0)
    }

    var isBusy: Bool {
        lock.lock()
        defer { lock.unlock() }
        return busy
    }

    private func load(_ cmd: [String: Any]) async {
        let started = Date()
        guard let path = (cmd["model"] as? String) ?? (cmd["diffusion_model"] as? String) else {
            out.emit(["event": "error", "scope": "load", "message": "load requires a model folder"])
            return
        }
        if (cmd["backend"] as? String) == "cpu" {
            Device.setDefault(device: Device(.cpu))
        }
        do {
            let fam = try await FamilyProbe.load(dir: URL(fileURLWithPath: path)) { component, i, n in
                if !component.isEmpty { out.emit(["event": "load_stage", "component": component]) }
                out.emit(["event": "load_progress", "step": i, "steps": n])
            }
            family = fam
            condCache = [:]
            out.emit([
                "event": "loaded", "version": fam.version, "supports_image": true, "supports_video": false,
                "default_sampler": "euler", "default_scheduler": fam.defaultScheduler,
                "samplers": Sampler.names, "schedulers": Schedule.names,
                "elapsed_ms": Int(Date().timeIntervalSince(started) * 1000),
            ])
        } catch {
            out.emit(["event": "error", "scope": "load", "message": error.localizedDescription])
        }
    }

    private func cond(_ fam: ImageFamily, _ text: String) throws -> (MLXArray, Bool) {
        if let c = condCache[text] { return (c, true) }
        let c = try fam.encode(text)
        eval(c)
        condCache[text] = c
        if condCache.count > 8 { condCache.removeValue(forKey: condCache.keys.first!) }
        return (c, false)
    }

    /// One generation. The closing event (done or error) is sent only once
    /// the server is free again: the app sends the next job the moment it
    /// reads it, and a job that arrives while `busy` is still set is refused.
    private func generate(_ cmd: [String: Any]) {
        lock.lock()
        busy = true
        cancelMode = nil
        lock.unlock()
        let closing = run(cmd)
        Memory.clearCache()
        lock.lock()
        busy = false
        lock.unlock()
        out.emit(closing)
    }

    private func run(_ cmd: [String: Any]) -> [String: Any] {
        let id = cmd["id"] as? String ?? ""
        let t0 = Date()
        func ms() -> Int { Int(Date().timeIntervalSince(t0) * 1000) }
        func fail(_ m: String, scope: String = "generate") -> [String: Any] {
            ["event": "error", "id": id, "scope": scope, "message": m]
        }
        guard let fam = family else { return fail("no model loaded") }

        func int(_ k: String, _ d: Int) -> Int { (cmd[k] as? NSNumber)?.intValue ?? d }
        func flt(_ k: String, _ d: Float) -> Float { (cmd[k] as? NSNumber)?.floatValue ?? d }
        func str(_ k: String, _ d: String = "") -> String { cmd[k] as? String ?? d }

        let align = fam.align
        let width = max(align, int("width", 1024) / align * align)
        let height = max(align, int("height", 1024) / align * align)
        let steps = max(1, int("steps", 20))
        let cfg = flt("cfg_scale", 1)
        let batch = max(1, int("batch_count", 1))
        let prompt = str("prompt"), negative = str("negative_prompt")
        let samplerIn = str("sampler")
        let sampler = Sampler.names.contains(samplerIn) ? samplerIn : "euler"
        var scheduler = str("scheduler")
        if sampler == "lcm" || sampler == "tcd" { if scheduler.isEmpty { scheduler = "lcm" } }
        if sampler == "ddim_trailing" && scheduler.isEmpty { scheduler = "simple" }
        if scheduler.isEmpty || !(Schedule.names.contains(scheduler) || scheduler == "flux" || scheduler == "flux2") {
            scheduler = fam.defaultScheduler
        }
        let shiftIn = flt("flow_shift", 0)
        let denoiser = FlowDenoiser(kind: fam.flowKind, shift: shiftIn > 0 && shiftIn.isFinite ? shiftIn : fam.defaultShift)
        var seed = Int64(int("seed", -1))
        if seed < 0 { seed = Int64(UInt32.random(in: 0 ... UInt32.max) & 0x7FFF_FFFF) }
        let previewMode = str("preview", "proj")
        let previewEvery = max(1, int("preview_interval", 1))
        let jpeg = ["jpg", "jpeg"].contains(str("format", "png"))
        let outDir = URL(fileURLWithPath: str("out_dir", "."))
        let stem = str("file_stem", "image")
        let metadata = str("metadata")
        let strength = flt("strength", 0.75)

        func stage(_ s: String, _ index: Int, _ count: Int, seed: Int64? = nil) {
            var ev: [String: Any] = ["event": "stage", "id": id, "stage": s, "index": index, "count": count]
            if let seed { ev["seed"] = seed }
            out.emit(ev)
        }

        fam.configure(cmd)
        stage("encode", 0, batch)
        let condC: MLXArray, condU: MLXArray?
        do {
            let (c, hitC) = try cond(fam, prompt)
            condC = c
            var hitU = true
            if cfg > 1 {
                let (u, h) = try cond(fam, negative)
                condU = u
                hitU = h
            } else {
                condU = nil
            }
            if hitC && hitU { out.emit(["event": "cache", "id": id, "kind": "conditioning"]) }
        } catch {
            return fail(error.localizedDescription)
        }

        // The schedule, cut short when starting from a picture.
        let lshape = fam.latentShape(width: width, height: height)
        let seqLen = (height / fam.vaeScale) * (width / fam.vaeScale)
        var sigmas = Schedule.sigmas(scheduler, steps: steps, denoiser: denoiser, imageSeqLen: seqLen)
        // For checking the pipeline against another implementation's schedule.
        if let d = cmd["debug_sigmas"] as? [NSNumber], d.count > 1 { sigmas = d.map { $0.floatValue } }
        var initLatent: MLXArray? = nil
        if let ip = cmd["init_image"] as? String, !ip.isEmpty {
            guard let img = Picture.load(ip, width: width, height: height) else {
                return fail("cannot read image: \(ip)")
            }
            guard let lat = fam.encodeImage(img) else { return fail("this model cannot start from a picture") }
            initLatent = lat.asType(.float32)
            if strength < 1 {
                let n = sigmas.count - 1
                var tEnc = Int(Float(n) * strength)
                if tEnc == n { tEnc -= 1 }
                sigmas = Array(sigmas[(n - tEnc - 1)...])
            }
        }
        let sampleSteps = sigmas.count - 1

        var latents: [(MLXArray, Int64)] = []
        var cancelledAll = false
        for b in 0 ..< batch {
            if cancel != nil { break }
            let s = seed + Int64(b)
            stage("sample", b, batch, seed: s)
            let noise = MLXRandom.normal(lshape, key: MLXRandom.key(UInt64(s)))
            let init0 = initLatent ?? zeros(lshape)
            let x0 = init0 * (1 - sigmas[0]) + noise * sigmas[0]
            let extra = NoiseSource(seed: UInt64(s) &+ 0x9E37_79B9, shape: lshape)
            var reported = 0
            var last = Date()
            // The step cache, when sampling acceleration is on (the app
            // resolves the level to a method and threshold per family).
            let cache: EasyCache? = str("cache_mode") == "easycache"
                ? EasyCache(threshold: flt("cache_threshold", 0.2), denoiser: denoiser) : nil
            let model: DenoiseFn = { x, sigma, step in
                if self.cancel == "all" { return nil }
                let index = abs(step)
                func predict(_ k: Int, _ c: MLXArray) -> MLXArray {
                    let f = { fam.velocity(x, sigma: sigma, cond: c).asType(.float32) }
                    return cache?.run(cond: k, x: x, sigma: sigma, index: index, compute: f) ?? f()
                }
                var v = predict(0, condC)
                if let condU {
                    let vu = predict(1, condU)
                    v = vu + cfg * (v - vu)
                }
                let den = x - sigma * v
                eval(den)
                let k = abs(step)
                if k > reported {
                    reported = k
                    let now = Date()
                    out.emit(["event": "progress", "id": id, "stage": "sample", "step": k, "steps": sampleSteps,
                              "time": now.timeIntervalSince(last)])
                    last = now
                    if previewMode != "none" && (k % previewEvery == 0 || k == sampleSteps) {
                        self.preview(fam, den, step: k, id: id, exact: previewMode == "vae")
                    }
                }
                return den
            }
            guard let x = Sampler.run(sampler, model: model, x: x0, sigmas: sigmas, noise: extra, eta: nil) else {
                cancelledAll = true
                break
            }
            eval(x)
            latents.append((x, s))
            if let cache {
                out.emit(["event": "cache", "id": id, "kind": "steps", "method": "easycache",
                          "skipped": cache.skipped, "total": cache.skipped > 0 ? sampleSteps : 0])
            }
            if cancel == "after_current" { break }
        }

        if cancelledAll || latents.isEmpty {
            return ["event": "done", "id": id, "count": 0, "cancelled": true, "elapsed_ms": ms()]
        }

        stage("decode", 0, latents.count)
        var pictures: [(Int, Int, [UInt8], Int64)] = []
        let tiled = (cmd["vae_tiling"] as? Bool) ?? false
        for (i, (lat, s)) in latents.enumerated() {
            let img = tiled ? Self.decodeTiled(fam, lat) : fam.decode(lat)
            let (w, h, rgb) = Picture.bytes(img)
            pictures.append((w, h, rgb, s))
            out.emit(["event": "progress", "id": id, "stage": "decode", "step": i + 1, "steps": latents.count, "time": 0])
        }

        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        var saved = 0
        for (i, (w, h, rgb, s)) in pictures.enumerated() {
            guard let cg = Picture.cgImage(width: w, height: h, rgb: rgb),
                var data = Picture.encode(cg, jpeg: jpeg, quality: Double(int("jpeg_quality", 95)) / 100)
            else { return fail("cannot encode the picture", scope: "save") }
            if !jpeg && !metadata.isEmpty {
                data = Picture.addText(data, key: "parameters", text: metadata + ", Seed: \(s)")
            }
            let name = pictures.count > 1 ? "\(stem)-\(i + 1)" : stem
            let path = Picture.uniquePath(outDir, name, jpeg ? ".jpg" : ".png")
            do { try data.write(to: path) } catch { return fail("cannot write \(path.path)", scope: "save") }
            out.emit(["event": "image", "id": id, "index": i, "path": path.path, "width": w, "height": h, "channels": 3,
                      "seed": s])
            saved += 1
        }
        return ["event": "done", "id": id, "count": saved, "cancelled": saved < batch, "elapsed_ms": ms()]
    }

    /// Decode in tiles, as stable-diffusion.cpp's VAE tiling does: 32 latent
    /// cells a side, half of each overlapping its neighbour, the overlaps
    /// feathered together. Much less memory for a large picture.
    static func decodeTiled(_ fam: ImageFamily, _ latent: MLXArray) -> MLXArray {
        let lat = fam.hwc(latent)
        let h = lat.dim(0), w = lat.dim(1)
        let tile = 32, stride = 16, s = fam.vaeScale
        if h <= tile && w <= tile { return fam.decodeHWC(lat) }
        func starts(_ n: Int) -> [Int] {
            if n <= tile { return [0] }
            var v = Array(Swift.stride(from: 0, to: n - tile, by: stride))
            v.append(n - tile)
            return v
        }
        func ramp(_ len: Int, _ lead: Bool, _ trail: Bool) -> MLXArray {
            let o = (tile - stride) * s
            let vals = (0 ..< len).map { i -> Float in
                var v: Float = 1
                if lead { v = min(v, Float(i + 1) / Float(o)) }
                if trail { v = min(v, Float(len - i) / Float(o)) }
                return v
            }
            return MLXArray(vals)
        }
        var acc = zeros([h * s, w * s, 3])
        var wsum = zeros([h * s, w * s, 1])
        let ys = starts(h), xs = starts(w)
        for y0 in ys {
            for x0 in xs {
                let th = min(tile, h), tw = min(tile, w)
                let part = fam.decodeHWC(lat[y0 ..< (y0 + th), x0 ..< (x0 + tw)]).asType(.float32)
                let wy = ramp(th * s, y0 > 0, y0 + th < h).reshaped([-1, 1, 1])
                let wx = ramp(tw * s, x0 > 0, x0 + tw < w).reshaped([1, -1, 1])
                let m = wy * wx
                let (py, px) = (y0 * s, x0 * s)
                acc[py ..< (py + th * s), px ..< (px + tw * s)] = acc[py ..< (py + th * s), px ..< (px + tw * s)] + part * m
                wsum[py ..< (py + th * s), px ..< (px + tw * s)] = wsum[py ..< (py + th * s), px ..< (px + tw * s)] + m
                eval(acc, wsum)
            }
        }
        return acc / wsum
    }

    /// The denoised estimate so far, as a small JPEG: the latent projected
    /// straight to RGB ("proj"), or decoded by the VAE ("vae").
    private func preview(_ fam: ImageFamily, _ den: MLXArray, step: Int, id: String, exact: Bool) {
        let rgbImg: MLXArray
        if exact {
            rgbImg = fam.decode(den)
        } else {
            let c = fam.latentChannels
            let lat = fam.hwc(den).asType(.float32)  // [h, w, c]
            let proj = MLXArray(fam.previewProj.flatMap { $0 }, [c, 3])
            let bias = MLXArray(fam.previewBias, [3])
            rgbImg = matmul(lat, proj) + bias
        }
        let (w, h, rgb) = Picture.bytes(rgbImg)
        guard let cg = Picture.cgImage(width: w, height: h, rgb: rgb),
            let jpg = Picture.encode(cg, jpeg: true, quality: 0.82)
        else { return }
        out.emit(["event": "preview", "id": id, "step": step, "width": w, "height": h,
                  "data": "data:image/jpeg;base64," + jpg.base64EncodedString()])
    }
}
