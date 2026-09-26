// A check of the image models against mflux, run by hand:
//   chaty-mlx --image-selftest <model folder> <reference.safetensors> <out.safetensors>
//   chaty-mlx --image-tokens <FLUX.1 folder> <prompts.json>   (ids only, no weights)
// The reference holds mflux's own intermediates for one prompt at 512×512
// (caption features, the transformer's output at two sigmas on the same
// noise, a VAE decode of that noise); this computes the same from the same
// inputs so the two can be compared.
import Foundation
import MLX

enum ImageSelfTest {
    static func run(model: String, ref: String, out outPath: String) async {
        do {
            let r = try loadArrays(url: URL(fileURLWithPath: ref))
            let fam = try await FamilyProbe.load(dir: URL(fileURLWithPath: model)) { c, i, n in
                log("loading \(c) \(i)/\(n)")
            }
            let prompt = "A neon sign that reads \"LATE NIGHT DINER\" on a rainy street corner, reflections on wet pavement, cinematic"
            _ = fam.latentShape(width: 512, height: 512)
            let cap = try fam.encode(prompt)
            // The transformer is fed the reference caption, so its check does
            // not depend on the encoder's.
            let refCap = r["cap"]!.asType(.bfloat16)
            let noise = r["noise"]!.asType(.bfloat16)
            // The reference may say which sigmas it used (its scheduler's).
            let sig = r["sigmas"].map { $0.asArray(Float.self) } ?? [1.0, 0.5]
            fam.configure(["guidance": 3.5])
            let v1 = fam.velocity(noise, sigma: sig[0], cond: refCap)
            let v05 = fam.velocity(noise, sigma: sig[1], cond: refCap)
            let dec = fam.decode(noise)
            eval(cap, v1, v05, dec)
            var arrays: [String: MLXArray] = [
                "cap": cap.asType(.float32), "v_s1": v1.asType(.float32), "v_s05": v05.asType(.float32),
                "dec": dec.asType(.float32),
            ]
            if let f = fam as? FluxFamily {
                let (c, t) = f.tokens(prompt)
                arrays["clip_ids"] = MLXArray(c.map { Int32($0) })
                arrays["t5_ids"] = MLXArray(t.map { Int32($0) })
            }
            if let z = fam as? ZImageFamily {
                let ids = try z.tokenizer.applyChatTemplate(
                    messages: [["role": "user", "content": prompt]], chatTemplate: nil, addGenerationPrompt: true,
                    truncation: true, maxLength: 512, tools: nil, additionalContext: ["enable_thinking": true])
                arrays["ids"] = MLXArray(ids.map { Int32($0) })
            }
            try save(arrays: arrays, url: URL(fileURLWithPath: outPath))
            log("selftest written: \(outPath)")
        } catch {
            log("selftest failed: \(error)")
        }
    }

    /// A FLUX.1 folder's CLIP and T5 ids for each prompt of a JSON list,
    /// printed as JSON, without loading a weight.
    static func tokens(model: String, prompts: String) async {
        do {
            let list = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: prompts))) as? [String] ?? []
            let (clip, t5) = try await FluxFamily.tokenizers(URL(fileURLWithPath: model))
            let out = list.map { ["clip": clip.encode($0), "t5": t5($0)] }
            print(String(decoding: try JSONSerialization.data(withJSONObject: out), as: UTF8.self))
            fflush(stdout)
        } catch {
            log("tokens failed: \(error)")
        }
    }
}
