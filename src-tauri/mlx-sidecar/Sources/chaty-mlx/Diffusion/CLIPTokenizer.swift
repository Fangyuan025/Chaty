// CLIP's tokenizer, read from its tokenizer.json.
//
// swift-transformers has no CLIPTokenizer, and its BPE does not know CLIP's
// end-of-word suffix: every word's last symbol carries "</w>", so a word-final
// "a" and a word-inner "a" are different tokens. This is the fast
// tokenizer's pipeline as the file describes it: NFC, whitespace collapsed,
// lower-cased; split with CLIP's pattern; GPT-2's byte-to-unicode map; BPE
// with the suffix; <|startoftext|> … <|endoftext|> around it.
import Foundation

final class CLIPTokenizer {
    let vocab: [String: Int]
    private let ranks: [String: Int]
    let bos: Int, eos: Int
    private let byteMap: [UInt8: Character]
    private let pattern = try! NSRegularExpression(
        pattern: "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+")
    private var cache: [String: [Int]] = [:]

    /// From a tokenizer folder: its tokenizer.json, or — in mflux's early
    /// saves — the slow tokenizer's vocab.json and merges.txt.
    convenience init(folder: URL) throws {
        let json = folder.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: json.path) {
            let data = try Data(contentsOf: json)
            guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let model = obj["model"] as? [String: Any],
                let v = model["vocab"] as? [String: Int]
            else { throw DiffusionError.badFolder("CLIP tokenizer.json is unreadable") }
            let merges: [String] = ((model["merges"] as? [Any]) ?? []).compactMap { m in
                if let pair = m as? [String], pair.count == 2 { return pair[0] + " " + pair[1] }
                return m as? String
            }
            try self.init(vocab: v, merges: merges)
        } else {
            let data = try Data(contentsOf: folder.appendingPathComponent("vocab.json"))
            guard let v = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
                throw DiffusionError.badFolder("CLIP vocab.json is unreadable")
            }
            let text = try String(contentsOf: folder.appendingPathComponent("merges.txt"), encoding: .utf8)
            let merges = text.split(separator: "\n", omittingEmptySubsequences: true)
                .map(String.init)
                .filter { !$0.hasPrefix("#version") }
            try self.init(vocab: v, merges: merges)
        }
    }

    init(vocab v: [String: Int], merges: [String]) throws {
        vocab = v
        var r: [String: Int] = [:]
        for (i, m) in merges.enumerated() { r[m] = i }
        ranks = r
        bos = v["<|startoftext|>"] ?? 49406
        eos = v["<|endoftext|>"] ?? 49407
        // GPT-2's bytes_to_unicode.
        var bs: [Int] = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var m: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) { m[UInt8(b)] = Character(UnicodeScalar(UInt32(c))!) }
        byteMap = m
    }

    private func bpe(_ word: String) -> [Int] {
        if let hit = cache[word] { return hit }
        var symbols = word.map { String($0) }
        guard !symbols.isEmpty else { return [] }
        symbols[symbols.count - 1] += "</w>"
        while symbols.count > 1 {
            var best: (rank: Int, i: Int)? = nil
            for i in 0 ..< symbols.count - 1 {
                if let r = ranks[symbols[i] + " " + symbols[i + 1]], best == nil || r < best!.rank { best = (r, i) }
            }
            guard let b = best else { break }
            let a = symbols[b.i], c = symbols[b.i + 1]
            var merged: [String] = []
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1 && symbols[i] == a && symbols[i + 1] == c {
                    merged.append(a + c)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
        }
        let ids = symbols.map { vocab[$0] ?? eos }
        cache[word] = ids
        return ids
    }

    /// Token ids for `text`, with the start and end tokens.
    func encode(_ text: String) -> [Int] {
        let norm = text.precomposedStringWithCanonicalMapping
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .lowercased()
        var ids = [bos]
        let ns = norm as NSString
        for m in pattern.matches(in: norm, range: NSRange(location: 0, length: ns.length)) {
            let word = ns.substring(with: m.range)
            if word == "<|startoftext|>" { ids.append(bos); continue }
            if word == "<|endoftext|>" { ids.append(eos); continue }
            let mapped = String(word.utf8.map { byteMap[$0]! })
            ids += bpe(mapped)
        }
        ids.append(eos)
        return ids
    }
}
