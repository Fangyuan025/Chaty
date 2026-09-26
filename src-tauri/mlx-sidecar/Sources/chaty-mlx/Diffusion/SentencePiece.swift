// T5's tokenizer, read from its SentencePiece model (`spiece.model`).
//
// mflux's early saves of FLUX.1 ship T5's tokenizer as SentencePiece's own
// file and nothing else, and swift-transformers reads only tokenizer.json.
// This is what the fast tokenizer converted from that file does: NFKC (the
// model's "nmt_nfkc" rules, for everything a prompt holds), split on
// whitespace, "▁" before each word, the best-scoring Unigram segmentation of
// each, a run of unknown characters as one <unk>, then </s>.
import Foundation

final class SentencePieceTokenizer {
    private var pieces: [String: (id: Int, score: Float)] = [:]
    private var maxLen = 1
    private var unk = 2
    private var unkScore: Float = -10
    let eos: Int

    init(file: URL) throws {
        let data = [UInt8](try Data(contentsOf: file))
        var minScore: Float = 0
        var id = 0
        var eosId = 1
        var r = ProtoReader(data)
        while let (field, wire) = r.key() {
            guard field == 1, wire == 2, let body = r.bytes() else {
                r.skip(wire)
                continue
            }
            // SentencePiece { string piece = 1; float score = 2; Type type = 3; }
            var p = ProtoReader(body)
            var piece = "", score: Float = 0, type = 1
            while let (f, w) = p.key() {
                switch (f, w) {
                case (1, 2): piece = String(decoding: p.bytes() ?? [], as: UTF8.self)
                case (2, 5): score = p.float32()
                case (3, 0): type = Int(p.varint())
                default: p.skip(w)
                }
            }
            switch type {
            case 1, 4:  // normal, user-defined
                pieces[piece] = (id, score)
                maxLen = max(maxLen, piece.unicodeScalars.count)
                minScore = min(minScore, score)
            case 2:
                unk = id
            case 3 where piece == "</s>":
                eosId = id
            default:
                break
            }
            id += 1
        }
        guard !pieces.isEmpty else { throw DiffusionError.badFolder("spiece.model is unreadable") }
        eos = eosId
        unkScore = minScore - 10
    }

    /// Token ids for `text`, ending with </s>.
    func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        let norm = text.precomposedStringWithCompatibilityMapping
        for word in norm.split(whereSeparator: { $0.unicodeScalars.allSatisfy { $0.properties.isWhitespace } }) {
            ids += segment(Array(("\u{2581}" + word).unicodeScalars))
        }
        ids.append(eos)
        return ids
    }

    /// The best segmentation of one word (Viterbi over the pieces).
    private func segment(_ s: [Unicode.Scalar]) -> [Int] {
        let n = s.count
        var best = [Float](repeating: -.infinity, count: n + 1)
        var back = [(start: Int, id: Int)](repeating: (0, 0), count: n + 1)
        best[0] = 0
        for i in 0 ..< n where best[i] > -.infinity {
            var single = false
            var key = ""
            for len in 1 ... min(maxLen, n - i) {
                key.unicodeScalars.append(s[i + len - 1])
                guard let (id, score) = pieces[key] else { continue }
                if len == 1 { single = true }
                if best[i] + score > best[i + len] {
                    best[i + len] = best[i] + score
                    back[i + len] = (i, id)
                }
            }
            if !single && best[i] + unkScore > best[i + 1] {
                best[i + 1] = best[i] + unkScore
                back[i + 1] = (i, unk)
            }
        }
        var out: [Int] = []
        var j = n
        while j > 0 {
            let (i, id) = back[j]
            if !(id == unk && out.last == unk) { out.append(id) }
            j = i
        }
        return out.reversed()
    }
}

/// Just enough protobuf to read a SentencePiece model.
private struct ProtoReader {
    let d: [UInt8]
    var i = 0
    init(_ d: [UInt8]) { self.d = d }

    mutating func varint() -> UInt64 {
        var v: UInt64 = 0, shift: UInt64 = 0
        while i < d.count {
            let b = d[i]
            i += 1
            v |= UInt64(b & 0x7f) << shift
            if b & 0x80 == 0 { break }
            shift += 7
        }
        return v
    }
    mutating func key() -> (Int, Int)? {
        guard i < d.count else { return nil }
        let k = varint()
        return (Int(k >> 3), Int(k & 7))
    }
    mutating func bytes() -> [UInt8]? {
        let n = Int(varint())
        guard n >= 0, i + n <= d.count else { i = d.count; return nil }
        defer { i += n }
        return Array(d[i ..< i + n])
    }
    mutating func float32() -> Float {
        guard i + 4 <= d.count else { i = d.count; return 0 }
        let bits = UInt32(d[i]) | UInt32(d[i + 1]) << 8 | UInt32(d[i + 2]) << 16 | UInt32(d[i + 3]) << 24
        i += 4
        return Float(bitPattern: bits)
    }
    mutating func skip(_ wire: Int) {
        switch wire {
        case 0: _ = varint()
        case 1: i += 8
        case 2: _ = bytes()
        case 5: i += 4
        default: i = d.count
        }
    }
}
