// swift-tools-version: 5.10
// chaty-mlx — MLX inference sidecar for Chaty (macOS, Apple Silicon).
//
// A tiny stdio-JSON server around mlx-swift-lm: the Rust backend spawns one
// instance per loaded model, streams commands on stdin and reads events on
// stdout. Process isolation doubles as the memory-release guarantee — ejecting
// an MLX model is `kill(child)`, so weights can never linger in the app.
//
// ⚠️ Building the Metal shaders requires Xcode (SwiftPM alone can't compile
// .metal): use `scripts/build-mlx-sidecar.sh`, which drives xcodebuild.
import PackageDescription

let package = Package(
    name: "chaty-mlx",
    platforms: [.macOS(.v14)],
    dependencies: [
        // Upstream, by revision: a runtime pack builds this against a newer
        // one (scripts/build-mlx-sidecar.sh, CHATY_MLX_LM_REV). Its manifests
        // and mlx-swift's need Xcode 26's toolchain.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "6759555f2bd8f18c242102d73d149efde9f2f2e0"),
        // mlx-swift-lm is tokenizer-agnostic; the swift-transformers tokenizer
        // is injected in OUR module via MLXHuggingFace's macros.
        // 1.4.0: BPE honours `ignore_merges` and splits by code point, which
        // LFM3's tokenizer needs (mlx-swift-lm refuses one that does not), and
        // it builds against swift-jinja 2.4.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.4.0"),
        // Root-level pin: the lowest swift-jinja 1.4.0 takes. Left to float it
        // resolves to 3.x, a major version of the engine every MLX chat
        // template is rendered with.
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.4.2"),
    ],
    targets: [
        .executableTarget(
            name: "chaty-mlx",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                // Natively-multimodal architectures (Qwen3.5+) only exist in
                // the VLM registry; we load them there and chat text-only.
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        )
    ]
)
