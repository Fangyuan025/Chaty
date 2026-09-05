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
        // Our fork, one commit ahead of 3.31.4: Qwen3.5 hands back the hidden
        // states its multi-token-prediction head consumes (`HiddenStateProviding`).
        // Everything that head needs is internal to MLXVLM upstream, so without
        // this the sidecar cannot drive it and the speedup is unreachable.
        // macOS-only target, so a git dependency costs nothing on Windows.
        .package(
            url: "https://github.com/Fangyuan025/mlx-swift-lm",
            revision: "c11767bc3aa01f33683ef39fa5adb9ad47f214ef"),
        // mlx-swift-lm is tokenizer-agnostic; the swift-transformers tokenizer
        // is injected in OUR module via MLXHuggingFace's macros.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        // Root-level pin: swift-jinja 2.4.0 changed template-object keys to
        // `ObjectKey`, which swift-transformers ≤1.3.3 doesn't compile
        // against. Drop this once a fixed swift-transformers ships.
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.3.6"),
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
