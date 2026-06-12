fn main() {
    // macOS: make the binary look for the sherpa-onnx / ONNX Runtime dylibs
    // next to itself and in the bundled Frameworks dir, so voice works from
    // inside the .app (sherpa-rs links them by install-name at build time).
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    }

    tauri_build::build()
}
