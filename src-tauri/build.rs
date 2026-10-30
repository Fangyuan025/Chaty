fn main() {
    // macOS: make the binary look for the sherpa-onnx / ONNX Runtime dylibs
    // next to itself and in the bundled Frameworks dir, so voice works from
    // inside the .app (sherpa-rs links them by install-name at build time).
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        // chaty-llama links the same library but uses none of the voice code:
        // without this it would still name the ONNX Runtime and sherpa dylibs,
        // and a runtime pack of it — run from the data folder, nowhere near
        // the app's Frameworks — would not start.
        println!("cargo:rustc-link-arg-bin=chaty-llama=-Wl,-dead_strip_dylibs");
        // The linker keeps sherpa's dylib all the same (no symbol comes from
        // it); weak, a missing one is no reason not to start.
        println!("cargo:rustc-link-arg-bin=chaty-llama=-Wl,-weak-lsherpa-onnx-c-api");
    }
    // Windows: the same voice DLLs, delay-loaded — fetched on a first call
    // that chaty-llama never makes, so a runtime pack starts without them.
    // (Linux: scripts/llama-standalone.sh.)
    #[cfg(windows)]
    {
        for dll in ["sherpa-onnx-c-api.dll", "sherpa-onnx-cxx-api.dll", "onnxruntime.dll", "cargs.dll"] {
            println!("cargo:rustc-link-arg-bin=chaty-llama=/DELAYLOAD:{dll}");
        }
        println!("cargo:rustc-link-arg-bin=chaty-llama=delayimp.lib");
    }

    tauri_build::build()
}
