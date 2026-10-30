//! chaty-llama: the llama.cpp engine in its own process, driven over stdin and
//! stdout by the app (inference/llama_sidecar.rs). Replaceable by a runtime
//! pack without an app release.

fn main() {
    chaty_lib::inference::llama_sidecar::serve()
}
