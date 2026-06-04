# Building Chaty

Tauri 2 + React-TS frontend, Rust backend with **llama.cpp** (`llama-cpp-2`) for
local GGUF inference.

## Prerequisites (Windows)

- **Rust** (MSVC toolchain) + **Node 18+**
- **Visual Studio Build Tools** with the *C++* and *C++ CMake tools* components
  (provides `cl`, `cmake`, the Windows SDK)
- **libclang** for `bindgen` (used by `llama-cpp-sys-2`). Any LLVM/libclang works;
  set `LIBCLANG_PATH` to the folder containing `libclang.dll`.

`cl`, `cmake`, and `libclang` usually aren't on `PATH` — `dev.ps1` wires them up.

## Run (dev)

```powershell
npm install
.\dev.ps1          # configures the build env, then `npm run tauri dev`
```

First run compiles llama.cpp from source (~3–4 min); afterwards it's cached.

## Headless engine smoke test

Verifies real inference without the GUI (load GGUF → chat template → stream):

```powershell
# inside a VS dev shell, or after dev.ps1's env setup:
cargo run --example smoke --manifest-path src-tauri/Cargo.toml -- "<path\to\model.gguf>" "你好"
```

## Project layout

```
src/                     React UI (chat, streaming, model picker)
  lib/ipc.ts             typed bridge to Rust (invoke + Channel)
src-tauri/src/
  inference/             InferenceBackend trait + types
    llama.rs             real llama.cpp engine (GGUF load, decode loop)
    mock.rs              fake streaming engine (test double)
  commands.rs            load_model / get_model / generate
  state.rs               shared app state
  examples/smoke.rs      headless inference test
```

## Notes

- CPU build for now (`n_gpu_layers = 0`). GPU offload + hardware auto-tuning are planned.
- The chat prompt uses the GGUF's **embedded chat template**, so a single `.gguf`
  file is all that's needed — no separate tokenizer.
