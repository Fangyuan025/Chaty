# Building Chaty

Tauri 2 + React-TS frontend, Rust backend with **llama.cpp** (`llama-cpp-2`) for
local GGUF inference.

## Prerequisites (Windows)

- **Rust** (MSVC toolchain) + **Node 18+**
- **Visual Studio Build Tools** with the *C++* and *C++ CMake tools* components
  (provides `cl`, `cmake`, the Windows SDK)
- **libclang** for `bindgen` (used by `llama-cpp-sys-2`). Any LLVM/libclang works;
  set `LIBCLANG_PATH` to the folder containing `libclang.dll`.
- **Vulkan SDK** (LunarG) for the GPU build — provides `glslc` + `vulkan-1.lib`.
  Install from <https://vulkan.lunarg.com/sdk/home> (needs admin). `dev.ps1`
  auto‑detects it under `C:\VulkanSDK`. For a **CPU‑only** build that doesn't
  need the Vulkan SDK, pass `--no-default-features` to cargo / tauri.

`cl`, `cmake`, `libclang`, and the Vulkan SDK usually aren't on `PATH` —
`dev.ps1` wires them up.

## Run (dev)

```powershell
npm install
.\dev.ps1          # configures the build env, then `npm run tauri dev`
```

First run compiles llama.cpp from source (~3–4 min); afterwards it's cached.

## Installer (Inno Setup)

The release installer is a modern **Inno Setup** wizard (per-user, no admin;
bundles the voice DLLs and auto-installs the WebView2 runtime if it's missing).
Build it with the **Inno Setup Compiler** (`ISCC.exe`):

```powershell
.\dev.ps1                              # (env only), then:
npm run tauri build -- --no-bundle     # builds chaty.exe + DLLs into the target dir
ISCC /DAppVersion=0.3.1 /DSrcDir=C:\ct\release src-tauri\installer\Chaty.iss
# → <SrcDir>\bundle\inno\Chaty_<ver>_x64-setup.exe
```

The script is `src-tauri/installer/Chaty.iss`. (Tauri's built-in NSIS target is
no longer used.)

## Build (macOS / Apple Silicon)

No Vulkan SDK, WebView2, or `MAX_PATH` workarounds — the Metal build of
llama.cpp is self-contained and macOS uses the system WKWebView.

```bash
xcode-select --install            # clang, metal toolchain, codesign
brew install cmake ninja
rustup target add aarch64-apple-darwin   # native arm64, NOT x86_64/Rosetta
npm install
npm run tauri dev                 # Metal build of llama.cpp (~a few min first time)
npm run tauri build               # → src-tauri/target/release/bundle/dmg/*.dmg
```

The GPU backend is selected **per target** via the `gpu-backend` shim crate
(`src-tauri/gpu-backend/`): the `gpu` feature activates the shim, whose
target-specific dependency tables add `metal` (macOS) or `vulkan`
(Windows/Linux) to `llama-cpp-2`. `--no-default-features` still gives a
pure-CPU build on either OS. Platform-specific Tauri config lives in
`tauri.macos.conf.json` / `tauri.windows.conf.json`, auto-merged over
`tauri.conf.json`.

⚠️ Do **not** put `rustflags = ["-C", "target-cpu=apple-m1"]` in a
`.cargo/config.toml` here: the `cc`/`cmake` build scripts translate it into
`-march=apple-m1` for Apple clang, which is rejected and breaks the llama.cpp
build. The Metal kernels do the heavy lifting anyway.

⚠️ **Voice dylibs (one-time, needs the Mac toolchain):** on Windows the
sherpa-onnx / ONNX Runtime libs are checked in under `src-tauri/libs/*.dll` and
bundled as resources. On macOS `sherpa-rs` links the `.dylib` equivalents from
its build output; `build.rs` adds the `@executable_path` / `../Frameworks`
rpaths so they resolve from inside the `.app`. After the first `tauri build`,
confirm `libonnxruntime*.dylib` and `libsherpa-onnx-*.dylib` are present next to
the binary or in `Contents/Frameworks` of the `.app`; if not, add them to
`bundle.macOS.frameworks` in `tauri.macos.conf.json` (or copy them in a bundling
hook) and rebuild.

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

- **GPU offload** uses the cross‑vendor **Vulkan** backend (NVIDIA / AMD / Intel).
  `n_gpu_layers` is **auto‑tuned** from detected VRAM (DXGI); the loader backs off
  to fewer layers if a GPU allocation fails, and falls back to CPU when there's no
  usable GPU. Override it in Settings → GPU acceleration. The hardware panel
  (top‑right) shows CPU/RAM/GPU and the current model's offload.
- The chat prompt uses the GGUF's **embedded chat template**, so a single `.gguf`
  file is all that's needed — no separate tokenizer.
