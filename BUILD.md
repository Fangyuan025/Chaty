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
