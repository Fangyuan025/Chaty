<div align="center">

<img src="icon.png" width="100" height="100" alt="Chaty" />

# Chaty

**A local, private desktop chat app for GGUF models.**

Load a `.gguf` and talk to it — everything runs on your machine, nothing leaves it.

Tauri 2 · React + TypeScript · Rust · llama.cpp · [**Releases**](../../releases)

</div>

---

## Features

**Inference**
- Run any `.gguf` locally — tokenizer & chat template come straight from the file.
- **GPU acceleration** — cross‑vendor **Vulkan**, auto‑tuned to your VRAM, graceful CPU fallback.
- Fast multi‑turn via a persistent context with KV‑cache reuse.
- **Adjustable context window** (up to the model's native length) with auto‑summary of older turns near the limit.
- **Model hot‑swap** from a `models/` folder; the last model auto‑loads on launch.
- **In‑app model downloader** — pull GGUF files straight from a HuggingFace repo with live progress.
- Full **sampling controls** (Top‑K, Min‑P, repeat penalty, stop sequences) and saveable **prompt presets**.

**Chat UI**
- Streaming, foldable `<think>` panel + thinking toggle (incl. Qwen3.5+), KaTeX, tables, **Mermaid** diagrams, per‑block code copy.
- **In‑app HTML preview** (zoomable) and a **`/webdesign`** mode for polished single‑file UIs.
- **Drag‑and‑drop** attachments, conversation **export** (Markdown/JSON) & **full‑text search**.
- **Light / dark / system** themes; compact **Tools** menu, plus **Model info** & **Hardware** panels.

**Voice** — English, runs on CPU so it never touches the LLM's VRAM
- Voice in/out with silence auto‑send and sentence‑by‑sentence read‑aloud; **11 voices** + speed control.
- **Live mode** — hands‑free continuous conversation with an animated orb.

**More**
- Web search + URL fetch · PDF / code attachments · Latin OCR.
- SQLite history with branching · system tray · global hotkey · EN / 简体中文.

> Offline‑first. Network is used only for optional web search and a one‑time voice‑model download (~0.5 GB).

## Install (Windows x64)

Download `Chaty_x.y.z_x64-setup.exe` from the [**Releases**](../../releases) page and run it — per‑user, no admin. Voice models download on first use.

## Build

See **[BUILD.md](BUILD.md)**. In short, on Windows:

```powershell
npm install
.\dev.ps1                            # dev
npm run tauri build -- --no-bundle   # release exe → then compile the Inno installer
```

## Stack

| Layer | |
|---|---|
| Shell | Tauri 2 — tray, global shortcut, single‑instance |
| Frontend | React 19 · Vite · react‑markdown · KaTeX |
| Inference | Rust · `llama-cpp-2` (llama.cpp) with Vulkan GPU offload |
| Voice | `sherpa-rs` (ONNX Runtime, CPU) — Whisper‑base.en + Kokoro‑82M |
| Storage | SQLite |

## License

MIT — see [LICENSE](LICENSE).
