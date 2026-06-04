<div align="center">

<img src="icon.png" width="104" height="104" alt="Chaty" />

# Chaty

**A local, private desktop chat app for GGUF models — with the polish of a closed‑source app.**

Load a `.gguf` file and talk to it. Everything runs on your machine; nothing leaves it.

Tauri 2 · React + TypeScript · Rust · llama.cpp

</div>

---

## Features

- **Local GGUF inference** — point it at any `.gguf` and chat. Tokenizer and chat template come straight from the file, so a single file just works. Powered by `llama-cpp-2` (llama.cpp).
- **GPU acceleration (auto‑tuned)** — cross‑vendor **Vulkan** offload (NVIDIA / AMD / Intel). It detects your VRAM and automatically offloads as many layers as fit, backing off gracefully and falling back to CPU when there's no GPU. A **hardware panel** (top‑right) shows your CPU / RAM / GPU and the current model's offload.
- **Fast multi‑turn** — a persistent context per model with KV‑cache prefix reuse, so a long conversation doesn't re‑process its whole history every turn.
- **Model hot‑swap** — drop `.gguf` files into the install's `models/` folder and switch between them from the title bar; the last model auto‑loads on launch.
- **Polished chat UI** — neutral ChatGPT‑style design, streaming tokens, a foldable `<think>` reasoning panel, a thinking‑mode toggle, KaTeX math, GFM tables, syntax‑highlighted code blocks with per‑block copy, and an **in‑app HTML preview** for HTML the model writes.
- **Voice (English)** — speak to it and hear it back, running entirely on **CPU** (Whisper‑base.en + Kokoro‑82M via ONNX Runtime) so it never touches the LLM's VRAM:
  - Voice input with automatic **silence detection** (speak, pause, it sends).
  - **Streaming read‑aloud** — replies are synthesized and played sentence‑by‑sentence as they generate.
  - **Live mode** — a hands‑free, Gemini‑style continuous conversation with an animated orb that reacts to the audio.
- **Web search & fetch** — optional DuckDuckGo search with page‑body extraction and context‑aware query rewriting; paste a URL to have it read the page.
- **Attachments & OCR** — attach `txt/md/pdf/code` (text extracted and used as context) or images (Latin OCR).
- **Native shell** — frameless custom title bar, system tray, global hotkey (`Ctrl+Shift+Space`), single‑instance, custom right‑click menu.
- **Conversations** — SQLite‑backed history, branching from any message, model‑generated titles.
- **Bilingual UI** — English / 简体中文, persisted.

> Everything is offline‑first. The only network use is the optional web search and the one‑time download of the voice models.

## Install (Windows)

Grab `Chaty_x.y.z_x64-setup.exe` from the [**Releases**](../../releases) page and run it. It installs per‑user (no admin needed). The voice module is bundled; the voice **models** (~0.5 GB) download automatically the first time you use a voice feature.

## Build from source

See **[BUILD.md](BUILD.md)**. In short, on Windows:

```powershell
npm install
.\dev.ps1     # wires up MSVC + libclang, then runs `npm run tauri dev`
```

To produce the installer:

```powershell
.\dev.ps1   # (env only); then:
npm run tauri build
```

## Tech stack

| Layer | What |
|-------|------|
| Shell | Tauri 2.x, system tray, global shortcut, single‑instance |
| Frontend | React 19, TypeScript, Vite 7, react‑markdown + KaTeX + highlight.js |
| Inference | Rust + `llama-cpp-2` (llama.cpp), persistent‑context actor with KV reuse |
| Voice | `sherpa-rs` (sherpa‑onnx / ONNX Runtime, CPU) — Whisper‑base.en STT + Kokoro‑82M TTS |
| Storage | SQLite (`rusqlite`) for conversations |
| Web/Docs | `reqwest` + `scraper`, `pdf-extract`, `ocrs` (Latin OCR) |

## Notes

- GPU offload uses **Vulkan** and is auto‑tuned from your VRAM; set it to Auto / Off / a manual layer count in Settings. With no usable GPU it runs on CPU.
- Voice is **English‑only** (the base.en / Kokoro‑en models). Voice controls are hidden when the UI language is set to Chinese.

## License

MIT — see [LICENSE](LICENSE).
