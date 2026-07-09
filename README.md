<div align="center">

**English** · [简体中文](README.zh-CN.md)

<img src="icon.png" width="92" height="92" alt="Chaty" />

# Chaty

### Private, on-device AI — your models, your data, your machine.

Chaty is a polished desktop app that runs open LLMs **100% offline**.
No account, no cloud, no telemetry — with a document knowledge base, multi-round
Deep Research, and hands-free voice built right in.

[![Windows · Vulkan](https://img.shields.io/badge/Windows-Vulkan-0078D6?logo=windows&logoColor=white)](../../releases)
[![macOS · Metal](https://img.shields.io/badge/macOS-Apple_Silicon_·_Metal-000000?logo=apple&logoColor=white)](../../releases)
[![100% offline](https://img.shields.io/badge/100%25-offline-19c37d)](https://chaty.ca)
[![Rust + Tauri 2](https://img.shields.io/badge/Rust_+_Tauri_2-CE412B?logo=rust&logoColor=white)](#architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-444)](LICENSE)

[**↓ Download**](../../releases) · [**Website**](https://chaty.ca) · [**Chaty model on Hugging Face**](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF)

</div>

---

## Why Chaty

- 🔒 **Truly private** — every model, document, and conversation stays on your device. No sign-up, no server, nothing phoned home.
- ⚡ **Native and fast** — a Rust + llama.cpp inference core with **Vulkan / Metal** GPU offload that auto-tunes to your hardware and falls back gracefully to CPU.
- 🧰 **More than a chat box** — a local knowledge base (RAG), agentic Deep Research, hands-free voice, and a self-healing **Design Canvas** — all offline.
- 🧠 **Runs almost anything** — Llama 3, Gemma 3 / 4, Qwen 3 / 3.5 / 3.6, or *any* GGUF from Hugging Face — plus **Chaty's own fine-tuned model**.
- 💻 **Friendly to modest hardware** — a first-launch *“Set up for me”* picks a model sized to your RAM, downloaded in one click.

## Screenshots

| | |
|---|---|
| <img src="docs/screenshots/shot-research.jpg" alt="Deep Research" /> | <img src="docs/screenshots/shot-live.jpg" alt="Live voice mode" /> |
| **Deep Research** — multi-round web search → a cited report. | **Live voice mode** — hands-free, continuous conversation. |
| <img src="docs/screenshots/shot-htmlpreview.jpg" alt="In-app HTML preview" /> | <img src="docs/screenshots/shot-podcast.jpg" alt="Deep-dive podcast" /> |
| **HTML preview** — render & play single-file web pages in-app. | **Deep-dive podcast** — turn documents into a two-host audio show. |

## Capabilities

### Local inference
- Run any `.gguf` — the tokenizer and chat template come straight from the file.
- **GPU acceleration**: cross-vendor **Vulkan** (Windows) and **Metal** (Apple Silicon, offload-all on unified memory), VRAM-aware auto-tuning with an OOM back-off and CPU fallback.
- First-class handling for **Llama 3**, **Gemma 3 / 4**, and **Qwen 3 / 3.5 / 3.6** — including Gemma 4's channel format and Qwen 3.5+ thinking control — with a robust template fallback for community models.
- **Adjustable context window** that auto-fits the model's trained length to your memory, summarizing older turns as you approach the limit.
- **Safe model switching** (the previous model is fully released first), hot-swap from a `models/` folder, and full sampling controls with saveable presets.
- **One-click model setup** — a hardware-fitted recommender and in-app Hugging Face downloader with live, cancelable progress.

### Code mode (agentic coding)
- A top-level **Chat | Code** switch turns Chaty into a **local coding agent**: point it at a folder, describe the task, and it explores, edits, and verifies the project by itself — file read/write, exact-string edits, glob/grep, shell commands (with **background jobs** for dev servers and long builds), and **web search** for unfamiliar errors — every step shown live.
- **Confined & sandboxed** — file access never leaves the workspace you picked, and on macOS shell commands run in a Seatbelt sandbox that can only write inside it.
- **You stay in control** — per-action approval with a real diff preview (or flip on **Bypass** for autonomy), a live task-plan checklist, and choice dialogs when a decision is yours to make.
- **Built for local models** — visible reasoning with an Off / Normal / Deep depth switch, a context-usage ring with automatic compaction, tolerant tool-call parsing so smaller models self-correct, plus persistent sessions, custom `/skills`, and slash commands.

### Knowledge base (RAG)
- Index **PDF, Word (.docx), Excel (.xlsx), Markdown, ~90 text/code/config formats, and images** (with OCR) into a private, on-device store — one file at a time or a **whole folder** (subdirectories included, with the project's structure preserved).
- **Hybrid retrieval**: bge-m3 multilingual vectors + BM25 keywords, fused with RRF, de-duplicated with MMR, and expanded with neighbor chunks.
- **Strict grounding** — answers come only from your documents; Chaty says when something isn't covered instead of guessing.
- **Per-file citations** with hover-preview of the source passage, and per-document query scope.
- **One-click report** — generate a cited, NotebookLM-style overview of your whole knowledge base (no topic needed), exportable to PDF or Markdown — fully offline.

### Deep Research & web
- Give a topic and Chaty plans queries, runs **multiple rounds** of web search interleaved with reasoning, and writes a structured, cited long-form report — **exportable to PDF or Markdown**.
- Topic-anchored and honest: the reference list contains only sources it actually cited.
- A free, key-less, multi-provider search chain (Brave → Bing → DuckDuckGo → Wikipedia) so a single blocked provider never breaks search.

### Voice & audio
- Hands-free **Live mode** — continuous conversation with an animated orb.
- Voice in/out with silence auto-send and read-aloud — **11 voices** with speed control.
- **Deep-dive podcast** — turn your knowledge base into a NotebookLM-style two-host audio show, with WAV export.
- All voice runs on the **CPU**, so it never competes with the LLM for VRAM.

### Design Canvas
- **Build a page by chatting** — open any single-file HTML Chaty generates into a split studio: a live preview on one side, an instruction box on the other. Ask for changes in plain language and Chaty edits the page **in place** (a fast search/replace patch, not a full re-render).
- **Self-healing** — the preview watches for runtime errors and offers a one-click **Fix** that hands the error to the model; every fix asks first, so there's no runaway loop.
- **Version history** with revert, plus export to a standalone `.html` or your browser. Pairs naturally with Chaty's own web-design fine-tune.

### Crafted chat experience
- A streaming, foldable `<think>` panel that follows the reasoning as it generates.
- KaTeX math, tables, **Mermaid** diagrams, per-block code copy, and in-app rendering of single-file HTML (including playable web games) through the Design Canvas.
- A **⌘K command palette**, pinnable / renameable conversations, in-app confirmations, and a crash-safe error boundary so an unexpected error never blanks the window.
- Drag-and-drop attachments, conversation export (Markdown / JSON), full-text search, branching history, light / dark / system themes, **reduced-motion** support, system tray, a global hotkey, and an **English / 简体中文** UI.

> **Offline-first.** The network is used only for optional web search and one-time model downloads.

## Chaty's own model

Beyond third-party models, Chaty ships its **own fine-tune** — a Qwen3.5-4B distilled
from a much larger teacher and tuned for leaner, on-device single-file web design,
with a baked-in Chaty identity and grounded citations. It's a one-click pick in
*“Set up for me”* and fully open on
**[Hugging Face](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF)**.

## Install

Grab the latest build from the [**Releases**](../../releases) page:

| Platform | File | Notes |
|---|---|---|
| Windows x64 | `Chaty_*_x64-setup.exe` | Per-user installer — no admin required |
| macOS (Apple Silicon) | `Chaty_*_aarch64.dmg` | See the first-launch note below |

**macOS first launch.** Chaty is ad-hoc signed but not notarized (there's no paid
Apple Developer account behind it), so Gatekeeper warns on first open. The app is
safe — everything runs locally. Clear the download quarantine once:

```sh
xattr -dr com.apple.quarantine /Applications/Chaty.app
```

then open Chaty normally. (Or: open it, dismiss the warning, and choose
**System Settings → Privacy & Security → Open Anyway**.) On macOS the writable
models folder lives in app data — use **Open models folder** in the model menu.

## Build

Full details in **[BUILD.md](BUILD.md)**.

```powershell
# Windows
npm install
.\dev.ps1                            # dev
npm run tauri build -- --no-bundle   # release exe → compile the Inno installer
```

```bash
# macOS (Apple Silicon)
npm install
npm run tauri dev      # dev (Metal)
npm run tauri build    # → .app + .dmg
```

Releases are produced by CI: bump with `scripts/bump-version.sh x.y.z`, then push a
`vx.y.z` tag — GitHub Actions builds both installers onto a single release.

## Architecture

| Layer | Stack |
|---|---|
| Shell | Tauri 2 — system tray, global shortcut, single-instance |
| Frontend | React 19 · Vite · react-markdown · KaTeX |
| Inference | Rust · `llama-cpp-2` (llama.cpp) — Vulkan (Windows) / Metal (macOS) |
| Voice | `sherpa-rs` (ONNX Runtime, CPU) — Whisper-base.en + Kokoro-82M |
| Knowledge base | bge-m3 embeddings + BM25 · hybrid RRF / MMR retrieval · SQLite vector store |
| Storage | SQLite — conversations, messages, full-text search |

## License

MIT — see [LICENSE](LICENSE). Built with [llama.cpp](https://github.com/ggml-org/llama.cpp), [Tauri](https://tauri.app), and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
