<div align="center">

<img src="icon.png" width="100" height="100" alt="Chaty" />

# Chaty

**A local, private desktop chat app for GGUF models.**

Load a `.gguf` and talk to it — everything runs on your machine, nothing leaves it.

**Windows (Vulkan) · macOS Apple Silicon (Metal)**

Tauri 2 · React + TypeScript · Rust · llama.cpp · [**Releases**](../../releases)

</div>

---

## Features

**Inference**
- Run any `.gguf` locally — tokenizer & chat template come straight from the file.
- **GPU acceleration** — cross‑vendor **Vulkan** on Windows, **Metal** on Apple Silicon (offload‑all on unified memory), auto‑tuned with graceful CPU fallback.
- First‑class support for **Llama 3**, **Gemma 3 / 4**, **Qwen 3 / 3.5 / 3.6** — incl. Gemma 4's `<|turn>`/`<|channel>` format and Qwen 3.5+ thinking control.
- Fast multi‑turn via a persistent context with KV‑cache reuse.
- **Adjustable context window** — auto‑fits the model's trained length to your memory (KV‑cache aware), with auto‑summary of older turns near the limit.
- **Safe model switching** — the old model is fully ejected (and verified released) before the next loads; live load progress bar.
- **Model hot‑swap** from a `models/` folder; the last model auto‑loads on launch.
- **In‑app model downloader** — pull GGUF files straight from a HuggingFace repo with live progress.
- Full **sampling controls** (Top‑K, Min‑P, repeat penalty, stop sequences) and saveable **prompt presets**.

**Chat UI**
- Streaming, foldable `<think>` panel with **focus‑follows‑generation** view + thinking toggle (incl. Qwen3.5+/Gemma 4), KaTeX, tables, **Mermaid** diagrams, per‑block code copy.
- **In‑app HTML preview** (zoomable, **playable** — single‑file web games work) and a **`/webdesign`** mode for polished single‑file UIs.
- **Drag‑and‑drop** attachments, conversation **export** (Markdown/JSON) & **full‑text search**.
- **Light / dark / system** themes; compact **Tools** menu, plus **Model info** & **Hardware** panels.

**Voice** — English, runs on CPU so it never touches the LLM's VRAM
- Voice in/out with silence auto‑send and sentence‑by‑sentence read‑aloud; **11 voices** + speed control. (Native CoreAudio capture on macOS.)
- **Live mode** — hands‑free continuous conversation with an animated orb.

**Local knowledge base (RAG)** — fully offline, high‑precision hybrid retrieval
- Index **PDF / text / Markdown / code / images** (with OCR) into a private, local store.
- **Hybrid retrieval**: bge‑m3 multilingual vectors + BM25 keywords, RRF fusion, MMR de‑duplication and neighbor‑chunk expansion — all on‑device.
- **Strict grounding** — answers come only from your documents; the model says "not covered by the current documents" instead of guessing.
- **Inline citations** — numbered superscripts with hover‑preview of the cited passage; per‑document **query scope** (pick which files to search).
- **Deep‑dive podcast** — turn your knowledge base into a NotebookLM‑style two‑host **English** audio show (alternating Kokoro voices), with progress + ETA, cancel anytime, and **WAV export**.

**More**
- Web search + URL fetch (with inline citations) · PDF / code / image attachments · Latin OCR.
- **Cancelable downloads** everywhere — model recommender, HuggingFace repos, and the embedding model.
- SQLite history with branching · system tray · global hotkey · EN / 简体中文.

> Offline‑first. Network is used only for optional web search and one‑time model downloads (voice ~0.5 GB; the bge‑m3 embedding model ~0.7 GB, only if you use the knowledge base).

## Install

From the [**Releases**](../../releases) page:

| Platform | File | Notes |
|---|---|---|
| Windows x64 | `Chaty_x.y.z_x64-setup.exe` | Per‑user installer, no admin |
| macOS (Apple Silicon) | `Chaty_x.y.z_aarch64.dmg` | See first‑launch note below |

### Running on macOS the first time

Chaty is **ad‑hoc signed but not notarized** (there's no paid Apple Developer account behind it), so macOS Gatekeeper will warn that it "cannot be verified" or is "damaged" on first launch. The app is safe — everything runs locally. Clear the download quarantine once, in Terminal:

```sh
xattr -dr com.apple.quarantine /Applications/Chaty.app
```

then open Chaty normally. (Alternatively: double‑click it, dismiss the warning, then go to **System Settings → Privacy & Security → Open Anyway**.)

Voice models download on first use. On macOS the writable models folder is in app data — use **Open models folder** in the model menu.

## Build

See **[BUILD.md](BUILD.md)**.

```powershell
# Windows
npm install
.\dev.ps1                            # dev
npm run tauri build -- --no-bundle   # release exe → then compile the Inno installer
```

```bash
# macOS (Apple Silicon)
npm install
npm run tauri dev     # dev (Metal)
npm run tauri build   # → .app + .dmg
```

Releases are produced by CI: bump with `scripts/bump-version.sh x.y.z`, commit, push a `vx.y.z` tag — GitHub Actions builds both installers onto one release.

## Stack

| Layer | |
|---|---|
| Shell | Tauri 2 — tray, global shortcut, single‑instance |
| Frontend | React 19 · Vite · react‑markdown · KaTeX |
| Inference | Rust · `llama-cpp-2` (llama.cpp) — Vulkan (Windows) / Metal (macOS) |
| Voice | `sherpa-rs` (ONNX Runtime, CPU) — Whisper‑base.en + Kokoro‑82M; cpal capture on macOS |
| Knowledge base | bge‑m3 embeddings (llama.cpp) + BM25, hybrid RRF/MMR retrieval · SQLite vector store |
| Storage | SQLite |

## License

MIT — see [LICENSE](LICENSE).
