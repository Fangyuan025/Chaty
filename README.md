<div align="center">

**English** · [简体中文](README.zh-CN.md)

<img src="icon.png" width="88" height="88" alt="Chaty" />

# Chaty

### Private, on-device AI — your models, your data, your machine.

Chaty runs open LLMs **100% offline** in a polished desktop app.
No account, no cloud, no telemetry — with a local coding agent, a document
knowledge base, Deep Research, and hands-free voice built right in.

[![Windows · Vulkan](https://img.shields.io/badge/Windows-Vulkan-0078D6?logo=windows&logoColor=white)](../../releases)
[![macOS · Metal](https://img.shields.io/badge/macOS-Apple_Silicon_·_Metal-000000?logo=apple&logoColor=white)](../../releases)
[![100% offline](https://img.shields.io/badge/100%25-offline-19c37d)](https://chaty.ca)
[![Rust + Tauri 2](https://img.shields.io/badge/Rust_+_Tauri_2-CE412B?logo=rust&logoColor=white)](#architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-444)](LICENSE)

[**↓ Download**](../../releases) · [**Website**](https://chaty.ca) · [**Chaty model on Hugging Face**](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF)

<br />

<img src="docs/screenshots/shot-code.jpg" width="860" alt="Chaty's local coding agent searching GitHub, editing files, and running tests" />

<sub>A local coding agent — searches GitHub, reads the source, edits your files, and runs the tests. **All on your machine.**</sub>

</div>

---

## Why Chaty

- 🔒 **Truly private** — every model, document, and conversation stays on your device. No sign-up, no server, nothing phoned home.
- ⚡ **Native and fast** — a Rust + llama.cpp core with **Vulkan / Metal** GPU offload that auto-tunes to your hardware and falls back gracefully to CPU.
- 🧰 **More than a chat box** — a coding agent, a knowledge base (RAG), Deep Research, hands-free voice, and a self-healing Design Canvas — all offline.
- 🧠 **Runs almost anything** — Llama 3, Gemma 3 / 4, Qwen 3 / 3.5 / 3.6, or *any* GGUF from Hugging Face — plus **Chaty's own fine-tuned model**.
- 💻 **Friendly to modest hardware** — a first-launch *“Set up for me”* picks a model sized to your RAM and downloads it in one click.

<br />

## A local coding agent

Flip the **Chat · Code** switch and Chaty becomes an agent for your codebase. Point it
at a folder, describe the task, and it explores, edits, and verifies the project by
itself — every step shown live, every change behind an approval + diff.

- 🌐 **The whole web as a tool** — key-less search of **GitHub** (repos, issues, *and code*), Reddit, YouTube, Bilibili, and any domain; fetching adapts to the content (articles → Markdown, GitHub pages → raw source, PDFs → text, videos → transcripts).
- 🧭 **Drives a real browser** — with a vision model, the agent opens pages, clicks by visible text, fills forms, scrolls, reads the console, and takes **screenshots it actually looks at** — watch it work in a real Chrome, logins and all.
- ✏️ **Precise edits** — exact-string patches with a diff preview and “did-you-mean” hints, file outlines to navigate big files, and `search_files` to find by name or content.
- 🖥️ **Real shell** — run commands and long **background jobs** (dev servers, builds), sandboxed to the workspace (Seatbelt on macOS).
- ⏪ **You stay in control** — per-action approval, a command allowlist, a live task-plan checklist, and **one-click checkpoint rewind** that restores files *and* rolls back the conversation.

<details>
<summary>More Code-mode details</summary>

- Built for local models: an **Off / Normal / Deep** reasoning switch, a **prompt-processing progress ring**, a context-usage ring with automatic compaction, whole-file reads sized to your context window, ranked `search_code` + knowledge-base `search_docs`, and loop-breaking for repetitive small models.
- Persistent sessions, project memory (**AGENTS.md**), custom **/skills**, and slash commands.
- Tune it under **Settings → Code**: step limit, command timeout, step temperature, an auto-approve-edits toggle, a headless-browser toggle, and a command allowlist.
- File access never leaves the folder you pick; downloads land in the workspace and are covered by checkpoints too.

</details>

<br />

## Chat that renders everything

<table>
<tr>
<td width="50%"><img src="docs/screenshots/shot-chat.jpg" alt="Rich chat rendering — syntax-highlighted code, tables, and KaTeX math" /></td>
<td width="50%"><img src="docs/screenshots/shot-chat-light.jpg" alt="The same conversation in Chaty's light theme" /></td>
</tr>
</table>

- A streaming, foldable **`<think>`** panel that follows the model's reasoning as it generates.
- **KaTeX** math, tables, **Mermaid** diagrams, per-block code copy, and in-app rendering of single-file HTML — including playable web games.
- A **⌘K command palette**, pinnable / renameable conversations, drag-and-drop attachments, export (Markdown / JSON), and full-text search.
- Four palettes (two dark, two light) with system-theme following, native UI zoom, reduced-motion support, and an **English / 简体中文** UI.

<br />

## Chaty can see

Load a **vision model** (its weights and `mmproj` encoder live together in one folder, paired automatically) and image understanding turns on everywhere:

- **Chat** — attach a picture and ask about it; follow-ups stay fast (already-seen images aren't re-encoded).
- **Code** — the agent reads screenshots and can look at any image with `view_image`; the composer takes images and documents just like chat.
- **Knowledge base** — imported images get a written description beside their OCR text, so search finds what's *in* them.
- **Canvas** — the model sees the live rendered page when you ask for an edit.

Text-only models keep the OCR path, so nothing regresses — and updating from an older version, a one-time prompt tidies your existing loose `.gguf` files into the one-folder-per-model layout with a single click.

<br />

## A private knowledge base

<table>
<tr>
<td width="52%">

- Index **PDF, Word, Excel, Markdown, ~90 text/code formats, and images** into an on-device store — one file or a whole folder. Images are read by **OCR *and*, with a vision model, described in words** so you can search what's *in* the picture.
- **Hybrid retrieval**: bge-m3 vectors + BM25 keywords, fused with RRF, de-duplicated with MMR, expanded with neighbors.
- **Strict grounding** — answers come only from your files, with **per-file citations** and hover-preview of the source passage. Chaty says when something isn't covered instead of guessing.
- **One-click report** — a cited, NotebookLM-style overview of the whole base, exportable to PDF or Markdown.

</td>
<td width="48%"><img src="docs/screenshots/shot-knowledge.jpg" alt="Local knowledge base — indexed documents with per-file toggles and one-click report / podcast" /></td>
</tr>
</table>

<br />

## Deep Research & the web

- Give a topic and Chaty plans queries, runs **multiple rounds** of web search interleaved with reasoning, and writes a structured, cited report — **exportable to PDF or Markdown**.
- Honest by design: the reference list contains only sources it actually cited.
- A free, key-less, multi-provider search chain (Brave → Bing → DuckDuckGo → Wikipedia) so one blocked provider never breaks search.

<br />

## Hands-free voice

<table>
<tr>
<td width="48%"><img src="docs/screenshots/shot-live.jpg" alt="Live voice mode — an animated orb for continuous, hands-free conversation" /></td>
<td width="52%">

- **Live mode** — continuous, hands-free conversation with an animated orb.
- Voice in/out with silence auto-send and read-aloud — **11 voices** with speed control.
- **Deep-dive podcast** — turn your knowledge base into a NotebookLM-style two-host audio show, with WAV export.
- All voice runs on the **CPU**, so it never competes with the LLM for VRAM.

</td>
</tr>
</table>

<br />

## Everything stays on your machine

<table>
<tr>
<td width="52%">

- Conversations, models, and indexes live in one **local data folder** — copy it to back up, clear it in a click.
- **GPU acceleration**: cross-vendor **Vulkan** (Windows) and **Metal** (Apple Silicon, offload-all on unified memory), VRAM-aware auto-tuning with OOM back-off and CPU fallback.
- **Any `.gguf`** — the tokenizer and chat template come from the file; first-class handling for Llama 3, Gemma 3 / 4, and Qwen 3 / 3.5 / 3.6.
- **Adjustable context** that auto-fits the model's trained length to your memory and summarizes older turns near the limit; **safe model switching** and full sampling controls with saveable presets.

</td>
<td width="48%"><img src="docs/screenshots/shot-settings.jpg" alt="Settings — a local data dashboard showing conversations, models, and knowledge-base stats" /></td>
</tr>
</table>

> **Offline-first.** The network is used only for optional web search and one-time model downloads.

<br />

## Design Canvas

- **Build a page by chatting** — open any single-file HTML Chaty generates into a split studio: live preview on one side, an instruction box on the other. Ask for changes in plain language and Chaty edits the page **in place** (a fast search/replace patch, not a full re-render).
- **Self-healing** — the preview watches for runtime errors and offers a one-click **Fix**; every fix asks first, so there's no runaway loop. With a vision model, Chaty also *sees* the rendered page (a live screenshot + console) when you ask for a change.
- **Version history** with revert, plus export to a standalone `.html`. Pairs naturally with Chaty's own web-design fine-tune.

<br />

## Chaty's own model

Beyond third-party models, Chaty ships its **own fine-tune** — a Qwen3.5-4B distilled from
a much larger teacher and tuned for leaner, on-device single-file web design, with a baked-in
Chaty identity and grounded citations. It's a one-click pick in *“Set up for me”* and fully
open on **[Hugging Face](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF)**.

<br />

## Install

Grab the latest build from the [**Releases**](../../releases) page:

| Platform | File | Notes |
|---|---|---|
| Windows x64 | `Chaty_*_x64-setup.exe` | Per-user installer — no admin required |
| macOS (Apple Silicon) | `Chaty_*_aarch64.dmg` | See the first-launch note below |

**macOS first launch.** Chaty is ad-hoc signed but not notarized (there's no paid Apple
Developer account behind it), so Gatekeeper warns on first open. The app is safe — everything
runs locally. Clear the download quarantine once:

```sh
xattr -dr com.apple.quarantine /Applications/Chaty.app
```

then open Chaty normally. (Or: open it, dismiss the warning, and choose **System Settings →
Privacy & Security → Open Anyway**.) On macOS the writable models folder lives in app data —
use **Open models folder** in the model menu.

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

Releases are produced by CI: bump with `scripts/bump-version.sh x.y.z`, then push a `vx.y.z`
tag — GitHub Actions builds both installers onto a single release.

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
