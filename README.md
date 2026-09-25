<div align="center">

**English** · [简体中文](README.zh-CN.md) · [Português (BR)](README.pt-BR.md)

<img src="icon.png" width="84" height="84" alt="Chaty" />

# Chaty

**The models on your disk, put to work.**

Chat, a coding agent, an image studio, answers from your own documents, and a voice to talk to —<br />
on open models running entirely on your own Mac or PC. No account, no cloud, no telemetry.

[![Release](https://img.shields.io/github/v/release/Fangyuan025/Chaty?label=release&color=3a3a3a)](../../releases/latest)
[![Downloads](https://img.shields.io/github/downloads/Fangyuan025/Chaty/total?color=3a3a3a&cacheSeconds=3600)](../../releases)
[![CI](https://img.shields.io/github/actions/workflow/status/Fangyuan025/Chaty/ci.yml?branch=main&label=CI)](../../actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-3a3a3a)](LICENSE)

[**Download**](../../releases/latest) · [**Website**](https://chaty.ca) · [**Docs**](https://chaty.ca/docs.html) · [**Changelog**](CHANGELOG.md)

<sub>macOS (Apple Silicon) · Windows 10/11 · Linux (AppImage) — GGUF on llama.cpp, MLX natively on Apple Silicon, text-to-image on stable-diffusion.cpp</sub>

<br />

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/screenshots/shot-code-light.jpg" />
  <img src="docs/screenshots/shot-code.jpg" width="900" alt="Chaty's Code mode: a task plan, tool steps, a real diff and the files the turn changed" />
</picture>

<sub>Code mode: the agent found the upstream fix, patched the parser, added a test and ran the suite — on a model running on the same laptop.</sub>

</div>

---

## Contents

[Chat](#chat) · [Code](#code) · [Image](#image) · [Documents and research](#documents-and-research) · [Voice](#voice) · [Built for small models](#built-around-what-small-models-get-wrong) · [Models](#models) · [Privacy](#privacy) · [Install](#install) · [Build](#build) · [Architecture](#architecture)

## Chat

A chat that shows its work. Reasoning streams into a panel you can fold away; on models with an effort ladder — Qwen3.8's low · medium · xhigh, Muse-Glimmer's four rungs, K2 Horizon's — you choose how hard it thinks, message by message.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/screenshots/shot-chat-light.jpg" />
  <img src="docs/screenshots/shot-chat.jpg" width="860" alt="A chat answer with highlighted Rust code, a table and a KaTeX formula" />
</picture>

- **Rendered as it streams** — highlighted code, tables, KaTeX, Mermaid, and HTML you can run in place. Blocks render one by one, so a fast model doesn't make the window stutter.
- **What's on, at a glance** — a quiet line above the composer names what the next message will use (thinking, web search, the knowledge base), each one click from off.
- **Pictures in** — any vision model: a GGUF with its projector, or an MLX build.
- **Canvas** — ask for a page and watch it build in a live preview beside its source; later changes land as patches.
- **Web search without a key**, full-text search across your history, and export to Markdown or JSON.
- **Your way** — warm or cool dark, paper or cream light, four code themes, English · 简体中文 · Português.

## Code

A coding agent for the model you can actually run. Switch to **Code**, open a folder, describe the task. The agent reads, searches, edits and runs commands in your project — each step shown as it happens, each edit as a real diff — and ends the turn with a card listing every file it changed, each one undoable.

- **You stay in charge** — a live task plan; edits and commands wait for approval unless you allow them. File access is confined to the workspace; on macOS its shell runs in the system sandbox.
- **Commands that keep running** — dev servers and watchers move to the background; a command that asks a question (`[y/N]`, `Password:`, a REPL, a scaffolder's menu) gets a terminal to answer in.
- **It remembers** — project memory in plain Markdown; it can search its own past sessions, and you can @-mention one to bring it in.
- **Tools you add** — MCP servers (with a curated, live-tested list), skills written as Markdown, and a browser it can drive.
- **Nothing done on red** — a turn ends at a gate: whatever changed since the last passing run is verified first.

**Measured.** One local model for every row — Qwen3.5-35B-A3B (MoE, ~3 B active), mxfp8 on MLX, reasoning off, one machine:

| SWE-bench Verified, 45-task macOS-validated subset | Resolved |
| --- | --- |
| **Chaty agent** (v1.9, 16K context) | **15/45 (33 %)** |
| qwen-code 0.20 — the model family's own CLI (needs 32K) | 12/45 (27 %) |
| pi 0.81 — minimal 4-tool agent | 10/45 (22 %) |
| opencode 1.18 | 7/45 (16 %) |
| bare bash agent — single-tool ablation | 6/45 (13 %) |

A subset on a macOS harness, so not comparable with leaderboard numbers; the method, configs and caveats are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Image

Load a text-to-image model and the whole app becomes an image studio: sessions that read like conversations, a live preview while it draws, and each model's recommended settings filled in.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/plate-diner.jpg" alt="A neon sign reading LATE NIGHT DINER on a rainy street corner" /></td>
<td width="50%"><img src="docs/screenshots/plate-shokudo.jpg" alt="A red neon sign reading 深夜食堂 on a rainy street at night" /></td>
</tr>
<tr>
<td><sub>“A neon sign that reads LATE NIGHT DINER on a rainy street corner…” — Z-Image Turbo, Q4_K_M, 1024², 8 steps, 2 min 51 s</sub></td>
<td><sub>“雨夜街角的霓虹灯招牌，写着「深夜食堂」…” — Qwen-Image 2.1, Q4_K_M, 1024², 20 steps, 16 min 10 s</sub></td>
</tr>
</table>

<sub>Straight out of Chaty, unedited, on an Apple M4 Pro with 48 GB.</sub>

- **The families that matter** — Z-Image and Z-Image Turbo, Qwen-Image 2.1, FLUX.1 dev and schnell, Chroma, Stable Diffusion 1.x, 2.x, XL and 3.x.
- **The other files, found** — a text-to-image GGUF is only the denoiser; Chaty finds its VAE and text encoder beside it, or downloads the missing ones in one click.
- **Keep going from a picture** — start the next round from any image; models that edit take it as a reference and change only what you ask.
- **Saved with its recipe** — PNGs carry the prompt and settings inside, filed by date.

It runs on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) in a helper process — Metal on a Mac, Vulkan on Windows and Linux — so a driver crash ends the helper, not the app.

## Documents and research

Drop in PDFs, Word files, slides, spreadsheets, Markdown or code. Chaty indexes them on your machine, retrieves by meaning and by keyword, and cites the passage behind each claim — or says plainly when your documents don't cover the question.

- **Scans, read page by page** by a vision model; charts inside documents can be described so they're searchable too.
- **Deep Research** — several rounds of web search and reasoning, written up as a report that cites only what it used; export to PDF or Markdown.
- **A two-host podcast** from a knowledge base, saved as WAV.

## Voice

Live mode is a hands-free conversation: Whisper listens, the model answers, and a local voice reads the reply back sentence by sentence — in English or Chinese. Speech runs on the CPU, so it never competes with the model for GPU memory. Silence sends your turn; any answer can be read aloud.

## Built around what small models get wrong

A frontier model can hold a workflow together on its own. A model that fits on your laptop often can't: it repeats the call it just made, sends an empty argument, retypes the line it means to change slightly wrong, and calls the job done without running it. Chaty is engineered for that model.

| | |
|---|---|
| **Its own dialect** | Tool calls are taught and read in the format each model was trained on — XML, JSON, Gemma, LFM, K2, GLM, MiniCPM — taken from its chat template, not assumed. |
| **Edits that land** | An edit is checked against the file while it's still being written. A retyped line with the wrong spaces, escapes or line numbers still lands when exactly one place fits. |
| **Slips caught at the step** | Repeated calls, empty arguments and plans left unexecuted are noticed where they happen, with a correction that shows the model what it wrote. |
| **Nothing done on red** | Changes since the last passing run are verified before a turn may end. |
| **A cache that carries over** | Each turn's prompt is an append to the last, so the model's cache is reused instead of read again from the top. |
| **One app** | No server, no port, no API key, no config file. |

## Models

Any GGUF runs on llama.cpp (Metal, or Vulkan on NVIDIA, AMD and Intel); on Apple Silicon, MLX folders run natively too. Search and download from Hugging Face inside the app — or point Chaty at a folder you already have. These families get their templates, reasoning controls and tool-call formats wired in:

| Family | What's wired in |
|---|---|
| Qwen 3 · 3.5 · 3.6 · 3.8 | Thinking on or off; Qwen3.8's effort ladder; vision where the model has it |
| Gemma 3 · 4 | Vision; Gemma 4's own tool-call format and reasoning |
| K2 Horizon · MoVA | Its effort ladder and `<ifm\|arg_key>` tool calls, on both engines |
| GLM-4.5 · 4.6 · 4.7 | Its `<tool_call>name<arg_key>…` tool calls |
| Llama 3 · Muse-Glimmer | Vision, and Muse-Glimmer's four-rung effort ladder |
| MiniCPM5 · LFM 2.5 | Each family's own tool-call format |
| [Chaty · Qwen3.5-4B design](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF) | Our own fine-tune for single-file web pages — one click at first launch |

A community fine-tune whose chat template differs from llama.cpp's built-in guess is run with its own template, as transformers would run it.

## Privacy

Models, conversations, documents and pictures live in a folder on your disk. There's no account and no server of ours to trust — delete the folder, and it's gone.

The network is used only for: **web search and Deep Research** when you turn them on; **downloads you start** (models, voices, embedding files); and **one update check** — a request to GitHub for the latest release, a few seconds after launch. Details in [Privacy & data](https://chaty.ca/docs.html#privacy).

## Install

Download from the [latest release](../../releases/latest):

| Platform | File | Notes |
|---|---|---|
| macOS (Apple Silicon) | `Chaty_*_aarch64.dmg` | Metal and MLX. See the first-launch note below |
| Windows 10 / 11 (x64) | `Chaty_*_x64-setup.exe` | Vulkan. Per-user installer, no admin needed |
| Linux (x86-64) | `Chaty_*_amd64.AppImage` | Vulkan. Beta — `chmod +x`, then run it |

**macOS first launch.** Chaty is signed but not notarized (there's no paid Apple Developer account behind it), so Gatekeeper warns on first open. Clear the download quarantine once, then open Chaty as usual:

```sh
xattr -dr com.apple.quarantine /Applications/Chaty.app
```

Or open it, dismiss the warning, and choose **System Settings → Privacy & Security → Open Anyway**.

Small quantized language models run in 8 GB of memory; text-to-image models want 16 GB or more. Chaty sizes GPU offload to your memory and refuses a model that can't fit rather than freezing the machine. New to it? [Getting started](https://chaty.ca/docs.html#getting-started).

## Build

Full details in **[BUILD.md](BUILD.md)**.

```bash
# macOS (Apple Silicon)
npm install
npm run tauri dev      # dev (Metal)
npm run tauri build    # → .app + .dmg
```

```powershell
# Windows
npm install
.\dev.ps1                            # dev
npm run tauri build -- --no-bundle   # release exe → compile the Inno installer
```

The MLX and image engines are separate helpers — `scripts/build-mlx-sidecar.sh` and `scripts/build-sd-sidecar.{sh,ps1}`. Releases come from CI: bump with `scripts/bump-version.sh x.y.z`, push a `vx.y.z` tag, and GitHub Actions builds all three platforms onto one release.

## Architecture

| Layer | Stack |
|---|---|
| Shell | Tauri 2 — tray, global shortcut, single instance |
| Interface | React 19 · Vite · react-markdown · KaTeX · Mermaid |
| Language models | Rust · `llama-cpp-2` (llama.cpp — Metal / Vulkan) · MLX through an `mlx-swift-lm` helper on Apple Silicon · chat templates rendered with minijinja where a model's own differs |
| Image models | stable-diffusion.cpp in the `chaty-sd` helper |
| Voice | `sherpa-rs` (ONNX Runtime, CPU) — Whisper, Kokoro-82M, and a VITS Chinese voice |
| Knowledge base | bge-m3 embeddings + BM25 · hybrid RRF / MMR retrieval · SQLite vector store |
| Storage | SQLite — conversations, sessions, full-text search |

## Contributing

Bug reports are welcome — attaching the error log (**Settings → Data → Open error log**) usually turns days of guessing into minutes. See [Contributing](https://chaty.ca/docs.html#contributing) for building, testing and the benchmarks.

## License

MIT — see [LICENSE](LICENSE). Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), [MLX](https://github.com/ml-explore/mlx-swift), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [Tauri](https://tauri.app) and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
