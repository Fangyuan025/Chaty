# Changelog

## v0.7.0 — Local knowledge base & deep-dive podcast (2026-06-12)

A fully-offline RAG knowledge base, traceable citations, and a NotebookLM-style
audio "deep dive" — plus quality-of-life fixes across downloads and macOS install.

### Local knowledge base (RAG) — hushdoc-style, high precision, fully offline
- **Index your documents** — PDF, text, Markdown, code, and **images** (via the same OCR engine as attachments) are chunked (paragraph-aware, ~800 chars / 120 overlap) and embedded locally.
- **Multilingual embeddings** — bge-m3 (1024-d, zh+en) runs through llama.cpp on its own worker thread with a persistent embeddings context, GPU-accelerated and independent of the chat model. One-time ~0.7 GB download; **cancelable**.
- **Hybrid retrieval** — dense cosine + BM25 (ASCII words + CJK uni/bigrams) → reciprocal-rank fusion → MMR diversification → neighbor-chunk expansion. All on-device.
- **Strict grounding** — when the knowledge base is on, the model answers *only* from the retrieved passages and explicitly says "not covered by the current documents" instead of fabricating.
- **Custom query scope** — enable/disable individual documents to control exactly what's searched.
- **SQLite vector store** in app data (vectors as f32 blobs); add/remove/re-index documents from the knowledge-base panel with live indexing progress.

### Citations & traceability
- **Inline citation anchors** — the model marks each sourced sentence with a numbered superscript 【N】; hovering an anchor previews the cited passage. Works for both knowledge-base and web sources.
- **Source chips** now carry a hover preview of the cited snippet (not just the title).
- Knowledge-base retrieval shows its own "Searching the knowledge base…" status (no longer the web-search label); a combined label covers KB + web together.

### Deep-dive podcast (NotebookLM-style)
- Turn the enabled knowledge-base documents into an **English two-host conversation**, written by the chat model and grounded strictly in your sources.
- **Alternating Kokoro voices** (one female, one male) read it aloud, synthesized line-by-line.
- **Progress bar + estimated time remaining**; other LLM features are **locked** during generation for stability, and you can **cancel anytime**.
- **Export the audio** as a `.wav` file. (Podcast output is English-only, but the feature is fully usable from the 简体中文 UI.)

### Downloads
- **Cancelable everywhere** — the first-launch "Set up for me" recommender, the HuggingFace repo downloader, and the embedding-model download all have a cancel button; partial `.part` files are cleaned up.

### macOS install
- **Documented first-launch path** — the release page and README now explain clearing Gatekeeper quarantine (`xattr -dr com.apple.quarantine /Applications/Chaty.app`) or using **System Settings → Privacy & Security → Open Anyway**. The app is ad-hoc signed with the right entitlements (mic, JIT, library validation for bundled ONNX dylibs) and a hardened runtime.

## v0.6.0 — macOS (Apple Silicon) port (2026-06-11)

Chaty now runs natively on Apple Silicon Macs, alongside Windows.

### macOS support
- **Metal GPU backend** — selected per target via a feature-multiplexer crate; Windows keeps Vulkan unchanged. On unified memory, all layers are offloaded when the model fits (`recommendedMaxWorkingSetSize` budget), with P-core-only worker threads.
- **Native window chrome** — traffic lights (`titleBarStyle: Overlay`), Dock-icon reopen, menu-bar tray, `Cmd+Shift+Space` global hotkey.
- **Clean quit on every path** — tray Quit, app-menu Quit and Cmd+Q no longer trip ggml/ONNX teardown crashes ("Chaty quit unexpectedly").
- **`.dmg` packaging** with entitlements (mic, JIT, library validation for the bundled ONNX dylibs); CI builds the dmg headlessly via `hdiutil`.
- **Native microphone capture** (CoreAudio/cpal) — WKWebView never exposes capture devices to embedded apps, so recording bypasses it entirely; devices are scanned (no phantom default-device failures) and the system mic consent is requested properly.

### Model support
- **Gemma 4** — native renderer for the `<|turn>role …<turn|>` format, `<|think|>` thinking control, `<|channel>thought` reasoning folded into the think panel, turn-boundary stop insurance.
- **Qwen 3.5 / 3.6** — pre-opened `<think>` handling (synthetic open tag for the UI), reliable no-think (empty think block), detected by GGUF architecture so community finetunes with custom templates behave too; the dead `/no_think` soft switch is never sent to 3.5+.
- **Robust template fallback chain** — embedded template → system-role folding → per-architecture built-in → ChatML, so unusual GGUFs still chat.
- Families covered: **Llama 3, Gemma 3, Gemma 4, Qwen 3, Qwen 3.5/3.6**.

### Memory & model switching
- **Synchronous eject before load** — switching models fully tears down (and verifies release of) the old model before the new one loads; no more unified-memory swap freezes. mmap is disabled on macOS (Metal-wired pages of mmap'd MoE models never returned to the kernel).
- **Model load progress bar** (eject → weights % → ready) in the titlebar and model chip.
- **Pre-flight guard** — models that cannot physically fit in RAM are refused with a clear message.
- **Context auto-fit** — "Auto" now uses as much of the model's trained context as memory allows (KV-cache-size aware); custom values are capped to fit, with a visible notice when clamped; the settings slider adapts to the loaded model's trained length.

### Chat & UI
- **Stop reason** shown after each reply (finished / length / context full / stop sequence / cancelled).
- **Focused thinking view** — while reasoning streams, a small window follows the newest text with older lines fading out; expandable as before.
- Circular context-usage ring (amber > 80 %, red > 95 %).
- **Unlimited reply length by default** (opt-in cap in settings); "Reload to apply" button under the context setting.
- **Playable HTML preview** — single-file web games work (keyboard focus + localStorage shim in the sandbox).
- Mermaid: theme-aware, looser parsing, and visible error messages instead of silently showing raw code.
- Higher-contrast light theme; "Open models folder" in the model menu; backend errors are bilingual (中文/English).
- GPU/memory usage in the hardware panel now reports the app's real footprint.

### Release engineering
- **Cross-platform release CI** — pushing a `vx.y.z` tag builds the Windows installer and macOS dmg and publishes both to one GitHub Release (version consistency is checked against the tag).
- `scripts/bump-version.sh` syncs the version across `package.json`, `tauri.conf.json`, `Cargo.toml` and `Cargo.lock`.
- The in-app updater picks the right asset per platform (`.exe` / `.dmg`).

## v0.5.x and earlier

See the [release history](https://github.com/Fangyuan025/Chaty/releases) — drag-drop attachments, sampling controls & presets, voices, themes, export/search, model downloader, Mermaid, Qwen3.5 thinking control, and more.
