# Changelog

## v1.1.0 — Open from Hugging Face (2026-06-25)

- **`chaty://` deep links** — Chaty now registers a URL scheme and handles `chaty://open_from_hf?model=<repo>&file=<file>`: it focuses the window and opens the downloader pre-filled with that Hugging Face repo (auto-starting the file if one is named). This is the groundwork for a one-click **“Use this model → Chaty”** entry on Hugging Face GGUF pages, with a launcher page that sends people without Chaty to the download instead.

## v1.0.1 — Canvas fixes (2026-06-25)

- **Canvas edit & fix reliability** — the *Fix* and edit actions could fail with “no HTML in the output” on smaller local models (they couldn't reproduce an exact in-place patch). Chaty now accepts either a fast patch *or* a full-file rewrite — whichever the model produces — so an edit always lands.
- **Canvas export** — a page opened in the browser is now saved as `canvas-*.html` instead of being named like (and mixed in with) Deep Research reports.
- **CI** — the backend test job now builds in release mode so the bundled ONNX Runtime links correctly.

## v1.0.0 — Design Canvas, command palette, and a 1.0 polish pass (2026-06-24)

Chaty reaches 1.0: a new flagship **Design Canvas**, a command palette, conversation
and data management, and a round of reliability, accessibility and testing work.

### Design Canvas — a local, self-healing design studio
- **Iterate on a live page** — open any single-file HTML the model produces into a split studio: a live preview on one side, an instruction box on the other. Ask for changes in plain language and Chaty regenerates the whole page — all on-device, using your loaded model (great with Chaty's own web-design fine-tune).
- **Version history** — every generation is saved as a version you can switch between and revert to, labelled with what changed.
- **Self-healing** — the preview watches for runtime errors (uncaught exceptions, failed resource loads, unhandled rejections) and, when one happens, offers to send it to the model to fix — with a one-click **Fix**. Every fix asks first (it never auto-sends), so there's no runaway loop, and you can mute the prompt for the session.
- **Export** — save the current version as a standalone `.html` or open it in your browser.

### Command palette
- **⌘K / Ctrl+K** opens a fuzzy-searchable palette over actions, your loaded models, and recent conversations — new chat, switch/eject model, toggle the knowledge base or web search, open settings, jump to any conversation, and more.

### Conversation & data management
- **Pin, rename, and delete** conversations from the sidebar — pinned chats float to the top; renaming is inline; deleting now asks for confirmation.
- **Model files** — the model menu shows each model's size and lets you **delete** one you no longer need (guarded so it can't touch the model in use or anything outside the models folder), plus a one-click **eject** to unload the current model and return to the empty state.
- **Your data** — a new *Data* section in Settings to open the data folder for backup or clear all conversations.

### Reliability
- **No more white screens** — an error boundary catches any unexpected render error and shows a recovery screen (with a reload) instead of a blank window; your conversations are always safe on disk.

### Downloads
- **Time remaining** — the *Set up for me* recommender, the Hugging Face downloader, and the embedding-model download now all show an estimated time left alongside the progress bar.

### Accessibility
- Respects the OS **“reduce motion”** setting (animations and transitions are dropped), and adds a clear keyboard **focus ring** on controls.

### Under the hood
- A **unit-test baseline** (prompt/channel handling, search URL decoding, RAG chunking and ranking) plus a **CI workflow** that type-checks the frontend and runs the backend tests on every push.

## v0.9.0 — A redesigned interface (2026-06-23)

A major visual refresh that gives Chaty its own identity instead of the generic chat-app look.

- **New look** — a deeper, cooler dark palette, with the brand green used deliberately (focus rings, the send button, status), an ambient brand glow on the home screen, and refined typography.
- **Hand-drawn icon set** — every UI emoji is replaced with consistent stroke icons (settings, knowledge base, Deep Research, podcast, attachments, export, …) for a cleaner, more professional feel.
- **Composer** — a bolder send button, a green focus ring, and a subtle lift so the input reads as the main surface.
- **Sidebar** — a quiet status footer that shows the app version.
- **Light theme** brought to parity for all of the new elements.

## v0.8.2 — Chaty's own model in “Set up for me” (2026-06-21)

- **Built-in Chaty fine-tune** — the first-launch **“Set up for me”** recommender now offers **Chaty's own web-design model** alongside the Qwen3.5 and Gemma 4 picks: a Qwen3.5-4B fine-tune (Q4_K_M, ~2.7 GB) tuned for leaner, stronger single-file web/HTML design with built-in Chaty identity and grounded citations — pulled straight from [HuggingFace](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF) with live progress. It's small enough to run on light machines, so it's offered in every hardware tier. The recommender grid now flows responsively to fit the extra card.

## v0.8.1 — Qwen3.5 model loading (2026-06-20)

- **Qwen3.5 support** — updated the bundled llama.cpp engine (`llama-cpp-2` 0.1.150) so Chaty can load **Qwen3.5** GGUF models, including quantized builds with the new Gated-DeltaNet (hybrid-SSM) layers and multi-token-prediction (NextN) tensors. Earlier builds failed these with a *"null result from llama cpp"* (upstream [llama.cpp #23347](https://github.com/ggml-org/llama.cpp/issues/23347)); they now load and offload to the GPU normally.

## v0.8.0 — Deep Research & a web search that actually works (2026-06-14)

A new **Deep Research** mode that runs many rounds of web search interleaved with
the model's own reasoning and writes a long, cited report you can export to PDF —
plus a complete rebuild of the web-search backend after the old single provider
was blocked, and a batch of macOS stability and quality fixes.

### Deep Research
- **Topic → cited report** — give it a subject and the model plans search queries, runs **multiple rounds** of web search interleaved with reasoning about what's still missing, then synthesizes a structured long-form report with inline `[n]` citations.
- **Topic-anchored** — the verbatim topic is always searched first, so results stay on subject even when a model derails into unrelated queries (notably uncensored finetunes on sensitive topics).
- **Honest references** — the references list contains *only* the sources the report actually cited (renumbered to stay contiguous); off-topic junk from a stray query is dropped rather than padded in.
- **One-click export** — save the report as **PDF** (rendered HTML opened in the system browser, which handles CJK and print-to-PDF reliably) or as **Markdown**.
- **Docked in the chat panel** with live progress (planning → searching → reasoning → writing), the queries being run, and accumulating sources; **cancel anytime**.
- Works fully from the 简体中文 UI; the report is written in the UI language.

### Web search — rebuilt to be reliable and free
- **The old path was fully broken.** DuckDuckGo started returning a bot-challenge page (HTTP 202) to every request, so web search — and anything built on it — failed for *all* users.
- **Multi-provider fallback chain**, all free and key-less: **Brave Search** (primary; high quality, excellent for Chinese) → Bing → DuckDuckGo (HTML/Lite) → Wikipedia → DuckDuckGo Instant Answer. The first provider with results wins.
- **Correct CJK handling** — Bing now uses the right market locale for Chinese-vs-English queries (a wrong locale was turning "刘华强" into "Milwaukee Brewers"); Bing redirect URLs are decoded to their real destinations.
- **Resilient** — a malformed or rate-limited response from any single provider no longer aborts the whole search; queries are spaced out so a run doesn't trip rate limits into the fallbacks.
- Refreshed browser User-Agent so requests aren't rejected as stale.

### Tools menu
- Reorganized into hover submenus: **知识库 / Knowledge base** → (retrieve · manage) and **联网 / Web** → (Deep Research · Web search), so the composer toolbar is less crowded.

### Fixes & polish
- **PDF export no longer crashes.** Opening files/links went through a path that does a manual `fork()`; in this multithreaded WebKit process that trips the libmalloc fork-child assertion on macOS and crashed the app (the slow, often-failing export button). All "open in default app" actions — PDF export, "Open models folder", and source-link clicks — now use a fork-free `posix_spawn`.
- **Hardware panel VRAM** now shows whole-device usage on Apple Silicon's unified memory (used / total system RAM), not just the current model's slice.
- **Better podcast voices** — the deep-dive podcast now uses the highest-graded female and male Kokoro voices (af_bella · A-, am_michael · C+) instead of a low-graded male voice; the voice picker is labeled with each voice's overall grade.

### Maintenance
- Bumped the GitHub-maintained CI actions to v5 (Node 24 runtime).

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
