# Changelog

## v1.8.5 — The canvas opens up, the agent slims down (2026-07-18)

The Design Canvas stops being a black box, and the coding agent gets a deep
efficiency pass built for small local models. Plus a first benchmarks page.

### Design Canvas, rebuilt in the open (#canvas)

- **Preview | code, side by side.** Every canvas now shows the actual source next to the live preview — syntax-highlighted, following your code palette in light and dark. All three columns drag-resize (double-click resets), and a fullscreen toggle (animated, traffic-light-aware, window still draggable) takes the studio edge-to-edge.
- **Element ↔ code correspondence.** Inspect mode links the panes both ways: hover or click an element in the preview and the code jumps to its line; click a code line and the element flashes in the page. **Clicking selects** (⌘/Ctrl multi-select) — chips above the composer name each element, and your next instruction is scoped to exactly those elements.
- **Watch edits happen, Cursor-style.** While the model streams an iteration the code pane live-scans the document: the line being read pulses, deletions/additions grow token by token, and a full rewrite dims the not-yet-reached old code instead of pretending it was deleted. Every finished iteration lands on a **Changes** view — the same +N/−N red/green diff as Code mode — so version-to-version deltas are never a mystery.
- **Hand-edit the source.** An **Edit** button opens the code in place; saving lands as a "Manual edit" version and shows your own change as a diff.
- **Canvas sessions survive.** Closing the studio no longer wipes iterations — every reply keeps its own version history across close/reopen; an explicit **Reset canvas** (with the standard confirm) drops back to v1.
- **A real console + browser-honest pages.** A Console tab mirrors the page's logs, warnings and errors (count badge included), and a compatibility layer fixes the class of "works in my browser, errors in the canvas" bugs — history API routing, cookies and clipboard now behave inside the sandboxed preview. Plus a one-click page reload that re-runs scripts from scratch.

### The agent got cheaper to run

- **The system prompt slimmed by a third to a half.** Tool docs are one line of purpose + args each (loop-enforced contracts kept verbatim), and English sessions get real English docs instead of Chinese ones: 5,292→3,545 chars (zh), 8,801→4,432 (zh + vision), 9,873→6,073 bytes (en). The prompt re-prefills every agent step, so this is a per-step saving on slow local prefill.
- **Guidance moved to just-in-time.** The browser workflow guide and edit-failure recovery now ride into the conversation only when a browser tool first runs or an edit first fails — not on every step of every task.
- **Tool output speaks ONE language.** Results and errors used to carry Chinese and English side by side; they now render in the session language only (~80 strings converted, contract markers untouched).
- **Compaction that remembers what it removed.** Elided tool results carry a digest — file + range for reads, command head + exit code for bash, the query for searches — so the model stops re-reading files it already read; trimmed history leaves a bullet summary of the dropped turns.
- **Read-only commands skip the approval dialog** (Settings → Code, default on): `ls`, `grep`, `git log` and friends run immediately; anything that writes, deletes, or is uncertain still asks. Fail-closed by design — redirection, substitution, and unknown commands always fall through to the dialog.

### Chat polish

- **Long code blocks fold** to a header with a first-lines preview (Settings → Chat, default on); while streaming they show a focus window pinned to the newest lines, think-panel style, and clicking expands at any moment.
- **Light-theme code, properly.** GitHub and Atom One palettes switch to their light siblings under light app themes; inline code chips and table headers — invisible white-on-white before — now follow the appearance. Code blocks carry a language tag and tab-size 4.
- **Reading ergonomics.** A floating jump-to-latest pill appears when you scroll up mid-stream (and no longer haunts empty conversations); user messages get a hover copy button, and pasted walls of text clamp to a preview with "show all".

### Benchmarks, published

- First numbers in the repo — same local model (Qwen3.5-35B-A3B MoE, ~3B active) on all rows: **SWE-bench Verified 45-task macOS subset 9/45 with the full Coder loop vs 6/45 bare-bash ablation** (django slice: 7/24 vs 2/24), Terminal-Bench core 15/77. Methodology and honest-comparison notes in `docs/BENCHMARKS.md`.

### Fixed

- **Quitting no longer leaks the MLX model.** The exit path skips destructors (a ggml teardown workaround), which orphaned the MLX sidecar — every quit with an MLX model loaded left a process holding the whole model in unified memory, silently stacking up across sessions until the machine choked. Sidecars are now reaped explicitly on exit.
- The dev-only double model load at startup (two 35B sidecars racing for RAM) is gated off.
- `tauri dev` runs again on a fresh clone (`default-run` was ambiguous since the bench tool server joined the crate).

## v1.8.4 — Scroll free while it streams (2026-07-17)

The streaming-scroll fix (thanks @sprite5, #4), plus benchmark and CI groundwork from this cycle.

### Scroll up any time while the model is streaming (#4)

- **Scrolling up during a streaming reply now sticks immediately.** Auto-follow used to be a distance check — within ~140px of the bottom every new token re-pinned the view, and on a fast stream you couldn't escape the threshold before the next token dragged you back: the scrollbar only moved with the stream, and on a touchpad it felt impossible to leave the bottom at all. Chat now treats following as an *intent*, the same model Code mode has shipped since v1.5.0: any upward wheel or touchpad motion releases auto-follow instantly and the view stays exactly where you put it; dragging the scrollbar away from the bottom releases it too, and parking back at the bottom re-arms it. Sending a message or switching conversations still jumps to the bottom as always.

### Under the hood

- **ChatyCoder-Bench** — a headless harness (`chaty-headless`) that drives the real Code-mode agent loop against SWE-bench-style task sets: gold-validated tasks, per-repo dependency quirks codified, heredoc-safe wiring with per-command timeout recovery, and temp workspaces cleaned up after grading.
- **CI catches more before a release**: the frontend now really builds in CI, clippy runs as a correctness gate (in release profile so native deps link), and Windows gets a compile check.
- Scaffold leftovers swept out — the app page carries Chaty's own favicon and title instead of the Vite/Tauri defaults.

## v1.8.3 — The models folder tells the truth, HF from anywhere (2026-07-17)

Two community-reported fixes (thanks @sprite5) plus a packaging root-fix.

### The models folder finally tells the truth (#1)

- **"Open models folder" opens the folder your models are actually in.** Several roots are scanned for models but the menu always opened the app-data one — for users upgrading from an older install (models next to the exe) that folder is empty, which read as "my models are gone". The menu now opens the first root that really contains models, falling back to the download target only when none does.
- **Models dropped in mid-session appear without a restart.** Loose `.gguf` files used to be organized into the folder layout only at startup — drop one in while Chaty runs and it stayed invisible until a relaunch. The picker now organizes on every open: drop the file, reopen the model picker, it's there.
- A small toast after "Open models folder" explains the drop-in flow — the original report read the button as a file picker.

### HuggingFace endpoint setting (#2)

- **Settings → Model → HuggingFace endpoint: Official / hf-mirror.com / custom URL.** Model search, quantization lists, READMEs, downloads (GGUF and MLX), and the knowledge-base embedding model all follow the chosen endpoint — for networks where huggingface.co is unreachable (mainland China: pick hf-mirror.com).
- Mirrors don't speak the xet fallback protocol (the official-CDN-block workaround), so on a mirror a rejected file now says exactly that — switch back to Official — instead of a cryptic HTTP error.

### Windows-only bug sweep

- **No more flashing console windows.** Every console child a GUI app spawns on Windows pops a black window without `CREATE_NO_WINDOW` — opening a folder or link, every Code-mode command, every post-edit syntax check, and killing background jobs/the agent browser all flashed one. All spawn sites are now silent.
- **Links with query strings open whole.** "Open externally" went through `cmd /C start`, whose parser splits an unquoted URL at `&` — links died at their first query parameter. URLs now go through `rundll32 FileProtocolHandler` and files/folders through `explorer` (both console-free; verified with a local listener that the full query string reaches the browser — explorer alone silently drops it).
- **Code mode speaks the right shell.** The agent's instructions described a POSIX shell, but Windows executes commands via `cmd.exe` — models wrote `ls`/`cat`/`$VAR` and watched them fail. On Windows the prompt now says to use Windows commands (`dir`, `type`, `findstr`) or cross-platform tools (git, npm, python), and `%VAR%` syntax.
- **Chinese-locale command output is readable.** Windows consoles emit GBK, not UTF-8 — `dir` listings and error messages reached the agent as mojibake. Output that isn't valid UTF-8 is now decoded as GBK.
- **Python syntax checks work with real Windows Pythons.** The post-edit checker invoked `python3`, which official Windows installers don't create — and the Microsoft-Store stub *named* python3 exits with an error, flagging every `.py` edit as a syntax failure. The checker now prefers `python` and ignores the store stub.
- **Per-user Chrome installs are found.** The agent-browser looked only in `Program Files`; Chrome's default per-user location (`%LOCALAPPDATA%`) — the usual non-admin install — plus 64-bit Edge and Brave are now on the candidate list.
- **Failed commands no longer wear a green check.** A Code-mode command that ran but exited non-zero (e.g. `ls` on stock Windows) showed the same ✓ as a success — the failure hid in a small exit badge, so broken commands read as working. Non-zero exits now show a red ✗ on the step card.
- **The `list_dir` step card no longer masquerades as `ls`.** Its summary label was literally "ls <path>" — on Windows that read as a shell `ls` succeeding where cmd.exe has no such command, and sent our own debugging down a phantom-bug hole. It now says `list <path>`.
- **Malformed tool calls show as errors.** A hallucinated tool name from a small model (e.g. `{"name":"tool"}`) returned a polite "unknown tool" note with a green check; it's now a red-✗ error step, and the model is pointed at the tool list.

### Packaging

- **`vulkan-1.dll` ships in the Windows installer.** Machines without a GPU driver (clean VMs — winget validation among them) lack the system Vulkan loader and the app failed to launch. The runtime loader now sits next to the exe as a fallback; on normal machines the driver's own loader still wins.

## v1.8.2 — Smart tools: the agent's tools do the thinking (2026-07-16)

Small local models shouldn't burn steps on planning grunt work — this release moves that intelligence **into the tools**. Verified end to end with a real 8B model: locate → fix → validate a bug in a multi-file project in 18 seconds.

### The toolbox got smarter

- **`understand_repo`** — one call returns the whole workspace orientation: README lede, manifests (name + scripts), a two-level tree, language census, and entry-point candidates. The agent's first move in an unfamiliar project, replacing a chain of directory listings.
- **`search_code` now ranks files, not chunks.** Ask "where is email validation handled" and get a relevance-ranked file list (semantic score fused with filename matches and exact-phrase hits), each file carrying its matching definition lines and best snippet — the filtering the model used to fumble across grep rounds is done inside the tool.
- **`read_file` reads symbols.** Pass `symbol: "refreshToken"` and get exactly that definition block plus every call site across the workspace — instead of paging through a 5000-line file. Unknown symbol? The error lists the definitions the file actually has.
- **Edits pass through a syntax gate.** After every write, cheap per-language checks run (JSON/TOML, Python, shell, JS); breaking a previously-parsable file warns loudly with a pointer to the checkpoint rewind.
- **`validate_change`** — after editing, one call finds the tests related to what changed (pytest / vitest / jest / cargo conventions), runs just that minimal set, and summarizes failures. No arguments needed: it knows what was touched this turn. Also fixes a real agent-speed hazard — Python's bytecode cache could serve stale code when an edit and its test run land in the same second.

### Documents, both surfaces

- **The Code agent reads documents.** `read_file` on a pdf / docx / xlsx / pptx extracts the text (same pipeline as chat attachments); scanned PDFs with no text layer get their pages OCR'd automatically, and embedded charts/photos are cached for `view_image` — no directory-grant friction.
- **The chat attachment picker accepts docx / xlsx / pptx.** The extractor always supported them; the picker's filter didn't — only drag-and-drop worked.

## v1.8.1 — Hotfix: oversized screenshots could kill the MLX engine (2026-07-14)

- **Huge screenshots can no longer crash the MLX engine.** A 2x full-page browser screenshot (15+ megapixels) fed to an MLX vision model — the Code agent's normal "look at the page" flow — could kill the inference sidecar mid-answer ("MLX engine exited unexpectedly"), or come back with an empty answer on models whose processor tolerated the size. Images now pass through the same 2-megapixel downscale as the GGUF engine before reaching the sidecar, with the downscaled copy cached so image reuse across turns still works. Verified end to end on the exact model and flow that crashed.
- **Healed community quants resize like stock models.** The processor config Chaty synthesizes for community VLM quants that ship without one now includes the official smart-resize band, so oversized inputs are constrained the same way as on official releases.
- **The model-load progress bar only moves forward.** Loading progress is estimated from memory growth, which dips mid-load (page eviction, GPU staging frees, out-of-memory back-off) — the bar visibly jumped backwards. A high-water mark on both engines plus a guard in the UI keeps it monotonic.
- **The Code agent knows when to change tack.** When the search backend degrades into irrelevant results, models used to rephrase the same query forever — now guidance plus a consecutive-search breaker (a reminder rides the 3rd/4th search, the 5th is intercepted) steer it to fetch a guessable URL directly or open the browser; verified end to end with a real model against a rigged always-irrelevant search backend. Same principle applied across the board: an unclear page digest → take a screenshot and look; the same approach failing twice → switch approach.
- **sudo passwords actually arrive.** The password dialog also appeared for background (`bash_bg`) sudo commands, whose sandboxed jobs can't receive stdin — the password was silently dropped and sudo reported a confusing "no password was provided". sudo is now foreground-only (the agent is told to re-issue the command with `bash`), and when a delivered password is *rejected*, the result now says exactly that instead of hiding behind sudo's misleading last line.

## v1.8.0 — Apple-Silicon native: MLX models (2026-07-13)

Chaty now runs **MLX models** — the Apple-Silicon-native format from mlx-community — side by side with GGUF, with the exact same feature surface. Plus a brand-new model store and a root fix for the wired-memory balloon.

### MLX, first-class

- **Folder models, fully wired in.** Drop an `mlx-community` model folder into `models/` (or download one in-app) and it loads like any GGUF: streaming, sampling controls, stop sequences, reasoning on/off per family (Qwen3's `enable_thinking`, Qwen3.5's prefilled think block, plain models unaffected), KV prefix reuse across turns, and the prompt-processing ring.
- **Vision included.** MLX vision models (the Qwen3.5+ and Qwen3-VL families) carry their vision tower in the weights — attach an image in chat, let the Code agent look at screenshots, caption knowledge-base imports, see Canvas pages. No separate encoder file to manage.
- **Vision that keeps up with Code mode.** Image prompts show the same prompt-processing **percentage ring** as text, and the image KV cache mirrors the GGUF engine: on models whose cache can rewind (Qwen3-VL), follow-up turns and agent tool loops reuse the cached image — screenshots aren't re-encoded every round. Positions are M-RoPE-exact end to end (a from-scratch decode loop threads the rope state through every token — this also fixed Qwen3-VL models answering with an instant EOS, and long multi-chunk prompts drifting off-position on Qwen3.5).
- **Memory safety by design.** MLX inference runs in an isolated sidecar process — ejecting the model kills it, so the memory *always* comes back (verified by a repeated load/eject e2e). Runs at full Metal speed via Apple's mlx-swift-lm.
- **Community quants included.** Some third-party VLM quants ship without their processor config files and would refuse to load — Chaty now heals the folder automatically (the missing preprocessing config is synthesized from the model's own `config.json`), so those models chat *and* see like any other.
- Windows builds politely decline MLX folders with a clear message; everything GGUF is unchanged.

### A real model store

- **Search & browse HuggingFace in-app** — by keyword or author, filtered by format (GGUF / MLX) and sorted by trending, downloads, likes or recency.
- **Models, not file lists.** Pick a quantization from a dropdown (`Q4_K_M · 7.15 GB`), see parameter/architecture/vision badges and the repo's README rendered right there, and get a "fits fully in memory" hint sized to your machine. Multi-part quants download and list correctly; vision models grab their encoder automatically.
- Pasting a repo link still works — Enter opens it directly. `chaty://` deep links now handle MLX repos too.

### Conversations stay in their lane

- **Cross-conversation isolation, guaranteed.** On hybrid-attention models (Qwen3.5 / 3.6, GGUF *and* MLX alike) the caches can't partially rewind and also misreport how much they hold — switching conversations could leave the previous one's KV in place, so the model literally saw two conversations at once (spiralling into confused, endless "thinking"). Both engines now keep their own exact ledger of what's in the cache and fall back to a full clear whenever a partial rewind isn't possible — regression-tested with cross-conversation canary prompts on hybrid, dense and standard models, both formats.
- **Deleting a coding session mid-run resets instantly.** The agent is stopped, any pending approval dialog is dismissed, queued messages are dropped, and the finished turn can no longer resurrect the file you just deleted.
- **Coding sessions get real titles.** After the first turn the model writes a concise session title (same as chat conversations, honouring the auto-title setting) instead of the sidebar echoing your raw first message.

### One way to load, no more freezes

- **"Load from folder" is the single local entry point** for both formats (one folder per model has been the canonical layout since vision landed) — the backend figures out what's inside.
- **Auto-load only reloads *your last* model** — it no longer picks the alphabetically-first folder, which could be a 32 GB giant on a 16 GB machine.
- **Wired-memory root fix:** ggml's Metal residency sets (enabled by newer build SDKs) pinned entire models into wired memory and could freeze the machine on big models. Chaty now disables them at startup on every build, so locally-built and CI binaries behave identically. Verified live with a 32B model: wired memory flat throughout load, chat and eject.

## v1.7.1 — Agentic browsing that finishes the job (2026-07-13)

A deep pass on Code-mode browser automation — driven by testing the local model against **real websites**, not just fixtures — plus embedded-image vision, prompt-injection defense, and a batch of quality-of-life fixes.

### Browser automation you can hand a task to

- **Reads pages as text, screenshots when it matters.** `browser_read` now returns the page's full visible text (dynamic rules, validation messages, results) plus the interactive elements *and their current values* — so the agent tracks state without a screenshot, and it saves the vision model for what actually needs eyes (rendering, layout, images, and confirming an answer before an irreversible submit). Read tasks fly; visual checks still happen where they count.
- **Real mouse clicks, on the right element.** Clicks are dispatched as genuine mouse events (so React-style widgets that ignore synthetic clicks respond), and when several elements share the same text — a nav "Login" link next to a form's "Login" button — the actionable control wins, so the form actually submits. After a click that navigates, the agent waits for the new page before deciding what's next, instead of re-clicking a stale one.
- **Do more per call.** `browser_click` and `browser_type` take a `steps` array to click a sequence or fill a whole form in one call — pick words in order for a Duolingo-style exercise, or fill six fields at once — instead of one call each. **Dropdowns** work too: `browser_type` with the option's text selects it.
- **Verified on real sites.** New end-to-end suites drive the 35B model through real pages (quotes.toscrape login / pagination / tag / search-form, Wikipedia, Hacker News) and local replicas (a long scrollable form, a progressive password game, a word-order sentence builder) — checking the *actual page state*, not the model's self-report.

### See inside documents

- **Embedded images are read, not skipped.** Charts, photos and screenshots *inside* PDFs, Word, Excel and PowerPoint files are now extracted and shown to a vision model — both as chat/Code attachments and when building the knowledge base (each figure gets a searchable description). Word/Excel/PowerPoint also get proper text extraction as chat attachments (previously docx/xlsx fell back to raw bytes), including per-slide text for decks.

### Trust & control

- **Prompt-injection defense.** Web pages, search results and file contents the agent reads are treated as data, never instructions: any control tokens they contain are neutralized (a page can't forge a tool call or break out of a result), and a firm rule tells the model to ignore embedded "commands."
- **Out-of-workspace access asks first.** When the coding agent needs a file outside your workspace it requests permission (once per directory, per session); granted folders show as removable chips in the header, and you can add one yourself.
- **`sudo` needs your OK — with a password field.** A privileged command pops a distinct high-risk dialog with a masked password box; the password is piped straight to `sudo` and never shown, logged, saved, or sent to the model, and the command runs outside the workspace sandbox only after you confirm.

### Comfort

- **A real progress ring.** The quiet moment before the first token — while a long prompt (or an image) is processed — shows a smooth circular **percentage** in Code mode, so you can tell working from stuck. Image turns are faster too: oversized screenshots are downscaled before the vision encoder, and already-seen images aren't re-processed.
- **Background downloads.** `web_download` returns immediately and streams in the background with a live progress badge; the agent keeps working and is told when it finishes.
- **New Code settings** (step temperature, auto-approve edits, run the browser hidden), a one-time prompt to tidy loose model files into the folder layout after updating, the Code agent now knows the current date/time, and the coding-session delete confirmation no longer says "conversation."

## v1.7.0 — Chaty can see (2026-07-12)

The big one: local **vision** models now reach every surface, and Code mode gains a **real browser** it can drive and see. Everything still runs on your machine, nothing leaves it.

### Vision, everywhere

- **Image understanding across the whole app.** Load a vision GGUF (with its `mmproj` encoder) and Chaty *sees* — attach an image in **Chat** and ask about it; in **Code** the agent has a `view_image` tool and reads screenshots; the **knowledge base** captions imported images (a real description alongside OCR text); **Canvas** shows the model the live rendered page. Text-only models keep the OCR path, so nothing regresses.
- **One folder per model.** Vision models keep their weights and `mmproj` encoder together in `models/<Name>/`, and Chaty pairs them automatically — in the downloader, in *"Set up for me"*, and for models you load from disk. Updating from an older version? A **one-time prompt** offers to tidy your existing loose `.gguf` files into this layout with a single click (files are only moved, never deleted).
- **No re-encoding on every turn.** A media cache keeps already-seen images in the KV cache, so a follow-up question doesn't re-process the picture — later turns stay fast.

### A browser the agent can drive

- **Full browser automation in Code mode.** The agent can open pages, click by visible text, fill forms by field label, scroll lazy-loaded pages, read the interactive-element list, check the JS console, run JavaScript, and take **full-page or viewport screenshots it actually looks at** — driving a real Chrome so you can watch it work. Logins persist in a dedicated profile, so a site you sign into once stays signed in.
- **See before it acts.** After any click or navigation the agent re-checks the page state before the next step instead of guessing, and it leads page research with a single full-page screenshot to locate what it needs — fewer, surer steps. Research still prefers fast `web_fetch`/`web_search`; the browser is for real interaction and visual verification.
- **Robust by construction.** If you close the browser window mid-task it transparently relaunches and continues; a repeated scroll counts as progress, not a stuck loop; and an `Organize`-grade multi-step suite (shop checkout, gated login, lazy-load feed, on-page lookup) is verified end-to-end against the real model, checking the actual page state rather than the model's word.

### Code mode, more comfortable

- **A progress ring for the quiet moment.** The pause before the first token — while a long prompt is being processed — now shows a circular **percentage** ring instead of a blank spinner, so you can tell processing from stalling.
- **New Code settings.** A **step temperature** slider (steadier vs. more creative), **auto-approve file edits** (checkpoints still cover rollback), and **run the browser hidden** (headless, no window) — all under Settings → Code, alongside the existing step limit, command timeout, allowlist and skills.
- **Attachments match Chat.** The Code composer now takes the same documents *and* images as chat — PDFs/Word/Excel/~90 formats extracted to text, images sent to vision models (or OCR'd for text-only ones) — and screenshot/`view_image` steps are clickable for a full-resolution preview you can save.

## v1.6.2 — Accurate diffs & whole-file reads (2026-07-10)

- **Code-mode diffs are now exact.** Edit previews were computed with a rough prefix/suffix heuristic and hard-capped at 60 lines — so the +N/−M badge silently under-counted big edits, scattered changes mislabeled untouched lines as changed, and large diffs were cut off. Chaty now runs a real line-level (LCS) diff: the +/− counts are exact over the whole change, only genuinely changed lines are marked, and the rendered hunk shows more with a clear "… N changed lines total" note when it's very long. The before/after snapshots are also read in full, so a big file's diff is no longer polluted by pagination text or truncated.
- **Long files read in one call.** The per-read budget now uses most of the model's real context window (leaving headroom for the system prompt and room to act), so files up to ~100 KB come back in a single `read_file` instead of paging — the agent stops re-reading the same file in pieces. Genuinely huge files still paginate with a followable offset.
- **New `search_files` tool.** A single literal-keyword search across both file **names** and file **contents** — "find anything to do with X" in one call, filling the gap between `glob` (name patterns) and `grep` (content regex). Pass `names_only` to search paths alone.
- **One edit tool.** `edit_file` and `multi_edit` were merged into a single `edit_file`: pass `old_string`/`new_string` for one change, or an `edits` array to apply several atomically — one fewer near-identical tool for the model to pick wrong. (`multi_edit` still works as an alias.)
- **Edits go through edit, not rewrite.** The agent used to sometimes reach for `write_file` to change an existing file — regenerating the whole thing to tweak a few lines, which risks dropping content it didn't retype. The prompt now scopes `write_file` to new files and true full rewrites, and a guardrail intercepts a small change delivered as a full rewrite of a sizable file, steering it to `edit_file` instead.

## v1.6.1 — Bilibili in-site & a think-loop gate (2026-07-10)

- **Bilibili** joins the in-site search family: `web_search` with `site="bilibili.com"` returns structured videos (title / UP / duration / views), and fetching any `bilibili.com/video/BV…` link returns the video's public metadata and description — all through Bilibili's own key-less public API, no cookie or login. Weibo / Xiaohongshu / X keep the search-engine snapshot fallback, since their public read endpoints are now login-walled and Chaty won't circumvent authentication.
- **Think-loop gate (Code mode)** — small models sometimes get stuck reasoning forever, never emitting a tool call or an answer. Chaty now cuts a runaway mid-stream once reasoning runs long past a budget with no output (or falls into degenerate repetition), then recovers by forcing reasoning **off** for one step, sampling hotter, and demanding a concrete action — instead of burning the whole token budget or returning a blank reply. A persistent loop pauses cleanly with a **Continue** button rather than spinning to the step limit.
- **Web search, hardened for agents** — the search chain was reworked so it stays alive under heavy, repeated use. A per-provider **circuit breaker** sits a source out for a while the moment it blocks us (so rapid-fire searches stop re-hitting a dead engine), an in-memory **cache** serves repeat queries instantly (agents loop over the same terms constantly), results are **validated** so a challenge/consent page can't short-circuit the chain with junk, and the providers were reordered to the ones currently answering (DuckDuckGo first; Brave, which now hard-blocks, demoted). Net effect: real results instead of the noisy last-resort fallback, and no more collapse when the agent searches in a tight loop.

## v1.6.0 — The agent levels up (2026-07-10)

Code mode grew in two directions at once: a no-blind-spots web research layer, and a set of editing power tools that make the agent faster and far harder to derail. All key-less, talking to sites directly, nothing routed through third-party services.

### Sharper coding

- **`multi_edit`** — several exact-match edits to one file in a single atomic call: every edit is validated (later ones against the result of earlier ones) and the file is written only when all of them land — a failure changes nothing. Multi-site changes that used to burn five steps now take one.
- **"Did you mean" edits** — when an exact-match edit misses, the error now shows the most similar line in the file with numbered context, so the next attempt copies the real text instead of re-reading the whole file. Successful edits echo the modified neighborhood back for free verification.
- **`outline`** — the definition lines of a file (functions/classes/structs/… with line numbers) across Rust/TS/JS/Python/Go and friends, so the agent grasps a big file's structure without reading it whole, then jumps straight to the right region with a ranged read.
- **Failures stay visible** — long command output used to be tail-chopped exactly where the panic/test summary lives; bash results now keep the head *and* the tail, eliding the middle instead.

### Everywhere online

- **In-site search** — `web_search` gained a `site` parameter: **GitHub** returns structured repositories (stars/language), issues/PRs, *and code matches* (via Sourcegraph's public index, since GitHub's own code search requires auth); **Reddit** searches posts through its still-open RSS endpoints (scope to a subreddit with `reddit.com/r/xxx`); **YouTube** returns structured videos (title/length/channel/views); **any other domain** — docs sites, Stack Overflow, **x.com** — is searched via a `site:` query over the multi-engine chain (for X, the engines' snapshot index is the only key-less view that exists).
- **Video understanding** — fetch any YouTube link and the agent gets the video's metadata plus its **full caption transcript** with periodic timestamps (auto-generated captions included, any language; manual tracks preferred, Chinese/English first). Search YouTube in-site, pick a video, and the agent can reason over what is actually *said* in it — no API key, no external transcription service.
- **Fetch anything** — `web_fetch` is now content-type aware: articles become clean Markdown (Readability extraction + HTML→MD), code/JSON/config files pass through as source, **GitHub file pages auto-rewrite to the raw file**, Reddit posts return the thread with comments, PDFs are text-extracted, and binaries report their metadata. `raw=true` returns the page's HTML source.
- **Walk into sub-pages** — every fetched page returns its harvested links (same-host first) and image URLs, so the agent can navigate a docs site or repository page by page.
- **`web_download`** — a new tool that saves any URL (images, archives, assets) into the workspace: sandboxed by the same path resolver as every write, approval-gated, and journaled so rewind removes downloaded files too.
- **Proven against the live internet** — 8 real-network integration tests (GitHub/Reddit/X search, article extraction, raw source, blob rewrite, binary metadata) plus a real-model end-to-end run where the agent researched an obscure Rust crate it couldn't know the URL of: searched GitHub, found the right repo, wrote up its findings, and downloaded an image from the page — all verified against live data.

## v1.5.0 — The design release (2026-07-10)

A full production-level visual overhaul — every surface, both themes, zero feature regressions.

- **New design system** — the entire UI was rebuilt on a single token system: a warm charcoal dark theme and a paper-white light theme, a recalibrated juniper accent, four fixed elevation levels, a unified radius/motion scale, and one icon language (every text-glyph button replaced with stroke icons). Consistent buttons, close buttons, focus rings, and semantic colours everywhere.
- **Four palettes** — Settings → General now offers two dark schemes (**Warm charcoal** / **Cool charcoal**, the pre-1.5 look) and two light schemes (**Paper** / **Cream**, a softer low-glare tone). Following the system theme picks your chosen scheme for each appearance; switching cross-fades instead of snapping.
- **Reading typography** — answers, suggestion cards, the greeting, and About share a reading serif (Source Serif 4) for Latin text while Chinese stays in the system sans; code and UI chrome keep their own faces.
- **Code highlighting, your pick** — four switchable palettes for chat code blocks (GitHub, Atom One, Monokai, Nord); the block canvas follows the palette, which also fixes code readability in the light themes.
- **Settings, enriched** — LM-Studio-style layout with a fixed header per category and much more control: UI scale, send shortcut (Enter or ⌘/Ctrl+Enter, wired into both composers), reduce motion, answer text size, auto-titling, auto-load last model, a voice preview button, and a **Data dashboard** with live statistics tiles (conversations, messages, Code sessions, models with total size, KB documents/chunks, database size) plus one-click clear for chats and the knowledge base.
- **Native UI zoom** — interface scaling now uses the webview's real page zoom instead of CSS zoom, so 90–120 % scales cleanly with no white edges or overflowing panels.
- **Micro-motion** — one motion language across the app: segmented controls pop to their new selection, toggles overshoot playfully, settings categories cascade in, messages fade-rise on load, each Code-mode step card enters as the agent works, and menus/modals share the same enter/exit curves. Everything respects Reduce motion (in-app or OS-level).
- **Code mode scrolling, unpinned** — while the agent streams, scrolling up now releases the auto-follow instantly (no more being dragged back to the bottom); returning to the bottom re-engages it, and opening a session lands on the latest progress.
- **Fixes** — the model chip lost its `.gguf` suffix and redundant engine badge (now in Model info); the download dialog is properly centered; scrollbars follow the theme; a font-stack bug that rendered UI text in Times on macOS is gone.

## v1.4.0 — Code mode: rewind, trust controls & a smarter loop (2026-07-09)

- **Checkpoints & rewind** — every turn journals the original state of each file the agent touches. Hover any of your messages and hit **↩** to rewind: edited files are restored, created files removed, later messages dropped, and your message lands back in the composer to refine and re-send. (bash side effects aren't journaled — same trade-off as the majors.)
- **Fine-grained permissions** — the approval dialog gains **"Always allow …"**: a two-word command prefix (`npm test`, `cargo build`) or all file edits, remembered for the session. A permanent **command allowlist** lives in Settings → Code. Priority: Bypass → allowlist → session grants → ask.
- **Project memory** — `AGENTS.md` / `PROJECT.md` / `CLAUDE.md` from the workspace is auto-injected into the agent's system prompt each turn, so `/init` pays off on every future task and your conventions are always followed.
- **`search_code`** — ranked, meaning-aware code search ("where is login handled?") over the workspace: BM25 with camelCase/snake_case splitting, line-window chunks, per-file result caps. **`search_docs`** lets the agent consult your knowledge base (requirements PDFs, design notes) while coding — fully offline.
- **Whole files in one read** — the read budget now scales with the model's actual context window instead of a tiny fixed cap, so a 1,500-line source file is a single `read` call; only files that genuinely exceed the window page, with a footer stating the exact next offset. Pathological single-line files (minified JS) are truncated safely.
- **Loop breaker** — a model repeating the exact same call is intercepted (not executed) with a corrective hint and one hotter-sampled step to escape the pattern; a third repeat pauses cleanly with a Continue button. Lone `cd` commands are caught with an explanation (there is no persistent cwd), and the prompt now teaches path-based navigation.
- **Message queue** — type while the agent works: Enter queues your messages (removable chips) and they auto-run in order after the turn.
- **Background-jobs badge** — running background commands (dev servers) show in the header with one-click **kill all**, so nothing the agent started is ever orphaned invisibly.
- **Polish** — reasoning panel finishes as a clean "Reasoned" (no emoji); write/edit steps have distinct icons; a copy button on the final answer; the sampling **Reset** moved from Data into Sampling and now resets only sampling parameters (theme, skills, and GPU settings are no longer wiped).

## v1.3.1 — Code mode: background commands, web access & fixes (2026-07-09)

- **Background commands** — the agent can now start long-running commands (dev servers, big builds) in the background: it gets an id back immediately and keeps working on other steps, can check progress or stop the job, and is **told automatically the moment a job finishes** (exit code + output tail, shown as a step card). Background jobs are cleaned up when you switch workspaces.
- **Web access for the agent** — new `web_search` / `web_fetch` tools reuse the chat's key-less multi-provider search chain, so the agent can look up unfamiliar errors and library docs instead of guessing.
- **Fixed: `npm` / `node` "command not found"** — the agent shell now includes the common tool locations (Homebrew, cargo, nvm, bun, …) that Finder-launched apps don't inherit, so project tooling just works.
- **Fixed: thinking across model families** — Code mode now uses the same per-model reasoning control as chat (Qwen3's `/no_think` soft switch, the think flag for Qwen3.5+, Gemma 4's channel-marker normalization), so switching models no longer breaks the reasoning panel.
- **Fixed: Bypass mid-run** — flipping Bypass (or pressing `Shift+Tab`) while a task is running now takes effect immediately, releasing any approval dialog that was already waiting.
- The agent is now explicitly guided to run servers in the background (never through the blocking shell) and to verify unfamiliar errors on the web.

## v1.3.0 — Code mode: a local coding agent (2026-07-08)

- **Code mode** — a new top-level **Chat | Code** switch turns Chaty into an agentic coding tool that runs entirely on your local model. Point it at a folder, describe a task, and the agent explores, edits, and verifies the project by itself: reading and writing files, exact-string edits, glob/grep search, and shell commands — every step shown live as it happens.
- **Sandboxed & confined** — all file access is locked to the workspace folder you picked (path traversal and symlink escapes are rejected), and on macOS shell commands run inside a Seatbelt sandbox that can only write within the workspace.
- **Approve or bypass** — file changes and commands ask for permission first, with a real diff/content preview in the approval dialog (`Enter` allows, `Esc` denies); flip on **Bypass** to let the agent run autonomously.
- **Task plan** — for non-trivial tasks the agent lays out a live todo checklist and keeps it updated as it progresses (with a done-count), so you always know where it is.
- **It asks *you*** — when a decision is genuinely yours (naming, language, approach), the agent raises a choice dialog: pick with a click or number key, or type a custom answer.
- **Visible reasoning, adjustable depth** — thinking streams into a collapsible panel per step, and an **Off / Normal / Deep** switch controls how much the model reasons before each action.
- **Skills** — six built-in task templates (`/init`, `/review`, `/test`, `/fix`, `/explain`, `/commit`), each toggleable in Settings → Code, plus your own custom skills invoked the same way — alongside built-in slash commands (`/clear`, `/think`, `/bypass`, `/help`).
- **@-mention files** — type `@` in the composer to fuzzy-pick any workspace file and drop its path into the task.
- **One-click continue** — when a turn pauses at the step limit, a **Continue** button resumes from the actual progress (the agent remembers which tools it already ran); `Shift+Tab` toggles auto-approve, mirroring Claude Code.
- **Context under control** — a live context-usage ring, real-time token count and speed, and automatic compaction: older tool results and conversation are elided as the window fills, marked by a small "context auto-compacted" chip.
- **Persistent sessions** — Code conversations (including plans, steps, and diffs) are saved to the local database with their workspace, separate from your chats.
- **Tolerant of imperfect models** — argument aliases (`file_path`, `filename`, `old_str`, …), malformed-tool-call retries, and clear self-correction hints mean smaller models recover in one step instead of stalling.
- **Settings, rebuilt** — settings now open as a floating window with categories (General / Chat / Sampling / Model / Code / Voice / Data / About), including new Code options (max steps per turn, command timeout, skills) and an About page with the logo, version, and an update check.

## v1.2.0 — Knowledge base, leveled up (2026-06-30)

- **Import a folder** — alongside picking individual files, you can now choose a folder and Chaty ingests every supported file inside it *and its subfolders*. Hidden files and symlinks are skipped, unreadable/empty files are silently passed over; for large folders it asks for confirmation first and shows per-file progress (3/20). Files keep their **path relative to the folder** (e.g. `myproject/src/lib/ipc.ts`), so the knowledge base — and the model — can see the project's structure, not just bare file names.
- **Per-file citations** — when a knowledge-base answer draws several passages from the same document, you now see *one* citation for that file instead of one per chunk (e.g. `report.pdf` rather than `report.pdf · §3`, `report.pdf · §5`).
- **One-click knowledge-base report** — a new “Generate report” action immediately writes a cited overview of your whole knowledge base (NotebookLM-style — no topic to type). It reads your *local documents only* (fully offline), grounds on per-file content plus the folder structure, cites one reference per file, and exports to PDF or Markdown.
- **Tidier knowledge-base actions** — the footer buttons (Add documents / Import folder / Generate report / Generate podcast) are now a clean 2×2 grid instead of an overflowing row. A **Clear all** action empties the whole knowledge base in one step.
- **KB report wording** — the report view no longer borrows Deep Research's web-search labels (no “Searching the web…”, no “0 searches” counter); it shows the document count and knowledge-base-appropriate status instead.
- **Many more knowledge-base file types** — added **.docx** (Word) and **.xlsx** (Excel, read as tab-separated rows) plus a broad set of text/code/markup/config formats: `js/ts/jsx/tsx`, `css/scss/less`, `py/rs/go/java/c/cpp/cs/rb/php/swift/kt/scala/sql/sh`, `vue/svelte`, `xml/yaml/toml/ini`, `tex/rst/mdx`, and more — on top of the existing PDF, text/markdown/HTML/CSV/JSON and images (OCR).
- **Fixed: knowledge base pinned a large chunk of memory (macOS)** — the embedding model (bge-m3, ~730 MB) was offloaded to the GPU, which on macOS left it stuck in *wired* memory that never came back, even after ejecting the chat model. It now runs on the CPU on macOS (no wired memory), and a full **Eject** frees it as well. Other platforms are unaffected.
- **Fixed: confirmation dialogs opened from a panel** (e.g. *Clear all*) could appear *behind* that panel and be unclickable — they now always sit on top.

## v1.1.2 — Smoother streaming + polish (2026-06-27)

- **Much smoother streaming** — a reply no longer re-renders the whole conversation on every token: messages are memoized so only the one being written updates, and those updates are coalesced to one render per frame. Long answers — and typing while a long conversation is on screen — no longer stutter.
- **Resizable sidebar** — drag the conversation sidebar's right edge to set its width (double-click the handle to reset); the width is remembered between launches.
- **Native dropdowns** — the in-app option menus (Deep Research depth, voice picker) now use Chaty's own themed dropdown instead of the OS-native `<select>`, so they match the app on every platform and in both themes.
- **Cleaner light theme** — removed the gray drop-shadow halo under menus and popovers in light mode; they now read as elevated via crisp borders.
- **Small polish** — the model menu's refresh button is larger and easier to hit, and the composer's **＋** button now spins as its menu opens and closes.
- **Fixed: deleting the conversation you're viewing now clears the chat area** even mid-reply (it previously left the old messages on screen until you clicked “New chat”).
- **Fixed: Deep Research reports could show the title twice** when opened to print (the topic was prepended even when the report already started with its own heading).
- **Canvas pages** opened in the browser now go to their own folder instead of the Deep Research `reports` folder.

## v1.1.1 — Links open in your browser (2026-06-26)

- **Fixed a serious bug**: clicking a link in a model's reply (or anywhere — Deep Research reports, the knowledge base, etc.) navigated the in-app window to that page and left the app stuck/unusable. Links now open in your default system browser, as expected.

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
