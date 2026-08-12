# Changelog

## v2.0.8 — Imperfect models, met halfway (2026-08-10)

Started as v2.0.7 hotfix rounds and grew into its own release: a wave of
model-adaptation work — broken quants healed, mute models explained,
alien reasoning markup parsed — plus coding-and-canvas diagnostics that
point at real lines instead of looping or shrugging.

### Model adaptation: heal, diagnose, adapt

- **A mute model now explains itself.** A generation that degenerates into
  pure whitespace (the broken-conversion signature) is aborted after 32
  tokens with a plain-language diagnosis streamed into the chat, instead
  of a silent screenful of nothing — and model families with known engine
  compatibility gaps (MiniCPM5 GGUFs: even the official file degenerates
  on the bundled llama.cpp; MiniCPM 1–3 exported as plain llama) get a
  warning at load that points macOS users to the MLX builds, which run
  perfectly. Traced end-to-end against upstream llama.cpp with the owner's
  own downloads; healthy models verified unaffected.
- **LFM2.5 MLX quants load now — two converter artifacts, two heals.**
  The community quants ship a text-only weight tree with every key wearing
  a VLM-style `language_model.` wrapper (the loader died with
  `keyNotFound(model.embed_tokens.weight)`), and a config that says
  `intermediate_size` where the engine's LFM2 reader only understands
  `block_ff_dim` (weights then failed shape validation). Chaty now strips
  a UNIFORM wrapper in place — safetensors headers are renamed and padded
  back to their exact original length, so a multi-gigabyte model costs a
  kilobyte-sized write and the tensor data never moves — and mirrors the
  FFN width into the field the engine reads. A real VLM always carries
  vision-tower keys, so the uniform-wrapper condition never touches one.
  Verified on the failing 2.6B (loads, chats, computes) with the
  previously-working 8B MoE confirmed byte-untouched.
- **A vision model missing its processor config now heals itself.**
  Community MLX quants sometimes ship a VLM checkpoint without
  `preprocessor_config.json` — the load died with a bare
  `configurationFileError` and the model was unusable. For families whose
  preprocessing values are architecture constants (Qwen3-VL, and now the
  whole Gemma 4 line) the sidecar synthesizes a minimal config next to the
  weights and vision simply works — verified end-to-end on a real 26B
  Gemma 4 quant that used to fail: it now loads in seconds and answers
  both text and image questions correctly. Families without a healing
  recipe degrade to text-only chat with a plain-language notice instead of
  refusing to load.
- **Gemma 4's reasoning chain parses completely now, everywhere.** The
  channel-style thought markup Gemma 4 (and Harmony) stream had three
  leaks: a runaway generation that re-opens its thought channel spilled the
  second round of reasoning into the visible answer; a close marker
  followed by prose starting with "final"/"thought" swallowed that word as
  markup; and a generation cut mid-thought showed the whole unfinished
  reasoning as the answer (Code mode's step text was the loudest victim).
  The normalizer now walks every reasoning span sequentially, half-typed
  markers are held back while streaming instead of flashing `<|chan` into
  the answer area, and Deep Research + Podcast — which stripped reasoning
  with a bare regex — now go through the same normalizer. Locked by a
  12-case unit suite over the real template shapes and re-verified live in
  Code mode on both Gemma 4 engines (GGUF E4B and MLX 26B): thought panel,
  prose and finals all clean.

### Coding & Canvas: errors that point somewhere

- **No canvas error stays anonymous — the whole map, not just the easy
  half.** Verified class-by-class against a 17-way error matrix on both
  engines (Chromium replica + a WKWebView hand-pass): classic scripts run
  inside a line-preserving guard so TOP-LEVEL throws finally name their
  line (a reported error-collection page turned out to be one giant
  invisible SyntaxError); inline `on*=` attributes are rewritten in place
  (which also cured WebKit's off-by-one for attribute-compiled code);
  dynamically-injected handlers — `el.onclick = fn`,
  `setAttribute('onclick', …)`, `innerHTML` with handlers, string-form
  timers — are instrumented at their entry points; observers
  (Mutation/Resize/Intersection/Performance), media and Worker `on*`
  properties and `requestIdleCallback` join the trap. The console itself
  grew up too: `console.error(new Error(…))` shows the real stack instead
  of `{}` (WebKit stacks carry no message line — one is prepended),
  objects and DOM nodes render DevTools-style, every console line carries
  its `@canvas:LINE` call site, and eval'd code is labeled `canvas:eval`
  instead of impersonating line 1. Scripts whose wrapping would change
  scoping semantics ('use strict', let/const shared across scripts,
  modules) are left alone and named in the Fix digest. And a repeated
  error folds into a ×N badge devtools-style instead of flooding the
  console — the frame budget counts unique lines, so one broken interval
  can't silence every later, different error.
- **Canvas console errors now point at the exact line.** WebKit anonymizes
  every uncaught error inside the sandboxed preview to a bare
  "Script error." — no line, no stack, and several distinct bugs collapse
  into identical, useless entries. The preview now routes the async entry
  points where interaction bugs live (timers, rAF, microtasks, event
  listeners) through a same-realm guard that catches each error with its
  full stack, reports it with USER-source line numbers (the injected
  shims' own line offset subtracted), and rethrows unchanged. Duplicate
  follow-ups are dropped on both engines (WebKit's anonymized echo,
  WebView2's detailed one), shim-internal frames never leak into stacks,
  and the remaining anonymized class (top-level statements, inline on*=
  handlers) is named in the Fix digest so the model audits the right
  places. The whole injected stack (storage shim included) now rides
  inside ONE counted block, so the calibration survives every wrapper —
  the first cut left the storage shim outside it and a 4-line repro
  reported line 16. And a page that throws several errors says so in the
  fix banner ("… — N errors total") instead of showing only the first;
  the Fix button already handed the model every error. Proven by behavior
  unit tests, a live-browser replica of the sandboxed preview, and a
  hand-verified WKWebView pass: throws reported once each, at their exact
  source lines (inline on*= attribute code can sit one line high — a
  WebKit quirk of attribute-compiled scripts).
- **`browser_navigate index.html` now opens the file instead of looping.**
  Models routinely pass a bare filename when asked to test the page they
  just wrote — and the resolver checked it against the app process's
  working directory (never the agent's workspace), so `index.html` became
  `https://index.html`, a DNS error, and an endless retry loop. Relative
  paths now resolve against the workspace, a bare name with no path finds
  its unique match anywhere in the project (`app.html` → `dist/app.html`;
  several matches ask which one), a file that resolves nowhere gets a
  plain-language error teaching the right form instead of a DNS guess, and
  scheme-less `localhost:8000` finally gets `http://` instead of an
  `https://` guess that dies on TLS. Real URLs, absolute paths and
  websites behave exactly as before — proven by a resolver unit suite and
  a live-Chrome pass over all four shapes.

- **The tiktok-video skill now follows its upstream on its own.** The
  skill's support files (scripts, references, examples) have always been
  byte-for-byte mirrors of the public upstream repo — and every upstream
  fix used to wait for a Chaty release just to ride along. Online machines
  now check upstream quietly (at most once a day), download complete
  verified trees into app data, and `use_skill` materializes the freshest
  layer; offline or failing networks fall back to the bundled files
  without a sound, incomplete or oversized fetches are rejected outright,
  and `CHATY_SKILL_SYNC=0` turns the whole mechanism off. The model-facing
  SKILL.md stays bundled (it is a Chaty-owned rewrite; script CLIs are
  stable upstream). Proven end-to-end against the live repo — the very
  first sync picked up an upstream Ken Burns judder fix that no longer
  needs a repack to reach users.
- **tiktok-video skill: the video speaks the USER'S language.** Upstream
  sync — `lang` no longer silently defaults to Chinese: the storyboard
  must state it explicitly (the pipeline stops with a pointed message
  otherwise), and the skill doc now opens with the rule: an English brief
  means an English video — hook, captions, badges, voice — whatever
  language the docs themselves are written in.

## v2.0.7 — Falado em português, auditado por completo (2026-08-08)

### The UI speaks Brazilian Portuguese

- **Community-contributed pt-BR localization** by [@magisph](https://github.com/magisph)
  (#6, #7): 194 UI strings covering the core UI, settings, Code mode and
  Canvas — with a new locale architecture where community languages are
  optional per key and fall back to English, so partial coverage never
  shows a blank. The language switch is now table-driven; the agent/model
  layer deliberately stays zh/en (prompt quality follows model training
  data, not the UI language). `npx tsx scripts/l10n-status.mts pt` shows
  what's left, and the README now ships in Portuguese too.

### A full-app audit, before users could find any of it

- **Downloads can no longer hang forever.** Six HTTP clients (model
  downloads, the repo-metadata call, the embedding-model fetch, the update
  check, the installer download) had **no timeout at all** — a stalled CDN
  read as "download stuck at 43%" until restart, an unreachable GitHub
  held "checking for updates…" indefinitely. Streaming paths now use a
  connect + between-chunks timeout (a big model on a slow line is still
  fine; a silent connection errors out), metadata calls a plain one.
- **The frontend test suite finally runs in CI.** 577 unit tests — the
  agent-loop harness, the wrap-up gates, the i18n contract, the
  skill-bundle drift lock — had never been wired into CI; guards designed
  to turn CI red only ever ran on the dev machine.
- **Windows and Linux now RUN the tests before a release, not just compile
  them.** Development happens on a Mac; the platform jobs executed nothing.
  Both now run the full no-GPU test suite (the Windows job exercises the
  tasklist pid checks, APPDATA paths and the GPU crash guard for real),
  the frontend job builds and tests on Windows too, and a new
  Windows-native orphan-browser sweep test covers what the Mac can't.
  Line endings are pinned so a CRLF checkout can't fake a drift failure.
- **Images inside Chrome-made PDFs finally extract.** Every PDF produced by
  Chromium's "print to PDF" (web receipts, statements, saved articles — a
  huge share of real documents) had its embedded images silently dropped:
  the PDF library's own decompressor rejects those streams. Chaty now
  inflates them itself, so the knowledge base and chat attachments can see
  charts inside such files. Found by finally running the repo's
  real-machine test suite — 60 ignored integration tests across web fetch
  (13 live sites), search, downloads (xet fallback + cancel), MCP (live
  server + store certification), browser CDP, llama agent E2E, MLX
  (8 sidecar lifecycle/vision tests), PDF/OCR — all green after fixes.
- **The knowledge base and voice pipelines got their first real tests
  ever**: a semantic probe through the actual bge-m3 embedder (kitten≈cat
  must beat kitten≈carburetor) and a full Kokoro-TTS → Whisper-STT
  round-trip — both green, both staying in the suite.
- **Headless-browser hardening for new Chrome**: the standard
  anti-throttling launch flags plus CDP focus emulation (the same defaults
  Puppeteer ships), guarding screenshot capture against Chrome ≥150's
  frame-parking on static pages.
- **Real-surface smokes now part of the audit**: the GGUF/llama.cpp path
  (load → generate → cancel mid-stream → regenerate), the key-less search
  chain, article extraction, and a full UI walkthrough in English and
  Portuguese (every settings tab, Code mode, Canvas) — all green.
- Engineering debt swept: 29 dead i18n keys deleted (translators would
  have translated them for nothing) and a **dead-key CI guard** added — a
  defined-but-unreferenced key now fails the build, closing the asymmetry
  that let feature rewrites orphan their strings silently (the guard's
  first run caught 8 keys the hand audit had waved through). The last
  copy-pasted browser UA was consolidated, and skills materialized into a
  workspace now carry a `.gitignore` so derived scripts never pollute the
  user's repo.
- **The new platform defenses paid for themselves on their first flight**:
  Windows CI's first real test run caught the Windows `build_command`
  skipping the npm/electron cache redirect that macOS and Linux apply —
  fixed, three platforms now behave identically.

## v2.0.6 — One sentence in, a finished video out (2026-08-06)

### New official skill: `tiktok-video` — 一句话出成片

- **One-line brief → finished vertical video.** The coding agent can now
  deliver complete TikTok/抖音/Shorts videos: it writes the script and
  storyboard, downloads free stock footage (Openverse, Wikimedia, NASA;
  Pexels/Pixabay with a key), synthesizes the voiceover (Chinese & English,
  word-level timestamps), renders karaoke captions, picks CC-BY music, and
  composes a loudness-normalized 1080×1920 MP4 with ffmpeg — attribution
  block included. Requires `ffmpeg` + `python3`; zero API keys needed.
- **Directory-shaped skills.** Official skills can now ship runnable support
  files (scripts, references) alongside their SKILL.md. The scripts are
  materialized into `<workspace>/.chaty/skills/<name>/` the first time the
  skill is used — content-hash keyed, so upgrades refresh and unchanged
  bundles cost one read — while model context only ever carries the one-page
  procedure. A user skill with the same name still shadows the whole thing.
- **Editor-grade review is part of the recipe.** The pipeline halts for a
  storyboard-vs-assets review (`view_image` on generated contact sheets for
  vision models), and `check.py` audits every asset's source title against
  the scene's keywords — an off-topic pick (an oil painting titled nothing
  like your subject) is flagged with a ready-to-run refetch command, which
  is how a text-only model reviews footage it cannot see.
- **Pipeline v2 (upstream merge): the videos stop feeling static.** Scenes
  are now multi-shot — each keyword is one visual and the video cuts every
  ~3 seconds — with crossfade transitions, a synthesized transition whoosh
  (`sfx`), stamped listicle badges (`badge: "第1名"`), an opt-in sticky
  topic bar, and a raised publish bar (45–75s, 6–9 scenes, 2–3 shots each).
  Refetch got shot-precise (`--scene 3 --shot 2`), and a refetched shot
  remembers the query that chose it, so a hand-picked replacement never
  re-flags against the storyboard's original words.
- **Pipeline v3 (upstream merge): rhythm and rigor.** Cuts snap to the
  music's beat grid (measured: 10/10 cuts within 15ms) — BGM can now be
  style-searched from CC catalogs ("trap", "lofi chill"; ccMixter/Jamendo,
  rhythm-screened so beatless tracks are skipped) with the classic mood
  table as default, plus 抖音-signature `spedup`/`slowed` vibes. And a
  verify-first workflow: every concrete claim in the script must be checked
  against live sources before writing, with URLs recorded in the storyboard
  and surfaced in the report for the user to audit.
- **Real video footage without an API key (upstream merge).** Three keyless
  motion sources join the providers: Wikimedia Commons video transcodes,
  NASA's public-domain video library, and the Prelinger historical archive —
  with negative-word query filtering ("lightning storm -satellite") to dodge
  the satellite-map flood, and an `"approved": true` manifest override so a
  hand-verified shot can silence a false title flag (eyes outrank text).
- **The off-topic audit fires at fetch time, not just delivery time.**
  `assets.py` prints the same OFF-TOPIC flags the moment assets land —
  before a compose cycle is spent on a wrong pick — and the flags print
  ahead of the preview-sheet build so a sheet failure can't eat them.
- Proven end-to-end with real-model sessions (35B): skill discovery →
  use_skill materialization → venv setup → pipeline → delivered 27–41s
  videos, three for three; the bench driver ships as
  `bench/coder/tvbench.mts`.
- Audit hardening: the skill bundle is drift-locked by test (edit
  `resources/skills/` without rebundling → CI red), `__pycache__`/`.pyc`
  can never ride the bundle again, and the Windows venv path
  (`.venv/Scripts/python`) is called out in the procedure.

### Full-app audit

- **Orphaned headless browsers are reaped at startup.** Every exit path
  that skips destructors — the exit handler's `_exit()`, a crash, a killed
  bench bridge — used to leave the headless Chrome tree running and its
  profile dir behind (16 helper processes and 14 profile dirs stood on the
  author's machine when this was found). Startup now kills any browser
  whose owning Chaty is gone — by creator-pid liveness, so a concurrently
  running instance keeps its own — and removes the leftover profiles.
- **Chat history that fails to save says so.** A failed conversation/message
  write used to vanish into the console; the message sat in the UI and was
  gone after restart. Now it lands in the error log and warns once per
  conversation — same treatment code sessions already had.
- Audited clean: the TS↔Rust IPC contract (three-way: calls, commands,
  registration), i18n key completeness, every non-test `unwrap` in hot
  paths, JSON.parse guards, event-listener balance, async-blocking sleeps,
  store/download error propagation and partial-file handling.
- **Unit tests can no longer write the real error log.** `append_error`
  compiles to a no-op under `cargo test` — after a third test run stamped a
  false "gpu crashed" entry into a real user log, the guarantee moved from
  discipline to the compiler. The GPU guard's state machine is pure now;
  its Windows caller does the logging.
- The video skill's asset fetch dedupes by content hash, so the same image
  arriving via two different URLs can't appear twice in one video.

## v2.0.5 — Functions or it didn't happen (2026-08-05)

v2.0.4 made the agent ship an app that launches. This release makes it ship
an app that **works**: every basic function must be executed — tested,
clicked, or invoked with real inputs — before the turn may end, on every
stack. Proven the hard way: a six-wave, five-stack matrix of real-model
deliveries (SwiftUI desktop, browser app, Python CLI, Node API, Rust CLI),
each product rebuilt, launched, and function-tested by hand.

### The functional bar

- **Executed proof or it isn't done.** An app-scale delivery with a green
  build but zero test runs, zero real invocations, and zero browser
  walkthroughs is stopped at the door — twice if needed. Green test suites,
  real CLI runs, `curl` probes, and browser click-throughs count; compile
  receipts don't.
- **A single-file page is an app too.** Delivering an `.html` now asks for
  the browser walkthrough even with no dev server running — the wave-1
  escape (a 387-line todo app shipped sight-unseen in two steps) turned into
  wave-2's model clicking through every feature and reloading to prove
  persistence, unprompted.
- **New official skill: `debug-playbook`.** The right debug method per
  symptom — compile errors by file:line, crashes by stack frame, logic bugs
  by failing-test-first — plus per-stack functional verification recipes,
  the `curl -f` 404 trap, route-params-are-strings, and a cleanup rule so
  test data doesn't ship.
- **The mac-app skill grew a functional chapter.** Core logic in plain
  testable types, an XCTest target wired into `Package.swift`, platform
  pinned to macOS 14 up front, and the two SwiftUI patterns small models
  reliably fumble (editable list bindings, master–detail selection) as
  copy-ready snippets.
- **Swift joins the post-edit syntax gate.** A brace-broken file now warns
  the moment it is written — syntax-only parsing is banned as verification
  but is exactly right as a tripwire.

### The delivered app must be the delivered code

- **Stale artifacts stop counting.** Editing sources after the last build
  and then packaging/launching the OLD binary (the minesweeper session:
  `swift test` refreshes debug, the release binary in the bundle stayed
  old) now draws an immediate warning, clears nothing, and the wrap-up gate
  demands rebuild → re-package → re-launch before delivery — release paths
  require a release-flavored build, test-file edits don't stale anything.
- **No-op repeats soft-lock instead of killing the turn.** Identical
  re-sends of a failed command, an unchanged file write, or an unchanged
  file read get step-consuming redirections (fix the code at the file:line /
  move on / act) — the turn survives to actually finish.
- **Hand-rolled `.xcodeproj` gets intercepted at the keystroke** and steered
  to SwiftPM (with husk cleanup); the mac-app skill now carries the
  three-target `Package.swift` template (Core lib + executable + tests),
  the `public`/`import` cross-module rules, and the platform pin — the
  recurring Swift scaffold deaths, each mechanized away.

### Windows: a broken GPU driver no longer takes the app with it

- **Crash-marker fallback (issue #5).** A model load that kills the process
  (Vulkan driver abort) is detected on the next start; GPU offload is
  blocked persistently, the load reruns on CPU, and a notice explains what
  happened. One crash maximum, then it just works.
- **Automatic error log.** Panics, front-end errors, model-load failures
  (GGUF and MLX sidecar, every OS), and crash post-mortems are recorded to
  `logs/chaty-error.log` (size-capped): open it from Settings → General,
  and the new GitHub issue template asks reporters to attach it. On macOS,
  a native crash that kills the process before any hook can run (Metal,
  sidecar, WebKit) is picked up on the next start from the system's
  DiagnosticReports and pointed to from the log.

### Delivery discipline tune-ups

- Leaked planning prose ("当前编译错误…解决方案…") is intercepted before it
  can pose as a final answer — widened trigger set from live transcripts.
- Functional receipts, walkthrough detection, and the second-shot gate all
  ride the same wrap-up machinery; small tasks stay entirely friction-free
  (verified: a simple script task runs 6 steps, zero interventions).

## v2.0.4 — Ship something that runs (2026-08-04)

A release about the difference between "it compiles" and "it works". We took
one real failed delivery — a calendar app that shipped with compile errors
after the model waved itself through with a syntax check — reproduced it
twenty-four times against the live model, and turned every failure into a
mechanism. The bar at the end: a **packaged app that actually launches**,
built the incremental way, on every stack.

### The deliverable is a running app

- **New official skill: `mac-app`.** The full recipe for macOS app delivery:
  scaffold → verify → feature → verify, SwiftPM-to-`.app` packaging, and a
  launch check that must print `LAUNCH OK` before the turn may end. The loop
  surfaces it the moment an app entry point is written, and the wrap-up gate
  refuses an app delivery with no `.app` bundle in the tree.
- **Feature → verify → next feature.** After four unverified source files the
  loop reminds once; after a red result it names exactly the files edited
  since the last green check ("suspect these — don't touch what already
  passed"). Small tasks never see either.
- **`validate_change` now speaks Swift, TypeScript, and Go.** Xcode projects
  get a real `xcodebuild`, SwiftPM gets `swift build`, bare Swift gets a
  whole-set typecheck; TS projects without tests get `tsc --noEmit` (vitest
  transpiles without typechecking); Go modules get `go build ./...`. Package
  manifests and `*Tests` are excluded from bare typechecks — no more phantom
  errors on healthy projects.
- **Verification can't be faked.** A failed build is a debt, not a receipt:
  only a green run clears the run-check ledger. Syntax-only probes
  (`swiftc -parse`, `node --check`, `py_compile`, `ruby -c`, `php -l`) and
  `--version` calls don't count, and the model is told so the moment it tries.
  Re-sending a failed build verbatim gets the honest advice: the error lives
  in the code at the file:line named, not in the arguments.

### The sandbox stops sabotaging builds

- **Xcode, SwiftPM, and macro builds work inside the agent sandbox.**
  DerivedData writes are allowed, SwiftPM and the Swift macro plugin server
  no longer die on nested sandboxing (`SwiftData` / `@Observable` code used
  to false-red with "external macro implementation could not be found"), and
  npm/electron caches are redirected to writable temp dirs — `npm install`
  and electron's postinstall download now succeed on the first try.

### The agent wastes fewer turns

- **Default step budget: 32 → 64.** App-scale one-shots kept dying one error
  from green at the old ceiling; existing installs on the old default are
  migrated automatically. Two steps before the ceiling the loop orders a
  wind-down: verify and deliver, don't start new files.
- **Trance breakers.** Plan-echo now shows a literal first-action call;
  re-sending an identical plan soft-locks the tool instead of killing the
  turn; five consecutive look-only steps get an "act now" break; a
  near-identical full-file rewrite is accepted with a tip instead of bounced;
  an `rm` that swallows files written this turn triggers an immediate
  accounting; and an "answer" that opens with planning prose ("让我先…" /
  "The user wants me to…") is intercepted once — act, or write a real
  summary.

### Small fixes that rode along

- The updater now ranks `2.0.4-beta` below `2.0.4` instead of equal to it.
- MCP over HTTP parses SSE events whose JSON spans multiple `data:` lines.
- A second call to a busy MCP server now says "busy — retry", not the
  misleading "not connected" that sent models off reconnecting.

### The sandbox was innocent (audit follow-up)

- **Permission errors get correct attribution.** When a command hits
  "Operation not permitted", the loop now explains on the spot what the
  sandbox actually restricts (writes outside the workspace — nothing else)
  and that screen capture / automation denials are macOS privacy
  authorization, not the sandbox — so the model adapts instead of declaring
  the task impossible and switching stacks.
- **A blank browser refuses to play system camera.** `browser_screenshot` /
  `browser_snapshot` with no page open now return a teaching error: browser
  captures show the embedded web page only, never native windows — verify a
  native GUI with the launch + stay-alive check.
- **Parallel half-implementations get called out.** Starting a second app
  stack while the first is still in the tree triggers a warning: fix the
  existing one, or state why and delete it — one complete implementation per
  delivery.
- **Byte-identical rewrites soft-lock instead of killing the turn**, with a
  "that file is already on disk — move on" redirect.
- **A failing session save is no longer silent.** If the transcript cannot
  be written to disk you now get one clear warning instead of discovering
  the loss later.
- **Raw `swift build` / `swiftc` now just work in the agent shell.** The
  nested-sandbox defusal that `validate_change` already used is applied to
  the model's own commands too — no more "SwiftPM has sandbox issues, let me
  try Electron" defections.
- **Sessions exist from the first message.** Sending the first message now
  writes the session to disk, names it, and shows it in the sidebar
  immediately; every step persists (debounced) during the run — a paused or
  crashed first turn no longer loses the whole transcript.
- **A pipe can't launder a failed build anymore.** `swift build | tail`
  exits 0 through the pipe; an exit 0 whose output carries compiler-failure
  signatures no longer counts as verification.
- **The whole toolchain class is whitelisted, not one victim at a time.**
  Cargo's registry, rustup, Go modules and build cache, Gradle/Maven, Dart
  pub, user gems, bun/pnpm/yarn/deno stores, pip/uv/poetry caches,
  Playwright/Puppeteer browser downloads, node-gyp, CocoaPods, composer —
  writes to these tool homes and caches are now allowed inside the agent
  sandbox (live-verified: cold-crate `cargo build` and
  `gem install --user-install` both worked end to end). Raw `xcodebuild`
  additionally gets script-phase sandboxing and Swift macro flags threaded
  through automatically. The hard edge is unchanged: Documents, dotfiles,
  `~/.ssh`, system paths, and other apps' data stay off limits.

## v2.0.3 — Cards on the table (2026-08-02)

A small release with two very tangible things: the agent learns a new craft,
and the canvas stops forgetting.

### New official skill: cardlet

- **Text in, carousel cards out.** The `cardlet` skill turns notes, threads,
  or an article into polished social-media card images (小红书 / Instagram /
  square / story sizes) through the Cardlet HTTP API — plain bash + curl, no
  browser involved, PNGs saved straight into the workspace. Ships enabled;
  toggle it in Settings → Code → Skill files. Free tier renders with a small
  watermark; a Cardlet Pro code removes it and unlocks the extra templates.
- The `use_skill` correction example now names a skill that actually exists —
  the made-up example taught small models to call a tool that wasn't there.

### The canvas remembers

- **Canvas version history survives an app restart.** Every design-canvas
  session (v1, v2, …) now mirrors to disk as you iterate; reopening the same
  document after a restart brings the whole version rail back. The newest
  hundred sessions are kept; persistence failures never interrupt the canvas
  itself.


## v2.0.2 — Spin control (2026-07-29)

A patch release about wasted rounds. We instrumented the agent loop, ran the
current model against real tasks, and fixed every failure shape the raw
transcripts showed — plus the delivery habit the webapp release didn't cover.

### The agent stops spinning

- **The parser now reads eleven more real-world tool-call shapes.** Local
  models fuse XML attributes into JSON (`{"name="grep"…}`), drop the
  `arguments` key, stutter the opener, ship the arguments as a second object —
  or as a whole second `<tool_call>` block. Every shape came from a raw
  transcript dump; every one now parses with the arguments intact. A sweep
  test drives all shapes across every tool with required arguments, so no
  repair is tuned to one lucky tool.
- **An empty completion is a glitch, not an answer.** Zero tokens — or an
  empty think block and nothing else — used to end the task silently with a
  blank reply. Now it retries hotter; three in a row pause with a real
  message and a working Continue.
- **The missing-argument ladder learned to forgive.** A tool disabled after
  empty-argument calls re-arms once the model makes real progress with other
  tools — with a total cap so the second chance can't become an infinite one.
  Seven more tools (outline, glob, bg_output, bg_kill, browser_click,
  browser_type, browser_eval) join the guarded set, with alias awareness so a
  legitimate `code`/`expr`/`expression` never gets blamed.

### Run it before you ship it

- **Code that was never run gets one honest question.** Ending the turn after
  writing real code (two files, or about a screenful) with zero execution —
  and `ls` does not count as execution — earns exactly one nudge: validate,
  run it, or say why a run isn't needed. Small single-file edits, docs and
  config never trigger it; it fires once, then your answer stands.

### Your budgets, your rules

- **Think budget (Settings → Code).** A hard per-step ceiling on thinking
  tokens. Over budget the think block closes gracefully: the reasoning stays
  in context and the model acts on it — nothing discarded. The old built-in
  3000-token cutoff that beheaded legitimate long reasoning is gone; if you
  want a ceiling, it's yours to set.
- **Per-step output limit (Settings → Code).** The generation budget per
  agent step is now yours too: 0 = auto by think depth, any value clamped to
  the context window.

### Small fixes

- An open dropdown in Settings no longer hides under the cards below it (the
  pane entrance animation trapped its stacking order — the voice picker was
  the visible victim).
- Chat now always sends ONE system message. Qwen3.5/3.6 chat templates
  reject a second system turn, so any two of attachment / web-design mode /
  custom system prompt / web search together failed on MLX with
  "System message must be at the beginning" — the fragments merge now
  (and GGUF models get a cleaner, standard prompt out of it too).
- An MLX runtime error no longer kills the engine. Metal-level failures
  (out-of-memory on a big model, mid-eval errors) used to crash the whole
  sidecar ("MLX 引擎意外退出"); they now surface as a plain per-round
  "generation failed" with the real reason, the model stays loaded, and a
  retry just works.
- Screenshots respect the memory you actually have left. When the Metal
  working-set ceiling minus the loaded weights leaves under 8 GB, the vision
  pixel budget halves to 0.5 MP — the same screenshot round measured 43%
  faster with element reading intact, and the encode transient that pushed a
  35 GB model over the line on a 48 GB box halves with it. The same model on
  a bigger machine keeps the full budget; a mid-size model on a small box is
  protected too.
- Tall pages arrive in legible pieces, not one blurry strip. A full-page
  screenshot of a many-screen page used to squeeze into ~350 px of width —
  the model misread prices off it. It now splits into viewport-sized
  segments (boundaries seek the blank gaps between sections, so nothing is
  ever cut in half), every pixel kept, each segment sharp. Snapshot gets its
  proper billing for quick viewport re-checks.
- The camera waits for the page to finish appearing. Reveal-on-scroll
  animations raced the capture and whole sections photographed blank — the
  model then faithfully reported content that "wasn't there". Animations are
  frozen at their end state before the shot; on the page that exposed all of
  this, element extraction went from partial to perfect.
- The canvas preview scrollbar finally matches the page. Native subframe
  scrollbars answer to the top document's color scheme, the window, and the
  stage behind them — never to the previewed page — so a dark page kept a
  glaring white bar through two earlier fixes. The preview now paints its
  own: track in the page's background, thumb in translucent ink, on any
  page, in any theme. The app also declares a proper color-scheme per theme,
  so native widgets everywhere stop assuming light mode.


## v2.0.1 — The webapp workshop (2026-07-28)

A patch release focused on one thing: making Code mode genuinely good at
building and debugging local web pages. Every fix below came out of live
walkthroughs — reported, reproduced, fixed, and re-verified the same night.

### The agent stops stalling on dev servers

- **Foreground dev servers auto-move to the background.** Start a server with
  plain `bash` and ~10s later it's a tracked background job — output so far,
  job id, and how to continue, instead of a 120s stall that ended with the
  server killed. Detection is two-track: server banners are the fast path, a
  listening-socket probe is the ground truth (banner-silent servers like a
  piped `python3 -m http.server` are caught too).
- **No more zombie servers.** A shell that exits after `server &` used to
  leave the server squatting its port forever. Survivors of the command's
  process group are now adopted as `[detached]` background jobs — visible,
  killable, and reaped on workspace switch or app exit.
- **`browser_refresh`** — the missing reload verb. True hard refresh
  (cache ignored) with the same page digest as navigate; after editing local
  files, re-screenshotting the stale DOM proves nothing and the agent now
  knows it.
- **Page errors come to the model by themselves.** On pages you own
  (localhost / local files), browser interactions auto-attach new console
  errors, exceptions and dialogs to the tool result — deduplicated, and
  labeled honestly (a dialog your click triggered is not an "error"). Other
  people's websites stay noise-free; `browser_console` still shows everything.
- **Plans stop being decoration.** `update_plan` echoes live statuses back to
  the model, and a one-shot wrap-up check bounces a premature final answer
  when todos are unfinished or page edits were never walked in the browser.
- **Fewer stuck loops.** Empty-argument tool calls get an escalating
  correction ladder (example → switch tools → tool disabled this turn) and are
  never executed; repeat-intercept advice no longer suggests the very tool
  being repeated; the tool-call parser survives real local-model output shapes
  (flat fields with an empty `arguments`, missing closing brace) instead of
  burning rounds on them.

### Canvas: a debugger you can trust

- **The console pipeline reports again** — a regression that silently killed
  error capture inside the preview is fixed, with a syntax gate on every
  injected shim so that class of bug can't return.
- **Complete errors, one honest Fix.** Error entries carry full stacks; the
  Fix button bundles every distinct problem as a numbered list and instructs
  the model to fix all of them in one pass. A parent-side syntax precheck
  recovers the real SyntaxError text (with the script-block index) that
  WebKit anonymizes to "Script error." inside the sandboxed preview.
- **The red badge means broken.** Uncaught exceptions, unhandled rejections,
  failed resources and syntax-precheck hits light it; `console.error()` /
  `console.warn()` prints render in the tab but never badge.
- **Hand-editing feels right.** Selection no longer drifts off the glyphs on
  long documents (integer line metrics, no soft-wrap), and the caret's line
  carries a highlight stripe.
- **Close means clean.** Reopening a canvas no longer stacks duplicate
  console entries — closing the panel resets its console state.

### Small fixes

- The MCP server toggle knob sits correctly on its track again (Settings →
  Code).
- The Canvas syntax precheck now compiles only classic scripts —
  `type="module"`, JSON and template blocks are pages, not faults, and no
  longer light a phantom badge or feed bogus entries to Fix.
- The agent's PATH resolves the newest nvm Node numerically instead of
  alphabetically (v20 beats v9 again).


## v2.0.0 — A local agent platform (2026-07-26)

v1.9 made the case that on a small local model the intelligence has to live in
the tools. 2.0 opens the platform: **bring your own tools, your own procedures,
your own memory — and it still fits a 16K model.** Everything below runs
entirely on one machine; nothing leaves it.

### MCP: connect any tool server

- **Model Context Protocol client**, hand-written in Rust — stdio and
  Streamable HTTP transports, one in-flight call per server, hard timeouts so a
  wedged server can't wedge the agent. Add servers in **Settings → Code**: a
  command (stdio) or an `https://` URL (HTTP, optional bearer token).
- **A curated store** of version-pinned servers (GitHub, filesystem, memory,
  the reference server), each **live-certified** against Chaty's own client —
  the badge is a passing probe, not a promise. One click to add; entries that
  need a directory or token ask inline.
- **Lean by design for small models.** Community MCP schemas run to thousands
  of tokens; Chaty synthesizes a one-line doc per tool and holds the full
  schema back until it's needed. A server bringing more than the core-tier
  limit collapses into a single index line, so the prompt stays flat no matter
  how many tools you connect.
- **Safe by default.** Every MCP result is treated as untrusted content
  (injection-defended), and every call needs your approval unless you mark the
  server trusted. Proven on **ChatyMCP-Bench** — the agent drives real servers
  (filesystem / knowledge-graph memory / reference) end to end, graded on
  server state.

### Skills: procedural knowledge as files

- **`SKILL.md` files** — a page of steps the model loads only when it's
  relevant. One index line per skill rides in the prompt; the body loads on
  demand via `use_skill`. Put them in `~/.chaty/skills/` (global) or
  `<project>/.chaty/skills/` (project, which shadows global).
- Three official skills ship enabled (verify-before-push, debug-by-mechanism,
  investigate-first), each toggleable in **Settings → Code**.
- Measured on quick15: skills don't raise the ceiling much, but on tasks both
  sides solve the skilled agent wandered far less (roughly half the steps).

### Memory: facts that outlive the session

- **`remember` writes non-obvious findings** to `<workspace>/.chaty/memory/` —
  plain markdown you can read and edit, that never leaves the machine. A capped
  index rides in the prompt; the model reads a fact only when a line looks
  relevant. On by default; toggle in **Settings → Code**. The value scales
  with how expensive a fact is to re-derive — a large codebase, or a gotcha
  learned the hard way — so it earns its keep on real projects more than on
  toy ones.

### Under the hood

- **One tool registry** — natives, MCP tools, and skills all register through a
  single source of truth, with permission classes and a context budget. The
  system prompt is byte-for-byte unchanged for anyone using none of the above.
- **Linux** joins macOS and Windows: an AppImage now ships on every release.
- A round of reinvented-wheel cleanup (one size formatter, one toggle style,
  one HTTP client factory, one bilingual macro) and several real bugs found by
  the benchmarks along the way — a timed-out shell command no longer leaks a
  runaway process, and the MLX sidecar always loads its Metal bundle.

## v1.9.1 — Browser tools for every model (2026-07-24)

A small release with one big unlock and a set of browser-agent fixes, all
proven on ChatyWeb-Bench — a new oracle-validated, 23-task local web-agent
benchmark that ships in `bench/web/` (19/23 → 22/23 across the fixes below).

### The browser is no longer vision-only

- **Text-only models can now drive the browser in Code mode.** Models
  without a vision encoder get the browser suite minus the two screenshot
  tools; `browser_read`'s rich digest (visible text + interactive elements +
  current input values) serves as the model's eyes. A text-only 35B-A3B
  (MoE) resolves 22/23 ChatyWeb-Bench tasks in this mode.
- If such a model still hallucinates a screenshot call, it gets a text
  redirect instead of an image its engine can't embed (previously a latent
  error path).

### Browser agent, text-first ergonomics

- **Glyph-only buttons are clickable by name.** Repeated ▶ / ✕ / ◀ controls
  used to be indistinguishable in the page digest; they now surface their
  `aria-label`, and click-by-text matches that label too — the board-style
  "move card / delete card" tasks went from unsolvable to solved.
- **Clicking a `<select>` redirects usefully** — instead of a silent no-op
  click, the agent is told to choose with `browser_type` and the option's
  visible label.
- **Legitimate repeat clicks aren't "loops" anymore.** Pagination
  (Next × 3), add-to-cart × 2, and wizard steps repeat the same call on a
  changed page; the repeat breaker now allows identical browser clicks/types
  after a *successful* one (repeats after an error still intercept).
- **Single-language browser output.** The browser tool results and page
  digests now follow the session language — the one model-visible surface
  that had missed the v1.8.5 single-language pass.

### Windowing (macOS)

- **Closing Chaty from fullscreen no longer leaves a black screen.** The close
  button hides Chaty to the tray, but a fullscreen window lives in its own
  Space — hiding just the window left that Space up with nothing in it. Chaty
  now hides the whole app instead, exactly as ⌘H does: one motion out of the
  Space, no windowed frame flashing on the way, and reopening from the tray or
  Dock slides back in **still fullscreen**. Quitting from fullscreen also asks
  to leave it first, so the Space is never stranded.

### Design Canvas

- **The preview's scrollbar matches the page.** With scrollbars set to
  always-show (or a mouse plugged in), the srcdoc frame drew a bright white
  track that glared beside the dark pages models like to build — WebKit only
  paints a dark scrollbar when the document declares a `color-scheme`. The
  preview now infers one from what the page actually renders (background
  brightness, or the text colour when the background is a gradient or image)
  and lets the engine draw its own native scrollbar; a page that declares its
  own `color-scheme` is left untouched. Measured on the failing case: track
  rgb(250,250,250) → rgb(44,44,44), light pages unchanged.

### The browser waits for the page

Found by driving this release by hand on a 4B text-only model:

- **Async confirmations are no longer invisible.** Three separate reasons a
  submitted form looked like it had failed, all fixed: a click now waits for
  the page to finish reacting (in-flight fetch/XHR plus a DOM-quiet period,
  not just `readyState`); the change watcher notices **attribute** reveals
  (`style.display = "block"` on a banner that was already in the DOM changes
  no text nodes); and when the page text is truncated, the result adds the
  text of **the region around the element just used** — on a long page the
  confirmation sits far below a top-anchored window, which is why the agent
  kept resubmitting a real site's contact form. The wait also surfaces async
  validation after typing.
- **A submit blocked by native validation says so.** Clicking submit on a form
  with an empty `required` field fires no handler, no request, and changes
  nothing on screen — the agent had no way to know why. The click result now
  names the blocking fields and their validation messages. Scoped to the form
  the agent actually used, and judged on validity *before* the click (a
  successful submit that calls `form.reset()` must not look "blocked").
- **Repeating a click is allowed only while the page keeps changing.** The
  stateful-UI allowance (pagination "Next" × 3) now requires the previous
  identical click to have *changed* something; on an unchanged page the agent
  is stopped with "this may have already worked — read the page before
  clicking again." Pagination still works; duplicate submits don't.
- **Toggling Settings → Code → hidden browser restarts the browser**, instead
  of applying only to the next launch (mid-session switches looked ignored).

### Benchmarks

- **ChatyWeb-Bench** (`bench/web/`): six deterministic single-file fixture
  apps, a zero-dependency state server, 23 tasks graded on server state or
  final answers, and per-task oracles replayed through the real tool chain —
  fixtures and graders prove themselves with no model involved.
- The coder benchmark story grew a field: same model, same grading —
  Chaty 15/45 vs qwen-code 12/45, pi 10/45, opencode 7/45, bare bash 6/45.
  Full methodology and fairness notes in docs/BENCHMARKS.md.

## v1.9.0 — The agent proves itself (2026-07-22)

A reliability release built the hard way: every agent change gated by A/B runs
on the real 35B-A3B (MoE) bench model, full-suite reruns on a repaired
harness, and one honest negative result. **v1.9 resolves 15/45 on the
SWE-bench Verified subset vs 12/45 for v1.8.4** — same model, same harness,
fewer steps (median 24 vs 29). Details and disclosures: docs/BENCHMARKS.md.

### Coding agent: recovery and self-awareness

- **Format slips can't spiral anymore.** A tool call missing a required argument is corrected with a filled-in example and retried without ever entering the conversation — small no-think models used to imitate their own empty-argument calls into a dead loop. Repeats of a just-errored call get "fix the arguments" advice instead of a generic lecture; missing-arg errors are single-language with a copyable example.
- **Post-edit diagnostics.** Every edit/write confirmation now catches the bug class that compiles fine: a flat-scope AST scan flags possibly-undefined Python names (typos) as a soft note, and syntax-gate failures attach the offending region with line numbers — no re-read needed to locate the error.
- **A progress ledger survives compaction.** The first time context compaction kicks in, the files already edited are pinned into the transcript, so the model stops redoing work it can no longer see.
- **Your project rules, whoever they were written for.** The project guide now also reads `.cursorrules`, `.github/copilot-instructions.md`, and `.cursor/rules/*.mdc`.
- **Experimental: anchor-based line editing** (`edit_lines`, hashline-style `LINE:HASH` anchors with shift recovery). Off by default: our A/B showed the 3B-active bench model reads worse with anchor prefixes — kept for bigger models, documented honestly.

### Design Canvas

- **Generation is interruptible** — Send becomes Stop while an iteration streams; stopping discards the partial quietly.
- **Hand-editing keeps syntax colors** (highlighted backdrop under a transparent editor, scroll-synced, both themes).
- **The live scan respects you**: wheel/touch breaks follow so you can inspect mid-generation, a pill resumes it, long documents render windowed instead of rebuilding every row per tick, and version switching locks during generation (it would diff against the wrong base).
- **HTML edit mode setting** — *Patch* (search/replace, fast) or *Rewrite* (the model streams the full document and Chaty computes the live diff). **Rewrite is the reliable choice for smaller models.**

### Benchmarks & harness

- Fixed two harness bugs that shaped the v1.8.5 published numbers: step counts were double-counted, and "Continue" turns after the step limit carried no history (the task itself vanished). Old numbers stay published as artifacts of that harness; the new comparison reruns **both** versions on the fixed harness.
- Bench tooling grew per-task transcripts, language/anchor switches, and watchdog patterns — all documented in the repo.

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
