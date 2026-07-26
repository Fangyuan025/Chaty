# Chaty benchmarks

Every number on this page comes from **one local model** — the same artifact a
user downloads inside the app, running entirely on one machine:

- **Model**: Qwen3.5-35B-A3B mxfp8 (MLX, Apple Silicon) — a **MoE** model:
  35 B total parameters, only **~3 B active per token** · reasoning **off** ·
  context **16384**
- **No cloud, no API fallback** — inference is Chaty's own MLX sidecar.

## Headline

One table is the whole story — the same local model behind five agent
designs, one machine, identical grading:

| SWE-bench Verified — 45-task macOS-validated subset | Context | Resolved |
| --- | --- | --- |
| **Chaty agent (v1.9)** — the full tool loop | 16K | **15/45 (33 %)** |
| [qwen-code](https://github.com/QwenLM/qwen-code) 0.20 — the model family's first-party CLI | 32K¹ | 12/45 (27 %) |
| [pi](https://github.com/badlogic/pi-mono) 0.81 — minimal 4-tool agent CLI | 16K | 10/45 (22 %) |
| [opencode](https://github.com/anomalyco/opencode) 1.18 | 16K | 7/45 (15.6 %) |
| bare bash agent — single-tool ablation | 16K | 6/45 (13.3 %) |

¹ qwen-code's own system prompt is ~19K tokens and does not fit a 16K
window; it received the minimum context it can operate in.

Chaty leads the field — including the model family's own first-party CLI,
while using **half its context window** — and resolves 2.5× the bare-bash
ablation (django 9/24 vs 2/24). The delta is the tool loop: repo-aware
search, symbol-level reads, precise edits, recovery guards, post-edit
diagnostics, targeted test runs. Five tasks were solved by Chaty and no
third-party agent; the union of all five agents is 19/45. At N = 45
individual tasks flip both ways; the aggregate and the repo slices are the
signal, not any single instance.

Comparability note: the bare-bash pipeline is single-turn by design and was
never affected by the two Coder-harness bugs fixed during v1.9 (inflated
step counts; history-less Continue turns), so its 2026-07-17 run remains
valid under the fixed harness that produced the v1.9 number. Version-over-
version data (v1.9 vs v1.8.4: 15/45 vs 12/45) and the Terminal-Bench run
(15/77) live in the sections below.

These numbers are **not** leaderboard submissions and are not directly
comparable to leaderboard entries (subset + harness deviations below).

## Third-party agent CLIs on the same model (2026-07-23/24)

To place the ablation on a real-world scale, the same 45 tasks ran through
well-known open-source agent CLIs, each on its own scaffold and defaults:

- **Serving** (identical for every agent): a local OpenAI-compatible shim
  over Chaty's own in-process MLX engine, plus a translation layer
  rendering OpenAI function-calling into the model's native `<tool_call>`
  XML dialect (bench tooling only, not shipped). The shim retries a
  truly-empty completion once with hotter sampling — a real cloud endpoint
  never returns an empty string, and some CLIs treat one empty as fatal;
  the retry applies uniformly to every agent, and every scored run below
  used it.
- **Config**: identical model artifact and quantization, think off; each
  agent's own system prompt, tools, and defaults; one fresh session and
  workspace per task; identical grading; 45-minute wall cap per task; no
  step caps imposed (Chaty's envelope is 40 steps × ≤3 turns). Context is
  16K except where an agent structurally cannot run there (table above).
- **Results**: qwen-code **12/45**, pi **10/45**, opencode **7/45**.
  Chaty-only vs all third parties: django-15814, django-16901, pytest-7571,
  sympy-13757, sympy-15345. Solved by a third party but not Chaty v1.9:
  django-15525, requests-1142, sympy-18211 (pi), sphinx-11445 (qwen-code —
  the only sphinx solve any agent has produced here).

Attempted but not scorable on this hardware, for transparency: Hermes
(hard-requires a 64K context declaration; serving that reliably was not
possible on this 48 GB machine), mini-swe-agent (requires a tool call in
every model response and aborts after repeated prose turns — a protocol
built for larger models), goose (abandons the session on a single empty
completion; the retry shim rescued transient cases but not deterministic
ones), and OpenAI's codex CLI (current versions require the Responses API;
the last chat-completions build was blocked by macOS Gatekeeper).

Reading this fairly: these scaffolds are sound trades with frontier-class
models, where the model itself carries planning, repo comprehension, and
protocol discipline. At ~3 B active parameters those assumptions weaken —
and a tool loop that carries part of the intelligence itself, plus the
error tolerance small models need, is what separates 15/45 from the field.
On small local models, the tools are the product.

## v1.9 agent vs v1.8.4 — fixed-harness rerun (2026-07-22)

While building v1.9's reliability work we found and fixed two harness bugs:
recorded step counts were inflated ~2× (double-counted step events), and
"Continue" turns after the step limit carried **no conversation history** —
the task text itself was gone, so turns 2–3 wandered. Numbers published
before the fix (the v1.8.5-era 9/45 baseline) are retired from this page
and are **not comparable** to any fixed-harness run.

Both agent versions, full 45-task subset, identical fixed harness, same
model and parameters, one fresh process per task:

| Agent | Resolved | django | sympy | pytest | median steps |
| --- | --- | --- | --- | --- | --- |
| **v1.9** (arg-guard, JIT hints, post-edit diagnostics, progress ledger) | **15/45 (33.3 %)** | 9/24 | 4/10 | 2/4 | 24 |
| v1.8.4 | 12/45 (26.7 %) | 9/24 | 1/10 | 1/4 | 29 |

v1.9-only solves: django-15814, django-16901, pytest-7432, sympy-13757,
sympy-15345, sympy-23950 (13757 and 7432 had never been solved in any prior
run). v1.8.4-only: django-13925, django-15525, requests-1142. The gap
concentrates in sympy — the slice where post-edit diagnostics (typo-level
name scan) and the recovery guards bite hardest — and v1.9 solves with
fewer steps. Run disclosures: one v1.9 task (django-14034) was killed by a
faulty run-watchdog and rerun solo; one v1.9 result row (django-13925) was
lost to a runner crash at grading and rerun solo; both retries scored ✗.
Artifacts: `runs/ab-final45-v19-2026-07-22.jsonl`,
`runs/ab-final45-old184-2026-07-22.jsonl`.

## Memory — does a remembered fact help the next session? (2026-07-26)

A tool (`remember`) writes non-obvious findings to `.chaty/memory/`; a capped
index rides in the next session's prompt, and the model reads a fact when a
line looks relevant. ChatyMemory-Bench runs each scenario twice — the only
variable is whether the relevant fact is already in memory:

| Scenario (tiny fixture) | Control (empty) | Seeded (fact in memory) |
| --- | --- | --- |
| port-owner — buried port constant | ✓ 3 steps | ✓ 1 step |
| config-gotcha — non-obvious env condition | ✓ 2 | ✓ 3 |
| magic-timeout — commented magic number | ✓ 1 | ✓ 1 |
| test-command — non-standard test target | ✓ 7 | ✓ 21† |

Read this honestly: on three-file fixtures the answer is cheap to find even
without memory, so the lift is small and within run-to-run noise (†the
test-command seeded run wandered — variance, not signal). The clean case is
port-owner, where the seeded agent answers in one step instead of three. The
mechanism is verified end to end: 8/8 real-model runs, every seeded agent
reads its fact and answers, no regressions.

Memory's value scales with how expensive the fact is to re-derive — a large
codebase, or a gotcha learned the hard way in a prior session — which these
deliberately small fixtures under-represent. The most valuable thing this
bench did was catch a real crash before release: the index linked facts by
bare filename, so the model's `read_file` missed and a later `undefined`
crashed the whole turn. Both are fixed (full workspace-relative paths;
tool results can never be undefined) with regression tests.

Reproduce: `CHATY_BENCH_MODEL=… npx tsx bench/memory/runner.mts`.

## Skills A/B — what a page of procedure is worth (2026-07-26)

2.0 adds file-based skills: procedural knowledge as markdown, one index line
per skill in the prompt, the body loaded only on use. Same 15-task screen,
same code, same binary — the only variable is `CHATY_BENCH_SKILLS`:

| quick15 subset | Resolved | Steps on commonly-solved tasks |
| --- | --- | --- |
| **Skills on** (3 official skills) | **6/15** | **62** |
| Skills off (control) | 5/15 | 135 |

Read the resolve delta honestly: +1 at N=15 is within single-run variance.
The step numbers are the finding — on the four tasks both sides solved, the
skilled side needed **2.2× fewer steps** (django-16333: 4 vs 23;
sympy-23950: 30 vs 92), and both skills-only solves were endurance wins the
control gave up on (django-16901 at 10 steps) or timed out of (sympy-15345
at 78). Procedure doesn't make the model smarter; it stops it wandering.

Cost: the index is ~4 prompt lines. Bodies never load unless invoked.

Reproduce: `bench/coder/runs/quick15-skills.sh {skills|noskills}`.

## ChatyMCP-Bench — do MCP servers work in a small model's hands? (2026-07-25)

Connecting to an MCP server is table stakes. The question that decides whether
the ecosystem is usable on a ~3B-active model is whether the model can *drive*
those tools — read their schemas, fill nested arguments, verify its own work.

Six tasks against three real servers, run through the real agent loop and the
real MCP client, graded on **server state** (read back through a different tool
than the one under test) or final-answer substring:

| Server | Tasks | Resolved | Steps (median) |
| --- | --- | --- | --- |
| [filesystem](https://github.com/modelcontextprotocol/servers) (14 tools) | write / read+answer / edit-in-place | **3/3** | 1 |
| memory — knowledge graph | store entity / store+recall | **2/2** | 3 |
| everything — reference server (13 tools) | tool-chained arithmetic | **1/1** | 1 |
| **Total** | | **6/6** | **2** |

Two properties made this work at 16K, both from the 2.0 tool registry:

1. **Lean docs.** Community MCP schemas run to thousands of tokens; Chaty
   synthesizes a one-line doc per tool (first sentence + compact argument
   signature). The full schema is held back and returned only as the
   correction when a call is missing an argument — so the expensive text
   appears exactly when the model needs it.
2. **A tool budget.** A server bringing more tools than the core-tier limit
   collapses into a single index line, keeping the prompt flat as servers are
   added.

The hardest task is `mem-remember-entity`: `create_entities` takes a nested
array of objects, the shape small models most often flatten. It passed in 3
steps.

Reproduce: `CHATY_BENCH_MODEL=… npx tsx bench/mcp/runner.mts` (servers are
version-pinned in the runner; results land in `bench/mcp/runs/`, gitignored).

Store certification — every catalog entry connecting and answering a probe
through the real client — is a separate live gate:
`cargo test -p chaty store_cert -- --ignored`.

## Harness & methodology (ChatyCoder-Bench)

Harness: [`bench/coder/`](../bench/coder/README.md). It drives the **real
production agent loop** (`src/lib/agentLoop.ts` + the real Rust tool layer
via a headless stdio server) — not a reimplementation.

- **Dataset**: deterministic 50-instance subset (seed 42) of SWE-bench
  Verified — pure-Python repos that install on Apple-Silicon macOS
  (django / sympy / sphinx / pytest / pylint / requests / flask),
  difficulty ≤ 4 h. 5 instances excluded as env-incompatible after
  gold-patch validation (`pallets__flask-5014`, `pylint-dev__pylint-7080`,
  `sphinx-doc__sphinx-7985`, `-8120`, `-9711`) → **N = 45**.
- **Budgets**: Chaty — 40 steps/turn, auto-continue up to 3 turns (mirrors
  the in-app Continue button); bare — single turn, 40 steps; third-party
  CLIs — own defaults, 45-min wall cap (section above). All: temperature
  0.2, think off.
- **Grading** mirrors the official harness: reset test files → apply held-out
  `test_patch` → run the repo's own test command → parse with log parsers
  vendored verbatim from `swebench.harness.log_parsers`; resolved iff every
  FAIL_TO_PASS **and** PASS_TO_PASS entry passes.
- **Current run artifacts** (per-task JSONL: resolved, steps, turns, wall
  time): Chaty v1.9 — `runs/ab-final45-v19-2026-07-22.jsonl`; bare —
  `runs/bare-2026-07-17-23-22-32.jsonl` (single-turn pipeline, unaffected
  by the harness fixes above); third-party — `runs/agents45/*-results.jsonl`.

### Per-repo breakdown (current)

| Repo | Chaty v1.9 | qwen-code | pi | bare |
| --- | --- | --- | --- | --- |
| django (24) | **9** | 7 | 6 | 2 |
| sympy (10) | **4** | 2 | 3 | 2 |
| pytest (4) | **2** | 1 | 0 | 1 |
| sphinx (3) | 0 | **1** | 0 | 0 |
| pylint (2) | 0 | 0 | 0 | 0 |
| requests (2) | 0 | **1** | **1** | 1 |
| **total (45)** | **15** | 12 | 10 | 6 |

### Deviations from the official harness (read before comparing)

- Host **macOS** execution, not the official Docker images.
- Specs pinned to Python 3.8 run on 3.9 (no 3.8 arm64 build exists).
- sphinx's `tox --current-env` wrapper is invoked as plain pytest in-env.
- Instances whose **gold patch** fails to grade green on macOS are excluded
  by validation (listed above) — N = 45, not 50, and not the full 500.
- The headless tool server is a debug build (inference runs in the MLX
  engine either way; affects tool-layer overhead only, not model output).
- Ablation budget asymmetry: the bare agent has no continue mechanism, so
  its effective step budget is lower than Chaty's (40 vs up to 3 × 40).

## Terminal-Bench (v1.8.5-era run)

Harness: **terminal-bench v0.2.18**, dataset **terminal-bench-core v0.1.1**,
`--n-concurrent 1`, run `2026-07-16__16-49-17` (~9.6 h wall). Agent:
[`bench/chaty_tb_agent.py`](../bench/chaty_tb_agent.py) — Chaty's production
ReAct protocol and loop guards ported to a bash-only terminal surface;
inference on the host via the MLX sidecar (model loaded once per run).

- **15 / 77 resolved (19.5 %)**. 3 of the 80 core tasks excluded up front
  (`eval-mteb`, `eval-mteb.hard`, `build-linux-kernel-qemu` — resource-bound
  on a laptop under a local 35B).

<details>
<summary>Resolved tasks (15)</summary>

`swe-bench-fsspec` · `swe-bench-langcodes` ·
`incompatible-python-fasttext.base_with_hint` · `simple-web-scraper` ·
`fix-permissions` · `hello-world` · `openssl-selfsigned-cert` ·
`prove-plus-comm` · `git-workflow-hack` · `crack-7z-hash.easy` ·
`create-bucket` · `tmux-advanced-workflow` · `new-encrypt-command` ·
`csv-to-parquet` · `heterogeneous-dates`

</details>
