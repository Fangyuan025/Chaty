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

While building v1.9's reliability work we found and fixed two harness bugs
that affected the published runs above: recorded step counts were inflated
~2× (double-counted step events), and "Continue" turns after the step limit
carried **no conversation history** — the task text itself was gone, so
turns 2–3 wandered. The published numbers stand as historical artifacts of
that harness; they are **not comparable** to the rerun below.

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

## SWE-bench Verified subset (ChatyCoder-Bench)

Harness: [`bench/coder/`](../bench/coder/README.md) at commit `57adac9`. It
drives the **real production agent loop**
(`src/lib/agentLoop.ts` + the real Rust tool layer via a headless stdio
server) — not a reimplementation.

- **Dataset**: deterministic 50-instance subset (seed 42) of SWE-bench
  Verified — pure-Python repos that install on Apple-Silicon macOS
  (django / sympy / sphinx / pytest / pylint / requests / flask),
  difficulty ≤ 4 h. 5 instances excluded as env-incompatible after
  gold-patch validation (`pallets__flask-5014`, `pylint-dev__pylint-7080`,
  `sphinx-doc__sphinx-7985`, `-8120`, `-9711`) → **N = 45**.
- **Budgets**: Coder — 40 steps/turn, auto-continue up to 3 turns (mirrors
  the in-app Continue button); bare — single turn, 40 steps. Both:
  temperature 0.2, think off, nCtx 16384.
- **Grading** mirrors the official harness: reset test files → apply held-out
  `test_patch` → run the repo's own test command → parse with log parsers
  vendored verbatim from `swebench.harness.log_parsers`; resolved iff every
  FAIL_TO_PASS **and** PASS_TO_PASS entry passes.
- **Runs** (per-task JSONL: resolved, steps, turns, wall time): Coder —
  `bench/coder/runs/2026-07-17-17-56-57.jsonl` (tasks 1–31) plus 14
  single-task resume files `runs/2026-07-18-*.jsonl`, merged as
  `runs/coder-merged-2026-07-18.jsonl`; bare —
  `bench/coder/runs/bare-2026-07-17-23-22-32.jsonl` (one uninterrupted pass).

### Per-repo breakdown

| Repo | Coder | bare |
| --- | --- | --- |
| django (24) | **7** | 2 |
| sympy (10) | 2 | 2 |
| pytest (4) | 0 | 1 |
| sphinx (3) | 0 | 0 |
| pylint (2) | 0 | 0 |
| requests (2) | 0 | 1 |
| **total (45)** | **9** | **6** |

### Deviations from the official harness (read before comparing)

- Host **macOS** execution, not the official Docker images.
- Specs pinned to Python 3.8 run on 3.9 (no 3.8 arm64 build exists).
- sphinx's `tox --current-env` wrapper is invoked as plain pytest in-env.
- Instances whose **gold patch** fails to grade green on macOS are excluded
  by validation (listed above) — N = 45, not 50, and not the full 500.
- The headless tool server was a debug build (inference runs in the MLX
  sidecar either way; affects tool-layer overhead only, not model output).
- Ablation budget asymmetry: the bare agent has no continue mechanism, so
  its effective step budget is lower than Coder's (40 vs up to 3 × 40).
- The Coder pass was interrupted after task 31 (the runner process exited
  between tasks; no results were affected) and the remaining 14 tasks ran
  as isolated single-task invocations with a fresh engine each. The bare
  pass ran uninterrupted end to end. Grading is per-task and independent,
  so the merge is sound — disclosed for completeness.

## Terminal-Bench

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
