# Chaty benchmarks

Every number on this page comes from **one local model** — the same artifact a
user downloads inside the app, running entirely on one machine:

- **Model**: Qwen3.5-35B-A3B mxfp8 (MLX, Apple Silicon) — a **MoE** model:
  35 B total parameters, only **~3 B active per token** · reasoning **off** ·
  context **16384**
- **No cloud, no API fallback** — inference is Chaty's own MLX sidecar.

## Headline

One comparison is the whole story:

| SWE-bench Verified — 45-task macOS-validated subset | Resolved |
| --- | --- |
| **Chaty agent (v1.9)** — the full tool loop | **15/45 (33 %)** |
| bare bash agent — same model, same tasks (ablation) | 6/45 (13.3 %) |

Same model, same tasks, same grading: the Chaty tool loop resolves **2.5×**
what a bare bash loop does — django **9/24 vs 2/24**, sympy 4/10 vs 2/10.
That delta — repo-aware search, symbol-level reads, precise edits, recovery
guards, post-edit diagnostics, targeted test runs — is the product, measured.
At N = 45 individual tasks flip both ways; the aggregate and the repo slices
are the signal, not any single instance.

Comparability note: the bare-bash pipeline is single-turn by design and was
never affected by the two Coder-harness bugs fixed during v1.9 (inflated
step counts; history-less Continue turns), so its 2026-07-17 run remains
valid under the fixed harness that produced the v1.9 number. Version-over-
version data (v1.9 vs v1.8.4: 15/45 vs 12/45) and the Terminal-Bench run
(15/77) live in the sections below.

These numbers are **not** leaderboard submissions and are not directly
comparable to leaderboard entries (subset + harness deviations below).

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
