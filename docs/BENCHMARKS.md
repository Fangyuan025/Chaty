# Chaty benchmarks

Every number on this page comes from **one local model** — the same artifact a
user downloads inside the app, running entirely on one machine:

- **Model**: Qwen3.5-35B-A3B mxfp8 (MLX, Apple Silicon) — a **MoE** model:
  35 B total parameters, only **~3 B active per token** · reasoning **off** ·
  context **16384**
- **No cloud, no API fallback** — inference is Chaty's own MLX sidecar.

## Headline

| Track | Agent | Result |
| --- | --- | --- |
| SWE-bench Verified — 45-task macOS-validated subset | **Chaty Coder** (full tool loop) | **9/45 (20 %)** |
| SWE-bench Verified — same subset, same model | bare bash agent (ablation) | 6/45 (13.3 %) |
| Terminal-Bench core v0.1.1 | Chaty agent protocol, bash-only surface | **15/77 (19.5 %)** |

Same model, same tasks, same grading: Chaty's tool loop resolves **half again
as many** instances as a bare bash agent (9 vs 6) — and on django, the
largest and most structured slice, **3.5×** (7/24 vs 2/24). That delta —
repo-aware search, symbol-level reads, precise edits, targeted test runs —
is the product, measured. At N = 45 individual tasks do flip both ways
(the bare agent solved 3 the full loop missed); the aggregate and the
django slice are the signal, not any single instance.

These numbers are **not** leaderboard submissions and are not directly
comparable to leaderboard entries (subset + harness deviations below).

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
