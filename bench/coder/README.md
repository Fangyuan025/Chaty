# ChatyCoder-Bench

Benchmarks the **complete Chaty Coder product** — the real production agent
loop (`src/lib/agentLoop.ts`: fused code search, symbol reads, syntax gate,
validate_change, repeat-breakers, think gate) driven headless against real
coding tasks. This is different from `bench/chaty_tb_agent.py`
(Terminal-Bench), which measures the *model* through a minimal bash agent.

## Architecture

```
runner.mts  ──(real agentLoop.ts, tsx + mockIPC)──►  ipc.ts invoke()
                                                        │ stdio JSON-lines
                                              chaty-headless (Rust bin)
                                          agent.rs tools + inference engine
```

- `src-tauri/src/bin/chaty-headless.rs` — stdio JSON server exposing the real
  tool layer + engines (MLX dir / GGUF / `mock`). Streaming `generate` events,
  verbatim error strings (`NEED_DIR_GRANT` markers survive).
- `runner.mts` — full-Coder runner: per-task workspace copy, step-limit
  continuation (mirrors the Continue button), grade.sh verdicts, JSONL results.
- `bare.mts` — ablation baseline: same model/engine/params/tasks, but a
  minimal bash-only agent. `runner − bare` isolates the tool layer's value.
- `swebench/` — SWE-bench Verified task track (see below).

## Quick start

```bash
cd src-tauri && cargo build --release --bin chaty-headless
cd bench/coder
CHATY_BENCH_MODEL="$HOME/Library/Application Support/com.chaty.desktop/models/<MLX-dir>" \
  npx tsx runner.mts --tasks swebench/tasks            # full Coder
CHATY_BENCH_MODEL=... npx tsx bare.mts --tasks swebench/tasks   # ablation
```

`CHATY_HEADLESS_BIN` overrides the binary path (defaults to release build).
`--only <task>` runs one task. Results land in `runs/*.jsonl`.

## SWE-bench Verified subset

`swebench/prepare.py` builds a deterministic 50-instance subset (seed 42) of
SWE-bench Verified, restricted to pure-Python repos that install on
Apple-Silicon macOS (django/sympy/sphinx/pytest/pylint/requests/flask),
difficulty ≤ 4h, spec Python ≥ 3.8.

```bash
python3 prepare.py select        # subset.json from verified_full.json
python3 prepare.py materialize   # clone repos @base_commit, venvs, grading files
python3 prepare.py validate      # gold-patch validation → validated.json
python3 prepare.py regrade       # refresh grading files only
```

Each task dir: `task.md` (problem statement), `workspace/` (repo at
base_commit, future refs stripped), `grade.sh` → `grade.py`, `grade_config.json`,
`test_patch.diff`, `gold_patch.diff` (self-validation only — outside workspace/).

**Grading** mirrors the official harness: reset test files → apply held-out
test_patch → run the repo's own test command with directives derived from the
test_patch files → parse the log with parsers vendored verbatim from
`swebench.harness.log_parsers` (`grade_lib.py`) → resolved iff every
FAIL_TO_PASS **and** PASS_TO_PASS entry is PASSED.

**Deviations from the official Linux harness (disclose when reporting):**
- Host macOS execution, not the official Docker images.
- Specs pinned to Python 3.8 run on 3.9 (no 3.8 arm64 build exists).
- sphinx's `tox --current-env` wrapper is invoked as plain pytest in-env.
- Instances where the **gold patch** fails to grade green on macOS (e.g.
  Apple's sqlite lacks column-metadata introspection) are excluded by
  `validate` and recorded in `validated.json`. Report N as the validated count.

## Reporting

Report: model + quant, think mode, nCtx, maxSteps, dataset (SWE-bench Verified
macOS-validated subset, N instances, seed 42), harness = ChatyCoder-Bench
(this directory, cite commit), and both numbers: full Coder vs bare-bash
ablation.
