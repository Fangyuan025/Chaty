# Chaty × Terminal-Bench

Benchmarks Chaty's local agent on [Terminal-Bench](https://www.tbench.ai/): the
`ChatyAgent` adapter runs **Chaty's MLX sidecar on the host** for inference and
drives the task container through the harness `TmuxSession`, using the same
`<tool_call>` ReAct protocol and loop guards as Chaty's production Code mode
(ported to a bash-only terminal surface).

## Prerequisites

- macOS (Apple Silicon), Docker runtime (`brew install colima docker && colima start`)
- `uv tool install terminal-bench`
- The MLX sidecar binary built (`./scripts/build-mlx-sidecar.sh` → `src-tauri/binaries/`)
- An MLX model folder (mlx-community layout)

## Run

```sh
cd bench
CHATY_TB_MODEL="$HOME/Library/Application Support/com.chaty.desktop/models/<model-dir>" \
tb run --dataset terminal-bench-core==head \
   --agent-import-path chaty_tb_agent:ChatyAgent \
   --n-concurrent 1
```

- Single task: add `--task-id hello-world`
- Reasoning toggle: `CHATY_TB_THINK=true` (default off)
- Model loads ONCE per run (class-level sidecar singleton) — the first task
  pays the load, the rest reuse it. Keep `--n-concurrent 1`: one Metal GPU.

## Reporting numbers

Per 2026 norms: always disclose harness (terminal-bench version), dataset
version, model + quantization, and think mode. Pair Terminal-Bench with a
SWE-Bench Verified subset (same harness, `--dataset swebench-verified`) when
quoting agent capability.
