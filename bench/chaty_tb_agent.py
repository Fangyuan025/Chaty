"""Terminal-Bench adapter for Chaty's local agent.

Runs Chaty's MLX sidecar (chaty-mlx) on the host for inference and drives the
Terminal-Bench task container through the harness TmuxSession — the same
ReAct <tool_call> protocol and loop guards as Chaty's production Code mode,
ported to a bash-only terminal surface.

Usage:
  CHATY_TB_MODEL=/path/to/mlx-model-dir \
  tb run --dataset terminal-bench-core==head --task-id hello-world \
     --agent-import-path chaty_tb_agent:ChatyAgent

Env / --agent-kwarg:
  CHATY_TB_MODEL   / model_dir : MLX model folder (required)
  CHATY_MLX_SIDECAR / sidecar  : sidecar binary (default: repo binaries build)
  CHATY_TB_THINK   / think     : "true"/"false" reasoning toggle (default false)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SIDECAR = REPO / "src-tauri" / "binaries" / "chaty-mlx-aarch64-apple-darwin"

SYS = """You are Chaty's terminal agent, completing a task inside a Linux shell.

You have ONE tool:
- bash: run a shell command in the task terminal. args: {"command": string, "timeout_secs"?: number}

Rules (follow strictly):
- Call ONE tool at a time. To call it, output a single line
  <tool_call>{"name":"bash","arguments":{"command":"..."}}</tool_call>
  and STOP immediately — nothing else in that message.
- You'll get the terminal output back as <tool_result>...</tool_result>, then continue.
- The shell is persistent (same session throughout): cd, exports and files stay.
- Prefer non-interactive flags (-y, --yes); never launch editors or pagers —
  write files with heredocs (cat > file <<'EOF' ... EOF).
- If the same approach fails twice, change approach — never repeat a failing
  command unchanged a third time.
- When the task is fully done, output a one-line summary WITHOUT any tool call."""

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)


class Sidecar:
    """Minimal stdio-JSON client for chaty-mlx. One instance per process —
    the model loads once and serves every task in the run."""

    _lock = threading.Lock()
    _shared: "Sidecar | None" = None

    @classmethod
    def shared(cls, sidecar: str, model_dir: str, n_ctx: int) -> "Sidecar":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls(sidecar, model_dir, n_ctx)
            return cls._shared

    def __init__(self, sidecar: str, model_dir: str, n_ctx: int):
        self.proc = subprocess.Popen(
            [sidecar],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send({"cmd": "load", "path": model_dir, "nCtx": n_ctx})
        for ev in self._events():
            if ev.get("event") == "loaded":
                return
            if ev.get("event") == "error":
                raise RuntimeError(f"sidecar load failed: {ev.get('message')}")
        raise RuntimeError("sidecar exited during load")

    def _send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _events(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def generate(
        self, messages: list[dict], think: bool, max_tokens: int = 3072
    ) -> tuple[str, int, int]:
        """Returns (text, prompt_tokens, completion_tokens). Client-side stop:
        the sidecar streams tokens; on seeing </tool_call> we cancel (stop
        sequences are enforced host-side in Chaty as well)."""
        self._send(
            {
                "cmd": "generate",
                "messages": messages,
                "params": {
                    "temperature": 0.2,
                    "topP": 0.9,
                    "repeatPenalty": 1.05,
                    "maxTokens": max_tokens,
                    "think": think,
                },
            }
        )
        text, cancelled = "", False
        p_tok = c_tok = 0
        for ev in self._events():
            kind = ev.get("event")
            if kind == "token":
                text += ev.get("text", "")
                if not cancelled and "</tool_call>" in text:
                    self._send({"cmd": "cancel"})
                    cancelled = True
            elif kind == "done":
                p_tok = int(ev.get("promptTokens", 0))
                c_tok = int(ev.get("completionTokens", 0))
                break
            elif kind == "error":
                raise RuntimeError(f"sidecar generate failed: {ev.get('message')}")
        else:
            raise RuntimeError("sidecar exited during generate")
        return text, p_tok, c_tok


def _parse_tool_call(raw: str) -> dict | None:
    body = THINK_RE.sub("", raw)
    m = TOOL_RE.search(body)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return call if call.get("name") == "bash" else None


def _cap(output: str, limit: int = 8000) -> str:
    if len(output) <= limit:
        return output
    head, tail = int(limit * 0.3), int(limit * 0.7)
    return f"{output[:head]}\n… (output truncated) …\n{output[-tail:]}"


class ChatyAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "chaty"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_dir = kwargs.get("model_dir") or os.environ.get("CHATY_TB_MODEL")
        if not self._model_dir:
            raise ValueError("set CHATY_TB_MODEL or --agent-kwarg model_dir=…")
        self._sidecar = kwargs.get("sidecar") or os.environ.get(
            "CHATY_MLX_SIDECAR", str(DEFAULT_SIDECAR)
        )
        self._n_ctx = int(kwargs.get("n_ctx", 16384))
        self._max_steps = int(kwargs.get("max_steps", 40))
        think = kwargs.get("think") or os.environ.get("CHATY_TB_THINK", "false")
        self._think = str(think).lower() == "true"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        sc = Sidecar.shared(self._sidecar, self._model_dir, self._n_ctx)
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": self._render_instruction(instruction)},
        ]
        in_tok = out_tok = 0
        markers: list[tuple[float, str]] = []
        last_cmd, repeats = "", 0
        log = (logging_dir / "chaty-agent.log").open("a") if logging_dir else None

        def note(s: str):
            if log:
                log.write(s + "\n")
                log.flush()

        failure = FailureMode.NONE
        try:
            for step in range(self._max_steps):
                raw, p, c = sc.generate(messages, self._think)
                in_tok += p
                out_tok += c
                note(f"--- step {step} raw ---\n{raw[:1500]}")
                call = _parse_tool_call(raw)
                if call is None:
                    note("final answer — done")
                    break

                cmd = str(call.get("arguments", {}).get("command", "")).strip()
                if not cmd:
                    result = 'ERROR: missing "command"'
                else:
                    # Chaty's identical-call breaker, ported.
                    if cmd == last_cmd:
                        repeats += 1
                    else:
                        last_cmd, repeats = cmd, 0
                    if repeats >= 2:
                        failure = FailureMode.AGENT_TIMEOUT
                        note("loop breaker: identical command 3x — stopping")
                        break
                    if repeats == 1:
                        result = (
                            "Intercepted: identical to the previous command — the "
                            "result cannot change. Do something different."
                        )
                    else:
                        timeout = min(
                            float(call.get("arguments", {}).get("timeout_secs", 120)),
                            180.0,
                        )
                        markers.append(
                            (session.get_asciinema_timestamp(), cmd[:80])
                        )
                        session.send_keys(
                            [cmd, "Enter"], block=True, max_timeout_sec=timeout
                        )
                        result = _cap(session.get_incremental_output())
                note(f"--- step {step} result ---\n{result[:1200]}")
                closed = raw if "</tool_call>" in raw else raw + "</tool_call>"
                messages.append({"role": "assistant", "content": THINK_RE.sub("", closed)})
                messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_result>\n{result}\n</tool_result>",
                    }
                )
            else:
                failure = FailureMode.AGENT_TIMEOUT
        except Exception as e:  # sidecar death, tmux errors — report, don't crash the run
            note(f"exception: {e}")
            failure = FailureMode.UNKNOWN_AGENT_ERROR
        finally:
            if log:
                log.close()

        return AgentResult(
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            failure_mode=failure,
            timestamped_markers=markers,
        )
