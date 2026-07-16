#!/usr/bin/env python3
"""Official-style per-test SWE-bench grading.

Run from inside a workspace copy (the runner does `cwd=ws`):
    python3 grade.py <task_dir>

Steps: reset the test files touched by test_patch → apply test_patch → run the
repo's own test command → parse the log with the vendored official parser →
resolved iff every FAIL_TO_PASS and PASS_TO_PASS entry is PASSED.
Exit code 0 = resolved.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_lib import MAP_REPO_TO_PARSER  # noqa: E402


def main():
    task_dir = Path(sys.argv[1]).resolve()
    cfg = json.load(open(task_dir / "grade_config.json"))
    ws = Path.cwd()

    subprocess.run(["git", "checkout", "-q", "--", *cfg["test_files"]], cwd=ws)
    r = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(task_dir / "test_patch.diff")],
        cwd=ws,
    )
    if r.returncode:
        sys.exit("grade: test_patch failed to apply")

    env = os.environ.copy()
    env["PATH"] = cfg["env_bin"] + ":" + env["PATH"]
    p = subprocess.run(
        ["bash", "-c", cfg["cmd"]],
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log = p.stdout + "\n" + p.stderr
    status = MAP_REPO_TO_PARSER[cfg["repo"]](log, None)

    f2p_bad = [t for t in cfg["f2p"] if status.get(t) != "PASSED"]
    p2p_bad = [t for t in cfg["p2p"] if status.get(t) != "PASSED"]
    print(
        f"grade: parsed {len(status)} tests — "
        f"FAIL_TO_PASS {len(cfg['f2p']) - len(f2p_bad)}/{len(cfg['f2p'])}, "
        f"PASS_TO_PASS {len(cfg['p2p']) - len(p2p_bad)}/{len(cfg['p2p'])}"
    )
    for label, bad in (("F2P", f2p_bad), ("P2P", p2p_bad)):
        for t in bad[:5]:
            print(f"  not passed [{label}] {t} → {status.get(t, 'missing')}")
    sys.exit(0 if not (f2p_bad or p2p_bad) else 1)


if __name__ == "__main__":
    main()
