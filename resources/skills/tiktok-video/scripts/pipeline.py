"""One-shot pipeline: tts -> assets -> bgm -> compose -> check.

Usage: python scripts/pipeline.py <project_dir> [--skip-assets] [--skip-tts]
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stage(name, args_):
    print(f"\n===== {name} =====", flush=True)
    r = subprocess.run([sys.executable, str(HERE / f"{name}.py"), *args_])
    if r.returncode != 0:
        print(f"pipeline stopped at stage '{name}' (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--skip-assets", action="store_true")
    a = ap.parse_args()
    if not a.skip_tts:
        stage("tts", [a.project_dir])
    if not a.skip_assets:
        stage("assets", [a.project_dir])
    stage("bgm", [a.project_dir])
    stage("compose", [a.project_dir])
    stage("check", [a.project_dir])


if __name__ == "__main__":
    main()
