#!/usr/bin/env python3
"""Decide which runtime packs to build (.github/workflows/runtimes.yml).

For each engine asked for, the revision to build: given by hand, or the
newest commit upstream (sd, audio), or the app's own pin (mlx — its engine is
our fork, moved by hand). A revision already in the published index, or the
one the app ships with, is skipped. Prints the build matrix as JSON.

Env: ENGINES ("sd,audio" …), SD_REV, AUDIO_REV (blank = newest upstream),
     INDEX (path of the published runtimes.json, may be missing).
"""
import json
import os
import re
import subprocess
import sys

UPSTREAM = {
    "sd": ("leejet/stable-diffusion.cpp", "master"),
    "audio": ("0xShug0/audio.cpp", "main"),
}
PLATFORMS = {
    "sd": [("macos-arm64", "macos-15"), ("windows-x64", "windows-latest"), ("linux-x64", "ubuntu-22.04")],
    "audio": [("macos-arm64", "macos-15"), ("windows-x64", "windows-latest"), ("linux-x64", "ubuntu-22.04")],
    "mlx": [("macos-arm64", "macos-15")],
}


def gh(path):
    out = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def bundled():
    """engine id -> (upstream, rev) the app ships with, from runtimes.rs."""
    src = open("src-tauri/src/runtimes.rs", encoding="utf-8").read()
    found = {}
    for m in re.finditer(r'Engine::(\w+) => Bundled \{ upstream: "([^"]+)", rev: "([0-9a-f]+)"', src):
        found[m.group(1).lower()] = (m.group(2), m.group(3))
    return found


def main():
    engines = [e.strip() for e in os.environ.get("ENGINES", "sd,audio").split(",") if e.strip()]
    index = {"packs": []}
    path = os.environ.get("INDEX", "")
    if path and os.path.exists(path):
        index = json.load(open(path, encoding="utf-8"))
    published = {(p["engine"], p["rev"]) for p in index.get("packs", [])}
    ships = bundled()
    matrix = []
    for engine in engines:
        if engine in UPSTREAM:
            repo, branch = UPSTREAM[engine]
            rev = os.environ.get(f"{engine.upper()}_REV", "").strip() or gh(f"repos/{repo}/commits/{branch}")["sha"]
        elif engine == "mlx":
            repo = ships["mlx"][0]
            rev = ships["mlx"][1]
        else:
            print(f"unknown engine {engine}", file=sys.stderr)
            sys.exit(1)
        commit = gh(f"repos/{repo}/commits/{rev}")
        rev, date = commit["sha"], commit["commit"]["committer"]["date"]
        if (engine, rev) in published:
            print(f"{engine}: {rev[:7]} is published already", file=sys.stderr)
            continue
        if engine != "mlx" and ships.get(engine, ("", ""))[1] == rev:
            print(f"{engine}: {rev[:7]} is what the app ships with", file=sys.stderr)
            continue
        version = f"{date[:10].replace('-', '.')}-{rev[:7]}"
        for platform, runner in PLATFORMS[engine]:
            matrix.append({"engine": engine, "platform": platform, "runner": runner,
                           "upstream": repo, "rev": rev, "date": date, "version": version})
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
