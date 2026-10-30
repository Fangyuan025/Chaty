#!/usr/bin/env python3
"""Decide which runtime packs to build (.github/workflows/runtimes.yml).

A pack is the engine wrapper of the newest Chaty release (its source, by tag:
what users run, not what main has yet) built against a newer upstream engine:

  llama  the newest build ggml-org/llama.cpp publishes (one per commit that
         passed its CI)
  sd     the newest stable-diffusion.cpp commit
  audio  the newest audio.cpp commit
  mlx    the newest mlx-swift-lm commit

A revision already in the published index, or the one that release ships
with, is skipped. Prints the build matrix as JSON.

Env: ENGINES ("llama,sd,audio,mlx" …), LLAMA_REV, SD_REV, AUDIO_REV, MLX_REV
     (blank = newest upstream), SOURCE (a tag; blank = the latest release),
     INDEX (path of the published runtimes.json, may be missing),
     GITHUB_REPOSITORY.
"""
import base64
import json
import os
import re
import subprocess
import sys

UPSTREAM = {
    "llama": ("ggml-org/llama.cpp", None),
    "sd": ("leejet/stable-diffusion.cpp", "master"),
    "audio": ("0xShug0/audio.cpp", "main"),
    "mlx": ("ml-explore/mlx-swift-lm", "main"),
}
DESKTOP = [("macos-arm64", "macos-15"), ("windows-x64", "windows-latest"), ("linux-x64", "ubuntu-22.04")]
PLATFORMS = {
    "llama": DESKTOP,
    "sd": DESKTOP,
    "audio": DESKTOP,
    # The newer mlx-swift manifests need Xcode 26's toolchain.
    "mlx": [("macos-arm64", "macos-26")],
}


def gh(path):
    out = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def bundled(source):
    """engine id -> (upstream, rev) the release ships with, from its runtimes.rs."""
    repo = os.environ.get("GITHUB_REPOSITORY", "Fangyuan025/Chaty")
    try:
        src = base64.b64decode(gh(f"repos/{repo}/contents/src-tauri/src/runtimes.rs?ref={source}")["content"]).decode()
    except subprocess.CalledProcessError:
        return None
    found = {}
    for m in re.finditer(r'Engine::(\w+) => Bundled \{ upstream: "([^"]+)", rev: "([0-9a-f]+)"', src):
        found[m.group(1).lower()] = (m.group(2), m.group(3))
    return found


def newest(engine):
    repo, branch = UPSTREAM[engine]
    if branch is None:
        # llama.cpp tags every build it publishes; the newest one, prerelease or not.
        tag = gh(f"repos/{repo}/releases?per_page=1")[0]["tag_name"]
        return gh(f"repos/{repo}/commits/{tag}")["sha"]
    return gh(f"repos/{repo}/commits/{branch}")["sha"]


def main():
    repo = os.environ.get("GITHUB_REPOSITORY", "Fangyuan025/Chaty")
    source = os.environ.get("SOURCE", "").strip() or gh(f"repos/{repo}/releases/latest")["tag_name"]
    ships = bundled(source)
    if ships is None:
        # A release from before runtime packs: nothing to build them from.
        print(f"{source} has no runtime packs", file=sys.stderr)
        print("[]")
        return
    engines = [e.strip() for e in os.environ.get("ENGINES", "llama,sd,audio,mlx").split(",") if e.strip()]
    index = {"packs": []}
    path = os.environ.get("INDEX", "")
    if path and os.path.exists(path):
        index = json.load(open(path, encoding="utf-8"))
    published = {(p["engine"], p["rev"]) for p in index.get("packs", [])}
    matrix = []
    for engine in engines:
        if engine not in UPSTREAM:
            print(f"unknown engine {engine}", file=sys.stderr)
            sys.exit(1)
        repo_up = UPSTREAM[engine][0]
        rev = os.environ.get(f"{engine.upper()}_REV", "").strip() or newest(engine)
        commit = gh(f"repos/{repo_up}/commits/{rev}")
        rev, date = commit["sha"], commit["commit"]["committer"]["date"]
        if (engine, rev) in published:
            print(f"{engine}: {rev[:7]} is published already", file=sys.stderr)
            continue
        if ships.get(engine, ("", ""))[1] == rev:
            print(f"{engine}: {rev[:7]} is what {source} ships with", file=sys.stderr)
            continue
        version = f"{date[:10].replace('-', '.')}-{rev[:7]}"
        for platform, runner in PLATFORMS[engine]:
            matrix.append({"engine": engine, "platform": platform, "runner": runner, "source": source,
                           "upstream": repo_up, "rev": rev, "date": date, "version": version})
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
