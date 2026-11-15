#!/usr/bin/env python3
"""Decide which runtime packs to build (.github/workflows/runtimes.yml).

A pack is the engine wrapper of the newest Chaty release (its source, by tag:
what users run, not what main has yet) built against a newer upstream engine:

  llama  the newest build ggml-org/llama.cpp publishes (one per commit that
         passed its CI)
  sd     the newest stable-diffusion.cpp commit
  audio  the newest audio.cpp commit
  mlx    the newest mlx-swift-lm commit

A revision already published for a platform from this release's own commit,
or the one that release ships with, is skipped — one published from an older
release, or from this release before a re-cut moved its tag, is built again:
an app runs only packs of its own build (runtimes.rs, built_for_this_app).
Prints the build matrix as JSON.

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
    # mlx-swift needs Swift tools 6.3: Xcode 26.4+, on macOS 26 (release.yml).
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


# stable-diffusion.cpp made its C API opaque (options, pictures and results set
# and read by key) in 532d2c1; a wrapper from before that cannot build against
# it. The header's marker, and the wrapper's, for the opaque form.
SD_OPAQUE_HEADER = "sd_ctx_opt_create"
SD_OPAQUE_WRAPPER = "CHATY_SD_OPT_API"


def file_at(repo, path, ref):
    try:
        return base64.b64decode(gh(f"repos/{repo}/contents/{path}?ref={ref}")["content"]).decode()
    except subprocess.CalledProcessError:
        return ""


def sd_rev_for(source, rev):
    """The newest stable-diffusion.cpp commit the release's wrapper builds
    against: `rev`, unless upstream's API moved past the wrapper — then the
    commit before the move. A release from before the opaque API keeps its
    packs coming, not a build that fails every day until the next release."""
    repo = os.environ.get("GITHUB_REPOSITORY", "Fangyuan025/Chaty")
    upstream = UPSTREAM["sd"][0]
    if SD_OPAQUE_WRAPPER in file_at(repo, "src-tauri/sd-sidecar/main.cpp", source):
        return rev
    if SD_OPAQUE_HEADER not in file_at(upstream, "include/stable-diffusion.h", rev):
        return rev
    # The commit that brought the opaque header in: the oldest of the header's
    # recent changes that has it. Its parent is the newest without.
    changes = gh(f"repos/{upstream}/commits?path=include/stable-diffusion.h&sha={rev}&per_page=30")
    introduced = None
    for c in changes:
        if SD_OPAQUE_HEADER in file_at(upstream, "include/stable-diffusion.h", c["sha"]):
            introduced = c
        else:
            break
    if introduced is None or not introduced.get("parents"):
        return rev
    before = introduced["parents"][0]["sha"]
    print(f"sd: {source}'s wrapper predates upstream's opaque API ({introduced['sha'][:7]}); building {before[:7]}", file=sys.stderr)
    return before


def main():
    repo = os.environ.get("GITHUB_REPOSITORY", "Fangyuan025/Chaty")
    source = os.environ.get("SOURCE", "").strip() or gh(f"repos/{repo}/releases/latest")["tag_name"]
    ships = bundled(source)
    source_sha = gh(f"repos/{repo}/commits/{source}")["sha"]
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
    # By platform: one that failed to build is tried again next time, while
    # the others of that revision stay published.
    published = {(p["engine"], p["rev"], p["platform"], p.get("source", "")) for p in index.get("packs", [])}
    matrix = []
    for engine in engines:
        if engine not in UPSTREAM:
            print(f"unknown engine {engine}", file=sys.stderr)
            sys.exit(1)
        repo_up = UPSTREAM[engine][0]
        rev = os.environ.get(f"{engine.upper()}_REV", "").strip()
        if not rev:
            rev = newest(engine)
            if engine == "sd":
                rev = sd_rev_for(source, rev)
        commit = gh(f"repos/{repo_up}/commits/{rev}")
        rev, date = commit["sha"], commit["commit"]["committer"]["date"]
        if ships.get(engine, ("", ""))[1] == rev:
            print(f"{engine}: {rev[:7]} is what {source} ships with", file=sys.stderr)
            continue
        version = f"{date[:10].replace('-', '.')}-{rev[:7]}"
        for platform, runner in PLATFORMS[engine]:
            if (engine, rev, platform, source_sha) in published:
                print(f"{engine}: {rev[:7]} for {platform} is published from {source} already", file=sys.stderr)
                continue
            matrix.append({"engine": engine, "platform": platform, "runner": runner, "source": source,
                           "upstream": repo_up, "rev": rev, "date": date, "version": version})
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
