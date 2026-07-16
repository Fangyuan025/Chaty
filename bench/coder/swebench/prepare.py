#!/usr/bin/env python3
"""SWE-bench Verified → ChatyCoder-Bench task materializer.

Selects a deterministic, macOS-friendly 50-instance subset of SWE-bench
Verified (pure-Python repos, spec Python >= 3.8, difficulty <= 4h) and
materializes each instance into the runner's task layout:

    tasks/<instance_id>/task.md      problem statement + standard preamble
    tasks/<instance_id>/workspace/   repo at base_commit (local shared clone,
                                     future refs stripped — no peeking at the fix)
    tasks/<instance_id>/grade.sh     official-style grading: reset test files,
                                     apply test_patch, run FAIL_TO_PASS +
                                     PASS_TO_PASS via the repo's own test_cmd

Usage:
    python3 prepare.py select                 # write subset.json (no network)
    python3 prepare.py materialize [id ...]   # clone/build tasks (network)

Needs: git, uv (for pinned Pythons). Run `select` first, review subset.json,
then `materialize` (all instances, or specific ids).
"""
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE / "verified_full.json"
SPECS = HERE / "repo_specs.json"
SUBSET = HERE / "subset.json"
MIRRORS = HERE / "mirrors"
TASKS = HERE / "tasks"
ENVS = HERE / "envs"

# repo → target sample size. Pure-Python, arm64-macOS-installable repos only.
QUOTA = {
    "django/django": 24,
    "sympy/sympy": 10,
    "sphinx-doc/sphinx": 6,
    "pytest-dev/pytest": 4,
    "pylint-dev/pylint": 3,
    "psf/requests": 2,
    "pallets/flask": 1,
}
# python-build-standalone has no 3.8 for arm64 macOS — run 3.8 specs on 3.9
# (supported by every selected repo version; disclosed in the report).
PY_MAP = {"3.5": None, "3.6": None, "3.7": None, "3.8": "3.9"}


def spec_for(specs, repo, version):
    s = specs.get(repo, {}).get(str(version))
    if not s or not s.get("python"):
        return None
    py = PY_MAP.get(str(s["python"]), str(s["python"]))
    if py is None:
        return None
    return {"python": py, "test_cmd": s["test_cmd"]}


def select():
    rows = json.load(open(FULL))
    specs = json.load(open(SPECS))
    pool = {}
    for r in rows:
        if r["repo"] not in QUOTA or r["difficulty"] == ">4 hours":
            continue
        s = spec_for(specs, r["repo"], r["version"])
        if not s:
            continue
        r["_spec"] = s
        pool.setdefault(r["repo"], []).append(r)
    rng = random.Random(42)
    chosen = []
    for repo, quota in QUOTA.items():
        cands = sorted(pool.get(repo, []), key=lambda r: r["instance_id"])
        rng.shuffle(cands)
        chosen += cands[:quota]
        print(f"{repo}: {len(cands)} eligible → {min(quota, len(cands))} selected")
    json.dump(chosen, open(SUBSET, "w"), indent=1)
    print(f"\n{len(chosen)} instances → {SUBSET}")


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def write_grading(r, tdir, env):
    """Grading = per-test log parsing against F2P/P2P (official criterion).
    Directives mirror the official get_test_directives(): the test FILES
    touched by test_patch; django converts them to dotted module labels.
    (FAIL_TO_PASS entries are log-parser output — docstring descriptions for
    django — and are NOT usable as runner arguments.)"""
    repo = r["repo"]
    test_files = sorted({
        ln.split()[1][2:] for ln in r["test_patch"].splitlines()
        if ln.startswith("+++ b/") or (ln.startswith("--- a/") and not ln.endswith("/dev/null"))
    })
    py_files = [f for f in test_files if f.endswith(".py")]
    if repo == "django/django":
        labels = sorted({f[len("tests/"):-3].replace("/", ".") for f in py_files if f.startswith("tests/")})
        cmd = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 " + " ".join(labels)
    elif repo == "sympy/sympy":
        cmd = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose " + " ".join(shlex.quote(f) for f in py_files)
    else:
        # sphinx's official cmd is a tox --current-env wrapper around pytest;
        # invoke pytest directly in the same env. -rA prints the per-test
        # summary the official parser reads.
        cmd = "python -m pytest -rA " + " ".join(shlex.quote(f) for f in py_files)
    json.dump({
        "repo": repo,
        "cmd": cmd,
        "env_bin": str(env / "bin"),
        "test_files": test_files,
        "f2p": json.loads(r["FAIL_TO_PASS"]),
        "p2p": json.loads(r["PASS_TO_PASS"]),
    }, open(tdir / "grade_config.json", "w"), indent=1)
    (tdir / "grade.sh").write_text(
        f'#!/bin/bash\nexec python3 "{HERE / "grade.py"}" "{tdir}"\n'
    )
    os.chmod(tdir / "grade.sh", 0o755)


def materialize(only_ids):
    subset = json.load(open(SUBSET))
    if only_ids:
        subset = [r for r in subset if r["instance_id"] in only_ids]
    MIRRORS.mkdir(exist_ok=True)
    TASKS.mkdir(exist_ok=True)
    ENVS.mkdir(exist_ok=True)
    for r in subset:
        iid = r["instance_id"]
        tdir = TASKS / iid
        if (tdir / "grade.sh").exists():
            print(f"= {iid} (already materialized)")
            continue
        # Incomplete leftovers from an aborted run — start clean.
        if tdir.exists():
            import shutil
            shutil.rmtree(tdir)
        if (ENVS / iid).exists():
            import shutil
            shutil.rmtree(ENVS / iid)
        print(f"+ {iid}", flush=True)
        repo = r["repo"]
        mirror = MIRRORS / (repo.replace("/", "__") + ".git")
        if not mirror.exists():
            sh(["git", "clone", "--bare", f"https://github.com/{repo}.git", str(mirror)])
        ws = tdir / "workspace"
        tdir.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--shared", "--no-checkout", str(mirror), str(ws)])
        sh(["git", "checkout", "-q", r["base_commit"]], cwd=ws)
        # Strip refs so the future fix commit is unreachable by name.
        sh(["git", "remote", "remove", "origin"], cwd=ws)
        # Keep tags that are ancestors of base_commit (setuptools-scm derives
        # versions from them — deleting ALL tags gave pytest "0.1.dev" and its
        # own minversion check refused to run). Only future tags could leak
        # the fix, so only those are dropped.
        all_tags = set(subprocess.run(["git", "tag"], cwd=ws, capture_output=True, text=True).stdout.split())
        past_tags = set(subprocess.run(["git", "tag", "--merged", "HEAD"], cwd=ws,
                                       capture_output=True, text=True).stdout.split())
        future_tags = sorted(all_tags - past_tags)
        if future_tags:
            sh(["git", "tag", "-d", *future_tags], cwd=ws, stdout=subprocess.DEVNULL)
        # NOTE: `git branch` prints "(HEAD detached at …)" on a detached HEAD —
        # for-each-ref lists only real branch refs.
        for b in subprocess.run(["git", "for-each-ref", "refs/heads", "--format=%(refname:short)"],
                                cwd=ws, capture_output=True, text=True).stdout.split():
            sh(["git", "branch", "-D", b], cwd=ws, stdout=subprocess.DEVNULL)

        # Per-instance interpreter + editable install (env lives OUTSIDE the
        # workspace so the agent can't touch it).
        env = ENVS / iid
        py = r["_spec"]["python"]
        sh(["uv", "venv", "--python", py, str(env)])
        # uv venvs ship without pip — install through uv against the env's python.
        pip = ["uv", "pip", "install", "-q", "--python", str(env / "bin" / "python")]
        try:
            sh([*pip, "-e", str(ws) + ("[test]" if repo == "sphinx-doc/sphinx" else "")])
        except subprocess.CalledProcessError:
            sh([*pip, "-e", str(ws)])

        (tdir / "task.md").write_text(
            "You are working in a checkout of an open-source Python project. "
            "Fix the issue described below by modifying the source code. Do not "
            "modify any test files — the fix will be verified by the project's "
            "own test suite. Use the project's virtualenv at "
            f"`{env}/bin/python` for any commands.\n\n---\n\n"
            + r["problem_statement"]
        )
        (tdir / "test_patch.diff").write_text(r["test_patch"])
        # Gold patch is for grade-pipeline self-validation only — it lives
        # OUTSIDE workspace/, the agent never sees it.
        (tdir / "gold_patch.diff").write_text(r["patch"])

        write_grading(r, tdir, env)


def fixdeps():
    """Install each instance's official spec pip_packages (test deps) into its
    env — `pip install -e .` alone doesn't bring pytest & friends. Best-effort
    per package so one exotic dep (pyenchant needs a brew lib) doesn't sink
    the rest; validate decides what's usable."""
    subset = {r["instance_id"]: r for r in json.load(open(SUBSET))}
    specs = json.load(open(SPECS))
    for tdir in sorted(TASKS.iterdir()):
        r = subset.get(tdir.name)
        env = ENVS / tdir.name
        if not r or not env.exists():
            continue
        s = specs.get(r["repo"], {}).get(str(r["version"]), {})
        pkgs = list(s.get("pip_packages") or [])
        cmd = s.get("test_cmd") or ""
        if "pytest" in cmd and not any(p.split("==")[0].lower() == "pytest" for p in pkgs):
            pkgs.append("pytest")
        if not pkgs:
            continue
        py = str(env / "bin" / "python")
        done, failed = [], []
        for p in pkgs:
            ok_ = subprocess.run(["uv", "pip", "install", "-q", "--python", py, p],
                                 capture_output=True).returncode == 0
            (done if ok_ else failed).append(p)
        print(f"{tdir.name}: +{len(done)}" + (f" (failed: {failed})" if failed else ""))


def validate():
    """Gold-patch validation: an instance only counts for the benchmark if the
    OFFICIAL fix makes grading pass on this machine (red without fix is implied
    by F2P semantics). Instances failing here are macOS-env-incompatible and
    get excluded — recorded in validated.json."""
    import shutil
    import tempfile
    results = {}
    if (HERE / "validated.json").exists():
        results = json.load(open(HERE / "validated.json"))
    for tdir in sorted(TASKS.iterdir()):
        iid = tdir.name
        if not (tdir / "grade.sh").exists():
            continue
        if iid in results:
            print(f"= {iid}: {results[iid]} (cached)")
            continue
        tmp = Path(tempfile.mkdtemp(prefix=f"gold-{iid}-"))
        ws = tmp / "ws"
        shutil.copytree(tdir / "workspace", ws)
        try:
            sh(["git", "apply", "--whitespace=nowarn", str(tdir / "gold_patch.diff")], cwd=ws)
            g = subprocess.run(["bash", str(tdir / "grade.sh")], cwd=ws,
                               capture_output=True, text=True, timeout=1800)
            results[iid] = "ok" if g.returncode == 0 else "env-incompatible"
            if g.returncode != 0:
                tail = "\n".join(g.stdout.splitlines()[-3:])
                print(f"✗ {iid}: env-incompatible\n{tail}")
            else:
                print(f"✓ {iid}: ok")
        except Exception as e:
            results[iid] = f"error: {e}"
            print(f"✗ {iid}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        json.dump(results, open(HERE / "validated.json", "w"), indent=1)
    ok = sum(1 for v in results.values() if v == "ok")
    print(f"\n{ok}/{len(results)} instances validated ok")


def regrade():
    """Refresh grade_config.json/grade.sh for already-materialized tasks."""
    subset = {r["instance_id"]: r for r in json.load(open(SUBSET))}
    for tdir in sorted(TASKS.iterdir()):
        r = subset.get(tdir.name)
        if r and (tdir / "workspace").exists():
            write_grading(r, tdir, ENVS / tdir.name)
            print(f"regraded {tdir.name}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "select"
    if mode == "select":
        select()
    elif mode == "materialize":
        materialize(set(sys.argv[2:]))
    elif mode == "regrade":
        regrade()
    elif mode == "fixdeps":
        fixdeps()
    elif mode == "validate":
        validate()
    else:
        sys.exit(f"unknown mode {mode}")
