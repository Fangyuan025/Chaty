#!/usr/bin/env python3
"""Merge newly built runtime packs into the index (.github/workflows/runtimes.yml).

  runtimes-index.py <published runtimes.json or ""> <packs dir> <out runtimes.json>

Each pack in <packs dir> is <name>.zip with <name>.json beside it: its index
entry, less the URL. The three newest packs of each engine and platform are
kept; the zip names no longer listed are printed, to be removed from the
release.
"""
import json
import os
import sys

BASE = "https://github.com/Fangyuan025/Chaty/releases/download/runtimes/"
KEEP = 3


def main():
    old_path, packs_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    packs = []
    if old_path and os.path.exists(old_path):
        packs = json.load(open(old_path, encoding="utf-8")).get("packs", [])
    before = {p["url"].rsplit("/", 1)[-1] for p in packs}
    for name in sorted(os.listdir(packs_dir)):
        if not name.endswith(".json"):
            continue
        entry = json.load(open(os.path.join(packs_dir, name), encoding="utf-8"))
        zip_name = name[: -len(".json")] + ".zip"
        entry["url"] = BASE + zip_name
        packs = [p for p in packs if not (p["engine"] == entry["engine"] and p["platform"] == entry["platform"] and p["rev"] == entry["rev"])]
        packs.append(entry)
    kept = []
    for key in sorted({(p["engine"], p["platform"]) for p in packs}):
        group = sorted((p for p in packs if (p["engine"], p["platform"]) == key), key=lambda p: (p["date"], p["version"]), reverse=True)
        kept.extend(group[:KEEP])
    kept.sort(key=lambda p: (p["engine"], p["platform"], p["date"]))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "packs": kept}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    after = {p["url"].rsplit("/", 1)[-1] for p in kept}
    for gone in sorted(before - after):
        print(gone)


if __name__ == "__main__":
    main()
