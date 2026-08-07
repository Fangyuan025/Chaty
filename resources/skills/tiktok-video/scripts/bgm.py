"""Stage 3 — background music.

Curated CC-BY tracks by Kevin MacLeod (incompetech.com), fetched primarily
from Wikimedia Commons (reliable CDN) with incompetech.com as fallback, and
cached locally. Attribution is recorded into the project credits (required by
CC-BY — the check stage prints a ready-to-paste credit block).

You can also drop your own music into assets/bgm/ and reference it via
storyboard: {"bgm": {"file": "assets/bgm/mytrack.mp3"}}, or set mood "none".

Usage: python scripts/bgm.py <project_dir> [--mood upbeat] [--list]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

from common import (CACHE_DIR, SKILL_ROOT, audio_duration, die, download, http_get,
                    load_storyboard, log, project_paths)

INCOMPETECH = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"

# mood -> ordered candidates; each is a Wikimedia Commons file title
MOODS = {
    "upbeat": ["Monkeys Spinning Monkeys (ISRC USUAN1400011).mp3",
               "Kevin MacLeod - Carefree.ogg",
               "Kevin MacLeod - Early Riser.ogg"],
    "funny": ["Sneaky Snitch (ISRC USUAN1100772).mp3",
              "Fluffing a Duck (ISRC USUAN1100768).mp3",
              "Kevin MacLeod - Pixel Peeker Polka - faster.ogg"],
    "inspiring": ["Kevin MacLeod - Clean Soul.ogg",
                  "Kevin MacLeod - Dream Culture.ogg",
                  "Kevin MacLeod - Reaching Out.ogg"],
    "chill": ["Kevin MacLeod - Tranquility.ogg",
              "Kevin MacLeod - Autumn Day.ogg",
              "Kevin MacLeod - Dream Culture.ogg"],
    "tech": ["Kevin MacLeod - Lift Motif.ogg",
             "Kevin MacLeod - Unity.ogg",
             "Kevin MacLeod - Smoother Move.ogg"],
    "mystery": ["Kevin MacLeod - Ghost Dance.ogg",
                "Sugar Plum Dark Mix (Kevin MacLeod) (ISRC USUAN1100623).oga",
                "Investigations (ISRC USUAN1100646).mp3"],
    "epic": ["Kevin MacLeod - Call to Adventure.ogg",
             "Kevin MacLeod - Virtutes Instrumenti.ogg",
             "Kevin MacLeod - Sovereign Quarter.ogg"],
    "sad": ["Kevin MacLeod - Mourning Song.ogg",
            "Kevin MacLeod - Winter Reflections.ogg",
            "Kevin MacLeod - Resignation.ogg"],
    "horror": ["Kevin MacLeod - Horroriffic.ogg",
               "Kevin MacLeod - Ghost Dance.ogg",
               "Sugar Plum Dark Mix (Kevin MacLeod) (ISRC USUAN1100623).oga"],
}


def clean_name(file_title: str) -> str:
    n = re.sub(r"^(File:)?(Kevin MacLeod ?[-~] ?)?", "", file_title)
    n = re.sub(r"\s*\((ISRC [^)]*|Kevin MacLeod)\)", "", n)
    return re.sub(r"\.(ogg|oga|mp3|wav)$", "", n).strip()


def commons_direct_url(file_title: str) -> str | None:
    try:
        r = http_get("https://commons.wikimedia.org/w/api.php?action=query"
                     f"&titles=File:{file_title}&prop=imageinfo&iiprop=url&format=json",
                     timeout=20)
        pages = (r.json().get("query") or {}).get("pages") or {}
        for p in pages.values():
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("url"):
                return ii["url"]
    except Exception:
        pass
    return None


def get_track(file_title: str) -> Path | None:
    cache = CACHE_DIR / "bgm"
    cache.mkdir(parents=True, exist_ok=True)
    ext = Path(file_title).suffix or ".mp3"
    dest = cache / (re.sub(r"[^A-Za-z0-9]+", "_", clean_name(file_title)) + ext)
    if not (dest.exists() and dest.stat().st_size > 100_000):
        url = commons_direct_url(file_title)
        ok = url and download(url, dest, max_bytes=40_000_000, attempts=3)
        if not ok:  # fallback: incompetech direct mp3
            alt = INCOMPETECH + clean_name(file_title).replace(" ", "%20") + ".mp3"
            dest = dest.with_suffix(".mp3")
            log(f"[bgm] commons failed, trying incompetech for {clean_name(file_title)}")
            if not download(alt, dest, max_bytes=40_000_000, attempts=4):
                return None
    try:
        if audio_duration(dest) < 20:
            dest.unlink(missing_ok=True)
            return None
    except Exception:
        dest.unlink(missing_ok=True)
        return None
    return dest


def install(paths, src: Path):
    for old in paths["root"].glob("bgm.*"):
        old.unlink()
    dest = paths["root"] / ("bgm" + src.suffix)
    shutil.copyfile(src, dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--mood", help="override storyboard mood")
    ap.add_argument("--track", help="exact Wikimedia Commons file title")
    ap.add_argument("--list", action="store_true", help="list moods and exit")
    args = ap.parse_args()
    if args.list:
        for m, ts in MOODS.items():
            print(f"{m}: {', '.join(clean_name(t) for t in ts)}")
        return

    sb = load_storyboard(args.project_dir)
    paths = project_paths(args.project_dir)
    bgm_cfg = sb["bgm"]
    credit_file = paths["root"] / "bgm_credit.json"

    if bgm_cfg.get("file"):
        src = Path(bgm_cfg["file"]).expanduser()
        if not src.is_absolute():
            for base in (paths["root"], SKILL_ROOT):
                if (base / src).exists():
                    src = base / src
                    break
        if not src.exists():
            die(f"bgm file not found: {bgm_cfg['file']}")
        install(paths, src)
        credit_file.write_text(json.dumps({"credit": f"Music: {src.name} (user-provided)"}))
        log(f"[bgm] using user file {src.name}")
        return

    mood = args.mood or bgm_cfg.get("mood", "upbeat")
    if mood == "none":
        for old in paths["root"].glob("bgm.*"):
            old.unlink()
        credit_file.write_text(json.dumps({"credit": ""}))
        log("[bgm] disabled")
        return

    titles = [args.track] if args.track else MOODS.get(mood)
    if not titles:
        die(f"unknown mood '{mood}'. Available: {', '.join(MOODS)} | none")
    for t in titles:
        p = get_track(t)
        if p:
            install(paths, p)
            credit_file.write_text(json.dumps({
                "credit": f'Music: "{clean_name(t)}" — Kevin MacLeod (incompetech.com), '
                          f"CC BY, via Wikimedia Commons"}))
            log(f"[bgm] ready: {clean_name(t)} ({audio_duration(p):.0f}s)")
            return
    die("all bgm candidates failed to download — check network, or use bgm mood 'none' "
        "or drop a local mp3 into assets/bgm/ and set bgm.file")


if __name__ == "__main__":
    main()
