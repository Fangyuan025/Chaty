"""Stage 3 — background music.

Three sources, in priority order:
  1. bgm.file  — your own track (e.g. a trending sound you are licensed to use)
  2. bgm.query — style search on Openverse audio (Jamendo CC music, no key):
                 "trap beat", "lofi hip hop", "epic cinematic", "synthwave"…
                 ND/NC licenses are excluded automatically.
  3. bgm.mood  — curated Kevin MacLeod tracks (Wikimedia Commons mirror)

Whatever the source, the chosen track is beat-analyzed (scripts/beats.py) and
the grid is saved to <project>/bgm_beats.json so compose.py can cut on the
beat (卡点). Attribution is recorded for the credit block.

Usage: python scripts/bgm.py <project_dir> [--mood upbeat | --query "trap beat"] [--list]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

from common import (CACHE_DIR, SKILL_ROOT, audio_duration, die, download, http_get,
                    load_storyboard, log, project_paths, run)

INCOMPETECH = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"

# mood -> candidates (Wikimedia Commons file titles). These MacLeod staples
# are the exact tracks countless faceless/营销号 channels run on — a proven
# default. One is picked per project (seeded by title) so repeated use of the
# same mood doesn't always sound identical; use bgm.query for wider variety.
MOODS = {
    "upbeat": ["Monkeys Spinning Monkeys (ISRC USUAN1400011).mp3",
               "Kevin MacLeod - Carefree.ogg",
               "Kevin MacLeod - Early Riser.ogg",
               "Kevin MacLeod - Smoother Move.ogg",
               "Kevin MacLeod - Danse Morialta.ogg"],
    "funny": ["Sneaky Snitch (ISRC USUAN1100772).mp3",
              "Fluffing a Duck (ISRC USUAN1100768).mp3",
              "Kevin MacLeod - Pixel Peeker Polka - faster.ogg",
              "Kevin MacLeod - Scheming Weasel (faster).wav"],
    "inspiring": ["Kevin MacLeod - Clean Soul.ogg",
                  "Kevin MacLeod - Dream Culture.ogg",
                  "Kevin MacLeod - Reaching Out.ogg",
                  "Kevin MacLeod - Inner Light.ogg",
                  "Kevin MacLeod - Windswept.ogg"],
    "chill": ["Kevin MacLeod - Tranquility.ogg",
              "Kevin MacLeod - Autumn Day.ogg",
              "Kevin MacLeod - Dream Culture.ogg",
              "Kevin MacLeod - Snow Drop.ogg",
              "Kevin MacLeod - Calmant.ogg"],
    "tech": ["Kevin MacLeod - Lift Motif.ogg",
             "Kevin MacLeod - Unity.ogg",
             "Kevin MacLeod - Smoother Move.ogg",
             "Stratosphere, (MacLeod, Kevin).oga"],
    "mystery": ["Kevin MacLeod - Ghost Dance.ogg",
                "Sugar Plum Dark Mix (Kevin MacLeod) (ISRC USUAN1100623).oga",
                "Investigations (ISRC USUAN1100646).mp3"],
    "epic": ["Kevin MacLeod - Call to Adventure.ogg",
             "Kevin MacLeod - Virtutes Instrumenti.ogg",
             "Kevin MacLeod - Sovereign Quarter.ogg",
             "Kevin MacLeod - Enchanted Journey.ogg"],
    "sad": ["Kevin MacLeod - Mourning Song.ogg",
            "Kevin MacLeod - Winter Reflections.ogg",
            "Kevin MacLeod - Resignation.ogg",
            "Kevin MacLeod - Sonatina.ogg"],
    "horror": ["Kevin MacLeod - Horroriffic.ogg",
               "Kevin MacLeod - Ghost Dance.ogg",
               "Sugar Plum Dark Mix (Kevin MacLeod) (ISRC USUAN1100623).oga"],
}


def clean_name(file_title: str) -> str:
    n = re.sub(r"^(File:)?(Kevin MacLeod ?[-~] ?)?", "", file_title)
    n = re.sub(r"\s*\((ISRC [^)]*|Kevin MacLeod|MacLeod, Kevin)\)", "", n)
    return re.sub(r"\.(ogg|oga|mp3|wav)$", "", n).strip(" ,")


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


# style words found in a query -> ccMixter tag sets (AND semantics, lic=open
# guarantees commercial-safe CC). ccMixter is remix culture: real produced
# beats with BPM metadata — much closer to a 抖音 sound than stock libraries.
CCMIXTER_STYLES = [
    (("phonk", "trap", "drift"), "trap,instrumental"),
    (("hip", "hop", "rap", "boom"), "hip_hop,instrumental"),
    (("lofi", "chill", "lo-fi", "mellow"), "chill,instrumental"),
    (("edm", "dance", "club", "house"), "electronic,dance"),
    (("synthwave", "retro", "electro"), "electronic,instrumental"),
    (("epic", "cinematic", "orchestral", "trailer"), "cinematic"),
    (("funny", "quirky", "comedy"), "quirky,instrumental"),
]


def ccmixter_search(query: str, n=15):
    qtok = set(re.findall(r"[a-z]+", query.lower()))
    tags = next((t for words, t in CCMIXTER_STYLES if qtok & set(words)), None)
    if not tags:
        return []
    try:
        r = http_get(f"http://ccmixter.org/api/query?f=json&tags={tags}&lic=open"
                     f"&sort=rank&limit={n}", timeout=25)
        out = []
        for u in r.json():
            mp3 = next((f.get("download_url") for f in u.get("files", [])
                        if str(f.get("download_url", "")).endswith(".mp3")), None)
            if not mp3:
                continue
            bpm = (u.get("upload_extra") or {}).get("bpm")
            try:
                bpm = float(bpm)
            except (TypeError, ValueError):
                bpm = None
            if bpm and not 70 <= bpm <= 180:
                continue
            out.append({"url": mp3, "title": u.get("upload_name") or "", "dur": 0,
                        "creator": u.get("user_name") or "",
                        "license": u.get("license_name") or "CC BY",
                        "source": u.get("file_page_url") or mp3,
                        "meta_bpm": bpm, "provider": "ccmixter",
                        "headers": {"Referer": "http://ccmixter.org/"}})
        return out
    except Exception as e:
        log(f"[bgm] ccmixter error: {e}")
        return []


def openverse_audio(query: str, n=20):
    """CC music search (Jamendo etc.). Only remix-safe commercial licenses."""
    try:
        r = http_get("https://api.openverse.org/v1/audio/"
                     f"?q={query}&license=by,by-sa,cc0,pdm&category=music&page_size={n}",
                     timeout=25)
        out = []
        qtok = set(re.findall(r"[a-z]+", query.lower()))
        for it in r.json().get("results", []):
            if not it.get("url"):
                continue
            dur = (it.get("duration") or 0) / 1000
            if not 45 <= dur <= 600:
                continue
            title = it.get("title") or ""
            genres = " ".join(it.get("genres") or [])
            ttok = set(re.findall(r"[a-z]+", f"{title} {genres}".lower()))
            score = (2.0 * len(qtok & ttok)
                     + (1.5 if it.get("source") == "jamendo" else 0)
                     + (1.0 if 90 <= dur <= 420 else 0)
                     + (1.2 if {"instrumental", "beat", "bgm"} & ttok else 0))
            out.append((score, {"url": it["url"], "title": title, "dur": dur,
                                "creator": it.get("creator") or "",
                                "license": f"CC {it.get('license', '').upper()}",
                                "source": it.get("foreign_landing_url") or it["url"]}))
        return [c for _, c in sorted(out, key=lambda x: -x[0])]
    except Exception as e:
        log(f"[bgm] openverse audio error: {e}")
        return []


BEAT_CONF_GATE = 3.0     # candidates below this rhythm confidence are rejected
BEAT_BPM_RANGE = (75, 175)


def get_query_track(query: str):
    """Search ccMixter (style tags) + Openverse, then SCREEN candidates with
    the beat tracker: a track only qualifies if it has a confident, danceable
    beat grid — the objective filter that separates produced beat music from
    stock ambience and vocal ballads. Returns (path, candidate, beats)."""
    from beats import analyze

    cache = CACHE_DIR / "bgm"
    cache.mkdir(parents=True, exist_ok=True)
    cands = (ccmixter_search(query) + openverse_audio(query))[:8]
    best = None   # (confidence, path, cand, beats) fallback if nobody passes
    for c in cands:
        ext = ".mp3" if ".mp3" in c["url"].lower() else ".ogg"
        dest = cache / (re.sub(r"[^A-Za-z0-9]+", "_", c["title"])[:40] + ext)
        if not (dest.exists() and dest.stat().st_size > 100_000):
            if not download(c["url"], dest, max_bytes=60_000_000, attempts=3,
                            headers=c.get("headers")):
                continue
        try:
            if audio_duration(dest) < 40:
                dest.unlink(missing_ok=True)
                continue
            res = analyze(dest)
        except Exception:
            dest.unlink(missing_ok=True)
            continue
        ok = res["confidence"] >= BEAT_CONF_GATE and \
            BEAT_BPM_RANGE[0] <= res["bpm"] <= BEAT_BPM_RANGE[1]
        log(f"[bgm] candidate: {c['title'][:38]!r} [{c.get('provider','openverse')}] "
            f"{res['bpm']:.0f}BPM conf={res['confidence']}"
            + ("  ✓" if ok else "  (weak beat, skipping)"))
        if ok:
            return dest, c, res
        if best is None or res["confidence"] > best[0]:
            best = (res["confidence"], dest, c, res)
    if best:
        log("[bgm] no candidate passed the beat gate — using the strongest one")
        return best[1], best[2], best[3]
    return None, None, None


# 抖音 signature treatments. spedup ≈ 1.25x with raised pitch (the ubiquitous
# "sped up" sound); slowed ≈ 0.88x with echo tail ("slowed + reverb").
VIBES = {
    "spedup": "aresample=48000,asetrate=48000*1.16,aresample=48000,atempo=1.08",
    "slowed": "aresample=48000,asetrate=48000*0.88,aresample=48000,"
              "aecho=0.7:0.65:60|110:0.30|0.22",
}
VIBE_SPEED = {"spedup": 1.16 * 1.08, "slowed": 0.88}


def apply_vibe(paths, installed: Path, vibe: str) -> Path:
    if vibe not in VIBES:
        die(f"unknown vibe '{vibe}'. Available: {', '.join(VIBES)}")
    out = paths["root"] / "bgm.m4a"
    tmp = paths["root"] / "bgm_vibe_tmp.m4a"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", installed,
         "-af", VIBES[vibe], "-c:a", "aac", "-b:a", "192k", tmp])
    for old in paths["root"].glob("bgm.*"):
        old.unlink()
    tmp.rename(out)
    log(f"[bgm] vibe '{vibe}' applied")
    return out


def analyze_beats(paths, track_path, track_name, bpm_hint=None):
    """Beat-grid the installed BGM for cut-on-beat compose (best effort)."""
    out = paths["root"] / "bgm_beats.json"
    out.unlink(missing_ok=True)
    try:
        from beats import analyze
        res = analyze(track_path, bpm_hint=bpm_hint)
        res["track"] = track_name
        out.write_text(json.dumps(res))
        log(f"[bgm] beat grid: {res['bpm']} BPM, {len(res['beats'])} beats, "
            f"confidence {res['confidence']}"
            + ("" if res["confidence"] >= 2 else " (low — compose will skip beat sync)"))
    except Exception as e:
        log(f"[bgm] beat analysis failed ({e}) — composing without beat sync")


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
    ap.add_argument("--query", help="style search, e.g. \"trap\" \"lofi\" \"epic cinematic\"")
    ap.add_argument("--track", help="exact Wikimedia Commons file title")
    ap.add_argument("--vibe", help="post-process: spedup | slowed")
    ap.add_argument("--list", action="store_true", help="list moods and exit")
    args = ap.parse_args()
    if args.list:
        for m, ts in MOODS.items():
            print(f"{m}: {', '.join(clean_name(t) for t in ts)}")
        print('query: any style words, e.g. "trap beat" "lofi hip hop" "epic cinematic"')
        return

    sb = load_storyboard(args.project_dir)
    paths = project_paths(args.project_dir)
    bgm_cfg = sb["bgm"]
    credit_file = paths["root"] / "bgm_credit.json"
    vibe = args.vibe or bgm_cfg.get("vibe")

    def finish(dest, track_name, beats_res=None):
        if vibe:
            hint = beats_res["bpm"] * VIBE_SPEED[vibe] if beats_res else None
            dest = apply_vibe(paths, dest, vibe)
            track_name = f"{track_name} ({vibe})"
            analyze_beats(paths, dest, track_name, bpm_hint=hint)
            return
        if beats_res is not None:
            beats_res["track"] = track_name
            (paths["root"] / "bgm_beats.json").write_text(json.dumps(beats_res))
            log(f"[bgm] beat grid: {beats_res['bpm']} BPM, confidence {beats_res['confidence']}")
        else:
            analyze_beats(paths, dest, track_name)

    if bgm_cfg.get("file") and not (args.query or args.mood or args.track):
        src = Path(bgm_cfg["file"]).expanduser()
        if not src.is_absolute():
            for base in (paths["root"], SKILL_ROOT):
                if (base / src).exists():
                    src = base / src
                    break
        if not src.exists():
            die(f"bgm file not found: {bgm_cfg['file']}")
        dest = install(paths, src)
        credit_file.write_text(json.dumps({"credit": f"Music: {src.name} (user-provided)"}))
        log(f"[bgm] using user file {src.name}")
        finish(dest, src.name)
        return

    query = args.query or bgm_cfg.get("query")
    if query and not (args.mood or args.track):
        p, c, beats_res = get_query_track(query)
        if p:
            dest = install(paths, p)
            credit_file.write_text(json.dumps({
                "credit": f'Music: "{c["title"]}" — {c["creator"]} ({c["license"]}), '
                          f'{c["source"]}'}))
            log(f"[bgm] ready: {c['title'][:48]} ({audio_duration(p):.0f}s)")
            finish(dest, c["title"], beats_res)
            return
        log(f"[bgm] no usable result for query '{query}' — falling back to mood table")

    mood = args.mood or bgm_cfg.get("mood", "upbeat")
    if mood == "none":
        for old in paths["root"].glob("bgm.*"):
            old.unlink()
        (paths["root"] / "bgm_beats.json").unlink(missing_ok=True)
        credit_file.write_text(json.dumps({"credit": ""}))
        log("[bgm] disabled")
        return

    titles = [args.track] if args.track else MOODS.get(mood)
    if not titles:
        die(f"unknown mood '{mood}'. Available: {', '.join(MOODS)} | none, "
            f"or use a style query instead")
    if not args.track:
        # vary the pick across projects, reproducibly within one
        import random
        titles = list(titles)
        random.Random(sb.get("title", "")).shuffle(titles)
    for t in titles:
        p = get_track(t)
        if p:
            dest = install(paths, p)
            credit_file.write_text(json.dumps({
                "credit": f'Music: "{clean_name(t)}" — Kevin MacLeod (incompetech.com), '
                          f"CC BY, via Wikimedia Commons"}))
            log(f"[bgm] ready: {clean_name(t)} ({audio_duration(p):.0f}s)")
            finish(dest, clean_name(t))
            return
    die("all bgm candidates failed to download — check network, or use bgm mood 'none' "
        "or drop a local mp3 into assets/bgm/ and set bgm.file")


if __name__ == "__main__":
    main()
