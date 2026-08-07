"""Stage 2 — find, score, download and validate one visual asset per scene.

Keyless providers: openverse (Flickr/museums CC images), wikimedia (encyclopedic
photos), nasa (space imagery). With PEXELS_API_KEY / PIXABAY_API_KEY set, real
stock video clips and photos are used and preferred.

Usage:
  python scripts/assets.py <project_dir>                 # fetch all scenes
  python scripts/assets.py <project_dir> --scene 3 --keywords "steam train"
  python scripts/assets.py <project_dir> --scene 3       # refetch (next candidate)
Writes: media/scene_NN.<ext>, media/manifest.json, media/assets_sheet.jpg
URLs listed in media/exclude.txt are never used again.
"""
import argparse
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path

from common import (ASPECTS, die, download, ensure_dirs, http_get, load_storyboard, log,
                    media_info, project_paths, title_audit)

MIN_SHORT_SIDE = 620        # images
MIN_SHORT_VIDEO = 360       # video clips — motion forgives lower resolution
PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "").strip()
PIXABAY_KEY = os.environ.get("PIXABAY_API_KEY", "").strip()


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


# ------------------------------------------------------------------ providers
# Each returns a list of candidates:
# {url, kind(image|video), w, h, dur, title, creator, license, source, provider}

def p_openverse(q, n=20):
    try:
        r = http_get("https://api.openverse.org/v1/images/",
                     headers={"Accept": "application/json"}, timeout=20)
        r = http_get(f"https://api.openverse.org/v1/images/?q={q}&license_type=commercial"
                     f"&page_size={n}", timeout=20)
        out = []
        for it in r.json().get("results", []):
            url = it.get("url") or ""
            if not re.search(r"\.(jpe?g|png)([?#].*)?$", url, re.I):
                continue
            out.append({"url": url, "kind": "image",
                        "w": it.get("width") or 0, "h": it.get("height") or 0, "dur": 0,
                        "title": it.get("title") or "", "creator": it.get("creator") or "",
                        "license": f"CC {it.get('license','').upper()} {it.get('license_version') or ''}".strip(),
                        "source": it.get("foreign_landing_url") or url,
                        "provider": "openverse"})
        return out
    except Exception as e:
        log(f"  [openverse] error: {e}")
        return []


def p_wikimedia(q, n=20):
    try:
        r = http_get("https://commons.wikimedia.org/w/api.php?action=query"
                     f"&generator=search&gsrsearch=filetype:bitmap {q}&gsrnamespace=6"
                     f"&gsrlimit={n}&prop=imageinfo&iiprop=url|size|mime|extmetadata"
                     "&iiurlwidth=2200&format=json", timeout=25)
        pages = (r.json().get("query") or {}).get("pages") or {}
        out = []
        for pg in sorted(pages.values(), key=lambda p: p.get("index", 99)):
            ii = (pg.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            meta = ii.get("extmetadata") or {}
            url = ii.get("thumburl") or ii.get("url")
            w = ii.get("thumbwidth") or ii.get("width") or 0
            h = ii.get("thumbheight") or ii.get("height") or 0
            out.append({"url": url, "kind": "image", "w": w, "h": h, "dur": 0,
                        "title": strip_tags(pg.get("title", "").replace("File:", "")),
                        "creator": strip_tags((meta.get("Artist") or {}).get("value", ""))[:80],
                        "license": strip_tags((meta.get("LicenseShortName") or {}).get("value", "")),
                        "source": ii.get("descriptionurl") or url,
                        "provider": "wikimedia"})
        return out
    except Exception as e:
        log(f"  [wikimedia] error: {e}")
        return []


def p_nasa(q, n=10):
    try:
        r = http_get(f"https://images-api.nasa.gov/search?q={q}&media_type=image", timeout=25)
        items = (r.json().get("collection") or {}).get("items", [])[:n]
        out = []
        for it in items:
            data = (it.get("data") or [{}])[0]
            nasa_id = data.get("nasa_id")
            if not nasa_id:
                continue
            try:
                a = http_get(f"https://images-api.nasa.gov/asset/{nasa_id}", timeout=20)
                files = [x.get("href", "") for x in (a.json().get("collection") or {}).get("items", [])]
            except Exception:
                files = []
            url = next((f for f in files if "~large.jpg" in f),
                       next((f for f in files if "~orig.jpg" in f), None))
            if not url:
                continue
            out.append({"url": url, "kind": "image", "w": 0, "h": 0, "dur": 0,
                        "title": data.get("title") or "", "creator": data.get("center") or "NASA",
                        "license": "Public Domain (NASA)",
                        "source": f"https://images.nasa.gov/details/{nasa_id}",
                        "provider": "nasa"})
        return out
    except Exception as e:
        log(f"  [nasa] error: {e}")
        return []


def p_wikimedia_video(q, n=10):
    """Real CC/PD video clips from Wikimedia Commons, using the server-side
    transcodes (240p/480p/720p/1080p) so downloads stay small."""
    try:
        r = http_get("https://commons.wikimedia.org/w/api.php?action=query"
                     f"&generator=search&gsrsearch=filetype:video {q}&gsrnamespace=6"
                     f"&gsrlimit={n}&prop=videoinfo&viprop=url|size|mime|derivatives|extmetadata"
                     "&format=json", timeout=30)
        pages = (r.json().get("query") or {}).get("pages") or {}
        out = []
        for pg in sorted(pages.values(), key=lambda p: p.get("index", 99)):
            vi = (pg.get("videoinfo") or [{}])[0]
            derivs = [d for d in (vi.get("derivatives") or [])
                      if d.get("src") and (d.get("height") or 0) >= 360
                      and "vp9" in str(d.get("transcodekey", "")) + str(d.get("src", ""))]
            best = None
            for d in sorted(derivs, key=lambda d: d.get("height") or 0, reverse=True):
                if (d.get("height") or 0) <= 1100:
                    best = d
                    break
            if not best and derivs:
                best = min(derivs, key=lambda d: d.get("height") or 9999)
            if not best:
                continue
            meta = vi.get("extmetadata") or {}
            out.append({"url": best["src"], "kind": "video",
                        "w": best.get("width") or 0, "h": best.get("height") or 0, "dur": 0,
                        "title": strip_tags(pg.get("title", "").replace("File:", "")),
                        "creator": strip_tags((meta.get("Artist") or {}).get("value", ""))[:80],
                        "license": strip_tags((meta.get("LicenseShortName") or {}).get("value", "")),
                        "source": f"https://commons.wikimedia.org/wiki/{pg.get('title','').replace(' ', '_')}",
                        "provider": "wikimedia_video"})
        return out
    except Exception as e:
        log(f"  [wikimedia_video] error: {e}")
        return []


def p_nasa_video(q, n=6):
    """NASA video library (public domain) — picks the medium/mobile mp4."""
    try:
        r = http_get(f"https://images-api.nasa.gov/search?q={q}&media_type=video", timeout=25)
        items = (r.json().get("collection") or {}).get("items", [])[:n]
        out = []
        for it in items:
            data = (it.get("data") or [{}])[0]
            nasa_id = data.get("nasa_id")
            if not nasa_id:
                continue
            try:
                a = http_get(f"https://images-api.nasa.gov/asset/{nasa_id}", timeout=20)
                files = [x.get("href", "") for x in (a.json().get("collection") or {}).get("items", [])]
            except Exception:
                continue
            url = next((f for suffix in ("~medium.mp4", "~mobile.mp4", "~small.mp4", "~preview.mp4")
                        for f in files if f.endswith(suffix)), None)
            if not url:
                continue
            from urllib.parse import quote
            url = quote(url, safe=":/~.%-_")
            out.append({"url": url, "kind": "video", "w": 0, "h": 0, "dur": 0,
                        "title": data.get("title") or "", "creator": data.get("center") or "NASA",
                        "license": "Public Domain (NASA)",
                        "source": f"https://images.nasa.gov/details/{nasa_id}",
                        "provider": "nasa_video"})
        return out
    except Exception as e:
        log(f"  [nasa_video] error: {e}")
        return []


def p_archive_video(q, n=6):
    """Prelinger Archives (public-domain historical film) via archive.org.
    Restricted to that curated collection because general archive.org license
    metadata is user-supplied and often wrong."""
    try:
        r = http_get("https://archive.org/advancedsearch.php?q="
                     f"collection:(prelinger) AND ({q})&fl=identifier,title&rows={n}"
                     "&sort=-downloads&output=json", timeout=30)
        out = []
        for doc in (r.json().get("response") or {}).get("docs", []):
            ident = doc.get("identifier")
            if not ident:
                continue
            try:
                meta = http_get(f"https://archive.org/metadata/{ident}", timeout=25).json()
            except Exception:
                continue
            mp4s = [f for f in meta.get("files", [])
                    if f.get("name", "").endswith(".mp4") and int(f.get("size") or 0) > 1_000_000]
            if not mp4s:
                continue
            best = min(mp4s, key=lambda f: int(f.get("size") or 1 << 40))
            if int(best.get("size") or 0) > 90_000_000:
                continue
            out.append({"url": f"https://archive.org/download/{ident}/{best['name']}",
                        "kind": "video", "w": 0, "h": 0,
                        "dur": float(best.get("length") or 0),
                        "title": str(doc.get("title") or ident)[:80], "creator": "Prelinger Archives",
                        "license": "Public Domain",
                        "source": f"https://archive.org/details/{ident}",
                        "provider": "archive_video"})
        return out
    except Exception as e:
        log(f"  [archive_video] error: {e}")
        return []


def p_pexels_video(q, n=12, portrait=True):
    if not PEXELS_KEY:
        return []
    try:
        ori = "portrait" if portrait else "landscape"
        r = http_get(f"https://api.pexels.com/videos/search?query={q}&per_page={n}&orientation={ori}",
                     headers={"Authorization": PEXELS_KEY}, timeout=25)
        out = []
        for v in r.json().get("videos", []):
            files = sorted(v.get("video_files", []),
                           key=lambda f: (f.get("height") or 0), reverse=True)
            best = next((f for f in files if (f.get("height") or 0) <= 2200 and
                         f.get("file_type") == "video/mp4"), None)
            if not best:
                continue
            out.append({"url": best["link"], "kind": "video",
                        "w": best.get("width") or 0, "h": best.get("height") or 0,
                        "dur": v.get("duration") or 0,
                        "title": (v.get("url") or "").rstrip("/").split("/")[-1].replace("-", " "),
                        "creator": (v.get("user") or {}).get("name", ""),
                        "license": "Pexels License", "source": v.get("url") or best["link"],
                        "provider": "pexels_video"})
        return out
    except Exception as e:
        log(f"  [pexels_video] error: {e}")
        return []


def p_pexels_photo(q, n=15, portrait=True):
    if not PEXELS_KEY:
        return []
    try:
        ori = "portrait" if portrait else "landscape"
        r = http_get(f"https://api.pexels.com/v1/search?query={q}&per_page={n}&orientation={ori}",
                     headers={"Authorization": PEXELS_KEY}, timeout=25)
        out = []
        for p in r.json().get("photos", []):
            out.append({"url": (p.get("src") or {}).get("large2x") or (p.get("src") or {}).get("original"),
                        "kind": "image", "w": p.get("width") or 0, "h": p.get("height") or 0,
                        "dur": 0, "title": p.get("alt") or "", "creator": p.get("photographer") or "",
                        "license": "Pexels License", "source": p.get("url") or "",
                        "provider": "pexels_photo"})
        return out
    except Exception as e:
        log(f"  [pexels_photo] error: {e}")
        return []


def p_pixabay_video(q, n=12):
    if not PIXABAY_KEY:
        return []
    try:
        r = http_get(f"https://pixabay.com/api/videos/?key={PIXABAY_KEY}&q={q}&per_page={n}&safesearch=true",
                     timeout=25)
        out = []
        for v in r.json().get("hits", []):
            f = (v.get("videos") or {}).get("large") or (v.get("videos") or {}).get("medium") or {}
            if not f.get("url"):
                continue
            out.append({"url": f["url"], "kind": "video", "w": f.get("width") or 0,
                        "h": f.get("height") or 0, "dur": v.get("duration") or 0,
                        "title": v.get("tags") or "", "creator": v.get("user") or "",
                        "license": "Pixabay License", "source": v.get("pageURL") or f["url"],
                        "provider": "pixabay_video"})
        return out
    except Exception as e:
        log(f"  [pixabay_video] error: {e}")
        return []


PROVIDERS = {
    "openverse": p_openverse, "wikimedia": p_wikimedia, "nasa": p_nasa,
    "wikimedia_video": p_wikimedia_video, "nasa_video": p_nasa_video,
    "archive_video": p_archive_video,
    "pexels_video": p_pexels_video, "pexels_photo": p_pexels_photo,
    "pixabay_video": p_pixabay_video,
}
PROVIDER_RANK = {"pexels_video": 5, "pixabay_video": 4, "wikimedia_video": 4,
                 "nasa_video": 4, "archive_video": 3, "pexels_photo": 3,
                 "wikimedia": 2, "openverse": 2, "nasa": 2}


def default_providers():
    if PEXELS_KEY or PIXABAY_KEY:
        return ["pexels_video", "pixabay_video", "pexels_photo", "openverse", "wikimedia"]
    return ["wikimedia_video", "openverse", "wikimedia"]


# ------------------------------------------------------------------ scoring

STOPWORDS = {"the", "a", "an", "of", "in", "on", "and", "or", "with", "at", "dark", "close", "up"}


def score(c, W, H, keywords):
    s = 0.0
    short = min(c["w"], c["h"]) if c["w"] and c["h"] else 0
    if short:
        if short < (MIN_SHORT_VIDEO if c["kind"] == "video" else MIN_SHORT_SIDE):
            return -1
        s += min(short, 1600) / 1600 * 1.1
        target = W / H
        ar = c["w"] / c["h"]
        s += max(0.0, 1.0 - abs(ar - target))
    else:
        s += 1.0  # unknown dims (nasa large) — usually fine
    # relevance: keyword tokens appearing in the asset title matter most
    kw_tokens = set(re.findall(r"[a-z]{3,}", " ".join(keywords).lower())) - STOPWORDS
    title_tokens = set(re.findall(r"[a-z]{3,}", (c["title"] or "").lower())) - STOPWORDS
    overlap = len(kw_tokens & title_tokens)
    s += 0.9 * min(overlap, 3)
    if kw_tokens and overlap == 0:
        s -= 1.3  # untitled / unrelated-title assets are risky picks
    s += PROVIDER_RANK.get(c["provider"], 1) * 0.8
    if c["kind"] == "video":
        s += 1.2
        if c["dur"]:
            if c["dur"] < 2:
                return -1
            s += 0.4 if 4 <= c["dur"] <= 180 else (-0.5 if c["dur"] > 600 else 0)
    return s


def gather(scene, W, H):
    q = " ".join(scene["keywords"]) or scene["text"][:40]
    q = q.replace("&", " ").replace("#", " ")
    provs = scene.get("providers") or default_providers()
    cands = []
    for pname in provs:
        fn = PROVIDERS.get(pname)
        if not fn:
            log(f"  unknown provider '{pname}', skipping")
            continue
        got = fn(q)
        if not got and len(q.split()) > 2:
            short_q = " ".join(q.split()[:2])
            got = fn(short_q)
            if got:
                log(f"  [{pname}] 0 for full query, {len(got)} for '{short_q}'")
        log(f"  [{pname}] {len(got)} candidates")
        cands.extend(got)
    seen, out = set(), []
    for c in cands:
        if not c["url"] or c["url"] in seen:
            continue
        seen.add(c["url"])
        out.append(c)
    return sorted(out, key=lambda c: score(c, W, H, scene["keywords"]), reverse=True)


def shot_name(i, j):
    return f"{i:02d}{chr(96 + j)}"  # 01a, 01b, …


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_shot(i, j, query, scene, sb, paths, excluded, used_urls, used_hashes):
    """Fetch one shot (j is 1-based within scene i).

    Dedupe happens on two levels: candidate URL (cheap, pre-download) and
    content sha1 (post-download) — the same picture often lives under several
    URLs (mirrors, thumb vs original, different landing pages)."""
    W, H = ASPECTS[sb["aspect"]]
    sc = dict(scene)
    sc["keywords"] = [query]
    cands = gather(sc, W, H)
    tried = 0
    for c in cands:
        if c["url"] in excluded or c["url"] in used_urls or score(c, W, H, sc["keywords"]) < 0:
            continue
        tried += 1
        if tried > 6:
            break
        ext = ".mp4" if c["kind"] == "video" else \
            (".png" if ".png" in c["url"].lower() else ".jpg")
        dest = paths["media"] / f"scene_{shot_name(i, j)}{ext}"
        log(f"  [{shot_name(i, j)}] trying [{c['provider']}] {c['title'][:46]!r} {c['w']}x{c['h']}")
        if not download(c["url"], dest):
            log("    download failed")
            continue
        try:
            kind, w, h, dur = media_info(dest)
        except Exception as e:
            log(f"    undecodable: {e}")
            dest.unlink(missing_ok=True)
            continue
        if min(w, h) < (MIN_SHORT_VIDEO if kind == "video" else MIN_SHORT_SIDE):
            log(f"    too small after download: {w}x{h}")
            dest.unlink(missing_ok=True)
            excluded.add(c["url"])
            continue
        if kind == "video" and dur < 1.5:
            log(f"    video too short: {dur:.1f}s")
            dest.unlink(missing_ok=True)
            excluded.add(c["url"])
            continue
        digest = file_sha1(dest)
        if digest in used_hashes:
            log("    duplicate content (same bytes as another shot), skipping")
            dest.unlink(missing_ok=True)
            excluded.add(c["url"])
            continue
        used_urls.add(c["url"])
        used_hashes.add(digest)
        return {"file": dest.name, "kind": kind, "w": w, "h": h, "dur": round(dur, 2),
                "sha1": digest, "query": query,
                **{k: c[k] for k in ("provider", "title", "creator", "license", "source", "url")}}
    die(f"scene {i} shot {j}: no usable asset for query '{query}'. Try: "
        f"python scripts/assets.py <dir> --scene {i} --shot {j} --keywords \"other english nouns\"")


def local_shot(i, j, src_path, paths, used_hashes=None):
    src = Path(src_path).expanduser()
    if not src.exists():
        die(f"scene {i}: media override not found: {src}")
    dest = paths["media"] / f"scene_{shot_name(i, j)}{src.suffix.lower()}"
    dest.write_bytes(src.read_bytes())
    kind, w, h, dur = media_info(dest)
    digest = file_sha1(dest)
    if used_hashes is not None:
        used_hashes.add(digest)
    return {"file": dest.name, "kind": kind, "w": w, "h": h, "dur": round(dur, 2),
            "sha1": digest,
            "provider": "local", "title": src.name, "creator": "", "license": "user-provided",
            "source": str(src), "url": ""}


def shots_needed(scene, est_dur):
    """One visual change every ~3.2s, driven by keyword count and VO length."""
    by_dur = max(1, round(est_dur / 3.2)) if est_dur else 1
    return min(4, max(len(scene["keywords"]) or 1, by_dur))


def build_sheet(manifest, paths, sb):
    """Grid of labeled shot thumbnails so the agent can eyeball relevance."""
    from PIL import Image, ImageDraw, ImageFont

    from common import main_font, run
    W, H = ASPECTS[sb["aspect"]]
    tw = 300
    th = int(tw * H / W)
    flat = [(m["i"], j + 1, s) for m in manifest for j, s in enumerate(m["shots"])]
    cols = min(5, max(1, len(flat)))
    rows = (len(flat) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 34)), (18, 18, 18))
    font = ImageFont.truetype(str(main_font(sb["lang"])), 24)
    for idx, (i, j, s) in enumerate(flat):
        src = paths["media"] / s["file"]
        frame = paths["work"] / f"thumb_{shot_name(i, j)}.jpg"
        seek = ["-ss", str(max(0, s["dur"] / 3))] if s["kind"] == "video" else []
        run(["ffmpeg", "-y", *seek, "-i", src, "-frames:v", "1",
             "-vf", f"scale={tw}:-2", frame])
        im = Image.open(frame).convert("RGB")
        im.thumbnail((tw, th))
        x, y = (idx % cols) * tw, (idx // cols) * (th + 34)
        sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
        d = ImageDraw.Draw(sheet)
        d.text((x + 8, y + th + 4),
               f"{shot_name(i, j)} {s['kind']} {s['provider']}", font=font, fill=(255, 225, 77))
    out = paths["media"] / "assets_sheet.jpg"
    sheet.save(out, quality=88)
    log(f"[assets] preview sheet -> {out}")


def est_durations(paths, sb):
    """Per-scene VO length estimates from timing.json (pipeline runs tts first)."""
    est = {}
    if paths["timing"].exists():
        for s in json.loads(paths["timing"].read_text())["scenes"]:
            est[s["i"]] = s["vo_end"] + 0.36
    return est


def load_manifest(paths):
    if not paths["manifest"].exists():
        return {}
    raw = json.loads(paths["manifest"].read_text())
    by_i = {}
    for m in raw:
        shots = m.get("shots") or [{k: v for k, v in m.items() if k != "i"}]  # legacy
        by_i[m["i"]] = {"i": m["i"], "shots": shots}
    return by_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--scene", type=int, help="refetch within a single scene (1-based)")
    ap.add_argument("--shot", type=int, default=1, help="which shot of --scene to refetch")
    ap.add_argument("--keywords", help="override query for --scene/--shot")
    args = ap.parse_args()

    sb = load_storyboard(args.project_dir)
    paths = project_paths(args.project_dir)
    ensure_dirs(paths)
    excluded = set()
    if paths["exclude"].exists():
        excluded = {l.strip() for l in paths["exclude"].read_text().splitlines() if l.strip()}

    by_i = load_manifest(paths)
    used_urls = {s.get("url", "") for m in by_i.values() for s in m["shots"] if s.get("url")}
    est = est_durations(paths, sb)

    # content-level dedupe: hashes of everything already in the video, plus
    # hashes blacklisted by earlier refetches (persisted across runs)
    blacklisted_hashes = set()
    if paths["exclude_hashes"].exists():
        blacklisted_hashes = {l.strip() for l in paths["exclude_hashes"].read_text().splitlines()
                              if l.strip()}
    used_hashes = set(blacklisted_hashes)
    for m in by_i.values():
        for s in m["shots"]:
            if s.get("sha1"):
                used_hashes.add(s["sha1"])
            elif (paths["media"] / s["file"]).exists():
                s["sha1"] = file_sha1(paths["media"] / s["file"])
                used_hashes.add(s["sha1"])

    def save_state():
        manifest = [by_i[i] for i in sorted(by_i)]
        paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        paths["exclude"].write_text("\n".join(sorted(excluded)))
        paths["exclude_hashes"].write_text("\n".join(sorted(blacklisted_hashes)))
        return manifest

    try:
        if args.scene:
            i, j = args.scene, args.shot
            scene = sb["scenes"][i - 1]
            entry = by_i.setdefault(i, {"i": i, "shots": []})
            if j <= len(entry["shots"]):        # refetch: blacklist current asset
                old = entry["shots"][j - 1]
                if old.get("url"):
                    excluded.add(old["url"])
                    used_urls.discard(old["url"])
                if old.get("sha1"):
                    blacklisted_hashes.add(old["sha1"])
                    used_hashes.add(old["sha1"])
            if j > len(entry["shots"]) + 1:
                die(f"scene {i} has {len(entry['shots'])} shot(s); --shot must be "
                    f"<= {len(entry['shots']) + 1} (or run a full assets pass to add shots)")
            query = args.keywords or scene["keywords"][(j - 1) % max(1, len(scene["keywords"]))]
            log(f"[assets] scene {i} shot {j}: query='{query}'")
            shot = fetch_shot(i, j, query, scene, sb, paths, excluded, used_urls, used_hashes)
            if j == len(entry["shots"]) + 1:
                entry["shots"].append(shot)
            else:
                entry["shots"][j - 1] = shot
        else:
            for i in range(1, len(sb["scenes"]) + 1):
                scene = sb["scenes"][i - 1]
                entry = by_i.setdefault(i, {"i": i, "shots": []})
                if scene.get("media") and not entry["shots"]:
                    entry["shots"] = [local_shot(i, 1, scene["media"], paths, used_hashes)]
                    continue
                need = shots_needed(scene, est.get(i))
                kws = scene["keywords"] or [scene["text"][:40]]
                for j in range(len(entry["shots"]) + 1, need + 1):
                    query = kws[(j - 1) % len(kws)]
                    log(f"[assets] scene {i} shot {j}/{need}: query='{query}'")
                    entry["shots"].append(
                        fetch_shot(i, j, query, scene, sb, paths, excluded, used_urls, used_hashes))
                if len(entry["shots"]) >= need:
                    log(f"[assets] scene {i}: {len(entry['shots'])} shot(s) ready")
    finally:
        # keep fetched shots + blacklists even when a later shot fails, so a
        # rerun (or --scene refetch) resumes instead of redownloading
        manifest = save_state()

    # Flag off-topic picks NOW, before a compose cycle is spent on them —
    # for a text-only agent these lines ARE the asset review. Before the
    # sheet build so a sheet failure can never eat the flags.
    for i, j, t in title_audit(sb, manifest):
        log(f'[assets] [!] scene {i} shot {j}: "{t}" looks OFF-TOPIC for its keywords — refetch it '
            f'with --scene {i} --shot {j} --keywords "<concrete english nouns>" before composing')
    build_sheet(manifest, paths, sb)
    n = sum(len(m["shots"]) for m in manifest)
    log(f"[assets] done: {n} shots across {len(manifest)} scenes. "
        "REVIEW media/assets_sheet.jpg before composing!")


if __name__ == "__main__":
    main()
