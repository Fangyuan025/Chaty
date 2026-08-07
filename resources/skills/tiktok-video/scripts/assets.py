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
import html
import json
import os
import re
import sys
from pathlib import Path

from common import (ASPECTS, die, download, ensure_dirs, http_get, load_storyboard, log,
                    media_info, project_paths)

MIN_SHORT_SIDE = 620
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
    "pexels_video": p_pexels_video, "pexels_photo": p_pexels_photo,
    "pixabay_video": p_pixabay_video,
}
PROVIDER_RANK = {"pexels_video": 5, "pixabay_video": 4, "pexels_photo": 3,
                 "wikimedia": 2, "openverse": 2, "nasa": 2}


def default_providers():
    if PEXELS_KEY or PIXABAY_KEY:
        return ["pexels_video", "pixabay_video", "pexels_photo", "openverse", "wikimedia"]
    return ["openverse", "wikimedia"]


# ------------------------------------------------------------------ scoring

STOPWORDS = {"the", "a", "an", "of", "in", "on", "and", "or", "with", "at", "dark", "close", "up"}


def score(c, W, H, keywords):
    s = 0.0
    short = min(c["w"], c["h"]) if c["w"] and c["h"] else 0
    if short:
        if short < MIN_SHORT_SIDE:
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


def fetch_scene(i, scene, sb, paths, excluded, used_urls):
    W, H = ASPECTS[sb["aspect"]]
    # explicit local file override
    if scene.get("media"):
        src = Path(scene["media"]).expanduser()
        if not src.exists():
            die(f"scene {i}: media override not found: {src}")
        dest = paths["media"] / f"scene_{i:02d}{src.suffix.lower()}"
        dest.write_bytes(src.read_bytes())
        kind, w, h, dur = media_info(dest)
        return {"i": i, "file": dest.name, "kind": kind, "w": w, "h": h, "dur": dur,
                "provider": "local", "title": src.name, "creator": "", "license": "user-provided",
                "source": str(src), "url": ""}

    cands = gather(scene, W, H)
    tried = 0
    for c in cands:
        if c["url"] in excluded or c["url"] in used_urls or score(c, W, H, scene["keywords"]) < 0:
            continue
        tried += 1
        if tried > 6:
            break
        ext = ".mp4" if c["kind"] == "video" else \
            (".png" if ".png" in c["url"].lower() else ".jpg")
        dest = paths["media"] / f"scene_{i:02d}{ext}"
        log(f"  trying [{c['provider']}] {c['title'][:46]!r} {c['w']}x{c['h']}")
        if not download(c["url"], dest):
            log("    download failed")
            continue
        try:
            kind, w, h, dur = media_info(dest)
        except Exception as e:
            log(f"    undecodable: {e}")
            dest.unlink(missing_ok=True)
            continue
        if min(w, h) < MIN_SHORT_SIDE:
            log(f"    too small after download: {w}x{h}")
            dest.unlink(missing_ok=True)
            excluded.add(c["url"])
            continue
        used_urls.add(c["url"])
        return {"i": i, "file": dest.name, "kind": kind, "w": w, "h": h, "dur": round(dur, 2),
                **{k: c[k] for k in ("provider", "title", "creator", "license", "source", "url")}}
    die(f"scene {i}: no usable asset found for keywords={scene['keywords']}. "
        f"Try different English keywords: python scripts/assets.py <dir> --scene {i} --keywords \"...\"")


def build_sheet(manifest, paths, sb):
    """Grid of numbered thumbnails so the agent can eyeball relevance."""
    from PIL import Image, ImageDraw, ImageFont

    from common import main_font
    W, H = ASPECTS[sb["aspect"]]
    tw = 300
    th = int(tw * H / W)
    cols = min(4, max(1, len(manifest)))
    rows = (len(manifest) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 34)), (18, 18, 18))
    font = ImageFont.truetype(str(main_font(sb["lang"])), 24)
    for idx, m in enumerate(manifest):
        src = paths["media"] / m["file"]
        frame = paths["work"] / f"thumb_{m['i']:02d}.jpg"
        from common import run
        if m["kind"] == "video":
            run(["ffmpeg", "-y", "-ss", str(max(0, m["dur"] / 3)), "-i", src,
                 "-frames:v", "1", "-vf", f"scale={tw}:-2", frame])
        else:
            run(["ffmpeg", "-y", "-i", src, "-frames:v", "1", "-vf", f"scale={tw}:-2", frame])
        im = Image.open(frame).convert("RGB")
        im.thumbnail((tw, th))
        x, y = (idx % cols) * tw, (idx // cols) * (th + 34)
        sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
        d = ImageDraw.Draw(sheet)
        d.text((x + 8, y + th + 4),
               f"{m['i']:02d} {m['kind']} {m['provider']}", font=font, fill=(255, 225, 77))
    out = paths["media"] / "assets_sheet.jpg"
    sheet.save(out, quality=88)
    log(f"[assets] preview sheet -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--scene", type=int, help="refetch a single scene (1-based)")
    ap.add_argument("--keywords", help="override keywords for --scene")
    args = ap.parse_args()

    sb = load_storyboard(args.project_dir)
    paths = project_paths(args.project_dir)
    ensure_dirs(paths)
    excluded = set()
    if paths["exclude"].exists():
        excluded = {l.strip() for l in paths["exclude"].read_text().splitlines() if l.strip()}

    manifest = []
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text())
    by_i = {m["i"]: m for m in manifest}
    used_urls = {m.get("url", "") for m in manifest if m.get("url")}

    targets = [args.scene] if args.scene else list(range(1, len(sb["scenes"]) + 1))
    for i in targets:
        scene = dict(sb["scenes"][i - 1])
        if args.scene and args.keywords:
            scene["keywords"] = [args.keywords]
        if args.scene and i in by_i:  # refetch: blacklist the current asset
            old = by_i[i]
            if old.get("url"):
                excluded.add(old["url"])
                used_urls.discard(old["url"])
        elif not args.scene and i in by_i and (paths["media"] / by_i[i]["file"]).exists():
            log(f"[assets] scene {i}: already have {by_i[i]['file']}, skipping")
            continue
        log(f"[assets] scene {i}: keywords={scene['keywords']}")
        by_i[i] = fetch_scene(i, scene, sb, paths, excluded, used_urls)

    manifest = [by_i[i] for i in sorted(by_i)]
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    paths["exclude"].write_text("\n".join(sorted(excluded)))
    build_sheet(manifest, paths, sb)
    log(f"[assets] done: {len(manifest)} assets. REVIEW media/assets_sheet.jpg before composing!")


if __name__ == "__main__":
    main()
