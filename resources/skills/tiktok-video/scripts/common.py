"""Shared helpers for the tiktok-video skill pipeline."""
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = SKILL_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
CACHE_DIR = Path(os.environ.get("TIKTOK_SKILL_CACHE", Path.home() / ".cache" / "tiktok-video-skill"))

USER_AGENT = "tiktok-video-skill/1.0 (open-source agent skill; https://github.com/tiktok-video-skill)"

ASPECTS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}

DEFAULT_VOICES = {
    "zh": "zh-CN-YunjianNeural",
    "en": "en-US-ChristopherNeural",
}

# Matches emoji / pictographs (kept in captions, stripped from TTS input).
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+"
)

ZH_PUNCT = ",。!?、;:!?,.;:…~—"


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list, quiet: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, surfacing stderr tail on failure."""
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if check and p.returncode != 0:
        tail = "\n".join((p.stderr or "").strip().splitlines()[-14:])
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(str(c) for c in cmd[:6])}...\n{tail}")
    if not quiet and p.stderr:
        print(p.stderr, file=sys.stderr)
    return p


def ffprobe_json(path) -> dict:
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(p.stdout)


def media_info(path):
    """Return (kind, width, height, duration) for a media file; kind in {image, video, audio}."""
    info = ffprobe_json(path)
    dur = float(info.get("format", {}).get("duration") or 0)
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if vstreams:
        v = vstreams[0]
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        nb = int(v.get("nb_frames") or 0)
        is_image = (v.get("codec_name") in ("mjpeg", "png", "webp", "bmp") or nb == 1) and dur < 1.5
        return ("image" if is_image else "video"), w, h, dur
    if astreams:
        return "audio", 0, 0, dur
    raise RuntimeError(f"no decodable streams in {path}")


def audio_duration(path) -> float:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(p.stdout.strip())


def strip_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text)


def clean_for_tts(text: str) -> str:
    t = strip_emoji(text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = re.sub(r"[^a-zA-Z0-9一-鿿]+", "-", t).strip("-").lower()
    return t[:48] or "video"


def http_get(url: str, timeout: int = 30, headers: dict | None = None, stream: bool = False):
    import requests

    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    return requests.get(url, timeout=timeout, headers=h, stream=stream)


def download(url: str, dest: Path, max_bytes: int = 120_000_000, headers: dict | None = None,
             attempts: int = 3) -> bool:
    """Stream-download url to dest with retries. Returns False on failure (never raises)."""
    import time

    for i in range(attempts):
        if _download_once(url, dest, max_bytes, headers):
            return True
        time.sleep(1.5 * (i + 1))
    return False


def _download_once(url: str, dest: Path, max_bytes: int, headers: dict | None) -> bool:
    try:
        r = http_get(url, timeout=60, headers=headers, stream=True)
        if r.status_code != 200:
            return False
        expected = int(r.headers.get("Content-Length") or 0)
        total = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    tmp.unlink(missing_ok=True)
                    return False
                f.write(chunk)
        # some CDNs close early with 200 — reject truncated bodies
        if total < 1024 or (expected and total < expected):
            tmp.unlink(missing_ok=True)
            return False
        tmp.rename(dest)
        return True
    except Exception:
        return False


TITLE_GENERIC = {"night", "day", "dark", "light", "lights", "running", "run", "fast", "close",
                 "closeup", "view", "silhouette", "background", "old", "big", "small", "little",
                 "photo", "image", "detail"}


def title_audit(sb: dict, manifest: list) -> list:
    """(scene_i, shot_j, title) triples whose asset title shares no meaningful
    word with the scene's keywords — how a text-only agent spots an off-topic
    stock pick (an oil painting, an archival boat photo) without seeing a
    single pixel. Word-prefix matching covers plurals (cat/cats, eye/eyes);
    the shot is compared against the union of the scene's queries so a shot
    matching a sibling query never false-flags."""
    flags = []
    scenes = sb.get("scenes", [])
    for m in manifest:
        if not (1 <= m.get("i", 0) <= len(scenes)):
            continue
        kw = " ".join(scenes[m["i"] - 1].get("keywords", []))
        scene_strong = {t for t in re.findall(r"[a-z]+", kw.lower()) if len(t) >= 3} - TITLE_GENERIC
        for j, s in enumerate(m.get("shots") or [m], 1):
            if s.get("provider") == "local":
                continue
            # A refetched shot records the query that actually chose it —
            # judge against that, or the agent's hand-picked override would
            # flag forever against the storyboard's original words.
            own = {t for t in re.findall(r"[a-z]+", str(s.get("query", "")).lower()) if len(t) >= 3} - TITLE_GENERIC
            strong = own or scene_strong
            if not strong:
                continue
            title = set(re.findall(r"[a-z]+", str(s.get("title", "")).lower()))
            if not any(t.startswith(k) or k.startswith(t) for k in strong for t in title):
                flags.append((m["i"], j, str(s.get("title", ""))[:70]))
    return flags


def project_paths(project_dir) -> dict:
    p = Path(project_dir).resolve()
    return {
        "root": p,
        "storyboard": p / "storyboard.json",
        "audio": p / "audio",
        "timing": p / "audio" / "timing.json",
        "media": p / "media",
        "manifest": p / "media" / "manifest.json",
        "exclude": p / "media" / "exclude.txt",
        "exclude_hashes": p / "media" / "exclude_hashes.txt",
        "work": p / "work",
        "review": p / "review",
        "bgm": p / "bgm.mp3",
        "final": p / "final.mp4",
    }


def load_storyboard(project_dir) -> dict:
    paths = project_paths(project_dir)
    if not paths["storyboard"].exists():
        die(f"storyboard.json not found in {paths['root']}. Write it first (see SKILL.md).")
    try:
        sb = json.loads(paths["storyboard"].read_text())
    except json.JSONDecodeError as e:
        die(f"storyboard.json is not valid JSON: {e}")

    # ---- validate + defaults ----
    if not isinstance(sb.get("scenes"), list) or not sb["scenes"]:
        die("storyboard needs a non-empty 'scenes' list")
    sb.setdefault("title", "untitled")
    lang = sb.setdefault("lang", "zh")
    if lang not in ("zh", "en"):
        die("lang must be 'zh' or 'en'")
    aspect = sb.setdefault("aspect", "9:16")
    if aspect not in ASPECTS:
        die(f"aspect must be one of {list(ASPECTS)}")
    sb.setdefault("fps", 30)
    sb.setdefault("voice", DEFAULT_VOICES[lang])
    sb.setdefault("rate", "+8%" if lang == "zh" else "+5%")
    sb.setdefault("caption_style", "karaoke")
    if sb["caption_style"] not in ("karaoke", "pop", "none"):
        die("caption_style must be karaoke | pop | none")
    bgm = sb.setdefault("bgm", {"mood": "upbeat"})
    if isinstance(bgm, str):
        sb["bgm"] = {"mood": bgm}
    sb["bgm"].setdefault("gain_db", -16)
    cap = sb.setdefault("caption", {})
    cap.setdefault("position", 0.70)   # vertical center of captions, fraction of H
    cap.setdefault("highlight", "#FFE14D")
    cap.setdefault("uppercase", lang == "en")
    hook = sb.get("hook")
    if hook and not hook.get("text"):
        sb["hook"] = None
    sb.setdefault("sfx", True)          # whoosh on scene transitions
    st = sb.get("sticky_title")
    if st and not st.get("text"):
        sb["sticky_title"] = None
    sb.setdefault("sticky_title", None)

    for i, sc in enumerate(sb["scenes"], 1):
        if not sc.get("text", "").strip():
            die(f"scene {i} has empty text")
        kw = sc.setdefault("keywords", [])
        if isinstance(kw, str):
            sc["keywords"] = [kw]
        sc.setdefault("effect", "auto")
        if sc["effect"] not in ("auto", "kb_in", "kb_out", "pan_left", "pan_right", "static"):
            die(f"scene {i}: bad effect '{sc['effect']}'")
        sc.setdefault("emphasis", [])
        sc.setdefault("media", None)
        sc.setdefault("providers", None)
        sc.setdefault("badge", None)
    return sb


def ensure_dirs(paths: dict) -> None:
    for k in ("audio", "media", "work", "review"):
        paths[k].mkdir(parents=True, exist_ok=True)


def find_font(names: list[str]):
    """Find first existing font in assets/fonts by candidate filenames."""
    for n in names:
        p = FONTS_DIR / n
        if p.exists():
            return p
    return None


def main_font(lang: str):
    p = find_font(["NotoSansCJKsc-Black.otf", "NotoSansCJKsc-Bold.otf"])
    if not p:
        die("Fonts missing. Run: bash scripts/setup.sh")
    return p


EMOJI_FONT_CANDIDATES = [
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
]


def emoji_font_path():
    for p in EMOJI_FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None
