#!/usr/bin/env bash
# One-time setup for the tiktok-video skill: venv + deps + CJK font.
# Idempotent — safe to rerun. The font is kept in the machine-wide cache
# (~/.cache/tiktok-video-skill) so a fresh workspace doesn't re-download it.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install it first (macOS: brew install ffmpeg | debian: apt install ffmpeg)"
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating venv…"
  python3 -m venv .venv
fi
echo "[setup] installing python deps…"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q edge-tts requests pillow

mkdir -p assets/fonts assets/bgm
FONT=assets/fonts/NotoSansCJKsc-Black.otf
CACHE="${TIKTOK_SKILL_CACHE:-$HOME/.cache/tiktok-video-skill}"
CACHED_FONT="$CACHE/fonts/NotoSansCJKsc-Black.otf"
if [ ! -s "$FONT" ]; then
  if [ -s "$CACHED_FONT" ]; then
    echo "[setup] font from cache…"
    cp "$CACHED_FONT" "$FONT"
  else
    echo "[setup] downloading Noto Sans CJK SC Black (~17MB, one time)…"
    curl -fsSL -o "$FONT" \
      "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Black.otf" \
      || { echo "font download failed — retry or place any bold CJK .otf at $FONT"; exit 1; }
    mkdir -p "$CACHE/fonts" && cp "$FONT" "$CACHED_FONT" || true
  fi
fi

.venv/bin/python - <<'EOF'
from PIL import ImageFont
f = ImageFont.truetype("assets/fonts/NotoSansCJKsc-Black.otf", 40)
print("[setup] font OK:", f.getname())
EOF

echo "[setup] done. Optional: export PEXELS_API_KEY / PIXABAY_API_KEY for stock video clips."
