#!/usr/bin/env bash
# Build the chaty-audio music-generation sidecar (audio.cpp) and stage it
# where Tauri's externalBin expects it: src-tauri/binaries/chaty-audio-<triple>.
#
# Usage: scripts/build-audio-sidecar.sh [cpu|vulkan|metal|cuda]
#   default backend: metal on macOS; vulkan on Linux when glslc is installed,
#   cpu otherwise.
# Env: AUDIOCPP_SOURCE_DIR=/path/to/audio.cpp builds from a local checkout
#      instead of fetching the pinned revision.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src-tauri/audio-sidecar"
BUILD="$SRC/build"

BACKEND="${1:-}"
if [[ -z "$BACKEND" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    BACKEND=metal
  elif command -v glslc >/dev/null 2>&1; then
    BACKEND=vulkan
  else
    BACKEND=cpu
  fi
fi

TRIPLE="$(rustc -vV 2>/dev/null | sed -n 's/^host: //p')"
if [[ -z "$TRIPLE" ]]; then
  echo "error: rustc not found — the target triple names the staged binary" >&2
  exit 1
fi

GEN=()
command -v ninja >/dev/null 2>&1 && GEN=(-G Ninja)
EXTRA=()
[[ -n "${AUDIOCPP_SOURCE_DIR:-}" ]] && EXTRA+=("-DFETCHCONTENT_SOURCE_DIR_AUDIOCPP=$AUDIOCPP_SOURCE_DIR")
if [[ "$(uname -s)" == "Darwin" ]]; then
  # audio.cpp's Metal code wants macOS 13.3 (its own release builds use that
  # floor); on anything older the app says the music engine cannot start.
  EXTRA+=(-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 -DCMAKE_OSX_ARCHITECTURES=arm64)
fi

echo "chaty-audio: backend=$BACKEND triple=$TRIPLE"
cmake -S "$SRC" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCHATY_AUDIO_BACKEND="$BACKEND" "${EXTRA[@]}"
cmake --build "$BUILD" --config Release --target chaty-audio -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

BIN="$BUILD/bin/chaty-audio"
[[ -x "$BIN" ]] || { echo "error: built binary not found at $BIN" >&2; exit 1; }
STAGE="$ROOT/src-tauri/binaries"
mkdir -p "$STAGE"
cp -f "$BIN" "$STAGE/chaty-audio-$TRIPLE"
echo "staged: $STAGE/chaty-audio-$TRIPLE"
"$STAGE/chaty-audio-$TRIPLE" --version || true
