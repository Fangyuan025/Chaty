#!/usr/bin/env bash
# Build the chaty-sd image-generation sidecar (stable-diffusion.cpp) and stage
# it where Tauri's externalBin expects it: src-tauri/binaries/chaty-sd-<triple>.
#
# Usage: scripts/build-sd-sidecar.sh [cpu|vulkan|metal|cuda]
#   default backend: metal on macOS; vulkan on Linux when glslc is installed,
#   cpu otherwise.
# Env: SDCPP_SOURCE_DIR=/path/to/stable-diffusion.cpp builds from a local
#      checkout instead of fetching the pinned revision.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src-tauri/sd-sidecar"
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
[[ -n "${SDCPP_SOURCE_DIR:-}" ]] && EXTRA+=("-DFETCHCONTENT_SOURCE_DIR_SDCPP=$SDCPP_SOURCE_DIR")
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Match the app's floor (tauri.macos.conf.json minimumSystemVersion).
  EXTRA+=(-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_OSX_ARCHITECTURES=arm64)
fi

echo "chaty-sd: backend=$BACKEND triple=$TRIPLE"
cmake -S "$SRC" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCHATY_SD_BACKEND="$BACKEND" "${EXTRA[@]}"
cmake --build "$BUILD" --config Release --target chaty-sd -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

BIN="$BUILD/bin/chaty-sd"
[[ -x "$BIN" ]] || { echo "error: built binary not found at $BIN" >&2; exit 1; }
STAGE="$ROOT/src-tauri/binaries"
mkdir -p "$STAGE"
cp -f "$BIN" "$STAGE/chaty-sd-$TRIPLE"
echo "staged: $STAGE/chaty-sd-$TRIPLE"
"$STAGE/chaty-sd-$TRIPLE" --version || true
