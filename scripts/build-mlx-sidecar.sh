#!/usr/bin/env bash
# Build the chaty-mlx sidecar (macOS arm64) and stage it where Tauri's
# externalBin expects it. Requires full Xcode (SwiftPM alone cannot compile
# the MLX Metal shaders) — plus the Metal Toolchain component on macOS 26+:
#   xcodebuild -downloadComponent MetalToolchain
#
# Usage: scripts/build-mlx-sidecar.sh [--debug]
# Env: CHATY_MLX_LM_REV=<ml-explore/mlx-swift-lm commit> builds against that
#      revision instead of the pin — what a runtime pack is built from
#      (.github/workflows/runtimes.yml). Package.swift and Package.resolved
#      are put back afterwards.
set -euo pipefail
cd "$(dirname "$0")/../src-tauri/mlx-sidecar"

if [[ -n "${CHATY_MLX_LM_REV:-}" ]]; then
  mkdir -p .build
  cp Package.swift .build/Package.swift.pin
  cp Package.resolved .build/Package.resolved.pin
  trap 'cp .build/Package.swift.pin Package.swift; cp .build/Package.resolved.pin Package.resolved' EXIT
  perl -0pi -e 's|(mlx-swift-lm",\s*revision: ")[0-9a-f]+|${1}'"$CHATY_MLX_LM_REV"'|' Package.swift
  grep -q "revision: \"$CHATY_MLX_LM_REV\"" Package.swift || { echo "error: no mlx-swift-lm revision to move" >&2; exit 1; }
  rm -f Package.resolved
  echo "chaty-mlx: mlx-swift-lm $CHATY_MLX_LM_REV"
fi

CONFIG=Release
[[ "${1:-}" == "--debug" ]] && CONFIG=Debug

if ! xcodebuild -version >/dev/null 2>&1; then
  echo "error: xcodebuild unavailable — install Xcode (App Store), then:" >&2
  echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
  echo "(or export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer)" >&2
  exit 1
fi

DERIVED=.build/xcode
# -skipPackagePluginValidation: mlx-swift ships SwiftPM build plugins (e.g.
# CudaBuild) that unattended xcodebuild refuses to run without a per-user
# trust prompt; CI has no way to click it.
xcodebuild build \
  -scheme chaty-mlx \
  -configuration "$CONFIG" \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$DERIVED" \
  -skipPackagePluginValidation \
  -skipMacroValidation \
  -quiet

PRODUCTS="$DERIVED/Build/Products/$CONFIG"
BIN="$PRODUCTS/chaty-mlx"
[[ -x "$BIN" ]] || { echo "error: built binary not found at $BIN" >&2; exit 1; }

# Stage for Tauri: externalBin wants `binaries/<name>-<target-triple>`, and any
# SwiftPM resource bundles (mlx-swift's Cmlx bundle carries the compiled
# mlx.metallib) must travel next to the binary at runtime.
STAGE=../binaries
mkdir -p "$STAGE"
cp -f "$BIN" "$STAGE/chaty-mlx-aarch64-apple-darwin"
rm -rf "$STAGE"/*.bundle
find "$PRODUCTS" -maxdepth 1 -name '*.bundle' -exec cp -R {} "$STAGE/" \;

echo "staged:"
ls -la "$STAGE"
