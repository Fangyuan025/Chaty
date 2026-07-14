#!/usr/bin/env bash
# Copy the MLX sidecar's SwiftPM resource bundles (mlx-swift's Cmlx bundle
# carries the compiled Metal library) into a built Chaty.app and re-sign.
# Tauri's `resources` glob can't map directories, so this runs AFTER
# `tauri build --config src-tauri/tauri.mlx.conf.json`.
# Usage: scripts/bundle-mlx-resources.sh <path/to/Chaty.app>
set -euo pipefail
APP="${1:?usage: bundle-mlx-resources.sh <Chaty.app>}"
SRC="$(dirname "$0")/../src-tauri/binaries"

[ -d "$APP/Contents/Resources" ] || { echo "error: $APP is not an app bundle" >&2; exit 1; }
BUNDLES=$(find "$SRC" -maxdepth 1 -name '*.bundle')
[ -n "$BUNDLES" ] || { echo "error: no *.bundle staged in $SRC — run build-mlx-sidecar.sh first" >&2; exit 1; }

for B in $BUNDLES; do
  rm -rf "$APP/Contents/Resources/$(basename "$B")"
  cp -R "$B" "$APP/Contents/Resources/"
  echo "bundled: $(basename "$B")"
done

# Adding files invalidates the signature; re-sign ad-hoc (matches the
# project's "-" signingIdentity).
codesign --force --deep --sign - "$APP"
echo "re-signed: $APP"
