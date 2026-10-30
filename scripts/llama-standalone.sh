#!/usr/bin/env bash
# Make a Linux chaty-llama start where the app's libraries are not.
#
# chaty-llama links the app's crate, and with it the voice library's
# (sherpa-onnx, ONNX Runtime) DT_NEEDED entries, though none of their
# symbols is left in it once unused code is dropped. A runtime pack runs
# from the data folder, so those entries would keep it from starting. They
# are removed here; the macOS build weak-links the same library (build.rs),
# and Windows delay-loads it.
#
# Usage: scripts/llama-standalone.sh path/to/chaty-llama [rpath]
#   rpath: also set this RPATH (the AppImage's '$ORIGIN/../lib').
set -euo pipefail
BIN="$1"
[ "$(uname -s)" = Linux ] || exit 0
command -v patchelf >/dev/null || { echo "error: patchelf is needed" >&2; exit 1; }
for lib in $(patchelf --print-needed "$BIN" | grep -Ei 'sherpa|onnxruntime' || true); do
  patchelf --remove-needed "$lib" "$BIN"
  echo "chaty-llama: no longer needs $lib"
done
if [ -n "${2:-}" ]; then
  patchelf --set-rpath "$2" "$BIN"
fi
echo "chaty-llama needs: $(patchelf --print-needed "$BIN" | tr '\n' ' ')"
