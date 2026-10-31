#!/usr/bin/env bash
# Check the AppImage before it is published: it carries no libwayland-client
# or xdg-open of its own, every library its binary needs resolves, its chat
# engine (chaty-llama) answers, a program it starts sees none of the image's
# paths, and it starts — still running after 20 s under a virtual display,
# with no missing library said.
#
# Usage: scripts/appimage-check.sh path/to/Chaty_x.y.z_amd64.AppImage
set -euo pipefail

IMG=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
cd "$WORK"
"$IMG" --appimage-extract >/dev/null
ROOT="$WORK/squashfs-root"

if find "$ROOT" -name 'libwayland-client.so*' | grep -q .; then
  echo "FAIL: the AppImage still carries libwayland-client"
  exit 1
fi

for tool in xdg-open xdg-mime; do
  if [ -e "$ROOT/usr/bin/$tool" ]; then
    echo "FAIL: the AppImage still carries $tool"
    exit 1
  fi
done

BIN=$(find "$ROOT/usr/bin" -maxdepth 1 -type f -name chaty | head -1)
missing=$(LD_LIBRARY_PATH="$ROOT/usr/lib" ldd "$BIN" | grep "not found" || true)
if [ -n "$missing" ]; then
  echo "FAIL: unresolved libraries"
  echo "$missing"
  exit 1
fi

LLAMA=$(find "$ROOT/usr/bin" -maxdepth 1 -type f -name chaty-llama | head -1)
[ -n "$LLAMA" ] || { echo "FAIL: no chaty-llama in the AppImage"; exit 1; }
# As the app starts it: no library path but the binary's own.
ready=$(printf '{"cmd":"quit"}\n' | "$LLAMA" 2>/dev/null | head -1 || true)
if ! printf '%s' "$ready" | grep -q '"event":"ready"'; then
  echo "FAIL: chaty-llama did not answer: $ready"
  exit 1
fi

# A command the agent runs, in the environment AppRun gives the app (issue
# #28): nothing of the image may reach it.
seen=$(python3 -I "$HERE/appimage-hostenv.py" "$ROOT" "$WORK") || {
  echo "FAIL: the agent's command did not run: $seen"
  exit 1
}
if printf '%s\n' "$seen" | grep -F "$ROOT"; then
  echo "FAIL: a program the app starts still sees the image's paths (above)"
  exit 1
fi

set +e
xvfb-run -a timeout 20 "$ROOT/AppRun" > run.log 2>&1
code=$?
set -e
if grep -q "error while loading shared libraries" run.log; then
  echo "FAIL: a library is missing at start"
  cat run.log
  exit 1
fi
# 124: still running when the timeout stopped it — it started.
if [ "$code" -ne 124 ]; then
  echo "FAIL: exited with $code within 20 s"
  tail -40 run.log
  exit 1
fi
echo "OK: no bundled libwayland-client or xdg-open, every library resolves, the chat engine answers, programs it starts see the host's environment, and it starts"
