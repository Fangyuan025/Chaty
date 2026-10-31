#!/usr/bin/env bash
# Check the AppImage before it is published: it carries no libwayland-client
# or xdg-open of its own, its icon and desktop file pass what the AppImage
# catalog checks, every library its binary needs resolves, its chat engine
# (chaty-llama) answers, a program it starts sees none of the image's paths,
# its files are open to every user, and it starts — its window shown and
# still running after 20 s under a virtual display, with no missing library
# said, and handling chaty:// links.
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

# Readable by everyone, and runnable where it is a program: firejail and
# others mount the image themselves, owned by root, and a file only its
# owner may run (AppRun.wrapped was) stopped the app before it started. Read
# from the image, as it is stored: extracting does not keep folders' modes.
closed=$(unsquashfs -o "$("$IMG" --appimage-offset)" -lls "$IMG" | awk '
  length($1) == 10 && $1 ~ /^[-dcbps][-rwxsStT]+$/ {
    if (substr($1, 8, 1) != "r" || (substr($1, 4, 1) == "x" && substr($1, 10, 1) != "x")) print $1, $NF
  }' | head -5)
if [ -n "$closed" ]; then
  echo "FAIL: not open to every user:"
  echo "$closed"
  exit 1
fi

for tool in xdg-open xdg-mime; do
  if [ -e "$ROOT/usr/bin/$tool" ]; then
    echo "FAIL: the AppImage still carries $tool"
    exit 1
  fi
done

# What the AppImage catalog (appimage.github.io) checks first.
[ -e "$ROOT/.DirIcon" ] || { echo "FAIL: .DirIcon does not lead to an icon: $(readlink "$ROOT/.DirIcon")"; exit 1; }
DESKTOP=$(find "$ROOT" -maxdepth 1 -name '*.desktop' | head -1)
desktop-file-validate "$DESKTOP" || { echo "FAIL: desktop-file-validate"; exit 1; }
for key in Icon Categories; do
  [ "$(grep -c "^$key=." "$DESKTOP")" = 1 ] || { echo "FAIL: $key is not set once in the desktop file"; exit 1; }
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

# As on a desktop: a 24-bit display (at xvfb-run's default 8 bits the window
# never came up) and a session bus of its own (without one, GLib launches a
# bus whose services never answer, and GTK's start waits 25 s on it). It is
# still running after 20 s (124: stopped by the timeout) and has shown its
# window.
set +e
xvfb-run -a -s "-screen 0 1280x800x24" dbus-run-session -- bash -c '
  timeout 20 "$1/AppRun" > run.log 2>&1 &
  for _ in $(seq 1 18); do
    sleep 1
    xdotool search --onlyvisible --name "^Chaty$" >/dev/null 2>&1 && { touch shown; break; }
  done
  [ -e shown ] || xwininfo -root -tree > windows.txt 2>&1
  wait $!' _ "$ROOT"
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
if [ ! -e shown ]; then
  echo "FAIL: no Chaty window within 18 s"
  echo "windows on the display:"
  grep -E '"[^"]+"' windows.txt | head -20 || true
  tail -40 run.log
  exit 1
fi
# As it started, it made itself the handler of chaty:// links, with the
# user's tools in the user's environment (src-tauri/src/scheme.rs).
HANDLER="${XDG_DATA_HOME:-$HOME/.local/share}/applications/chaty-handler.desktop"
if [ ! -f "$HANDLER" ]; then
  echo "FAIL: no $HANDLER"
  ls -la "$(dirname "$HANDLER")" 2>&1 || true
  grep -iE "scheme|panic|error" run.log | tail -20 || true
  exit 1
fi
desktop-file-validate "$HANDLER" || { echo "FAIL: desktop-file-validate on the link handler"; exit 1; }
if command -v xdg-mime >/dev/null; then
  said=$(xdg-mime query default x-scheme-handler/chaty)
  [ "$said" = chaty-handler.desktop ] || { echo "FAIL: chaty:// opens with '$said'"; exit 1; }
fi
echo "OK: no bundled libwayland-client or xdg-open, its icon and desktop file pass, its files are open to every user, every library resolves, the chat engine answers, programs it starts see the host's environment, and it shows its window and handles chaty:// links"
