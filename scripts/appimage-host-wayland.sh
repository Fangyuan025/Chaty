#!/usr/bin/env bash
# Repack the AppImage Tauri built without its own libwayland-client, so the
# host's is used, and with its chat engine able to start.
#
# The AppImage community's excludelist names it: a newer Mesa on the host
# does not work with the older libwayland-client bundled from the build
# machine. On Fedora that is a gray window (issue #26), which preloading the
# system's copy cures. linuxdeploy's GTK plugin bundles it anyway, so it is
# taken out of the AppDir here and the image is packed again from it.
#
# chaty-llama, the chat engine, is a second binary Tauri copies in as it is:
# it gets the RPATH the app's own binary has, and loses the voice library's
# entries it never uses (scripts/llama-standalone.sh).
#
# Usage: scripts/appimage-host-wayland.sh   (after `tauri build --bundles appimage`)
set -euo pipefail

DIR=src-tauri/target/release/bundle/appimage
APPDIR=$(ls -d "$DIR"/*.AppDir | head -1)
IMG=$(ls "$DIR"/*.AppImage | head -1)

removed=$(find "$APPDIR" -name 'libwayland-client.so*' -print -delete)
echo "removed:"
echo "${removed:-(no bundled libwayland-client)}"
LLAMA="$APPDIR/usr/bin/chaty-llama"
[ -f "$LLAMA" ] || { echo "error: no chaty-llama in the AppDir" >&2; exit 1; }
./scripts/llama-standalone.sh "$LLAMA" '$ORIGIN/../lib'

TOOL="$HOME/.cache/appimagetool-1.9.1-x86_64.AppImage"
if [ ! -x "$TOOL" ]; then
  mkdir -p "$(dirname "$TOOL")"
  curl -fsSL -o "$TOOL" \
    https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage
  chmod +x "$TOOL"
fi
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --no-appstream "$APPDIR" "$IMG"
echo "repacked $IMG"
