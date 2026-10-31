#!/usr/bin/env bash
# Repack the AppImage Tauri built without its own libwayland-client and
# xdg-open, so the host's are used, and with its chat engine able to start.
#
# The AppImage community's excludelist names it: a newer Mesa on the host
# does not work with the older libwayland-client bundled from the build
# machine. On Fedora that is a gray window (issue #26), which preloading the
# system's copy cures. linuxdeploy's GTK plugin bundles it anyway, so it is
# taken out of the AppDir here and the image is packed again from it.
#
# The image's own xdg-open and xdg-mime go too (issue #28): they found no
# way to open a folder on Fedora, and what they did start loaded the image's
# libraries. The host's are used; Tauri's bundler stopped shipping them as
# well (tauri-apps/tauri#15804).
#
# .DirIcon, the image's icon for file managers and the AppImage catalog, is a
# link Tauri before 2.11.4 makes to the build machine's absolute path — broken
# anywhere else, and the catalog's check fails on it (tauri-apps/tauri#15596).
# It is pointed at the icon beside it instead.
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
rm -fv "$APPDIR/usr/bin/xdg-open" "$APPDIR/usr/bin/xdg-mime"
ICON=$(find "$APPDIR" -maxdepth 1 -name '*.png' | head -1)
[ -n "$ICON" ] || { echo "error: no icon at the AppDir's top" >&2; exit 1; }
ln -sfn "$(basename "$ICON")" "$APPDIR/.DirIcon"
echo ".DirIcon -> $(readlink "$APPDIR/.DirIcon")"
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
