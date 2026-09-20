#!/usr/bin/env bash
# Bump the app version in the four places that must stay in sync, then refresh
# Cargo.lock. Uses sed per repo convention (PowerShell Set-Content writes a BOM
# that breaks the JSON parsers).
#
#   ./scripts/bump-version.sh 0.6.0
#   git commit -am "v0.6.0: ..." && git tag v0.6.0 && git push --tags
set -euo pipefail
cd "$(dirname "$0")/.."

VER="${1:?usage: bump-version.sh <x.y.z>}"
case "$VER" in
  *[!0-9.]*) echo "not a x.y.z version: $VER" >&2; exit 1 ;;
esac

# package.json + tauri.conf.json: first "version" field
sed -i.bak -E "s/(\"version\": \")[0-9]+\.[0-9]+\.[0-9]+(\")/\1$VER\2/" package.json
sed -i.bak -E "s/(\"version\": \")[0-9]+\.[0-9]+\.[0-9]+(\")/\1$VER\2/" src-tauri/tauri.conf.json
# package-lock.json carries the app's own version twice, at the root and in
# packages[""] — npm rewrites them on its next install, so a lock left behind
# turns up as an unrelated diff under someone else's change.
sed -i.bak -E "1,12 s/(\"version\": \")[0-9]+\.[0-9]+\.[0-9]+(\")/\1$VER\2/" package-lock.json
# Cargo.toml: the [package] version (first match only; BSD sed lacks GNU's
# 0,/re/ address form, so use perl)
perl -pi -e 'if (!$done && s/^version = "[0-9.]+"/version = "'"$VER"'"/) { $done = 1 }' src-tauri/Cargo.toml
rm -f package.json.bak src-tauri/tauri.conf.json.bak package-lock.json.bak

# Refresh Cargo.lock to match
(cd src-tauri && cargo update -p chaty --precise "$VER" 2>/dev/null || cargo check -q >/dev/null)

echo "bumped to $VER:"
grep -H '"version"' package.json src-tauri/tauri.conf.json | head -2
grep -Hm2 '"version"' package-lock.json
grep -H '^version' src-tauri/Cargo.toml | head -1
