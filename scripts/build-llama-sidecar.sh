#!/usr/bin/env bash
# Build chaty-llama — the llama.cpp engine in its own process
# (src-tauri/src/bin/chaty-llama.rs) — into src-tauri/target/release/.
# `tauri build` builds it along with the app; this is for building it alone.
#
# Env: CHATY_LLAMA_REV=<llama.cpp commit> builds it against that llama.cpp
#      instead of the one the bindings pin — what a runtime pack is built from
#      (.github/workflows/runtimes.yml). The bindings stay at the revision
#      Cargo.toml pins; only their llama.cpp checkout moves.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src-tauri"

ARGS=(--release --bin chaty-llama)
if [[ -n "${CHATY_LLAMA_REV:-}" ]]; then
  URL=https://github.com/Fangyuan025/llama-cpp-rs
  PIN=$(sed -n 's/^llama-cpp-2 = .*rev = "\([0-9a-f]*\)".*/\1/p' Cargo.toml | head -1)
  [[ -n "$PIN" ]] || { echo "error: no llama-cpp-rs revision in Cargo.toml" >&2; exit 1; }
  SRC="$ROOT/src-tauri/target/llama-rs-$PIN-${CHATY_LLAMA_REV:0:12}"
  if [[ ! -d "$SRC/.git" ]]; then
    git clone --filter=blob:none "$URL" "$SRC"
    git -C "$SRC" checkout --quiet "$PIN"
    git -C "$SRC" submodule update --init --filter=blob:none llama-cpp-sys-2/llama.cpp
    git -C "$SRC/llama-cpp-sys-2/llama.cpp" fetch --quiet origin "$CHATY_LLAMA_REV"
    git -C "$SRC/llama-cpp-sys-2/llama.cpp" checkout --quiet FETCH_HEAD
  fi
  echo "chaty-llama: bindings $PIN, llama.cpp $(git -C "$SRC/llama-cpp-sys-2/llama.cpp" rev-parse HEAD)"
  ARGS+=(--config "patch.\"$URL\".llama-cpp-2.path=\"$SRC/llama-cpp-2\""
         --config "patch.\"$URL\".llama-cpp-sys-2.path=\"$SRC/llama-cpp-sys-2\"")
fi

cargo build "${ARGS[@]}"
echo "built: ${CARGO_TARGET_DIR:-$ROOT/src-tauri/target}/release/chaty-llama"
