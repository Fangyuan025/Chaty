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
    SUB="$SRC/llama-cpp-sys-2/llama.cpp"
    # The pinned llama.cpp can be upstream plus a fix upstream has not merged
    # yet (on our fork of it). Those commits are carried onto the newer
    # upstream: one upstream already has is skipped, and one that no longer
    # applies fails the build — a pack without the fix would replace the
    # bundled engine that has it. A commit marked `Chaty-Carry: optional` (a
    # speed-up, never a fix of results) is left out instead.
    CARRY_TIP=$(git -C "$SUB" rev-parse HEAD)
    git -C "$SUB" fetch --quiet https://github.com/ggml-org/llama.cpp master
    UPSTREAM=$(git -C "$SUB" rev-parse FETCH_HEAD)
    git -C "$SUB" fetch --quiet origin "$CHATY_LLAMA_REV"
    git -C "$SUB" checkout --quiet FETCH_HEAD
    for C in $(git -C "$SUB" rev-list --reverse --no-merges "$UPSTREAM..$CARRY_TIP"); do
      if git -C "$SUB" diff "$C^" "$C" | git -C "$SUB" apply --reverse --check - 2>/dev/null; then
        echo "llama.cpp: $(git -C "$SUB" log -1 --format='%h %s' "$C") — upstream has it"
      elif git -C "$SUB" -c user.name=chaty -c user.email=chaty@localhost cherry-pick "$C" >/dev/null; then
        echo "llama.cpp: carried $(git -C "$SUB" log -1 --format='%h %s' "$C")"
      elif [[ "$(git -C "$SUB" log -1 --format='%(trailers:key=Chaty-Carry,valueonly)' "$C" | tr -d '[:space:]')" == optional ]]; then
        git -C "$SUB" cherry-pick --abort
        echo "llama.cpp: left out $(git -C "$SUB" log -1 --format='%h %s' "$C") — does not apply, optional"
      else
        echo "error: $(git -C "$SUB" log -1 --format='%h %s' "$C") does not apply to llama.cpp $CHATY_LLAMA_REV" >&2
        exit 1
      fi
    done
  fi
  echo "chaty-llama: bindings $PIN, llama.cpp $(git -C "$SRC/llama-cpp-sys-2/llama.cpp" rev-parse HEAD)"
  ARGS+=(--config "patch.\"$URL\".llama-cpp-2.path=\"$SRC/llama-cpp-2\""
         --config "patch.\"$URL\".llama-cpp-sys-2.path=\"$SRC/llama-cpp-sys-2\"")
  # The patch makes the bindings path dependencies, and cargo writes that into
  # Cargo.lock (their `source` lines go). Put the lock back afterwards, so a
  # local build against another llama.cpp leaves nothing to commit by mistake.
  cp Cargo.lock "$SRC.Cargo.lock"
  trap 'cp "$SRC.Cargo.lock" "$ROOT/src-tauri/Cargo.lock"' EXIT
fi

cargo build "${ARGS[@]}"
BIN="${CARGO_TARGET_DIR:-$ROOT/src-tauri/target}/release/chaty-llama"
"$ROOT/scripts/llama-standalone.sh" "$BIN"
echo "built: $BIN"
