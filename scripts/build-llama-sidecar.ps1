# Build chaty-llama — the llama.cpp engine in its own process
# (src-tauri\src\bin\chaty-llama.rs) — on Windows. `tauri build` builds it
# along with the app; this is for building it alone.
#
# Usage (from a VS developer shell, with the Vulkan SDK for the GPU build):
#   .\scripts\build-llama-sidecar.ps1
# $env:CHATY_LLAMA_REV builds it against that llama.cpp commit instead of the
# one the bindings pin — what a runtime pack is built from
# (.github/workflows/runtimes.yml). The bindings stay at the revision
# Cargo.toml pins; only their llama.cpp checkout moves.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location (Join-Path $root "src-tauri")

$cargoArgs = @("build", "--release", "--bin", "chaty-llama")
if ($env:CHATY_LLAMA_REV) {
  $url = "https://github.com/Fangyuan025/llama-cpp-rs"
  $pin = (Select-String -Path Cargo.toml -Pattern '^llama-cpp-2 = .*rev = "([0-9a-f]+)"').Matches[0].Groups[1].Value
  if (-not $pin) { throw "no llama-cpp-rs revision in Cargo.toml" }
  # Short: llama.cpp's tree nests deep enough to hit MAX_PATH.
  $base = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
  $src = Join-Path $base ("lrs-" + $pin.Substring(0, 8) + "-" + $env:CHATY_LLAMA_REV.Substring(0, 8))
  if (-not (Test-Path (Join-Path $src ".git"))) {
    git clone --filter=blob:none $url $src
    if ($LASTEXITCODE -ne 0) { throw "clone failed" }
    git -C $src checkout --quiet $pin
    git -C $src submodule update --init --filter=blob:none llama-cpp-sys-2/llama.cpp
    $sub = "$src\llama-cpp-sys-2\llama.cpp"
    # The pinned llama.cpp can be upstream plus a fix upstream has not merged
    # yet (on our fork of it). Those commits are carried onto the newer
    # upstream: one upstream already has is skipped, and one that no longer
    # applies fails the build — a pack without the fix would replace the
    # bundled engine that has it. A commit marked `Chaty-Carry: optional` (a
    # speed-up, or an architecture upstream does not have yet, whose models then
    # fall back to the bundled engine — never a fix of results) is left out
    # instead.
    $carryTip = (git -C $sub rev-parse HEAD).Trim()
    git -C $sub fetch --quiet https://github.com/ggml-org/llama.cpp master
    if ($LASTEXITCODE -ne 0) { throw "upstream llama.cpp not reachable" }
    $upstream = (git -C $sub rev-parse FETCH_HEAD).Trim()
    git -C $sub fetch --quiet origin $env:CHATY_LLAMA_REV
    git -C $sub checkout --quiet FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp $env:CHATY_LLAMA_REV not found" }
    $carried = @(git -C $sub rev-list --reverse --no-merges "$upstream..$carryTip")
    foreach ($c in $carried) {
      if (-not $c) { continue }
      $what = (git -C $sub log -1 --format="%h %s" $c).Trim()
      $patch = Join-Path $base ("carry-" + $c.Substring(0, 12) + ".patch")
      git -C $sub diff --output=$patch "$c^" $c
      git -C $sub apply --reverse --check $patch 2>$null
      if ($LASTEXITCODE -eq 0) { Write-Host "llama.cpp: $what — upstream has it"; continue }
      git -C $sub -c user.name=chaty -c user.email=chaty@localhost cherry-pick $c | Out-Null
      if ($LASTEXITCODE -ne 0) {
        $carry = (git -C $sub log -1 --format="%(trailers:key=Chaty-Carry,valueonly)" $c | Out-String).Trim()
        if ($carry -ne "optional") { throw "$what does not apply to llama.cpp $env:CHATY_LLAMA_REV" }
        git -C $sub cherry-pick --abort
        Write-Host "llama.cpp: left out $what — does not apply, optional"
        continue
      }
      Write-Host "llama.cpp: carried $what"
    }
  }
  $fwd = $src -replace '\\', '/'
  Write-Host "chaty-llama: bindings $pin, llama.cpp $env:CHATY_LLAMA_REV"
  $cargoArgs += @("--config", "patch.`"$url`".llama-cpp-2.path=`"$fwd/llama-cpp-2`"",
                  "--config", "patch.`"$url`".llama-cpp-sys-2.path=`"$fwd/llama-cpp-sys-2`"")
}

cargo @cargoArgs
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }
Write-Host "built: $target\release\chaty-llama.exe"
