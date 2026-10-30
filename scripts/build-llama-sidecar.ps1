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
    git -C "$src\llama-cpp-sys-2\llama.cpp" fetch --quiet origin $env:CHATY_LLAMA_REV
    git -C "$src\llama-cpp-sys-2\llama.cpp" checkout --quiet FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp $env:CHATY_LLAMA_REV not found" }
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
