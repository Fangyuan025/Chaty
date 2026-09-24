# Build the chaty-sd image-generation sidecar (stable-diffusion.cpp) on Windows
# and stage it where Tauri's externalBin expects it:
#   src-tauri\binaries\chaty-sd-x86_64-pc-windows-msvc.exe
#
# Usage (from a VS developer shell, or after dev.ps1's environment setup):
#   .\scripts\build-sd-sidecar.ps1 [-Backend vulkan|cpu|cuda]
# The default is vulkan when the Vulkan SDK is installed, cpu otherwise.
# $env:SDCPP_SOURCE_DIR builds from a local stable-diffusion.cpp checkout.
param([string]$Backend = "")
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$src = Join-Path $root "src-tauri\sd-sidecar"
# Short build dir: ggml's Vulkan shader tree nests deep enough to hit MAX_PATH.
$build = if ($env:CHATY_SD_BUILD_DIR) { $env:CHATY_SD_BUILD_DIR } else { "C:\ct-sd" }

if (-not $Backend) {
  $Backend = if ($env:VULKAN_SDK) { "vulkan" } else { "cpu" }
}
$triple = (rustc -vV | Select-String '^host: ').ToString().Substring(6).Trim()

$extra = @()
if ($env:SDCPP_SOURCE_DIR) { $extra += "-DFETCHCONTENT_SOURCE_DIR_SDCPP=$env:SDCPP_SOURCE_DIR" }
$gen = @()
if (Get-Command ninja -ErrorAction SilentlyContinue) { $gen = @("-G", "Ninja") }

Write-Host "chaty-sd: backend=$Backend triple=$triple build=$build"
cmake -S $src -B $build @gen -DCMAKE_BUILD_TYPE=Release "-DCHATY_SD_BACKEND=$Backend" @extra
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build $build --config Release --target chaty-sd
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

$bin = Get-ChildItem -Recurse (Join-Path $build "bin") -Filter chaty-sd.exe | Select-Object -First 1 -ExpandProperty FullName
if (-not $bin) { throw "built binary not found under $build\bin" }
$stage = Join-Path $root "src-tauri\binaries"
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item $bin (Join-Path $stage "chaty-sd-$triple.exe") -Force
Write-Host "staged: $stage\chaty-sd-$triple.exe"
