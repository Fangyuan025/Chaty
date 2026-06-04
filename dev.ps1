# Launch Chaty in dev mode with the native build environment configured.
#
#   .\dev.ps1
#
# Sets up the MSVC toolchain (cl/cmake/INCLUDE/LIB) and libclang (for the
# llama-cpp-sys-2 bindgen step), then runs `npm run tauri dev`.
$ErrorActionPreference = "Stop"

# 1. MSVC environment via vcvars64 (auto-located through vswhere).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -property installationPath
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:\$($matches[1])" -Value $matches[2] }
}

# 2. libclang for bindgen. Override by setting $env:LIBCLANG_PATH before running.
if (-not $env:LIBCLANG_PATH) {
  $env:LIBCLANG_PATH = "C:\Users\25289\AppData\Local\Programs\Python\Python39\Lib\site-packages\clang\native"
}

Write-Host "cmake    : $((Get-Command cmake -ErrorAction SilentlyContinue).Source)"
Write-Host "cl       : $((Get-Command cl    -ErrorAction SilentlyContinue).Source)"
Write-Host "libclang : $env:LIBCLANG_PATH"
Write-Host ""

# 3. Run the app (Vite dev server + Tauri window, hot-reloads on changes).
npm run tauri dev
