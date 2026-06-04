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

# 3. Vulkan SDK for the GPU (Vulkan) llama.cpp build. Auto-detect the newest
#    install under C:\VulkanSDK. Build with `--no-default-features` for CPU-only.
if (-not $env:VULKAN_SDK) {
  $vk = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
  if ($vk) { $env:VULKAN_SDK = $vk.FullName }
}
if ($env:VULKAN_SDK) { $env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH" }

# 4. Short target dir + Ninja: the Vulkan llama.cpp build nests very deep
#    (vulkan-shaders-gen/...), so the default target path blows past Windows
#    MAX_PATH (260) and the shader compile fails with C1083. A short root fixes it.
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = "C:\ct" }
if (-not $env:CMAKE_GENERATOR) { $env:CMAKE_GENERATOR = "Ninja" }
$ninjaDir = "C:\ProgramData\anaconda3\Library\bin"
if ((Test-Path "$ninjaDir\ninja.exe") -and ($env:PATH -notlike "*$ninjaDir*")) {
  $env:PATH = "$env:PATH;$ninjaDir"
}

Write-Host "cmake    : $((Get-Command cmake -ErrorAction SilentlyContinue).Source)"
Write-Host "cl       : $((Get-Command cl    -ErrorAction SilentlyContinue).Source)"
Write-Host "libclang : $env:LIBCLANG_PATH"
Write-Host "vulkan   : $env:VULKAN_SDK"
Write-Host ""

# 3. Run the app (Vite dev server + Tauri window, hot-reloads on changes).
npm run tauri dev
