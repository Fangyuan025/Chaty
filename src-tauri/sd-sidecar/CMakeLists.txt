# chaty-sd — the image-generation engine, a sidecar process around
# stable-diffusion.cpp.
#
# A separate process rather than a library linked into the app: llama.cpp (the
# chat engine) and stable-diffusion.cpp each vendor their own ggml, at
# different revisions, and two copies of ggml in one binary collide at link
# time. As a process it also gives the same guarantees the MLX sidecar does —
# ejecting the model is killing the process, so every byte comes back, and a
# GPU driver that aborts takes the sidecar down instead of the app.
#
# Build (see scripts/build-sd-sidecar.sh / .ps1, which stage the result where
# Tauri's externalBin expects it):
#   cmake -S src-tauri/sd-sidecar -B build -DCHATY_SD_BACKEND=vulkan
#   cmake --build build --config Release
#
# Offline / local checkout of stable-diffusion.cpp:
#   -DFETCHCONTENT_SOURCE_DIR_SDCPP=/path/to/stable-diffusion.cpp
cmake_minimum_required(VERSION 3.16)
project(chaty-sd C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CHATY_SD_BACKEND "cpu" CACHE STRING "GPU backend: cpu | vulkan | metal | cuda")
set_property(CACHE CHATY_SD_BACKEND PROPERTY STRINGS cpu vulkan metal cuda)

# The pinned engine. Bumping it is a deliberate act: a newer revision is what
# brings new model families, and it is also what can move the C API under us.
set(CHATY_SDCPP_REPO "https://github.com/leejet/stable-diffusion.cpp.git" CACHE STRING "")
set(CHATY_SDCPP_REV "88411ef1e0688ff2df1010aeeb5d92b2d8cea2be" CACHE STRING "")

# Library only — no CLI, no server, no WebP/WebM (output is PNG/JPEG).
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# A shipped binary must not be tuned to whatever CPU built it: GGML_NATIVE
# compiles for the CI runner, and the first user with an older CPU gets an
# illegal-instruction crash. x86-64 gets a fixed AVX2 baseline instead (the
# sidecar checks for it at startup and says so plainly when it is missing);
# Apple Silicon and other ARM targets keep ggml's defaults.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_BMI2 ON CACHE BOOL "" FORCE)
    set(CHATY_SD_NEEDS_AVX2 ON)
endif()

if (CHATY_SD_BACKEND STREQUAL "vulkan")
    set(SD_VULKAN ON CACHE BOOL "" FORCE)
elseif (CHATY_SD_BACKEND STREQUAL "metal")
    set(SD_METAL ON CACHE BOOL "" FORCE)
    # Embed the Metal library in the binary so the sidecar needs no
    # default.metallib beside it inside the .app bundle.
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
elseif (CHATY_SD_BACKEND STREQUAL "cuda")
    set(SD_CUDA ON CACHE BOOL "" FORCE)
elseif (NOT CHATY_SD_BACKEND STREQUAL "cpu")
    message(FATAL_ERROR "CHATY_SD_BACKEND must be cpu, vulkan, metal or cuda (got '${CHATY_SD_BACKEND}')")
endif()

include(FetchContent)
FetchContent_Declare(sdcpp
    GIT_REPOSITORY ${CHATY_SDCPP_REPO}
    GIT_TAG ${CHATY_SDCPP_REV}
    # ggml is the only submodule the library needs; the web UI, libwebp and
    # libwebm are hundreds of MB of history for features switched off above.
    GIT_SUBMODULES ggml
    GIT_PROGRESS ON
)
FetchContent_MakeAvailable(sdcpp)

if (WIN32)
    add_executable(chaty-sd main.cpp chaty-sd.manifest)
else()
    add_executable(chaty-sd main.cpp)
endif()
target_link_libraries(chaty-sd PRIVATE stable-diffusion)
if (CHATY_SD_NEEDS_AVX2)
    target_compile_definitions(chaty-sd PRIVATE CHATY_SD_NEEDS_AVX2=1)
endif()
if (MSVC)
    target_compile_options(chaty-sd PRIVATE /utf-8 /EHsc)
    # A console subsystem binary is what a pipe-driven child wants; the app
    # spawns it with CREATE_NO_WINDOW so no console ever flashes.
endif()
if (APPLE)
    set_target_properties(chaty-sd PROPERTIES
        BUILD_RPATH "@executable_path"
        INSTALL_RPATH "@executable_path")
endif()
set_target_properties(chaty-sd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
