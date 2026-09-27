# chaty-audio — the music-generation engine, a sidecar process around
# audio.cpp (https://github.com/0xShug0/audio.cpp).
#
# A separate process for the reasons chaty-sd is one: audio.cpp vendors its own
# ggml at its own revision, which cannot share a binary with llama.cpp's; and a
# song is minutes of GPU work that audio.cpp gives no way to interrupt, so
# stopping one is ending the process — which is also what hands every byte of
# the model back at once.
#
# Only the families Chaty drives are compiled in (AUDIOCPP_MODELS), which keeps
# the build to a fraction of the full engine's eighty.
#
# Build (see scripts/build-audio-sidecar.sh / .ps1, which stage the result where
# Tauri's externalBin expects it):
#   cmake -S src-tauri/audio-sidecar -B build -DCHATY_AUDIO_BACKEND=vulkan
#   cmake --build build --config Release --target chaty-audio
#
# Offline / local checkout of audio.cpp:
#   -DFETCHCONTENT_SOURCE_DIR_AUDIOCPP=/path/to/audio.cpp
cmake_minimum_required(VERSION 3.20)
project(chaty-audio C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CHATY_AUDIO_BACKEND "cpu" CACHE STRING "GPU backend: cpu | vulkan | metal | cuda")
set_property(CACHE CHATY_AUDIO_BACKEND PROPERTY STRINGS cpu vulkan metal cuda)

# The pinned engine. Bumping it is a deliberate act: a newer revision is what
# brings new model families, and it is also what can move the C API under us.
set(CHATY_AUDIOCPP_REPO "https://github.com/0xShug0/audio.cpp.git" CACHE STRING "")
set(CHATY_AUDIOCPP_REV "955c8725c611d511774e6be132aff6609163b2d2" CACHE STRING "")
# The model families compiled in. Each one Chaty can load is listed in
# src-tauri/src/musicgen/probe.rs as well.
set(CHATY_AUDIO_MODELS "yue2,ace_step,heartmula,stable_audio,minimax_music3,midashenglm_gen" CACHE STRING "")

# Only the runtime: no CLI, server, tests, model manager or C API library (its
# one source file is compiled into the sidecar instead, below).
set(AUDIOCPP_MODEL_SET custom CACHE STRING "" FORCE)
set(AUDIOCPP_MODELS "${CHATY_AUDIO_MODELS}" CACHE STRING "" FORCE)
# Model specs compiled into the runtime: the sidecar ships as one file, with
# no model_specs/ folder beside it.
set(AUDIOCPP_DEPLOYMENT_BUILD ON CACHE BOOL "" FORCE)
set(AUDIOCPP_BUILD_C_API OFF CACHE BOOL "" FORCE)
set(AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER OFF CACHE BOOL "" FORCE)
set(AUDIOCPP_STATIC_ESPEAK OFF CACHE BOOL "" FORCE)
set(ENGINE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ENGINE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENGINE_BUILD_EXTENDED_TESTS OFF CACHE BOOL "" FORCE)
set(ENGINE_BUILD_MODEL_TESTS OFF CACHE BOOL "" FORCE)
set(ENGINE_BUILD_WARMBENCH OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)

# A shipped binary must not be tuned to whatever CPU built it (see chaty-sd):
# x86-64 gets a fixed AVX2 baseline, checked at startup.
set(ENGINE_ENABLE_NATIVE_CPU OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_BMI2 ON CACHE BOOL "" FORCE)
    set(CHATY_AUDIO_NEEDS_AVX2 ON)
endif()

set(ENGINE_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_HIP OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_METAL OFF CACHE BOOL "" FORCE)
if (CHATY_AUDIO_BACKEND STREQUAL "vulkan")
    set(ENGINE_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
elseif (CHATY_AUDIO_BACKEND STREQUAL "metal")
    set(ENGINE_ENABLE_METAL ON CACHE BOOL "" FORCE)
    # Embed the Metal library in the binary so the sidecar needs no
    # default.metallib beside it inside the .app bundle.
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    # Apple's clang has no OpenMP; ggml's own thread pool does the work.
    set(ENGINE_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
elseif (CHATY_AUDIO_BACKEND STREQUAL "cuda")
    set(ENGINE_ENABLE_CUDA ON CACHE BOOL "" FORCE)
elseif (NOT CHATY_AUDIO_BACKEND STREQUAL "cpu")
    message(FATAL_ERROR "CHATY_AUDIO_BACKEND must be cpu, vulkan, metal or cuda (got '${CHATY_AUDIO_BACKEND}')")
endif()
# Windows too: MSVC's OpenMP runtime (vcomp140.dll) is not on a machine
# without the Visual C++ redistributable, and the installer ships no copy.
# The music families' heavy work is ggml's, whose own thread pool does it.
if (APPLE OR WIN32)
    set(ENGINE_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()

# What audio.cpp's own Windows build (scripts/build_windows.ps1) gives every
# target: its sources are UTF-8 (tokenizers, prompt markers), and C++
# exceptions carry the engine's errors.
if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
endif()

include(FetchContent)
FetchContent_Declare(audiocpp
    GIT_REPOSITORY ${CHATY_AUDIOCPP_REPO}
    GIT_TAG ${CHATY_AUDIOCPP_REV}
    # ggml and the rest are vendored in the tree; the one submodule is the
    # server's web frontends (and an ssh URL a CI runner cannot fetch).
    GIT_SUBMODULES ""
    GIT_PROGRESS ON
    # LF as committed, whatever the runner's git says: the patches below
    # apply to the files as they are in the repository.
    GIT_CONFIG core.autocrlf=false
    # The engine's own targets (CLI, server, converter) are built only when
    # something here asks for them — nothing does.
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(audiocpp)

# Chaty's changes to the engine, applied to the checkout once (a checkout that
# already has them is left alone). They only add log lines: audio.cpp says
# nothing while a song is made, and these lines — a count of music tokens, a
# rendering step, a decoded tile — are what the studio's percentage is made
# from (see main.cpp, on_log_line).
find_package(Git REQUIRED)
file(GLOB CHATY_AUDIOCPP_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
list(SORT CHATY_AUDIOCPP_PATCHES)
foreach(patch IN LISTS CHATY_AUDIOCPP_PATCHES)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --check --reverse --ignore-whitespace "${patch}"
        WORKING_DIRECTORY "${audiocpp_SOURCE_DIR}"
        RESULT_VARIABLE already OUTPUT_QUIET ERROR_QUIET)
    if (NOT already EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply --ignore-whitespace --whitespace=nowarn "${patch}"
            WORKING_DIRECTORY "${audiocpp_SOURCE_DIR}"
            RESULT_VARIABLE applied ERROR_VARIABLE why)
        if (NOT applied EQUAL 0)
            message(FATAL_ERROR "chaty-audio: ${patch} does not apply to audio.cpp at ${CHATY_AUDIOCPP_REV}:\n${why}")
        endif()
        message(STATUS "chaty-audio: applied ${patch}")
    endif()
endforeach()

set(CHATY_AUDIO_SOURCES
    main.cpp
    # The C ABI, linked in statically: the sidecar talks to the engine through
    # the same surface an embedding app would, and no libaudiocpp ships.
    ${audiocpp_SOURCE_DIR}/src/capi/audiocpp.cpp
)
if (WIN32)
    list(APPEND CHATY_AUDIO_SOURCES chaty-audio.manifest)
endif()
add_executable(chaty-audio ${CHATY_AUDIO_SOURCES})
target_include_directories(chaty-audio PRIVATE
    ${audiocpp_SOURCE_DIR}/include
    ${audiocpp_SOURCE_DIR}/external/cJSON
)
target_link_libraries(chaty-audio PRIVATE engine_runtime ggml cjson_vendor)
# The engine is built with OpenMP (except on Apple) and needs its runtime at
# the final link; its imported target is scoped to the engine's directory.
if (ENGINE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(chaty-audio PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(chaty-audio PRIVATE
    AUDIOCPP_VERSION_STRING="${CHATY_AUDIOCPP_REV}"
    CHATY_AUDIOCPP_REV="${CHATY_AUDIOCPP_REV}"
)
if (CHATY_AUDIO_NEEDS_AVX2)
    target_compile_definitions(chaty-audio PRIVATE CHATY_AUDIO_NEEDS_AVX2=1)
endif()
if (MSVC)
    target_compile_options(chaty-audio PRIVATE /utf-8 /EHsc)
endif()
if (APPLE)
    set_target_properties(chaty-audio PROPERTIES
        BUILD_RPATH "@executable_path"
        INSTALL_RPATH "@executable_path")
endif()
set_target_properties(chaty-audio PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
