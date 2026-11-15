// chaty-sd — stable-diffusion.cpp as a sidecar process for Chaty.
//
// Protocol: one JSON object per line. Commands arrive on stdin, events leave
// on stdout. The engine's own chatter goes to stderr: stdout is re-pointed at
// stderr at startup and the protocol writes to a private duplicate of the
// original, so a stray printf anywhere in the engine can never corrupt a line
// the app parses.
//
// Commands
//   {"cmd":"load", ...paths and options...}          → load_stage* load_progress* (loaded | error)
//   {"cmd":"generate","id":"…", ...params...}        → stage/progress/preview* image* (done | error)
//   {"cmd":"cancel","mode":"all"|"after_current"}    → the running generate ends with done{cancelled}
//   {"cmd":"unload"}                                 → unloaded
//   {"cmd":"ping"}                                   → pong
//   {"cmd":"quit"}                                   → the process exits
// End of stdin (the app went away) exits too.
//
// One job at a time: load and generate run on a worker thread so the reader
// stays free to deliver a cancel; a second job while one runs is refused.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "stable-diffusion.h"
#include "json.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
#include <intrin.h>
#endif

using json   = nlohmann::json;
namespace fs = std::filesystem;

static const char* CHATY_SD_PROTOCOL = "1";

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static FILE* g_out = nullptr;
static std::mutex g_out_mu;

static void emit(const json& j) {
    // Invalid UTF-8 (a path from a legacy code page, a broken prompt) must not
    // throw out of a callback deep inside the engine — replace it instead.
    std::string s = j.dump(-1, ' ', false, json::error_handler_t::replace);
    std::lock_guard<std::mutex> lk(g_out_mu);
    fwrite(s.data(), 1, s.size(), g_out);
    fputc('\n', g_out);
    fflush(g_out);
}

static void setup_io() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    int fd = _dup(_fileno(stdout));
    _dup2(_fileno(stderr), _fileno(stdout));
    _setmode(fd, _O_BINARY);
    g_out = _fdopen(fd, "wb");
#else
    int fd = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);
    g_out = fdopen(fd, "w");
#endif
    if (g_out == nullptr) {
        g_out = stderr;  // better a protocol on the wrong pipe than none
    }
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

#ifdef _WIN32
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty())
        return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
static std::string wide_to_acp(const std::wstring& w) {
    if (w.empty())
        return {};
    int n = WideCharToMultiByte(CP_ACP, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_ACP, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
#endif

/// A UTF-8 path in the form the engine can open.
///
/// stable-diffusion.cpp opens files through narrow `std::ifstream`, which on
/// Windows reads the string in the process code page. The embedded manifest
/// makes that code page UTF-8 on Windows 10 1903 and later, and then the path
/// passes through untouched. On anything older a user folder with a Chinese
/// name would not open, so such a path is handed over as its 8.3 short form,
/// which is plain ASCII whenever the volume keeps short names.
static std::string engine_path(const std::string& utf8) {
#ifdef _WIN32
    if (utf8.empty() || GetACP() == CP_UTF8)
        return utf8;
    bool ascii = std::all_of(utf8.begin(), utf8.end(), [](unsigned char c) { return c < 0x80; });
    if (ascii)
        return utf8;
    std::wstring w = utf8_to_wide(utf8);
    DWORD n        = GetShortPathNameW(w.c_str(), nullptr, 0);
    if (n > 0) {
        std::wstring shortw(n, L'\0');
        DWORD got = GetShortPathNameW(w.c_str(), shortw.data(), n);
        if (got > 0 && got < n) {
            shortw.resize(got);
            return wide_to_acp(shortw);
        }
    }
    return wide_to_acp(w);
#else
    return utf8;
#endif
}

static fs::path fs_path(const std::string& utf8) {
#ifdef _WIN32
    return fs::path(utf8_to_wide(utf8));
#else
    return fs::path(utf8);
#endif
}

static std::string path_utf8(const fs::path& p) {
#ifdef _WIN32
    const std::wstring& w = p.native();
    if (w.empty())
        return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
#else
    return p.string();
#endif
}

static bool read_file(const std::string& utf8, std::vector<unsigned char>& out) {
    std::ifstream f(fs_path(utf8), std::ios::binary);
    if (!f)
        return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static bool write_file(const fs::path& p, const std::vector<unsigned char>& bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f)
        return false;
    f.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
    return (bool)f;
}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------

enum class Job { None, Load, Generate };

static sd_ctx_t* g_ctx = nullptr;

static void ctx_free(sd_ctx_t* ctx) {
#if CHATY_SD_OPT_API
    sd_ctx_free(ctx);
#else
    free_sd_ctx(ctx);
#endif
}
static std::atomic<bool> g_busy{false};
static std::thread g_worker;

// What the callbacks are reporting for. Written by the worker between jobs and
// by the log callback (which runs on engine threads) during one.
static std::mutex g_state_mu;
static Job g_job = Job::None;
static std::string g_job_id;
static std::string g_stage;          // encode | sample | decode | weights
static std::string g_stage_before;   // the stage a lazy weight load interrupted
static int g_batch_index   = 0;
static int g_batch_count   = 1;
static int g_sample_steps  = 0;      // the schedule's length, learned from its first report
static bool g_steps_known  = false;
static std::atomic<bool> g_cancel_requested{false};
static int64_t g_cur_seed  = 0;
static std::string g_last_error;    // the first error of the running job
static std::string g_version_name;
static std::chrono::steady_clock::time_point g_last_load_emit;

static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    size_t i = 0;
    while (i < s.size() && s[i] == ' ')
        i++;
    return s.substr(i);
}

static bool starts_with(const std::string& s, const char* p) {
    return s.rfind(p, 0) == 0;
}

static void emit_stage_locked() {
    json ev = {{"event", "stage"}, {"id", g_job_id}, {"stage", g_stage}, {"index", g_batch_index}, {"count", g_batch_count}};
    if (g_stage == "sample")
        ev["seed"] = g_cur_seed;
    emit(ev);
}

/// Read the engine's own log for what it is doing — it announces each phase
/// of a generation, and nothing else does.
/// The engine prefixes each line with where it was logged ("image.cpp:859  -
/// generating image…"); the phase markers below match what follows.
static std::string strip_source(const std::string& t) {
    size_t dash = t.find(" - ");
    if (dash == std::string::npos || dash > 48)
        return t;
    std::string head = t.substr(0, dash);
    size_t colon     = head.rfind(':');
    if (colon == std::string::npos || colon + 1 >= head.size())
        return t;
    for (size_t i = colon + 1; i < head.size(); i++) {
        if (head[i] != ' ' && (head[i] < '0' || head[i] > '9'))
            return t;
    }
    return trim(t.substr(dash + 3));
}

static void on_log(enum sd_log_level_t level, const char* text, void*) {
    std::string raw = trim(text ? text : "");
    if (raw.empty())
        return;
    const std::string t = strip_source(raw);
    const char* lvl = level == SD_LOG_ERROR ? "ERROR" : level == SD_LOG_WARN ? "WARN" : level == SD_LOG_INFO ? "INFO" : "DEBUG";
    if (level >= SD_LOG_INFO)
        fprintf(stderr, "[%s] %s\n", lvl, raw.c_str());

    std::lock_guard<std::mutex> lk(g_state_mu);
    // The first error of a job is its cause; what follows is the engine
    // unwinding ("cannot inspect model source … No such file" before "get sd
    // version from file failed").
    if (level == SD_LOG_ERROR && g_last_error.empty())
        g_last_error = t;
    if (level >= SD_LOG_WARN)
        emit({{"event", "log"}, {"level", level == SD_LOG_ERROR ? "error" : "warn"}, {"text", t}});
    if (level != SD_LOG_INFO)
        return;

    if (g_job == Job::Load) {
        static const std::pair<const char*, const char*> parts[] = {
            {"loading model from", "model"},
            {"loading diffusion model from", "diffusion"},
            {"loading clip_l from", "clip_l"},
            {"loading clip_g from", "clip_g"},
            {"loading t5xxl from", "t5xxl"},
            {"loading llm from", "llm"},
            {"loading llm vision from", "llm_vision"},
            {"loading vae from", "vae"},
            {"loading tae from", "taesd"},
        };
        for (const auto& p : parts) {
            if (starts_with(t, p.first)) {
                emit({{"event", "load_stage"}, {"component", p.second}});
                return;
            }
        }
        if (starts_with(t, "Version: ")) {
            g_version_name = trim(t.substr(9));
        }
        return;
    }

    if (g_job != Job::Generate)
        return;
    int b = 0, n = 0;
    long long seed = 0;
    if (sscanf(t.c_str(), "generating image: %d/%d - seed %lld", &b, &n, &seed) >= 2) {
        g_stage       = "sample";
        g_steps_known = false;
        g_batch_index = std::max(0, b - 1);
        g_batch_count = std::max(1, n);
        g_cur_seed    = seed;
        emit_stage_locked();
    } else if (starts_with(t, "decoding ")) {
        g_stage       = "decode";
        g_batch_index = 0;
        emit_stage_locked();
    } else if (starts_with(t, "latent ") && t.find(" decoded") != std::string::npos) {
        int i = 0;
        if (sscanf(t.c_str(), "latent %d decoded", &i) == 1) {
            emit({{"event", "progress"}, {"id", g_job_id}, {"stage", "decode"}, {"step", i}, {"steps", g_batch_count}, {"time", 0}});
        }
    } else if (t == "conditioning cache hit") {
        emit({{"event", "cache"}, {"id", g_job_id}, {"kind", "conditioning"}});
    } else if (t.find(" skipped ") != std::string::npos || t.find("completed without skipping steps") != std::string::npos) {
        // "EasyCache skipped 5/20 steps (1.33x estimated speedup)"
        char method[32] = {0};
        int skipped = 0, total = 0;
        if (sscanf(t.c_str(), "%31s skipped %d/%d steps", method, &skipped, &total) == 3) {
            emit({{"event", "cache"}, {"id", g_job_id}, {"kind", "steps"}, {"method", method}, {"skipped", skipped}, {"total", total}});
        } else if (t.find("completed without skipping steps") != std::string::npos) {
            emit({{"event", "cache"}, {"id", g_job_id}, {"kind", "steps"}, {"skipped", 0}, {"total", 0}});
        }
    } else if (starts_with(t, "loading tensors completed") && g_stage == "weights") {
        // A lazy weight load inside the generation finished; carry on with
        // the phase it interrupted.
        g_stage = g_stage_before.empty() ? "encode" : g_stage_before;
        emit_stage_locked();
    }
}

static void on_progress(int step, int steps, float time, void*) {
    std::lock_guard<std::mutex> lk(g_state_mu);
    if (g_job == Job::Load) {
        // Tensor loading reports every few hundred ms per group of files;
        // the bar is the app's to smooth, this only keeps the pipe quiet.
        auto now = std::chrono::steady_clock::now();
        if (step < steps && now - g_last_load_emit < std::chrono::milliseconds(150))
            return;
        g_last_load_emit = now;
        emit({{"event", "load_progress"}, {"step", step}, {"steps", steps}});
        return;
    }
    if (g_job != Job::Generate)
        return;
    // Weights not resident yet are loaded on first use, and that load reports
    // through this same callback — counting tensors, not steps. The sampler
    // always reports first, (0, schedule length), before the model runs and
    // so before any such load; a count that differs afterwards is the load.
    if (g_stage == "sample" && !g_steps_known) {
        g_sample_steps = steps;
        g_steps_known  = true;
    } else if (g_stage == "encode" || (g_stage == "sample" && steps != g_sample_steps)) {
        g_stage_before = g_stage;
        g_stage        = "weights";
        emit_stage_locked();
    } else if (g_stage == "weights" && g_stage_before == "sample" && g_steps_known && steps == g_sample_steps) {
        g_stage = "sample";
        emit_stage_locked();
    }
    emit({{"event", "progress"}, {"id", g_job_id}, {"stage", g_stage}, {"step", step}, {"steps", steps}, {"time", std::isfinite(time) ? time : 0.0f}});
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

static void to_vec(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<unsigned char>*>(ctx);
    auto* p = static_cast<unsigned char*>(data);
    v->insert(v->end(), p, p + size);
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64(const std::vector<unsigned char>& in) {
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t n = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out += B64[(n >> 18) & 63];
        out += B64[(n >> 12) & 63];
        out += B64[(n >> 6) & 63];
        out += B64[n & 63];
    }
    if (i + 1 == in.size()) {
        uint32_t n = in[i] << 16;
        out += B64[(n >> 18) & 63];
        out += B64[(n >> 12) & 63];
        out += "==";
    } else if (i + 2 == in.size()) {
        uint32_t n = (in[i] << 16) | (in[i + 1] << 8);
        out += B64[(n >> 18) & 63];
        out += B64[(n >> 12) & 63];
        out += B64[(n >> 6) & 63];
        out += '=';
    }
    return out;
}

static uint32_t crc32_png(const unsigned char* data, size_t len, uint32_t crc = 0xffffffffu) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        init = true;
    }
    for (size_t i = 0; i < len; i++)
        crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return crc;
}

/// Insert an iTXt chunk (UTF-8, uncompressed) right after IHDR. The prompt and
/// the settings travel inside the picture, where every other Stable Diffusion
/// tool looks for them under the key "parameters".
static void png_add_text(std::vector<unsigned char>& png, const std::string& key, const std::string& text) {
    if (png.size() < 33 || text.empty())
        return;
    std::vector<unsigned char> body;
    body.insert(body.end(), key.begin(), key.end());
    body.push_back(0);  // keyword terminator
    body.push_back(0);  // not compressed
    body.push_back(0);  // compression method
    body.push_back(0);  // empty language tag
    body.push_back(0);  // empty translated keyword
    body.insert(body.end(), text.begin(), text.end());

    std::vector<unsigned char> chunk;
    uint32_t len = (uint32_t)body.size();
    chunk.push_back((len >> 24) & 0xff);
    chunk.push_back((len >> 16) & 0xff);
    chunk.push_back((len >> 8) & 0xff);
    chunk.push_back(len & 0xff);
    const char type[4] = {'i', 'T', 'X', 't'};
    chunk.insert(chunk.end(), type, type + 4);
    chunk.insert(chunk.end(), body.begin(), body.end());
    uint32_t crc = crc32_png(reinterpret_cast<const unsigned char*>(type), 4);
    crc          = crc32_png(body.data(), body.size(), crc) ^ 0xffffffffu;
    chunk.push_back((crc >> 24) & 0xff);
    chunk.push_back((crc >> 16) & 0xff);
    chunk.push_back((crc >> 8) & 0xff);
    chunk.push_back(crc & 0xff);
    // signature (8) + IHDR chunk (4 len + 4 type + 13 data + 4 crc = 25)
    png.insert(png.begin() + 33, chunk.begin(), chunk.end());
}

/// A picture as both forms of the engine's API hand one over: interleaved
/// 8-bit pixels, row by row. Before upstream's opaque API (532d2c1,
/// 2026-11-15) a struct with these fields; since, a U8 tensor shaped
/// [channels, width, height] (CMakeLists.txt says which this build has).
struct Pix {
    uint32_t width        = 0;
    uint32_t height       = 0;
    uint32_t channel      = 0;
    const uint8_t* data   = nullptr;
};

#if CHATY_SD_OPT_API
static Pix pix_of(const sd_tensor_t* t) {
    Pix p;
    if (t == nullptr || sd_tensor_type(t) != SD_TENSOR_U8 || sd_tensor_rank(t) != 3)
        return p;
    p.channel = (uint32_t)sd_tensor_dim(t, 0);
    p.width   = (uint32_t)sd_tensor_dim(t, 1);
    p.height  = (uint32_t)sd_tensor_dim(t, 2);
    p.data    = static_cast<const uint8_t*>(sd_tensor_data(t));
    return p;
}
#else
static Pix pix_of(const sd_image_t& im) {
    return Pix{im.width, im.height, im.channel, im.data};
}
#endif

static void emit_preview(int step, const Pix& img) {
    if (img.data == nullptr)
        return;
    std::string id;
    {
        std::lock_guard<std::mutex> lk(g_state_mu);
        if (g_job != Job::Generate)
            return;
        id = g_job_id;
    }
    std::vector<unsigned char> jpg;
    if (img.channel < 1 || img.channel > 4)
        return;
    if (!stbi_write_jpg_to_func(to_vec, &jpg, (int)img.width, (int)img.height, (int)img.channel, img.data, 82))
        return;
    emit({{"event", "preview"},
          {"id", id},
          {"step", step},
          {"width", img.width},
          {"height", img.height},
          {"data", "data:image/jpeg;base64," + base64(jpg)}});
}

#if CHATY_SD_OPT_API
static void on_preview(int step, int frame_count, const sd_tensor_t* const* frames, bool is_noisy, void*) {
    if (frames == nullptr || frame_count <= 0 || is_noisy)
        return;
    emit_preview(step, pix_of(frames[0]));
}
#else
static void on_preview(int step, int frame_count, sd_image_t* frames, bool is_noisy, void*) {
    if (frames == nullptr || frame_count <= 0 || is_noisy)
        return;
    emit_preview(step, pix_of(frames[0]));
}
#endif

/// A picture read from disk, its pixels owned until it goes.
struct Loaded {
    Pix pix;
    unsigned char* owned = nullptr;
    Loaded()             = default;
    Loaded(const Loaded&) = delete;
    Loaded& operator=(const Loaded&) = delete;
    Loaded(Loaded&& o) noexcept : pix(o.pix), owned(o.owned) { o.owned = nullptr; }
    ~Loaded() {
        if (owned != nullptr)
            stbi_image_free(owned);
    }
};

static bool load_image(const std::string& path, Loaded& out, int want_channels, std::string& err) {
    std::vector<unsigned char> bytes;
    if (!read_file(path, bytes)) {
        err = "cannot read image: " + path;
        return false;
    }
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load_from_memory(bytes.data(), (int)bytes.size(), &w, &h, &c, want_channels);
    if (data == nullptr) {
        err = "cannot decode image: " + path;
        return false;
    }
    out.owned       = data;
    out.pix.width   = (uint32_t)w;
    out.pix.height  = (uint32_t)h;
    out.pix.channel = (uint32_t)(want_channels ? want_channels : c);
    out.pix.data    = data;
    return true;
}

#if CHATY_SD_OPT_API
/// A picture as the engine takes one: a U8 tensor [channels, width, height]
/// holding a copy of the pixels (the options retain it once set).
static sd_tensor_t* tensor_of(const Pix& p) {
    const int64_t shape[] = {(int64_t)p.channel, (int64_t)p.width, (int64_t)p.height};
    return sd_tensor_create(SD_TENSOR_U8, 3, shape, p.data, (size_t)p.width * p.height * p.channel);
}

/// The engine's word for why the last call failed on this thread, if any.
static std::string engine_error() {
    const char* e = sd_get_last_error();
    return e != nullptr ? std::string(e) : std::string();
}

/// A setter's refusal, said on stderr: an option the engine no longer takes
/// is not a reason to fail the job — what it governs falls back to the
/// engine's default.
static void check_opt(sd_status_t st, const char* key) {
    if (st != SD_STATUS_OK)
        fprintf(stderr, "[WARN] option %s not taken (%d): %s\n", key, (int)st, engine_error().c_str());
}
#else
static sd_image_t image_of(const Pix& p) {
    sd_image_t im{};
    im.width   = p.width;
    im.height  = p.height;
    im.channel = p.channel;
    im.data    = const_cast<uint8_t*>(p.data);
    return im;
}
#endif

static fs::path unique_path(const fs::path& dir, const std::string& stem, const std::string& ext) {
    fs::path p = dir / fs_path(stem + ext);
    for (int i = 2; fs::exists(p) && i < 10000; i++)
        p = dir / fs_path(stem + "-" + std::to_string(i) + ext);
    return p;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

static std::string jstr(const json& j, const char* key, const std::string& def = "") {
    auto it = j.find(key);
    return it != j.end() && it->is_string() ? it->get<std::string>() : def;
}
static bool jbool(const json& j, const char* key, bool def) {
    auto it = j.find(key);
    return it != j.end() && it->is_boolean() ? it->get<bool>() : def;
}
static int jint(const json& j, const char* key, int def) {
    auto it = j.find(key);
    return it != j.end() && it->is_number() ? it->get<int>() : def;
}
static int64_t jint64(const json& j, const char* key, int64_t def) {
    auto it = j.find(key);
    return it != j.end() && it->is_number() ? it->get<int64_t>() : def;
}
static float jfloat(const json& j, const char* key, float def) {
    auto it = j.find(key);
    return it != j.end() && it->is_number() ? it->get<float>() : def;
}

static std::string take_last_error() {
    std::lock_guard<std::mutex> lk(g_state_mu);
    std::string e = g_last_error;
    g_last_error.clear();
    return e;
}

static void begin_job(Job job, const std::string& id) {
    std::lock_guard<std::mutex> lk(g_state_mu);
    g_job            = job;
    g_job_id         = id;
    g_stage          = job == Job::Generate ? "encode" : "";
    g_stage_before.clear();
    g_steps_known    = false;
    g_sample_steps   = 0;
    g_batch_index    = 0;
    g_batch_count    = 1;
    g_last_error.clear();
    g_last_load_emit = std::chrono::steady_clock::time_point{};
}

static void end_job() {
    std::lock_guard<std::mutex> lk(g_state_mu);
    g_job = Job::None;
    g_job_id.clear();
}

static void do_load(json cmd) {
    auto t0 = std::chrono::steady_clock::now();
    begin_job(Job::Load, "");
    {
        std::lock_guard<std::mutex> lk(g_state_mu);
        g_version_name.clear();
    }
    if (g_ctx != nullptr) {
        ctx_free(g_ctx);
        g_ctx = nullptr;
    }

    // Keep every string alive for the duration of the context's creation.
    std::string model        = engine_path(jstr(cmd, "model"));
    std::string diffusion    = engine_path(jstr(cmd, "diffusion_model"));
    std::string vae          = engine_path(jstr(cmd, "vae"));
    std::string llm          = engine_path(jstr(cmd, "llm"));
    std::string llm_vision   = engine_path(jstr(cmd, "llm_vision"));
    std::string clip_l       = engine_path(jstr(cmd, "clip_l"));
    std::string clip_g       = engine_path(jstr(cmd, "clip_g"));
    std::string t5xxl        = engine_path(jstr(cmd, "t5xxl"));
    std::string taesd        = engine_path(jstr(cmd, "taesd"));
    std::string max_vram     = jstr(cmd, "max_vram");
    std::string backend      = jstr(cmd, "backend");
    std::string params_backend = jstr(cmd, "params_backend");

    // The CLI's --offload-to-cpu / --clip-on-cpu / --vae-on-cpu, spelled the
    // way the library takes them now: assignments prepended to the lists.
    auto prepend = [](std::string& list, const std::string& item) {
        list = list.empty() ? item : item + "," + list;
    };
    if (jbool(cmd, "offload_to_cpu", false))
        prepend(params_backend, "*=cpu");
    if (jbool(cmd, "clip_on_cpu", false))
        prepend(backend, "te=cpu");
    if (jbool(cmd, "vae_on_cpu", false))
        prepend(backend, "vae=cpu");

    const int threads             = jint(cmd, "threads", -1);
    const bool flash_attn         = jbool(cmd, "flash_attn", false);
    const bool diffusion_fa       = jbool(cmd, "diffusion_fa", true);
    const bool mmap               = jbool(cmd, "mmap", false);
    const bool vae_conv_direct    = jbool(cmd, "vae_conv_direct", false);
    const bool diff_conv_direct   = jbool(cmd, "diffusion_conv_direct", false);
    // Everything resident at load time, the way a chat model loads: the bar
    // the user watches is the whole cost, and the first picture is not
    // mysteriously slower than the rest. The library falls back to lazy
    // loading by itself when the model has to be split across memories.
    const bool eager_load         = jbool(cmd, "eager_load", true);
    const bool auto_fit           = jbool(cmd, "auto_fit", true);
    // The prompt's encoding (and, for editing models, the reference
    // picture's) is kept for the rounds that ask for it again — "again", a
    // batch re-run, a negative prompt that did not change — so the text and
    // vision encoders are skipped for them. Explicit rather than trusting
    // the library's default, which differs between its front ends.
    const int conditioning_cache  = std::max(0, jint(cmd, "conditioning_cache", 8));
    const std::string rng         = jstr(cmd, "rng");
    const std::string prediction  = jstr(cmd, "prediction");

    sd_ctx_t* ctx = nullptr;
    std::string engine_err;
#if CHATY_SD_OPT_API
    {
        sd_ctx_opt_t* o = sd_ctx_opt_create();
        if (o == nullptr) {
            end_job();
            emit({{"event", "error"}, {"scope", "load"}, {"message", "out of memory"}});
            return;
        }
        // A path or list not given is left unset — the engine's default, as
        // the struct's empty fields were — not set to "".
        auto set_str = [&](const char* key, const std::string& v) {
            if (!v.empty())
                check_opt(sd_opt_set_string(o, key, v.c_str()), key);
        };
        auto set_bool = [&](const char* key, bool v) { check_opt(sd_opt_set_bool(o, key, v), key); };
        set_str("model_path", model);
        set_str("diffusion_model_path", diffusion);
        set_str("vae_path", vae);
        set_str("llm_path", llm);
        set_str("llm_vision_path", llm_vision);
        set_str("clip_l_path", clip_l);
        set_str("clip_g_path", clip_g);
        set_str("t5xxl_path", t5xxl);
        set_str("taesd_path", taesd);
        set_bool("tae_preview_only", !taesd.empty());
        if (threads > 0)
            check_opt(sd_opt_set_int(o, "n_threads", threads), "n_threads");
        set_bool("flash_attn", flash_attn);
        set_bool("diffusion_flash_attn", diffusion_fa);
        set_bool("enable_mmap", mmap);
        set_bool("vae_conv_direct", vae_conv_direct);
        set_bool("diffusion_conv_direct", diff_conv_direct);
        set_bool("eager_load", eager_load);
        set_bool("auto_fit", auto_fit);
        check_opt(sd_opt_set_int(o, "conditioning_cache_size", conditioning_cache), "conditioning_cache_size");
        set_str("max_vram", max_vram);
        set_str("backend", backend);
        set_str("params_backend", params_backend);
        // A name the engine does not know is refused by its setter and leaves
        // the engine's own choice, as an unknown name did before.
        set_str("rng_type", rng);
        set_str("prediction", prediction);
        if (sd_ctx_create(o, &ctx) != SD_STATUS_OK) {
            ctx        = nullptr;
            engine_err = engine_error();
        }
        sd_opt_free(o);
    }
#else
    {
        sd_ctx_params_t p;
        sd_ctx_params_init(&p);
        p.model_path            = model.c_str();
        p.diffusion_model_path  = diffusion.c_str();
        p.vae_path              = vae.c_str();
        p.llm_path              = llm.c_str();
        p.llm_vision_path       = llm_vision.c_str();
        p.clip_l_path           = clip_l.c_str();
        p.clip_g_path           = clip_g.c_str();
        p.t5xxl_path            = t5xxl.c_str();
        p.taesd_path            = taesd.c_str();
        p.tae_preview_only      = !taesd.empty();
        if (threads > 0)
            p.n_threads = threads;
        p.flash_attn            = flash_attn;
        p.diffusion_flash_attn  = diffusion_fa;
        p.enable_mmap           = mmap;
        p.vae_conv_direct       = vae_conv_direct;
        p.diffusion_conv_direct = diff_conv_direct;
        p.eager_load            = eager_load;
        p.auto_fit              = auto_fit;
        p.conditioning_cache_size = conditioning_cache;
        p.max_vram       = max_vram.empty() ? nullptr : max_vram.c_str();
        p.backend        = backend.empty() ? nullptr : backend.c_str();
        p.params_backend = params_backend.empty() ? nullptr : params_backend.c_str();
        if (!rng.empty()) {
            enum rng_type_t r = str_to_rng_type(rng.c_str());
            if (r != RNG_TYPE_COUNT)
                p.rng_type = r;
        }
        if (!prediction.empty()) {
            enum prediction_t pr = str_to_prediction(prediction.c_str());
            if (pr != PREDICTION_COUNT)
                p.prediction = pr;
        }
        ctx = new_sd_ctx(&p);
    }
#endif

    std::string err = take_last_error();
    end_job();
    if (ctx == nullptr) {
        if (err.empty())
            err = engine_err;
        emit({{"event", "error"}, {"scope", "load"}, {"message", err.empty() ? "failed to load the model" : err}});
        return;
    }
    g_ctx = ctx;
    std::string version;
    {
        std::lock_guard<std::mutex> lk(g_state_mu);
        version = g_version_name;
    }
    const char* vname = sd_get_model_version_name(ctx);
    if (vname != nullptr && strcmp(vname, "Unknown") != 0)
        version = vname;
    // The sampler and scheduler the engine picks for this model when none is
    // asked for. The opaque API resolves them inside each generation and has
    // no way to ask beforehand; the app then names the family's own (its
    // catalogue's) beside "Auto", as it does whenever this is empty.
#if CHATY_SD_OPT_API
    const std::string default_sampler, default_scheduler;
#else
    enum sample_method_t method = sd_get_default_sample_method(ctx);
    const std::string default_sampler   = sd_sample_method_name(method);
    const std::string default_scheduler = sd_scheduler_name(sd_get_default_scheduler(ctx, method));
#endif
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    emit({{"event", "loaded"},
          {"version", version},
          {"supports_image", sd_ctx_supports_image_generation(ctx)},
          {"supports_video", sd_ctx_supports_video_generation(ctx)},
          {"default_sampler", default_sampler},
          {"default_scheduler", default_scheduler},
          {"elapsed_ms", ms}});
}

// The pictures one generation made, freed when it goes. Upstream changed the
// call to hand back a result that owns them (a85db96, 2026-10-24), and then
// to opaque options and results (532d2c1, 2026-11-15); CMakeLists.txt says
// which form this build's header has, so a runtime pack built from a newer
// upstream compiles as the pinned one does.
#if CHATY_SD_OPT_API
struct Generated {
    sd_result_t* r     = nullptr;
    sd_status_t status = SD_STATUS_OK;
    bool run(sd_ctx_t* ctx, const sd_img_gen_opt_t* g) {
        status = sd_generate_image(ctx, g, &r);
        return status == SD_STATUS_OK;
    }
    int count() const { return r ? (int)sd_result_get_image_count(r) : 0; }
    Pix at(int i) const { return pix_of(sd_result_get_image(r, (size_t)i)); }
    bool cancelled() const { return status == SD_STATUS_CANCELLED; }
    ~Generated() { sd_result_free(r); }
};
#elif CHATY_SD_RESULT_API
struct Generated {
    sd_image_generation_result_t* r = nullptr;
    bool run(sd_ctx_t* ctx, const sd_img_gen_params_t* g) { return generate_image(ctx, g, &r); }
    int count() const { return r ? r->count : 0; }
    Pix at(int i) const { return pix_of(r->items[i].image); }
    bool cancelled() const { return false; }
    ~Generated() { free_sd_image_generation_result(r); }
};
#else
struct Generated {
    sd_image_t* images = nullptr;
    int n              = 0;
    bool run(sd_ctx_t* ctx, const sd_img_gen_params_t* g) { return generate_image(ctx, g, &images, &n); }
    int count() const { return images ? n : 0; }
    Pix at(int i) const { return pix_of(images[i]); }
    bool cancelled() const { return false; }
    ~Generated() { free_sd_images(images, n); }
};
#endif

static void do_generate(json cmd) {
    const std::string id = jstr(cmd, "id");
    auto t0              = std::chrono::steady_clock::now();
    if (g_ctx == nullptr) {
        emit({{"event", "error"}, {"id", id}, {"scope", "generate"}, {"message", "no model loaded"}});
        return;
    }

    std::string prompt   = jstr(cmd, "prompt");
    std::string negative = jstr(cmd, "negative_prompt");
    std::string out_dir  = jstr(cmd, "out_dir");
    std::string stem     = jstr(cmd, "file_stem", "image");
    std::string format   = jstr(cmd, "format", "png");
    std::string metadata = jstr(cmd, "metadata");
    int jpeg_quality     = std::clamp(jint(cmd, "jpeg_quality", 95), 1, 100);

    const int width       = std::max(64, jint(cmd, "width", 1024));
    const int height      = std::max(64, jint(cmd, "height", 1024));
    const int clip_skip   = jint(cmd, "clip_skip", -1);
    const int batch_count = std::clamp(jint(cmd, "batch_count", 1), 1, 16);
    const float strength  = jfloat(cmd, "strength", 0.75f);

    int64_t seed = jint64(cmd, "seed", -1);
    if (seed < 0) {
        // Chosen here rather than by the library so the app knows it: a
        // picture worth keeping has to be reproducible from what was saved.
        std::random_device rd;
        std::mt19937_64 gen(((uint64_t)rd() << 32) ^ rd() ^ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
        seed = (int64_t)(gen() % 4294967295ull);
    }

    const int steps           = std::clamp(jint(cmd, "steps", 20), 1, 200);
    const float cfg_scale     = jfloat(cmd, "cfg_scale", 7.0f);
    const float guidance      = jfloat(cmd, "guidance", 3.5f);
    const float flow_shift    = jfloat(cmd, "flow_shift", 0.0f);
    const float eta           = jfloat(cmd, "eta", -1.0f);
    const std::string sampler   = jstr(cmd, "sampler");
    const std::string scheduler = jstr(cmd, "scheduler");
    const bool vae_tiling     = jbool(cmd, "vae_tiling", false);

    // Sampling acceleration: denoiser work reused between steps whose input
    // barely moved (EasyCache for DiT models, UCache for UNet, Spectrum for
    // either). Off unless asked for; the threshold is the engine's own when
    // not given.
    std::string cache_mode    = jstr(cmd, "cache_mode");
    if (cache_mode == "off")
        cache_mode.clear();
    const float cache_threshold = jfloat(cmd, "cache_threshold", -1.0f);

    // Optional img2img source and reference pictures (editing models).
    std::string err;
    std::vector<Loaded> refs;
    Loaded init;
    bool has_init         = false;
    std::string init_path = jstr(cmd, "init_image");
    if (!init_path.empty()) {
        if (!load_image(init_path, init, 3, err)) {
            emit({{"event", "error"}, {"id", id}, {"scope", "generate"}, {"message", err}});
            return;
        }
        has_init = true;
    }
    if (cmd.contains("ref_images") && cmd["ref_images"].is_array()) {
        for (const auto& r : cmd["ref_images"]) {
            if (!r.is_string())
                continue;
            Loaded im;
            if (!load_image(r.get<std::string>(), im, 0, err)) {
                emit({{"event", "error"}, {"id", id}, {"scope", "generate"}, {"message", err}});
                return;
            }
            refs.push_back(std::move(im));
        }
    }

    // Live preview of the denoised estimate, every `preview_interval` steps.
    std::string preview        = jstr(cmd, "preview", "proj");
    const int preview_interval = std::max(1, jint(cmd, "preview_interval", 1));

#if CHATY_SD_OPT_API
    sd_img_gen_opt_t* g = sd_img_gen_opt_create();
    if (g == nullptr) {
        emit({{"event", "error"}, {"id", id}, {"scope", "generate"}, {"message", "out of memory"}});
        return;
    }
    struct OptGuard {
        sd_img_gen_opt_t* o;
        ~OptGuard() { sd_opt_free(o); }
    } guard{g};
    check_opt(sd_opt_set_string(g, "prompt", prompt.c_str()), "prompt");
    check_opt(sd_opt_set_string(g, "negative_prompt", negative.c_str()), "negative_prompt");
    check_opt(sd_opt_set_int(g, "width", width), "width");
    check_opt(sd_opt_set_int(g, "height", height), "height");
    if (clip_skip > 0)
        check_opt(sd_opt_set_int(g, "clip_skip", clip_skip), "clip_skip");
    check_opt(sd_opt_set_int(g, "batch_count", batch_count), "batch_count");
    check_opt(sd_opt_set_float(g, "strength", std::clamp(strength, 0.0f, 1.0f)), "strength");
    check_opt(sd_opt_set_int(g, "seed", seed), "seed");
    check_opt(sd_opt_set_int(g, "sample.steps", steps), "sample.steps");
    check_opt(sd_opt_set_float(g, "sample.guidance.txt_cfg", cfg_scale), "sample.guidance.txt_cfg");
    check_opt(sd_opt_set_float(g, "sample.guidance.distilled_guidance", guidance), "sample.guidance.distilled_guidance");
    if (flow_shift > 0.0f)
        check_opt(sd_opt_set_float(g, "sample.flow_shift", flow_shift), "sample.flow_shift");
    if (eta >= 0.0f)
        check_opt(sd_opt_set_float(g, "sample.eta", eta), "sample.eta");
    // Unset — or a name the engine does not know, which its setter refuses —
    // is the engine's choice for the model, as before.
    if (!sampler.empty())
        check_opt(sd_opt_set_string(g, "sample.method", sampler.c_str()), "sample.method");
    if (!scheduler.empty())
        check_opt(sd_opt_set_string(g, "sample.scheduler", scheduler.c_str()), "sample.scheduler");
    if (vae_tiling)
        check_opt(sd_opt_set_bool(g, "vae_tiling.enabled", true), "vae_tiling.enabled");
    if (!cache_mode.empty()) {
        // The app's names, in the engine's spelling ("cache-dit" is
        // "cache_dit" there).
        std::string mode = cache_mode;
        std::replace(mode.begin(), mode.end(), '-', '_');
        check_opt(sd_opt_set_string(g, "cache.mode", mode.c_str()), "cache.mode");
        if (cache_threshold > 0.0f) {
            if (mode == "easycache" || mode == "ucache")
                check_opt(sd_opt_set_float(g, "cache.reuse_threshold", cache_threshold), "cache.reuse_threshold");
            else if (mode == "dbcache" || mode == "cache_dit")
                check_opt(sd_opt_set_float(g, "cache.residual_diff_threshold", cache_threshold), "cache.residual_diff_threshold");
        }
    }
    // The options keep their own reference to each picture once set.
    if (has_init) {
        sd_tensor_t* t = tensor_of(init.pix);
        if (t == nullptr) {
            emit({{"event", "error"}, {"id", id}, {"scope", "generate"}, {"message", engine_error()}});
            return;
        }
        check_opt(sd_opt_set_tensor(g, "init_image", t), "init_image");
        sd_tensor_free(t);
    }
    if (!refs.empty()) {
        std::vector<sd_tensor_t*> ts;
        for (const auto& r : refs) {
            sd_tensor_t* t = tensor_of(r.pix);
            if (t != nullptr)
                ts.push_back(t);
        }
        std::vector<const sd_tensor_t*> views(ts.begin(), ts.end());
        check_opt(sd_opt_set_tensors(g, "ref_images", views.data(), views.size()), "ref_images");
        for (auto* t : ts)
            sd_tensor_free(t);
    }
    // A mode the engine does not know keeps the projection preview.
    if (preview == "none") {
        sd_set_preview_callback(nullptr, "none", 1, false, false, nullptr);
    } else if (sd_set_preview_callback(on_preview, preview.c_str(), preview_interval, true, false, nullptr) != SD_STATUS_OK) {
        sd_set_preview_callback(on_preview, "proj", preview_interval, true, false, nullptr);
    }
#else
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    const sd_img_gen_params_t* g = &gp;
    gp.prompt          = prompt.c_str();
    gp.negative_prompt = negative.c_str();
    gp.width           = width;
    gp.height          = height;
    gp.clip_skip       = clip_skip;
    gp.batch_count     = batch_count;
    gp.strength        = strength;
    gp.seed            = seed;

    sd_sample_params_t& sp = gp.sample_params;
    sp.sample_steps        = steps;
    sp.guidance.txt_cfg    = cfg_scale;
    sp.guidance.distilled_guidance = guidance;
    if (flow_shift > 0.0f)
        sp.flow_shift = flow_shift;
    if (eta >= 0.0f)
        sp.eta = eta;
    sp.sample_method = sampler.empty() ? SAMPLE_METHOD_COUNT : str_to_sample_method(sampler.c_str());
    if (sp.sample_method == SAMPLE_METHOD_COUNT)
        sp.sample_method = sd_get_default_sample_method(g_ctx);
    sp.scheduler = scheduler.empty() ? SCHEDULER_COUNT : str_to_scheduler(scheduler.c_str());
    if (sp.scheduler == SCHEDULER_COUNT)
        sp.scheduler = sd_get_default_scheduler(g_ctx, sp.sample_method);

    gp.vae_tiling_params = {vae_tiling, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};

    if (!cache_mode.empty()) {
        static const std::pair<const char*, sd_cache_mode_t> modes[] = {
            {"easycache", SD_CACHE_EASYCACHE}, {"ucache", SD_CACHE_UCACHE},         {"dbcache", SD_CACHE_DBCACHE},
            {"taylorseer", SD_CACHE_TAYLORSEER}, {"cache-dit", SD_CACHE_CACHE_DIT}, {"spectrum", SD_CACHE_SPECTRUM},
        };
        for (const auto& m : modes) {
            if (cache_mode == m.first)
                gp.cache.mode = m.second;
        }
        if (cache_threshold > 0.0f) {
            if (gp.cache.mode == SD_CACHE_EASYCACHE || gp.cache.mode == SD_CACHE_UCACHE)
                gp.cache.reuse_threshold = cache_threshold;
            else if (gp.cache.mode == SD_CACHE_DBCACHE || gp.cache.mode == SD_CACHE_CACHE_DIT)
                gp.cache.residual_diff_threshold = cache_threshold;
        }
    }

    std::vector<sd_image_t> ref_images;
    if (has_init)
        gp.init_image = image_of(init.pix);
    for (const auto& r : refs)
        ref_images.push_back(image_of(r.pix));
    if (!ref_images.empty()) {
        gp.ref_images       = ref_images.data();
        gp.ref_images_count = (int)ref_images.size();
    }

    enum preview_t pmode = preview == "none" ? PREVIEW_NONE : str_to_preview(preview.c_str());
    if (pmode == PREVIEW_COUNT)
        pmode = PREVIEW_PROJ;
    sd_set_preview_callback(pmode == PREVIEW_NONE ? nullptr : on_preview, pmode, preview_interval, true, false, nullptr);
#endif

    g_cancel_requested = false;
    begin_job(Job::Generate, id);
    {
        std::lock_guard<std::mutex> lk(g_state_mu);
        g_batch_count = batch_count;
        g_cur_seed    = seed;
        emit_stage_locked();
    }

    Generated images;
    bool ok             = images.run(g_ctx, g);
    const int count     = images.count();
    std::string gen_err = take_last_error();
#if CHATY_SD_OPT_API
    if (!ok && gen_err.empty())
        gen_err = engine_error();
#endif
    end_job();

    if (!ok || count <= 0) {
        // A cancel ends the call as a failure; it is not one to report.
        if (g_cancel_requested || images.cancelled()) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            emit({{"event", "done"}, {"id", id}, {"count", 0}, {"cancelled", true}, {"elapsed_ms", ms}});
        } else {
            emit({{"event", "error"}, {"id", id}, {"scope", "generate"}, {"message", gen_err.empty() ? "generation failed" : gen_err}});
        }
        return;
    }

    std::error_code ec;
    fs::path dir = fs_path(out_dir.empty() ? "." : out_dir);
    fs::create_directories(dir, ec);
    const bool jpeg = format == "jpg" || format == "jpeg";
    int saved       = 0;
    for (int i = 0; i < count; i++) {
        const Pix im = images.at(i);
        if (im.data == nullptr)
            continue;
        std::vector<unsigned char> bytes;
        bool enc_ok;
        if (jpeg) {
            enc_ok = stbi_write_jpg_to_func(to_vec, &bytes, (int)im.width, (int)im.height, (int)im.channel, im.data, jpeg_quality) != 0;
        } else {
            enc_ok = stbi_write_png_to_func(to_vec, &bytes, (int)im.width, (int)im.height, (int)im.channel, im.data,
                                            (int)(im.width * im.channel)) != 0;
            if (enc_ok) {
                std::string meta = metadata;
                if (!meta.empty())
                    meta += ", Seed: " + std::to_string(seed + i);
                png_add_text(bytes, "parameters", meta);
            }
        }
        std::string name = count > 1 ? stem + "-" + std::to_string(i + 1) : stem;
        fs::path path    = unique_path(dir, name, jpeg ? ".jpg" : ".png");
        if (!enc_ok || !write_file(path, bytes)) {
            emit({{"event", "error"}, {"id", id}, {"scope", "save"}, {"message", "cannot write " + path_utf8(path)}});
            return;
        }
        emit({{"event", "image"},
              {"id", id},
              {"index", i},
              {"path", path_utf8(path)},
              {"width", im.width},
              {"height", im.height},
              {"channels", im.channel},
              {"seed", seed + i}});
        saved++;
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    // "after_current" returns the pictures finished so far — fewer than asked.
    emit({{"event", "done"}, {"id", id}, {"count", saved}, {"cancelled", saved < batch_count}, {"elapsed_ms", ms}});
}

static void run_job(void (*fn)(json), json cmd) {
    if (g_worker.joinable())
        g_worker.join();
    g_busy = true;
    g_worker = std::thread([fn, cmd]() {
        try {
            fn(cmd);
        } catch (const std::exception& e) {
            end_job();
            emit({{"event", "error"}, {"id", jstr(cmd, "id")}, {"scope", jstr(cmd, "cmd")}, {"message", e.what()}});
        }
        g_busy = false;
    });
}

static bool cpu_ok(std::string& why) {
#if defined(CHATY_SD_NEEDS_AVX2)
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    int max_leaf = r[0];
    bool avx2    = false;
    if (max_leaf >= 7) {
        __cpuidex(r, 7, 0);
        avx2 = (r[1] & (1 << 5)) != 0;
    }
    __cpuid(r, 1);
    bool fma = (r[2] & (1 << 12)) != 0;
    bool osxsave = (r[2] & (1 << 27)) != 0;
    bool ok = avx2 && fma && osxsave;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    bool ok = true;
#endif
    if (!ok) {
        why = "this CPU lacks AVX2/FMA, which the image engine is built for";
        return false;
    }
#endif
    (void)why;
    return true;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        printf("chaty-sd %s (stable-diffusion.cpp %s %s)\n", CHATY_SD_PROTOCOL, sd_version(), sd_commit());
        return 0;
    }
    setup_io();

    std::string why;
    if (!cpu_ok(why)) {
        emit({{"event", "fatal"}, {"message", why}});
        return 3;
    }

    sd_set_log_callback(on_log, nullptr);
    sd_set_progress_callback(on_progress, nullptr);

    json devices = json::array();
    {
        size_t need = sd_list_devices(nullptr, 0);
        std::string buf(need + 1, '\0');
        sd_list_devices(buf.data(), buf.size());
        buf.resize(strlen(buf.c_str()));
        size_t start = 0;
        while (start < buf.size()) {
            size_t end       = buf.find('\n', start);
            std::string line = buf.substr(start, end == std::string::npos ? std::string::npos : end - start);
            size_t tab       = line.find('\t');
            if (!line.empty())
                devices.push_back({{"name", line.substr(0, tab)}, {"description", tab == std::string::npos ? "" : line.substr(tab + 1)}});
            if (end == std::string::npos)
                break;
            start = end + 1;
        }
    }
    emit({{"event", "ready"},
          {"protocol", CHATY_SD_PROTOCOL},
          {"version", sd_version()},
          {"commit", sd_commit()},
          {"system", sd_get_system_info()},
          {"devices", devices}});

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (trim(line).empty())
            continue;
        json cmd;
        try {
            cmd = json::parse(line);
        } catch (const std::exception& e) {
            emit({{"event", "error"}, {"scope", "protocol"}, {"message", std::string("bad command: ") + e.what()}});
            continue;
        }
        const std::string c = jstr(cmd, "cmd");
        if (c == "ping") {
            emit({{"event", "pong"}, {"busy", g_busy.load()}});
        } else if (c == "cancel") {
            g_cancel_requested = true;
            if (g_ctx != nullptr)
                sd_cancel_generation(g_ctx, jstr(cmd, "mode") == "after_current" ? SD_CANCEL_NEW_LATENTS : SD_CANCEL_ALL);
        } else if (c == "quit") {
            fflush(g_out);
            std::_Exit(0);
        } else if (g_busy) {
            emit({{"event", "error"}, {"id", jstr(cmd, "id")}, {"scope", c}, {"message", "busy"}});
        } else if (c == "load") {
            run_job(do_load, cmd);
        } else if (c == "generate") {
            run_job(do_generate, cmd);
        } else if (c == "unload") {
            if (g_worker.joinable())
                g_worker.join();
            if (g_ctx != nullptr) {
                ctx_free(g_ctx);
                g_ctx = nullptr;
            }
            emit({{"event", "unloaded"}});
        } else {
            emit({{"event", "error"}, {"scope", "protocol"}, {"message", "unknown command: " + c}});
        }
    }
    // The app is gone. Whatever is running dies with the process — there is
    // nobody left to receive it.
    fflush(g_out);
    std::_Exit(0);
}
