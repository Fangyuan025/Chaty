// chaty-audio — audio.cpp as a sidecar process for Chaty's music studio.
//
// Protocol: one JSON object per line. Commands arrive on stdin, events leave
// on stdout. The engine's own chatter goes to stderr: stdout is re-pointed at
// stderr at startup and the protocol writes to a private duplicate of the
// original, so a stray printf anywhere in the engine can never corrupt a line
// the app parses.
//
// Commands
//   {"cmd":"load", "model_path", "family", "backend", "threads",
//    "session_options":{…}}                      → loaded | error
//   {"cmd":"generate", "id", "text", "audio_path", "seed", "out_path", "options":{…}}
//                                                → stage* progress* info* audio (done | error)
//   {"cmd":"ping"}                               → pong
//   {"cmd":"quit"}                               → the process exits
// End of stdin (the app went away) exits too.
//
// Nothing here knows one model family from another. `model_path` is a GGUF or
// a package folder, `session_options` and `options` are audio.cpp's own
// session and request options, passed through as they are; which ones a
// family takes is reported back on `loaded` — the engine's own list, with
// each option's type, default and range — for the app to offer.
//
// There is no cancel: audio.cpp runs a song to its end and offers no way in.
// The app stops one by ending this process, and starts another for the next.
//
// Progress: audio.cpp reports nothing while it works. Chaty patches it
// (patches/chaty-progress.patch) to write `chaty.progress <stage> <done>/<total>`
// and `chaty.info <key> <value>` lines to its log as a song is made; the log is
// std::cout, captured below, and those lines become events.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include "audiocpp.h"
#include "cJSON.h"
#include "engine/framework/debug/trace.h"
#include "ggml-backend.h"

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

namespace fs = std::filesystem;

static const char* CHATY_AUDIO_PROTOCOL = "1";

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

/// An owned cJSON tree.
struct Json {
    cJSON* p;
    explicit Json(cJSON* v = cJSON_CreateObject()) : p(v) {}
    ~Json() { cJSON_Delete(p); }
    Json(const Json&) = delete;
    Json& operator=(const Json&) = delete;
    Json& str(const char* k, const std::string& v) {
        cJSON_AddStringToObject(p, k, v.c_str());
        return *this;
    }
    Json& num(const char* k, double v) {
        cJSON_AddNumberToObject(p, k, v);
        return *this;
    }
    Json& boolean(const char* k, bool v) {
        cJSON_AddBoolToObject(p, k, v);
        return *this;
    }
    /// Takes ownership of `child`.
    Json& put(const char* k, cJSON* child) {
        cJSON_AddItemToObject(p, k, child);
        return *this;
    }
};

static std::string jstr(const cJSON* o, const char* k, const std::string& def = "") {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsString(v) && v->valuestring != nullptr ? std::string(v->valuestring) : def;
}

static double jnum(const cJSON* o, const char* k, double def) {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(v) ? v->valuedouble : def;
}

/// Each `key: value` of the object `k` as strings (numbers written plainly,
/// booleans as true/false); anything else is skipped.
template <typename F>
static void each_option(const cJSON* o, const char* k, F&& f) {
    const cJSON* m = cJSON_GetObjectItemCaseSensitive(o, k);
    if (!cJSON_IsObject(m))
        return;
    for (const cJSON* v = m->child; v != nullptr; v = v->next) {
        if (v->string == nullptr)
            continue;
        std::string value;
        if (cJSON_IsString(v) && v->valuestring != nullptr) {
            value = v->valuestring;
        } else if (cJSON_IsNumber(v)) {
            char buf[64];
            double d = v->valuedouble;
            if (d == (double)(long long)d && d > -9e15 && d < 9e15)
                snprintf(buf, sizeof buf, "%lld", (long long)d);
            else
                snprintf(buf, sizeof buf, "%.9g", d);
            value = buf;
        } else if (cJSON_IsBool(v)) {
            value = cJSON_IsTrue(v) ? "true" : "false";
        } else {
            continue;
        }
        f(std::string(v->string), value);
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static FILE* g_out = nullptr;
static std::mutex g_out_mu;

/// Replace bytes that are not valid UTF-8 — a path in a legacy code page, a
/// broken prompt — so the app's JSON reader never refuses a line.
static std::string valid_utf8(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        bool ok = n > 0 && i + n <= s.size();
        for (size_t k = 1; ok && k < n; ++k)
            ok = ((unsigned char)s[i + k] >> 6) == 0x2;
        if (ok) {
            out.append(s, i, n);
            i += n;
        } else {
            out += "\xEF\xBF\xBD";
            i += 1;
        }
    }
    return out;
}

static void emit(const Json& j) {
    char* text = cJSON_PrintUnformatted(j.p);
    if (text == nullptr)
        return;
    std::string s = valid_utf8(text);
    cJSON_free(text);
    std::lock_guard<std::mutex> lk(g_out_mu);
    fwrite(s.data(), 1, s.size(), g_out);
    fputc('\n', g_out);
    fflush(g_out);
}

static void emit_error(const std::string& id, const std::string& scope, const std::string& message) {
    Json e;
    e.str("event", "error").str("scope", scope).str("message", message);
    if (!id.empty())
        e.str("id", id);
    emit(e);
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
// The engine's log, read for progress
// ---------------------------------------------------------------------------

/// The song being made, as its log lines report it.
struct Watch {
    std::mutex mu;
    std::string id;     // empty = no song running
    std::string stage;  // the stage last reported
};
static Watch g_watch;

static void emit_stage(const std::string& id, const std::string& name) {
    Json e;
    e.str("event", "stage").str("id", id).str("stage", name);
    emit(e);
}

/// One line of the engine's log: passed on to stderr (for the error report),
/// and read for Chaty's progress lines — `[TIMING ts=…] chaty.progress
/// <stage> <done>/<total>` and `[TIMING ts=…] chaty.info <key> <value>`.
static void on_log_line(const std::string& line) {
    fprintf(stderr, "%s\n", line.c_str());
    const std::string tag = "[TIMING ";
    if (line.compare(0, tag.size(), tag) != 0)
        return;
    size_t close = line.find("] ");
    if (close == std::string::npos)
        return;
    const std::string rest = line.substr(close + 2);
    size_t sp = rest.find(' ');
    if (sp == std::string::npos)
        return;
    const std::string name  = rest.substr(0, sp);
    const std::string value = rest.substr(sp + 1);

    std::lock_guard<std::mutex> lk(g_watch.mu);
    const std::string id = g_watch.id;
    if (id.empty())
        return;
    if (name == "chaty.progress") {
        char stage[32] = {0};
        double done = 0, total = 0;
        if (sscanf(value.c_str(), "%31s %lf/%lf", stage, &done, &total) != 3 || total <= 0)
            return;
        // A stage begins where its first line arrives.
        if (g_watch.stage != stage) {
            g_watch.stage = stage;
            emit_stage(id, stage);
        }
        Json e;
        e.str("event", "progress").str("id", id).str("stage", stage).num("done", done).num("total", total);
        emit(e);
    } else if (name == "chaty.info") {
        size_t s2 = value.find(' ');
        if (s2 == std::string::npos)
            return;
        Json e;
        e.str("event", "info").str("id", id).str("key", value.substr(0, s2)).num("value", std::atof(value.c_str() + s2 + 1));
        emit(e);
    }
}

/// A stream buffer standing in for std::cout's: audio.cpp's logger writes
/// there, a line at a time.
class LogTap : public std::streambuf {
public:
    int_type overflow(int_type ch) override {
        if (ch == traits_type::eof())
            return traits_type::not_eof(ch);
        put((char)ch);
        return ch;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        for (std::streamsize i = 0; i < n; ++i)
            put(s[i]);
        return n;
    }

private:
    void put(char c) {
        std::lock_guard<std::mutex> lk(mu_);
        if (c == '\n') {
            std::string line;
            line.swap(line_);
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            on_log_line(line);
        } else if (line_.size() < 64 * 1024) {
            line_ += c;
        }
    }
    std::mutex mu_;
    std::string line_;
};

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
/// audio.cpp builds its paths from narrow strings, which Windows reads in the
/// process code page. The embedded manifest makes that code page UTF-8 on
/// Windows 10 1903 and later, and then the path passes through untouched. On
/// anything older a folder with a Chinese name would not open, so such a path
/// is handed over as its 8.3 short form, which is plain ASCII whenever the
/// volume keeps short names.
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

/// 16-bit PCM WAV of interleaved float samples.
static bool write_wav(const std::string& utf8, const float* samples, size_t frames, int rate, int channels) {
    std::ofstream f(fs_path(utf8), std::ios::binary | std::ios::trunc);
    if (!f)
        return false;
    const uint64_t data_bytes64 = (uint64_t)frames * (uint64_t)channels * 2;
    if (data_bytes64 > 0xFFFFFFF0ull)
        return false;
    const uint32_t data_bytes = (uint32_t)data_bytes64;
    auto u32 = [&](uint32_t v) {
        char b[4] = {(char)(v & 0xFF), (char)((v >> 8) & 0xFF), (char)((v >> 16) & 0xFF), (char)((v >> 24) & 0xFF)};
        f.write(b, 4);
    };
    auto u16 = [&](uint16_t v) {
        char b[2] = {(char)(v & 0xFF), (char)((v >> 8) & 0xFF)};
        f.write(b, 2);
    };
    f.write("RIFF", 4);
    u32(36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    u32(16);
    u16(1);  // PCM
    u16((uint16_t)channels);
    u32((uint32_t)rate);
    u32((uint32_t)(rate * channels * 2));
    u16((uint16_t)(channels * 2));
    u16(16);
    f.write("data", 4);
    u32(data_bytes);
    std::vector<char> buf;
    buf.reserve(64 * 1024);
    const size_t total = frames * (size_t)channels;
    for (size_t i = 0; i < total; ++i) {
        float s = samples[i];
        if (!(s == s))
            s = 0.0f;  // NaN
        s       = std::max(-1.0f, std::min(1.0f, s));
        int16_t v = (int16_t)std::lrint(s * 32767.0f);
        buf.push_back((char)(v & 0xFF));
        buf.push_back((char)((v >> 8) & 0xFF));
        if (buf.size() >= 64 * 1024) {
            f.write(buf.data(), (std::streamsize)buf.size());
            buf.clear();
        }
    }
    f.write(buf.data(), (std::streamsize)buf.size());
    return (bool)f;
}

/// A WAV's samples, interleaved float — 16/24/32-bit PCM or 32-bit float, as
/// this sidecar and most tools write them. For the edits that start from an
/// earlier song (a repaint, a cover, a variation).
static bool read_wav(const std::string& utf8, std::vector<float>& out, size_t& frames, int& rate, int& channels) {
    std::ifstream f(fs_path(utf8), std::ios::binary);
    if (!f)
        return false;
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u16 = [&](size_t o) { return (uint32_t)b[o] | ((uint32_t)b[o + 1] << 8); };
    auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
    if (b.size() < 12 || memcmp(b.data(), "RIFF", 4) != 0 || memcmp(b.data() + 8, "WAVE", 4) != 0)
        return false;
    int format = 0, bits = 0;
    channels = 0;
    rate     = 0;
    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        const uint32_t size = u32(pos + 4);
        const size_t body   = pos + 8;
        if (memcmp(b.data() + pos, "fmt ", 4) == 0 && body + 16 <= b.size()) {
            format   = (int)u16(body);
            channels = (int)u16(body + 2);
            rate     = (int)u32(body + 4);
            bits     = (int)u16(body + 14);
            if (format == 0xFFFE && size >= 26 && body + 26 <= b.size())
                format = (int)u16(body + 24);  // WAVE_FORMAT_EXTENSIBLE: the sub-format
        } else if (memcmp(b.data() + pos, "data", 4) == 0) {
            if (channels <= 0 || rate <= 0 || bits <= 0)
                return false;
            const size_t bytes = std::min<size_t>(size, b.size() - body);
            const size_t width = (size_t)bits / 8;
            if (width == 0)
                return false;
            const size_t n = bytes / width;
            frames         = n / (size_t)channels;
            out.resize(frames * (size_t)channels);
            const unsigned char* d = b.data() + body;
            for (size_t i = 0; i < out.size(); ++i) {
                const unsigned char* x = d + i * width;
                if (format == 3 && bits == 32) {
                    float v;
                    memcpy(&v, x, 4);
                    out[i] = v;
                } else if (format == 1 && bits == 16) {
                    out[i] = (float)(int16_t)(x[0] | (x[1] << 8)) / 32768.0f;
                } else if (format == 1 && bits == 24) {
                    int32_t v = (int32_t)((uint32_t)x[0] << 8 | (uint32_t)x[1] << 16 | (uint32_t)x[2] << 24) >> 8;
                    out[i] = (float)v / 8388608.0f;
                } else if (format == 1 && bits == 32) {
                    int32_t v;
                    memcpy(&v, x, 4);
                    out[i] = (float)((double)v / 2147483648.0);
                } else {
                    return false;
                }
            }
            return frames > 0;
        }
        pos = body + size + (size & 1);
    }
    return false;
}

static bool write_text(const std::string& utf8, const void* data, size_t n) {
    std::ofstream f(fs_path(utf8), std::ios::binary | std::ios::trunc);
    if (!f)
        return false;
    f.write(static_cast<const char*>(data), (std::streamsize)n);
    return (bool)f;
}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------

static audiocpp_registry* g_registry = nullptr;
static audiocpp_model* g_model       = nullptr;
static audiocpp_session* g_session   = nullptr;
static std::atomic<bool> g_busy{false};
static std::thread g_worker;

static std::string last_error(audiocpp_status st) {
    std::string d = audiocpp_last_error();
    std::string s = audiocpp_status_string(st);
    return d.empty() ? s : d;
}

static void free_engine() {
    audiocpp_session_free(g_session);
    audiocpp_model_free(g_model);
    audiocpp_registry_free(g_registry);
    g_session  = nullptr;
    g_model    = nullptr;
    g_registry = nullptr;
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

/// Every option of one scope, as the engine describes it.
static cJSON* option_list(audiocpp_option_scope scope) {
    cJSON* list = cJSON_CreateArray();
    const size_t n = audiocpp_model_option_count(g_model, scope);
    for (size_t i = 0; i < n; ++i) {
        const char *name = nullptr, *value_name = nullptr, *description = nullptr, *fallback = nullptr, *min_value = nullptr,
                   *max_value = nullptr;
        int required = 0;
        if (audiocpp_model_option(g_model, scope, i, &name, &value_name, &description, &fallback, &min_value, &max_value,
                                  &required) != AUDIOCPP_OK ||
            name == nullptr || name[0] == '\0')
            continue;
        cJSON* o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "name", name);
        cJSON_AddStringToObject(o, "type", value_name ? value_name : "");
        cJSON_AddStringToObject(o, "description", description ? description : "");
        cJSON_AddStringToObject(o, "default", fallback ? fallback : "");
        cJSON_AddStringToObject(o, "min", min_value ? min_value : "");
        cJSON_AddStringToObject(o, "max", max_value ? max_value : "");
        cJSON_AddBoolToObject(o, "required", required != 0);
        cJSON_AddItemToArray(list, o);
    }
    return list;
}

static void do_load(const cJSON* cmd) {
    free_engine();
    const std::string path   = jstr(cmd, "model_path");
    const std::string family = jstr(cmd, "family");
    if (path.empty())
        throw std::runtime_error("no model given");

    audiocpp_status st = audiocpp_registry_create(nullptr, &g_registry);
    if (st != AUDIOCPP_OK)
        throw std::runtime_error(last_error(st));

    audiocpp_model_config config = {family.empty() ? nullptr : family.c_str(), nullptr, nullptr, nullptr};
    st = audiocpp_model_load(g_registry, engine_path(path).c_str(), &config, nullptr, &g_model);
    if (st != AUDIOCPP_OK)
        throw std::runtime_error(last_error(st));
    if (!audiocpp_model_supports(g_model, "gen", "offline"))
        throw std::runtime_error("this model does not generate audio");

    // The session names the files of a package the user picked (a
    // quantization) and carries the engine settings (weight types, memory
    // saving, attention kernels) — whatever the family calls them.
    audiocpp_options* opts = audiocpp_options_create();
    each_option(cmd, "session_options", [&](const std::string& k, const std::string& v) {
        if (!v.empty())
            audiocpp_options_set(opts, k.c_str(), v.c_str());
    });
    const std::string backend = jstr(cmd, "backend", "best");
    int threads               = (int)jnum(cmd, "threads", 0);
    if (threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        threads     = hc == 0 ? 4 : (int)std::max(1u, std::min(hc, 16u));
    }
    audiocpp_backend_config bc = {backend.c_str(), 0, threads};
    st = audiocpp_session_create(g_model, "gen", "offline", &bc, opts, &g_session);
    audiocpp_options_free(opts);
    if (st != AUDIOCPP_OK)
        throw std::runtime_error(last_error(st));

    Json e;
    e.str("event", "loaded")
        .str("family", audiocpp_model_family(g_model))
        .str("description", audiocpp_model_description(g_model))
        .str("backend", backend)
        .num("threads", threads)
        .put("request_options", option_list(AUDIOCPP_OPTION_SCOPE_REQUEST))
        .put("session_options", option_list(AUDIOCPP_OPTION_SCOPE_SESSION));
    emit(e);
}

static void do_generate(const cJSON* cmd) {
    const std::string id = jstr(cmd, "id");
    if (g_session == nullptr)
        throw std::runtime_error("no model is loaded");
    const std::string out_path = jstr(cmd, "out_path");
    if (out_path.empty())
        throw std::runtime_error("no output path given");

    // Negative = a random one, chosen here and reported back.
    int64_t seed = (int64_t)jnum(cmd, "seed", -1);
    if (seed < 0) {
        std::random_device rd;
        seed = (int64_t)((((uint64_t)rd() << 31) ^ (uint64_t)rd()) & 0x7FFFFFFFull);
    }

    audiocpp_request* req = audiocpp_request_create();
    struct Free {
        audiocpp_request* r;
        ~Free() { audiocpp_request_free(r); }
    } free_req{req};
    const std::string text = jstr(cmd, "text");
    if (!text.empty())
        audiocpp_request_set_text(req, text.c_str(), nullptr);
    // An earlier song to work from: repainted, covered, varied, continued.
    const std::string audio_path = jstr(cmd, "audio_path");
    if (!audio_path.empty()) {
        std::vector<float> samples;
        size_t frames = 0;
        int rate = 0, channels = 0;
        if (!read_wav(audio_path, samples, frames, rate, channels))
            throw std::runtime_error("cannot read " + audio_path);
        audiocpp_status as = audiocpp_request_set_audio(req, samples.data(), frames, rate, channels);
        if (as != AUDIOCPP_OK)
            throw std::runtime_error(last_error(as));
    }
    auto opt = [&](const std::string& k, const std::string& v) {
        audiocpp_status os = audiocpp_request_set_option(req, k.c_str(), v.c_str());
        if (os != AUDIOCPP_OK)
            throw std::runtime_error("option " + k + ": " + last_error(os));
    };
    // Empty values are kept: an empty `lyrics` is how a song is asked to be
    // instrumental.
    each_option(cmd, "options", [&](const std::string& k, const std::string& v) { opt(k, v); });
    opt("seed", std::to_string(seed));

    {
        std::lock_guard<std::mutex> lk(g_watch.mu);
        g_watch.id    = id;
        g_watch.stage = "prepare";
    }
    struct Unwatch {
        ~Unwatch() {
            std::lock_guard<std::mutex> lk(g_watch.mu);
            g_watch.id.clear();
        }
    } unwatch;

    const auto t0 = std::chrono::steady_clock::now();
    {
        Json e;
        e.str("event", "stage").str("id", id).str("stage", "prepare").num("seed", (double)seed);
        emit(e);
    }

    audiocpp_result* result = nullptr;
    audiocpp_status st      = audiocpp_session_run(g_session, req, &result);
    struct FreeResult {
        audiocpp_result** r;
        ~FreeResult() { audiocpp_result_free(*r); }
    } free_result{&result};
    if (st != AUDIOCPP_OK)
        throw std::runtime_error(last_error(st));

    const float* samples = nullptr;
    size_t frames        = 0;
    int rate = 0, channels = 0;
    st = audiocpp_result_audio(result, &samples, &frames, &rate, &channels);
    if (st != AUDIOCPP_OK || samples == nullptr || frames == 0 || rate <= 0 || channels <= 0)
        throw std::runtime_error("the engine returned no audio");

    {
        std::lock_guard<std::mutex> lk(g_watch.mu);
        g_watch.stage = "save";
    }
    emit_stage(id, "save");
    if (!write_wav(out_path, samples, frames, rate, channels))
        throw std::runtime_error("cannot write " + out_path);

    // What the song is made of, kept beside it where the family says: the
    // score YuE2 wrote for itself (sheet music a notation program opens, and
    // what a rearrangement follows) and its music tokens (what a later round
    // continues from).
    std::string abc, tokens;
    for (size_t i = 0; i < audiocpp_result_artifact_count(result); ++i) {
        audiocpp_artifact_kind kind;
        const char* aid     = nullptr;
        const void* payload = nullptr;
        size_t bytes        = 0;
        if (audiocpp_result_artifact(result, i, &kind, &aid, &payload, &bytes) != AUDIOCPP_OK || aid == nullptr ||
            payload == nullptr || bytes == 0)
            continue;
        if (std::string(aid) == "score")
            abc.assign(static_cast<const char*>(payload), bytes);
        else if (std::string(aid) == "semantic")
            tokens.assign(static_cast<const char*>(payload), bytes);
    }
    auto beside = [&](const char* ext, const std::string& body) -> std::string {
        if (body.empty())
            return {};
        fs::path p = fs_path(out_path);
        p.replace_extension(ext);
        std::string utf8 = path_utf8(p);
        return write_text(utf8, body.data(), body.size()) ? utf8 : std::string();
    };
    const std::string abc_path    = beside(".abc", abc);
    const std::string tokens_path = beside(".tokens.json", tokens);

    Json a;
    a.str("event", "audio")
        .str("id", id)
        .str("path", out_path)
        .num("sample_rate", rate)
        .num("channels", channels)
        .num("seconds", (double)frames / (double)rate)
        .num("seed", (double)seed);
    if (!abc_path.empty())
        a.str("score_path", abc_path);
    if (!tokens_path.empty())
        a.str("tokens_path", tokens_path);
    emit(a);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    Json d;
    d.str("event", "done").str("id", id).num("elapsed_ms", (double)ms);
    emit(d);
}

static void run_job(void (*fn)(const cJSON*), cJSON* cmd) {
    if (g_worker.joinable())
        g_worker.join();
    g_busy   = true;
    g_worker = std::thread([fn, cmd]() {
        std::string id = jstr(cmd, "id");
        std::string c  = jstr(cmd, "cmd");
        try {
            fn(cmd);
        } catch (const std::exception& e) {
            emit_error(id, c, e.what());
        } catch (...) {
            emit_error(id, c, "unknown error");
        }
        cJSON_Delete(cmd);
        g_busy = false;
    });
}

static bool cpu_ok(std::string& why) {
#if defined(CHATY_AUDIO_NEEDS_AVX2)
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
    bool fma     = (r[2] & (1 << 12)) != 0;
    bool osxsave = (r[2] & (1 << 27)) != 0;
    bool ok      = avx2 && fma && osxsave;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    bool ok = true;
#endif
    if (!ok) {
        why = "this CPU lacks AVX2/FMA, which the music engine is built for";
        return false;
    }
#endif
    (void)why;
    return true;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        printf("chaty-audio %s (audio.cpp %s)\n", CHATY_AUDIO_PROTOCOL, CHATY_AUDIOCPP_REV);
        return 0;
    }
    setup_io();

    std::string why;
    if (!cpu_ok(why)) {
        Json e;
        e.str("event", "fatal").str("message", why);
        emit(e);
        return 3;
    }

    // audio.cpp logs to std::cout; the tap reads it (see LogTap) and passes
    // it on to stderr. The log is what carries the stage timings.
    static LogTap tap;
    std::cout.rdbuf(&tap);
    engine::debug::configure_logging(engine::debug::LoggingConfig{true, std::nullopt});

    cJSON* devices = cJSON_CreateArray();
    if (ggml_backend_reg_count() == 0)
        ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr)
            continue;
        cJSON* d = cJSON_CreateObject();
        cJSON_AddStringToObject(d, "name", ggml_backend_dev_name(dev));
        cJSON_AddStringToObject(d, "description", ggml_backend_dev_description(dev));
        const auto type = ggml_backend_dev_type(dev);
        cJSON_AddStringToObject(d, "type", type == GGML_BACKEND_DEVICE_TYPE_CPU ? "cpu" : "gpu");
        cJSON_AddItemToArray(devices, d);
    }
    {
        Json e;
        e.str("event", "ready")
            .str("protocol", CHATY_AUDIO_PROTOCOL)
            .str("version", audiocpp_build_version())
            .put("devices", devices);
        emit(e);
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty())
            continue;
        cJSON* cmd = cJSON_Parse(line.c_str());
        if (cmd == nullptr || !cJSON_IsObject(cmd)) {
            cJSON_Delete(cmd);
            emit_error("", "protocol", "bad command");
            continue;
        }
        const std::string c = jstr(cmd, "cmd");
        if (c == "ping") {
            Json e;
            e.str("event", "pong").boolean("busy", g_busy.load());
            emit(e);
            cJSON_Delete(cmd);
        } else if (c == "quit") {
            fflush(g_out);
            std::_Exit(0);
        } else if (g_busy) {
            emit_error(jstr(cmd, "id"), c, "busy");
            cJSON_Delete(cmd);
        } else if (c == "load") {
            run_job(do_load, cmd);
        } else if (c == "generate") {
            run_job(do_generate, cmd);
        } else {
            emit_error(jstr(cmd, "id"), "protocol", "unknown command: " + c);
            cJSON_Delete(cmd);
        }
    }
    // The app is gone. Whatever is running dies with the process — there is
    // nobody left to receive it.
    fflush(g_out);
    std::_Exit(0);
}
