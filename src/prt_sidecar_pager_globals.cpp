// Phase 28BM: PRT sidecar pager global variables
// Phase 28BQ: added decoder + application counters
// Phase 28BR-A: add decode-once cached residual buffer
// Strong symbols defined here (not inline) — linked into libllama.so

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

#include "prt_sidecar_pager.h"
#include "prt_trit_decode.h"
#include <unordered_map>
#include <cstring>
#include <vector>

// ── Decode-once cache for residual buffers ─────────────────────────────────

struct prt_decode_cache_entry {
    float * decoded_buf = nullptr;
    size_t decoded_size = 0;       // bytes
    size_t rows = 0, cols = 0;
    size_t block_rows = 0, block_cols = 0;
    size_t n_scales = 0;
    size_t checksum = 0;
    bool valid = false;
};

struct prt_decode_cache {
    std::map<std::string, prt_decode_cache_entry> entries;
    size_t cache_misses = 0;
    size_t cache_hits = 0;
    size_t total_decoded_bytes = 0;
    size_t cache_frees = 0;
};

static prt_decode_cache g_prt_decode_cache;

// SidecarLoad — duplicated from prt_shadow.h to avoid circular include.
struct SidecarLoad {
    int layer;
    std::string path;
    float * data;
    size_t size;
};

// Global variable definitions (strong symbols)
std::unordered_map<int, SidecarLoad> g_sidecars;
bool g_sidecars_loaded = false;
prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;

// Phase 28BQ: guarded residual application
bool g_prt_sidecar_apply_enabled = false;  // OFF by default
int  g_prt_sidecar_apply_layer = -1;        // -1 = all layers
std::string g_prt_sidecar_apply_family;    // empty = all families

// Phase 28BR-A: prt_decode_cached — decode once, cache, reuse on repeated hits
float* prt_decode_cached(const char* raw_view, size_t raw_size,
                         int layer, const char* family,
                         prt_trit_decoder* decoder) {
    std::string key = std::to_string(layer) + ":" + family;

    auto it = g_prt_decode_cache.entries.find(key);
    if (it != g_prt_decode_cache.entries.end() && it->second.valid) {
        g_prt_decode_cache.cache_hits++;
        return it->second.decoded_buf;
    }

    // Cache miss — decode
    g_prt_decode_cache.cache_misses++;

    if (raw_size < 32) return nullptr;

    const uint8_t* hdr = reinterpret_cast<const uint8_t*>(raw_view);
    uint32_t rows = *(const uint32_t*)(hdr + 8);
    uint32_t cols = *(const uint32_t*)(hdr + 12);
    uint16_t block_rows = *(const uint16_t*)(hdr + 16);
    uint16_t block_cols = *(const uint16_t*)(hdr + 18);
    uint16_t n_scales = *(const uint16_t*)(hdr + 20);
    uint32_t scale_offset = *(const uint32_t*)(hdr + 26);

    if (rows == 0 || cols == 0 || n_scales == 0) return nullptr;

    std::vector<float> scales(n_scales);
    if (scale_offset > 0 && scale_offset < raw_size) {
        memcpy(scales.data(), hdr + scale_offset, n_scales * sizeof(float));
    }

    prt_decoded_view dv = decoder->decode_bytes(
        reinterpret_cast<const uint8_t*>(raw_view), raw_size,
        rows, cols, block_rows, block_cols,
        n_scales, scales.data()
    );

    if (dv.is_null || dv.data == nullptr) return nullptr;

    // Allocate owned buffer and COPY decoded data into it
    size_t n_floats = (size_t)rows * cols;
    float* owned = new float[n_floats];
    memcpy(owned, dv.data, n_floats * sizeof(float));
    delete[] dv.data;  // free decoder's buffer

    // Store in cache
    prt_decode_cache_entry entry;
    entry.decoded_buf = owned;
    entry.decoded_size = n_floats * sizeof(float);
    entry.rows = rows;
    entry.cols = cols;
    entry.block_rows = block_rows;
    entry.block_cols = block_cols;
    entry.n_scales = n_scales;
    entry.valid = true;
    g_prt_decode_cache.entries[key] = entry;
    g_prt_decode_cache.total_decoded_bytes += entry.decoded_size;

    return owned;
}

// Phase 28BR-A: shutdown — free all cached decoded buffers exactly once
void prt_decode_cache_shutdown() {
    for (auto& [key, entry] : g_prt_decode_cache.entries) {
        if (entry.valid && entry.decoded_buf != nullptr) {
            delete[] entry.decoded_buf;
            entry.decoded_buf = nullptr;
            entry.valid = false;
            g_prt_decode_cache.cache_frees++;
        }
    }
    g_prt_decode_cache.entries.clear();
}

// Phase 28BQ: application counters (defined as strong symbol here)
struct prt_apply_counters {
    size_t decoded_views = 0;
    size_t application_attempts = 0;
    size_t application_successes = 0;
    size_t application_skipped_wrong_target = 0;
    size_t application_failures = 0;
    size_t decode_errors = 0;
    bool sidecar_math_influenced_output = false;
    // Phase 28BR-A: decode-once cache counters
    size_t decode_cache_misses = 0;
    size_t decode_cache_hits = 0;
    size_t decode_cache_entries = 0;
    size_t decoded_bytes_total = 0;
    bool raw_bytes_cast_to_float = false;
};

static struct {
    size_t decoded_views = 0;
    size_t application_attempts = 0;
    size_t application_successes = 0;
    size_t application_skipped_wrong_target = 0;
    size_t application_failures = 0;
    size_t decode_errors = 0;
    bool sidecar_math_influenced_output = false;
    // Phase 28BR-A: decode-once cache counters
    size_t decode_cache_misses = 0;
    size_t decode_cache_hits = 0;
    size_t decode_cache_entries = 0;
    size_t decoded_bytes_total = 0;
    bool raw_bytes_cast_to_float = false;
} g_prt_apply_stats;

// Defined in header — implemented here (strong symbol for use in llama-graph.cpp)
prt_apply_counters prt_get_apply_stats() {
    prt_apply_counters r;
    r.decoded_views = g_prt_apply_stats.decoded_views;
    r.application_attempts = g_prt_apply_stats.application_attempts;
    r.application_successes = g_prt_apply_stats.application_successes;
    r.application_skipped_wrong_target = g_prt_apply_stats.application_skipped_wrong_target;
    r.application_failures = g_prt_apply_stats.application_failures;
    r.decode_errors = g_prt_apply_stats.decode_errors;
    r.sidecar_math_influenced_output = g_prt_apply_stats.sidecar_math_influenced_output;
    // Phase 28BR-A: cache counters from g_prt_decode_cache
    r.decode_cache_misses = g_prt_decode_cache.cache_misses;
    r.decode_cache_hits = g_prt_decode_cache.cache_hits;
    r.decode_cache_entries = g_prt_decode_cache.entries.size();
    r.decoded_bytes_total = g_prt_decode_cache.total_decoded_bytes;
    r.raw_bytes_cast_to_float = g_prt_apply_stats.raw_bytes_cast_to_float;
    return r;
}

// Defined in header — implemented here (strong symbol for use in llama-graph.cpp)
prt_decoded_view prt_shadow_apply(int layer_idx, const std::string& tensor_family, const prt_residual_view& raw_view) {
    prt_decoded_view dec;
    dec.is_null = true;

    if (g_prt_sidecar_apply_layer >= 0 && layer_idx != g_prt_sidecar_apply_layer) {
        g_prt_apply_stats.application_skipped_wrong_target++;
        dec.reason = "skipped_wrong_layer";
        return dec;
    }
    if (!g_prt_sidecar_apply_family.empty() && tensor_family != g_prt_sidecar_apply_family) {
        g_prt_apply_stats.application_skipped_wrong_target++;
        dec.reason = "skipped_wrong_family";
        return dec;
    }

    if (raw_view.is_null || raw_view.data == nullptr || raw_view.size < 32) {
        g_prt_apply_stats.application_failures++;
        dec.reason = "null_raw_view";
        return dec;
    }

    // Phase 28BR-A: use decode-once cache — decode once, reuse on repeated hits
    prt_trit_decoder decoder;
    float* cached = prt_decode_cached(
        reinterpret_cast<const char*>(raw_view.data), raw_view.size,
        layer_idx, tensor_family.c_str(), &decoder
    );

    g_prt_apply_stats.decoded_views++;
    g_prt_apply_stats.application_attempts++;

    if (cached != nullptr) {
        g_prt_apply_stats.application_successes++;
        g_prt_apply_stats.sidecar_math_influenced_output = false;
        // Shadow mode: do NOT feed into model compute. Buffer stays in cache until shutdown.
        // DO NOT deallocate — cache owns it.
        dec.is_null = false;
        dec.reason = "shadow_compute_cached";
        dec.format = prt_data_format::DECODED_F32;
        dec.data = cached;  // point to cached buffer (NOT owned by caller)
        dec.rows = decoder.get_stats().files_decoded > 0 ? g_prt_decode_cache.entries[
            std::to_string(layer_idx) + ":" + tensor_family].rows : 0;
        dec.cols = decoder.get_stats().files_decoded > 0 ? g_prt_decode_cache.entries[
            std::to_string(layer_idx) + ":" + tensor_family].cols : 0;
    } else {
        g_prt_apply_stats.decode_errors++;
        g_prt_apply_stats.application_failures++;
        dec.reason = "decode_failed";
    }

    return dec;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL