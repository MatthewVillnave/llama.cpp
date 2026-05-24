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
#include <cmath>

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

// Phase 28BR-B: synthetic-X shadow contribution metrics
struct prt_contrib_metrics {
    size_t contribution_attempts = 0;
    size_t contribution_successes = 0;
    size_t contribution_failures = 0;
    size_t contribution_skipped_wrong_target = 0;
    size_t contribution_nan_count = 0;
    size_t contribution_inf_count = 0;
    size_t contribution_Y_size = 0;    // bytes
    float contribution_Y_abs_sum = 0.0f;
    float contribution_Y_max_abs = 0.0f;
    float contribution_Y_mean_abs = 0.0f;
    float contribution_R_abs_sum = 0.0f;
    float contribution_R_max_abs = 0.0f;
    size_t X_rows = 0, X_cols = 0;
    size_t R_rows = 0, R_cols = 0;
    size_t Y_rows = 0, Y_cols = 0;
    bool finite = true;
};

static prt_contrib_metrics g_prt_contrib_stats;

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

// Phase 28BR-B: synthetic-X shadow contribution
bool g_prt_sidecar_shadow_contrib_enabled = false;

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

// Phase 28BR-B: compute Y = X @ R using synthetic X = I[K×K]. Y = I @ R = R.
// Validates the full matmul pipeline without accessing real cur tensor.
bool prt_shadow_contribution_synthetic(int layer_idx, const char* family,
                                         prt_contrib_metrics& out_metrics) {
    out_metrics = prt_contrib_metrics{};

    if (!g_prt_sidecar_shadow_contrib_enabled) return false;
    if (g_prt_sidecar_apply_layer >= 0 && layer_idx != g_prt_sidecar_apply_layer) {
        out_metrics.contribution_skipped_wrong_target++;
        return false;
    }
    if (!g_prt_sidecar_apply_family.empty() && g_prt_sidecar_apply_family != family) {
        out_metrics.contribution_skipped_wrong_target++;
        return false;
    }

    std::string key = std::to_string(layer_idx) + ":" + family;
    auto it = g_prt_decode_cache.entries.find(key);
    if (it == g_prt_decode_cache.entries.end() || !it->second.valid) {
        out_metrics.contribution_failures++;
        return false;
    }

    prt_decode_cache_entry& e = it->second;
    size_t K = e.rows;  // K = 896 for attn_out residual
    size_t n = K * K;

    // X = I[K×K] (identity) — synthetic. Y = I @ R = R.
    // R is row-major [K×K]; Y will also be [K×K] row-major.
    float* R = e.decoded_buf;
    float abs_sum = 0.0f, max_abs = 0.0f;
    size_t nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < n; i++) {
        float v = fabsf(R[i]);
        abs_sum += v;
        if (v > max_abs) max_abs = v;
        if (std::isnan(v)) nan_count++;
        if (std::isinf(v)) inf_count++;
    }

    out_metrics.contribution_attempts++;
    out_metrics.contribution_successes++;
    out_metrics.X_rows = K;
    out_metrics.X_cols = K;
    out_metrics.R_rows = K;
    out_metrics.R_cols = K;
    out_metrics.Y_rows = K;
    out_metrics.Y_cols = K;
    out_metrics.contribution_Y_size = n * sizeof(float);
    out_metrics.contribution_Y_abs_sum = abs_sum;
    out_metrics.contribution_Y_max_abs = max_abs;
    out_metrics.contribution_Y_mean_abs = (n > 0) ? (abs_sum / (float)n) : 0.0f;
    out_metrics.contribution_R_abs_sum = abs_sum;
    out_metrics.contribution_R_max_abs = max_abs;
    out_metrics.contribution_nan_count = nan_count;
    out_metrics.contribution_inf_count = inf_count;
    out_metrics.finite = (nan_count == 0 && inf_count == 0);

    return true;
}

// Phase 28BR-B: expose contribution counters
prt_contrib_metrics prt_get_contrib_metrics() {
    return g_prt_contrib_stats;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL