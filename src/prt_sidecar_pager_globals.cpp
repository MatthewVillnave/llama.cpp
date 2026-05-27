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
#include <cstdio>
#include <string>

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

#include <cstdlib>
#include <cstdarg>

// Phase 28BR-D: file-based forensic log — avoids stderr capture issues
extern prt_sidecar_pager* g_prt_pager;
extern bool g_prt_pager_enabled;

static void prt_forensic_log(const char* fmt, ...) {
    const char* path = getenv("PRT_FORENSIC_LOG");
    if (!path || !path[0]) return;
    FILE* fp = fopen(path, "a");
    if (!fp) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fclose(fp);
}

// Phase 28BR-D: checkpoint scan helper — file + stderr
static void prt_checkpoint_scan(const char* checkpoint_name, const float* buf, size_t n_floats,
                                   const void* ptr, size_t rows, size_t cols,
                                   const char* key, const char* family,
                                   bool cache_hit) {
    if (!buf || n_floats == 0) {
        prt_forensic_log("{\"checkpoint\":\"%s\",\"key\":\"%s\",\"family\":\"%s\",\"result\":\"NO_DATA\"}\n",
                checkpoint_name, key, family);
        fprintf(stderr, "[PRT-CHECKPOINT] %s ptr=%p float_count=0 rows=%zu cols=%zu finite=NO_DATA\n",
                checkpoint_name, ptr, rows, cols);
        return;
    }
    size_t nan_count = 0, inf_count = 0;
    double abs_sum = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < n_floats; i++) {
        float v = fabsf(buf[i]);
        abs_sum += v;
        if (v > max_abs) max_abs = v;
        if (std::isnan(buf[i])) nan_count++;
        if (std::isinf(buf[i])) inf_count++;
    }
    bool finite = (nan_count == 0 && inf_count == 0);
    // File-based JSONL
    prt_forensic_log(
        "{\"checkpoint\":\"%s\",\"key\":\"%s\",\"family\":\"%s\",\"ptr\":\"%p\","
        "\"rows\":%zu,\"cols\":%zu,\"float_count\":%zu,\"byte_count\":%zu,"
        "\"pager_ptr\":\"%p\",\"pager_global_addr\":\"%p\","
        "\"pager_enabled\":%d,\"pager_enabled_addr\":\"%p\","
        "\"expected_float\":%u,\"expected_byte\":%u,"
        "\"nan\":%zu,\"inf\":%zu,\"finite\":%d,"
        "\"abs_sum\":%.6e,\"max_abs\":%.6e,\"mean_abs\":%.6e,"
        "\"cache_hit\":%d,"
        "\"first8\":[%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e],"
        "\"last8\":[%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e],"
        "\"samples\":{\"0\":%.4e,\"1\":%.4e,\"895\":%.4e,\"896\":%.4e,\"897\":%.4e,\"last\":%.4e}}\n",
        checkpoint_name, key, family, (const void*)buf,
        rows, cols, n_floats, n_floats * 4u,
        (void*)g_prt_pager, (void*)&g_prt_pager,
        g_prt_pager_enabled ? 1 : 0, (void*)&g_prt_pager_enabled,
        (uint32_t)(rows * cols), (uint32_t)(rows * cols * 4u),
        nan_count, inf_count, finite ? 1 : 0,
        abs_sum, max_abs, (n_floats > 0 ? abs_sum / (double)n_floats : 0.0),
        cache_hit ? 1 : 0,
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        buf[n_floats-8], buf[n_floats-7], buf[n_floats-6], buf[n_floats-5],
        buf[n_floats-4], buf[n_floats-3], buf[n_floats-2], buf[n_floats-1],
        buf[0], buf[1],
        (n_floats > 895 ? buf[895] : 0.0f),
        (n_floats > 896 ? buf[896] : 0.0f),
        (n_floats > 897 ? buf[897] : 0.0f),
        buf[n_floats - 1]
    );
    // Stderr fallback
    fprintf(stderr, "[PRT-CHECKPOINT] %s key=%s ptr=%p float_count=%zu nan=%zu inf=%zu finite=%d abs_sum=%.3e\n",
            checkpoint_name, key, (const void*)buf, n_floats, nan_count, inf_count, finite ? 1 : 0, abs_sum);
    (void)checkpoint_name; (void)key; (void)family; (void)ptr;
}

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

// Phase 28BR-F: true injection canary — guarded, mutates only after graph-side checks
bool g_prt_sidecar_true_injection_enabled = false;

// Phase 28BR-T: scale factor for residual injection (0.0 to 2.0)
float g_prt_sidecar_scale_env = 1.0f;

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

    // Phase 28BR-D Checkpoint A: immediately after decode_bytes() returns, BEFORE cache insert
    {
        std::string keyA = std::to_string(layer) + ":" + family + ":A";
        size_t n_floats_A = (size_t)rows * cols;
        prt_checkpoint_scan("CHECKPOINT_A_AFTER_DECODE", dv.data, n_floats_A,
                            dv.data, rows, cols, keyA.c_str(), family, false);
    }

    if (dv.is_null || dv.data == nullptr) return nullptr;

    // Allocate owned buffer and COPY decoded data into it
    size_t n_floats = (size_t)rows * cols;
    float* owned = new float[n_floats];
    memcpy(owned, dv.data, n_floats * sizeof(float));
    delete[] dv.data;  // free decoder's buffer

    // Phase 28BR-D Checkpoint B: immediately after cache insert (owned copy)
    {
        std::string keyB = std::to_string(layer) + ":" + family + ":B";
        prt_checkpoint_scan("CHECKPOINT_B_AFTER_COPY", owned, n_floats,
                            owned, rows, cols, keyB.c_str(), family, false);
    }

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
    // Phase 28BR-F: true injection counters
    size_t injection_attempts = 0;
    size_t injection_successes = 0;
    size_t injection_failures = 0;
    size_t injection_skipped = 0;
    size_t injection_shape_mismatch = 0;
    size_t injection_nonfinite_blocked = 0;
    size_t injection_skipped_wrong_target = 0;
    size_t injection_skipped_nonfinite = 0;
    size_t injection_skipped_shape_mismatch = 0;
    bool contribution_finite_before_injection = false;
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
    // Phase 28BR-F: true injection counters
    size_t injection_attempts = 0;
    size_t injection_successes = 0;
    size_t injection_failures = 0;
    size_t injection_skipped = 0;
    size_t injection_shape_mismatch = 0;
    size_t injection_nonfinite_blocked = 0;
    size_t injection_skipped_wrong_target = 0;
    size_t injection_skipped_nonfinite = 0;
    size_t injection_skipped_shape_mismatch = 0;
    bool contribution_finite_before_injection = false;
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
    // Phase 28BR-F: true injection counters
    r.injection_attempts = g_prt_apply_stats.injection_attempts;
    r.injection_successes = g_prt_apply_stats.injection_successes;
    r.injection_failures = g_prt_apply_stats.injection_failures;
    r.injection_skipped = g_prt_apply_stats.injection_skipped;
    r.injection_shape_mismatch = g_prt_apply_stats.injection_shape_mismatch;
    r.injection_nonfinite_blocked = g_prt_apply_stats.injection_nonfinite_blocked;
    r.injection_skipped_wrong_target = g_prt_apply_stats.injection_skipped_wrong_target;
    r.injection_skipped_nonfinite = g_prt_apply_stats.injection_skipped_nonfinite;
    r.injection_skipped_shape_mismatch = g_prt_apply_stats.injection_skipped_shape_mismatch;
    r.contribution_finite_before_injection = g_prt_apply_stats.contribution_finite_before_injection;
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

// Phase 28BR-F: prepare true injection by decoding .trit and proving R is finite.
// This does not record an injection attempt or success; graph code does that only
// after target, shape, and finite checks allow wiring a real contribution.
prt_decoded_view prt_true_apply(int layer_idx, const std::string& tensor_family,
                                 const prt_residual_view& raw_view) {
    prt_decoded_view dec;
    dec.is_null = true;

    if (g_prt_sidecar_apply_layer >= 0 && layer_idx != g_prt_sidecar_apply_layer) {
        g_prt_apply_stats.injection_skipped++;
        g_prt_apply_stats.injection_skipped_wrong_target++;
        dec.reason = "injection_skipped_wrong_layer";
        return dec;
    }
    if (!g_prt_sidecar_apply_family.empty() && tensor_family != g_prt_sidecar_apply_family) {
        g_prt_apply_stats.injection_skipped++;
        g_prt_apply_stats.injection_skipped_wrong_target++;
        dec.reason = "injection_skipped_wrong_family";
        return dec;
    }

    if (raw_view.is_null || raw_view.data == nullptr || raw_view.size < 32) {
        g_prt_apply_stats.injection_skipped++;
        g_prt_apply_stats.injection_failures++;
        dec.reason = "injection_null_raw_view";
        return dec;
    }

    // Phase 28BR-A: use decode-once cache — decode once, reuse on repeated hits
    prt_trit_decoder decoder;
    float* cached = prt_decode_cached(
        reinterpret_cast<const char*>(raw_view.data), raw_view.size,
        layer_idx, tensor_family.c_str(), &decoder
    );

    if (cached != nullptr) {
        // Phase 28BR-F: full decoded-R finiteness check before graph mutation.
        size_t n = dec.rows * dec.cols;
        if (dec.rows > 0 && dec.cols > 0) {
            n = (size_t)dec.rows * (size_t)dec.cols;
        } else {
            auto it = g_prt_decode_cache.entries.find(std::to_string(layer_idx) + ":" + tensor_family);
            if (it != g_prt_decode_cache.entries.end()) {
                dec.rows = it->second.rows;
                dec.cols = it->second.cols;
                n = (size_t)dec.rows * (size_t)dec.cols;
            }
        }

        size_t nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < n; i++) {
            float v = cached[i];
            if (std::isnan(v)) nan_count++;
            if (std::isinf(v)) inf_count++;
        }

        if (nan_count > 0 || inf_count > 0) {
            g_prt_apply_stats.injection_skipped++;
            g_prt_apply_stats.injection_nonfinite_blocked++;
            g_prt_apply_stats.injection_skipped_nonfinite++;
            g_prt_apply_stats.injection_failures++;
            dec.reason = "injection_skipped_nonfinite";
            dec.data = cached;
            dec.is_null = true;
            return dec;
        }

        dec.is_null = false;
        dec.reason = "true_injection_ready";
        dec.format = prt_data_format::DECODED_F32;
        dec.data = cached;  // point to cached buffer (NOT owned by caller)
        g_prt_apply_stats.contribution_finite_before_injection = true;

        auto it = g_prt_decode_cache.entries.find(std::to_string(layer_idx) + ":" + tensor_family);
        if (it != g_prt_decode_cache.entries.end()) {
            dec.rows = it->second.rows;
            dec.cols = it->second.cols;
        }
    } else {
        g_prt_apply_stats.injection_skipped++;
        g_prt_apply_stats.injection_failures++;
        dec.reason = "injection_decode_failed";
    }

    return dec;
}

void prt_true_injection_record_shape_mismatch(int layer_idx, const char * family,
                                              int64_t r_rows, int64_t r_cols,
                                              int64_t x_rows, int64_t x_cols,
                                              int64_t out_rows, int64_t out_cols) {
    g_prt_apply_stats.injection_skipped++;
    g_prt_apply_stats.injection_shape_mismatch++;
    g_prt_apply_stats.injection_skipped_shape_mismatch++;
    prt_forensic_log(
        "{\"event\":\"TRUE_INJECTION_SHAPE_MISMATCH\",\"layer\":%d,\"family\":\"%s\","
        "\"R_rows\":%lld,\"R_cols\":%lld,\"X_rows\":%lld,\"X_cols\":%lld,"
        "\"out_rows\":%lld,\"out_cols\":%lld}\n",
        layer_idx, family ? family : "",
        (long long) r_rows, (long long) r_cols,
        (long long) x_rows, (long long) x_cols,
        (long long) out_rows, (long long) out_cols);
}

void prt_true_injection_record_attempt_result(bool success, const char * reason) {
    g_prt_apply_stats.injection_attempts++;
    if (success) {
        g_prt_apply_stats.injection_successes++;
        g_prt_apply_stats.sidecar_math_influenced_output = true;
    } else {
        g_prt_apply_stats.injection_failures++;
    }
    prt_forensic_log(
        "{\"event\":\"TRUE_INJECTION_ATTEMPT\",\"success\":%d,\"reason\":\"%s\","
        "\"injection_attempts\":%zu,\"injection_successes\":%zu,\"injection_failures\":%zu,"
        "\"sidecar_math_influenced_output\":%d}\n",
        success ? 1 : 0, reason ? reason : "",
        g_prt_apply_stats.injection_attempts,
        g_prt_apply_stats.injection_successes,
        g_prt_apply_stats.injection_failures,
        g_prt_apply_stats.sidecar_math_influenced_output ? 1 : 0);
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

    // Phase 28BR-D Checkpoint C: immediately after cache retrieval (before contribution loop reads R)
    {
        std::string keyC = std::to_string(layer_idx) + ":" + family + ":C";
        prt_checkpoint_scan("CHECKPOINT_C_AFTER_RETRIEVAL", e.decoded_buf, n,
                            e.decoded_buf, e.rows, e.cols, keyC.c_str(), family, true);
    }

    // X = I[K×K] (identity) — synthetic. Y = I @ R = R.
    // R is row-major [K×K]; Y will also be [K×K] row-major.
    float* R = e.decoded_buf;

    // Phase 28BR-D Checkpoint D: immediately before contribution loop reads R
    {
        std::string keyD = std::to_string(layer_idx) + ":" + family + ":D";
        prt_checkpoint_scan("CHECKPOINT_D_BEFORE_LOOP", R, n,
                            R, e.rows, e.cols, keyD.c_str(), family, true);
    }

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
