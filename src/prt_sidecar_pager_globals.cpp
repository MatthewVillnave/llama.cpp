// Phase 28BM: PRT sidecar pager global variables
// Phase 28BQ: added decoder + application counters
// Strong symbols defined here (not inline) — linked into libllama.so

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

#include "prt_sidecar_pager.h"
#include "prt_trit_decode.h"
#include <unordered_map>
#include <cstring>
#include <vector>

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

// Phase 28BQ: application counters (defined as strong symbol here)
struct prt_apply_counters {
    size_t decoded_views = 0;
    size_t application_attempts = 0;
    size_t application_successes = 0;
    size_t application_skipped_wrong_target = 0;
    size_t application_failures = 0;
    size_t decode_errors = 0;
    bool sidecar_math_influenced_output = false;
};

static struct {
    size_t decoded_views = 0;
    size_t application_attempts = 0;
    size_t application_successes = 0;
    size_t application_skipped_wrong_target = 0;
    size_t application_failures = 0;
    size_t decode_errors = 0;
    bool sidecar_math_influenced_output = false;
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

    // Extract header from raw .trit view
    const uint8_t * hdr = raw_view.data;
    uint32_t rows = *(const uint32_t *)(hdr + 8);
    uint32_t cols = *(const uint32_t *)(hdr + 12);
    uint16_t block_rows = *(const uint16_t *)(hdr + 16);
    uint16_t block_cols = *(const uint16_t *)(hdr + 18);
    uint16_t n_scales = *(const uint16_t *)(hdr + 20);
    uint32_t scale_offset = *(const uint32_t *)(hdr + 26);

    if (rows == 0 || cols == 0 || n_scales == 0) {
        g_prt_apply_stats.decode_errors++;
        dec.reason = "invalid_header";
        return dec;
    }

    // Extract scales
    std::vector<float> scales(n_scales);
    if (scale_offset > 0 && scale_offset < raw_view.size) {
        memcpy(scales.data(), hdr + scale_offset, n_scales * sizeof(float));
    }

    // Decode .trit to float
    prt_trit_decoder decdr;
    prt_decoded_view dv = decdr.decode_bytes(
        raw_view.data, raw_view.size,
        rows, cols, block_rows, block_cols,
        n_scales, scales.data()
    );

    g_prt_apply_stats.decoded_views++;

    if (dv.is_null) {
        g_prt_apply_stats.decode_errors++;
        g_prt_apply_stats.application_failures++;
        dec.reason = "decode_failed:" + dv.reason;
        return dec;
    }

    // Option B (shadow compute): decode succeeds, we have decoded residual buffer,
    // but we do NOT feed it into the model compute path.
    // This proves the application path executed without corrupting generation.
    g_prt_apply_stats.application_attempts++;
    g_prt_apply_stats.application_successes++;
    g_prt_apply_stats.sidecar_math_influenced_output = false;

    // Clean up decoded buffer (shadow — not used further)
    delete[] dv.data;

    dec.is_null = false;
    dec.reason = "shadow_compute";
    dec.format = prt_data_format::DECODED_F32;
    return dec;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL