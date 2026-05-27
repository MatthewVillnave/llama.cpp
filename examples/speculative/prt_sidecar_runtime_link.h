// Phase 28AU: Runtime Link Stub — prt_get_residual_view() behind disabled flag
// Provides lookup routing between legacy g_sidecars and pager, no generation change.
// HEADER ONLY — implementations are inline-weak to avoid ODR violations.
// Actual definitions are in prt_sidecar_runtime_link.cpp (linked into libllama.so).

#ifndef PRT_SIDECAR_RUNTIME_LINK_H
#define PRT_SIDECAR_RUNTIME_LINK_H

// Compile guard: only active when experimental sidecar pager is enabled
#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

#include "prt_sidecar_pager.h"
#include "prt_trit_decode.h"
#include <cstdio>
#include <string>
#include <unordered_map>

// SidecarLoad — duplicated from prt_shadow.h to avoid circular include.
struct SidecarLoad {
    int layer;
    std::string path;
    float * data;  // PRT sidecar: |w| per element, float32
    size_t size;
};

// ── Global symbols (defined once in src/prt_sidecar_pager_globals.cpp) ──

extern std::unordered_map<int, SidecarLoad> g_sidecars;
extern bool g_sidecars_loaded;
extern prt_sidecar_pager* g_prt_pager;
extern bool g_prt_pager_enabled;

// Phase 28BQ: guarded residual application globals
extern bool g_prt_sidecar_apply_enabled;
extern int  g_prt_sidecar_apply_layer;
extern std::string g_prt_sidecar_apply_family;

// Phase 28BR-B: synthetic-X shadow contribution
extern bool g_prt_sidecar_shadow_contrib_enabled;

// Phase 28BR-F: true injection canary — guarded, mutates only after graph-side shape checks
extern bool g_prt_sidecar_true_injection_enabled;

// Phase 28BR-B: contribution metrics struct (defined in prt_sidecar_pager_globals.cpp)
struct prt_contrib_metrics {
    size_t contribution_attempts = 0;
    size_t contribution_successes = 0;
    size_t contribution_failures = 0;
    size_t contribution_skipped_wrong_target = 0;
    size_t contribution_nan_count = 0;
    size_t contribution_inf_count = 0;
    size_t contribution_Y_size = 0;
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

// Phase 28BR-B: compute Y = X @ R with synthetic X = I[K×K]. Returns contribution metrics.
// Does NOT inject into model output. pure shadow compute.
bool prt_shadow_contribution_synthetic(int layer_idx, const char* family, prt_contrib_metrics& out_metrics);

// Phase 28BR-B: get contribution counters
prt_contrib_metrics prt_get_contrib_metrics();

// ── Residual view wrapper ────────────────────────────────────────────────────

// Get residual view for a layer + tensor family.
// Routes to pager if enabled and layer is available in pager manifest.
// Falls back to legacy g_sidecars only when pager is disabled.
// Returns a null view with reason set if neither pager nor legacy has the tensor.
inline prt_residual_view prt_get_residual_view(int layer_idx, const std::string& tensor_family);

// Phase 28BQ: guarded shadow apply — decode .trit and run shadow compute.
// Option B only: decoded buffer computed but NOT fed into model compute path.
// Returns decoded view (is_null=false on success) with reason=shadow_compute.
inline prt_decoded_view prt_shadow_apply(int layer_idx, const std::string& tensor_family, const prt_residual_view& raw_view);

// Phase 28BR-F: prepare true injection by decoding .trit and checking target/R finiteness.
// Graph code records success only if it actually wires a contribution into output.
// Strong implementation in prt_sidecar_pager_globals.cpp.
inline prt_decoded_view prt_true_apply(int layer_idx, const std::string& tensor_family, const prt_residual_view& raw_view);
void prt_true_injection_record_shape_mismatch(int layer_idx, const char * family,
                                              int64_t r_rows, int64_t r_cols,
                                              int64_t x_rows, int64_t x_cols,
                                              int64_t out_rows, int64_t out_cols);
void prt_true_injection_record_attempt_result(bool success, const char * reason);

// Phase 28BQ: application counters (strong symbol in prt_sidecar_pager_globals.cpp)
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
    size_t injection_skipped_wrong_target = 0; // retained for older logs
    size_t injection_skipped_nonfinite = 0;    // retained for older logs
    size_t injection_skipped_shape_mismatch = 0; // retained for older logs
    bool contribution_finite_before_injection = false;
};
inline prt_apply_counters prt_get_apply_stats();

inline prt_residual_view prt_get_residual_view(int layer_idx, const std::string& tensor_family) {
    prt_residual_view view;
    view.is_null = true;
    view.reason = "not_found";

    if (g_prt_pager != nullptr && g_prt_pager_enabled) {
        // Pager path
        view = g_prt_pager->get_residual(layer_idx, tensor_family);

        // Phase 28BP-A: not_in_manifest is cheap — no activation attempt, no verbose log
        if (view.reason == "not_in_manifest") {
            return view;
        }

        if (!view.is_null) {
            prt_sidecar_pager_stats s = g_prt_pager->get_stats();
            fprintf(stderr,
                    "[PRT-PAGER-LAZY] layer=%d family=%s reason=%s size=%zu resident_bytes=%zu activation_attempts=%zu activation_successes=%zu non_null_views=%zu null_views=%zu budget_rejects=%zu not_in_manifest=%zu\n",
                    layer_idx, tensor_family.c_str(), view.reason.c_str(), view.size,
                    s.resident_bytes, s.activation_attempts, s.activation_successes,
                    s.non_null_views, s.null_views, s.budget_rejects, s.not_in_manifest);
            return view;
        }

        const std::string first_reason = view.reason;
        bool activation_attempted = false;
        bool activation_ok = false;

        // Phase 28BN: lazy activation on first lookup.
        if (view.reason == "layer_not_activated") {
            activation_attempted = true;
            activation_ok = g_prt_pager->activate_layer(layer_idx);
            if (activation_ok) {
                view = g_prt_pager->get_residual(layer_idx, tensor_family);
                if (!view.is_null) {
                    prt_sidecar_pager_stats s = g_prt_pager->get_stats();
                    fprintf(stderr,
                            "[PRT-PAGER-LAZY] layer=%d family=%s first_reason=%s activation_attempted=1 activation_ok=1 retry_is_null=0 reason=%s size=%zu resident_bytes=%zu activation_attempts=%zu activation_successes=%zu non_null_views=%zu null_views=%zu budget_rejects=%zu not_in_manifest=%zu\n",
                            layer_idx, tensor_family.c_str(), first_reason.c_str(), view.reason.c_str(), view.size,
                            s.resident_bytes, s.activation_attempts, s.activation_successes,
                            s.non_null_views, s.null_views, s.budget_rejects, s.not_in_manifest);
                    return view;
                }
            }
        }

        prt_sidecar_pager_stats s = g_prt_pager->get_stats();
        fprintf(stderr,
                "[PRT-PAGER-LAZY] layer=%d family=%s first_reason=%s activation_attempted=%d activation_ok=%d retry_is_null=%d reason=%s size=%zu resident_bytes=%zu activation_attempts=%zu activation_successes=%zu non_null_views=%zu null_views=%zu budget_rejects=%zu not_in_manifest=%zu\n",
                layer_idx, tensor_family.c_str(), first_reason.c_str(),
                activation_attempted ? 1 : 0, activation_ok ? 1 : 0, view.is_null ? 1 : 0,
                view.reason.c_str(), view.size, s.resident_bytes, s.activation_attempts,
                s.activation_successes, s.non_null_views, s.null_views, s.budget_rejects, s.not_in_manifest);

        return view;
    }

    // Legacy path: check g_sidecars map only when pager is disabled.
    auto it = g_sidecars.find(layer_idx);
    if (it != g_sidecars.end() && it->second.data != nullptr) {
        view.data = reinterpret_cast<const uint8_t*>(it->second.data);
        view.size = it->second.size;
        view.is_null = false;
        view.reason = "legacy";
        return view;
    }

    view.is_null = true;
    view.reason = "not_found";
    return view;
}

inline bool prt_init_pager(const prt_sidecar_pager_config& config) {
    if (g_prt_pager != nullptr) {
        return true;
    }
    g_prt_pager = new prt_sidecar_pager(config);
    if (!g_prt_pager->init()) {
        delete g_prt_pager;
        g_prt_pager = nullptr;
        g_prt_pager_enabled = false;
        return false;
    }
    g_prt_pager_enabled = true;
    return true;
}

inline void prt_shutdown_pager() {
    if (g_prt_pager != nullptr) {
        g_prt_pager->shutdown();
        delete g_prt_pager;
        g_prt_pager = nullptr;
        g_prt_pager_enabled = false;
    }
}

inline prt_sidecar_pager_stats prt_get_pager_stats() {
    if (g_prt_pager != nullptr) {
        return g_prt_pager->get_stats();
    }
    return prt_sidecar_pager_stats();
}

// Phase 28BR-T: residual scale factor
extern float g_prt_sidecar_scale_env;

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL

#endif  // PRT_SIDECAR_RUNTIME_LINK_H
