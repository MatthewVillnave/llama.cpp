// Phase 11BB: Route A — GGML Custom Op for True Replacement
//
// In build_ffn, when PRT true replacement is active for a layer,
// replace build_lora_mm(up, cur) with build_prt_ffn_up(cur).
// This inserts a GGML custom op node that produces [ffn, n_tokens]
// directly — the native FFN_UP matmul is never inserted into the graph.
//
// Mode 5700: all 36 layers use PRT true replacement
// Mode 5600+L: layer L uses PRT true replacement
// Mode < 5600: native build_lora_mm (existing behavior)
//
// Custom op userdata pool (per-layer, re-used across calls)

// Phase 13U: log level gating — externs declared in llama-graph.cpp
extern FILE * g_prt_log_file;
extern int g_prt_log_level;

// Phase 13V: per-call timing instrumentation
#include <chrono>
#include <float.h>  // for FLT_MAX
#if defined(__AVX2__)
#include <immintrin.h>  // for AVX2 intrinsics (__m256, _mm256_*)
#endif

// Per-layer timing accumulators (array indexed by layer_id)
static double g_prt_call_time_total[36] = {0.0};
static double g_prt_call_time_min[36]   = {DBL_MAX};
static double g_prt_call_time_max[36]   = {0.0};
static int    g_prt_call_count[36]      = {0};

// Phase 13W: nested timing — separate kernel time from setup/dispatch overhead
static double g_prt_kernel_time_total[36] = {0.0};
static double g_prt_kernel_time_min[36]   = {DBL_MAX};
static double g_prt_kernel_time_max[36]   = {0.0};
static int    g_prt_kernel_count[36]      = {0};

// Phase 13W: first-call vs subsequent-call tracking
static bool   g_prt_layer_called_first[36] = {false};  // tracks if layer had a "first" call
static double g_prt_first_call_time[36]   = {0.0};     // time of first call per layer
static double g_prt_later_call_time_total[36] = {0.0}; // accumulated time of non-first calls
static int    g_prt_later_call_count[36]  = {0};

static bool   g_prt_timing_initialized   = false;

static void prt_init_timing(void) {
    if (!g_prt_timing_initialized) {
        for (int i = 0; i < 36; i++) {
            g_prt_call_time_min[i] = DBL_MAX;
            g_prt_kernel_time_min[i] = DBL_MAX;
            g_prt_layer_called_first[i] = false;
        }
        g_prt_timing_initialized = true;
    }
}

// Phase 13W: reset timing state (called at start of each run)
static void prt_reset_timing(void) {
    for (int i = 0; i < 36; i++) {
        g_prt_call_time_total[i] = 0.0;
        g_prt_call_time_min[i]   = DBL_MAX;
        g_prt_call_time_max[i]   = 0.0;
        g_prt_call_count[i]      = 0;
        g_prt_kernel_time_total[i] = 0.0;
        g_prt_kernel_time_min[i]   = DBL_MAX;
        g_prt_kernel_time_max[i]   = 0.0;
        g_prt_kernel_count[i]      = 0;
        g_prt_layer_called_first[i] = false;
        g_prt_first_call_time[i]   = 0.0;
        g_prt_later_call_time_total[i] = 0.0;
        g_prt_later_call_count[i]  = 0;
    }
}

// Emit per-layer timing summary (called at process exit or on demand)
static void prt_dump_timing_summary(void) {
    if (g_prt_log_level < 1) return;  // summary or debug only
    double grand_total_cb = 0.0, grand_total_kern = 0.0;
    double grand_first = 0.0, grand_later = 0.0;
    int total_calls = 0;
    for (int i = 0; i < 36; i++) {
        if (g_prt_call_count[i] > 0) {
            double avg_ms = g_prt_call_time_total[i] / g_prt_call_count[i];
            double avg_kern = g_prt_kernel_count[i] > 0 ?
                g_prt_kernel_time_total[i] / g_prt_kernel_count[i] : 0.0;
            if (g_prt_log_file) {
                fprintf(g_prt_log_file,
                    "[PRT-13V-TIMING] IL=%d calls=%d avg_ms=%.3f min_ms=%.3f max_ms=%.3f "
                    "kern_avg=%.3f kern_total=%.3f first_ms=%.3f later_avg=%.3f\n",
                    i, g_prt_call_count[i], avg_ms,
                    g_prt_call_time_min[i] * 1000.0,
                    g_prt_call_time_max[i] * 1000.0,
                    avg_kern, g_prt_kernel_time_total[i] * 1000.0,
                    g_prt_first_call_time[i] * 1000.0,
                    g_prt_later_call_count[i] > 0 ?
                        g_prt_later_call_time_total[i] / g_prt_later_call_count[i] * 1000.0 : 0.0);
                fflush(g_prt_log_file);
            }
            grand_total_cb += g_prt_call_time_total[i];
            grand_total_kern += g_prt_kernel_time_total[i];
            grand_first += g_prt_first_call_time[i];
            grand_later += g_prt_later_call_time_total[i];
            total_calls += g_prt_call_count[i];
        }
    }
    if (g_prt_log_file && total_calls > 0) {
        fprintf(g_prt_log_file,
            "[PRT-13V-SUMMARY] total_calls=%d cb_total=%.1fms kern_total=%.1fms "
            "overhead_total=%.1fms first_total=%.1fms later_total=%.1fms\n",
            total_calls, grand_total_cb * 1000.0, grand_total_kern * 1000.0,
            (grand_total_cb - grand_total_kern) * 1000.0,
            grand_first * 1000.0, grand_later * 1000.0);
        fflush(g_prt_log_file);
    }
}

// Phase 13W: pre-touch sidecar memory to force page faults before generation
static void prt_pretouch_sidecars(void) {
    if (g_prt_log_level < 1) return;  // summary or debug only
    extern int g_prt_sidecar_M[36];
    extern int g_prt_sidecar_N[36];
    extern const float * g_prt_sidecar_data[36];
    extern bool g_prt_force_native_enabled;
    extern bool g_prt_force_native_layer[36];

    auto pretouch_start = std::chrono::high_resolution_clock::now();
    volatile double checksum = 0.0;
    int touched_layers = 0;
    size_t touched_bytes = 0;

    for (int i = 0; i < 36; i++) {
        if (!g_prt_sidecar_data[i]) continue;
        if (g_prt_force_native_enabled && g_prt_force_native_layer[i]) continue; // skip force-native
        int M = g_prt_sidecar_M[i];
        int N = g_prt_sidecar_N[i];
        if (M <= 0 || N <= 0) continue;
        const float * ptr = g_prt_sidecar_data[i];
        // Touch one float per 4096-byte page (every 1024 floats)
        // Use volatile to prevent dead-code elimination
        for (size_t j = 0; j < (size_t)M * N; j += 1024) {
            checksum += ptr[j];
        }
        touched_bytes += (size_t)M * N * sizeof(float);
        touched_layers++;
    }

    auto pretouch_end = std::chrono::high_resolution_clock::now();
    double pretouch_ms = std::chrono::duration<double, std::milli>(
        pretouch_end - pretouch_start).count();

    if (g_prt_log_file) {
        fprintf(g_prt_log_file,
            "[PRT-13W-PRETOUCH] layers=%d bytes=%zu checksum=%.4f time_ms=%.2f\n",
            touched_layers, touched_bytes, checksum, pretouch_ms);
        fflush(g_prt_log_file);
    }
}

// Public API to reset timing (call at start of each run)
extern "C" LLAMA_API void llama_reset_prt_timing(void) {
    prt_reset_timing();
}

// Public API to trigger pretouch (call after sidecar load, before generation)
extern "C" LLAMA_API void llama_pretouch_prt_sidecars(void) {
    prt_pretouch_sidecars();
}
struct PRTUserData {
    const float * sidecar;   // [ffn * hidden] float32
    const int8_t * int8_data; // Phase 14B: INT8 weights
    const float * int8_scales; // Phase 14B: per-row scales [M]
    int format;              // Phase 14B: 0=float32, 1=int8
    int M;                   // hidden = 2048
    int N;                   // ffn = 11008
    int layer_id;
    int batch;               // n_tokens
    int kernel_mode;         // 0=scalar, 1=AVX2 (float32 only)
};

static PRTUserData g_prt_ud_pool[36];

// PRT custom op — replaces native ffn_up matmul
// src[0] = cur [hidden, n_tokens]
// dst = [ffn, n_tokens] — consumed by SwiGLU
static void prt_ffn_up_custom_op(
    struct ggml_tensor * dst,
    int ith, int nth, void * userdata
) {
    if (ith != 0) return;

    PRTUserData * ud = (PRTUserData *)userdata;
    if (!ud) {
        if (g_prt_log_file) { fprintf(g_prt_log_file, "[PRT-11BB] ERROR: custom op called without userdata!\n"); fflush(g_prt_log_file); }
        else { fprintf(stderr, "[PRT-11BB] ERROR: custom op called without userdata!\n"); }
        return;
    }
    // Accept float32 OR INT8 sidecar
    if (ud->format == 0 && !ud->sidecar) {
        if (g_prt_log_file) { fprintf(g_prt_log_file, "[PRT-11BB] ERROR: custom op called without float32 sidecar!\n"); fflush(g_prt_log_file); }
        else { fprintf(stderr, "[PRT-11BB] ERROR: custom op called without float32 sidecar!\n"); }
        return;
    }
    if (ud->format == 1 && (!ud->int8_data || !ud->int8_scales)) {
        if (g_prt_log_file) { fprintf(g_prt_log_file, "[PRT-11BB] ERROR: custom op called without INT8 sidecar data/scales!\n"); fflush(g_prt_log_file); }
        else { fprintf(stderr, "[PRT-11BB] ERROR: custom op called without INT8 sidecar data/scales!\n"); }
        return;
    }

    // Phase 19C: shape audit log (fires before crash)
    {
        const struct ggml_tensor * _src0 = dst->src[0];
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT_SHAPE] IL=%d dst_ne=[%lld,%lld] src_ne=[%lld,%lld] ud_M=%d ud_N=%d format=%d\n",
                    ud->layer_id, (long long)dst->ne[0], (long long)dst->ne[1],
                    (long long)_src0->ne[0], (long long)_src0->ne[1], ud->M, ud->N, ud->format);
            fflush(g_prt_log_file);
        } else {
            fprintf(stderr, "[PRT_SHAPE] IL=%d dst_ne=[%lld,%lld] src_ne=[%lld,%lld] ud_M=%d ud_N=%d format=%d\n",
                    ud->layer_id, (long long)dst->ne[0], (long long)dst->ne[1],
                    (long long)_src0->ne[0], (long long)_src0->ne[1], ud->M, ud->N, ud->format);
        }
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const float * X = (const float *)src0->data;
    int hidden = ud->M;   // 2048
    int ffn    = ud->N;   // 11008
    int n_tokens = ud->batch;
    float * Y = (float *)dst->data;

    // Phase 19U: Activation audit — layer 0, token 0, first call only
    {
        bool do_audit = (ud->layer_id == 0 && n_tokens >= 1);
        if (do_audit) {
            static bool s_audit_fired = false;
            if (!s_audit_fired) {
                s_audit_fired = true;
                float x_min = X[0], x_max = X[0], x_sum = 0.0f, x_abssum = 0.0f;
                int x_nan = 0, x_inf = 0;
                for (int i = 0; i < 16; i++) {
                    if (std::isnan((double)X[i])) x_nan++;
                    if (std::isinf((double)X[i])) x_inf++;
                }
                for (int i = 0; i < std::min(16, hidden); i++) {
                    float v = X[i];
                    if (v < x_min) x_min = v;
                    if (v > x_max) x_max = v;
                    x_sum += v;
                    x_abssum += fabsf(v);
                }
                if (g_prt_log_file) {
                    fprintf(g_prt_log_file, "[PRT_ACT_AUDIT] layer=0 shape=[%d,%d] first16=",
                            hidden, n_tokens);
                    for (int i = 0; i < 16 && i < hidden; i++) fprintf(g_prt_log_file, " %.4g", X[i]);
                    fprintf(g_prt_log_file, " min=%.4g max=%.4g mean=%.4g abassum=%.4g nan=%d inf=%d\n",
                            x_min, x_max, x_sum/std::min(16,hidden), x_abssum, x_nan, x_inf);
                    fflush(g_prt_log_file);
                }
            }
        }
    }

    // Per-call debug logs: only at debug level (2)
    if (g_prt_log_level >= 2) {
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT-11BB] custom op: dst=%s src0=%s ne=[%lld,%lld] src_ne=[%lld,%lld]\n",
                    dst->name, src0->name,
                    (long long)dst->ne[0], (long long)dst->ne[1],
                    (long long)src0->ne[0], (long long)src0->ne[1]);
            fflush(g_prt_log_file);
        } else {
            fprintf(stderr, "[PRT-11BB] custom op: dst=%s src0=%s ne=[%lld,%lld] src_ne=[%lld,%lld]\n",
                    dst->name, src0->name,
                    (long long)dst->ne[0], (long long)dst->ne[1],
                    (long long)src0->ne[0], (long long)src0->ne[1]);
        }
    }
    
    // Dump first 4 input values and cur norm
    float sum_in = 0.0f, sum_out = 0.0f;
    for (int i = 0; i < std::min(4, hidden * n_tokens); i++) sum_in += X[i];
    for (int i = 0; i < std::min(4, ffn * n_tokens); i++) sum_out += Y[i];
    if (g_prt_log_level >= 2) {
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT-11BB] IL=%d hidden=%d ffn=%d tokens=%d in_sum(4)=%.4f out_sum(4)=%.4f\n",
                    ud->layer_id, hidden, ffn, n_tokens, sum_in, sum_out);
            fflush(g_prt_log_file);
        } else {
            fprintf(stderr, "[PRT-11BB] IL=%d hidden=%d ffn=%d tokens=%d in_sum(4)=%.4f out_sum(4)=%.4f\n",
                    ud->layer_id, hidden, ffn, n_tokens, sum_in, sum_out);
        }
    }

    // Phase 13V/13W: per-call nested timing
    prt_init_timing();
    auto prt_total_start = std::chrono::high_resolution_clock::now();
    auto prt_kernel_end = prt_total_start;  // default if AVX2 not used
    auto prt_kernel_start = prt_total_start;

#if defined(__AVX2__)
    if (ud->kernel_mode == 1) {
        prt_kernel_start = std::chrono::high_resolution_clock::now();
        const int VLEN = 8;
        for (int t = 0; t < n_tokens; t++) {
            const float * X_t = X + t * hidden;
            float * Y_t = Y + t * ffn;

            int j = 0;
            for (; j + VLEN - 1 < ffn; j += VLEN) {
                __m256 sum[VLEN];
                for (int v = 0; v < VLEN; v++) sum[v] = _mm256_setzero_ps();

                int k = 0;
                for (; k + 31 < hidden; k += 32) {
                    __m256 x0 = _mm256_loadu_ps(X_t + k);
                    __m256 x1 = _mm256_loadu_ps(X_t + k + 8);
                    __m256 x2 = _mm256_loadu_ps(X_t + k + 16);
                    __m256 x3 = _mm256_loadu_ps(X_t + k + 24);

                    for (int v = 0; v < VLEN; v++) {
                        const float * W_row = ud->sidecar + (j + v) * hidden + k;
                        sum[v] = _mm256_fmadd_ps(x0, _mm256_loadu_ps(W_row),     sum[v]);
                        sum[v] = _mm256_fmadd_ps(x1, _mm256_loadu_ps(W_row + 8),  sum[v]);
                        sum[v] = _mm256_fmadd_ps(x2, _mm256_loadu_ps(W_row + 16), sum[v]);
                        sum[v] = _mm256_fmadd_ps(x3, _mm256_loadu_ps(W_row + 24), sum[v]);
                    }
                }
                // k tail
                for (; k < hidden; k++) {
                    float xv = X_t[k];
                    __m256 xvec = _mm256_set1_ps(xv);
                    for (int v = 0; v < VLEN; v++) {
                        sum[v] = _mm256_fmadd_ps(xvec,
                            _mm256_set1_ps(ud->sidecar[(j + v) * hidden + k]), sum[v]);
                    }
                }
                // Horizontal sum → store
                for (int v = 0; v < VLEN; v++) {
                    __m128 lo = _mm256_castps256_ps128(sum[v]);
                    __m128 hi = _mm256_extractf128_ps(sum[v], 1);
                    __m128 s = _mm_add_ps(lo, hi);
                    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
                    s = _mm_add_ss(s, _mm_movehdup_ps(s));
                    Y_t[j + v] = _mm_cvtss_f32(s);
                }
            }
            // j tail
            for (; j < ffn; j++) {
                float s = 0.0f;
                int k = 0;
                for (; k + 7 < hidden; k += 8) {
                    __m256 x = _mm256_loadu_ps(X_t + k);
                    __m256 w = _mm256_loadu_ps(ud->sidecar + j * hidden + k);
                    __m256 p = _mm256_mul_ps(x, w);
                    __m128 lo = _mm256_castps256_ps128(p);
                    __m128 hi = _mm256_extractf128_ps(p, 1);
                    __m128 ps = _mm_add_ps(lo, hi);
                    ps = _mm_add_ps(ps, _mm_movehl_ps(ps, ps));
                    ps = _mm_add_ss(ps, _mm_movehdup_ps(ps));
                    s += _mm_cvtss_f32(ps);
                }
                for (; k < hidden; k++) {
                    s += X_t[k] * ud->sidecar[j * hidden + k];
                }
                Y_t[j] = s;
            }
        }
        prt_kernel_end = std::chrono::high_resolution_clock::now();
    } else
#endif
    {
        // Phase 14B: INT8 path with dequantization
        // Phase 19D: INT6 (format=2) uses same int8_data buffer but packed 6-bit values
        if ((ud->format == 1 || ud->format == 2) && ud->int8_data && ud->int8_scales) {
            for (int t = 0; t < n_tokens; t++) {
                const float * X_t = X + t * hidden;
                float * Y_t = Y + t * ffn;
                for (int j = 0; j < ffn; j++) {
                    float s = 0.0f;
                    float sc = ud->int8_scales[j];
                    for (int k = 0; k < hidden; k++) {
                        // W_float[j*K+k] = int8[j*K+k] * scale[j]
                        s += X_t[k] * (float)ud->int8_data[j * hidden + k] * sc;
                    }
                    Y_t[j] = s;
                }
            }
        } else {
            // Scalar fallback for float32
            for (int t = 0; t < n_tokens; t++) {
                const float * X_t = X + t * hidden;
                float * Y_t = Y + t * ffn;
                for (int j = 0; j < ffn; j++) {
                    float s = 0.0f;
                    for (int k = 0; k < hidden; k++) {
                        s += X_t[k] * ud->sidecar[j * hidden + k];
                    }
                    Y_t[j] = s;
                }
            }
        }
    }

    // Phase 13V/13W: record per-call timing (total + nested)
    auto prt_total_end = std::chrono::high_resolution_clock::now();
    double prt_call_sec = std::chrono::duration<double>(
        prt_total_end - prt_total_start).count();
    double prt_kern_sec = std::chrono::duration<double>(
        prt_kernel_end - prt_kernel_start).count();
    int lid = ud->layer_id;

    // Phase 19U: FFN_UP output audit — layer 0, token 0, first call only
    {
        bool do_audit = (lid == 0 && n_tokens >= 1);
        if (do_audit) {
            static bool s_up_audit_fired = false;
            if (!s_up_audit_fired) {
                s_up_audit_fired = true;
                float y_min = Y[0], y_max = Y[0], y_sum = 0.0f, y_abssum = 0.0f;
                int y_nan = 0, y_inf = 0;
                for (int i = 0; i < 16; i++) {
                    if (std::isnan((double)Y[i])) y_nan++;
                    if (std::isinf((double)Y[i])) y_inf++;
                }
                for (int i = 0; i < 16; i++) {
                    float v = Y[i];
                    if (v < y_min) y_min = v;
                    if (v > y_max) y_max = v;
                    y_sum += v;
                    y_abssum += fabsf(v);
                }
                if (g_prt_log_file) {
                    fprintf(g_prt_log_file, "[PRT_UP_AUDIT] layer=0 shape=[%d,%d] first16=",
                            ffn, n_tokens);
                    for (int i = 0; i < 16; i++) fprintf(g_prt_log_file, " %.4g", Y[i]);
                    fprintf(g_prt_log_file, " min=%.4g max=%.4g mean=%.4g abassum=%.4g nan=%d inf=%d\n",
                            y_min, y_max, y_sum/16.0f, y_abssum, y_nan, y_inf);
                    fflush(g_prt_log_file);
                }
            }
        }
    }

    g_prt_call_time_total[lid] += prt_call_sec;
    if (prt_call_sec < g_prt_call_time_min[lid]) g_prt_call_time_min[lid] = prt_call_sec;
    if (prt_call_sec > g_prt_call_time_max[lid]) g_prt_call_time_max[lid] = prt_call_sec;
    g_prt_call_count[lid]++;

    // Phase 13W: kernel time separately (AVX2 only — for scalar, kern=total)
    g_prt_kernel_time_total[lid] += prt_kern_sec;
    if (prt_kern_sec < g_prt_kernel_time_min[lid]) g_prt_kernel_time_min[lid] = prt_kern_sec;
    if (prt_kern_sec > g_prt_kernel_time_max[lid]) g_prt_kernel_time_max[lid] = prt_kern_sec;
    g_prt_kernel_count[lid]++;

    // Phase 13W: first-call vs subsequent-call tracking
    if (!g_prt_layer_called_first[lid]) {
        g_prt_first_call_time[lid] = prt_call_sec;
        g_prt_layer_called_first[lid] = true;
    } else {
        g_prt_later_call_time_total[lid] += prt_call_sec;
        g_prt_later_call_count[lid]++;
    }

    // Count true replacement invocations
    extern int g_prt_true_replacement_calls;
    g_prt_true_replacement_calls++;
}

// Check if PRT true replacement is active for a given layer
// Mode 5700: all layers
// Mode 5600+L: layer L only
static bool prt_is_true_replacement_layer(int il) {
    extern int g_prt_debug_mode;
    extern int g_prt_only_layer;
    extern bool g_prt_only_layers_set[36];
    extern bool g_prt_disable_layers_set[36];
    // Phase 19X: prt_disable_layers takes priority — if set, these layers are always native
    if (g_prt_disable_layers_set[il]) return false;
    // Phase 19X: multi-layer set — if any layers are set, only those use PRT
    bool any_only_layers_set = false;
    for (int i = 0; i < 36; i++) { if (g_prt_only_layers_set[i]) { any_only_layers_set = true; break; } }
    if (any_only_layers_set) return g_prt_only_layers_set[il];
    // Phase 19W: single-layer mode — if set to >=0, only this layer uses PRT
    if (g_prt_only_layer >= 0 && g_prt_only_layer <= 35) {
        return (il == g_prt_only_layer);
    }
    if (g_prt_debug_mode >= 5700) return true;                          // all layers
    if (g_prt_debug_mode >= 5600 && g_prt_debug_mode < 5700) {
        return (g_prt_debug_mode == 5600 + il);                         // specific layer
    }
    return false;
}

// Build PRT FFN_UP custom op — drop-in for build_lora_mm(up, cur)
// Returns a [ffn, n_tokens] tensor computed via PRT sidecar matmul
// Returns nullptr if PRT is not active for this layer (caller uses build_lora_mm)
static ggml_tensor * build_prt_ffn_up(
    struct ggml_context * ctx,
    ggml_tensor * cur,    // [hidden, n_tokens] input activations
    int layer_id          // layer index (il)
) {
    extern int g_prt_debug_mode;
    extern const float * g_prt_sidecar_data[36];
    extern int g_prt_sidecar_M[36];
    extern int g_prt_sidecar_N[36];
    extern int g_prt_kernel_mode;  // 0=scalar, 1=AVX2

    // Phase 14B: also check INT8 sidecar data (float32 or INT8 must be present)
    if (layer_id < 0 || layer_id >= 36) return nullptr;
    if (!g_prt_sidecar_data[layer_id] && !g_prt_int8_data[layer_id]) return nullptr;

    // Phase 19C-FIX: Swap M/K - sidecar header has M=ffn (output), K=hidden (input)
    // For 0.5B: header M=4864 (FFN), K=896 (hidden)
    // But the variable names are swapped in header, so swap them here:
    // g_prt_sidecar_M = rows/sidecar M = FFN output size
    // g_prt_sidecar_N = cols/sidecar N = hidden input size
    int hidden = g_prt_sidecar_N[layer_id];   // 896 (hidden/input)
    int ffn    = g_prt_sidecar_M[layer_id];   // 4864 (FFN/output)
    int n_tokens = (int)cur->ne[1];           // from cur shape

    // Set up per-layer userdata
    PRTUserData * ud = &g_prt_ud_pool[layer_id];
    ud->format    = g_prt_sidecar_format[layer_id];  // 0=float32, 1=int8
    ud->sidecar   = g_prt_sidecar_data[layer_id];   // nullptr for INT8
    ud->int8_data = g_prt_int8_data[layer_id];     // nullptr for float32
    ud->int8_scales = g_prt_int8_scales[layer_id];  // nullptr for float32
    ud->M        = hidden;
    ud->N        = ffn;
    ud->layer_id = layer_id;
    ud->batch    = n_tokens;
    // kernel_mode: AVX2 only for float32 (INT8 uses scalar path)
    ud->kernel_mode = (ud->format == 0 && g_prt_kernel_mode == 1) ? 1 : 0;

    // args[0] = cur (src[0] in the custom op)
    // args[1] = cur (dummy — custom op requires ≥1 src, ggml_custom_4d needs ≥2)
    struct ggml_tensor * args[2];
    args[0] = cur;
    args[1] = cur;

    // Output shape: [ffn, n_tokens] — identical to native ffn_up output
    ggml_tensor * result = ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        ffn,        // ne0 = ffn
        n_tokens,   // ne1 = n_tokens
        1,          // ne2
        1,          // ne3
        args,
        2,
        prt_ffn_up_custom_op,
        1,          // n_tasks = 1 (single-threaded, SIMD inside)
        ud
    );

    char name[32];
    snprintf(name, sizeof(name), "prt_ffn_up.%d", layer_id);
    ggml_set_name(result, name);

    return result;
}
