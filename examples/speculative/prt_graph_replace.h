// PRT Phase 11BB: Route A — GGML Custom Op for True Replacement
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
struct PRTUserData {
    const float * sidecar;   // [ffn * hidden] float32
    int M;                   // hidden = 2048
    int N;                   // ffn = 11008
    int layer_id;
    int batch;               // n_tokens
    int kernel_mode;         // 0=scalar, 1=AVX2
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
    if (!ud || !ud->sidecar) {
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT-11BB] ERROR: custom op called without sidecar!\n");
            fflush(g_prt_log_file);
        } else {
            fprintf(stderr, "[PRT-11BB] ERROR: custom op called without sidecar!\n");
        }
        return;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const float * X = (const float *)src0->data;
    int hidden = ud->M;   // 2048
    int ffn    = ud->N;   // 11008
    int n_tokens = ud->batch;
    float * Y = (float *)dst->data;

    // Auth debug: dump tensor metadata every call
    if (g_prt_log_file) {
        fprintf(g_prt_log_file, "[PRT-11BB] custom op: dst=%s src0=%s ne=[%lld,%lld] src_ne=[%lld,%lld]\n",
                dst->name,
                src0->name,
                (long long)dst->ne[0], (long long)dst->ne[1],
                (long long)src0->ne[0], (long long)src0->ne[1]);
        fflush(g_prt_log_file);
    } else {
        fprintf(stderr, "[PRT-11BB] custom op: dst=%s src0=%s ne=[%lld,%lld] src_ne=[%lld,%lld]\n",
                dst->name,
                src0->name,
                (long long)dst->ne[0], (long long)dst->ne[1],
                (long long)src0->ne[0], (long long)src0->ne[1]);
    }
    
    // Dump first 4 input values and cur norm
    float sum_in = 0.0f, sum_out = 0.0f;
    for (int i = 0; i < std::min(4, hidden * n_tokens); i++) sum_in += X[i];
    for (int i = 0; i < std::min(4, ffn * n_tokens); i++) sum_out += Y[i];
    if (g_prt_log_file) {
        fprintf(g_prt_log_file, "[PRT-11BB] IL=%d hidden=%d ffn=%d tokens=%d in_sum(4)=%.4f out_sum(4)=%.4f\n",
                ud->layer_id, hidden, ffn, n_tokens, sum_in, sum_out);
        fflush(g_prt_log_file);
    } else {
        fprintf(stderr, "[PRT-11BB] IL=%d hidden=%d ffn=%d tokens=%d in_sum(4)=%.4f out_sum(4)=%.4f\n",
                ud->layer_id, hidden, ffn, n_tokens, sum_in, sum_out);
    }

#if defined(__AVX2__)
    if (ud->kernel_mode == 1) {
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
    } else
#endif
    {
        // Scalar fallback
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

    // Count true replacement invocations
    extern int g_prt_true_replacement_calls;
    g_prt_true_replacement_calls++;
}

// Check if PRT true replacement is active for a given layer
// Mode 5700: all layers
// Mode 5600+L: layer L only
static bool prt_is_true_replacement_layer(int il) {
    extern int g_prt_debug_mode;
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

    if (layer_id < 0 || layer_id >= 36) return nullptr;
    if (!g_prt_sidecar_data[layer_id]) return nullptr;

    int hidden = g_prt_sidecar_M[layer_id];   // 2048
    int ffn    = g_prt_sidecar_N[layer_id];   // 11008
    int n_tokens = (int)cur->ne[1];           // from cur shape

    // Set up per-layer userdata
    PRTUserData * ud = &g_prt_ud_pool[layer_id];
    ud->sidecar  = g_prt_sidecar_data[layer_id];
    ud->M        = hidden;
    ud->N        = ffn;
    ud->layer_id = layer_id;
    ud->batch    = n_tokens;
    ud->kernel_mode = (g_prt_kernel_mode == 1) ? 1 : 0;

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
