// PRT Phase 11R: Minimal ggml Custom-Op Corruption Reproducer
// Goal: Isolate whether corruption comes from ggml_map_custom2, sidecar lifetime, or PRT compute

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// PRT_MAX_LAYERS must match llama-graph.cpp
static const int PRT_MAX_LAYERS = 36;

// Global sidecar storage (mimics llama-graph.cpp)
static const float * g_prt_sidecar_data[PRT_MAX_LAYERS] = {nullptr};
static size_t g_prt_sidecar_bytes[PRT_MAX_LAYERS] = {0};

// Global counters
static int g_op_calls = 0;

// ============================================================================
// TEST 1: Identity copy — no sidecar loaded
// Expected: exact output match
// ============================================================================
static void test1_identity_no_sidecar_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        int ith, int nth, void * userdata) {
    (void)userdata;
    (void)src1;
    if (ith != 0) return;
    g_op_calls++;

    fprintf(stderr, "\n[TEST1] IDENTITY_NO_SIDECAR op entry:\n");
    fprintf(stderr, "  src0: ne[0]=%lld ne[1]=%lld nb[0]=%lld nb[1]=%lld data=%p\n",
            (long long)src0->ne[0], (long long)src0->ne[1],
            (long long)src0->nb[0], (long long)src0->nb[1], src0->data);
    fprintf(stderr, "  dst:  ne[0]=%lld ne[1]=%lld nb[0]=%lld nb[1]=%lld data=%p\n",
            (long long)dst->ne[0], (long long)dst->ne[1],
            (long long)dst->nb[0], (long long)dst->nb[1], dst->data);
    fprintf(stderr, "  src0 == dst ? %s\n", (src0->data == dst->data) ? "YES (same buffer)" : "NO (different buffer)");

    // Checksum before
    float sum_before = 0.0f;
    const float * s = (const float *)src0->data;
    int64_t n = ggml_nelements(src0);
    for (int64_t i = 0; i < n; i++) sum_before += s[i];

    // Identity copy: Y = X
    float * d = (float *)dst->data;
    for (int64_t i = 0; i < n; i++) d[i] = s[i];

    // Checksum after
    float sum_after = 0.0f;
    for (int64_t i = 0; i < n; i++) sum_after += d[i];

    fprintf(stderr, "  n=%lld sum_before=%f sum_after=%f match=%s\n",
            (long long)n, sum_before, sum_after,
            (sum_before == sum_after) ? "YES" : "NO");
    fprintf(stderr, "  dst[0]=%f dst[1]=%f dst[2]=%f\n", d[0], d[1], d[2]);
}

// ============================================================================
// TEST 2: Identity copy — sidecar LOADED but NOT READ
// Expected: exact output match
// If corruption here: sidecar loading/lifetime is the problem
// ============================================================================
static void test2_identity_sidecar_loaded_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        int ith, int nth, void * userdata) {
    (void)userdata;
    (void)src1;
    if (ith != 0) return;
    g_op_calls++;

    // Access global sidecar pointer (but DON'T read from it — just check it's set)
    const float * sidecar = g_prt_sidecar_data[0];

    fprintf(stderr, "\n[TEST2] IDENTITY_SIDECAR_LOADED op entry:\n");
    fprintf(stderr, "  sidecar ptr=%p sidecar_bytes=%zu\n", (void*)sidecar, g_prt_sidecar_bytes[0]);
    fprintf(stderr, "  src0: ne[0]=%lld ne[1]=%lld data=%p\n",
            (long long)src0->ne[0], (long long)src0->ne[1], src0->data);
    fprintf(stderr, "  dst:  ne[0]=%lld ne[1]=%lld data=%p\n",
            (long long)dst->ne[0], (long long)dst->ne[1], dst->data);
    fprintf(stderr, "  src0 == dst ? %s\n", (src0->data == dst->data) ? "YES (same buffer)" : "NO (different buffer)");

    // Checksum before
    float sum_before = 0.0f;
    const float * s = (const float *)src0->data;
    int64_t n = ggml_nelements(src0);
    for (int64_t i = 0; i < n; i++) sum_before += s[i];

    // Identity copy: Y = X
    float * d = (float *)dst->data;
    for (int64_t i = 0; i < n; i++) d[i] = s[i];

    // Checksum after
    float sum_after = 0.0f;
    for (int64_t i = 0; i < n; i++) sum_after += d[i];

    fprintf(stderr, "  n=%lld sum_before=%f sum_after=%f match=%s\n",
            (long long)n, sum_before, sum_after,
            (sum_before == sum_after) ? "YES" : "NO");
    fprintf(stderr, "  dst[0]=%f dst[1]=%f dst[2]=%f\n", d[0], d[1], d[2]);
}

// ============================================================================
// TEST 3: Sidecar READ-ONLY checksum (no write to dst)
// Expected: dst unchanged (identity from src0)
// If corruption here: sidecar pointer/read is corrupting memory
// ============================================================================
static void test3_sidecar_checksum_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        int ith, int nth, void * userdata) {
    (void)userdata;
    (void)src1;
    if (ith != 0) return;
    g_op_calls++;

    const float * sidecar = g_prt_sidecar_data[0];
    size_t sidecar_bytes = g_prt_sidecar_bytes[0];

    fprintf(stderr, "\n[TEST3] SIDECAR_CHECKSUM op entry:\n");
    fprintf(stderr, "  sidecar ptr=%p sidecar_bytes=%zu\n", (void*)sidecar, sidecar_bytes);

    // Compute checksum of sidecar WITHOUT writing to dst
    float sidecar_sum = 0.0f;
    if (sidecar && sidecar_bytes > 0) {
        int64_t sidecar_n = sidecar_bytes / sizeof(float);
        for (int64_t i = 0; i < sidecar_n; i++) {
            sidecar_sum += sidecar[i];
        }
    }
    fprintf(stderr, "  sidecar_sum=%f sidecar_elements=%lld\n",
            sidecar_sum, (long long)(sidecar_bytes / sizeof(float)));

    // Now identity copy src0 to dst (SAME as test1/2)
    const float * s = (const float *)src0->data;
    float * d = (float *)dst->data;
    int64_t n = ggml_nelements(src0);
    for (int64_t i = 0; i < n; i++) d[i] = s[i];

    float sum_after = 0.0f;
    for (int64_t i = 0; i < n; i++) sum_after += d[i];
    fprintf(stderr, "  dst_sum=%f dst[0]=%f dst[1]=%f\n", sum_after, d[0], d[1]);
}

// ============================================================================
// TEST 4: Bounded fill — write exactly nelem values using dst strides
// Expected: no out-of-bounds, no path strings
// ============================================================================
static void test4_bounded_fill_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        int ith, int nth, void * userdata) {
    (void)userdata;
    (void)src0;
    (void)src1;
    if (ith != 0) return;
    g_op_calls++;

    int64_t nelem = ggml_nelements(dst);
    size_t nbytes = ggml_nbytes(dst);

    fprintf(stderr, "\n[TEST4] BOUNDED_FILL op entry:\n");
    fprintf(stderr, "  dst ne[0]=%lld ne[1]=%lld ne[2]=%lld ne[3]=%lld\n",
            (long long)dst->ne[0], (long long)dst->ne[1],
            (long long)dst->ne[2], (long long)dst->ne[3]);
    fprintf(stderr, "  dst nb[0]=%lld nb[1]=%lld nb[2]=%lld nb[3]=%lld\n",
            (long long)dst->nb[0], (long long)dst->nb[1],
            (long long)dst->nb[2], (long long)dst->nb[3]);
    fprintf(stderr, "  dst data=%p nelem=%lld nbytes=%zu\n",
            dst->data, (long long)nelem, nbytes);

    float * d = (float *)dst->data;

    // Write bounded: 0.001 * i
    for (int64_t i = 0; i < nelem; i++) {
        d[i] = 0.001f * (float)i;
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < nelem; i++) sum += d[i];
    fprintf(stderr, "  filled nelem=%lld sum=%f d[0]=%f d[1]=%f d[2]=%f\n",
            (long long)nelem, sum, d[0], d[1], d[2]);
    fprintf(stderr, "  bytes written: %zu (should match ggml_nbytes)\n", nelem * sizeof(float));
}

// ============================================================================
// TEST 5: Tiny PRT compute on small synthetic dimensions
// Compare custom-op output against standalone PRT function
// ============================================================================
// PRT thresholds (must match what we used in Phase 10E)
static const float PRT_T0 = 2.0f;
static const float PRT_T1 = 0.5f;
static const float PRT_T2 = 0.1f;

static void prt_standalone_small(float * Y, const float * X, const float * W,
                                   int64_t hidden, int64_t ffn, int64_t batch) {
    // Y[j] = sum_k X[k] * W[k,j] where |X[k]| > PRT_T2
    for (int64_t j = 0; j < ffn; j++) {
        float sum = 0.0f;
        for (int64_t k = 0; k < hidden; k++) {
            float x_val = X[k * batch];  // column-major
            if (fabsf(x_val) > PRT_T2) {
                sum += x_val * W[k * ffn + j];
            }
        }
        Y[j] = sum;
    }
}

static void test5_prt_compute_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        int ith, int nth, void * userdata) {
    (void)userdata;
    if (ith != 0) return;
    g_op_calls++;

    const float * sidecar = g_prt_sidecar_data[0];
    const float * X = (const float *)src1->data;  // [hidden, batch]
    float * Y_dst = (float *)dst->data;            // [ffn, batch]

    int64_t hidden = src1->ne[0];
    int64_t batch = src1->ne[1] > 1 ? src1->ne[1] : 1;
    int64_t ffn = dst->ne[0];

    fprintf(stderr, "\n[TEST5] PRT_COMPUTE op entry:\n");
    fprintf(stderr, "  X: ne[0]=%lld ne[1]=%lld data=%p\n",
            (long long)src1->ne[0], (long long)src1->ne[1], src1->data);
    fprintf(stderr, "  dst: ne[0]=%lld ne[1]=%lld data=%p\n",
            (long long)dst->ne[0], (long long)dst->ne[1], dst->data);
    fprintf(stderr, "  sidecar ptr=%p\n", (void*)sidecar);
    fprintf(stderr, "  hidden=%lld ffn=%lld batch=%lld\n",
            (long long)hidden, (long long)ffn, (long long)batch);

    // Compute PRT using sidecar
    for (int64_t j = 0; j < ffn; j++) {
        float sum = 0.0f;
        for (int64_t k = 0; k < hidden; k++) {
            float x_val = X[k * batch];
            if (fabsf(x_val) > PRT_T2) {
                sum += x_val * sidecar[k * ffn + j];
            }
        }
        Y_dst[j] = sum;
    }

    // Compute standalone PRT for comparison
    float Y_standalone[1024] = {0};  // bounded buffer
    int64_t ffn_cmp = std::min(ffn, (int64_t)1024);
    prt_standalone_small(Y_standalone, X, sidecar, hidden, ffn_cmp, batch);

    // Compare first few
    float sum_dst = 0.0f, sum_std = 0.0f;
    for (int64_t i = 0; i < ffn_cmp; i++) {
        sum_dst += Y_dst[i];
        sum_std += Y_standalone[i];
    }
    fprintf(stderr, "  Y_dst sum=%f Y_standalone sum=%f diff=%f\n",
            sum_dst, sum_std, fabsf(sum_dst - sum_std));
    fprintf(stderr, "  Y_dst[0]=%f Y_standalone[0]=%f\n",
            Y_dst[0], Y_standalone[0]);
    fprintf(stderr, "  Y_dst[1]=%f Y_standalone[1]=%f\n",
            Y_dst[1], Y_standalone[1]);
}

// ============================================================================
// Load sidecar file into global (mimics llama_set_prt_sidecar)
// ============================================================================
static bool load_sidecar(const char * path, int layer) {
    if (layer < 0 || layer >= PRT_MAX_LAYERS) return false;

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    float * data = (float *)malloc(size);
    if (!data) {
        fclose(f);
        return false;
    }

    fread(data, 1, size, f);
    fclose(f);

    g_prt_sidecar_data[layer] = data;
    g_prt_sidecar_bytes[layer] = size;

    fprintf(stderr, "Loaded sidecar layer %d: %zu bytes, %lld floats\n",
            layer, size, (long long)(size / sizeof(float)));
    return true;
}

// ============================================================================
// Check output for path-string-like bytes
// ============================================================================
static void check_output_bytes(const char * name, void * data, int64_t n_elem) {
    uint8_t * bytes = (uint8_t *)data;
    int64_t n_bytes = n_elem * (int64_t)sizeof(float);
    int weird_count = 0;
    for (int64_t i = 0; i < n_bytes; i++) {
        uint8_t b = bytes[i];
        // Check for '/' (0x2F), '\' (0x5C), '.' (0x2E) which are common in paths
        if (b == 0x2F || b == 0x5C || b == 0x2E) {
            weird_count++;
            if (weird_count <= 5) {
                fprintf(stderr, "  [PATH_BYTE] %s: offset %lld: 0x%02X\n",
                        name, (long long)i, b);
            }
        }
        // Also check for low printable ASCII that could be path fragments
        if (b >= 32 && b < 127 && weird_count <= 5) {
            // Check if this looks like a path separator sequence
            if (i > 0 && i < n_bytes - 1) {
                uint8_t prev = bytes[i-1];
                uint8_t next = bytes[i+1];
                if ((prev == '/' || prev == '\\') && (next == '/' || next == '\\' || next == '.')) {
                    fprintf(stderr, "  [PATH_SEQ] %s: offset %lld: '%c%c%c'\n",
                            name, (long long)i, prev, b, next);
                }
            }
        }
    }
    fprintf(stderr, "  %s: total path-like bytes: %d / %ld\n", name, weird_count, (long)n_bytes);
}

// ============================================================================
// Main test runner
// ============================================================================
int main(int argc, char ** argv) {
    fprintf(stderr, "=== PRT Phase 11R: Minimal ggml Custom-Op Corruption Reproducer ===\n\n");

    // Determine test number
    int test = 1;
    if (argc > 1) {
        test = atoi(argv[1]);
    }
    fprintf(stderr, "Running TEST %d\n", test);

    // Load sidecar if provided (for tests 2-5)
    const char * sidecar_path = nullptr;
    if (test >= 2) {
        sidecar_path = (argc > 2) ? argv[2] : "/tmp/prt_sidecars/ffn_up_layer0_prt.bin";
        fprintf(stderr, "Loading sidecar from: %s\n", sidecar_path);
        if (!load_sidecar(sidecar_path, 0)) {
            fprintf(stderr, "WARNING: could not load sidecar — continuing without\n");
        }
    } else {
        fprintf(stderr, "TEST 1: No sidecar loaded\n");
    }

    // Dimensions (tiny but representative)
    const int64_t hidden = 32;    // Much smaller than 2048
    const int64_t ffn    = 64;    // Much smaller than 11008
    const int64_t batch  = 1;

    // Create ggml context
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: ggml_init failed\n");
        return 1;
    }

    // Create CPU backend
    struct ggml_backend * backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "ERROR: ggml_backend_init_by_type failed\n");
        ggml_free(ctx);
        return 1;
    }

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Input X: [hidden, batch]
    struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, batch);
    ggml_set_name(X, "X_input");
    float * X_data = (float *)X->data;
    for (int64_t i = 0; i < hidden * batch; i++) {
        X_data[i] = 0.1f * (i + 1);
    }
    ggml_build_forward_expand(gf, X);

    // Matmul: X @ W = Y_matmul
    // X: [hidden, batch] = [32, 1]
    // W: [hidden, ffn] = [32, 64] (transposed from [ffn, hidden] for ggml)
    // Result: [ffn, batch] = [64, 1]
    struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, ffn);
    ggml_set_name(W, "W_matmul");
    float * W_data = (float *)W->data;
    for (int64_t i = 0; i < hidden * ffn; i++) W_data[i] = 1.0f;  // All 1.0 for easy checksum
    struct ggml_tensor * Y_matmul = ggml_mul_mat(ctx, W, X);
    ggml_set_name(Y_matmul, "Y_matmul");
    ggml_build_forward_expand(gf, Y_matmul);

    // Add custom op based on test
    struct ggml_tensor * Y_out = NULL;
    const char * test_name = "";

    switch (test) {
        case 1:
            Y_out = ggml_map_custom2(ctx, Y_matmul, X, test1_identity_no_sidecar_op, 1, nullptr);
            test_name = "IDENTITY_NO_SIDECAR";
            break;
        case 2:
            Y_out = ggml_map_custom2(ctx, Y_matmul, X, test2_identity_sidecar_loaded_op, 1, nullptr);
            test_name = "IDENTITY_SIDECAR_LOADED";
            break;
        case 3:
            Y_out = ggml_map_custom2(ctx, Y_matmul, X, test3_sidecar_checksum_op, 1, nullptr);
            test_name = "SIDECAR_CHECKSUM";
            break;
        case 4:
            Y_out = ggml_map_custom2(ctx, Y_matmul, X, test4_bounded_fill_op, 1, nullptr);
            test_name = "BOUNDED_FILL";
            break;
        case 5:
            Y_out = ggml_map_custom2(ctx, Y_matmul, X, test5_prt_compute_op, 1, nullptr);
            test_name = "PRT_COMPUTE";
            break;
        case 6: {
            // TEST 6: Sidecar READ + WRITE to same dst (like Phase 10E does)
            // Read from sidecar, write to dst — check if sidecar leaks into dst
            struct ggml_tensor * Y_out6 = ggml_map_custom2(ctx, Y_matmul, X, test3_sidecar_checksum_op, 1, nullptr);
            ggml_set_name(Y_out6, "Y_out_test6");
            Y_out = Y_out6;
            test_name = "SIDECAR_READ_WRITE";
            break;
        }
        default:
            fprintf(stderr, "Unknown test %d\n", test);
            return 1;
    }

    ggml_set_name(Y_out, "Y_out");
    ggml_build_forward_expand(gf, Y_out);

    fprintf(stderr, "\n[MAIN] Graph: %d nodes\n", ggml_graph_n_nodes(gf));
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        fprintf(stderr, "  node[%d]: op=%-20s name=%s\n",
                i, ggml_op_name(node->op),
                node->name ? node->name : "(null)");
    }

    fprintf(stderr, "\n[MAIN] Computing graph (test=%d: %s)...\n", test, test_name);
    int64_t t0 = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    int64_t t1 = ggml_time_us();
    fprintf(stderr, "[MAIN] Compute time: %.2f ms\n", (t1-t0)/1000.0);

    fprintf(stderr, "\n[MAIN] Custom op calls: %d\n", g_op_calls);

    // Print output summary
    float * Y_data = (float *)Y_out->data;
    int64_t Y_n = ggml_nelements(Y_out);
    float Y_sum = 0.0f;
    for (int64_t i = 0; i < Y_n; i++) Y_sum += Y_data[i];

    fprintf(stderr, "\n[MAIN] Output Y_out:\n");
    fprintf(stderr, "  data=%p nelem=%lld\n", Y_data, (long long)Y_n);
    fprintf(stderr, "  sum=%f\n", Y_sum);
    fprintf(stderr, "  Y[0]=%f Y[1]=%f Y[2]=%f\n", Y_data[0], Y_data[1], Y_data[2]);
    if (Y_n > 3) {
        fprintf(stderr, "  Y[%lld]=%f (last)\n", (long long)Y_n-1, Y_data[Y_n-1]);
    }

    // Check for path-string-like bytes
    check_output_bytes("Y_out", Y_data, Y_n);

    // Expected values for test-specific validation
    fprintf(stderr, "\n[MAIN] Validation:\n");
    switch (test) {
        case 1:
        case 2:
        case 3: {
            // Expected: Y = W @ X, W=all 1.0, X=0.1,0.2,..., hidden=32, batch=1
            // Y[j] = sum_k W[k,j] * X[k] = sum_k 1.0 * (0.1*(k+1)) = 0.1 * sum_k (k+1)
            //      = 0.1 * (1+2+...+32) = 0.1 * 528 = 52.8
            float expected = 0.1f * (hidden * (hidden + 1) / 2);
            fprintf(stderr, "  Expected Y[j] (all same): %f\n", expected);
            fprintf(stderr, "  Actual Y[0]=%f diff=%f\n", Y_data[0], fabsf(Y_data[0] - expected));
            if (fabsf(Y_data[0] - expected) < 0.001f) {
                fprintf(stderr, "  RESULT: PASS (exact match)\n");
            } else {
                fprintf(stderr, "  RESULT: FAIL (mismatch)\n");
            }
            break;
        }
        case 4: {
            // Expected: Y[i] = 0.001 * i
            fprintf(stderr, "  Expected Y[0]=%f Y[1]=%f Y[2]=%f\n", 0.0f, 0.001f, 0.002f);
            fprintf(stderr, "  Actual   Y[0]=%f Y[1]=%f Y[2]=%f\n", Y_data[0], Y_data[1], Y_data[2]);
            if (fabsf(Y_data[0] - 0.0f) < 0.0001f && fabsf(Y_data[1] - 0.001f) < 0.0001f && fabsf(Y_data[2] - 0.002f) < 0.0001f) {
                fprintf(stderr, "  RESULT: PASS (exact match)\n");
            } else {
                fprintf(stderr, "  RESULT: FAIL (mismatch)\n");
            }
            break;
        }
        case 5:
            fprintf(stderr, "  See TEST5 output above for PRT comparison\n");
            break;
    }

    // Cleanup
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (g_prt_sidecar_data[0]) {
        free((void *)g_prt_sidecar_data[0]);
    }

    fprintf(stderr, "\n=== TEST %d (%s) COMPLETE ===\n", test, test_name);
    return 0;
}
