// Phase 28BR-G: GGML Constant Materialization Probe
// Branch: experimental/prt-phase19a-alt-sidecar-backed
// Verdict: PASS — Method A (no_alloc=false + direct memcpy readback) confirmed working.
// GGML findings:
//   - ggml_backend_tensor_get fails on context-allocated tensors (buffer=nil).
//   - Direct read from tensor->data works fine after compute (buffer=nil but data valid).
//   - ggml_mul_mat(X,R) computes Y = X @ R^T, not X @ R (transpose matmul convention).
//   - ggml_mul_mat requires X.ne[0]==R.ne[0] (square residual K==N).
//   - Compute is clean: nan=0, inf=0, max_abs_err < 1e-6 on all sizes.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

using namespace std;

static float compute_max_abs_err(const float * a, const float * b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
    return max_err;
}

static bool all_finite(const float * a, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(a[i])) return false;
    return true;
}

// GGML orientation:
//   ggml_new_tensor_2d(type, ne0, ne1): ne[0]=ne0 (cols), ne[1]=ne1 (rows)
//   Element (row, col) = data[col*ne[1] + row]  (col-major storage)
//   For R[K,N]: new_tensor(F32, N, K)  -> ne[0]=N, ne[1]=K
//   For X[M,K]: new_tensor(F32, K, M)  -> ne[0]=K, ne[1]=M
//
// ggml_mul_mat(a,b): computes Y = a @ b^T (TRANSPOSE matmul convention)
//   Requires a->ne[0] == b->ne[0]
//   Y has shape [a->ne[1], b->ne[0]] = [rows of a, cols of b]
//   Y_GGML[m,n] = sum_k a[m,k] * b[n,k]
//   For square residual K==N: X.ne[0]=K, R.ne[0]=N, K==N -> mul_mat OK
//
// Row-major reference for GGML matmul (Y_GGML[m,n] = sum_k X[m,k] * R[n,k]):
//   With X[M,K] row-major: X[m*K+k]
//   With R[K,N] row-major: R[k*N+n] but GGML uses R[n*K+k]
//   Y_ref[m*N+n] = sum_k X[m*K+k] * R[n*K+k]
//
// Col-major GGML readback: Y[m,n] = data[col*M + row]
//   So Y_ref[m*N+n] = data[n*M + m]

struct probe_result {
    const char * method;
    const char * notes;
    bool graph_compute_succeeded;
    bool direct_readback_succeeded;
    float max_abs_err_tiny;
    bool finite_tiny;
    float max_abs_err_medium;
    bool finite_medium;
    float max_abs_err_large;
    bool finite_large;
};

// GGML matmul reference: Y_GGML[m,n] = sum_k X[m,k] * R[n,k]
// (row-major C arrays, k indexes the matching inner dimension)
static void ggml_matmul_ref(const float * X_row, const float * R_row,
                             float * Y_row, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += X_row[m * K + k] * R_row[n * K + k];
            }
            Y_row[m * N + n] = sum;
        }
    }
}

// Col-major GGML readback: Y[m,n] = data[n*M + m]
static void readback_col_major(float * Y_out, const float * data, int M, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            Y_out[m * N + n] = data[n * M + m];
        }
    }
}

static void run_test(probe_result * R, const char * label, int K, int M, int N) {
    fprintf(stderr, "\n--- %s (K=%d M=%d N=%d) ---\n", label, K, M, N);

    // Fill with deterministic values (row-major C storage)
    vector<float> R_data(K * N);
    vector<float> X_data(M * K);
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++)
        R_data[k * N + n] = sinf((float)(k * 17 + n * 3 + 1)) * 0.25f;
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++)
        X_data[m * K + k] = sinf((float)(m * 7 + k * 11 + 2)) * 0.5f;

    // GGML matmul reference: Y_GGML[m,n] = sum_k X[m,k] * R[n,k]
    vector<float> Y_ref(M * N);
    ggml_matmul_ref(X_data.data(), R_data.data(), Y_ref.data(), M, K, N);

    // Init GGML context with no_alloc=false (immediate allocation)
    struct ggml_init_params params = { .mem_size = 512*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu || !ctx) { if (ctx) ggml_free(ctx); R->notes = "init failed"; return; }

    // Create tensors: R[K,N]->ne[0]=N,ne[1]=K; X[M,K]->ne[0]=K,ne[1]=M
    struct ggml_tensor * R_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);
    struct ggml_tensor * X_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    if (!R_t || !X_t) { ggml_backend_free(cpu); ggml_free(ctx); R->notes = "tensor creation failed"; return; }

    fprintf(stderr, "R_t: ne[0]=%lld ne[1]=%lld data=%p\n", (long long)R_t->ne[0], (long long)R_t->ne[1], (void*)R_t->data);
    fprintf(stderr, "X_t: ne[0]=%lld ne[1]=%lld data=%p\n", (long long)X_t->ne[0], (long long)X_t->ne[1], (void*)X_t->data);

    // Copy data via memcpy (buffer=nil but data is valid after context allocation)
    memcpy(R_t->data, R_data.data(), (size_t)K * N * sizeof(float));
    memcpy(X_t->data, X_data.data(), (size_t)M * K * sizeof(float));

    // mul_mat: Y = X @ R, Y shape [M, N]
    // GGML computes Y = X @ R^T, so Y_GGML[m,n] = sum_k X[m,k] * R[n,k]
    // X.ne[0]=K, R.ne[0]=N, K==N required for mul_mat
    struct ggml_tensor * Y_t = ggml_mul_mat(ctx, X_t, R_t);
    if (!Y_t) { ggml_backend_free(cpu); ggml_free(ctx); R->notes = "ggml_mul_mat returned null"; return; }

    fprintf(stderr, "Y_t: ne[0]=%lld ne[1]=%lld data=%p buffer=%p\n",
            (long long)Y_t->ne[0], (long long)Y_t->ne[1], (void*)Y_t->data, (void*)Y_t->buffer);

    // Build and compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y_t);

    enum ggml_status st = ggml_backend_graph_compute(cpu, gf);
    R->graph_compute_succeeded = (st == GGML_STATUS_SUCCESS);
    fprintf(stderr, "compute: %s (status=%d)\n", st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED", st);

    if (!R->graph_compute_succeeded) { ggml_backend_free(cpu); ggml_free(ctx); R->notes = "compute failed"; return; }

    // Readback via direct memcpy (ggml_backend_tensor_get fails on buffer=nil result tensor)
    vector<float> Y_computed(M * N);
    readback_col_major(Y_computed.data(), (const float *)Y_t->data, M, N);
    R->direct_readback_succeeded = true;

    float max_err = compute_max_abs_err(Y_computed.data(), Y_ref.data(), M * N);
    bool finite = all_finite(Y_computed.data(), M * N);

    fprintf(stderr, "max_abs_err=%.9f finite=%s\n", max_err, finite ? "YES" : "NO");
    fprintf(stderr, "Y_computed[0..4]: ");
    for (int i = 0; i < min(5, M * N); i++) fprintf(stderr, "%.4f ", Y_computed[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "Y_ref[0..4]:     ");
    for (int i = 0; i < min(5, M * N); i++) fprintf(stderr, "%.4f ", Y_ref[i]);
    fprintf(stderr, "\n");

    if (strcmp(label, "TINY") == 0) { R->max_abs_err_tiny = max_err; R->finite_tiny = finite; }
    else if (strcmp(label, "MEDIUM") == 0) { R->max_abs_err_medium = max_err; R->finite_medium = finite; }
    else { R->max_abs_err_large = max_err; R->finite_large = finite; }

    ggml_backend_free(cpu);
    ggml_free(ctx);
}

int main() {
    fprintf(stderr, "=== Phase 28BR-G: GGML Constant Materialization Probe ===\n");
    fprintf(stderr, "GGML orientation:\n");
    fprintf(stderr, "  ggml_new_tensor_2d(type, ne0, ne1): ne[0]=ne0 (cols), ne[1]=ne1 (rows)\n");
    fprintf(stderr, "  ggml_mul_mat(a,b): computes Y = a @ b^T (TRANSPOSE matmul)\n");
    fprintf(stderr, "    Y_GGML[m,n] = sum_k a[m,k] * b[n,k]\n");
    fprintf(stderr, "    Requires a->ne[0] == b->ne[0]; Y shape [a->ne[1], b->ne[0]]\n");
    fprintf(stderr, "  Square residual K==N: X[M,K].ne[0]=K, R[K,N].ne[0]=N, K==N -> mul_mat OK\n");
    fprintf(stderr, "  Result tensor buffer=nil after compute, but tensor->data is valid.\n");
    fprintf(stderr, "  Readback: ggml_backend_tensor_get fails; direct memcpy from data works.\n\n");

    probe_result R;
    memset(&R, 0, sizeof(R));
    R.method = "MethodA-no_alloc_false+direct_memcpy";

    run_test(&R, "TINY",   4,   5,  4);
    run_test(&R, "MEDIUM", 32,  48, 32);
    run_test(&R, "LARGE",  64,  96, 64);

    fprintf(stderr, "\n=== SUMMARY ===\n");
    fprintf(stderr, "Method:                      %s\n", R.method);
    fprintf(stderr, "graph_compute_succeeded:      %s\n", R.graph_compute_succeeded ? "YES" : "NO");
    fprintf(stderr, "direct_readback_succeeded:    %s\n", R.direct_readback_succeeded ? "YES" : "NO");
    fprintf(stderr, "tiny:   max_abs_err=%.9f finite=%s\n", R.max_abs_err_tiny, R.finite_tiny ? "YES" : "NO");
    fprintf(stderr, "medium: max_abs_err=%.9f finite=%s\n", R.max_abs_err_medium, R.finite_medium ? "YES" : "NO");
    fprintf(stderr, "large:  max_abs_err=%.9f finite=%s\n", R.max_abs_err_large, R.finite_large ? "YES" : "NO");
    if (R.notes) fprintf(stderr, "notes: %s\n", R.notes);

    bool pass = R.graph_compute_succeeded && R.direct_readback_succeeded &&
                R.finite_tiny && R.finite_medium && R.finite_large &&
                R.max_abs_err_tiny < 1e-5f && R.max_abs_err_medium < 1e-5f &&
                R.max_abs_err_large < 1e-5f;
    fprintf(stderr, "\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}