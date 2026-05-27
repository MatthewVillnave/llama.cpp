// Phase 28BR-Z: Synthetic non-square orientation probe
// Tests GGML ggml_mul_mat orientation to determine correct layout for non-square residual matmul
#include "ggml/ggml.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

static void mm_ref(const float* A, const float* B, float* C, int N, int K, int M) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++)
                sum += A[i*K + k] * B[k*M + j];
            C[i*M + j] = (float)sum;
        }
}

static float max_abs_err(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(a[i] - b[i]));
    return mx;
}

static float rmse(const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; s += d*d; }
    return (float)sqrt(s / n);
}

static bool all_finite(const float* a, int n) {
    for (int i = 0; i < n; i++) if (std::isnan((double)a[i]) || std::isinf((double)a[i])) return false;
    return true;
}

int main() {
    printf("=== GGML matmul orientation probe ===\n\n");

    // --- Tiny probe: N=2, K=3, M=5 ---
    {
        printf("--- TINY: N=2 K=3 M=5 ---\n");
        const int N = 2, K = 3, M = 5;
        std::vector<float> X(N*K);
        std::vector<float> R(K*M);
        // Simple pattern: X[i*K+j] = i+j, R[k*M+m] = k*m
        for (int i=0;i<N;i++) for(int j=0;j<K;j++) X[i*K+j] = (float)(i+j);
        for (int k=0;k<K;k++) for(int m=0;m<M;m++) R[k*M+m] = (float)(k*m+1);

        float ref[N*M]; memset(ref, 0, sizeof(ref));
        mm_ref(X.data(), R.data(), ref, N, K, M);
        printf("Reference: largest=%.4f, finite=%d\n", ref[0], all_finite(ref,N*M));

        // GGML: Create tensors with explicit shapes
        struct ggml_init_params params = { 0 };
        ggml_context* ctx = ggml_init(params);
        ggml_tensor* gX = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, (int64_t[]){(int64_t)K, (int64_t)N});
        ggml_tensor* gR = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, (int64_t[]){(int64_t)M, (int64_t)K});
        memcpy(gX->data, X.data(), sizeof(float)*N*K);
        memcpy(gR->data, R.data(), sizeof(float)*K*M);
        ggml_tensor* gY = ggml_mul_mat(ctx, gR, gX); // A=ggml_mul_mat(a,b) → a @ b^T ??
        ggml_build_forward_expand(ctx, gY);
        ggml_graph_compute(ctx, &(ggml_cgraph){0});

        // Compare
        float* Y_ggml = (float*)gY->data;
        printf("Shape of result: ggml_mul_mat(R [M×K], X [K×N]) => Y [%lld×%lld]\n",
               (long long)gY->ne[0], (long long)gY->ne[1]);
        float e = max_abs_err(ref, Y_ggml, N*M);
        float r = rmse(ref, Y_ggml, N*M);
        printf("max_abs_err=%.6f  rmse=%.6f  finite=%s\n", e, r, all_finite(Y_ggml,N*M) ? "YES" : "NO");
        printf("\n");

        // Also try: R stored as [K,M] (transposed interpretation)
        ggml_tensor* gR2 = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, (int64_t[]){(int64_t)K, (int64_t)M});
        memcpy(gR2->data, R.data(), sizeof(float)*K*M);
        ggml_tensor* gY2 = ggml_mul_mat(ctx, gR2, gX);
        ggml_build_forward_expand(ctx, gY2);
        ggml_graph_compute(ctx, &(ggml_cgraph){0});
        float* Y_ggml2 = (float*)gY2->data;
        float e2 = max_abs_err(ref, Y_ggml2, N*M);
        float r2 = rmse(ref, Y_ggml2, N*M);
        printf("Shape of result: ggml_mul_mat(R [K×M], X [K×N]) [transposed interp] => Y [%lld×%lld]\n",
               (long long)gY2->ne[0], (long long)gY2->ne[1]);
        printf("max_abs_err=%.6f  rmse=%.6f  finite=%s\n", e2, r2, all_finite(Y_ggml2,N*M) ? "YES" : "NO");

        ggml_free(ctx);
    }

    // --- Medium probe: N=2, K=16, M=32 ---
    {
        printf("\n--- MEDIUM: N=2 K=16 M=32 ---\n");
        const int N = 2, K = 16, M = 32;
        std::vector<float> X(N*K);
        std::vector<float> R(K*M);
        for (int i=0;i<N;i++) for(int j=0;j<K;j++) X[i*K+j] = (float)(i*1.0f + j*0.1f);
        for (int k=0;k<K;k++) for(int m=0;m<M;m++) R[k*M+m] = (float)(k*0.1f + m*0.01f);

        float ref[N*M]; memset(ref, 0, sizeof(ref));
        mm_ref(X.data(), R.data(), ref, N, K, M);

        struct ggml_init_params params = { 0 };
        ggml_context* ctx = ggml_init(params);
        // X as [K,N] col-major in GGML (ne[0]=K, ne[1]=N)
        ggml_tensor* gX = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, (int64_t[]){(int64_t)K, (int64_t)N});
        // R as [M,K]
        ggml_tensor* gR = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, (int64_t[]){(int64_t)M, (int64_t)K});
        memcpy(gX->data, X.data(), sizeof(float)*N*K);
        memcpy(gR->data, R.data(), sizeof(float)*K*M);

        ggml_tensor* gY = ggml_mul_mat(ctx, gR, gX);
        ggml_build_forward_expand(ctx, gY);
        ggml_graph_compute(ctx, &(ggml_cgraph){0});
        float* Y_ggml = (float*)gY->data;
        printf("ggml_mul_mat(R[M×K], X[K×N]) => Y[%lld×%lld] max_err=%.6f rmse=%.6f finite=%s\n",
               (long long)gY->ne[0], (long long)gY->ne[1],
               max_abs_err(ref, Y_ggml, N*M), rmse(ref, Y_ggml, N*M),
               all_finite(Y_ggml,N*M) ? "YES" : "NO");

        // Wrong orientation: R as [K,M]
        ggml_tensor* gR2 = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, (int64_t[]){(int64_t)K, (int64_t)M});
        memcpy(gR2->data, R.data(), sizeof(float)*K*M);
        ggml_tensor* gY2 = ggml_mul_mat(ctx, gR2, gX);
        ggml_build_forward_expand(ctx, gY2);
        ggml_graph_compute(ctx, &(ggml_cgraph){0});
        float* Y_ggml2 = (float*)gY2->data;
        printf("ggml_mul_mat(R[K×M], X[K×N]) => Y[%lld×%lld] max_err=%.6f rmse=%.6f finite=%s\n",
               (long long)gY2->ne[0], (long long)gY2->ne[1],
               max_abs_err(ref, Y_ggml2, N*M), rmse(ref, Y_ggml2, N*M),
               all_finite(Y_ggml2,N*M) ? "YES" : "NO");

        ggml_free(ctx);
    }

    return 0;
}
