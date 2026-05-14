// PRT Phase 21B-R: synthetic correctness test for GGML_OP_PRT_FFN_UP
// Each test gets its own fresh context

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

static int run_test(int K, int M, int N,
                    const float * x_vals,
                    const float * w_vals,
                    const float * s_vals) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL: ggml_init\n");
        return -1;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        printf("FAIL: ggml_backend_cpu_init\n");
        ggml_free(ctx);
        return -1;
    }

    // Create all tensors
    struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, 1);
    struct ggml_tensor * result = ggml_prt_ffn_up(ctx, X, W, scales, K, M);
    if (!result) {
        printf("FAIL: ggml_prt_ffn_up returned NULL\n");
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return -1;
    }

    // Allocate buffer
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        printf("FAIL: alloc_buffer\n");
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return -1;
    }

    // Initialize tensor data
    ggml_backend_tensor_set(X, x_vals, 0, sizeof(float) * K * N);
    ggml_backend_tensor_set(W, w_vals, 0, sizeof(float) * K * M);
    ggml_backend_tensor_set(scales, s_vals, 0, sizeof(float) * M);

    // Verify op params
    int32_t * op_params = (int32_t *)result->op_params;
    printf("  K=%d M=%d (stored op_params[0]=%d [1]=%d)\n",
           K, M, op_params[0], op_params[1]);

    // Compute expected
    float expected[M * N];
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < M; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float x_val = x_vals[k * N + n];
                float w_val = w_vals[k * M + j];
                float s_val = s_vals[j];
                acc += x_val * w_val * s_val;
            }
            expected[j * N + n] = acc;
        }
    }

    printf("  Expected:");
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < M; j++) {
            printf(" %.4f", expected[j * N + n]);
        }
    }
    printf("\n");

    // Build and compute
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_backend_graph_compute(cpu, gf);

    // Get result
    float output_vals[M * N];
    ggml_backend_tensor_get(result, output_vals, 0, sizeof(output_vals));

    printf("  Computed:");
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < M; j++) {
            printf(" %.4f", output_vals[j * N + n]);
        }
    }
    printf("\n");

    float max_abs_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(output_vals[i] - expected[i]);
        if (err > max_abs_error) max_abs_error = err;
    }

    printf("  Max abs error: %.9f\n", (double)max_abs_error);
    int pass = (max_abs_error < 1e-5f);
    printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    return pass;
}

int main(void) {
    printf("=== PRT Phase 21B-R Synthetic Correctness Test ===\n\n");

    int n_pass = 0;
    int n_total = 0;

    // Test N=1
    {
        n_total++;
        const int K = 8, M = 4, N = 1;
        printf("--- Test N=%d ---\n", N);

        float x_vals[K * N];
        float w_vals[K * M];
        float s_vals[M];

        for (int i = 0; i < K * N; i++) x_vals[i] = 0.0f;
        x_vals[0] = 1.0f; x_vals[1] = 0.5f;

        for (int i = 0; i < K * M; i++) w_vals[i] = (float)(i + 1);

        s_vals[0] = 1.0f; s_vals[1] = 0.5f; s_vals[2] = -1.0f; s_vals[3] = 2.0f;

        int r = run_test(K, M, N, x_vals, w_vals, s_vals);
        if (r > 0) n_pass++;
    }

    // Test N=2
    {
        n_total++;
        const int K = 8, M = 4, N = 2;
        printf("--- Test N=%d ---\n", N);

        float x_vals[K * N];
        float w_vals[K * M];
        float s_vals[M];

        for (int i = 0; i < K * N; i++) x_vals[i] = 0.0f;
        x_vals[0] = 1.0f; x_vals[1] = 0.5f;  // X[0,0]=1, X[1,0]=0.5

        for (int i = 0; i < K * M; i++) w_vals[i] = (float)(i + 1);

        s_vals[0] = 1.0f; s_vals[1] = 0.5f; s_vals[2] = -1.0f; s_vals[3] = 2.0f;

        int r = run_test(K, M, N, x_vals, w_vals, s_vals);
        if (r > 0) n_pass++;
    }

    printf("=== Summary: %d/%d passed ===\n", n_pass, n_total);
    return (n_pass == n_total) ? 0 : 1;
}