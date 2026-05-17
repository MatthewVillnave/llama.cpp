// PRT Phase 22K-S: AVX2 microkernel for GGML_OP_PRT_FFN_UP (TWO-PASS + TEMP STORE)
// Strategy A: vectorize over output columns j
// Y[M,N] = W[K,M]^T @ X[K,N]
// Row-major: Y[j,n] = sum_k W[k,j] * X[k,n]
//
// AVX2: 8 floats per __m256, FMA available
//
// Phase 22K-S: Temp-array store fix for N=2.
// Accumulate with AVX2, store via temp array + scalar strided writes.
// This ensures Y[j*2+n] layout matching the scalar reference exactly.

#include <stdint.h>

#ifdef __AVX2__
#include <immintrin.h>

void ggml_compute_forward_prt_ffn_up_avx2(
    int K, int M, int n_tokens,
    const float* X, const float* W, const float* scales,
    float* Y, int n_threads
) {
    (void)scales;
    (void)n_threads;

    if (n_tokens == 1) {
        // ================================================
        // N=1 fast path (unchanged, working correctly)
        // ================================================
        int j = 0;

        // 64 at a time (8 blocks of 8 columns)
        for (; j + 64 <= M; j += 64) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
            __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                __m256 xb = _mm256_set1_ps(X[k]);
                const float* Wk = W + k * M + j;
                a0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+ 0), a0);
                a1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+ 8), a1);
                a2 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+16), a2);
                a3 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+24), a3);
                a4 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+32), a4);
                a5 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+40), a5);
                a6 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+48), a6);
                a7 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+56), a7);
            }

            _mm256_storeu_ps(Y + j,      a0);
            _mm256_storeu_ps(Y + j + 8,  a1);
            _mm256_storeu_ps(Y + j + 16, a2);
            _mm256_storeu_ps(Y + j + 24, a3);
            _mm256_storeu_ps(Y + j + 32, a4);
            _mm256_storeu_ps(Y + j + 40, a5);
            _mm256_storeu_ps(Y + j + 48, a6);
            _mm256_storeu_ps(Y + j + 56, a7);
        }

        // 16 at a time (2 blocks)
        for (; j + 16 <= M; j += 16) {
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 xb = _mm256_set1_ps(X[k]);
                const float* Wk = W + k * M + j;
                a0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a0);
                a1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+8), a1);
            }
            _mm256_storeu_ps(Y + j,     a0);
            _mm256_storeu_ps(Y + j + 8, a1);
        }

        // 8 at a time (1 block)
        for (; j + 8 <= M; j += 8) {
            __m256 a0 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(X[k]), _mm256_loadu_ps(W + k*M + j), a0);
            }
            _mm256_storeu_ps(Y + j, a0);
        }

        // Scalar tail
        for (; j < M; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += X[k] * W[k*M + j];
            }
            Y[j] = acc;
        }
    }
    // ================================================
    // N=2 fast path (Phase 22K-S: TEMP STORE FIX)
    // X: [K, 2] row-major, X[k*2 + n]
    // Y: [M, 2] row-major, Y[j*2 + n] for token n at column j
    //
    // Temp-array store approach:
    //   1. Accumulate token0 with AVX2 into acc0[8]
    //   2. Accumulate token1 with AVX2 into acc1[8]
    //   3. Store via scalar strided writes to dst
    //      dst[(j+lane)*2 + 0] = acc0[lane]
    //      dst[(j+lane)*2 + 1] = acc1[lane]
    // This ensures exact Y[j*2+n] layout matching scalar reference.
    // ================================================
    else if (n_tokens == 2) {
        fprintf(stderr, "[PRT_V2_AVX2] path=N2_TEMP_STORE K=%d M=%d N=2\n", K, M);

        int j = 0;
        float tmp0[8];
        float tmp1[8];

        // ---------------------------------------------
        // 64 at a time (8 blocks of 8 columns)
        // ---------------------------------------------
        for (; j + 64 <= M; j += 64) {
            // Process each 8-wide sub-block
            for (int block = 0; block < 8; block++) {
                int j_base = j + block * 8;

                // PASS 0: Token0
                __m256 a0 = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    float x0 = X[k*2 + 0];
                    __m256 xb = _mm256_set1_ps(x0);
                    const float* Wk = W + k * M + j_base;
                    a0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a0);
                }
                _mm256_storeu_ps(tmp0, a0);

                // PASS 1: Token1
                __m256 a1 = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    float x1 = X[k*2 + 1];
                    __m256 xb = _mm256_set1_ps(x1);
                    const float* Wk = W + k * M + j_base;
                    a1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a1);
                }
                _mm256_storeu_ps(tmp1, a1);

                // Scalar strided stores: Y[(j_base+lane)*2 + n] = tmpn[lane]
                for (int lane = 0; lane < 8; lane++) {
                    Y[(j_base + lane)*2 + 0] = tmp0[lane];
                    Y[(j_base + lane)*2 + 1] = tmp1[lane];
                }
            }
        }

        // ---------------------------------------------
        // 16 at a time (2 blocks of 8)
        // ---------------------------------------------
        for (; j + 16 <= M; j += 16) {
            for (int block = 0; block < 2; block++) {
                int j_base = j + block * 8;

                // PASS 0: Token0
                __m256 a0 = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    float x0 = X[k*2 + 0];
                    __m256 xb = _mm256_set1_ps(x0);
                    const float* Wk = W + k * M + j_base;
                    a0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a0);
                }
                _mm256_storeu_ps(tmp0, a0);

                // PASS 1: Token1
                __m256 a1 = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    float x1 = X[k*2 + 1];
                    __m256 xb = _mm256_set1_ps(x1);
                    const float* Wk = W + k * M + j_base;
                    a1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a1);
                }
                _mm256_storeu_ps(tmp1, a1);

                // Scalar strided stores
                for (int lane = 0; lane < 8; lane++) {
                    Y[(j_base + lane)*2 + 0] = tmp0[lane];
                    Y[(j_base + lane)*2 + 1] = tmp1[lane];
                }
            }
        }

        // ---------------------------------------------
        // 8 at a time (1 block of 8)
        // ---------------------------------------------
        for (; j + 8 <= M; j += 8) {
            // PASS 0: Token0
            __m256 a0 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                float x0 = X[k*2 + 0];
                __m256 xb = _mm256_set1_ps(x0);
                const float* Wk = W + k * M + j;
                a0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a0);
            }
            _mm256_storeu_ps(tmp0, a0);

            // PASS 1: Token1
            __m256 a1 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                float x1 = X[k*2 + 1];
                __m256 xb = _mm256_set1_ps(x1);
                const float* Wk = W + k * M + j;
                a1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a1);
            }
            _mm256_storeu_ps(tmp1, a1);

            // Scalar strided stores
            for (int lane = 0; lane < 8; lane++) {
                Y[(j + lane)*2 + 0] = tmp0[lane];
                Y[(j + lane)*2 + 1] = tmp1[lane];
            }
        }

        // Scalar tail (also strided layout)
        for (; j < M; j++) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (int k = 0; k < K; k++) {
                acc0 += X[k*2 + 0] * W[k*M + j];
                acc1 += X[k*2 + 1] * W[k*M + j];
            }
            Y[j*2 + 0] = acc0;
            Y[j*2 + 1] = acc1;
        }

        return;
    }
    else {
        fprintf(stderr, "[PRT_V2_AVX2_REJECT] N=%d > 2, falling back to scalar\n", n_tokens);
    }

    // Scalar fallback (for N != 1,2)
    for (int n = 0; n < n_tokens; n++) {
        for (int j = 0; j < M; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += X[k * n_tokens + n] * W[k * M + j];
            }
            Y[j * n_tokens + n] = acc;
        }
    }
}

#endif // __AVX2__ && __FMA__