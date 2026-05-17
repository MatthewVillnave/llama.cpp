// PRT Phase 22A: AVX2 microkernel for GGML_OP_PRT_FFN_UP
// Strategy A: vectorize over output columns j
// Y[M,N] = W[K,M]^T @ X[K,N]
// Row-major: Y[j,n] = sum_k W[k,j] * X[k,n]
//
// AVX2: 8 floats per __m256, FMA available
//
// Constraints:
//   - X: [K, N] f32, contiguous row-major
//   - W: [K, M] f32, contiguous row-major
//   - Y: [M, N] f32, contiguous row-major
//   - scales: MUST be NULL (scales baked into f32 W during sidecar decode)
//   - N=1 and N=2 fast paths; N>2 falls back to scalar
//   - Only enabled when PRT_V2_AVX2=1 env var is set

// Phase 22K: N=2 AVX2 kernel implementation
#include <time.h>
static int prt_ffn_up_avx2_call_count = 0;

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

// Handles N=1 and N=2; N>2 falls back to scalar in caller
static inline void ggml_compute_forward_prt_ffn_up_avx2(
        const int K,
        const int M,
        const int n_tokens,
        const float * X,    // [K, n_tokens] row-major
        const float * W,    // [K, M] row-major
        const float * scales, // [M] or NULL - MUST BE NULL for AVX2 path
        float * Y)          // [M, n_tokens] row-major
{
    prt_ffn_up_avx2_call_count++;
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    
    // N=1 or N=2 fast paths; N>2 falls back to scalar
    if (n_tokens > 2) {
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        long long us = (ts_end.tv_sec - ts_start.tv_sec) * 1000000LL + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000LL;
        fprintf(stderr, "[PRT_V2_AVX2_KERNEL] call=%d K=%d M=%d N=%d us=%lld REJECTED_N_GT_2\n",
                prt_ffn_up_avx2_call_count, K, M, n_tokens, us);
        return;
    }

    int j = 0;
    
    // =====================
    // N=1 fast path
    // =====================
    if (n_tokens == 1) {
        // 64 at a time (8 blocks of 8)
        for (; j + 64 <= M; j += 64) {
            __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
            __m256 a4=_mm256_setzero_ps(), a5=_mm256_setzero_ps(), a6=_mm256_setzero_ps(), a7=_mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                float xv = X[k];
                __m256 xb = _mm256_set1_ps(xv);
                const float* Wk = W + k * M + j;
                a0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a0);
                a1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+8), a1);
                a2 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+16), a2);
                a3 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+24), a3);
                a4 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+32), a4);
                a5 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+40), a5);
                a6 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+48), a6);
                a7 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+56), a7);
            }
            _mm256_storeu_ps(Y+j, a0);
            _mm256_storeu_ps(Y+j+8, a1);
            _mm256_storeu_ps(Y+j+16, a2);
            _mm256_storeu_ps(Y+j+24, a3);
            _mm256_storeu_ps(Y+j+32, a4);
            _mm256_storeu_ps(Y+j+40, a5);
            _mm256_storeu_ps(Y+j+48, a6);
            _mm256_storeu_ps(Y+j+56, a7);
        }
        
        // 16 at a time (2 blocks)
        for (; j + 16 <= M; j += 16) {
            __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                float xv = X[k];
                __m256 xb = _mm256_set1_ps(xv);
                const float* Wk = W + k * M + j;
                a0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+0), a0);
                a1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(Wk+8), a1);
            }
            _mm256_storeu_ps(Y+j, a0);
            _mm256_storeu_ps(Y+j+8, a1);
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
            for (int k = 0; k < K; k++) acc += X[k] * W[k*M + j];
            Y[j] = acc;
        }
    }
    
    // =====================
    // N=2 fast path (Phase 22K)
    // X: [K, 2] row-major, X[k*2 + n]
    // Y: [M, 2] row-major, Y[j*2 + n]
    // =====================
    else { // n_tokens == 2
        // 64 at a time (8 blocks of 8 columns)
        for (; j + 64 <= M; j += 64) {
            // Separate accumulators for token0 and token1
            __m256 a0t=_mm256_setzero_ps(), a1t=_mm256_setzero_ps(), a2t=_mm256_setzero_ps(), a3t=_mm256_setzero_ps();
            __m256 a4t=_mm256_setzero_ps(), a5t=_mm256_setzero_ps(), a6t=_mm256_setzero_ps(), a7t=_mm256_setzero_ps();
            __m256 a0n=_mm256_setzero_ps(), a1n=_mm256_setzero_ps(), a2n=_mm256_setzero_ps(), a3n=_mm256_setzero_ps();
            __m256 a4n=_mm256_setzero_ps(), a5n=_mm256_setzero_ps(), a6n=_mm256_setzero_ps(), a7n=_mm256_setzero_ps();
            
            for (int k = 0; k < K; k++) {
                float x0 = X[k*2 + 0];
                float x1 = X[k*2 + 1];
                __m256 xb0 = _mm256_set1_ps(x0);
                __m256 xb1 = _mm256_set1_ps(x1);
                const float* Wk = W + k * M + j;
                __m256 w0 = _mm256_loadu_ps(Wk+0);
                __m256 w1 = _mm256_loadu_ps(Wk+8);
                __m256 w2 = _mm256_loadu_ps(Wk+16);
                __m256 w3 = _mm256_loadu_ps(Wk+24);
                __m256 w4 = _mm256_loadu_ps(Wk+32);
                __m256 w5 = _mm256_loadu_ps(Wk+40);
                __m256 w6 = _mm256_loadu_ps(Wk+48);
                __m256 w7 = _mm256_loadu_ps(Wk+56);
                // Token 0: acc += x0 * W
                a0t = _mm256_fmadd_ps(xb0, w0, a0t);
                a1t = _mm256_fmadd_ps(xb0, w1, a1t);
                a2t = _mm256_fmadd_ps(xb0, w2, a2t);
                a3t = _mm256_fmadd_ps(xb0, w3, a3t);
                a4t = _mm256_fmadd_ps(xb0, w4, a4t);
                a5t = _mm256_fmadd_ps(xb0, w5, a5t);
                a6t = _mm256_fmadd_ps(xb0, w6, a6t);
                a7t = _mm256_fmadd_ps(xb0, w7, a7t);
                // Token 1: acc += x1 * W
                a0n = _mm256_fmadd_ps(xb1, w0, a0n);
                a1n = _mm256_fmadd_ps(xb1, w1, a1n);
                a2n = _mm256_fmadd_ps(xb1, w2, a2n);
                a3n = _mm256_fmadd_ps(xb1, w3, a3n);
                a4n = _mm256_fmadd_ps(xb1, w4, a4n);
                a5n = _mm256_fmadd_ps(xb1, w5, a5n);
                a6n = _mm256_fmadd_ps(xb1, w6, a6n);
                a7n = _mm256_fmadd_ps(xb1, w7, a7n);
            }
            // Store token0: Y[j*2 + 0], Y[(j+8)*2 + 0], ...
            _mm256_storeu_ps(Y + j*2, a0t);
            _mm256_storeu_ps(Y + (j+8)*2, a1t);
            _mm256_storeu_ps(Y + (j+16)*2, a2t);
            _mm256_storeu_ps(Y + (j+24)*2, a3t);
            _mm256_storeu_ps(Y + (j+32)*2, a4t);
            _mm256_storeu_ps(Y + (j+40)*2, a5t);
            _mm256_storeu_ps(Y + (j+48)*2, a6t);
            _mm256_storeu_ps(Y + (j+56)*2, a7t);
            // Store token1: Y[j*2 + 1], Y[(j+8)*2 + 1], ...
            _mm256_storeu_ps(Y + j*2 + 1, a0n);
            _mm256_storeu_ps(Y + (j+8)*2 + 1, a1n);
            _mm256_storeu_ps(Y + (j+16)*2 + 1, a2n);
            _mm256_storeu_ps(Y + (j+24)*2 + 1, a3n);
            _mm256_storeu_ps(Y + (j+32)*2 + 1, a4n);
            _mm256_storeu_ps(Y + (j+40)*2 + 1, a5n);
            _mm256_storeu_ps(Y + (j+48)*2 + 1, a6n);
            _mm256_storeu_ps(Y + (j+56)*2 + 1, a7n);
        }
        
        // 16 at a time (2 blocks)
        for (; j + 16 <= M; j += 16) {
            __m256 a0t=_mm256_setzero_ps(), a1t=_mm256_setzero_ps();
            __m256 a0n=_mm256_setzero_ps(), a1n=_mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                float x0 = X[k*2 + 0];
                float x1 = X[k*2 + 1];
                __m256 xb0 = _mm256_set1_ps(x0);
                __m256 xb1 = _mm256_set1_ps(x1);
                const float* Wk = W + k * M + j;
                __m256 w0 = _mm256_loadu_ps(Wk+0);
                __m256 w1 = _mm256_loadu_ps(Wk+8);
                a0t = _mm256_fmadd_ps(xb0, w0, a0t);
                a1t = _mm256_fmadd_ps(xb0, w1, a1t);
                a0n = _mm256_fmadd_ps(xb1, w0, a0n);
                a1n = _mm256_fmadd_ps(xb1, w1, a1n);
            }
            // Store token0 first, then token1 (avoid overlapping stores)
            _mm256_storeu_ps(Y + j*2 + 0, a0t);
            _mm256_storeu_ps(Y + (j+8)*2 + 0, a1t);
            _mm256_storeu_ps(Y + j*2 + 1, a0n);
            _mm256_storeu_ps(Y + (j+8)*2 + 1, a1n);
        }
        
        // 8 at a time (1 block)
        for (; j + 8 <= M; j += 8) {
            __m256 a0t = _mm256_setzero_ps();
            __m256 a0n = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                float x0 = X[k*2 + 0];
                float x1 = X[k*2 + 1];
                __m256 w = _mm256_loadu_ps(W + k*M + j);
                a0t = _mm256_fmadd_ps(_mm256_set1_ps(x0), w, a0t);
                a0n = _mm256_fmadd_ps(_mm256_set1_ps(x1), w, a0n);
            }
            // Store token0 then token1 (avoid overlapping stores)
            _mm256_storeu_ps(Y + j*2 + 0, a0t);
            _mm256_storeu_ps(Y + j*2 + 1, a0n);
        }
        
        // Scalar tail
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
        
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        long long us = (ts_end.tv_sec - ts_start.tv_sec) * 1000000LL + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000LL;
        fprintf(stderr, "[PRT_V2_AVX2_KERNEL] call=%d K=%d M=%d N=%d us=%lld\n",
                prt_ffn_up_avx2_call_count, K, M, n_tokens, us);
        return;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    long long us = (ts_end.tv_sec - ts_start.tv_sec) * 1000000LL + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000LL;
    fprintf(stderr, "[PRT_V2_AVX2_KERNEL] call=%d K=%d M=%d N=%d us=%lld\n",
            prt_ffn_up_avx2_call_count, K, M, n_tokens, us);
}

#endif // __AVX2__ && __FMA__