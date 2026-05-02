// PRT Phase 11AY: AVX2 SIMD Kernel for Corrected Orientation PRT
// Computes Y[j] = sum_k X[k] * W[j*hidden + k]
// Sidecar is stored as [ffn, hidden], accessed with transposed orientation j*hidden+k
#pragma once

#if defined(__AVX2__)
#include <immintrin.h>

// Helper: horizontal sum of __m256
static inline float hsum256(__m256 v) {
    // Sum even and odd lanes
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

// AVX2 SIMD matmul: Y[j] = sum_k X[k] * W[j*M + k]
// X: [batch, M] input activations, batch is the batch dimension
// W: [N, M] sidecar weights stored as [N*M] (transposed from normal [M,N])
// Y: [batch, N] output
static void matmul_prt_avx2(const float * X, const float * W, float * Y, int batch, int M, int N) {
    // Process 8 outputs at a time (AVX2 vector width = 8 floats)
    const int VLEN = 8;
    
    for (int b = 0; b < batch; b++) {
        const float * X_batch = X + b * M;
        float * Y_batch = Y + b * N;
        
        int j = 0;
        // Main loop: 8 outputs at a time
        for (; j + VLEN - 1 < N; j += VLEN) {
            // For each of the 8 outputs, we compute dot(X, W_row_j)
            // Since W[j*M+k] is contiguous over k, we can load 8 x values
            // and multiply with 8 corresponding weights, then accumulate
            
            __m256 sum[VLEN];
            for (int v = 0; v < VLEN; v++) {
                sum[v] = _mm256_setzero_ps();
            }
            
            int k = 0;
            // Unroll k loop by 4 for better ILP (4 * 8 = 32 elements per iteration)
            for (; k + 31 < M; k += 32) {
                // Load X values (reused for all 8 output computations)
                __m256 x0 = _mm256_loadu_ps(X_batch + k);
                __m256 x1 = _mm256_loadu_ps(X_batch + k + 8);
                __m256 x2 = _mm256_loadu_ps(X_batch + k + 16);
                __m256 x3 = _mm256_loadu_ps(X_batch + k + 24);
                
                // Load W values for each of the 8 outputs
                // W[(j+v)*M + k] for v in 0..7
                for (int v = 0; v < VLEN; v++) {
                    const float * W_row = W + (j + v) * M + k;
                    __m256 w0 = _mm256_loadu_ps(W_row);
                    __m256 w1 = _mm256_loadu_ps(W_row + 8);
                    __m256 w2 = _mm256_loadu_ps(W_row + 16);
                    __m256 w3 = _mm256_loadu_ps(W_row + 24);
                    
                    sum[v] = _mm256_fmadd_ps(x0, w0, sum[v]);
                    sum[v] = _mm256_fmadd_ps(x1, w1, sum[v]);
                    sum[v] = _mm256_fmadd_ps(x2, w2, sum[v]);
                    sum[v] = _mm256_fmadd_ps(x3, w3, sum[v]);
                }
            }
            
            // Handle remaining k values
            for (; k < M; k++) {
                float x_val = X_batch[k];
                for (int v = 0; v < VLEN; v++) {
                    float w_val = W[(j + v) * M + k];
                    sum[v] = _mm256_fmadd_ps(_mm256_set1_ps(x_val), 
                                           _mm256_set1_ps(w_val), 
                                           sum[v]);
                }
            }
            
            // Horizontal sum of each accumulator and store
            for (int v = 0; v < VLEN; v++) {
                Y_batch[j + v] = hsum256(sum[v]);
            }
        }
        
        // Handle tail (N not divisible by 8)
        for (; j < N; j++) {
            float s = 0.0f;
            int k = 0;
            // Unroll by 4
            __m256 sum = _mm256_setzero_ps();
            for (; k + 7 < M; k += 8) {
                __m256 x = _mm256_loadu_ps(X_batch + k);
                __m256 w = _mm256_loadu_ps(W + j * M + k);
                sum = _mm256_fmadd_ps(x, w, sum);
            }
            s = hsum256(sum);
            for (; k < M; k++) {
                s += X_batch[k] * W[j * M + k];
            }
            Y_batch[j] = s;
        }
    }
}

#else
// No AVX2: provide a stub that forces a compile error if called
static void matmul_prt_avx2_placeholder(void) {
    // If this is called on a system without AVX2, it's a build error
    extern void matmul_prt_avx2_requires_AVX2(void);
    matmul_prt_avx2_requires_AVX2();
}
#endif
