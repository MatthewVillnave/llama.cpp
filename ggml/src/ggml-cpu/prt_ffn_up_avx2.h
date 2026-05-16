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
//   - N=1 fast path, N>1 falls back to scalar
//   - Only enabled when PRT_V2_AVX2=1 env var is set

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

// Only handles N=1 (single token), scales must be NULL
// N>1 falls back to scalar in caller
static inline void ggml_compute_forward_prt_ffn_up_avx2(
        const int K,
        const int M,
        const int n_tokens,
        const float * X,    // [K, n_tokens] row-major
        const float * W,    // [K, M] row-major  
        const float * scales, // [M] or NULL - MUST BE NULL for AVX2 path
        float * Y)          // [M, n_tokens] row-major
{
    // Only fast path for N=1; scalar fallback for N>1
    if (n_tokens != 1) {
        return; // caller falls back to scalar
    }

    int j = 0;
    
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

#endif // __AVX2__ && __FMA__