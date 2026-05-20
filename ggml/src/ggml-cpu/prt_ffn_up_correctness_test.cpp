// PRT Phase 22A: Synthetic correctness test for AVX2 vs scalar GGML_OP_PRT_FFN_UP
// Run with: g++ -O3 -mavx2 -mfma -std=c++17 prt_ffn_up_correctness_test.cpp -o /tmp/test && /tmp/test

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

// Simple timing utility
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Scalar reference implementation
void scalar_prt_ffn_up(int K, int M, int N, const float* X, const float* W, float* Y) {
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < M; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += X[k * N + n] * W[k * M + j];
            }
            Y[j * N + n] = acc;
        }
    }
}

// AVX2 implementation (matches ops.cpp pattern)
void avx2_prt_ffn_up(int K, int M, int N, const float* X, const float* W, float* Y) {
    static constexpr int JBLOCK = 8;
    
    for (int n = 0; n < N; n++) {
        int j = 0;
        
        // 56 columns at a time (7 blocks of 8)
        for (; j + 56 <= M; j += 56) {
            __m256 acc[7];
            for (int b = 0; b < 7; b++) acc[b] = _mm256_setzero_ps();
            
            for (int k = 0; k < K; k++) {
                float x_val = X[k * N + n];
                __m256 x_bc = _mm256_set1_ps(x_val);
                const float* W_row = W + k * M + j;
                
                acc[0] = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 0), acc[0]);
                acc[1] = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 8), acc[1]);
                acc[2] = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 16), acc[2]);
                acc[3] = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 24), acc[3]);
                acc[4] = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 32), acc[4]);
                acc[5] = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 40), acc[5]);
                acc[6] = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 48), acc[6]);
            }
            
            float* Y_base = Y + n;
            _mm256_storeu_ps(Y_base + (j + 0) * N, acc[0]);
            _mm256_storeu_ps(Y_base + (j + 8) * N, acc[1]);
            _mm256_storeu_ps(Y_base + (j + 16) * N, acc[2]);
            _mm256_storeu_ps(Y_base + (j + 24) * N, acc[3]);
            _mm256_storeu_ps(Y_base + (j + 32) * N, acc[4]);
            _mm256_storeu_ps(Y_base + (j + 40) * N, acc[5]);
            _mm256_storeu_ps(Y_base + (j + 48) * N, acc[6]);
        }
        
        // 8 columns at a time (2 blocks)
        for (; j + 16 <= M; j += 16) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            
            for (int k = 0; k < K; k++) {
                float x_val = X[k * N + n];
                __m256 x_bc = _mm256_set1_ps(x_val);
                const float* W_row = W + k * M + j;
                
                acc0 = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 0), acc0);
                acc1 = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 8), acc1);
            }
            
            float* Y_base = Y + n;
            _mm256_storeu_ps(Y_base + (j + 0) * N, acc0);
            _mm256_storeu_ps(Y_base + (j + 8) * N, acc1);
        }
        
        // 8 columns (1 block)  
        for (; j + 8 <= M; j += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            
            for (int k = 0; k < K; k++) {
                float x_val = X[k * N + n];
                __m256 x_bc = _mm256_set1_ps(x_val);
                const float* W_row = W + k * M + j;
                
                acc0 = _mm256_fmadd_ps(x_bc, _mm256_loadu_ps(W_row + 0), acc0);
            }
            
            float* Y_base = Y + n;
            _mm256_storeu_ps(Y_base + j * N, acc0);
        }
        
        // Scalar tail
        for (; j < M; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += X[k * N + n] * W[k * M + j];
            }
            Y[j * N + n] = acc;
        }
    }
}

float abs_sum(const float* Y, int M, int N, int max_j = 4, int max_n = 4) {
    float s = 0.0f;
    for (int n = 0; n < N && n < max_n; n++) {
        for (int j = 0; j < M && j < max_j; j++) {
            s += fabsf(Y[j * N + n]);
        }
    }
    return s;
}

float max_abs_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float mae(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum / n;
}

bool test_shape(const char* name, int K, int M, int N, float tolerance = 1e-4) {
    printf("\n=== %s: K=%d M=%d N=%d ===\n", name, K, M, N);
    
    // Calculate sizes
    size_t X_size = (size_t)K * N;
    size_t W_size = (size_t)K * M;
    size_t Y_size = (size_t)M * N;
    size_t X_bytes = X_size * sizeof(float);
    size_t W_bytes = W_size * sizeof(float);
    size_t Y_bytes = Y_size * sizeof(float);
    
    printf("  X=%.1fMB W=%.1fMB Y=%.1fMB total=%.1fMB\n",
           X_bytes/1e6, W_bytes/1e6, Y_bytes/1e6, (X_bytes+W_bytes+2*Y_bytes)/1e6);
    
    // Allocate with malloc (not aligned_alloc for better compatibility)
    float* X = (float*)malloc(X_bytes);
    float* W = (float*)malloc(W_bytes);
    float* Y_scalar = (float*)malloc(Y_bytes);
    float* Y_avx2 = (float*)malloc(Y_bytes);
    
    if (!X || !W || !Y_scalar || !Y_avx2) {
        printf("  FAIL: allocation error\n");
        free(X); free(W); free(Y_scalar); free(Y_avx2);
        return false;
    }
    
    // Initialize with deterministic pattern
    srand(42);
    for (size_t i = 0; i < X_size; i++) X[i] = (rand() % 1000) / 100.0f - 5.0f;
    for (size_t i = 0; i < W_size; i++) W[i] = (rand() % 1000) / 100.0f - 5.0f;
    
    // Run scalar
    memset(Y_scalar, 0, Y_bytes);
    double t0 = get_time_ms();
    scalar_prt_ffn_up(K, M, N, X, W, Y_scalar);
    double t_scalar = get_time_ms() - t0;
    
    // Run AVX2
    memset(Y_avx2, 0, Y_bytes);
    double t1 = get_time_ms();
    avx2_prt_ffn_up(K, M, N, X, W, Y_avx2);
    double t_avx2 = get_time_ms() - t1;
    
    // Compare
    float err_max = max_abs_error(Y_scalar, Y_avx2, (int)Y_size);
    float err_mae = mae(Y_scalar, Y_avx2, (int)Y_size);
    float sum_scalar = abs_sum(Y_scalar, M, N);
    float sum_avx2 = abs_sum(Y_avx2, M, N);
    
    printf("  Scalar: %.3f ms | abs_sum=%.6f\n", t_scalar, sum_scalar);
    printf("  AVX2:   %.3f ms | abs_sum=%.6f\n", t_avx2, sum_avx2);
    if (t_avx2 > 0.1) printf("  Speedup: %.2fx\n", t_scalar / t_avx2);
    printf("  max_abs_err=%.2e | mae=%.2e | tol=%.2e\n", err_max, err_mae, tolerance);
    bool pass = err_max < tolerance;
    printf("  %s\n", pass ? "PASS" : "FAIL");
    
    free(X); free(W); free(Y_scalar); free(Y_avx2);
    return pass;
}

int main() {
    printf("PRT Phase 22A: AVX2 correctness test\n");
    printf("====================================\n");
    
    bool all_pass = true;
    
    // Small shapes
    all_pass &= test_shape("tiny", 8, 16, 1);
    all_pass &= test_shape("small", 17, 19, 1);
    all_pass &= test_shape("medium", 64, 128, 1);
    
    // 0.5B shapes (K=896, M=4864)
    all_pass &= test_shape("0.5B_shape", 896, 4864, 1);
    all_pass &= test_shape("0.5B_shape_N2", 896, 4864, 2);
    
    // Skip 7B shape (68M elements = 272MB per tensor, 1GB+ total) unless we have lots of RAM
    // Uncomment if you want to test it:
    // all_pass &= test_shape("7B_shape", 3584, 18944, 1);
    
    printf("\n=== Overall: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}