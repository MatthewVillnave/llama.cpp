/*
PRT Phase 19S: Standalone 7B INT6 Kernel Harness
================================================
Tests the scalar INT6 compute path without loading the 7B model.

Loads one INT6 sidecar file, parses PRT6 header, decodes unpacked int8_data,
and runs the exact same matvec formula as the runtime kernel:

  out[j] = sum_k X[k] * int8_data[j*K + k] * scales[j]

Expected:
  M=18944, K=3584, scale_off=20, file_size=50997268

No full model load. No KV cache. One sidecar at a time.
*/

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// ============ PRT6 Header Parse ============
struct PRT6Header {
    uint32_t magic;
    uint32_t version;
    uint32_t M;
    uint32_t K;
    uint32_t reserved;
    size_t scale_off;
    size_t packed_off;
    int64_t file_size;
};

static PRT6Header parse_prt6_header(const char * path) {
    PRT6Header h = {};
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open %s\n", path); return h; }
    
    fseek(f, 0, SEEK_END);
    h.file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint8_t hdr[24];
    if (fread(hdr, 1, 24, f) != 24) { fprintf(stderr, "[ERROR] Cannot read header\n"); fclose(f); return h; }
    
    h.magic = hdr[0] | (hdr[1]<<8) | (hdr[2]<<16) | (hdr[3]<<24);
    h.version = hdr[4] | (hdr[5]<<8) | (hdr[6]<<16) | (hdr[7]<<24);
    h.M = hdr[8] | (hdr[9]<<8) | (hdr[10]<<16) | (hdr[11]<<24);
    h.K = hdr[12] | (hdr[13]<<8) | (hdr[14]<<16) | (hdr[15]<<24);
    h.reserved = hdr[16] | (hdr[17]<<8) | (hdr[18]<<16) | (hdr[19]<<24);
    
    // Schema-aware scale_off: 0.5B (M=4864/K=896) -> 16, else 20
    h.scale_off = (h.M == 4864 && h.K == 896) ? 16 : 20;
    h.packed_off = h.scale_off + (size_t)h.M * 4;
    
    fclose(f);
    return h;
}

// ============ INT6 Unpack (matches runtime) ============
// 64-entry LUT: 6-bit -> offset-32 int8
// Unpacks M*K packed 6-bit values into M*K int8 values
// Matches exactly: tools/cli/cli.cpp Phase 15H optimized unpack

static void unpack_int6(const uint8_t * packed, int64_t M, int64_t K,
                         int8_t * int8_data) {
    int8_t lut[64];
    for (int i = 0; i < 64; i++) lut[i] = (int8_t)(i - 32);
    
    int64_t total = M * K;
    const uint8_t * p_src = packed;
    int64_t i = 0;
    
    // 4x unrolled: 16 elements per 12 bytes (same as runtime)
    for (; i + 15 < total; i += 16) {
        uint8_t b0 = *p_src++; uint8_t b1 = *p_src++; uint8_t b2 = *p_src++;
        int8_data[i]     = lut[b0 & 0x3F];
        int8_data[i + 1] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
        int8_data[i + 2] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
        int8_data[i + 3] = lut[(b2 >> 2) & 0x3F];
        b0 = *p_src++; b1 = *p_src++; b2 = *p_src++;
        int8_data[i + 4] = lut[b0 & 0x3F];
        int8_data[i + 5] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
        int8_data[i + 6] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
        int8_data[i + 7] = lut[(b2 >> 2) & 0x3F];
        b0 = *p_src++; b1 = *p_src++; b2 = *p_src++;
        int8_data[i + 8]  = lut[b0 & 0x3F];
        int8_data[i + 9]  = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
        int8_data[i + 10] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
        int8_data[i + 11] = lut[(b2 >> 2) & 0x3F];
        b0 = *p_src++; b1 = *p_src++; b2 = *p_src++;
        int8_data[i + 12] = lut[b0 & 0x3F];
        int8_data[i + 13] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
        int8_data[i + 14] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
        int8_data[i + 15] = lut[(b2 >> 2) & 0x3F];
    }
    // Scalar tail
    for (; i < total; i += 4) {
        uint8_t b0 = *p_src++; uint8_t b1 = *p_src++; uint8_t b2 = *p_src++;
        int8_data[i]     = lut[b0 & 0x3F];
        if (i + 1 < total) int8_data[i + 1] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
        if (i + 2 < total) int8_data[i + 2] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
        if (i + 3 < total) int8_data[i + 3] = lut[(b2 >> 2) & 0x3F];
    }
}

// ============ Scalar INT6 Matvec (matches runtime) ============
// out[j] = sum_k X[k] * int8_data[j*K + k] * scales[j]
// j runs over M (ffn dim), k runs over K (hidden dim)

static void matvec_int6_scalar(const float * X, const int8_t * int8_data,
                                const float * scales, int64_t M, int64_t K,
                                float * out) {
    for (int64_t j = 0; j < M; j++) {
        float s = 0.0f;
        float sc = scales[j];
        const int8_t * W_row = int8_data + j * K;
        for (int64_t k = 0; k < K; k++) {
            s += X[k] * (float)W_row[k] * sc;
        }
        out[j] = s;
    }
}

// Sparse matvec: only one k value is nonzero
static void matvec_int6_sparse_k(const int8_t * int8_data, const float * scales,
                                  int64_t M, int64_t K, int64_t k_target, float X_val,
                                  float * out) {
    for (int64_t j = 0; j < M; j++) {
        out[j] = X_val * (float)int8_data[j * K + k_target] * scales[j];
    }
}

// ============ Load Sidecar ============
static bool load_sidecar(const char * path,
                         std::vector<float> & scales,
                         std::vector<int8_t> & int8_data,
                         PRT6Header & h) {
    h = parse_prt6_header(path);
    if (h.M == 0 || h.K == 0) return false;
    
    fprintf(stderr, "[LOAD] file=%s size=%ld magic=0x%08x M=%u K=%u scale_off=%zu packed_off=%zu\n",
            path, h.file_size, h.magic, h.M, h.K, h.scale_off, h.packed_off);
    
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    
    // Read scales
    scales.resize(h.M);
    fseek(f, h.scale_off, SEEK_SET);
    if (fread(scales.data(), 4, h.M, f) != (size_t)h.M) { fclose(f); return false; }
    fprintf(stderr, "[LOAD] scales read=%u first8=", h.M);
    for (int i = 0; i < 8; i++) fprintf(stderr, " %.6f", scales[i]);
    fprintf(stderr, "\n");
    
    // Read packed payload
    int64_t packed_n = ((h.M * h.K) + 3) / 4 * 3;
    std::vector<uint8_t> packed(packed_n);
    fseek(f, h.packed_off, SEEK_SET);
    if (fread(packed.data(), 1, packed_n, f) != (size_t)packed_n) { fclose(f); return false; }
    fclose(f);
    
    // Unpack
    int8_data.resize(h.M * h.K);
    unpack_int6(packed.data(), h.M, h.K, int8_data.data());
    
    fprintf(stderr, "[LOAD] unpacked %ld values\n", (long)int8_data.size());
    fprintf(stderr, "[LOAD] int8_data[0..15]=");
    for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)int8_data[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "[LOAD] int8_data[M*K-16..M*K-1]=");
    for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)int8_data[int8_data.size() - 16 + i]);
    fprintf(stderr, "\n");
    
    return true;
}

// ============ Synthetic Sidecar ============
static bool create_synthetic_sidecar(std::vector<float> & scales,
                                      std::vector<int8_t> & int8_data,
                                      int64_t M, int64_t K) {
    scales.assign(M, 1.0f);
    int8_data.resize(M * K);
    for (int64_t i = 0; i < M * K; i++) {
        int8_data[i] = (int8_t)((i % 64) - 32);  // Simple pattern: cycle through -32..+31
    }
    return true;
}

// ============ Analysis ============
struct VecStats {
    float min_val, max_val, mean_val, abs_sum;
    int64_t nonzero_count, nan_count, inf_count;
};

static VecStats compute_stats(const float * v, int64_t n) {
    VecStats s = {};
    s.min_val = 1e38f; s.max_val = -1e38f;
    double sum = 0.0; s.abs_sum = 0.0; s.nonzero_count = 0;
    for (int64_t i = 0; i < n; i++) {
        float x = v[i];
        if (x < s.min_val) s.min_val = x;
        if (x > s.max_val) s.max_val = x;
        sum += x;
        s.abs_sum += fabsf(x);
        if (x != 0.0f) s.nonzero_count++;
        if (std::isnan(x)) s.nan_count++;
        if (std::isinf(x)) s.inf_count++;
    }
    s.mean_val = (float)(sum / n);
    return s;
}

static void print_stats(const char * label, const VecStats & s) {
    fprintf(stderr, "[STATS] %s: min=%.6g max=%.6g mean=%.6g abs_sum=%.6g nonzero=%ld nan=%ld inf=%ld\n",
            label, s.min_val, s.max_val, s.mean_val, s.abs_sum,
            (long)s.nonzero_count, (long)s.nan_count, (long)s.inf_count);
}

static void print_vec(const char * label, const float * v, int64_t n, int64_t max_show = 16) {
    fprintf(stderr, "[VEC] %s[0..%ld]=", label, (long)(max_show - 1));
    for (int64_t i = 0; i < max_show && i < n; i++) fprintf(stderr, " %.6g", v[i]);
    fprintf(stderr, "\n");
}

static void print_top_k(const float * v, int64_t n, int top_n = 10) {
    fprintf(stderr, "[TOP] top %d abs values:\n", top_n);
    std::vector<std::pair<float,int64_t>> idx;
    idx.reserve(n);
    for (int64_t i = 0; i < n; i++) idx.emplace_back(fabsf(v[i]), i);
    std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(),
                      [](auto &a, auto &b){ return a.first > b.first; });
    for (int i = 0; i < top_n && i < (int)idx.size(); i++) {
        fprintf(stderr, "  [%ld]=%.6g\n", (long)idx[i].second, idx[i].first);
    }
}

// ============ Main ============
int main(int argc, char ** argv) {
    const char * usage = "Usage: %s <sidecar_file> [--synthetic M K]\n"
                         "  sidecar_file: e.g. /tmp/prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6\n"
                         "  --synthetic M K: create in-memory synthetic sidecar (M rows, K cols)\n";
    
    bool synthetic = false;
    int64_t syn_M = 0, syn_K = 0;
    const char * sidecar_path = nullptr;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--synthetic") == 0 && i + 2 < argc) {
            synthetic = true;
            syn_M = atoll(argv[++i]);
            syn_K = atoll(argv[++i]);
        } else {
            sidecar_path = argv[i];
        }
    }
    
    if (!synthetic && !sidecar_path) {
        fprintf(stderr, usage, argv[0]);
        return 1;
    }
    
    std::vector<float> scales;
    std::vector<int8_t> int8_data;
    PRT6Header h = {};
    
    if (synthetic) {
        fprintf(stderr, "[SYNTHETIC] Creating M=%ld K=%ld synthetic sidecar\n", (long)syn_M, (long)syn_K);
        create_synthetic_sidecar(scales, int8_data, syn_M, syn_K);
        h.M = syn_M; h.K = syn_K;
    } else {
        if (!load_sidecar(sidecar_path, scales, int8_data, h)) {
            fprintf(stderr, "[ERROR] Failed to load sidecar\n");
            return 1;
        }
    }
    
    int64_t M = h.M, K = h.K;
    fprintf(stderr, "\n=== SIDEARM: M=%ld K=%ld ===\n", (long)M, (long)K);
    fprintf(stderr, "scale[0]=%g scale[1]=%g scale[M/2]=%g\n",
            scales[0], scales[1], scales[M/2]);
    fprintf(stderr, "q[0..7]=%d %d %d %d %d %d %d %d\n",
            (int)int8_data[0], (int)int8_data[1], (int)int8_data[2], (int)int8_data[3],
            (int)int8_data[4], (int)int8_data[5], (int)int8_data[6], (int)int8_data[7]);
    
    std::vector<float> out(M);
    auto start_all = std::chrono::high_resolution_clock::now();
    
    // ============ Test 1: All-ones activation ============
    fprintf(stderr, "\n=== TEST: all-ones activation ===\n");
    std::vector<float> X_ones(K, 1.0f);
    matvec_int6_scalar(X_ones.data(), int8_data.data(), scales.data(), M, K, out.data());
    auto stats = compute_stats(out.data(), M);
    print_stats("all_ones", stats);
    print_vec("out[0..15]", out.data(), M, 16);
    
    // ============ Test 2: Sparse k=0 ============
    fprintf(stderr, "\n=== TEST: sparse k=0 (X[0]=1, others=0) ===\n");
    memset(out.data(), 0, M * sizeof(float));
    matvec_int6_sparse_k(int8_data.data(), scales.data(), M, K, 0, 1.0f, out.data());
    stats = compute_stats(out.data(), M);
    print_stats("sparse_k0", stats);
    print_vec("out[0..15]", out.data(), M, 16);
    // Verify: out[j] should = int8_data[j*K + 0] * scales[j]
    int64_t mismatches = 0;
    for (int64_t j = 0; j < M; j++) {
        float expected = (float)int8_data[j * K + 0] * scales[j];
        if (fabsf(out[j] - expected) > 1e-6) mismatches++;
    }
    fprintf(stderr, "[VERIFY] sparse_k0: %ld mismatches out of %ld\n", (long)mismatches, (long)M);
    
    // ============ Test 3: Sparse k=K/2 ============
    fprintf(stderr, "\n=== TEST: sparse k=K/2 (X[K/2]=1, others=0) ===\n");
    int64_t k_half = K / 2;
    memset(out.data(), 0, M * sizeof(float));
    matvec_int6_sparse_k(int8_data.data(), scales.data(), M, K, k_half, 1.0f, out.data());
    stats = compute_stats(out.data(), M);
    print_stats("sparse_k_half", stats);
    print_vec("out[0..15]", out.data(), M, 16);
    mismatches = 0;
    for (int64_t j = 0; j < M; j++) {
        float expected = (float)int8_data[j * K + k_half] * scales[j];
        if (fabsf(out[j] - expected) > 1e-6) mismatches++;
    }
    fprintf(stderr, "[VERIFY] sparse_k_half: %ld mismatches out of %ld\n", (long)mismatches, (long)M);
    
    // ============ Test 4: Random normal activation ============
    fprintf(stderr, "\n=== TEST: random normal activation (seed=42) ===\n");
    std::mt19937 rng(42);
    std::vector<float> X_rand(K);
    for (int64_t k = 0; k < K; k++) X_rand[k] = (rng() - 0.5f) / (float)2147483648.0f;
    matvec_int6_scalar(X_rand.data(), int8_data.data(), scales.data(), M, K, out.data());
    stats = compute_stats(out.data(), M);
    print_stats("random_normal", stats);
    print_vec("out[0..15]", out.data(), M, 16);
    print_top_k(out.data(), M, 10);
    
    auto end_all = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(end_all - start_all).count();
    fprintf(stderr, "\n=== TOTAL TIME: %.1f ms ===\n", ms_total);
    fprintf(stderr, "Throughput: %.2f GMACS (approx)\n",
            (M * K * 2) / (ms_total * 1e6));
    
    // ============ Summary ============
    fprintf(stderr, "\n=== SUMMARY ===\n");
    fprintf(stderr, "M=%ld K=%ld scale_off=%zu\n", (long)M, (long)K, h.scale_off);
    fprintf(stderr, "scale[0]=%.6g\n", scales[0]);
    fprintf(stderr, "q[0..7]=%d,%d,%d,%d,%d,%d,%d,%d\n",
            (int)int8_data[0], (int)int8_data[1], (int)int8_data[2], (int)int8_data[3],
            (int)int8_data[4], (int)int8_data[5], (int)int8_data[6], (int)int8_data[7]);
    
    return 0;
}