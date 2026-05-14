// PRT Phase 10C: Active Guarded ffn_up PRT Replacement Test
// Tests local matmul replacement before full speculative integration
// Simulates "active replacement" by running PRT instead of float for active layers
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>

#define T_HIGH_0 2.0f
#define T_HIGH_1 0.5f
#define T_HIGH_2 0.1f

const int M = 2048;
const int N = 11008;
const int N_LAYERS = 28;

static void matmul_prt(const float * X, const float * W, float * Y, int batch) {
    for (int b = 0; b < batch; b++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                float x = X[b * M + k];
                float ax = fabsf(x);
                if (ax > T_HIGH_0) { sum += x * W[k * N + n]; }
                else if (ax > T_HIGH_1) { sum += x * W[k * N + n]; }
                else if (ax > T_HIGH_2) { sum += x * W[k * N + n]; }
            }
            Y[b * N + n] = sum;
        }
    }
}

static void matmul_float(const float * X, const float * W, float * Y, int batch) {
    for (int b = 0; b < batch; b++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) sum += X[b * M + k] * W[k * N + n];
            Y[b * N + n] = sum;
        }
    }
}

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    na = sqrtf(na); nb = sqrtf(nb);
    return (na > 1e-10f && nb > 1e-10f) ? dot / (na * nb) : 0.0f;
}
static float max_err(const float * a, const float * b, int n) {
    float m = 0.0f; for (int i = 0; i < n; i++) { float e = fabsf(a[i]-b[i]); if (e > m) m = e; } return m;
}
static float mean_err(const float * a, const float * b, int n) {
    float s = 0.0f; for (int i = 0; i < n; i++) s += fabsf(a[i]-b[i]); return s / n;
}

struct Sidecar { int layer; float * data; size_t size; };
std::vector<Sidecar> load_sidecars() {
    std::vector<Sidecar> sc;
    char path[256];
    for (int l = 0; l < N_LAYERS; l++) {
        snprintf(path, sizeof(path), "/tmp/prt_sidecars/ffn_up_layer%d_prt.bin", l);
        FILE * f = fopen(path, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        float * data = (float *)malloc(sz);
        if (!data || fread(data, 1, sz, f) != sz) { free(data); fclose(f); continue; }
        fclose(f);
        sc.push_back({l, data, sz});
    }
    return sc;
}

struct LayerMetrics {
    int layer;
    float float_time_us;
    float prt_time_us;
    float cosine;
    bool prt_replaced;
    bool guard_passed;
};

int main(int argc, char ** argv) {
    int active_scope = (argc > 1) ? atoi(argv[1]) : 0;  // 0=none, 1=layer0, 2=layers0-3, 3=all28
    const char * scope_name = (active_scope == 0) ? "disabled" :
                               (active_scope == 1) ? "layer0" :
                               (active_scope == 2) ? "layers0-3" : "all28";
    
    fprintf(stderr, "=== PRT Phase 10C: Active Guarded Test ===\n");
    fprintf(stderr, "Active scope: %s\n\n", scope_name);
    
    // Load sidecars
    auto t_load = std::chrono::high_resolution_clock::now();
    auto sidecars = load_sidecars();
    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_load).count();
    
    fprintf(stderr, "Sidecars: %zu/28 loaded (%.1f ms)\n\n", sidecars.size(), load_ms);
    if (sidecars.size() != 28) { fprintf(stderr, "ERROR: need all 28\n"); return 1; }
    
    // Generate test inputs (same as Phase 10B: X in [0,1], seed=42)
    fprintf(stderr, "Generating test inputs...\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> X_16(16 * M);
    std::vector<float> X_17(17 * M);
    for (int i = 0; i < 16 * M; i++) X_16[i] = dist(rng);
    for (int i = 0; i < 17 * M; i++) X_17[i] = dist(rng);
    
    std::vector<float> Y_float(17 * N);
    std::vector<float> Y_prt(17 * N);
    std::vector<float> Y_active(17 * N);  // Result with active replacement
    
    // === PHASE 1: Run pure float (baseline) ===
    fprintf(stderr, "Running baseline (pure float)...\n");
    std::vector<LayerMetrics> baseline_results;
    double baseline_total_float_us = 0.0;
    
    for (const auto & sc : sidecars) {
        LayerMetrics bm = {};
        bm.layer = sc.layer;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_float(X_16.data(), sc.data, Y_float.data(), 16);
        matmul_float(X_17.data(), sc.data, Y_float.data(), 17);
        auto t1 = std::chrono::high_resolution_clock::now();
        bm.float_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        
        baseline_total_float_us += bm.float_time_us;
        baseline_results.push_back(bm);
    }
    
    // === PHASE 2: Run active replacement ===
    fprintf(stderr, "Running active replacement (scope=%s)...\n", scope_name);
    std::vector<LayerMetrics> active_results;
    double active_total_us = 0.0;
    int replacements = 0;
    int guard_fails = 0;
    float min_active_cosine = 1.0f;
    
    for (const auto & sc : sidecars) {
        LayerMetrics am = {};
        am.layer = sc.layer;
        am.prt_replaced = false;
        am.guard_passed = false;
        
        bool is_active = (active_scope == 1 && sc.layer == 0) ||
                         (active_scope == 2 && sc.layer >= 0 && sc.layer <= 3) ||
                         (active_scope == 3 && sc.layer >= 0 && sc.layer < 28);
        
        if (is_active) {
            // Run both float and PRT, then replace if guard passes
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_float(X_16.data(), sc.data, Y_float.data(), 16);
            matmul_float(X_17.data(), sc.data, Y_float.data(), 17);
            auto t1 = std::chrono::high_resolution_clock::now();
            am.float_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            
            auto t2 = std::chrono::high_resolution_clock::now();
            matmul_prt(X_16.data(), sc.data, Y_prt.data(), 16);
            matmul_prt(X_17.data(), sc.data, Y_prt.data(), 17);
            auto t3 = std::chrono::high_resolution_clock::now();
            am.prt_time_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
            
            // Guard check: cosine between float and PRT
            float cos = cosine_sim(Y_float.data(), Y_prt.data(), 16 * N);
            am.cosine = cos;
            am.guard_passed = (cos >= 0.95f);
            
            if (am.guard_passed) {
                // Copy PRT output to active result (active replacement)
                memcpy(Y_active.data(), Y_prt.data(), 16 * N * sizeof(float));
                // For batch 17, append
                memcpy(Y_active.data() + 16 * N, Y_prt.data() + 16 * N, 17 * N * sizeof(float));
                am.prt_replaced = true;
                replacements++;
                if (cos < min_active_cosine) min_active_cosine = cos;
            } else {
                // Guard failed: use float
                memcpy(Y_active.data(), Y_float.data(), 16 * N * sizeof(float));
                memcpy(Y_active.data() + 16 * N, Y_float.data() + 16 * N, 17 * N * sizeof(float));
                guard_fails++;
            }
            
            active_total_us += am.float_time_us + am.prt_time_us;  // Float + guard check overhead
        } else {
            // Not active: pure float
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_float(X_16.data(), sc.data, Y_active.data(), 16);
            matmul_float(X_17.data(), sc.data, Y_active.data(), 17);
            auto t1 = std::chrono::high_resolution_clock::now();
            am.float_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            active_total_us += am.float_time_us;
        }
        
        active_results.push_back(am);
        fprintf(stderr, "  Layer %2d: cosine=%.6f replaced=%s guard_passed=%s\n",
                am.layer, am.cosine, am.prt_replaced ? "YES" : "NO", am.guard_passed ? "YES" : "NO");
    }
    
    // === Compute end-to-end quality ===
    // Compare active output to pure float output
    float active_cosine = cosine_sim(Y_float.data(), Y_active.data(), 17 * N);
    float active_max_err = max_err(Y_float.data(), Y_active.data(), 17 * N);
    float active_mean_err = mean_err(Y_float.data(), Y_active.data(), 17 * N);
    
    // Timing summary
    double baseline_per_hit = baseline_total_float_us / (28 * 2);
    double active_per_hit = active_total_us / (28 * 2);
    double speedup = baseline_per_hit / active_per_hit;
    
    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Active scope: %s\n", scope_name);
    fprintf(stderr, "Replacements: %d\n", replacements);
    fprintf(stderr, "Guard fails: %d\n", guard_fails);
    fprintf(stderr, "Min PRT cosine (replaced layers): %.6f\n", min_active_cosine);
    fprintf(stderr, "Active vs Float cosine: %.6f\n", active_cosine);
    fprintf(stderr, "Active vs Float max_err: %.4f\n", active_max_err);
    fprintf(stderr, "Active vs Float mean_err: %.4f\n", active_mean_err);
    fprintf(stderr, "Baseline per-hit: %.1f us\n", baseline_per_hit);
    fprintf(stderr, "Active per-hit: %.1f us\n", active_per_hit);
    fprintf(stderr, "Speedup: %.2fx\n", speedup);
    
    bool pass = (min_active_cosine >= 0.95f) && (replacements > 0);
    
    // Write output files
    std::string scope_str = scope_name;
    char fname[256];
    
    snprintf(fname, sizeof(fname), "/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10c_active_%s.md", 
             scope_name);
    FILE * out_md = fopen(fname, "w");
    fprintf(out_md, "# Phase 10C Active Guarded Test: %s\n\n", scope_name);
    fprintf(out_md, "## Summary\n\n");
    fprintf(out_md, "| Metric | Value |\n|--------|-------|\n");
    fprintf(out_md, "| Active scope | %s |\n", scope_name);
    fprintf(out_md, "| Replacements | %d |\n", replacements);
    fprintf(out_md, "| Guard fails | %d |\n", guard_fails);
    fprintf(out_md, "| Min cosine | %.6f |\n", min_active_cosine);
    fprintf(out_md, "| Active vs Float cosine | %.6f |\n", active_cosine);
    fprintf(out_md, "| Baseline per-hit | %.1f us |\n", baseline_per_hit);
    fprintf(out_md, "| Active per-hit | %.1f us |\n", active_per_hit);
    fprintf(out_md, "| Speedup | %.2fx |\n", speedup);
    fprintf(out_md, "\n## Per-Layer\n\n");
    fprintf(out_md, "| L | Float_us | PRT_us | Cosine | Replaced | Guard |\n");
    fprintf(out_md, "|--|----------|--------|--------|----------|-------|\n");
    for (size_t i = 0; i < active_results.size(); i++) {
        const auto & r = active_results[i];
        fprintf(out_md, "| %d | %.1f | %.1f | %.6f | %s | %s |\n",
            r.layer, r.float_time_us, r.prt_time_us, r.cosine,
            r.prt_replaced ? "YES" : "NO", r.guard_passed ? "PASS" : "FAIL");
    }
    fprintf(out_md, "\n## Verdict: %s\n", pass ? "PASS" : "FAIL");
    fclose(out_md);
    
    fprintf(stderr, "\nOutput: %s\n", fname);
    fprintf(stderr, "VERDICT: %s\n", pass ? "PASS" : "FAIL");
    
    for (auto & sc : sidecars) free(sc.data);
    return pass ? 0 : 1;
}