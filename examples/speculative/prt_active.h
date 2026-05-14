// PRT Phase 10C: Active Guarded ffn_up PRT Replacement
// Integrates PRT_3P into speculative verification loop
// Replaces float ffn_up output with PRT output if cosine >= threshold
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

// PRT_3P thresholds
#define T_HIGH_0 2.0f
#define T_HIGH_1 0.5f
#define T_HIGH_2 0.1f

const int M = 2048;
const int N = 11008;

struct SidecarLoad {
    int layer;
    float * data;
    size_t size;
};

struct PRTState {
    bool loaded = false;
    std::unordered_map<int, SidecarLoad> sidecars;
    int active_scope = 0;  // 0 = none, 1 = layer0, 2 = layers0-3, 3 = all28
    float guard_threshold = 0.95f;
    float warn_threshold = 0.99f;
    
    // Per-generation stats
    int replacements = 0;
    int fallbacks = 0;
    int guard_fails = 0;
    float min_cosine = 1.0f;
    double total_float_us = 0.0;
    double total_prt_us = 0.0;
    
    std::ofstream log_json;
    bool first_entry = true;
};

static PRTState g_prt;

bool prt_load_sidecars(const char * map_path) {
    fprintf(stderr, "=== PRT Phase 10C: Loading sidecars ===\n");
    
    std::ifstream f(map_path);
    if (!f.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    
    int loaded = 0;
    for (int layer = 0; layer < 28; layer++) {
        char search_key[32];
        snprintf(search_key, sizeof(search_key), "\"%d\":", layer);
        size_t pos = content.find(search_key);
        if (pos == std::string::npos) continue;
        
        size_t path_pos = content.find("\"path\":", pos);
        if (path_pos == std::string::npos || path_pos > pos + 1000) continue;
        
        size_t path_start = content.find("\"", path_pos + 7);
        size_t path_end = content.find("\"", path_start + 1);
        if (path_start == std::string::npos) continue;
        
        std::string path = content.substr(path_start + 1, path_end - path_start - 1);
        
        FILE * sf = fopen(path.c_str(), "rb");
        if (!sf) continue;
        fseek(sf, 0, SEEK_END);
        size_t sz = ftell(sf);
        fseek(sf, 0, SEEK_SET);
        float * data = (float *)malloc(sz);
        if (!data || fread(data, 1, sz, sf) != sz) { free(data); fclose(sf); continue; }
        fclose(sf);
        
        g_prt.sidecars[layer] = {layer, data, sz};
        loaded++;
    }
    
    fprintf(stderr, "Loaded %d/28 sidecars\n", loaded);
    g_prt.loaded = (loaded == 28);
    
    if (g_prt.loaded) {
        g_prt.log_json.open("/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10c_active_logs.json");
        if (g_prt.log_json.is_open()) {
            g_prt.log_json << "[\n";
        }
    }
    
    return g_prt.loaded;
}

void prt_set_active_scope(int scope) {
    g_prt.active_scope = scope;
    fprintf(stderr, "PRT active scope: %d (%s)\n", scope,
        scope == 0 ? "disabled" : scope == 1 ? "layer0 only" : scope == 2 ? "layers0-3" : "all28");
}

void prt_reset_stats() {
    g_prt.replacements = 0;
    g_prt.fallbacks = 0;
    g_prt.guard_fails = 0;
    g_prt.min_cosine = 1.0f;
    g_prt.total_float_us = 0.0;
    g_prt.total_prt_us = 0.0;
    g_prt.first_entry = true;
}

bool prt_is_layer_active(int layer) {
    if (!g_prt.loaded || g_prt.active_scope == 0) return false;
    if (g_prt.active_scope == 1) return layer == 0;
    if (g_prt.active_scope == 2) return layer >= 0 && layer <= 3;
    return layer >= 0 && layer < 28;
}

bool prt_has_sidecar(int layer) {
    return g_prt.sidecars.find(layer) != g_prt.sidecars.end();
}

// PRT_3P matmul: Y[b,n] = sum over k where |X[b,k]| > threshold of: X[b,k] * |W[k,n]|
static void matmul_prt_3plane(const float * X, const float * W_prt, float * Y_prt, int batch) {
    for (int b = 0; b < batch; b++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                float x = X[b * M + k];
                float ax = fabsf(x);
                if (ax > T_HIGH_0) {
                    sum += x * W_prt[k * N + n];
                } else if (ax > T_HIGH_1) {
                    sum += x * W_prt[k * N + n];
                } else if (ax > T_HIGH_2) {
                    sum += x * W_prt[k * N + n];
                }
            }
            Y_prt[b * N + n] = sum;
        }
    }
}

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    na = sqrtf(na); nb = sqrtf(nb);
    return (na > 1e-10f && nb > 1e-10f) ? dot / (na * nb) : 0.0f;
}

// Run PRT replacement for one layer, one batch
// Returns true if PRT output should be used (cos >= threshold)
bool prt_try_replace(int layer, int batch, float * X_input, float * Y_float, float * Y_prt_buf) {
    if (!prt_is_layer_active(layer)) return false;
    if (!prt_has_sidecar(layer)) return false;
    
    const SidecarLoad & sc = g_prt.sidecars[layer];
    
    // Time PRT computation
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_prt_3plane(X_input, sc.data, Y_prt_buf, batch);
    auto t1 = std::chrono::high_resolution_clock::now();
    double prt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    
    int n = batch * N;
    float cos = cosine_sim(Y_float, Y_prt_buf, n);
    
    g_prt.total_prt_us += prt_us;
    if (cos < g_prt.min_cosine) g_prt.min_cosine = cos;
    
    // Log to JSON
    if (g_prt.log_json.is_open()) {
        if (!g_prt.first_entry) g_prt.log_json << ",\n";
        g_prt.first_entry = false;
        g_prt.log_json << "  {\"layer\":" << layer << ",\"batch\":" << batch
            << ",\"cosine\":" << cos << ",\"pass\":" << (cos >= g_prt.guard_threshold ? "true" : "false")
            << ",\"prt_us\":" << prt_us << "}";
    }
    
    if (cos >= g_prt.guard_threshold) {
        // Copy PRT output to float output (replacement)
        memcpy(Y_float, Y_prt_buf, n * sizeof(float));
        g_prt.replacements++;
        return true;
    } else {
        g_prt.guard_fails++;
        g_prt.fallbacks++;
        return false;
    }
}

void prt_print_stats() {
    fprintf(stderr, "\n=== PRT Phase 10C Stats ===\n");
    fprintf(stderr, "Active scope: %d\n", g_prt.active_scope);
    fprintf(stderr, "Replacements: %d\n", g_prt.replacements);
    fprintf(stderr, "Fallbacks: %d\n", g_prt.fallbacks);
    fprintf(stderr, "Guard fails: %d\n", g_prt.guard_fails);
    fprintf(stderr, "Min cosine: %.6f\n", g_prt.min_cosine);
    fprintf(stderr, "Total PRT time: %.1f ms\n", g_prt.total_prt_us / 1000.0);
}

void prt_cleanup() {
    for (auto & kv : g_prt.sidecars) free(kv.second.data);
    g_prt.sidecars.clear();
    g_prt.loaded = false;
    if (g_prt.log_json.is_open()) {
        g_prt.log_json << "\n]\n";
        g_prt.log_json.close();
    }
}

int prt_get_replacements() { return g_prt.replacements; }
int prt_get_fallbacks() { return g_prt.fallbacks; }
int prt_get_guard_fails() { return g_prt.guard_fails; }
float prt_get_min_cosine() { return g_prt.min_cosine; }
bool prt_is_loaded() { return g_prt.loaded; }