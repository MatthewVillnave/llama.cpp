// PRT Phase 10B: Runtime Shadow Compute
// Hooks into speculative decoding verification loop for ffn_up ops
// Does NOT modify model computation, runs PRT in parallel as shadow
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

// PRT_3P thresholds
#define T_HIGH_0 2.0f
#define T_HIGH_1 0.5f
#define T_HIGH_2 0.1f

// Sidecar storage
struct SidecarLoad {
    int layer;
    std::string path;
    float * data;  // PRT sidecar: |w| per element, float32
    size_t size;
};

static std::unordered_map<int, SidecarLoad> g_sidecars;
static bool g_sidecars_loaded = false;
static FILE * g_log_file = nullptr;

// Load sidecar map and all sidecar files
bool prt_load_sidecars(const char * map_path) {
    fprintf(stderr, "=== PRT Phase 10B: Loading sidecars ===\n");
    
    std::ifstream f(map_path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", map_path);
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    f.close();
    
    // Parse JSON and load each sidecar
    int loaded = 0;
    for (int layer = 0; layer < 28; layer++) {
        // Find this layer's entry
        char search_key[32];
        snprintf(search_key, sizeof(search_key), "\"%d\":", layer);
        size_t pos = content.find(search_key);
        if (pos == std::string::npos) continue;
        
        // Find "path" near this layer entry (within 1000 chars)
        size_t path_pos = content.find("\"path\":", pos);
        if (path_pos == std::string::npos || path_pos > pos + 1000) continue;
        
        size_t path_start = content.find("\"", path_pos + 7);
        size_t path_end = content.find("\"", path_start + 1);
        if (path_start == std::string::npos) continue;
        
        std::string path = content.substr(path_start + 1, path_end - path_start - 1);
        
        FILE * sf = fopen(path.c_str(), "rb");
        if (!sf) {
            fprintf(stderr, "WARNING: cannot open %s\n", path.c_str());
            continue;
        }
        
        fseek(sf, 0, SEEK_END);
        size_t size = ftell(sf);
        fseek(sf, 0, SEEK_SET);
        
        float * data = (float *) malloc(size);
        if (!data) { fclose(sf); continue; }
        
        if (fread(data, 1, size, sf) != size) {
            free(data); fclose(sf); continue;
        }
        fclose(sf);
        
        SidecarLoad sl;
        sl.layer = layer;
        sl.path = path;
        sl.data = data;
        sl.size = size;
        g_sidecars[layer] = sl;
        loaded++;
    }
    
    fprintf(stderr, "Loaded %d/28 sidecars\n", loaded);
    g_sidecars_loaded = (loaded == 28);
    
    // Open log file
    g_log_file = fopen("/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10b_runtime_shadow_logs.json", "w");
    if (g_log_file) {
        fprintf(g_log_file, "[\n");
    }
    
    return g_sidecars_loaded;
}

// PRT_3P matmul: Y_prt = X @ |W| using only non-pruned activations
// Y_prt[b,n] = sum over k where |X[b,k]| > T of: |X[b,k]| * |W[k,n]|
static void matmul_prt_3plane(
        const float * X,
        const float * W_sidecar,
        float * Y_prt,
        int batch, int M, int N) {
    
    for (int b = 0; b < batch; b++) {
        const float * Xb = X + b * M;
        float * Ybn = Y_prt + b * N;
        
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            const float * Wk = W_sidecar + n;
            
            for (int k = 0; k < M; k++) {
                float x = Xb[k];
                float ax = fabsf(x);
                
                if (ax > T_HIGH_0) {
                    sum += ax * Wk[k * N];
                } else if (ax > T_HIGH_1) {
                    sum += ax * Wk[k * N];
                } else if (ax > T_HIGH_2) {
                    sum += ax * Wk[k * N];
                }
            }
            Ybn[n] = sum;
        }
    }
}

// Cosine similarity
static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    na = sqrtf(na); nb = sqrtf(nb);
    return (na > 0.0f && nb > 0.0f) ? dot / (na * nb) : 0.0f;
}

// Max absolute error
static float max_abs_err(const float * a, const float * b, int n) {
    float max_e = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > max_e) max_e = e;
    }
    return max_e;
}

// Mean absolute error
static float mean_abs_err(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabsf(a[i] - b[i]);
    return sum / n;
}

// Run shadow comparison for one layer, one batch
struct ShadowResult {
    int layer;
    int batch;
    bool computed;
    bool has_sidecar;
    float cosine;
    float max_abs_err;
    float mean_abs_err;
    double float_us;
    double prt_us;
    int64_t float_crc;
    int64_t prt_crc;
};

ShadowResult run_shadow_test(int layer, int batch, const float * X_act, const float * Y_float, int M, int N) {
    ShadowResult r = {};
    r.layer = layer;
    r.batch = batch;
    r.computed = false;
    r.cosine = -1.0f;
    r.max_abs_err = -1.0f;
    r.mean_abs_err = -1.0f;
    r.float_us = 0.0;
    r.prt_us = 0.0;
    r.float_crc = 0;
    r.prt_crc = 0;
    
    if (!g_sidecars_loaded || g_sidecars.find(layer) == g_sidecars.end()) {
        r.has_sidecar = false;
        return r;
    }
    r.has_sidecar = true;
    
    const SidecarLoad & sc = g_sidecars[layer];
    
    // Allocate PRT output
    float * Y_prt = (float *) calloc(batch * N, sizeof(float));
    if (!Y_prt) return r;
    
    // Run PRT_3P shadow matmul
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_prt_3plane(X_act, sc.data, Y_prt, batch, M, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.prt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    
    // Compare PRT to float
    r.cosine = cosine_sim(Y_float, Y_prt, batch * N);
    r.max_abs_err = max_abs_err(Y_float, Y_prt, batch * N);
    r.mean_abs_err = mean_abs_err(Y_float, Y_prt, batch * N);
    
    // CRC32 for both outputs
    uint32_t fcrc = 0, prcrc = 0;
    for (int i = 0; i < batch * N; i++) {
        fcrc ^= (uint32_t)(Y_float[i] * 1000.0f) ^ (i * 37);
        prcrc ^= (uint32_t)(Y_prt[i] * 1000.0f) ^ (i * 37);
    }
    r.float_crc = fcrc;
    r.prt_crc = prcrc;
    
    r.computed = true;
    
    free(Y_prt);
    return r;
}

// Cleanup
void prt_unload_sidecars() {
    for (auto & kv : g_sidecars) free(kv.second.data);
    g_sidecars.clear();
    g_sidecars_loaded = false;
    if (g_log_file) { fprintf(g_log_file, "\n]\n"); fclose(g_log_file); g_log_file = nullptr; }
}

// Log result
void prt_log_result(const ShadowResult & r) {
    if (!g_log_file) return;
    static bool first = true;
    if (!first) fprintf(g_log_file, ",\n");
    first = false;
    fprintf(g_log_file,
        "  {\"layer\":%d,\"batch\":%d,\"computed\":%s,\"cosine\":%.6f,\"max_err\":%.6f,\"mean_err\":%.6f,\"prt_us\":%.1f}",
        r.layer, r.batch,
        r.computed ? "true" : "false",
        r.cosine, r.max_abs_err, r.mean_abs_err, r.prt_us);
}

// Get loaded sidecar count
int prt_get_sidecar_count() { return (int)g_sidecars.size(); }

// Check if sidecar for layer exists
bool prt_has_sidecar(int layer) { return g_sidecars.find(layer) != g_sidecars.end(); }

// Run shadow test with provided activation and float output data
ShadowResult prt_shadow_test(int layer, int batch, const float * X_act, const float * Y_float, int M, int N) {
    ShadowResult r = run_shadow_test(layer, batch, X_act, Y_float, M, N);
    prt_log_result(r);
    return r;
}