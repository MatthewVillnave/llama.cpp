// PRT Phase 24F-X: Extract 3B layer0 FFN_UP - using ggml to_float
// Adapted from Phase15B-D

#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// 3B constants  
static const int64_t K = 2048;
static const int64_t M = 11008;

// Model paths  
static const char * MODEL_PATH = 
    "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
static const char * OUTPUT_PATH = 
    "/media/matthew-villnave/VL_usb/prt_scratch/f32_refs/prt_phase24f_3b_layer0_W_f32.bin";

int main(int argc, char ** argv) {
    const char * model_path = (argc > 1) ? argv[1] : MODEL_PATH;
    const char * out_path = (argc > 2) ? argv[2] : OUTPUT_PATH;
    
    fprintf(stderr, "=== PRT Phase 24F-X: Extract 3B FFN_UP ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Output: %s\n", out_path);
    fprintf(stderr, "K=%lld M=%lld\n", (long long)K, (long long)M);
    
    // Load GGUF using ggml-base
    struct gguf_init_params params = {
        .no_alloc = false,
    };
    gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "FAIL: cannot load GGUF\n");
        return 1;
    }
    
    // Get tensor using gguf_find_tensor
    const char * tensor_name = "blk.0.ffn_up.weight";
    int tensor_idx = gguf_find_tensor(ctx, tensor_name);
    if (tensor_idx < 0) {
        fprintf(stderr, "FAIL: tensor '%s' not found\n", tensor_name);
        gguf_free(ctx);
        return 1;
    }
    
    struct ggml_tensor * t = ggml_get_tensor(ctx, tensor_name);
    if (!t) {
        fprintf(stderr, "FAIL: cannot get tensor\n");
        gguf_free(ctx);
        return 1;
    }
    
    fprintf(stderr, "Found: %s [%lld, %lld] type=%d\n", 
            tensor_name, (long long)t->ne[0], (long long)t->ne[1], t->type);
    
    // Get type traits for dequantization
    const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "FAIL: no dequant for type %d\n", t->type);
        gguf_free(ctx);
        return 1;
    }
    
    // Allocate output
    std::vector<float> f32_data(K * M);
    
    // Use ggml to_float for proper dequantization
    int64_t nelems = K * M;
    traits->to_float(t->data, f32_data.data(), nelems);
    
    // Validate
    float min_val = 1e9, max_val = -1e9, sum = 0;
    int inf_count = 0;
    for (int64_t i = 0; i < nelems; i++) {
        float v = f32_data[i];
        if (!std::isfinite(v)) {
            inf_count++;
            if (inf_count <= 5) fprintf(stderr, "INF at %lld: %e\n", i, v);
        }
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    
    fprintf(stderr, "Stats: min=%.4f max=%.4f mean=%.4f inf=%d\n", 
            min_val, max_val, sum / nelems, inf_count);
    
    // Compute norm
    float norm = 0;
    for (int64_t i = 0; i < nelems; i++) norm += f32_data[i] * f32_data[i];
    norm = std::sqrt(norm);
    fprintf(stderr, "F32 norm: %.4f\n", norm);
    
    // Write output
    FILE * fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "FAIL: write %s\n", out_path);
        gguf_free(ctx);
        return 1;
    }
    
    fwrite(f32_data.data(), sizeof(float), nelems, fout);
    fclose(fout);
    
    fprintf(stderr, "Wrote %s (%lld bytes)\n", out_path, (long long)(nelems * 4));
    
    gguf_free(ctx);
    fprintf(stderr, "=== DONE ===\n");
    return 0;
}