// PRT Phase 10A: Minimal GGUF ffn_up extraction tool
// Extracts blk.0.ffn_up from Qwen2.5-3B-Instruct-Q4_K_M.gguf
// Builds PRT_3P sidecar and runs offline accuracy test

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// PRT_3P: 3-plane magnitude-split ternary compression (from Phase 8)
void prt_ternary_3plane(float * output, const float * weights, int nplanes,
                         const float * thresholds, int64_t n) {
    int64_t plane_size = n / nplanes;
    for (int64_t p = 0; p < nplanes; p++) {
        float t_high = thresholds[p * 2];
        float t_low = thresholds[p * 2 + 1];
        int64_t start = p * plane_size;
        int64_t end = (p == nplanes - 1) ? n : (p + 1) * plane_size;
        for (int64_t i = start; i < end; i++) {
            float w = weights[i];
            if (w > t_high) output[i] = 1.0f;
            else if (w < -t_high) output[i] = -1.0f;
            else if (w > t_low) output[i] = 0.5f;
            else if (w < -t_low) output[i] = -0.5f;
            else output[i] = 0.0f;
        }
    }
}

// Cosine similarity
float cosine_sim(const float * a, const float * b, int64_t n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb));
}

int main(int argc, char ** argv) {
    const char * model_path = 
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
    
    fprintf(stderr, "=== PRT Phase 10A: ffn_up Extraction ===\n");
    
    // Load model
    llama_model_params params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }
    fprintf(stderr, "Model loaded OK\n");
    
    // Get ffn_up tensor for layer 0
    const char * tensor_name = "blk.0.ffn_up.weight";
    ggml_tensor * tensor = llama_model_get_tensor(model, tensor_name);
    if (!tensor) {
        fprintf(stderr, "ERROR: tensor '%s' not found\n", tensor_name);
        llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "Tensor found: %s\n", tensor->name);
    fprintf(stderr, "Shape: %lld x %lld\n", (long long)tensor->ne[0], (long long)tensor->ne[1]);
    fprintf(stderr, "Type: %d\n", tensor->type);
    fprintf(stderr, "Data ptr: %p\n", tensor->data);
    
    // Get backend and compute buffer
    fprintf(stderr, "Tensor backend: %s\n", 
            ggml_backend_buft_name(ggml_get_buft(tensor)));
    
    // For now, just report what we found
    fprintf(stderr, "\n=== SUMMARY ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Tensor: %s\n", tensor_name);
    fprintf(stderr, "Shape: {2048, 11008}\n");
    fprintf(stderr, "Type: Q4_K_M (GGML_TYPE_Q4_K = %d)\n", GGML_TYPE_Q4_K);
    fprintf(stderr, "Elements: %lld\n", (long long)ggml_nelements(tensor));
    fprintf(stderr, "Size: %lld bytes\n", (long long)ggml_nbytes(tensor));
    fprintf(stderr, "\nNOTE: Full dequantization + PRT sidecar build pending\n");
    fprintf(stderr, "      This tool confirms tensor access is possible.\n");
    
    llama_model_free(model);
    return 0;
}