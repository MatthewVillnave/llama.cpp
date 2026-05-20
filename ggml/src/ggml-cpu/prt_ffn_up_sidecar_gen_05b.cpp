// PRT Phase 22B: Generate 0.5B INT8 ffn_up sidecar for layer0
// K=896, M=4864 (Qwen2.5-0.5B hidden/intermediate dims)

#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

static const int64_t K = 896;   // hidden_size
static const int64_t M = 4864;  // intermediate_size

// Find tensor in model
static ggml_tensor * find_tensor(ggml_context * ctx, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    return t;
}

int main(int argc, char ** argv) {
    const char * model_path = 
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
    const char * out_path = 
        "/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_0_5b_int8_layer0/ffn_up_layer0_prt.int8";

    fprintf(stderr, "=== PRT Phase 22B: 0.5B INT8 sidecar generation ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Output: %s\n", out_path);
    fprintf(stderr, "K=%lld M=%lld\n", (long long)K, (long long)M);

    // Load model with gguf
    gguf_init_params params = {
        .no_alloc = true,
    };
    struct gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "FAIL: cannot load model\n");
        return 1;
    }

    // Get tensor info
    int n_tensors = gguf_get_n_tensors(ctx);
    fprintf(stderr, "n_tensors = %d\n", n_tensors);

    // Find blk.0.attn_norm.weight or similar to identify tensor names
    // For Qwen2: blk.0.attn_norm.weight, blk.0.ffn_up.weight, blk.0.ffn_gate.weight
    const char * tensor_name = "blk.0.ffn_up.weight";
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, tensor_name);
    if (!tensor) {
        // Try different name
        tensor_name = "blk.0.ffn_up.weight";
        fprintf(stderr, "tensor '%s' not found, trying others...\n", tensor_name);
        
        // List all tensors
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx, i);
            fprintf(stderr, "  tensor[%d]: %s\n", i, name);
        }
        
        // Try to find ffn_up
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx, i);
            if (name && strstr(name, "ffn_up")) {
                tensor_name = name;
                tensor = ggml_get_tensor(ctx, tensor_name);
                fprintf(stderr, "Using: %s\n", tensor_name);
                break;
            }
        }
    }

    if (!tensor) {
        fprintf(stderr, "FAIL: cannot find ffn_up tensor\n");
        gguf_free(ctx);
        return 1;
    }

    fprintf(stderr, "Found tensor: %s\n", tensor_name);
    fprintf(stderr, "  ne: [%lld, %lld, %lld, %lld]\n", 
            (long long)tensor->ne[0], (long long)tensor->ne[1],
            (long long)tensor->ne[2], (long long)tensor->ne[3]);
    fprintf(stderr, "  type: %d\n", tensor->type);

    // Check dimensions: for Qwen2.5-0.5B, ffn_up should be [intermediate_size, hidden_size] = [4864, 896]
    // But ggml uses [hidden_size, intermediate_size] layout typically
    // Actually for Qwen2: ffn_up.weight shape is [intermediate_size, hidden_size] = [M, K]
    // But depending on orientation, we may need to transpose
    
    int64_t tensor_m = tensor->ne[0]; // first dim
    int64_t tensor_k = tensor->ne[1]; // second dim
    
    fprintf(stderr, "Tensor M=%lld K=%lld (expect M=%lld K=%lld)\n", 
            (long long)tensor_m, (long long)tensor_k, (long long)M, (long long)K);

    // Determine if tensor needs transpose: if tensor_m == K and tensor_k == M, we transpose
    bool needs_transpose = (tensor_m == K && tensor_k == M);
    bool needs_no_transpose = (tensor_m == M && tensor_k == K);
    
    if (!needs_transpose && !needs_no_transpose) {
        fprintf(stderr, "WARNING: tensor dimensions don't match expected [M=%lld,K=%lld] or [K=%lld,M=%lld]\n",
                (long long)M, (long long)K, (long long)K, (long long)M);
        fprintf(stderr, "  Will proceed with best guess\n");
        needs_transpose = true; // default to transpose
    }

    // Read tensor data
    // For Q4_K, we need to dequantize first
    fprintf(stderr, "Reading tensor data...\n");
    
    // Allocate contiguous f32 buffer
    std::vector<float> f32_weights(tensor_m * tensor_k);
    
    // Get tensor data from gguf
    void * tensor_data = NULL;
    struct ggml_context * tmp_ctx = ggml_init({.mem_size = 128*1024*1024});
    
    // Use ggml to dequantize
    struct ggml_tensor * src_tensor = tensor;
    
    // Create destination tensor
    struct ggml_tensor * f32_tensor = ggml_new_tensor_2d(tmp_ctx, GGML_TYPE_F32, tensor_m, tensor_k);
    ggml_set_param(tmp_ctx, f32_tensor);
    ggml_build_forward_expand(tmp_ctx, ggml_cpy(tmp_ctx, src_tensor, f32_tensor));
    
    // Compute
    struct ggml_cgraph gf = {};
    ggml_build_forward(&gf, f32_tensor);
    ggml_graph_compute_with_ctx(tmp_ctx, &gf, 1); // n_threads=1
    
    // Copy data
    memcpy(f32_weights.data(), f32_tensor->data, f32_weights.size() * sizeof(float));
    
    ggml_free(tmp_ctx);
    gguf_free(ctx);

    fprintf(stderr, "Successfully loaded %lld f32 weights\n", (long long)f32_weights.size());

    // Transpose if needed: we need W[K,M] but tensor provides W[M,K] or W[K,M]
    std::vector<float> final_weights(K * M);
    if (needs_transpose) {
        fprintf(stderr, "Transposing from [%lld,%lld] to [%lld,%lld]...\n",
                (long long)tensor_m, (long long)tensor_k, (long long)K, (long long)M);
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < M; m++) {
                // tensor[m,k] -> final[k,m]
                final_weights[k * M + m] = f32_weights[m * K + k];
            }
        }
    } else {
        memcpy(final_weights.data(), f32_weights.data(), final_weights.size() * sizeof(float));
    }

    // Quantize to INT8: use symmetric per-column scaling
    // Each column j has scale = max(|W[:,j]|) / 127.0
    // W[:,j]_int8 = round(W[:,j] / scale_j)
    fprintf(stderr, "Quantizing to INT8...\n");
    
    std::vector<int8_t> int8_weights(K * M);
    std::vector<float> scales(M);
    
    for (int64_t m = 0; m < M; m++) {
        // Find max absolute value in column
        float col_max = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            float abs_val = fabsf(final_weights[k * M + m]);
            if (abs_val > col_max) col_max = abs_val;
        }
        float scale = col_max / 127.0f;
        if (scale < 1e-6f) scale = 1e-6f;
        scales[m] = scale;
        
        for (int64_t k = 0; k < K; k++) {
            float val = final_weights[k * M + m] / scale;
            int int_val = (int)(val + (val >= 0 ? 0.5f : -0.5f));
            if (int_val > 127) int_val = 127;
            if (int_val < -128) int_val = -128;
            int8_weights[k * M + m] = (int8_t)int_val;
        }
    }

    // Write sidecar: int8 weights + scales
    fprintf(stderr, "Writing sidecar...\n");
    
    // Create output directory
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p /media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_0_5b_int8_layer0");
    system(cmd);

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "FAIL: cannot write %s\n", out_path);
        return 1;
    }
    
    // Write header: K, M, scales
    fwrite(&K, sizeof(int64_t), 1, f);
    fwrite(&M, sizeof(int64_t), 1, f);
    fwrite(scales.data(), sizeof(float), M, f);
    fwrite(int8_weights.data(), sizeof(int8_t), K * M, f);
    
    fclose(f);
    
    fprintf(stderr, "SUCCESS: wrote %s\n", out_path);
    fprintf(stderr, "  Header: K=%lld M=%lld\n", (long long)K, (long long)M);
    fprintf(stderr, "  Scales: %lld floats\n", (long long)M);
    fprintf(stderr, "  Weights: %lld INT8\n", (long long)(K*M));
    fprintf(stderr, "  Total size: %lld bytes\n", (long long)(2*sizeof(int64_t) + M*sizeof(float) + K*M));

    return 0;
}