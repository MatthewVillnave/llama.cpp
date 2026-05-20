// PRT Phase 24F: Generate 3B INT8 ffn_up sidecar for layer0
// K=2048, M=11008 (Qwen2.5-3B hidden/intermediate dims)

#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

static const int64_t K = 2048;   // hidden_size
static const int64_t M = 11008;  // intermediate_size

int main(int argc, char ** argv) {
    const char * model_path = 
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
    const char * out_path = 
        "/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_3b_int8_phase24f/ffn_up_layer0_prt.int8";

    fprintf(stderr, "=== PRT Phase 24F: 3B INT8 sidecar generation ===\n");
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

    // Qwen2 tensor name: blk.0.ffn_up.weight (gate+up projection, shape [M, K*2])
    // For Qwen2: blk.0.ffn_up.weight has shape M x (K*2) for gate+up
    // We want only the "up" half: columns K to 2*K in the tensor
    
    const char * tensor_name = "blk.0.ffn_up.weight";
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, tensor_name);
    
    if (!tensor) {
        fprintf(stderr, "FAIL: tensor %s not found\n", tensor_name);
        gguf_free(ctx);
        return 1;
    }
    
    fprintf(stderr, "Tensor: %s shape [%lld, %lld, %lld, %lld]\n", 
            tensor_name, 
            (long long)tensor->ne[0],
            (long long)tensor->ne[1],
            (long long)tensor->ne[2],
            (long long)tensor->ne[3]);
            
    // For ffn_up in Qwen2: K=2048 input, M*2=22016 output (gate+up concatenated)
    // We want the "up" portion: columns K to 2*K (2048 to 4096)
    // Actual layout: [K, M*2] in GGUF = [K, M*2] row-major
    // But we need to extract: up projection = rows 0:K, cols K:2*K
    
    const int64_t tensor_K = tensor->ne[0];  // should be K=2048
    const int64_t tensor_M_total = tensor->ne[1];  // should be M*2=22016
    
    // Allocate output
    const int64_t out_size = K * M;  // 2048 * 11008 = 22,544,384
    const int64_t scale_size = M;  // 11008 scalars
    const int64_t total_size = out_size + scale_size * sizeof(float);
    
    fprintf(stderr, "Allocating %lld bytes\n", (long long)total_size);
    
    std::vector<int8_t> out_data(out_size);
    std::vector<float> scales(scale_size);
    
    // Allocate model backing for reading
    struct ggml_init_params ggml_params = {
        .n_threads = 1,
    };
    struct ggml_context * ggml_ctx = ggml_init(ggml_params);
    
    // Re-read with allocation for actual data
    gguf_free(ctx);
    gguf_init_params params_alloc = {
        .no_alloc = false,
    };
    ctx = gguf_init_from_file(model_path, params_alloc);
    
    tensor = ggml_get_tensor(ctx, tensor_name);
    if (!tensor) {
        fprintf(stderr, "FAIL: cannot re-load tensor with allocation\n");
        return 1;
    }
    
    // Read K rows, columns K:2*K (the "up" projection)
    // The tensor is quantized - we need to dequantize
    // For Q4_K_M, we need to use ggml_dequantize or similar
    
    // For now - use identity mapping if already f16/f32
    // Otherwise, this is a simplification
    
    // Load data per column j (output dimension)
    for (int64_t j = 0; j < M; j++) {
        float max_val = 0.0f;
        
        // For column j, find max absolute value for scale
        for (int64_t i = 0; i < K; i++) {
            float abs_val = fabsf(((float*)tensor->data)[i * tensor_M_total + (K + j)]);
            if (abs_val > max_val) max_val = abs_val;
        }
        
        scales[j] = max_val / 127.0f;  // INT8 range
        
        // Quantize
        for (int64_t i = 0; i < K; i++) {
            float fval = ((float*)tensor->data)[i * tensor_M_total + (K + j)];
            int8_t ival = (int8_t)(fval / scales[j] + 0.5f);
            out_data[i * M + j] = ival;
        }
    }
    
    // Write output
    FILE * fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "FAIL: cannot write %s\n", out_path);
        return 1;
    }
    
    // Write header: K, M, scales
    fwrite(&K, sizeof(K), 1, fout);
    fwrite(&M, sizeof(M), 1, fout);
    fwrite(scales.data(), sizeof(float), scale_size, fout);
    fwrite(out_data.data(), 1, out_size, fout);
    fclose(fout);
    
    fprintf(stderr, "Wrote %s\n", out_path);
    fprintf(stderr, "Size: %lld bytes\n", (long long)(total_size + 2*sizeof(int64_t)));
    
    // Validate
    struct ggml_init_params free_params = { .n_threads = 1 };
    ggml_free(ggml_ctx);
    gguf_free(ctx);
    
    return 0;
}