// PRT Phase 22B: Generate 0.5B ffn_up f32 sidecar
// Uses GGUF reader + ggml dequantization
// Output: f32 row-major [K,M] = [896, 4864]

#include "llama.h"
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

int main(int argc, char ** argv) {
    const char * model_path = 
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
    const char * out_path = 
        "/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_0_5b_int8_layer0/ffn_up_layer0_prt.f32";

    fprintf(stderr, "=== PRT Phase 22B: 0.5B f32 sidecar ===\n");
    fprintf(stderr, "Model: %s\n", model_path);

    // Load model via llama.cpp
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "FAIL: cannot load model\n");
        return 1;
    }
    fprintf(stderr, "Model loaded\n");

    // Get model context to find tensor
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1;
    llama_context * ctx = llama_new_context(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: cannot create context\n");
        llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "Context created\n");

    // Find tensor via ggml
    struct ggml_context * ggml_ctx = llama_get_gf()->ctx;
    struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, "blk.0.ffn_up.weight");
    if (!tensor) {
        fprintf(stderr, "FAIL: tensor blk.0.ffn_up.weight not found\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "Found tensor: ne=[%lld, %lld]\n", 
            (long long)tensor->ne[0], (long long)tensor->ne[1]);
    fprintf(stderr, "  type: %d (Q5_0=6)\n", tensor->type);

    // Dump tensor data
    void * data = tensor->data;
    int64_t ne0 = tensor->ne[0]; // K
    int64_t ne1 = tensor->ne[1]; // M

    // For f32 sidecar: W[K,M] row-major
    // tensor is [K, M] in ggml layout (already correct orientation!)
    // But stored as Q5_0, need to dequantize

    // Q5_0 dequantization:
    // 256 elements per block, 176 bytes per block
    // Format per block:
    //   2 bytes: d (float16 scale)
    //   2 bytes: dmin (float16 min)
    //   16 bytes: 8 scale pairs (4 bits each) for 8x32 sub-blocks
    //   160 bytes: 5-bit quantized values (256 elements)
    
    // We need to dequantize Q5_0 -> f32
    
    int64_t n_elements = ne0 * ne1; // K * M
    int64_t n_blocks = n_elements / 256;
    fprintf(stderr, "n_elements=%lld n_blocks=%lld\n", (long long)n_elements, (long long)n_blocks);

    // Allocate f32 buffer
    std::vector<float> f32_weights(n_elements);

    // Manually dequantize Q5_0
    // Block: 256 elements, 176 bytes
    // Byte layout (176 bytes):
    //   [0,1]:   d (float16, big-endian?)
    //   [2,3]:   dmin (float16)
    //   [4..19]: 16 bytes of scales (8 pairs, 4 bits each)
    //   [20..175]: 156 bytes of 5-bit values (249.6 bits... rounded to 256)

    // Actually Q5_0 packs 256 elements into 176 bytes:
    // 256 * 5 = 1280 bits = 160 bytes for values
    // Plus 16 bytes for scales (8 floats stored as 4-bit each)
    // Total = 176 bytes

    // Per sub-block (32 elements, 20 bytes):
    //   4 bytes: scale (float16)
    //   16 bytes: 5-bit values (32 * 5 = 160 bits)

    // Let me just use the ggml dequant directly
    // Create a ggml tensor and run to_float
    
    struct ggml_init_params ggml_params = {.mem_size = 128*1024*1024, .no_alloc = false};
    struct ggml_context * tmp_ctx = ggml_init({.mem_size = 128*1024*1024});
    
    // Create f32 tensor  
    struct ggml_tensor * f32_tensor = ggml_new_tensor_2d(tmp_ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_param(tmp_ctx, f32_tensor);
    
    // Copy source to f32
    struct ggml_tensor * src_tensor = tensor;
    struct ggml_cgraph gf = {};
    ggml_build_forward_expand(tmp_ctx, ggml_cpy(tmp_ctx, src_tensor, f32_tensor));
    ggml_graph_compute_with_ctx(tmp_ctx, &gf, 1);
    
    // Copy result
    memcpy(f32_weights.data(), f32_tensor->data, n_elements * sizeof(float));
    ggml_free(tmp_ctx);
    llama_free(ctx);
    llama_model_free(model);

    fprintf(stderr, "Dequantized to f32\n");

    // Write f32 sidecar (raw f32, no header for now)
    // For PRT ffn_up: W[K,M] row-major, 4 bytes per element
    fprintf(stderr, "Writing f32 sidecar to %s\n", out_path);
    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "FAIL: cannot write %s\n", out_path);
        return 1;
    }
    fwrite(f32_weights.data(), sizeof(float), n_elements, f);
    fclose(f);

    fprintf(stderr, "SUCCESS: wrote %lld bytes\n", (long long)(n_elements * sizeof(float)));
    return 0;
}