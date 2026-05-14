// PRT Phase 16E-REPAIR: Batch INT6 sidecar regeneration for Qwen2.5-14B
// GGUF Q4_K direct path — no intermediate float files
//
// Usage:
//   ./build/bin/llama-prt-ffn-up-extract \
//     -m /path/to/model.gguf \
//     -o /tmp/prt_sidecars_14b_int6/ \
//     --layers 1-39
//
// Or use this tool for batch regen with inline parity check.

#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

// Constants for Qwen2.5-14B ffn_up
static const int N_LAYERS = 40;
static const int M = 13824;   // ffn (rows)
static const int K = 5120;    // hidden (cols)
static const int QK_K = 256;  // Q4_K block size
static const float PRESCALE = 31.0f;

struct SidecarStats {
    int layer;
    int64_t file_size;
    bool finite_ok;
    float weight_cosine;
    float matvec_cosine;
    float mae;
    float rmse;
    float max_error;
    float norm_ratio;
    std::string sha_prefix;
};

// Read entire file into buffer
static std::vector<char> read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(len);
    fread(buf.data(), 1, len, f);
    fclose(f);
    return buf;
}

// Dequantize one Q4_K_M block (256 elements, 144 bytes) -> float32
// Based on Phase 15B working dequant path
static void dequant_q4_k_block(const uint8_t* block, float* result) {
    // d and dmin as float16
    float d  = Eigen::half::unpack(*(const uint16_t*)&block[0]).value;
    float dm = Eigen::half::unpack(*(const uint16_t*)&block[2]).value;
    
    // 8 scale values from 16 bytes
    for (int j = 0; j < 4; j++) {
        uint8_t s0 = block[4 + j*2];
        uint8_t s1 = block[4 + j*2 + 1];
        
        float sc0 = (s0 & 0xF);
        float m0  = (s0 >> 4);
        float sc1 = (s1 & 0xF);
        float m1v = (s1 >> 4);
        
        float d1 = d * sc0;
        float m1_val = dm * m0;
        float d2 = d * sc1;
        float m2_val = dm * m1v;
        
        int qp = j * 64;
        
        // 32 low nibbles
        for (int l = 0; l < 32; l++) {
            float q = block[20 + j*64 + l] & 0xF;
            result[qp++] = d1 * q - m1_val;
        }
        // 32 high nibbles
        for (int l = 0; l < 32; l++) {
            float q = (block[20 + j*64 + l] >> 4);
            result[qp++] = d2 * q - m2_val;
        }
    }
}

// Extract one ffn_up layer from GGUF
static bool extract_ffn_up_layer(ggml_context* ctx_data, 
                                  const char* tensor_name,
                                  float* W_out) {
    struct ggml_tensor* t = ggml_get_tensor(ctx_data, tensor_name);
    if (!t) {
        fprintf(stderr, "ERROR: tensor '%s' not found\n", tensor_name);
        return false;
    }
    
    // Get data pointer
    float* data = (float*)t->data;
    if (!data) {
        fprintf(stderr, "ERROR: tensor data is NULL\n");
        return false;
    }
    
    // Copy (the tensor should already be dequantized by gguf_init)
    // But we need to handle Q4_K -> float ourselves
    // Actually GGUF doesn't auto-dequant. We need to use ggml_type_traits.
    // For now, just copy and hope it's float. But for Q4_K we need manual dequant.
    
    return false; // Need proper approach
}

// Main: batch regenerate
int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* out_dir = "/tmp/prt_sidecars_14b_int6";
    int layer_start = -1, layer_end = -1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--layers") == 0 && i+1 < argc) {
            if (strchr(argv[++i], '-')) {
                sscanf(argv[i], "%d-%d", &layer_start, &layer_end);
            } else {
                layer_start = layer_end = atoi(argv[i]);
            }
        }
    }
    
    if (!model_path) {
        fprintf(stderr, "Usage: %s -m model.gguf [-o out_dir] [--layers 1-39]\n", argv[0]);
        return 1;
    }
    
    if (layer_start < 0) layer_start = 1;
    if (layer_end < 0) layer_end = 39;
    
    fprintf(stderr, "=== Phase 16E-REPAIR: Batch INT6 Regen ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Output: %s\n", out_dir);
    fprintf(stderr, "Layers: %d-%d\n", layer_start, layer_end);
    
    return 0;
}
