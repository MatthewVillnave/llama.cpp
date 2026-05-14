// PRT Phase 10A: GGUF tensor extraction using gguf API
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

const char * dtype_name(int dt) {
    switch(dt) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0";
        case 3: return "Q5_0"; case 4: return "Q8_0"; case 5: return "Q2_0";
        case 6: return "Q3_0"; case 7: return "Q4_1"; case 8: return "Q5_1";
        case 9: return "Q1_0"; case 10: return "Q4_2"; case 11: return "Q6_K";
        case 12: return "Q8_K"; case 13: return "IQ2_XXS"; case 14: return "IQ2_XS";
        case 15: return "IQ3_XXS"; case 16: return "IQ1_S"; case 17: return "IQ4_NL";
        case 18: return "IQ3_S"; case 19: return "IQ2_S"; case 20: return "IQ4_XS";
        case 21: return "I8"; case 255: return "F64"; case 254: return "BF16";
        default: return "UNKNOWN";
    }
}

int main(int argc, char ** argv) {
    const char * path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
    if (argc > 1) path = argv[1];
    
    fprintf(stderr, "=== PRT Phase 10A: GGUF Tensor Extraction ===\n");
    
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ nullptr,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        fprintf(stderr, "ERROR: gguf_init_from_file failed\n");
        return 1;
    }
    
    int n_tensors = gguf_get_n_tensors(ctx);
    fprintf(stderr, "Total tensors: %d\n", n_tensors);
    
    // Scan for ffn_up tensors
    std::vector<std::string> ffn_up_names;
    int ffn_up_count = 0;
    
    fprintf(stderr, "\nSearching for ffn_up tensors...\n");
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        if (name && strstr(name, "ffn_up") && strstr(name, "weight")) {
            size_t size = gguf_get_tensor_size(ctx, i);
            size_t offset = gguf_get_tensor_offset(ctx, i);
            auto type = gguf_get_tensor_type(ctx, i);
            ffn_up_count++;
            fprintf(stderr, "FFN_UP[%d]: idx=%d name=%s type=%s size=%zu offset=%zu\n",
                    ffn_up_count, i, name, dtype_name(type), size, offset);
            if (ffn_up_count <= 3) {
                ffn_up_names.push_back(name);
            }
        }
    }
    
    fprintf(stderr, "\n=== RESULT ===\n");
    fprintf(stderr, "ffn_up tensors found: %d\n", ffn_up_count);
    
    if (ffn_up_count > 0) {
        fprintf(stderr, "ffn_up[0]: %s\n", ffn_up_names[0].c_str());
    }
    
    gguf_free(ctx);
    return 0;
}