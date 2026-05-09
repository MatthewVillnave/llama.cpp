// PRT Phase 16C: 14B Single-Tensor FFN_UP Extraction Probe
// Extracts one ffn_up tensor from Qwen2.5-14B-Instruct Q4_K_M GGUF
// Usage: ./build/bin/phase16c_14b_single_tensor_probe -m <model.gguf> -t <tensor_name> -o <output.bin>
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

int main(int argc, char ** argv) {
    const char * model_path = NULL;
    const char * tensor_name = "blk.0.ffn_up.weight";
    const char * output_path = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) tensor_name = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) output_path = argv[++i];
    }
    
    if (!model_path || !output_path) {
        fprintf(stderr, "Usage: %s -m <model.gguf> -t <tensor> -o <output.bin>\n", argv[0]);
        return 1;
    }
    
    fprintf(stderr, "=== PRT Phase 16C: 14B Single-Tensor Extraction ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Tensor: %s\n", tensor_name);
    
    // Init ggml
    struct ggml_init_params params = {
        /*.mem_size   = */ 512ull*1024ull*1024ull,
        /*.mem_buffer = */ NULL,
        /*.no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    // Load GGUF
    struct ggml_context * ctx_data = NULL;
    struct gguf_init_params uf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path, uf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "ERROR: gguf_init_from_file failed\n");
        return 1;
    }
    
    // Get tensor
    struct ggml_tensor * t = ggml_get_tensor(ctx_data, tensor_name);
    if (!t) {
        fprintf(stderr, "ERROR: tensor '%s' not found\n", tensor_name);
        gguf_free(ctx_gguf);
        return 1;
    }
    
    fprintf(stderr, "Shape: {%lld, %lld}\n", (long long)t->ne[0], (long long)t->ne[1]);
    fprintf(stderr, "Type: %s (%d)\n", ggml_type_name(t->type), t->type);
    fprintf(stderr, "Elements: %lld\n", (long long)ggml_nelements(t));
    
    int64_t n = ggml_nelements(t);
    
    // Get dequant function
    const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
    
    if (tt && tt->to_float) {
        float * float_data = (float *) malloc((size_t)n * sizeof(float));
        if (!float_data) {
            fprintf(stderr, "ERROR: malloc failed\n");
            return 1;
        }
        
        tt->to_float(t->data, float_data, n);
        
        // Basic stats
        int64_t n_finite = 0;
        float min_v = 0, max_v = 0, sum_v = 0;
        for (int64_t i = 0; i < n; i++) {
            float v = float_data[i];
            if (std::isfinite(v)) {
                n_finite++;
                sum_v += v;
                if (n_finite == 1) { min_v = max_v = v; }
                else {
                    if (v < min_v) min_v = v;
                    if (v > max_v) max_v = v;
                }
            }
        }
        
        fprintf(stderr, "\n=== LAYER STATS ===\n");
        fprintf(stderr, "Finite values: %lld / %lld\n", n_finite, n);
        fprintf(stderr, "Min: %.6f, Max: %.6f, Mean: %.6f\n", min_v, max_v, sum_v / n_finite);
        
        // Write output
        FILE * f = fopen(output_path, "wb");
        if (!f) {
            fprintf(stderr, "ERROR: cannot write %s\n", output_path);
            free(float_data);
            return 1;
        }
        fwrite(float_data, sizeof(float), (size_t)n, f);
        fclose(f);
        
        fprintf(stderr, "Wrote %zu bytes to %s\n", (size_t)n * sizeof(float), output_path);
        free(float_data);
    } else {
        fprintf(stderr, "ERROR: type %s cannot be dequantized\n", ggml_type_name(t->type));
        return 1;
    }
    
    gguf_free(ctx_gguf);
    return 0;
}