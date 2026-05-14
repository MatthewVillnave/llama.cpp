// PRT Phase 10A: Extract and dequant any ffn_up tensor
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char ** argv) {
    const char * model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
    const char * tensor_name = "blk.0.ffn_up.weight";
    const char * output_path = "/tmp/ffn_up_layer_float.bin";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) tensor_name = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) output_path = argv[++i];
    }
    
    fprintf(stderr, "=== PRT Phase 10A: Extract + Dequant ===\n");
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
    
    fprintf(stderr, "Shape: {%d, %d}\n", (int)t->ne[0], (int)t->ne[1]);
    fprintf(stderr, "Type: %s (%d)\n", ggml_type_name(t->type), t->type);
    fprintf(stderr, "Elements: %lld\n", (long long)ggml_nelements(t));
    
    int64_t n = ggml_nelements(t);
    
    // Get dequant function
    const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
    
    if (tt && tt->to_float) {
        float * float_data = (float *) malloc(n * sizeof(float));
        if (!float_data) {
            fprintf(stderr, "ERROR: malloc failed\n");
            return 1;
        }
        
        tt->to_float(t->data, float_data, n);
        
        FILE * f = fopen(output_path, "wb");
        if (!f) {
            fprintf(stderr, "ERROR: cannot write %s\n", output_path);
            free(float_data);
            return 1;
        }
        fwrite(float_data, sizeof(float), n, f);
        fclose(f);
        
        fprintf(stderr, "Wrote %zu bytes to %s\n", n * sizeof(float), output_path);
        free(float_data);
    } else {
        fprintf(stderr, "ERROR: type %s cannot be dequantized\n", ggml_type_name(t->type));
        return 1;
    }
    
    gguf_free(ctx_gguf);
    return 0;
}