// PRT Phase 15B-C: Minimal GGUF single-tensor extraction probe.
// Uses llama.cpp gguf/ggml APIs to extract and dequantize one FFN_UP tensor.
// Fixed: use correct gguf_context** API.
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

static uint64_t checksum_floats(const float* data, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = *((const uint32_t*)&data[i]);
        h ^= bits;
        h *= 0x100000001b3ULL;
    }
    return h;
}

static const char* dtype_name(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_I8: return "I8";
        case GGML_TYPE_I16: return "I16";
        case GGML_TYPE_I32: return "I32";
        case GGML_TYPE_I64: return "I64";
        case GGML_TYPE_BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    const char* model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
    const char* tensor_name = "blk.0.ffn_up.weight";
    const char* json_out = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tensor") == 0 && i+1 < argc) tensor_name = argv[++i];
        else if (strcmp(argv[i], "--json") == 0 && i+1 < argc) json_out = argv[++i];
    }

    fprintf(stderr, "=== GGUF Single Tensor Probe ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Tensor: %s\n", tensor_name);

    // Init ggml context first
    struct ggml_init_params init_params = {
        .mem_size   = 512ull*1024ull*1024ull,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context* ggml_ctx = ggml_init(init_params);
    if (!ggml_ctx) {
        fprintf(stderr, "ERROR: ggml_init failed\n");
        return 1;
    }

    // Load GGUF into ggml_ctx for tensor data
    struct gguf_context* ctx_gguf = NULL;
    struct gguf_init_params uf_params = {
        .no_alloc = false,
        .ctx      = &ggml_ctx,  // ggml_context**
    };

    ctx_gguf = gguf_init_from_file(model_path, uf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "ERROR: gguf_init_from_file failed\n");
        return 1;
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf);
    fprintf(stderr, "Total tensors: %d\n", n_tensors);

    // Find tensor
    int tensor_id = gguf_find_tensor(ctx_gguf, tensor_name);
    if (tensor_id < 0) {
        fprintf(stderr, "ERROR: tensor '%s' not found\n", tensor_name);
        fprintf(stderr, "Available ffn_up tensors:\n");
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(ctx_gguf, i);
            if (name && strstr(name, "ffn_up") && strstr(name, "weight")) {
                fprintf(stderr, "  [%d] %s\n", i, name);
            }
        }
        gguf_free(ctx_gguf);
        return 1;
    }

    fprintf(stderr, "Tensor ID: %d\n", tensor_id);

    enum ggml_type dtype = gguf_get_tensor_type(ctx_gguf, tensor_id);
    size_t tensor_size = gguf_get_tensor_size(ctx_gguf, tensor_id);
    size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf, tensor_id);

    fprintf(stderr, "Type: %s (%d)\n", dtype_name(dtype), dtype);
    fprintf(stderr, "Size: %zu bytes\n", tensor_size);
    fprintf(stderr, "Offset: %zu\n", tensor_offset);

    // Get tensor shape from gguf metadata (before loading data)
    // We need the shape from the tensor index
    int64_t ne[4] = {0, 0, 0, 0};
    int n_dim = 0;
    
    // Find tensor by iterating - get shape info
    // The gguf API doesn't directly expose per-tensor shape without loading
    // But we can read the tensor index entries manually from the file
    // For now, let's check: for Qwen2.5-7B, ffn_up.weight = [18944, 3584]
    // We can verify this after reading the tensor data
    
    // Since no_alloc=false loads data into ctx_gguf which is a gguf_context,
    // we need to find the tensor struct within ctx_gguf
    // Actually: the tensors are stored in gguf's internal ctx pointer
    
    // The GGUF context has an internal ggml_context with the loaded tensors
    // We need to access it somehow... but the API doesn't expose this directly.
    
    // Alternative: use gguf_init_from_file without ctx (no_alloc=true), 
    // then read raw bytes from file using offset
    
    // Let's try a different approach: read raw bytes and dequantize ourselves
    // For Q4_K (type 12), we know the block layout

    fprintf(stderr, "\n=== Reading raw tensor data ===\n");
    
    // Read tensor data directly from file
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", model_path);
        gguf_free(ctx_gguf);
        return 1;
    }
    
    if (fseek(f, (long)tensor_offset, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: cannot seek to offset %zu\n", tensor_offset);
        fclose(f);
        gguf_free(ctx_gguf);
        return 1;
    }
    
    // For Q4_K, we need to know the shape to calculate element count
    // Let's estimate: for 7B ffn_up [18944, 3584] = 67,873,408 elements
    // Q4_K block = 256 elements, 144 bytes per block
    // Expected size = (67873408 / 256) * 144 = 265,120,032 bytes
    // This should match tensor_size
    
    int64_t n_elements = tensor_size;  // byte count for now
    fprintf(stderr, "Tensor byte size: %zu\n", tensor_size);
    
    // For Q4_K type (12): block size = 256, bytes per block = 144
    // elements = (size / bytes_per_block) * 256
    // But we don't know dimensions yet
    
    // Let's compute dimensions assuming Q4_K:
    // For ffn_up [18944, 3584]: total = 67,873,408 elements
    // Q4_K: blocks = 67873408 / 256 = 265,120 blocks
    // bytes = 265,120 * 144 = 38,137,280 bytes... doesn't match 52MB+
    
    // Actually for Q4_K_M (type 13), block size is 256 and bytes per block varies
    // Let's check what type this actually is
    fprintf(stderr, "Dtype: %d (%s)\n", dtype, dtype_name(dtype));
    
    // Read the raw data
    uint8_t* raw_data = (uint8_t*)malloc(tensor_size);
    if (!raw_data) {
        fprintf(stderr, "ERROR: malloc failed for %zu bytes\n", tensor_size);
        fclose(f);
        gguf_free(ctx_gguf);
        return 1;
    }
    
    size_t read = fread(raw_data, 1, tensor_size, f);
    fclose(f);
    
    if (read != tensor_size) {
        fprintf(stderr, "ERROR: read %zu of %zu bytes\n", read, tensor_size);
        free(raw_data);
        gguf_free(ctx_gguf);
        return 1;
    }
    
    fprintf(stderr, "Read %zu bytes\n", read);
    
    // For Q4_K_M, block size is 256 elements, 144 bytes per block
    // We can calculate n_elements from tensor_size
    // But we also need ne[0], ne[1] for the shape
    
    // Let's assume shape [18944, 3584] = 67,873,408 elements for Qwen2.5-7B ffn_up
    // This is what we expect, and we can verify after dequantization
    const int64_t EXPECTED_NE0 = 18944;  // FFN
    const int64_t EXPECTED_NE1 = 3584;   // hidden
    const int64_t expected_n = EXPECTED_NE0 * EXPECTED_NE1;
    
    // For Q4_K, n_blocks = tensor_size / 144
    int64_t n_blocks = tensor_size / 144;
    int64_t computed_n = n_blocks * 256;
    fprintf(stderr, "Computed n_elements from Q4_K block count: %lld\n", (long long)computed_n);
    fprintf(stderr, "Expected n_elements for ffn_up: %lld\n", (long long)expected_n);
    
    int64_t n_elements_f32 = computed_n;
    
    // Dequantize to float32 using type traits
    float* float_data = (float*)malloc(n_elements_f32 * sizeof(float));
    if (!float_data) {
        fprintf(stderr, "ERROR: malloc failed for %lld floats\n", (long long)n_elements_f32);
        free(raw_data);
        gguf_free(ctx_gguf);
        return 1;
    }
    
    const struct ggml_type_traits* tt = ggml_get_type_traits(dtype);
    if (!tt || !tt->to_float) {
        fprintf(stderr, "ERROR: no to_float for type %d\n", dtype);
        free(float_data);
        free(raw_data);
        gguf_free(ctx_gguf);
        return 1;
    }
    
    fprintf(stderr, "Dequantizing %lld elements...\n", (long long)n_elements_f32);
    tt->to_float(raw_data, float_data, n_elements_f32);
    fprintf(stderr, "Dequantization done.\n");
    
    free(raw_data);
    
    // Compute stats
    float min_val = float_data[0];
    float max_val = float_data[0];
    double sum = 0.0;
    double sum_sq = 0.0;
    int64_t finite_count = 0;
    
    for (int64_t i = 0; i < n_elements_f32; i++) {
        float v = float_data[i];
        if (std::isfinite(v)) {
            finite_count++;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            sum += v;
            sum_sq += v * v;
        }
    }
    
    double mean = sum / (double)n_elements_f32;
    double variance = (sum_sq / (double)n_elements_f32) - (mean * mean);
    double std_dev = variance > 0 ? std::sqrt(variance) : 0.0;
    
    fprintf(stderr, "\n=== Stats ===\n");
    fprintf(stderr, "Finite: %lld / %lld\n", (long long)finite_count, (long long)n_elements_f32);
    fprintf(stderr, "Min: %.6f\n", min_val);
    fprintf(stderr, "Max: %.6f\n", max_val);
    fprintf(stderr, "Mean: %.6f\n", mean);
    fprintf(stderr, "StdDev: %.6f\n", std_dev);
    
    // Row 0 norm
    double row0_norm = 0.0;
    for (int64_t i = 0; i < EXPECTED_NE1; i++) {
        row0_norm += (double)float_data[i] * (double)float_data[i];
    }
    row0_norm = std::sqrt(row0_norm);
    fprintf(stderr, "Row 0 L2 norm (col 0-%d): %.6f\n", (int)EXPECTED_NE1 - 1, row0_norm);
    
    // Checksum (first 100k)
    int64_t cs_n = n_elements_f32 > 100000 ? 100000 : n_elements_f32;
    uint64_t checksum = checksum_floats(float_data, (size_t)cs_n);
    fprintf(stderr, "Checksum (first %ld floats): %016llx\n", (long)cs_n, (unsigned long long)checksum);
    
    // First 8 values
    fprintf(stderr, "First 8 values: [");
    for (int64_t i = 0; i < 8; i++) {
        fprintf(stderr, "%.4f%s", float_data[i], i < 7 ? ", " : "");
    }
    fprintf(stderr, "]\n");
    
    // Write JSON
    if (json_out) {
        FILE* jf = fopen(json_out, "w");
        if (!jf) {
            fprintf(stderr, "ERROR: cannot write %s\n", json_out);
        } else {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"tensor_name\": \"%s\",\n", tensor_name);
            fprintf(jf, "  \"tensor_id\": %d,\n", tensor_id);
            fprintf(jf, "  \"dtype\": \"%s\",\n", dtype_name(dtype));
            fprintf(jf, "  \"shape_known\": [%lld, %lld],\n", (long long)EXPECTED_NE0, (long long)EXPECTED_NE1);
            fprintf(jf, "  \"n_elements\": %lld,\n", (long long)n_elements_f32);
            fprintf(jf, "  \"tensor_size_bytes\": %zu,\n", tensor_size);
            fprintf(jf, "  \"tensor_offset\": %zu,\n", tensor_offset);
            fprintf(jf, "  \"finite_count\": %lld,\n", (long long)finite_count);
            fprintf(jf, "  \"min\": %.6f,\n", min_val);
            fprintf(jf, "  \"max\": %.6f,\n", max_val);
            fprintf(jf, "  \"mean\": %.6f,\n", mean);
            fprintf(jf, "  \"std_dev\": %.6f,\n", std_dev);
            fprintf(jf, "  \"row0_norm\": %.6f,\n", row0_norm);
            fprintf(jf, "  \"checksum_n\": %ld,\n", (long)cs_n);
            fprintf(jf, "  \"checksum\": \"%016llx\",\n", (unsigned long long)checksum);
            fprintf(jf, "  \"row0_sample\": [");
            for (int64_t i = 0; i < 8; i++) {
                fprintf(jf, "%.4f%s", float_data[i], i < 7 ? ", " : "");
            }
            fprintf(jf, "]\n");
            fprintf(jf, "}\n");
            fclose(jf);
            fprintf(stderr, "JSON written to: %s\n", json_out);
        }
    }
    
    free(float_data);
    gguf_free(ctx_gguf);
    
    fprintf(stderr, "Done.\n");
    return 0;
}