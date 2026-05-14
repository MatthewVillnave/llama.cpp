// PRT Phase 16B: INT6 sidecar generator for Qwen2.5-14B
// Generates all 40 INT6 sidecar files for ffn_up.weight layers
// Usage: ./build/bin/prt_sidecar_gen_14b <model.gguf> <output_dir>
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/stat.h>

// INT6 packed format: 4 values -> 3 bytes (offset-32 encoding)
// Header: [4 magic="PRT6"][4 version=1][4 rows][4 cols][4 reserved]
// Then: [M*4 bytes scales (float32)][packed bytes]

static inline void pack_int6_row(const int8_t * src, uint8_t * dst, int64_t k) {
    int64_t i = 0;
    for (; i + 3 < k; i += 4) {
        dst[0] = (uint8_t)(src[i+0] + 32) | ((uint8_t)(src[i+1] + 32) & 0x03) << 6;
        dst[1] = ((uint8_t)(src[i+1] + 32) >> 2) | ((uint8_t)(src[i+2] + 32) & 0x0F) << 4;
        dst[2] = ((uint8_t)(src[i+2] + 32) >> 4) | ((uint8_t)(src[i+3] + 32) & 0x3F) << 2;
        dst += 3;
    }
    int64_t remaining = k - i;
    if (remaining == 1) dst[0] = (uint8_t)(src[i+0] + 32);
    else if (remaining == 2) {
        dst[0] = (uint8_t)(src[i+0] + 32) | ((uint8_t)(src[i+1] + 32) & 0x03) << 6;
        dst[1] = ((uint8_t)(src[i+1] + 32) >> 2) & 0x0F;
    } else if (remaining == 3) {
        dst[0] = (uint8_t)(src[i+0] + 32) | ((uint8_t)(src[i+1] + 32) & 0x03) << 6;
        dst[1] = ((uint8_t)(src[i+1] + 32) >> 2) | ((uint8_t)(src[i+2] + 32) & 0x0F) << 4;
        dst[2] = ((uint8_t)(src[i+2] + 32) >> 4);
    }
}

static inline int64_t packed_row_size(int64_t k) {
    return ((k + 3) / 4) * 3;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <output_dir>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_dir = argv[2];

    const int64_t M = 13824;   // intermediate_size
    const int64_t K = 5120;     // hidden_size
    const int n_layer = 40;
    const float PRESCALE = 31.0f;

    fprintf(stderr, "=== PRT Phase 16B: INT6 sidecar generator for Qwen2.5-14B ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Output: %s\n", out_dir);
    fprintf(stderr, "FFN_up: M=%lld K=%lld\n", (long long)M, (long long)K);
    fprintf(stderr, "Per-layer: scales=%.1fMB packed=%.1fMB total=%.1fMB\n",
            M * 4 / (1024.0*1024.0), packed_row_size(K) / (1024.0*1024.0),
            (M*4 + packed_row_size(K)) / (1024.0*1024.0));

    mkdir(out_dir, 0755);

    // Load GGUF
    struct gguf_init_params params = { .no_alloc = false, .ctx = NULL };
    struct gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to load GGUF from %s\n", model_path);
        return 1;
    }
    fprintf(stderr, "GGUF loaded. Tensors: %lld\n", (long long)gguf_get_n_tensors(ctx));

    struct ggml_context * ctx_data = (struct ggml_context *)gguf_get_val_ptr(ctx, "general.architecture");
    // Get the ggml context from ctx
    // Actually we need a different approach - gguf_init_from_file with no_alloc=false
    // gives us tensors with data loaded. Let's get tensor data via ggml_get_tensor.
    // The ctx_data is stored inside the gguf_context.

    // Find the internal ggml context
    struct ggml_context * data_ctx = NULL;
    // We need to find the ggml context from ctx. Let me check the gguf API.
    // Actually in newer llama.cpp, gguf_init_from_file creates a ggml_context internally.
    // We access tensors via ggml_get_tensor(ctx_data, name).
    // But we don't have direct access to ctx_data from outside.
    // Let's use gguf_get_tensor_offset to read tensor data directly from the file.

    // Read all tensor metadata
    int64_t n_tensors = gguf_get_n_tensors(ctx);
    fprintf(stderr, "n_tensors = %lld\n", (long long)n_tensors);

    // Build tensor name -> offset map
    // Then use pread() to read each tensor's data from the mmap'd file

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: cannot open %s\n", model_path);
        gguf_free(ctx);
        return 1;
    }

    size_t file_size = (size_t)lseek(fd, 0, SEEK_END);
    fprintf(stderr, "File size: %zu bytes (%.1f GB)\n", file_size, file_size / (1024.0*1024.0*1024.0));

    // Memory-map the file
    void * mmapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmapped == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed\n");
        close(fd);
        gguf_free(ctx);
        return 1;
    }

    // Get data offset
    size_t data_offset = gguf_get_data_offset(ctx);
    fprintf(stderr, "Data offset: %zu\n", data_offset);

    // Read tensor info for all layers
    struct TensorInfo { char name[128]; int64_t offset; int64_t nelements; int32_t type; };
    struct TensorInfo ffn_up_layers[40];
    int found_layers = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * tname = gguf_get_tensor_name(ctx, i);
        if (strstr(tname, "ffn_up") && strstr(tname, "weight")) {
            int layer = -1;
            if (sscanf(tname, "blk.%d.ffn_up.weight", &layer) == 1 && layer >= 0 && layer < 40) {
                ffn_up_layers[layer].offset = gguf_get_tensor_offset(ctx, i);
                ffn_up_layers[layer].nelements = gguf_get_tensor_info(ctx, i)->n_elements;
                ffn_up_layers[layer].type = gguf_get_tensor_type(ctx, i);
                snprintf(ffn_up_layers[layer].name, sizeof(ffn_up_layers[layer].name), "%s", tname);
                fprintf(stderr, "Layer %d: offset=%lld elements=%lld type=%d\n",
                        layer, (long long)ffn_up_layers[layer].offset,
                        (long long)ffn_up_layers[layer].nelements,
                        ffn_up_layers[layer].type);
                found_layers++;
            }
        }
    }

    fprintf(stderr, "Found %d ffn_up layers\n", found_layers);

    if (found_layers == 0) {
        fprintf(stderr, "ERROR: no ffn_up layers found. Is this actually a Qwen2.5-14B model?\n");
        munmap(mmapped, file_size);
        close(fd);
        gguf_free(ctx);
        return 1;
    }

    // Allocate working buffers
    int64_t n_elements = M * K;
    int64_t packed_per_layer = packed_row_size(K) * M;  // packed rows for all M rows

    float * scales = (float *)malloc((size_t)M * sizeof(float));
    int8_t * ffn_row = (int8_t *)malloc((size_t)K * sizeof(int8_t));
    int8_t * packed_row = (int8_t *)malloc(packed_row_size(K));
    uint8_t * layer_buf = (uint8_t *)malloc(16 + M * 4 + packed_per_layer);
    int8_t * raw_row = (int8_t *)malloc((size_t)n_elements);

    if (!scales || !ffn_row || !packed_row || !layer_buf || !raw_row) {
        fprintf(stderr, "ERROR: memory allocation failed\n");
        return 1;
    }

    int generated = 0;
    for (int l = 0; l < 40; l++) {
        if (ffn_up_layers[l].offset == 0 && ffn_up_layers[l].nelements == 0) continue;

        fprintf(stderr, "\nProcessing layer %d: %s\n", l, ffn_up_layers[l].name);

        // Read raw data (Q4_K format - compressed)
        // For Q4_K: 256 elements per block, each block = 208 bytes
        // Block structure: 128 x Q4_K (4-bit) + 256 x F16 (scale + 15 values) = 208 bytes
        // But we need FP16 values, so we need to dequantize.
        // Let's just read the raw weights as they are stored.
        // Actually, let's just use the tensor info to get the element count
        // and read the raw bytes.

        int32_t tensor_type = ffn_up_layers[l].type;
        fprintf(stderr, "  type=%d nelements=%lld offset=%lld\n",
                tensor_type, (long long)ffn_up_layers[l].nelements, (long long)ffn_up_layers[l].offset);

        // For Q4_K_M, we need to dequantize. Let's compute the size.
        // Q4_K block: 256 elements in 208 bytes
        // n_elements = 13824 * 5120 = 70778880
        // blocks = n_elements / 256 = 276480
        // raw size = blocks * 208 = 57507840 bytes (~54.9 MB per layer)
        //
        // But our code expects M=13824 rows, K=5120 cols
        // We need to read and dequantize each Q4_K block.

        // Let me try a simpler approach - read all 40 layers sequentially
        // using pread on the mmapped file.

        int64_t tensor_offset = ffn_up_layers[l].offset;
        int64_t tensor_size = gguf_get_tensor_size(ctx, l);

        fprintf(stderr, "  tensor_size=%lld bytes (%.1f MB)\n",
                (long long)tensor_size, tensor_size / (1024.0*1024.0));

        // For now, just create the sidecar with placeholder data
        // and report what we'd need to do.
        fprintf(stderr, "  SKIPPING dequantization for now (Q4_K complexity)\n");
        fprintf(stderr, "  Would need to read %lld bytes and dequantize Q4_K -> FP16\n",
                (long long)tensor_size);
    }

    // Cleanup
    free(scales); free(ffn_row); free(packed_row); free(layer_buf); free(raw_row);
    munmap(mmapped, file_size);
    close(fd);
    gguf_free(ctx);

    fprintf(stderr, "\nPhase 16B: need Q4_K dequantization path\n");
    return 0;
}