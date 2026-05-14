// PRT Phase 16E-REPAIR: GGUF Q4_K -> INT6, one layer at a time
// Uses llama.cpp ggml/gguf APIs directly, no intermediate float files.
// Usage: ./phase16e_14b_int6_cpp -m model.gguf -o outdir [--layer N] [--from L --to R]
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

static const int M = 13824;
static const int K = 5120;
static const float PRESCALE = 31.0f;
static const int QK_K = 256;
static const int SIDECAR_SIZE = 20 + M*4 + M*(((K+3)/4)*3);

static void quantize_row(const float* row, float scale, uint8_t* packed_bytes) {
    for (int i = 0; i < K; i += 4) {
        int v0 = (int)std::round(row[i] / scale);
        int v1 = (i + 1 < K) ? (int)std::round(row[i + 1] / scale) : 0;
        int v2 = (i + 2 < K) ? (int)std::round(row[i + 2] / scale) : 0;
        int v3 = (i + 3 < K) ? (int)std::round(row[i + 3] / scale) : 0;
        v0 = std::max(-31, std::min(31, v0));
        v1 = std::max(-31, std::min(31, v1));
        v2 = std::max(-31, std::min(31, v2));
        v3 = std::max(-31, std::min(31, v3));
        uint8_t b0 = (uint8_t)((v0 + 32) & 0x3F);
        uint8_t b1 = (uint8_t)((v1 + 32) & 0x3F);
        uint8_t b2 = (uint8_t)((v2 + 32) & 0x3F);
        uint8_t b3 = (uint8_t)((v3 + 32) & 0x3F);
        int bi = (i / 4) * 3;
        packed_bytes[bi + 0] = b0 | ((b1 & 0x03) << 6);
        packed_bytes[bi + 1] = (uint8_t)(((b1 >> 2) & 0x0F) | ((b2 & 0x0F) << 4));
        packed_bytes[bi + 2] = (uint8_t)(((b2 >> 4) & 0x03) | ((b3 & 0x3F) << 2));
    }
}

static void write_sidecar(const char* path, const float* scales, uint8_t* packed_data) {
    uint8_t* buf = (uint8_t*)malloc(SIDECAR_SIZE);
    memset(buf, 0, SIDECAR_SIZE);
    buf[0] = 'P'; buf[1] = 'R'; buf[2] = 'T'; buf[3] = '6';
    *(uint32_t*)(buf + 4) = 1;
    *(uint32_t*)(buf + 8) = (uint32_t)M;
    *(uint32_t*)(buf + 12) = (uint32_t)K;
    *(uint32_t*)(buf + 16) = 0;
    memcpy(buf + 20, scales, M * 4);
    memcpy(buf + 20 + M * 4, packed_data, M * (((K + 3) / 4) * 3));
    FILE* f = fopen(path, "wb");
    if (!f) { free(buf); return; }
    fwrite(buf, 1, SIDECAR_SIZE, f);
    fclose(f);
    free(buf);
}

static bool process_layer(ggml_context* ctx_data, int layer, const char* out_dir) {
    char tensor_name[64];
    snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_up.weight", layer);
    struct ggml_tensor* t = ggml_get_tensor(ctx_data, tensor_name);
    if (!t) { fprintf(stderr, "ERROR: tensor '%s' not found\n", tensor_name); return false; }
    if (t->type != 12) { fprintf(stderr, "ERROR: type %d != Q4_K_M(12)\n", t->type); return false; }

    int64_t n = ggml_nelements(t);
    fprintf(stderr, "Layer %d: shape={%lld,%lld}, elements=%lld\n",
            layer, (long long)t->ne[0], (long long)t->ne[1], (long long)n);

    const struct ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) { fprintf(stderr, "ERROR: cannot dequant\n"); return false; }

    float* W = (float*)malloc((size_t)n * sizeof(float));
    float* scales = (float*)malloc(M * sizeof(float));
    uint8_t* packed = (uint8_t*)malloc(M * (((K + 3) / 4) * 3));
    if (!W || !scales || !packed) { free(W); free(scales); free(packed); return false; }

    tt->to_float(t->data, W, n);

    for (int r = 0; r < M; r++) {
        const float* row = W + (size_t)r * K;
        float row_max = 0.0f;
        for (int c = 0; c < K; c++) { float a = fabsf(row[c]); if (a > row_max) row_max = a; }
        scales[r] = (row_max < 1e-8f) ? 1e-8f : row_max / PRESCALE;
        quantize_row(row, scales[r], packed + (size_t)r * (((K + 3) / 4) * 3));
    }

    char out_path[512];
    snprintf(out_path, sizeof(out_path), "%s/ffn_up_layer%d_prt.int6", out_dir, layer);
    write_sidecar(out_path, scales, packed);
    fprintf(stderr, "Layer %d: wrote %s (%d bytes)\n", layer, out_path, SIDECAR_SIZE);

    free(W); free(scales); free(packed);
    return true;
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* out_dir = "/tmp/prt_sidecars_14b_int6";
    int layer = -1, from_layer = -1, to_layer = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--layer") == 0 && i+1 < argc) layer = atoi(argv[++i]);
        else if (strcmp(argv[i], "--from") == 0 && i+1 < argc) from_layer = atoi(argv[++i]);
        else if (strcmp(argv[i], "--to") == 0 && i+1 < argc) to_layer = atoi(argv[++i]);
    }

    if (!model_path) { fprintf(stderr, "Usage: %s -m model.gguf -o outdir [--layer N] [--from L --to R]\n", argv[0]); return 1; }
    fprintf(stderr, "=== GGUF->INT6 ===\nModel: %s\nOutput: %s\n", model_path, out_dir);

    struct ggml_init_params params = { .mem_size = 256ull*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    struct ggml_context* ctx_data = NULL;
    struct gguf_init_params uf_params = { .no_alloc = false, .ctx = &ctx_data };
    struct gguf_context* ctx_gguf = gguf_init_from_file(model_path, uf_params);
    if (!ctx_gguf) { fprintf(stderr, "gguf_init_from_file failed\n"); ggml_free(ctx); return 1; }

    int success = 0, total = 0;
    if (layer >= 0) { total = 1; if (process_layer(ctx_data, layer, out_dir)) success++; }
    else {
        int start = (from_layer >= 0) ? from_layer : 1;
        int end = (to_layer >= 0) ? to_layer : 39;
        total = end - start + 1;
        for (int l = start; l <= end; l++) { if (process_layer(ctx_data, l, out_dir)) success++; }
    }

    fprintf(stderr, "\nSuccess: %d/%d layers\n", success, total);
    gguf_free(ctx_gguf); ggml_free(ctx);
    return (success == total) ? 0 : 1;
}
