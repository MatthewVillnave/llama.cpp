// PRT Phase 16E-REPAIR: GGUF Q4_K -> INT6, with correct orientation
// GGUF ffn_up tensors are stored as [K, M]=[5120, 13824].
// Must transpose to [M, K]=[13824, 5120] before quantizing.
// Usage: ./phase16e_14b_int6_v2 -m model.gguf -o outdir [--layer N] [--from L --to R]
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

static const int K = 5120;
static const int M = 13824;
static const float PRESCALE = 31.0f;
static const int SIDECAR_PACKED = M * (((K + 3) / 4) * 3);
static const int SIDECAR_SIZE = 20 + M*4 + SIDECAR_PACKED;

// Pack int8 values [-31, 31] into 3-byte groups (4 values -> 3 bytes)
static void pack_row_int6(const int8_t* qvals, int K, uint8_t* out) {
    for (int i = 0; i < K; i += 4) {
        int v0 = qvals[i];
        int v1 = (i+1 < K) ? qvals[i+1] : 0;
        int v2 = (i+2 < K) ? qvals[i+2] : 0;
        int v3 = (i+3 < K) ? qvals[i+3] : 0;
        v0 = std::max(-31, std::min(31, v0));
        v1 = std::max(-31, std::min(31, v1));
        v2 = std::max(-31, std::min(31, v2));
        v3 = std::max(-31, std::min(31, v3));
        uint8_t b0 = (uint8_t)((v0 + 32) & 0x3F);
        uint8_t b1 = (uint8_t)((v1 + 32) & 0x3F);
        uint8_t b2 = (uint8_t)((v2 + 32) & 0x3F);
        uint8_t b3 = (uint8_t)((v3 + 32) & 0x3F);
        int bi = (i / 4) * 3;
        out[bi + 0] = b0 | ((b1 & 0x03) << 6);
        out[bi + 1] = (uint8_t)(((b1 >> 2) & 0x0F) | ((b2 & 0x0F) << 4));
        out[bi + 2] = (uint8_t)(((b2 >> 4) & 0x03) | ((b3 & 0x3F) << 2));
    }
}

static bool process_layer(ggml_context* ctx_data, int layer, const char* out_dir) {
    char name[64];
    snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
    struct ggml_tensor* t = ggml_get_tensor(ctx_data, name);
    if (!t) { fprintf(stderr, "ERROR: tensor '%s' not found\n", name); return false; }
    if (t->type != 12) { fprintf(stderr, "ERROR: type %d != Q4_K_M(12)\n", t->type); return false; }

    fprintf(stderr, "Layer %d: GGUF shape={%lld,%lld}, elements=%lld\n",
            layer, (long long)t->ne[0], (long long)t->ne[1], (long long)ggml_nelements(t));

    // ne[0]=K=5120, ne[1]=M=13824
    int64_t ne0 = t->ne[0];  // K
    int64_t ne1 = t->ne[1];  // M

    const struct ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) { fprintf(stderr, "ERROR: no to_float\n"); return false; }

    // Dequantize GGUF tensor -> W_gguf of shape [ne0, ne1] = [K, M]
    float* W_gguf = (float*)malloc((size_t)ne0 * ne1 * sizeof(float));
    if (!W_gguf) { fprintf(stderr, "ERROR: malloc W_gguf\n"); return false; }
    tt->to_float(t->data, W_gguf, ne0 * ne1);

    // Transpose: W_sidecar[m, k] = W_gguf[k, m]
    // W_gguf is [K, M] in row-major C order: W_gguf[k * ne1 + m]
    // W_sidecar is [M, K] in row-major: W_sidecar[m * K + k] = W_gguf[k * M + m]
    float* W_sidecar = (float*)malloc((size_t)M * K * sizeof(float));
    float* scales = (float*)malloc(M * sizeof(float));
    int8_t* qvals = (int8_t*)malloc(M * K * sizeof(int8_t));
    uint8_t* packed = (uint8_t*)malloc(SIDECAR_PACKED);

    if (!W_sidecar || !scales || !qvals || !packed) {
        free(W_gguf); free(W_sidecar); free(scales); free(qvals); free(packed);
        return false;
    }

    // Compute scales and quantize
    // For each row m in W_sidecar (which corresponds to column m in W_gguf):
    // scale[m] = max_k(|W_gguf[k, m]|) / PRESCALE
    for (int m = 0; m < M; m++) {
        float col_max = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = fabsf(W_gguf[k * ne1 + m]);  // W_gguf[k, m]
            if (v > col_max) col_max = v;
        }
        scales[m] = (col_max < 1e-8f) ? 1e-8f : col_max / PRESCALE;

        float inv_scale = 1.0f / scales[m];
        int8_t* qrow = qvals + (size_t)m * K;
        float* wrow = W_sidecar + (size_t)m * K;

        for (int k = 0; k < K; k++) {
            float v = W_gguf[k * ne1 + m];  // W_gguf[k, m]
            int q = (int)std::round(v * inv_scale);
            qrow[k] = (int8_t)std::max(-31, std::min(31, q));
            wrow[k] = (float)qrow[k] * scales[m];  // reconstructed
        }
    }

    // Pack
    for (int m = 0; m < M; m++) {
        pack_row_int6(qvals + (size_t)m * K, K, packed + (size_t)m * (((K + 3) / 4) * 3));
    }

    // Write sidecar
    uint8_t* buf = (uint8_t*)malloc(SIDECAR_SIZE);
    memset(buf, 0, SIDECAR_SIZE);
    buf[0] = 'P'; buf[1] = 'R'; buf[2] = 'T'; buf[3] = '6';
    *(uint32_t*)(buf + 4) = 1;
    *(uint32_t*)(buf + 8) = (uint32_t)M;
    *(uint32_t*)(buf + 12) = (uint32_t)K;
    *(uint32_t*)(buf + 16) = 0;
    memcpy(buf + 20, scales, M * 4);
    memcpy(buf + 20 + M * 4, packed, SIDECAR_PACKED);

    char path[512];
    snprintf(path, sizeof(path), "%s/ffn_up_layer%d_prt.int6", out_dir, layer);
    FILE* f = fopen(path, "wb");
    if (!f) { free(buf); return false; }
    fwrite(buf, 1, SIDECAR_SIZE, f);
    fclose(f);
    fprintf(stderr, "Layer %d: wrote %s (%d bytes)\n", layer, path, SIDECAR_SIZE);

    free(W_gguf); free(W_sidecar); free(scales); free(qvals); free(packed); free(buf);
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
    fprintf(stderr, "=== GGUF->INT6 (transposed) ===\nModel: %s\nOutput: %s\n", model_path, out_dir);

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