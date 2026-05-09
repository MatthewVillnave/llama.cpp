// PRT Phase 16E-REPAIR v4: GGUF Q4_K -> INT6, matching Phase 16D raw-split approach
// GGUF tensor: ne={5120, 13824}=[K,M], dequantizes to raw[70778880]
// Phase 16D approach: reshape raw as (M=13824, K=5120) row-major,
//                    scale[r] = max(|raw[r*K + k]|) for k=0..K-1
//                    q[r,k] = round(raw[r*K + k] / scale[r])
// Usage: ./phase16e_14b_int6_v4 -m model.gguf -o outdir [--layer N]
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

static const int K = 5120;      // GGUF ne[0]
static const int M = 13824;     // GGUF ne[1]
static const float PRESCALE = 31.0f;
static const int SIDECAR_PACKED = M * (((K + 3) / 4) * 3);  // 53084160
static const int SIDECAR_SIZE = 16 + M*4 + SIDECAR_PACKED; // 53139472 (16-byte header)

// Pack int8 [-31,31] -> 3 bytes per 4 values
static inline void pack_row_int6(const int8_t* qvals, uint8_t* out) {
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

    fprintf(stderr, "Layer %d: GGUF ne={%lld,%lld}, elements=%lld\n",
            layer, (long long)t->ne[0], (long long)t->ne[1], (long long)ggml_nelements(t));

    const struct ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) { fprintf(stderr, "ERROR: no to_float\n"); return false; }

    int64_t ne0 = t->ne[0];  // K=5120
    int64_t ne1 = t->ne[1];  // M=13824

    // Dequantize entire tensor
    int64_t n = ggml_nelements(t);
    float* raw = (float*)malloc((size_t)n * sizeof(float));
    if (!raw) { fprintf(stderr, "ERROR: malloc raw\n"); return false; }
    tt->to_float(t->data, raw, n);
    fprintf(stderr, "Layer %d: dequantized %.1f GB\n", layer, n * sizeof(float) / 1024.0 / 1024.0 / 1024.0);

    // Allocate output buffers
    float* scales = (float*)malloc(M * sizeof(float));
    uint8_t* packed = (uint8_t*)malloc(SIDECAR_PACKED);
    if (!scales || !packed) { free(raw); free(scales); free(packed); return false; }

    // Phase 16D approach: reshape raw as (M, K) row-major
    // For sidecar row r (0..M-1), column k (0..K-1):
    //   raw_index = r * K + k
    //   scale[r] = max_k(|raw[r*K + k]|)
    //   q[r,k] = round(raw[r*K + k] / scale[r])
    for (int r = 0; r < M; r++) {
        // Compute scale for row r: max over K consecutive elements starting at r*K
        float row_max = 0.0f;
        int base_idx = r * K;
        for (int k = 0; k < K; k++) {
            float v = fabsf(raw[base_idx + k]);
            if (v > row_max) row_max = v;
        }
        scales[r] = (row_max < 1e-8f) ? 1e-8f : row_max / PRESCALE;

        // Quantize row r
        float inv_scale = 1.0f / scales[r];
        int8_t qrow[5120];
        for (int k = 0; k < K; k++) {
            int q = (int)std::round(raw[base_idx + k] * inv_scale);
            qrow[k] = (int8_t)std::max(-31, std::min(31, q));
        }

        // Pack row r
        pack_row_int6(qrow, packed + (size_t)r * (((K + 3) / 4) * 3));

        if (r % 2000 == 0 && r > 0) {
            fprintf(stderr, "  layer %d: row %d/%d done\n", layer, r, M);
        }
    }

    free(raw);

    // Write sidecar
    uint8_t* buf = (uint8_t*)malloc(SIDECAR_SIZE);
    memset(buf, 0, SIDECAR_SIZE);
    buf[0] = 'P'; buf[1] = 'R'; buf[2] = 'T'; buf[3] = '6';
    *(uint32_t*)(buf + 4) = 1;
    *(uint32_t*)(buf + 8) = (uint32_t)M;
    *(uint32_t*)(buf + 12) = (uint32_t)K;
    memcpy(buf + 16, scales, M * 4);
    memcpy(buf + 16 + M*4, packed, SIDECAR_PACKED);

    char path[512];
    snprintf(path, sizeof(path), "%s/ffn_up_layer%d_prt.int6", out_dir, layer);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); free(buf); return false; }
    size_t written = fwrite(buf, 1, SIDECAR_SIZE, f);
    if (written != (size_t)SIDECAR_SIZE) {
        fprintf(stderr, "ERROR: partial write %zu / %d\n", written, SIDECAR_SIZE);
        fclose(f); free(buf); return false;
    }
    fclose(f);
    fprintf(stderr, "Layer %d: wrote %s (%d bytes)\n", layer, path, SIDECAR_SIZE);

    free(scales); free(packed); free(buf);
    return true;
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* out_dir = "/tmp/prt_phase16e_forensic";
    int layer = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--layer") == 0 && i+1 < argc) layer = atoi(argv[++i]);
    }

    if (!model_path) { fprintf(stderr, "Usage: %s -m model.gguf -o outdir [--layer N]\n", argv[0]); return 1; }
    fprintf(stderr, "=== GGUF->INT6 (Phase 16D raw-split) ===\nModel: %s\nOutput: %s\n", model_path, out_dir);

    struct ggml_init_params params = { .mem_size = 256ull*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    struct ggml_context* ctx_data = NULL;
    struct gguf_init_params uf_params = { .no_alloc = false, .ctx = &ctx_data };
    struct gguf_context* ctx_gguf = gguf_init_from_file(model_path, uf_params);
    if (!ctx_gguf) { fprintf(stderr, "gguf_init_from_file failed\n"); ggml_free(ctx); return 1; }

    int success = 0;
    if (layer >= 0) { if (process_layer(ctx_data, layer, out_dir)) success++; }
    else { fprintf(stderr, "ERROR: must specify --layer N\n"); }

    fprintf(stderr, "\nSuccess: %d/1 layers\n", success);
    gguf_free(ctx_gguf); ggml_free(ctx);
    return (success == 1) ? 0 : 1;
}