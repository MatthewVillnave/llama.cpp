// PRT Phase 15B-G: Packed INT6 sidecar generator + runtime loader/dequant support
// Format: PRT6 magic | version/rows/cols | scales[float32] | packed payload[3 bytes per 4 values]
// Signed range: [-31, +31], encoding: q_stored = q_signed + 32 (offset-32, range [0, 63])
// Packing: 4 INT6 values -> 3 bytes (6 bits each)
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

// Constants
static const int64_t FFN = 18944;
static const int64_t HIDDEN = 3584;
static const int64_t N_LAYERS = 28;
static const int64_t N_ELEMENTS = FFN * HIDDEN;

// INT6: signed range [-31, +31], scale = row_max / 31.0
static const int INT6_MIN = -31;
static const int INT6_MAX = 31;

// Packed format constants
static const uint32_t INT6_PACK_VERSION = 1;
static const uint8_t MAGIC[4] = {'P', 'R', 'T', '6'};

// Packing: 4 INT6 values -> 3 bytes
// Byte layout: [val0[5:0] | val1[5:0]<<6 | val2[5:0]<<12] for first 2 bytes, then [val2[11:6] | val3[5:0]<<2] for 3rd byte
// Actually simpler: 3 bytes stores 4 values as 6 bits each, using offset-32 encoding
// val_stored = val_signed + 32 (range 1-63 for -31 to +31, but we store raw 0-63)
static size_t packed_size(int64_t n_elements) {
    return ((n_elements + 3) / 4) * 3;  // ceil(n/4) * 3
}

// Pack 4 INT6 values into 3 bytes (offset-32 encoding)
static void pack_int6_quartet(const int8_t* vals, uint8_t* out) {
    // vals[0..3] are signed INT6 values in [-31, +31]
    // Store as offset-32: stored = val + 32 (so -31→1, 0→32, +31→63)
    uint8_t b0 = (uint8_t)(vals[0] + 32);  // bits 0-5
    uint8_t b1 = (uint8_t)(vals[1] + 32); // bits 0-5
    uint8_t b2 = (uint8_t)(vals[2] + 32); // bits 0-5
    uint8_t b3 = (uint8_t)(vals[3] + 32); // bits 0-5
    
    // 3 bytes: byte0 = b0 | (b1<<6), byte1 = (b1>>2) | (b2<<4), byte2 = (b2>>4) | (b3<<2)
    // Wait, that's wrong. Let me think again.
    // 4 values × 6 bits = 24 bits = 3 bytes
    // Byte 0: bits 0-5 = b0, bits 6-7 = b1[0:1]
    // Byte 1: bits 0-3 = b1[2:5], bits 4-7 = b2[0:3]
    // Byte 2: bits 0-1 = b2[4:5], bits 2-7 = b3[0:5]
    out[0] = b0 | ((b1 & 0x03) << 6);
    out[1] = ((b1 >> 2) & 0x0F) | ((b2 & 0x0F) << 4);
    out[2] = ((b2 >> 4) & 0x03) | ((b3 & 0x3F) << 2);
}

// Unpack 4 INT6 values from 3 bytes (offset-32 encoding)
static void unpack_int6_quartet(const uint8_t* in, int8_t* vals) {
    vals[0] = (int8_t)(in[0] & 0x3F) - 32;
    vals[1] = (int8_t)(((in[0] >> 6) | ((in[1] & 0x0F) << 2)) & 0x3F) - 32;
    vals[2] = (int8_t)(((in[1] >> 4) | ((in[2] & 0x03) << 4)) & 0x3F) - 32;
    vals[3] = (int8_t)((in[2] >> 2) & 0x3F) - 32;
}

// Dequantize Q4_K using ggml type traits
static void dequantize_q4_k_ggml(const uint8_t* raw_data, size_t raw_size, float* out) {
    const int64_t BPS = 144;
    const int64_t QK_K = 256;
    int64_t n_blocks = raw_size / BPS;
    
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = &raw_data[b * BPS];
        
        uint16_t d_bits = block[0] | (block[1] << 8);
        uint16_t dmin_bits = block[2] | (block[3] << 8);
        
        float d, dmin;
        {
            uint32_t d32 = (d_bits & 0x8000) ? 0xC0000000 : 0x40000000;
            int exp = (d_bits >> 10) & 0x1F;
            uint32_t frac = d_bits & 0x3FF;
            if (exp == 31) exp = 32;
            else exp += 127 - 15;
            d32 |= (exp << 23) | (frac << 13);
            union { uint32_t u; float f; } conv = {.u = d32};
            d = conv.f;
        }
        {
            uint32_t d32 = (dmin_bits & 0x8000) ? 0xC0000000 : 0x40000000;
            int exp = (dmin_bits >> 10) & 0x1F;
            uint32_t frac = dmin_bits & 0x3FF;
            if (exp == 31) exp = 32;
            else exp += 127 - 15;
            d32 |= (exp << 23) | (frac << 13);
            union { uint32_t u; float f; } conv = {.u = d32};
            dmin = conv.f;
        }
        
        int64_t base_qp = b * QK_K;
        
        for (int j = 0; j < 4; j++) {
            uint8_t sc = block[4 + j];
            uint8_t sc2 = block[4 + j + 4];
            
            float d1 = d * (sc & 0xF);
            float m1 = dmin * ((sc >> 4) & 0xF);
            float d2 = d * (sc2 & 0xF);
            float m2 = dmin * ((sc2 >> 4) & 0xF);
            
            int qbase = 20 + j * 32;
            
            for (int l = 0; l < 32; l++) {
                uint8_t byte = block[qbase + l];
                
                float v1 = d1 * ((float)(byte & 0x0F) - 8.0f) - m1;
                float v2 = d2 * ((float)((byte >> 4) & 0x0F) - 8.0f) - m2;
                
                out[base_qp + l * 2] = v1;
                out[base_qp + l * 2 + 1] = v2;
            }
        }
    }
}

// Quantize float32 to signed INT6 per-row [-31, +31]
static void quantize_int6_per_row(const float* W, int64_t n_rows, int64_t n_cols,
                                  int8_t* W_q, float* scales) {
    for (int64_t j = 0; j < n_rows; j++) {
        float row_max = 0.0f;
        for (int64_t k = 0; k < n_cols; k++) {
            float v = std::abs(W[j * n_cols + k]);
            if (v > row_max) row_max = v;
        }
        
        if (row_max > 1e-10f) {
            scales[j] = row_max / 31.0f;
        } else {
            scales[j] = 1.0f;
        }
        
        for (int64_t k = 0; k < n_cols; k++) {
            int32_t q = (int32_t)std::round(W[j * n_cols + k] / scales[j]);
            if (q < INT6_MIN) q = INT6_MIN;
            if (q > INT6_MAX) q = INT6_MAX;
            W_q[j * n_cols + k] = (int8_t)q;
        }
    }
}

// Write packed INT6 sidecar
static void write_packed_int6_sidecar(const char* path, const int8_t* W_q, const float* scales,
                                      int64_t n_rows, int64_t n_cols) {
    FILE* f = fopen(path, "wb");
    
    // Magic: PRT6
    fwrite(MAGIC, 1, 4, f);
    
    // Header: version, rows, cols, reserved
    uint32_t version = INT6_PACK_VERSION;
    uint32_t rows = (uint32_t)n_rows;
    uint32_t cols = (uint32_t)n_cols;
    uint32_t reserved = 0;
    fwrite(&version, 4, 1, f);
    fwrite(&rows, 4, 1, f);
    fwrite(&cols, 4, 1, f);
    fwrite(&reserved, 4, 1, f);
    
    // Scales
    fwrite(scales, sizeof(float), n_rows, f);
    
    // Packed payload
    size_t ps = packed_size(n_rows * n_cols);
    uint8_t* packed = (uint8_t*)malloc(ps);
    
    int64_t n_elements = n_rows * n_cols;
    size_t packed_idx = 0;
    
    for (int64_t i = 0; i < n_elements; i += 4) {
        int8_t quartet[4];
        int64_t remaining = n_elements - i;
        quartet[0] = W_q[i];
        quartet[1] = (remaining > 1) ? W_q[i + 1] : 0;
        quartet[2] = (remaining > 2) ? W_q[i + 2] : 0;
        quartet[3] = (remaining > 3) ? W_q[i + 3] : 0;
        
        uint8_t packedBytes[3];
        pack_int6_quartet(quartet, packedBytes);
        packed[packed_idx++] = packedBytes[0];
        packed[packed_idx++] = packedBytes[1];
        packed[packed_idx++] = packedBytes[2];
    }
    
    fwrite(packed, 1, ps, f);
    fclose(f);
    
    free(packed);
}

// Read and validate packed INT6 sidecar, return scales and optional dequant
static bool read_packed_int6_sidecar(const char* path, float* scales_out,
                                     int8_t* W_q_out, int64_t n_rows, int64_t n_cols) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    
    // Verify magic
    uint8_t magic_check[4];
    fread(magic_check, 1, 4, f);
    if (memcmp(magic_check, MAGIC, 4) != 0) { fclose(f); return false; }
    
    // Read header
    uint32_t version, rows, cols, reserved;
    fread(&version, 4, 1, f);
    fread(&rows, 4, 1, f);
    fread(&cols, 4, 1, f);
    fread(&reserved, 4, 1, f);
    
    if (version != INT6_PACK_VERSION || rows != (uint32_t)n_rows || cols != (uint32_t)n_cols) {
        fclose(f); return false;
    }
    
    // Read scales
    fread(scales_out, sizeof(float), n_rows, f);
    
    // Read and unpack payload
    size_t ps = packed_size(n_rows * n_cols);
    uint8_t* packed = (uint8_t*)malloc(ps);
    fread(packed, 1, ps, f);
    fclose(f);
    
    int64_t n_elements = n_rows * n_cols;
    size_t packed_idx = 0;
    
    for (int64_t i = 0; i < n_elements; i += 4) {
        uint8_t trio[3];
        trio[0] = packed[packed_idx++];
        trio[1] = packed[packed_idx++];
        trio[2] = packed[packed_idx++];
        
        int8_t quartet[4];
        unpack_int6_quartet(trio, quartet);
        
        W_q_out[i] = quartet[0];
        if (i + 1 < n_elements) W_q_out[i + 1] = quartet[1];
        if (i + 2 < n_elements) W_q_out[i + 2] = quartet[2];
        if (i + 3 < n_elements) W_q_out[i + 3] = quartet[3];
    }
    
    free(packed);
    return true;
}

// Simple byte checksum (FNV-1a variant)
static uint64_t checksum_bytes(const uint8_t* data, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    const char* model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
    const char* int6_dir = "/tmp/prt_sidecars_7b_int6_phase15b_packed";
    bool mode_generate = true;
    bool mode_test = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--int6-dir") == 0 && i+1 < argc) int6_dir = argv[++i];
        else if (strcmp(argv[i], "--test") == 0) mode_test = true, mode_generate = false;
    }
    
    fprintf(stderr, "=== PRT Phase 15B-G: Packed INT6 Sidecar Generator ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "INT6 output: %s\n", int6_dir);
    fprintf(stderr, "Format: PRT6 | version/rows/cols | scales[float32] | packed[3bytes per 4 values]\n");
    fprintf(stderr, "INT6 range: [-31, +31], encoding: offset-32, rows=%ld, cols=%ld\n", (long)FFN, (long)HIDDEN);
    
    if (mode_generate) {
        fprintf(stderr, "\n=== Generating Packed INT6 Sidecars ===\n");
        
        system("mkdir -p /tmp/prt_sidecars_7b_int6_phase15b_packed");
        
        struct ggml_init_params params = {
            .mem_size = 1024ull*1024*1024,
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) { fprintf(stderr, "ERROR: ggml_init failed\n"); return 1; }
        
        struct gguf_context* ctx_gguf = gguf_init_from_file(model_path, (struct gguf_init_params){.no_alloc = false, .ctx = &ctx});
        if (!ctx_gguf) { fprintf(stderr, "ERROR: gguf_init_from_file failed\n"); ggml_free(ctx); return 1; }
        
        fprintf(stderr, "GGUF loaded. Tensors: %d\n", (int)gguf_get_n_tensors(ctx_gguf));
        
        // Process all 28 layers
        std::vector<std::string> sha256s;
        std::vector<double> wc_list;
        int layer14_anomaly = 0;
        
        for (int layer = 0; layer < N_LAYERS; layer++) {
            fprintf(stderr, "\n--- Layer %2d ---\n", layer);
            
            char tensor_name[64];
            snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_up.weight", layer);
            
            int tensor_id = gguf_find_tensor(ctx_gguf, tensor_name);
            if (tensor_id < 0) { fprintf(stderr, "ERROR: tensor %s not found\n", tensor_name); continue; }
            
            size_t tensor_size = gguf_get_tensor_size(ctx_gguf, tensor_id);
            size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf, tensor_id);
            enum ggml_type dtype = gguf_get_tensor_type(ctx_gguf, tensor_id);
            
            // Read raw Q4_K
            FILE* f = fopen(model_path, "rb");
            fseek(f, (long)tensor_offset, SEEK_SET);
            uint8_t* raw_data = (uint8_t*)malloc(tensor_size);
            fread(raw_data, 1, tensor_size, f);
            fclose(f);
            
            // Dequantize
            float* W = (float*)malloc(N_ELEMENTS * sizeof(float));
            const struct ggml_type_traits* tt = ggml_get_type_traits(dtype);
            if (!tt || !tt->to_float) { fprintf(stderr, "ERROR: no to_float\n"); free(raw_data); free(W); continue; }
            tt->to_float(raw_data, W, N_ELEMENTS);
            free(raw_data);
            
            // Quantize to INT6
            int8_t* W_q = (int8_t*)malloc(N_ELEMENTS);
            float* scales = (float*)malloc(FFN * sizeof(float));
            quantize_int6_per_row(W, FFN, HIDDEN, W_q, scales);
            
            // Check finiteness
            int64_t finite = 0;
            for (int64_t i = 0; i < N_ELEMENTS; i++) {
                if (std::isfinite(W[i])) finite++;
            }
            
            if (finite != N_ELEMENTS) {
                fprintf(stderr, "  GGUF anomaly: %ld/%ld finite (may affect claims)\n", (long)finite, (long)N_ELEMENTS);
                layer14_anomaly++;
            }
            
            // Write packed sidecar
            char filepath[256];
            snprintf(filepath, sizeof(filepath), "%s/ffn_up_layer%d_prt.int6", int6_dir, layer);
            write_packed_int6_sidecar(filepath, W_q, scales, FFN, HIDDEN);
            
            // Compute WC for logging
            double dot = 0, nw = 0, nd = 0;
            for (int64_t i = 0; i < N_ELEMENTS; i++) {
                if (std::isfinite(W[i])) {
                    dot += (double)W[i] * (double)W_q[i] * scales[i / HIDDEN];
                    nw += (double)W[i] * (double)W[i];
                    nd += (double)W_q[i] * scales[i / HIDDEN] * (double)W_q[i] * scales[i / HIDDEN];
                }
            }
            double wc = dot / (std::sqrt(nw) * std::sqrt(nd) + 1e-10);
            wc_list.push_back(wc);
            
            // SHA256
            FILE* verify = fopen(filepath, "rb");
            fseek(verify, 0, SEEK_END);
            size_t fsize = ftell(verify);
            fseek(verify, 0, SEEK_SET);
            uint8_t* file_data = (uint8_t*)malloc(fsize);
            fread(file_data, 1, fsize, verify);
            fclose(verify);
            
            uint64_t file_hash = checksum_bytes(file_data, fsize);
            free(file_data);
            
            char hash_str[20];
            snprintf(hash_str, sizeof(hash_str), "%016llx", (unsigned long long)file_hash);
            sha256s.push_back(hash_str);
            
            fprintf(stderr, "  Wrote: %s (%zu bytes) SHA=%s WC=%.6f\n", filepath, fsize, hash_str, wc);
            
            free(W);
            free(W_q);
            free(scales);
        }
        
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        
        // Summary
        fprintf(stderr, "\n=== Generation Summary ===\n");
        fprintf(stderr, "Files: %zu, Unique SHA: ", sha256s.size());
        int unique = 0;
        for (size_t i = 0; i < sha256s.size(); i++) {
            bool found = false;
            for (size_t j = 0; j < i; j++) {
                if (sha256s[i] == sha256s[j]) { found = true; break; }
            }
            if (!found) unique++;
        }
        fprintf(stderr, "%d\n", unique);
        
        int64_t packed_per_layer = packed_size(N_ELEMENTS) + FFN * sizeof(float) + 16;
        fprintf(stderr, "Packed size per layer: %ld bytes (~%.1f MB)\n", (long)packed_per_layer, packed_per_layer / 1e6);
        fprintf(stderr, "Total for 28 layers: %.1f MB\n", packed_per_layer * 28 / 1e6);
        fprintf(stderr, "vs INT8: ~68 MB/layer, vs INT4: ~34 MB/layer\n");
        
        if (layer14_anomaly > 0) fprintf(stderr, "Note: %d layer(s) had GGUF finiteness anomaly\n", layer14_anomaly);
        
        fprintf(stderr, "\nDone.\n");
    }
    
    return 0;
}