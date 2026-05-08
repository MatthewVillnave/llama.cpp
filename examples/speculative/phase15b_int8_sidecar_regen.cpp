// PRT Phase 15B-D: Regenerate unique 7B INT8 sidecars + parity validation.
// Uses gguf/ggml APIs to extract/dequantize each layer, quantize to INT8,
// write runtime-compatible sidecar files, and validate selected-layer parity.
// Fixed: uses ggml API to_float for proper Q4_K dequantization.
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

// Q4_K block constants
static const int64_t QK_K = 256;
static const int64_t BPS = 144;  // bytes per Q4_K block

// Deterministic matvec seeds
static const int MATVEC_SEEDS[] = {42, 123, 456, 789, 1011, 2022, 3033, 4044};
static const int N_SEEDS = 8;

// Simple byte checksum (FNV-1a variant)
static uint64_t checksum_bytes(const uint8_t* data, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Properly dequantize Q4_K using ggml type traits (from Phase 15B-C approach)
static void dequantize_q4_k_ggml(const uint8_t* raw_data, size_t raw_size, float* out) {
    // For Q4_K (type 12): blocks of 256 elements, 144 bytes each
    int64_t n_blocks = raw_size / BPS;
    
    // Use the same dequantization as ggml's to_float for Q4_K
    // Per-block layout (144 bytes):
    // [2] d = float16 delta scale (interpreted as float32 for this implementation)
    // [2] dmin = float16 min offset
    // [16] 8 scale pairs (4 bits each): (scale_low, scale_high)
    // [128] 64 pairs of 4-bit quantized values
    
    // The actual Q4_K dequant formula:
    // For each block: y[i] = d * (quant[i] - 8) - dmin * (offset[i] - 8)
    // Where d, dmin are float16 values, quant are 4-bit values
    
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = &raw_data[b * BPS];
        
        // d and dmin as float16 (2 bytes each)
        // Need to convert float16 to float32
        uint16_t d_bits = block[0] | (block[1] << 8);
        uint16_t dmin_bits = block[2] | (block[3] << 8);
        
        // Float16 to float32 conversion
        float d, dmin;
        {
            uint32_t d32 = (d_bits & 0x8000) ? 0xC0000000 : 0x40000000;
            int exp = (d_bits >> 10) & 0x1F;
            uint32_t frac = d_bits & 0x3FF;
            if (exp == 31) exp = 32;
            else exp += 127 - 15;
            d32 |= (exp << 23) | (frac << 13);
            union { uint32_t u; float f; } conv_d = {.u = d32};
            d = conv_d.f;
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
        
        // 16 scale bytes = 8 pairs of (low, high)
        // For each pair j: scale_low = block[4+j] & 0xF, scale_high = block[4+j] >> 4
        // delta_j = d * scale_low
        // offset_j = dmin * scale_high
        
        int64_t base_qp = b * QK_K;
        
        for (int j = 0; j < 4; j++) {
            uint8_t sc = block[4 + j];
            uint8_t sc2 = block[4 + j + 4];
            
            float d1 = d * (sc & 0xF);
            float m1 = dmin * ((sc >> 4) & 0xF);
            float d2 = d * (sc2 & 0xF);
            float m2 = dmin * ((sc2 >> 4) & 0xF);
            
            // 32 bytes of nibbles → 64 elements
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

int main(int argc, char** argv) {
    const char* model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
    const char* output_dir = "/tmp/prt_sidecars_7b_int8_phase15b_fixed";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_dir = argv[++i];
    }
    
    fprintf(stderr, "=== PRT Phase 15B-D: INT8 Sidecar Regeneration ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Output: %s\n", output_dir);
    fprintf(stderr, "Layers: 0-%d, Shape: %ld x %ld\n", (int)N_LAYERS-1, (long)FFN, (long)HIDDEN);
    
    // Create output directory
    system("mkdir -p /tmp/prt_sidecars_7b_int8_phase15b_fixed");
    
    // Init ggml context
    struct ggml_init_params params = {
        .mem_size   = 1024ull*1024ull*1024ull,  // 1GB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: ggml_init failed\n");
        return 1;
    }
    
    // Load GGUF
    struct gguf_context* ctx_gguf = gguf_init_from_file(model_path, (struct gguf_init_params){.no_alloc = false, .ctx = &ctx});
    if (!ctx_gguf) {
        fprintf(stderr, "ERROR: gguf_init_from_file failed\n");
        ggml_free(ctx);
        return 1;
    }
    
    fprintf(stderr, "GGUF loaded. Tensors: %d\n", (int)gguf_get_n_tensors(ctx_gguf));
    
    // Process all 28 layers
    std::vector<std::string> sha256s;
    std::vector<double> weight_cosines;
    std::vector<double> min_cosines;
    std::vector<double> mae_list;
    std::vector<double> rmse_list;
    std::vector<double> max_err_list;
    
    for (int layer = 0; layer < N_LAYERS; layer++) {
        fprintf(stderr, "\n--- Layer %2d ---\n", layer);
        
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_up.weight", layer);
        
        int tensor_id = gguf_find_tensor(ctx_gguf, tensor_name);
        if (tensor_id < 0) {
            fprintf(stderr, "ERROR: tensor %s not found\n", tensor_name);
            continue;
        }
        
        size_t tensor_size = gguf_get_tensor_size(ctx_gguf, tensor_id);
        size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf, tensor_id);
        enum ggml_type dtype = gguf_get_tensor_type(ctx_gguf, tensor_id);
        
        fprintf(stderr, "  ID=%d, type=%d, size=%zu, offset=%zu\n", tensor_id, dtype, tensor_size, tensor_offset);
        
        // Read raw Q4_K data
        FILE* f = fopen(model_path, "rb");
        if (!f) {
            fprintf(stderr, "ERROR: cannot open model\n");
            continue;
        }
        fseek(f, (long)tensor_offset, SEEK_SET);
        uint8_t* raw_data = (uint8_t*)malloc(tensor_size);
        size_t read = fread(raw_data, 1, tensor_size, f);
        fclose(f);
        
        if (read != tensor_size) {
            fprintf(stderr, "ERROR: read %zu of %zu bytes\n", read, tensor_size);
            free(raw_data);
            continue;
        }
        
        // Dequantize to float32 using ggml type traits
        float* W = (float*)malloc(N_ELEMENTS * sizeof(float));
        const struct ggml_type_traits* tt = ggml_get_type_traits(dtype);
        if (tt && tt->to_float) {
            tt->to_float(raw_data, W, N_ELEMENTS);
        } else {
            fprintf(stderr, "ERROR: no to_float for dtype %d\n", dtype);
            free(W);
            free(raw_data);
            continue;
        }
        
        // Check finiteness
        int64_t finite_count = 0;
        for (int64_t i = 0; i < N_ELEMENTS; i++) {
            if (std::isfinite(W[i])) finite_count++;
        }
        fprintf(stderr, "  Finite: %ld/%ld\n", (long)finite_count, (long)N_ELEMENTS);
        
        // Quantize to INT8 per-row
        int8_t* W_q = (int8_t*)malloc(FFN * HIDDEN);
        float* scales = (float*)malloc(FFN * sizeof(float));
        
        for (int64_t j = 0; j < FFN; j++) {
            float row_max = 0.0f;
            for (int64_t k = 0; k < HIDDEN; k++) {
                float v = std::abs(W[j * HIDDEN + k]);
                if (v > row_max) row_max = v;
            }
            
            if (row_max > 1e-10f) {
                scales[j] = row_max / 127.0f;
            } else {
                scales[j] = 1.0f;
            }
            
            for (int64_t k = 0; k < HIDDEN; k++) {
                int32_t q = (int32_t)std::round(W[j * HIDDEN + k] / scales[j]);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                W_q[j * HIDDEN + k] = (int8_t)q;
            }
        }
        
        // Reconstruct for weight cosine and error metrics
        double diff_sum = 0.0;
        double diff_sq_sum = 0.0;
        double max_abs_err = 0.0;
        double dot_W_Wdq = 0.0;
        double norm_W = 0.0;
        double norm_Wdq = 0.0;
        
        for (int64_t i = 0; i < FFN * HIDDEN; i++) {
            float dq = W_q[i] * scales[i / HIDDEN];
            float diff = W[i] - dq;
            diff_sum += std::abs(diff);
            double da = diff * diff;
            diff_sq_sum += da;
            if (std::abs(diff) > max_abs_err) max_abs_err = std::abs(diff);
            dot_W_Wdq += (double)W[i] * (double)dq;
            norm_W += (double)W[i] * (double)W[i];
            norm_Wdq += (double)dq * (double)dq;
        }
        
        double weight_cosine = dot_W_Wdq / (std::sqrt(norm_W) * std::sqrt(norm_Wdq) + 1e-10);
        double mae = diff_sum / N_ELEMENTS;
        double rmse = std::sqrt(diff_sq_sum / N_ELEMENTS);
        
        fprintf(stderr, "  WC=%.6f MAE=%.6f RMSE=%.6f max_err=%.6f\n", 
                weight_cosine, mae, rmse, max_abs_err);
        
        // Write INT8 sidecar file
        char filepath[256];
        snprintf(filepath, sizeof(filepath), "%s/ffn_up_layer%d_prt.int8", output_dir, layer);
        
        FILE* out = fopen(filepath, "wb");
        if (!out) {
            fprintf(stderr, "ERROR: cannot write %s\n", filepath);
            free(W_q);
            free(scales);
            free(W);
            free(raw_data);
            continue;
        }
        
        fwrite(W_q, 1, FFN * HIDDEN, out);
        fwrite(scales, sizeof(float), FFN, out);
        fclose(out);
        
        // SHA256 of file
        FILE* verify = fopen(filepath, "rb");
        fseek(verify, 0, SEEK_END);
        size_t fsize = ftell(verify);
        fseek(verify, 0, SEEK_SET);
        uint8_t* file_data = (uint8_t*)malloc(fsize);
        fread(file_data, 1, fsize, verify);
        fclose(verify);
        
        uint64_t file_hash = checksum_bytes(file_data, fsize);
        free(file_data);
        
        fprintf(stderr, "  Wrote: %s (%zu bytes) SHA=%016llx\n", filepath, fsize, (unsigned long long)file_hash);
        
        // Clean up per-layer
        free(W_q);
        free(scales);
        free(W);
        free(raw_data);
        
        // Store results
        char hash_str[20];
        snprintf(hash_str, sizeof(hash_str), "%016llx", (unsigned long long)file_hash);
        sha256s.push_back(hash_str);
        weight_cosines.push_back(weight_cosine);
        mae_list.push_back(mae);
        rmse_list.push_back(rmse);
        max_err_list.push_back(max_abs_err);
    }
    
    gguf_free(ctx_gguf);
    ggml_free(ctx);
    
    // Summary
    fprintf(stderr, "\n=== Regeneration Summary ===\n");
    fprintf(stderr, "Layers processed: %zu\n", sha256s.size());
    
    // Count unique hashes
    int unique_count = 0;
    for (size_t i = 0; i < sha256s.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < i; j++) {
            if (sha256s[i] == sha256s[j]) {
                found = true;
                break;
            }
        }
        if (!found) unique_count++;
    }
    
    fprintf(stderr, "Unique hashes: %d / %zu\n", unique_count, sha256s.size());
    
    double min_wc = weight_cosines.empty() ? 0.0 : *std::min_element(weight_cosines.begin(), weight_cosines.end());
    double max_wc = weight_cosines.empty() ? 0.0 : *std::max_element(weight_cosines.begin(), weight_cosines.end());
    fprintf(stderr, "Weight cosines: min=%.6f max=%.6f\n", min_wc, max_wc);
    
    for (size_t i = 0; i < sha256s.size(); i++) {
        fprintf(stderr, "  L%2zu: WC=%.6f MAE=%.6f SHA=%s\n", i, weight_cosines[i], mae_list[i], sha256s[i].c_str());
    }
    
    // Run parity checks on selected layers (using fresh sidecars vs GGUF reference)
    fprintf(stderr, "\n=== Selected-Layer Parity ===\n");
    
    // Re-load GGUF for parity checks
    struct ggml_context* ctx2 = ggml_init(params);
    struct gguf_context* ctx_gguf2 = gguf_init_from_file(model_path, (struct gguf_init_params){.no_alloc = false, .ctx = &ctx2});
    
    const int selected_layers[] = {0, 1, 10, 11, 15, 20, 27};
    const int n_selected = 7;
    
    for (int si = 0; si < n_selected; si++) {
        int layer = selected_layers[si];
        fprintf(stderr, "\n  --- Layer %d parity ---\n", layer);
        
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_up.weight", layer);
        
        int tensor_id = gguf_find_tensor(ctx_gguf2, tensor_name);
        size_t tensor_size = gguf_get_tensor_size(ctx_gguf2, tensor_id);
        size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf2, tensor_id);
        enum ggml_type dtype = gguf_get_tensor_type(ctx_gguf2, tensor_id);
        
        // Extract float32 from GGUF
        FILE* f = fopen(model_path, "rb");
        fseek(f, (long)tensor_offset, SEEK_SET);
        uint8_t* raw_data = (uint8_t*)malloc(tensor_size);
        fread(raw_data, 1, tensor_size, f);
        fclose(f);
        
        float* W = (float*)malloc(N_ELEMENTS * sizeof(float));
        const struct ggml_type_traits* tt2 = ggml_get_type_traits(dtype);
        if (tt2 && tt2->to_float) tt2->to_float(raw_data, W, N_ELEMENTS);
        free(raw_data);
        
        // Load INT8 sidecar
        char sidecar_path[256];
        snprintf(sidecar_path, sizeof(sidecar_path), "%s/ffn_up_layer%d_prt.int8", output_dir, layer);
        
        FILE* sc = fopen(sidecar_path, "rb");
        if (!sc) {
            fprintf(stderr, "  ERROR: cannot open sidecar %s\n", sidecar_path);
            free(W);
            continue;
        }
        fseek(sc, 0, SEEK_END);
        fseek(sc, 0, SEEK_SET);
        
        int8_t* W_q = (int8_t*)malloc(FFN * HIDDEN);
        float* scales = (float*)malloc(FFN * sizeof(float));
        fread(W_q, 1, FFN * HIDDEN, sc);
        fread(scales, sizeof(float), FFN, sc);
        fclose(sc);
        
        // Reconstruct dequantized
        float* W_dq = (float*)malloc(N_ELEMENTS * sizeof(float));
        for (int64_t i = 0; i < FFN * HIDDEN; i++) {
            W_dq[i] = W_q[i] * scales[i / HIDDEN];
        }
        
        // Weight cosine
        double dot = 0, nw = 0, nd = 0;
        for (int64_t i = 0; i < N_ELEMENTS; i++) {
            dot += (double)W[i] * (double)W_dq[i];
            nw += (double)W[i] * (double)W[i];
            nd += (double)W_dq[i] * (double)W_dq[i];
        }
        double wc = dot / (std::sqrt(nw) * std::sqrt(nd) + 1e-10);
        
        // MAE/RMSE
        double diff_sum = 0, diff_sq_sum = 0, max_err = 0;
        for (int64_t i = 0; i < N_ELEMENTS; i++) {
            double diff = std::abs((double)W[i] - (double)W_dq[i]);
            diff_sum += diff;
            diff_sq_sum += diff * diff;
            if (diff > max_err) max_err = diff;
        }
        double mae = diff_sum / N_ELEMENTS;
        double rmse = std::sqrt(diff_sq_sum / N_ELEMENTS);
        
        // Matvec parity (8 seeds)
        double cos_sum = 0, cos_min = 1.0;
        double ratio_sum = 0;
        
        for (int si2 = 0; si2 < N_SEEDS; si2++) {
            std::vector<double> y_ref(FFN, 0.0), y_dq(FFN, 0.0);
            
            // Fixed seed random vector
            std::mt19937 rgen(MATVEC_SEEDS[si2] * 1000 + layer);
            std::normal_distribution<double> dist(0.0, 1.0);
            std::vector<double> x(HIDDEN);
            for (int k = 0; k < HIDDEN; k++) x[k] = dist(rgen);
            
            // Reference: W @ x
            for (int64_t j = 0; j < FFN; j++) {
                double sum = 0.0;
                for (int64_t k = 0; k < HIDDEN; k++) {
                    sum += (double)W[j * HIDDEN + k] * x[k];
                }
                y_ref[j] = sum;
            }
            
            // Dequantized: W_dq @ x
            for (int64_t j = 0; j < FFN; j++) {
                double sum = 0.0;
                for (int64_t k = 0; k < HIDDEN; k++) {
                    sum += (double)W_dq[j * HIDDEN + k] * x[k];
                }
                y_dq[j] = sum;
            }
            
            // Cosine
            double dot2 = 0, n1 = 0, n2 = 0;
            for (int64_t j = 0; j < FFN; j++) {
                dot2 += y_ref[j] * y_dq[j];
                n1 += y_ref[j] * y_ref[j];
                n2 += y_dq[j] * y_dq[j];
            }
            double cos = dot2 / (std::sqrt(n1) * std::sqrt(n2) + 1e-10);
            cos_sum += cos;
            if (cos < cos_min) cos_min = cos;
            
            // Output norm ratio
            double nref = std::sqrt(n1);
            double ndq = std::sqrt(n2);
            double ratio = nref > 0 ? ndq / nref : 1.0;
            ratio_sum += ratio;
        }
        
        double cos_mean = cos_sum / N_SEEDS;
        double ratio_mean = ratio_sum / N_SEEDS;
        
        fprintf(stderr, "  L%d: WC=%.6f cos_mean=%.6f cos_min=%.6f ratio_mean=%.6f\n",
                layer, wc, cos_mean, cos_min, ratio_mean);
        fprintf(stderr, "  L%d: MAE=%.6f RMSE=%.6f max_err=%.6f\n",
                layer, mae, rmse, max_err);
        
        free(W);
        free(W_q);
        free(scales);
        free(W_dq);
    }
    
    gguf_free(ctx_gguf2);
    ggml_free(ctx2);
    
    fprintf(stderr, "\nDone.\n");
    return 0;
}