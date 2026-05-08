// PRT Phase 15B-E: INT4 offline parity with fixed GGUF extraction
// Uses working Phase 15B-C/D GGUF API path.
// Extracts GGUF weights, quantizes to INT4 per-row, computes parity vs
// true GGUF reference and vs fresh INT8 sidecar references.
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

// Deterministic matvec seeds
static const int MATVEC_SEEDS[] = {42, 123, 456, 789, 1011, 2022, 3033, 4044};
static const int N_SEEDS = 8;

// Selected layers for parity testing
static const int SELECTED_LAYERS[] = {0, 1, 10, 11, 15, 20, 27};
static const int N_SELECTED = 7;

// Dequantize Q4_K using ggml type traits (same as Phase 15B-C/D)
static void dequantize_q4_k_ggml(const uint8_t* raw_data, size_t raw_size, float* out) {
    const int64_t BPS = 144;
    const int64_t QK_K = 256;
    int64_t n_blocks = raw_size / BPS;
    
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = &raw_data[b * BPS];
        
        // d and dmin as float16
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

// Quantize float32 to signed INT4 per-row
// Range: [-7, +7], scale = row_max / 7.0
static void quantize_int4_per_row(const float* W, int64_t n_rows, int64_t n_cols,
                                  int8_t* W_q, float* scales) {
    for (int64_t j = 0; j < n_rows; j++) {
        float row_max = 0.0f;
        for (int64_t k = 0; k < n_cols; k++) {
            float v = std::abs(W[j * n_cols + k]);
            if (v > row_max) row_max = v;
        }
        
        if (row_max > 1e-10f) {
            scales[j] = row_max / 7.0f;
        } else {
            scales[j] = 1.0f;
        }
        
        for (int64_t k = 0; k < n_cols; k++) {
            int32_t q = (int32_t)std::round(W[j * n_cols + k] / scales[j]);
            if (q < -7) q = -7;
            if (q > 7) q = 7;
            W_q[j * n_cols + k] = (int8_t)q;
        }
    }
}

// Dequantize INT4 back to float32
static void dequantize_int4_per_row(const int8_t* W_q, const float* scales,
                                    int64_t n_rows, int64_t n_cols, float* out) {
    for (int64_t i = 0; i < n_rows * n_cols; i++) {
        out[i] = W_q[i] * scales[i / n_cols];
    }
}

// Compute parity metrics between W_ref and W_test
static void compute_parity(const float* W_ref, const float* W_test, int64_t n_elements,
                          double* wc, double* mae, double* rmse, double* max_err,
                          int64_t* finite_count, int64_t* zero_count) {
    double dot = 0, nw = 0, nd = 0;
    double diff_sum = 0, diff_sq_sum = 0;
    double max_abs = 0;
    int64_t finite = 0, zero = 0;
    
    for (int64_t i = 0; i < n_elements; i++) {
        if (std::isfinite(W_ref[i]) && std::isfinite(W_test[i])) {
            finite++;
            dot += (double)W_ref[i] * (double)W_test[i];
            nw += (double)W_ref[i] * (double)W_ref[i];
            nd += (double)W_test[i] * (double)W_test[i];
            
            double diff = std::abs((double)W_ref[i] - (double)W_test[i]);
            diff_sum += diff;
            diff_sq_sum += diff * diff;
            if (diff > max_abs) max_abs = diff;
            
            if (std::abs(W_ref[i]) < 1e-10f && std::abs(W_test[i]) < 1e-10f) zero++;
        }
    }
    
    *wc = dot / (std::sqrt(nw) * std::sqrt(nd) + 1e-10);
    *mae = diff_sum / n_elements;
    *rmse = std::sqrt(diff_sq_sum / n_elements);
    *max_err = max_abs;
    *finite_count = finite;
    *zero_count = zero;
}

// Compute matvec cosine between W_ref and W_test with deterministic seeds
static void compute_matvec_parity(const float* W_ref, const float* W_test,
                                  int64_t n_rows, int64_t n_cols,
                                  double* cos_mean, double* cos_min,
                                  double* ratio_mean, double* ratio_min, double* ratio_max) {
    double cos_sum = 0;
    double ratio_sum = 0;
    double ratio_min_val = 1e10;
    double ratio_max_val = 0;
    double cos_min_val = 1.0;
    
    for (int si = 0; si < N_SEEDS; si++) {
        std::mt19937 rgen(MATVEC_SEEDS[si] * 1000);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> x(n_cols);
        for (int k = 0; k < n_cols; k++) x[k] = dist(rgen);
        
        // Reference: W_ref @ x
        std::vector<double> y_ref(n_rows, 0.0);
        for (int64_t j = 0; j < n_rows; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < n_cols; k++) {
                sum += (double)W_ref[j * n_cols + k] * x[k];
            }
            y_ref[j] = sum;
        }
        
        // Test: W_test @ x
        std::vector<double> y_test(n_rows, 0.0);
        for (int64_t j = 0; j < n_rows; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < n_cols; k++) {
                sum += (double)W_test[j * n_cols + k] * x[k];
            }
            y_test[j] = sum;
        }
        
        // Cosine
        double dot = 0, n1 = 0, n2 = 0;
        for (int64_t j = 0; j < n_rows; j++) {
            dot += y_ref[j] * y_test[j];
            n1 += y_ref[j] * y_ref[j];
            n2 += y_test[j] * y_test[j];
        }
        double cos = dot / (std::sqrt(n1) * std::sqrt(n2) + 1e-10);
        cos_sum += cos;
        if (cos < cos_min_val) cos_min_val = cos;
        
        // Norm ratio
        double nref = std::sqrt(n1);
        double ndq = std::sqrt(n2);
        double ratio = nref > 0 ? ndq / nref : 1.0;
        ratio_sum += ratio;
        if (ratio < ratio_min_val) ratio_min_val = ratio;
        if (ratio > ratio_max_val) ratio_max_val = ratio;
    }
    
    *cos_mean = cos_sum / N_SEEDS;
    *cos_min = cos_min_val;
    *ratio_mean = ratio_sum / N_SEEDS;
    *ratio_min = ratio_min_val;
    *ratio_max = ratio_max_val;
}

// Compute INT4-specific quantization stats
static void compute_int4_stats(const int8_t* W_q, int64_t n_elements,
                              int64_t* zero_count, int64_t* sat_abs7_count,
                              float* scale_min, float* scale_max, float* scale_mean) {
    int64_t zero = 0, sat7 = 0;
    float sc_min = 1e10f, sc_max = 0.0f, sc_sum = 0.0f;
    int64_t n_rows = n_elements / HIDDEN;
    
    for (int64_t i = 0; i < n_elements; i++) {
        if (W_q[i] == 0) zero++;
        if (W_q[i] == 7 || W_q[i] == -7) sat7++;
    }
    
    *zero_count = zero;
    *sat_abs7_count = sat7;
    
    // Scales stats computed externally (pass here)
    (void)sc_min; (void)sc_max; (void)sc_sum;
    (void)n_rows;
}

int main(int argc, char** argv) {
    const char* model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
    const char* int8_ref_dir = "/tmp/prt_sidecars_7b_int8_phase15b_fixed_v2";
    const char* int4_out_dir = "/tmp/prt_sidecars_7b_int4_phase15b_probe";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--int8-dir") == 0 && i+1 < argc) int8_ref_dir = argv[++i];
        else if (strcmp(argv[i], "--int4-out") == 0 && i+1 < argc) int4_out_dir = argv[++i];
    }
    
    fprintf(stderr, "=== PRT Phase 15B-E: INT4 Offline Parity ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "INT8 ref: %s\n", int8_ref_dir);
    fprintf(stderr, "INT4 out: %s\n", int4_out_dir);
    fprintf(stderr, "Layers: 0-%d, Shape: %ld x %ld\n", (int)N_LAYERS-1, (long)FFN, (long)HIDDEN);
    fprintf(stderr, "Selected layers: ");
    for (int i = 0; i < N_SELECTED; i++) fprintf(stderr, "%d ", SELECTED_LAYERS[i]);
    fprintf(stderr, "\n");
    
    // Create output dir
    system("mkdir -p /tmp/prt_sidecars_7b_int4_phase15b_probe");
    
    // Init ggml
    struct ggml_init_params params = {
        .mem_size = 1024ull*1024*1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ERROR: ggml_init failed\n"); return 1; }
    
    // Load GGUF
    struct gguf_context* ctx_gguf = gguf_init_from_file(model_path, (struct gguf_init_params){.no_alloc = false, .ctx = &ctx});
    if (!ctx_gguf) { fprintf(stderr, "ERROR: gguf_init_from_file failed\n"); ggml_free(ctx); return 1; }
    
    fprintf(stderr, "GGUF loaded. Tensors: %d\n", (int)gguf_get_n_tensors(ctx_gguf));
    
    // Process selected layers
    fprintf(stderr, "\n=== Selected-Layer INT4 Parity ===\n");
    
    // JSON output buffer
    FILE* json_fp = fopen("/tmp/prt_phase15b_e_int4_offline/int4_selected_layer_parity.json", "w");
    fprintf(json_fp, "{\n  \"phase\": \"15B-E\",\n  \"selected_layers\": [");
    for (int i = 0; i < N_SELECTED; i++) {
        fprintf(json_fp, "%d%s", SELECTED_LAYERS[i], i < N_SELECTED-1 ? ", " : "");
    }
    fprintf(json_fp, "],\n  \"layers\": [\n");
    
    int first_layer = 1;
    double min_wc = 1.0, min_mc = 1.0;
    int weakest_layer = -1;
    int pass_count = 0, maybe_count = 0, fail_count = 0;
    
    for (int si = 0; si < N_SELECTED; si++) {
        int layer = SELECTED_LAYERS[si];
        fprintf(stderr, "\n--- Layer %d ---\n", layer);
        
        // Build tensor name
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_up.weight", layer);
        
        int tensor_id = gguf_find_tensor(ctx_gguf, tensor_name);
        if (tensor_id < 0) { fprintf(stderr, "ERROR: tensor %s not found\n", tensor_name); continue; }
        
        size_t tensor_size = gguf_get_tensor_size(ctx_gguf, tensor_id);
        size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf, tensor_id);
        enum ggml_type dtype = gguf_get_tensor_type(ctx_gguf, tensor_id);
        
        fprintf(stderr, "  GGUF: ID=%d, type=%d, size=%zu, offset=%zu\n", tensor_id, dtype, tensor_size, tensor_offset);
        
        // Read raw Q4_K data from GGUF
        FILE* f = fopen(model_path, "rb");
        fseek(f, (long)tensor_offset, SEEK_SET);
        uint8_t* raw_data = (uint8_t*)malloc(tensor_size);
        fread(raw_data, 1, tensor_size, f);
        fclose(f);
        
        // Dequantize to float32 (true GGUF reference)
        float* W_gguf = (float*)malloc(N_ELEMENTS * sizeof(float));
        const struct ggml_type_traits* tt = ggml_get_type_traits(dtype);
        if (!tt || !tt->to_float) {
            fprintf(stderr, "ERROR: no to_float for dtype %d\n", dtype);
            free(raw_data); free(W_gguf);
            continue;
        }
        tt->to_float(raw_data, W_gguf, N_ELEMENTS);
        free(raw_data);
        
        // Check finiteness
        int64_t finite_gguf = 0;
        for (int64_t i = 0; i < N_ELEMENTS; i++) {
            if (std::isfinite(W_gguf[i])) finite_gguf++;
        }
        fprintf(stderr, "  GGUF finite: %ld/%ld\n", (long)finite_gguf, (long)N_ELEMENTS);
        
        // Quantize to INT4 per-row
        int8_t* W_q4 = (int8_t*)malloc(N_ELEMENTS);
        float* scales4 = (float*)malloc(FFN * sizeof(float));
        quantize_int4_per_row(W_gguf, FFN, HIDDEN, W_q4, scales4);
        
        // Dequantize back to float32
        float* W_int4 = (float*)malloc(N_ELEMENTS * sizeof(float));
        dequantize_int4_per_row(W_q4, scales4, FFN, HIDDEN, W_int4);
        
        // INT4 stats
        int64_t zero4 = 0, sat7_4 = 0;
        for (int64_t i = 0; i < N_ELEMENTS; i++) {
            if (W_q4[i] == 0) zero4++;
            if (W_q4[i] == 7 || W_q4[i] == -7) sat7_4++;
        }
        float sc4_min = 1e10f, sc4_max = 0.0f, sc4_sum = 0.0f;
        for (int64_t j = 0; j < FFN; j++) {
            if (scales4[j] < sc4_min) sc4_min = scales4[j];
            if (scales4[j] > sc4_max) sc4_max = scales4[j];
            sc4_sum += scales4[j];
        }
        float sc4_mean = sc4_sum / FFN;
        
        // INT4 vs GGUF parity
        double wc4, mae4, rmse4, max_err4;
        compute_parity(W_gguf, W_int4, N_ELEMENTS, &wc4, &mae4, &rmse4, &max_err4, &finite_gguf, &zero4);
        
        // INT4 matvec cosine
        double cos_mean4, cos_min4, ratio_mean4, ratio_min4, ratio_max4;
        compute_matvec_parity(W_gguf, W_int4, FFN, HIDDEN, &cos_mean4, &cos_min4, &ratio_mean4, &ratio_min4, &ratio_max4);
        
        fprintf(stderr, "  INT4 vs GGUF: WC=%.6f cos_mean=%.6f cos_min=%.6f MAE=%.6f RMSE=%.6f max_err=%.6f\n",
                wc4, cos_mean4, cos_min4, mae4, rmse4, max_err4);
        fprintf(stderr, "  INT4 stats: zero=%.4f sat7=%.4f scale=[%.6f, %.6f, %.6f]\n",
                (double)zero4/N_ELEMENTS, (double)sat7_4/N_ELEMENTS, sc4_min, sc4_max, sc4_mean);
        
        // Load INT8 reference sidecar
        char int8_path[256];
        snprintf(int8_path, sizeof(int8_path), "%s/ffn_up_layer%d_prt.int8", int8_ref_dir, layer);
        
        float* W_int8 = NULL;
        double wc8 = 0.0, mae8 = 0.0, rmse8 = 0.0;
        double cos_mean8 = 0.0, cos_min8 = 0.0;
        double ratio_mean8 = 0.0, ratio_min8 = 0.0, ratio_max8 = 0.0;
        bool int8_loaded = false;
        
        FILE* sc = fopen(int8_path, "rb");
        if (sc) {
            int8_t* W_q8 = (int8_t*)malloc(FFN * HIDDEN);
            float* scales8 = (float*)malloc(FFN * sizeof(float));
            fread(W_q8, 1, FFN * HIDDEN, sc);
            fread(scales8, sizeof(float), FFN, sc);
            fclose(sc);
            
            W_int8 = (float*)malloc(N_ELEMENTS * sizeof(float));
            dequantize_int4_per_row(W_q8, scales8, FFN, HIDDEN, W_int8);  // reuse INT4 dequant
            
            // INT8 vs GGUF parity
            int64_t finite_int8 = N_ELEMENTS;
            compute_parity(W_gguf, W_int8, N_ELEMENTS, &wc8, &mae8, &rmse8, &max_err4, &finite_int8, &zero4);
            
            // INT8 matvec
            compute_matvec_parity(W_gguf, W_int8, FFN, HIDDEN, &cos_mean8, &cos_min8, &ratio_mean8, &ratio_min8, &ratio_max8);
            
            fprintf(stderr, "  INT8 vs GGUF: WC=%.6f cos_mean=%.6f cos_min=%.6f MAE=%.6f RMSE=%.6f\n",
                    wc8, cos_mean8, cos_min8, mae8, rmse8);
            
            free(W_q8);
            free(scales8);
            int8_loaded = true;
        } else {
            fprintf(stderr, "  WARNING: INT8 ref not found at %s\n", int8_path);
        }
        
        // Verdict
        const char* verdict;
        if (wc4 >= 0.999 && cos_min4 >= 0.999) {
            verdict = "PASS";
            pass_count++;
        } else if (wc4 >= 0.980 && cos_min4 >= 0.980) {
            verdict = "MAYBE";
            maybe_count++;
        } else {
            verdict = "NO_GO";
            fail_count++;
        }
        
        fprintf(stderr, "  Verdict: %s\n", verdict);
        
        // Track weakest
        if (wc4 < min_wc) { min_wc = wc4; weakest_layer = layer; }
        if (cos_min4 < min_mc) min_mc = cos_min4;
        
        // Write temporary INT4 probe file
        // Format: [FFN*HIDDEN int4 bytes (packed 2 per byte)][FFN*4 float32 scales]
        char int4_path[256];
        snprintf(int4_path, sizeof(int4_path), "%s/ffn_up_layer%d_prt.int4", int4_out_dir, layer);
        
        FILE* i4 = fopen(int4_path, "wb");
        if (i4) {
            // Pack 2 int4 values per byte: low nibble first, high nibble second
            for (int64_t j = 0; j < FFN; j++) {
                for (int64_t k = 0; k < HIDDEN; k += 2) {
                    uint8_t byte = (W_q4[j * HIDDEN + k] + 8) & 0x0F;
                    byte |= ((W_q4[j * HIDDEN + k + 1] + 8) & 0x0F) << 4;
                    fwrite(&byte, 1, 1, i4);
                }
            }
            fwrite(scales4, sizeof(float), FFN, i4);
            fclose(i4);
            fprintf(stderr, "  Wrote INT4 probe: %s\n", int4_path);
        }
        
        // JSON output
        if (!first_layer) fprintf(json_fp, ",\n");
        first_layer = 0;
        
        fprintf(json_fp, "    {\n");
        fprintf(json_fp, "      \"layer\": %d,\n", layer);
        fprintf(json_fp, "      \"finite\": %s,\n", finite_gguf == N_ELEMENTS ? "true" : "false");
        fprintf(json_fp, "      \"int4_vs_gguf\": {\n");
        fprintf(json_fp, "        \"weight_cosine\": %.6f,\n", wc4);
        fprintf(json_fp, "        \"cos_mean\": %.6f,\n", cos_mean4);
        fprintf(json_fp, "        \"cos_min\": %.6f,\n", cos_min4);
        fprintf(json_fp, "        \"mae\": %.6f,\n", mae4);
        fprintf(json_fp, "        \"rmse\": %.6f,\n", rmse4);
        fprintf(json_fp, "        \"max_err\": %.6f,\n", max_err4);
        fprintf(json_fp, "        \"zero_frac\": %.6f,\n", (double)zero4/N_ELEMENTS);
        fprintf(json_fp, "        \"sat_7_frac\": %.6f,\n", (double)sat7_4/N_ELEMENTS);
        fprintf(json_fp, "        \"scale_min\": %.6f,\n", sc4_min);
        fprintf(json_fp, "        \"scale_max\": %.6f,\n", sc4_max);
        fprintf(json_fp, "        \"scale_mean\": %.6f\n", sc4_mean);
        fprintf(json_fp, "      },\n");
        
        if (int8_loaded) {
            fprintf(json_fp, "      \"int8_vs_gguf\": {\n");
            fprintf(json_fp, "        \"weight_cosine\": %.6f,\n", wc8);
            fprintf(json_fp, "        \"cos_mean\": %.6f,\n", cos_mean8);
            fprintf(json_fp, "        \"cos_min\": %.6f,\n", cos_min8);
            fprintf(json_fp, "        \"mae\": %.6f,\n", mae8);
            fprintf(json_fp, "        \"rmse\": %.6f\n", rmse8);
            fprintf(json_fp, "      },\n");
            
            double wc_gap = wc8 - wc4;
            double mc_gap = cos_min8 - cos_min4;
            fprintf(json_fp, "      \"int4_vs_int8_gap\": {\n");
            fprintf(json_fp, "        \"wc_gap\": %.6f,\n", wc_gap);
            fprintf(json_fp, "        \"mc_gap\": %.6f\n", mc_gap);
            fprintf(json_fp, "      },\n");
        }
        
        fprintf(json_fp, "      \"verdict\": \"%s\"\n", verdict);
        fprintf(json_fp, "    }");
        
        // Cleanup
        free(W_gguf);
        free(W_q4);
        free(scales4);
        free(W_int4);
        if (W_int8) free(W_int8);
    }
    
    fprintf(json_fp, "\n  ],\n");
    fprintf(json_fp, "  \"summary\": {\n");
    fprintf(json_fp, "    \"min_weight_cosine\": %.6f,\n", min_wc);
    fprintf(json_fp, "    \"min_matvec_cosine\": %.6f,\n", min_mc);
    fprintf(json_fp, "    \"weakest_layer\": %d,\n", weakest_layer);
    fprintf(json_fp, "    \"pass_count\": %d,\n", pass_count);
    fprintf(json_fp, "    \"maybe_count\": %d,\n", maybe_count);
    fprintf(json_fp, "    \"fail_count\": %d,\n", fail_count);
    fprintf(json_fp, "    \"n_selected\": %d\n", N_SELECTED);
    fprintf(json_fp, "  }\n");
    fprintf(json_fp, "}\n");
    fclose(json_fp);
    
    gguf_free(ctx_gguf);
    ggml_free(ctx);
    
    // Overall verdict
    fprintf(stderr, "\n=== Overall Verdict ===\n");
    fprintf(stderr, "Min WC: %.6f, Min MC: %.6f, Weakest: L%d\n", min_wc, min_mc, weakest_layer);
    fprintf(stderr, "PASS: %d, MAYBE: %d, NO_GO: %d\n", pass_count, maybe_count, fail_count);
    
    const char* overall_verdict;
    if (fail_count > 0 || min_wc < 0.980 || min_mc < 0.980) {
        overall_verdict = "NO_GO_INT4_OFFLINE_PARITY";
    } else if (maybe_count > 0 || min_wc < 0.999 || min_mc < 0.999) {
        overall_verdict = "MAYBE_INT4_OFFLINE_PARITY";
    } else {
        overall_verdict = "PASS_INT4_OFFLINE_PARITY";
    }
    fprintf(stderr, "Overall: %s\n", overall_verdict);
    
    // Estimate INT4 size
    size_t int4_probe_size = (FFN * HIDDEN / 2) + (FFN * 4);  // packed nibbles + scales
    fprintf(stderr, "INT4 probe size per layer: %zu bytes (~%.1f MB)\n", int4_probe_size, int4_probe_size/1e6);
    fprintf(stderr, "Total for 28 layers: %.1f MB\n", int4_probe_size * 28 / 1e6);
    
    fprintf(stderr, "\nDone.\n");
    return 0;
}