// PRT Phase 10E-0: Layer 0 ffn_up Replacement via Eval Callback Hook

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>

// Phase 11AY: AVX2 SIMD kernel
#include "prt_avx2_kernel.h"

#define T_HIGH_0 2.0f
#define T_HIGH_1 0.5f
#define T_HIGH_2 0.1f
static const int TOTAL_LAYERS = 36;  // Phase 10E-5: all 36 layers
static int g_n_layer = 0;       // Phase 13B: set from llama_model_n_layer() at load
static int g_prt_M = 0;          // Phase 13B: hidden dim (from ffn_up tensor ne[1])
static int g_prt_N = 0;          // Phase 13B: ffn dim (from ffn_up tensor ne[0])

// Phase 11AY: SIMD kernel selection (0=scalar, 1=AVX2)
extern int g_prt_debug_mode;
static int g_prt_kernel_mode = 0;
extern int g_native_fallback_calls;  // Phase 11BD
extern "C" int llama_get_native_ffn_up_calls(void);
extern "C" int llama_get_prt_true_replacement_calls(void);
extern "C" int llama_get_native_fallback_calls(void);
extern "C" float llama_get_sidecar_checksum(int layer);
extern "C" void llama_set_prt_force_native_layers(int n_layers, const int * layer_ids);
extern "C" void llama_clear_prt_force_native(void);
extern "C" void llama_set_prt_sidecar(int layer, const float * data, int M, int N);
extern "C" void llama_set_prt_debug_mode(int mode);
extern "C" int llama_get_prt_replacement_count(void);
extern "C" int llama_get_prt_fallback_count(void);

// Scalar kernel (threshold-sparse, corrected orientation)
static void matmul_prt_scalar(const float * X, const float * W_prt, float * Y, int batch, int M, int N) {
    for (int b = 0; b < batch; b++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                float x = X[b * M + k];
                float ax = fabsf(x);
                if (ax > T_HIGH_0) sum += x * W_prt[n * M + k];
                else if (ax > T_HIGH_1) sum += x * W_prt[n * M + k];
                else if (ax > T_HIGH_2) sum += x * W_prt[n * M + k];
            }
            Y[b * N + n] = sum;
        }
    }
}

// Phase 11AY: SIMD wrapper - dispatches to correct kernel
static void matmul_prt(const float * X, const float * W_prt, float * Y, int batch, int M, int N) {
    if (g_prt_kernel_mode == 1) {
#if defined(__AVX2__)
        matmul_prt_avx2(X, W_prt, Y, batch, M, N);
#else
        matmul_prt_scalar(X, W_prt, Y, batch, M, N);
#endif
    } else {
        matmul_prt_scalar(X, W_prt, Y, batch, M, N);
    }
}

struct Sidecar { int layer; float * data; int M, N; };
// PRT Phase 11W: selectable sidecar loader
// LOADER_TYPE: 0=POSIX open/read (DEFAULT/PRODUCTION), 1=mmap, 2=static/synthetic
// NOTE: fopen/fread REMOVED — Phase 11W: fopen causes filename-leak corruption in logits
#ifndef PRT_LOADER_TYPE
#define PRT_LOADER_TYPE 0  // PRODUCTION DEFAULT: POSIX open/read (no stdio)
#endif

static std::map<int, Sidecar> g_sidecars;
static bool g_sidecars_loaded = false;

static bool load_sidecar_posix(int layer) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/prt_sidecars/ffn_up_layer%d_prt.bin", layer);
    // POSIX open/read — no stdio FILE* buffering
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return false; }
    size_t sz = st.st_size;
    float * data = (float *)mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        // fallback: malloc + read
        data = (float *)malloc(sz);
        if (!data) { close(fd); return false; }
        ssize_t got = read(fd, data, sz);
        if ((size_t)got != sz) { free(data); close(fd); return false; }
    }
    close(fd);
    g_sidecars[layer] = {layer, data, g_prt_M, g_prt_N};
    (void)sz;
    return true;
}

static bool load_sidecar_mmap(int layer) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/prt_sidecars/ffn_up_layer%d_prt.bin", layer);
    // Direct mmap — no read(), no FILE*
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return false; }
    size_t sz = st.st_size;
    float * data = (float *)mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // fd can be closed after mmap
    if (data == MAP_FAILED) return false;
    g_sidecars[layer] = {layer, data, g_prt_M, g_prt_N};
    return true;
}

static bool load_sidecar_fopen(int layer) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/prt_sidecars/ffn_up_layer%d_prt.bin", layer);
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    float * data = (float *)malloc(sz);
    if (!data || fread(data, 1, sz, f) != sz) { free(data); fclose(f); return false; }
    fclose(f);
    g_sidecars[layer] = {layer, data, g_prt_M, g_prt_N};
    return true;
}

static bool load_sidecar_static(int layer) {
    // Synthetic sidecar: all zeros (no file access at all)
    // Phase 13B: use dynamic M/N instead of hardcoded 2048x11008
    int M = g_prt_M > 0 ? g_prt_M : 2048;
    int N = g_prt_N > 0 ? g_prt_N : 11008;
    static float * s_zero_sidecar = nullptr;
    static int s_alloc_M = 0, s_alloc_N = 0;
    if (M != s_alloc_M || N != s_alloc_N) {
        free(s_zero_sidecar);
        s_zero_sidecar = (float *)calloc(M * N, sizeof(float));
        s_alloc_M = M; s_alloc_N = N;
    }
    if (!s_zero_sidecar) return false;
    g_sidecars[layer] = {layer, s_zero_sidecar, M, N};
    return true;
}

static bool load_sidecar(int layer) {
    if (PRT_LOADER_TYPE == 1) {
        //
        return load_sidecar_mmap(layer);
    }
    if (PRT_LOADER_TYPE == 2) {
        //
        return load_sidecar_static(layer);
    }
    //
    return load_sidecar_posix(layer);  // type 0 = POSIX open/read (PRODUCTION)
}

static void load_all_sidecars(struct llama_model * model) {
    fprintf(stderr, "[PRT-DEBUG] load_all_sidecars called\n");
    if (g_sidecars_loaded) return;
    // Phase 13B: derive dynamic dims from loaded model
    g_n_layer = llama_model_n_layer(model);
    // Read M (hidden) and N (ffn) from the ffn_up tensor shape
    // Shape: blk.0.ffn_up.weight = [n_ff, n_embd] in the file
    // We need to find the tensor via ggml context - use the model loader's ctx
    // Since we don't have ctx yet, use the file-size heuristic: check one sidecar
    // The sidecar file size tells us M*N directly:
    char path[256];
    snprintf(path, sizeof(path), "/tmp/prt_sidecars/ffn_up_layer0_prt.bin", 0);
    struct stat st;
    if (stat(path, &st) == 0) {
        int64_t expected_bytes = st.st_size;
        // Try to derive M/N from known model shapes
        // For Qwen2.5-3B: 2048*11008*4 = 90,113,024
        // For Qwen2.5-0.5B: 896*4864*4 = 17,432,576
        // For Qwen2.5-1.5B: 1536*8960*4 = 55,049,216
        if (expected_bytes == (int64_t)2048 * 11008 * 4) { g_prt_M = 2048; g_prt_N = 11008; }
        else if (expected_bytes == (int64_t)896 * 4864 * 4) { g_prt_M = 896; g_prt_N = 4864; }
        else if (expected_bytes == (int64_t)1536 * 8960 * 4) { g_prt_M = 1536; g_prt_N = 8960; }
        else {
            // Fallback: assume 3B dims
            g_prt_M = 2048; g_prt_N = 11008;
            fprintf(stderr, "[PRT WARNING] Unknown sidecar size %ld, assuming M=%d N=%d\n",
                    (long)expected_bytes, g_prt_M, g_prt_N);
        }
    } else {
        g_prt_M = 2048; g_prt_N = 11008;
    }
    fprintf(stderr, "[PRT] Dynamic shape: n_layer=%d, M=%d, N=%d\n", g_n_layer, g_prt_M, g_prt_N);
    int64_t expected_bytes = (int64_t)g_prt_M * g_prt_N * 4;
    fprintf(stderr, "[PRT_SHAPE] n_layer=%d M=%d N=%d expected_bytes=%ld sidecars=%d force_native=11,15\n",
            g_n_layer, g_prt_M, g_prt_N, (long)expected_bytes, g_n_layer);
    int loaded = 0;
    for (int l = 0; l < g_n_layer; l++) {
        if (load_sidecar(l)) {
            loaded++;
            auto it = g_sidecars.find(l);
            if (it != g_sidecars.end()) {
                llama_set_prt_sidecar(l, it->second.data, it->second.M, it->second.N);
            }
        }
    }
    fprintf(stderr, "[PRT] Loaded %d/%d sidecars\n", loaded, g_n_layer);
    g_sidecars_loaded = true;
}

// Phase 11BP: loud startup validation for PRT Route A sidecars
static bool validate_prt_sidecars(void) {
    extern int g_prt_debug_mode;
    extern bool g_prt_force_native_layer[36];
    extern bool g_prt_force_native_enabled;


    // Mode 5700 = Route A (all layers PRT by default)
    bool route_a_mode = (g_prt_debug_mode >= 5700);
    if (!route_a_mode) {
        fprintf(stderr, "[PRT-11BP] PRT Route A not active (mode=%d) — skipping validation\n", g_prt_debug_mode);
        return true;  // no validation needed for non-Route-A modes
    }


    fprintf(stderr, "\n[PRT-11BP] === Sidecar Validation ===\n");
    fprintf(stderr, "[PRT-11BP] PRT mode=%d, Route A active\n", g_prt_debug_mode);


    int required = 0;
    int missing = 0;
    int force_native_count = 0;

    // Phase 13B: use dynamic g_n_layer instead of TOTAL_LAYERS
    for (int l = 0; l < g_n_layer; l++) {
        bool force_native = g_prt_force_native_enabled && g_prt_force_native_layer[l];
        if (force_native) force_native_count++;


        // In Route A mode, ALL layers need PRT except force-native layers
        if (!force_native) {
            required++;
            if (g_sidecars.find(l) == g_sidecars.end()) {
                fprintf(stderr, "[PRT-ERROR] Layer %d: required PRT sidecar MISSING\n", l);
                missing++;
            }
        }
    }

    fprintf(stderr, "[PRT-11BP] Required PRT layers: %d\n", required);
    fprintf(stderr, "[PRT-11BP] Force-native layers: %d", force_native_count);
    if (force_native_count > 0) {
        fprintf(stderr, " (skipped — allowed via --prt-force-native)");
    }
    fprintf(stderr, "\n");

    if (missing > 0) {
        fprintf(stderr, "\n[PRT-ERROR] FATAL: %d required sidecar(s) missing. Exiting.\n", missing);
        fprintf(stderr, "[PRT-ERROR] Use --prt-force-native to mark layers as native if sidecar is unavailable.\n");
        return false;
    }

    fprintf(stderr, "\n[PRT-11BP] === Sidecar Checksums ===\n");
    fprintf(stderr, "[PRT-11BP] L0:  %.6f\n", llama_get_sidecar_checksum(0));
    fprintf(stderr, "[PRT-11BP] L12: %.6f\n", llama_get_sidecar_checksum(12));
    fprintf(stderr, "[PRT-11BP] L15: %.6f\n", llama_get_sidecar_checksum(15));
    fprintf(stderr, "[PRT-11BP] L35: %.6f\n", llama_get_sidecar_checksum(35));


    fprintf(stderr, "\n[PRT-11BP] VALIDATION PASSED — all required sidecars present\n");
    return true;
}

struct PrtState {
    int total_replacements = 0;
    int callback_overwrites = 0;  // Phase 11BD
    float last_cosine = 0.0f;
};

static PrtState g_prt_state;

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na < 1e-10 || nb < 1e-10) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

static bool prt_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    PrtState * state = (PrtState *)user_data;
    
    if (ask && t->op == GGML_OP_MUL_MAT) {
        fprintf(stderr, "  [DEBUG] mul_mat: %s\n", t->name);
    }
    
    // Phase 11BB: Skip callback in Route A modes (5700+), only use for callback modes (5435, 5600-5635)
    // Route A uses build_ffn replacement, not callback
    bool callback_mode = (g_prt_debug_mode < 5700);
    
extern int g_prt_debug_mode;
    if (ask) {
        if (callback_mode && t->op == GGML_OP_MUL_MAT && strstr(t->name, "ffn_up")) {
            int layer = -1;
            if ((sscanf(t->name, "ffn_up-%d", &layer) == 1 || sscanf(t->name, "blk.%d.ffn_up", &layer) == 1) && layer >= 0 && layer < g_n_layer) {
                fprintf(stderr, "  [ASK] ffn_up tensor found: name='%s' layer=%d sidecar_loaded=%d\n",
                        t->name, layer, g_sidecars.find(layer) != g_sidecars.end());
                if (g_sidecars.find(layer) != g_sidecars.end()) {
                    return true;
                }
            } else {
                fprintf(stderr, "  [ASK] ffn_up tensor NO MATCH: name='%s'\n", t->name);
            }
        }
        return false;
    }
    
    int layer = -1;
    if ((sscanf(t->name, "ffn_up-%d", &layer) != 1 && sscanf(t->name, "blk.%d.ffn_up", &layer) != 1)) return true;
    if (layer < 0 || layer >= g_n_layer) return true;
    
    auto it = g_sidecars.find(layer);
    if (it == g_sidecars.end()) return true;
    const Sidecar & sc = it->second;
    
    const struct ggml_tensor * src1 = t->src[1];
    if (!src1) return true;
    
    int M = (int)src1->ne[0];
    int batch = (int)src1->ne[1];
    int N = (int)t->ne[0];
    
    if (M != sc.M || N != sc.N) return true;
    
    std::vector<float> input_data(batch * M);
    std::vector<float> float_output(batch * N);
    std::vector<float> prt_output(batch * N);
    
    bool src1_on_host = ggml_backend_buffer_is_host(src1->buffer);
    if (src1_on_host) {
        memcpy(input_data.data(), src1->data, batch * M * sizeof(float));
    } else {
        ggml_backend_tensor_get(src1, input_data.data(), 0, batch * M * sizeof(float));
    }
    
    bool dst_on_host = ggml_backend_buffer_is_host(t->buffer);
    if (dst_on_host) {
        memcpy(float_output.data(), t->data, batch * N * sizeof(float));
    } else {
        ggml_backend_tensor_get(t, float_output.data(), 0, batch * N * sizeof(float));
    }
    
    matmul_prt(input_data.data(), sc.data, prt_output.data(), batch, M, N);
    
    float cos = cosine_sim(float_output.data(), prt_output.data(), batch * N);
    float max_err = 0.0f;
    for (int i = 0; i < batch * N; i++) {
        float e = fabsf(float_output[i] - prt_output[i]);
        if (e > max_err) max_err = e;
    }
    
    if (dst_on_host) {
        memcpy(t->data, prt_output.data(), batch * N * sizeof(float));
    } else {
        ggml_backend_tensor_set(t, prt_output.data(), 0, batch * N * sizeof(float));
    }
    
    state->total_replacements++;
    state->callback_overwrites++;
    state->last_cosine = cos;
    
    fprintf(stderr, "  [PRT] Layer %d REPLACED: batch=%d cos=%.6f max_err=%.6f\n", layer, batch, cos, max_err);
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
    std::string prompt = "Hello";
    int n_predict = 10;
    int n_ctx = 512;
    
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0) n_predict = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prt-mode") == 0) {
            int mode = atoi(argv[++i]);
            llama_set_prt_debug_mode(mode);
        }
        else if (strcmp(argv[i], "--prt-threshold") == 0) {
            float thresh = atof(argv[++i]);
            extern float g_prt_threshold;
            g_prt_threshold = thresh;
            fprintf(stderr, "[PRT-11AV] threshold set to %.4f\n", thresh);
        }
        else if (strcmp(argv[i], "--prt-kernel") == 0) {
            int kmode = atoi(argv[++i]);
            g_prt_kernel_mode = kmode;
            fprintf(stderr, "[PRT-11AY] kernel mode set to %d (%s)\n", kmode, kmode == 1 ? "AVX2" : "scalar");
        }
        else if (strcmp(argv[i], "--prt-force-native") == 0) {
            // Format: --prt-force-native L0,L1,L5 (comma-separated layer IDs)
            const char * layers_str = argv[++i];
            std::vector<int> layers;
            char * copy = strdup(layers_str);
            char * token = strtok(copy, ",");
            while (token) {
                layers.push_back(atoi(token));
                token = strtok(NULL, ",");
            }
            free(copy);
            if (!layers.empty()) {
                llama_set_prt_force_native_layers(layers.size(), layers.data());
            }
        }
    }

    fprintf(stderr, "=== PRT Phase 11S ===\n");
    
    fprintf(stderr, "=== PRT Phase 10E-0 ===\n"); //
    
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) return 1;
    fprintf(stderr, "Model: n_layers=%d\n", llama_model_n_layer(model));
    
    load_all_sidecars(model);
    if (g_sidecars.empty()) return 1;

    // Phase 11BP: validate sidecars before generation
    if (!validate_prt_sidecars()) return 1;
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.cb_eval = prt_eval_callback;
    cparams.cb_eval_user_data = &g_prt_state;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) return 1;
    tokens.resize(n_tokens);
    
    fprintf(stderr, "\n=== First decode ===\n");
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) return 1;
    fprintf(stderr, "Promp decoded (replacements: %d)\n", llama_get_prt_replacement_count());
    
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    
    fprintf(stderr, "\nGenerating...\n");
    char buf[256];
    for (int i = 0; i < n_predict; i++) {
        fprintf(stderr, "[GEN] step %d: calling decode...\n", i);
        if (llama_decode(ctx, llama_batch_get_one(&tokens.back(), 1))) { fprintf(stderr, "[GEN] decode failed at step %d\n", i); break; }
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        fprintf(stderr, "[GEN] step %d: token_id=%d is_eog=%d\n", i, id, llama_vocab_is_eog(vocab, id));
        char tbuf[256];
        int tn = llama_token_to_piece(vocab, id, tbuf, sizeof(tbuf), 0, true);
        fprintf(stderr, "[GEN] step %d: tn=%d tbuf_hex: ", i, tn);
        for (int j = 0; j < tn && j < 32; j++) fprintf(stderr, "%02X ", (unsigned char)tbuf[j]);
        fprintf(stderr, "\n");
        if (llama_vocab_is_eog(vocab, id)) { fprintf(stderr, "[GEN] EOG at step %d\n", i); break; }
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            // Hex dump the first 16 bytes of buf
            fprintf(stderr, "[DEBUG] token %d: n=%d buf_hex: ", i, n);
            for (int j = 0; j < n && j < 16; j++) fprintf(stderr, "%02X ", (unsigned char)buf[j]);
            fprintf(stderr, " str='%s'\n", buf);
            fprintf(stderr, "  %d: '%s' (replaced: %d)\n", i, buf, llama_get_prt_replacement_count());
            tokens.back() = id;
        }
    }
    llama_sampler_free(smpl);
    
    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Total PRT replacements: %d  [fallback: %d]\n", llama_get_prt_replacement_count(), llama_get_prt_fallback_count());
    fprintf(stderr, "Last cosine: %.6f\n", g_prt_state.last_cosine);
    fprintf(stderr, "[11BD] callback_overwrites: %d\n", g_prt_state.callback_overwrites);
    fprintf(stderr, "[11BD] native_ffn_up_calls: %d\n", llama_get_native_ffn_up_calls());
    fprintf(stderr, "[11BD] prt_true_replacement_calls: %d\n", llama_get_prt_true_replacement_calls());
    fprintf(stderr, "[11BD] native_fallback_calls: %d\n", llama_get_native_fallback_calls());
    fprintf(stderr, "[11BD] sidecar_L0_checksum: %.6f\n", llama_get_sidecar_checksum(0));
    fprintf(stderr, "[11BD] sidecar_L35_checksum: %.6f\n", llama_get_sidecar_checksum(35));
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}
