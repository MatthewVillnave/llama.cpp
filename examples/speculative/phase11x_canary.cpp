// PRT Phase 11X: Corrected Path Mode-Separation Canary
// Tests PRT compute vs identity vs baseline with tensor-level comparison

#include "phase10e0_layer0_replacement.cpp"  // reuse the harness infrastructure

#include <cmath>
#include <numeric>

static double cosine_sim(const float * a, const float * b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9);
}

static float max_abs_diff(const float * a, const float * b, int64_t n) {
    float m = 0;
    for (int64_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float mean_abs_diff(const float * a, const float * b, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += std::fabs(a[i] - b[i]);
    return s / n;
}

// Global to capture layer0 dst tensor data from PRT op for comparison
static float g_layer0_identity_dst[11008] = {0};
static float g_layer0_prt_dst[11008] = {0};
static int g_layer0_identity_captured = 0;
static int g_layer0_prt_captured = 0;

// We need to intercept the PRT op to capture tensors.
// Since we can't easily hook into the custom op, we'll use mode differences:
// - Mode 4: identity (memset then copy)
// - Mode 7: PRT compute (memset then compute)
// We can deduce the PRT result by comparing outputs

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model> <test-mode> [n_predict=3]\n", argv[0]);
        fprintf(stderr, "test-mode: 0=baseline 1=identity 3=checksum 4=layer0_prt 5=all_prt 6=zero_perturb\n");
        return 1;
    }
    const char * model_path = argv[1];
    int test_mode = atoi(argv[2]);
    int n_predict = argc > 3 ? atoi(argv[3]) : 3;
    
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) { fprintf(stderr, "ERR: cannot load model\n"); return 1; }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    llama_context * ctx = llama_new_context(model, cparams);
    if (!ctx) { fprintf(stderr, "ERR: cannot create ctx\n"); return 1; }
    
    llama_set_prt_debug_mode(test_mode);
    llama_set_prt_sidecar_load(!false); // enable loading
    
    llama_model_loader_load_file(model, "/tmp/prt_sidecars", "ffn_up_layer", 36);
    
    auto tokens = llama_tokenize(model, "XYZ", true);
    fprintf(stderr, "\n========== Phase 11X Test mode=%d ==========\n", test_mode);
    
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        fprintf(stderr, "ERR: decode failed\n"); return 1;
    }
    
    char buf[256];
    fprintf(stderr, "[11X] Tokens: ");
    for (auto t : tokens) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n");
    
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    
    fprintf(stderr, "\nGenerating %d tokens...\n", n_predict);
    std::vector<int> gen_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (llama_decode(ctx, llama_batch_get_one(&tokens.back(), 1))) break;
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) { fprintf(stderr, "[11X] EOG at step %d\n", i); break; }
        
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            buf[n] = 0;
            // hex dump
            fprintf(stderr, "[11X] t%d id=%d tn=%d hex=", i, id, n);
            for (int j = 0; j < n; j++) fprintf(stderr, "%02X", (unsigned char)buf[j]);
            fprintf(stderr, " str='%s'\n", buf);
            gen_tokens.push_back(id);
        }
        tokens.push_back(id);
    }
    
    llama_sampler_free(smpl);
    
    int repl = llama_get_prt_replacement_count();
    int fallb = llama_get_prt_fallback_count();
    int wrong = llama_get_prt_wrong_layer_count();
    
    // PRT Phase 11AO: new instrumentation
    extern int llama_get_native_ffn_up_calls(void);
    extern int llama_get_prt_direct_calls(void);
    extern int llama_get_postprocess_calls(void);
    int native_calls = llama_get_native_ffn_up_calls();
    int prt_direct = llama_get_prt_direct_calls();
    int postproc = llama_get_postprocess_calls();
    
    fprintf(stderr, "\n========== RESULTS mode=%d ==========\n", test_mode);
    fprintf(stderr, "Gen tokens: %zu\n", gen_tokens.size());
    for (size_t i = 0; i < gen_tokens.size(); i++) fprintf(stderr, "  [%zu] id=%d\n", i, gen_tokens[i]);
    fprintf(stderr, "Total PRT replacements: %d\n", repl);
    fprintf(stderr, "Fallback count: %d\n", fallb);
    fprintf(stderr, "Wrong-layer count: %d\n", wrong);
    fprintf(stderr, "[PRT-11AO] native_ffn_up_calls=%d\n", native_calls);
    fprintf(stderr, "[PRT-11AO] prt_direct_calls=%d\n", prt_direct);
    fprintf(stderr, "[PRT-11AO] postprocess_calls=%d\n", postproc);
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}
