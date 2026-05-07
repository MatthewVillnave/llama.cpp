// PRT Phase 13AI: In-process warm harness
// Loads model + PRT sidecars once, runs multiple sequential requests in same process.
// Compares native vs PRT warm request latency.

#include "llama.h"
#include <sys/stat.h>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// PRT API (only available when libllama has PRT support)
extern "C" {
    void llama_set_prt_debug_mode(int mode);
    void llama_set_prt_sidecar(int layer, const float * data, int M, int N);
    void llama_set_prt_force_native_layers(int n_layers, const int * layer_ids);
}

static double us_to_ms(int64_t us) { return us / 1000.0; }

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * sidecar_dir = nullptr;
    int prt_mode = 0;
    int n_tokens_predict = 80;
    int n_ctx = 256;
    int n_threads = 4;
    int n_requests = 5;
    bool prt_active = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--sidecar-dir") == 0 && i+1 < argc) sidecar_dir = argv[++i];
        else if (strcmp(argv[i], "--prt-mode") == 0 && i+1 < argc) prt_mode = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prt-force-native") == 0) prt_active = true;
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) n_tokens_predict = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i+1 < argc) n_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--requests") == 0 && i+1 < argc) n_requests = atoi(argv[++i]);
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s -m <model.gguf> [--sidecar-dir <dir>] [--prt-mode N] [--prt-force-native] [--requests N]\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "=== WARM HARNESS ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "PRT active: %d (mode=%d)\n", prt_active ? 1 : 0, prt_mode);
    fprintf(stderr, "Requests: %d\n\n", n_requests);

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // Load model
    auto model_load_start = std::chrono::high_resolution_clock::now();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) { fprintf(stderr, "ERROR: failed to load model\n"); return 1; }
    double model_load_ms = us_to_ms(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - model_load_start).count());
    fprintf(stderr, "[harness] model loaded: %.1fms\n", model_load_ms);

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { fprintf(stderr, "ERROR: failed to create context\n"); return 1; }

    // If PRT active, load sidecars via mmap (same as cli.cpp)
    double sidecar_load_ms = 0;
    if (prt_active && sidecar_dir) {
        auto sc_start = std::chrono::high_resolution_clock::now();
        llama_set_prt_debug_mode(prt_mode);
        int fn_layers[] = {11, 15};
        llama_set_prt_force_native_layers(2, fn_layers);

        int n_layer = llama_model_n_layer(model);
        int loaded = 0;

        for (int l = 0; l < n_layer; l++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/ffn_up_layer%d_prt.bin", sidecar_dir, l);
            FILE * f = fopen(path, "rb");
            if (!f) continue;

            struct stat st;
            if (stat(path, &st) != 0) { fclose(f); continue; }
            int64_t bytes = st.st_size;

            int M = 0, N = 0;
            if      (bytes == (int64_t)2048 * 11008 * 4) { M = 2048; N = 11008; }
            else if (bytes == (int64_t)896  * 4864 * 4) { M = 896;  N = 4864; }
            else {
                fprintf(stderr, "[harness] unknown sidecar size %ld for layer %d\n", (long)bytes, l);
                fclose(f); continue;
            }

            float * data = nullptr;
            int fd = fileno(f);
            void * mapped = mmap(nullptr, (size_t)M * N * sizeof(float),
                PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapped != MAP_FAILED) {
                data = (float *)mapped;
            } else {
                data = (float *)malloc((size_t)M * N * sizeof(float));
                if (data) {
                    fseek(f, 0, SEEK_SET);
                    if (fread(data, sizeof(float), (size_t)M * N, f) != (size_t)M * N) {
                        free(data); data = nullptr;
                    }
                }
            }
            fclose(f);

            if (data) {
                llama_set_prt_sidecar(l, data, M, N);
                loaded++;
            }
        }

        auto sc_end = std::chrono::high_resolution_clock::now();
        sidecar_load_ms = us_to_ms(std::chrono::duration_cast<std::chrono::microseconds>(sc_end - sc_start).count());
        fprintf(stderr, "[harness] loaded %d/%d sidecars: %.2fms (mmap)\n", loaded, n_layer, sidecar_load_ms);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Greedy sampler (temp=0)
    llama_sampler * sampler = llama_sampler_init_greedy();

    // Test prompts
    const char * prompts[] = {
        "The capital of France is",
        "The capital of France is",
        "The capital of France is",
        "The capital of France is",
        "The capital of France is",
        "Write a Python function that reverses a list.",
        "Return JSON with keys name and status.",
        "Explain CPU inference in one sentence.",
    };
    if (n_requests > 8) n_requests = 8;

    fprintf(stderr, "\n=== Running %d warm requests ===\n", n_requests);

    struct request_result {
        int idx;
        double prompt_ms;
        double gen_ms;
        double total_ms;
        double gen_tok_s;
        int n_tokens;
        bool clean;
        std::string output;
    };
    std::vector<request_result> results;

    // Token buffer (max tokens for any prompt)
    const int MAX_TOKENS = 4096;
    std::vector<llama_token> token_buf(MAX_TOKENS);

    for (int r = 0; r < n_requests; r++) {
        const char * prompt = prompts[r % 8];
        request_result res;
        res.idx = r;
        auto req_start = std::chrono::high_resolution_clock::now();

        // Tokenize
        auto tok_start = std::chrono::high_resolution_clock::now();
        int n_tokens = llama_tokenize(vocab, prompt, -1, token_buf.data(), MAX_TOKENS, true, false);
        auto tok_end = std::chrono::high_resolution_clock::now();
        res.prompt_ms = us_to_ms(std::chrono::duration_cast<std::chrono::microseconds>(tok_end - tok_start).count());

        if (n_tokens < 0) {
            fprintf(stderr, "ERROR: tokenization failed\n");
            break;
        }
        int n_prompt_tokens = n_tokens;

        // Build batch for prompt (fill all fields manually)
        llama_batch batch;
        batch.n_tokens = n_prompt_tokens;
        batch.token = token_buf.data();
        batch.embd = nullptr;
        batch.pos = (llama_pos *)malloc(sizeof(llama_pos) * n_prompt_tokens);
        batch.n_seq_id = (int32_t *)malloc(sizeof(int32_t) * n_prompt_tokens);
        batch.seq_id = (llama_seq_id **)malloc(sizeof(llama_seq_id *) * n_prompt_tokens);
        batch.logits = (int8_t *)malloc(sizeof(int8_t) * n_prompt_tokens);
        for (int i = 0; i < n_prompt_tokens; i++) {
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = (llama_seq_id *)malloc(sizeof(llama_seq_id));
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_prompt_tokens - 1) ? 1 : 0;
        }

        // Decode prompt
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "ERROR: llama_decode failed on prompt\n");
            free(batch.pos); free(batch.n_seq_id);
            for (int i = 0; i < n_prompt_tokens; i++) free(batch.seq_id[i]);
            free(batch.seq_id); free(batch.logits);
            break;
        }
        free(batch.pos); free(batch.n_seq_id);
        for (int i = 0; i < n_prompt_tokens; i++) free(batch.seq_id[i]);
        free(batch.seq_id); free(batch.logits);

        // Generation loop
        auto gen_start = std::chrono::high_resolution_clock::now();
        int n_gen = 0;
        std::string output_text = prompt;
        llama_token new_token;

        for (int i = 0; i < n_tokens_predict; i++) {
            new_token = llama_sampler_sample(sampler, ctx, -1);
            if (new_token == llama_vocab_eos(vocab)) break;

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
            if (n > 0) output_text.append(buf, n);

            // Build next batch for single token
            llama_batch batch_next;
            batch_next.n_tokens = 1;
            batch_next.token = &new_token;
            batch_next.embd = nullptr;
            batch_next.pos = (llama_pos *)malloc(sizeof(llama_pos));
            batch_next.pos[0] = n_prompt_tokens + i;
            batch_next.n_seq_id = (int32_t *)malloc(sizeof(int32_t));
            batch_next.n_seq_id[0] = 1;
            batch_next.seq_id = (llama_seq_id **)malloc(sizeof(llama_seq_id *));
            batch_next.seq_id[0] = (llama_seq_id *)malloc(sizeof(llama_seq_id));
            batch_next.seq_id[0][0] = 0;
            batch_next.logits = (int8_t *)malloc(sizeof(int8_t));
            batch_next.logits[0] = 1;

            if (llama_decode(ctx, batch_next)) break;
            free(batch_next.pos); free(batch_next.n_seq_id);
            free(batch_next.seq_id[0]); free(batch_next.seq_id); free(batch_next.logits);
            n_gen++;
        }

        auto gen_end = std::chrono::high_resolution_clock::now();
        res.gen_ms = us_to_ms(std::chrono::duration_cast<std::chrono::microseconds>(gen_end - gen_start).count());
        res.n_tokens = n_gen;
        res.gen_tok_s = n_gen > 0 ? (n_gen / (res.gen_ms / 1000.0)) : 0;

        auto req_end = std::chrono::high_resolution_clock::now();
        res.total_ms = us_to_ms(std::chrono::duration_cast<std::chrono::microseconds>(req_end - req_start).count());
        res.clean = output_text.find("undefined") == std::string::npos &&
                   output_text.find("NaN") == std::string::npos &&
                   output_text.find("/tmp/prt") == std::string::npos &&
                   output_text.find("nan") == std::string::npos;
        res.output = output_text;

        results.push_back(res);

        fprintf(stderr, "[harness] req %d: prompt_ms=%.1f gen_ms=%.1f total_ms=%.1f tok/s=%.1f tokens=%d clean=%d\n",
            r, res.prompt_ms, res.gen_ms, res.total_ms, res.gen_tok_s, n_gen, res.clean ? 1 : 0);
        fprintf(stderr, "[harness] output: %s\n", output_text.c_str());

        // KV cache cleared via context reuse - no explicit clear needed
    }

    llama_sampler_free(sampler);

    // Summary
    fprintf(stderr, "\n=== SUMMARY ===\n");
    double total_avg = 0, gen_avg = 0, prompt_avg = 0;
    int clean_count = 0;
    for (auto & res : results) {
        total_avg += res.total_ms;
        gen_avg += res.gen_ms;
        prompt_avg += res.prompt_ms;
        if (res.clean) clean_count++;
    }
    int n = (int)results.size();
    if (n > 0) {
        fprintf(stderr, "avg prompt_ms: %.1f\n", prompt_avg / n);
        fprintf(stderr, "avg gen_ms: %.1f\n", gen_avg / n);
        fprintf(stderr, "avg total_ms: %.1f\n", total_avg / n);
        fprintf(stderr, "avg gen tok/s: %.2f\n", gen_avg > 0 ? n / (gen_avg / 1000.0) : 0);
        fprintf(stderr, "clean: %d/%d\n", clean_count, n);
    }

    // JSON output for parsing
    printf("\n{\"mode\":\"%s\",\"model_load_ms\":%.1f,\"sidecar_load_ms\":%.2f,\"n_requests\":%d,\"results\":[",
           prt_active ? "prt" : "native", model_load_ms, sidecar_load_ms, n);
    for (int i = 0; i < n; i++) {
        auto & res = results[i];
        printf("{\"idx\":%d,\"prompt_ms\":%.1f,\"gen_ms\":%.1f,\"total_ms\":%.1f,\"tok_s\":%.1f,\"n_tokens\":%d,\"clean\":%d}",
            res.idx, res.prompt_ms, res.gen_ms, res.total_ms, res.gen_tok_s, res.n_tokens, res.clean ? 1 : 0);
        if (i < n-1) printf(",");
    }
    printf("]}\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}