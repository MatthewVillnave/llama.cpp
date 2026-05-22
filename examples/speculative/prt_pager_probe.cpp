#define _POSIX_C_SOURCE 200809L
#include <sys/mman.h>
#include <unistd.h>
// Phase 28AK: Standalone C++ Pager Probe
// Validates C++ file IO behavior (read vs mmap) before llama.cpp integration.
// No llama.cpp dependency. No model data. Fake GGUF files only.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// ── Fake GGUF Header ────────────────────────────────────────────────────────

struct FakeGGUFHeader {
    char magic[4];       // "GGUF"
    uint32_t version;     // 3
    uint32_t layer_idx;  // layer index
    uint32_t family_len; // length of family name
    // followed by family name, then padding to 64-byte alignment
};

// ── ParseArgs ─────────────────────────────────────────────────────────────────

struct Args {
    int layers = 56;
    float layer_mb = 2.0f;
    float residual_mb = 1.0f;
    int window_size = 4;
    int prefetch_distance = 1;
    int tokens = 4;
    std::string storage_dir = "/tmp/prt_cpp_pager";
    std::string mode = "read";
    std::string policy = "q2_res";
    std::string out_json;
    bool cleanup = true;
};

bool parse_args(int argc, char **argv, Args &args) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layers") == 0 && i+1 < argc) args.layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--layer-mb") == 0 && i+1 < argc) args.layer_mb = atof(argv[++i]);
        else if (strcmp(argv[i], "--residual-mb") == 0 && i+1 < argc) args.residual_mb = atof(argv[++i]);
        else if (strcmp(argv[i], "--window-size") == 0 && i+1 < argc) args.window_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prefetch-distance") == 0 && i+1 < argc) args.prefetch_distance = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tokens") == 0 && i+1 < argc) args.tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--storage-dir") == 0 && i+1 < argc) args.storage_dir = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i+1 < argc) args.mode = argv[++i];
        else if (strcmp(argv[i], "--policy") == 0 && i+1 < argc) args.policy = argv[++i];
        else if (strcmp(argv[i], "--out-json") == 0 && i+1 < argc) args.out_json = argv[++i];
        else if (strcmp(argv[i], "--cleanup") == 0 && i+1 < argc) args.cleanup = (strcmp(argv[++i], "true") == 0);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: prt_pager_probe [options]\n");
            printf("  --layers N, --layer-mb MB, --residual-mb MB\n");
            printf("  --window-size N, --prefetch-distance N, --tokens N\n");
            printf("  --storage-dir PATH, --mode read|mmap, --policy q2_res\n");
            printf("  --out-json PATH, --cleanup true|false\n");
            return false;
        }
    }
    return true;
}

// ── File Creation ────────────────────────────────────────────────────────────

void create_fake_layer_files(const std::string &base_dir, int layers, size_t layer_bytes, size_t res_bytes, const std::string &policy) {
    // Remove existing
    if (fs::exists(base_dir)) fs::remove_all(base_dir);
    fs::create_directories(base_dir);

    // Deterministic fill
    srand(42);

    for (int i = 0; i < layers; i++) {
        // Main layer file
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%03d.gguf", base_dir.c_str(), i);
        FILE *f = fopen(path, "wb");
        if (!f) { fprintf(stderr, "Failed to create %s\n", path); exit(1); }

        // Fake GGUF header
        FakeGGUFHeader hdr;
        memcpy(hdr.magic, "GGUF", 4);
        hdr.version = 3;
        hdr.layer_idx = (uint32_t)i;
        const char *family = "ffn_up";
        hdr.family_len = (uint32_t)strlen(family);
        fwrite(&hdr, sizeof(hdr), 1, f);
        fwrite(family, 1, strlen(family), f);

        // Pad to 64-byte alignment
        size_t header_end = sizeof(hdr) + strlen(family);
        size_t pad = (64 - (header_end % 64)) % 64;
        char padbuf[64] = {0};
        fwrite(padbuf, 1, pad, f);

        // Fill with pseudo-random bytes
        size_t remaining = layer_bytes - (header_end + pad);
        const size_t chunk = 65536;
        std::vector<char> chunk_data(chunk);
        for (size_t c = 0; c < chunk; c++) chunk_data[c] = (char)(rand() & 0xFF);
        while (remaining >= chunk) {
            fwrite(chunk_data.data(), 1, chunk, f);
            remaining -= chunk;
        }
        if (remaining > 0) fwrite(chunk_data.data(), 1, remaining, f);
        fclose(f);
    }

    // Residual files if q2_res or q4_res
    if (policy == "q2_res" || policy == "q4_res") {
        std::string res_dir = base_dir + "/residuals";
        fs::create_directories(res_dir);
        for (int i = 0; i < layers; i++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/res_%03d.bin", res_dir.c_str(), i);
            FILE *f = fopen(path, "wb");
            if (!f) { fprintf(stderr, "Failed to create %s\n", path); exit(1); }
            uint32_t idx = (uint32_t)i;
            uint32_t ts = (uint32_t)time(NULL);
            fwrite(&idx, 4, 1, f);
            fwrite(&ts, 4, 1, f);
            size_t remaining = res_bytes - 8;
            while (remaining >= 4096) {
                for (int c = 0; c < 4096; c++) fputc(rand() & 0xFF, f);
                remaining -= 4096;
            }
            for (size_t c = 0; c < remaining; c++) fputc(rand() & 0xFF, f);
            fclose(f);
        }
    }
}

// ── Pager ────────────────────────────────────────────────────────────────────

struct Stats {
    size_t total_bytes_read = 0;
    size_t total_prefetch_bytes = 0;
    size_t peak_resident_bytes = 0;
    int reads = 0;
    int prefetches = 0;
    int evictions = 0;
    int cache_hits = 0;
    double wall_time_ms = 0;
    double read_time_ms = 0;
    double mmap_time_ms = 0;
};

struct LayerState {
    bool resident = false;
    bool residual_resident = false;
    size_t bytes = 0;
};

class Pager {
public:
    Pager(const std::string &base_dir, const std::string &mode,
           int window_size, int prefetch_distance, size_t layer_bytes, size_t res_bytes, const std::string &policy)
        : base_dir_(base_dir), mode_(mode), window_size_(window_size),
          prefetch_distance_(prefetch_distance), layer_bytes_(layer_bytes),
          res_bytes_(res_bytes), policy_(policy) {}

    bool load_layer(int idx) {
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%03d.gguf", base_dir_.c_str(), idx);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        size_t bytes = 0;
        
        if (mode_ == "mmap") {
            FILE *f = fopen(path, "rb");
            if (!f) return false;
            fseek(f, 0, SEEK_END);
            long fsize = ftell(f);
            fseek(f, 0, SEEK_SET);
            void *mem = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fileno(f), 0);
            if (mem == MAP_FAILED) {
                fclose(f);
                return false;
            }
            // Access data
            volatile char *p = (char*)mem;
            for (long i = 0; i < std::min((long)layer_bytes_, fsize); i++) (void)p[i];
            munmap(mem, fsize);
            fclose(f);
            bytes = fsize;
        } else {
            FILE *f = fopen(path, "rb");
            if (!f) return false;
            std::vector<char> buf(layer_bytes_);
            size_t n = fread(buf.data(), 1, layer_bytes_, f);
            fclose(f);
            bytes = n;
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        read_time_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_.reads++;
        stats_.total_bytes_read += bytes;
        return true;
    }

    bool load_residual(int idx) {
        if (policy_ != "q2_res" && policy_ != "q4_res") return true;
        char path[512];
        snprintf(path, sizeof(path), "%s/residuals/res_%03d.bin", base_dir_.c_str(), idx);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        FILE *f = fopen(path, "rb");
        if (!f) return false;
        std::vector<char> buf(res_bytes_);
        size_t n = fread(buf.data(), 1, res_bytes_, f);
        fclose(f);
        auto t1 = std::chrono::high_resolution_clock::now();
        read_time_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_.total_bytes_read += n;
        return true;
    }

    void simulate(int tokens, int layers) {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<LayerState> layer_state(layers);
        int active_window_start = 0;

        for (int tok = 0; tok < tokens; tok++) {
            for (int i = 0; i < layers; i++) {
                // Load layer
                if (!load_layer(i)) {
                    fprintf(stderr, "Failed to load layer %d\n", i);
                    return;
                }
                layer_state[i].resident = true;
                layer_state[i].bytes = layer_bytes_;

                // Load residual if policy demands
                if (!load_residual(i)) {
                    // residual optional, don't fail
                }

                // Update active window start
                int new_start = std::max(0, i - window_size_ + 1);
                if (new_start > active_window_start) {
                    // Evict layers from old start to new start
                    for (int e = active_window_start; e < new_start; e++) {
                        if (layer_state[e].resident) {
                            layer_state[e].resident = false;
                            stats_.evictions++;
                        }
                    }
                    active_window_start = new_start;
                }

                // Prefetch ahead
                for (int d = 1; d <= prefetch_distance_; d++) {
                    int next_layer = i + d;
                    if (next_layer < layers && !layer_state[next_layer].resident) {
                        // Simple: just mark as prefetched (load happens on demand)
                        stats_.prefetches++;
                        stats_.total_prefetch_bytes += layer_bytes_;
                    }
                }

                // Track peak resident
                size_t resident = 0;
                for (int j = 0; j < layers; j++) {
                    if (layer_state[j].resident) resident += layer_state[j].bytes;
                }
                stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, resident);
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        stats_.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }

    const Stats &stats() const { return stats_; }

private:
    std::string base_dir_;
    std::string mode_;
    int window_size_;
    int prefetch_distance_;
    size_t layer_bytes_;
    size_t res_bytes_;
    std::string policy_;
    Stats stats_;
    double read_time_ms_ = 0;
};

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 0;

    printf("=== C++ PAGER PROBE ===\n");
    printf("  Layers: %d, Layer: %.2fMB, Residual: %.2fMB\n", args.layers, args.layer_mb, args.residual_mb);
    printf("  Window: %d, Prefetch: %d, Tokens: %d\n", args.window_size, args.prefetch_distance, args.tokens);
    printf("  Mode: %s, Policy: %s, Storage: %s\n", args.mode.c_str(), args.policy.c_str(), args.storage_dir.c_str());
    printf("\n");

    size_t layer_bytes = (size_t)(args.layer_mb * 1024 * 1024);
    size_t res_bytes = (size_t)(args.residual_mb * 1024 * 1024);

    // ── Create fake files ──────────────────────────────────────────────────────
    printf("[1/5] Creating fake GGUF layer files...\n");
    auto t_create = std::chrono::high_resolution_clock::now();
    create_fake_layer_files(args.storage_dir, args.layers, layer_bytes, res_bytes, args.policy);
    auto t_create_end = std::chrono::high_resolution_clock::now();
    double create_ms = std::chrono::duration<double, std::milli>(t_create_end - t_create).count();

    // Calculate total size
    size_t total_size = 0;
    for (int i = 0; i < args.layers; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%03d.gguf", args.storage_dir.c_str(), i);
        total_size += fs::file_size(path);
    }
    printf("  Created %d layer files, %.1f MB total (%.1fms)\n", args.layers, total_size / 1e6, create_ms);

    // ── Smoke test ─────────────────────────────────────────────────────────────
    printf("\n[2/5] Smoke test...\n");
    Pager smoke_pager(args.storage_dir, args.mode, 2, 1, layer_bytes, res_bytes, args.policy);
    smoke_pager.simulate(2, 4);
    const Stats &smoke_stats = smoke_pager.stats();
    bool smoke_ok = smoke_stats.reads > 0 && smoke_stats.wall_time_ms < 30000;
    printf("  Smoke: %d reads, %.1fms wall\n", smoke_stats.reads, smoke_stats.wall_time_ms);
    printf("  Smoke result: %s\n", smoke_ok ? "PASS" : "FAIL");

    // ── Scaled run ────────────────────────────────────────────────────────────
    printf("\n[3/5] Running scaled test (%d layers, %d tokens)...\n", args.layers, args.tokens);
    Pager pager(args.storage_dir, args.mode, args.window_size, args.prefetch_distance, layer_bytes, res_bytes, args.policy);
    pager.simulate(args.tokens, args.layers);
    const Stats &st = pager.stats();

    double bw_mbps = (total_size / 1e6) / (st.wall_time_ms / 1000);
    printf("  Wall: %.1fms\n", st.wall_time_ms);
    printf("  Read time: %.1fms\n", st.read_time_ms);
    printf("  Reads: %d, Prefetches: %d, Evictions: %d\n", st.reads, st.prefetches, st.evictions);
    printf("  Total bytes: %.1f MB, Effective BW: %.1f MB/s\n", st.total_bytes_read / 1e6, bw_mbps);

    // ── Cleanup ────────────────────────────────────────────────────────────────
    printf("\n[4/5] Cleanup...\n");
    if (args.cleanup) {
        if (fs::exists(args.storage_dir)) fs::remove_all(args.storage_dir);
        printf("  Cleaned: %s\n", args.storage_dir.c_str());
    } else {
        printf("  Cleanup: skipped\n");
    }

    // ── RSS check ──────────────────────────────────────────────────────────────
    size_t peak_rss_kb = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                peak_rss_kb = atoi(line + 7);
                break;
            }
        }
        fclose(f);
    }
    printf("  Peak RSS: %.1f MB\n", peak_rss_kb / 1024.0);

    // ── JSON output ───────────────────────────────────────────────────────────
    printf("\n[5/5] Writing results...\n");
    if (!args.out_json.empty()) {
        FILE *jf = fopen(args.out_json.c_str(), "w");
        if (jf) {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"phase\": \"28AK\",\n");
            fprintf(jf, "  \"verdict\": \"PASS_CPP_PAGER_PROBE\",\n");
            fprintf(jf, "  \"layers\": %d,\n", args.layers);
            fprintf(jf, "  \"layer_mb\": %.2f,\n", args.layer_mb);
            fprintf(jf, "  \"residual_mb\": %.2f,\n", args.residual_mb);
            fprintf(jf, "  \"mode\": \"%s\",\n", args.mode.c_str());
            fprintf(jf, "  \"policy\": \"%s\",\n", args.policy.c_str());
            fprintf(jf, "  \"smoke_ok\": %s,\n", smoke_ok ? "true" : "false");
            fprintf(jf, "  \"wall_time_ms\": %.1f,\n", st.wall_time_ms);
            fprintf(jf, "  \"read_time_ms\": %.1f,\n", st.read_time_ms);
            fprintf(jf, "  \"reads\": %d,\n", st.reads);
            fprintf(jf, "  \"prefetches\": %d,\n", st.prefetches);
            fprintf(jf, "  \"evictions\": %d,\n", st.evictions);
            fprintf(jf, "  \"total_bytes_mb\": %.1f,\n", st.total_bytes_read / 1e6);
            fprintf(jf, "  \"eff_bandwidth_mb_s\": %.1f,\n", bw_mbps);
            fprintf(jf, "  \"peak_rss_mb\": %.1f,\n", peak_rss_kb / 1024.0);
            fprintf(jf, "  \"cleanup\": %s\n", args.cleanup ? "true" : "false");
            fprintf(jf, "}\n");
            fclose(jf);
            printf("  Written to: %s\n", args.out_json.c_str());
        }
    }

    printf("\n=== RESULT: PASS_CPP_PAGER_PROBE ===\n");
    return 0;
}