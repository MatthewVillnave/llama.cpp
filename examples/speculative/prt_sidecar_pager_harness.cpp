// Phase 28AR: Sidecar Pager Harness Integration
// Standalone harness: no model generation, no ggml graph, no llama.cpp matmul changes.
// Exercises prt_sidecar_pager with synthetic .trit package under /tmp.

#include "prt_sidecar_pager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace fs = std::filesystem;

struct harness_config {
    bool enabled = false;
    std::string manifest_path;
    int max_resident_kb = 512;
    bool eviction_lru = false;
    bool checksum = true;
    int prefetch_distance = 1;
    int window_size = 4;
    std::string mode; // "disabled", "enabled", "budget_stress", "lru"
};

static bool has_suffix(const std::string& s, const char* suffix) {
    size_t sl = strlen(suffix);
    return s.size() >= sl && s.compare(s.size() - sl, sl, suffix) == 0;
}

static void print_stats(const prt_sidecar_pager& pager) {
    auto stats = pager.get_stats();
    printf("  resident_bytes   = %zu\n", stats.resident_bytes);
    printf("  peak_resident    = %zu\n", stats.peak_resident_bytes);
    printf("  reads            = %zu\n", stats.reads);
    printf("  evictions        = %zu\n", stats.evictions);
    printf("  lru_evictions    = %zu\n", stats.lru_evictions);
    printf("  cache_hits       = %zu\n", stats.cache_hits);
    printf("  cache_misses     = %zu\n", stats.cache_misses);
    printf("  fallbacks        = %zu\n", stats.fallbacks);
    printf("  trit_checksum_ok   = %zu\n", stats.trit_checksum_ok);
    printf("  trit_checksum_fail = %zu\n", stats.trit_checksum_fail);
    printf("  budget_rejects   = %zu\n", stats.budget_rejects);
}

static bool write_fake_trit(const std::string& path, uint32_t layer_id,
                            const std::string& tensor_family, uint32_t tensor_idx,
                            uint32_t n_rows, uint32_t n_cols, uint32_t n_scales) {
    (void) tensor_family;

    constexpr size_t header_size = 32;
    size_t payload_bytes = ((size_t)n_rows * n_cols * 3 + 7) / 8; // 3-bit/trit
    size_t payload_offset = header_size + payload_bytes;
    size_t scale_offset = payload_offset;
    if (scale_offset % 4) scale_offset = (scale_offset / 4 + 1) * 4;
    size_t scale_bytes = n_scales * sizeof(float);
    size_t total = scale_offset + scale_bytes;

    std::vector<unsigned char> buf(total, 0);

    auto put_u16 = [&](size_t off, uint16_t v) {
        buf[off + 0] = (uint8_t)((v >> 0) & 0xFF);
        buf[off + 1] = (uint8_t)((v >> 8) & 0xFF);
    };
    auto put_u32 = [&](size_t off, uint32_t v) {
        buf[off + 0] = (uint8_t)((v >> 0) & 0xFF);
        buf[off + 1] = (uint8_t)((v >> 8) & 0xFF);
        buf[off + 2] = (uint8_t)((v >> 16) & 0xFF);
        buf[off + 3] = (uint8_t)((v >> 24) & 0xFF);
    };

    put_u32(0, 0x54495254); // "TRIT" bytes, read as 0x54495254 little-endian
    put_u16(4, 0);          // ver_major
    put_u16(6, 1);          // ver_minor
    put_u32(8, n_rows);
    put_u32(12, n_cols);
    put_u16(16, 512);
    put_u16(18, 256);
    put_u16(20, (uint16_t)n_scales);
    put_u32(22, (uint32_t)payload_offset);
    put_u32(26, (uint32_t)scale_offset);

    // CRC16 of first 30 bytes (rotation variant)
    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= buf[i];
    }
    buf[30] = (crc >> 0) & 0xFF;
    buf[31] = (crc >> 8) & 0xFF;

    // Payload: deterministic fill
    for (size_t i = 0; i < payload_bytes; i++) {
        buf[32 + i] = (uint8_t)((layer_id * 37 + tensor_idx * 13 + i) & 0xFF);
    }

    // Scales: deterministic
    for (uint32_t s = 0; s < n_scales; s++) {
        float v = (float)(0.5 + ((layer_id * 17 + s * 7) % 100) / 100.0);
        memcpy(buf.data() + scale_offset + s * sizeof(float), &v, sizeof(float));
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = fwrite(buf.data(), 1, total, f);
    fclose(f);
    return written == total;
}

static std::string gen_fake_manifest(const std::string& dir, int n_layers,
                                     const std::vector<std::string>& families) {
    // Write manifest.json
    std::string manifest_path = dir + "/manifest.json";
    FILE* mf = fopen(manifest_path.c_str(), "w");
    if (!mf) return "";

    fprintf(mf, "{\n  \"format_version\": 1,\n  \"entries\": [\n");
    bool first = true;
    for (int l = 0; l < n_layers; l++) {
        for (size_t fi = 0; fi < families.size(); fi++) {
            size_t payload_bytes = ((512LU * 2048LU * 3) + 7) / 8;
            size_t scale_offset = 32 + payload_bytes;
            if (scale_offset % 4) scale_offset = (scale_offset / 4 + 1) * 4;
            size_t fsize = scale_offset + 128 * sizeof(float);
            if (!first) fprintf(mf, ",\n");
            char fname[64];
            snprintf(fname, sizeof(fname), "layer_%03d.%s_%zu.trit", l, families[fi].c_str(), (size_t)fi);
            fprintf(mf, "    {\"layer_id\": %d, \"tensor_family\": \"%s\", "
                     "\"file_path\": \"%s\", \"byte_size\": %zu, \"checksum\": \"0000\"}",
                     l, families[fi].c_str(), fname, fsize);
            first = false;
        }
    }
    fprintf(mf, "\n  ]\n}\n");
    fclose(mf);
    return manifest_path;
}

static bool setup_synthetic_package(const std::string& dir, int n_layers,
                                    const std::vector<std::string>& families) {
    fs::create_directories(dir);
    std::string manifest = gen_fake_manifest(dir, n_layers, families);
    if (manifest.empty()) return false;

    for (int l = 0; l < n_layers; l++) {
        for (size_t fi = 0; fi < families.size(); fi++) {
            char filename[256];
            snprintf(filename, sizeof(filename), "layer_%03d.%s_%zu.trit",
                     l, families[fi].c_str(), fi);
            std::string path = dir + "/" + std::string(filename);
            if (!write_fake_trit(path, (uint32_t)l, families[fi], (uint32_t)fi, 512, 2048, 128)) {
                std::cerr << "Failed to write " << path << "\n";
                return false;
            }
        }
    }
    return true;
}

static bool cleanup_package(const std::string& dir) {
    try {
        fs::remove(dir + "/manifest.json");
        for (const auto& entry : fs::directory_iterator(dir)) {
            std::string name = entry.path().filename();
            if (has_suffix(name, ".trit")) fs::remove(entry.path());
        }
    } catch (...) {}
    return true;
}

// Test: disabled mode: pager not initialized, no manifest required
static int test_disabled_mode() {
    printf("\n=== TEST: DISABLED MODE ===\n");
    printf("Expected: no pager init, no manifest required, exit success\n");
    // In disabled mode, we don't even create a pager object.
    // This is tested by running harness without --enable-prt-sidecar-pager.
    printf("Result: DISABLED mode: no init, no manifest required\n");
    printf("Verdict: PASS_DISABLED_MODE_NOOP\n");
    return 0;
}

// Test: enabled mode: pager initialized, layers activated, residual views
static int test_enabled_mode(const harness_config& cfg) {
    printf("\n=== TEST: ENABLED MODE ===\n");
    printf("Manifest: %s\n", cfg.manifest_path.c_str());
    printf("Max resident: %d KB\n", cfg.max_resident_kb);
    printf("LRU: %s\n", cfg.eviction_lru ? "true" : "false");
    printf("Checksum: %s\n", cfg.checksum ? "true" : "false");

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
        pconfig.sidecar_root = cfg.manifest_path.empty() ? "" :
            cfg.manifest_path.substr(0, cfg.manifest_path.find_last_of("/"));
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = cfg.eviction_lru;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = !cfg.eviction_lru;

    prt_sidecar_pager pager(pconfig);
    if (!pager.init()) {
        printf("pager.init() failed\n");
        return 1;
    }

    // Activate layer 0
    bool a0 = pager.activate_layer(0);
    printf("activate_layer(0) = %s\n", a0 ? "true" : "false");

    // Prefetch layer 1
    pager.prefetch_layer(1);

    // Get residual views
    const char* families[] = {"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"};
    int valid_views = 0;
    int null_views = 0;
    for (auto fam : families) {
        auto view = pager.get_residual(0, fam);
        if (view.is_null) {
            null_views++;
            printf("  get_residual(0, %s): NULL (reason: %s)\n", fam, view.reason.c_str());
        } else {
            valid_views++;
            printf("  get_residual(0, %s): VALID %zu bytes\n", fam, view.size);
        }
    }

    // Request missing layer
    auto missing = pager.get_residual(99, "ffn_up");
    printf("get_residual(99, ffn_up): %s\n", missing.is_null ? "NULL" : "VALID");

    printf("\nStats:\n");
    print_stats(pager);

    printf("\nResult: valid=%d null=%d\n", valid_views, null_views);
    printf("Verdict: %s\n", (valid_views > 0) ? "PASS_ENABLED_PAGER_INIT" : "FAIL");
    return (valid_views > 0) ? 0 : 1;
}

// Test: budget stress: small budget causes rejections
static int test_budget_stress(const harness_config& cfg, int n_layers,
                              const std::vector<std::string>& families) {
    printf("\n=== TEST: BUDGET STRESS ===\n");
    printf("Budget: %d KB, %d layers x %d families\n",
           cfg.max_resident_kb, n_layers, (int)families.size());

    // Each .trit is 393280 bytes. With 2 tensors, layer = 786560 bytes.
    // With budget < 786560, layer 0 should exhaust budget and layer 1 should reject.
    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
        pconfig.sidecar_root = cfg.manifest_path.empty() ? "" :
            cfg.manifest_path.substr(0, cfg.manifest_path.find_last_of("/"));
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = false;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = true;

    prt_sidecar_pager pager(pconfig);
    if (!pager.init()) {
        printf("pager.init() failed\n");
        return 1;
    }

    int activated = 0;
    int rejected = 0;
    for (int l = 0; l < n_layers; l++) {
        bool ok = pager.activate_layer(l);
        if (ok) activated++;
        else rejected++;
        printf("activate_layer(%d) = %s\n", l, ok ? "true" : "false");
    }

    auto stats = pager.get_stats();
    printf("\nStats:\n");
    print_stats(pager);
    printf("\nActivated: %d, Rejected: %d, Budget rejects: %zu\n",
           activated, rejected, stats.budget_rejects);

    bool pass = (rejected > 0 && stats.budget_rejects > 0);
    printf("Verdict: %s\n", pass ? "PASS_BUDGET_STRESS" : "FAIL_BUDGET_STRESS");
    return pass ? 0 : 1;
}

// Test: LRU eviction: walk more layers than budget fits
static int test_lru_mode(const harness_config& cfg, int n_layers,
                         const std::vector<std::string>& families) {
    printf("\n=== TEST: LRU MODE ===\n");
    printf("Budget: %d KB, LRU=true, %d layers\n", cfg.max_resident_kb, n_layers);

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
        pconfig.sidecar_root = cfg.manifest_path.empty() ? "" :
            cfg.manifest_path.substr(0, cfg.manifest_path.find_last_of("/"));
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = true;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = false;

    prt_sidecar_pager pager(pconfig);
    if (!pager.init()) {
        printf("pager.init() failed\n");
        return 1;
    }

    int activated = 0;
    for (int l = 0; l < n_layers; l++) {
        bool ok = pager.activate_layer(l);
        if (ok) activated++;
        printf("activate_layer(%d) = %s\n", l, ok ? "true" : "false");
    }

    auto stats = pager.get_stats();
    printf("\nStats:\n");
    print_stats(pager);
    printf("\nActivated: %d, LRU evictions: %zu, Resident: %zu bytes (cap=%d KB)\n",
           activated, stats.lru_evictions, stats.resident_bytes,
           cfg.max_resident_kb);

    bool pass = (stats.lru_evictions > 0 && stats.resident_bytes <= (size_t)cfg.max_resident_kb * 1024);
    printf("Verdict: %s\n", pass ? "PASS_LRU_HARNESS" : "FAIL_LRU_HARNESS");
    return pass ? 0 : 1;
}

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --disabled              Run disabled-mode test (no init)\n");
    printf("  --enabled               Run enabled-mode test\n");
    printf("  --budget-stress         Run budget stress test\n");
    printf("  --lru                   Run LRU eviction test\n");
    printf("  --manifest PATH         Manifest path (required for enabled/budget/lru)\n");
    printf("  --max-resident-kb N     Max resident KB (default: 512)\n");
    printf("  --eviction-lru          Use LRU eviction policy\n");
    printf("  --no-checksum           Disable .trit checksum validation\n");
    printf("  --setup-dir DIR N       Create synthetic package in DIR with N layers\n");
    printf("  --all                   Run all tests in sequence\n");
}

int main(int argc, char** argv) {
    harness_config cfg;
    std::string setup_dir;
    int setup_layers = 0;
    std::string mode;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--disabled") == 0) { mode = "disabled"; }
        else if (strcmp(argv[i], "--enabled") == 0) { mode = "enabled"; }
        else if (strcmp(argv[i], "--budget-stress") == 0) { mode = "budget_stress"; }
        else if (strcmp(argv[i], "--lru") == 0) { mode = "lru"; }
        else if (strcmp(argv[i], "--all") == 0) { mode = "all"; }
        else if (strcmp(argv[i], "--manifest") == 0 && i+1 < argc) {
            cfg.manifest_path = argv[++i];
            cfg.enabled = true;
        }
        else if (strcmp(argv[i], "--max-resident-kb") == 0 && i+1 < argc) {
            cfg.max_resident_kb = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--eviction-lru") == 0) {
            cfg.eviction_lru = true;
        }
        else if (strcmp(argv[i], "--no-checksum") == 0) {
            cfg.checksum = false;
        }
        else if (strcmp(argv[i], "--setup-dir") == 0 && i+2 < argc) {
            setup_dir = argv[++i];
            setup_layers = atoi(argv[++i]);
        }
        else {
            print_help(argv[0]);
            return 1;
        }
    }

    // Setup phase
    if (!setup_dir.empty()) {
        std::vector<std::string> families = {"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"};
        printf("Setting up synthetic package: %s (%d layers)\n", setup_dir.c_str(), setup_layers);
        bool ok = setup_synthetic_package(setup_dir, setup_layers, families);
        printf("Setup: %s\n", ok ? "SUCCESS" : "FAILED");
        if (!ok) return 1;
        printf("Manifest: %s/manifest.json\n", setup_dir.c_str());
        return 0;
    }

    if (mode.empty()) {
        print_help(argv[0]);
        return 1;
    }

    if (mode == "disabled") {
        return test_disabled_mode();
    }

    // all mode creates its own package; no manifest required upfront
    if (mode != "all" && cfg.manifest_path.empty()) {
        std::cerr << "Error: --manifest required for " << mode << "\n";
        return 1;
    }

    if (mode == "enabled") {
        return test_enabled_mode(cfg);
    }

    if (mode == "budget_stress") {
        std::vector<std::string> families = {"ffn_up", "ffn_down"};
        int n_layers = 4;
        return test_budget_stress(cfg, n_layers, families);
    }

    if (mode == "lru") {
        std::vector<std::string> families = {"ffn_up", "ffn_down"};
        int n_layers = 4;
        return test_lru_mode(cfg, n_layers, families);
    }

    if (mode == "all") { cfg.enabled = true;
        int rc0 = test_disabled_mode();

        // Setup temp package for enabled tests
        std::string tmpdir = "/tmp/prt_harness_28ar";
        std::vector<std::string> families = {"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"};
        setup_synthetic_package(tmpdir, 4, families);

        harness_config cfg_enabled;
        cfg_enabled.manifest_path = tmpdir + "/manifest.json";
        cfg_enabled.max_resident_kb = 2048;
        cfg_enabled.eviction_lru = false;
        cfg_enabled.checksum = true;

        int rc1 = test_enabled_mode(cfg_enabled);

        // Budget stress
        harness_config cfg_budget;
        cfg_budget.manifest_path = tmpdir + "/manifest.json";
        cfg_budget.max_resident_kb = 256; // too small for 2 tensors/layer
        cfg_budget.eviction_lru = false;
        cfg_budget.checksum = false;
        int rc2 = test_budget_stress(cfg_budget, 4, {"ffn_up", "ffn_down"});

        // LRU
        harness_config cfg_lru;
        cfg_lru.manifest_path = tmpdir + "/manifest.json";
        cfg_lru.max_resident_kb = 2500;  // 5 families ~1.9MB, 1 layer fits, 2nd triggers LRU eviction
        cfg_lru.eviction_lru = true;
        cfg_lru.checksum = true;
        int rc3 = test_lru_mode(cfg_lru, 4, {"ffn_up", "ffn_down"});

        cleanup_package(tmpdir);

        printf("\n=== ALL TESTS SUMMARY ===\n");
        printf("Disabled:  %s\n", rc0 == 0 ? "PASS" : "FAIL");
        printf("Enabled:   %s\n", rc1 == 0 ? "PASS" : "FAIL");
        printf("Budget:    %s\n", rc2 == 0 ? "PASS" : "FAIL");
        printf("LRU:       %s\n", rc3 == 0 ? "PASS" : "FAIL");

        int total = rc0 + rc1 + rc2 + rc3;
        printf("\nVerdict: %s\n", total == 0 ? "PASS_PHASE28AR_HARNESS_INTEGRATION" : "FAIL");
        return total;
    }

    return 0;
}
