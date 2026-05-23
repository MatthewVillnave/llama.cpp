// Phase 28AU: Runtime Link Harness
// Tests prt_get_residual_view() routing: pager vs legacy, enabled vs disabled.
// No generation, no ggml graph.

// prt_shadow.h must be included BEFORE prt_sidecar_pager.h
// so that SidecarLoad is fully defined when g_sidecars is used
#include "prt_shadow.h"
#include "prt_sidecar_pager.h"
#include "prt_sidecar_runtime_link.h"

// Definitions for globals declared extern in prt_sidecar_runtime_link.h
prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// ── G_SIDECARS MOCK (minimal, mirrors prt_shadow.h interface) ────────────────

// SidecarLoad is defined in prt_shadow.h
// Legacy sidecar map: layer 0 only, 1234 bytes of mock data
static uint8_t g_legacy_data[1234] = {0};

// Override g_sidecars for harness: only layer 0
static SidecarLoad g_legacy_sidecar_0 = {0, "/mock/legacy/sidecar_0.bin", (float*)g_legacy_data, 1234};
std::unordered_map<int, SidecarLoad> g_sidecars = {{0, g_legacy_sidecar_0}};  // override weak def in prt_shadow.h

// g_prt_pager and g_prt_pager_enabled declared extern in prt_sidecar_runtime_link.h
// No need to redeclare here

// Required by prt_shadow.h (minimal stub for linking)
bool g_sidecars_loaded_flag() { return g_sidecars_loaded; }

// ── Config ────────────────────────────────────────────────────────────────────

struct link_config {
    bool pager_enabled = false;
    std::string manifest_path;
    std::string sidecar_root;
    int budget_kb = 2048;
    int layers = 4;
    bool lru = false;
    bool cleanup = false;
    std::vector<std::string> families{"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"};
};

// ── Synthetic .trit package ──────────────────────────────────────────────────

static bool write_trit(const std::string& path, uint32_t layer_id,
                       const std::string& family, uint32_t idx) {
    uint32_t rows = 512, cols = 2048, n_scales = 128;
    size_t payload = (rows * cols * 3) / 8;
    size_t total = 32 + payload + n_scales * 2;
    std::vector<unsigned char> buf(total, 0);
    // Magic "TRIT" = 0x54495254 in LE
    buf[0] = 0x54; buf[1] = 0x52; buf[2] = 0x49; buf[3] = 0x54;
    buf[4] = 0x00; buf[5] = 0x01;
    buf[6] = (rows >> 0) & 0xFF; buf[7] = (rows >> 8) & 0xFF;
    buf[8] = (cols >> 0) & 0xFF; buf[9] = (cols >> 8) & 0xFF;
    buf[10] = (n_scales >> 0) & 0xFF; buf[11] = (n_scales >> 8) & 0xFF;
    buf[12] = 0; buf[13] = 0;  // block_rows=0
    buf[14] = 0; buf[15] = 0;  // block_cols=0
    buf[16] = 32 & 0xFF; buf[17] = (32 >> 8) & 0xFF; buf[18] = (32 >> 16) & 0xFF; buf[19] = (32 >> 24) & 0xFF;
    size_t scale_off = 32 + payload;
    buf[20] = scale_off & 0xFF; buf[21] = (scale_off >> 8) & 0xFF;
    buf[22] = (scale_off >> 16) & 0xFF; buf[23] = (scale_off >> 24) & 0xFF;
    // CRC over bytes 0-29
    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= buf[i];
    }
    buf[30] = (crc >> 0) & 0xFF; buf[31] = (crc >> 8) & 0xFF;
    for (size_t i = 0; i < payload; i++)
        buf[32 + i] = (uint8_t)((layer_id * 37 + idx * 13 + i) & 0xFF);
    for (uint32_t s = 0; s < n_scales; s++) {
        uint16_t v = (uint16_t)((layer_id * 17 + s * 7) & 0xFFFF);
        buf[32 + payload + s*2 + 0] = (v >> 0) & 0xFF;
        buf[32 + payload + s*2 + 1] = (v >> 8) & 0xFF;
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(buf.data(), 1, total, f);
    fclose(f);
    return w == total;
}

static bool setup_package(const std::string& dir, int n_layers,
                          const std::vector<std::string>& families) {
    fs::create_directories(dir);
    std::string manifest_path = dir + "/manifest.json";
    FILE* mf = fopen(manifest_path.c_str(), "w");
    if (!mf) { fprintf(stderr, "ERROR: fopen(\%s\") failed: %s\n", manifest_path.c_str(), strerror(errno)); return false; }
    fprintf(mf, "{\n  \"format_version\": 1,\n  \"entries\": [\n");
    bool first = true;
    for (int l = 0; l < n_layers; l++) {
        for (size_t fi = 0; fi < families.size(); fi++) {
            size_t fsize = 32 + ((512LU * 2048LU * 3) / 8) + (128 * 2);
            char fname[64];
            snprintf(fname, sizeof(fname), "layer_%03d.%s_%zu.trit", l, families[fi].c_str(), fi);
            std::string path = dir + "/" + std::string(fname);
            write_trit(path, (uint32_t)l, families[fi], (uint32_t)fi);
            if (!first) fprintf(mf, ",\n");
            fprintf(mf, "    {\"layer_id\": %d, \"tensor_family\": \"%s\", "
                     "\"file_path\": \"%s\", \"byte_size\": %zu, \"checksum\": \"0000\"}",
                     l, families[fi].c_str(), fname, fsize);
            first = false;
        }
    }
    fprintf(mf, "\n  ]\n}\n");
    fclose(mf);
    return true;
}

static void cleanup_package(const std::string& dir) {
    try {
        for (const auto& e : fs::directory_iterator(dir)) {
            fs::remove(e.path());
        }
        fs::remove(dir + "/manifest.json");
        fs::remove(dir);
    } catch (...) {}
}

// ── Test: Legacy Disabled ─────────────────────────────────────────────────────

static int test_legacy_disabled(const link_config& cfg) {
    printf("\n=== TEST: LEGACY DISABLED (pager NOT initialized) ===\n");

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
    // Explicitly ensure pager is off
    g_prt_pager_enabled = false;
    if (g_prt_pager != nullptr) {
        prt_shutdown_pager();
    }

    // Layer 0 exists in legacy map
    auto view = prt_get_residual_view(0, "ffn_up");
    printf("prt_get_residual_view(0, ffn_up): is_null=%s reason=%s\n",
           view.is_null ? "true" : "false", view.reason.c_str());

    bool pass = !view.is_null && view.reason == "legacy" && view.size == 1234;
    printf("Layer 0 via legacy: %s\n", pass ? "PASS" : "FAIL");

    // Layer 99 does NOT exist in legacy or pager
    auto view_missing = prt_get_residual_view(99, "ffn_up");
    printf("prt_get_residual_view(99, ffn_up): is_null=%s reason=%s\n",
           view_missing.is_null ? "true" : "false", view_missing.reason.c_str());
    bool pass2 = view_missing.is_null;
    printf("Missing layer: %s\n", pass2 ? "PASS" : "FAIL");

    printf("Result: %s\n", (pass && pass2) ? "PASS_LEGACY_DISABLED" : "FAIL_LEGACY_DISABLED");
    return (pass && pass2) ? 0 : 1;
#else
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined — stub only\n");
    printf("Result: PASS_LEGACY_DISABLED (stub mode, no runtime link)\n");
    (void)cfg;
    return 0;
#endif
}

// ── Test: Pager Enabled ──────────────────────────────────────────────────────

static int test_pager_enabled(const link_config& cfg) {
    printf("\n=== TEST: PAGER ENABLED ===\n");

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
    // Init pager
    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.budget_kb * 1024;
    pconfig.eviction_lru = cfg.lru;
    pconfig.checksum_enabled = false;
    pconfig.validate_trit_header = false;
    pconfig.strict_budget = !cfg.lru;

    bool ok = prt_init_pager(pconfig);
    printf("prt_init_pager() = %s\n", ok ? "true" : "false");
    if (!ok) {
        printf("FAIL: pager init failed (error=%d)\n", (int)g_prt_pager->last_error());
        return 1;
    }

    prt_sidecar_pager_stats stats = prt_get_pager_stats();
    printf("Pager stats: resident=%zu reads=%zu\n", stats.resident_bytes, stats.reads);

    // Activate layer 0
    g_prt_pager->activate_layer(0);

    // Pager has this tensor
    auto view = prt_get_residual_view(0, "ffn_up");
    printf("prt_get_residual_view(0, ffn_up): is_null=%s size=%zu reason=%s\n",
           view.is_null ? "true" : "false", view.size, view.reason.c_str());
    bool pass = !view.is_null && view.reason.empty();  // empty reason = pager hit

    printf("Result: %s\n", pass ? "PASS_PAGER_ENABLED" : "FAIL_PAGER_ENABLED");

    prt_shutdown_pager();
    return pass ? 0 : 1;
#else
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    printf("Result: FAIL_PAGER_ENABLED (stub only)\n");
    (void)cfg;
    return 1;
#endif
}

// ── Test: Missing Tensor Fallback ───────────────────────────────────────────

static int test_missing_fallback(const link_config& cfg) {
    printf("\n=== TEST: MISSING TENSOR FALLBACK ===\n");

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.budget_kb * 1024;
    pconfig.eviction_lru = cfg.lru;
    pconfig.checksum_enabled = false;
    pconfig.validate_trit_header = false;
    pconfig.strict_budget = !cfg.lru;

    prt_init_pager(pconfig);
    g_prt_pager->activate_layer(0);

    // Unknown family on known layer
    auto view_unknown = prt_get_residual_view(0, "nonexistent_family");
    printf("prt_get_residual_view(0, nonexistent): is_null=%s reason=%s\n",
           view_unknown.is_null ? "true" : "false", view_unknown.reason.c_str());
    bool pass_unknown = !view_unknown.is_null && view_unknown.reason == "legacy";

    // Unknown layer
    auto view_layer = prt_get_residual_view(99, "ffn_up");
    printf("prt_get_residual_view(99, ffn_up): is_null=%s reason=%s\n",
           view_layer.is_null ? "true" : "false", view_layer.reason.c_str());
    bool pass_layer = view_layer.is_null;

    printf("Result: %s\n", (pass_unknown && pass_layer) ? "PASS_MISSING_FALLBACK" : "FAIL_MISSING_FALLBACK");
    prt_shutdown_pager();
    return (pass_unknown && pass_layer) ? 0 : 1;
#else
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    return 1;
#endif
}

// ── Test: Budget Reject ──────────────────────────────────────────────────────

static int test_budget_reject(const link_config& cfg) {
    printf("\n=== TEST: BUDGET REJECT ===\n");

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
    // Very small budget
    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = 256 * 1024;  // 256 KB — too small
    pconfig.eviction_lru = false;
    pconfig.checksum_enabled = false;
    pconfig.validate_trit_header = false;
    pconfig.strict_budget = true;

    prt_init_pager(pconfig);

    // Try activating layers — should be rejected
    bool activated = g_prt_pager->activate_layer(0);
    printf("activate_layer(0) with 256KB budget: %s\n", activated ? "activated" : "REJECTED");

    prt_sidecar_pager_stats stats = prt_get_pager_stats();
    printf("budget_rejects = %zu\n", stats.budget_rejects);

    bool pass = !activated && stats.budget_rejects > 0;
    printf("Result: %s\n", pass ? "PASS_BUDGET_REJECT" : "FAIL_BUDGET_REJECT");
    prt_shutdown_pager();
    return pass ? 0 : 1;
#else
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    return 1;
#endif
}

// ── Main ─────────────────────────────────────────────────────────────────────

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --legacy          Test legacy disabled mode\n");
    printf("  --pager           Test pager enabled mode\n");
    printf("  --fallback        Test missing tensor fallback\n");
    printf("  --budget          Test budget reject\n");
    printf("  --all             Run all tests\n");
    printf("  --manifest PATH   Manifest path (required for pager tests)\n");
    printf("  --sidecar-root DIR  Sidecar root dir\n");
    printf("  --budget-kb N     Budget in KB (default: 2048)\n");
    printf("  --lru             Use LRU eviction\n");
    printf("  --setup-dir DIR N Setup synthetic package with N layers\n");
    printf("  --cleanup         Clean up package after\n");
}

int main(int argc, char** argv) {
    link_config cfg;
    std::string mode;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--legacy") == 0) mode = "legacy";
        else if (strcmp(argv[i], "--pager") == 0) mode = "pager";
        else if (strcmp(argv[i], "--fallback") == 0) mode = "fallback";
        else if (strcmp(argv[i], "--budget") == 0) mode = "budget";
        else if (strcmp(argv[i], "--all") == 0) mode = "all";
        else if (strcmp(argv[i], "--manifest") == 0 && i+1 < argc)
            { cfg.manifest_path = argv[++i]; }
        else if (strcmp(argv[i], "--sidecar-root") == 0 && i+1 < argc)
            { cfg.sidecar_root = argv[++i]; }
        else if (strcmp(argv[i], "--budget-kb") == 0 && i+1 < argc)
            { cfg.budget_kb = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--lru") == 0)
            { cfg.lru = true; }
        else if (strcmp(argv[i], "--cleanup") == 0)
            { cfg.cleanup = true; }
        else if (strcmp(argv[i], "--setup-dir") == 0 && i+2 < argc)
            { const char* dir = argv[++i]; int n = atoi(argv[++i]); printf("DEBUG: setup dir=%s n=%d\n", dir, n); setup_package(dir, n, cfg.families); return 0; }
        else { print_help(argv[0]); return 1; }
    }

    if (mode == "all") {
        // Always can run legacy test
        int rc0 = test_legacy_disabled(cfg);

        // Pager tests need manifest
        if (cfg.manifest_path.empty()) {
            fprintf(stderr, "Error: --manifest required for pager tests\n");
            return rc0 ? rc0 : 1;
        }

        std::string tmpdir = cfg.sidecar_root.empty() ? "/tmp/prt_link_28au" : cfg.sidecar_root;
        if (cfg.sidecar_root.empty() || !fs::exists(cfg.manifest_path)) {
            setup_package(tmpdir, 4, cfg.families);
            cfg.manifest_path = tmpdir + "/manifest.json";
            cfg.sidecar_root = tmpdir;
        }

        int rc1 = test_pager_enabled(cfg);
        int rc2 = test_missing_fallback(cfg);
        int rc3 = test_budget_reject(cfg);

        if (cfg.cleanup) cleanup_package(tmpdir);

        printf("\n=== SUMMARY ===\n");
        printf("Legacy:    %s\n", rc0==0?"PASS":"FAIL");
        printf("Pager:     %s\n", rc1==0?"PASS":"FAIL");
        printf("Fallback:  %s\n", rc2==0?"PASS":"FAIL");
        printf("Budget:    %s\n", rc3==0?"PASS":"FAIL");
        int total = rc0+rc1+rc2+rc3;
        printf("\nVerdict: %s\n", total==0 ? "PASS_PHASE28AU_RUNTIME_LINK_STUB" : "FAIL");
        return total;
    }

    if (mode == "legacy") return test_legacy_disabled(cfg);

    if (mode == "pager" || mode == "fallback" || mode == "budget") {
        if (cfg.manifest_path.empty()) {
            fprintf(stderr, "Error: --manifest required for %s mode\n", mode.c_str());
            return 1;
        }
    }

    if (mode == "pager") return test_pager_enabled(cfg);
    if (mode == "fallback") return test_missing_fallback(cfg);
    if (mode == "budget") return test_budget_reject(cfg);

    print_help(argv[0]);
    return 1;
}