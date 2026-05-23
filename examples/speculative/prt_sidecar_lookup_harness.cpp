// Phase 28AS: Sidecar Pager Lookup Harness
// Tests the prt_get_residual_view() wrapper that routes between pager and legacy paths.
// No model generation, no ggml graph, no active matmul changes.

#include "prt_sidecar_pager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace fs = std::filesystem;

// ── Wrapper under test ──────────────────────────────────────────────────────

// Simulates the proposed prt_get_residual_view() wrapper.
// In the actual integration this would live in prt_shadow.h or a helper file.
// Here we test the routing logic standalone.

struct legacy_sidecar_data {
    const float* data = nullptr;
    size_t size = 0;
    bool loaded = false;
};

static legacy_sidecar_data g_legacy_sidecars[32]; // simulated legacy map
static bool g_legacy_loaded = false;

static prt_residual_view wrapper_get_residual(
    int layer,
    const std::string& tensor_family,
    bool pager_enabled,
    prt_sidecar_pager* pager,
    bool& used_pager,
    bool& used_legacy
) {
    used_pager = false;
    used_legacy = false;

    if (pager_enabled && pager != nullptr) {
        auto view = pager->get_residual(layer, tensor_family);
        if (!view.is_null) {
            used_pager = true;
            return view;
        }
        // fall through to legacy
    }

    // Legacy path
    if (layer >= 0 && layer < 32 && g_legacy_sidecars[layer].loaded) {
        used_legacy = true;
        prt_residual_view v;
        v.data = (const uint8_t*)g_legacy_sidecars[layer].data;
        v.size = g_legacy_sidecars[layer].size;
        v.is_null = false;
        v.reason = "legacy";
        return v;
    }

    prt_residual_view null_view;
    null_view.is_null = true;
    null_view.reason = "not_found";
    return null_view;
}

// ── Harness config ───────────────────────────────────────────────────────────

struct harness_config {
    std::string mode; // "legacy", "pager", "fallback", "budget"
    std::string manifest_path;
    std::string sidecar_root;
    int max_resident_kb = 512;
    bool pager_enabled = false;
    bool eviction_lru = false;
    bool checksum = true;
};

static bool has_suffix(const std::string& s, const char* suffix) {
    size_t sl = strlen(suffix);
    return s.size() >= sl && s.compare(s.size() - sl, sl, suffix) == 0;
}

// ── Synthetic .trit generation ───────────────────────────────────────────────

static bool write_fake_trit(const std::string& path, uint32_t layer_id,
                            const std::string& tensor_family, uint32_t tensor_idx) {
    uint32_t n_rows = 512, n_cols = 2048, n_scales = 128;
    size_t payload_bytes = (n_rows * n_cols * 3) / 8;
    size_t total = 32 + payload_bytes + n_scales * 2;

    std::vector<unsigned char> buf(total, 0);
    buf[0] = 0x54; buf[1] = 0x49; buf[2] = 0x52; buf[3] = 0x52;
    buf[4] = 0x00; buf[5] = 0x01;
    buf[6] = (n_rows >> 0) & 0xFF; buf[7] = (n_rows >> 8) & 0xFF;
    buf[8] = (n_cols >> 0) & 0xFF; buf[9] = (n_cols >> 8) & 0xFF;
    buf[10] = (n_scales >> 0) & 0xFF; buf[11] = (n_scales >> 8) & 0xFF;

    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= buf[i];
    }
    buf[30] = (crc >> 0) & 0xFF;
    buf[31] = (crc >> 8) & 0xFF;

    for (size_t i = 0; i < payload_bytes; i++)
        buf[32 + i] = (uint8_t)((layer_id * 37 + tensor_idx * 13 + i) & 0xFF);
    for (uint32_t s = 0; s < n_scales; s++) {
        uint16_t v = (uint16_t)((layer_id * 17 + s * 7) & 0xFFFF);
        buf[32 + payload_bytes + s*2 + 0] = (v >> 0) & 0xFF;
        buf[32 + payload_bytes + s*2 + 1] = (v >> 8) & 0xFF;
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = fwrite(buf.data(), 1, total, f);
    fclose(f);
    return written == total;
}

static bool setup_synthetic_package(const std::string& dir, int n_layers,
                                   const std::vector<std::string>& families) {
    fs::create_directories(dir);
    FILE* mf = fopen((dir + "/manifest.json").c_str(), "w");
    if (!mf) return false;
    fprintf(mf, "{\n  \"format_version\": 1,\n  \"entries\": [\n");
    bool first = true;
    for (int l = 0; l < n_layers; l++) {
        for (size_t fi = 0; fi < families.size(); fi++) {
            size_t fsize = 32 + ((512LU * 2048LU * 3) / 8) + (128 * 2);
            char fname[64];
            snprintf(fname, sizeof(fname), "layer_%03d.%s_%zu.trit", l, families[fi].c_str(), fi);
            std::string path = dir + "/" + std::string(fname);
            write_fake_trit(path, (uint32_t)l, families[fi], (uint32_t)fi);
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

// ── Test: legacy mode (pager disabled) ───────────────────────────────────────

static int test_legacy_mode() {
    printf("\n=== TEST: LEGACY MODE (pager disabled) ===\n");
    bool used_pager = false, used_legacy = false;

    // Simulate legacy loaded
    g_legacy_sidecars[0].loaded = true;
    g_legacy_sidecars[99].loaded = true;
    g_legacy_sidecars[99].data = (const float*)0x9999;
    g_legacy_sidecars[99].size = 99999;
    g_legacy_sidecars[0].data = (const float*)0x1234; // fake pointer
    g_legacy_sidecars[0].size = 12345;

    auto view = wrapper_get_residual(0, "ffn_up", false, nullptr, used_pager, used_legacy);
    printf("get_residual(0, ffn_up): is_null=%s, reason=%s\n",
           view.is_null ? "true" : "false", view.reason.c_str());
    printf("used_pager=%s, used_legacy=%s\n",
           used_pager ? "true" : "false", used_legacy ? "true" : "false");

    // Request layer with no legacy data
    auto view2 = wrapper_get_residual(5, "ffn_up", false, nullptr, used_pager, used_legacy);
    printf("get_residual(5, ffn_up): is_null=%s, reason=%s\n",
           view2.is_null ? "true" : "false", view2.reason.c_str());

    printf("DEBUG: used_legacy=%s view.is_null=%s view.reason=%s\n",
           used_legacy?"true":"false", view.is_null?"true":"false", view.reason.c_str());
    // Pass if we got a non-null view from the legacy path
    bool pass = (view.reason == "legacy") && !view.is_null;
    printf("Verdict: %s\n", pass ? "PASS_LEGACY_MODE ✅" : "FAIL_LEGACY_MODE");
    return pass ? 0 : 1;
}

// ── Test: pager mode (pager enabled, working) ───────────────────────────────

static int test_pager_mode(const harness_config& cfg) {
    printf("\n=== TEST: PAGER MODE (pager enabled, synthetic manifest) ===\n");
    printf("Manifest: %s\n", cfg.manifest_path.c_str());
    printf("Max resident: %d KB\n", cfg.max_resident_kb);
    printf("Pager enabled: %s\n", cfg.pager_enabled ? "true" : "false");

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = cfg.eviction_lru;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = !cfg.eviction_lru;

    prt_sidecar_pager pager(pconfig);
    bool init_ok = pager.init();
    printf("pager.init() = %s\n", init_ok ? "true" : "false");

    if (!init_ok) {
        printf("Pager init failed — cannot test pager mode\n");
        return 1;
    }

    pager.activate_layer(0);
    pager.get_residual(0, "ffn_up"); // prime cache

    bool used_pager = false, used_legacy = false;
    auto view = wrapper_get_residual(0, "ffn_up", true, &pager, used_pager, used_legacy);

    printf("get_residual(0, ffn_up): is_null=%s, size=%zu, reason=%s\n",
           view.is_null ? "true" : "false", view.size, view.reason.c_str());
    printf("used_pager=%s, used_legacy=%s\n",
           used_pager ? "true" : "false", used_legacy ? "true" : "false");

    auto stats = pager.get_stats();
    printf("pager stats: resident=%zu, reads=%zu, misses=%zu\n",
           stats.resident_bytes, stats.reads, stats.cache_misses);

    bool pass = used_pager && !view.is_null;
    printf("Verdict: %s\n", pass ? "PASS_PAGER_MODE ✅" : "FAIL_PAGER_MODE");
    return pass ? 0 : 1;
}

// ── Test: fallback (pager enabled, tensor not in pager, fall through to legacy) ─

static int test_fallback_mode(const harness_config& cfg) {
    printf("\n=== TEST: FALLBACK (pager returns null for unknown layer, routes correctly) ===\n");

    // Setup: pager with synthetic manifest (only layers 0-3)
    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = cfg.eviction_lru;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = !cfg.eviction_lru;

    prt_sidecar_pager pager(pconfig);
    bool init_ok = pager.init();
    printf("pager.init() = %s\n", init_ok ? "true" : "false");

    pager.activate_layer(0);  // prime layer 0

    // Test 1: layer 99 not in manifest → pager returns null
    bool used_pager = false, used_legacy = false;
    auto view_pager_miss = wrapper_get_residual(99, "ffn_up", true, &pager, used_pager, used_legacy);
    printf("get_residual(99, ffn_up): is_null=%s, used_pager=%s, used_legacy=%s\n",
           view_pager_miss.is_null ? "true" : "false",
           used_pager ? "true" : "false", used_legacy ? "true" : "false");

    bool pass_pager_null = view_pager_miss.is_null && view_pager_miss.reason == "not_found";
    printf("Pager returns null for unknown layer: %s\n", pass_pager_null ? "PASS ✅" : "FAIL");

    // Test 2: with pager disabled, unknown layer returns null
    used_pager = used_legacy = false;
    auto view_disabled = wrapper_get_residual(99, "ffn_up", false, nullptr, used_pager, used_legacy);
    printf("get_residual(99, ffn_up) [pager disabled]: is_null=%s, used_pager=%s, used_legacy=%s\n",
           view_disabled.is_null ? "true" : "false",
           used_pager ? "true" : "false", used_legacy ? "true" : "false");

    bool pass_disabled_null = view_disabled.is_null && view_disabled.reason == "not_found";
    printf("Pager disabled returns null for unknown layer: %s\n", pass_disabled_null ? "PASS ✅" : "FAIL");

    // Test 3: known layer via pager returns valid view
    used_pager = used_legacy = false;
    auto view_known = wrapper_get_residual(0, "ffn_up", true, &pager, used_pager, used_legacy);
    printf("get_residual(0, ffn_up): is_null=%s, size=%zu, used_pager=%s\n",
           view_known.is_null ? "true" : "false", view_known.size, used_pager ? "true" : "false");

    bool pass_known_valid = !view_known.is_null && used_pager;
    printf("Known layer via pager returns valid view: %s\n", pass_known_valid ? "PASS ✅" : "FAIL");

    bool pass = pass_pager_null && pass_disabled_null && pass_known_valid;
    printf("Verdict: %s\n", pass ? "PASS_FALLBACK_ROUTING ✅" : "FAIL_FALLBACK_ROUTING");
    return pass ? 0 : 1;
}

// ── Test: budget (pager enabled, small budget, reject/evict) ─────────────────

static int test_budget_mode(const harness_config& cfg) {
    printf("\n=== TEST: BUDGET (tiny budget, pager rejects) ===\n");
    printf("Budget: %d KB, pager=%s\n", cfg.max_resident_kb, cfg.pager_enabled ? "true" : "false");

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = cfg.eviction_lru;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = !cfg.eviction_lru;

    prt_sidecar_pager pager(pconfig);
    bool init_ok = pager.init();
    if (!init_ok) { printf("pager init failed\n"); return 1; }

    bool a0 = pager.activate_layer(0);
    bool a1 = pager.activate_layer(1);
    bool a2 = pager.activate_layer(2);
    bool a3 = pager.activate_layer(3);

    auto stats = pager.get_stats();
    printf("activate: layer0=%s layer1=%s layer2=%s layer3=%s\n",
           a0?"true":"false", a1?"true":"false", a2?"true":"false", a3?"true":"false");
    printf("resident=%zu, budget_rejects=%zu, lru_evictions=%zu\n",
           stats.resident_bytes, stats.budget_rejects, stats.lru_evictions);

    bool pass = (stats.budget_rejects > 0 || stats.lru_evictions > 0);
    printf("Verdict: %s\n", pass ? "PASS_BUDGET_BEHAVIOR ✅" : "FAIL_BUDGET_BEHAVIOR");
    return pass ? 0 : 1;
}

// ── Main ───────────────────────────────────────────────────────────────────────

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --legacy              Run legacy-mode test\n");
    printf("  --pager               Run pager-mode test\n");
    printf("  --fallback            Run fallback test\n");
    printf("  --budget             Run budget test\n");
    printf("  --all                 Run all tests\n");
    printf("  --manifest PATH       Manifest path (for pager/fallback/budget)\n");
    printf("  --sidecar-root DIR    Sidecar root dir (default: derived from manifest)\n");
    printf("  --max-resident-kb N   Max resident KB\n");
    printf("  --eviction-lru        Use LRU eviction\n");
    printf("  --no-checksum         Disable checksum validation\n");
    printf("  --setup-dir DIR N     Setup synthetic package with N layers\n");
}

int main(int argc, char** argv) {
    harness_config cfg;
    std::string mode;
    std::string setup_dir;
    int setup_layers = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--legacy") == 0) { mode = "legacy"; }
        else if (strcmp(argv[i], "--pager") == 0) { mode = "pager"; cfg.pager_enabled = true; }
        else if (strcmp(argv[i], "--fallback") == 0) { mode = "fallback"; cfg.pager_enabled = true; }
        else if (strcmp(argv[i], "--budget") == 0) { mode = "budget"; cfg.pager_enabled = true; }
        else if (strcmp(argv[i], "--all") == 0) { mode = "all"; cfg.pager_enabled = true; }
        else if (strcmp(argv[i], "--manifest") == 0 && i+1 < argc)
            { cfg.manifest_path = argv[++i]; }
        else if (strcmp(argv[i], "--sidecar-root") == 0 && i+1 < argc)
            { cfg.sidecar_root = argv[++i]; }
        else if (strcmp(argv[i], "--max-resident-kb") == 0 && i+1 < argc)
            { cfg.max_resident_kb = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--eviction-lru") == 0)
            { cfg.eviction_lru = true; }
        else if (strcmp(argv[i], "--no-checksum") == 0)
            { cfg.checksum = false; }
        else if (strcmp(argv[i], "--setup-dir") == 0 && i+2 < argc)
            { setup_dir = argv[++i]; setup_layers = atoi(argv[++i]); }
        else { print_help(argv[0]); return 1; }
    }

    if (!setup_dir.empty()) {
        std::vector<std::string> families = {"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"};
        printf("Setup: %s with %d layers\n", setup_dir.c_str(), setup_layers);
        bool ok = setup_synthetic_package(setup_dir, setup_layers, families);
        printf("Setup: %s\n", ok ? "SUCCESS" : "FAILED");
        return ok ? 0 : 1;
    }

    if (mode.empty()) { print_help(argv[0]); return 1; }

    if (mode == "legacy") return test_legacy_mode();

    if ((mode == "pager" || mode == "fallback" || mode == "budget") && cfg.manifest_path.empty()) {
        std::cerr << "Error: --manifest required for " << mode << "\n";
        return 1;
    }

    if (mode == "pager") return test_pager_mode(cfg);
    if (mode == "fallback") return test_fallback_mode(cfg);
    if (mode == "budget") return test_budget_mode(cfg);

    if (mode == "all") {
        int rc0 = test_legacy_mode();

        // Setup synthetic package
        std::string tmpdir = "/tmp/prt_lookup_28as";
        setup_synthetic_package(tmpdir, 4, {"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"});

        harness_config cfg_pager;
        cfg_pager.manifest_path = tmpdir + "/manifest.json";
        cfg_pager.sidecar_root = tmpdir;
        cfg_pager.max_resident_kb = 2048;
        cfg_pager.pager_enabled = true;
        cfg_pager.eviction_lru = false;
        cfg_pager.checksum = false;
        int rc1 = test_pager_mode(cfg_pager);

        harness_config cfg_fallback;
        cfg_fallback.manifest_path = tmpdir + "/manifest.json";
        cfg_fallback.sidecar_root = tmpdir;
        cfg_fallback.max_resident_kb = 2048;
        cfg_fallback.pager_enabled = true;
        cfg_fallback.eviction_lru = false;
        cfg_fallback.checksum = false;
        int rc2 = test_fallback_mode(cfg_fallback);

        harness_config cfg_budget;
        cfg_budget.manifest_path = tmpdir + "/manifest.json";
        cfg_budget.sidecar_root = tmpdir;
        cfg_budget.max_resident_kb = 256;
        cfg_budget.pager_enabled = true;
        cfg_budget.eviction_lru = false;
        cfg_budget.checksum = false;
        int rc3 = test_budget_mode(cfg_budget);

        printf("\n=== ALL TESTS SUMMARY ===\n");
        printf("Legacy:  %s\n", rc0==0?"PASS":"FAIL");
        printf("Pager:   %s\n", rc1==0?"PASS":"FAIL");
        printf("Fallback:%s\n", rc2==0?"PASS":"FAIL");
        printf("Budget:  %s\n", rc3==0?"PASS":"FAIL");
        int total = rc0+rc1+rc2+rc3;
        printf("\nVerdict: %s\n", total==0 ? "PASS_PHASE28AS_LOOKUP_INTEGRATION ✅" : "FAIL");
        return total;
    }

    return 0;
}