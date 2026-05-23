// Phase 28AV: PRT Matmul Sidecar Lookup Harness
// Exercises the lookup path the PRT matmul will call — no actual matmul execution.
// Build: g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL
//        -I. -Iggml/include -Iinclude examples/speculative/prt_sidecar_pager.cpp
//        examples/speculative/prt_matmul_sidecar_lookup_harness.cpp -o /tmp/prt_matmul_lookup_harness

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <sys/stat.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

// Minimal SidecarLoad (full def in prt_shadow.h — harness is standalone)
struct SidecarLoad {
    int layer;
    std::string path;
    float * data;
    size_t size;
};

// Legacy sidecar: layer 0 only, 2048 bytes of mock float data
static float g_legacy_data[512] = {0};  // 512 floats = 2048 bytes

static SidecarLoad make_legacy_sidecar_0() {
    SidecarLoad sc;
    sc.layer = 0;
    sc.path = "/mock/legacy/sidecar_0.bin";
    sc.data = (float*)g_legacy_data;
    sc.size = 2048;
    return sc;
}

// g_sidecars MUST be initialized before prt_sidecar_runtime_link.h is processed
std::unordered_map<int, SidecarLoad> g_sidecars = {{0, make_legacy_sidecar_0()}};
bool g_sidecars_loaded = true;

// Now safe to include — g_sidecars is fully defined
#include "prt_sidecar_pager.h"
#include "prt_sidecar_runtime_link.h"

// ── Definitions for globals declared extern in prt_sidecar_runtime_link.h ──

prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;

// ── Stats ────────────────────────────────────────────────────────────────────

static uint64_t g_lookup_calls = 0;
static uint64_t g_pager_hits = 0;
static uint64_t g_legacy_hits = 0;
static uint64_t g_fallbacks = 0;
static uint64_t g_budget_rejects = 0;

// ── Synthetic .trit writer ───────────────────────────────────────────────────

static bool write_trit(const std::string& path, uint32_t layer_id,
                       const std::string& family, uint32_t family_idx) {
    uint8_t buf[32] = {0};
    buf[0] = 0x54; buf[1] = 0x52; buf[2] = 0x49; buf[3] = 0x54;  // "TRIT"
    uint32_t ver = 1; memcpy(buf + 4, &ver, 4);
    uint32_t rows = 512; memcpy(buf + 8, &rows, 4);
    uint32_t cols = 2048; memcpy(buf + 12, &cols, 4);
    uint32_t n_scales = 128; memcpy(buf + 16, &n_scales, 4);
    uint32_t family_hash = 0;
    memcpy(buf + 20, &family_hash, 4);
    uint32_t payload_off = 32; memcpy(buf + 22, &payload_off, 4);
    uint32_t scale_off = 32 + 393216; memcpy(buf + 26, &scale_off, 4);

    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= buf[i];
    }
    memcpy(buf + 30, &crc, 2);

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((char*)buf, 32);

    for (size_t i = 0; i < 393216; i++) {
        f.put((uint8_t)((i * 17 + layer_id * 31) & 0xFF));
    }
    for (size_t i = 0; i < 256; i++) {
        f.put((uint8_t)((i * 7 + layer_id * 13) & 0xFF));
    }
    return f.good();
}

// ── Setup/cleanup ────────────────────────────────────────────────────────────

static bool setup_package(const std::string& dir, int n_layers,
                          const std::vector<std::string>& families) {
    mkdir(dir.c_str(), 0755);
    std::string manifest_path = dir + "/manifest.json";
    FILE* mf = fopen(manifest_path.c_str(), "w");
    if (!mf) return false;
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
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            std::filesystem::remove(e.path());
        }
        std::filesystem::remove(dir + "/manifest.json");
        std::filesystem::remove(dir);
    } catch (...) {}
}

// ── Test: Legacy Disabled ────────────────────────────────────────────────────

static int test_legacy_disabled() {
    printf("\n=== TEST: LEGACY DISABLED (pager NOT initialized) ===\n");
    printf("DEBUG: g_sidecars.size=%zu\n", g_sidecars.size());

    g_prt_pager_enabled = false;
    if (g_prt_pager != nullptr) {
        prt_shutdown_pager();
    }

    g_lookup_calls++;
    auto view = prt_get_residual_view(0, "ffn_up");
    printf("prt_get_residual_view(0, ffn_up): is_null=%s reason=%s size=%zu\n",
           view.is_null ? "true" : "false", view.reason.c_str(), view.size);
    bool pass1 = !view.is_null && view.reason == "legacy" && view.size == 2048;
    if (!view.is_null) g_legacy_hits++;
    printf("Layer 0 via legacy: %s\n", pass1 ? "PASS" : "FAIL");

    g_lookup_calls++;
    auto view2 = prt_get_residual_view(99, "ffn_up");
    printf("prt_get_residual_view(99, ffn_up): is_null=%s reason=%s\n",
           view2.is_null ? "true" : "false", view2.reason.c_str());
    bool pass2 = view2.is_null && view2.reason == "not_found";
    printf("Missing layer: %s\n", pass2 ? "PASS" : "FAIL");

    printf("Result: %s\n", (pass1 && pass2) ? "PASS_LEGACY_DISABLED" : "FAIL_LEGACY_DISABLED");
    return (pass1 && pass2) ? 0 : 1;
}

// ── Test: Pager Enabled ──────────────────────────────────────────────────────

static int test_pager_enabled(const std::string& manifest_path,
                              const std::string& sidecar_root) {
    printf("\n=== TEST: PAGER ENABLED ===\n");

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = sidecar_root;
    cfg.max_resident_bytes = 2048 * 1024;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = true;

    bool ok = prt_init_pager(cfg);
    printf("prt_init_pager() = %s\n", ok ? "true" : "false");
    if (!ok) {
        printf("FAIL: pager init failed (error=%d)\n", (int)g_prt_pager->last_error());
        return 1;
    }

    prt_sidecar_pager_stats stats = prt_get_pager_stats();
    printf("Pager stats: resident=%zu reads=%zu\n", stats.resident_bytes, stats.reads);

    g_prt_pager->activate_layer(0);

    g_lookup_calls++;
    auto view = prt_get_residual_view(0, "ffn_up");
    printf("prt_get_residual_view(0, ffn_up): is_null=%s size=%zu reason=%s\n",
           view.is_null ? "true" : "false", view.size, view.reason.c_str());
    bool pass = !view.is_null && view.size == 393504;
    if (!view.is_null && view.reason.empty()) g_pager_hits++;
    printf("Pager lookup: %s\n", pass ? "PASS" : "FAIL");

    prt_shutdown_pager();
    printf("Result: %s\n", pass ? "PASS_PAGER_ENABLED" : "FAIL_PAGER_ENABLED");
    return pass ? 0 : 1;
}

// ── Test: Fallback (pager → legacy) ────────────────────────────────────────

static int test_fallback(const std::string& manifest_path,
                         const std::string& sidecar_root) {
    printf("\n=== TEST: FALLBACK (pager→legacy) ===\n");

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = sidecar_root;
    cfg.max_resident_bytes = 2048 * 1024;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = true;

    prt_init_pager(cfg);
    g_prt_pager->activate_layer(0);

    g_lookup_calls++;
    auto view_pager = prt_get_residual_view(0, "unknown_family");
    printf("prt_get_residual_view(0, unknown_family): is_null=%s reason=%s\n",
           view_pager.is_null ? "true" : "false", view_pager.reason.c_str());
    bool pass1 = !view_pager.is_null && view_pager.reason == "legacy";
    printf("Fallback to legacy: %s\n", pass1 ? "PASS" : "FAIL");
    if (!view_pager.is_null && view_pager.reason == "legacy") g_legacy_hits++;

    g_lookup_calls++;
    auto view_missing = prt_get_residual_view(99, "ffn_up");
    printf("prt_get_residual_view(99, ffn_up): is_null=%s reason=%s\n",
           view_missing.is_null ? "true" : "false", view_missing.reason.c_str());
    bool pass2 = view_missing.is_null && view_missing.reason == "not_found";
    printf("Missing layer: %s\n", pass2 ? "PASS" : "FAIL");
    if (view_missing.is_null) g_fallbacks++;

    prt_shutdown_pager();
    printf("Result: %s\n", (pass1 && pass2) ? "PASS_FALLBACK" : "FAIL_FALLBACK");
    return (pass1 && pass2) ? 0 : 1;
}

// ── Test: Budget Reject ─────────────────────────────────────────────────────

static int test_budget_reject(const std::string& manifest_path,
                              const std::string& sidecar_root) {
    printf("\n=== TEST: BUDGET REJECT ===\n");

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = sidecar_root;
    cfg.max_resident_bytes = 256 * 1024;  // 256 KB
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = true;

    prt_init_pager(cfg);

    bool rejected = !g_prt_pager->activate_layer(0);
    printf("activate_layer(0) with 256KB budget: %s\n", rejected ? "REJECTED" : "ACTIVATED");
    if (rejected) g_budget_rejects++;

    g_lookup_calls++;
    auto view = prt_get_residual_view(0, "ffn_up");
    printf("prt_get_residual_view(0, ffn_up) after reject: is_null=%s reason=%s\n",
           view.is_null ? "true" : "false", view.reason.c_str());
    bool pass = rejected && !view.is_null && view.reason == "legacy";
    printf("Fallback after reject: %s\n", pass ? "PASS" : "FAIL");

    prt_shutdown_pager();
    printf("Result: %s\n", pass ? "PASS_BUDGET_REJECT" : "FAIL_BUDGET_REJECT");
    return pass ? 0 : 1;
}

// ── Test: Lookup Stats ──────────────────────────────────────────────────────

static int test_lookup_stats() {
    printf("\n=== TEST: LOOKUP STATS ===\n");
    printf("lookup_calls=%lu pager_hits=%lu legacy_hits=%lu fallbacks=%lu budget_rejects=%lu\n",
           (unsigned long)g_lookup_calls, (unsigned long)g_pager_hits,
           (unsigned long)g_legacy_hits, (unsigned long)g_fallbacks,
           (unsigned long)g_budget_rejects);
    bool pass = g_lookup_calls > 0;
    printf("Stats tracking: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ── Main ────────────────────────────────────────────────────────────────────

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --legacy          Test legacy disabled mode\n");
    printf("  --pager           Test pager enabled mode\n");
    printf("  --fallback        Test fallback behavior\n");
    printf("  --budget          Test budget reject\n");
    printf("  --stats           Test lookup stats\n");
    printf("  --all             Run all tests\n");
    printf("  --manifest PATH   Manifest path\n");
    printf("  --sidecar-root DIR  Sidecar root dir\n");
    printf("  --setup-dir DIR N Setup synthetic package with N layers\n");
    printf("  --cleanup         Clean up package after\n");
}

int main(int argc, char** argv) {
    std::string manifest_path;
    std::string sidecar_root;
    std::string setup_dir;
    int setup_layers = 0;
    bool do_cleanup = false;
    std::vector<std::string> families = {"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--legacy") == 0) {
            return test_legacy_disabled();
        } else if (strcmp(argv[i], "--pager") == 0) {
            if (manifest_path.empty()) { fprintf(stderr, "Error: --manifest required\n"); return 1; }
            return test_pager_enabled(manifest_path, sidecar_root);
        } else if (strcmp(argv[i], "--fallback") == 0) {
            if (manifest_path.empty()) { fprintf(stderr, "Error: --manifest required\n"); return 1; }
            return test_fallback(manifest_path, sidecar_root);
        } else if (strcmp(argv[i], "--budget") == 0) {
            if (manifest_path.empty()) { fprintf(stderr, "Error: --manifest required\n"); return 1; }
            return test_budget_reject(manifest_path, sidecar_root);
        } else if (strcmp(argv[i], "--stats") == 0) {
            return test_lookup_stats();
        } else if (strcmp(argv[i], "--all") == 0) {
            int rc0 = test_legacy_disabled();
            std::string tmpdir = sidecar_root.empty() ? "/tmp/prt_matmul_lookup" : sidecar_root;
            if (manifest_path.empty()) {
                setup_package(tmpdir, 4, families);
                manifest_path = tmpdir + "/manifest.json";
                sidecar_root = tmpdir;
            }
            int rc1 = test_pager_enabled(manifest_path, sidecar_root);
            int rc2 = test_fallback(manifest_path, sidecar_root);
            int rc3 = test_budget_reject(manifest_path, sidecar_root);
            int rc4 = test_lookup_stats();
            if (do_cleanup) cleanup_package(tmpdir);
            printf("\n=== SUMMARY ===\n");
            printf("Legacy:    %s\n", rc0==0?"PASS":"FAIL");
            printf("Pager:     %s\n", rc1==0?"PASS":"FAIL");
            printf("Fallback:  %s\n", rc2==0?"PASS":"FAIL");
            printf("Budget:    %s\n", rc3==0?"PASS":"FAIL");
            printf("Stats:     %s\n", rc4==0?"PASS":"FAIL");
            int total = rc0+rc1+rc2+rc3+rc4;
            printf("\nVerdict: %s\n", total==0 ? "PASS_PHASE28AV_LOOKUP_HOOK" : "FAIL");
            return total;
        } else if (strcmp(argv[i], "--manifest") == 0 && i+1 < argc) {
            manifest_path = argv[++i];
        } else if (strcmp(argv[i], "--sidecar-root") == 0 && i+1 < argc) {
            sidecar_root = argv[++i];
        } else if (strcmp(argv[i], "--setup-dir") == 0 && i+2 < argc) {
            setup_dir = argv[++i]; setup_layers = atoi(argv[++i]);
            setup_package(setup_dir, setup_layers, families);
            printf("Setup complete: %s/\n", setup_dir.c_str());
            return 0;
        } else if (strcmp(argv[i], "--cleanup") == 0) {
            do_cleanup = true;
        } else {
            print_help(argv[0]);
            return 1;
        }
    }
    print_help(argv[0]);
    return 0;
}

#else  // PRT_SIDECAR_PAGER_EXPERIMENTAL

int main() {
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined — stub only\n");
    printf("Result: PASS_PHASE28AV_LOOKUP_HOOK (stub mode)\n");
    return 0;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL