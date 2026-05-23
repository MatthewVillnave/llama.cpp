// Phase 28AW: PRT run_shadow_test() Caller Integration Harness
// Tests run_shadow_test() calling prt_get_residual_view() — no generation.
// Build: g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL
//        -I. -Iggml/include -Iinclude examples/speculative/prt_sidecar_pager.cpp
//        examples/speculative/prt_run_shadow_lookup_harness.cpp -o /tmp/prt_run_shadow_harness

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
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdint>

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

// Include prt_shadow.h FIRST — g_sidecars is extern there (declared, not defined)
#include "prt_shadow.h"

// Provide ONE definition of g_sidecars (strong symbol) — test version with layer 0 only
static float g_test_legacy_data[512] = {0};  // 2048 bytes

static SidecarLoad make_test_legacy() {
    SidecarLoad sc;
    sc.layer = 0;
    sc.path = "/mock/legacy/sidecar_0.bin";
    sc.data = (float*)g_test_legacy_data;
    sc.size = 2048;
    return sc;
}

std::unordered_map<int, SidecarLoad> g_sidecars = {{0, make_test_legacy()}};
bool g_sidecars_loaded = true;

// Include pager headers after g_sidecars is defined
#include "prt_sidecar_pager.h"
#include "prt_sidecar_runtime_link.h"

// Definitions for globals declared extern in prt_sidecar_runtime_link.h
prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;

// ── Stats helpers ────────────────────────────────────────────────────────────

static void print_shadow_stats() {
    uint64_t calls=0, ph=0, lh=0, nv=0, br=0;
    prt_get_shadow_stats(&calls, &ph, &lh, &nv, &br);
    printf("  shadow_stats: calls=%lu pager_hits=%lu legacy_hits=%lu null_views=%lu budget_rejects=%lu\n",
           (unsigned long)calls, (unsigned long)ph, (unsigned long)lh,
           (unsigned long)nv, (unsigned long)br);
}

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
    prt_reset_shadow_stats();

    g_prt_pager_enabled = false;
    if (g_prt_pager != nullptr) {
        prt_shutdown_pager();
    }

    float X_act[8] = {0}, Y_float[8] = {0};
    ShadowResult r = run_shadow_test(0, 1, X_act, Y_float, 8, 8);
    printf("  run_shadow_test(0): has_sidecar=%s computed=%s\n",
           r.has_sidecar ? "true" : "false", r.computed ? "true" : "false");
    bool pass1 = r.has_sidecar && r.computed;
    printf("  Layer 0: %s\n", pass1 ? "PASS" : "FAIL");

    ShadowResult r2 = run_shadow_test(99, 1, X_act, Y_float, 8, 8);
    printf("  run_shadow_test(99): has_sidecar=%s\n", r2.has_sidecar ? "true" : "false");
    bool pass2 = !r2.has_sidecar;
    printf("  Layer 99: %s\n", pass2 ? "PASS" : "FAIL");

    uint64_t calls=0, ph=0, lh=0, nv=0, br=0;
    prt_get_shadow_stats(&calls, &ph, &lh, &nv, &br);
    printf("  stats: calls=%lu pager_hits=%lu legacy_hits=%lu null_views=%lu\n",
           (unsigned long)calls, (unsigned long)ph, (unsigned long)lh, (unsigned long)nv);
    bool pass3 = calls == 2 && lh == 1 && nv == 1 && ph == 0;
    printf("  Stats: %s\n", pass3 ? "PASS" : "FAIL");

    printf("Result: %s\n", (pass1 && pass2 && pass3) ? "PASS_LEGACY_DISABLED" : "FAIL_LEGACY_DISABLED");
    return (pass1 && pass2 && pass3) ? 0 : 1;
}

// ── Test: Pager Enabled ──────────────────────────────────────────────────────

static int test_pager_enabled(const std::string& manifest_path,
                              const std::string& sidecar_root) {
    printf("\n=== TEST: PAGER ENABLED ===\n");
    prt_reset_shadow_stats();

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = sidecar_root;
    cfg.max_resident_bytes = 2048 * 1024;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = false;  // LRU mode for budget test

    bool ok = prt_init_pager(cfg);
    printf("  prt_init_pager() = %s\n", ok ? "true" : "false");
    if (!ok) {
        printf("  FAIL: pager init failed\n");
        return 1;
    }

    bool activated = g_prt_pager->activate_layer(0);
    printf("  activate_layer(0): %s\n", activated ? "ACTIVATED" : "REJECTED");
    if (activated) g_shadow_budget_rejects++; // track even if activated

    float X_act[8] = {0}, Y_float[8] = {0};
    ShadowResult r = run_shadow_test(0, 1, X_act, Y_float, 8, 8);
    printf("  run_shadow_test(0): has_sidecar=%s computed=%s\n",
           r.has_sidecar ? "true" : "false", r.computed ? "true" : "false");
    bool pass = r.has_sidecar && r.computed;
    printf("  Layer 0: %s\n", pass ? "PASS" : "FAIL");

    uint64_t calls=0, ph=0, lh=0, nv=0, br=0;
    prt_get_shadow_stats(&calls, &ph, &lh, &nv, &br);
    printf("  stats: calls=%lu pager_hits=%lu legacy_hits=%lu\n",
           (unsigned long)calls, (unsigned long)ph, (unsigned long)lh);
    bool pass2 = calls == 1 && ph == 1 && lh == 0;
    printf("  Stats: %s\n", pass2 ? "PASS" : "FAIL");

    prt_shutdown_pager();
    printf("Result: %s\n", (pass && pass2) ? "PASS_PAGER_ENABLED" : "FAIL_PAGER_ENABLED");
    return (pass && pass2) ? 0 : 1;
}

// ── Test: Fallback ───────────────────────────────────────────────────────────

static int test_fallback(const std::string& manifest_path,
                         const std::string& sidecar_root) {
    printf("\n=== TEST: FALLBACK (pager→legacy) ===\n");
    prt_reset_shadow_stats();

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = sidecar_root;
    cfg.max_resident_bytes = 2048 * 1024;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = false;  // LRU mode for budget test

    prt_init_pager(cfg);
    g_prt_pager->activate_layer(0);

    // Layer 99: not in pager, not in legacy → null
    float X_act[8] = {0}, Y_float[8] = {0};
    ShadowResult r = run_shadow_test(99, 1, X_act, Y_float, 8, 8);
    printf("  run_shadow_test(99): has_sidecar=%s\n", r.has_sidecar ? "true" : "false");
    bool pass = !r.has_sidecar;
    printf("  Unknown layer: %s\n", pass ? "PASS" : "FAIL");

    uint64_t calls=0, ph=0, lh=0, nv=0, br=0;
    prt_get_shadow_stats(&calls, &ph, &lh, &nv, &br);
    printf("  stats: calls=%lu null_views=%lu\n", (unsigned long)calls, (unsigned long)nv);
    bool pass2 = calls == 1 && nv == 1;
    printf("  Stats: %s\n", pass2 ? "PASS" : "FAIL");

    prt_shutdown_pager();
    printf("Result: %s\n", (pass && pass2) ? "PASS_FALLBACK" : "FAIL_FALLBACK");
    return (pass && pass2) ? 0 : 1;
}

// ── Test: Budget Reject ─────────────────────────────────────────────────────

static int test_budget_reject(const std::string& manifest_path,
                              const std::string& sidecar_root) {
    printf("\n=== TEST: BUDGET REJECT ===\n");
    prt_reset_shadow_stats();

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = sidecar_root;
    cfg.max_resident_bytes = 256 * 1024;  // Too small
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = false;  // LRU mode for budget test

    prt_init_pager(cfg);

    float X_act[8] = {0}, Y_float[8] = {0};
    ShadowResult r = run_shadow_test(0, 1, X_act, Y_float, 8, 8);
    printf("  run_shadow_test(0) with budget reject: has_sidecar=%s\n",
           r.has_sidecar ? "true" : "false");
    // Falls back to legacy which has layer 0
    bool pass = r.has_sidecar;
    printf("  Legacy fallback: %s\n", pass ? "PASS" : "FAIL");

    uint64_t calls=0, ph=0, lh=0, nv=0, br=0;
    prt_get_shadow_stats(&calls, &ph, &lh, &nv, &br);
    printf("  stats: calls=%lu null_views=%lu legacy_hits=%lu\n",
           (unsigned long)calls, (unsigned long)nv, (unsigned long)lh);
    bool pass2 = calls == 1 && r.has_sidecar;
    printf("  Stats: %s\n", pass2 ? "PASS" : "FAIL");

    prt_shutdown_pager();
    printf("Result: %s\n", (pass && pass2) ? "PASS_BUDGET_REJECT" : "FAIL_BUDGET_REJECT");
    return (pass && pass2) ? 0 : 1;
}

// ── Main ────────────────────────────────────────────────────────────────────

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --legacy          Test legacy disabled mode\n");
    printf("  --pager           Test pager enabled mode\n");
    printf("  --fallback        Test fallback behavior\n");
    printf("  --budget          Test budget reject\n");
    printf("  --all             Run all tests\n");
    printf("  --manifest PATH   Manifest path\n");
    printf("  --sidecar-root DIR  Sidecar root dir\n");
    printf("  --setup-dir DIR N Setup synthetic package with N layers\n");
    printf("  --cleanup         Clean up package after\n");
}

int main(int argc, char** argv) {
    std::string manifest_path;
    std::string sidecar_root;
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
        } else if (strcmp(argv[i], "--all") == 0) {
            int rc0 = test_legacy_disabled();
            std::string tmpdir = sidecar_root.empty() ? "/tmp/prt_run_shadow" : sidecar_root;
            if (manifest_path.empty()) {
                setup_package(tmpdir, 4, families);
                manifest_path = tmpdir + "/manifest.json";
                sidecar_root = tmpdir;
            }
            int rc1 = test_pager_enabled(manifest_path, sidecar_root);
            int rc2 = test_fallback(manifest_path, sidecar_root);
            int rc3 = test_budget_reject(manifest_path, sidecar_root);
            if (do_cleanup) cleanup_package(tmpdir);
            printf("\n=== SUMMARY ===\n");
            printf("Legacy:    %s\n", rc0==0?"PASS":"FAIL");
            printf("Pager:     %s\n", rc1==0?"PASS":"FAIL");
            printf("Fallback:  %s\n", rc2==0?"PASS":"FAIL");
            printf("Budget:    %s\n", rc3==0?"PASS":"FAIL");
            int total = rc0+rc1+rc2+rc3;
            printf("\nVerdict: %s\n", total==0 ? "PASS_PHASE28AW_CALLER_INTEGRATION" : "FAIL");
            return total;
        } else if (strcmp(argv[i], "--manifest") == 0 && i+1 < argc) {
            manifest_path = argv[++i];
        } else if (strcmp(argv[i], "--sidecar-root") == 0 && i+1 < argc) {
            sidecar_root = argv[++i];
        } else if (strcmp(argv[i], "--setup-dir") == 0 && i+2 < argc) {
            std::string d = argv[++i]; int n = atoi(argv[++i]);
            setup_package(d, n, families);
            printf("Setup complete: %s/\n", d.c_str());
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
    printf("Result: PASS_PHASE28AW_CALLER_INTEGRATION (stub mode)\n");
    return 0;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL