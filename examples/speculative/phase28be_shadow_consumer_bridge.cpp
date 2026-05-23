// Phase 28BE: Runtime Shadow Lookup Consumer Bridge
// Proves the runtime-adjacent shadow consumer path (run_shadow_test() / prt_shadow.h)
// can request and consume a pager-backed residual view correctly.
//
// NO generation. NO speedup claims. NO quality parity claims.
//
// Build:
//   cd /home/matthew-villnave/llama.cpp && \
//   g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
//       -I. -Iggml/include -Iinclude \
//       -c examples/speculative/prt_sidecar_pager.cpp -o /tmp/prt_pager_28be.o && \
//   g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
//       -I. -Iggml/include -Iinclude \
//       -c examples/speculative/prt_trit_decode.cpp -o /tmp/prt_decode_28be.o && \
//   g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L \
//       -I. -Iggml/include -Iinclude \
//       examples/speculative/phase28be_shadow_consumer_bridge.cpp \
//       /tmp/prt_pager_28be.o /tmp/prt_decode_28be.o \
//       -o /tmp/phase28be_consumer_bridge

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <errno.h>
#include <cstdlib>
#include <algorithm>
#include <numeric>

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

#include "prt_shadow.h"          // shadow counters: g_shadow_*
#include "prt_sidecar_pager.h"   // prt_residual_view, prt_init_pager, prt_shutdown_pager
#include "prt_sidecar_runtime_link.h"  // prt_get_residual_view()
#include "prt_trit_decode.h"     // prt_trit_decoder, prt_decoded_view

// ── Strong definitions for runtime-link globals ──────────────────────────
// (prt_sidecar_runtime_link.h declares these extern — provide strong defs)
prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;

// ── Mock g_sidecars (empty — we test pager path only) ───────────────────
std::unordered_map<int, SidecarLoad> g_sidecars;
bool g_sidecars_loaded = false;

// ── Math helpers ─────────────────────────────────────────────────────────
static double max_abs_err(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, fabs(a[i] - b[i]));
    return m;
}
static double rmse(const float* a, const float* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) { double d = a[i] - b[i]; s += d*d; }
    return sqrt(s / n);
}

// ── Reset shadow counters (must use extern decl from prt_shadow.h) ─────
static void reset_shadow_counters() {
    // Use the official API — prt_shadow.h has static globals, not extern
    prt_reset_shadow_stats();
}

// ── Decode a .trit file directly (baseline) ──────────────────────────────
static prt_decoded_view decode_direct(const std::string& path) {
    prt_trit_decoder dec;
    return dec.decode_file(path);
}

// ── Extract header fields from raw .trit bytes ───────────────────────────
struct TritHeaderFields {
    uint32_t rows = 0, cols = 0;
    uint16_t block_rows = 0, block_cols = 0;
    uint16_t n_scales = 0;
    uint32_t scale_offset = 0;
    bool valid = false;
};
static TritHeaderFields parse_trit_header(const uint8_t* hdr, size_t size) {
    TritHeaderFields f;
    if (size < 32) return f;
    uint32_t magic = *(const uint32_t*)(hdr + 0);
    if (magic != 0x54524954 && magic != 0x54495254) return f;
    f.rows = *(const uint32_t*)(hdr + 8);
    f.cols = *(const uint32_t*)(hdr + 12);
    f.block_rows = *(const uint16_t*)(hdr + 16);
    f.block_cols = *(const uint16_t*)(hdr + 18);
    f.n_scales = *(const uint16_t*)(hdr + 20);
    f.scale_offset = *(const uint32_t*)(hdr + 26);
    f.valid = true;
    return f;
}

// ── Result structure per test case ───────────────────────────────────────
struct TestResult {
    std::string case_name;
    bool pass = false;
    double R_max_abs_err = -1;
    double R_rmse = -1;
    bool pager_view_is_null = true;
    bool pager_view_size_gt_0 = false;
    uint64_t shadow_lookup_calls = 0;
    uint64_t shadow_pager_hits = 0;
    uint64_t shadow_legacy_hits = 0;
    uint64_t shadow_null_views = 0;
    uint64_t shadow_budget_rejects = 0;
    std::string fail_reason;
};

// ── Test one positive case ────────────────────────────────────────────────
static TestResult test_positive_case(const std::string& pkg_dir,
                                     const std::string& case_name,
                                     int layer_idx,
                                     const std::string& tensor_family) {
    TestResult r;
    r.case_name = case_name;
    reset_shadow_counters();

    std::string manifest_path = pkg_dir + "/manifest.json";
    std::string trit_path = pkg_dir + "/layers/layer_000/ffn_up.trit";

    // Direct decode baseline
    prt_decoded_view direct = decode_direct(trit_path);
    if (direct.is_null) {
        r.fail_reason = "direct_decode_failed: " + direct.reason;
        return r;
    }

    // Init pager
    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    if (!prt_init_pager(cfg)) {
        r.fail_reason = "pager_init_failed";
        delete[] direct.data;
        return r;
    }

    // Activate layer
    if (!g_prt_pager->activate_layer(layer_idx)) {
        prt_shutdown_pager();
        r.fail_reason = "activate_layer_failed";
        delete[] direct.data;
        return r;
    }

    // ── POSITIVE TEST (via run_shadow_test to exercise counter path) ────
    // run_shadow_test() calls prt_get_residual_view() internally and increments
    // shadow counters. We provide dummy X_act/Y_float just to exercise the path.
    // Decode correctness is verified via direct decode comparison above.
    // The shadow counter state (pager_hits vs legacy_hits) confirms the routing path.

    // Provide dummy activation and float output for run_shadow_test()
    // M=rows, N=cols from the .trit header
    uint32_t M_rows = (uint32_t)direct.rows;
    uint32_t N_cols = (uint32_t)direct.cols;
    size_t n = (size_t)M_rows * (size_t)N_cols;

    std::vector<float> dummy_X(n, 0.5f);   // synthetic activation
    std::vector<float> dummy_Y(n, 0.0f);   // synthetic float output

    // Get raw residual view via prt_get_residual_view (runtime-adjacent path)
    prt_residual_view raw = prt_get_residual_view(layer_idx, tensor_family);
    r.pager_view_is_null = raw.is_null;
    r.pager_view_size_gt_0 = !raw.is_null && raw.size > 0;

    // Also call run_shadow_test() to exercise the full shadow counter path
    ShadowResult sr = run_shadow_test(layer_idx, 1, dummy_X.data(), dummy_Y.data(),
                                      (int)M_rows, (int)N_cols);

    // Capture shadow counter state AFTER both calls
    r.shadow_lookup_calls = g_shadow_lookup_calls;
    r.shadow_pager_hits = g_shadow_pager_hits;
    r.shadow_legacy_hits = g_shadow_legacy_hits;
    r.shadow_null_views = g_shadow_null_views;
    r.shadow_budget_rejects = g_shadow_budget_rejects;

    // Decode via runtime-adjacent shadow path
    if (raw.is_null) {
        prt_shutdown_pager();
        r.fail_reason = "pager_view_null";
        delete[] direct.data;
        return r;
    }
    TritHeaderFields hdr = parse_trit_header(raw.data, raw.size);
    if (!hdr.valid) {
        prt_shutdown_pager();
        r.fail_reason = "header_parse_failed";
        delete[] direct.data;
        return r;
    }

    // Extract scales
    std::vector<float> scales(hdr.n_scales, 0.0f);
    if (hdr.n_scales > 0 && hdr.scale_offset > 0 &&
        hdr.scale_offset + hdr.n_scales * 4 <= raw.size) {
        memcpy(scales.data(), raw.data + hdr.scale_offset, hdr.n_scales * 4);
    }

    prt_trit_decoder dec;
    prt_decoded_view consumer_decoded = dec.decode_bytes(
        raw.data, raw.size,
        hdr.rows, hdr.cols,
        hdr.block_rows, hdr.block_cols,
        hdr.n_scales, scales.data());

    prt_shutdown_pager();

    if (consumer_decoded.is_null) {
        r.fail_reason = "consumer_decode_failed: " + consumer_decoded.reason;
        delete[] direct.data;
        return r;
    }

    // Compare decoded vs direct baseline
    if (consumer_decoded.rows != direct.rows || consumer_decoded.cols != direct.cols) {
        r.fail_reason = "dimension_mismatch";
        delete[] direct.data; delete[] consumer_decoded.data;
        return r;
    }

    r.R_max_abs_err = max_abs_err(direct.data, consumer_decoded.data, n);
    r.R_rmse = rmse(direct.data, consumer_decoded.data, n);

    // Shadow counter assertions: pager path should have pager_hits > 0
    // (the call to run_shadow_test above incremented them)
    bool counters_ok = (r.shadow_pager_hits > 0) && (r.shadow_legacy_hits == 0);
    bool null_ok = (r.shadow_null_views == 0);
    bool budget_ok = (r.shadow_budget_rejects == 0);

    r.pass = (r.R_max_abs_err < 1e-4f) && counters_ok && null_ok && budget_ok;
    if (!r.pass) {
        std::ostringstream oss;
        oss << "max_err=" << r.R_max_abs_err;
        if (!counters_ok) oss << " pager_hits=" << r.shadow_pager_hits << " legacy_hits=" << r.shadow_legacy_hits;
        if (!null_ok) oss << " null_views=" << r.shadow_null_views;
        if (!budget_ok) oss << " budget_rejects=" << r.shadow_budget_rejects;
        r.fail_reason = oss.str();
    }

    delete[] direct.data; delete[] consumer_decoded.data;
    return r;
}

// ── CONTROL TEST A: DISABLED MODE ────────────────────────────────────────
// After prt_init_pager() succeeds, set g_prt_pager_enabled=false,
// confirm prt_get_residual_view falls through (null view or legacy).
struct ControlResult {
    std::string name;
    bool pass = false;
    std::string detail;
};
static ControlResult test_disabled_mode(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "DISABLED_MODE";
    reset_shadow_counters();

    std::string manifest_path = pkg_dir + "/manifest.json";

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    bool ok = prt_init_pager(cfg);
    if (!ok) { r.detail = "init_failed"; r.pass = false; return r; }

    // Force disable BEFORE calling lookup
    g_prt_pager_enabled = false;

    g_prt_pager->activate_layer(0);
    prt_residual_view raw = prt_get_residual_view(0, "ffn_up");

    // With pager disabled and g_sidecars empty, expect null view with "not_found" reason
    r.pass = raw.is_null && (raw.reason == "not_found" || raw.reason == "legacy");

    std::ostringstream oss;
    oss << "is_null=" << raw.is_null << " reason=" << raw.reason;
    oss << " lookup_calls=" << g_shadow_lookup_calls;
    oss << " legacy_hits=" << g_shadow_legacy_hits;
    r.detail = oss.str();

    prt_shutdown_pager();
    g_prt_pager_enabled = false;
    return r;
}

// ── CONTROL TEST B: MISSING SIDECAR ───────────────────────────────────────
static ControlResult test_missing_sidecar() {
    ControlResult r;
    r.name = "MISSING_SIDECAR";
    reset_shadow_counters();

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = "/tmp/nonexistent_28be_manifest.json";
    cfg.sidecar_root = "/tmp/nonexistent_28be_root";
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    bool ok = prt_init_pager(cfg);
    // Expect deterministic failure
    r.pass = !ok;
    r.detail = ok ? "init_succeeded_unexpectedly" : "init_correctly_failed";

    prt_shutdown_pager();
    return r;
}

// ── CONTROL TEST C: BAD TENSOR FAMILY ────────────────────────────────────
static ControlResult test_bad_tensor_family(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "BAD_TENSOR_FAMILY";
    reset_shadow_counters();

    std::string manifest_path = pkg_dir + "/manifest.json";

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    bool ok = prt_init_pager(cfg);
    if (!ok) { r.detail = "init_failed"; r.pass = false; return r; }

    g_prt_pager->activate_layer(0);
    prt_residual_view raw = prt_get_residual_view(0, "nonexistent_family");
    r.pass = raw.is_null;
    std::ostringstream oss;
    oss << "is_null=" << raw.is_null << " reason=" << raw.reason;
    r.detail = oss.str();

    prt_shutdown_pager();
    return r;
}

// ── CONTROL TEST D: BUDGET REJECT ─────────────────────────────────────────
static ControlResult test_budget_reject(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "BUDGET_REJECT";
    reset_shadow_counters();

    std::string manifest_path = pkg_dir + "/manifest.json";

    // Very small budget — single layer residual will exceed it
    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 64;  // 64 bytes — far too small for any real residual
    cfg.checksum_enabled = false;
    cfg.strict_budget = true;    // Reject rather than evict
    cfg.eviction_lru = false;
    cfg.validate_trit_header = false;

    bool ok = prt_init_pager(cfg);
    if (!ok) { r.detail = "init_failed"; r.pass = false; return r; }

    // activate_layer may succeed if it doesn't load, but get_residual should budget-reject
    g_prt_pager->activate_layer(0);
    prt_residual_view raw = prt_get_residual_view(0, "ffn_up");

    // Either pager returns null (budget exceeded), or we get null view
    bool null_view = raw.is_null;
    r.pass = null_view;  // Budget reject → null view
    std::ostringstream oss;
    oss << "is_null=" << raw.is_null << " reason=" << raw.reason;
    oss << " budget_rejects=" << g_shadow_budget_rejects;
    r.detail = oss.str();

    prt_shutdown_pager();
    return r;
}

// ── Main ──────────────────────────────────────────────────────────────────
int main() {
    printf("Phase 28BE: Runtime Shadow Lookup Consumer Bridge\n");
    printf("===================================================\n");

    struct CaseDef {
        const char* pkg;
        const char* name;
        int layer;
    };
    CaseDef cases[] = {
        {"row_split",     "row_split",     0},
        {"col_split",     "col_split",     0},
        {"row_col_split","row_col_split", 0},
        {"awkward_edge",  "awkward_edge",  0},
    };

    std::vector<TestResult> results;
    std::vector<ControlResult> controls;
    int fails = 0;

    // ── POSITIVE TESTS ──────────────────────────────────────────────────
    for (const auto& c : cases) {
        std::string pkg_dir = "/tmp/phase28bd_pkg_" + std::string(c.pkg);
        printf("\n=== 28BE-POS: %s ===\n", c.name);
        TestResult r = test_positive_case(pkg_dir, c.name, c.layer, "ffn_up");
        results.push_back(r);

        printf("  direct_vs_consumer_R_max_abs_err: %.8e\n", r.R_max_abs_err);
        printf("  direct_vs_consumer_R_rmse:        %.8e\n", r.R_rmse);
        printf("  pager_view_is_null:               %s\n", r.pager_view_is_null ? "true" : "false");
        printf("  pager_view_size_gt_0:             %s\n", r.pager_view_size_gt_0 ? "true" : "false");
        printf("  shadow_lookup_calls:               %lu\n", (unsigned long)r.shadow_lookup_calls);
        printf("  shadow_pager_hits:                 %lu\n", (unsigned long)r.shadow_pager_hits);
        printf("  shadow_legacy_hits:                %lu\n", (unsigned long)r.shadow_legacy_hits);
        printf("  shadow_null_views:                 %lu\n", (unsigned long)r.shadow_null_views);
        printf("  shadow_budget_rejects:             %lu\n", (unsigned long)r.shadow_budget_rejects);
        printf("  pass: %s\n", r.pass ? "true" : "false");
        if (!r.pass) {
            printf("  FAIL_REASON: %s\n", r.fail_reason.c_str());
            fails++;
        }
        printf("Result: %s\n", r.pass ? "PASS_28BE_POS" : "FAIL_28BE_POS");
    }

    // ── CONTROL TESTS ──────────────────────────────────────────────────
    std::string ref_pkg = "/tmp/phase28bd_pkg_row_split";

    ControlResult ctrl_a = test_disabled_mode(ref_pkg);
    printf("\n=== 28BE-CTRL-A: %s ===\n", ctrl_a.name.c_str());
    printf("  pass: %s\n", ctrl_a.pass ? "true" : "false");
    printf("  detail: %s\n", ctrl_a.detail.c_str());
    printf("Result: %s\n", ctrl_a.pass ? "PASS_28BE_CTRL_A" : "FAIL_28BE_CTRL_A");
    if (!ctrl_a.pass) fails++;
    controls.push_back(ctrl_a);

    ControlResult ctrl_b = test_missing_sidecar();
    printf("\n=== 28BE-CTRL-B: %s ===\n", ctrl_b.name.c_str());
    printf("  pass: %s\n", ctrl_b.pass ? "true" : "false");
    printf("  detail: %s\n", ctrl_b.detail.c_str());
    printf("Result: %s\n", ctrl_b.pass ? "PASS_28BE_CTRL_B" : "FAIL_28BE_CTRL_B");
    if (!ctrl_b.pass) fails++;
    controls.push_back(ctrl_b);

    ControlResult ctrl_c = test_bad_tensor_family(ref_pkg);
    printf("\n=== 28BE-CTRL-C: %s ===\n", ctrl_c.name.c_str());
    printf("  pass: %s\n", ctrl_c.pass ? "true" : "false");
    printf("  detail: %s\n", ctrl_c.detail.c_str());
    printf("Result: %s\n", ctrl_c.pass ? "PASS_28BE_CTRL_C" : "FAIL_28BE_CTRL_C");
    if (!ctrl_c.pass) fails++;
    controls.push_back(ctrl_c);

    ControlResult ctrl_d = test_budget_reject(ref_pkg);
    printf("\n=== 28BE-CTRL-D: %s ===\n", ctrl_d.name.c_str());
    printf("  pass: %s\n", ctrl_d.pass ? "true" : "false");
    printf("  detail: %s\n", ctrl_d.detail.c_str());
    printf("Result: %s\n", ctrl_d.pass ? "PASS_28BE_CTRL_D" : "FAIL_28BE_CTRL_D");
    if (!ctrl_d.pass) fails++;
    controls.push_back(ctrl_d);

    // ── Summary ────────────────────────────────────────────────────────
    printf("\n===================================================\n");
    printf("Phase 28BE SUMMARY\n");
    for (const auto& r : results) {
        printf("  %s: %s (R_max_err=%.4e)\n", r.case_name.c_str(),
               r.pass ? "PASS" : "FAIL", r.R_max_abs_err);
    }
    printf("Controls: DISABLED=%s MISSING=%s BAD_FAMILY=%s BUDGET=%s\n",
           ctrl_a.pass ? "PASS" : "FAIL",
           ctrl_b.pass ? "PASS" : "FAIL",
           ctrl_c.pass ? "PASS" : "FAIL",
           ctrl_d.pass ? "PASS" : "FAIL");
    printf("Total failures: %d\n", fails);
    printf("Verdict: %s\n", fails==0 ? "PASS_PHASE28BE_RUNTIME_SHADOW_LOOKUP_CONSUMER_BRIDGE" : "FAIL_PHASE28BE");

    // ── Write JSON output ───────────────────────────────────────────────
    std::string json_path = "/home/matthew-villnave/llama.cpp/examples/speculative/results/phase28be_runtime_shadow_lookup_consumer_bridge.json";
    FILE* jf = fopen(json_path.c_str(), "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"phase\": \"28BE\",\n");
        fprintf(jf, "  \"verdict\": \"%s\",\n", fails==0 ? "PASS" : "FAIL");
        fprintf(jf, "  \"positive_tests\": [\n");
        for (size_t i = 0; i < results.size(); i++) {
            const auto& r = results[i];
            fprintf(jf, "    {\n");
            fprintf(jf, "      \"case_name\": \"%s\",\n", r.case_name.c_str());
            fprintf(jf, "      \"pass\": %s,\n", r.pass ? "true" : "false");
            fprintf(jf, "      \"R_max_abs_err\": %.8e,\n", r.R_max_abs_err);
            fprintf(jf, "      \"R_rmse\": %.8e,\n", r.R_rmse);
            fprintf(jf, "      \"pager_view_is_null\": %s,\n", r.pager_view_is_null ? "true" : "false");
            fprintf(jf, "      \"pager_view_size_gt_0\": %s,\n", r.pager_view_size_gt_0 ? "true" : "false");
            fprintf(jf, "      \"shadow_lookup_calls\": %lu,\n", (unsigned long)r.shadow_lookup_calls);
            fprintf(jf, "      \"shadow_pager_hits\": %lu,\n", (unsigned long)r.shadow_pager_hits);
            fprintf(jf, "      \"shadow_legacy_hits\": %lu,\n", (unsigned long)r.shadow_legacy_hits);
            fprintf(jf, "      \"shadow_null_views\": %lu,\n", (unsigned long)r.shadow_null_views);
            fprintf(jf, "      \"shadow_budget_rejects\": %lu,\n", (unsigned long)r.shadow_budget_rejects);
            fprintf(jf, "      \"fail_reason\": \"%s\"\n", r.fail_reason.c_str());
            fprintf(jf, "    }%s\n", (i < results.size() - 1) ? "," : "");
        }
        fprintf(jf, "  ],\n");
        fprintf(jf, "  \"control_tests\": [\n");
        for (size_t i = 0; i < controls.size(); i++) {
            const auto& c = controls[i];
            fprintf(jf, "    {\"name\": \"%s\", \"pass\": %s, \"detail\": \"%s\"}%s\n",
                    c.name.c_str(), c.pass ? "true" : "false", c.detail.c_str(),
                    (i < controls.size() - 1) ? "," : "");
        }
        fprintf(jf, "  ]\n");
        fprintf(jf, "}\n");
        fclose(jf);
        printf("JSON written to: %s\n", json_path.c_str());
    } else {
        printf("WARNING: could not write JSON to %s\n", json_path.c_str());
    }

    return fails;
}

#else  // PRT_SIDECAR_PAGER_EXPERIMENTAL

int main() {
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    printf("Result: FAIL_PHASE28BE (stub)\n");
    return 1;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL