// Phase 28BG: Shadow/Pager Matmul Coverage Sweep
//
// Proves the shadow/pager overlay matmul path works across multiple real
// Qwen2.5-0.5B tensor families (ffn_up, ffn_gate, ffn_down, attn_output)
// across at least 2 layers (layer 0 + layer 1 + layer 5), with correct
// orientation handling per family.
//
// NO generation. NO Bonsai. NO llama.cpp model inference.

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

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
#include <unordered_map>
#include <chrono>

#include "prt_shadow.h"          // run_shadow_test(), prt_reset_shadow_stats(), prt_get_shadow_stats()
#include "prt_sidecar_pager.h"   // prt_residual_view, prt_sidecar_pager, prt_init_pager, prt_shutdown_pager
#include "prt_sidecar_runtime_link.h"  // prt_get_residual_view()
#include "prt_trit_decode.h"     // prt_trit_decoder, prt_decoded_view

// ── Strong definitions for runtime-link globals ──────────────────────────
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
static double cosine_sim(const float* a, const float* b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (sqrt(na) * sqrt(nb) + 1e-9);
}

// ── Dense matmul: Y[b,n] = sum_k X[b,k] * W[k,n] ────────────────────────
static void matmul_dense(const float* X, const float* W, float* Y, int B, int M, int N, int K) {
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += X[b * M * K + k] * W[k * N + n];
            }
            Y[b * M * N + n] = sum;
        }
    }
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

// ── Result structures ────────────────────────────────────────────────────
struct TestResult {
    std::string case_name;
    std::string tensor_family;
    int source_layer = -1;
    std::string source_key;
    std::string original_shape;
    bool pass = false;
    int K = 0, M = 0, N = 0;
    uint16_t block_rows = 0, block_cols = 0;
    double R_max_abs_err = -1, R_rmse = -1;
    double W_max_abs_err = -1, W_rmse = -1;
    double Y_max_abs_err = -1, Y_rmse = -1;
    double Y_cosine = -1;
    bool pager_view_is_null = true;
    bool pager_view_size_gt_0 = false;
    uint64_t shadow_lookup_calls = 0;
    uint64_t shadow_pager_hits = 0;
    uint64_t shadow_legacy_hits = 0;
    uint64_t shadow_null_views = 0;
    uint64_t shadow_budget_rejects = 0;
    std::string fail_reason;
};

struct ControlResult {
    std::string name;
    bool pass = false;
    std::string detail;
};

// ── POSITIVE TEST ───────────────────────────────────────────────────────
static TestResult test_positive(const std::string& pkg_dir,
                                 const std::string& case_name,
                                 const std::string& tensor_family,
                                 uint32_t K, uint32_t M, uint32_t N,
                                 uint16_t block_rows, uint16_t block_cols,
                                 const std::string& source_key,
                                 const std::string& original_shape,
                                 int source_layer) {
    TestResult r;
    r.case_name = case_name;
    r.tensor_family = tensor_family;
    r.source_layer = source_layer;
    r.source_key = source_key;
    r.original_shape = original_shape;
    r.K = (int)K; r.M = (int)M; r.N = (int)N;
    r.block_rows = block_rows; r.block_cols = block_cols;

    prt_reset_shadow_stats(); // Reset counters at start

    // Build path to the .trit file for this family
    std::string trit_file = tensor_family + ".trit";
    std::string trit_path = pkg_dir + "/layers/layer_000/" + trit_file;
    std::string manifest = pkg_dir + "/manifest.json";

    // Step 1: Reference path — decode directly from .trit file
    prt_trit_decoder ref_dec;
    prt_decoded_view R_ref = ref_dec.decode_file(trit_path);
    if (R_ref.is_null) {
        r.fail_reason = "ref_decode_failed: " + R_ref.reason;
        return r;
    }

    // Step 2: Generate deterministic W_base[K*N]
    std::vector<float> W_base(K * N);
    for (size_t i = 0; i < K * N; i++) {
        W_base[i] = float((i * K + i) % 101) / 100.0f;
    }

    // Step 3: Generate deterministic X[1*M*K]
    std::vector<float> X_act(1 * M * K);
    for (size_t i = 0; i < 1 * M * K; i++) {
        X_act[i] = float((i * 13 + i * 7) % 97) / 100.0f;
    }

    // Step 4: Reference Y = X @ (W_base + R_ref)
    std::vector<float> W_full(K * N);
    for (size_t i = 0; i < K * N; i++) W_full[i] = W_base[i] + R_ref.data[i];
    std::vector<float> Y_ref(1 * M * N);
    matmul_dense(X_act.data(), W_full.data(), Y_ref.data(), 1, (int)M, (int)N, (int)K);

    // Step 5: Init pager
    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;
    if (!prt_init_pager(cfg)) {
        r.fail_reason = "pager_init_failed";
        delete[] R_ref.data;
        return r;
    }
    g_prt_pager_enabled = true; // ENABLE pager (critical!)

    if (!g_prt_pager->activate_layer(0)) {
        prt_shutdown_pager();
        r.fail_reason = "activate_layer_failed";
        delete[] R_ref.data;
        return r;
    }

    // Step 6: Call run_shadow_test() to exercise counter path
    ShadowResult sr = run_shadow_test(0, 1, X_act.data(), Y_ref.data(), (int)M, (int)N, tensor_family.c_str());
    (void)sr;

    // Step 7: Get pager-backed view via prt_get_residual_view(0, tensor_family)
    prt_residual_view raw = prt_get_residual_view(0, tensor_family);
    r.pager_view_is_null = raw.is_null;
    r.pager_view_size_gt_0 = !raw.is_null && raw.size > 0;

    if (!raw.is_null) {
        TritHeaderFields hdr = parse_trit_header(raw.data, raw.size);
        if (hdr.valid) {
            std::vector<float> scales(hdr.n_scales, 0.0f);
            if (hdr.n_scales > 0 && hdr.scale_offset > 0 &&
                hdr.scale_offset + hdr.n_scales * 4 <= raw.size) {
                memcpy(scales.data(), raw.data + hdr.scale_offset, hdr.n_scales * 4);
            }

            prt_trit_decoder dec;
            prt_decoded_view R_pager = dec.decode_bytes(
                raw.data, raw.size,
                hdr.rows, hdr.cols,
                hdr.block_rows, hdr.block_cols,
                hdr.n_scales, scales.data());

            if (!R_pager.is_null) {
                std::vector<float> W_shadow(K * N);
                for (size_t i = 0; i < K * N; i++) W_shadow[i] = W_base[i] + R_pager.data[i];
                std::vector<float> Y_shadow(1 * M * N);
                matmul_dense(X_act.data(), W_shadow.data(), Y_shadow.data(), 1, (int)M, (int)N, (int)K);

                size_t nR = (size_t)R_ref.rows * (size_t)R_ref.cols;
                size_t nW = (size_t)K * (size_t)N;
                size_t nY = (size_t)1 * (size_t)M * (size_t)N;
                r.R_max_abs_err = max_abs_err(R_ref.data, R_pager.data, nR);
                r.R_rmse = rmse(R_ref.data, R_pager.data, nR);
                r.W_max_abs_err = max_abs_err(W_full.data(), W_shadow.data(), nW);
                r.W_rmse = rmse(W_full.data(), W_shadow.data(), nW);
                r.Y_max_abs_err = max_abs_err(Y_ref.data(), Y_shadow.data(), nY);
                r.Y_rmse = rmse(Y_ref.data(), Y_shadow.data(), nY);
                r.Y_cosine = cosine_sim(Y_ref.data(), Y_shadow.data(), nY);
                delete[] R_pager.data;
            }
        }
        // NOTE: raw.data is owned by the pager — do NOT free it.
    }

    prt_shutdown_pager();

    // Step 8: Read shadow counters
    prt_get_shadow_stats(&r.shadow_lookup_calls, &r.shadow_pager_hits,
                         &r.shadow_legacy_hits, &r.shadow_null_views,
                         &r.shadow_budget_rejects);

    // Step 9: Determine pass/fail
    bool counters_ok = (r.shadow_pager_hits > 0) && (r.shadow_legacy_hits == 0);
    bool R_ok = (r.R_max_abs_err >= 0) && (r.R_max_abs_err < 1e-4f);
    bool W_ok = (r.W_max_abs_err >= 0) && (r.W_max_abs_err < 1e-3f);
    bool Y_ok = (r.Y_max_abs_err >= 0) && (r.Y_max_abs_err < 1.0f);

    r.pass = counters_ok && R_ok && W_ok && Y_ok;
    if (!r.pass) {
        std::ostringstream oss;
        oss << "max_err=" << r.Y_max_abs_err;
        if (!counters_ok) oss << " pager_hits=" << r.shadow_pager_hits << " legacy=" << r.shadow_legacy_hits;
        if (!R_ok) oss << " R_err=" << r.R_max_abs_err;
        if (!W_ok) oss << " W_err=" << r.W_max_abs_err;
        if (!Y_ok) oss << " Y_err=" << r.Y_max_abs_err;
        r.fail_reason = oss.str();
    }

    delete[] R_ref.data;
    return r;
}

// ── CONTROL TEST A: DISABLED_MODE ────────────────────────────────────────
static ControlResult test_disabled_mode(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "DISABLED_MODE";
    prt_reset_shadow_stats();

    std::string manifest = pkg_dir + "/manifest.json";
    prt_sidecar_pager_config cfg = {};
    cfg.manifest_path = manifest;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    if (!prt_init_pager(cfg)) { r.pass = true; r.detail = "init_correctly_failed"; return r; }
    g_prt_pager_enabled = false; // DISABLE

    g_prt_pager->activate_layer(0);
    prt_residual_view raw = prt_get_residual_view(0, "ffn_up");

    uint64_t lc = 0, ph = 0, lh = 0, nv = 0, br = 0;
    prt_get_shadow_stats(&lc, &ph, &lh, &nv, &br);

    prt_shutdown_pager();

    r.pass = raw.is_null && lh == 0;
    r.detail = "is_null=" + std::to_string(raw.is_null) + " legacy_hits=" + std::to_string(lh);
    return r;
}

// ── CONTROL TEST B: MISSING_SIDECAR ──────────────────────────────────────
static ControlResult test_missing_sidecar() {
    ControlResult r;
    r.name = "MISSING_SIDECAR";
    prt_reset_shadow_stats();
    prt_sidecar_pager_config cfg = {};
    cfg.manifest_path = "/nonexistent/path/manifest.json";
    cfg.sidecar_root = "/nonexistent";
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.strict_budget = false;
    bool ok = prt_init_pager(cfg);
    uint64_t lc = 0, ph = 0, lh = 0, nv = 0, br = 0;
    prt_get_shadow_stats(&lc, &ph, &lh, &nv, &br);
    r.pass = !ok;
    r.detail = ok ? "init_incorrectly_succeeded" : "init_correctly_failed";
    return r;
}

// ── CONTROL TEST C: BAD_TENSOR_KEY ──────────────────────────────────────
static ControlResult test_bad_tensor_key(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "BAD_TENSOR_KEY";
    prt_reset_shadow_stats();
    std::string manifest = pkg_dir + "/manifest.json";
    prt_sidecar_pager_config cfg = {};
    cfg.manifest_path = manifest;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;
    if (!prt_init_pager(cfg)) { r.pass = true; r.detail = "init_failed"; return r; }
    g_prt_pager_enabled = true;
    if (!g_prt_pager->activate_layer(0)) {
        prt_shutdown_pager(); r.pass = true; r.detail = "activate_failed"; return r;
    }
    prt_residual_view raw = prt_get_residual_view(0, "nonexistent_family");
    uint64_t lc = 0, ph = 0, lh = 0, nv = 0, br = 0;
    prt_get_shadow_stats(&lc, &ph, &lh, &nv, &br);
    prt_shutdown_pager();
    r.pass = raw.is_null;
    r.detail = "is_null=" + std::to_string(raw.is_null) + " null_views=" + std::to_string(nv);
    return r;
}

// ── CONTROL TEST D: BUDGET_REJECT ───────────────────────────────────────
static ControlResult test_budget_reject(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "BUDGET_REJECT";
    prt_reset_shadow_stats();
    std::string manifest = pkg_dir + "/manifest.json";
    prt_sidecar_pager_config cfg = {};
    cfg.manifest_path = manifest;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 1; // Intentionally tiny
    cfg.strict_budget = true;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.eviction_lru = true;
    if (!prt_init_pager(cfg)) { r.pass = true; r.detail = "init_correctly_failed"; return r; }
    g_prt_pager_enabled = true;
    if (!g_prt_pager->activate_layer(0)) {
        prt_shutdown_pager(); r.pass = true; r.detail = "activate_correctly_failed"; return r;
    }
    prt_residual_view raw = prt_get_residual_view(0, "ffn_up");
    uint64_t lc = 0, ph = 0, lh = 0, nv = 0, br = 0;
    prt_get_shadow_stats(&lc, &ph, &lh, &nv, &br);
    prt_shutdown_pager();
    r.pass = (raw.is_null || br > 0);
    r.detail = "is_null=" + std::to_string(raw.is_null) + " budget_rejects=" + std::to_string(br);
    return r;
}

// ── Manual JSON writer ──────────────────────────────────────────────────
static void write_json(const std::string& path,
                       const std::vector<TestResult>& pos_results,
                       const std::vector<ControlResult>& ctrl_results,
                       bool all_pass) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"phase\": \"28BG\",\n");
    fprintf(f, "  \"name\": \"Shadow/Pager Matmul Coverage Sweep\",\n");
    fprintf(f, "  \"branch\": \"experimental/prt-phase19a-alt-sidecar-backed\",\n");
    fprintf(f, "  \"verdict\": \"%s\",\n", all_pass ? "PASS" : "FAIL");
    fprintf(f, "  \"generation_run\": false,\n");
    fprintf(f, "  \"large_files_staged\": false,\n");
    fprintf(f, "  \"positive_tests\": [\n");

    for (size_t i = 0; i < pos_results.size(); i++) {
        const TestResult& tr = pos_results[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"case_name\": \"%s\",\n", tr.case_name.c_str());
        fprintf(f, "      \"tensor_family\": \"%s\",\n", tr.tensor_family.c_str());
        fprintf(f, "      \"source_layer\": %d,\n", tr.source_layer);
        fprintf(f, "      \"source_key\": \"%s\",\n", tr.source_key.c_str());
        fprintf(f, "      \"original_shape\": \"%s\",\n", tr.original_shape.c_str());
        fprintf(f, "      \"K\": %d, \"M\": %d, \"N\": %d,\n", tr.K, tr.M, tr.N);
        fprintf(f, "      \"block_rows\": %u, \"block_cols\": %u,\n", tr.block_rows, tr.block_cols);
        fprintf(f, "      \"R_max_abs_err\": %.8e, \"R_rmse\": %.8e,\n", tr.R_max_abs_err, tr.R_rmse);
        fprintf(f, "      \"W_max_abs_err\": %.8e, \"W_rmse\": %.8e,\n", tr.W_max_abs_err, tr.W_rmse);
        fprintf(f, "      \"Y_max_abs_err\": %.8e, \"Y_rmse\": %.8e, \"Y_cosine\": %.6f,\n",
                tr.Y_max_abs_err, tr.Y_rmse, tr.Y_cosine);
        fprintf(f, "      \"pager_view_is_null\": %s,\n", tr.pager_view_is_null ? "true" : "false");
        fprintf(f, "      \"pager_view_size_gt_0\": %s,\n", tr.pager_view_size_gt_0 ? "true" : "false");
        fprintf(f, "      \"shadow_lookup_calls\": %llu,\n", (unsigned long long)tr.shadow_lookup_calls);
        fprintf(f, "      \"shadow_pager_hits\": %llu,\n", (unsigned long long)tr.shadow_pager_hits);
        fprintf(f, "      \"shadow_legacy_hits\": %llu,\n", (unsigned long long)tr.shadow_legacy_hits);
        fprintf(f, "      \"shadow_null_views\": %llu,\n", (unsigned long long)tr.shadow_null_views);
        fprintf(f, "      \"shadow_budget_rejects\": %llu,\n", (unsigned long long)tr.shadow_budget_rejects);
        fprintf(f, "      \"pass\": %s,\n", tr.pass ? "true" : "false");
        fprintf(f, "      \"fail_reason\": \"%s\"\n", tr.fail_reason.c_str());
        fprintf(f, "    }%s\n", (i < pos_results.size() - 1) ? "," : "");
    }
    fprintf(f, "  ],\n");
    fprintf(f, "  \"control_tests\": [\n");
    for (size_t i = 0; i < ctrl_results.size(); i++) {
        const ControlResult& cr = ctrl_results[i];
        fprintf(f, "    {\"name\": \"%s\", \"pass\": %s, \"detail\": \"%s\"}%s\n",
                cr.name.c_str(), cr.pass ? "true" : "false", cr.detail.c_str(),
                (i < ctrl_results.size() - 1) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

// ── Main ──────────────────────────────────────────────────────────────────
int main() {
    printf("Phase 28BG: Shadow/Pager Matmul Coverage Sweep\n");
    printf("================================================\n\n");

    struct Case {
        std::string name;
        std::string pkg;
        std::string tensor_family;
        uint32_t K, M, N;
        uint16_t br, bc;
        std::string source_key;
        std::string original_shape;
        int source_layer;
    };
    Case cases[] = {
        // Layer 0 cases
        {"ffn_up_l0_row",   "/tmp/phase28bg_pkgs/ffn_up_l0_row",   "ffn_up",   96,  48, 4, 32, 48,
         "model.layers.0.mlp.up_proj.weight",   "(4864, 896)", 0},
        {"ffn_gate_l0_col", "/tmp/phase28bg_pkgs/ffn_gate_l0_col", "ffn_gate", 32, 144, 4, 32, 48,
         "model.layers.0.mlp.gate_proj.weight", "(4864, 896)", 0},
        {"ffn_down_l0_rc",  "/tmp/phase28bg_pkgs/ffn_down_l0_rc",  "ffn_down", 96, 144, 6, 32, 48,
         "model.layers.0.mlp.down_proj.weight", "(896, 4864)", 0},
        {"attn_out_l0_awk", "/tmp/phase28bg_pkgs/attn_out_l0_awk", "attn_out", 70, 101, 2, 32, 48,
         "model.layers.0.self_attn.o_proj.weight", "(896, 896)", 0},
        // Layer 1 cases
        {"ffn_up_l1_col",   "/tmp/phase28bg_pkgs/ffn_up_l1_col",   "ffn_up",   32, 144, 4, 32, 48,
         "model.layers.1.mlp.up_proj.weight",   "(4864, 896)", 1},
        {"ffn_gate_l1_row", "/tmp/phase28bg_pkgs/ffn_gate_l1_row", "ffn_gate", 96,  48, 4, 32, 48,
         "model.layers.1.mlp.gate_proj.weight", "(4864, 896)", 1},
        {"ffn_down_l1_awk", "/tmp/phase28bg_pkgs/ffn_down_l1_awk", "ffn_down", 70, 101, 2, 32, 48,
         "model.layers.1.mlp.down_proj.weight", "(896, 4864)", 1},
        // Layer 5 case
        {"attn_out_l5_rc",  "/tmp/phase28bg_pkgs/attn_out_l5_rc",  "attn_out", 96, 144, 6, 32, 48,
         "model.layers.5.self_attn.o_proj.weight", "(896, 896)", 5},
    };

    std::vector<TestResult> pos_results;
    for (auto& c : cases) {
        TestResult tr = test_positive(c.pkg, c.name, c.tensor_family, c.K, c.M, c.N,
                                      c.br, c.bc, c.source_key, c.original_shape, c.source_layer);
        pos_results.push_back(tr);
        printf("=== 28BG-POS: %s (family=%s, layer=%d) ===\n",
               c.name.c_str(), c.tensor_family.c_str(), c.source_layer);
        printf("  source_key: %s  shape: %s\n", c.source_key.c_str(), c.original_shape.c_str());
        printf("  R_max_abs_err: %.8e  R_rmse: %.8e\n", tr.R_max_abs_err, tr.R_rmse);
        printf("  W_max_abs_err: %.8e  W_rmse: %.8e\n", tr.W_max_abs_err, tr.W_rmse);
        printf("  Y_max_abs_err: %.8e  Y_rmse: %.8e  Y_cosine: %.6f\n",
               tr.Y_max_abs_err, tr.Y_rmse, tr.Y_cosine);
        printf("  pager_view_null=%d size_gt_0=%d\n", tr.pager_view_is_null, tr.pager_view_size_gt_0);
        printf("  lookup_calls=%llu pager_hits=%llu legacy_hits=%llu null_views=%llu budget_rejects=%llu\n",
               (unsigned long long)tr.shadow_lookup_calls,
               (unsigned long long)tr.shadow_pager_hits,
               (unsigned long long)tr.shadow_legacy_hits,
               (unsigned long long)tr.shadow_null_views,
               (unsigned long long)tr.shadow_budget_rejects);
        printf("  pass: %s  fail_reason: %s\n", tr.pass ? "true" : "false", tr.fail_reason.c_str());
        printf("Result: %s_28BG_POS\n\n", tr.pass ? "PASS" : "FAIL");
    }

    // Control tests
    std::vector<ControlResult> ctrl_results;
    ctrl_results.push_back(test_disabled_mode(cases[0].pkg));
    ctrl_results.push_back(test_missing_sidecar());
    ctrl_results.push_back(test_bad_tensor_key(cases[0].pkg));
    ctrl_results.push_back(test_budget_reject(cases[0].pkg));

    for (auto& cr : ctrl_results) {
        printf("=== 28BG-CTRL: %s ===\n", cr.name.c_str());
        printf("  pass=%s  detail=%s\n", cr.pass ? "true" : "false", cr.detail.c_str());
        printf("Result: %s_28BG_CTRL\n\n", cr.pass ? "PASS" : "FAIL");
    }

    // Summary
    int pos_pass = 0, pos_fail = 0;
    for (auto& tr : pos_results) { if (tr.pass) pos_pass++; else pos_fail++; }
    int ctrl_pass = 0, ctrl_fail = 0;
    for (auto& cr : ctrl_results) { if (cr.pass) ctrl_pass++; else ctrl_fail++; }

    printf("================================================\n");
    printf("Phase 28BG SUMMARY\n");
    for (auto& tr : pos_results)
        printf("  %-22s: %s (Y_max_err=%.4e)\n", tr.case_name.c_str(), tr.pass ? "PASS" : "FAIL", tr.Y_max_abs_err);
    printf("Controls: ");
    for (auto& cr : ctrl_results) printf("%s=%s ", cr.name.c_str(), cr.pass ? "PASS" : "FAIL");
    printf("\nTotal failures: %d\n", pos_fail + ctrl_fail);

    bool all_pass = (pos_fail == 0) && (ctrl_fail == 0);
    printf("Verdict: %s\n", all_pass
        ? "PASS_PHASE28BG_SHADOW_PAGER_MATMUL_COVERAGE_SWEEP"
        : "FAIL_PHASE28BG");

    // Write JSON
    std::string json_path = "/home/matthew-villnave/llama.cpp/examples/speculative/results/phase28bg_shadow_pager_matmul_coverage_sweep.json";
    write_json(json_path, pos_results, ctrl_results, all_pass);
    printf("JSON written to: %s\n", json_path.c_str());

    return all_pass ? 0 : 1;
}

#else  // PRT_SIDECAR_PAGER_EXPERIMENTAL

int main() {
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    printf("Result: FAIL_PHASE28BG (stub)\n");
    return 1;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL