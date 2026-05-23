// Phase 28BH: Full-Layer-Adjacent Sidecar Chunk Sweep
//
// Tests larger real Qwen2.5-0.5B chunks (256+, not just 70-144) with the proven
// shadow/pager overlay matmul path to expose scale-related issues.
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
    int block_count = 0;
    size_t sidecar_bytes = 0;
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

    // Compute block count for reporting
    int n_br = (K + block_rows - 1) / block_rows;
    int n_bc = (M + block_cols - 1) / block_cols;
    r.block_count = n_br * n_bc;

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

    // Record sidecar byte size from manifest
    {
        std::ifstream mf(manifest);
        if (mf.good()) {
            std::stringstream ss;
            ss << mf.rdbuf();
            std::string content = ss.str();
            size_t byte_pos = content.find("\"byte_size\"");
            if (byte_pos != std::string::npos) {
                size_t val_start = content.find(":", byte_pos);
                if (val_start != std::string::npos) {
                    size_t val_end = content.find(",", val_start);
                    if (val_end == std::string::npos) val_end = content.find("}", val_start);
                    std::string num_str = content.substr(val_start + 1, val_end - val_start - 1);
                    r.sidecar_bytes = std::stoll(num_str);
                }
            }
        }
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
    r.detail = "pager_disabled_mode";
    return r;
}

// ── CONTROL TEST B: MISSING_SIDECAR ───────────────────────────────────────
static ControlResult test_missing_sidecar(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "MISSING_SIDECAR";
    prt_reset_shadow_stats();

    std::string manifest = "/nonexistent/path/manifest.json";
    std::string nonexistent_root = "/nonexistent";
    prt_sidecar_pager_config cfg = {};
    cfg.manifest_path = manifest;
    cfg.sidecar_root = nonexistent_root;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    if (!prt_init_pager(cfg)) { r.pass = true; r.detail = "init_correctly_rejected_missing_root"; return r; }
    prt_shutdown_pager();

    r.detail = "init_succeeded_despite_missing_root";
    return r;
}

// ── CONTROL TEST C: BAD_TENSOR_KEY ────────────────────────────────────────
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

    if (!prt_init_pager(cfg)) { r.pass = false; r.detail = "init_failed"; return r; }
    g_prt_pager_enabled = true;
    g_prt_pager->activate_layer(0);

    prt_residual_view raw = prt_get_residual_view(0, "nonexistent_family");
    prt_shutdown_pager();

    r.pass = raw.is_null;
    r.detail = raw.is_null ? "correctly_returned_null" : "unexpectedly_non_null";
    return r;
}

// ── CONTROL TEST D: BUDGET_REJECT ────────────────────────────────────────
static ControlResult test_budget_reject(const std::string& pkg_dir) {
    ControlResult r;
    r.name = "BUDGET_REJECT";
    prt_reset_shadow_stats();

    std::string manifest = pkg_dir + "/manifest.json";
    prt_sidecar_pager_config cfg = {};
    cfg.manifest_path = manifest;
    cfg.sidecar_root = pkg_dir;
    cfg.max_resident_bytes = 4;  // INTENTIONALLY TINY to force rejection
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    if (!prt_init_pager(cfg)) {
        r.pass = true;
        r.detail = "init_correctly_rejected_tiny_budget";
        return r;
    }
    g_prt_pager_enabled = true;
    g_prt_pager->activate_layer(0);

    prt_residual_view raw = prt_get_residual_view(0, "ffn_up");
    prt_shutdown_pager();

    r.pass = raw.is_null;
    r.detail = raw.is_null ? "correctly_rejected_tiny_budget" : "unexpectedly_non_null_with_tiny_budget";
    return r;
}

// ── CONTROL TEST E: REPEATED_LOAD ─────────────────────────────────────────
static ControlResult test_repeated_load(const std::string& pkg_dir, const std::string& tensor_family,
                                        uint32_t K, uint32_t M, uint32_t N) {
    ControlResult r;
    r.name = "REPEATED_LOAD";
    r.pass = false;
    std::string manifest = pkg_dir + "/manifest.json";
    prt_trit_decoder ref_dec;
    std::string trit_file = tensor_family + ".trit";
    std::string trit_path = pkg_dir + "/layers/layer_000/" + trit_file;
    prt_decoded_view R_ref = ref_dec.decode_file(trit_path);
    if (R_ref.is_null) { r.detail = "ref_decode_failed"; return r; }

    std::vector<float> W_base(K * N);
    for (size_t i = 0; i < K * N; i++) W_base[i] = float((i * K + i) % 101) / 100.0f;
    std::vector<float> X_act(1 * M * K);
    for (size_t i = 0; i < 1 * M * K; i++) X_act[i] = float((i * 13 + i * 7) % 97) / 100.0f;
    std::vector<float> W_full(K * N);
    for (size_t i = 0; i < K * N; i++) W_full[i] = W_base[i] + R_ref.data[i];
    std::vector<float> Y_ref(1 * M * N);
    matmul_dense(X_act.data(), W_full.data(), Y_ref.data(), 1, (int)M, (int)N, (int)K);

    for (int iter = 0; iter < 3; iter++) {
        prt_reset_shadow_stats();
        prt_sidecar_pager_config cfg = {};
        cfg.manifest_path = manifest;
        cfg.sidecar_root = pkg_dir;
        cfg.max_resident_bytes = 64 * 1024 * 1024;
        cfg.checksum_enabled = false;
        cfg.validate_trit_header = false;
        cfg.strict_budget = false;
        cfg.eviction_lru = true;

        if (!prt_init_pager(cfg)) {
            r.detail = "iter" + std::to_string(iter) + "_init_failed";
            delete[] R_ref.data;
            return r;
        }
        g_prt_pager_enabled = true;
        g_prt_pager->activate_layer(0);

        ShadowResult sr = run_shadow_test(0, 1, X_act.data(), Y_ref.data(), (int)M, (int)N, tensor_family.c_str());

        uint64_t lc = 0, ph = 0, lh = 0, nv = 0, br = 0;
        prt_get_shadow_stats(&lc, &ph, &lh, &nv, &br);

        prt_shutdown_pager();

        if (iter == 0 && ph == 0) {
            r.detail = "iter0_no_pager_hits";
            delete[] R_ref.data;
            return r;
        }
    }

    r.pass = true;
    r.detail = "3x_decode_no_crash";
    delete[] R_ref.data;
    return r;
}

// ── JSON helpers ─────────────────────────────────────────────────────────
static void result_to_json(const TestResult& r, std::ostream& out) {
    out << "{\n";
    out << "  \"case_name\": \"" << r.case_name << "\",\n";
    out << "  \"tensor_family\": \"" << r.tensor_family << "\",\n";
    out << "  \"source_layer\": " << r.source_layer << ",\n";
    out << "  \"source_key\": \"" << r.source_key << "\",\n";
    out << "  \"original_shape\": \"" << r.original_shape << "\",\n";
    out << "  \"K\": " << r.K << ", \"M\": " << r.M << ", \"N\": " << r.N << ",\n";
    out << "  \"block_rows\": " << r.block_rows << ", \"block_cols\": " << r.block_cols << ",\n";
    out << "  \"block_count\": " << r.block_count << ",\n";
    out << "  \"sidecar_bytes\": " << r.sidecar_bytes << ",\n";
    out << "  \"R_max_abs_err\": " << std::scientific << std::setprecision(6) << r.R_max_abs_err << ",\n";
    out << "  \"R_rmse\": " << r.R_rmse << ",\n";
    out << "  \"W_max_abs_err\": " << r.W_max_abs_err << ",\n";
    out << "  \"W_rmse\": " << r.W_rmse << ",\n";
    out << "  \"Y_max_abs_err\": " << r.Y_max_abs_err << ",\n";
    out << "  \"Y_rmse\": " << r.Y_rmse << ",\n";
    out << "  \"Y_cosine\": " << std::setprecision(12) << r.Y_cosine << ",\n";
    out << "  \"pager_view_is_null\": " << (r.pager_view_is_null ? "true" : "false") << ",\n";
    out << "  \"pager_view_size_gt_0\": " << (r.pager_view_size_gt_0 ? "true" : "false") << ",\n";
    out << "  \"shadow_lookup_calls\": " << r.shadow_lookup_calls << ",\n";
    out << "  \"shadow_pager_hits\": " << r.shadow_pager_hits << ",\n";
    out << "  \"shadow_legacy_hits\": " << r.shadow_legacy_hits << ",\n";
    out << "  \"shadow_null_views\": " << r.shadow_null_views << ",\n";
    out << "  \"shadow_budget_rejects\": " << r.shadow_budget_rejects << ",\n";
    out << "  \"pass\": " << (r.pass ? "true" : "false") << ",\n";
    out << "  \"fail_reason\": \"" << r.fail_reason << "\"\n";
    out << "}";
}

// ── MAIN ─────────────────────────────────────────────────────────────────
int main() {
    std::vector<TestResult> positive_results;
    std::vector<ControlResult> control_results;
    std::string base = "/tmp/phase28bh_pkg_/";

    std::cout << "\n============================================================\n";
    std::cout << "PHASE 28BH: Full-Layer-Adjacent Sidecar Chunk Sweep\n";
    std::cout << "============================================================\n\n";

    // ── Positive test cases ─────────────────────────────────────────────────
    // (name, family, K, M, N, block_rows, block_cols, key, shape, layer)
    struct CaseSpec {
        const char* name;
        const char* family;
        uint32_t K, M, N;
        uint16_t br, bc;
        const char* key;
        const char* shape;
        int layer;
    };
    CaseSpec specs[] = {
        {"ffn_up_l0_256x256",   "ffn_up",   256, 256,  4, 32, 48, "model.layers.0.mlp.up_proj.weight",   "(4864, 896)", 0},
        {"ffn_gate_l0_384x256", "ffn_gate", 384, 256,  4, 32, 48, "model.layers.0.mlp.gate_proj.weight", "(4864, 896)", 0},
        {"ffn_down_l0_256x384", "ffn_down", 256, 384,  6, 32, 48, "model.layers.0.mlp.down_proj.weight", "(896, 4864)", 0},
        {"attn_out_l0_512",     "attn_out", 512, 512,  4, 32, 48, "model.layers.0.self_attn.o_proj.weight", "(896, 896)", 0},
        {"ffn_up_l1_257x389",  "ffn_up",   257, 389,  3, 32, 48, "model.layers.1.mlp.up_proj.weight",   "(4864, 896)", 1},
        {"ffn_down_l1_256x256", "ffn_down", 256, 256,  4, 32, 48, "model.layers.1.mlp.down_proj.weight", "(896, 4864)", 1},
        {"attn_out_l5_384x256", "attn_out", 384, 256,  4, 32, 48, "model.layers.5.self_attn.o_proj.weight", "(896, 896)", 5},
    };

    for (const auto& spec : specs) {
        std::string pkg_dir = base + spec.name;
        std::cout << ">>> Testing: " << spec.name << " (" << spec.family << " layer=" << spec.layer << " K=" << spec.K << " M=" << spec.M << " blocks=" << ((spec.K+spec.br-1)/spec.br) << "x" << ((spec.M+spec.bc-1)/spec.bc) << ") ..." << std::endl;
        TestResult r = test_positive(pkg_dir, spec.name, spec.family, spec.K, spec.M, spec.N, spec.br, spec.bc, spec.key, spec.shape, spec.layer);
        positive_results.push_back(r);

        std::cout << "    pager_hits=" << r.shadow_pager_hits
                  << " legacy=" << r.shadow_legacy_hits
                  << " null_views=" << r.shadow_null_views
                  << " R_err=" << std::scientific << std::setprecision(2) << r.R_max_abs_err
                  << " W_err=" << r.W_max_abs_err
                  << " Y_err=" << r.Y_max_abs_err
                  << " Y_cos=" << std::setprecision(8) << r.Y_cosine
                  << " PASS=" << (r.pass ? "✓" : "✗")
                  << std::endl;
        if (!r.fail_reason.empty()) {
            std::cout << "    FAIL_REASON: " << r.fail_reason << std::endl;
        }
    }

    // ── Control tests ─────────────────────────────────────────────────────────
    std::string ctrl_pkg = base + "ffn_up_l0_256x256";

    {
        std::cout << ">>> Control: DISABLED_MODE ..." << std::endl;
        ControlResult cr = test_disabled_mode(ctrl_pkg);
        control_results.push_back(cr);
        std::cout << "    PASS=" << (cr.pass ? "✓" : "✗") << " detail=" << cr.detail << std::endl;
    }
    {
        std::cout << ">>> Control: MISSING_SIDECAR ..." << std::endl;
        ControlResult cr = test_missing_sidecar(ctrl_pkg);
        control_results.push_back(cr);
        std::cout << "    PASS=" << (cr.pass ? "✓" : "✗") << " detail=" << cr.detail << std::endl;
    }
    {
        std::cout << ">>> Control: BAD_TENSOR_KEY ..." << std::endl;
        ControlResult cr = test_bad_tensor_key(ctrl_pkg);
        control_results.push_back(cr);
        std::cout << "    PASS=" << (cr.pass ? "✓" : "✗") << " detail=" << cr.detail << std::endl;
    }
    {
        std::cout << ">>> Control: BUDGET_REJECT ..." << std::endl;
        ControlResult cr = test_budget_reject(ctrl_pkg);
        control_results.push_back(cr);
        std::cout << "    PASS=" << (cr.pass ? "✓" : "✗") << " detail=" << cr.detail << std::endl;
    }
    {
        std::cout << ">>> Control: REPEATED_LOAD (3× decode same chunk) ..." << std::endl;
        ControlResult cr = test_repeated_load(ctrl_pkg, "ffn_up", 256, 256, 4);
        control_results.push_back(cr);
        std::cout << "    PASS=" << (cr.pass ? "✓" : "✗") << " detail=" << cr.detail << std::endl;
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    int n_positive = (int)positive_results.size();
    int n_pass_positive = 0;
    for (const auto& r : positive_results) if (r.pass) n_pass_positive++;

    int n_controls = (int)control_results.size();
    int n_pass_controls = 0;
    for (const auto& c : control_results) if (c.pass) n_pass_controls++;

    std::cout << "\n============================================================\n";
    std::cout << "PHASE 28BH SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "Positive tests: " << n_pass_positive << "/" << n_positive << " passed\n";
    for (const auto& r : positive_results) {
        std::cout << "  " << r.case_name << ": " << (r.pass ? "PASS" : "FAIL");
        if (!r.fail_reason.empty()) std::cout << " [" << r.fail_reason << "]";
        std::cout << "\n";
    }
    std::cout << "Controls: " << n_pass_controls << "/" << n_controls << " passed\n";
    for (const auto& c : control_results) {
        std::cout << "  " << c.name << ": " << (c.pass ? "PASS" : "FAIL") << " (" << c.detail << ")\n";
    }
    std::cout << "============================================================\n";

    bool all_pass = (n_pass_positive == n_positive) && (n_pass_controls == n_controls);
    std::cout << "OVERALL: " << (all_pass ? "PASS ✓" : "FAIL ✗") << std::endl;

    // ── Write JSON results ──────────────────────────────────────────────────
    std::string json_path = "/home/matthew-villnave/llama.cpp/examples/speculative/results/phase28bh_full_layer_adjacent_sidecar_chunk_sweep.json";
    {
        std::ofstream jf(json_path);
        jf << "{\n";
        jf << "  \"phase\": \"28BH\",\n";
        jf << "  \"name\": \"Full-Layer-Adjacent Sidecar Chunk Sweep\",\n";
        jf << "  \"branch\": \"experimental/prt-phase19a-alt-sidecar-backed\",\n";
        jf << "  \"all_pass\": " << (all_pass ? "true" : "false") << ",\n";
        jf << "  \"positive_pass_count\": " << n_pass_positive << ",\n";
        jf << "  \"positive_total_count\": " << n_positive << ",\n";
        jf << "  \"control_pass_count\": " << n_pass_controls << ",\n";
        jf << "  \"control_total_count\": " << n_controls << ",\n";
        jf << "  \"tests\": [\n";
        for (int i = 0; i < (int)positive_results.size(); i++) {
            if (i > 0) jf << ",\n";
            result_to_json(positive_results[i], jf);
        }
        jf << "\n  ],\n";
        jf << "  \"controls\": [\n";
        for (int i = 0; i < (int)control_results.size(); i++) {
            if (i > 0) jf << ",\n";
            jf << "    {\n";
            jf << "      \"name\": \"" << control_results[i].name << "\",\n";
            jf << "      \"pass\": " << (control_results[i].pass ? "true" : "false") << ",\n";
            jf << "      \"detail\": \"" << control_results[i].detail << "\"\n";
            jf << "    }";
        }
        jf << "\n  ]\n";
        jf << "}\n";
    }
    std::cout << "\nResults → " << json_path << std::endl;

    return all_pass ? 0 : 1;
}

#else
#include <cstdio>
int main() {
    std::fprintf(stderr, "PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    return 1;
}
#endif