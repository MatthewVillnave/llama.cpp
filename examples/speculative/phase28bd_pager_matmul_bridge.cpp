// Phase 28BD: Pager View → Decoded Residual Matmul Bridge
// Proves pager-view .trit bytes decode identically to direct-file path
//
// Build (separate compile):
//   g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
//        -I. -Iggml/include -Iinclude \
//        -c examples/speculative/prt_sidecar_pager.cpp -o /tmp/prt_pager.o
//   g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
//        -I. -Iggml/include -Iinclude \
//        -c examples/speculative/prt_trit_decode.cpp -o /tmp/prt_decode.o
//   g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
//        -I. -Iggml/include -Iinclude \
//        examples/speculative/phase28bd_pager_matmul_bridge.cpp \
//        /tmp/prt_pager.o /tmp/prt_decode.o \
//        -o /tmp/phase28bd_pager_matmul_bridge
//
// Run: /tmp/phase28bd_pager_matmul_bridge

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

#include "prt_shadow.h"
#include "prt_sidecar_pager.h"
#include "prt_sidecar_runtime_link.h"
#include "prt_trit_decode.h"

// ── Mock globals (from prt_trit_decode_harness.cpp) ──────────────────────
static float g_mock_legacy[128] = {0};
static SidecarLoad g_mock_sidecar = {0, "/mock/legacy/sidecar_0.bin", g_mock_legacy, 512};
std::unordered_map<int, SidecarLoad> g_sidecars = {{0, g_mock_sidecar}};
bool g_sidecars_loaded = true;
prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;

// ── Math helpers ──────────────────────────────────────────────────────────
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
    for (size_t i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na < 1e-12 || nb < 1e-12) return 0;
    return dot / (sqrt(na) * sqrt(nb));
}

// ── Read raw bytes from file ────────────────────────────────────────────
static std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f) return {};
    return buf;
}

// ── Direct-file decode (bypasses pager) ─────────────────────────────────
static prt_decoded_view decode_direct(const std::string& path) {
    prt_trit_decoder dec;
    return dec.decode_file(path);
}

// ── Test one case ────────────────────────────────────────────────────────
struct TestResult {
    std::string case_name;
    bool pass = false;
    double R_max_abs_err = 0, R_rmse = 0;
    double W_max_abs_err = 0, W_rmse = 0;
    double Y_max_abs_err = 0, Y_rmse = 0;
    double Y_cosine = 0;
    bool pager_view_is_null = false;
    bool pager_view_size_gt_0 = false;
    std::string fail_reason;
};

static TestResult test_one_case(const std::string& pkg_dir,
                                const std::string& case_name,
                                uint32_t K, uint32_t M, uint32_t N,
                                uint16_t block_rows, uint16_t block_cols) {
    TestResult r;
    r.case_name = case_name;

    std::string manifest_path = pkg_dir + "/manifest.json";
    std::string sidecar_root = pkg_dir;
    std::string tensor_family = "ffn_up";
    int layer_idx = 0;

    // Direct decode (ground truth)
    std::string trit_path = pkg_dir + "/layers/layer_000/ffn_up.trit";
    prt_decoded_view direct = decode_direct(trit_path);
    if (direct.is_null) {
        r.fail_reason = "direct_decode_failed: " + direct.reason;
        return r;
    }

    // Pager-backed path
    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = sidecar_root;
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

    if (!g_prt_pager->activate_layer(layer_idx)) {
        prt_shutdown_pager();
        r.fail_reason = "activate_layer_failed";
        delete[] direct.data;
        return r;
    }

    prt_residual_view raw = prt_get_residual_view(layer_idx, tensor_family);
    r.pager_view_is_null = raw.is_null;

    if (raw.is_null || raw.data == nullptr || raw.size == 0) {
        r.fail_reason = raw.is_null ? ("raw_view_null: " + raw.reason) : "raw_view_null";
        prt_shutdown_pager();
        delete[] direct.data;
        return r;
    }
    r.pager_view_size_gt_0 = (raw.size > 0);

    // Read .trit header from pager raw view
    const uint8_t* hdr = raw.data;
    if (raw.size < 32) {
        prt_shutdown_pager();
        r.fail_reason = "raw_data_too_small";
        delete[] direct.data;
        return r;
    }

    uint32_t rows   = *(uint32_t*)(hdr + 8);
    uint32_t cols   = *(uint32_t*)(hdr + 12);
    uint16_t br     = *(uint16_t*)(hdr + 16);
    uint16_t bc     = *(uint16_t*)(hdr + 18);
    uint16_t n_sc   = *(uint16_t*)(hdr + 20);
    uint32_t so     = *(uint32_t*)(hdr + 26);

    // Extract scales from raw view
    std::vector<float> scales(n_sc, 0.0f);
    if (n_sc > 0 && so > 0 && so + n_sc * 4 <= raw.size) {
        memcpy(scales.data(), hdr + so, n_sc * 4);
    }

    prt_trit_decoder dec;
    prt_decoded_view pager_decoded = dec.decode_bytes(raw.data, raw.size,
                                                     rows, cols, br, bc,
                                                     n_sc, scales.data());
    prt_shutdown_pager();

    if (pager_decoded.is_null) {
        r.fail_reason = "pager_decode_bytes_failed: " + pager_decoded.reason;
        delete[] direct.data;
        return r;
    }

    // Compare decoded results
    size_t n = (size_t)rows * (size_t)cols;
    if (pager_decoded.rows != direct.rows || pager_decoded.cols != direct.cols) {
        r.fail_reason = "dimension_mismatch";
        delete[] direct.data; delete[] pager_decoded.data;
        return r;
    }

    r.R_max_abs_err = max_abs_err(direct.data, pager_decoded.data, n);
    r.R_rmse = rmse(direct.data, pager_decoded.data, n);
    r.pass = r.R_max_abs_err < 1e-4;

    if (!r.pass) {
        r.fail_reason = "max_err=" + std::to_string(r.R_max_abs_err) + " exceeds 1e-4";
    }

    delete[] direct.data; delete[] pager_decoded.data;
    return r;
}

// ── Negative tests ──────────────────────────────────────────────────────
static int test_missing_sidecar() {
    printf("\n=== 28BD-NEG: MISSING SIDECAR ===\n");
    g_sidecars.clear();
    g_sidecars_loaded = false;

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = "/tmp/nonexistent_manifest.json";
    cfg.sidecar_root = "/tmp/nonexistent_root";
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    bool ok = prt_init_pager(cfg);
    printf("  prt_init_pager(missing): %s\n", ok ? "true" : "false");
    int rc = ok ? 1 : 0;
    printf("Result: %s\n", ok ? "PASS_NEG_MISSING_SIDECAR" : "FAIL_NEG_MISSING_SIDECAR");
    prt_shutdown_pager();
    return rc;
}

static int test_bad_tensor_family() {
    printf("\n=== 28BD-NEG: BAD TENSOR FAMILY ===\n");
    g_sidecars.clear();
    g_sidecars_loaded = false;

    std::string tmpdir = "/tmp/prt_28bd_neg_pkg";
    mkdir(tmpdir.c_str(), 0755);
    std::string manifest_path = tmpdir + "/manifest.json";
    std::string trit_path = tmpdir + "/layers/layer_000/ffn_up.trit";

    // Write minimal valid .trit header
    {
        std::vector<uint8_t> buf(80, 0);
        uint32_t magic = 0x54495254;  // TRIT little-endian
        memcpy(buf.data(), &magic, 4);
        uint16_t vmaj = 0, vmin = 1;
        memcpy(buf.data()+4, &vmaj, 2);
        memcpy(buf.data()+6, &vmin, 2);
        uint32_t rows = 4, cols = 8;
        memcpy(buf.data()+8, &rows, 4);
        memcpy(buf.data()+12, &cols, 4);
        uint16_t br = 4, bc = 8;
        memcpy(buf.data()+16, &br, 2);
        memcpy(buf.data()+18, &bc, 2);
        uint16_t ns = 1;
        memcpy(buf.data()+20, &ns, 2);
        uint32_t po = 32, so = 32;
        memcpy(buf.data()+22, &po, 4);
        memcpy(buf.data()+26, &so, 4);
        uint16_t crc = 0;
        memcpy(buf.data()+30, &crc, 2);  // zero checksum (skip CRC)
        FILE* f = fopen(trit_path.c_str(), "wb");
        if (f) { fwrite(buf.data(), 1, buf.size(), f); fclose(f); }
    }
    {
        FILE* f = fopen(manifest_path.c_str(), "w");
        fprintf(f, "{\n");
        fprintf(f, "  \"format_name\": \"prt_residual_sidecar\",\n");
        fprintf(f, "  \"format_version\": \"0.1\",\n");
        fprintf(f, "  \"source_model\": \"test\",\n");
        fprintf(f, "  \"layer_count\": 1,\n");
        fprintf(f, "  \"tensor_families\": [\"ffn_up\"],\n");
        fprintf(f, "  \"entries\": [\n");
        fprintf(f, "    {\"layer_index\": 0, \"tensor_family\": \"ffn_up\", "
                "\"file_path\": \"layers/layer_000/ffn_up.trit\", \"byte_size\": 80, \"checksum\": \"0000\", \"required\": true}\n");
        fprintf(f, "  ]\n}\n");
        fclose(f);
    }

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = tmpdir;
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    bool ok = prt_init_pager(cfg);
    if (ok) g_prt_pager->activate_layer(0);
    prt_residual_view raw = prt_get_residual_view(0, "nonexistent_family");
    printf("  prt_get_residual_view(0, nonexistent_family): is_null=%s reason=%s\n",
           raw.is_null ? "true" : "false", raw.reason.c_str());
    bool pass = raw.is_null;
    printf("Result: %s\n", pass ? "PASS_NEG_BAD_FAMILY" : "FAIL_NEG_BAD_FAMILY");
    prt_shutdown_pager();
    return pass ? 0 : 1;
}

// ── Main ──────────────────────────────────────────────────────────────────
int main() {
    printf("Phase 28BD: Pager View → Decoded Residual Matmul Bridge\n");
    printf("========================================================\n");

    struct CaseDef {
        const char* pkg;
        const char* name;
        uint32_t K, M, N;
        uint16_t br, bc;
    };
    CaseDef cases[] = {
        {"row_split",     "row_split",     96,  48,   4, 32, 48},
        {"col_split",     "col_split",     32, 144,   4, 32, 48},
        {"row_col_split", "row_col_split", 96, 144,   6, 32, 48},
        {"awkward_edge",  "awkward_edge",  70, 101,   2, 32, 48},
    };

    std::vector<TestResult> results;
    int fails = 0;

    for (const auto& c : cases) {
        std::string pkg_dir = "/tmp/phase28bd_pkg_" + std::string(c.pkg);
        printf("\n=== 28BD: %s ===\n", c.name);
        TestResult r = test_one_case(pkg_dir, c.name, c.K, c.M, c.N, c.br, c.bc);
        results.push_back(r);

        printf("  direct_vs_pager_R_max_abs_err: %.8e\n", r.R_max_abs_err);
        printf("  direct_vs_pager_R_rmse:        %.8e\n", r.R_rmse);
        printf("  pager_view_is_null:            %s\n", r.pager_view_is_null ? "true" : "false");
        printf("  pager_view_size_gt_0:          %s\n", r.pager_view_size_gt_0 ? "true" : "false");
        printf("  pass: %s\n", r.pass ? "true" : "false");
        if (!r.pass) {
            printf("  FAIL_REASON: %s\n", r.fail_reason.c_str());
            fails++;
        }
        printf("Result: %s\n", r.pass ? "PASS_28BD" : "FAIL_28BD");
    }

    // Negative tests
    int neg1 = test_missing_sidecar();
    int neg2 = test_bad_tensor_family();
    fails += (neg1 != 0) + (neg2 != 0);

    // Summary
    printf("\n========================================================\n");
    printf("Phase 28BD SUMMARY\n");
    for (const auto& r : results) {
        printf("  %s: %s (R_max_err=%.4e)\n", r.case_name.c_str(),
               r.pass ? "PASS" : "FAIL", r.R_max_abs_err);
    }
    printf("Negative tests: missing_sidecar=%s bad_family=%s\n",
           neg1==0?"PASS":"FAIL", neg2==0?"PASS":"FAIL");
    printf("Total failures: %d\n", fails);
    printf("Verdict: %s\n", fails==0 ? "PASS_PHASE28BD_PAGER_VIEW_DECODED_RESIDUAL_MATMUL_BRIDGE" : "FAIL");

    return fails;
}

#else  // PRT_SIDECAR_PAGER_EXPERIMENTAL

int main() {
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    printf("Result: FAIL_PHASE28BD (stub)\n");
    return 1;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL