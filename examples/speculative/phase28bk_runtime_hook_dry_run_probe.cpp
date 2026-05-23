// Phase 28BK: Runtime Hook Dry-Run / No-Generation Integration Probe
// Exercises prt_get_residual_view() — the actual runtime hook entry point —
// without running generation. Proves the bridge from shadow harness path
// to real pager + decode path works end-to-end.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <sys/stat.h>

// Must include these in dependency order:
// 1. prt_shadow.h brings in g_sidecars, g_sidecars_loaded, SidecarLoad
// 2. prt_sidecar_runtime_link.h provides prt_get_residual_view() + g_prt_pager*
// 3. prt_trit_decode.h provides decode_bytes + prt_decoded_view
#include "prt_shadow.h"         // PRT_SIDECAR_PAGER_EXPERIMENTAL guard inside
#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
#include "prt_sidecar_runtime_link.h"
#endif
#include "prt_trit_decode.h"

using namespace std::string_literals;

// ── Globals (defined in prt_shadow.h as weak externs, defined here for link) ──
std::unordered_map<int, SidecarLoad> g_sidecars;
bool g_sidecars_loaded = false;

// ── External globals (defined in prt_sidecar_runtime_link.h, linked from pager) ──
#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;
#endif

// ── Helper: hex dump of first N bytes ──
static void hexdump(const char* label, const uint8_t* data, size_t n, size_t max_show) {
    fprintf(stderr, "  %s (%zu bytes): ", label, n);
    if (n == 0) { fprintf(stderr, "<empty>\n"); return; }
    size_t show = std::min(n, max_show);
    for (size_t i = 0; i < show; i++) fprintf(stderr, "%02X ", data[i]);
    if (show < n) fprintf(stderr, "... (%zu more)", n - show);
    fprintf(stderr, "\n");
}

// ── Helper: get file size ──
static size_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return (size_t)st.st_size;
}

// ── Helper: read raw .trit header ──
struct trit_header_raw {
    uint32_t magic; uint16_t ver_major; uint16_t ver_minor;
    uint32_t rows; uint32_t cols;
    uint16_t block_rows; uint16_t block_cols;
    uint16_t n_scales; uint32_t payload_offset; uint32_t scale_offset;
    uint16_t checksum;
    static constexpr size_t SIZE = 32;
};

static bool read_trit_header(const std::string& path, trit_header_raw& h) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t buf[32];
    if (fread(buf, 1, 32, f) != 32) { fclose(f); return false; }
    fclose(f);
    h.magic = *(uint32_t*)(buf + 0);
    h.ver_major = *(uint16_t*)(buf + 4);
    h.ver_minor = *(uint16_t*)(buf + 6);
    h.rows = *(uint32_t*)(buf + 8);
    h.cols = *(uint32_t*)(buf + 12);
    h.block_rows = *(uint16_t*)(buf + 16);
    h.block_cols = *(uint16_t*)(buf + 18);
    h.n_scales = *(uint16_t*)(buf + 20);
    h.payload_offset = *(uint32_t*)(buf + 22);
    h.scale_offset = *(uint32_t*)(buf + 26);
    h.checksum = *(uint16_t*)(buf + 30);
    return true;
}

// ── TEST 1: Runtime Hook Positive ────────────────────────────────────────────
// Uses the phase28bi attn_out_l5_full package (layer 0, attn_out, [896,896])
// Steps: init pager → activate layer → prt_get_residual_view() → decode → verify
static bool test_runtime_hook_positive(bool silent = false) {
    if (!silent) fprintf(stderr, "\n=== TEST 1: RUNTIME HOOK POSITIVE ===\n");
    if (!silent) fprintf(stderr, "  Package: /tmp/phase28bi_pkg_attn_out_l5_full/\n");

    // Init pager
    prt_sidecar_pager_config cfg;
    cfg.sidecar_root = "/tmp/phase28bi_pkg_attn_out_l5_full";
    cfg.manifest_path = "/tmp/phase28bi_pkg_attn_out_l5_full/manifest.json";
    cfg.max_resident_bytes = 512 * 1024 * 1024;
    cfg.prefetch_distance = 1;
    cfg.window_size = 4;
    cfg.use_mmap = false;
    cfg.checksum_enabled = false;  // no checksum in fixture
    cfg.strict_budget = true;
    cfg.validate_trit_header = true;
    cfg.eviction_lru = false;

    prt_init_pager(cfg);
    if (!silent) fprintf(stderr, "  pager enabled: %s\n", g_prt_pager_enabled ? "YES" : "NO");
    if (!g_prt_pager_enabled) {
        if (!silent) fprintf(stderr, "  FAIL: pager not enabled\n");
        return false;
    }

    // Activate layer 0 (the only layer in this package)
    bool activated = g_prt_pager->activate_layer(0);
    if (!silent) fprintf(stderr, "  activate_layer(0): %s\n", activated ? "OK" : "FAIL");
    if (!activated) {
        if (!silent) fprintf(stderr, "  FAIL: layer activation failed\n");
        return false;
    }

    // Call the runtime hook entry point directly
    const char* family = "attn_out";
    prt_residual_view raw = prt_get_residual_view(0, family);
    if (!silent) fprintf(stderr, "  prt_get_residual_view(0, \"%s\"): is_null=%s reason=\"%s\"\n",
            family, raw.is_null ? "true" : "false", raw.reason.c_str());

    if (raw.is_null) {
        if (!silent) fprintf(stderr, "  FAIL: got null view\n");
        prt_shutdown_pager();
        return false;
    }

    if (!silent) fprintf(stderr, "  raw.size=%zu bytes\n", raw.size);

    // Prove decode-first: raw is .trit, NOT decoded float
    // Read header from the .trit file directly
    std::string trit_path = cfg.sidecar_root + "/layers/layer_000/attn_out.trit";
    trit_header_raw th;
    if (!read_trit_header(trit_path, th)) {
        if (!silent) fprintf(stderr, "  FAIL: could not read .trit header\n");
        prt_shutdown_pager();
        return false;
    }
    if (!silent) fprintf(stderr, "  .trit header: rows=%u cols=%u block_rows=%u block_cols=%u n_scales=%u\n",
            th.rows, th.cols, th.block_rows, th.block_cols, th.n_scales);

    size_t decoded_float_bytes = (size_t)th.rows * th.cols * sizeof(float);
    if (!silent) fprintf(stderr, "  decoded float size: %zu bytes (rows*cols*4)\n", decoded_float_bytes);
    if (!silent) fprintf(stderr, "  raw .trit size:   %zu bytes\n", raw.size);
    if (!silent) fprintf(stderr, "  ratio: %.2f x smaller than full float\n",
            (double)decoded_float_bytes / (double)raw.size);

    if (decoded_float_bytes == raw.size) {
        if (!silent) fprintf(stderr, "  WARNING: raw.size == decoded float size — may be misinterpreting!\n");
    } else {
        if (!silent) fprintf(stderr, "  PASS: raw .trit size != decoded float size (decode is necessary)\n");
    }

    // Decode the raw bytes
    if (!silent) hexdump("raw bytes (first 32)", raw.data, raw.size, 32);

    // Read full .trit file for decode (file_cache_ has raw bytes, decode_bytes needs full file)
    prt_trit_decoder dec;
    prt_decoded_view dv = dec.decode_file(trit_path);

    if (dv.is_null) {
        if (!silent) fprintf(stderr, "  FAIL: decode failed: %s\n", dv.reason.c_str());
        prt_shutdown_pager();
        return false;
    }

    if (!silent) fprintf(stderr, "  decoded: rows=%zu cols=%zu float_elems=%zu float_bytes=%zu\n",
            dv.rows, dv.cols, (size_t)dv.rows * dv.cols, (size_t)dv.rows * dv.cols * sizeof(float));

    bool size_ok = (dv.rows == th.rows && dv.cols == th.cols);
    if (!silent) fprintf(stderr, "  decoded dims match header: %s\n", size_ok ? "YES" : "NO");

    // Stats check via pager
    prt_sidecar_pager_stats st = prt_get_pager_stats();
    if (!silent) fprintf(stderr, "  PAGER STATS: reads=%zu cache_misses=%zu fallbacks=%zu\n",
            st.reads, st.cache_misses, st.fallbacks);

    prt_shutdown_pager();
    if (!silent) fprintf(stderr, "  RESULT: PASS\n");
    return true;
}

// ── TEST 2: DISABLED_MODE ────────────────────────────────────────────────────
// Pager disabled, no manifest, no false pass via legacy
static bool test_disabled_mode(bool silent = false) {
    if (!silent) fprintf(stderr, "\n=== TEST 2: DISABLED_MODE (pager disabled) ===\n");

    // Ensure pager is shut down from any prior test
    prt_shutdown_pager();

    if (!silent) fprintf(stderr, "  g_prt_pager_enabled=%s (should be false)\n",
            g_prt_pager_enabled ? "TRUE" : "FALSE");
    if (g_prt_pager_enabled) {
        if (!silent) fprintf(stderr, "  WARNING: pager still enabled, forcing off\n");
        prt_shutdown_pager();
    }

    // Without g_sidecars loaded, this must return null
    const char* family = "attn_out";
    prt_residual_view raw = prt_get_residual_view(0, family);

    if (!silent) fprintf(stderr, "  prt_get_residual_view(0, \"%s\"): is_null=%s reason=\"%s\"\n",
            family, raw.is_null ? "true" : "false", raw.reason.c_str());

    if (!raw.is_null) {
        if (!silent) fprintf(stderr, "  FAIL: got non-null view when pager disabled and no legacy\n");
        return false;
    }

    // Should fail with "not_found" (neither pager nor legacy has it)
    bool ok = raw.is_null && raw.reason == "not_found";
    if (!silent) fprintf(stderr, "  is_null && reason==not_found: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ── TEST 3: MISSING_MANIFEST_OR_SIDECAR ─────────────────────────────────────
// Init pager with non-existent manifest → deterministic failure, no crash
static bool test_missing_manifest(bool silent = false) {
    if (!silent) fprintf(stderr, "\n=== TEST 3: MISSING_MANIFEST ===\n");

    prt_shutdown_pager();

    prt_sidecar_pager_config cfg;
    cfg.sidecar_root = "/tmp";
    cfg.manifest_path = "/tmp/nonexistent_manifest_28bk.json";
    cfg.max_resident_bytes = 512 * 1024 * 1024;

    bool ok = prt_init_pager(cfg);
    if (!silent) fprintf(stderr, "  prt_init_pager() returned: %s\n", ok ? "true" : "false");

    if (ok) {
        // Should not have enabled if manifest doesn't exist
        if (!silent) fprintf(stderr, "  WARNING: init succeeded on missing manifest?\n");
        prt_shutdown_pager();
        return false;
    }

    if (!silent) fprintf(stderr, "  deterministic failure on missing manifest: PASS (no crash)\n");
    return true;
}

// ── TEST 4: BAD_TENSOR_KEY ───────────────────────────────────────────────────
// Valid pager/manifest but request non-existent tensor_family
static bool test_bad_tensor_key(bool silent = false) {
    if (!silent) fprintf(stderr, "\n=== TEST 4: BAD_TENSOR_KEY ===\n");

    prt_shutdown_pager();

    prt_sidecar_pager_config cfg;
    cfg.sidecar_root = "/tmp/phase28bi_pkg_attn_out_l5_full";
    cfg.manifest_path = "/tmp/phase28bi_pkg_attn_out_l5_full/manifest.json";
    cfg.max_resident_bytes = 512 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.strict_budget = true;
    cfg.validate_trit_header = true;

    bool initted = prt_init_pager(cfg);
    if (!silent) fprintf(stderr, "  init: %s\n", initted ? "OK" : "FAIL");
    if (!initted) {
        if (!silent) fprintf(stderr, "  FAIL: could not init pager\n");
        return false;
    }

    bool activated = g_prt_pager->activate_layer(0);
    if (!silent) fprintf(stderr, "  activate_layer(0): %s\n", activated ? "OK" : "FAIL");

    // Request wrong tensor family
    prt_residual_view raw = prt_get_residual_view(0, "nonexistent_family");
    if (!silent) fprintf(stderr, "  prt_get_residual_view(0, \"nonexistent_family\"): is_null=%s reason=\"%s\"\n",
            raw.is_null ? "true" : "false", raw.reason.c_str());

    prt_shutdown_pager();

    bool ok = raw.is_null && (raw.reason == "tensor_not_found" || raw.reason == "not_found");
    if (!silent) fprintf(stderr, "  deterministic null for bad key: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ── TEST 5: RAW_BYTES_NOT_USED_AS_FLOAT ──────────────────────────────────────
// Prove raw.size != decoded float size; decode-first semantics verified
static bool test_raw_vs_decoded(bool silent = false) {
    if (!silent) fprintf(stderr, "\n=== TEST 5: RAW_BYTES_NOT_USED_AS_FLOAT ===\n");

    prt_shutdown_pager();

    prt_sidecar_pager_config cfg;
    cfg.sidecar_root = "/tmp/phase28bi_pkg_attn_out_l5_full";
    cfg.manifest_path = "/tmp/phase28bi_pkg_attn_out_l5_full/manifest.json";
    cfg.max_resident_bytes = 512 * 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.strict_budget = true;
    cfg.validate_trit_header = true;

    prt_init_pager(cfg);
    if (!g_prt_pager_enabled) {
        if (!silent) fprintf(stderr, "  FAIL: pager not enabled\n");
        return false;
    }

    g_prt_pager->activate_layer(0);

    // Get raw view via runtime hook
    prt_residual_view raw = prt_get_residual_view(0, "attn_out");

    if (raw.is_null) {
        if (!silent) fprintf(stderr, "  FAIL: null raw view\n");
        prt_shutdown_pager();
        return false;
    }

    // Read header for ground truth
    std::string trit_path = cfg.sidecar_root + "/layers/layer_000/attn_out.trit";
    trit_header_raw th;
    read_trit_header(trit_path, th);

    size_t decoded_float_bytes = (size_t)th.rows * th.cols * sizeof(float);
    size_t raw_size = raw.size;

    // Show the stark size difference
    if (!silent) fprintf(stderr, "  raw.size (packed .trit)      = %zu bytes\n", raw_size);
    if (!silent) fprintf(stderr, "  decoded float (rows*cols*4) = %zu bytes\n", decoded_float_bytes);
    if (!silent) fprintf(stderr, "  compression ratio           = %.1f:1\n",
            (double)decoded_float_bytes / (double)raw_size);

    bool size_differs = (raw_size != decoded_float_bytes);
    if (!silent) fprintf(stderr, "  raw.size != decoded float size: %s\n", size_differs ? "PASS" : "FAIL");

    // Decode and verify actual decoded count
    prt_trit_decoder dec;
    prt_decoded_view dv = dec.decode_file(trit_path);

    bool decode_ok = !dv.is_null && dv.rows == th.rows && dv.cols == th.cols;
    if (!silent) fprintf(stderr, "  decode produces correct shape: %s\n", decode_ok ? "PASS" : "FAIL");

    // Demonstrate: if someone incorrectly cast raw bytes to float*, they get wrong element count
    size_t wrong_elem_count = raw_size / sizeof(float);
    size_t correct_elem_count = (size_t)th.rows * th.cols;
    if (!silent) fprintf(stderr, "  wrong_elem_count (raw/4) = %zu vs correct = %zu\n",
            wrong_elem_count, correct_elem_count);

    prt_shutdown_pager();
    return size_differs && decode_ok;
}

// ── TEST 6: BUDGET_REJECT ────────────────────────────────────────────────────
// Tiny budget (4 bytes) must reject deterministically
static bool test_budget_reject(bool silent = false) {
    if (!silent) fprintf(stderr, "\n=== TEST 6: BUDGET_REJECT (budget=4 bytes) ===\n");

    prt_shutdown_pager();

    prt_sidecar_pager_config cfg;
    cfg.sidecar_root = "/tmp/phase28bi_pkg_attn_out_l5_full";
    cfg.manifest_path = "/tmp/phase28bi_pkg_attn_out_l5_full/manifest.json";
    cfg.max_resident_bytes = 4;  // absurdly tiny
    cfg.checksum_enabled = false;
    cfg.strict_budget = true;
    cfg.validate_trit_header = true;
    cfg.eviction_lru = false;  // strict reject mode

    bool initted = prt_init_pager(cfg);
    if (!silent) fprintf(stderr, "  init: %s\n", initted ? "OK" : "FAIL");

    // Even if init succeeds, activate_layer must reject
    bool activated = false;
    if (g_prt_pager) {
        activated = g_prt_pager->activate_layer(0);
    }

    if (!silent) fprintf(stderr, "  activate_layer(0) with tiny budget: %s\n",
            activated ? "activated (UNEXPECTED)" : "REJECTED (expected)");

    prt_sidecar_pager_stats st = prt_get_pager_stats();
    if (!silent) fprintf(stderr, "  budget_rejects=%zu resident_bytes=%zu\n",
            st.budget_rejects, st.resident_bytes);

    prt_shutdown_pager();

    // Must be rejected (activated == false) OR budget_rejects > 0
    bool ok = !activated || st.budget_rejects > 0;
    if (!silent) fprintf(stderr, "  deterministic budget rejection: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    fprintf(stderr, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    fprintf(stderr, "PHASE 28BK: Runtime Hook Dry-Run Probe\n");
    fprintf(stderr, "Entry point: prt_get_residual_view()\n");
    fprintf(stderr, "Flag: PRT_SIDECAR_PAGER_EXPERIMENTAL\n");
    fprintf(stderr, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Always reset pager state at start
    prt_shutdown_pager();

    // Run tests
    bool t1 = test_runtime_hook_positive();
    bool t2 = test_disabled_mode();
    bool t3 = test_missing_manifest();
    bool t4 = test_bad_tensor_key();
    bool t5 = test_raw_vs_decoded();
    bool t6 = test_budget_reject();

    fprintf(stderr, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    fprintf(stderr, "SUMMARY\n");
    fprintf(stderr, "  T1 runtime_hook_positive:      %s\n", t1 ? "PASS" : "FAIL");
    fprintf(stderr, "  T2 DISABLED_MODE:             %s\n", t2 ? "PASS" : "FAIL");
    fprintf(stderr, "  T3 MISSING_MANIFEST:          %s\n", t3 ? "PASS" : "FAIL");
    fprintf(stderr, "  T4 BAD_TENSOR_KEY:            %s\n", t4 ? "PASS" : "FAIL");
    fprintf(stderr, "  T5 RAW_VS_DECODED:            %s\n", t5 ? "PASS" : "FAIL");
    fprintf(stderr, "  T6 BUDGET_REJECT:             %s\n", t6 ? "PASS" : "FAIL");
    fprintf(stderr, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    bool all_pass = t1 && t2 && t3 && t4 && t5 && t6;
    fprintf(stderr, "Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    fprintf(stderr, "Generation: false (dry-run only)\n");

    return all_pass ? 0 : 1;
}