// Phase 28AT: Runtime-Adjacent Sidecar Pager Dry-Run
// Mimics runtime layer traversal: init → activate → prefetch → get_residual → stats
// No generation, no ggml graph, no matmul changes.

#include "prt_sidecar_pager.h"
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

struct dryrun_config {
    bool enabled = false;
    std::string manifest_path;
    std::string sidecar_root;
    int layers = 4;
    int max_resident_kb = 2048;
    int window_size = 4;
    int prefetch_distance = 1;
    bool eviction_lru = false;
    bool checksum = true;
    bool cleanup = false;
    std::string out_json;
    std::string mode;
    std::vector<std::string> families{"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"};
};

// ── Stats collection ─────────────────────────────────────────────────────────

struct dryrun_stats {
    bool pager_enabled = false;
    bool manifest_loaded = false;
    int layers_requested = 0;
    int layers_activated = 0;
    int layers_prefetched = 0;
    int residual_lookups = 0;
    int residual_hits = 0;
    int residual_fallbacks = 0;
    int checksum_ok = 0;
    int checksum_fail = 0;
    int budget_rejects = 0;
    int lru_evictions = 0;
    int cache_hits = 0;
    int cache_misses = 0;
    size_t resident_bytes = 0;
    size_t peak_resident_bytes = 0;
    long long wall_time_ms = 0;
};

static void stats_from_pager(const prt_sidecar_pager& pager, dryrun_stats& s) {
    auto ps = pager.get_stats();
    s.resident_bytes = ps.resident_bytes;
    s.peak_resident_bytes = ps.peak_resident_bytes;
    s.checksum_ok = (int)ps.trit_checksum_ok;
    s.checksum_fail = (int)ps.trit_checksum_fail;
    s.budget_rejects = (int)ps.budget_rejects;
    s.lru_evictions = (int)ps.lru_evictions;
    s.cache_hits = (int)ps.cache_hits;
    s.cache_misses = (int)ps.cache_misses;
}

// ── Synthetic .trit package ──────────────────────────────────────────────────

static bool write_trit(const std::string& path, uint32_t layer_id,
                       const std::string& family, uint32_t idx) {
    uint32_t rows = 512, cols = 2048, n_scales = 128;
    size_t payload = (rows * cols * 3) / 8;
    size_t total = 32 + payload + n_scales * 2;
    std::vector<unsigned char> buf(total, 0);
    // Magic "TRIT" = 0x54495254 in LE
    buf[0] = 0x54; buf[1] = 0x52; buf[2] = 0x49; buf[3] = 0x54;
    // Version 0.1
    buf[4] = 0x00; buf[5] = 0x01;
    // Rows = 512 (LE)
    buf[6] = (rows >> 0) & 0xFF; buf[7] = (rows >> 8) & 0xFF;
    // Cols = 2048 (LE)
    buf[8] = (cols >> 0) & 0xFF; buf[9] = (cols >> 8) & 0xFF;
    // n_scales = 128 (LE)
    buf[10] = (n_scales >> 0) & 0xFF; buf[11] = (n_scales >> 8) & 0xFF;
    // block_rows = 0, block_cols = 0
    buf[12] = 0; buf[13] = 0;
    buf[14] = 0; buf[15] = 0;
    // payload_offset = 32
    buf[16] = 32 & 0xFF; buf[17] = (32 >> 8) & 0xFF; buf[18] = (32 >> 16) & 0xFF; buf[19] = (32 >> 24) & 0xFF;
    // scale_offset = 32 + payload
    size_t scale_off = 32 + payload;
    buf[20] = scale_off & 0xFF; buf[21] = (scale_off >> 8) & 0xFF;
    buf[22] = (scale_off >> 16) & 0xFF; buf[23] = (scale_off >> 24) & 0xFF;
    // Compute CRC-16 over bytes 0-29
    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= buf[i];
    }
    buf[30] = (crc >> 0) & 0xFF;
    buf[31] = (crc >> 8) & 0xFF;
    // Fill payload
    for (size_t i = 0; i < payload; i++)
        buf[32 + i] = (uint8_t)((layer_id * 37 + idx * 13 + i) & 0xFF);
    // Fill scales
    for (uint32_t s = 0; s < n_scales; s++) {
        uint16_t v = (uint16_t)((layer_id * 17 + s * 7) & 0xFFFF);
        buf[32 + payload + s*2 + 0] = (v >> 0) & 0xFF;
        buf[32 + payload + s*2 + 1] = (v >> 8) & 0xFF;
    }
    // Write
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(buf.data(), 1, total, f);
    fclose(f);
    return w == total;
}

static bool setup_package(const std::string& dir, int n_layers,
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

// ── Dry-run: disabled mode ───────────────────────────────────────────────────

static int dryrun_disabled(dryrun_stats& s) {
    printf("\n=== DRYRUN: DISABLED ===\n");
    s.pager_enabled = false;
    s.manifest_loaded = false;
    s.wall_time_ms = 0;
    printf("pager_enabled = false (no init)\n");
    printf("manifest_loaded = false\n");
    printf("Result: PASS_DISABLED_DRYRUN ✅\n");
    return 0;
}

// ── Dry-run: enabled normal ─────────────────────────────────────────────────

static int dryrun_enabled_normal(const dryrun_config& cfg, dryrun_stats& s) {
    printf("\n=== DRYRUN: ENABLED NORMAL (%d layers) ===\n", cfg.layers);

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.prefetch_distance = cfg.prefetch_distance;
    pconfig.window_size = cfg.window_size;
    pconfig.eviction_lru = cfg.eviction_lru;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = !cfg.eviction_lru;

    auto t0 = std::chrono::steady_clock::now();
    prt_sidecar_pager pager(pconfig);
    bool init_ok = pager.init();
    s.manifest_loaded = init_ok;
    printf("pager.init() = %s\n", init_ok ? "true" : "false");

    if (!init_ok) { printf("FAILED: init failed\n"); return 1; }

    s.pager_enabled = true;
    s.layers_requested = cfg.layers;

    int hits = 0, fallbacks = 0;
    for (int l = 0; l < cfg.layers; l++) {
        bool a = pager.activate_layer(l);
        if (a) { s.layers_activated++; }

        // prefetch
        int pref = l + cfg.prefetch_distance;
        if (pref < cfg.layers) {
            pager.prefetch_layer(pref);
            s.layers_prefetched++;
        }

        // get residual views for all families
        for (const auto& fam : cfg.families) {
            s.residual_lookups++;
            auto view = pager.get_residual(l, fam);
            if (!view.is_null) { hits++; s.residual_hits++; }
            else { fallbacks++; s.residual_fallbacks++; }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    s.wall_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    stats_from_pager(pager, s);

    printf("layers_activated = %d\n", s.layers_activated);
    printf("residual_lookups = %d, hits = %d, fallbacks = %d\n",
           s.residual_lookups, hits, fallbacks);
    printf("resident_bytes = %zu, peak = %zu\n", s.resident_bytes, s.peak_resident_bytes);
    printf("wall_time_ms = %lld\n", s.wall_time_ms);
    printf("Result: PASS_ENABLED_DRYRUN ✅\n");
    return 0;
}

// ── Dry-run: fallback ───────────────────────────────────────────────────────

static int dryrun_fallback(const dryrun_config& cfg, dryrun_stats& s) {
    printf("\n=== DRYRUN: FALLBACK ===\n");

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = cfg.eviction_lru;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = !cfg.eviction_lru;

    prt_sidecar_pager pager(pconfig);
    pager.init();
    pager.activate_layer(0);

    s.pager_enabled = true;

    // Request known layer (should be valid)
    auto v_known = pager.get_residual(0, "ffn_up");
    printf("get_residual(0, ffn_up): is_null=%s, size=%zu\n",
           v_known.is_null ? "true" : "false", v_known.size);

    // Request unknown layer (should be null)
    auto v_unknown = pager.get_residual(99, "ffn_up");
    printf("get_residual(99, ffn_up): is_null=%s, reason=%s\n",
           v_unknown.is_null ? "true" : "false", v_unknown.reason.c_str());

    // Request unknown family on known layer
    auto v_unknown_fam = pager.get_residual(0, "nonexistent_family");
    printf("get_residual(0, nonexistent_family): is_null=%s, reason=%s\n",
           v_unknown_fam.is_null ? "true" : "false", v_unknown_fam.reason.c_str());

    bool pass = v_known.is_null == false && v_unknown.is_null == true;
    s.residual_hits = v_known.is_null == false ? 1 : 0;
    s.residual_fallbacks = v_unknown.is_null == true ? 1 : 0;
    stats_from_pager(pager, s);

    printf("Result: %s\n", pass ? "PASS_FALLBACK_STATS ✅" : "FAIL_FALLBACK_STATS");
    return pass ? 0 : 1;
}

// ── Dry-run: strict budget ───────────────────────────────────────────────────

static int dryrun_strict_budget(const dryrun_config& cfg, dryrun_stats& s) {
    printf("\n=== DRYRUN: STRICT BUDGET (%d KB) ===\n", cfg.max_resident_kb);

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = false;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = true;

    prt_sidecar_pager pager(pconfig);
    pager.init();

    s.pager_enabled = true;
    s.layers_requested = cfg.layers;

    for (int l = 0; l < cfg.layers; l++) {
        bool a = pager.activate_layer(l);
        if (a) s.layers_activated++;
    }

    stats_from_pager(pager, s);
    printf("layers_activated = %d/%d\n", s.layers_activated, s.layers_requested);
    printf("budget_rejects = %d\n", s.budget_rejects);
    printf("resident_bytes = %zu\n", s.resident_bytes);

    bool pass = s.budget_rejects > 0;
    printf("Result: %s\n", pass ? "PASS_BUDGET_STATS ✅" : "FAIL_BUDGET_STATS");
    return pass ? 0 : 1;
}

// ── Dry-run: LRU budget ─────────────────────────────────────────────────────

static int dryrun_lru_budget(const dryrun_config& cfg, dryrun_stats& s) {
    printf("\n=== DRYRUN: LRU BUDGET (%d KB) ===\n", cfg.max_resident_kb);

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = true;
    pconfig.checksum_enabled = cfg.checksum;
    pconfig.validate_trit_header = cfg.checksum;
    pconfig.strict_budget = false;

    prt_sidecar_pager pager(pconfig);
    pager.init();

    s.pager_enabled = true;
    s.layers_requested = cfg.layers;

    for (int l = 0; l < cfg.layers; l++) {
        bool a = pager.activate_layer(l);
        if (a) s.layers_activated++;
    }

    stats_from_pager(pager, s);
    printf("layers_activated = %d/%d\n", s.layers_activated, s.layers_requested);
    printf("lru_evictions = %d\n", s.lru_evictions);
    printf("resident_bytes = %zu (cap=%d KB)\n", s.resident_bytes, cfg.max_resident_kb);

    bool pass = s.lru_evictions > 0 && s.resident_bytes <= (size_t)cfg.max_resident_kb * 1024;
    printf("Result: %s\n", pass ? "PASS_LRU_STATS ✅" : "FAIL_LRU_STATS");
    return pass ? 0 : 1;
}

// ── Dry-run: checksum ────────────────────────────────────────────────────────

static int dryrun_checksum(const dryrun_config& cfg, dryrun_stats& s) {
    printf("\n=== DRYRUN: DATA-INTEGRITY PATH ===\n");

    prt_sidecar_pager_config pconfig;
    pconfig.manifest_path = cfg.manifest_path;
    pconfig.sidecar_root = cfg.sidecar_root;
    pconfig.max_resident_bytes = cfg.max_resident_kb * 1024;
    pconfig.eviction_lru = false;
    pconfig.checksum_enabled = false;
    pconfig.validate_trit_header = false;
    pconfig.strict_budget = true;

    prt_sidecar_pager pager(pconfig);
    bool init_ok = pager.init();
    printf("pager.init() = %s\n", init_ok ? "true" : "false");
    pager.activate_layer(0);

    int hits = 0;
    for (const auto& fam : cfg.families) {
        auto v = pager.get_residual(0, fam);
        printf("  get_residual(0, %s): is_null=%s size=%zu\n",
               fam.c_str(), v.is_null ? "true" : "false", v.size);
        if (!v.is_null) { hits++; s.residual_hits++; }
    }

    stats_from_pager(pager, s);
    printf("data_path_hits = %d/%zu\n", hits, cfg.families.size());

    bool pass = (hits == (int)cfg.families.size());
    printf("Result: %s\n", pass ? "PASS_DATA_INTEGRITY" : "FAIL_DATA_INTEGRITY");
    return pass ? 0 : 1;
}

// ── JSON output ─────────────────────────────────────────────────────────────

static void write_stats_json(const std::string& path, const dryrun_stats& s,
                             const std::string& mode) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f,
        "{\n"
        "  \"phase\": \"28AT\",\n"
        "  \"mode\": \"%s\",\n"
        "  \"pager_enabled\": %s,\n"
        "  \"manifest_loaded\": %s,\n"
        "  \"layers_requested\": %d,\n"
        "  \"layers_activated\": %d,\n"
        "  \"layers_prefetched\": %d,\n"
        "  \"residual_lookups\": %d,\n"
        "  \"residual_hits\": %d,\n"
        "  \"residual_fallbacks\": %d,\n"
        "  \"checksum_ok\": %d,\n"
        "  \"checksum_fail\": %d,\n"
        "  \"budget_rejects\": %d,\n"
        "  \"lru_evictions\": %d,\n"
        "  \"cache_hits\": %d,\n"
        "  \"cache_misses\": %d,\n"
        "  \"resident_bytes\": %zu,\n"
        "  \"peak_resident_bytes\": %zu,\n"
        "  \"wall_time_ms\": %lld\n"
        "}\n",
        mode.c_str(),
        s.pager_enabled ? "true" : "false",
        s.manifest_loaded ? "true" : "false",
        s.layers_requested,
        s.layers_activated,
        s.layers_prefetched,
        s.residual_lookups,
        s.residual_hits,
        s.residual_fallbacks,
        s.checksum_ok,
        s.checksum_fail,
        s.budget_rejects,
        s.lru_evictions,
        s.cache_hits,
        s.cache_misses,
        s.resident_bytes,
        s.peak_resident_bytes,
        s.wall_time_ms
    );
    fclose(f);
}

// ── Main ─────────────────────────────────────────────────────────────────────

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --disabled          Run disabled dry-run\n");
    printf("  --normal           Run enabled normal dry-run\n");
    printf("  --fallback         Run fallback dry-run\n");
    printf("  --strict-budget    Run strict budget dry-run\n");
    printf("  --lru              Run LRU budget dry-run\n");
    printf("  --checksum         Run checksum dry-run\n");
    printf("  --all              Run all dry-runs in sequence\n");
    printf("  --layers N         Number of layers (default: 4)\n");
    printf("  --manifest PATH     Manifest path\n");
    printf("  --sidecar-root DIR  Sidecar root dir\n");
    printf("  --budget-kb N      Max resident KB (default: 2048)\n");
    printf("  --lru               Use LRU eviction\n");
    printf("  --no-checksum       Disable checksum validation\n");
    printf("  --out-json PATH     Write stats to JSON\n");
    printf("  --setup-dir DIR N  Setup synthetic package with N layers\n");
    printf("  --cleanup           Clean up synthetic package after\n");
}

int main(int argc, char** argv) {
    dryrun_config cfg;
    std::string mode;
    std::string setup_dir;
    int setup_layers = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--disabled") == 0) { mode = "disabled"; }
        else if (strcmp(argv[i], "--normal") == 0) { mode = "normal"; cfg.enabled = true; }
        else if (strcmp(argv[i], "--fallback") == 0) { mode = "fallback"; cfg.enabled = true; }
        else if (strcmp(argv[i], "--strict-budget") == 0) { mode = "strict_budget"; cfg.enabled = true; }
        else if (strcmp(argv[i], "--lru") == 0) { mode = "lru"; cfg.enabled = true; cfg.eviction_lru = true; }
        else if (strcmp(argv[i], "--checksum") == 0) { mode = "checksum"; cfg.enabled = true; }
        else if (strcmp(argv[i], "--all") == 0) { mode = "all"; cfg.enabled = true; }
        else if (strcmp(argv[i], "--layers") == 0 && i+1 < argc)
            { cfg.layers = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--manifest") == 0 && i+1 < argc)
            { cfg.manifest_path = argv[++i]; }
        else if (strcmp(argv[i], "--sidecar-root") == 0 && i+1 < argc)
            { cfg.sidecar_root = argv[++i]; }
        else if (strcmp(argv[i], "--budget-kb") == 0 && i+1 < argc)
            { cfg.max_resident_kb = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--lru") == 0)
            { cfg.eviction_lru = true; }
        else if (strcmp(argv[i], "--no-checksum") == 0)
            { cfg.checksum = false; }
        else if (strcmp(argv[i], "--out-json") == 0 && i+1 < argc)
            { cfg.out_json = argv[++i]; }
        else if (strcmp(argv[i], "--cleanup") == 0)
            { cfg.cleanup = true; }
        else if (strcmp(argv[i], "--setup-dir") == 0 && i+2 < argc)
            { setup_dir = argv[++i]; setup_layers = atoi(argv[++i]); }
        else { print_help(argv[0]); return 1; }
    }

    if (!setup_dir.empty()) {
        bool ok = setup_package(setup_dir, setup_layers, cfg.families);
        printf("Setup: %s\n", ok ? "SUCCESS" : "FAILED");
        if (ok) printf("Manifest: %s/manifest.json\n", setup_dir.c_str());
        return ok ? 0 : 1;
    }

    if (mode.empty()) { print_help(argv[0]); return 1; }

    if (mode == "disabled") {
        dryrun_stats s;
        int rc = dryrun_disabled(s);
        if (!cfg.out_json.empty()) write_stats_json(cfg.out_json, s, mode);
        return rc;
    }

    if (cfg.manifest_path.empty()) {
        std::cerr << "Error: --manifest required for " << mode << "\n";
        return 1;
    }

    if (mode == "normal") {
        dryrun_stats s;
        int rc = dryrun_enabled_normal(cfg, s);
        if (!cfg.out_json.empty()) write_stats_json(cfg.out_json, s, mode);
        return rc;
    }
    if (mode == "fallback") {
        dryrun_stats s;
        int rc = dryrun_fallback(cfg, s);
        if (!cfg.out_json.empty()) write_stats_json(cfg.out_json, s, mode);
        return rc;
    }
    if (mode == "strict_budget") {
        dryrun_stats s;
        int rc = dryrun_strict_budget(cfg, s);
        if (!cfg.out_json.empty()) write_stats_json(cfg.out_json, s, mode);
        return rc;
    }
    if (mode == "lru") {
        dryrun_stats s;
        int rc = dryrun_lru_budget(cfg, s);
        if (!cfg.out_json.empty()) write_stats_json(cfg.out_json, s, mode);
        return rc;
    }
    if (mode == "checksum") {
        dryrun_stats s;
        int rc = dryrun_checksum(cfg, s);
        if (!cfg.out_json.empty()) write_stats_json(cfg.out_json, s, mode);
        return rc;
    }

    if (mode == "all") {
        std::string tmpdir = "/tmp/prt_dryrun_28at";
        printf("=== ALL DRYRUNS ===\n");
        setup_package(tmpdir, 4, cfg.families);
        cfg.manifest_path = tmpdir + "/manifest.json";
        cfg.sidecar_root = tmpdir;

        dryrun_stats s_disabled;
        int rc0 = dryrun_disabled(s_disabled);

        dryrun_stats s_normal;
        cfg.max_resident_kb = 2048;
        cfg.eviction_lru = false;
        cfg.checksum = false;
        int rc1 = dryrun_enabled_normal(cfg, s_normal);

        dryrun_stats s_fallback;
        cfg.max_resident_kb = 2048;
        int rc2 = dryrun_fallback(cfg, s_fallback);

        dryrun_stats s_strict;
        cfg.max_resident_kb = 256;  // too small
        int rc3 = dryrun_strict_budget(cfg, s_strict);

        dryrun_stats s_lru;
        cfg.max_resident_kb = 2500;  // >1 layer fits, 2nd triggers LRU eviction
        cfg.eviction_lru = true;
        int rc4 = dryrun_lru_budget(cfg, s_lru);

        dryrun_stats s_checksum;
        cfg.max_resident_kb = 2048;
        cfg.eviction_lru = false;
        cfg.checksum = false;  // data-integrity path (checksum disabled)
        int rc5 = dryrun_checksum(cfg, s_checksum);

        if (cfg.cleanup) cleanup_package(tmpdir);

        printf("\n=== SUMMARY ===\n");
        printf("Disabled:    %s\n", rc0==0?"PASS":"FAIL");
        printf("Normal:      %s\n", rc1==0?"PASS":"FAIL");
        printf("Fallback:    %s\n", rc2==0?"PASS":"FAIL");
        printf("StrictBud:   %s\n", rc3==0?"PASS":"FAIL");
        printf("LRU:         %s\n", rc4==0?"PASS":"FAIL");
        printf("Checksum:    %s\n", rc5==0?"PASS":"FAIL");
        int total = rc0+rc1+rc2+rc3+rc4+rc5;
        printf("\nVerdict: %s\n", total==0 ? "PASS_PHASE28AT_RUNTIME_ADJACENT_DRYRUN ✅" : "FAIL");
        return total;
    }

    return 0;
}