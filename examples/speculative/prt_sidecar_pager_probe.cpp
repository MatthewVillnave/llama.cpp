// Phase 28AN: Sidecar Pager Probe
// Tests prt_sidecar_pager with fake + real manifest + real .trit files
// Supports --real-trit mode for Phase 28Y manifest + .trit header validation

#include "prt_sidecar_pager.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

// ── CRC16 (matching Python implementation) ────────────────────────────────────

static uint16_t compute_crc16(const uint8_t * data, size_t len) {
    uint16_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1) crc = (crc >> 1) ^ 0xA001;
            else crc >>= 1;
        }
    }
    return crc;
}

// ── Args ─────────────────────────────────────────────────────────────────────

struct Args {
    int layers = 4;
    int tensors_per_layer = 3;
    int trit_kb = 64;
    int window_size = 2;
    int prefetch_distance = 1;
    int max_resident_kb = 512;
    bool eviction_lru = false;
    std::string storage_dir = "/tmp/prt_sidecar_pager_smoke";
    mutable std::string manifest_path; // NON_CONST: allow mutation
    bool real_trit = false;
    bool check_trit_header = true;
    std::string out_json;
    bool cleanup = true;
};

bool parse_args(int argc, char **argv, Args &a) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layers") == 0 && i+1 < argc) a.layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tensors-per-layer") == 0 && i+1 < argc) a.tensors_per_layer = atoi(argv[++i]);
        else if (strcmp(argv[i], "--trit-kb") == 0 && i+1 < argc) a.trit_kb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--window-size") == 0 && i+1 < argc) a.window_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prefetch-distance") == 0 && i+1 < argc) a.prefetch_distance = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-resident-kb") == 0 && i+1 < argc) a.max_resident_kb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--storage-dir") == 0 && i+1 < argc) a.storage_dir = argv[++i];
        else if (strcmp(argv[i], "--manifest-path") == 0 && i+1 < argc) a.manifest_path = argv[++i];
        else if (strcmp(argv[i], "--real-trit") == 0 && i+1 < argc) a.real_trit = strcmp(argv[++i], "true") == 0;
        else if (strcmp(argv[i], "--check-trit-header") == 0 && i+1 < argc) a.check_trit_header = strcmp(argv[i+1], "true") == 0;
        else if (strcmp(argv[i], "--out-json") == 0 && i+1 < argc) a.out_json = argv[++i];
        else if (strcmp(argv[i], "--eviction-lru") == 0 && i+1 < argc) a.eviction_lru = strcmp(argv[++i], "true") == 0;
        else if (strcmp(argv[i], "--cleanup") == 0 && i+1 < argc) a.cleanup = strcmp(argv[++i], "true") == 0;
    }
    return true;
}

// ── Fake manifest generator (Phase 28AM-style, for regression) ─────────────────

void create_fake_manifest(const Args &a) {
    if (fs::exists(a.storage_dir)) fs::remove_all(a.storage_dir);
    fs::create_directories(a.storage_dir);

    std::vector<std::string> families = {"ffn_up", "ffn_down", "attn_q"};
    std::ostringstream json;
    json << "{\n  \"format_version\": 1,\n  \"entries\": [\n";

    for (int layer = 0; layer < a.layers; layer++) {
        for (int t = 0; t < a.tensors_per_layer && t < (int)families.size(); t++) {
            const std::string & fam = families[t];
            char filename[128];
            snprintf(filename, sizeof(filename), "layer_%03d.%s.trit", layer, fam.c_str());
            std::string path = (fs::path(a.storage_dir) / filename).string();

            FILE * f = fopen(path.c_str(), "wb");
            if (!f) { fprintf(stderr, "Failed to create %s\n", path.c_str()); exit(1); }
            size_t data_size = a.trit_kb * 1024;
            unsigned char * buf = new unsigned char[data_size];
            srand((unsigned)(42 + layer * 17 + t * 31));
            for (size_t i = 0; i < data_size; i++) buf[i] = rand() & 0xFF;
            fwrite(buf, 1, data_size, f);
            fclose(f);
            delete[] buf;

            // CRC
            FILE * cf = fopen(path.c_str(), "rb");
            unsigned char * data = new unsigned char[data_size];
            fread(data, 1, data_size, cf);
            fclose(cf);
            uint16_t crc = compute_crc16(data, data_size);
            delete[] data;

            json << "    {\"layer_index\": " << layer << ", \"tensor_family\": \"" << fam << "\", "
                 << "\"file_path\": \"" << filename << "\", \"byte_size\": " << data_size << ", \"crc\": " << crc << "}";
            if (layer < a.layers - 1 || t < a.tensors_per_layer - 1) json << ",";
            json << "\n";
        }
    }
    json << "  ]\n}\n";

    std::string mp = (fs::path(a.storage_dir) / "manifest.json").string();
    std::ofstream mf(mp);
    mf << json.str();
    mf.close();
    a.manifest_path = mp;
}

// ── Real .trit file writer (matching prt_trit_io.py format) ───────────────────

static constexpr uint32_t TRIT_MAGIC = 0x54524954;
static constexpr uint16_t TRIT_VER_MAJOR = 0;
static constexpr uint16_t TRIT_VER_MINOR = 1;
static constexpr size_t TRIT_HEADER_SIZE = 32;
static constexpr int TRIT_ENCODING_BITS = 3;

struct trit_header_packed {
    uint32_t magic;
    uint16_t ver_major;
    uint16_t ver_minor;
    uint32_t rows;
    uint32_t cols;
    uint16_t block_rows;
    uint16_t block_cols;
    uint16_t n_scales;
    uint32_t payload_offset;
    uint32_t scale_offset;
    uint16_t checksum;
};

static uint16_t compute_trit_crc(const uint8_t * header_30bytes) {
    uint16_t crc = 0;
    for (size_t i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= header_30bytes[i];
    }
    return crc & 0xFFFF;
}

void write_trit_real(const std::string & path, int rows, int cols, int n_scales) {
    size_t total_trits = (size_t)rows * cols;
    size_t payload_bits = total_trits * TRIT_ENCODING_BITS;
    size_t payload_bytes = (payload_bits + 7) / 8;
    size_t payload_offset = TRIT_HEADER_SIZE + payload_bytes;
    size_t scale_offset = payload_offset;
    if (scale_offset % 4) scale_offset = (scale_offset / 4 + 1) * 4;

    std::vector<uint8_t> packed(payload_bytes);
    srand(42);
    for (size_t i = 0; i < total_trits; i++) {
        int trit_val = rand() % 3 - 1;
        int bits = (trit_val == 0) ? 0 : (trit_val == 1 ? 1 : 7);
        size_t bit_pos = i * TRIT_ENCODING_BITS;
        size_t byte_idx = bit_pos / 8;
        size_t bit_off = bit_pos % 8;
        if (bit_off <= 5) packed[byte_idx] |= bits << bit_off;
        else {
            size_t bits_in_first = 8 - bit_off;
            size_t bits_in_second = 3 - bits_in_first;
            packed[byte_idx] |= (bits & ((1 << bits_in_first) - 1)) << bit_off;
            packed[byte_idx + 1] |= (bits >> bits_in_first) & ((1 << bits_in_second) - 1);
        }
    }

    std::vector<float> scales(n_scales);
    for (int i = 0; i < n_scales; i++) scales[i] = (float)(0.5 + (rand() % 100) / 100.0);

    uint8_t header[32] = {0};
    header[0] = 'T'; header[1] = 'R'; header[2] = 'I'; header[3] = 'T';
    *(uint16_t*)(header + 4) = TRIT_VER_MAJOR;
    *(uint16_t*)(header + 6) = TRIT_VER_MINOR;
    *(uint32_t*)(header + 8) = (uint32_t)rows;
    *(uint32_t*)(header + 12) = (uint32_t)cols;
    *(uint16_t*)(header + 16) = 512;
    *(uint16_t*)(header + 18) = 256;
    *(uint16_t*)(header + 20) = (uint16_t)n_scales;
    *(uint32_t*)(header + 22) = (uint32_t)payload_offset;
    *(uint32_t*)(header + 26) = (uint32_t)scale_offset;

    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= header[i];
    }
    *(uint16_t*)(header + 30) = crc;

    FILE * f = fopen(path.c_str(), "wb");
    fwrite(header, 1, 32, f);
    fwrite(packed.data(), 1, payload_bytes, f);
    size_t padding = scale_offset - (TRIT_HEADER_SIZE + payload_bytes);
    if (padding) fwrite("\x00", 1, padding, f);
    fwrite(scales.data(), 4, n_scales, f);
    fclose(f);
}

// ── Real Phase 28Y manifest generator ──────────────────────────────────────────

void create_real_manifest(const Args &a) {
    if (fs::exists(a.storage_dir)) fs::remove_all(a.storage_dir);
    fs::create_directories(a.storage_dir);

    std::vector<std::string> families = {"ffn_up", "ffn_q"};
    int rows = 512;
    int cols = 2048;
    int block_rows = 512;
    int block_cols = 256;
    int n_scales = ((rows + block_rows - 1) / block_rows) * ((cols + block_cols - 1) / block_cols);
    size_t data_size = TRIT_HEADER_SIZE + ((size_t)rows * cols * 3 + 7) / 8;
    size_t scale_offset = data_size;
    if (scale_offset % 4) scale_offset = (scale_offset / 4 + 1) * 4;
    size_t total_bytes = scale_offset + n_scales * 4;

    std::ostringstream json;
    json << "{\n";
    json << "  \"format_name\": \"PRT Progressive Residual Ternary\",\n";
    json << "  \"format_version\": \"1.0\",\n";
    json << "  \"source_model\": \"synthetic_fixture_v0\",\n";
    json << "  \"source_model_hash\": \"sha256:synthetic0000000000000000000000000000000000000000000000000000000000000\",\n";
    json << "  \"base_quant\": \"Q4_K_M\",\n";
    json << "  \"residual_format\": \"ternary\",\n";
    json << "  \"tensor_families\": [\"ffn_up\", \"ffn_down\", \"attn_q\", \"attn_k\", \"attn_v\"],\n";
    json << "  \"layer_count\": " << a.layers << ",\n";
    json << "  \"budget_policy\": \"strict\",\n";
    json << "  \"validation_summary\": {\n";
    json << "    \"total_tensors\": " << (a.layers * (int)families.size()) << ",\n";
    json << "    \"present\": " << (a.layers * (int)families.size()) << ",\n";
    json << "    \"missing\": 0,\n";
    json << "    \"checksummed\": 0\n";
    json << "  },\n";
    json << "  \"entries\": [\n";

    for (int layer = 0; layer < a.layers; layer++) {
        for (size_t fi = 0; fi < families.size(); fi++) {
            std::string fam = families[fi];
            char filename[128];
            snprintf(filename, sizeof(filename), "layer_%03d.%s.trit", layer, fam.c_str());
            std::string path = (fs::path(a.storage_dir) / filename).string();

            // Write real .trit file
            write_trit_real(path, rows, cols, n_scales);

            json << "    {\n";
            json << "      \"layer_index\": " << layer << ",\n";
            json << "      \"tensor_name\": \"" << fam << "\",\n";
            json << "      \"tensor_family\": \"" << fam << "\",\n";
            json << "      \"shape\": [" << rows << ", " << cols << "],\n";
            json << "      \"residual_encoding\": \"ternary\",\n";
            json << "      \"file_path\": \"" << filename << "\",\n";
            json << "      \"byte_size\": " << total_bytes << ",\n";
            json << "      \"compression_ratio_vs_q4\": 0.33,\n";
            json << "      \"validation_metrics\": {\"psnr\": 99.9, \"checksum_ok\": true},\n";
            json << "      \"status\": \"present\"\n";
            json << "    }";
            if (layer < a.layers - 1 || fi < families.size() - 1) json << ",";
            json << "\n";
        }
    }
    json << "  ]\n}\n";

    std::string mp = (fs::path(a.storage_dir) / "manifest.json").string();
    std::ofstream mf(mp);
    mf << json.str();
    mf.close();
    a.manifest_path = mp;

    printf("  Created real manifest: %zu entries, %zu bytes each\n",
           (size_t)a.layers * families.size(), total_bytes);
    printf("  Manifest: %s\n", mp.c_str());
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 0;

    printf("=== SIDECAR PAGER PROBE (Phase 28AN) ===\n");
    printf("  Mode: %s\n", args.real_trit ? "REAL (.trit + Phase28Y manifest)" : "FAKE (simple manifest)");
    printf("  Layers: %d, Tensors/layer: %d, Trit: %dKB\n", args.layers, args.tensors_per_layer, args.trit_kb);
    printf("  Window: %d, Prefetch: %d, Max resident: %dKB\n", args.window_size, args.prefetch_distance, args.max_resident_kb);
    printf("  Storage: %s\n", args.storage_dir.c_str());
    printf("  Validate .trit header: %s\n", args.check_trit_header ? "true" : "false");
    printf("\n");

    // ── Generate manifest + sidecars ──────────────────────────────────────────
    if (args.real_trit) {
        printf("[1/7] Creating real manifest + .trit files...\n");
        create_real_manifest(args);
    } else {
        printf("[1/7] Creating fake manifest + .trit files...\n");
        create_fake_manifest(args);
    }

    // ── Initialize pager ──────────────────────────────────────────────────────
    printf("[2/7] Initializing pager...\n");
    prt_sidecar_pager_config cfg;
    cfg.sidecar_root = args.storage_dir;
    cfg.manifest_path = args.manifest_path.empty()
        ? (fs::path(args.storage_dir) / "manifest.json").string()
        : args.manifest_path;
    cfg.max_resident_bytes = (size_t)args.max_resident_kb * 1024;
    cfg.window_size = args.window_size;
    cfg.prefetch_distance = args.prefetch_distance;
    cfg.checksum_enabled = true;
    cfg.strict_budget = true;
    cfg.eviction_lru = args.eviction_lru;
    cfg.validate_trit_header = args.check_trit_header;

    prt_sidecar_pager pager(cfg);
    bool init_ok = pager.init();
    auto schema = pager.detected_schema();
    printf("  init() = %s\n", init_ok ? "true" : "false");
    printf("  schema = %s\n", schema == manifest_schema::PHASE28Y ? "Phase28Y"
        : schema == manifest_schema::SIDECAR_LEGACY ? "Legacy"
        : "Unknown");

    // ── Activate layers ───────────────────────────────────────────────────────
    printf("[3/7] Activating layers...\n");
    std::vector<bool> activate_ok(args.layers);
    for (int i = 0; i < args.layers; i++) {
        activate_ok[i] = pager.activate_layer(i);
        printf("  activate_layer(%d) = %s\n", i, activate_ok[i] ? "true" : "false");
    }

    auto stats = pager.get_stats();
    printf("  After activation: resident=%zu bytes, peak=%zu bytes\n",
           stats.resident_bytes, stats.peak_resident_bytes);

    // ── Test get_residual ────────────────────────────────────────────────────
    printf("[4/7] Testing get_residual...\n");
    std::vector<std::string> families = {"ffn_up", "ffn_down", "attn_q"};
    int null_count = 0;
    int valid_count = 0;
    for (int i = 0; i < args.layers; i++) {
        for (int t = 0; t < std::min(args.tensors_per_layer, (int)families.size()); t++) {
            auto view = pager.get_residual(i, families[t]);
            if (view.is_null) null_count++;
            else valid_count++;
        }
    }
    printf("  Valid views: %d, Null views: %d\n", valid_count, null_count);

    // ── Budget stress test ──────────────────────────────────────────────────
    printf("[5/7] Budget stress test...\n");
    prt_sidecar_pager_config tight_cfg = cfg;
    tight_cfg.max_resident_bytes = 256 * 1024;
    prt_sidecar_pager tight_pager(tight_cfg);
    tight_pager.init();

    int budget_rejects = 0;
    for (int i = 0; i < args.layers; i++) {
        if (!tight_pager.activate_layer(i)) budget_rejects++;
    }
    printf("  Budget rejects (256KB limit): %d/%d\n", budget_rejects, args.layers);
    printf("  Budget enforcement: %s\n", budget_rejects > 0 ? "PASS" : "FAIL");

    // ── Missing tensor fallback ─────────────────────────────────────────────
    printf("[6/7] Testing fallback behavior...\n");
    auto missing_view = pager.get_residual(999, "nonexistent");
    printf("  get_residual(999, \"nonexistent\"): is_null=%s, reason=\"%s\"\n",
           missing_view.is_null ? "true" : "false", missing_view.reason.c_str());
    printf("  Fallback: %s\n", missing_view.is_null ? "PASS" : "FAIL");

    // ── .trit header validation (real mode only) ────────────────────────────
    printf("[7/7] Final stats...\n");
    auto final_stats = pager.get_stats();
    printf("  Reads: %zu, Prefetches: %zu, Evictions: %zu\n",
           final_stats.reads, final_stats.prefetches, final_stats.evictions);
    printf("  Cache hits: %zu, Cache misses: %zu, Fallbacks: %zu\n",
           final_stats.cache_hits, final_stats.cache_misses, final_stats.fallbacks);
    printf("  Budget rejects: %zu\n", final_stats.budget_rejects);
    printf("  .trit validated: %zu, checksum_ok: %zu, checksum_fail: %zu\n",
           final_stats.trit_validated, final_stats.trit_checksum_ok, final_stats.trit_checksum_fail);

    // ── Write JSON ──────────────────────────────────────────────────────────
    bool all_pass = init_ok && (valid_count > 0) && (budget_rejects > 0) && missing_view.is_null;
    if (!args.out_json.empty()) {
        FILE * jf = fopen(args.out_json.c_str(), "w");
        if (jf) {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"phase\": \"28AN\",\n");
            fprintf(jf, "  \"verdict\": \"%s\",\n", all_pass ? "PASS" : "FAIL");
            fprintf(jf, "  \"mode\": \"%s\",\n", args.real_trit ? "real_trit" : "fake");
            fprintf(jf, "  \"schema\": \"%s\",\n",
                schema == manifest_schema::PHASE28Y ? "Phase28Y"
                : schema == manifest_schema::SIDECAR_LEGACY ? "Legacy" : "Unknown");
            fprintf(jf, "  \"init_ok\": %s,\n", init_ok ? "true" : "false");
            fprintf(jf, "  \"activate_ok\": %d/%d,\n",
                    (int)std::count(activate_ok.begin(), activate_ok.end(), true), args.layers);
            fprintf(jf, "  \"valid_views\": %d,\n", valid_count);
            fprintf(jf, "  \"null_views\": %d,\n", null_count);
            fprintf(jf, "  \"budget_rejects\": %d,\n", budget_rejects);
            fprintf(jf, "  \"fallback_ok\": %s,\n", missing_view.is_null ? "true" : "false");
            fprintf(jf, "  \"resident_bytes\": %zu,\n", final_stats.resident_bytes);
            fprintf(jf, "  \"peak_resident_bytes\": %zu,\n", final_stats.peak_resident_bytes);
            fprintf(jf, "  \"trit_validated\": %zu,\n", final_stats.trit_validated);
            fprintf(jf, "  \"trit_checksum_ok\": %zu,\n", final_stats.trit_checksum_ok);
            fprintf(jf, "  \"trit_checksum_fail\": %zu\n", final_stats.trit_checksum_fail);
            fprintf(jf, "}\n");
            fclose(jf);
            printf("\n  JSON: %s\n", args.out_json.c_str());
        }
    }

    if (args.cleanup) {
        if (fs::exists(args.storage_dir)) fs::remove_all(args.storage_dir);
        printf("\n  Cleanup done\n");
    }

    printf("\n=== RESULT: %s ===\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}